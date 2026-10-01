# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 FFN sub-layers: dense SwiGLU MLP (layer 0) and the 256-expert top-8 sigmoid MoE (layers 1..47).

Hidden states are ``[1, 1, S_local, H]``: sequence block-cyclic over SP rows, replicated over TP cols.

* Dense MLP: gate/up column-parallel, down row-parallel + TP all-reduce.
* Gate (``noaux_tc``, n_group = topk_group = 1): logits = x @ Wg (replicated), then the DeepSeek
  ``moe_grouped_topk`` op — sigmoid, + e_score_correction_bias for selection, top-8, weights = unbiased
  sigmoid scores renormalised (``norm_topk_prob``), x routed_scaling_factor (1.0). Same routing rule as Kimi.
* MoE: DeepSeek EP substrate (routing_setup -> dispatch -> unified_routed_expert_ffn(Silu) -> combine ->
  reduce), experts spread over all chips (2x2: 64/chip, Galaxy 8x4: 8/chip). No shared expert.
  Default: the all-gather MoE block (``tt/moe_ag.py``; ``MiMoRuntimeOptions.moe_ag=False`` for dispatch / combine)
  with the flat streamed expert op (``tt/flat_expert.py``; ``routed_expert="unified"`` for unified_routed_expert_moe).
"""

import math

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_mapper,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.tt.flat_expert import FlatExpert, FlatRoutedExpert
from models.demos.mimo_v2_d_p.tt.mm_configs import best_mm_config, router_mm_config
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions
from models.demos.mimo_v2_d_p.tt.weight_cache import cache_dir, cache_name


def _rep(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


class TtRMSNorm:
    def __init__(self, mesh_device, weight: torch.Tensor | None, eps: float):
        self.w = (
            None
            if weight is None
            else ttnn.from_torch(
                weight.float().reshape(1, 1, -1, ttnn.TILE_SIZE),
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=_rep(mesh_device),
            )
        )
        self.eps = eps
        self.cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )

    def __call__(self, x):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.w, compute_kernel_config=self.cfg)


def all_reduce_tp(x, mesh_device, num_links=None):
    """Sum over the TP (col) axis; deallocates ``x``. ``num_links``: MiMoRuntimeOptions.ar_links (None: op default)."""
    if mesh_device.shape[1] == 1:
        return x
    out = ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_links=num_links)
    x.deallocate(True)
    return out


class TtDenseMLP:
    def __init__(self, mesh_device, sd, weight_dtype=ttnn.bfloat8_b, cache_prefix=None, options=None):
        self.mesh_device = mesh_device
        self.options = options = options or MiMoRuntimeOptions()
        shape = tuple(mesh_device.shape)
        col = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(None, 3))
        row = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(None, 2))
        to = lambda t, m, name: ttnn.as_tensor(
            t.float()[None, None],
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=m,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(mesh_device, cache_prefix, f"mlp.{name}", options),
        )
        self.w_gate = to(sd["gate_proj.weight"].T, col, "gate")
        self.w_up = to(sd["up_proj.weight"].T, col, "up")
        self.w_down = to(sd["down_proj.weight"].T, row, "down")
        self.cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._pcs = {}

    def __call__(self, x):
        M, K = x.shape[2], x.shape[3]
        if M not in self._pcs:
            self._pcs[M] = (
                best_mm_config(self.mesh_device, M, K, self.w_gate.shape[3]),
                best_mm_config(self.mesh_device, M, self.w_down.shape[2], self.w_down.shape[3]),
            )
        pc_up, pc_down = self._pcs[M]
        gate = ttnn.linear(x, self.w_gate, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg, program_config=pc_up)
        up = ttnn.linear(x, self.w_up, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg, program_config=pc_up)
        h = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(h, self.w_down, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg, program_config=pc_down)
        h.deallocate(True)
        return all_reduce_tp(out, self.mesh_device, self.options.ar_links)


class TtGate:
    """noaux_tc sigmoid router -> (indices uint16 RM [S, K], weights [S, K]); fp32 logits + bias (see __init__)."""

    def __init__(self, mesh_device, sd, cfg: MiMoTextConfig, seq_len_per_chip: int, cache_prefix=None, options=None):
        self.K, self.E = cfg.num_experts_per_tok, cfg.n_routed_experts
        self.route_scale = cfg.routed_scaling_factor
        self.w = ttnn.as_tensor(
            sd["weight"].float().T.contiguous()[None, None],
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=_rep(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(mesh_device, cache_prefix, "gate.w", options),
        )
        bias = sd["e_score_correction_bias"].float().view(1, 1, 1, -1).expand(1, 1, seq_len_per_chip, -1).contiguous()
        # fp32 bias + logits: the bias sits at ~1-2 where the bf16 step (0.008-0.016) exceeds the typical
        # top-8/top-9 score gap (~0.003-0.01) -> bf16 flips ~16% of tokens' expert choice.
        self.bias = ttnn.from_torch(
            bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            mesh_mapper=_rep(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.cfg = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )
        # tuned 2D config (bit-identical logits; 32 / 90 us vs 105 / 234 us default at 640 / 2048 tokens per chip).
        # HiFi4 stays: the matmul is x-read bound (HiFi2 is no faster and flips ~4% of tokens' top-8)
        self._pcs = {}
        self.mesh_device = mesh_device

    def __call__(self, x, row_major=False, tiles=False):
        """``row_major``: (indices, weights) both uint16 / bf16 ROW_MAJOR [1, 1, S, K] (the all-gather block's input);
        ``tiles``: the same as TILE (the all-gather block gathers tiles and untilizes after)."""
        M = x.shape[-2]
        if M not in self._pcs:
            self._pcs[M] = router_mm_config(self.mesh_device, M, N=self.E)
        logits = ttnn.linear(x, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg, program_config=self._pcs[M])
        w, idx = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            self.bias,
            n_groups=1,
            summed_experts_per_group=1,
            topk_groups=1,
            n_activated_experts=self.K,
            route_scale=self.route_scale,
            stable_sort=True,
            epsilon=1e-20,
            score_func="sigmoid",
        )
        logits.deallocate(True)
        if tiles:
            return idx, w
        if row_major:
            return ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT), ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT)
        S = x.shape[2]
        idx = ttnn.reshape(ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT), (S, self.K))
        w = ttnn.reshape(w, (S, self.K))
        return idx, w


def prepare_expert_weights(sd, cfg: MiMoTextConfig):
    return [
        {n: sd[f"experts.{e}.{n}.weight"] for n in ("gate_proj", "up_proj", "down_proj")}
        for e in range(cfg.n_routed_experts)
    ]


def build_flat_expert(
    mesh_device,
    sd,
    cfg: MiMoTextConfig,
    experts_per_chip,
    dgs,
    ndg,
    max_tok,
    weights_dtype,
    cache_prefix=None,
    options=None,
):
    """One FlatExpert over the mesh: device (r, c) runs the experts the EP table puts there (the same global ids the
    unified op reads through its global expert idx table: table[c, r], get_ep_mesh_mapper's sharding)."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (dgs, ndg), (rows, cols, dgs, ndg)
    table = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=experts_per_chip, dispatch_group_size=dgs, num_dispatch_groups=ndg
    )
    gids = [[int(g) for g in table[c, r]] for r in range(rows) for c in range(cols)]
    # nn.Linear [out, in] -> x @ W [in, out]; bf16 (the checkpoint values are bf16-exact); built only on a cache miss
    tw = lambda g, n: sd[f"experts.{g}.{n}.weight"].T.contiguous()
    weights = lambda: [[(tw(g, "gate_proj"), tw(g, "up_proj"), tw(g, "down_proj")) for g in gl] for gl in gids]
    wdtype = {ttnn.bfloat4_b: "bf4", ttnn.bfloat8_b: "bf8"}[weights_dtype]
    options = options or MiMoRuntimeOptions()
    cls = FlatExpert if options.routed_expert == "py" else FlatRoutedExpert
    return cls(
        mesh_device,
        weights() if cls is FlatExpert else weights,  # the Python builder takes the weights themselves
        m=max_tok,
        H=cfg.hidden_size,
        I=cfg.moe_intermediate_size,
        gids=gids,
        n_global=cfg.n_routed_experts,
        wdtype=wdtype,
        act="silu",
        pin=1,
        **(
            {"cache_prefix": cache_name(mesh_device, cache_prefix, "experts", options)}
            if cls is FlatRoutedExpert
            else {}
        ),
    )


def moe_capacity_factor(K: int, E: int, n_dev: int, override: int | None = None) -> int:
    """Dispatch-buffer capacity factor (buffer = dispatch_group * seq * factor tokens per chip).

    A token lands on K * (E / n_dev) / E = K / n_dev experts per chip on average (2x2: 2, Galaxy 8x4: 0.25).
    The buffer, and with it dispatch / tilize / combine time, scales with the factor, so size it at 2x the
    expected load (floor 2 = DeepSeek production, cap K = exact worst case); overflow tokens are dropped by
    the dispatch kernel, not corrupted. ``override`` (MiMoRuntimeOptions.moe_capacity) wins.
    """
    if override is not None:
        return override
    return min(K, E // n_dev, max(2, math.ceil(2 * K / n_dev)))


class TtMoE:
    """Expert-parallel routed experts. Default (``options.use_moe_ag``): the all-gather MoE block (tt/moe_ag.py:
    high_bw_all_gather of x / top-k over the dispatch axis, on-device route plan, the flat expert in indexed mode,
    local weighted reduce, gather + add send-back). Else the DeepSeek dispatch / combine substrate."""

    def __init__(
        self,
        mesh_device,
        sd,
        cfg: MiMoTextConfig,
        *,
        seq_len_per_chip,
        num_links=1,
        topology=ttnn.Topology.Linear,
        weights_dtype=None,
        cache_prefix=None,
        options=None,
    ):
        self.mesh_device = mesh_device
        self.options = options = options or MiMoRuntimeOptions()
        weights_dtype = weights_dtype or options.expert_dtype
        E, K, H = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.hidden_size
        self.E, self.K, self.H = E, K, H
        self.gate = TtGate(
            mesh_device,
            {k[len("gate.") :]: v for k, v in sd.items() if k.startswith("gate.")},
            cfg,
            seq_len_per_chip,
            cache_prefix,
            options,
        )
        mc = extract_mesh_config(mesh_device)
        dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
        n_dev = mesh_device.get_num_devices()
        cap = moe_capacity_factor(K, E, n_dev, options.moe_capacity)
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(seq_len_per_chip, E, K, n_dev, dgs, cap)
        table = ExpertMapping.create_dispatch_table(E, dgs, ndg)
        self.routing_setup = TtMoERoutingSetup(
            mesh_device, table, num_links=num_links, experts_per_chip=experts_per_chip
        )
        self.tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, table, dispatch_axis=0)
        self.dispatch = TtDispatchModule(
            mesh_device=mesh_device,
            dispatch_group_size=dgs,
            experts_per_chip=experts_per_chip,
            num_routed_experts=E,
            num_experts_per_tok=K,
            metadata_len=metadata_len,
            max_dispatch_buffer_token_size=max_buf,
            seq_len_per_chip=seq_len_per_chip,
            emb_dim=H,
            cluster_axis=0,
            num_links=num_links,
            topology=topology,
            subdevice_id=None,
        )
        self.combine = TtCombineModule(
            mesh_device=mesh_device,
            dispatch_group_size=dgs,
            num_dispatch_groups=ndg,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=K,
            seq_len_per_chip=seq_len_per_chip,
            cluster_axis=0,
            num_links=num_links,
            topology=topology,
            init_zeros=True,
        )
        self.flat = None
        self.ag = None
        if options.use_moe_ag:
            from models.demos.mimo_v2_d_p.tt.moe_ag import MoeAgBlock, flat_rows

            table_g = ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=experts_per_chip, dispatch_group_size=dgs, num_dispatch_groups=ndg
            )
            gids = [[int(g) for g in table_g[c, r]] for r in range(dgs) for c in range(ndg)]
            # worst-case flat rows (every pair of the column's tokens on this chip): the dispatch capacity factor
            # would let adversarial routing overrun token_index / y (the route plan cannot drop pairs)
            ag_rows = flat_rows(dgs * seq_len_per_chip, K, experts_per_chip)
            self.ag = MoeAgBlock.get(
                mesh_device,
                chunk_size_per_chip=seq_len_per_chip,
                hidden=H,
                k=K,
                n_global=E,
                gids=gids,
                buf_rows=ag_rows,
                options=options,
            )
        if options.flat_expert:
            self.flat = build_flat_expert(
                mesh_device, sd, cfg, experts_per_chip, dgs, ndg, max_tok, weights_dtype, cache_prefix, options
            )
            self.expert = None
        else:
            self._init_unified(mesh_device, sd, cfg, experts_per_chip, dgs, ndg, max_tok, weights_dtype, cache_prefix)
        self.reduce = TtReduceModule(
            mesh_device=mesh_device, topk_dim=3, cluster_axis=1, num_links=num_links, topology=topology
        )

    def _init_unified(self, mesh_device, sd, cfg, experts_per_chip, dgs, ndg, max_tok, weights_dtype, cache_prefix):
        H, I = cfg.hidden_size, cfg.moe_intermediate_size
        gidx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=experts_per_chip, dispatch_group_size=dgs, num_dispatch_groups=ndg
            ),
            mesh_mapper=get_ep_mesh_mapper(mesh_device),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )
        gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)
        # A complete expert cache loads without touching the torch weights (no gather / transpose / stack).
        ec_dir = cache_dir(mesh_device, self.options)
        ec_prefix = None if ec_dir is None or cache_prefix is None else f"{cache_prefix}.experts"
        ec_hit = False
        if ec_prefix is not None and ec_dir.is_dir():
            init_checker(ec_dir)
            ec_hit = TtRoutedExpert.check_cache_complete(ec_dir, ec_prefix, experts_per_chip, weights_dtype)
        self.expert = TtRoutedExpert(
            mesh_device=mesh_device,
            experts_per_chip=experts_per_chip,
            global_expert_idx_table=gidx,
            emb_dim=H,
            hidden_dim=I,
            max_tokens=max_tok,
            torch_weights=None if ec_hit else prepare_expert_weights(sd, cfg),
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype,
            activation=ttnn.RoutedExpertActivation.Silu,
            weight_cache_path=ec_dir if ec_prefix else None,
            cache_name_prefix=ec_prefix,
            hybrid_token_threshold=self.options.re_hybrid_threshold,
        )

    def _call_ag(self, x):
        """All-gather block: x [1,1,S,H] TILE -> [1,1,S,H] TILE (replicated over TP)."""
        tiles = self.ag.tile_topk and self.ag.rows > 1
        idx4, w_rm = self.gate(x, row_major=not tiles, tiles=tiles)
        x_rm = self.ag.to_rm(x)
        gx, _, _ = self.ag.gather(x_rm, idx4, w_rm)
        gathered = self.ag.rows > 1  # one mesh row: gather returns x_rm / idx / w themselves (used until the reduce)
        if gathered:
            ttnn.deallocate(x_rm)
            ttnn.deallocate(w_rm)
        counts, regions, token_index, _ = self.ag.plan()
        y = self.expert_indexed(gx, counts, regions, token_index)
        out = self.ag.reduce(y)
        ttnn.deallocate(y)
        if not gathered:
            ttnn.deallocate(x_rm)
            ttnn.deallocate(w_rm)
        return out

    def expert_indexed(self, gx, counts, regions, token_index):
        """The routed experts on the gathered tokens: flat row r of the output reads gathered row token_index[r].
        ``options.moe_ag_embedding`` (A/B): build the flat buffer locally with ttnn.embedding and run the flat expert
        on it."""
        gx2 = ttnn.reshape(gx, (gx.shape[-2], gx.shape[-1]))
        if self.options.moe_ag_embedding:
            buf = ttnn.embedding(token_index, gx2, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            buf = ttnn.reshape(buf, (buf.shape[-2], buf.shape[-1]))
            y = self.flat(buf, counts, regions, y_row_major=self.ag.y_rm)
            ttnn.deallocate(buf)
            return y
        return self.flat(
            gx2, counts, regions, token_index=token_index, x_pages_per_row=self.ag.xppr, y_row_major=self.ag.y_rm
        )

    def __call__(self, x):
        """x [1,1,S,H] (post-attention-normed, replicated over TP) -> [1,1,S,H]."""
        if self.ag is not None:
            return self._call_ag(x)
        idx, w = self.gate(x)
        offsets, counts, regions, _ = self.routing_setup(
            ttnn_top_k_experts_indices=idx, num_routed_experts=self.E, num_experts_per_tok=self.K
        )
        S = x.shape[2]
        idx = ttnn.reshape(idx, (1, S, self.K))
        w = ttnn.reshape(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (1, S, self.K))
        x3 = ttnn.squeeze(x, 0)
        buf, meta = self.dispatch(x3, w, idx, offsets, self.tt_table)
        if self.flat is not None:  # reads the row-major bf16 buffer itself
            y = self.flat(ttnn.squeeze(ttnn.squeeze(buf, 0), 0), counts, regions)
            ttnn.deallocate(buf)
        else:
            tiled = ttnn.to_layout(ttnn.squeeze(ttnn.squeeze(buf, 0), 0), ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
            ttnn.deallocate(buf)
            y = self.expert(tiled, counts, regions)
        y = ttnn.unsqueeze(ttnn.unsqueeze(y, 0), 0)
        comb = self.combine(y, meta, counts, regions, seq_len_per_chip=S)
        w = ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)
        idx = ttnn.to_memory_config(idx, ttnn.DRAM_MEMORY_CONFIG)
        out = self.reduce(
            comb, weights=w, indices=idx, expert_dispatch_table=self.tt_table
        )  # [1,S,H/tp] reduce-scattered over TP
        out = ttnn.unsqueeze(ttnn.squeeze(out, 0), 0) if len(out.shape) == 4 else ttnn.unsqueeze(out, 0)
        if self.mesh_device.shape[1] > 1 and out.shape[-1] < self.H:
            out = ttnn.all_gather(out, dim=-1, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out
