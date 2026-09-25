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
"""

import math
import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, extract_mesh_config, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.tt.mm_configs import best_mm_config
from models.demos.mimo_v2_d_p.tt.weight_cache import cache_dir, cache_name


def _rep(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device)


class TtRMSNorm:
    def __init__(self, mesh_device, weight: torch.Tensor | None, eps: float):
        self.w = (
            None
            if weight is None
            else ttnn.from_torch(
                weight.float().reshape(1, 1, -1, ttnn.TILE_SIZE), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.bfloat16, mesh_mapper=_rep(mesh_device),
            )
        )
        self.eps = eps
        self.cfg = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)

    def __call__(self, x):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.w, compute_kernel_config=self.cfg)


# TP all-reduce links: op default (measured on 2x2: 2 links made the reduce-scatter ~1.8x slower).
AR_LINKS = int(os.environ["MIMO_AR_LINKS"]) if os.environ.get("MIMO_AR_LINKS") else None


def all_reduce_tp(x, mesh_device):
    """Sum over the TP (col) axis; deallocates ``x``."""
    if mesh_device.shape[1] == 1:
        return x
    out = ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_links=AR_LINKS)
    x.deallocate(True)
    return out


class TtDenseMLP:
    def __init__(self, mesh_device, sd, weight_dtype=ttnn.bfloat8_b, cache_prefix=None):
        self.mesh_device = mesh_device
        shape = tuple(mesh_device.shape)
        col = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(None, 3))
        row = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(None, 2))
        to = lambda t, m, name: ttnn.as_tensor(
            t.float()[None, None], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=weight_dtype, mesh_mapper=m,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_file_name=cache_name(mesh_device, cache_prefix, f"mlp.{name}"),
        )
        self.w_gate = to(sd["gate_proj.weight"].T, col, "gate")
        self.w_up = to(sd["up_proj.weight"].T, col, "up")
        self.w_down = to(sd["down_proj.weight"].T, row, "down")
        self.cfg = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True)
        self._pcs = {}

    def __call__(self, x):
        M, K = x.shape[2], x.shape[3]
        if M not in self._pcs:
            self._pcs[M] = (best_mm_config(self.mesh_device, M, K, self.w_gate.shape[3]),
                            best_mm_config(self.mesh_device, M, self.w_down.shape[2], self.w_down.shape[3]))
        pc_up, pc_down = self._pcs[M]
        gate = ttnn.linear(x, self.w_gate, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg, program_config=pc_up)
        up = ttnn.linear(x, self.w_up, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg, program_config=pc_up)
        h = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=ttnn.bfloat16)
        gate.deallocate(True)
        up.deallocate(True)
        out = ttnn.linear(h, self.w_down, dtype=ttnn.bfloat16, compute_kernel_config=self.cfg, program_config=pc_down)
        h.deallocate(True)
        return all_reduce_tp(out, self.mesh_device)


class TtGate:
    """noaux_tc sigmoid router -> (indices uint16 RM [S, K], weights [S, K]); fp32 logits + bias (see __init__)."""

    def __init__(self, mesh_device, sd, cfg: MiMoTextConfig, seq_len_per_chip: int, cache_prefix=None):
        self.K, self.E = cfg.num_experts_per_tok, cfg.n_routed_experts
        self.route_scale = cfg.routed_scaling_factor
        self.w = ttnn.as_tensor(
            sd["weight"].float().T.contiguous()[None, None], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            mesh_mapper=_rep(mesh_device), memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(mesh_device, cache_prefix, "gate.w"),
        )
        bias = sd["e_score_correction_bias"].float().view(1, 1, 1, -1).expand(1, 1, seq_len_per_chip, -1).contiguous()
        # fp32 bias + logits: the bias sits at ~1-2 where the bf16 step (0.008-0.016) exceeds the typical
        # top-8/top-9 score gap (~0.003-0.01) -> bf16 flips ~16% of tokens' expert choice.
        self.bias = ttnn.from_torch(bias, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, mesh_mapper=_rep(mesh_device),
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.cfg = ttnn.init_device_compute_kernel_config(mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)

    def __call__(self, x):
        logits = ttnn.linear(x, self.w, dtype=ttnn.float32, compute_kernel_config=self.cfg)
        w, idx = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits, self.bias, n_groups=1, summed_experts_per_group=1, topk_groups=1, n_activated_experts=self.K,
            route_scale=self.route_scale, stable_sort=True, epsilon=1e-20, score_func="sigmoid",
        )
        logits.deallocate(True)
        S = x.shape[2]
        idx = ttnn.reshape(ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT), (S, self.K))
        w = ttnn.reshape(w, (S, self.K))
        return idx, w


def prepare_expert_weights(sd, cfg: MiMoTextConfig):
    return [
        {n: sd[f"experts.{e}.{n}.weight"] for n in ("gate_proj", "up_proj", "down_proj")} for e in range(cfg.n_routed_experts)
    ]


def default_expert_dtype():
    """Routed-expert weight dtype (``MIMO_EXPERT_DTYPE`` = bf4 | bf8). bf4 (DeepSeek's production default) halves
    the expert weight bandwidth, which bounds the MoE at 64 experts/chip on 2x2 (experts 3.66 -> 2.73 ms at
    640 tok/chip); the source is MXFP4, so the loss is small (6-layer stitched KV PCC >= 0.997 vs >= 0.9994 at bf8)."""
    return {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b}[os.environ.get("MIMO_EXPERT_DTYPE", "bf4")]


def moe_capacity_factor(K: int, E: int, n_dev: int) -> int:
    """Dispatch-buffer capacity factor (buffer = dispatch_group * seq * factor tokens per chip).

    A token lands on K * (E / n_dev) / E = K / n_dev experts per chip on average (2x2: 2, Galaxy 8x4: 0.25).
    The buffer, and with it dispatch / tilize / combine time, scales with the factor, so size it at 2x the
    expected load (floor 2 = DeepSeek production, cap K = exact worst case); overflow tokens are dropped by
    the dispatch kernel, not corrupted. ``MIMO_MOE_CAPACITY`` overrides.
    """
    if os.environ.get("MIMO_MOE_CAPACITY"):
        return int(os.environ["MIMO_MOE_CAPACITY"])
    return min(K, E // n_dev, max(2, math.ceil(2 * K / n_dev)))


class TtMoE:
    """Expert-parallel routed experts (DeepSeek substrate, SiLU fused expert FFN)."""

    def __init__(self, mesh_device, sd, cfg: MiMoTextConfig, *, seq_len_per_chip, num_links=1, topology=ttnn.Topology.Linear,
                 weights_dtype=None, cache_prefix=None):
        self.mesh_device = mesh_device
        weights_dtype = weights_dtype or default_expert_dtype()
        E, K, H, I = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.hidden_size, cfg.moe_intermediate_size
        self.E, self.K, self.H = E, K, H
        self.gate = TtGate(mesh_device, {k[len("gate.") :]: v for k, v in sd.items() if k.startswith("gate.")}, cfg, seq_len_per_chip,
                           cache_prefix)
        mc = extract_mesh_config(mesh_device)
        dgs, ndg = mc.dispatch_group_size, mc.num_dispatch_groups
        n_dev = mesh_device.get_num_devices()
        cap = moe_capacity_factor(K, E, n_dev)
        experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(seq_len_per_chip, E, K, n_dev, dgs, cap)
        table = ExpertMapping.create_dispatch_table(E, dgs, ndg)
        self.routing_setup = TtMoERoutingSetup(mesh_device, table, num_links=num_links, experts_per_chip=experts_per_chip)
        self.tt_table = TtDispatchModule.shard_expert_dispatch_table(mesh_device, table, dispatch_axis=0)
        self.dispatch = TtDispatchModule(
            mesh_device=mesh_device, dispatch_group_size=dgs, experts_per_chip=experts_per_chip, num_routed_experts=E,
            num_experts_per_tok=K, metadata_len=metadata_len, max_dispatch_buffer_token_size=max_buf,
            seq_len_per_chip=seq_len_per_chip, emb_dim=H, cluster_axis=0, num_links=num_links, topology=topology, subdevice_id=None,
        )
        self.combine = TtCombineModule(
            mesh_device=mesh_device, dispatch_group_size=dgs, num_dispatch_groups=ndg, experts_per_chip=experts_per_chip,
            num_experts_per_tok=K, seq_len_per_chip=seq_len_per_chip, cluster_axis=0, num_links=num_links, topology=topology, init_zeros=True,
        )
        gidx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(experts_per_chip=experts_per_chip, dispatch_group_size=dgs, num_dispatch_groups=ndg),
            mesh_mapper=get_ep_mesh_mapper(mesh_device), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32,
        )
        gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)
        # A complete expert cache loads without touching the torch weights (no gather / transpose / stack).
        ec_dir = cache_dir(mesh_device)
        ec_prefix = None if ec_dir is None or cache_prefix is None else f"{cache_prefix}.experts"
        ec_hit = False
        if ec_prefix is not None and ec_dir.is_dir():
            init_checker(ec_dir)
            ec_hit = TtRoutedExpert.check_cache_complete(ec_dir, ec_prefix, experts_per_chip, weights_dtype)
        self.expert = TtRoutedExpert(
            mesh_device=mesh_device, experts_per_chip=experts_per_chip, global_expert_idx_table=gidx, emb_dim=H, hidden_dim=I,
            max_tokens=max_tok, torch_weights=None if ec_hit else prepare_expert_weights(sd, cfg), activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype, activation=ttnn.RoutedExpertActivation.Silu,
            weight_cache_path=ec_dir if ec_prefix else None, cache_name_prefix=ec_prefix,
        )
        self.reduce = TtReduceModule(mesh_device=mesh_device, topk_dim=3, cluster_axis=1, num_links=num_links, topology=topology)

    def __call__(self, x):
        """x [1,1,S,H] (post-attention-normed, replicated over TP) -> [1,1,S,H]."""
        idx, w = self.gate(x)
        offsets, counts, regions, _ = self.routing_setup(ttnn_top_k_experts_indices=idx, num_routed_experts=self.E, num_experts_per_tok=self.K)
        S = x.shape[2]
        idx = ttnn.reshape(idx, (1, S, self.K))
        w = ttnn.reshape(ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT), (1, S, self.K))
        x3 = ttnn.squeeze(x, 0)
        buf, meta = self.dispatch(x3, w, idx, offsets, self.tt_table)
        tiled = ttnn.to_layout(ttnn.squeeze(ttnn.squeeze(buf, 0), 0), ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(buf)
        y = self.expert(tiled, counts, regions)
        y = ttnn.unsqueeze(ttnn.unsqueeze(y, 0), 0)
        comb = self.combine(y, meta, counts, regions, seq_len_per_chip=S)
        w = ttnn.to_memory_config(w, ttnn.DRAM_MEMORY_CONFIG)
        idx = ttnn.to_memory_config(idx, ttnn.DRAM_MEMORY_CONFIG)
        out = self.reduce(comb, weights=w, indices=idx, expert_dispatch_table=self.tt_table)  # [1,S,H/tp] reduce-scattered over TP
        out = ttnn.unsqueeze(ttnn.squeeze(out, 0), 0) if len(out.shape) == 4 else ttnn.unsqueeze(out, 0)
        if self.mesh_device.shape[1] > 1 and out.shape[-1] < self.H:
            out = ttnn.all_gather(out, dim=-1, cluster_axis=1, topology=ttnn.Topology.Linear)
        return out
