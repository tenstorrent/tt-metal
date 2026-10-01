# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 routed experts on a 1x4 mesh: DeepSeek EP pipeline with the fused ``unified_routed_expert_moe`` kernel.

    routing as dense [S, 256] (8 nonzeros per row) or (idx [S, 8], weights [S, 8])
      -> masked_bincount + offset_cumsum -> TtDispatchModule (1-chip dispatch group, local dispatch)
      -> mode 'unified' (default): one unified_routed_expert_moe over every local expert with high_precision=True
         (the factory honours the fidelity / fp32 dest for Silu, keeps x, intermediates and output bf16, fp32 partials);
         'loop': per local expert extract -> down(silu(x @ gate) * (x @ up)) via ttnn.linear (fp32 dest) -> insert;
         'unified_lofi': the op without high_precision (LoFi, bf16 dest, bf8 x / output)
      -> TtCombineModule -> TtReduceModule (fused weighted top-k sum over this chip's experts)
      -> all_reduce (cluster_axis=1) -> experts_out [1, 1, S, H] replicated

EP=4: chip c holds experts 64c..64c+63 (bfp8 gate/up [4096, 2048], down [2048, 4096] per expert, never bfp4). The
checkpoint's mxfp4 x e8m0 weights are dequantized on the host at load (reference/weights.py, exact in fp32) one expert
at a time, and the bfp8 device tensors are cached under generated/mimo_v2_6_d_p/tt_cache/experts; a complete cache
skips the dequant. expert_token_counts / expert_region_offsets are per chip [1, 256] (global expert ids). No shared
expert, so the routed partial is the only thing all-reduced.

Adapted from models/demos/gemma4_a4b_d_p/tt/experts.py (itself from ernie45_d_p/tt/moe_unified.py). Uses the bring-up
forks (ttnn/ttnn/bringup): offset_cumsum, dispatch and combine for the 1-device dispatch axis, and
unified_routed_expert_ffn for high_precision (mode 'unified').
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

REPO = Path(__file__).resolve().parents[4]
CACHE_ROOT = REPO / "generated/mimo_v2_6_d_p/tt_cache"

DGS = 1  # dispatch group = one chip (mesh rows)
# Dispatch/combine hold no per-layer state: one pair per (mesh, chunk length) is shared by every MoE layer.
_SEQ_MODULES = {}


def _dispatch(m, x, indices, offsets, table):
    """TtDispatchModule.forward on the bring-up fork ttnn.bringup.dispatch (1-device dispatch axis, no fabric;
    ttnn/ttnn/bringup/INDEX.md). m is the TtDispatchModule holding the sizes; returns (buffer, metadata)."""
    return ttnn.bringup.dispatch(
        input_tensor=x,
        indices_tensor=indices,
        expert_offsets_tensor=offsets,
        expert_dispatch_table_tensor=table,
        dispatch_group_size=m.dispatch_group_size,
        experts_per_chip=m.experts_per_chip,
        num_routed_experts=m.num_routed_experts,
        num_experts_per_tok=m.num_experts_per_tok,
        metadata_len=m.metadata_len,
        max_dispatch_buffer_token_size=m.max_dispatch_buffer_token_size,
        cluster_axis=m.cluster_axis,
        num_links=m.num_links,
        topology=m.topology,
        fp8_output=m.fp8_output,
        subdevice_id=m.subdevice_id,
        num_workers_per_sender=m.num_workers_per_sender,
    )


def _combine(m, buf, meta, counts, region_offsets, seq_len):
    """TtCombineModule.forward on the bring-up fork ttnn.bringup.combine (1-device dispatch axis, no fabric)."""
    return ttnn.bringup.combine(
        buf,
        meta,
        counts,
        region_offsets,
        dispatch_group_size=m.dispatch_group_size,
        experts_per_chip=m.experts_per_chip,
        num_experts_per_tok=m.num_experts_per_tok,
        seq_len_per_chip=seq_len,
        cluster_axis=m.cluster_axis,
        num_links=m.num_links,
        topology=m.topology,
        memory_config=m.memory_config,
        init_zeros=m.init_zeros,
        use_fp8_combine=m.fp8_output,
    )


class LazyExpertWeights:
    """Sequence of {'gate_proj', 'up_proj', 'down_proj'} dicts (HF (out, in) layout, global expert order), each expert
    dequantized from mxfp4 when it is indexed, so the 256 dense experts are never all held on the host."""

    def __init__(self, loader, prefix: str, num_experts: int):
        self.loader, self.prefix, self.n = loader, prefix, num_experts

    def __len__(self):
        return self.n

    def __getitem__(self, e):
        from models.demos.mimo_v2_6_d_p.reference.weights import PackedExpert

        g, u, d = PackedExpert(self.loader, f"{self.prefix}{e}.").weights(torch.float32)
        return {"gate_proj": g, "up_proj": u, "down_proj": d}


class TtExperts:
    def __init__(
        self,
        mesh,
        layer: int,
        torch_weights,
        emb_dim: int,
        hidden_dim: int,
        top_k: int = 8,
        max_seq_len: int = 8192,
        num_links: int = 1,
        weights_dtype=ttnn.bfloat8_b,
        cache: bool = True,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        mode: str = "unified",
        loop_act_dtype=ttnn.bfloat16,
        loop_mid_dtype=ttnn.float32,
    ):
        """torch_weights: sequence (len E) of {'gate_proj' [I, H], 'up_proj' [I, H], 'down_proj' [H, I]}, or None to
        load a complete cache. mode 'unified' (default): unified_routed_expert_moe with high_precision=True at
        math_fidelity + fp32 dest (layer 5 at HiFi4: rel 0.0071, ratio [0.987, 1.021]; HiFi2 fails the ratio);
        'loop': per local expert extract -> ttnn.linear SwiGLU at math_fidelity with fp32 dest -> insert (10x slower);
        'unified_lofi': the op without high_precision (LoFi + bf16 dest + bf8 x: rel 0.019, norm ratio [0.963, 1.054]
        on the layer-1 golden, fails); 'fused': moe_fused_swiglu for every expert (rel 0.030, ratio [0.961, 1.055],
        fails)."""
        n = mesh.get_num_devices()
        E = len(torch_weights) if torch_weights is not None else 256
        assert tuple(mesh.shape) == (1, n) and E % n == 0
        assert weights_dtype != ttnn.bfloat4_b, "owner rule: experts are bfp8 on device, never bfp4"
        self.mesh, self.layer, self.num_links = mesh, layer, num_links
        self.E, self.K, self.H, self.I = E, top_k, emb_dim, hidden_dim
        self.n, self.epc = n, E // n
        # Worst case all K of a token's experts live on one chip: its flat dispatch buffer must hold K*S rows.
        self.capacity_factor = top_k
        self.max_seq_len = max_seq_len

        self.dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh, ExpertMapping.create_dispatch_table(E, DGS, n), dispatch_axis=0
        )
        gidx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=self.epc, dispatch_group_size=DGS, num_dispatch_groups=n
            ),
            mesh_mapper=get_ep_mesh_mapper(mesh),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            dtype=ttnn.uint32,
        )
        gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)
        cache_path = CACHE_ROOT / "experts" if cache else None
        prefix = f"layer_{layer}.experts.{weights_dtype.name}" if cache else None
        if cache:
            from models.demos.deepseek_v3_d_p.utils import fast_cache_checker

            cache_path.mkdir(parents=True, exist_ok=True)
            fast_cache_checker.init_checker(cache_path)
            try:
                if TtRoutedExpert.check_cache_complete(cache_path, prefix, self.epc, weights_dtype):
                    torch_weights = None  # every tensorbin exists: skip the mxfp4 dequant
            finally:
                fast_cache_checker._checker = None
        assert torch_weights is not None or cache, "no weights and no cache"
        assert mode in ("unified", "unified_lofi", "fused", "loop"), mode
        self.mode = mode
        # Loop mode keeps the expert input in bf16: layer 5's MoE input has outlier channels (|x| up to 131, median
        # 0.009) and bfp8 x (one exponent per 16 values) flushes their neighbours (rel 0.045, see known issues).
        self.loop_act_dtype = loop_act_dtype
        # fp32 gate/up/silu*up intermediates: bf16 ones left layer-5 norm ratio [0.976, 1.028] vs [0.988, 1.016].
        self.loop_mid_dtype = loop_mid_dtype
        self.hybrid = max_seq_len if mode == "fused" else None
        # Loop mode: per local expert extract -> routed_expert_ffn (ttnn.matmul, honours this config) -> insert.
        self.loop_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.routed = TtRoutedExpert(
            mesh_device=mesh,
            experts_per_chip=self.epc,
            global_expert_idx_table=gidx,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=max_seq_len,  # per-expert cap = chunk length (a token picks an expert at most once)
            torch_weights=torch_weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype,
            weight_cache_path=cache_path,
            cache_name_prefix=prefix,
            # Used by mode 'unified_lofi' (TtRoutedExpert.__call__, no high_precision: Silu runs LoFi + bf16 dst);
            # moe_fused_swiglu (mode 'fused') takes math_fidelity with fp32_dest / packer_l1_acc cleared.
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            ),
            activation=ttnn.RoutedExpertActivation.Silu,
            hybrid_token_threshold=self.hybrid,
        )
        # cluster_axis=0 has 1 device -> no reduce-scatter: fused weighted top-k sum over local experts only.
        self.reduce = TtReduceModule(
            mesh_device=mesh, topk_dim=3, cluster_axis=0, num_links=num_links, topology=ttnn.Topology.Linear
        )

    def _seq_modules(self, S: int):
        key = (self.E, self.K, self.H, S)
        hit = _SEQ_MODULES.get(key)
        if hit is None or hit[0] is not self.mesh:  # tests reopen the mesh: rebuild on a new one
            assert S <= self.max_seq_len, f"chunk {S} > max_seq_len {self.max_seq_len}"
            epc, metadata_len, max_buf, _ = compute_constants(S, self.E, self.K, self.n, DGS, self.capacity_factor)
            dispatch = TtDispatchModule(
                mesh_device=self.mesh,
                dispatch_group_size=DGS,
                experts_per_chip=epc,
                num_routed_experts=self.E,
                num_experts_per_tok=self.K,
                metadata_len=metadata_len,
                max_dispatch_buffer_token_size=max_buf,
                seq_len_per_chip=S,
                emb_dim=self.H,
                cluster_axis=0,
                num_links=self.num_links,
                topology=ttnn.Topology.Linear,
                subdevice_id=None,
            )
            combine = TtCombineModule(
                mesh_device=self.mesh,
                dispatch_group_size=DGS,
                num_dispatch_groups=self.n,
                experts_per_chip=epc,
                num_experts_per_tok=self.K,
                seq_len_per_chip=S,
                cluster_axis=0,
                num_links=self.num_links,
                topology=ttnn.Topology.Linear,
                init_zeros=True,  # (token, slot) pairs routed to other chips' experts must read as 0
            )
            _SEQ_MODULES[key] = (self.mesh, dispatch, combine)
        return _SEQ_MODULES[key][1:]

    def topk_from_dense(self, dense):
        """dense [1,1,S,E] (exactly K nonzeros per row) -> (idx [1,1,S,K], weights bf16 [1,1,S,K])."""
        if dense.dtype != ttnn.bfloat16:
            dense = ttnn.typecast(dense, ttnn.bfloat16)
        wts, idx = ttnn.topk(dense, k=self.K, dim=-1, largest=True, sorted=True)
        return idx, wts

    def routed_partial(self, x, idx, wts):
        """This chip's partial over its local experts. x [1,1,S,H] bf16; idx/wts [1,1,S,K] -> [1,1,S,H] TILE."""
        S, K = x.shape[-2], self.K
        dispatch, combine = self._seq_modules(S)
        idx2 = ttnn.reshape(ttnn.typecast(idx, ttnn.uint16) if idx.dtype != ttnn.uint16 else idx, (S, K))
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(idx2, self.dispatch_table, self.E, K)
        # 4th output (#57859, every device's offsets row along the axis) is not used here
        offsets, counts, region_offsets, _ = ttnn.bringup.offset_cumsum(
            hist,
            cluster_axis=0,
            num_links=self.num_links,
            experts_per_chip=self.epc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ind = ttnn.reshape(ttnn.to_layout(idx2, ttnn.ROW_MAJOR_LAYOUT), (1, S, K))
        wb = ttnn.typecast(wts, ttnn.bfloat16) if wts.dtype != ttnn.bfloat16 else wts
        scores = ttnn.reshape(ttnn.to_layout(wb, ttnn.ROW_MAJOR_LAYOUT), (1, S, K))
        buf, meta = _dispatch(dispatch, ttnn.squeeze(x, dim=0), ind, offsets, self.dispatch_table)
        buf2 = ttnn.squeeze(ttnn.squeeze(buf, dim=0), dim=0)
        if self.hybrid is not None:
            # moe_fused_swiglu reads a TILE bf8 buffer (at each expert's region offset).
            rm = buf2
            buf2 = ttnn.to_layout(rm, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
            ttnn.deallocate(rm)
        # ROW_MAJOR bf16 buffer -> the unified op's fused path (tilize + bf8 pack inside the kernel, fresh output).
        if self.mode == "loop":
            out = self._loop_experts(buf2, counts, region_offsets, S)
        elif self.mode == "unified":
            out = self._unified_experts(buf2, counts, region_offsets)
        else:
            out = self.routed(buf2, counts, region_offsets)
        ttnn.deallocate(buf2)
        out = ttnn.unsqueeze(ttnn.unsqueeze(out, dim=0), dim=0)
        comb = _combine(combine, out, meta, counts, region_offsets, S)
        ttnn.deallocate(out)
        red = self.reduce(comb, weights=scores, indices=ind, expert_dispatch_table=self.dispatch_table)
        ttnn.deallocate(comb)
        for t in (hist, offsets, counts, region_offsets, meta):
            ttnn.deallocate(t)
        red = ttnn.to_layout(red, ttnn.TILE_LAYOUT)
        return ttnn.reshape(red, (1, 1, S, self.H))

    def _unified_experts(self, buf, counts, region_offsets):
        """One fused unified_routed_expert_moe over every local expert with high_precision=True: the factory honours
        loop_cfg (fidelity, fp32 dest) for Silu and keeps x, intermediates and the output in bf16 (layer 5's outlier
        channels fail with a bf8 x). buf is ROW_MAJOR bf16; returns a fresh TILE bf16 buffer."""
        r = self.routed
        return ttnn.bringup.unified_routed_expert_moe(
            buf,
            region_offsets,
            counts,
            r.global_expert_idx_table,
            r.gate_projs,
            r.up_projs,
            r.down_projs,
            max_dispatched_tokens_per_expert=r.max_tokens,
            compute_kernel_config=self.loop_cfg,
            activation=ttnn.bringup.RoutedExpertActivation.Silu,
            high_precision=True,
        )

    def _ffn(self, x, wg, wu, wd):
        """down(silu(x @ wg) * (x @ wu)) with ttnn.linear (auto program config) at loop_cfg; x [M, H] TILE."""
        cfg = self.loop_cfg
        mid = self.loop_mid_dtype
        g = ttnn.linear(x, wg, compute_kernel_config=cfg, dtype=mid)
        u = ttnn.linear(x, wu, compute_kernel_config=cfg, dtype=mid)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=mid)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        y = ttnn.linear(h, wd, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        return y

    def _loop_experts(self, buf, counts, region_offsets, S):
        """Per local expert: extract its rows ([S, H] slab, S = per-expert cap = chunk length), the SwiGLU FFN at the
        module's fidelity with fp32 dest, insert back in place. Returns a fresh TILE buffer (buf is left alone)."""
        r = self.routed
        x = ttnn.to_layout(buf, ttnn.TILE_LAYOUT, dtype=self.loop_act_dtype)
        out = ttnn.to_layout(buf, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)  # bf16 output slab (insert matches dtypes)
        for le in range(self.epc):
            tok = ttnn.experimental.deepseek_prefill.extract(
                x,
                region_offsets,
                counts,
                r.global_expert_idx_table,
                local_expert_id=le,
                max_dispatched_tokens_per_expert=S,
            )
            y = self._ffn(tok, r.gate_projs[le], r.up_projs[le], r.down_projs[le])
            ttnn.deallocate(tok)
            out = ttnn.experimental.deepseek_prefill.insert(
                out, y, region_offsets, counts, r.global_expert_idx_table, local_expert_id=le
            )
            ttnn.deallocate(y)
        ttnn.deallocate(x)
        return out

    def __call__(self, x, dense=None, idx=None, wts=None):
        """x [1,1,S,H] replicated; routing as dense [1,1,S,E] or (idx, wts) -> experts_out [1,1,S,H] replicated."""
        tmp = []
        if idx is None:
            idx, wts = self.topk_from_dense(dense)
            tmp += [idx, wts]
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
            tmp.append(x)
        part = self.routed_partial(x, idx, wts)
        out = ttnn.all_reduce(part, cluster_axis=1)
        ttnn.deallocate(part)
        for t in tmp:
            ttnn.deallocate(t)
        return out
