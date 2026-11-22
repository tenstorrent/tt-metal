# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 routed experts on a 2x2 mesh: DeepSeek 2D EP pipeline with the fused ``unified_routed_expert_moe`` kernel.

2x2 port of models/demos/mimo_v2_6_d_p/tt/experts.py (1x4, local 1-chip dispatch groups). The 2x2 mesh has no
size-1 axis, so this uses the layout the DeepSeek ops were built for (deepseek_v3_d_p/tt/moe/README.md):

    x [1,1,S,H], dense routing [1,1,S,E] replicated
      -> ttnn.mesh_partition(dim -2, cluster_axis 0): chip (r, c) keeps rows [r*S/2, (r+1)*S/2) (local, no transfer)
      -> ttnn.topk on the dense row half -> (idx, wts) [1,1,S/2,8]
      -> masked_bincount + ttnn.bringup.offset_cumsum (cluster_axis 0, histograms all_gathered over the 2 rows)
      -> ttnn.bringup.dispatch (dispatch_group_size 2, cluster_axis 0, fabric, Topology.Linear): each column c is a
         dispatch group holding experts 128c .. 128c+127; chip (r, c) holds experts 128c + 64r .. +63
         (ExpertMapping col-major: device_idx = 2c + r)
      -> mode 'unified' (default): ttnn.bringup.unified_routed_expert_moe, high_precision=True, HiFi4 + fp32 dest
         (bf16 x / intermediates / output, fp32 partials); 'loop': per local expert extract -> ttnn.linear SwiGLU
         -> insert; 'unified_lofi': the op without high_precision (fails the norm ratio, known issues)
      -> ttnn.bringup.combine (cluster_axis 0, fabric) -> post_combine_reduce (fused weighted top-k sum over group
         c's experts; the dispatch table masks the other group) -> [1,1,S/2,H]
      -> ttnn.all_reduce(cluster_axis 1): add the other dispatch group
      -> ttnn.all_gather(dim -2, cluster_axis 0): experts_out [1,1,S,H] replicated (component boundary unchanged)

Per-expert cap (max_dispatched_tokens_per_expert) = 2 x S/2 = S; flat dispatch buffer capacity factor 8 (worst
case: all 8 of every token's experts on one chip). expert_token_counts / expert_region_offsets are [1, 256] (global
expert ids). Weights: mxfp4 x e8m0 dequantized on the host (reference/weights.py, exact in fp32) one expert at a
time, bfp8 on device (never bfp4), cached under generated/mimo_v2_6_d_p_2x2/tt_cache/experts (the 2x2 device order
differs from the 1x4 cache's).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

REPO = Path(__file__).resolve().parents[4]
CACHE_ROOT = REPO / "generated/mimo_v2_6_d_p_2x2/tt_cache"

DISPATCH_AXIS = 0  # dispatch along mesh rows (SP axis); each column is one dispatch group
CAPACITY_FACTOR = 8  # flat dispatch buffer = 8 x per-expert cap (all top-8 experts of every token on one chip)
# Dispatch/combine hold no per-layer state: one pair per (mesh, chunk length) is shared by every MoE layer.
_SEQ_MODULES = {}


def _dispatch(m, x, indices, offsets, table):
    """TtDispatchModule.forward on the bring-up fork ttnn.bringup.dispatch (with 2 devices on the dispatch axis it
    behaves as the source op: fabric on). m is the TtDispatchModule holding the sizes; returns (buffer, metadata)."""
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
    """TtCombineModule.forward on the bring-up fork ttnn.bringup.combine (2-device dispatch axis: fabric on)."""
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
        math_fidelity=ttnn.MathFidelity.HiFi4,
        mode: str = "unified",
        loop_act_dtype=ttnn.bfloat16,
        loop_mid_dtype=ttnn.float32,
    ):
        """torch_weights: sequence (len E) of {'gate_proj' [I, H], 'up_proj' [I, H], 'down_proj' [H, I]}, or None to
        load a complete cache. max_seq_len: the longest chunk (full S, before the row split)."""
        rows, cols = tuple(mesh.shape)
        n = rows * cols
        E = len(torch_weights) if torch_weights is not None else 256
        assert rows > 1 and cols > 1 and E % n == 0, f"2D mesh expected, got {tuple(mesh.shape)}"
        assert weights_dtype != ttnn.bfloat4_b, "owner rule: experts are bfp8 on device, never bfp4"
        assert mode in ("unified", "unified_lofi", "loop"), mode
        self.mesh, self.layer, self.num_links, self.mode = mesh, layer, num_links, mode
        self.E, self.K, self.H, self.I = E, top_k, emb_dim, hidden_dim
        self.dgs, self.ngroups, self.n, self.epc = rows, cols, n, E // n
        self.max_seq_len = max_seq_len

        # (num_groups, E + 1): column c's experts -> their chip row, -1 elsewhere; sharded over columns,
        # replicated over rows -> per chip (1, E + 1).
        self.dispatch_table = TtDispatchModule.shard_expert_dispatch_table(
            mesh, ExpertMapping.create_dispatch_table(E, self.dgs, self.ngroups), dispatch_axis=DISPATCH_AXIS
        )
        gidx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=self.epc, dispatch_group_size=self.dgs, num_dispatch_groups=self.ngroups
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
        # bf16 x in loop mode: layer 5's MoE input has outlier channels (known issues).
        self.loop_act_dtype = loop_act_dtype
        self.loop_mid_dtype = loop_mid_dtype
        # HiFi4 + fp32 dest for every expert matmul (owner rule; high_precision honours it for Silu).
        self.cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.routed = TtRoutedExpert(
            mesh_device=mesh,
            experts_per_chip=self.epc,
            global_expert_idx_table=gidx,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            # Per-expert cap = dispatch group size x rows per chip = the full chunk (a token picks an expert once).
            max_tokens=max_seq_len,
            torch_weights=torch_weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype,
            weight_cache_path=cache_path,
            cache_name_prefix=prefix,
            compute_kernel_config=self.cfg,  # used by mode 'unified_lofi' only (the factory forces LoFi there)
            activation=ttnn.RoutedExpertActivation.Silu,
        )

    def _seq_modules(self, S_chip: int):
        key = (self.E, self.K, self.H, S_chip)
        hit = _SEQ_MODULES.get(key)
        if hit is None or hit[0] is not self.mesh:  # tests reopen the mesh: rebuild on a new one
            assert self.dgs * S_chip <= self.max_seq_len, f"chunk {self.dgs * S_chip} > max_seq_len {self.max_seq_len}"
            epc, metadata_len, max_buf, _ = compute_constants(S_chip, self.E, self.K, self.n, self.dgs, CAPACITY_FACTOR)
            assert epc == self.epc
            dispatch = TtDispatchModule(
                mesh_device=self.mesh,
                dispatch_group_size=self.dgs,
                experts_per_chip=epc,
                num_routed_experts=self.E,
                num_experts_per_tok=self.K,
                metadata_len=metadata_len,
                max_dispatch_buffer_token_size=max_buf,
                seq_len_per_chip=S_chip,
                emb_dim=self.H,
                cluster_axis=DISPATCH_AXIS,
                num_links=self.num_links,
                topology=ttnn.Topology.Linear,
                subdevice_id=None,
            )
            combine = TtCombineModule(
                mesh_device=self.mesh,
                dispatch_group_size=self.dgs,
                num_dispatch_groups=self.ngroups,
                experts_per_chip=epc,
                num_experts_per_tok=self.K,
                seq_len_per_chip=S_chip,
                cluster_axis=DISPATCH_AXIS,
                num_links=self.num_links,
                topology=ttnn.Topology.Linear,
                init_zeros=True,  # (token, slot) pairs routed to the other group's experts read as 0
            )
            _SEQ_MODULES[key] = (self.mesh, dispatch, combine)
        return _SEQ_MODULES[key][1:]

    def topk_from_dense(self, dense):
        """dense [1,1,s,E] (exactly K nonzeros per row) -> (idx uint16 [1,1,s,K], weights bf16 [1,1,s,K])."""
        if dense.dtype != ttnn.bfloat16:
            dense = ttnn.typecast(dense, ttnn.bfloat16)
        wts, idx = ttnn.topk(dense, k=self.K, dim=-1, largest=True, sorted=True)
        return idx, wts

    def routed_half(self, x, idx, wts):
        """Group c's experts over this chip's row half. x [1,1,s,H] bf16 (s = S/2); idx/wts [1,1,s,K]
        -> [1,1,s,H] TILE (sum over the 128 experts of this chip's column)."""
        s, K = x.shape[-2], self.K
        dispatch, combine = self._seq_modules(s)
        idx2 = ttnn.reshape(ttnn.typecast(idx, ttnn.uint16) if idx.dtype != ttnn.uint16 else idx, (s, K))
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(idx2, self.dispatch_table, self.E, K)
        # 4th output (#57859, every device's offsets row along the axis) is not used here
        offsets, counts, region_offsets, _ = ttnn.bringup.offset_cumsum(
            hist,
            cluster_axis=DISPATCH_AXIS,
            num_links=self.num_links,
            experts_per_chip=self.epc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ind = ttnn.reshape(ttnn.to_layout(idx2, ttnn.ROW_MAJOR_LAYOUT), (1, s, K))
        wb = ttnn.typecast(wts, ttnn.bfloat16) if wts.dtype != ttnn.bfloat16 else wts
        scores = ttnn.reshape(ttnn.to_layout(wb, ttnn.ROW_MAJOR_LAYOUT), (1, s, K))
        buf, meta = _dispatch(dispatch, ttnn.squeeze(x, dim=0), ind, offsets, self.dispatch_table)
        buf2 = ttnn.squeeze(ttnn.squeeze(buf, dim=0), dim=0)
        if self.mode == "loop":
            out = self._loop_experts(buf2, counts, region_offsets)
        elif self.mode == "unified":
            out = self._unified_experts(buf2, counts, region_offsets)
        else:
            out = self.routed(buf2, counts, region_offsets)
        ttnn.deallocate(buf2)
        out = ttnn.unsqueeze(ttnn.unsqueeze(out, dim=0), dim=0)
        comb = _combine(combine, out, meta, counts, region_offsets, s)
        ttnn.deallocate(out)
        # Fused weighted top-k sum (TtReduceModule's kernel without its reduce_scatter): slots whose expert is not in
        # this chip's dispatch group are skipped via the dispatch table.
        w5 = ttnn.unsqueeze(ttnn.unsqueeze(scores, dim=-1), dim=0)
        red = ttnn.experimental.deepseek_prefill.post_combine_reduce(
            comb,
            w5,
            ind,
            self.dispatch_table,
            expert_dim=3,
            output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(comb)
        for t in (hist, offsets, counts, region_offsets, meta):
            ttnn.deallocate(t)
        red = ttnn.to_layout(red, ttnn.TILE_LAYOUT)
        return ttnn.reshape(red, (1, 1, s, self.H))

    def _unified_experts(self, buf, counts, region_offsets):
        """One fused unified_routed_expert_moe over every local expert with high_precision=True (bf16 x, intermediates
        and output, fp32 partials, cfg honoured). buf is ROW_MAJOR bf16; returns a fresh TILE bf16 buffer."""
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
            compute_kernel_config=self.cfg,
            activation=ttnn.bringup.RoutedExpertActivation.Silu,
            high_precision=True,
        )

    def _ffn(self, x, wg, wu, wd):
        """down(silu(x @ wg) * (x @ wu)) with ttnn.linear at cfg; x [M, H] TILE."""
        cfg, mid = self.cfg, self.loop_mid_dtype
        g = ttnn.linear(x, wg, compute_kernel_config=cfg, dtype=mid)
        u = ttnn.linear(x, wu, compute_kernel_config=cfg, dtype=mid)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=mid)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        y = ttnn.linear(h, wd, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        return y

    def _loop_experts(self, buf, counts, region_offsets):
        """Per local expert: extract (cap = per-expert cap), SwiGLU via ttnn.linear, insert. Returns a fresh buffer."""
        r = self.routed
        x = ttnn.to_layout(buf, ttnn.TILE_LAYOUT, dtype=self.loop_act_dtype)
        out = ttnn.to_layout(buf, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for le in range(self.epc):
            tok = ttnn.experimental.deepseek_prefill.extract(
                x,
                region_offsets,
                counts,
                r.global_expert_idx_table,
                local_expert_id=le,
                max_dispatched_tokens_per_expert=r.max_tokens,
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
        """x [1,1,S,H] replicated; routing as dense [1,1,S,E] or (idx, wts) [1,1,S,K], all replicated
        -> experts_out [1,1,S,H] replicated."""
        tmp = []
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
            tmp.append(x)
        # Row half of this chip's mesh row (local slice, no transfer).
        xh = ttnn.mesh_partition(x, dim=-2, cluster_axis=DISPATCH_AXIS)
        if idx is None:
            dh = ttnn.mesh_partition(dense, dim=-2, cluster_axis=DISPATCH_AXIS)
            ih, wh = self.topk_from_dense(dh)
            tmp += [dh]
        else:
            ih = ttnn.mesh_partition(idx, dim=-2, cluster_axis=DISPATCH_AXIS)
            wh = ttnn.mesh_partition(wts, dim=-2, cluster_axis=DISPATCH_AXIS)
        tmp += [xh, ih, wh]
        part = self.routed_half(xh, ih, wh)
        # Add the other dispatch group's experts, then restore the full sequence on every chip.
        summed = ttnn.all_reduce(part, cluster_axis=1)
        ttnn.deallocate(part)
        out = ttnn.all_gather(summed, dim=-2, cluster_axis=DISPATCH_AXIS)
        ttnn.deallocate(summed)
        for t in tmp:
            ttnn.deallocate(t)
        return out
