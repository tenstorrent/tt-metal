# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hy4 routed experts (HF HYV4Experts: 256 experts, top-8, SwiGLU 2048 clamped at 10) on the 2x2 mesh, per plan.md.

DeepSeek 2D expert parallelism (deepseek_v3_d_p/tt/moe/README.md), the layout of mimo_v2_6_d_p_2x2/tt/experts.py:

    x [1,1,S/2,H] bf16 (ffn_norm: rows split over axis 0, replicated over axis 1), (idx, wts) [1,1,S/2,8] (router)
      -> masked_bincount + ttnn.bringup.offset_cumsum (cluster_axis 0: the 2 row halves of a column)
      -> ttnn.bringup.dispatch (dispatch_group_size 2, cluster_axis 0, fabric on the FABRIC_2D mesh): column c is a
         dispatch group holding experts 128c .. 128c+127; chip (r, c) holds experts 128c + 64r .. +63
         (ExpertMapping column-major: device index 2c + r)
      -> mode 'unified' (default): ttnn.bringup.unified_routed_expert_moe, ClampedSiluGlu (silu(min(g, 10)) *
         clamp(u, -10, 10), limit baked in the kernel), high_precision=True, HiFi4 + fp32 dest (bf16 x /
         intermediates / output, fp32 partials); 'loop': per local expert extract -> ttnn.linear gate / up (fp32)
         -> ttnn.clamp -> multiply (SILU) -> ttnn.linear down -> insert
      -> ttnn.bringup.combine (cluster_axis 0) -> post_combine_reduce (weighted top-k sum over column c's experts;
         the dispatch table masks the other group's slots) -> [1,1,S/2,H] partial (group c only)
      -> ttnn.reduce_scatter(dim 3, cluster_axis 1), fp32: add the other group's partial, keep this chip's hidden
         half -> experts_out [1,1,S/2,H/2] fp32 (the residual's column split, like tt/mlp.py)

Weights: gate_up_proj [256, 4096, 6144] (bf16) is split on the host into gate (rows 0-2047) and up (rows 2048-4095);
down_proj [256, 6144, 2048]; bfp8 on device (never bfp4), read one expert at a time from the checkpoint and cached
under generated/hy4_preview_d_p/tt_cache/experts. The expert input stays bf16 (ffn_norm on layers 1-5 has no
outlier channels, max |x| <= 0.85, but bfp8 x buys nothing here). Per-expert cap = max chunk (a token picks an
expert once); expert_token_counts / region offsets are [1, 256] (global ids, known issues).

No host work in __call__: the dispatch / combine modules (sizes only) are built once per chunk length and reused.
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
CACHE_ROOT = REPO / "generated/hy4_preview_d_p/tt_cache"

DISPATCH_AXIS = 0  # dispatch along mesh rows (SP axis); each column is one dispatch group
TP_AXIS = 1  # the two dispatch groups (columns); experts_out is reduce-scattered over it
CAPACITY_FACTOR = 8  # flat dispatch buffer = 8 x per-expert cap (all top-8 experts of every token on one chip)
# Dispatch / combine hold no per-layer state: one pair per (mesh, chunk rows per chip) is shared by every MoE layer.
_SEQ_MODULES = {}


class LazyExpertWeights:
    """Sequence of {'gate_proj' [I, H], 'up_proj' [I, H], 'down_proj' [H, I]} (HF (out, in) layout, bf16, global
    expert order) read from the checkpoint's fused gate_up_proj / down_proj one expert at a time."""

    def __init__(self, loader, prefix: str, num_experts: int, inter: int):
        from models.demos.hy4_preview_d_p.reference.weights import ExpertSlab

        self.gate_up = ExpertSlab(loader, prefix + "gate_up_proj", torch.bfloat16)
        self.down = ExpertSlab(loader, prefix + "down_proj", torch.bfloat16)
        assert len(self.gate_up) == num_experts and self.gate_up.shape[1] == 2 * inter, self.gate_up.shape
        self.n, self.inter = num_experts, inter
        self._last = None

    def __len__(self):
        return self.n

    def __getitem__(self, e):
        e = int(e)
        if self._last is None or self._last[0] != e:  # gather_weights_for_mesh_distribution indexes 3x per expert
            gu = self.gate_up[e]
            self._last = (
                e,
                {"gate_proj": gu[: self.inter], "up_proj": gu[self.inter :], "down_proj": self.down[e]},
            )
        return self._last[1]


def _dispatch(m, x, indices, offsets, table):
    """TtDispatchModule.forward on the bring-up fork ttnn.bringup.dispatch (2 devices on the dispatch axis: fabric
    on, as the source op). Returns (buffer, metadata)."""
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
    """TtCombineModule.forward on the bring-up fork ttnn.bringup.combine."""
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


class TtHy4Experts:
    def __init__(
        self,
        mesh,
        layer: int,
        torch_weights,
        emb_dim: int,
        hidden_dim: int,
        top_k: int = 8,
        max_seq_len: int = 8192,
        swiglu_limit: float = 10.0,
        num_links: int = 1,
        weights_dtype=ttnn.bfloat8_b,
        cache: bool = True,
        mode: str = "unified",
        out_dtype=ttnn.float32,
    ):
        """torch_weights: sequence (len E) of {'gate_proj', 'up_proj', 'down_proj'} (HF layout), or None to load a
        complete cache. max_seq_len: the longest chunk (full S, before the row split)."""
        rows, cols = tuple(mesh.shape)
        n = rows * cols
        E = len(torch_weights) if torch_weights is not None else 256
        assert rows > 1 and cols > 1 and E % n == 0, f"2D mesh expected, got {tuple(mesh.shape)}"
        assert weights_dtype != ttnn.bfloat4_b, "owner rule: experts are bfp8 on device, never bfp4"
        assert mode in ("unified", "loop"), mode
        # The fused kernel bakes ClampedSiluGluConfigDsV4 (limit 10); the loop path takes any limit.
        assert mode == "loop" or swiglu_limit == 10.0, f"unified ClampedSiluGlu bakes limit 10, got {swiglu_limit}"
        self.mesh, self.layer, self.num_links, self.mode = mesh, layer, num_links, mode
        self.E, self.K, self.H, self.I = E, top_k, emb_dim, hidden_dim
        self.dgs, self.ngroups, self.n, self.epc = rows, cols, n, E // n
        self.max_seq_len, self.limit, self.out_dtype = max_seq_len, float(swiglu_limit), out_dtype

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
                    torch_weights = None  # every tensorbin exists: skip the checkpoint reads
            finally:
                fast_cache_checker._checker = None
        assert torch_weights is not None or cache, "no weights and no cache"
        # HiFi4 + fp32 dest for every expert matmul (owner rule; high_precision honours it for every activation).
        self.cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.routed = TtRoutedExpert(
            mesh_device=mesh,
            experts_per_chip=self.epc,
            global_expert_idx_table=gidx,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            max_tokens=max_seq_len,  # per-expert cap = dispatch group size x rows per chip = the full chunk
            torch_weights=torch_weights,
            activations_dtype=ttnn.bfloat8_b,  # unused: _unified_experts / _loop_experts call the ops directly
            weights_dtype=weights_dtype,
            weight_cache_path=cache_path,
            cache_name_prefix=prefix,
            compute_kernel_config=self.cfg,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
        )

    def setup(self, chunk: int):
        """Build (once) the dispatch / combine modules for a chunk of ``chunk`` rows (both mesh rows together)."""
        self._seq_modules(chunk // self.dgs)

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
                topology=ttnn.Topology.Linear,  # 2 chips on the dispatch axis; the fabric itself is FABRIC_2D
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

    def routed_half(self, x, idx, wts):
        """Group c's experts over this chip's row half. x [1,1,s,H] bf16 (s = S/2); idx / wts [1,1,s,K]
        -> [1,1,s,H] TILE bf16 (sum over the 128 experts of this chip's column)."""
        s, K = x.shape[-2], self.K
        dispatch, combine = self._seq_modules(s)
        idx2 = ttnn.reshape(ttnn.typecast(idx, ttnn.uint16) if idx.dtype != ttnn.uint16 else idx, (s, K))
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(idx2, self.dispatch_table, self.E, K)
        offsets, counts, region_offsets = ttnn.bringup.offset_cumsum(
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
        if self.mode == "unified":
            out = self._unified_experts(buf2, counts, region_offsets)
        else:
            out = self._loop_experts(buf2, counts, region_offsets)
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
        # idx2 / ind / scores / w5 are reshapes (views) of the caller's or each other's buffers: not freed here.
        for t in (hist, offsets, counts, region_offsets, meta):
            ttnn.deallocate(t)
        red = ttnn.to_layout(red, ttnn.TILE_LAYOUT)
        return ttnn.reshape(red, (1, 1, s, self.H))

    def _unified_experts(self, buf, counts, region_offsets):
        """One fused unified_routed_expert_moe over every local expert: ClampedSiluGlu (limit 10), high_precision
        (bf16 x, intermediates and output, fp32 partials, HiFi4 + fp32 dest honoured). buf is ROW_MAJOR bf16; returns a
        fresh TILE bf16 buffer."""
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
            activation=ttnn.bringup.RoutedExpertActivation.ClampedSiluGlu,
            high_precision=True,
        )

    def _ffn(self, x, wg, wu, wd):
        """down(silu(min(x @ wg, L)) * clamp(x @ wu, -L, L)) with ttnn.linear at cfg, fp32 intermediates."""
        cfg, mid, L = self.cfg, ttnn.float32, self.limit
        g = ttnn.linear(x, wg, compute_kernel_config=cfg, dtype=mid)
        u = ttnn.linear(x, wu, compute_kernel_config=cfg, dtype=mid)
        gc = ttnn.clamp(g, max=L)
        uc = ttnn.clamp(u, min=-L, max=L)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        h = ttnn.multiply(gc, uc, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=mid)
        ttnn.deallocate(gc)
        ttnn.deallocate(uc)
        y = ttnn.linear(h, wd, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        return y

    def _loop_experts(self, buf, counts, region_offsets):
        """Per local expert: extract (cap = per-expert cap), clamped SwiGLU via ttnn.linear, insert."""
        r = self.routed
        x = ttnn.to_layout(buf, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
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

    def __call__(self, x, idx, wts):
        """x: ffn_norm [1,1,S/2,H] per chip (rows split over axis 0, replicated over axis 1), bf16 or fp32 TILE;
        idx / wts: the router's [1,1,S/2,K] (same placement). Returns experts_out [1,1,S/2,H/2] (``out_dtype``,
        rows over axis 0, hidden columns over axis 1)."""
        xb = x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)
        part = self.routed_half(xb, idx, wts)
        if xb is not x:
            ttnn.deallocate(xb)
        if self.out_dtype != part.dtype:
            p2 = ttnn.typecast(part, self.out_dtype)
            ttnn.deallocate(part)
            part = p2
        # Add the other dispatch group's experts and keep this chip's hidden half.
        out = ttnn.reduce_scatter(part, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(part)
        return out
