# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 routed experts (HF Xing4_0MoE: 64 experts, top-4, SwiGLU 1024, unclamped) on the 4x2 mesh, per plan.md.

DeepSeek 2D expert parallelism (deepseek_v3_d_p/tt/moe/README.md: dispatch groups = mesh columns, dispatch along the
rows), the pipeline of hy4_preview_d_p/tt/experts.py:TtHy4Experts with 4 chips per group:

    x [1,1,S/4,H] bf16 (ffn_norm: rows split over axis 0, replicated over axis 1), (idx, wts) [1,1,S/4,4] (router)
      -> masked_bincount + ttnn.bringup.offset_cumsum (cluster_axis 0: the 4 row quarters of a column)
      -> ttnn.bringup.dispatch (dispatch_group_size 4, cluster_axis 0, fabric on the FABRIC_2D mesh): column c is a
         dispatch group holding experts 32c .. 32c+31; chip (r, c) holds experts 32c + 8r .. +7
         (ExpertMapping column-major)
      -> mode 'unified' (default): ttnn.bringup.unified_routed_expert_moe, Silu (silu(g) * u), high_precision=True,
         HiFi4 + fp32 dest (bf16 x / intermediates / output, fp32 partials); 'loop' (XING_EXPERTS_MODE=loop): per local
         expert extract -> ttnn.linear gate / up (fp32) -> multiply (SILU) -> ttnn.linear down -> insert
      -> ttnn.bringup.combine (cluster_axis 0) -> post_combine_reduce (weighted top-k sum over column c's experts; the
         dispatch table masks the other group's slots) -> [1,1,S/4,H] partial (group c only)
      -> typecast fp32 -> ttnn.reduce_scatter(dim 3, cluster_axis 1): add the other group's partial, keep this chip's
         hidden half -> experts_out [1,1,S/4,H/2] fp32 (the residual's column split, like tt/mlp.py)

Weights: bf16 in the checkpoint, bfp8 on device (never bfp4), read one expert at a time and cached under
generated/xing40_a4b_d_p/tt_cache/experts/<rows>x<cols> (each tensorbin stacks one local slot over the whole mesh, so a
cache is only valid for the mesh shape that wrote it). Per-expert cap = the chunk (a token picks an expert once); the flat
dispatch buffer has capacity factor 4 (all 4 of a token's experts on one chip). expert_token_counts / region offsets
are [1, 64] (global ids, known issues).

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
from models.demos.xing40_a4b_d_p.tt.settings import settings

REPO = Path(__file__).resolve().parents[4]
CACHE_ROOT = REPO / "generated/xing40_a4b_d_p/tt_cache"

DISPATCH_AXIS = 0  # dispatch along mesh rows (SP axis); each column is one dispatch group
TP_AXIS = 1  # the two dispatch groups (columns); experts_out is reduce-scattered over it
CAPACITY_FACTOR = 4  # flat dispatch buffer = 4 x per-expert cap (all top-4 experts of every token on one chip)
# Dispatch / combine hold no per-layer state: one pair per (mesh, rows per chip) is shared by every MoE layer.
_SEQ_MODULES = {}


def experts_fidelity():
    """Routed-expert matmul fidelity: HiFi2 by owner decision (2026-10-01, after P.3's A/B: experts 256.6 -> 242.4 ms,
    s56320 top5 1.0, final hidden 0.99803 with the bfp8 KV cache); XING_EXPERTS_FIDELITY=hifi4 for the previous path."""
    f = settings.get("EXPERTS_FIDELITY")
    assert f in ("hifi2", "hifi4"), f"XING_EXPERTS_FIDELITY={f!r}, expected hifi2 or hifi4"
    return ttnn.MathFidelity.HiFi2 if f == "hifi2" else ttnn.MathFidelity.HiFi4


class LazyExpertWeights:
    """Sequence of {'gate_proj' [I, H], 'up_proj' [I, H], 'down_proj' [H, I]} (HF (out, in) layout, bf16, global
    expert order) read from the checkpoint one expert at a time."""

    def __init__(self, loader, layer: int, num_experts: int):
        self.loader, self.p, self.n = loader, f"model.layers.{layer}.mlp.experts.", num_experts
        self._last = None

    def __len__(self):
        return self.n

    def __getitem__(self, e):
        e = int(e)
        if self._last is None or self._last[0] != e:  # gather_weights_for_mesh_distribution indexes 3x per expert
            self._last = (
                e,
                {
                    m: self.loader.get(f"{self.p}{e}.{m}.weight").to(torch.bfloat16)
                    for m in ("gate_proj", "up_proj", "down_proj")
                },
            )
        return self._last[1]


def _dispatch(m, x, indices, offsets, table):
    """TtDispatchModule.forward on the bring-up fork ttnn.bringup.dispatch. Returns (buffer, metadata)."""
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


class TtExperts:
    def __init__(
        self,
        mesh,
        layer: int,
        torch_weights,
        num_experts: int,
        emb_dim: int,
        hidden_dim: int,
        top_k: int = 4,
        max_seq_len: int = 8192,
        num_links: int = 1,
        weights_dtype=ttnn.bfloat8_b,
        cache: bool = True,
        mode: str = "unified",
        out_dtype=ttnn.float32,
    ):
        """torch_weights: sequence (len E) of {'gate_proj', 'up_proj', 'down_proj'} (HF layout). max_seq_len: the
        longest chunk (full S, before the row split)."""
        rows, cols = tuple(mesh.shape)
        n = rows * cols
        E = num_experts
        assert rows > 1 and cols > 1 and E % n == 0, f"2D mesh expected, got {tuple(mesh.shape)}"
        assert weights_dtype != ttnn.bfloat4_b, "owner rule: bf16 checkpoint experts are never bfp4"
        assert mode in ("unified", "loop"), mode
        self.mesh, self.layer, self.num_links, self.mode = mesh, layer, num_links, mode
        self.E, self.K, self.H, self.I = E, top_k, emb_dim, hidden_dim
        self.dgs, self.ngroups, self.n, self.epc = rows, cols, n, E // n
        self.max_seq_len, self.out_dtype = max_seq_len, out_dtype

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
        cache_path = CACHE_ROOT / "experts" / f"{rows}x{cols}" if cache else None
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
        # HiFi4 + fp32 dest for every expert matmul (owner rule; high_precision honours it).
        self.cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=experts_fidelity(), math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
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
            activation=ttnn.RoutedExpertActivation.Silu,
        )

    def setup(self, chunk: int):
        """Build (once) the dispatch / combine modules for a chunk of ``chunk`` rows (all mesh rows together)."""
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
                topology=ttnn.Topology.Linear,  # 4 chips on the dispatch axis (no torus); the fabric is FABRIC_2D
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

    def routed_part(self, x, idx, wts):
        """Group c's experts over this chip's row quarter. x [1,1,s,H] bf16 (s = S/4); idx / wts [1,1,s,K]
        -> [1,1,s,H] TILE (sum over the 32 experts of this chip's column)."""
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
        for t in (hist, offsets, counts, region_offsets, meta):
            ttnn.deallocate(t)
        if wb is not wts:
            ttnn.deallocate(wb)
        # to_layout may hand back the same buffer (post_combine_reduce already writes TILE): not freed here.
        red = ttnn.to_layout(red, ttnn.TILE_LAYOUT)
        return ttnn.reshape(red, (1, 1, s, self.H))

    def _unified_experts(self, buf, counts, region_offsets):
        """One fused unified_routed_expert_moe over every local expert: Silu SwiGLU, high_precision (bf16 x,
        intermediates and output, fp32 partials, HiFi4 + fp32 dest honoured). buf is ROW_MAJOR bf16."""
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
        """down(silu(x @ wg) * (x @ wu)) with ttnn.linear at cfg, fp32 intermediates."""
        cfg, mid = self.cfg, ttnn.float32
        g = ttnn.linear(x, wg, compute_kernel_config=cfg, dtype=mid)
        u = ttnn.linear(x, wu, compute_kernel_config=cfg, dtype=mid)
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=mid)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        y = ttnn.linear(h, wd, compute_kernel_config=cfg, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        return y

    def _loop_experts(self, buf, counts, region_offsets):
        """Per local expert: extract (cap = per-expert cap), SwiGLU via ttnn.linear, insert."""
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
        """x: ffn_norm [1,1,S/4,H] per chip (rows split over axis 0, replicated over axis 1), bf16 or fp32 TILE;
        idx / wts: the router's [1,1,S/4,K] (same placement). Returns experts_out [1,1,S/4,H/2] (``out_dtype``, rows
        over axis 0, hidden columns over axis 1)."""
        xb = x if x.dtype == ttnn.bfloat16 else ttnn.typecast(x, ttnn.bfloat16)
        part = self.routed_part(xb, idx, wts)
        if xb is not x:
            ttnn.deallocate(xb)
        if self.out_dtype != part.dtype:
            p2 = ttnn.typecast(part, self.out_dtype)
            ttnn.deallocate(part)
            part = p2
        # Add the other dispatch group's experts (fp32) and keep this chip's hidden half.
        out = ttnn.reduce_scatter(part, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(part)
        return out


def build_experts(mesh, loader, cfg, layer: int, max_chunk: int) -> TtExperts:
    """TtExperts for one MoE layer (8 experts per chip, bfp8). XING_EXPERTS_MODE=loop selects the per-expert path."""
    return TtExperts(
        mesh,
        layer,
        LazyExpertWeights(loader, layer, cfg.n_routed_experts),
        num_experts=cfg.n_routed_experts,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        top_k=cfg.num_experts_per_tok,
        max_seq_len=max_chunk,
        mode=settings.get("EXPERTS_MODE"),
    )
