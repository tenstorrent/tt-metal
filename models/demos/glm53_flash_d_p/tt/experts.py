# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 routed experts (288, top-8, clamped SwiGLU at 10, width 2048) on a 2x2 mesh: DeepSeek 2D EP.

    x [1,1,S,H], routing as dense [1,1,S,E] or (idx, wts) [1,1,S,K], replicated
      -> ttnn.mesh_partition(dim -2, cluster_axis 0): chip (r, c) keeps rows [r*S/2, (r+1)*S/2)
      -> (dense only) ttnn.topk on the row half -> (idx, wts) [1,1,S/2,8]
      -> masked_bincount + ttnn.bringup.offset_cumsum (cluster_axis 0)
      -> ttnn.bringup.dispatch (dispatch_group_size 2, cluster_axis 0, fabric): column c is a dispatch group holding
         experts 144c .. 144c+143; chip (r, c) holds experts 144c + 72r .. +71 (ExpertMapping col-major, 2c + r)
      -> mode 'unified' (default): ttnn.bringup.unified_routed_expert_moe (ClampedSiluGlu = silu(min(g, 10)) *
         clamp(u, +-10), high_precision=True, HiFi4 + fp32 dest: bf16 x / intermediates / output, fp32 partials);
         'loop': per local expert extract -> ttnn.linear gate / up (fp32) -> clamps -> silu * up -> down -> insert
      -> ttnn.bringup.combine (cluster_axis 0) -> post_combine_reduce (weighted top-k sum over group c's experts)
      -> ttnn.all_reduce(cluster_axis 1, fp32) -> ttnn.all_gather(dim -2, cluster_axis 0): experts_out [1,1,S,H] replicated

The routed scale (2.5) is already in the router weights. Counts / offsets are [1, 288] (global expert ids).
Weights: fp8 e4m3 x 128x128 block scale dequantized on the host (reference/weights.py) one expert at a time, bfp8 on
device (never bfp4), cached under generated/glm53_flash_d_p/tt_cache/experts.
Port of models/demos/mimo_v2_6_d_p_2x2/tt/experts.py:TtExperts.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.glm53_flash_d_p.reference.weights import PackedExpert
from models.demos.glm53_flash_d_p.tt.common import hifi4_config

REPO = Path(__file__).resolve().parents[4]
CACHE_ROOT = REPO / "generated/glm53_flash_d_p/tt_cache"

DISPATCH_AXIS = 0  # dispatch along mesh rows; each column is one dispatch group
CAPACITY_FACTOR = 8  # flat dispatch buffer = 8 x per-expert cap (all top-8 experts of every token on one chip)
_SEQ_MODULES = {}  # (mesh, dispatch, combine) per (E, K, H, rows per chip); no per-layer state


def _dispatch(m, x, indices, offsets, table):
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
    """Sequence of {'gate_proj', 'up_proj', 'down_proj'} (HF (out, in) layout, global order), each expert
    dequantized from fp8 when indexed."""

    def __init__(self, loader, layer: int, num_experts: int):
        self.loader, self.layer, self.n = loader, layer, num_experts

    def __len__(self):
        return self.n

    def __getitem__(self, e):
        g, u, d = PackedExpert(self.loader, self.layer, e).weights(torch.float32)
        return {"gate_proj": g, "up_proj": u, "down_proj": d}


class TtExperts:
    def __init__(
        self,
        mesh,
        layer: int,
        torch_weights,
        num_experts: int,
        emb_dim: int,
        hidden_dim: int,
        top_k: int = 8,
        limit: float = 10.0,
        max_seq_len: int = 8192,
        num_links: int = 1,
        weights_dtype=ttnn.bfloat8_b,
        cache: bool = True,
        mode: str = "unified",
    ):
        """torch_weights: sequence (len E) of HF expert weight dicts, or None to load a complete cache.
        max_seq_len: the longest chunk (full S, before the row split)."""
        rows, cols = tuple(mesh.shape)
        n = rows * cols
        E = num_experts
        assert rows > 1 and cols > 1 and E % n == 0, f"2D mesh expected, got {tuple(mesh.shape)}"
        assert weights_dtype != ttnn.bfloat4_b, "owner rule: experts are bfp8 on device, never bfp4"
        assert mode in ("unified", "loop"), mode
        # ClampedSiluGlu bakes the DeepSeek-V4 limit 10 into the kernel.
        assert mode != "unified" or float(limit) == 10.0, f"ClampedSiluGlu is fixed at limit 10, got {limit}"
        self.mesh, self.layer, self.num_links, self.mode = mesh, layer, num_links, mode
        self.E, self.K, self.H, self.I, self.limit = E, top_k, emb_dim, hidden_dim, float(limit)
        self.dgs, self.ngroups, self.n, self.epc = rows, cols, n, E // n
        self.max_seq_len = max_seq_len

        # (num_groups, E + 1): column c's experts -> their chip row, -1 elsewhere; per chip (1, E + 1).
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
                    torch_weights = None  # every tensorbin exists: skip the fp8 dequant
            finally:
                fast_cache_checker._checker = None
        assert torch_weights is not None or cache, "no weights and no cache"
        # HiFi4 + fp32 dest for every expert matmul (owner rule; high_precision honours it).
        self.cfg = hifi4_config()
        self.cfg_fused = ttnn.types.BlackholeComputeKernelConfig(
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
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype,
            weight_cache_path=cache_path,
            cache_name_prefix=prefix,
            compute_kernel_config=self.cfg_fused,
            activation=ttnn.RoutedExpertActivation.ClampedSiluGlu,
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
        """dense [1,1,s,E] (exactly K nonzeros per row) -> (idx [1,1,s,K], weights bf16 [1,1,s,K])."""
        if dense.dtype != ttnn.bfloat16:
            dense = ttnn.typecast(dense, ttnn.bfloat16)
        wts, idx = ttnn.topk(dense, k=self.K, dim=-1, largest=True, sorted=True)
        return idx, wts

    def routed_half(self, x, idx, wts):
        """Group c's experts over this chip's row half. x [1,1,s,H] bf16; idx / wts [1,1,s,K] -> [1,1,s,H] TILE."""
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
        else:
            out = self._unified_experts(buf2, counts, region_offsets)
        ttnn.deallocate(buf2)
        out = ttnn.unsqueeze(ttnn.unsqueeze(out, dim=0), dim=0)
        comb = _combine(combine, out, meta, counts, region_offsets, s)
        ttnn.deallocate(out)
        # Fused weighted top-k sum; slots whose expert is outside this chip's dispatch group are masked by the table.
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
        """Fused unified_routed_expert_moe over every local expert (ClampedSiluGlu, high_precision). buf ROW_MAJOR
        bf16; returns a fresh TILE bf16 buffer."""
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
            compute_kernel_config=self.cfg_fused,
            activation=ttnn.bringup.RoutedExpertActivation.ClampedSiluGlu,
            high_precision=True,
        )

    def _ffn(self, x, wg, wu, wd):
        """down(silu(min(x @ wg, L)) * clamp(x @ wu, +-L)), fp32 intermediates, HiFi4; x [M, H] TILE."""
        mc, cfg, lim = ttnn.DRAM_MEMORY_CONFIG, self.cfg, self.limit
        g = ttnn.linear(x, wg, compute_kernel_config=cfg, dtype=ttnn.float32, memory_config=mc)
        gc = ttnn.minimum(g, lim, memory_config=mc)
        ttnn.deallocate(g)
        a = ttnn.silu(gc, memory_config=mc)
        ttnn.deallocate(gc)
        u = ttnn.linear(x, wu, compute_kernel_config=cfg, dtype=ttnn.float32, memory_config=mc)
        uc = ttnn.clamp(u, min=-lim, max=lim, memory_config=mc)
        ttnn.deallocate(u)
        h = ttnn.multiply(a, uc, dtype=ttnn.float32, memory_config=mc)
        ttnn.deallocate(a)
        ttnn.deallocate(uc)
        y = ttnn.linear(h, wd, compute_kernel_config=cfg, dtype=ttnn.bfloat16, memory_config=mc)
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

    def __call__(self, x, dense=None, idx=None, wts=None, split=False):
        """x [1,1,S,H] replicated; routing as dense [1,1,S,E] or (idx, wts) [1,1,S,K], replicated
        -> experts_out [1,1,S,H] bf16 replicated.
        split: x and the routing are already mesh row r's half [r S/2, (r+1) S/2) (on both of its chips); the output is
        this chip's quarter [r S/2 + c S/4, +S/4) (the split residual layout: reduce_scatter on axis 1, no gather)."""
        tmp = []
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
            tmp.append(x)
        part_rows = (lambda t: t) if split else (lambda t: ttnn.mesh_partition(t, dim=-2, cluster_axis=DISPATCH_AXIS))
        xh = part_rows(x)
        if idx is None:
            dh = part_rows(dense)
            ih, wh = self.topk_from_dense(dh)
            tmp += [ih, wh] + ([] if split else [dh])
        else:
            ih, wh = part_rows(idx), part_rows(wts)
            tmp += [] if split else [ih, wh]
        if not split:
            tmp.append(xh)
        part = self.routed_half(xh, ih, wh)
        # Add the other dispatch group's experts in fp32 (a bf16 all_reduce biases the sum upwards, known issues).
        pf = ttnn.typecast(part, ttnn.float32)
        ttnn.deallocate(part)
        if split:
            sf = ttnn.reduce_scatter(pf, dim=-2, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pf)
            out = ttnn.typecast(sf, ttnn.bfloat16)
            ttnn.deallocate(sf)
            for t in tmp:
                ttnn.deallocate(t)
            return out
        sf = ttnn.all_reduce(pf, cluster_axis=1)
        ttnn.deallocate(pf)
        summed = ttnn.typecast(sf, ttnn.bfloat16)
        ttnn.deallocate(sf)
        out = ttnn.all_gather(summed, dim=-2, cluster_axis=DISPATCH_AXIS)
        ttnn.deallocate(summed)
        for t in tmp:
            ttnn.deallocate(t)
        return out


def build_experts(mesh, loader, cfg, layer: int, max_chunk: int) -> TtExperts:
    """TtExperts for one MoE layer (72 experts per chip, bfp8). GLM_EXPERTS_MODE=loop selects the per-expert path."""
    return TtExperts(
        mesh,
        layer,
        LazyExpertWeights(loader, layer, cfg.n_routed_experts),
        num_experts=cfg.n_routed_experts,
        emb_dim=cfg.hidden_size,
        hidden_dim=cfg.moe_intermediate_size,
        top_k=cfg.num_experts_per_tok,
        limit=cfg.swiglu_limit,
        max_seq_len=max_chunk,
        mode=os.environ.get("GLM_EXPERTS_MODE", "unified"),
    )
