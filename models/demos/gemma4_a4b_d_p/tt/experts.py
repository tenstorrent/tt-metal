# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 routed experts on a 1x4 mesh: DeepSeek EP pipeline with the fused ``unified_routed_expert_moe`` kernel.

    dense routing [S, 128] (8 nonzeros per row) -> topk(8) on device -> idx [S, 8], weights [S, 8]
      -> masked_bincount + offset_cumsum -> TtDispatchModule (1-chip dispatch group, local dispatch)
      -> ttnn.bringup.unified_routed_expert_moe (fork, RoutedExpertActivation.GeluTanh: down(gelu_tanh(gate) * up))
      -> TtCombineModule -> TtReduceModule (fused weighted top-k sum over this chip's experts)
      -> all_reduce (cluster_axis=1) -> experts_out [1, 1, S, H] replicated

EP=4: chip c holds experts 32c..32c+31 (bf16 gate/up [704, 2816], down [2816, 704]). expert_token_counts and
expert_region_offsets are per chip [1, 128] (global expert ids, see known issues). The routed partial is all-reduced on
its own: post_feedforward_layernorm_2 comes next and is nonlinear, so it cannot be folded into another reduction.

Adapted from models/demos/ernie45_d_p/tt/moe_unified.py (routed_partial). Uses the bring-up forks (ttnn/ttnn/bringup):
offset_cumsum, dispatch and combine for the 1-device dispatch axis, unified_routed_expert_ffn for GeluTanh.
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
CACHE_ROOT = REPO / "generated/gemma4_a4b_d_p/tt_cache"

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


class TtExperts:
    def __init__(
        self,
        mesh,
        layer: int,
        gate_up: torch.Tensor,
        down: torch.Tensor,
        top_k: int = 8,
        max_seq_len: int = 8192,
        num_links: int = 1,
        weights_dtype=ttnn.bfloat16,
        cache: bool = True,
        math_fidelity=ttnn.MathFidelity.HiFi2,
    ):
        """HF weights: gate_up [E, 2I, H] (gate rows first), down [E, H, I]."""
        E, I2, H = gate_up.shape
        I = I2 // 2
        n = mesh.get_num_devices()
        assert tuple(mesh.shape) == (1, n) and E % n == 0
        self.mesh, self.layer, self.num_links = mesh, layer, num_links
        self.E, self.K, self.H, self.I = E, top_k, H, I
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
        torch_weights = [
            {"gate_proj": gate_up[e, :I], "up_proj": gate_up[e, I:], "down_proj": down[e]} for e in range(E)
        ]  # HF (out, in) layout, global expert order
        self.routed = TtRoutedExpert(
            mesh_device=mesh,
            experts_per_chip=self.epc,
            global_expert_idx_table=gidx,
            emb_dim=H,
            hidden_dim=I,
            max_tokens=max_seq_len,  # per-expert cap = chunk length (a token picks an expert at most once)
            torch_weights=torch_weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=weights_dtype,
            weight_cache_path=(CACHE_ROOT / "experts") if cache else None,
            cache_name_prefix=f"layer_{layer}.experts.{weights_dtype.name}" if cache else None,
            # The op honours math_fidelity and fp32_dest_acc_en for GeluTanh only (LoFi + bf16 dst otherwise).
            # Measured on expert 47 (one-hot): LoFi bf16-dst ratio 0.96, HiFi2 bf16-dst 1.036, HiFi2 fp32-dst 1.011.
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=math_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
            ),
            # TtRoutedExpert only holds the weights and tables here: its forward calls the original op, which has no
            # GeluTanh; _routed calls the fork ttnn.bringup.unified_routed_expert_moe with it.
            activation=ttnn.RoutedExpertActivation.Silu,
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

    def _routed(self, buf, counts, region_offsets):
        """TtRoutedExpert.forward (Blackhole, ROW_MAJOR bf16 buffer) on the fork with GeluTanh."""
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
            compute_kernel_config=r.compute_kernel_config,
            activation=ttnn.bringup.RoutedExpertActivation.GeluTanh,
        )

    def topk_from_dense(self, dense):
        """dense [1,1,S,E] bf16 (exactly K nonzeros per row) -> (idx uint16 [1,1,S,K], weights bf16 [1,1,S,K])."""
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
        offsets, counts, region_offsets = ttnn.bringup.offset_cumsum(
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
        # ROW_MAJOR bf16 buffer -> the op's fused path (tilize + bf8 pack inside the kernel, fresh output).
        buf2 = ttnn.squeeze(ttnn.squeeze(buf, dim=0), dim=0)
        out = self._routed(buf2, counts, region_offsets)
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

    def __call__(self, x, dense=None, idx=None, wts=None):
        """x [1,1,S,H] replicated; routing as dense [1,1,S,E] or (idx, wts) -> experts_out [1,1,S,H] replicated."""
        if idx is None:
            idx, wts = self.topk_from_dense(dense)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        part = self.routed_partial(x, idx, wts)
        out = ttnn.all_reduce(part, cluster_axis=1)
        ttnn.deallocate(part)
        return out
