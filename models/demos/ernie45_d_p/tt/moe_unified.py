# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ERNIE MoE on the DeepSeek EP pipeline with the fused ``unified_routed_expert_moe`` kernel.

    TtRouter (fp32 softmax, bias-corrected top-6, renorm) -> topk idx/weights [S, 6]
      -> masked_bincount + offset_cumsum -> TtDispatchModule -> TtRoutedExpert (unified_routed_expert_moe, Silu)
      -> TtCombineModule -> TtReduceModule (fused weighted top-k sum)
    + shared experts (TP4 partial) -> ONE all_reduce -> [1,1,S,H] replicated

Mesh (1,4): dispatch_group_size = 1 (mesh rows), num_dispatch_groups = 4 (mesh cols). Every chip holds all
S tokens (replicated residual) and dispatches them LOCALLY to its own 16 experts (16c..16c+15, same as tt/moe.py);
the weighted top-k sum yields this chip's partial [S, H] (its experts only), which is added to its shared-expert
TP partial and reduced with one all_reduce -- the same single CCL per MoE as the dense-EP path.

Needs three small ttnn patches for a 1-device dispatch axis (see BREADCRUMBS P3.2): offset_cumsum skips its
internal all_gather; dispatch and combine build their existing no-fabric kernel variants (no neighbour lookup).
Dispatch/combine buffers are sized per chunk length and built lazily; expert weights are shared.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants, get_ep_mesh_mapper
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.ernie45_d_p.reference.ernie_ref import ErnieConfig, LayerWeights
from models.demos.ernie45_d_p.tt.common import CACHE_ROOT, COMPUTE_HIFI2, signpost
from models.demos.ernie45_d_p.tt.moe import TtRouter
from models.demos.ernie45_d_p.tt.ops import TtSwiGLU

# Worst case all 6 of a token's experts live on one chip, so its flat dispatch buffer must hold 6*S rows.
DISPATCH_CAPACITY_FACTOR = 6
DGS = 1  # dispatch group = one chip (mesh rows)
# Dispatch/combine hold no per-layer state: one pair per (mesh, chunk length) is shared by all 27 MoE layers.
_SEQ_MODULES = {}


class TtMoEUnified:
    def __init__(
        self,
        mesh,
        cfg: ErnieConfig,
        layer: int,
        w: LayerWeights,
        max_seq_len: int = 8192,
        num_links: int = 1,
        weights_dtype=ttnn.bfloat16,
        activations_dtype=ttnn.bfloat8_b,
    ):
        self.mesh, self.cfg, self.layer, self.num_links = mesh, cfg, layer, num_links
        E, H, I = cfg.moe_num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        n = mesh.get_num_devices()
        assert tuple(mesh.shape) == (1, n)
        self.n, self.epc = n, E // n

        self.router = TtRouter(mesh, cfg, layer, w)
        self.shared = TtSwiGLU(mesh, w.w_gate, w.w_up, w.w_down, name=f"L{layer}/shared")
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
        self.routed = TtRoutedExpert(
            mesh_device=mesh,
            experts_per_chip=self.epc,
            global_expert_idx_table=gidx,
            emb_dim=H,
            hidden_dim=I,
            max_tokens=max_seq_len,  # per-expert cap = chunk length; the kernel sizes work from the live counts
            torch_weights=[
                {"gate_proj": w.e_gate[e], "up_proj": w.e_up[e], "down_proj": w.e_down[e]} for e in range(E)
            ],  # HF (out, in) layout, global expert order
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
            compute_kernel_config=COMPUTE_HIFI2,
            weight_cache_path=CACHE_ROOT / "unified_moe",
            cache_name_prefix=f"layer_{layer}.routed_expert.{weights_dtype.name}",
            activation=ttnn.RoutedExpertActivation.Silu,
        )
        # cluster_axis=0 has 1 device -> TtReduceModule skips its reduce-scatter: fused weighted top-k sum only
        # (a per-chip partial over local experts; the cross-chip sum is the all_reduce in __call__).
        self.reduce = TtReduceModule(
            mesh_device=mesh, topk_dim=3, cluster_axis=0, num_links=num_links, topology=ttnn.Topology.Linear
        )
        self.max_seq_len = max_seq_len

    def _seq_modules(self, S: int):
        """Dispatch/combine sized for S-token chunks, shared by every layer on this mesh."""
        key = (self.cfg.moe_num_experts, self.cfg.moe_k, self.cfg.hidden_size, S)
        hit = _SEQ_MODULES.get(key)
        if hit is None or hit[0] is not self.mesh:  # tests reopen the mesh: rebuild on a new one
            assert S <= self.max_seq_len, f"chunk {S} > max_seq_len {self.max_seq_len}"
            E, K, H = self.cfg.moe_num_experts, self.cfg.moe_k, self.cfg.hidden_size
            epc, metadata_len, max_buf, _ = compute_constants(S, E, K, self.n, DGS, DISPATCH_CAPACITY_FACTOR)
            dispatch = TtDispatchModule(
                mesh_device=self.mesh,
                dispatch_group_size=DGS,
                experts_per_chip=epc,
                num_routed_experts=E,
                num_experts_per_tok=K,
                metadata_len=metadata_len,
                max_dispatch_buffer_token_size=max_buf,
                seq_len_per_chip=S,
                emb_dim=H,
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
                num_experts_per_tok=K,
                seq_len_per_chip=S,
                cluster_axis=0,
                num_links=self.num_links,
                topology=ttnn.Topology.Linear,
                init_zeros=True,  # (token, slot) pairs routed to other chips' experts must read as 0
            )
            _SEQ_MODULES[key] = (self.mesh, dispatch, combine)
        return _SEQ_MODULES[key][1:]

    def _routing_setup(self, idx2):
        """TtMoERoutingSetup.forward (bincount mask = the sharded dispatch table, as in TtMoERoutingSetup)."""
        hist = ttnn.experimental.deepseek_prefill.masked_bincount(
            idx2, self.dispatch_table, self.cfg.moe_num_experts, self.cfg.moe_k
        )
        return ttnn.experimental.deepseek_prefill.offset_cumsum(
            hist,
            cluster_axis=0,
            num_links=self.num_links,
            experts_per_chip=self.epc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def routed_partial(self, x, idx, wts):
        """This chip's routed-expert partial. x [1,1,S,H]; idx/wts [1,1,S,K] (replicated) -> [1,1,S,H] TILE."""
        S, K = x.shape[-2], self.cfg.moe_k
        dispatch, combine = self._seq_modules(S)
        signpost("moe.dispatch")
        idx2 = ttnn.reshape(ttnn.typecast(idx, ttnn.uint16) if idx.dtype != ttnn.uint16 else idx, (S, K))
        offsets, counts, region_offsets = self._routing_setup(idx2)
        ind = ttnn.reshape(ttnn.to_layout(idx2, ttnn.ROW_MAJOR_LAYOUT), (1, S, K))
        scores = ttnn.reshape(ttnn.to_layout(ttnn.typecast(wts, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT), (1, S, K))
        buf, meta = dispatch(ttnn.squeeze(x, dim=0), scores, ind, offsets, self.dispatch_table)
        # ROW_MAJOR bf16 buffer -> the op's fused fast path (tilize + bf8 pack inside the kernel, fresh output).
        buf2 = ttnn.squeeze(ttnn.squeeze(buf, dim=0), dim=0)
        signpost("moe.experts")
        out = self.routed(buf2, counts, region_offsets)
        ttnn.deallocate(buf2)
        signpost("moe.combine_reduce")
        out = ttnn.unsqueeze(ttnn.unsqueeze(out, dim=0), dim=0)
        comb = combine(out, meta, counts, region_offsets, seq_len_per_chip=S)
        ttnn.deallocate(out)
        red = self.reduce(comb, weights=scores, indices=ind, expert_dispatch_table=self.dispatch_table)
        ttnn.deallocate(comb)
        red = ttnn.to_layout(red, ttnn.TILE_LAYOUT)
        return ttnn.reshape(red, (1, 1, S, self.cfg.hidden_size))

    def __call__(self, x, debug: dict | None = None):
        """x [1,1,S,H] replicated -> MoE output [1,1,S,H] replicated (routed + shared)."""
        signpost("moe.router")
        dense, idx, wts = self.router(x)
        ttnn.deallocate(dense)
        routed = self.routed_partial(x, idx, wts)
        signpost("moe.shared")
        shared = self.shared(x)
        tot = ttnn.add(routed, shared if shared.dtype == routed.dtype else ttnn.typecast(shared, routed.dtype))
        if debug is not None:
            debug.update(routed=ttnn.all_reduce(routed, cluster_axis=1), topk_idx=idx, topk_w=wts)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        signpost("moe.all_reduce")
        out = ttnn.all_reduce(tot, cluster_axis=1)
        ttnn.deallocate(tot)
        return out
