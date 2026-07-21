# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Expert-parallel MoE for the HunyuanImage-3.0 backbone. The 64 experts are sharded
# across the mesh so each device holds 64/ndev experts RESIDENT in bf8 (experts are
# ~97% of per-layer weight memory; 16 experts/device ~= 19GB is the only layout that
# fits the 80GB bf8 model — see MEMORY_FIT_PLAN.md).
#
# SPARSE by default. There is ONE resident copy of the expert weights (split
# gate/up/down, DeepSeek-V3 d_p col-major placement), consumed two ways:
#
#   * PREFILL (large M): SP-dispatch — route each token only to its top-k experts
#     (routing_setup -> dispatch -> fused unified_routed_expert_moe -> combine ->
#     reduce), so an expert computes ONLY its routed tokens (~8x less FLOP than the
#     old dense-over-all-tokens loop). ~10x on the expert compute; PCC 0.999 @ CF=3.
#   * DECODE / ineligible shapes (M~1): a plain dense loop over the SAME per-expert
#     weights + expert all-reduce. At tiny M there is no compute to save, and the
#     dispatch/combine fabric round-trips would only add latency — so decode skips
#     them. Both paths read the identical resident weights: no double residency.
#
# Expert weight disk cache (same technique as DeepSeek-V3 d_p / MiniMax-M3):
# ``TtRoutedExpert`` writes/loads per-local-expert ``.tensorbin`` files under
# ``weight_cache_path`` (from ``TT_DIT_CACHE_DIR``). First process converts host
# torch -> device and populates the cache; later processes skip the host stack /
# transpose when the cache is complete and load tilized tensors directly.
#
# See SPARSE_MOE_PLAN.md and tests/{pcc/test_moe_sparse.py, perf/test_sparse_moe_*.py}.

import os
from pathlib import Path

import torch
import ttnn
from loguru import logger
from models.common.lightweightmodule import LightweightModule

from .gate import HunyuanTtTopKGate
from .mlp import HunyuanTtMLP
from ..parallel_utils import (
    decode_mm_program_config,
    moe_full_seq_mem_config,
    resid_mem_config,
    sp_gather,
    sp_shard,
    wide_mm_program_config,
)


class HunyuanTtMoEParallel(LightweightModule):
    def __init__(
        self,
        mesh_device,
        ccl_manager,
        state_dict,
        prefix,
        *,
        num_experts: int,
        hidden_size: int,
        moe_topk: int,
        num_shared_expert: int = 1,
        norm_topk_prob: bool = True,
        use_mixed_mlp_moe: bool = True,
        mesh_axis: int = 0,
        weight_dtype=ttnn.bfloat8_b,
        gate_dtype=ttnn.bfloat8_b,
        sp_axis: int = 0,
        sp_factor: int = 1,
        weight_cache_path=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.ccl = ccl_manager
        self.mesh_axis = mesh_axis
        self.sp_axis = sp_axis
        self.sp_factor = sp_factor
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.norm_topk_prob = norm_topk_prob
        self.weight_dtype = weight_dtype
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        # Capacity factor for the per-chip dispatch buffer (prefill). CF=3 covers the
        # summed tile-aligned per-expert counts under routing skew (CF=2 overflows and
        # drops tokens -> PCC 0.973 vs 0.999; measured in test_sparse_moe_e2e.py).
        self.sparse_cf = int(os.environ.get("HY_SPARSE_MOE_CF", "3"))

        ndev = mesh_device.get_num_devices()
        assert num_experts % ndev == 0, f"{num_experts} experts not divisible by mesh device count {ndev}"
        self.ndev = ndev
        self.experts_per_dev = num_experts // ndev
        mesh_shape = tuple(mesh_device.shape)
        # Experts are split across every non-trivial mesh axis (disjoint 16/device on a
        # 2x2 or 1x4); the decode dense-loop partial is summed over these axes.
        self.ep_reduce_axes = [a for a, n in enumerate(mesh_shape) if n > 1]

        # --- single resident expert-weight copy (split gate|up|down). Host torch
        # list is only built on a cache miss; on a hit TtRoutedExpert loads .tensorbin
        # directly (DeepSeek/MiniMax cache-only path). Hunyuan stores a FUSED
        # gate_and_up_proj [2I,H]; split into up|gate halves in HF (out,in) layout.
        # SwiGLU is silu(x2)*x1 with x1=first half => up=FIRST, gate=SECOND —
        # matches RoutedExpertActivation.Silu = silu(gate)*up.
        self._prefix = prefix
        self._cache_name_prefix = f"{prefix}.routed_expert"
        self._weight_cache_path = Path(weight_cache_path) if weight_cache_path is not None else None
        self._inter = self._resolve_routed_intermediate(state_dict, prefix)
        self._expert_torch_weights = self._load_or_skip_host_experts(state_dict, prefix, num_experts)
        self.routed_expert = None  # TtRoutedExpert weight holder (built lazily, once)
        self.dec_ids_experts = None  # per-local-expert global id (bf16) for the decode mask
        self._module_cache = {}  # prefill seq_len -> dispatch/combine/reduce modules

        # Gate + shared MLP run replicated (input is replicated). Gate signature is
        # positional: (device, num_experts, moe_topk, state_dict, weight_key, ...).
        self.gate = HunyuanTtTopKGate(
            mesh_device,
            hidden_size,
            num_experts,
            moe_topk,
            state_dict,
            f"{prefix}.gate.wg",
            norm_topk_prob=norm_topk_prob,
            weight_dtype=gate_dtype,
            weight_cache_path=weight_cache_path,
        )
        self.shared_mlp = None
        if use_mixed_mlp_moe:
            self.shared_mlp = HunyuanTtMLP(
                mesh_device,
                hidden_size,
                state_dict,
                f"{prefix}.shared_mlp",
                weight_dtype=weight_dtype,
                weight_cache_path=weight_cache_path,
            )

        # Expert matmuls are compute-bound on low-precision weights; HiFi2 (2 passes) +
        # bf16 accumulation is PCC-safe and ~2x faster than HiFi4. Used by the decode
        # dense loop (the fused prefill op carries its own kernel config).
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    # ---------------------------------------------------------------- weight residency
    @staticmethod
    def _resolve_routed_intermediate(state_dict, prefix: str) -> int:
        """Routed intermediate I from fused gate_and_up [2I, H] (or raise if absent)."""
        gu = state_dict.get(f"{prefix}.experts.0.gate_and_up_proj.weight")
        if gu is None:
            raise KeyError(
                f"Missing {prefix}.experts.0.gate_and_up_proj.weight — needed for routed "
                f"intermediate dim (even on a TTNN cache hit, so empty state_dicts must "
                f"still supply expert.0 shapes or a future dim sidecar)."
            )
        return gu.shape[0] // 2

    def _expert_cache_complete(self) -> bool:
        """True when every local expert gate/up/down .tensorbin exists under the cache dir."""
        if self._weight_cache_path is None:
            return False
        from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker

        self._weight_cache_path.mkdir(parents=True, exist_ok=True)
        init_checker(self._weight_cache_path)
        return TtRoutedExpert.check_cache_complete(
            self._weight_cache_path, self._cache_name_prefix, self.experts_per_dev
        )

    def _load_or_skip_host_experts(self, state_dict, prefix: str, num_experts: int):
        """Build the host torch_weights list, or None when the TTNN expert cache is complete.

        Mirrors DeepSeek/MiniMax: on a cache hit, skip stacking/transposing all experts on
        host — ``TtRoutedExpert`` loads tilized tensors from ``.tensorbin`` with
        ``torch_weights=None``.
        """
        if self._expert_cache_complete():
            verbose = os.environ.get("HY_VERBOSE", "1") != "0"
            if verbose:
                logger.info(
                    f"[moe] {self._cache_name_prefix}: TTNN expert cache hit "
                    f"({self.experts_per_dev} local x gate/up/down) — skipping host stack"
                )
            return None

        weights = []
        for e in range(num_experts):
            gu = state_dict[f"{prefix}.experts.{e}.gate_and_up_proj.weight"]  # [2I, H]
            half = gu.shape[0] // 2
            weights.append(
                {
                    "up_proj": gu[:half].contiguous(),
                    "gate_proj": gu[half:].contiguous(),
                    "down_proj": state_dict[f"{prefix}.experts.{e}.down_proj.weight"].contiguous(),  # [H, I]
                }
            )
        return weights

    def _ensure_experts(self):
        """Build the single resident expert-weight copy (once). TtRoutedExpert loads the
        split weights in DeepSeek col-major placement (from host or .tensorbin cache) and
        exposes gate_projs/up_projs/down_projs that BOTH the fused prefill op and the
        decode loop consume."""
        if self.routed_expert is not None:
            return
        from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, get_ep_mesh_mapper
        from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

        md = self.mesh_device
        dgs = md.shape[self.sp_axis]
        ndg = md.shape[1 - self.sp_axis]
        epc = self.experts_per_dev

        gidx = ExpertMapping.create_global_expert_idx_table(epc, dgs, ndg)  # [ndg, dgs, epc]
        gidx_tt = ttnn.from_torch(
            gidx, mesh_mapper=get_ep_mesh_mapper(md), layout=ttnn.ROW_MAJOR_LAYOUT, device=md, dtype=ttnn.uint32
        )
        gidx_tt = ttnn.squeeze(ttnn.squeeze(gidx_tt, 0), 0)  # per-device [epc]

        # Same col-major placement as gidx, as a bf16 [epc,1] for the decode eq-mask.
        dec = torch.as_tensor(gidx).reshape(ndg, dgs, epc, 1).float()
        dec_tt = ttnn.from_torch(
            dec, mesh_mapper=get_ep_mesh_mapper(md), layout=ttnn.TILE_LAYOUT, device=md, dtype=ttnn.bfloat16
        )
        dec_tt = ttnn.squeeze(ttnn.squeeze(dec_tt, 0), 0)  # per-device [epc,1]
        self.dec_ids_experts = [dec_tt[el] for el in range(epc)]

        if self._weight_cache_path is not None:
            self._weight_cache_path.mkdir(parents=True, exist_ok=True)
            mode = "cache-load" if self._expert_torch_weights is None else "convert+cache"
            logger.info(
                f"[moe] {self._cache_name_prefix}: building TtRoutedExpert ({mode}) -> {self._weight_cache_path}"
            )

        # max_tokens is a fused-op runtime arg (per-prefill); set it per call in
        # _forward_prefill. Weights do not depend on it, so any positive placeholder is fine.
        # weight_cache_path + cache_name_prefix: same as DeepSeek TtMoe / MiniMax TtMiniMaxMoE —
        # first run writes .tensorbin; later runs load them (torch_weights may be None).
        self.routed_expert = TtRoutedExpert(
            mesh_device=md,
            experts_per_chip=epc,
            global_expert_idx_table=gidx_tt,
            emb_dim=self.hidden_size,
            hidden_dim=self._inter,
            max_tokens=1,
            torch_weights=self._expert_torch_weights,
            activations_dtype=ttnn.bfloat8_b,
            weights_dtype=self.weight_dtype,
            weight_cache_path=self._weight_cache_path,
            cache_name_prefix=self._cache_name_prefix,
            activation=ttnn.RoutedExpertActivation.Silu,
        )
        self._expert_torch_weights = None  # free host copy (if any)

    def _num_links(self):
        from models.common.modules.tt_ccl import get_num_links

        return max(1, get_num_links(self.mesh_device))

    def _prefill_eligible(self, x: ttnn.Tensor) -> bool:
        """SP-dispatch prefill needs a replicated full sequence we can shard across the SP
        mesh axis into tile-aligned per-chip shards. Decode (S=1) and non-tile-aligned
        shards fall back to the dense loop."""
        dgs = self.mesh_device.shape[self.sp_axis]
        s = x.shape[1]
        return self.sp_factor == 1 and x.shape[0] == 1 and dgs > 1 and s % dgs == 0 and (s // dgs) % 64 == 0

    def _ep_all_reduce(self, partial: ttnn.Tensor) -> ttnn.Tensor:
        """Sum a [B,S,H] per-device partial over every non-trivial mesh axis. Each device
        computed the partial over ITS disjoint local experts; reducing over the shard axes
        yields the full combined output on every device (all-gather(dim=0)+sum per axis)."""
        out = partial
        for axis in self.ep_reduce_axes:
            n = self.mesh_device.shape[axis]
            gathered = self.ccl.all_gather(out, dim=0, mesh_axis=axis, use_hyperparams=True)  # [n*B,S,H]
            ttnn.deallocate(out)
            B = gathered.shape[0] // n
            S, H = gathered.shape[1], gathered.shape[2]
            gathered = ttnn.reshape(gathered, (n, B, S, H))
            out = ttnn.sum(gathered, dim=0, memory_config=moe_full_seq_mem_config(S))  # [B,S,H]
            ttnn.deallocate(gathered)
        return out

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        self._ensure_experts()
        if self._prefill_eligible(x):
            return self._forward_prefill(x)
        return self._forward_decode(x)

    # -------------------------------------------------------------------- prefill (sparse)
    def _build_modules(self, seq_len: int):
        """Lazily build (cache by seq_len) the seq-dependent EP dispatch/combine/reduce
        modules + capacity constants. Weights are NOT here — they live in self.routed_expert."""
        if seq_len in self._module_cache:
            return self._module_cache[seq_len]
        from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping, compute_constants
        from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
        from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
        from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
        from models.demos.deepseek_v3_d_p.tt.moe.tt_reduce import TtReduceModule

        md = self.mesh_device
        nl = self._num_links()
        dgs = md.shape[self.sp_axis]  # SP / dispatch-group axis
        ep_axis = 1 - self.sp_axis  # EP-group / TP axis
        ndg = md.shape[ep_axis]
        spc = seq_len // dgs
        epc, metadata_len, max_buf, max_tok = compute_constants(
            spc, self.num_experts, self.moe_topk, md.get_num_devices(), dgs, self.sparse_cf
        )

        edt = ExpertMapping.create_dispatch_table(self.num_experts, dgs, ndg)
        mods = {
            "dgs": dgs,
            "ep_axis": ep_axis,
            "spc": spc,
            "max_tok": max_tok,
            "tt_edt": TtDispatchModule.shard_expert_dispatch_table(md, edt, self.sp_axis),
            "routing_setup": TtMoERoutingSetup(md, edt, num_links=nl, experts_per_chip=epc),
            "dispatch": TtDispatchModule(
                md,
                dgs,
                epc,
                self.num_experts,
                self.moe_topk,
                metadata_len,
                max_buf,
                spc,
                emb_dim=self.hidden_size,
                cluster_axis=self.sp_axis,
                num_links=nl,
                topology=ttnn.Topology.Linear,
            ),
            "combine": TtCombineModule(
                md,
                dgs,
                ndg,
                epc,
                self.moe_topk,
                spc,
                cluster_axis=self.sp_axis,
                num_links=nl,
                topology=ttnn.Topology.Linear,
                init_zeros=True,
            ),
            "reduce": TtReduceModule(md, topk_dim=3, cluster_axis=ep_axis, num_links=nl, topology=ttnn.Topology.Linear),
        }
        self._module_cache[seq_len] = mods
        return mods

    def _forward_prefill(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """SP-dispatch MoE: shard the replicated sequence across the SP axis, route each
        token only to its top-k experts, run the fused sparse expert on the routed tokens,
        combine + weighted-reduce, regather emb, add the (SP-sharded) shared MLP, and
        regather the sequence to the replicated [1,S,H] the layer residual expects."""
        md = self.mesh_device
        m = self._build_modules(x.shape[1])
        dgs, ep_axis, spc = m["dgs"], m["ep_axis"], m["spc"]

        # replicated [1,S,H] -> per-device row shard [1, S/dgs, H]. DRAM: the dispatch /
        # fused-expert chain requires DRAM-interleaved tensors (fused op TT_FATALs on L1).
        xs = sp_shard(self.ccl, x, dim=1, mesh_axis=self.sp_axis, n=dgs, out_memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # router on the sharded tokens; slice off the gate's tile-pad (sentinel id = E).
        topk_w, topk_idx = self.gate(xs)
        idx_rm = ttnn.to_layout(topk_idx, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(topk_idx)
        w_rm = ttnn.to_layout(topk_w, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(topk_w)
        idx_rm = idx_rm[:, :, : self.moe_topk]
        w_rm = w_rm[:, :, : self.moe_topk]
        if idx_rm.get_dtype() != ttnn.uint16:
            idx_rm = ttnn.typecast(idx_rm, ttnn.uint16)
        if w_rm.get_dtype() != ttnn.bfloat16:
            w_rm = ttnn.typecast(w_rm, ttnn.bfloat16)
        idx3 = ttnn.reshape(idx_rm, (1, spc, self.moe_topk))
        scores3 = ttnn.reshape(w_rm, (1, spc, self.moe_topk))
        idx2 = ttnn.reshape(idx_rm, (spc, self.moe_topk))

        offsets, counts, region_offsets, _ = m["routing_setup"](
            ttnn_top_k_experts_indices=idx2,
            num_routed_experts=self.num_experts,
            seq_len_per_chip=spc,
            num_experts_per_tok=self.moe_topk,
        )

        buf, meta = m["dispatch"](xs, scores3, idx3, offsets, m["tt_edt"])
        buf_tiled = ttnn.to_layout(ttnn.squeeze(ttnn.squeeze(buf, 0), 0), ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(buf)
        if buf_tiled.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            buf_tiled = ttnn.to_memory_config(buf_tiled, ttnn.DRAM_MEMORY_CONFIG)  # fused op requires DRAM-interleaved
        self.routed_expert.max_tokens = m["max_tok"]  # per-prefill capacity for the fused op
        eo = self.routed_expert(buf_tiled, counts, region_offsets)
        eo = ttnn.unsqueeze(ttnn.unsqueeze(eo, 0), 0)
        comb = m["combine"](eo, meta, counts, region_offsets)
        routed = m["reduce"](comb, weights=scores3, indices=idx3, expert_dispatch_table=m["tt_edt"])

        # reduce reduce-scatters emb across the EP/TP axis -> regather to full emb
        if md.shape[ep_axis] > 1:
            routed = self.ccl.all_gather(routed, dim=-1, mesh_axis=ep_axis, use_hyperparams=False)
        routed = ttnn.reshape(routed, (1, spc, self.hidden_size))

        # shared MLP on the SAME sharded tokens (halves its M), added here
        if self.shared_mlp is not None:
            shared = self.shared_mlp(xs)
            shared = ttnn.reshape(shared, (1, spc, self.hidden_size))
            out_sharded = ttnn.add(shared, routed, memory_config=moe_full_seq_mem_config(spc))
            ttnn.deallocate(shared)
            ttnn.deallocate(routed)
        else:
            out_sharded = routed
        ttnn.deallocate(xs)

        # sequence-sharded -> replicated full [1,S,H] for the layer residual add
        out = sp_gather(
            self.ccl,
            out_sharded,
            dim=1,
            mesh_axis=self.sp_axis,
            n=dgs,
            out_memory_config=moe_full_seq_mem_config(x.shape[1]),
        )
        ttnn.deallocate(out_sharded)
        return out

    # ---------------------------------------------------------------------- decode (dense)
    def _expert_dense(self, x, el):
        """Run local expert `el` on all tokens using the resident SPLIT weights
        (silu(gate)*up @ down). Used for decode / small-M where dispatch overhead would
        dominate — same weights as the fused prefill op."""
        wg = self.routed_expert.gate_projs[el]  # [H, I]
        wu = self.routed_expert.up_projs[el]  # [H, I]
        wd = self.routed_expert.down_projs[el]  # [I, H]
        mc = moe_full_seq_mem_config(x.shape[1])
        Mt = (x.shape[-2] + 31) // 32

        def pc(M, K, N):
            return (
                wide_mm_program_config(self.mesh_device, M, K, N)
                if Mt >= 8
                else decode_mm_program_config(self.mesh_device, M, K, N)
            )

        g = ttnn.linear(
            x,
            wg,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mc,
            program_config=pc(x.shape[-2], x.shape[-1], wg.shape[-1]),
        )
        u = ttnn.linear(
            x,
            wu,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mc,
            program_config=pc(x.shape[-2], x.shape[-1], wu.shape[-1]),
        )
        # silu(gate) * up (gate=g from the second fused half, up=u from the first).
        h = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=mc)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(
            h,
            wd,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=mc,
            program_config=pc(h.shape[-2], h.shape[-1], wd.shape[-1]),
        )
        ttnn.deallocate(h)
        return out

    def _forward_decode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Dense loop over the resident experts + expert all-reduce. For decode (M~1) and
        any shape the SP-dispatch path is not eligible for. Handles sp>1 by gathering to
        the full sequence and resharding the result (the EP all-reduce needs replicated
        tokens), mirroring the previous dense path."""
        sp = self.sp_factor > 1
        global_seq = x.shape[1] * self.sp_factor if sp else x.shape[1]
        xf = (
            sp_gather(
                self.ccl,
                x,
                dim=1,
                mesh_axis=self.sp_axis,
                n=self.sp_factor,
                out_memory_config=moe_full_seq_mem_config(global_seq),
            )
            if sp
            else x
        )

        topk_w, topk_idx_raw = self.gate(xf)  # [B,S,k] replicated
        topk_idx = ttnn.typecast(topk_idx_raw, ttnn.bfloat16)  # ids <= 63 exact in bf16
        ttnn.deallocate(topk_idx_raw)

        mc = moe_full_seq_mem_config(xf.shape[1])
        partial = None
        for el in range(self.experts_per_dev):
            gid = self.dec_ids_experts[el]  # global id of this device's el-th expert (col-major)
            sel = ttnn.eq(topk_idx, gid)  # [B,S,k]
            contrib = ttnn.multiply(sel, topk_w)
            ttnn.deallocate(sel)
            w_e = ttnn.sum(contrib, dim=-1, keepdim=True)  # [B,S,1]
            ttnn.deallocate(contrib)

            oe = self._expert_dense(xf, el)
            if partial is None:
                partial = ttnn.multiply(oe, w_e, memory_config=mc)
            else:
                tmp = ttnn.addcmul(partial, oe, w_e, memory_config=mc)
                ttnn.deallocate(partial)
                partial = tmp
            ttnn.deallocate(oe)
            ttnn.deallocate(w_e)

        ttnn.deallocate(topk_w)
        ttnn.deallocate(topk_idx)

        combined = self._ep_all_reduce(partial)
        if self.shared_mlp is not None:
            shared = self.shared_mlp(xf)
            out = ttnn.add(shared, combined, memory_config=moe_full_seq_mem_config(xf.shape[1]))
            ttnn.deallocate(shared)
            ttnn.deallocate(combined)
        else:
            out = combined

        if sp:
            ttnn.deallocate(xf)
            out = sp_shard(
                self.ccl,
                out,
                dim=1,
                mesh_axis=self.sp_axis,
                n=self.sp_factor,
                out_memory_config=resid_mem_config(out.shape[1] // self.sp_factor),
            )
        return out
