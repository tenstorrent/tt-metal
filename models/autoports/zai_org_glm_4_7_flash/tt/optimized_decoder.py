# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Optimized TTNN decoder layer for zai-org/GLM-4.7-Flash (optimized-decoder stage).

Same public contract as ``fused_decoder.FusedDecoder`` (prefill/decode
semantics, paged latent cache geometry/ops, determinism, batch handling); this
subclass optimizes the per-device execution of the same op graph for a single
Blackhole p150-class chip (11x10 compute grid, 8 DRAM banks):

Precision/fidelity policy (per tensor group, defaults):
- activations / residual / norms: bf16 (norm gamma bf16, HiFi4 norm kernel)
- attention weights: bfloat4_b for the decode DRAM-sharded copies (wqkv_a,
  wq_b, wo) and the absorbed w_uk/w_uv (OPT-007 real-weight evidence: decode
  PCC unchanged at every tested context incl. 202k, faster; synthetic-weight
  arms use bf8 attention because gaussian weights lose ~2x more to bf4 block
  quantization - OPT-012); prefill flat projection copies stay bfloat8_b
- shared-expert weights: bfloat4_b (real-weight trial: moe decode
  0.500 -> 0.492 ms/tok at PCC 0.997+, 202k rows-at-bar unchanged vs the
  fused control); dense-MLP weights: bfloat8_b - the bf4 dense arm was
  measured (0.449 -> 0.443 ms/tok) and REJECTED on real-weight 202k
  dense-control regression (decode@202751 0.99865 vs 0.99993) for a
  negligible 1-of-47-layers win; synthetic arms pin bf8 per OPT-012
- routed experts: bfloat4_b (deployment contract: 30.6B fits one 32 GB chip
  only with bf4 experts, doc/probe/README.md) with LoFi
- router gate: fp32 weights, fp32 accumulation/output (selection semantics)
- latent KV cache: bfloat8_b deployment arm (real-weight 202k evidence equal
  to bf16 cache; bf16 remains supported and is the synthetic-ladder arm)
- decode matmul fidelity: LoFi for bf8/bf4 weight groups (isolated-op and
  layer PCC evidence in doc/optimized_decoder/), HiFi2 fallback per group
- prefill projections: HiFi2 + fp32 dest acc (flash MLA prefill keeps the
  functional stage's HiFi4 + fp32 acc: long-context drift evidence)

Decode program configs (from doc/optimized_decoder/ isolated-op sweeps):
- wqkv_a / wq_b / wo / shared_down: DRAM-sharded matmuls
  (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig, weights
  width-sharded across the 8 DRAM banks, activations width-sharded in L1 on
  the matching core count); wqkv_a N is zero-padded 1344 -> 1536 so
  per-core N divides the 8-bank grid (padding sliced off logically; the
  q/kv/rope slice offsets are below the pad).
- shared gate/up + dense gate/up: 1D mcast wide-grid matmuls on interleaved
  weights (the qwen36-blackhole idiom) with SiLU fused into the gate matmul.
- absorbed per-head matmuls (w_uk, w_uv): explicit
  MatmulMultiCoreReuseProgramConfig on a 5x4 grid, one head per core
  (default configs ran these at 43-73 GB/s; this is the biggest single fix).
- router: small 2-core 1D mcast config (25.8 -> 8.5 us), fp32 semantics kept.
- RMS norms: width-sharded L1 + LayerNormShardedMultiCoreProgramConfig
  (the default 1-core LayerNorm was 10-35 us per call).
- decode residual and most intermediates live in L1 (width-sharded across
  the wqkv in0 grid for the residual; interleaved L1 for the MoE glue).
- routed expert sparse matmuls: tuned 1D mcast configs (see class attrs),
  L1 outputs.

Weight-layout duplicates (decode DRAM-sharded copies alongside the prefill
interleaved copies) add ~0.55 GiB at the final dtypes over 47 layers; the
context contract budget still holds (see doc/context_contract.json).
"""

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import TILE, _ck
from models.autoports.zai_org_glm_4_7_flash.tt.fused_decoder import FusedDecoder

FIDELITY = {
    "lofi": (ttnn.MathFidelity.LoFi, False),
    "hifi2": (ttnn.MathFidelity.HiFi2, False),
    "hifi2_fp32": (ttnn.MathFidelity.HiFi2, True),
    "hifi4_fp32": (ttnn.MathFidelity.HiFi4, True),
}


def _dram_ws_weight_cfg(dev, k, n):
    """Weight width-sharded across all DRAM banks (n padded to a bank multiple)."""
    banks = dev.dram_grid_size().x
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    padded = -(-n // (TILE * banks)) * (TILE * banks)
    spec = ttnn.ShardSpec(grid, (k, padded // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def _dram_pc(in0_block_w, per_core_n, act=None):
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w, per_core_M=1, per_core_N=per_core_n, fused_activation=act
    )


def _rect_grid(cores):
    for cols in range(min(cores, 11), 0, -1):
        if cores % cols == 0 and cores // cols <= 10:
            return cols, cores // cols
    raise ValueError(f"no rectangular grid for {cores} cores")


def _mcast_2d_pc(m, k, n, act=None, gx=11, gy=10, bw=8, osh=2, osw=2):
    """2D mcast prefill matmul config (large-M flat projections). The
    tt-perf-report advice pass flagged the default-config prefill flat
    matmuls at in0_block_w=1; explicit 11x10 bw8 configs measured 1.3-1.8x
    faster per role (probe, work_log.md). Returns None for small M where the
    2D grid would idle most rows (default config kept there)."""
    mt, nt, kt = -(-m // TILE), -(-n // TILE), k // TILE
    if mt < gy:
        return None
    if kt % bw != 0:
        bw = max(d for d in range(1, bw + 1) if kt % d == 0)
    pcm, pcn = -(-mt // gy), -(-nt // gx)
    # Cap the out blocks (CB footprint scales with in0_block_w*out_block_w for
    # in1 and out_block_h*out_block_w for the output): per_core-sized blocks
    # fit on an empty chip but clash with resident L1 tensors in-layer for
    # very wide N (dense gate per_core_N=30).
    obh = max(d for d in range(1, min(pcm, 4) + 1) if pcm % d == 0)
    obw = max(d for d in range(1, min(pcn, 8) + 1) if pcn % d == 0)
    osh = max(d for d in range(1, min(osh, obh) + 1) if obh % d == 0)
    osw = max(d for d in range(1, min(osw, obw) + 1) if obw % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=bw,
        out_subblock_h=osh,
        out_subblock_w=osw,
        out_block_h=obh,
        out_block_w=obw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=act,
    )


def _mcast_1d_pc(nt, kt, target_cores, act=None, osw_cap=4):
    """Wide 1D mcast decode matmul config, grid sized exactly to the blocks."""
    per_core_n = -(-nt // target_cores)
    blocks = -(-nt // per_core_n)
    cols, rows = _rect_grid(blocks)
    in0_bw = max(d for d in range(1, min(kt, 48) + 1) if kt % d == 0)
    osw = max(d for d in range(1, osw_cap + 1) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
        in0_block_w=in0_bw,
        out_subblock_h=1,
        out_subblock_w=osw,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=act,
        mcast_in0=True,
    )


class OptimizedDecoder(FusedDecoder):
    """Per-device-optimized GLM-4.7-Flash decoder layer (same contract as fused)."""

    # Per-group decode fidelities (overridable before from_state_dict for A/B).
    attn_fidelity = "lofi"  # wqkv_a, wq_b, w_uk, w_uv, wo
    mlp_fidelity = "lofi"  # shared expert + dense MLP
    expert_fidelity = "lofi"  # routed experts (bf4)
    # Router/gate decode fidelity ($datatype-sweep policy field; the router
    # matmul stays fp32-weight regardless of this knob). Default matches the
    # value FunctionalDecoder.__init__ hard-coded before this attribute
    # existed (ck_hifi4 == FIDELITY["hifi4_fp32"]), so leaving this unset
    # reproduces the exact pre-existing behavior.
    router_fidelity = "hifi4_fp32"
    # Attention weight dtype (None -> weight_dtype). Applies to the decode
    # DRAM-sharded copies AND the absorbed w_uk/w_uv (which prefill shares).
    # Default bfloat4_b per the OPT-007 real-weight trial: real-checkpoint
    # decode PCC is unchanged (moe 0.9975, dense 0.9999) and traced decode
    # improves 0.513 -> 0.500 (moe) / 0.463 -> 0.449 (dense) ms/token; the
    # prefill interleaved projection copies stay at weight_dtype (bf8).
    attn_weight_dtype = ttnn.bfloat4_b
    # MLP weight dtypes (None -> weight_dtype). mlp_*_dtype covers the SHARED
    # expert; dense_mlp_dtype covers the dense-layer MLP separately. The P1
    # real-weight trial (work_log.md) measured both at bf4/LoFi:
    # - shared expert bf4: moe decode 0.500 -> 0.492 ms/tok (x46 layers at
    #   full model), 202k rows-at-bar unchanged vs the fused control -> KEPT.
    # - dense MLP bf4: dense decode 0.449 -> 0.443 (1 of 47 layers,
    #   negligible model-level win) but the real-weight 202k dense control
    #   loses measurably (decode@202751 0.9987 vs 0.9999, end window 28/32 vs
    #   29/32 rows) -> REJECTED on the accuracy-for-nothing trade; the dense
    #   MLP stays at weight_dtype (bf8).
    # Synthetic arms pin bf8 (OPT-012, same convention as attention weights).
    mlp_gateup_dtype = ttnn.bfloat4_b
    mlp_down_dtype = ttnn.bfloat4_b
    dense_mlp_dtype = None
    prefill_proj_fidelity = "hifi2_fp32"
    prefill_expert_fidelity = "hifi2_fp32"
    # Routed sparse matmul geometry (per_core_N, in0_block_w, out_subblock_w)
    # from the isolated sweep: gate_up 8x6 grid pcn2 bw8 osw2 (125->58 us at
    # bf4), down 8x8 grid pcn1 bw16 (79->35 us at bf4).
    # NOTE on out_subblock_w: in the NON-indexed sparsity-walk mode,
    # out_subblock_w>1 corrupts multi-group outputs (PCC 0.82-0.87 vs 0.9939
    # at osw=1; minimal repro in doc/optimized_decoder/work_log.md). The
    # indexed/gather mode is immune (bit-identical PCC osw1 vs osw2), so only
    # the batch-1 indexed decode path may use osw=2; the union (batch>1) and
    # prefill paths must keep osw=1.
    sparse_gu_geom = (2, 8, 2)
    sparse_dn_geom = (1, 16, 1)
    # Prefill sparse geometries (per_core_N, in0_block_w, out_subblock_w=1);
    # None = fall back to the functional-stage config for that matmul.
    sparse_gu_prefill_geom = (3, 32, 1)
    sparse_dn_prefill_geom = (1, 24, 1)

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=1,
        max_context=None,
        expert_dtype=ttnn.bfloat4_b,
        weight_dtype=ttnn.bfloat8_b,
        prefill_chunk_size=2048,
        paged_config=None,
    ):
        import torch

        self = FusedDecoder.from_state_dict.__func__(
            cls,
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_context=max_context,
            expert_dtype=expert_dtype,
            weight_dtype=weight_dtype,
            prefill_chunk_size=prefill_chunk_size,
            paged_config=paged_config,
        )
        dev = mesh_device
        grid = dev.compute_with_storage_grid_size()
        banks = dev.dram_grid_size().x

        # ---- compute kernel configs per tensor group ----
        self.ck_attn = _ck(dev, *FIDELITY[cls.attn_fidelity])
        self.ck_mlp = _ck(dev, *FIDELITY[cls.mlp_fidelity])
        self.ck_expert = _ck(dev, *FIDELITY[cls.expert_fidelity])
        self.ck_prefill_proj = _ck(dev, *FIDELITY[cls.prefill_proj_fidelity])
        self.ck_prefill_expert = _ck(dev, *FIDELITY[cls.prefill_expert_fidelity])
        self.ck_norm = _ck(dev, ttnn.MathFidelity.HiFi4, True)  # norms always HiFi4+fp32acc
        # Router decode fidelity is now a policy field (datatype-sweep); ck_hifi4
        # itself stays bound to true HiFi4+fp32acc for any inherited functional-
        # stage codepath that still reads it directly (e.g. prefill routing).
        self.ck_router = _ck(dev, *FIDELITY[cls.router_fidelity])
        # The inherited _moe_prefill consumes ck_hifi2 for the sparse expert
        # matmuls; repoint it at the prefill expert fidelity policy.
        self.ck_hifi2 = self.ck_prefill_expert

        # ---- decode DRAM-sharded weight copies + program configs ----
        attn_dtype = cls.attn_weight_dtype or weight_dtype
        gu_dtype = cls.mlp_gateup_dtype or weight_dtype
        dn_dtype = cls.mlp_down_dtype or weight_dtype
        dense_dtype = cls.dense_mlp_dtype or weight_dtype

        def dram_ws_from(key_t, k, n, dtype=attn_dtype):
            w = key_t.to(torch.float32)
            n_pad = -(-n // (TILE * banks)) * (TILE * banks)
            if n_pad != n:
                w = torch.nn.functional.pad(w, (0, n_pad - n))
            return ttnn.from_torch(
                w.contiguous(),
                device=dev,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram_ws_weight_cfg(dev, k, n),
            )

        wq_a_t = state_dict["self_attn.q_a_proj.weight"].to(torch.float32).T
        wkv_a_t = state_dict["self_attn.kv_a_proj_with_mqa.weight"].to(torch.float32).T
        wqkv_cat = torch.cat([wq_a_t, wkv_a_t], dim=-1)  # [2048, 1344]
        self.qkv_n = wqkv_cat.shape[-1]
        self.qkv_n_pad = -(-self.qkv_n // (TILE * banks)) * (TILE * banks)
        self.wqkv_a_ds = dram_ws_from(wqkv_cat, self.hidden, self.qkv_n)
        self.wqkv_pc = _dram_pc(in0_block_w=8, per_core_n=self.qkv_n_pad // TILE // banks)

        wq_b_t = state_dict["self_attn.q_b_proj.weight"].to(torch.float32).T  # [768, 5120]
        self.wq_b_ds = dram_ws_from(wq_b_t, self.q_lora_rank, self.num_heads * self.qk_head_dim)
        self.wq_b_pc = _dram_pc(in0_block_w=3, per_core_n=self.num_heads * self.qk_head_dim // TILE // banks)
        # decode no longer uses the flat interleaved wq_b
        ttnn.deallocate(self.wq_b)
        del self.wq_b

        wo_t = state_dict["self_attn.o_proj.weight"].to(torch.float32).T  # [5120, 2048]
        self.wo_ds = dram_ws_from(wo_t, self.num_heads * self.v_head_dim, self.hidden)
        # bf4 halves the weight bytes and shifts the block-geometry winner
        # (precision-locked sweep, work_log.md): bw20 at bf4, bw10 at bf8.
        self.wo_pc = _dram_pc(
            in0_block_w=20 if attn_dtype == ttnn.bfloat4_b else 10, per_core_n=self.hidden // TILE // banks
        )

        if attn_dtype != weight_dtype:
            # OPT-007 arm: absorbed per-head weights follow the attention dtype
            # (prefill shares these tensors).
            kv_b = state_dict["self_attn.kv_b_proj.weight"].to(torch.float32)
            kv_b = kv_b.reshape(self.num_heads, self.qk_nope + self.v_head_dim, self.kv_lora_rank)
            w_uk = kv_b[:, : self.qk_nope, :].unsqueeze(0).contiguous()
            w_uv_t = kv_b[:, self.qk_nope :, :].transpose(-1, -2).unsqueeze(0).contiguous()
            for name, w in (("w_uk", w_uk), ("w_uv_t", w_uv_t)):
                ttnn.deallocate(getattr(self, name))
                setattr(
                    self,
                    name,
                    ttnn.from_torch(
                        w, device=dev, dtype=attn_dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                    ),
                )

        if self.layer_kind == "moe":
            sd_t = state_dict["mlp.shared_experts.down_proj.weight"].to(torch.float32).T  # [1536, 2048]
            self.shared_down_ds = dram_ws_from(sd_t, sd_t.shape[0], self.hidden, dtype=dn_dtype)
            self.shared_down_pc = _dram_pc(in0_block_w=6, per_core_n=self.hidden // TILE // banks)
            self.shared_inter = sd_t.shape[0]
            # shared gate/up: 1D mcast on the interleaved weights (kept from base)
            si_t = self.shared_inter // TILE
            self.shared_gate_pc = _mcast_1d_pc(si_t, self.hidden // TILE, 24, act=ttnn.UnaryOpType.SILU)
            self.shared_up_pc = _mcast_1d_pc(si_t, self.hidden // TILE, 24)
            # router: tiny 2-core 1D config, fp32 out/acc kept
            self.router_pc = _mcast_1d_pc(self.n_experts // TILE, self.hidden // TILE, 2, osw_cap=1)
            # routed sparse matmul configs (indexed batch-1 decode may use
            # osw>1; the union batch>1 path runs the non-indexed sparsity walk
            # where osw>1 corrupts multi-group outputs, so it gets osw=1)
            gu_pcn, gu_bw, gu_osw = cls.sparse_gu_geom
            dn_pcn, dn_bw, dn_osw = cls.sparse_dn_geom
            gu_nt, dn_nt = 2 * self.moe_inter // TILE, self.hidden // TILE
            gu_kt, dn_kt = self.hidden // TILE, self.moe_inter // TILE
            self.sparse_gu_pc = self._sparse_pc_exact(gu_nt, gu_kt, gu_pcn, gu_bw, gu_osw)
            self.sparse_dn_pc = self._sparse_pc_exact(dn_nt, dn_kt, dn_pcn, dn_bw, dn_osw)
            self.sparse_gu_pc_union = self._sparse_pc_exact(gu_nt, gu_kt, gu_pcn, gu_bw, 1)
            self.sparse_dn_pc_union = self._sparse_pc_exact(dn_nt, dn_kt, dn_pcn, dn_bw, 1)
        else:
            dg_t = state_dict["mlp.gate_proj.weight"].to(torch.float32).T  # [2048, 10240]
            self.dense_inter = dg_t.shape[-1]
            di_t = self.dense_inter // TILE
            self.dense_gate_pc = _mcast_1d_pc(di_t, self.hidden // TILE, 88, act=ttnn.UnaryOpType.SILU)
            self.dense_up_pc = _mcast_1d_pc(di_t, self.hidden // TILE, 88)
            dd_t = state_dict["mlp.down_proj.weight"].to(torch.float32).T  # [10240, 2048]
            self.dense_down_ds = dram_ws_from(dd_t, self.dense_inter, self.hidden, dtype=dense_dtype)
            self.dense_down_pc = _dram_pc(
                in0_block_w=20 if dense_dtype == ttnn.bfloat4_b else 10, per_core_n=self.hidden // TILE // banks
            )
            # At bf4, DRAM-sharded dense gate/up becomes legal (half-size CBs)
            # and beats the wide-1D family (42.5 vs 45.6 us isolated); at bf8
            # the 1D family stays (DRAM-sharded CB-clashes at bw8 / is slower).
            self.dense_gu_dram = dense_dtype == ttnn.bfloat4_b
            if self.dense_gu_dram:
                du_t = state_dict["mlp.up_proj.weight"].to(torch.float32).T
                self.dense_gate_ds = dram_ws_from(dg_t, self.hidden, self.dense_inter, dtype=dense_dtype)
                self.dense_up_ds = dram_ws_from(du_t, self.hidden, self.dense_inter, dtype=dense_dtype)
                self.dense_gu_pc_silu = _dram_pc(
                    in0_block_w=8, per_core_n=self.dense_inter // TILE // banks, act=ttnn.UnaryOpType.SILU
                )
                self.dense_gu_pc_plain = _dram_pc(in0_block_w=8, per_core_n=self.dense_inter // TILE // banks)

        # ---- decode sharded memory configs ----
        def ws_cfg(width, cores):
            return ttnn.create_sharded_memory_config(
                shape=(TILE, width // cores),
                core_grid=ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

        self.res_mem = ws_cfg(self.hidden, banks)  # residual on the DRAM-sharded in0 grid
        self.q_norm_mem = ws_cfg(self.q_lora_rank, banks)
        self.kv_norm_mem = ws_cfg(self.kv_lora_rank, banks)
        if self.layer_kind == "moe":
            self.shared_down_in_mem = ws_cfg(self.shared_inter, banks)
        else:
            self.dense_down_in_mem = ws_cfg(self.dense_inter, banks)

        def norm_pc(width, cores):
            bw = width // TILE // cores
            cols, rows = _rect_grid(cores)
            return ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
                subblock_w=max(d for d in (1, 2, 4) if bw % d == 0),
                block_h=1,
                block_w=bw,
                inplace=False,
            )

        self.res_norm_pc = norm_pc(self.hidden, banks)
        self.q_norm_pc = norm_pc(self.q_lora_rank, banks)
        self.kv_norm_pc = norm_pc(self.kv_lora_rank, banks)

        # re-upload the interleaved gate/up copies (prefill + 1D decode) when
        # the MLP gate/up dtype differs from the base weight dtype
        for name, key, dt in (
            ("shared_gate", "mlp.shared_experts.gate_proj.weight", gu_dtype),
            ("shared_up", "mlp.shared_experts.up_proj.weight", gu_dtype),
            ("mlp_gate", "mlp.gate_proj.weight", dense_dtype),
            ("mlp_up", "mlp.up_proj.weight", dense_dtype),
        ):
            if dt == weight_dtype or not hasattr(self, name):
                continue
            w = state_dict[key].to(torch.float32).T.contiguous()
            ttnn.deallocate(getattr(self, name))
            setattr(
                self,
                name,
                ttnn.from_torch(
                    w, device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
                ),
            )

        # absorbed per-head matmuls: one head per core on a 5x4 grid
        self.w_uk_pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
            in0_block_w=self.qk_nope // TILE,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=self.kv_lora_rank // TILE,
        )
        self.w_uv_pc = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
            # precision-locked sweep: bw8 wins at bf4 (12.8 vs 13.5 us at bw16)
            in0_block_w=8 if attn_dtype == ttnn.bfloat4_b else self.kv_lora_rank // TILE,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=self.v_head_dim // TILE,
        )
        return self

    def _sparse_pc(self, m, n, k, cores=(8, 4), in0_block_w=8, out_subblock_w=1):
        """Prefill sparse expert matmul configs (consumed by the inherited
        _moe_prefill): tuned grid/in0_block_w from the prefill-shaped sweep.
        gate_up: 48 cores (pcn 2), in0_block_w 32 (61.4 -> 33.3 ms per
        1024-token all-ones chunk); down: pcn 2, in0_block_w 24 (37.8 -> 18.2)."""
        nt, kt = -(-n // TILE), k // TILE
        geom = None
        if self.layer_kind == "moe" and n == 2 * self.moe_inter:
            geom = self.sparse_gu_prefill_geom
        elif self.layer_kind == "moe" and n == self.hidden and k == self.moe_inter:
            geom = self.sparse_dn_prefill_geom
        if geom is None:
            return super()._sparse_pc(m, n, k, cores=cores, in0_block_w=in0_block_w, out_subblock_w=out_subblock_w)
        pcn, bw, osw = geom
        blocks = -(-nt // pcn)
        cols, rows = _rect_grid(blocks)
        if kt % bw != 0:
            bw = max(d for d in range(1, bw + 1) if kt % d == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=osw,
            out_block_h=1,
            out_block_w=osw,
            per_core_M=max(TILE, m) // TILE,
            per_core_N=pcn,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    @staticmethod
    def _sparse_pc_exact(nt, kt, per_core_n, in0_bw, osw=1):
        """sparse_matmul 1D config with the grid sized exactly to the blocks."""
        blocks = -(-nt // per_core_n)
        cols, rows = _rect_grid(blocks)
        if kt % in0_bw != 0:
            in0_bw = max(d for d in range(1, in0_bw + 1) if kt % d == 0)
        if per_core_n % osw != 0:
            osw = max(d for d in range(1, osw + 1) if per_core_n % d == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
            in0_block_w=in0_bw,
            out_subblock_h=1,
            out_subblock_w=osw,
            out_block_h=1,
            out_block_w=osw,
            per_core_M=1,
            per_core_N=per_core_n,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )

    # ------------------------------------------------------------------ norms

    def _rms(self, x, w, eps):
        """Interleaved RMS norm (prefill path): pinned to the norm kernel config
        so the prefill projection-fidelity swap never touches norm numerics."""
        return ttnn.rms_norm(x, epsilon=eps, weight=w, compute_kernel_config=self.ck_norm)

    def _rms_sharded(self, x, w, eps, mem, pc):
        """Width-sharded RMS norm; reshards x if it is not already in mem."""
        if x.memory_config() != mem:
            x = ttnn.to_memory_config(x, mem)
        return ttnn.rms_norm(
            x, epsilon=eps, weight=w, program_config=pc, compute_kernel_config=self.ck_norm, memory_config=mem
        )

    # ------------------------------------------------------------------ decode attention

    def _attn_decode(self, x, kv_cache, page_table, cur_pos_tensor, rot_idxs):
        """x: [1, 1, B, hidden] normed, width-sharded on the residual grid."""
        B = x.shape[2]
        cos, sin, trans = self._decode_rope_mats(rot_idxs, B)

        # --- fused QKV-A projection: DRAM-sharded, output L1 width-sharded ---
        qkv_a = ttnn.linear(
            x,
            self.wqkv_a_ds,
            program_config=self.wqkv_pc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.ck_attn,
        )  # [1,1,B,qkv_n_pad] logical width includes the zero pad
        qkv_a = ttnn.sharded_to_interleaved(qkv_a, ttnn.L1_MEMORY_CONFIG)
        q = ttnn.slice(qkv_a, [0, 0, 0, 0], [1, 1, B, self.q_lora_rank])
        kv_nope = ttnn.slice(qkv_a, [0, 0, 0, self.q_lora_rank], [1, 1, B, self.q_lora_rank + self.kv_lora_rank])
        kv_rope = ttnn.slice(
            qkv_a, [0, 0, 0, self.q_lora_rank + self.kv_lora_rank], [1, 1, B, self.q_lora_rank + self.kvpe_dim]
        )
        ttnn.deallocate(qkv_a)

        # --- KV path ---
        kv_nope = self._rms_sharded(kv_nope, self.kv_norm_w, self.lora_norm_eps, self.kv_norm_mem, self.kv_norm_pc)
        kv_nope = ttnn.sharded_to_interleaved(kv_nope, ttnn.L1_MEMORY_CONFIG)
        if B > 1:
            kv_nope = ttnn.transpose(kv_nope, 1, 2)  # [1, B, 1(32), kv_lora]
            kv_rope = ttnn.transpose(kv_rope, 1, 2, memory_config=self.rope_in_decode_mem)
        else:
            kv_rope = ttnn.to_memory_config(kv_rope, self.rope_in_decode_mem)
        kv_rope = ttnn.experimental.rotary_embedding_llama(kv_rope, cos, sin, trans, is_decode_mode=True)
        kv_rope = ttnn.to_memory_config(kv_rope, ttnn.DRAM_MEMORY_CONFIG)
        # kvpe stays in DRAM (as in the fused stage): at batch 32 the
        # paged_update_cache static CBs need the L1 headroom.
        kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, B, 1(32), 576]
        ttnn.deallocate(kv_nope)
        ttnn.deallocate(kv_rope)

        # --- Q path ---
        q = self._rms_sharded(q, self.q_norm_w, self.lora_norm_eps, self.q_norm_mem, self.q_norm_pc)
        q = ttnn.linear(
            q,
            self.wq_b_ds,
            program_config=self.wq_b_pc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.ck_attn,
        )  # [1,1,B,nh*qk_head] width-sharded
        q = ttnn.sharded_to_interleaved(q, ttnn.L1_MEMORY_CONFIG)
        q = ttnn.untilize(q)
        q = ttnn.reshape(q, (1, B, self.num_heads, self.qk_head_dim))
        q = ttnn.tilize_with_zero_padding(q, use_multicore=True)  # [1, B, nh(32), qk_head]
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, B, self.num_heads, self.qk_nope])
        q_rope = ttnn.slice(q, [0, 0, 0, self.qk_nope], [1, B, self.num_heads, self.qk_head_dim])
        ttnn.deallocate(q)
        q_nope = ttnn.transpose(q_nope, 1, 2)  # [1, nh, B, qk_nope]
        q_lat = ttnn.matmul(
            q_nope, self.w_uk, program_config=self.w_uk_pc, compute_kernel_config=self.ck_attn
        )  # [1, nh, B, kv_lora]
        ttnn.deallocate(q_nope)
        q_lat = ttnn.transpose(q_lat, 1, 2)  # [1, B, nh, kv_lora]
        q_rope = ttnn.to_memory_config(q_rope, self.rope_in_decode_mem)
        q_rope = ttnn.experimental.rotary_embedding_llama(q_rope, cos, sin, trans, is_decode_mode=True)
        q_rope = ttnn.to_memory_config(q_rope, ttnn.DRAM_MEMORY_CONFIG)
        q_abs = ttnn.concat([q_lat, q_rope], dim=-1)  # [1, B, nh, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(trans)

        # --- cache update (unchanged fused path) ---
        for g0, g1, mem in self.kvpe_update_groups:
            single = g0 == 0 and g1 == B and len(self.kvpe_update_groups) == 1
            kvpe_g = kvpe if single else ttnn.slice(kvpe, [0, g0, 0, 0], [1, g1, kvpe.shape[2], self.kvpe_dim])
            pos_g = cur_pos_tensor if single else ttnn.slice(cur_pos_tensor, [g0], [g1])
            pt_g = page_table if single else ttnn.slice(page_table, [g0, 0], [g1, page_table.shape[1]])
            kvpe_sh = ttnn.to_memory_config(kvpe_g, mem)
            ttnn.experimental.paged_update_cache(kv_cache, kvpe_sh, update_idxs_tensor=pos_g, page_table=pt_g)
            ttnn.deallocate(kvpe_sh)
        ttnn.deallocate(kvpe)

        attn_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_abs,
            kv_cache,
            head_dim_v=self.kv_lora_rank,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            program_config=self.flash_decode_pc,
            compute_kernel_config=self.ck_flash_decode,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, B, nh(padded), kv_lora]
        ttnn.deallocate(q_abs)
        attn_lat = ttnn.slice(attn_lat, [0, 0, 0, 0], [1, B, self.num_heads, self.kv_lora_rank])
        attn_lat = ttnn.transpose(attn_lat, 1, 2)  # [1, nh, B, kv_lora]
        v = ttnn.matmul(
            attn_lat, self.w_uv_t, program_config=self.w_uv_pc, compute_kernel_config=self.ck_attn
        )  # [1, nh, B, v_head]
        ttnn.deallocate(attn_lat)
        v = ttnn.transpose(v, 1, 2)  # [1, B, nh, v_head]
        v = ttnn.to_memory_config(v, self.concat_heads_mem(B))
        v = ttnn.experimental.nlp_concat_heads_decode(v, num_heads=self.num_heads)  # [1, 1, 32, nh*v_head]
        v = ttnn.to_memory_config(v, ttnn.L1_MEMORY_CONFIG)
        if v.shape[2] != B:
            v = ttnn.slice(v, [0, 0, 0, 0], [1, 1, B, self.num_heads * self.v_head_dim])
        # wo consumes a width-sharded input on the DRAM-sharded in0 grid
        wo_in_mem = getattr(self, "_wo_in_mem", None)
        if wo_in_mem is None:
            grid = self.device.compute_with_storage_grid_size()
            wo_in_mem = ttnn.create_sharded_memory_config(
                shape=(TILE, self.num_heads * self.v_head_dim // self.device.dram_grid_size().x),
                core_grid=ttnn.num_cores_to_corerangeset(self.device.dram_grid_size().x, grid, row_wise=True),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self._wo_in_mem = wo_in_mem
        v = ttnn.to_memory_config(v, wo_in_mem)
        out = ttnn.linear(
            v,
            self.wo_ds,
            program_config=self.wo_pc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.ck_attn,
        )  # [1,1,B,hidden] width-sharded on the residual grid
        ttnn.deallocate(v)
        return out

    # ------------------------------------------------------------------ decode mlp pieces

    def _swiglu_shared_decode(self, x_l1):
        """Shared expert at decode: 1D-mcast gate/up (SiLU fused) + DRAM-sharded down."""
        g = ttnn.linear(
            x_l1,
            self.shared_gate,
            program_config=self.shared_gate_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_mlp,
        )
        u = ttnn.linear(
            x_l1,
            self.shared_up,
            program_config=self.shared_up_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_mlp,
        )
        h = ttnn.multiply(g, u, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        h = ttnn.to_memory_config(h, self.shared_down_in_mem)
        out = ttnn.linear(
            h,
            self.shared_down_ds,
            program_config=self.shared_down_pc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.ck_mlp,
        )
        ttnn.deallocate(h)
        return out  # width-sharded on the residual grid

    def _swiglu_dense_decode(self, x_l1):
        if getattr(self, "dense_gu_dram", False):
            # bf4 arm: all-DRAM-sharded dense MLP on the width-sharded stream
            # (norm output feeds gate/up directly; the elementwise output is
            # already the down matmul's in0 shard).
            xs = x_l1 if x_l1.memory_config() == self.res_mem else ttnn.to_memory_config(x_l1, self.res_mem)
            g = ttnn.linear(
                xs,
                self.dense_gate_ds,
                program_config=self.dense_gu_pc_silu,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.ck_mlp,
            )
            u = ttnn.linear(
                xs,
                self.dense_up_ds,
                program_config=self.dense_gu_pc_plain,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.ck_mlp,
            )
            if xs is not x_l1:
                ttnn.deallocate(xs)
            h = ttnn.multiply(g, u)  # same width-sharded spec
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            out = ttnn.linear(
                h,
                self.dense_down_ds,
                program_config=self.dense_down_pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self.ck_mlp,
            )
            ttnn.deallocate(h)
            return out
        g = ttnn.linear(
            x_l1,
            self.mlp_gate,
            program_config=self.dense_gate_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_mlp,
        )
        u = ttnn.linear(
            x_l1,
            self.mlp_up,
            program_config=self.dense_up_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_mlp,
        )
        h = ttnn.multiply(g, u, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        h = ttnn.to_memory_config(h, self.dense_down_in_mem)
        out = ttnn.linear(
            h,
            self.dense_down_ds,
            program_config=self.dense_down_pc,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.ck_mlp,
        )
        ttnn.deallocate(h)
        return out

    def _router_scores_decode(self, x_l1):
        """Router with the tuned 2-core config; math identical to fused."""
        logits = ttnn.linear(
            x_l1,
            self.gate_w,
            program_config=self.router_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_router,
            dtype=ttnn.float32,
        )
        scores = ttnn.sigmoid_accurate(logits)
        ttnn.deallocate(logits)
        biased = ttnn.add(scores, self.gate_bias)
        row_mean = ttnn.mean(biased, dim=-1, keepdim=True)
        centered = ttnn.subtract(biased, row_mean)
        ttnn.deallocate(row_mean)
        ttnn.deallocate(biased)
        centered_bf16 = ttnn.typecast(centered, ttnn.bfloat16)
        ttnn.deallocate(centered)
        return scores, centered_bf16

    def _moe_decode_indexed(self, x):
        """Batch-1 decode MoE: fused-decoder graph with tuned configs, L1 glue,
        bf4 experts by default. x: [1,1,B,hidden] interleaved L1.

        The compact routing weights come from an embedding-table lookup of the
        bf16 sigmoid scores at the topk ids instead of ttnn.gather: the gather
        op is a ~37 us single-core kernel at [1,1,32,64] while the embedding
        chain measures ~18 us (probe role_qpath). Normalization then runs on
        the compact [1,k,1,1] picks (bf16; the weights were already applied in
        bf16, and layer PCC is unchanged)."""
        B = x.shape[2]
        scores, centered_bf16 = self._router_scores_decode(x)
        _, idx = ttnn.topk(centered_bf16, k=self.top_k, dim=-1, sorted=True)
        ttnn.deallocate(centered_bf16)

        idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)  # uint16 [1,1,1,k] for sparse_matmul
        idx_u32 = ttnn.typecast(idx, ttnn.uint32)
        ttnn.deallocate(idx)
        idx_u32 = ttnn.to_layout(idx_u32, ttnn.ROW_MAJOR_LAYOUT)
        idx_u32 = ttnn.reshape(idx_u32, (1, self.top_k))

        scores_bf16 = ttnn.typecast(scores, ttnn.bfloat16)
        ttnn.deallocate(scores)
        table = ttnn.to_layout(scores_bf16, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(scores_bf16)
        table = ttnn.reshape(table, (self.n_experts, 1))
        picked = ttnn.embedding(idx_u32, table)  # [1, k, 1] bf16 RM
        ttnn.deallocate(idx_u32)
        ttnn.deallocate(table)
        picked = ttnn.to_layout(ttnn.reshape(picked, (1, self.top_k, 1, 1)), ttnn.TILE_LAYOUT)
        denom = ttnn.sum(picked, dim=1, keepdim=True)  # [1,1,1,1]
        denom = ttnn.add(denom, 1e-20)
        rw = ttnn.div(picked, denom)  # [1, k, 1, 1]
        ttnn.deallocate(picked)
        ttnn.deallocate(denom)

        inter = self.moe_inter
        gu = ttnn.sparse_matmul(
            x,
            self.experts_gate_up,
            sparsity=self.ones_e,
            indices=idx_rm,
            is_input_b_sparse=True,
            program_config=self.sparse_gu_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_expert,
            dtype=ttnn.bfloat16,
        )  # [1, 1, 1, k, B, 2*inter] compact
        gu = ttnn.reshape(gu, (1, self.top_k, B, 2 * inter))
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, self.top_k, B, inter], memory_config=ttnn.L1_MEMORY_CONFIG)
        up = ttnn.slice(gu, [0, 0, 0, inter], [1, self.top_k, B, 2 * inter], memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gu)
        h = ttnn.multiply(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        h = ttnn.multiply(h, rw, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(rw)
        down = ttnn.sparse_matmul(
            h,
            self.experts_down,
            sparsity=self.ones_e,
            indices=idx_rm,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self.sparse_dn_pc,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_expert,
            dtype=ttnn.bfloat16,
        )  # [1, k, B, hidden] compact
        ttnn.deallocate(h)
        ttnn.deallocate(idx_rm)
        routed = ttnn.sum(down, dim=1, keepdim=True)  # [1, 1, B, hidden]
        ttnn.deallocate(down)

        shared = self._swiglu_shared_decode(x)  # width-sharded residual grid
        routed = ttnn.to_memory_config(routed, self.res_mem)
        out = ttnn.add(routed, shared, memory_config=self.res_mem)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    def _moe_decode_union(self, x):
        """Batch>1 decode MoE (union sparsity), tuned configs + L1 glue."""
        B = x.shape[2]
        scores, centered_bf16 = self._router_scores_decode(x)
        _, idx = ttnn.topk(centered_bf16, k=self.top_k, dim=-1, sorted=True)
        T = idx.shape[2]
        src = self.scatter_ones
        if src.shape[2] != T:
            src = ttnn.slice(self.scatter_ones, [0, 0, 0, 0], [1, 1, T, self.top_k])
        mask_bf16 = ttnn.scatter(ttnn.zeros_like(centered_bf16), dim=-1, index=idx, src=src)
        ttnn.deallocate(centered_bf16)
        ttnn.deallocate(idx)
        mask = ttnn.typecast(mask_bf16, ttnn.float32)
        ttnn.deallocate(mask_bf16)
        picked = ttnn.multiply(scores, mask)
        ttnn.deallocate(scores)
        ttnn.deallocate(mask)
        denom = ttnn.sum(picked, dim=-1, keepdim=True)
        denom = ttnn.add(denom, 1e-20)
        inv = ttnn.reciprocal(denom)
        ttnn.deallocate(denom)
        weights = ttnn.multiply(picked, inv)
        ttnn.deallocate(picked)
        ttnn.deallocate(inv)
        routing = ttnn.typecast(weights, ttnn.bfloat16)
        ttnn.deallocate(weights)

        union = ttnn.max(routing, dim=2, keepdim=True)  # [1,1,1,E]
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(union)

        inter = self.moe_inter
        gu = ttnn.sparse_matmul(
            x,
            self.experts_gate_up,
            sparsity=sparsity,
            nnz=None,
            program_config=self.sparse_gu_pc_union,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_expert,
            dtype=ttnn.bfloat16,
        )
        gu = ttnn.reshape(gu, (1, self.n_experts, B, 2 * inter))
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, self.n_experts, B, inter], memory_config=ttnn.L1_MEMORY_CONFIG)
        up = ttnn.slice(gu, [0, 0, 0, inter], [1, self.n_experts, B, 2 * inter], memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(gu)
        h = ttnn.multiply(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        rw = ttnn.permute(routing, (0, 3, 2, 1))  # [1, E, B, 1]
        ttnn.deallocate(routing)
        h = ttnn.multiply(h, rw, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(rw)
        down = ttnn.sparse_matmul(
            h,
            self.experts_down,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=True,
            program_config=self.sparse_dn_pc_union,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.ck_expert,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(sparsity)
        routed = ttnn.sum(down, dim=1, keepdim=True)
        ttnn.deallocate(down)

        shared = self._swiglu_shared_decode(x)
        routed = ttnn.to_memory_config(routed, self.res_mem)
        out = ttnn.add(routed, shared, memory_config=self.res_mem)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    def _mlp_decode(self, x):
        """x: interleaved L1 normed hidden. Returns width-sharded residual-grid output."""
        if self.layer_kind == "dense":
            return self._swiglu_dense_decode(x)
        if self.max_batch == 1:
            return self._moe_decode_indexed(x)
        return self._moe_decode_union(x)

    # ------------------------------------------------------------------ public decode

    def decode_forward(self, x, *, kv_cache, page_table, cur_pos_tensor, rot_idxs):
        if x.memory_config() != self.res_mem:
            x = ttnn.to_memory_config(x, self.res_mem)
        h = self._rms_sharded(x, self.input_norm_w, self.rms_eps, self.res_mem, self.res_norm_pc)
        attn = self._attn_decode(h, kv_cache, page_table, cur_pos_tensor, rot_idxs)
        ttnn.deallocate(h)
        res = ttnn.add(x, attn, memory_config=self.res_mem)
        ttnn.deallocate(attn)
        h2 = self._rms_sharded(res, self.post_norm_w, self.rms_eps, self.res_mem, self.res_norm_pc)
        if self.layer_kind == "dense" and getattr(self, "dense_gu_dram", False):
            # the DRAM-sharded dense MLP consumes the width-sharded norm output directly
            mlp = self._mlp_decode(h2)
        else:
            h2 = ttnn.sharded_to_interleaved(h2, ttnn.L1_MEMORY_CONFIG)
            mlp = self._mlp_decode(h2)
        ttnn.deallocate(h2)
        out = ttnn.add(res, mlp, memory_config=self.res_mem)
        ttnn.deallocate(res)
        ttnn.deallocate(mlp)
        return out  # width-sharded L1 on the residual grid

    forward = decode_forward

    # ------------------------------------------------------------------ prefill

    def _swiglu_linear(self, x, w_gate, w_up, w_down, ck):
        """Prefill swiglu (shared expert / dense MLP) with the prefill
        projection fidelity and explicit 2D mcast configs for large M."""
        ckp = self.ck_prefill_proj
        m, k = x.shape[2], x.shape[3]
        inter = w_gate.shape[-1]
        g = ttnn.linear(
            x,
            w_gate,
            program_config=_mcast_2d_pc(m, k, inter, act=ttnn.UnaryOpType.SILU),
            compute_kernel_config=ckp,
            activation="silu" if _mcast_2d_pc(m, k, inter) is None else None,
        )
        u = ttnn.linear(x, w_up, program_config=_mcast_2d_pc(m, k, inter), compute_kernel_config=ckp)
        h = ttnn.multiply(g, u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(h, w_down, program_config=_mcast_2d_pc(m, inter, w_down.shape[-1]), compute_kernel_config=ckp)
        ttnn.deallocate(h)
        return out

    def _attn_prefill_chunk(self, x, kv_cache, page_table, user_id, chunk_start):
        """Fused-decoder prefill chunk with the prefill projection fidelity and
        reduced-dtype cache support: the kvpe fill tensor is typecast to the
        cache dtype before paged_fill_cache (decode paged_update_cache keeps
        its bf16 input; the op writes into the cache dtype)."""
        S_c = x.shape[2]
        ckp = self.ck_prefill_proj
        cos, sin, trans = self.rope.prefill_mats(chunk_start, chunk_start + S_c)

        qkv_a = ttnn.linear(
            x,
            self.wqkv_a,
            program_config=_mcast_2d_pc(S_c, self.hidden, self.q_lora_rank + self.kvpe_dim),
            compute_kernel_config=ckp,
        )  # [1,1,S,q_lora+576]
        q = ttnn.slice(qkv_a, [0, 0, 0, 0], [1, 1, S_c, self.q_lora_rank])
        kv_nope = ttnn.slice(qkv_a, [0, 0, 0, self.q_lora_rank], [1, 1, S_c, self.q_lora_rank + self.kv_lora_rank])
        kv_rope = ttnn.slice(
            qkv_a, [0, 0, 0, self.q_lora_rank + self.kv_lora_rank], [1, 1, S_c, self.q_lora_rank + self.kvpe_dim]
        )
        ttnn.deallocate(qkv_a)

        kv_nope = self._rms(kv_nope, self.kv_norm_w, self.lora_norm_eps)
        kv_rope = ttnn.experimental.rotary_embedding_llama(kv_rope, cos, sin, trans, is_decode_mode=False)
        kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,S,576]
        ttnn.deallocate(kv_nope)
        ttnn.deallocate(kv_rope)
        if kv_cache.dtype != kvpe.dtype:
            kvpe_fill = ttnn.typecast(kvpe, kv_cache.dtype)
        else:
            kvpe_fill = kvpe

        block = self.paged_config.block_size
        start_block = chunk_start // block
        end_block = (chunk_start + S_c) // block
        # NB: page-table slices may alias the page table; never deallocate them.
        chunk_pt = ttnn.slice(page_table, [0, start_block], [page_table.shape[0], end_block])
        ttnn.experimental.paged_fill_cache(kv_cache, kvpe_fill, page_table=chunk_pt, batch_idx=user_id)
        if kvpe_fill is not kvpe:
            ttnn.deallocate(kvpe_fill)

        q = self._rms(q, self.q_norm_w, self.lora_norm_eps)
        qh = ttnn.matmul(q, self.wq_b_heads, compute_kernel_config=ckp)  # [1, nh, S, qk_head]
        ttnn.deallocate(q)
        q_nope = ttnn.slice(qh, [0, 0, 0, 0], [1, self.num_heads, S_c, self.qk_nope])
        q_rope = ttnn.slice(qh, [0, 0, 0, self.qk_nope], [1, self.num_heads, S_c, self.qk_head_dim])
        ttnn.deallocate(qh)
        q_lat = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=ckp)  # [1, nh, S, kv_lora]
        ttnn.deallocate(q_nope)
        q_rope = ttnn.experimental.rotary_embedding_llama(q_rope, cos, sin, trans, is_decode_mode=False)
        q_abs = ttnn.concat([q_lat, q_rope], dim=-1)  # [1, nh, S, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        # cos/sin may alias the persistent rope tables; never deallocate.

        user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
        attn_lat = ttnn.transformer.chunked_flash_mla_prefill(
            q_abs,
            kv_cache,
            self.kv_lora_rank,
            user_pt,
            chunk_start_idx=chunk_start,
            scale=self.scale,
            compute_kernel_config=self.ck_flash_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, nh, S, kv_lora]
        ttnn.deallocate(q_abs)
        ttnn.deallocate(kvpe)

        v = ttnn.matmul(attn_lat, self.w_uv_t, compute_kernel_config=ckp)  # [1, nh, S, v_head]
        ttnn.deallocate(attn_lat)
        v = ttnn.transformer.concatenate_heads(v)  # [1, S, nh*v_head]
        v = ttnn.reshape(v, (1, 1, S_c, self.num_heads * self.v_head_dim))
        out = ttnn.linear(
            v,
            self.wo,
            program_config=_mcast_2d_pc(S_c, self.num_heads * self.v_head_dim, self.hidden),
            compute_kernel_config=ckp,
        )  # [1,1,S,hidden]
        ttnn.deallocate(v)
        return out
