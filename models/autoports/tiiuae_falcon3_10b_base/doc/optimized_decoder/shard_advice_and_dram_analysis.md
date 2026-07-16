# Shard-Advice Usage & DRAM-Sharding Analysis — Falcon3-10B-Base

**Model:** `tiiuae/Falcon3-10B-Base` (autoport dir `tiiuae_falcon3_10b_base`)
**Artifacts:** `doc/optimized_decoder/shard_advise/report.json` (advisor input), `tt/optimized_decoder.py` (final), `doc/optimized_decoder/work_log.md` (disposition).

This report answers two questions for the optimized single-device decode layer: (1) how much of the shard-advisor output was actually used, under a strict classification rubric, and (2) which matmuls use DRAM-sharded weights, which do not and why, and what candidate versions were measured.

---

## 1. Shard-advice usage (strict rubric)

**Rubric.**
- **A — used as-is:** advisor config shipped unchanged.
- **D — used-but-modified:** advisor config taken as the starting candidate; the shipped default is a *re-tuned derivative of it* (same program-config class / memory placement, only numeric params altered).
- **B — not-used:** advisor config is **not** the shipped default and the shipped default is **not derived from it**; the advisor config survives (if at all) only behind a non-default flag. Reason sub-tags: `incorrect`, `superseded-on-perf` (implemented + measured correct, but an unrelated faster option shipped), `algorithm-change`.
- **C — could-not-use:** advisor emitted no valid config / hard technical blocker.

**Decision rule for B vs D (applied uniformly):** the shipped default is *derived* (D) only if it keeps the advisor's program-config class and memory placement and merely re-tunes numbers. If the shipped default uses a different program-config class or memory placement (e.g. a DRAM-sharded matmul instead of the advisor's L1 `MatmulMultiCoreReuseMultiCast1DProgramConfig`, or an independent `legacy_32core` residual/norm grid) and the advisor's exact config exists only behind a non-default flag, it is B/superseded-on-perf.

**Total advice items: 21** (`report.json` `final_choices=21`; plus 18 reshards + 1 spill as metadata, and 1 op the advisor itself declared unfixable).

| Advice item / group (indices) | Advisor recommendation | Verdict | Reason | Evidence (snippet + source) |
|---|---|---|---|---|
| reshape ×2 (0, 20), nlp_create_qkv_heads_decode (3), rotary_embedding ×2 (6, 7), slice_static ×4 (4, 5, 8, 9), SDPA-decode (10) | l1/interleaved, height-sharded, dram/interleaved layouts | **A** | used-as-is | Shipped `optimized_decoder.py` uses the same DRAM/interleaved + height-sharded layouts; composite SDPA + exact RoPE-row slices retained. |
| QKV matmul (2) | `matmul_multi_core_reuse_multi_cast_1d @11x8`, L1 width-sharded 1x80 | **B** | superseded-on-perf | Shipped default `qkv_decode_program_config = _dram_matmul_program_config(...)` → `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`, DRAM-sharded weight. Advisor 1D config exists only under `decode_matmul_mode="shard_advisor"`. work_log: report chain 0.816962–0.819488 ms vs 0.793291 ms DRAM. |
| O matmul (12) | `...1d @11x9`, L1 width-sharded 1x96 | **B** | superseded-on-perf | `o_decode_program_config = _dram_matmul_program_config(...)`. work_log: "Whole report residual candidate measured 0.816962 ms; DRAM-sharded selected." |
| gate matmul (15) | `...1d @11x10`, L1 width-sharded 1x103 | **B** | superseded-on-perf | `gate_decode_program_config = _dram_matmul_program_config(...)`. work_log: "Exact BFP4 advisor candidate measured 0.803918 ms; selected 24-core DRAM path measured 0.793291 ms." |
| up matmul (16) | `...1d @11x10`, L1 width-sharded 1x103 | **B** | superseded-on-perf | Up shares `gate_decode_program_config` (DRAM-sharded). Separate-family DRAM won vs packed (0.799328 vs 0.968070 ms). |
| down matmul (18) | `...1d @11x9`, L1 width-sharded 1x96 | **B** | superseded-on-perf | `down_decode_program_config = _dram_matmul_program_config(...)`. Aligned down input improved 0.804331→0.799328 ms; DRAM selected. |
| RMSNorm ×2 (1, 14) | block_sharded 1×11 on 11 cores | **B** | superseded-on-perf | Shipped default norm is block-sharded on the 32-core `residual_grid`; advisor 11-core norm only under `advisor_residual_mode="report"`. work_log: "Full `report_sharded_inputs` chain measured 0.819488 ms; phase-specific DRAM-matmul input grids selected." |
| residual add ×2 (13, 19) | width_sharded 1×96 on 96 cores | **B** | superseded-on-perf | Shipped adds use 32-core `residual_memory_config` (`advisor_residual_mode="legacy_32core"`, default); advisor 96-core chain only under `report`. |
| SiLU multiply (17) | width_sharded 1×103 | **B** | superseded-on-perf | Shipped multiply is width-sharded on the DRAM MLP-gate grid (24-core BFP4); advisor 103-core layout only under `report`. |
| nlp_concat_heads_decode (11) | dram/interleaved (no valid config emitted) | **C** | blocker | `report.json` `unfixable_ops`: `TT_FATAL … Input tensor must be sharded`. Decoder manually re-shards the input (`_prepare_decode_heads`) as a workaround. |

### Counts (strict rubric)

| Verdict | Count | of 21 |
|---|---|---|
| A used-as-is | 10 | 47.6% |
| B not-used | 10 | 47.6% |
| C could-not-use | 1 | 4.8% |
| D used-but-modified | 0 | 0% |

**B reason breakdown:** superseded-on-perf **10**, incorrect 0, algorithm-change 0.

**Note vs first pass:** an earlier pass labeled the 10 heavy ops **D** ("seeded then modified"). Under the strict rubric they are **B/superseded-on-perf**: the advisor configs were run and measured correct but were *not* the basis of the shipped code — the shipped defaults are an independent DRAM-sharded matmul family plus a `legacy_32core` residual/norm chain, and the advisor's exact configs survive only as flagged regression controls (`decode_matmul_mode="shard_advisor"`, `advisor_residual_mode="report"`, `advisor_matmul_input_mode="report_sharded"`).

---

## 2. DRAM-sharding analysis

**Shipped default:** `decode_matmul_mode="dram_sharded"`, `precision_policy="all_bfp4_lofi"` (BFP4 projection weights, LoFi fidelity; KV cache BFP8), separate gate/up family. A DRAM-sharded matmul is identified by `_dram_matmul_program_config()` → `ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with a `_dram_sharded_memory_config()` weight (`BufferType.DRAM`, `WIDTH_SHARDED`).

**All five in-scope decode matmuls use DRAM-sharded weights.** Falcon3-10B is dense (no MoE), so there are no expert/router matmuls; lm-head is out of scope.

| Matmul | Final decode layout | DRAM-sharded? | Versions tested (measured) | Why this choice |
|---|---|---|---|---|
| QKV (fused, packed) | DRAM-sharded, BFP4/LoFi, target-32 grid | **Yes** | Advisor L1 `1D @11x8` (report chain 0.816962–0.819488 ms; advisor all-BFP4 0.803918 ms); DRAM all-BFP4 24-core **0.793291 ms** | DRAM-sharded is the whole-layer winner; QKV kept packed (splitting adds 2 weight reads/launches). |
| O (o_proj) | DRAM-sharded, BFP4/LoFi, 32-core residual grid | **Yes** | Advisor `1D @11x9` width-sharded output/residual 0.816962 ms; DRAM O selected | Selected as part of the DRAM whole-layer winner; boundary reshard required by head helper + residual grid. |
| gate_proj | DRAM-sharded, BFP4/LoFi, 24-core (BFP4) / 48 (BFP8/BF16) | **Yes** | Advisor `1D @11x10` 0.803918 ms; packed DRAM BFP4 0.968070 ms; separate DRAM 0.799328/0.793291 ms; 16-core 0.809437 ms; 12-core & 6-core OOM | Separate-family DRAM BFP4 at 24 cores is fastest; packed loses to separate. |
| up_proj | DRAM-sharded (shares gate program config), BFP4/LoFi | **Yes** | Same sweep as gate; packed vs separate crossed | Separate family retained; fused SiLU into the multiply. |
| down_proj | DRAM-sharded, BFP4/LoFi, 24-core, input aligned to gate-output grid | **Yes** | Advisor `1D @11x9`; unaligned-down 0.804331 ms → aligned 0.799328 ms; DRAM selected | Aligning product/down shard removes an extra reshard. |

**Why some matmuls are *not* DRAM-sharded:** none — every dominant decode matmul is DRAM-sharded. The only "not DRAM-sharded" placements in the layer are *non-matmul* ops (RoPE/head transforms height-sharded in L1, SDPA + KV cache DRAM-interleaved, residual/norm on L1 sharded grids), which are not matmuls.

**Precision sweep (selected `all_bfp4_lofi`):** all-BFP4 LoFi **0.793496 ms** beat BFP8 (1.008006 ms), MLP-BFP4/attn-BFP8 (0.812733–0.814243 ms), attn-BFP4/MLP-BFP8 (0.999123 ms), and all-BFP4 attn-HiFi2 (0.805026 ms). Real-weight decode PCC ≥ 0.99932. KV cache stays BFP8 (PCC above bar, less persistent storage).

**Grid sweep (DRAM MLP core target):** 24-core (BFP4) selected — 0.793291 ms, reconfirmed 0.793096 ms over a 200-replay A/B vs 48-core 0.799102 ms and 16-core 0.809437/0.809589 ms. 12-core and 6-core targets failed with hard L1-capacity errors (gate/up `in0_block_w=8`/`16` requested 1,584,896 / 2,861,824 bytes vs 1,572,864 available). BFP8/BF16 precisions default to 48 cores (a BFP8 run reusing the BFP4 24-core geometry hit a repeatable L1 clash).

**Roofline:** the five projection weights are ~135.8 MB in BFP4; at Blackhole P300c 512 GB/s that is a 0.26528 ms DRAM-bandwidth floor ≈ 32.1% of the 0.826 ms decode wall — consistent with a DRAM-bandwidth-bound decode, which is exactly why DRAM-sharded weight streaming wins over L1 width-sharding here.

---

## Summary

The advisor was run as a mandatory gate and its full report was implemented and measured, but for Falcon3-10B the shipped decode layer takes an **independent DRAM-sharded matmul family + `legacy_32core` residual/norm chain** that beat every advisor candidate on measured whole-layer latency (0.793 ms vs 0.804–0.819 ms). Under the strict rubric: **A=10, B=10 (all superseded-on-perf), C=1, D=0** of 21 items. The advisor's contribution that reached production was the topology/layout scaffolding (reshapes, head ops, RoPE, slices, SDPA placement, 10 items); its compute-heavy L1 width-sharded proposals were correct but out-tuned and now live only as flagged regression controls. **All five dominant decode matmuls (QKV, O, gate, up, down) are DRAM-sharded** — none is left non-DRAM-sharded — because decode is DRAM-bandwidth-bound and DRAM-sharded BFP4 weight streaming on a 24-core grid is the measured optimum.
