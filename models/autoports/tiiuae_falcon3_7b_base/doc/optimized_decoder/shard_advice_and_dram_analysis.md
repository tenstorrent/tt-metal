# Shard-Advice Usage and DRAM-Sharding Analysis — tiiuae/Falcon3-7B-Base

**Model:** `tiiuae/Falcon3-7B-Base`
**Autoport dir:** `models/autoports/tiiuae_falcon3_7b_base`
**Artifacts analyzed:** `doc/optimized_decoder/shard_advise/report.json` (advisor input),
`tt/optimized_decoder.py` (shipped output), `doc/optimized_decoder/{work_log,README,AUTODEBUG,AUTOFIX}.md`
(narrative), and the optimize codex run log.

---

## 1. Summary

The shipped decoder runs **all five dominant projection matmuls as DRAM-sharded BFP4/LoFi**
(`decode_matmul_mode="dram_sharded"`, `precision_policy="all_bfp4_lofi"`). The shard-advisor's
L1 width-sharded 1D matmul family and its 11-core block-sharded norm / 96-core residual chain were
fully implemented, measured, and proven correct, but are retained only behind the non-default
`decode_matmul_mode="shard_advisor"` / `advisor_residual_mode="report"` flags because the DRAM-sharded
family (per OPT-004/OPT-015) won a like-for-like all-BFP4 comparison. All non-matmul topology/layout
advice (reshapes, slices, RoPE, QKV-head split, decode SDPA) was adopted unchanged.

---

## 2. Shard-advice usage (consistent rubric)

Rubric: **A** used-as-is (advisor config shipped unchanged) · **D** used-but-modified (advisor config is
the starting candidate; shipped config is a re-tuned derivative of it) · **B** not-used (advisor config
not in shipped code and nothing derived from it; sub-tagged *incorrect* / *superseded-on-perf* /
*algorithm-change*) · **C** could-not-use (no valid config / hard blocker).

Advisor `report.json`: `total_ops=24`, `final_choices=21` op layout items, plus 18 reshards, 1 spill,
1 unfixable op.

| Advice item(s) (report index) | Advisor recommendation | Verdict | Reason tag | Evidence |
|---|---|---|---|---|
| reshape (0, 20) | `l1/interleaved` | A | — | Shipped unchanged. |
| nlp_create_qkv_heads_decode (3) | `l1/height_sharded 32x1` | A | — | Height-sharded head split shipped as advised. |
| slice_static (4, 5) | `dram/interleaved` | A | — | Shipped unchanged. |
| rotary_embedding (6, 7) | `l1/height_sharded 32x1` | A | — | Shipped unchanged. |
| slice_static (8, 9) | `l1/interleaved` | A | — | Shipped unchanged. |
| scaled_dot_product_attention_decode (10) | `dram/interleaved` | A | — | Composite decode SDPA retained; "Maskless decode SDPA 0.782570 ms … Selected" (work_log). |
| **QKV matmul (2)** | L1 width-sharded 1D `@11x8` | **B** | superseded-on-perf | Shipped is `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (DRAM-sharded); advisor 1D config only under `shard_advisor` flag. "final default deliberately rejects the advisor matmul family after a like-for-like all-BFP4 cross … 0.768483 versus 0.773084 ms." |
| **O matmul (12)** | L1 width-sharded 1D `@11x9` | **B** | superseded-on-perf | Same DRAM-sharded family; advisor config behind flag. |
| **gate matmul (15)** | L1 width-sharded 1D `@11x10` | **B** | superseded-on-perf | Shipped split DRAM-sharded BFP4; "packed is 0.938405 ms versus 0.773380 ms split, so split wins." |
| **up matmul (16)** | L1 width-sharded 1D `@11x10` | **B** | superseded-on-perf | As gate. |
| **down matmul (18)** | L1 width-sharded 1D `@11x9` | **B** | superseded-on-perf | DRAM-48, down-input aligned: "0.773251/0.768429 ms … Aligned; removes one reshard." |
| **RMSNorm (1, 14)** | `l1/block_sharded 1x11` (11-core) | **B** | superseded-on-perf | Shipped default is coherent 32-core width-sharded (`advisor_residual_mode="legacy_32core"`); "Exact advisor residual and input modes are retained as controls at 0.786988/0.788957 ms; coherent 32-core residual is final." |
| **residual add (13, 19)** | `l1/width_sharded 1x96` (96-core) | **B** | superseded-on-perf | As norm; advisor 96-core residual behind `report` flag. |
| **gate·up multiply (17)** | `l1/width_sharded 1x103` | **B** | superseded-on-perf | Shipped multiply runs in the DRAM-48 MLP geometry, not the advisor's 103-wide L1 layout. |
| nlp_concat_heads_decode (11) | `dram/interleaved` (no valid config) | **C** | — | `unfixable_ops`: `TT_FATAL … Input tensor must be sharded`; advisor emitted no usable config; decoder supplies the required sharded input contract. |

### Counts (N = 21 op items)

| A used-as-is | B not-used | C could-not-use | D used-but-modified |
|---|---|---|---|
| **10** | **10** | **1** | **0** |

**B reason breakdown:** superseded-on-perf **10**, incorrect 0, algorithm-change 0.

> Note vs first pass: the 10 matmul/norm/residual/multiply items were previously tagged **D**. Under the
> strict rubric they are **B/superseded-on-perf** — the shipped configs are an independent DRAM-sharded /
> legacy-32-core implementation, not re-tuned derivatives of the advisor's L1 configs, which survive only
> behind non-default flags.

---

## 3. DRAM-sharding analysis (per matmul)

Shipped default: `decode_matmul_mode="dram_sharded"`, `precision_policy="all_bfp4_lofi"`. Each dominant
matmul uses `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` with 8-bank DRAM width-sharded
weights (`_dram_sharded_memory_config`, `BufferType.DRAM`, `WIDTH_SHARDED`), `per_core_M=1` for
batch-32 tile-padded decode.

| Matmul | Final layout | DRAM-sharded? | Versions tested (measured, batch 32 traced decode) | Why this choice |
|---|---|---|---|---|
| QKV (fused Q/K/V) | DRAM-sharded, BFP4/LoFi | **Yes** | Advisor L1 width-sharded 1D vs DRAM-48; BFP8 HiFi2/LoFi vs BFP4 | DRAM-48 all-BFP4 wins vs advisor (0.768483 vs 0.773084 ms b32; 0.644047 vs 0.652849 ms b1). |
| O proj | DRAM-sharded, BFP4/LoFi | **Yes** | Same family cross | Part of the winning coherent DRAM-48 all-BFP4 family. |
| gate proj | DRAM-sharded, BFP4/LoFi, split (not packed) | **Yes** | split vs packed; `in0_block_w=4, per_core_N=30` | Split 0.773380 ms vs packed 0.938405 ms → split. |
| up proj | DRAM-sharded, BFP4/LoFi, split | **Yes** | split vs packed | As gate. |
| down proj | DRAM-sharded, BFP4/LoFi | **Yes** | down-input unaligned vs aligned; `in0_block_w=30, per_core_N=4` | Aligned down-input 0.768429 vs 0.773251 ms, identical PCC; removes one reshard. |

**All five dominant matmuls are DRAM-sharded — none are excluded.** The advisor's L1 width-sharded 1D
family was the tested-and-rejected alternative (like-for-like all-BFP4).

**Not DRAM-sharded (deliberate):** the two RMSNorms, two residual adds, and the gate·up multiply stay
**L1 32-core width-sharded**. These are normalization / elementwise ops with no large streamed weight to
benefit from DRAM bandwidth; keeping them on a single coherent 32-core L1 chain minimizes reshards
("selected decode has four reshards per replay … 32-way residual chain; 32-way post-attention norm into
the 48-way MLP; and 48-way down output back into the 32-way residual").

### Precision sweep (all DRAM-48 geometry, batch 32 traced decode)

| Policy | Decode PCC | Decode latency | Decision |
|---|---|---|---|
| Functional BF16 | 0.99999972 | 1.797630 ms | Reference, slowest |
| DRAM BFP8 control | 0.99999995 | 1.134656 ms | Correct, slower |
| Attention BFP4 / MLP BFP8 | 0.99966209 | 0.969488 ms | MLP too slow |
| All BFP4, attention HiFi2 | 0.99905481 | 0.774677 ms | Fidelity control, slower |
| MLP BFP4 HiFi2 | — | 0.785517 ms | Slower than LoFi |
| **All BFP4 / LoFi (DRAM-48)** | 0.99904047 | **0.768483 ms** | **Selected** |

BFP4/LoFi selected for every dominant projection; attention BFP8 kept only as a higher-precision control.
Decode acceptance bar PCC ≥ 0.99 is met (0.99904).

---

## 4. Bottom line

- Advisor **topology/layout** advice (10/21 items): adopted verbatim (A).
- Advisor **matmul + norm + residual + multiply** advice (10/21 items): implemented, measured correct,
  then superseded on performance by the DRAM-sharded BFP4 / coherent-32-core family (B, superseded-on-perf);
  advisor configs retained behind non-default flags for regression A/B.
- **1/21** hard blocker: `nlp_concat_heads_decode`, advisor emitted no valid config (C).
- **DRAM-sharding pattern:** every dominant weight matmul (QKV, O, gate, up, down) is DRAM-sharded BFP4;
  norms/adds/multiply stay L1 32-core width-sharded by design (no weight to stream, residual-chain coherence).
