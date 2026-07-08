# Rewrite-bs32 vs Experiment-17 — Optimized-Decoder Stage Comparison (Llama-3.1-8B-Instruct)

Comparing this repo's **rewrite-bs32** optimized decoder against the [experiment-17 `meta_llama_llama_3_1_8b_instruct`](https://github.com/tenstorrent/agentic-research/tree/main/experiment-17/code/llama31-8b/snapshots/evidence-final-20260615T2301Z/models/autoports/meta_llama_llama_3_1_8b_instruct) autoport (`evidence-final-20260615T2301Z`) from `tenstorrent/agentic-research`.

This is a **like-for-like comparison of the single-chip `optimized_decoder` stage only** — one Llama decoder layer, batch 1, on **N300 Wormhole 1×1**. Both sides ran `forge-functional-decoder`/`functional-decoder` → `optimize`. (EXP17 also has later multichip / full-model stages; those run on T3K 1×8 and are out of scope here.)

- **LOCAL (rewrite-bs32)** — branch `mvasiljevic/llama-bs32-rerun`.
- **EXP17** — the `optimized_decoder` stage (repo commit `86f8bc0`).

**The question this document answers:** LOCAL's optimized decode is ≈1,020 µs; EXP17's is 750 µs on the same hardware and workload. *Why did LOCAL's optimization process leave ≈270 µs on the table, and how do we make sure it doesn't next time?* The sections below build the answer: **Part I** sizes the gap, **Part II** localizes it op-by-op, **Part III** explains why LOCAL's process made the choices it did, and **Part IV** lists the exact fixes plus the skill/review changes that would force them next run.

---

# Part I — The gap we're investigating

## 1. TL;DR

| | LOCAL (rewrite-bs32) | EXP17 (optimized_decoder) |
| --- | --- | --- |
| Hardware | N300 1×1, one decoder layer | N300 1×1, one decoder layer |
| Decode batch | 1 | 1 |
| Attention weights | **BFP4** | **BFP8** |
| MLP gate/up/down | BFP4 | BFP4 |
| MLP fidelity | LoFi | LoFi |
| KV cache | BFP8 | BFP8 |
| Decode matmul sharding | **down only** DRAM-sharded | **all 5** DRAM-sharded |
| Activations kept in L1 (vs bounced to DRAM) | **only the MLP tail** (residual + post-norm, 32-core); attention half runs in DRAM → 8 DRAM↔L1 bounces | **the whole layer** (norm→attn→residual→MLP→output stay L1-sharded) → 2 tiny bounces |
| Decode ops (single layer) | **38** | **19** |
| Decode device time | **≈1,020 µs** | **750 µs** |
| Decode roofline / DRAM util | ≈36% | **72–80%** (208–232 of 288 GB/s) |
| Decode PCC (real weights) | 0.99999 | 0.99949 |

**One line:** EXP17's decode is **≈27% faster than LOCAL (750 vs ≈1,020 µs) *even though it keeps attention at the higher BFP8 precision*** — because it DRAM-shards **every** decode matmul and keeps activations resident in L1 (72–80% of peak DRAM BW), whereas LOCAL leans on aggressive BFP4 precision but only shards the `down` matmul (≈36% roofline). Sharding, not precision, is the bigger lever here.

---

# Part II — Where the gap comes from (op-level diagnosis)

## 2. Rough breakdown — the gap in three buckets

Before the op-by-op detail, here is the whole 270 µs gap (LOCAL 1,020 − EXP17 750) sorted into three causes. **Buckets A and B are ops both sides run, where LOCAL just chose a worse config/layout; bucket C is ops LOCAL runs that EXP17 does not have at all** (fused or eliminated). The op-by-op tables in §3–§5 substantiate each row.

| Bucket | What it is | LOCAL cost | EXP17 cost | Lost |
| --- | --- | ---: | ---: | ---: |
| **A. Matmul configs** (same 5 matmuls, worse config) | QKV / O / gate / up left interleaved/1D instead of DRAM-sharded (`down` is a tie) — detail in §4 | 697 µs | ≈589 µs `est` | **≈108 µs** |
| **B. Other-op configs** (same op, worse layout) | Input RMSNorm run in DRAM on 1 core instead of L1-sharded (94 → ≈10 µs); post-attn norm, RoPE, cache, SDPA, residual adds are ties — detail in §5 | ≈130 µs | ≈100 µs `est` | **≈84 µs** |
| **C. Extra ops** (exist only in LOCAL, EXP17 fused them away) | QKV head glue = 6 Slice + 4 Transpose + 3 Reshape (≈86); 6 extra `Interleaved↔Sharded` bounces (≈16); separate SiLU `Unary` (≈14) — detail in §5 | ≈116 µs | ≈0 | **≈116 µs** |
| **Sum of identified causes** | | | | **≈308 µs** |
| *Estimate slack* | EXP17 per-op times are roofline estimates (its norm/RoPE/cache are not literally 0), so identified > net | | | *≈−38 µs* |
| **Net measured gap** | LOCAL 1,020 − EXP17 750 | | | **270 µs** |

**Read it as roughly one-third each:** ≈108 µs is pure **matmul config** (sharding), ≈84 µs is **one other op's config** (the input norm's placement), and ≈116 µs is **work EXP17 simply doesn't do** (head glue + layout churn + un-fused activation). So it is *not* that LOCAL's ops are individually slow — it leaves the big four matmuls and the input norm on non-sharded layouts (A+B ≈ 192 µs) **and** runs ~2× as many ops (C ≈ 116 µs) because it hand-built the head path instead of using fused ops. EXP17's `Attention1D` / `RMSNorm1D` modules fix all three at once: they DRAM-shard the matmuls (A), shard both norms (B), and fuse the heads/activation (C).

> The bucket µs are **approximate attribution**: LOCAL's side is measured exactly, but EXP17's per-op times are roofline estimates, so identified causes (≈308 µs) overshoot the net gap (270 µs) by the estimate slack. The *direction and rough magnitude* are solid — matmul sharding (A) and the fused head path (C) are the two dominant levers, with the input norm (B) a close third.

## 3. Op-by-type counts (single layer, batch-1 decode)

| Op type | LOCAL (38 ops) | EXP17 (19 ops) |
| --- | ---: | ---: |
| Matmul | **5** (QKV 102µs SLOW, O 58µs SLOW, gate 202µs SLOW, up 201µs SLOW, down 134µs ✅ — **1 of 5 optimized**) | **5** (QKV, O, gate, up, down — **all 5 DRAM-sharded ✅**) |
| RMSNorm (LayerNorm) | 2 (input 94µs, post-attn 10µs sharded) | 2 (input + post-attn, sharded) |
| Interleaved↔Sharded transitions | **8** (7× I2S + 1× S2I) | **2** (1× S2I 3µs + 1× I2S 1µs) |
| Slice | 6 | 0 (fused head ops) |
| Transpose | 4 | 0 (fused head ops) |
| ReshapeView | 3 | ≈0 |
| RotaryEmbedding | 2 | ≈2 |
| PagedUpdateCache | 2 | ≈2 |
| SdpaDecode | 1 (13µs) | 1 |
| NLP create/concat QKV heads | 1 (concat) | ≈2 (fused create + concat) |
| Residual add (BinaryNg) | 3 | ≈2 |
| SiLU (Unary) | 1 | ≈1 |
| **Layout + head glue subtotal** | **≈21** (slice + transpose + reshape + 8 layout transitions) | **≈2** |

LOCAL runs **2× the ops (38 vs 19)** for the *same* math. The entire gap is **layout/head glue**: LOCAL rebuilds the QKV heads from primitive `Slice`/`Transpose`/`ReshapeView` and bounces `Interleaved↔Sharded` **8 times**; EXP17 keeps activations width-sharded in L1 end-to-end, uses fused head ops, and has only **2** tiny layout transitions (and no tilize/untilize/copy/host at all). Combined with only 1-of-5 vs 5-of-5 matmuls optimized, this is exactly why LOCAL sits at 38% DRAM / 1,020 µs and EXP17 at 72–80% DRAM / 750 µs.

> ⚠️ EXP17's per-op tracy CSVs were **not committed** to the snapshot; its counts beyond the published totals (**19 device ops, 0 host ops, all 5 matmuls optimized, 2 layout transitions, 0 tilize/untilize/copy**) are reconstructed from the documented forward structure and marked `≈`. LOCAL's counts are exact from `tt_perf_report_decode_gate_up_geometry.txt`.

## 4. Matmul-by-matmul — config, what was tried, final perf

**Data availability:** LOCAL's per-matmul device times are **exact** (from `tt_perf_report_decode_gate_up_geometry.txt`). **EXP17 did *not* commit its per-op decode tracy CSVs** — only the aggregate (`decode 750 µs, 19 ops, all 5 matmuls ✅ Optimized, matmuls DRAM-bound at 208–232 GB/s`) plus a per-matmul **weight-byte accounting**. So EXP17's per-matmul times below are a **weight-read roofline estimate** (bytes ÷ ~220 GB/s measured), flagged `est`, not measured.

The five decode matmuls (batch 1, M tile-padded to 32). **Status key:** ✅ **optimized** (at the DRAM-sharded ceiling) · 🟡 **partially optimized** (a config was tuned but it still ends `SLOW`, bandwidth left on the table) · ❌ **not attempted** (left on the default interleaved config, no matmul tuning tried) · 🔴 a stronger config was tried and reverted.

| Matmul | Dims (K→N) | LOCAL precision / config | LOCAL device time / status | EXP17 precision / config | EXP17 time (est) / status |
| --- | --- | --- | ---: | --- | ---: |
| **QKV** (packed) | 4096 → 6144 | **BFP4**, **default interleaved (no program_config)** | **102 µs · 🟡 SLOW** — packing+precision tuned, config not sharded (45% DRAM) | **BFP8**, DRAM-sharded (via `Attention1D`) | ≈114 µs `est` · ✅ |
| **O** proj | 4096 → 4096 | **BFP4**, default interleaved | **58 µs · ❌ SLOW** — no config tuning at all (52% DRAM) | **BFP8**, DRAM-sharded (via `Attention1D`) | ≈76 µs `est` · ✅ |
| **gate** (FF1) | 4096 → 14336 | BFP4, **1D multicast** (`MatmulMultiCoreReuseMultiCast1D`, 64-core, `in0_block_w=4`, subblock `1x7`) | **202 µs · 🟡 SLOW** — geometry-tuned but 1D, not DRAM-sharded (52% DRAM) | BFP4, **DRAM-sharded** (`in0_block_w=4`, `per_core_N=14`, 32-core) | ≈133 µs `est` · ✅ |
| **up** (FF3) | 4096 → 14336 | BFP4, same 1D config as gate | **201 µs · 🟡 SLOW** (52% DRAM) | BFP4, DRAM-sharded (same as gate) | ≈133 µs `est` · ✅ |
| **down** (FF2) | 14336 → 4096 | BFP4, **DRAM-sharded** (`MatmulMultiCoreReuseMultiCastDRAMSharded`, `in0_block_w=14`) | **134 µs · ✅ Optimized** (76% DRAM); 🔴 `in0_block_w` 28/56 tried & reverted | BFP4, DRAM-sharded (`in0_block_w=7`, `per_core_N=4`, 32-core) | ≈133 µs `est` · ✅ |
| **Matmul subtotal** | | | **≈697 µs** (of 1,020) — 🟡×3, ❌×1, ✅×1 | | **≈589 µs** `est` (of 750) — ✅×5 |

### 4a. How each matmul got its config — what each side tried (narrative)

Detail behind the table above: the **A/B decisions, sweeps, and reverts** that produced each final config. (Same status key.)

- **QKV** (packed) — LOCAL 🟡 **partially optimized**. LOCAL A/B'd packed vs separate decode QKV and kept **packed** (separate was +9% slower), then dropped to **BFP4** — but left the matmul on the **default interleaved** config (never DRAM-sharded), so it stays **`SLOW`, 102 µs**. EXP17 kept **BFP8** and **DRAM-sharded** it inside `Attention1D` → **✅ Optimized**. Net: LOCAL moves fewer bytes (BFP4) on a worse config; EXP17 moves more bytes (BFP8) but DRAM-sharded — roughly a wash on time, and more accurate.
- **O** proj — LOCAL ❌ **not attempted**. LOCAL ran no sweep and applied no program config at all; BFP4 on the default interleaved config → **`SLOW`, 58 µs**. EXP17 used BFP8, DRAM-sharded via `Attention1D` → **✅ Optimized**.
- **gate / up** (FF1/FF3) — LOCAL 🟡 **partially optimized**, and the **dominant decode cost**. LOCAL A/B'd packed vs separate (kept separate) and ran a geometry sweep (64-core, `in0_block_w=4`, subblock `1x7`) that cut them from 258/270 → **202/201 µs** — but they stayed on a **1D-multicast** config (never DRAM-sharded), still **`SLOW`** at 52% DRAM. EXP17 **DRAM-sharded** them (`in0_block_w=4`, `per_core_N=14`) → **✅ Optimized**. This is the single biggest gap between the two.
- **down** (FF2) — ✅ **the one matmul both fully optimized**. LOCAL swept `in0_block_w` 14 / 28 / 56 (🔴 28 slower, 56 L1 clash — both reverted), kept **14** → **134 µs, ✅ Optimized** at 76% DRAM. EXP17 also DRAM-sharded it (`in0_block_w=7`, `per_core_N=4`) → **✅ Optimized**.

**Takeaway:** the race is decided by **QKV / O / gate / up**. LOCAL left all four on interleaved/1D configs (`SLOW`, 45–52% DRAM) → 102+58+202+201 = **563 µs**; EXP17 DRAM-sharded all four (`Optimized`, ~75–80% DRAM) → an estimated ~456 µs. **down** is the only tie. LOCAL's BFP4-everywhere shrinks the bytes but the un-sharded configs waste the bandwidth; EXP17's DRAM-sharding wins even carrying BFP8 QKV/O.

## 5. Every *other* op — where the rest of the device time goes

The matmuls are ~697 µs of LOCAL's 1,020 µs. The remaining **323 µs is non-matmul**, and this is where the *op-count* gap (38 vs 19) turns into time. LOCAL times below are **exact** (from `tt_perf_report_decode_gate_up_geometry.txt`, one decode layer). EXP17 per-op times were **not committed**, so its column is **structural** — whether the op exists, is fused away, or is done on a cheaper (sharded) layout — with a roofline `est` where useful.

**Δ key:** 🔺 = LOCAL carries extra cost EXP17 avoids (a gap source) · ➖ = roughly tied.

| Op (non-matmul) | LOCAL count · exact time | EXP17 (structural) | Δ / why |
| --- | ---: | --- | --- |
| **Input RMSNorm** | 1 · **94 µs** (DRAM, 1 core) | 1 · ≈10 µs — `RMSNorm1D`, **sharded** | 🔺 **≈84 µs** — LOCAL left the pre-attn norm interleaved in DRAM on a single core |
| **Post-attn RMSNorm** | 1 · 10 µs (sharded) | 1 · ≈10 µs (sharded) | ➖ both sharded |
| **Slice** (QKV head split) | 6 · **17 µs** | 0 — fused into `nlp_create_qkv_heads_decode` | 🔺 ≈17 µs — EXP17's `Attention1D` never materializes the slices |
| **Transpose** (head reorder) | 4 · **28 µs** | 0 — fused | 🔺 ≈28 µs |
| **ReshapeView** | 3 · **41 µs** | ≈0 — fused | 🔺 ≈41 µs |
| **Interleaved↔Sharded** | 8 · **20 µs** | 2 · ≈4 µs | 🔺 ≈16 µs — LOCAL bounces layouts between nearly every op |
| **RoPE** (`rotary_embedding`) | 2 · 32 µs | ≈2 (inside `Attention1D`) | ➖-ish — LOCAL wraps each in extra transpose/reshape (counted above) |
| **PagedUpdateCache** (K,V) | 2 · 27 µs | ≈2 | ➖ |
| **SdpaDecode** | 1 · 13 µs | 1 | ➖ |
| **NLPConcatHeadsDecode** | 1 · 2 µs | ≈1 | ➖ |
| **SiLU** (`Unary`) | 1 · **14 µs** | 0 — **fused** into the gate×up multiply | 🔺 ≈14 µs — EXP17 does `mul(gate, up, activations=[SILU])` in one op |
| **gate×up multiply** (`BinaryNg`) | 1 · 15 µs | 1 (SiLU fused in) | ➖ the multiply itself |
| **Residual adds** (`BinaryNg`) | 2 · 10 µs | ≈2 | ➖ |
| **Non-matmul subtotal** | **≈323 µs** (of 1,020) | **≈161 µs** `est` (of 750) | 🔺 **≈162 µs** |

This subtotal maps onto buckets **B** (the ≈84 µs input-norm placement) and **C** (the ≈116 µs of extra glue + un-fused SiLU) of the §2 rough breakdown; the matmul ≈108 µs is bucket **A** from §4.

---

# Part III — Why LOCAL's optimization didn't close the gap

## 6. Decode optimization mechanism — sharding vs precision

| Lever | LOCAL | EXP17 |
| --- | --- | --- |
| QKV / WO / gate / up matmuls | interleaved (packed QKV) | **DRAM-sharded** |
| down matmul | **DRAM-sharded** | **DRAM-sharded** |
| Decode activations/residual | residual + post-norm sharded (32-core); rest DRAM | **width-sharded L1 across norm/attn/residual/MLP/output** (no reshard churn) |
| RMSNorm | **post-attn sharded; input norm left DRAM (94 µs)** | **both norms sharded** |
| Layout transitions in decode | 8 (`Interleaved↔Sharded`) | only two tiny (`Sharded↔Interleaved`, 3 µs + 1 µs) forced by TTNN head APIs |
| Roofline / DRAM util | ≈36% | **72–80%** of 288 GB/s peak; all matmuls ✅ Optimized |
| Main win | BFP4 everywhere (fewer bytes) | full DRAM-sharding + L1-resident activations (bytes moved efficiently) |

**LOCAL got fast by moving fewer bytes; EXP17 got faster by moving bytes efficiently.** EXP17 proves the sharding lever dominates — it beats LOCAL while carrying higher-precision (BFP8) attention weights.

## 7. Every decode lever — who tried it, who kept it

Every decode optimization either side **actually tried**, cross-referenced. Grades: 🟢 great (big win, kept) · 🟡 ok (small/moderate win, kept) · 🔴 tried and rejected · ❌ never tried.

**Both try it?** column: ✅ = both tried **and reached the same outcome** · 🔀 = both tried **but reached different outcomes** · otherwise noted as *only LOCAL* / *only EXP17*.

| Lever | LOCAL (rewrite-bs32) | EXP17 (optimized_decoder) | Both try it? |
| --- | --- | --- | --- |
| Baseline BFP8 attn + BFP8 MLP | 🔴 baseline 1.845 ms, replaced by BFP4 | 🔴 1.140 ms, rejected slower | ✅ both, both rejected |
| **BFP4 attention weights** | 🟢 kept (1.373 → 1.287 ms, −6.3%) | ❌ never tried — kept attention **BFP8** | **only LOCAL** |
| BFP4 gate/up (FF1/FF3) | 🟢 kept | 🟢 kept | ✅ both kept |
| BFP4 down (FF2) | 🟢 kept (all MLP BFP4) | 🔴 BFP8-down tried (1.032 ms, slower) → 🟢 kept BFP4-down (0.989 ms) | ✅ both kept BFP4; only EXP17 A/B'd it |
| LoFi MLP fidelity (reject HiFi2/HiFi4) | 🔴 HiFi2 rejected → 🟢 LoFi | 🔴 HiFi rejected → 🟢 LoFi | ✅ both, same conclusion |
| BFP8 KV cache | 🟢 kept (swept BF16 vs BF8_B) | 🟢 kept (no dtype sweep logged) | ✅ both kept; only LOCAL swept BF16 |
| BF16 KV cache | 🔴 slower than BF8_B | ❌ not swept | only LOCAL |
| DRAM-sharded **down** matmul | 🟡 kept (`in0_block_w=14`; 28 slower, 56 L1 clash) | 🟢 kept | ✅ both kept |
| **DRAM-shard QKV / O / gate / up matmuls** | ❌ **never tried** — stayed SLOW interleaved (45–52% DRAM) | 🟢 kept — **all 5 DRAM-sharded, ✅ optimized (72–80% DRAM)** | **only EXP17** ← decisive |
| Width-sharded L1 residual / RMSNorm | 🟡 kept 32-core (residual + post-norm + MLP boundary) — but **left the input RMSNorm DRAM (94 µs)** | 🟢 kept — **end-to-end** across norm/attn/residual/MLP/output, **both norms sharded** | 🔀 both, **different outcome** (LOCAL partial, EXP17 end-to-end) |
| L1 matmul input movement | 🔴 rejected (+slower) | 🔴 short-prefill MLP L1 input rejected (added `CopyDeviceOperation`, 2437 vs 2387 µs) | ✅ both tried, both rejected |
| Gate/up matmul geometry (64-core, `in0_block_w=4`, `1x7`) | 🟡 kept (258/270 → 202/201 µs) | ❌ not separately logged | only LOCAL |
| Packed vs separate **decode** QKV | 🟡 kept packed (separate 🔴 +9%) | ❌ not logged | only LOCAL |
| Packed vs separate gate/up | 🟡 kept separate (packed 🔴 slower) | ❌ not logged | only LOCAL |

**Reading the table:**

- **LOCAL's only unique *kept* lever is BFP4 attention** — and EXP17 deliberately declined it (kept BFP8 attention) yet is still faster, so that lever did not decide the race.
- **EXP17's decisive unique lever is DRAM-sharding the other four matmuls** (QKV/O/gate/up). LOCAL never tried it, so four of LOCAL's five decode matmuls stay `SLOW`/interleaved at 45–52% DRAM while EXP17's five are all ✅ at 72–80%.
- Everything they *both* explored (LoFi, BFP4 gate/up+down, BFP8 KV, DRAM-shard down, L1 input) they resolved the **same way**; the one 🔀 is sharded residual/norm, where LOCAL went partial and EXP17 went end-to-end. The divergence is mostly in what each *failed to try*.

## 8. Source-level differences in `tt/optimized_decoder.py`

The decisive structural choice: **EXP17 composes shared, already-optimized library modules; LOCAL hand-rolls the whole decoder from primitive `ttnn` ops.**

| Aspect | LOCAL (865 lines, hand-rolled) | EXP17 (739 lines, library-composed) | Device-perf consequence |
| --- | --- | --- | --- |
| **Attention** | built from primitives: `matmul` + `slice` + `reshape` + `permute` + `experimental.rotary_embedding` + `paged_update_cache` + `paged_sdpa_decode` + `nlp_concat_heads_decode` | delegates to shared **`Attention1D`** module | EXP17 gets fused heads, DRAM-sharded QKV/O and L1-sharded activations *for free*; LOCAL's manual head path is the 6 Slice + 4 Transpose + 3 Reshape + 8 layout-bounce glue ops |
| **RMSNorm** | raw `ttnn.rms_norm`; **input norm left DRAM-interleaved** (`_decode_qkv`, line 701), only post-attn norm gets a sharded program config | **`RMSNorm1D(decode_in_sharded=True, decode_out_sharded=True)`** — both norms sharded | LOCAL's input RMSNorm = the **94 µs / 8.9%** report row; EXP17 shards both |
| **Decode matmul configs** | DRAM-sharded config only for **`down`** (`_decode_dram_matmul_program_config`, `in0_block_w≤14`); gate/up on a 1D config; **QKV/O on plain `ttnn.matmul` (no program_config)** | uniform `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` (`_dram_matmul_config`) for **all five** (QKV/O via `Attention1D`, gate/up/down via `_OptimizedMLP`) | **LOCAL 1/5 matmuls ✅**, QKV/O/gate/up `SLOW` at 45–52% DRAM; **EXP17 5/5 ✅** at 72–80% |
| **Weight memory** | only `down_proj_weight_dram_sharded` stored DRAM-width-sharded; the rest are default-interleaved `_to_device_tensor` | gate/up/down via `_dram_sharded_weight_memcfg` + QKV/O DRAM-sharded in `Attention1D` | enables the matmul row above |
| **Activation layout** | **DRAM for the whole attention half**, width-sharded L1 only for the residual→MLP tail (`_decode_residual_memory_config`) | **width-sharded L1 end-to-end** (`decode_residual_memcfg` reused across norm/attn/residual/MLP/output) | LOCAL bounces `Interleaved↔Sharded` **8×**; EXP17 has **2** tiny transitions, no tilize/untilize/copy |
| **RoPE** | manual `experimental.rotary_embedding` with `permute`-before/after + precomputed cos/sin tables | handled inside `Attention1D` (`rot_mats` passed in) | extra transpose/slice ops on LOCAL |
| **MLP** | hand-rolled `_mlp` (shared prefill/decode via flags) | self-contained `_OptimizedMLP` | both hand-rolled — the shared `mlp_1d` import failed in *both* checkouts |
| **Precision policy** | dict `optimization_profile`, BFP4 attn + BFP4 MLP, LoFi both | `OptimizedDecoderPolicy` dataclass, **BFP8 attn** + BFP4 MLP, LoFi MLP | §10 |
| **Batch** | runtime `batch_size = hidden_states.shape[-2]` (batch-agnostic) | `max_batch_size` param (default 1) baked into shard shapes | both measured batch 1 |

**Bottom line:** the file-level story is not "LOCAL tuned worse" so much as **LOCAL never adopted the shared `Attention1D`/`RMSNorm1D` modules that already bundle DRAM-sharded matmuls + fused heads + end-to-end L1 residency.** Hand-rolling attention from primitives left four matmuls interleaved, the input norm in DRAM (94 µs), and 8 layout bounces — exactly the 38-op / 38%-DRAM / 1,020 µs profile. EXP17's library composition is the 19-op / 72–80%-DRAM / 750 µs profile. (LOCAL couldn't use the shared MLP either — that import fails in both trees — but it *could* have used `Attention1D`/`RMSNorm1D`, which EXP17 did.)

## 9. Batch — both optimize decode at batch 1

- **EXP17** optimizes/measures decode at **batch 1** (M tile-padded to 32).
- **LOCAL** preserves batch 32 in the **functional** layer (real-weight PCC 0.99999805) but the `optimize` stage still tunes/measures decode at **batch 1** (`optimize/SKILL.md`).

So batch-1 decode is the common target; neither measures batch-32 decode device perf.

## 10. Precision — LOCAL is *more* aggressive

| Weight / tensor | LOCAL | EXP17 |
| --- | --- | --- |
| Attention QKV / WO | **BFP4** | **BFP8** |
| MLP gate / up | BFP4 | BFP4 |
| MLP down (FF2) | BFP4 | BFP4 |
| MLP mul intermediate | BF16 | BFP8 |
| KV cache | BFP8 | BFP8 |
| Matmul fidelity | LoFi | LoFi (HiFi rejected for BFP4 MLP) |
| Decode PCC (real, 1 layer) | 0.99999 | 0.99949 |

LOCAL drops **attention** to BFP4; EXP17 deliberately keeps attention at **BFP8**. Yet EXP17 is still faster (Part II), because it recovers bandwidth through **sharding** rather than precision. Both single-layer PCCs clear the 0.995 bar comfortably; LOCAL's slightly higher number is the nominal benefit of its more aggressive quantization, not a perf advantage.

## 11. Optimization method / process

| | LOCAL | EXP17 |
| --- | --- | --- |
| Search style | granular A/B sweeps on one layer (precision ladder, packed vs separate QKV/gate-up, cache dtype, down geometry, L1 movement) | strategy-first (shard every matmul + conservative attention precision), precision A/B with evidence |
| Precision decision | BFP4-everything (aggressive) | BFP8 attention + BFP4 MLP + LoFi; HiFi2/HiFi4 explicitly rejected with evidence |
| Rejected-with-evidence | yes | yes — e.g. short-prefill MLP L1 input (added a `CopyDeviceOperation`, 2437 vs 2387 µs → rejected); HiFi for BFP4 MLP (LoFi faster, PCC ok → rejected) |

## 12. Context — how the single-chip decoder matured (exp-5 → exp-17)

The single-chip decoder EXP17 is measured against got materially better between experiments, same base recipe. This is the target LOCAL is being held to:

| | exp-5 (`claude-wh-lb-80`) | **exp-17** |
| --- | ---: | ---: |
| Single-chip decode device time | 1,055 µs | **750 µs** (**−29%**) |
| Attention weights | BFP8 | BFP8 |
| `down` (FF2) weights | **BFP8** (kept high) | **BFP4** (dropped) |
| MLP fidelity | HiFi2 | **LoFi** |
| Decode matmul sharding | all DRAM-sharded | all DRAM-sharded |
| Decode PCC | 0.9999970 | 0.9995 |

exp-17 improved by pushing precision harder **where it's safe** — `down`→BFP4 and MLP kernels→LoFi — while *keeping attention at BFP8* and keeping the full DRAM-sharding. PCC dropped from 0.9999970 to 0.9995 but stayed above the 0.995 bar. Net: −29% decode latency. Note that **full DRAM-sharding was already present in exp-5** — it is the stable foundation of this line of work, which makes LOCAL's un-sharded QKV/O/gate/up the clear outlier.

---

# Part IV — Closing the gap: what to change and how to ensure it next time

## 13. Exact changes that would make LOCAL match EXP17

Ordered by measured/estimated decode saving. Items 1–5 are what separate LOCAL's 1,020 µs from EXP17's 750 µs, and **all five are delivered at once by adopting `Attention1D` + `RMSNorm1D`** — which is exactly how EXP17 got them.

| # | Change | Est. decode saving | Mechanism / evidence |
| --- | --- | ---: | --- |
| 1 | **DRAM-shard QKV, O, gate, up** (all four still interleaved/1D) | **≈108 µs** | §4 — moves them from 45–52% DRAM (`SLOW`) to ~75–80% (`✅`), like `down` already is |
| 2 | **Shard the input RMSNorm** (currently DRAM, 1 core) | **≈84 µs** | §5 — the single largest non-matmul row (94 → ≈10 µs), same treatment already applied to the post-attn norm |
| 3 | **Replace primitive head plumbing with `nlp_create_qkv_heads_decode`** | **≈86 µs** | §5 — removes 6 Slice + 4 Transpose + 3 Reshape (fused away in EXP17) |
| 4 | **Carry width-sharded L1 activations end-to-end** (attention half too) | **≈16 µs** | §5 — cuts 8 `Interleaved↔Sharded` bounces down to 2 |
| 5 | **Fuse SiLU into the gate×up multiply** (`mul(..., activations=[SILU])`) | **≈14 µs** | §5 — removes the separate `Unary` op |
| — | *(free correctness margin)* move attention back to **BFP8** | 0 (≈ wash) | §10 — sharding makes the precision drop unnecessary; buys PCC headroom |

Estimated combined effect: closing 1–5 would bring LOCAL from ≈1,020 µs into EXP17's 750 µs neighborhood. **The dominant two are item 1 (DRAM-shard the four matmuls) and the head-glue/input-norm cluster (items 2+3)** — together ≈278 µs, essentially the whole gap.

## 14. Why the process didn't do them (from the work log + the `optimize` skill)

**The `optimize` skill already mandates every one of these changes.** The gap is *execution and enforcement*, not missing guidance. Concretely, from `.agents/skills/optimize/SKILL.md`:

- **"DRAM-sharded decode matmuls"** is a checklist item, and the matmul rules say *"If `tt-perf-report` says a matmul is DRAM-bound and it is not DRAM-sharded, trying DRAM-sharded matmul is mandatory."* Item 1 above was required and skipped for 4 of 5 matmuls.
- **OPT-003** requires carrying the sharded residual layout **through the input RMSNorm**, not just the post-attention path — item 2.
- The checklist requires *"Used SDPA and other optimized composite ttnn ops instead of hand-built attention primitives"* — item 3.
- **Final Audit** says *"No unnecessary `InterleavedToSharded`, `ShardedToInterleaved`, reshard, tilize, untilize…"* — item 4 (LOCAL has 8).
- **"Code Paths Worth Reading"** explicitly lists `models/common/modules/attention/attention_1d.py` and `rms_norm` — the very modules that bundle items 1–5.

What the work log shows actually happened:

- **Hand-rolled from primitives (step 3).** The agent built attention out of `matmul + slice + reshape + permute + rotary_embedding + paged_update_cache + sdpa + concat_heads` rather than composing `Attention1D`/`RMSNorm1D`. The skill *lists* those modules but does not *force* adoption, so the agent wrote its own — which is what created all the head glue and left the matmuls/norm on defaults.
- **Matmul tuning stopped at `down` + a 1D gate/up tweak.** Steps 6, 17–18 and the "Gate/Up Decode Geometry Follow-Up" only ever DRAM-sharded `down`; gate/up got a **1D-multicast** geometry sweep (`in0_block_w=4`), and **QKV/O never received a program config at all.** The mandatory "DRAM-shard when SLOW" step was simply not carried out for four of the five matmuls.
- **Input norm was explicitly left in DRAM.** The "Sharded Residual/Norm Follow-Up" scoped its fix to the **post-attention** residual/norm path only; the log states outright: *"The remaining 94 us LayerNorm row is the input RMSNorm before packed QKV, not the reviewed post-attention residual/norm path."* Nobody went back for it.
- **Stage-review passed it anyway.** `$stage-review` raised gate/up geometry and sharded residual as P1s, but **accepted a 1D geometry fix and a post-attn-only norm as "done"** and returned **`clean-pass`** — with 4/5 matmuls `SLOW`, the input norm at 94 µs, and 8 layout bounces still in the trace. None of those tripped a gate, so the run shipped.

**In one sentence:** LOCAL hand-rolled the decoder instead of composing the shared modules, its matmul/norm optimization only reached `down` and the post-attn norm, and stage-review signed off without enforcing the skill's own mandatory sharding / composite-op / layout-audit items.

## 15. How to assure next time — make the skill's mandates hard gates

The fixes are all about *enforcement*, since the guidance already exists:

1. **Prefer composition over hand-rolling.** Add a directive to `forge-functional-decoder` and `optimize`: *before* hand-building attention/norm, attempt to compose `Attention1D` / `RMSNorm1D` / `mlp_1d`; only hand-roll if they demonstrably don't fit the target, and record the exact blocker. EXP17 is the proof this path wins on both perf and op count. This single rule would have delivered items 1–5 automatically.
2. **Gate on the perf-report CSV, not prose.** Make `$stage-review` fail (not warn) when the final decode `tt-perf-report` shows **any material matmul row that is `SLOW` or not DRAM-sharded**. The skill already calls DRAM-sharding mandatory; turn it into a machine-checked gate over the committed CSV.
3. **Gate on norm placement.** Fail if the input **or** post-attention RMSNorm row is DRAM/1-core (OPT-003) — both must be sharded.
4. **Budget layout transitions.** Fail if decode `Interleaved↔Sharded` transitions exceed a small budget (e.g. > 3), or if any `tilize`/`untilize`/`reshard`/`copy` appears in the decode window (Final Audit already forbids these — enforce it).
5. **Require the operation-topology audit as a committed artifact**, and require that primitive `slice`/`transpose`/`reshape` head-building be replaced by the fused `nlp_create_qkv_heads_decode` (checklist "composite ops not hand-built primitives").
6. **Verify sharding reached the measured op, alongside dtype.** OPT-013 already forces a check that the *dtype* policy shows up in the runtime rows; extend the same row-level check to *"row is DRAM-sharded"* for every dominant decode matmul.

With gates 2–4 in place, LOCAL's run could not have reached `clean-pass`: 4/5 `SLOW` matmuls, a 94 µs DRAM input norm, and 8 layout bounces would each have blocked signoff and forced the remediation that closes the ≈270 µs gap.

## 16. Verdict

- **On single-chip decode, EXP17 wins outright: 750 µs vs LOCAL's ≈1,020 µs** — while using *higher* attention precision (BFP8). The difference is **full DRAM-sharding + L1-resident activations (72–80% DRAM BW)** vs LOCAL's shard-`down`-only (≈36%).
- **The decisive lever is DRAM-sharding all decode matmuls**, which LOCAL never tried; LOCAL's unique BFP4-attention lever did *not* decide the race.
- **LOCAL's aggressive BFP4 attention is a false economy:** it raises nominal single-layer PCC but does *not* make LOCAL faster than EXP17's BFP8-attention + fully-sharded decode.
- **The root cause is process, not knowledge:** the `optimize` skill mandates everything EXP17 did; LOCAL hand-rolled instead of composing the shared modules, and stage-review signed off before the mandatory sharding/norm/layout items were met (§14). Enforcing those mandates as hard gates (§15) is what closes the gap next time.
- **exp-5 → exp-17 maturation:** single-chip decode 1,055 → 750 µs (−29%) by dropping `down`→BFP4 and MLP→LoFi while keeping attention BFP8 and full sharding — on a foundation of full DRAM-sharding that was already standard in exp-5.

---

*Evidence — EXP17: `experiment-17/.../evidence-final-20260615T2301Z/.../doc/optimized_decoder/{README,work_log}.md` and `tt/optimized_decoder.py`. LOCAL: this branch's `optimized_decoder/{README,work_log}.md`, `optimized_decoder/tt_perf_report_decode_gate_up_geometry.txt`, `tt/optimized_decoder.py`, and `.agents/skills/optimize/SKILL.md`. Companions: `REWRITE_BS32_vs_INPLACE_EMIT_decode.md`, `REWRITE_BS1_vs_REWRITE_BS32_decode.md`.*
