# Forcing the emitted memory configs onto the optimized decode — what it recovers

An experiment that answers a narrow question raised by
[`REWRITE_BS32_vs_EXP17_llama.md`](REWRITE_BS32_vs_EXP17_llama.md) §46 (bucket **B**) and
[`LOCAL_vs_EXP17_L1_chain.md`](LOCAL_vs_EXP17_L1_chain.md): **the tt-forge emit
(`ttnn-models/.../graph_0/model_ttnn.py`) keeps the whole layer L1-sharded, while LOCAL's
`optimized_decoder` left the input RMSNorm and QKV/O matmuls DRAM-interleaved. If we just
force the emit's memory configs back on, how much of the gap closes, and does PCC hold?**

New code: `tt/optimized_decoder_force_emitted_configs.py`
(`OptimizedDecoderForceEmittedConfigs`, subclass of `OptimizedDecoder`) +
`tests/test_optimized_decoder_force_emitted_configs.py`. dtype (BFP4 weights), fidelity
(LoFi), and the paged-KV/head structure are unchanged, so every delta is **memory layout
only**.

---

## 0. Setup — the emit's configs are not runnable verbatim

The emit was generated for an **≥11-wide compute grid** (`CoreCoord(11, 9)` program grids,
`CoreRange` up to `CoreCoord(10, 9)`). The target here is **Wormhole B0, 8×8 compute grid
(64 cores), 12 DRAM banks**, so the emit's exact `CoreRangeSet` / grid / `per_core_N` **do
not fit** and cannot be applied verbatim — the same "core-grid legalization for 8×8" the
in-place GRAPH tune hit (`REWRITE_BS32_vs_INPLACE_EMIT_decode.md` §5b). So "force the
emitted configs" means: **keep the emit's layout *kind* (L1 `WIDTH_SHARDED` everywhere) and
1D-multicast program-config *style*, legalized to the 8×8 grid** (`per_core_N = N_tiles/64`;
`4096/6144/14336` all divide `32·64` exactly, so nothing is padded and no config number is
invented beyond the grid legalization — the `in0_block_w`/subblock formulas reuse the parent
decoder's own helpers). This is a faithful, minimal legalization, not a re-tune.

Two scopes are measured:

- **`all`** — force the emitted layout on **every** decode compute op (this *reverts* the
  optimizer's DRAM-sharded `down` and tuned gate/up to the emit's plain 1D style).
- **`unchanged`** — force the emitted layout **only** on ops the optimizer left at the
  functional/DRAM default (input RMSNorm, packed QKV, O projection, head-glue), and **keep**
  the optimizer's tuned ops (sharded residual/post-norm, tuned gate/up, DRAM-sharded `down`).
  This is the "fill the gaps the optimizer left, don't overwrite its wins" variant.

---

## 1. TL;DR

| | baseline `optimized_decoder` | force **unchanged** | force **all** |
| --- | ---: | ---: | ---: |
| Decode device time | **1,020 µs** | **954 µs** | 1,047 µs |
| Decode host (traced) | 1.059 ms | 1.051 ms | 1.121 ms |
| Device ops | 38 | 43 | 43 |
| DRAM util | 38.1 % | 40.4 % | 36.2 % |
| Real-weight decode PCC | 0.9999948 | **0.9999960** | 0.9999957 |
| Matmul subtotal (5 mm) | ≈697 µs | ≈724 µs | ≈844 µs |

**One line:** forcing the emit's configs **only where the optimizer left DRAM defaults**
(`unchanged`) recovers **≈66 µs (−6.5% device)** at equal PCC — *entirely* from the input
RMSNorm (94 → 9 µs), exactly as `REWRITE_BS32_vs_EXP17_llama.md`:46 predicted. Forcing them
**everywhere** (`all`) is a net **loss** (+27 µs), because the emit's 1D-multicast matmul
style is worse than the optimizer's DRAM-sharded `down` and tuned gate/up. **The emit's L1
layout wins on norms/residuals; it does *not* win on matmuls — that lever is DRAM-sharding,
which the emit does not use.**

---

## 2. Op-by-op (single-layer, batch-1 decode, exact device µs)

Rows the two scopes touch; `=` = kept from the optimizer.

| Op | baseline | unchanged | all | note |
| --- | ---: | ---: | ---: | --- |
| **Input RMSNorm** | **94** (DRAM, 1 core) | **9** (L1 WS, 32c) | **8** (L1 WS, 32c) | 🟢 the win — emit shards it; §5.1 of L1-chain |
| QKV matmul | 102 (interleaved) | 95 (1D WS L1) | 95 (1D WS L1) | ➖ ~neutral (emit 1D ≈ interleaved) |
| **O matmul** | **58** (interleaved) | **94** (1D WS L1) | **94** (1D WS L1) | 🔺 emit's 1D config is *worse* than the plain default |
| Post-attn RMSNorm | 10 (WS) | 10 = | 10 = | ➖ already sharded |
| gate matmul | 202 (tuned 1D) | 201 = | 246 (emit 1D) | `all` reverts the tuned geometry → +44 µs |
| up matmul | 201 (tuned 1D) | 200 = | 239 (emit 1D) | +38 µs in `all` |
| SiLU | 14 (`Unary`) | 14 = | 0 (fused in gate) | emit fuses SiLU into gate mm |
| **down matmul** | **134** (DRAM-sharded ✅) | **134** = ✅ | **171** (emit 1D) | `all` loses the DRAM-shard → +37 µs |
| residual adds | sharded | = | = | ➖ already sharded |
| **Device total** | **1,020** | **954** | **1,047** | |

Reading it:
- **`unchanged` = baseline − 85 µs (input norm) − 7 µs (QKV) + 36 µs (O) + a few extra
  reshards ≈ −66 µs.** The input norm alone is the whole win; O is a self-inflicted loss.
- **`all`** adds, on top of `unchanged`, the reverted MLP: gate/up 202/201 → 246/239 and
  `down` 134 → 171 (SiLU-fusion claws back only ~14 µs), so the matmul subtotal jumps
  697 → 844 µs and the layer ends *slower* than baseline.

---

## 3. Relation to the two source docs

### 3.1 `REWRITE_BS32_vs_EXP17_llama.md` line 46 (bucket **B**)

That row: *"Input RMSNorm run in DRAM on 1 core instead of L1-sharded (94 → ≈10 µs); …
≈84 µs"*. **Confirmed, essentially exactly:** forcing the emit's `WIDTH_SHARDED` input norm
takes it **94 → 9 µs** (predicted ≈10), i.e. **≈85 µs** recovered (predicted ≈84). This is
the single change that makes `unchanged` faster than baseline, and it needs **no new tuning
work** — the emit already carried the right layout, and the optimizer's own
`_decode_residual_norm_program_config` + `residual_memcfg` (already used for the *post-attn*
norm) is the legal 8×8 realization of it.

### 3.2 `LOCAL_vs_EXP17_L1_chain.md`

- **Fix #1 ("Shard the input RMSNorm … reuse `_decode_residual_norm_program_config` +
  `residual_memcfg`")** — this experiment does exactly that and measures the promised
  94 → ≈10 µs. ✅
- **Fixes #2/#3 (DRAM-shard QKV/O, keep gate/up L1 with DRAM-sharded weights)** — the emit
  does **not** implement these. Its QKV/O/gate/up/down are **1D-multicast over interleaved
  weights**, i.e. the *same family* as LOCAL's already-`SLOW` gate/up (52% DRAM). Forcing
  them proves the L1-chain doc's core nuance the hard way: QKV stays ~neutral (95 vs 102),
  **O regresses (58 → 94)**, and reverting the tuned gate/up + DRAM-sharded `down` (the
  `all` scope) *loses* 119 µs. The matmul bandwidth lever the docs attribute to EXP17 is
  **DRAM-sharding**, which is absent from the emit — so "just use the emitted configs"
  cannot reach EXP17's 750 µs. It recovers the norm (bucket B), not the matmuls (bucket A).

### 3.3 Net

The emit's value for this decoder is its **norm/residual L1 residency**, not its matmul
configs. The correct takeaway for `optimize`: **shard the input RMSNorm** (a free ≈85 µs the
emit hands you) and **keep DRAM-sharding the matmuls** (the emit does not) — do not adopt the
emit's 1D-multicast matmul style wholesale.

---

## 4. Correctness

Real-weight single-layer paged decode (prefix 17) PCC vs HF: **unchanged 0.9999960**,
**all 0.9999957**, baseline 0.9999948 — all ≥ 0.99; the layout change is PCC-neutral.
Synthetic BFP4/LoFi paged decode ≈0.964 (matches baseline's random-input band).

---

*Evidence — this branch: `tt/optimized_decoder_force_emitted_configs.py`,
`tests/test_optimized_decoder_force_emitted_configs.py`,
`doc/optimized_decoder_force_emitted_configs/tt_perf_report_decode_unchanged.txt`,
`.../tt_perf_report_decode_force_all.txt`, `.../perf_host_timings*.csv`. Baseline:
`doc/optimized_decoder/tt_perf_report_decode_gate_up_geometry.txt`. Emit:
`ttnn-models/meta-llama/Llama-3.1-8B-Instruct/model/graph_0/model_ttnn.py`. Companions:
`REWRITE_BS32_vs_EXP17_llama.md`, `LOCAL_vs_EXP17_L1_chain.md`,
`LOCAL_vs_EXP17_op_shapes_configs.md`.*
