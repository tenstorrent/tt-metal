# Rewrite-bs32 vs In-Place Emit Optimization — Decode Differences

Two agentic decode optimizations of **the same** tt-forge emit for
`meta-llama/Llama-3.1-8B-Instruct`, compared:

- **LOCAL (rewrite-bs32)** = the **rewrite** with the batch-preserving skill —
  `forge-functional-decoder` rewrites the emit into a fresh module **keeping the emitted
  batch 32**, then `optimize` tunes it. (Branch `mvasiljevic/llama-bs32-rerun`.)
- **GRAPH** = the **in-place emit** tune — `optimize` only, on the codegen-emitted graph
  (`ttnn-models`).

| | LOCAL (rewrite-bs32) | GRAPH (in-place emit optimization) |
| --- | --- | --- |
| Where | `models/autoports/meta_llama_llama_3_1_8b_instruct` (branch `…llama-bs32-rerun`) | [`ttnn-models` branch `…decode-opt`](https://github.com/svuckovicTT/ttnn-models/tree/mvasiljevic/agentic-research-llama-3.1-8b-decode-opt/meta-llama/Llama-3.1-8B-Instruct/model/graph_0) |
| **Pipeline** | **`forge-functional-decoder` → `optimize`** | **`optimize` only** (no functional-decoder stage) |
| Effect | rewrote the emit into a clean, fused, single-layer TTNN module **at the emitted batch 32**, then tuned it | legalized + tuned the codegen-emitted `model_ttnn.py` **in place** |

**Same starting point:** the emitted `graph_0/model_ttnn.py` is the same for both, and it
is **batch 32** (`model_pt.BATCH_SIZE = 32`).

> **What changed vs. the first rewrite:** the earlier rewrite
> (`REWRITE_vs_INPLACE_EMIT_decode.md`) retargeted the module to **batch 1**. This run
> uses the updated `forge-functional-decoder` skill, so the functional layer now
> **preserves the emitted batch 32** (see §2). The optimized-decoder perf, however, is
> **still measured at batch 1** — that half of the batch problem lives in a different
> skill. §2 explains exactly which.

---

## 1. Scope — this is still the #1 source of confusion

| | Runs on device | Batch |
| --- | --- | --- |
| **LOCAL functional layer** | one bare decoder layer (norm → attn → MLP → residuals) | **32** (tested + default) |
| **LOCAL optimized decode (perf)** | one bare decoder layer | **1** (tile-padded to 32) |
| **GRAPH (device / reduced)** | embedding → one layer → final norm → LM head | 32 |
| **GRAPH (e2e headline)** | full 32-layer model | 32 |

LOCAL still runs **one bare decoder layer** (no embedding, no LM head, no 32-layer
stack — the run executed only pipeline stages 01–02). GRAPH never decomposed the emit,
so it is a full model throughout — which is why only GRAPH has a full-model e2e.

The new wrinkle this run: LOCAL's **functional** layer is exercised at batch 32, but the
**optimized** decode was tuned and timed at batch 1. So LOCAL now has a batch-32-capable
module whose headline decode latency is still a batch-1 number.

---

## 2. The batch-size problem — half fixed, half not (and which skill caused each)

This run was specifically meant to stop the rewrite from collapsing the emitted batch 32
down to batch 1. The result is a **split**:

### ✅ Fixed: `forge-functional-decoder` now preserves batch 32

The functional-decoder stage carried the emitted batch through:

- `doc/functional_decoder/work_log.md`: *"Preserved the emitted workload batch size `32`
  as the default and tested shape."*
- `from_state_dict(..., max_seq_len=128, batch=32, ...)` — **default batch is 32**, read
  from the emit's `model_pt.BATCH_SIZE`.
- Input/output shape is `[1, batch, seq_len, 4096]`; **all prefill tests ran at batch
  32** (real-weight seq-16 PCC `0.9999980`, synthetic seq 16/64 PCC ≈0.998).
- The optimized decode is structurally batch-agnostic too: `optimized_decoder.py`
  derives `batch_size = hidden_states.shape[-2]` at runtime, so it *can* run batch 32.

This is the direct effect of the skill edit committed as *"forge-functional-decoder:
preserve emitted workload batch size."*

### ❌ Not fixed: `optimize` still targets batch-1 latency

Despite the module being batch-32-capable, the optimize stage tuned and measured decode
at **batch 1**:

- `doc/optimized_decoder/README.md`: *"Batch capability preserved: complete for batch 2
  page isolation; **batch 1 remains primary latency target**."*
- Every headline decode number (final **1.059 ms**) is batch 1 at prefix 16. Batch 2 is
  used only for a page-isolation correctness test; **batch-32 decode was never measured**.
- Decode residual/norm was sharded for a `[32, 128]` shard shape — i.e. the 32-row *tile
  padding of a single token*, not 32 real users.

**Which skill caused it:** `optimize/SKILL.md` line 14 —
*"Optimize primarily for **batch-1 single-user latency**."* The `forge-functional-decoder`
skill was corrected this run, so the remaining batch-1 behaviour is entirely the
`optimize` skill's directive. To carry batch 32 all the way into the optimized/measured
perf, that line must change too.

**Net:** the emitted batch 32 now survives into the **module**, but not into the
**optimization target**. GRAPH kept the emit's batch-32 serving workload end-to-end.

---

## 3. Performance (measured on this Wormhole)

### Decode — each row measures a *different scope*

| | What it covers | Baseline | Final | Δ |
| --- | --- | ---: | ---: | ---: |
| **LOCAL** (host traced decode) | one decoder layer, **batch 1** | 1.845 ms (BFP8/BFP8) | **1.059 ms** (BFP4/BFP4 + sharded residual/norm + gate/up geometry) | **−42.6%** |
| **LOCAL** (profiler device) | one decoder layer, batch 1 | — | ≈1,020 μs, 38 ops, roofline 36.1% | — |
| **GRAPH** (reduced device window) | whole model, `num_layers=1`, batch 32 (embed + 1 layer + final norm + LM head) | 8.435 ms / 71 ops | 5.034 ms / 69 ops | −40% |
| **GRAPH** (full-model e2e) | **full 32-layer model, batch 32** | 0.0831 s/step | 0.0791 s/step | −4.8% |

The Δ within each row is valid (same scope, baseline→final), but absolute values across
rows are **not comparable** — GRAPH's window includes the one-time embedding + LM head
and 32 real users; LOCAL's is a single batch-1 layer.

This run's LOCAL decode (1.059 ms) is **faster than the first rewrite's** (1.229 ms
device / ~1.28 ms host) because it additionally landed **sharded residual/RMSNorm** and
**gate/up geometry** tuning that the batch-1 run never reached.

### Optimization mechanism (still opposite)

- **LOCAL got faster by moving fewer bytes** — BFP4 weights (plus sharded
  residual/norm and gate/up geometry this run). Roofline stays low (36.1%).
- **GRAPH got faster by moving bytes more efficiently** — DRAM width-sharded QKV at
  unchanged precision. Roofline rose 28.2% → 47.2%.

### End-to-end (full model) status — only GRAPH has a real one

| | Full-model e2e | How |
| --- | --- | --- |
| **GRAPH** | **measured: 0.0791 s/step** (≈404 TPS aggregate, ≈12.6 t/s/user) | `main.py` runs the full 32-layer batch-32 model and wall-clock-times `execute_trace` + `synchronize` |
| **LOCAL** | **none measured** | single decoder layer only — no 32-layer assembly, embedding, or LM head |

---

## 4. Op counts — why LOCAL has far fewer ops

Reduced windows: **LOCAL 38 ops** vs **GRAPH 69 ops**. Same two reasons as before:

1. **Scope:** GRAPH's window has embedding + final norm/LM-head/tail; LOCAL's doesn't.
2. **Per-layer glue:** the forge-emitted graph carries layout/format-conversion glue
   between every op (Interleaved↔Sharded, Typecast, Tilize/Untilize, batch-broadcast
   Repeat, mask-select), while LOCAL keeps consistent memory layouts and fuses decode
   ops.

The *math* is essentially identical (QKV/O/gate/up/down matmuls, norms, RoPE,
paged-cache, SDPA, concat-heads); GRAPH's extra ops are cheap layout/codegen glue
(1–3 μs each). LOCAL's 38 (vs the first rewrite's 37) reflects the added sharded
residual/norm ops. Detailed per-op categorization mirrors
`REWRITE_vs_INPLACE_EMIT_decode.md` §4.

| | Per layer | 32-layer decoder stack | + embed/final/LM head |
| --- | ---: | ---: | --- |
| **LOCAL** | ≈38 | ≈1,216 | not implemented (single-layer scope) |
| **GRAPH** | 61 | 1,952 | + ≈8 ≈ **1,960 total** |

---

## 5. Decode optimizations — every change, graded

Every decode change each run **actually tried**. Only LOCAL's decode path is included
(prefill excluded). Grades: 🟢 great (big win, kept) · 🟡 ok (small/moderate win, kept) ·
⚪ enabling (bring-up/correctness, not perf) · 🔴 rejected. ⚠️ LOCAL and GRAPH numbers are
**not comparable** across runs (different scope — see §3).

### 5a. LOCAL (rewrite-bs32) — decode changes

| Change | Decode effect |
| --- | --- |
| 🟢 BFP4 **MLP** weights (LoFi) | **1.845 → 1.373 ms (−25.6%)** |
| 🟢 Sharded residual + post-attn RMSNorm (32-core) | 1.284 → 1.130 ms (−12%); post-attn RMSNorm 94 μs → 10 μs |
| 🟡 Gate/up geometry (64-core, `in0_block_w=4`, `1x7`) | 1.130 → 1.059 ms; gate/up rows 258/270 → 202/201 μs |
| 🟡 Packed decode QKV (vs separate) | 1.413 → 1.294 ms (−8.5%) |
| 🟡 BFP4 **attention** weights (LoFi) | 1.373 → 1.287 ms (−6.3%, on top of BFP4 MLP) |
| 🟡 Separate gate/up (vs packed) | 1.333 → 1.289 ms (−3.4%) |
| 🟡 `down_proj` DRAM-sharded, `in0_block_w=14` | kept; `28` slower, `56` L1 clash |
| 🟡 BF8_B KV cache (vs BF16) | 1.285 → 1.276 ms (−0.7%) |
| 🔴 Separate **decode** QKV | +9% slower than packed |
| 🔴 Packed gate/up | +3.5% slower than separate |
| 🔴 HiFi2 fidelity | no gain or slower |
| 🔴 BF16 KV cache | slower than BF8_B |
| 🔴 L1 input movement | +1.9% slower |

### 5b. GRAPH (in-place emit) — decode changes

| Change | Decode effect |
| --- | --- |
| 🟢 **QKV DRAM width-sharding** | QKV row 128 → 107 μs; roofline 28.2 → 47.2%; **0.0831 → 0.0791 s/step (−4.8%)** |
| 🟡 Tail reshape + redundant L1 copy removal | reduced window 8.435 → 5.034 ms, 71 → 69 ops |
| ⚪ Harness entrypoint fix (`main()`) | made the graph runnable at all |
| ⚪ PCC gate `==` → `>=` | correctness hygiene |
| ⚪ Shard / core-grid legalization for 8×8 | defines the runnable baseline |
| 🔴 Remove-only the unused logits copy | no gain |
| 🔴 Reuse DRAM logits tensor for reshape | slower |
| 🔴 `create_qkv_heads` on sharded output | compile fail (4D shape contract) |
| 🔴 Pre-reshape sharded QKV `[1,1,32,6144]` | broke rotary (saw seq len 8, not 1) |
| 🔴 MLP-down DRAM-sharding | regressed reduced replay to 20.8 ms |

### 5c. New in rewrite-bs32 vs the first rewrite

This run reached two optimizations the batch-1 rewrite never did — both surfaced by a
stage-review of leftover DRAM-interleaved decode ops:

- **Sharded residual + post-attention RMSNorm** across 32 cores (the biggest new win).
- **Gate/up matmul geometry** (64-core, `in0_block_w=4`, output subblock `1x7`).

Together they took decode from ≈1.28 ms to **1.059 ms** on top of the shared BFP4 policy.

---

## 6. Which decode optimization was more successful?

### The two headline wins (unchanged)

- **LOCAL → BFP4 weight precision (LoFi).** Decode is one token, so every projection is a
  skinny matmul **DRAM-bandwidth bound on weights**; halving weight bytes cuts the
  dominant cost. GRAPH declined BFP4 as "unsafe," optimizing *around* precision.
- **GRAPH → DRAM width-sharded QKV.** Pulls QKV weights from all DRAM banks in parallel;
  raises effective bandwidth on the largest projection. LOCAL kept QKV packed and never
  DRAM-sharded it.

The wins remain **non-overlapping** — BFP4 (LOCAL) + DRAM-sharded QKV (GRAPH) stacked on
the full model would very likely beat either alone.

### Scorecard

| Axis | Winner | Why |
| --- | --- | --- |
| Raw % on its own baseline | LOCAL (−42.6% host decode layer) vs GRAPH (−4.8% e2e / −40% reduced window) | LOCAL's bigger % partly reflects a worse BFP8/BFP8 start with precision headroom GRAPH's emit had already spent |
| Biggest single lever captured | **LOCAL** | BFP4 precision; GRAPH left it untouched |
| Per-layer decode efficiency | **LOCAL** | BFP4 + sharded residual/norm + gate/up geometry; ≈38 vs 61 ops/layer |
| Complete, deployable decode | **GRAPH** | real full 32-layer batch-32 model, measured e2e, **PCC 1.0** |
| Batch fidelity to the emit | **GRAPH** (fully) / LOCAL (partial) | LOCAL preserves batch 32 in the *module* but optimizes at batch 1; GRAPH serves 32 users per step end-to-end |
| Correctness margin | GRAPH (1.0) vs LOCAL (0.99999) | both excellent; GRAPH exact because it kept precision |

### Verdict

- **Most effective single lever → LOCAL** (BFP4), now compounded by sharded residual/norm
  and gate/up geometry for −42.6% — but still **proven only on one batch-1 layer**.
- **Most successful finished result → GRAPH** — a running full 32-layer batch-32 model
  with measured e2e and exact PCC.
- **Batch:** rewrite-bs32 fixed *half* the batch problem (the module keeps batch 32); the
  optimize stage still tunes at batch 1, so LOCAL's headline decode number is not a
  batch-32 serving number.
- **Best move:** stack the levers (BFP4 + DRAM-sharded QKV) **and** change
  `optimize/SKILL.md` line 14 so the optimized decode is measured at the emitted batch.

---

## 7. One-paragraph summary

Both start from the identical batch-32 emit. GRAPH runs `optimize` only, directly on the
emitted graph, so it stays a batch-32 full model and wins ≈40% (reduced window) / −4.8%
(full e2e) mainly via DRAM-sharded QKV at unchanged precision. LOCAL rebuilds a single
decoder layer as a clean module — and, thanks to the updated `forge-functional-decoder`
skill, **preserves the emitted batch 32** in that module (tested at batch 32, PCC
0.99999805). But the `optimize` stage still tunes and times decode at **batch 1**
(`optimize/SKILL.md` line 14), reaching 1.059 ms (−42.6%) via BFP4 precision plus new
sharded-residual/norm and gate/up-geometry wins. So the emitted batch now survives into
the module but not into the optimization target; closing that last gap requires editing
the `optimize` skill's batch-1 directive.

---

*Evidence: `optimized_decoder/README.md` and `work_log.md` (LOCAL rerun),
`optimized_decoder/tt_perf_report_decode_*` (LOCAL device rows),
`functional_decoder/work_log.md` (batch-32 preservation),
`tt_perf_report_GRAPH_branch_reduced.txt` + GRAPH summary (GRAPH). Companion:
`REWRITE_vs_INPLACE_EMIT_decode.md` (the earlier batch-1 rewrite).*
