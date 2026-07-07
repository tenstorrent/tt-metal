# Rewrite vs In-Place Emit Optimization — Decode Differences

Two agentic decode optimizations of **the same** tt-forge emit for
`meta-llama/Llama-3.1-8B-Instruct`, compared:

- **LOCAL** = the **rewrite** — `forge-functional-decoder` rewrites the emit into a fresh module, then `optimize` tunes it
- **GRAPH** = the **in-place emit** tune — `optimize` only, on the codegen-emitted graph (`ttnn-models`)

| | LOCAL (`forge-functional-decoder`) | GRAPH (in-place emit optimization) |
| --- | --- | --- |
| Where | `models/autoports/meta_llama_llama_3_1_8b_instruct` | [`ttnn-models` branch `…decode-opt`](https://github.com/svuckovicTT/ttnn-models/tree/mvasiljevic/agentic-research-llama-3.1-8b-decode-opt/meta-llama/Llama-3.1-8B-Instruct/model/graph_0) |
| **Pipeline** | **`forge-functional-decoder` → `optimize`** | **`optimize` only** (no functional-decoder stage) |
| Effect | rewrote the emit into a clean, fused, single-layer TTNN module, then tuned it | legalized + tuned the codegen-emitted `model_ttnn.py` **in place** |

**Same starting point:** the emitted `graph_0/model_ttnn.py` is byte-identical for
both (verified), and it is **batch 32** (`model_pt.BATCH_SIZE = 32`).

> **Root cause of every difference below:** LOCAL runs the `forge-functional-decoder`
> skill *first*, which **rewrites** the emit into a clean single-layer module (drops
> codegen glue, retargets to single-user batch-1, scopes to one bare layer). GRAPH
> skips that stage and optimizes the **raw emitted graph directly**, so it keeps the
> glue, batch-32, the full model, and the LM head. Everything else — op counts, batch,
> scope, mechanism — follows from this one pipeline difference. (The GRAPH summary
> states it: *"No forge or functional-decoder bringup stage was run."*)

---

## 1. Scope — this is the #1 source of confusion

| | Runs on device | Batch |
| --- | --- | --- |
| **LOCAL** | **one bare decoder layer** (norm → attn → MLP → residuals) | 1 (tile-padded to 32) |
| **GRAPH (device / reduced)** | embedding → **one** layer → final norm → **LM head** | 32 |
| **GRAPH (e2e headline)** | **full 32-layer model** | 32 |

Both start from the **same emitted ttnn**. They diverge only because LOCAL runs one
extra step — `forge-functional-decoder` — before optimizing, and that rewrite changes
the scope (bare layer) and batch (1). So the *measured windows* end up different
(bare batch-1 layer vs batch-32 full-model window) and their absolute numbers aren't
directly comparable — but the origin is identical.

### Why LOCAL is only one layer (no full model)

LOCAL's `tt/` contains **only** `functional_decoder.py` and `optimized_decoder.py` —
a single decoder layer. No embedding, no LM head, no 32-layer stack (verified: no
`lm_head` / `embed_tokens` / model class anywhere in the autoport code).

This is because the `.agents` bringup pipeline has 10 stages and this run executed
**only stages 01–02**:

```
01 forge-functional-decoder   ← ran (produces ONE decoder layer)
02 optimized-decoder          ← ran (optimizes that ONE layer)
03 multichip-decoder
04 optimized-multichip-decoder
05 full-model                 ← would assemble embedding + N layers + norm + LM head (NOT run)
06 optimized-full-model       ← (NOT run)
07 datatype-sweep · 08 vllm · 09 optimized-vllm · 10 tti-release  (NOT run)
```

- **When it became single-layer:** at stage 01 — `forge-functional-decoder`'s scope is
  a single decoder layer by design.
- **Why it stayed single-layer:** the run stopped after stage 02; the full-model
  assembly (stage 05/06) was never invoked.

GRAPH never decomposed the emit, so it was a full model the whole time — which is why
only GRAPH has a full-model e2e.

---

## 2. Why LOCAL is batch 1 (not 32 like the emit)

By design of the skills, not inherited from the emit:

- `optimize/SKILL.md` line 14: *"Optimize primarily for **batch-1 single-user latency**."*
- `forge-functional-decoder/SKILL.md` builds a generic single-sequence layer and
  intentionally drops the emit's *"batch-32 / cache-128 reshape glue"*.

GRAPH kept the emit's batch-32 serving workload; LOCAL retargeted to single-user latency.

---

## 3. Performance (measured on this Wormhole)

### Decode device time — note each row measures a *different scope*

| | What the μs covers | Baseline | Final | Δ |
| --- | --- | ---: | ---: | ---: |
| **LOCAL** | **one decoder layer only** — no embedding, no final norm, **no LM head**; batch 1 | 1,817 μs (BFP8/BFP8) | 1,229 μs (BFP4/BFP4) | −32% |
| **GRAPH** | **whole model with `num_layers=1`** — embedding + **one** layer + final norm + **LM head**; batch 32 | 8,435 μs | 5,019 μs | −40% |

Neither is the full 32-layer model. The Δ within each row is valid (same scope,
baseline→final), but the absolute μs across rows are **not comparable** — GRAPH's
window includes the one-time embedding + LM head that LOCAL's bare layer doesn't. The
next table strips those out for a true layer-to-layer comparison.

### Apples-to-apples: decoder layer only (strip embedding + LM head from GRAPH)

| Decoder layer only | Device time |
| --- | ---: |
| LOCAL final (batch 1, BFP4/BFP4) | 1,229 μs |
| GRAPH layer (batch 32, BFP8 attn / BFP4 MLP) | ≈1,946 μs |

GRAPH's layer (≈1.95 ms) does **32× the tokens** of LOCAL's (≈1.23 ms). The raw
5,019 μs only looked ≈4× bigger because **the LM head alone is 2,941 μs (58%)** of
that window — an op LOCAL's layer doesn't contain.

### Optimization mechanism (opposite!)

- **LOCAL got faster by moving fewer bytes** — BFP4 weights. Roofline actually
  *dropped* 42.2% → 31.6%.
- **GRAPH got faster by moving bytes more efficiently** — DRAM width-sharded QKV at
  unchanged precision. Roofline *rose* 28.2% → 47.2%.

### End-to-end (full model) status — only GRAPH has a real one

| | Full-model e2e | How |
| --- | --- | --- |
| **GRAPH** | **measured: 0.0791 s/step** warmed trace (≈404 TPS aggregate, ≈12.6 tokens/s/user) | `main.py` runs the **full 32-layer model, batch 32**, and wall-clock-times `execute_trace` + `synchronize` (all layers + embedding + LM head + host dispatch) |
| **LOCAL** | **none measured** | single decoder layer only — no 32-layer assembly, embedding, or LM head |

The `1.289 ms × 32 ≈ 41 ms` full-model figure quoted for LOCAL elsewhere is a naive
estimate (batch 1, excludes embedding/LM head), **not a measurement**. So GRAPH's
device numbers have a real full-model e2e behind them; LOCAL's do not.

---

## 4. Op counts — why LOCAL has far fewer ops

Reduced windows: **LOCAL 37 ops** vs **GRAPH 69 ops**. Two reasons:

1. **Scope (8 ops):** GRAPH's window has embedding (5) + final norm/LM-head/tail (3);
   LOCAL's doesn't.
2. **Per-layer glue (24 ops):** layer-to-layer it's **LOCAL 37 vs GRAPH 61**.

### Per-layer breakdown

| Op group | LOCAL | GRAPH | Note |
| --- | ---: | ---: | --- |
| Core math (matmul, norm, rotary, paged-cache, SDPA, concat-heads) | 13 | 14 | 1:1 **except** GRAPH's +1 op below |
| — of which Matmul | 5 | 6 | both have QKV/O/gate/up/down; GRAPH adds a tiny `64×32×32` FP32 RoPE-position matmul (LOCAL precomputes RoPE tables at setup) |
| Layout conversions (Interleaved↔Sharded, Reshard) | 6 | 20 | **+14** — codegen inserts them between ops |
| Codegen-only glue (Typecast, Repeat, Tilize/Untilize, Concat, Ternary, CreateHeads) | 0 | 13 | FP32 position path, batch broadcast, mask select |
| Manual head/RoPE (Slice, Transpose) | 10 | 2 | LOCAL builds heads by hand |
| Unary / BinaryNg / Reshape / Copy | 8 | 12 | |
| **Total per layer** | **37** | **61** | |

**Takeaway:** the *math* is the same (13 vs 14: the only extra is GRAPH's tiny FP32
RoPE-position matmul; LOCAL precomputes those tables at setup). The whole difference is that the
forge-emitted graph carries layout/format-conversion glue between every op, while
LOCAL keeps consistent memory layouts and fuses decode ops.

### Full model (32 layers)

| | Per layer | 32-layer decoder stack | + embed/final/LM head |
| --- | ---: | ---: | --- |
| **LOCAL** | 37 | 1,184 | not implemented (single-layer scope) |
| **GRAPH** | 61 | 1,952 | + ≈8 ≈ **1,960 total** |

Across the decoder stack LOCAL runs **≈39% fewer ops** — all glue, not math.
(GRAPH's extra ops are mostly cheap, 1–3 μs each; op-to-op gap 69 μs vs 40 μs.)

---

## 5. Decode optimizations — every change, graded

Every decode change each run **actually tried**, from the trial CSVs
(`optimized_decoder/*.csv`, LOCAL) and GRAPH's summary. Only LOCAL's decode path is
included (prefill excluded). Grades:

- 🟢 **great** — big win, kept
- 🟡 **ok** — small/moderate win, kept
- ⚪ **enabling** — bring-up/correctness fix, not a perf change
- 🔴 **rejected** — tried and reverted

Each number is that run's own decode metric (LOCAL = traced-decode host ms; GRAPH =
reduced-window / full-step). ⚠️ LOCAL and GRAPH numbers are **not comparable** across
runs (different scope — see section 3).

### 5a. LOCAL (rewrite) — decode changes

| Change | Decode effect |
| --- | --- |
| 🟢 BFP4 **MLP** weights (LoFi) | **1.845 → 1.373 ms (−25.6%)** |
| 🟡 Packed decode QKV (vs separate) | 1.413 → 1.294 ms (−8.5%) |
| 🟡 BFP4 **attention** weights (LoFi) | 1.373 → 1.287 ms (−6.3%, on top of BFP4 MLP) |
| 🟡 Separate gate/up (vs packed) | 1.333 → 1.289 ms (−3.4%) |
| 🟡 `down_proj` `in0_block_w=14` tune | 1.298 → 1.286 ms (≈−1% across sweep) |
| 🟡 BF8_B KV cache (vs BF16) | 1.285 → 1.276 ms (−0.7%) |
| 🟡 DRAM-sharded `down_proj` (vs interleaved) | kept, but **never A/B'd vs interleaved** — gain unquantified |
| 🔴 Separate **decode** QKV | +9% slower than packed |
| 🔴 Packed gate/up | +3.5% slower than separate |
| 🔴 HiFi2 fidelity (attn and/or MLP) | no gain or slower |
| 🔴 BF16 KV cache | slower than BF8_B |
| 🔴 `down_proj` `in0_block_w` 28 / 56 | 28 slower; 56 L1 circular-buffer clash |
| 🔴 L1 input movement | 1.283 → 1.307 ms (+1.9%) |

### 5b. GRAPH (in-place emit) — decode changes

| Change | Decode effect |
| --- | --- |
| 🟢 **QKV DRAM width-sharding** | QKV row 128 → 107 μs; roofline 28.2 → 47.2%; **0.0831 → 0.0791 s/step (−4.8%)** |
| 🟡 Tail reshape + redundant L1 copy removal | reduced window 8.435 → 5.048 ms, 71 → 69 ops |
| ⚪ Harness entrypoint fix (`main()`) | made the graph runnable at all |
| ⚪ PCC gate `==` → `>=` | correctness hygiene |
| ⚪ Shard / core-grid legalization for 8×8 | defines the runnable baseline |
| 🔴 Remove-only the unused logits copy | 0.0831 s before & after (no gain) |
| 🔴 Reuse DRAM logits tensor for reshape | 0.0837 vs 0.0831 s (slower) |
| 🔴 `create_qkv_heads` on sharded output | compile fail (4D shape contract) |
| 🔴 Pre-reshape sharded QKV `[1,1,32,6144]` | broke rotary (saw seq len 8, not 1) |
| 🔴 MLP-down DRAM-sharding | regressed reduced replay to 20.8 ms |

### 5c. How many iterations before aborting a rejected idea

Both runs mostly treated rejection as a **one-shot A/B**: measure the candidate once,
and if it doesn't beat the baseline, revert and move on. Neither re-tuned a losing idea
in the hope of rescuing it — with two exceptions, both on GRAPH's side.

| Rejected idea | Iterations before abort |
| --- | --- |
| LOCAL: separate decode QKV / packed gate/up / L1 movement | 1 each (single trial row, reverted) |
| LOCAL: HiFi2 fidelity, BF16 cache | measured once each inside the 6-point `fidelity_cache` sweep — losing points not retried |
| LOCAL: `down_proj in0_block_w` 28 / 56 | measured once each inside the `4,7,8,14,28,56` sweep |
| GRAPH: remove-only logits copy / reuse DRAM logits tensor | 1 run each, reverted |
| **GRAPH: sharded-QKV head creation** | **2** — direct `create_qkv_heads` (compile fail) → pre-reshape `[1,1,32,6144]` (broke rotary) → abandoned |
| **GRAPH: MLP-down DRAM-sharding** | **<1** — aborted after one reduced run, *"rejecting it without a full run"* |

**Takeaway:** the only genuine "try → fail → try again → abort" sequence in either run
is GRAPH's two-step attempt to feed sharded QKV into head-creation. LOCAL's multi-point
sweeps (fidelity, down geometry) look like many trials, but each losing point was
measured exactly once — it swept a grid rather than iterating on a failure.

### 5d. Named but never actually tried

- **LOCAL:** did not attempt QKV DRAM-sharding at all (only `down_proj`).
- **GRAPH:** **o-proj DRAM-sharding** was named in the log as remaining advice but
  **never implemented or measured**; **precision lowering** was considered and
  **declined without any trial**.

### 5e. What only one side ever explored

| Only LOCAL explored | Only GRAPH explored |
| --- | --- |
| <ul><li>precision sweep BFP8 → BFP4</li><li>fidelity sweep LoFi vs HiFi2</li><li>KV-cache dtype BF16 vs BF8_B</li><li>packed vs separate QKV / gate-up</li><li>the whole prefill path + K/V geometry</li><li>`down_proj` `in0_block_w` sweep</li></ul> | <ul><li>DRAM width-sharding the QKV projection</li><li>`create_qkv_heads` on sharded output</li><li>pre-reshaping sharded QKV</li><li>graph-tail cleanup (logits reshape / L1 copy)</li><li>emit legalization for the 8×8 grid</li><li>harness fixes (entrypoint, PCC gate)</li></ul> |

**The pattern:** LOCAL explored the *numerical* axes (precision, fidelity, cache
dtype) + a whole prefill path; GRAPH explored *layout/sharding + emit-cleanup* axes.
Each left the other's headline lever completely untouched.

---

## 6. Which decode optimization was more successful?

### The two headline wins

#### 🟦 LOCAL → BFP4 weight precision (LoFi)

| | |
| --- | --- |
| **What** | Dropped attention + MLP weights BFP8 → **BFP4** at Low-Fidelity matmul |
| **Why it works** | Decode is one token, so every projection is a skinny matmul that is **DRAM-bandwidth bound on weights**. Halving weight bytes cuts the dominant cost directly. |
| **Impact** | ≈26% of the decode win from BFP4 MLP alone; PCC still 0.99999 |
| **Why GRAPH didn't** | It optimized **directly on the emit** and treated emitted precisions as a correctness contract → **declined BFP4 as "unsafe"**. Optimized *around* precision, not *through* it. |

#### 🟩 GRAPH → DRAM width-sharded QKV projection

| | |
| --- | --- |
| **What** | Width-sharded the QKV weights **across DRAM banks** instead of interleaved |
| **Why it works** | The skinny decode matmul pulls weights from all banks in parallel → higher effective DRAM bandwidth on the largest single projection |
| **Impact** | GRAPH's dominant lever (bulk of its ≈40% reduced-window win) |
| **Why LOCAL didn't** | Kept QKV **packed / DRAM-interleaved** and got bandwidth relief from BFP4. DRAM-sharding **regressed** when tried on MLP-down, so LOCAL generalized "keep interleaved" and never re-tried it on QKV. |

> **Mirror image:** each took the lever the other refused — LOCAL went *through*
> precision and away from sharding; GRAPH went *through* sharding and away from
> precision. The wins are **non-overlapping**, so **BFP4 (LOCAL) + DRAM-sharded QKV
> (GRAPH)** stacked on the full model would very likely beat either alone.

### Scorecard

| Axis | Winner | Why |
| --- | --- | --- |
| Raw % on its own baseline | LOCAL (−32% device layer) vs GRAPH (−4.8% e2e / −40% reduced window) | LOCAL's bigger % partly reflects a worse BFP8/BFP8 start with precision headroom GRAPH's emit had already spent |
| Biggest single lever captured | **LOCAL** | BFP4 precision is the larger decode lever; GRAPH left it untouched |
| Per-layer decode efficiency | **LOCAL** | BFP4 win + ≈39% leaner op graph (37 vs 61 ops/layer) |
| Complete, deployable decode | **GRAPH** | real full 32-layer batch-32 model, measured e2e (0.0791 s/step), **PCC 1.0** |
| Correctness margin | GRAPH (1.0) vs LOCAL (0.99999) | both excellent; GRAPH exact because it kept precision |

### Verdict

- **Most effective single lever → LOCAL.** BFP4 hit decode's real weight-bandwidth
  bottleneck for −32% per layer plus a leaner graph — but it's **proven only on one
  batch-1 layer**.
- **Most successful finished result → GRAPH.** A smaller, precision-safe lever, but on a
  running **full 32-layer batch-32 model** with measured e2e and exact PCC.
- **Why they diverged:** opposite constraints — LOCAL was free to change precision on a
  fresh module; GRAPH had to preserve the emit.
- **Best move:** stack them — **BFP4 (LOCAL) + DRAM-sharded QKV (GRAPH)** on the full
  model would very likely beat either decode alone.

---

## 7. One-paragraph summary

Both start from the identical batch-32 emit. GRAPH runs **`optimize` only, directly on
the emitted graph** (no `forge-functional-decoder` stage), so it stays a
batch-32 full model; its decode device time is dominated by the one-time LM head and
by layout-conversion glue that codegen emits between ops, and it wins ≈40% mainly via
DRAM-sharded QKV at unchanged precision. LOCAL rebuilds a single decoder layer as a
clean, single-user (batch-1) module — same core math but ≈40% fewer ops (no
embedding/LM head, and far less layout glue) — and wins ≈32% mainly via BFP4
precision. Once matched to a bare decoder layer, the two are comparable in device
time, but LOCAL is batch-1 while GRAPH serves 32 users per step. The single reason for
all of this is that LOCAL rewrites the emit via `forge-functional-decoder` first,
while GRAPH optimizes the emitted graph directly.

---

*Evidence: `optimized_decoder/tt_perf_report_decode.txt` (LOCAL final),
`tt_perf_report_baseline_decode.txt` (LOCAL baseline),
`tt_perf_report_GRAPH_branch_reduced.txt` (GRAPH). Fuller detail in
`DECODE_OPTIMIZATION_COMPARISON.md`.*
