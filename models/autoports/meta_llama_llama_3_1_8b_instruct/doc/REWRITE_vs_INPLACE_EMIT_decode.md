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

## 5. Decode optimizations, lever by lever

Scoped to **decode only** for LOCAL (its prefill path is excluded here), against
GRAPH's decode-only emit.

### Lever by lever

| Lever | LOCAL (decode) | GRAPH (decode) |
| --- | --- | --- |
| Precision | **BFP4** attn + MLP weights, LoFi — its dominant win | left as emitted (BFP8 attn/down/LM-head, BFP4 gate/up); BFP4 lowering explicitly declined |
| QKV projection | packed, DRAM-interleaved | **DRAM width-sharded** — its dominant win |
| MLP down matmul | DRAM-sharded, `in0_block_w=14` (kept) | tried DRAM-sharding → **rejected** (regressed to 20.8 ms) |
| Gate/up matmul | separate (unpacked) after A/B | left as emitted |
| KV cache | BF8_B paged | emitted paged cache |
| Attention | paged flash SDPA decode | emitted SDPA decode |
| Layout / glue | leaner hand-built single-layer graph | legalized emit; removed redundant logits reshape + L1 copy |

### Contribution to decode

| | LOCAL | GRAPH |
| --- | --- | --- |
| Headline metric | **−32%** device time per decoder layer | **−4.8%** full-model e2e (−40% on reduced 1-layer window) |
| Dominant lever | BFP4 precision (≈26% of decode from BFP4 MLP alone) | QKV DRAM-sharding |
| Op graph | ≈39% leaner (37 vs 61 ops/layer) | emitted op count, legalized |
| Scope proven | one decoder layer, batch 1 | full 32-layer, batch 32, measured e2e |
| Correctness | PCC 0.99999 (BFP4) | PCC 1.0 (precision preserved) |

### Rejected during decode search

| LOCAL | GRAPH |
| --- | --- |
| <ul><li>L1 input movement (slower)</li><li>packed gate/up</li><li>separate decode QKV</li><li>HiFi2</li><li>BF16 cache</li></ul> | <ul><li>remove-logits-copy-only (no gain)</li><li>reuse DRAM logits tensor (slower)</li><li>`create_qkv_heads` on sharded output (compile fail)</li><li>pre-reshape sharded QKV (broke rotary)</li><li>**MLP-down DRAM-sharding (regressed to 20.8 ms)**</li><li>**broad precision lowering (declined as unsafe)**</li></ul> |

### Explored by one, never even attempted by the other

Drawn from LOCAL's `work_log.md` trial artifacts and GRAPH's summary. This is about
what each **searched**, independent of what it kept.

**LOCAL explored, GRAPH never attempted**

- Weight precision sweep BFP8 → BFP4 (`precision_trials.csv`) — GRAPH declined precision lowering *without trialing* it
- Math-fidelity sweep LoFi vs HiFi2 (`fidelity_cache_trials.csv`)
- KV-cache dtype sweep BF16 vs BF8_B
- Projection topology A/B: packed vs separate QKV, and packed vs separate gate/up
- The **entire prefill path** + K/V core-geometry sweep (GRAPH's emit is decode-only)
- `down_proj` `in0_block_w` geometry sweep (14 vs 56)

**GRAPH explored, LOCAL never attempted**

- **DRAM width-sharding the QKV projection** (its headline win) — LOCAL only sharded `down_proj`
- `create_qkv_heads` on sharded output / pre-reshaping sharded QKV
- Graph-tail cleanup: removing redundant logits reshape + L1 copy, reusing the DRAM logits tensor
- Emit legalization for the 8×8 grid, entrypoint fix, PCC gate `==`→`>=`

**The asymmetry:** LOCAL explored the *numerical* axes (precision, fidelity, cache
dtype) + a whole prefill path; GRAPH explored *layout/sharding + emit-cleanup* axes.
The one lever each deliberately avoided is the other's headline win — precision (LOCAL
swept, GRAPH declined) and QKV sharding (GRAPH pushed, LOCAL never tried).

### The two headline wins don't overlap

Same tool (`tt-perf-report`) pointed both at DRAM-bound decode matmuls, but they
reached **opposite conclusions** on which projection benefits from DRAM-sharding:

- **GRAPH never touched precision** (declined BFP4) — that is LOCAL's biggest decode win.
- **LOCAL never DRAM-sharded QKV for decode** (kept it packed) — that is GRAPH's biggest win.

Precision (LOCAL) + QKV DRAM-sharding (GRAPH) are **complementary and non-overlapping**.

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
