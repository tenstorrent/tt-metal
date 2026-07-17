# AGENT INDEX — Optimization Guidelines Quick Lookup

**Purpose:** If you already know what you need to optimize, read this file first.
It maps your task to the 1–2 files you need — skip the rest.

---

## ⛔ PERSISTENCE CONTRACT — DO NOT STOP UNTIL THE GOAL METRIC IS MET

You are running an **optimization loop**, not a one-shot task. The most common failure
is stopping at the **first signal of progress** (one sweep winner, one faster bucket, one
passing PCC) and declaring done while the goal is still unmet. Do not do this.

**1. Define "done" as a number, up front.** Restate the target at the start
(e.g. "≤ 4.30 ms wall" / "≥ 6.45 FPS" / "match the perf target in the task"). Re-read this
target at the top of **every** iteration — do not let it drift as context grows.

**2. Run the loop until the number is hit — then loop again:**
```
 ┌─> profile + bucket the ops          (09 §1–4)
 │   pick the current top bucket       (09 §4)
 │   apply the right lever             (02–06, 08)
 │   sweep → PCC gate → in-model       (07 §1)
 │   measure WALL time, median-of-3+   (07 §3)
 │   did we hit the target number?
 │      NO  → re-profile and loop  ────┘   (there is ALWAYS a new top bucket;
 │                                          fixing one promotes the next)
 └── YES → run FULL-MODEL PCC (07 §5) → only now may you stop
```

**3. You are NOT done when (these are traps, keep going):**
- ❌ a sweep found a faster config — it is **not validated in-model** yet (07 §1 step 4).
- ❌ one bucket got faster — **re-profile**; the next bucket is now the bottleneck.
- ❌ single-layer PCC passed — **full-model** PCC is not yet gated (07 §5).
- ❌ one experiment improved the metric — more levers remain untried.
- ❌ device time dropped but wall time didn't move — host-bound, not a shippable win (07 §7).

**4. You ARE done (stopping criteria — to avoid looping forever):**
- ✅ the target number is met **AND** full-model PCC passes, **OR**
- ✅ the top buckets are irreducible — at the math/bandwidth ceiling or only
  kernel-level C++ could help (this is the documented floor — **09 §8**), **OR**
- ✅ successive iterations yield **less than the noise floor** (~10–20 µs / under 1× the
  run-to-run noise — 07 §3): diminishing returns, record the negative result and stop.

**Rule of thumb:** stop on **observable state** (the measured number, full-model PCC, the
irreducible-floor), never on a feeling of "that looks done." If unsure → re-profile and
attack the next bucket.

---

## HOW TO USE THIS INDEX

1. Find your task in one of the sections below.
2. Note the **file(s)** and **section(s)** listed.
3. Read only those. Each section has a "Quick reference" table at the end for fast answers.

---

## TASK → FILE MAP

### You need to profile / find what's slow
| Task | Read | Section(s) |
|---|---|---|
| Capture a Tracy device profile | `09_PROFILING_AND_OP_ANALYSIS.md` | §1 |
| Generate a human-readable perf report | `09_PROFILING_AND_OP_ANALYSIS.md` | §2 |
| Find which ops consume the most device time | `09_PROFILING_AND_OP_ANALYSIS.md` | §3, §4 |
| Understand if an op is compute-bound vs DRAM-bound vs dispatch-bound | `09_PROFILING_AND_OP_ANALYSIS.md` | §2, §4 |
| Separate matmul families (QKV vs FF1 vs FF2) in the CSV | `09_PROFILING_AND_OP_ANALYSIS.md` | §5 |
| Identify a mystery op by its neighbors | `09_PROFILING_AND_OP_ANALYSIS.md` | §5 |
| Understand reshape/tilize/reshard buckets | `09_PROFILING_AND_OP_ANALYSIS.md` | §6 |
| Sanity-check a capture before trusting it | `09_PROFILING_AND_OP_ANALYSIS.md` | §7 |

---

### You need to run an experiment / sweep
| Task | Read | Section(s) |
|---|---|---|
| First time — understand the hardware limits (L1, grid, DST) | `01_FOUNDATIONS.md` | §1, §2, §3 |
| Set up compute kernel config (`packer_l1_acc`, fidelity, `fp32_dest_acc_en`) | `01_FOUNDATIONS.md` | §3, §4, §5 |
| Sweep matmul configs without producing bogus winners | `07_METHODOLOGY.md` | §2, §4, §9 |
| Establish a noise floor / know when a win is real | `07_METHODOLOGY.md` | §3 |
| Wire a change into the model safely (in-model validation loop) | `07_METHODOLOGY.md` | §1 |
| Gate a reduction-op change through full-model PCC | `07_METHODOLOGY.md` | §5 |
| Bound a Tracy capture to a single forward (signposts) | `07_METHODOLOGY.md` | §6 |
| Diagnose a win in device time that didn't move wall time | `07_METHODOLOGY.md` | §7 |
| A downstream guard silently undid your change | `07_METHODOLOGY.md` | §8 |

---

### You are tuning a specific transformer component

#### LayerNorm / RMSNorm / GroupNorm
| Task | Read | Section(s) |
|---|---|---|
| Choose sharded vs interleaved for norm | `02_NORMALIZATION.md` | §2 |
| Configure sharded norm (program config knobs) | `02_NORMALIZATION.md` | §3 |
| Set norm fidelity and `fp32_dest_acc_en` | `02_NORMALIZATION.md` | §4 |
| Fix the bf8b precision-compounding bug | `02_NORMALIZATION.md` | §5 |
| Fuse the residual add into norm | `02_NORMALIZATION.md` | §6 |
| Chain LN output into the next op (avoid reshards) | `02_NORMALIZATION.md` | §7 |
| Tune GroupNorm (Swin / spatial) | `02_NORMALIZATION.md` | §8 |

#### QKV Projection (matmul before attention)
| Task | Read | Section(s) |
|---|---|---|
| Fuse Q, K, V into one matmul | `03_QKV_PROJECTION.md` | §1 |
| Choose 1D vs 2D-mcast program config | `03_QKV_PROJECTION.md` | §2 |
| Set `in0_block_w` and subblock for QKV | `03_QKV_PROJECTION.md` | §3, §4 |
| Set fidelity for QKV matmul | `03_QKV_PROJECTION.md` | §3 |
| Choose head-split strategy (encoder vs decode) | `03_QKV_PROJECTION.md` | §5 |
| Add RoPE after QKV (LLM) | `03_QKV_PROJECTION.md` | §5b, §5c |
| Resharding between QKV and attention | `03_QKV_PROJECTION.md` | §6 |
| When to use `minimal_matmul` for QKV (usually don't) | `03_QKV_PROJECTION.md` | §8 |

#### Attention / SDPA
| Task | Read | Section(s) |
|---|---|---|
| Decide SDPA vs manual (height-sharded BMMs) | `04_ATTENTION_SDPA.md` | §1 |
| Configure manual (BMM) attention | `04_ATTENTION_SDPA.md` | §2 |
| Configure SDPA (flash kernel, chunk sizes) | `04_ATTENTION_SDPA.md` | §3 |
| Configure flash-decode (autoregressive) | `04_ATTENTION_SDPA.md` | §3b |
| Choose `exp_approx_mode` | `04_ATTENTION_SDPA.md` | §3c, §6 |
| DRAM-stage Q/K/V for long sequences | `04_ATTENTION_SDPA.md` | §4 |
| Remove an unnecessary Q/K/V typecast (the −13.7 ms trap) | `04_ATTENTION_SDPA.md` | §5 |
| Set SDPA compute kernel (`fp32_dest_acc_en`, fidelity) | `04_ATTENTION_SDPA.md` | §7 |
| Handle the attention mask (DRAM, `None` fast-path) | `04_ATTENTION_SDPA.md` | §8 |

#### MLP / FFN (FF1, FF2, attention output projection)
| Task | Read | Section(s) |
|---|---|---|
| Fuse activation (GELU/SiLU) into FF1 | `05_MLP.md` | §2 |
| Configure FF1 / FF2 block-sharded (small/mid batch) | `05_MLP.md` | §3 |
| Choose prefill (2D-mcast) vs decode (DRAM-sharded) matmul variant | `05_MLP.md` | §3b |
| Use `minimal_matmul` when 2D-mcast blows L1 (large batch) | `05_MLP.md` | §4 |
| Unlock `h·w ≤ 8` subblocks with `fp32_dest_acc_en=False` | `05_MLP.md` | §5 |
| Walk fidelity for MLP matmuls | `05_MLP.md` | §6 |
| L1 handoff FF1 → FF2 (large batch) | `05_MLP.md` | §7 |
| Tune attention-output projection | `05_MLP.md` | §8 |
| The matmul sweep method (any family) | `05_MLP.md` | §9 |

---

### You are reducing op count / fusing ops
| Task | Read | Section(s) |
|---|---|---|
| Fuse residual add into norm | `06_FUSION_AND_RESIDUALS.md` | §1 |
| Fuse activation into producing matmul | `06_FUSION_AND_RESIDUALS.md` | §2 |
| Fuse dtype cast into a reshard | `06_FUSION_AND_RESIDUALS.md` | §3 |
| Fuse adjacent unary ops with `unary_chain` | `06_FUSION_AND_RESIDUALS.md` | §4 |
| Pre-fuse weight-dependent ops at load time | `06_FUSION_AND_RESIDUALS.md` | §5 |
| Skip op-chains with config flags (`attn_mask=None`, etc.) | `06_FUSION_AND_RESIDUALS.md` | §6 |
| Cross-block sharded handoff (no I→S reshard at block boundary) | `06_FUSION_AND_RESIDUALS.md` | §7 |
| Know when op-count fusion does NOT help (device-bound) | `06_FUSION_AND_RESIDUALS.md` | §8 |

---

### You are working on a generative LLM (prefill / decode / KV-cache / multi-device)
| Task | Read | Section(s) |
|---|---|---|
| Understand prefill vs decode bottleneck difference | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §1 |
| Choose matmul variant by phase (2D-mcast vs DRAM-sharded) | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §2 |
| Set up and update the KV cache | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §3 |
| Prefill vs decode attention ops (different op names!) | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §4 |
| Configure Grouped-Query Attention (GQA) | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §5 |
| Add RoPE (fused rotary embeddings, decode regenerates cos/sin) | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §6 |
| Fracture weights across devices (n300/T3K vs TG) | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §7 |
| Minimize host–device round-trips in decode | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §8 |
| Norm weight DRAM layout trick (no padding) | `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | §9 |

---

## DECISION TREES

### "My model is slow — where do I start?"
```
1. Capture + bucket the ops  →  09 §1–4
2. Is top bucket a matmul?   →  fidelity walk first  →  05 §6 or 03 §3
   Is it a norm?             →  02 §4, then §2
   Is it SDPA?               →  04 §5, then §3
   Is it data movement?      →  09 §6, then 06 §3
   Is it many tiny ops?      →  06 §4–5 (op-count / fusion)
3. For each change: sweep → PCC gate → in-model → wall  →  07 §1
```

### "My matmul config crashes in-model but works standalone"
```
L1 CB clash?  →  01 §2  (compute the budget; use minimal_matmul or shrink in0_block_w)
Wrong grid?   →  01 §1  (hard-coded 8×8 on BH discards 40% cores)
packer bug?   →  07 §4  (packer_l1_acc=False makes times 3.5× wrong)
Downstream guard reverted it?  →  07 §8
```

### "My change passes single-layer PCC but fails full-model"
```
Reduction op (norm, softmax)?  →  07 §5  (depth-compounding error — needs HiFi2 + fp32_dest=True)
Matmul only?                   →  single-layer is usually fine; re-check the broadcast / shard spec
```

### "I'm adding a new encoder model (not LLM)"
```
1. Read 01_FOUNDATIONS  (hardware limits, knobs)
2. Read 02 §2 (norm layout decision)
3. Read 03 §1–2 (QKV config)
4. Read 04 §1 (SDPA vs manual)
5. Read 05 §2–3 (MLP fuse + config)
6. Run 07 §1 loop for each component
```

### "I'm adding a generative LLM"
```
1. Read 01_FOUNDATIONS
2. Read 08 §1–2  (prefill/decode split, matmul variant)
3. Read 08 §3–6  (KV-cache, GQA, RoPE)
4. Read 03 §5b–5c (RoPE + decode head-split)
5. For multi-device: 08 §7
```

---

## FILE SUMMARIES (one-liner each)

| File | One-liner |
|---|---|
| `00_README.md` | Overview, source campaigns, the 5 universal rules, how to navigate |
| `01_FOUNDATIONS.md` | Hardware grid, L1/DST budget, CB-clash math, fidelity/precision knobs |
| `02_NORMALIZATION.md` | LN/RMSNorm/GroupNorm: sharding, fidelity, precision-compounding bug, residual fusion |
| `03_QKV_PROJECTION.md` | Fused QKV matmul, program configs, head-split strategies, RoPE |
| `04_ATTENTION_SDPA.md` | SDPA vs manual BMM, chunk sizes, softmax, score dtype, mask rules |
| `05_MLP.md` | FF1/FF2 configs, fused activation, `minimal_matmul`, subblock tuning, L1 handoff |
| `06_FUSION_AND_RESIDUALS.md` | Op-count reduction, residual folds, reshard/dtype fusion, `unary_chain` |
| `07_METHODOLOGY.md` | Sweep/PCC/wall-time loop, noise floor, harness bugs, signpost captures |
| `08_DECODE_PREFILL_AND_MULTIDEVICE.md` | Prefill vs decode phases, KV-cache, GQA, RoPE, multi-device fracturing |
| `09_PROFILING_AND_OP_ANALYSIS.md` | Tracy capture, tt-perf-report, CSV bucketing, data-movement analysis |
| `10_PROGRESS_REPORT.md` | Model-agnostic: how to run the optimization loop + keep the results log (TSV) and HTML progress report |

---

## QUICK ANSWERS (no need to open any file)

| Question | Answer | File/§ |
|---|---|---|
| Never hard-code the core grid | `device.compute_with_storage_grid_size()` | 01 §1 |
| `packer_l1_acc` in sweeps | always `True` (else 3.5× wrong times) | 01 §4 |
| `fp32_dest_acc_en` for matmul | try `False` first → subblock cap 4→8 | 01 §3 |
| `fp32_dest_acc_en` for SDPA/norm | `True` (softmax sum / LN reduction need fp32) | 02 §4, 04 §7 |
| Fidelity for bf8b matmul | LoFi | 01 §5 |
| Fidelity for normalization | HiFi2 minimum | 02 §4 |
| Cast Q/K/V to bf16 before SDPA? | **No** — costs −13.7 ms (BGE-M3) | 04 §5 |
| Attention mask location | DRAM (hard-asserted) | 04 §8 |
| `exp_approx_mode` for SDPA | False (exact is faster on BH at S=512) | 04 §6 |
| FF1 activation | fuse via `fused_activation=(GELU,True)` | 05 §2 |
| `minimal_matmul` — when? | only when 2D-mcast blows L1 CB | 05 §4 |
| Full-model PCC required for | any reduction op (LN, softmax, GN) | 07 §5 |
| Median-of-N, min threshold | median of 3+; ≥50 µs (small batch), ≥200 µs (large batch) | 07 §3 |
| Prefill matmul variant | Matmul 2D, DRAM interleaved | 08 §2 |
| Decode matmul variant | DRAM-sharded, L1 width-sharded activation | 08 §2 |
| Tracy op count sanity check | 4N matmuls, N SDPA, ~2N norms for N-layer model | 09 §7 |
