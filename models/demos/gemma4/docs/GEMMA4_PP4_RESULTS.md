# Gemma4 prefill PP=4 × [8,1] — measured

> **Start with [`GEMMA4_PP4_OVERVIEW.md`](GEMMA4_PP4_OVERVIEW.md)** — what PP=4 is, how it works,
> the headline numbers and the ranked next steps. This document is the detail behind it.

**Machine:** `bh-glx-120-b03u02`, one 32-chip Blackhole galaxy. **Date:** 2026-09-07.
**Branch:** `gemma4-pp4` off `gemma4-prefill-pr` @ `b7babca5380`.
**Ported 2026-09-08** to `kmabee/svuckovic/gemma4-prefill-freeze-sep-07` and fully re-verified there;
every number below reproduced within ~1%. The port's own results, and the one design change it
forced (`prefill_weights_only` collapsing with `is_last_rank`), are in the overview's §0.1.
**Design and rationale:** [`GEMMA4_PP4_PLAN.md`](GEMMA4_PP4_PLAN.md) — read that first; this records what
the plan got right, what it got wrong, and the numbers.

Every number below was measured in one session, baseline included, so the comparison is not
against an archived figure.

---

## 1. Headline

| config | 256k prefill | tok/s | vs baseline |
|---|---:|---:|---:|
| **8×4 single rank** (CP=8 × TP=4), chunk 8192 | **13.6 s** device / 14.1 s wall | **19,287** | — |
| **PP=4 × [8,1]** (CP=8 × TP=1), chunk 8192, 1 request | **11.64 s** | **22,517** | **1.17×** |
| PP=4 × [8,1], 4 requests back to back (1,048,576 tok) | 43.85 s | **23,912** | **1.24×** |

Both configurations use all 32 chips and the same chunk size. The baseline reproduced its
archived figure exactly (13.6 s / 19,287 vs 13.6 s / 19,271), which also confirms the
`concat_heads` change in §4.1 is a no-op at TP=4.

**The compute win is bigger than the end-to-end win, and the gap is pipeline bubble:**

```
max stage compute       9.78 s   ->  1.39x   (what TP=1 actually bought)
+ pipeline fill        +0.67 s             3 stages before the last rank starts
+ inbound socket sync  +0.56 s             ~18 ms x 31 chunks, an untraced device op
+ residual imbalance   +0.64 s             stages differ by 8%; the pipeline runs at the max
= measured span        11.64 s   ->  1.17x
```

The split of that bubble is measured, not inferred — see §7.1. The obvious reading, that the D2D
hop is free because `_d2d_send` reports 0.31 ms, is wrong: the send only ENQUEUES and the bytes
move while the host runs on.

The plan predicted a **1.44×** ceiling from removing the TP-axis collectives. The stage compute
came in at **1.39×** of it — so the model of *why* PP=4 × [8,1] should help was right. What the
estimate under-counted was the pipeline overhead: it assumed ~9% (fill only) and the real figure
is 16%.

---

## 2. Stage balance: the cost model, and that it holds

The plan's central Gemma4-specific claim was that stage balance here is by **global-layer count**,
not layer count, because `layer_types` is 50 sliding + 10 full with a global every 6th and a global
layer costs several times a sliding one at depth. Two splits were run:

| `PREFILL_PP_LAYER_COUNTS` | stage compute (s) | max | span | tok/s |
|---|---|---:|---:|---:|
| `17,13,17,13` (shipped) | 8.98 / 9.67 / 8.96 / **9.78** | 9.78 | 11.64 s | 22,517 |
| `15,15,15,15` (the default with no override) | 8.39 / 10.26 / 8.38 / **10.37** | 10.37 | 12.04 s | 21,780 |

Fitting `cost = S·s + G·g` (per-layer cost summed over the 32 chunks) to the `15,15,15,15` run
gives **s = 0.301 s** per sliding layer and **g = 2.234 s** per global layer — a **7.4× premium**.
Those two numbers then predict the `17,13,17,13` stages as 8.99 / 9.71 / 8.99 / 9.71 against a
measured 8.98 / 9.67 / 8.96 / 9.78. **Under 1% error on a split the model was not fitted to.**

That model also says `17,13,17,13` is **optimal**, not merely better. Enumerating all **32,509**
ways to cut 60 layers into 4 contiguous stages and scoring each by its max stage:

| rank | counts | globals per stage | max stage |
|---:|---|---|---:|
| **1** | **`17,13,17,13`** | 2 / 3 / 2 / 3 | **9.712 s** |
| 2–6 | `16,14,16,14`, `17,13,16,14`, … | 2 / 3 / 2 / 3 | 10.013 s |
| 7 | `15,15,15,15` | 2 / 3 / 2 / 3 | 10.314 s |

`17,13,17,13` is the **unique** minimum, and the reason is a boundary: a 2-global stage would want
18.2 layers to hit the balanced ideal, but `[0,18)` contains global layer 17 and is therefore a
*3*-global stage — **17 is the ceiling**. The residual 3.9% (9.712 vs the 9.348 s ideal) is
structural: **10 global layers do not divide by 4.** No split fixes it, and PP=2 × [8,2], which
would balance them exactly 5/5, loses far more on the collectives it keeps (§7.4).

---

## 3. Chunk 16384: a cold-JIT artifact, and then a real (smaller) loss

The first measurement of chunk 16384 read **18.19 s** with rank 0 alone accounting for +5.5 s, and
was written up here as a rank-0 embedding bug at TP=1. **That was wrong**, and the way it was wrong
is worth keeping.

Per-chunk timings show rank 0's *first* chunk at **5,904 ms** and every chunk after it within 1% of
its peer rank (349/379/410… against 347/376/404…). It was a **cold JIT kernel compile** for a shape
that had never been run, paid once, inside the first chunk. `TT_METAL_CACHE` persists it, so the
re-run is clean:

| chunk | span | tok/s | stage compute (s) | fill | bubble |
|---:|---:|---:|---|---:|---:|
| 8192 | **11.64 s** | 22,517 | 8.97 / 9.67 / 8.97 / 9.76 | 0.67 s | 1.87 s |
| 16384 (cold) | 18.19 s | 14,409 | 14.22 / 9.61 / 8.73 / 9.67 | 6.71 s | 3.97 s |
| 16384 (warm) | 12.62 s | 20,776 | 8.65 / 9.61 / 8.75 / 9.67 | 1.15 s | 2.94 s |

Warm, the two chunk sizes do the **same stage compute** (max 9.67 vs 9.76). 16384 still loses, by
1.0 s, and now for an understandable reason: pipeline fill scales with chunk size (1.15 s vs 0.67 s)
and there is nothing to win back, because the per-chunk overhead does not halve when the chunks do.

The embedding cost is real but small, and was measured directly by A/B-ing one 17-layer stage with
and without `--first` (i.e. embedding tokens vs receiving an activation): **12.5 ms/chunk at 8192
and 41.6 ms/chunk at 16384** — superlinear, but 0.67 s over a whole 16-chunk run, not 5.5 s.

The general lesson is one the Mistral PP=4 docs already state and this session re-learned anyway:
**a cold cell is not a slow measurement, it is a wrong one.** Re-run any new shape warm before
believing it.

## 4. What the plan got wrong

### 4.1 TP=1 does not fit L1 — `nlp_concat_heads`, not the matmuls

Predicted risk #1 was "TP=1 shapes / L1", with the matmul program configs named as the likely
culprit. The matmuls were fine. The op that broke was `nlp_concat_heads`, whose src0 circular
buffer is `2 × heads × head_dim/32` tiles with **no dependence on sequence length**. That head
count is per device, so a Gemma4 *global* layer costs 0.5 MB at TP=4 (8 heads × head_dim 512) and
**2.0 MB at TP=1**, against Blackhole's 1.5 MB. It died during compile at layer 5, the first
`full_attention` layer.

Fixed at the Gemma4 layer rather than in the shared op: `concat_heads` splits the heads into the
fewest even groups whose per-group CB fits a budget and joins the results on the embedding axis,
which is numerically identical. Sliding layers need no split even at TP=1 (0.5 MB), and every
TP ≥ 2 shape computes one group and takes the original single-call path — which is why the
baseline reproduced to the digit.

### 4.2 A later rank fed the model its own persistent trace input

`Gemma4DecoderLayer` **deallocates its input** (`layer.py:302` — it is the residual, freed after the
add). On a non-first rank the runtime was handing it `self._trace_input` directly, so layer 0 freed
the buffer the captured trace reads and that `prefill_chunk` copies each chunk into. The first rank
never hits this because `ttnn.embedding` builds a fresh tensor.

It surfaced as a **segfault inside the next forward's first `rms_norm`** — and, worse, only on the
stage whose first layer is global (`[17,30)`). The stage starting on a sliding layer (`[30,47)`)
read the freed buffer before anything reclaimed it and *passed*, with numbers that looked fine.
`_forward` now clones the received activation: one 11 MB device copy per chunk, inside the trace.

### 4.3 The topology is plain FABRIC_2D, not torus_y

The plan ported Mistral's `torus_y` descriptor and gave "torus_y routes the D2D socket" a 75%
prior. Wrong question. Gemma4's CP collective is
`ring_joint_scaled_dot_product_attention`, which passes `topology=ttnn.Topology.Linear`
*unconditionally* (`attention/ring_prefill.py:418`), and its 8×4 baseline opens plain `FABRIC_2D`
(`test_factory._fabric_config_for_shape`). There is no wrap to match, so asking for one would only
risk a Ring collective on an unwrapped axis — which hangs. Caught by reading the baseline's log
before the first pipeline run, not by a failure.

Mistral needed `torus_y` because *its* single-process comparison opened `FABRIC_2D_TORUS_XY`. The
lesson that transfers is the rule, not the file: **match the fabric mode to the descriptor's
`dim_types` and to what the model's collectives actually ask for.**

### 4.4 The global-index threading was not the hard part

Predicted at 70% to be the dominant bug. It worked on the first device run: rank 1 reported its
globals at `[17, 23, 29]` and rank 3 at `[47, 53, 59]`. The 12 rewritten `layer_types[i]` sites and
the `layer_scalar` assertion were cheap insurance that turned out not to be needed — which is the
correct outcome for insurance, and the assertion is worth keeping regardless (§5).

---

## 5. What the plan got right

* **Memory neutrality.** TP 4→1 quadruples per-device weight bytes, PP=4 quarters the layer count.
  The TP=1 cache came out at 35 GB against the TP=4 cache's 37 GB, and every stage fit with the
  same 256k / 2-user KV as the baseline. This is the fact that makes `[8,1]` legal at all.
* **The 1.44× ceiling and where it comes from.** Stage compute hit 1.39× of it.
* **Balance by global count.** Validated to under 1% by a model fitted on a different split (§2).
* **CP=8 and chunk 8192 stay.** Confirmed the hard way by §3.
* **D2D is not the cost.** Steady-state pushes measured 0.2–0.45 ms against a predicted < 5 ms.
* **Per-rank cold weight loading.** `load_layer_window_state` read 18.0 GiB for a first/last stage
  and 11.9 GiB for a middle one, against 62 GiB for the whole checkpoint.

---

## 6. Estimates vs. actuals

| quantity | estimated (before any code) | measured | verdict |
|---|---|---|---|
| 256k span | 9.4 – 10.5 s | 11.64 s | **10–20% optimistic** |
| 256k tok/s | 25,000 – 28,000 | 22,517 | 10–20% optimistic |
| speedup | 1.30 – 1.45× | 1.17× (1.39× on compute) | compute right, overhead under-counted |
| per-device weights | neutral | neutral | ✅ |
| per-device KV | neutral | neutral | ✅ |
| D2D hop | < 5 ms | 0.2 – 0.45 ms | ✅ (conservative) |
| loss from an even split | 15 – 20% | 6.1% on max stage, 3.4% on span | overestimated |
| loss from `17,13,17,13` | < 6% | 4.3% off an unreachable ideal | ✅ |
| TP=1 needs no op surgery | 60% likely | one L1 fix, no matmul work | half right |
| dominant bug = layer indexing | 70% likely | no | ❌ |
| `local_heads=1` bites at TP=1 | near certain | fixed pre-emptively; would have silently allocated ¼ the global KV heads | ✅ |
| files / new lines | ~8 files, ~400 lines | 9 source files, 6 new | ✅ |

---

## 7. Where the remaining 16% is — measured, then attacked

### 7.1 The bubble, attributed

`run_request_loop` now writes a per-chunk phase breakdown alongside the chunk timings whenever
`PREFILL_TIMING_DIR` is set. Medians over a 256k single-request run, first 4 chunks discarded:

| rank | lease wait (out) | lease wait (in) | inbound recv | compute + send | loop |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 ms | 0.0 ms | **17.7 ms** | 294.6 ms | 312.5 ms |
| 1 | 0.1 ms | 0.1 ms | **17.9 ms** | 321.7 ms | 339.8 ms |
| 2 | 0.1 ms | 0.1 ms | 52.5 ms | 293.3 ms | 346.2 ms |
| 3 | 0.0 ms | 0.2 ms | 35.9 ms | 325.4 ms | 357.6 ms |

**The fabric-link lease cycle is 0.1 ms.** It was the prime suspect and it is not the problem —
four synchronising calls per chunk that cost nothing. Ranks 2 and 3 spend 36–52 ms in `recv`, but
most of that is legitimate pipeline idle (their compute is below the bottleneck rank's, so they
wait). The diagnostic number is **rank 0's 17.7 ms**: rank 0 has no upstream, the producer had
finished pushing long before, and it still pays 17.7 ms. Rank 1 pays 17.9 ms while computing *more*
than rank 0, so it is not waiting either. That ~18 ms is a floor — the cost of an untraced
`inbound_socket_service_sync` device op, one full dispatch round trip per chunk.

Over 31 chunks on the critical rank that is 0.56 s of the 1.87 s bubble. Fill is 0.67 s. The
remaining 0.64 s is the 8% stage imbalance, which the pipeline pays because it runs at the max.

### 7.2 Stage balance: closed

Given the fitted per-layer costs (sliding 0.301 s, global 2.234 s), all **32,509** ways to cut 60
layers into 4 contiguous stages were enumerated. `17,13,17,13` is the **unique minimum** at
9.712 s; the runner-up family is 10.013 s and the naive `15,15,15,15` ranks 7th at 10.314 s
(measured 12.04 s end to end). The perfectly-balanced ideal is 9.348 s and is unreachable because
10 globals do not divide by 4. **There is nothing left in the split.**

### 7.3 Concurrency: +6.2%, and it converges

_(Pre-freeze numbers, as with the rest of this document. Re-measured on the freeze branch in the
overview's §7 — 22,236 / 23,571 / 23,832 tok/s, agreeing within 1.5%.)_

| in-flight requests | tokens | span | tok/s | max stage | fill | bubble | bubble/chunk |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 262,144 | 11.64 s | 22,517 | 9.78 s | 0.67 s | 1.87 s | 58 ms |
| 2 | 524,288 | 22.66 s | 23,136 | 19.75 s | 1.20 s | 2.91 s | 45 ms |
| 4 | 1,048,576 | 43.85 s | **23,912** | 40.13 s | 0.74 s | 3.72 s | 29 ms |

Only the *fill* part of the bubble amortises; the ~18 ms socket sync is paid per chunk regardless.
That is why the curve flattens at 1.24× rather than approaching the 1.39× the compute would allow.

### 7.4 PP=2 × [8,2]: ruled out by modelling, not worth a run

It would balance the 10 globals exactly 5/5 — the one structural defect PP=4 cannot fix. But a
per-layer TP=2 cost of roughly `2 × L4 × (1 − 0.306 × ⅓)` over 30 layers puts its stage period near
12.2 s against PP=4's 9.71 s. The balance it buys is worth 4%; the collectives it keeps cost 25%.

### 7.5 What is left

1. **The inbound socket sync** (§7.1). 0.56 s here, and it lives in the shared runner, so fixing it
   helps every pipelined model. Either capture it in the trace or find a cheaper sync.
2. **The global layers themselves.** 7.4× a sliding layer, and 10 of 60 layers are ~60% of a
   stage's cost. PP does not change that arithmetic — it makes it the dominant term. A per-stage
   Tracy capture is the way in (`tests/perf/pp4/analyze_layer_budget.py`; read its header — an op's
   cost across a stage's 8 concurrent chips is the MAX not the sum, and
   `InboundSocketServiceSyncOperation` is ~99% of a PP stage's device time and is pure idle).
3. **Double-buffer the hand-off.** Would take §7.1's cost off the critical path rather than
   shrinking it. Needs an async variant of the socket sync first; the current op blocks.
4. **Migration and a correctness gate.** Both off throughout this work. See the overview's §5.

## 8. Reproducing

```bash
# 0. once per machine -- the [8,1] column -> device map is per-galaxy and a wrong map does NOT error
python models/demos/gemma4/tests/perf/pp4/gen_gemma4_pp4_binding.py

# 1. once per machine -- build the TP=1 weight cache, one layer window at a time
#    (a single [8,1] mesh cannot hold all 60 layers at TP=1: ~29 GB/chip of weights)
export GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1 GEMMA4_WEIGHT_CACHE_MESH_ONLY=1
for w in "0 17 --first" "17 13" "30 17" "47 13 --last"; do
  set -- $w; python models/demos/gemma4/tests/perf/pp4/run_stage.py \
    --first-layer $1 --num-layers $2 ${3:-} --max-seq 32768 --chunks 4
done
unset GEMMA4_PREFILL_LOAD_FULL_WEIGHTS

# 2. the pipeline
RUN_TAG=pp4_256k_u1 PP_MAX_SEQ_LEN=262144 PP_USERS=1 PP_REQUESTS=1 \
  models/demos/gemma4/tests/perf/pp4/run_pp4_gemma4.sh

# 3. the baseline it is compared against
GEMMA4_PREFILL_LOAD_FULL_WEIGHTS=1 PYTEST_TIMEOUT=3600 pytest -sv \
  'models/demos/gemma4/demo/text_demo_prefill.py::test_prefill_long_context_traced[blackhole-readback_final-ctx_256k-chunk8192-text-8x4]'
```

Never run a device job in a foreground tool call with a timeout: a killed wrapper is a SIGKILL
mid-fabric and the next mesh open dies on an ethernet-core timeout. Use `setsid nohup … &` and poll
the log; `tt-smi -r` after any hard kill.
