# Borrowing from the independent `winter/` DiffusionGemma implementation — 2026-07-27

An independent TTNN implementation of this same model (`models/experimental/diffusion_gemma/winter/`,
dated 2026-06-15, ~3.3k lines, validated on a 4×P150 1×4 mesh) was diffed line-by-line against this
branch. This document records what the diff found, what was **measured on QB2**, and what was
changed. Winter's own headline (a ~0.15 s warm traced denoise step) has **no committed measurement
artifact** in that tree — it is a README assertion plus a code path capable of producing it — so
nothing here rests on it. Every number below is ours.

## Summary of what changed

| # | change | status | evidence |
|---|---|---|---|
| 1 | denoise SDPA no longer pinned to an 8-core grid | **default flipped**, bit-exact | −8.8% traced block, `arm_sweep_30L` |
| 2 | OPT-004 tuned MoE geometry made L1-legal at any capacity; the stale `C == 32` gate removed | **default-on** (was silently inert since 2026-07-15) | −0.8% traced block; decision-changing |
| 3 | encoder `layer_scalar` divergence guard | added, fail-loud | checkpoint measured tied (max\|Δ\| = 0.0 over 30 layers) |
| 4 | `DG_ROPE_FUSED` — fused `ttnn.experimental.rotary_embedding` on the denoise path | added, **default off** | −0.2% (noise) and decision-changing ⇒ not worth flipping |
| 5 | `DG_SDPA_EXP_APPROX` — ttnn's own SDPA softmax default, which we had hard-coded off | added, **default off** | decision-changing; unswept before today |
| 6 | `DG_MOE_CONCAT` — concat-experts MoE | added, **default off** pending the quality gate | **−29.9%** traced block at 30L; see §5 |
| 6b | `DG_TRACE_REGION_SIZE` 12 GiB → 6 GiB in the harness/gate scripts | done | 48 traces measure 3.04 GiB, not the 1.44 GiB our own doc claimed; see §8 |
| 7 | `DG_SKIP` — in-graph component zeroing | added, measurement-only | see §6 |
| 8 | `ttnn.topk` width-cliff coverage | test added | see §7 |
| 9 | `--entropy-stop-threshold` on `serving_smoke` | added | required for any per-step A/B; see §2 |
| 10 | corrected three docs that advertised measurements as current when they were not | done | §3, §4 |

## 1. Method — and one thing that invalidated the first attempt

`doc/optimize_perf/sweep_denoise_arms.sh`: interleaved arms over `demo/serving_smoke.py --upfront`,
30 layers, canvas 256, `reveal_pmax = 4096`, `max_seq_len = 4096`, `--gumbel-mode host --seed 0`,
3 blocks × 48 denoise steps, 3 repetitions. Steady state = `mean(per_block_latency_s[1:])`; block 0
carries the 48-trace capture (~85 s) and is discarded.

**The first run of this sweep was invalid and the reason generalizes.** With the shipped
`entropy_stop_threshold = 0.005`, the stable-and-confident early halt fired at
`denoise_steps_per_block = [9, 2, 2]` — so the "48-step" steady blocks ran **two** denoise steps
each, and what was being timed was mostly the fixed commit cost. Worse, a lever that changes the
numerics also moves where the halt fires, so the arms would not have been running the same amount
of work. `serving_smoke` grew `--entropy-stop-threshold`; a negative value disables the halt. Every
number below is with the halt off and `denoise_steps_per_block = [48, 48, 48]` verified per arm.

The sweep script now prints the per-arm step counts next to the latency for exactly this reason.

**Two things worth recording from the invalid run, because they are about the shipped path.** With
the halt at its shipped 0.005, this prompt ran `[9, 2, 2]` denoise steps and blocks 1 and 2 produced
the *identical* `per_block_sha256` — the model emitting the same degenerate block twice, the #48291
signature. And in that regime the block is roughly 55% denoise / 45% commit (2.38 s/block at ~660
ms/step plus a ~1.0 s commit), so a denoise-step lever has materially less end-to-end leverage than
a 48-step budget implies: the −8.8% below becomes roughly −6% on a block that halts at 2 steps. The
per-step number is still the right thing to optimize — it is what a non-degenerate model would
spend its time in — but the two should not be quoted interchangeably.

## 2. Results

All arms emitted 3 blocks × 48 steps. `committed_sha256` is perfectly reproducible per arm across
repetitions, so a sha difference is a real decision change and not run-to-run noise.

| arm | env | n | steady s/block | per-rep | vs `auto` | ms/step | committed_sha256 |
|---|---|---|---|---|---|---|---|
| `auto` | `DG_SPARSE_MOE_TUNED=0` — today's production path | 3 | 34.642 | 34.762 / 34.718 / 34.447 | — | 722 | `304e8023…` |
| `tuned` | `DG_SPARSE_MOE_TUNED=1` | 3 | 34.369 | 34.498 / 34.293 / 34.315 | −0.8% | 716 | `2ac3efcc…` |
| `tunedgrid` | `+ DG_SDPA_GRID=device` | 3 | 31.586 | 31.423 / 31.892 / 31.444 | **−8.8%** | 658 | `2ac3efcc…` |
| `tunedgridrope` | `+ DG_ROPE_FUSED=1` | 3 | 31.565 | 31.586 / 31.235 / 31.873 | −8.9% | 658 | `1615f91d…` |

Within-arm spread is ≤1.5%, and `auto` > `tuned` > `tunedgrid` holds in every repetition, so the
−0.8% and −8.8% are both resolved; the −0.1% between `tunedgrid` and `tunedgridrope` is not.

Reading:

* **The SDPA grid is the win, and it is free.** `tunedgrid` and `tuned` produce the *same* committed
  tokens, so reassigning the Q axis across cores is bit-exact — as expected, since it does not touch
  the flash K-reduction (`k_chunk_size` is unchanged). This is the same argument the 2026-07-24
  q-chunk sweep validated over 6 runs; here it is confirmed again at a 4× larger prefix span.
* **The tuned MoE geometry is worth ~1%, not the 3.47× the docs implied.** That figure was a C=32
  measurement (§3). It is a consistent win — `tuned` beat `auto` in every repetition — but small,
  and it changes the committed tokens.
* **Fused RoPE is noise here.** −0.2% over `tunedgrid`, well inside the spread, while changing the
  committed tokens. The op-count argument (8 ops → 1, ×2 ×30 = 420 fewer ops/step) is real but those
  ops are evidently overlap-hidden under trace. Kept as a flag, default off.
* **ms/step is not comparable to the ~428 ms in earlier docs.** Those were taken at
  `reveal_pmax = 1024`; this sweep runs 4096, i.e. 4352 key rows per layer instead of 1280. The
  span-proportional attention term is exactly what makes the grid lever matter, so the larger span
  is the right vehicle for it — but the absolute number is span-specific.

## 3. The tuned MoE configs had been inert on the production path for 12 days

`746cfe53cb6` (2026-07-15) moved the capacity default from 32 to the canvas length, because 32 was
"silently discarding 41-84% of active routes per layer". In the same commit the tuned program
configs gained a `C == DEFAULT_CAPACITY` condition (`DEFAULT_CAPACITY = 32`). Production passes 256.
So from that day every MoE matmul — gate, up, down, gather, combine — ran `program_config=None`,
while `sparse_moe.py`'s own docstring and `doc/optimize_perf/README.md` both continued to advertise
the tuned geometry as the default and warn that "a run that forgets the flag would silently take
the slow path".

The condition existed for a real reason: at C=256 the down matmul's per-core output block is
`per_core_M × per_core_N = 8 × 88` tiles = **2.9 MB** against ~1.4 MB of usable L1, so the tuned
config was illegal, not merely suboptimal. Our `_pick_in0_block_w` modelled only the in1 block
against a flat 176-tile budget — no in0 CB, no output CB, no partials.

Fix (`tt/sparse_moe.py`), taking the shape of winter's `_compat.make_block_sharded_matmul_config`:

* `_cb_tiles` / `_cb_fits_l1` model **all** the per-core CBs — double-buffered in0 and in1, plus
  output and partials.
* `_pick_in0_block_w` walks the K-block down through divisors of Kt until the whole set fits.
* `_pick_per_core_m` (new) walks the M-block down through divisors of Mt. The reuse factory forces
  `per_core_N == Nt`, so M is the only way to shrink the output block; handing `split_work_to_cores`
  `(Mt / per_core_M) × E` smaller blocks is legal.
* `_pick_out_block` (new) does the same for the 2D mcast gather/combine, which size their output CB
  from `out_block_*` rather than `per_core_*`.
* A matmul that still does not fit is **omitted** (auto-config for that matmul, warning logged)
  rather than emitted as a config the device would reject.

The resulting geometry, verified by construction:

| | C=32 (unchanged) | C=256 (newly legal) |
|---|---|---|
| gate/up | pcM 1, pcN 6, kw 22 — 0.66 MB | pcM 8, pcN 6, kw 11 — 0.83 MB |
| down | pcM 1, pcN 88, kw 2 — 1.09 MB | pcM 2, pcN 88, kw 1 — 1.09 MB |
| gather | pcM 13, pcN 8, out (13,8), kw 8 — 1.11 MB | pcM 103, pcN 8, out (103,2), kw 1 — 1.30 MB |
| combine | pcM 1, pcN 8, out (1,8), kw 16 — 0.62 MB | same |

**The C=32 configs come out byte-for-byte identical**, so the validated OPT-004 geometry is
preserved; only capacities that previously fell off the path are affected.

## 4. The sparse-MoE win was measured against the wrong baseline

`doc/optimize_perf/README.md` claimed the token-gather path is "~5× faster/step than dense-128" and
`path_to_100tps.md` "10.54 ms/layer (13.0× vs dense 137.6)". Winter's README claims the opposite —
that a token-gather sparse MoE benchmarked **2–12× slower** than its concat-experts dense matmul.

Both can be true, because they are about different capacities. At the shipped `C = S = 256`
(`E=128, H=2816, I_dev=192`):

| per layer per device | our gather path | concat-experts |
|---|---:|---:|
| expert gate+up+down MACs | 5.31e10 | 5.31e10 (**identical**) |
| routing / dispatch MACs | **4.72e10** (gather + combine) | 8.05e8 (one expand matmul) |
| **total** | **1.00e11 (+89%)** | 5.39e10 |
| activation DRAM | ~900 MiB (two ~184 MiB intermediates written *and* read) | ~40 MiB |

At C = S the gather cannot save any expert work — the gathered `[1,E,C,H]` is ~94% zero rows — so
the dispatch is pure overhead. The "13×" denominator was `gemma4`'s `PREFILL_CHUNK_SIZE=32` serial
per-expert `sparse_matmul` path, not a well-configured dense matmul; and the numerator was measured
at the capacity that dropped most of the routing. Both docs now carry a correction.

We have **not** established that concat-experts is faster on our stack — only that the argument for
the gather path does not hold at the capacity we ship. §5 is the experiment.

## 5. Concat-experts MoE (`DG_MOE_CONCAT`, default off)

`tt/concat_moe.py` implements winter's shape: relayout gate/up to `[1,1,H,E*I]` and down to
`[1,1,E*I,H]` once, then

    g    = geglu(x @ gate_cat, x @ up_cat)
    rexp = routing @ expand            # expand = repeat_interleave(I(E), I), a static [1,1,E,E*I]
    out  = (g * rexp) @ down_cat

The down fold is exact by linearity: `sum_e W_down_e @ (r_e * g_e) == (r ⊙ g) @ down_cat`. It also
never materializes the `[1,E,S,H]` per-expert output. `apply_geglu` is **ours** (tanh GeLU) — winter
uses `fast_and_approximate` GeLU in the shared MLP and erf-GeLU in the self-conditioning gate, both
of which disagree with this checkpoint, so that part is deliberately not copied.

Cost: `gate_cat` and `up_cat` are a second copy of those weights, 132 MiB per layer per device at
bf16 = **~7.7 GiB over 30 layers**. The originals cannot be freed — prefill and commit still run the
ragged top-8 path over them. `down_cat` should be free (same byte order at bf16 TILE);
`verify_down_concat_is_free` checks that on device instead of assuming it.

7.7 GiB does not fit beside a 12 GiB trace reservation, which is why §8 came first.

### Measured — this is the largest lever found

Same harness, 30L, `reveal_pmax = 4096`, 48 forced steps, `DG_TRACE_REGION_SIZE = 4 GiB`, 2 reps:

| arm | n | steady s/block | per-rep | ms/step | vs sparse |
|---|---|---|---|---|---|
| `sparse` (shipped token-gather) | 2 | 31.737 | 31.442 / 32.033 | 661 | — |
| `concat` (`DG_MOE_CONCAT=1`) | 2 | **22.234** | 22.317 / 22.150 | **463** | **−29.9%** |

Stacked on the SDPA grid fix, against the path production ran this morning (`auto`, 8×1 grid,
34.642 s/block): **22.234 s/block = −35.8%, a 1.56× speedup, 722 → 463 ms/step.**

All 30 layers built their concat weights; free DRAM went 27.87 GiB → 14.41 (model) → 4.93 (concat),
i.e. the relayout cost **7.8 GiB**, matching the estimate. At the new 6 GiB trace default that
leaves only ~2.9 GiB, so **concat currently wants the 4 GiB reservation, not 6 GiB** — the two
levers are coupled and must be set together.

The output is coherent, which matters as much as the latency: `sparse` opens "A diffusion language
model is a generative model that creates text by starting with a sequence of random noise and
iteratively refining it into coherent…", `concat` "Diffusion language model is a generative model
that creates text by starting with a sequence of noise and iteratively refining it into a coherent
sen…". Different trajectory, same content — this is a fast path, not a broken one.

**A 2L/6L extrapolation would have missed this and nearly did.** At 2 layers the delta reads −5.5%
and at 6 layers −5.4%, and the implied per-layer slope difference is ~0.6 ms — which would have
projected ~−3% at 30 layers. The real answer is −30%. At small layer counts the fixed terminal and
self-conditioning cost dominates and the DRAM/L1 pressure is different, so the MoE is simply not
the bottleneck being measured. Measure the lever at the depth it ships at.

**Still default off.** −30% is worth having, but this changes the committed tokens by more than any
other lever here, and the model is already in a decision-fidelity hole (#48291). It needs the
absolute GPQA arm against the CUDA bar in §11 before the default flips.

## 6. `DG_SKIP` — pricing a component's traced cost

A serial per-op profile does not price a traced step; under trace, host dispatch overlaps, so an
op's standalone time can be almost entirely hidden. `DG_SKIP="attn,shared,moe"` replaces a component
with a shape-preserving `ttnn.mul(x, 0.0)` at its seam so the rest of the graph is untouched. Output
is garbage by construction — never feed a `DG_SKIP` run into a `committed_sha256` comparison.

Winter's own tier-2 profiler was deliberately **not** copied: its `_pmark` calls
`ttnn.synchronize_device` at every stage boundary, which drains dispatch and inflates the sum. That
is why winter's serial profile (~570 ms) over-predicts its traced step (~150 ms) by 3.8× — an
artifact of the instrument, not a hardware property. `prof_step_breakdown.py` is async-pipelined
with a single final sync and lands within ~8% of the traced step.

## 7. `ttnn.topk` width cliff

Winter measured `ttnn.topk` returning a garbage index **and** value at a 32768-wide reduction
(32/256 rows matching a torch control, `inf` values), correct at ≥ 49152, and worked around it by
padding to 49152 with `-inf` for the index while taking the value from `ttnn.max`.

We do not hit that width today: the terminal uses `ttnn.argmax` on a ROW_MAJOR input, and the only
`ttnn.topk` on the denoise path is the router's over the 128-expert axis. But V=262144 over **tp=8**
is exactly 32768 — a Galaxy 4×8 bring-up lands on it, and a wrong argmax index is committed with no
temperature cushion.

**Measured on QB2, and it reproduces exactly** (`tests/test_device_topk_width_cliff.py`, 7 passed):

| shard width | `ttnn.topk(k=1)` index agreement vs torch | |
|---|---|---|
| 16384 | 0.129 | |
| 32768 | 0.129 | ← V/tp at tp=8 |
| 49152 | 1.000 | |
| 65536 | 1.000 | ← V/tp at tp=4, what we serve |

`ttnn.max` stayed finite and correct at every width, which is why winter's workaround (pad to 49152
with `-inf` for the *index*, take the *value* from `max`) is the right shape if a tp=8 mesh ever
needs it. `argmax_last_dim` — the op our terminal actually uses — is exact at 65536.

One note on the test itself: the router arm initially failed at 0.9961 set overlap on a plain random
input. That was tie-breaking, not an op error — the 8th and 9th routing values land within a bf16
ulp on a few rows. The test now separates the winners by a wide margin so it measures the op instead
of the tie rule, and passes at 1.0000.

## 8. Trace region — 8 GiB/chip of reserved-but-unusable DRAM (measured; the 1.44 GiB figure is wrong too)

Every serving script and gate reserved `DG_TRACE_REGION_SIZE = 12 GiB`, sized from an estimate that
`doc/vllm_integration/traced_serving.md` later refuted by measuring the 48 resident traces at
~1.41–1.44 GiB/chip. **That correction is also wrong at the span we serve.**
`doc/optimize_perf/bisect_trace_region.sh`, 30L, `reveal_pmax = 4096`:

| reservation | result | free DRAM after build |
|---|---|---|
| 12 GiB | OK | 4.702 GiB |
| 8 GiB | OK | 8.702 GiB |
| 6 GiB | OK | 10.702 GiB |
| 4 GiB | OK | 12.702 GiB |
| 3 GiB | **FAIL** — `TT_FATAL: Creating trace buffers of size 3259146240B … but only 3221225472B is allocated` | — |

So the 48 traces need **3.04 GiB** at this span, not 1.44 GiB, and the reservation was ~8 GiB of
DRAM that nothing could allocate — the free pool tracks the reservation one-for-one. The harness and
gate scripts now default to **6 GiB** (≈2× the measured requirement); 4 GiB is the verified floor.

The number is not a constant: the trace buffer scales with the captured op set and with
`reveal_pmax`, which is exactly how the 1.44 GiB figure came to disagree with this one. Re-run the
bisect when either changes rather than carrying a fixed value forward — that is the mistake this
whole item is.

This is also the DRAM the concat-experts MoE (§5) needs: 4.70 GiB free cannot hold its +7.7 GiB,
10.70 GiB can.

## 9. Encoder `layer_scalar` — checked, and it is tied

Winter loads a **separate** encoder `layer_scalar` (`winter/tt_model.py:260-266`), swaps it in for
the encode pass and back for denoise, and its vendored HF reference agrees. We classify the whole
`model.encoder.` prefix as ignorable. If the two copies differed, we would be applying the wrong
scalar on every prefill and commit — and because `layer_scalar` multiplies the entire layer output,
the error would compound per layer into the prompt KV, which is a shape a precision sweep cannot
move and would have been a candidate root cause for the ~0.85 backbone-PCC floor.

Measured on `diffusiongemma-26B-A4B-it`: **max |encoder − decoder| = 0.0 across all 30 layers.**
The prefix holds nothing else on the text path — its other 356 keys are vision tower / embed_vision.
So the loader is correct for this checkpoint, and the docstring reason ("NOT on the text-first
causal path") was wrong even though the conclusion was right.

`checkpoint.validate_encoder_layer_scalar_tie` now runs on every load and raises if a future
checkpoint diverges, so this cannot become a silent correctness bug.

## 10. What was NOT borrowed, and why

* **argmax-for-accepted / dropping the stochastic draw.** Winter uses no on-device RNG at all,
  justified by accepted positions carrying ≤ 0.1 nats total. Our own matched-seed A/B — changing
  *only* the Gumbel source, both statistically valid — flips GPQA answers, and `sampled` reaches
  the canvas only through accepted positions, so the draw demonstrably propagates at our scale. The
  prize is ~27 ms of a ~496 ms step. `--argmax-sampling` already exists as the deterministic A/B arm.
* **bf4 / bfp8 expert weights.** Winter runs bf4 experts. That is the configuration we measured and
  rejected: committed-argmax 0.227, entropy PCC 0.631, accept IoU 0.501 (`doc/datatype_sweep/`), for
  +6–9% end-to-end. Winter's PCC bands were measured at that precision.
* **Winter's activations.** `fast_and_approximate` GeLU in the shared MLP and erf-GeLU in the
  self-conditioning gate; this checkpoint needs tanh. Ours is right in both places.
* **Per-bucket trace release + recapture.** We shipped that design, measured 18 → 3.6 tok/s, and
  replaced it with the opposite one (constant `p_max`, capture-once/replay-many, 1.68×, zero
  recapture). It also collides with three later correctness fixes that all key off a fixed-shape
  reveal mask. Winter can afford bucket churn only because it re-encodes the whole prefix every
  canvas — an O(P²) cost it was paying anyway, and one we removed with the batched commit.
* **Winter's `_sc_buf`.** It persists the previous step's raw `[1,1,S,V]` logits (67–268 MB, copied
  every replay). Ours persists the already-reduced `[1,1,C,hidden]` signal, 1.4 MB, computed inside
  the same trace.
* **A vLLM-free OpenAI shim.** We serve through the real fork and run the 198-sample GPQA through
  it. Winter's shim serializes on one lock, so it adds no capability and would miss the
  degeneracy-guard terminal contract.

## 11. Where this leaves us against CUDA

Measured the same day on `ssh a100` (one NVIDIA A100 80GB PCIe, bf16, upstream vLLM, single stream,
same prompt, 768 tokens, default sampler — the reference server idle):

| | s per 256-token block | tok/s |
|---|---|---|
| A100 80GB, 1 GPU | **0.54 – 0.62** | 410 – 473 |
| QB2, 4× Blackhole, shipped early-halt path | ~2.3 | ~108 |

So today TT is **~4× slower per block than a single A100**, on 4 chips — ~17× per chip. The levers
in this document move the forced-48-step block by 1.56×, which is real and does not close that gap.

Two caveats that cut in opposite directions and should be stated together: the TT blocks in that
comparison halt after 2–9 denoise steps and blocks 1–2 emit *identical* committed tokens, so TT is
partly "fast" for the wrong reason (#48291); and the A100 number is single-stream on an 80 GB part
that holds the whole model, so it is a latency comparison, not a cost or throughput one.

GPQA-Diamond on the same reference server, 198 samples, flexible-extract: **70.71%** and **70.20%**
across two repetitions (thinking, 262k) — a 0.5 pp spread, which is the resolution any TT quality
arm has to beat before a difference means anything.

## 12. Open

* The GPQA arm for the two decision-changing defaults (tuned MoE geometry) — the CUDA reference bar
  is GPQA-Diamond flexible-extract **70.7%** (thinking, 262k), with the reference's own run-to-run
  spread measured at ~1.1 pp over three repetitions, so the gate cannot resolve differences below
  ~1–2 pp.
* Concat-experts MoE measurement (§5), after the trace region is right-sized (§8).
* `DG_SDPA_EXP_APPROX` has never been swept.
