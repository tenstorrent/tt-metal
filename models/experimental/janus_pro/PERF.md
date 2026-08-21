# Model performance and accuracy

Two separate things live in this document.

1. **Demo-level numbers** — Top-1/Top-5, decode speed, TTFT — measured end to end from
   `demo/vision_demo.py` and `demo/text_demo.py`. These describe what a user of the model
   experiences.
2. **[Vision tower device time](#vision-tower-device-time)** — the vision tower measured on its
   own as an optimization target. This is where the optimization work is documented, with a
   measured baseline, one entry per change, and a preserved profiler report per stage.

The tower is one component of TTFT, not the whole of it: see [How this relates to the TTFT
numbers above](#how-this-relates-to-the-ttft-numbers-above).

Note: accuracy and perf are gathered in separate runs. Accuracy uses tracing **off**
(`-k notrace`); perf uses tracing **on** (`-k trace`).

Note: unlike Gemma-3, Janus-Pro runs a **single fixed precision** — there is no separate
performance/accuracy mode, so accuracy and perf share one configuration. The decoder is
bfloat8_b. The vision tower is mixed: projection weights and matmul outputs bfloat8_b, residual
stream and biases bfloat16. See README.md for the per-tensor breakdown.

Note: the demo tables predate the vision-tower optimization work and are not refreshed here;
treat them as bring-up figures for the decoder path.

## Build type: Debug is fine for accuracy, Release is required for perf

Accuracy (Top-1 / Top-5) is **build-independent**: it measures *which* tokens the model
predicts, and the ops, dtypes, and numerics are identical between a `Debug` and a `Release`
build — Debug only runs slower, it does not change the result. So the accuracy numbers below,
collected on a `Debug` build, are valid as-is.

Perf (Speed / TTFT) is **not** valid on a `Debug` build: unoptimized host code plus extra
assertions and watcher overhead make it orders of magnitude slower than `Release`, so Debug
timings are meaningless as a performance figure. Perf must be measured on a `Release` build —
the `[janus_pro][demo] * perf benchmark` launch configs use the `build release` task for exactly
this reason.

## Text accuracy (LLaMA decode path)

Top-1/Top-5 are from the **text accuracy run** (`-k notrace`), scoring the device's greedy
predictions against the fp32 HF reference (256 tokens, teacher-forced) — a `Debug` build is fine
here (build-independent). Speed/TTFT are notrace/`Debug` values and are **not** valid perf
figures.

| Model        | Device | Build | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|--------------|--------|-------|-----------|-----------|---------------|-----------|
| Janus-Pro-7B | N150   | Debug | 97.66     | 100.00    | 1.13          | 866       |

## Vision perf (vision tower + LLaMA decode path)

This is the **perf test** — the vision perf benchmark, tracing **on**, covering both
single-image and multi-image prompts. No reference-token accuracy for the multimodal path
(image-conditioned generation); functional output is verified by inspection.

| Model        | Device | Build   | Scenario | Top-1 (%) | Top-5 (%) | Speed (t/s/u) | TTFT (ms) |
|--------------|--------|---------|----------|-----------|-----------|---------------|-----------|
| Janus-Pro-7B | N150   | Debug   | OCR      | N/A       | N/A       | 17.73         | 1439.6    |
| Janus-Pro-7B | N150   | Release | haiku    | N/A       | N/A       | 15.12         | 554.2     |
| Janus-Pro-7B | N150   | Release | OCR      | N/A       | N/A       | 17.93         | 763.5     |
| Janus-Pro-7B | N150   | Release | multi    | N/A       | N/A       | 12.94         | 1609.4    |

Only the **Release** rows are valid performance figures; the **Debug** row is kept for reference
(only the OCR scenario was captured for the Debug run). All rows use tracing **on**. Scenarios:
the two default single-image prompts (haiku on `dog.jpg`, OCR on `ocr_image.jpeg`) and the
multi-image prompt (`dog.jpg` + `ocr_image.jpeg` fed as one prompt, "Describe each of these
images in one sentence.").

The multi-image row runs a single parametrization; select it with
`-k "trace-multi and not notrace"`. Note `-k "trace and multi"` is too loose — `multi` matches
`multimodal` in the test function name and `trace` is a substring of `notrace`, so it also runs
the single-image and notrace cases.

Comparing the same scenario (OCR) across builds: **decode speed is nearly identical** (Debug
17.73 vs Release 17.93) because the traced decode loop is device-bound — trace replay runs
on-device, so the slow Debug host path barely matters. **TTFT roughly halves** on Release
(1439.6 → 763.5) because prefill runs **notrace** (host-dispatched) and a Release build cuts
that host overhead. TTFT includes the vision tower + prefill over the ~596-token image+prompt
sequence.

The **multi** row is slower on both axes than the single-image rows: **decode 12.94 vs ~15–18
t/s/u** because two images push the prefill to ~1172 tokens, so the traced decode attends over a
longer KV cache each step; and **TTFT 1609.4 ms vs ~554–763 ms** because prefill covers roughly
double the sequence (two image placeholder blocks + prompt).

Not directly comparable to Gemma-3's published vision perf, which is measured **multi-image**
(`batch1-multi-image-trace`, 8 images per prompt) in a tunable `performance` precision mode.
Janus multi-image is validated here with **2 images** per prompt at its one fixed precision;
Gemma-3 uses 8.

---

# Vision tower device time

Everything from here down is about the **vision tower alone** — patch embedding, transformer
encoder, aligner, no language model — measured on a Wormhole N150 and optimized as a standalone
target.

## Start here

**What this is.** Janus-Pro's vision tower turns one 384x384 image into 576 tokens for the
language model. It is 24 transformer layers of fixed shape, which makes it a good optimization
target: nothing about it varies at runtime. This section documents how long it takes on
device and why.

**Where it stands.** Kernel time — the sum of every op's compute time for one forward pass —
went from **29.501 ms to 9.230 ms, −68.7%**, and device ops from 393 to 293. Every number here
was measured on an N150; none is estimated. The tower's own metric ended at 0.967334 against its
0.95 gate, and the strictest gate (`test_vision_transformer`, 0.99) at 0.996628.

Accuracy held to change 29 without being spent — that gate *rose*, 0.998631 to 0.998811. Changes
30 and 31 are the first that buy time with precision rather than for free, and 31 is bounded by
that gate rather than by anything on the device: it narrows the residual on as many blocks as
0.99 allows and no more, which leaves 6.7e-3 of slack for whatever comes next.

**Where the 20.2 ms came from.** Four op families account for 96% of it. If you read nothing
else, read this table:

| what was slow | why | what fixed it | saved |
|---|---|---|---:|
| matmuls | bfloat16 weights at the most expensive math fidelity, no explicit configs | narrower dtypes, lower fidelity, sharded outputs, per-shape configs | **7.4 ms** |
| 148 elementwise ops | every bias was a separate op costing ~29 us regardless of its arithmetic | fold each bias into the matmul that precedes it | **5.6 ms** |
| 24 standalone gelus | gelu ran as its own op after the MLP's first matmul | fuse it into that matmul, then approximate it | **2.9 ms** |
| attention (SDPA) | chunk sizes did not divide 576 evenly, and it ran at the highest fidelity | chunk 192, one k-iteration, HiFi2 without fp32 accumulation | **2.8 ms** |

The pattern behind all four: **the tower was not slow because of bad algorithms, it was slow
because of unexamined defaults.** Nothing had *chosen* bfloat16, HiFi4, separate biases or
DRAM — those were simply what the inherited Llama-vision code did.

**The four techniques, if you are optimizing your own model on this hardware.** None of them is
specific to Janus:

1. **Fold every elementwise op you can into the matmul that produces its input.** Under trace a
   standalone bias add costs ~29 us of kernel time no matter how trivial its arithmetic. 96 of
   them went away.
2. **Match the numeric format to what the operands actually carry.** bfloat8_b holds a 7-bit
   mantissa, so a fidelity that makes four passes over it is paying for bits that do not exist.
   The rule that decided which tensors could be narrowed is **read-once versus accumulated** —
   a tensor with a single consumer narrows for free; an accumulator does not.
3. **Give matmuls explicit program configs and sweep them in the model.** ttnn's derivation is a
   safe default, not an optimum, and an isolated bench of the same shape disagreed with the
   in-model result on two of three shapes.
4. **Choose sharding for the chain of ops, not for one op.** The layer norm's grid is not what is
   fastest for the norm alone; it is what lets the following matmul read the shard in place and
   skip a conversion entirely.

## How this relates to the TTFT numbers above

**It is not a subset you can read off them.** TTFT is the demo's `inference_prefill` duration
(`demo/vision_demo.py:109-118`), and that call runs the vision tower *and* the language model's
prefill over the whole image+prompt sequence (`generator_vllm.py:249-252`). The tower is one
component inside a TTFT of 554-763 ms; the language model dominates the rest. Improving the tower
by 19.5 ms moves TTFT by at most 19.5 ms, and only if nothing else changes.

Two further reasons the figures are not directly comparable:

- **This section measures device kernel time under trace.** TTFT is wall-clock on the host and
  includes dispatch, host-side work and the transfer of pixel values.
- **Production calls the tower eagerly.** `compute_vision_token`
  (`janus_pro_e2e_model.py:73-75`) invokes `self.vision_model(pixel_values)`, i.e. the eager
  `forward`; the traced entry point `forward_device` is used only by
  `tests/test_vision_tower_janus.py`. The same tower code measures 129.6 ms eager against 27.2 ms
  traced, so trace state alone changes the tower's contribution by ~100 ms.

## What the tower is

You need these shapes to read anything below, because most results are consequences of them
(`model_config.py:145-168`):

| | |
|---|---|
| image | 384x384, patch 16 → 24x24 = **576 tokens** |
| hidden dim | **1024** (32 tiles of 32) |
| layers | **24**, pre-norm |
| heads | **16** x 64 |
| MLP | ratio 4 → inner dim **4096** |

576 tokens is **18 tile-rows** of 32. That single fact drives most of the core-count ceilings
later on: anything that parallelizes over rows cannot exceed 18 cores.

One encoder layer runs, in order:

```
ln_1 → qkv matmul → split heads → SDPA → concat heads → wo matmul → add
ln_2 → c_fc matmul (+gelu) → c_proj matmul → add
```

`c_fc` and `c_proj` are this codebase's names for HF SigLIP's `fc1` and `fc2`
(`janus_pro_image_mlp.py:74`). `wqkv` is the fused Q/K/V weight; `wo` is the output projection.
The two `add`s are the residual stream.

## Vocabulary

Written out because the rest of the document is unreadable without it.

| term | meaning |
|---|---|
| **kernel time** | Σ`DEVICE KERNEL DURATION` over one trace replay. Compute only, no dispatch. **Reproducible to two decimals; this is the figure to compare across sessions.** |
| **span** | kernel time + Σ`OP TO OP LATENCY`. Careful: slicing replays at the tower's first op folds that op's gap — the *inter-replay trace turnaround* — into the span. That turnaround measured 434-577 us across runs and 113-771 us between replays of one run. **`span_clean` excludes it; plain `span` does not.** |
| **in0 / in1** | a matmul's two operands: in0 is the activation, in1 the weight. |
| **BRISC / NCRISC / TRISC0-2** | the five RISC-V cores per Tensix tile. BRISC and NCRISC move data over the NOC; TRISC0/1/2 unpack, do math, and pack, and have **no NOC access** (`tech_reports/PerfCounters/perf-counters.md:23`). Comparing their busy time to the op's duration says which one bounds it. |
| **1D vs 2D reuse** | matmul strategies. 2D splits both M and N across a core grid; 1D multicasts in0 to every core and splits only N. Which wins depends on the shape. |
| **multicast** | one NOC write whose destination is a *rectangle* of cores (`dataflow_api.h:923`). Cores needing identical bytes lie in a grid row or column, so one read plus one multicast replaces N reads. |
| **sharded vs interleaved** | interleaved splits a tensor into pages and round-robins them over memory banks, ignoring shape. Sharded pins a *rectangle of the tensor* into a *specific core's L1*, so the kernel on that core reads it without touching the NOC. |
| **reader-bound** | the data-movement core is busy for the whole op, so the math engine waits. Cutting bytes helps; changing math fidelity does not. |
| **FLOPs %** | from `tt-perf-report`. Achieved FLOPs over `tflops_per_core(fidelity) x cores_used` (`perf_report.py:744,815`). **Not a target** — see [the trap](perf_reports/PROFILER_NOTES.md#flops--is-the-wrong-target). |

## How to reproduce

Every stage in this document, baseline included, was measured with the same command:

```bash
HF_MODEL=deepseek-community/Janus-Pro-7B MESH_DEVICE=N150 JANUS_VIT_DEVICE_PERF=1 \
  python -m tracy -p -r -v --op-support-count 10000 -m pytest \
  'models/experimental/janus_pro/tests/test_vision_tower_janus.py::test_janus_vision_tower[wormhole_b0-mesh_device0-device_params0-trace_pcache-0.0-0.1-0.1-0.95-1]' -v
```

Then condense the run into a committable stage report:

```bash
python -m models.experimental.janus_pro.tools.perf_stage_report \
  generated/profiler/reports/<stamp>/ops_perf_results_<stamp>.csv \
  --stage <slug> --sha <commit> --note "<one line>"
```

Or read it interactively:

```bash
tt-perf-report --start-signpost start --end-signpost stop --no-advice --no-summary "$CSV"
```

Four traps, each of which cost real time to find:

- **`--op-support-count 10000` is required.** The default per-RISC marker budget is 1000 programs
  (`tools/tracy/common.py:34`) and this path emits ~7,600 ops. Without it, post-processing fails
  to match a host op against the device report.
- **tracy swallows pytest's exit code.** Read the log; a green status means nothing.
- **`tt-perf-report` needs both signposts.** Without them it keeps only the ops *after* the last
  signpost, and everything here is *between* `start` and `stop`, so it reports "No device
  operations found".
- **`-k "expr with spaces"` does not survive tracy**, which re-splits the wrapped command. Use
  the full node id, quoted, as above.

The test warms up, captures a trace, then replays it 10 times inside the signposts. The reported
figure is the mean over replays 2-10.

**Read kernel time, not span.** Kernel time is the sum of the ops' compute durations and
reproduces to two decimals. Span adds the op-to-op gaps, which are dominated by the turnaround
between one trace replay and the next — measured anywhere from 113 to 771 us between replays of a
single run. Two runs of identical code can differ by 2.5% on span and not at all on kernel. Replay 1 is excluded by convention, but **not because it is
slow** — on several runs it measured *faster* than the mean. The convention is cheap insurance,
not a correction for a measured cold-start cost.

## Baseline: the unoptimized tower

The tower as it first worked, before any performance change. Concretely:

- every matmul at **HiFi4** (`compute_kernel_config_hifi4`), the most expensive fidelity
- every tensor **bfloat16**, weights included
- every tensor in **DRAM interleaved**, nothing sharded
- **no explicit program configs** — they are commented out, so ttnn derives its own
- bias and gelu as **separate elementwise ops** after each matmul
- layer norms on the inherited Llama `TtLayerNorm`, which falls through to the interleaved
  18-core path at this sequence length

One thing to know about how this was measured. The traced entry point the perf test uses did not
exist when the tower was first written, so the baseline could not simply be checked out and run.
Instead **only the files that carry the arithmetic** were put back to their original state, while
the ones that merely expose a device-side entry point stayed current. The compute measured is the
original; the harness is the same one every later stage uses, which is what makes the numbers
comparable at all. `model_config.py` is byte-identical across the whole range, so no program
config leaked either way. `perf_reports/README.md` has the mechanics.

Cross-checked two ways: the traced run reports 393 ops, while an eager single forward of
`test_vision_model.py` reports 400. The difference is exactly the 7 setup ops (Tilize,
Embeddings, Untilize, Typecast) that live outside the trace region. Both runs pass at PCC
0.995921.

Full report: **[`perf_reports/00-baseline-unoptimized.md`](perf_reports/00-baseline-unoptimized.md)**

### Baseline against current, by op family

| Op | inst then | ms then | inst now | ms now | Δ ms | Δ % |
|---|---:|---:|---:|---:|---:|---:|
| Matmul | 99 | 12.686 | 99 | 5.246 | −7.440 | **−58.6** |
| BinaryNg (elementwise) | 148 | 5.762 | 49 | 0.150 | −5.612 | **−97.4** |
| SDPA | 24 | 4.382 | 24 | 1.567 | −2.815 | **−64.2** |
| Unary (standalone gelu) | 25 | 3.064 | — | — | −3.064 | **gone** |
| LayerNorm | 49 | 1.581 | 49 | 0.943 | −0.638 | −40.4 |
| NlpCreateHeads | 24 | 1.526 | 24 | 0.856 | −0.670 | **−43.9** |
| NLPConcatHeads | 24 | 0.488 | 24 | 0.318 | −0.170 | **−34.8** |
| ShardedToInterleaved | — | — | 24 | 0.232 | +0.232 | new |
| InterleavedToSharded | — | — | — | — | — | — |
| **total** | **393** | **29.501** | **293** | **9.316** | **−20.185** | **−68.4** |

Read the `ShardedToInterleaved` row as the price of sharding: 24 unshards were *added*, and they
bought back many times their cost in the matmuls that follow them. The reshard that used to sit in
front of the first block is gone -- the patch projection now writes the shard `ln_1` wants.

### Which matmuls performed badly, and why

The original brief asked specifically for this. Matmul was 43.0% of baseline kernel time; here is
every shape in it.

| shape | what it is | inst | baseline us | now us | Δ % |
|---|---|---:|---:|---:|---:|
| 576 x 4096 x 4096 | aligner projection | 1 | **490.9** | 313.9 | −36.1 |
| 576 x 1024 x 4096 | `c_fc` (MLP up) | 25 | **179.2** | 81.6 | −54.5 |
| 576 x 1024 x 3072 | `qkv` (fused Q/K/V) | 24 | 138.5 | 48.8 | −64.8 |
| 576 x 4096 x 1024 | `c_proj` (MLP down) | 24 | 130.3 | 55.1 | −57.7 |
| 576 x 1024 x 1024 | `wo` (attn output) | 24 | 50.8 | 18.2 | −64.2 |
| 576 x 768 x 1024 | patch embedding | 1 | 43.4 | 44.2 | +1.8 |

Four findings from that table, each of which drove specific changes:

1. **`c_fc` was the worst per-instance body matmul at 179.2 us** — and the reason was *not* the
   matmul. `c_proj` does the identical 4.83 GFLOP in 130.3 us. The gap cannot be arithmetic; it
   is the separate `ttnn.gelu` that follows `c_fc`, which a FLOP count does not credit. That
   observation selected [change 4](perf_reports/04-approx-gelu.md).
2. **The aligner's single 490.9 us instance was the most expensive op in the whole tower**, and
   it had been left entirely in bfloat16 when nothing else was. Selected
   [change 12](perf_reports/12-bfp8-aligner-weights.md).
3. **Every body matmul ran at HiFi4 with bfloat16 operands.** Both are the most expensive
   available setting, and neither was chosen deliberately — they were defaults. Selected changes
   [1](perf_reports/01-hifi2-fused-gelu.md), [8](perf_reports/08-bfp8-projection-weights.md) and later
   [22](perf_reports/22-lofi-body-matmuls.md).
4. **The patch embedding is the one shape that did not improve.** It runs once, at 43 us, and
   nothing was aimed at it. It is now 0.4% of the total and correctly ignored.

### What the profiler said beyond matmul

- **BinaryNg was the second-largest family: 148 instances, 5.762 ms, 19.5%.** 148 elementwise ops
  for 24 layers is roughly 6 per layer — four biases and two residual adds. Under trace each one
  costs ~29 us of pure kernel time regardless of how little arithmetic it does. That selected the
  bias fusions, changes [5](perf_reports/05-qkv-bias-fused.md)–[7](perf_reports/07-aligner-biases-hifi2.md).
- **Unary was 25 instances at 122.57 us each.** These are the standalone gelus. At 10.4% of the
  tower they were a larger line item than LayerNorm. 24 of the 25 disappeared by being fused into
  `c_fc`'s matmul in change [1](perf_reports/01-hifi2-fused-gelu.md), and the last one — the
  aligner's — in change [25](perf_reports/25-aligner-activation-fused.md).
- **SDPA at 182.60 us was the most expensive repeated op after `c_fc`.** Its chunk sizes did not
  divide 576 evenly, which selected changes [2](perf_reports/02-sdpa-chunk-192.md) and
  [13](perf_reports/13-asymmetric-sdpa-chunks.md).
- **LayerNorm ran on 18 cores, not 48.** The inherited class pins its shard to a single tile-row,
  which at 576 tokens cannot hold the tensor, so it silently fell back to the interleaved path.
  Selected change [15](perf_reports/15-block-sharded-layernorm.md).

## Where the time goes now

Kernel time per op code, one trace replay:

| Op | inst | us each | ms | % |
|---|---:|---:|---:|---:|
| Matmul | 99 | 52.99 | 5.246 | 56.3 |
| SDPA | 24 | 65.29 | 1.567 | 16.8 |
| LayerNorm | 49 | 19.24 | 0.943 | 10.1 |
| NlpCreateHeads | 24 | 35.68 | 0.856 | 9.2 |
| NLPConcatHeads | 24 | 13.24 | 0.318 | 3.4 |
| ShardedToInterleaved | 24 | 9.66 | 0.232 | 2.5 |
| BinaryNg | 49 | 3.06 | 0.150 | 1.6 |
| **total** | **293** | | **9.316** | |

Matmul's *share* rose from 43.0% to 53.2% even though its absolute time fell 58%, because
everything around it shrank harder. **Share is not progress; absolute milliseconds are.**

## Change log

Row **0** is the unoptimized tower, so the table starts where the work started. Rows 1-24 are the
changes, in the order they landed. Each names the **layer** it touched, the **type** of
optimization, the **delta**, and **what in the profiler selected it**, and links to its own
explanation.

`kernel after` is the tower's whole kernel time once that change was in, so the column traces the
arc from 29.501 down to 9.316. `Δ kernel` is what that row alone changed, and `ops` is the device
op count per forward pass — worth watching, because two rows *add* ops and still come out ahead.

**Every row was measured the same way.** Each stage was checked out and re-run through the harness
in [How to reproduce](#how-to-reproduce) — same warm-up, same trace capture, same ten replays,
same test — so the rows compare directly and the deltas sum. `PCC after` is that same run's tower
figure against its 0.95 gate. The per-stage breakdowns are in
[`perf_reports/`](perf_reports/).

| # | Layer | Type | Change | Δ kernel | kernel after | ops | PCC after | Selected from profiler by |
|--:|---|---|---|---:|---:|---:|---:|---|
| **0** | — | — | **[baseline, unoptimized](#baseline-the-unoptimized-tower)** | — | **29.501** | 393 | 0.995921 | 393 ops; matmul 43.0%, elementwise 19.5%, SDPA 14.9%, gelu 10.4% |
| 1 | whole tower, MLP | fidelity + fusion | [HiFi2 across the tower, fused MLP gelu](perf_reports/01-hifi2-fused-gelu.md) | -3.327 ms | 26.174 | 345 | 0.995519 | every matmul at HiFi4 by default; Unary 25 inst / 10.4% |
| 2 | SDPA | progcfg | [SDPA chunk 256 → 192](perf_reports/02-sdpa-chunk-192.md) | -1.729 ms | 24.445 | 345 | 0.995034 | SDPA 182.6 us, 3rd largest family; 576 not divisible by 256 |
| 3 | MLP `c_fc` | strategy + memcfg | [c_fc as 1D reuse, output in L1](perf_reports/03-cfc-1d-reuse-l1.md) | -0.909 ms | 23.536 | 345 | 0.994531 | c_fc worst body matmul at 179.2 us; derived 2D reached only 48 cores |
| 4 | MLP `c_fc` | precision (SFPU) | [Approximate gelu in c_fc](perf_reports/04-approx-gelu.md) | -2.321 ms | 21.215 | 345 | 0.976968 | c_fc 235 us vs c_proj 125 us for identical FLOPs → gap is SFPU, not matmul |
| 5 | attn `qkv` | fusion | [qkv bias fused into its matmul](perf_reports/05-qkv-bias-fused.md) | -1.178 ms | 20.037 | 321 | 0.971488 | BinaryNg 148 inst / 19.5%, ~29 us each under trace |
| 6 | attn `wo`, MLP `c_proj` | fusion | [Post-reduce biases fused, single-device only](perf_reports/06-post-reduce-biases-fused.md) | -0.902 ms | 19.135 | 273 | 0.974878 | same BinaryNg census |
| 7 | aligner | fusion + fidelity | [Aligner biases fused, aligner to HiFi2](perf_reports/07-aligner-biases-hifi2.md) | -0.191 ms | 18.944 | 271 | 0.974864 | aligner was the only module left at HiFi4 |
| 8 | all body projections | dtype | [bfloat8_b projection weights](perf_reports/08-bfp8-projection-weights.md) | -1.373 ms | 17.571 | 271 | 0.974384 | 604 MB weight traffic per pass at only 10.6% of DRAM peak → latency, not bandwidth |
| 9 | attn `qkv` | dtype | [bfloat8_b fused qkv output](perf_reports/09-bfp8-qkv-output.md) | -1.276 ms | 16.295 | 271 | 0.975586 | read-once audit of every intermediate |
| 10 | MLP | dtype | [bfloat8_b c_fc intermediate](perf_reports/10-bfp8-cfc-intermediate.md) | -0.433 ms | 15.862 | 271 | 0.976103 | same audit |
| 11 | attn `wo`, MLP `c_proj` | dtype | [bfloat8_b wo and c_proj outputs](perf_reports/11-bfp8-branch-outputs.md) | -0.574 ms | 15.288 | 271 | 0.966546 | same audit |
| 12 | aligner | dtype | [bfloat8_b aligner weights](perf_reports/12-bfp8-aligner-weights.md) | -0.215 ms | 15.073 | 271 | 0.966521 | aligner's 490.9 us instance was the tower's most expensive single op |
| 13 | SDPA | progcfg | [Asymmetric SDPA chunks](perf_reports/13-asymmetric-sdpa-chunks.md) | -0.190 ms | 14.883 | 271 | 0.969296 | SDPA still 103 us after change 2; softmax reducing 3x per q block |
| 14 | both norms | dtype | [bfloat8_b layer-norm outputs](perf_reports/14-bfp8-norm-outputs.md) | -0.508 ms | 14.375 | 319 | 0.972620 | norm outputs are the in0 of the two dominant matmuls |
| 15 | both norms | sharding | [Block-sharded layer norm on 48 cores](perf_reports/15-block-sharded-layernorm.md) | -0.544 ms | 13.831 | 367 | 0.974037 | LayerNorm at 32.26 us on 18 cores; sharded path unusable at 576 tokens |
| 16 | `qkv`, `wo`, `c_proj` | progcfg | [Explicit 2D configs, in0_block_w per shape](perf_reports/16-explicit-2d-configs.md) | -0.350 ms | 13.481 | 367 | 0.974018 | in-model sweep over each shape's valid divisors |
| 17 | attn `wo`, MLP `c_proj` | sharding | [wo and c_proj outputs L1 block-sharded](perf_reports/17-wo-cproj-sharded.md) | -1.456 ms | 12.025 | 320 | 0.973980 | BRISC at ~100% of op duration; it hosts the writer |
| 18 | attn `qkv` | sharding | [qkv output L1 block-sharded](perf_reports/18-qkv-output-sharded.md) | -0.288 ms | 11.737 | 344 | 0.973980 | same, and qkv has the widest write burst |
| 19 | attn | memcfg | [qkv unshard into L1 rather than DRAM](perf_reports/19-qkv-unshard-to-l1.md) | -0.167 ms | 11.570 | 344 | 0.973980 | the unshard's consumer is not a matmul |
| 20 | SDPA | fidelity | [SDPA HiFi4 → HiFi2](perf_reports/20-sdpa-hifi2.md) | -0.170 ms | 11.400 | 344 | 0.972216 | SDPA the last HiFi4 op; TRISC0/1/2 all at 98.5% → genuinely math-bound |
| 21 | SDPA | compute cfg | [fp32 dest accumulation off on SDPA](perf_reports/21-sdpa-no-fp32-acc.md) | -0.509 ms | 10.891 | 344 | 0.974919 | same per-RISC read; DST halves under fp32 acc |
| 22 | `qkv`, `wo`, `c_fc`, `c_proj` | fidelity | [LoFi on the body matmuls](perf_reports/22-lofi-body-matmuls.md) | -0.693 ms | 10.198 | 344 | 0.969802 | all four take bfloat8_b on both sides, so HiFi2's 2nd pass reads absent mantissa bits |
| 23 | `ln_1` + `qkv` | sharding | [ln_1's shard fed to qkv in place](perf_reports/23-ln1-shard-into-qkv.md) | -0.109 ms | 10.089 | 320 | 0.973980 | ShardedToInterleaved 72 inst / 0.528 ms, and the two grids already matched |
| 24 | `ln_2` + `c_fc` | sharding | [ln_2's shard fed to c_fc in place](perf_reports/24-ln2-shard-into-cfc.md) | -0.073 ms | 10.016 | 296 | 0.966490 | same census, remaining 48 unshards |
| 25 | aligner | fusion | [Aligner activation fused into its matmul](perf_reports/25-aligner-activation-fused.md) | -0.033 ms | 9.983 | 295 | 0.966489 | the last standalone Unary, 1 inst / 0.124 ms at 1.2% |
| 26 | MLP `c_fc` | sharding | [c_fc output block-sharded in L1](perf_reports/26-cfc-output-block-sharded.md) | -0.142 ms | 9.841 | 295 | 0.966489 | per-RISC split: BRISC at 99-100% of every matmul, and c_fc's was the only unsharded output |
| 27 | attn heads | memcfg | [q/k/v written into L1](perf_reports/27-qkv-heads-output-l1.md) | -0.342 ms | 9.499 | 295 | 0.966489 | pure data movement, and SDPA reads all three straight back |
| 28 | SDPA | memcfg | [SDPA's output written into L1](perf_reports/28-sdpa-output-l1.md) | -0.098 ms | 9.401 | 295 | 0.966489 | same shape; `nlp_concat_heads` is its only consumer |
| 29 | patch embed, attn | progcfg + memcfg | [the encoder's activations all live in L1](perf_reports/29-encoder-activations-in-l1.md) | -0.085 ms | 9.316 | 293 | 0.970880 | the patch projection's 2D config lands on the norm's grid; `nlp_concat_heads` writes L1 because the in0 penalty is mcast, not L1 |
| 30 | aligner | dtype | [the aligner's intermediate in bfloat8_b](perf_reports/30-aligner-bfp8-intermediate.md) | -0.022 ms | 9.294 | 293 | 0.970875 | fc1's output has exactly one consumer, so the read-once rule applies to the aligner too. Halving a 4.72 MB write moved it 2% — the aligner is transaction-bound like the body, not write-bandwidth-bound |
| 31 | encoder block | dtype | [bfloat8_b residual on the last 12 blocks](perf_reports/31-bfp8-residual-last12.md) | -0.064 ms | **9.230** | 293 | 0.967334 | both norms inherit the residual's format, so `qkv`'s and `c_fc`'s in0 multicast halves with no typecast to pay for it. A suffix rather than all 24: the residual is summed across every layer, so its error compounds and the count is bounded by a gate |

The PCC column is the **tower unit test's** (`test_vision_tower_janus`), against its 0.95 gate,
**except rows 2 and 3 which only have an end-to-end figure** and are labelled `e2e`. Read the
column before attributing an accuracy loss: the gelu approximation alone accounts for 0.0176 of
the total drop, and seven of the 24 steps moved PCC *up*.

## Where the rest is

| file | what it holds |
|---|---|
| [`perf_reports/NN-*.md`](perf_reports/) | one per change-log row, linked from the Change column: that stage's explanation next to its own per-op and per-matmul breakdown |
| [`perf_reports/DEAD_ENDS.md`](perf_reports/DEAD_ENDS.md) | levers measured that did not pay, and one that **did** — the largest win found anywhere here — deliberately absent because it breaks an accuracy gate |
| [`perf_reports/PROFILER_NOTES.md`](perf_reports/PROFILER_NOTES.md) | three ways `tt-perf-report` and the per-RISC counters mislead on this tower, which ops are structurally closed, and what the profiling did *not* establish |
| [`perf_reports/OPTIMIZED_OP_LIST.md`](perf_reports/OPTIMIZED_OP_LIST.md) | every device op of one replay at 9.316 ms, as `tt-perf-report` prints it |
| [`perf_reports/README.md`](perf_reports/README.md) | how a stage report is produced and regenerated |

Read `DEAD_ENDS.md` before trying anything on this tower. It is the cheapest way to avoid
repeating a day of work.
