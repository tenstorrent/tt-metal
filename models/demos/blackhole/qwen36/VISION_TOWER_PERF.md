# Vision tower device-performance optimisation

Status of the matmul optimisation pass on `tt/vision/` (the `DropInVisionTransformer` tower:
patch/positional embed → 27 × `VisionBlock` → `PatchMerger`).

**Measured on Wormhole**, since that is the hardware the work was done on: Qwen3.5-9B on an **N300**
(TP=2, activations fractured) and Qwen3.6-27B on a **T3K** (TP=8, activations replicated — see
`tt/vision/vision_ccl.py`). The `README.md` deployment targets are Blackhole `P150` / `P150x4`;
those are **not swept** — see [Not covered](#not-covered).

**The tuning is gated to Wormhole N300 / T3K in code**: `VisionModelArgs.vision_mm_tuned` is
`is_wormhole_b0()`, and off-arch `vision_mm_plan` returns only its untuned plan — ttnn's auto matmul
config, DRAM in and out, and the pre-sweep fidelity (`decoders_optimizations` for `qkv`/`wo`).
CCL workers (`vision_ccl_tuning`) additionally require `device_name in ("N300", "T3K")`; Blackhole
keeps the pre-sweep `(chunks_per_sync=10, num_workers_per_link=2)`. `QWEN36_VISION_MM_TUNING=0`
and `QWEN36_VISION_CCL=0` force those fallbacks on any arch.

All numbers below are `tt-perf-report` device time for the demo image (patch grid `1×86×128` =
11008 patches → 12288 padded), which is what `demo/benchmark_vision.py` and `demo/vision_demo.py`
default to.

### Reading the "window" numbers — MIND THE DEPTH

A window total is only comparable at the same **depth**, because a window is
`head + depth × block + tail`. The *Result* table below is **depth 1**; the SDPA and data-movement
sections further down quote **depth 2** windows as well, so read the column headers.

`test_vision_tower_pcc.py` now pins this: profile with **`-k oneblock`** (depth 1, signposted, block
ops appear exactly once) and gate accuracy with **`-k fulldepth`**. Profiling `fulldepth` instead
would report a ~27× larger window and read as a massive regression.

Comparing a depth-1 window against a depth-2 one looks like a 55% regression and is not one. The
whole campaign at **one** depth (all cells measured at depth 1, demo grid, one session, both arms
back to back). `27-blk` is `head + 27 x block + tail`, the only honest projection of the shipping
tower:

**9B / N300**

| | head | block | tail | depth-1 window | 27-blk |
| --- | --- | --- | --- | --- | --- |
| tuning gated off (`QWEN36_VISION_MM_TUNING=0`) | 4.55 | 37.50 | 6.57 | 48.61 ms | 1023.5 ms |
| + matmul program configs | 4.37 | 34.19 | 6.62 | 45.18 ms | 934.0 ms |
| + SDPA + redundant-op removal | 3.03 | 30.42 | 6.51 | 39.96 ms | 831.0 ms |
| + tightened row padding, q/k 128/512 | 2.83 | 26.74 | 6.65 | 36.22 ms | 731.4 ms |
| + CCL `num_workers_per_link=4` (**current**) | 2.66 | 26.46 | 6.25 | **35.37 ms** | **723.3 ms** |

**27B / T3K**

| | head | block | tail | depth-1 window | 27-blk |
| --- | --- | --- | --- | --- | --- |
| tuning gated off (`QWEN36_VISION_MM_TUNING=0`) | 7.23 | 35.01 | 4.39 | 46.63 ms | 956.9 ms |
| + matmul program configs | 6.59 | 33.18 | 3.99 | 43.76 ms | 906.4 ms |
| + SDPA + redundant-op removal | 5.17 | 31.48 | 3.74 | 40.38 ms | 858.7 ms |
| + tightened row padding, q/k 128/512 | 4.85 | 27.70 | 3.81 | 36.35 ms | 756.6 ms |
| + CCL `num_workers_per_link=4` (**current**)† | 4.55 | 32.30 | 3.73 | **40.59 ms** | **880.4 ms** |

`patch_embed` and the two `merger` matmuls are in the head/tail — they run once per image at any
depth, which is why the head/block/tail split is the only honest way to project a 27-block tower.

† The 27B CCL row is a **different session** from the 36.35 ms cell. Same-session `wpl=2` was
41.55 ms, so the CCL win is **−1.0 ms**, not a regression vs 36.35. The 9B `wpl=2` arm this
session was 36.52 ms, within noise of the previous campaign's 36.22. See [CCL workers](#ccl-workers).

Two notes on reading these:

- The **block/tail boundary** moved in the last pass: it used to be marked by the pre-merger
  unpad `Slice`, which disappears once `seq_len == unpadded_seq_len` makes the unpad a no-op. Both
  rows above are now split at the last `LayerNorm`, so the earlier rows' block/tail shifted by
  ~0.13/0.26 ms against what was published before. Window totals are unaffected.
- The **block boundary** matters as much as the depth. On the 9B the block *opens* with the
  `vision_replicated_acts` `AllGather` that precedes the first `LayerNorm`; on the 27B it opens at
  the `LayerNorm` itself, because that mesh's all-reduce is `AllGather` + `FastReduceNC` *after*
  `wo`/`mlp_fc2`. Put the 9B's leading 2.1 ms `AllGather` in the head by mistake and the head/block
  split is wrong by that much while the window total still looks right.
- The gated-off row is **not** the same thing as the historical pre-campaign measurement (the 9B
  depth-1 window recorded earlier in this campaign was 52.48 ms). `QWEN36_VISION_MM_TUNING=0`
  restores the pre-tuning program configs and fidelities, but a few op-graph changes are committed
  unconditionally, so it is a *reproducible* baseline rather than an archaeological one. Use it for
  A/B; do not read it as "what the tower cost before anyone touched it".

### Accuracy is the opposite — gate it at FULL depth

Error compounds block over block, so a shallow PCC check flatters the tower. Measured, demo grid,
after all three passes:

| | depth 1 | **depth 27 (real)** | depth 27, before the padding fix |
| --- | --- | --- | --- |
| 9B / N300 | 0.99977 | **0.99929** | 0.98540 |
| 27B / T3K | 0.99965 | **0.99903** | 0.98696 |

**An earlier version of this document attributed the ~0.985 to `bfloat8_b` weight error and called
it pre-existing. That was wrong.** It was the sequence padding: SDPA ran `is_causal=False` with no
`attn_mask`, so the pad rows were unmasked *keys* — zeros, so every real query scored `q·0 = 0`
against each and summed `exp(0) = 1` into its softmax denominator, attenuating every output row.
Tightening the pad (see *Sequence padding* below) took the 9B from 0.98540 to **0.99921**, an 18x
reduction in error, and the 27B from 0.98696 to **0.99897**.

The control that produced the wrong conclusion is worth recording, because it looked sound: the
untuned tower measured 0.98495, *no better* than the tuned one, which correctly ruled out the tuning
as the cause — but `QWEN36_VISION_MM_TUNING=0` does not touch `seq_len`, so both arms carried the
same padding bug and the comparison could not see it. **A gate that holds both arms equally wrong
proves the two arms match, not that either is right.**

Depth still decides whether you can see it at all: depth-1 PCC moved by 5e-7 (0.9997659 ->
0.9997664) while depth-27 moved by 0.0138. The error compounds, so only the full-depth case gates.

---

## Result

This section is the **matmul pass only**; the SDPA and redundant-op passes come after it. For the
tower's current end-to-end numbers read the depth tables above, not this one — and note the two
`whole window` cells below are at **different depths**, which is why each is labelled.

| | matmul bucket | whole window | per block | over 27 blocks | PCC |
| --- | --- | --- | --- | --- | --- |
| **9B / N300** | 11,876 → **6,887 µs** (1.72x) | 52,477 → **45,343 µs** (−13.6%) **[depth 1]** | 11,710 → **4,691 µs** (2.50x) | **−189 ms** | 0.99853 → 0.99857 |
| **27B / T3K** | 10,171 → **4,876 µs** (2.09x) | 84,362 → **77,101 µs** (−8.6%) **[depth 2]** | 3,624 → **1,489 µs** (2.43x) | **−58 ms** | 0.99816 → 0.99808 |

The 27B cell is a depth-2 window because that is how it was captured at the time; its depth-1
equivalent, measured later, is 46.63 → 43.76 ms. PCC figures in this table are depth-1/2 and so
~0.013 optimistic — see the full-depth table above.

Per-op device time, 9B / N300, one instance of each. The `before` column folds in the separate op
each change absorbed, so the comparison is like-for-like:

| op | before | after | |
| --- | --- | --- | --- |
| `patch_embed` 11008→5504 × 1536 × 576 | 865 | **730** | 1.18x |
| `qkv` 2048→1536 × 1152 × 2304 | 3281 + 937 (bias add) | **1411** | 2.99x |
| `wo` 1024→4096 × 768 × 1152 | 2725 | **294** | 9.27x |
| `mlp_fc1` 1024→3072 × 1152 × 2176 | 1863 + 1234 (GELU) | **1991** | 1.56x |
| `mlp_fc2` 1024→1536 × 2176 × 1152 | 1670 | **995** | 1.68x |
| `merger_fc1` 2752 × 4608 × 2304 | 707 | 708 | left on auto |
| `merger_fc2` 2752 × 2304 × 4096 | 764 | 757 | left on auto |

Per-op device time, 27B / T3K, one instance of each:

| op | baseline | config-tuned | + `in0` in L1 | |
| --- | --- | --- | --- | --- |
| `patch_embed` 5504 × 1536 × 1152 | 1756 | 1125 | **1122** | 1.57x |
| `qkv` 768 × 1152 × 576 | 1061 | 562 | **391** | 2.71x |
| `wo` 3072 × 192 × 1152 | 893 | 95 | **94** | 9.50x |
| `mlp_fc1` 1536 × 1152 × 544 | 929 | 769 | **703** | 1.32x |
| `mlp_fc2` 1536 × 544 × 1152 | 741 | 298 | **296** | 2.50x |
| `merger_fc1` 2752 × 4608 × 576 | 531 | 531 | 539 | left on auto |
| `merger_fc2` 1376 × 576 × 5120 | 641 | 497 | **244** | 2.63x |

---

## What was wrong

Found from a tower profile (`tests/test_vision_tower_pcc.py` under Tracy), which tagged every
one of the seven matmuls **SLOW** — both `DRAM %` and `FLOPs %` low, i.e. neither bandwidth- nor
math-bound, so a program config was on the table.

| # | Problem | Cost |
| --- | --- | --- |
| 1 | `VISION_WO_PREFILL_PROGCFG` sized `per_core_M` for 2048 rows while the matmul it configures runs 1024 → `per_core_M=11` against a 32-tile M, so `wo` ran on **24 of 64 cores** | 2725 µs on 24 cores |
| 2 | `qkv` used a hand-written 8×8 config with `in0_block_w=1`, subblock 1×1 — 36 single-tile K blocks | 3281 µs at 30% FLOPs |
| 3 | `mlp_fc1`, `mlp_fc2`, `merger_fc1`, `merger_fc2`, `patch_embed` ran on ttnn's **auto** config | — |
| 4 | `qkv` and `wo` ran **HiFi4 on bfloat8_b weights**, from `DecodersPrecision.accuracy`. Gains nothing, halves math throughput, and on Wormhole HiFi4 + fp32 accumulate is a documented accuracy hazard (the runtime warns) | ~2x on both |
| 5 | `ttnn.linear(activation="gelu")` with no explicit core grid dispatches a **separate `unary_chain` op** (`matmul.cpp`, the `user_fused_activation && !user_core_coord` branch) | 1234 µs/block + 294 µs |
| 6 | The `qkv` bias was a separate elementwise add over the whole `[1, S, 2304]` tensor | 937 µs/block |

---

## What changed

One entry point: **`VisionModelArgs.vision_mm_plan`** (`tt/vision/vision_model_config.py`). It
derives grid, `in0_block_w`, subblock, row chunk, fidelity and both memory configs from the
matmul's *actual per-device shape*, checks them against the L1 budget, and falls back to
`program_config=None` (auto, unchunked) when nothing legal fits — so it can never crash a shape
nobody measured. Call sites: `vision_attention.py`, `vision_mlp.py`, `patch_embed.py`,
`patch_merger.py`.

Per-family tuning lives in `_VISION_MM_TUNING`, with per-device overrides in
`_VISION_MM_TUNING_BY_DEVICE` keyed on `ModelArgs.device_name`. `chunk` and `in0_block_w` are
**caps**, snapped down to what is legal for the shape being run.

Final decisions, 9B / N300:

| family | chunk | grid | ibw | subblock | fidelity | in0 | out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_embed` | 5504 | 6×8 | 6 | 1×3 | HiFi2 | DRAM | DRAM |
| `qkv` | 1536 | 8×8 | 18 | 1×3 | HiFi2 | DRAM | DRAM |
| `wo` | 4096 | 6×8 | 24 | 1×6 | LoFi | DRAM | DRAM |
| `mlp_fc1` | 3072 | 8×8 | 6 | 2×3 | HiFi2_fp16 | DRAM | DRAM |
| `mlp_fc2` | 1536 | 6×8 | 4 | 1×6 | HiFi2_fp16 | DRAM | **L1** |
| `merger_fc1/2` | — | auto | — | — | HiFi2_fp16 | DRAM | DRAM |

Final decisions, 27B / T3K (`[T3K override]` marks the ones that differ from the shared rules):

| family | chunk | grid | ibw | subblock | fidelity | in0 | out |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `patch_embed` | 5504 | 8×8 | 6 | 2×1 | HiFi2 | DRAM | DRAM |
| `qkv` | 768 | 8×8 | 18 | 1×3 | HiFi2 | **L1** | **L1** |
| `wo` | 3072 | 6×8 | 6 | 1×3 | HiFi2 | DRAM | **L1** |
| `mlp_fc1` | 1536 | 8×8 | 18 | 2×3 | HiFi2_fp16 | **L1** | **L1** |
| `mlp_fc2` | 1536 | 6×8 | 17 | 1×6 | HiFi2_fp16 | **L1** (from fc1) | **L1** |
| `merger_fc1` | — | auto | — | — | HiFi2_fp16 | DRAM | DRAM |
| `merger_fc2` | 1376 | 8×8 | 9 | 1×5 | HiFi2_fp16 | DRAM | **L1** |

Plus, on both meshes: **GELU folded into the program config** (`fused_activation`) instead of the
`activation=` kwarg, and the **`qkv` bias folded into `ttnn.linear(bias=...)`**. The `qkv` bias is
safe to fold because that projection is column-parallel and its output is final; the row-parallel
biases (`wo`, `mlp_fc2`, `merger_fc2`) must stay after the collective, which would otherwise sum
them TP times.

### Fidelity

Only where HiFi4 on bfloat8_b weights was plainly wasteful:

| family | before | after | PCC effect |
| --- | --- | --- | --- |
| `qkv` | HiFi4 | HiFi2 | 1.00000 → 0.99999 |
| `wo` (9B) | HiFi4 | LoFi | 0.99997 → 0.99984 |
| `wo` (27B) | HiFi4 | HiFi2 | faster *and* more accurate (0.99987 → 0.99997) |
| everything else | — | unchanged | digit-identical |

LoFi on the four `bf16 × bf8b` families is a further **≈−46 ms/tower** on the 9B but is an accuracy
call for the model owner, not a config tweak — recorded, not taken. See `_VISION_MM_TUNING`'s
comments for the per-family numbers.

---

## Tests added

| file | purpose |
| --- | --- |
| `tests/perf/vision_matmul_specs.py` | The matmul inventory, two ways: derived analytically from `VisionModelArgs`, and **captured** from a real forward with `ttnn.linear` monkey-patched. `assert_specs_match` diffs them. |
| `tests/perf/test_sweep_vision_matmuls.py` | `test_vision_matmul_specs_match_model` (the gate above) and `test_sweep_vision_matmuls` (the tuning sweep). |
| `tests/test_vision_tower_pcc.py` | Checkpoint-free PCC gate. Also signposted, so one Tracy run yields PCC **and** the device report. |

The shape gate is what keeps the sweep honest: change a reshape granularity, a weight dtype or a
fidelity in the tower and the gate fails instead of the sweep optimising a shape nothing runs. It
passes on both meshes.

`test_wrapped_model.py` cannot run for the 9B at all — `dummy_weights=True` routes the config
through `ModelArgs.LOCAL_HF_PARAMS`, which has no `Qwen3.5-9B` entry, so it raises
`KeyError: 'Qwen3.5-9B'` before reaching the device. That gap is why `test_vision_tower_pcc.py`
exists. (Pre-existing; `test_vision_block.py`, `test_vision_attention.py` and
`test_patch_merger.py` fail the same way.)

---

## How to reproduce

```bash
cd /path/to/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0
```

9B on N300:

```bash
export HF_MODEL=Qwen/Qwen3.5-9B MESH_DEVICE=N300
```

27B on T3K — `HF_MODEL` points at the **local config dir**, since `ModelArgs` takes
`CKPT_DIR = HF_MODEL` and `model_name` from its basename, so no checkpoint or hub fetch is needed
(the tower's reference weights are config-initialised either way):

```bash
export HF_MODEL=$(pwd)/models/tt_transformers/model_params/Qwen3.6-27B MESH_DEVICE=T3K
```

Then, with either environment set:

```bash
# the depth-1 device-perf report (~40 s). Use -k, not a node id: ids change when a param is
# added, and `-k pcc`/`-k perf` would match the module+function name and select BOTH cases.
python -m tracy -p -v -r -m \
  pytest models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s -k oneblock
tt-perf-report --start-signpost start --end-signpost stop \
  "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)"

# the numerical gate: all 27 blocks (~50 s)
pytest models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py -v -s -k fulldepth

# the shape gate
pytest models/demos/blackhole/qwen36/tests/perf/test_sweep_vision_matmuls.py::test_vision_matmul_specs_match_model -v

# the tuning sweeps (matmul ~5 min for all 7 families; SDPA ~1 min)
pytest models/demos/blackhole/qwen36/tests/perf/test_sweep_vision_matmuls.py -v -s -k sweep
QWEN36_SWEEP_FAMILIES=qkv,wo QWEN36_SWEEP_PASSES=2 pytest … -k sweep   # narrower / deeper
pytest models/demos/blackhole/qwen36/tests/perf/test_sweep_vision_sdpa.py -v -s
```

Before/after on either mesh — use the gate, **not** `git stash`. One env var, no working tree
surgery, and it flips the matmul *and* SDPA tuning together (so it lands on the pre-campaign
numbers, not an intermediate row of the depth table):

```bash
QWEN36_VISION_MM_TUNING=0 pytest … -k oneblock     # or -k fulldepth
```

### Operational warnings

- **Do not pass `--dump-device-data-mid-run` on T3K.** It livelocks on the `ARC_MSG` mutex — 12+
  minutes of `Waiting for lock` with no progress. `-p -v -r` alone works on both meshes.
- **Never SIGKILL a Tracy run mid-flight.** It wedges the board (a 15 s test went to a 300 s
  timeout). `tt-smi -r` recovers in ~1 min; re-run a cheap test to confirm health before trusting
  any number.
- A bare `tt-perf-report <csv>` anchors on the last signpost and prints "No device operations
  found". Always pass `--start-signpost start --end-signpost stop`.

---

## The L1 budget

Wormhole has **1464 KB usable L1 per Tensix core** (`MEM_L1_SIZE`, `wormhole/dev_mem_map.h:33`).
Three things compete for it, and `program.cpp:1779` fires when the circular-buffer region grows
past the lowest L1-buffer address:

1. the matmul's **circular buffers**, sized by the program config;
2. an L1-interleaved **`in0`** — paged across all 64 cores, so each core holds `total / 64`;
3. an L1-interleaved **output**, likewise.

9B `qkv` (`12288×1152×2304`, bf16 both sides, chunk 1536, grid 8×8, `ibw18`):

```
in0 resident   27.0 MB / 64 =  432 KB/core
out resident   54.0 MB / 64 =  864 KB/core
CBs   in0 432 + in1 344 + out 108 + interm 108 = 992 KB/core

out only :  864 +  992 = 1856 KB  → over by 392 KB
in0 only :  432 +  992 = 1424 KB  → fits, 39 KB spare
BOTH     : 1296 +  992 = 2288 KB  → over by 824 KB
```

Even with the CBs collapsed to the minimum (`ibw1` → 259 KB) both are at 1555 KB. So on the 9B the
output alone is 864 KB/core and the two genuinely cannot coexist. On the 27B, TP=8 makes each
output ~4x narrower and they fit with 200–450 KB/core spare — which is why `in0` in L1 is taken
there and not on the 9B.

Three facts the planner needed, each found a different way. Only the first is visible on paper:

| fact | found by |
| --- | --- |
| CBs + `in0` + output share 1464 KB/core | arithmetic |
| a consumer's `in0` is placed by its **producer**, and spends the consumer's budget | the tower crashing on the 6×8 core range |
| an L1 tensor's **lifetime** spans ops an isolated sweep never runs together | the tower crashing again, same range, after the budget was already right |

The second is why `LayerNorm.forward` now takes a `memory_config` (a matmul cannot relocate its own
input) and why `vision_mm_plan` takes `in0_already_l1` — `mlp_fc2` reads `mlp_fc1`'s L1 output, so
that L1 is already spent when `mlp_fc2`'s own output placement is budgeted.

The third: `ff_in` (the norm output, 432 KB/core) was held by `VisionBlock` until `MLP.forward`
returned, so `fc2` ran with three L1 tenants — `432 + 208 + 432 + 697 = 1769 KB` against 1432 — and
the tower died while the isolated sweep measured that exact config at 298 µs quite happily. `MLP`
now frees its own input after fc1, matching `VisionAttention`, which already did (precisely why
`qkv` worked and `fc2` did not).

Two corrections to the budget model itself, both load-bearing:

- The **intermediate CB aliases the output CB** when their formats match (bf16 out, no fp32
  accumulate). Counting it twice was ~144 KB/core too pessimistic and cost the 27B's `mlp_fc2` its
  L1 output — a measured 298 → 493 µs regression.
- `_L1_RESERVE` was 100 KB of guesswork; it is now 32 KB (the `l1_small_size=24576` the demo opens
  with, plus slack). The model now agrees with **all six** cases there are measurements for.

⚠️ That makes the estimate more aggressive for **unmeasured** shapes. A new image size on the 27B
could clash where the old constant would have quietly fallen back to DRAM. If image sizes change,
re-run the sweep's `in0`/output arms and re-profile the tower.

---

## Negative results — do not re-try

| tried | outcome |
| --- | --- |
| **`BLOCK_SHARDED` L1 activation** | The layout pins `in0_block_w = K_tiles/grid_x` and `per_core_M = rows_tiles/grid_y`, forcing the single-shot form (`per_core_M=48` at 12288 rows) whose CBs alone are 2.0–8.2 MB/core. 7 of 8 arms never built; the one that did (27B `wo`) measured **408 µs against 105 µs**. |
| `in0` in L1 on the **9B** | 1.02x on `qkv` (noise), ~0.7% on `mlp_fc1` (noise), nothing on `mlp_fc2`/`wo`. It also forces `in0_block_w` 18 → 6, and that deep K-block is worth more than the L1 read. |
| A 2D config for `merger_fc1` | Isolated sweep liked it (654 → 555 µs) but **in-model it measured 559 vs auto's 531**. Left on auto. Already ~60% of the FLOP ceiling. |
| Folding the **row-parallel** biases into the matmul | Numerically wrong — the collective would sum the bias TP times. |
| Converting `in0` to L1 rather than having the producer write it there | +139–277 µs, which wipes out the win. |

---

## SDPA (second pass)

After the matmul work, SDPA was the tower's largest single op. Two defects, both the same class as the
matmul ones -- a fidelity chosen for bf16 operands, applied to bfp8:

| | before | after |
| --- | --- | --- |
| fidelity | HiFi4 (`decoders_optimizations` `SDPA_PREFILL`) | **HiFi2** |
| K dtype | **BF16** (`kv_cache_dtype`), vs Q/V in BFP8 | **BFP8** |
| chunks | 256/256 (`get_attn_sdpa_prefill_program_config`) | **256/512** (9B), **128/256** (T3K) — later re-swept to **128/512 on both**, see *Sequence padding* |

`kv_cache_dtype` never applied here: the tower is single-pass non-causal attention with no KV cache.
Besides costing SDPA a mixed-precision QKᵀ it left a **BF16 → BF16 no-op typecast** on the profile
(2 instances on the 9B, 3 on the 27B — all gone now).

Isolated, at the demo shape (`tests/perf/test_sweep_vision_sdpa.py`, seq 12288, head_dim 96):

| | baseline | winner | | PCC |
| --- | --- | --- | --- | --- |
| 9B / N300, 8 heads/dev | 20.62 ms | **15.29 ms** @ 256/512 | 1.35x | 0.999917 (baseline 0.999911) |
| 27B / T3K, 2 heads/dev | 6.79 ms | **4.56 ms** @ 128/256 | 1.49x | 0.999886 (baseline 0.999898) |

In-model, per block, and over the 2-block signpost window (`test_vision_tower_pcc.py` under Tracy):

| | SDPA/block | window **[depth 2]** | tower PCC |
| --- | --- | --- | --- |
| **9B / N300** | 18,119 → **14,430 µs** (1.26x) | 79.46 → **71.81 ms** (−9.6%) | 0.9985658 → **0.9985879** |
| **27B / T3K** | 6,103 → **4,265 µs** (1.43x) | 77.06 → **73.31 ms** (−4.9%) | 0.9980751 → 0.9980751 |

These windows are depth 2 (this pass predates pinning the profile at depth 1). The SDPA/block and
PCC columns are depth-independent; the window column is not comparable to the depth-1 tables above.

Over 27 blocks that is **−100 ms** on the 9B and **−50 ms** on the 27B. The 9B's PCC *improved*: a
larger `k_chunk` means fewer flash accumulation steps, so less accumulated softmax error.

Why `k_chunk = 2 × q_chunk` but a different `q_chunk` per mesh: the kernel parallelises over
`heads × q_chunks` across 64 cores. 8 heads/device gives 8 × 48 = 384 units at q=256; 2 heads/device
gives only 96, so the 27B wants q=128 for 192. `512/512` is rejected on both meshes — the flash CBs
reach 1,949,888 B against L1's 1,499,136 B.

**LoFi was rejected.** It is ~3% faster than HiFi2 and lands at **PCC 0.9656 per op**, which no
27-block tower survives. The sweep enforces a `PCC_FLOOR` because fastest-wins picks LoFi every time.
`exp_approx_mode=True` measured marginally *slower* on both meshes (15.37 vs 15.29; 4.65 vs 4.56).

## Redundant data-movement ops (third pass)

Layout/dtype ops were 4.4 ms of the 2-block window. Three were genuinely redundant:

| what | why it was redundant | fix |
| --- | --- | --- |
| `FillPad` ×2 **per block** | `VisionAttention.forward_prefill` padded `rot_mats` 72 → 96 itself, so the same two tensors were re-padded 27× per image | pad once on host in `DropInVisionTransformer`; the in-attention pad now gates on the tensor's own last dim, so it is a no-op there and still correct for other callers |
| `Typecast FP32→BF16` ×4 + fp32 `Tilize` | `pixel_values`, the bilinear weights and cos/sin were uploaded as **fp32**, so ttnn tilized fp32 on device and then typecast | cast on host before `from_torch` — also halves the bytes over PCIe |
| `Pad` ×2 | cos/sin rows were padded to `seq_len` on device | folded into the same host pad |

| | TM ops | tower (27 blocks) |
| --- | --- | --- |
| **9B / N300** | 4,444 → **2,912 µs** (−34%) | 831.6 → **827.6 ms** |
| **27B / T3K** | 4,258 → **2,683 µs** (−37%) | 906.4 → **852.0 ms** |

**The 27B tower figure was previously recorded as 785.9 → 784.4 ms and that was wrong.** It implied a
28.7 ms block; the measured depth-1 split gives 31.22 ms. Both `tower` cells above are now
`head + 27 x block + tail` from the depth-1 tables at the top of this document, so they are
reproducible from a single `-k oneblock` profile per mesh. The 9B's own figure moved only 2.3 ms
(831.6 → 827.6), which is why the error went unnoticed on that mesh.

`FillPad` went 6 → 0 on both meshes. PCC is bit-identical on the 9B (0.9985878806669486 before and
after), so the host cast and host pad reproduce the device ops exactly. Note the tower delta is
smaller than the TM delta because most of what was removed sits in the once-per-image head (4.38 →
3.04 ms on the 9B); the per-block part is at the measurement noise floor.

**Not removed, and why** (each measured, so the cost is known):

- **`Typecast BF16→BF8B` ×3 per block, 473 µs/block ≈ 12.8 ms/tower.** Q/K must be bf16 —
  `rotary_embedding_llama` hard-asserts it (`rotary_embedding_llama_device_operation.cpp:74`) — and
  SDPA wants bf8b. Only V's cast is avoidable, and it is a wash: −160 µs of typecast for ~+150 µs of
  SDPA on bf16 V.
- **The merger's `Untilize` + `ReshapeView` + `Tilize`, 1,059 µs/image.** The ROW_MAJOR round trip is
  the documented workaround for the tilized-reshape hang (tt-metal#29932).
- **The merger's GELU as a separate 294 µs `UnaryDeviceOperation`/image**, because `merger_fc1` stays
  on ttnn's auto config and `activation=` cannot fuse without a program config — and a forced config
  measured slower in-model (see `_VISION_MM_TUNING`).
- `NlpCreateHeads` (664 µs/block), `NLPConcatHeads` (140 µs/block) and the pre-merger unpad `Slice`
  (128 µs/image) are structural.

Host-side per-image torch work is **1.48 ms** total (the RoPE index/trig math), so there is no host
prize here either: the tower is ~90% device-bound against the demo's measured 1.02 s.

## Sequence padding (fourth pass)

The tower padded its row count to a multiple of **2048**, on this justification:

```python
# Calculate padded sequence length (divisible by 2048) required by
# models/tt_transformers/tt/attention.py::forward_prefill
seq_len = ((unpadded_seq_len // 2048) + 1) * 2048
```

**The vision tower never calls that file** — it has its own `VisionAttention`. The actual constraints
are far looser:

| claimed | actual |
| --- | --- |
| multiple of 2048 | `VisionAttention.forward_prefill` asserts `seq_len % 128 == 0` |
| — | every matmul `chunk` is derived as a **divisor** of the row count, so any tile-aligned length is legal |
| — | ttnn's plain SDPA validates only `q_chunk_size % 32` and `k_chunk_size % 32`; there is **no** seq-vs-chunk divisibility requirement, the flash kernel handles a ragged tail |

So 128 is the whole requirement, and the demo grid needs **zero** padding: 11008 = 86 × 128. The fix
is `-(-n // 128) * 128`, which also repairs an off-by-one — `(n // m) + 1` over-pads exact multiples,
so a 4096-patch image went to 6144: 1.5x the rows and **2.25x the SDPA** for nothing.

| | SDPA/block | window (depth 1) | 27-blk | full-depth PCC |
| --- | --- | --- | --- | --- |
| **9B / N300** | 14,429 → **12,585 µs** (1.15x) | 39.96 → **36.22 ms** | 831.0 → **731.4 ms** | 0.98540 → **0.99921** |
| **27B / T3K** | 4,262 → **3,501 µs** (1.22x) | 40.38 → **36.35 ms** | 858.7 → **756.6 ms** | 0.98696 → **0.99897** |

Two ops vanish outright: the cos/sin row `Pad` (nothing left to pad) and the pre-merger unpad
`Slice` (a no-op once `seq_len == unpadded_seq_len`). Everything that moves `rows × dim` bytes drops
~10% with it — on the 27B the `AllGather` saving (20.5 → 18.3 ms/block) is worth **more than the
SDPA saving**.

### The accuracy half was the bigger win, and it was invisible by construction

SDPA runs `is_causal=False` with **no `attn_mask`**. The pad rows were therefore unmasked *keys*:
zeros, so every real query scored `q·0 = 0` against each of the 1280 of them and summed `exp(0) = 1`
into its softmax denominator. Removing them is worth 18x on the 9B's full-depth PCC — see *Accuracy*
near the top for why the earlier "pre-existing `bfloat8_b` error" conclusion was wrong.

### Re-sweeping the chunks was mandatory, not optional

The winning chunk pair is a function of the row count, and changing the row count invalidated it.
Both meshes now want **q=128 / k=512**, so `_VISION_SDPA_TUNING_BY_DEVICE` is empty:

| | was (@12288) | now (@11008) | |
| --- | --- | --- | --- |
| 9B / N300 | 256/512 → 14.22 ms | **128/512 → 13.43 ms** | 1.06x |
| 27B / T3K | 128/256 → 4.08 ms | **128/512 → 3.80 ms** | 1.07x |

The kernel parallelises over `heads × q_chunks` across 64 cores, and that is what moved:

- at 12288, `8 × (12288/256) = 384` units is **exactly** 6 rounds of 64 — `q=256` was optimal by
  coincidence of the row count, not because of the head count
- at 11008, `8 × (11008/256) = 344` still needs 6 rounds, so 40 slots idle (10.4% waste); `q=128`
  gives `8 × 86 = 688` in 11 rounds (2.3% waste)
- `k=512` still wins despite 11008 = 21×512 + 256 leaving a half-empty last chunk

This is also why SDPA fell only 1.15x when the row count fell 1.12x and the work is *quadratic*: the
predicted 1.25x never materialised because the q dimension is quantised to core-rounds. **The second
pass's stated rationale — "the winning q_chunk depends on the head count" — was a two-point fit that
happened to hold.** `exp_approx=True` measured 13.35 vs 13.43 ms on the 9B: 0.6%, inside noise, and
it accumulates error across `SEQ/k_chunk` flash chunks over 27 blocks. Not taken.

## CCL workers

The five AllGather / ReduceScatter sites hardcoded `chunks_per_sync=10, num_workers_per_link=2`.
Text prefill on this repo already moved N300 to `wpl=4` (−19% AllGather). Vision re-swept the same
knob on its own shapes (`QWEN36_VISION_CCL=cps,wpl`, demo grid, `-k oneblock`). `num_links` stays 1
on both N300 and T3K (`get_num_links` is hard-fatal above that).

| | AllGather | ReduceScatter | depth-1 window |
| --- | --- | --- | --- |
| **9B / N300** `wpl=2` | 1.90+1.93+1.93 = 5.77 ms (5 cores) | 1.60+1.49+1.28 = 4.37 ms | 36.52 ms |
| **9B / N300** `wpl=4` | 1.54+1.55+1.60 = **4.69 ms (−19%)** (9 cores) | 1.57+1.51+1.30 = 4.39 ms (wash) | **35.37 ms** |
| **9B / N300** `wpl=8` | 4.66 ms (17 cores) | 4.40 ms | 35.46 ms (noise vs 4) |
| **27B / T3K** `wpl=2` | 11.40+11.42 = 22.82 ms (6 cores) | 1.41 ms (merger) | 41.55 ms |
| **27B / T3K** `wpl=4` | 10.92+10.97 = **21.89 ms (−4%)** (10 cores) | 1.37 ms | **40.59 ms** |

Shipped default is `(10, 4)` from `vision_ccl_tuning()`, **only when** `is_wormhole_b0()` and
`device_name` is `N300` or `T3K`. Every other SKU (Blackhole `P150` / `P150x4` included) keeps
`(10, 2)`. `QWEN36_VISION_CCL=0` forces the untuned pair; `QWEN36_VISION_CCL=cps,wpl` overrides on
any arch for sweeps. Wired into LN AllGather, `wo`/`mlp`/`merger` `tt_all_reduce`, and
`all_reduce_replicated`. Full-depth PCC: 9B **0.99929**, 27B **0.99903**.
`chunks_per_sync` left at 10 (text already measured it as a no-op). `wpl=8` matches 4 and spends
more cores — not taken. Do **not** bf8 the 27B residual gather without a separate PCC decision.

The 27B win is the fabric floor talking: 7 ring hops on 1 link, ~25 MB/gather. Bytes are the only
remaining lever — pad `dim` 1152→1280 so fracture+reduce-scatter works, or a dtype drop.

## Remaining headroom, ranked

1. **27B: the two residual AllGathers are still ~54% of the depth-1 window** (21.9 ms of 40.6 ms;
   3.7 ms of matmuls). `wpl=4` took 4% off them and that is the end of this knob. The structural
   cost is `all_reduce_replicated`: vision `dim=1152` is 36 tiles and TP=8 cannot split that into
   whole tiles, so each `wo`/`mlp_fc2` all-gathers the full activation 8-ways on dim 0. Pad `dim` to
   `32×8=1280` to restore fracture+reduce-scatter, or gather in bf8 — neither is this pass.
2. **9B: SDPA is still the top bucket at 34.7%** of the depth-1 window (12.6 ms/block, down from
   18.1 across three passes). What is left is genuinely structural: `head_dim` 72 tile-pads to 96,
   so a quarter of every QKᵀ and PV is arithmetic on zeros, and 72 is not a tile multiple so 96 is
   the tightest legal pad. Config space is exhausted — fidelity, K dtype, chunk sizes and
   `exp_approx` have all been swept twice, and `fp32_dest_acc_en=False` is a ~0.94 PCC cliff.
   Against the ~66 TFLOP/s ceiling that `fp32_dest_acc_en=True` actually allows, SDPA runs at ~49%,
   with flash softmax accounting for most of the rest.
3. **The row padding is now tight for the demo grid but not in general.** 11008 = 86 × 128 needs no
   pad at all; an image whose patch count is not a multiple of 128 still gets up to 127 pad rows,
   and those rows are still **unmasked keys** in SDPA. The residual error is ~127/11008 of what was
   just removed, so it is small — but the clean fix is an `attn_mask` (or a `cu_seqlens`-aware SDPA
   variant), not more padding arithmetic. Worth doing before shipping non-demo image sizes.
4. **LoFi on the four `bf16 × bf8b` matmul families** — ≈−46 ms/tower on the 9B, at ~0.9998 per-matmul
   PCC over a 27-block tower. An accuracy decision. (LoFi on *SDPA* is a different story and already
   rejected — see above.)
5. **`fp32_dest_acc_en=True` on `wo` / `mlp_fc1` / `mlp_fc2`** — the opposite trade, and the one place
   where spending time buys accuracy. Those three run `lofi` / `hifi2_fp16` / `hifi2_fp16`, all of
   which accumulate the K dim in a bf16 dest. Forcing them to `hifi2` costs **+9 ms/tower** (798 → 806
   ms on the 9B/N300) and is worth **+0.002 real-weight PCC at full depth** (0.98850 → 0.99034). NOT
   taken: the two precision fixes that ship (`VisionAttention.attn_out_dtype`, and `fp32_dest_acc_en`
   on the block LayerNorms) already buy +0.0198 for +8.7 ms, so this is the same money for a tenth of
   the return. Revisit if the tower ever needs the last 0.002.

## Not covered

- **Blackhole `P150` / `P150x4`**, the `README.md` deployment targets. No BH card was available, so
  the tuning is **gated off** there (`vision_mm_tuned`, and `vision_ccl_tuning` only enables `(10, 4)`
  on Wormhole `N300`/`T3K`) rather than shipped unmeasured: its 13×10
  grid lets `_grid_extent` pick `grid_x=12`, and the L1 budget's `_L1_PER_CORE`/`_L1_RESERVE` and the
  fidelity walk are Wormhole measurements. To lift the gate: `MESH_DEVICE=P150x4 pytest … -k sweep`,
  add a `"P150x4"` entry to `_VISION_MM_TUNING_BY_DEVICE` wherever the swept winner beats what the
  shared rules derive, re-check the PCC test, then widen `vision_mm_tuned` to include the arch.
- Sharding modes other than block-sharded `in0`: height/width-sharded `in0`, **sharded weights**
  (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`), a sharded **output**,
  `ttnn.experimental.minimal_matmul`, and **1D mcast** (the sweep supports a `"1d"` variant but
  never generates candidates for it).
- SDPA at other image sizes: its chunk sizes are fixed constants, not caps snapped to the shape the
  way the matmul knobs are. They stay legal at any tile-aligned sequence length, but only 12288 has
  been swept and the winning `q_chunk` depends on head count.
- Image sizes other than the demo grid. The planner adapts (every knob is a cap snapped to what is
  legal), but only `1×86×128` has been swept and profiled.
