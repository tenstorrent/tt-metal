# Vision tower device-performance optimisation

Status of the matmul optimisation pass on `tt/vision/` (the `DropInVisionTransformer` tower:
patch/positional embed → 27 × `VisionBlock` → `PatchMerger`).

**Measured on Wormhole**, since that is the hardware the work was done on: Qwen3.5-9B on an **N300**
(TP=2, activations fractured) and Qwen3.6-27B on a **T3K** (TP=8, activations replicated — see
`tt/vision/vision_ccl.py`). The `README.md` deployment targets are Blackhole `P150` / `P150x4`;
those are **not swept** — see [Not covered](#not-covered).

**The tuning is gated to Wormhole B0 in code**: `VisionModelArgs.vision_mm_tuned` is
`is_wormhole_b0()`, and off-arch `vision_mm_plan` returns only its untuned plan — ttnn's auto matmul
config, DRAM in and out, and the pre-sweep fidelity (`decoders_optimizations` for `qkv`/`wo`). So a
Blackhole run gets the op graph it had before this pass. `QWEN36_VISION_MM_TUNING=0` forces that path
on any arch, which is how the fallback is tested (both meshes pass PCC on it).

All numbers below are `tt-perf-report` device time for the demo image (patch grid `1×86×128` =
11008 patches → 12288 padded), which is what `demo/benchmark_vision.py` and `demo/vision_demo.py`
default to. The tower runs 2 blocks under the profile, so block ops appear twice and the per-block
figures are one instance; `patch_embed` and the two `merger` matmuls run once per image regardless.

---

## Result

| | matmul bucket | whole window | per block | over 27 blocks | PCC |
| --- | --- | --- | --- | --- | --- |
| **9B / N300** | 11,876 → **6,887 µs** (1.72x) | 52,477 → **45,343 µs** (−13.6%) | 11,710 → **4,691 µs** (2.50x) | **−189 ms** | 0.99853 → 0.99857 |
| **27B / T3K** | 10,171 → **4,876 µs** (2.09x) | 84,362 → **77,101 µs** (−8.6%) | 3,624 → **1,489 µs** (2.43x) | **−58 ms** | 0.99816 → 0.99808 |

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
# PCC + the device-perf report from ONE run (~50-85 s)
python -m tracy -p -v -r -m \
  pytest "models/demos/blackhole/qwen36/tests/test_vision_tower_pcc.py::test_vision_tower_pcc[wormhole_b0-patches11008_depth2-mesh_device0-device_params0]"
tt-perf-report --start-signpost start --end-signpost stop \
  "$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)"

# the shape gate
pytest models/demos/blackhole/qwen36/tests/perf/test_sweep_vision_matmuls.py::test_vision_matmul_specs_match_model -v

# the tuning sweep (~5 min, all 7 families)
pytest models/demos/blackhole/qwen36/tests/perf/test_sweep_vision_matmuls.py -v -s -k sweep
QWEN36_SWEEP_FAMILIES=qkv,wo QWEN36_SWEEP_PASSES=2 pytest … -k sweep   # narrower / deeper
```

Before/after on either mesh:

```bash
git stash push -- models/demos/blackhole/qwen36/tt/vision/{vision_model_config,vision_attention,vision_mlp,patch_embed,patch_merger,vision_block,vision_layernorm,vision_distributed_layernorm}.py
# …run…
git stash pop
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

## Remaining headroom, ranked

1. **27B: collectives are 59% of the window.** `all_reduce_replicated` all-gathers the full
   activation 8-ways on dim 0 (8 × 28 MB per MLP call) because vision `dim=1152` is 36 tiles and
   TP=8 cannot split that into whole tiles. Per block: **AllGather 7.8 + 12.7 ms and FastReduceNC
   0.8 + 1.6 ms against 1.49 ms of matmuls.** This is worth an order of magnitude more than
   anything left in the 27B matmuls.
2. **9B: SDPA is 40%** of the window (18.1 ms/block, untouched). Levers: `get_attn_sdpa_program_config`
   chunking, `SDPA_PREFILL` fidelity (HiFi4 today), and the padding in (3).
3. **The sequence-length rounding is wasteful.** `seq_len = ((n // 2048) + 1) * 2048` rounds up even
   when `n` is already a multiple of 2048, so a 4096-patch image is padded to 6144 — 1.5x the rows
   and 2.25x the SDPA, for free. The demo grid (11008) is unaffected, which is why the profile does
   not show it. `ceil` would fix it and stays compatible with the `% 2048` requirement.
4. **LoFi on the four `bf16 × bf8b` families** — ≈−46 ms/tower on the 9B, at ~0.9998 per-matmul PCC
   over a 27-block tower. An accuracy decision.

## Not covered

- **Blackhole `P150` / `P150x4`**, the `README.md` deployment targets. No BH card was available, so
  the tuning is **gated off** there (`vision_mm_tuned`) rather than shipped unmeasured: its 13×10
  grid lets `_grid_extent` pick `grid_x=12`, and the L1 budget's `_L1_PER_CORE`/`_L1_RESERVE` and the
  fidelity walk are Wormhole measurements. To lift the gate: `MESH_DEVICE=P150x4 pytest … -k sweep`,
  add a `"P150x4"` entry to `_VISION_MM_TUNING_BY_DEVICE` wherever the swept winner beats what the
  shared rules derive, re-check the PCC test, then widen `vision_mm_tuned` to include the arch.
- Sharding modes other than block-sharded `in0`: height/width-sharded `in0`, **sharded weights**
  (`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`), a sharded **output**,
  `ttnn.experimental.minimal_matmul`, and **1D mcast** (the sweep supports a `"1d"` variant but
  never generates candidates for it).
- Image sizes other than the demo grid. The planner adapts (every knob is a cap snapped to what is
  legal), but only `1×86×128` has been swept and profiled.
