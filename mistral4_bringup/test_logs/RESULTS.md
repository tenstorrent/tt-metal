# Mistral-Small-4-119B prefill test sweep

Env: `TT_METAL_HOME=/data/ssalice/temp/tt-metal`, `MISTRAL4_HF_MODEL=/data/kmabee/models/Mistral-Small-4-119B-2603`,
`TT_METAL_OPERATION_TIMEOUT_SECONDS=120`. Mesh 8x4 (SP=8, TP=4), Blackhole galaxy.
Link health at sweep start: 297 UP / 87 DOWN.

## Pre-existing test (no new wiring needed)

### test_mla.py::test_mistral4_mla  -k "8x4"   -> log: mla_8x4_full.log
| case | result | output PCC | KVPE KV PCC | KVPE PE PCC | call time |
|---|---|---|---|---|---|
| seq5k random     | PASS | 0.998161 | 0.999901 | 0.999901 | 13.19s |
| seq5k pretrained | PASS | 0.999331 | 0.999886 | 0.999899 |  8.43s |
| seq25k random    | PASS | 0.998057 | 0.999903 | 0.999904 | 12.88s |
| seq25k pretrained| PASS | 0.998613 | 0.999891 | 0.999902 | 38.25s |

4 passed / 0 failed, 90.89s total.

## New test entries

### test_kv_cache_table.py::test_mistral4_kv_cache_table  -> log: kv_cache_table_mistral4.log
| case | result | call time | notes |
|---|---|---|---|
| `[blackhole-mistral4-seq5k-line-8x4]` | PASS | 11.84s | kvpe latent 320 wide, chunk 10880 B, TP-replicated |

1 passed / 0 failed, 38.43s wall.

### Variant-parametrized entries (one param added per file)

| test | selector | result | log | note |
|---|---|---|---|---|
| `op_unit_tests/test_ttnn_dispatch_combine.py::test_ttnn_dispatch_combine` | `-k "mistral4-640-avg"` | **8 passed**, 80 skipped | `vp_dispatch_combine.log` | runs on 8x4; scaledown `//4` → 1 expert/chip |
| `op_unit_tests/test_prefill_dispatch.py::test_ttnn_dispatch` | `-k "mistral4-perf_no_pcc and mesh-8x4"` | **8 passed**, 24 skipped | `vp_prefill_dispatch_perf.log` | ditto |
| `op_unit_tests/test_prefill_dispatch.py::test_ttnn_dispatch` | `-k "mistral4-pcc and mesh-8x4"` | 12 failed | `vp_prefill_dispatch.log` | **pre-existing**, not mistral4 — baseline `dsv3-pcc` fails identically (`diag_dsv3_dispatch_8x4.log`). `//16` → 8 experts over 32 chips → `experts_per_chip == 0` → `ZeroDivisionError` at `tt/moe/init_helpers.py:245`. See F7. |
| `op_unit_tests/test_prefill_combine.py::test_ttnn_combine` | `-k "mistral4 and mesh-8x4"` | 32 skipped | `vp_prefill_combine_8x4.log` | **pre-existing** — 8x4 entry needs `FABRIC_2D_TORUS_XY`; no wrap on the 4-wide TP axis here. `dsv3` skips 32/32 with the identical message (`diag_dsv3_combine_8x4.log`). |
| `op_unit_tests/test_reduce.py::test_ttnn_reduce_models` | `-k mistral4` | 4 skipped | `vp_reduce.log` | mesh axis offers only 4/8-chip shapes; *"Blackhole only supports 32-device mesh configs"*. Entry is correct, unreachable on this box. |
| `cache/test_mla_cache.py::test_mla_weights_cold_warm_cache` | `-k mistral4` | 2 skipped | `vp_mla_cache.log` | same reason |
| `pcc/test_parallel_embedding.py::test_parallel_embedding` | `-k mistral4` | 2 skipped | `vp_parallel_embedding.log` | same reason |

Passing total from this group: **16 passed**, 0 genuine mistral4 failures.

### test_mla.py::test_mla_chunked_prefill — mistral4 variant added (1404 params)

Selector: `-k "mistral4 and 8x4 and fabric2d and scalar and no_determinism and cpu and (plain-5k or rot-aligned_min or rot-midchip_straddle)"`
Log: `mla_chunked_mistral4.log`. **3 passed / 0 failed.**

These are the only three scenarios whose every position stays under 8192, so they are the ones where
the reference comparison is a fidelity statement rather than just a bookkeeping check (see F1).

| scenario | iter | out PCC | full measured PCC | KV k_nope | KV k_pe |
|---|---|---|---|---|---|
| `plain-5k` (5120) | 0 (kv=0, isl=640) | 0.999560 | 0.999286 | 0.999886 | 0.999899 |
| | 1 (kv=640, isl=5120, rotated) | 0.999133 | | | |
| `rot-aligned_min` (5760) | 0 (kv=0, isl=672) | 0.999554 | 0.999285 | 0.999886 | 0.999899 |
| | 1 (kv=672, isl=5120, rotated) | 0.999133 | | | |
| `rot-midchip_straddle` (5792) | 0 (kv=0, isl=5120) | 0.999302 | 0.999302 | 0.999886 | 0.999899 |

This is the first evidence that the **chunked path runs for mistral4 on device**. `ring_mla` sees the
absorbed widths (DH = 256+64 = 320, VDH = kv_lora_rank = 256), not the 128-wide per-head dims, so all
five latent-V asserts in `ring_joint_sdpa_device_operation.cpp:573-595` hold. Chunked mode is selected
by a plain `is_chunked` constructor kwarg (`tt/mla/mla.py:608-614`) — mistral4 was only ever
single-shot because every call site used the default.

### test_prefill_block.py::test_mistral4_prefill_block (NEW) -> log: prefill_block_mistral4.log

Composed CPU reference (GLM pattern) via `mistral4_decoder_layer_reference`, threshold 0.98,
`apply_llama4_attn_scale=False` to match the device. Random weights.

| case | result | block output PCC | call time |
|---|---|---|---|
| `[blackhole-mistral4-dense-seq5120-mesh-8x4]` | PASS | **0.999878** | 41.27s |
| `[blackhole-mistral4-moe-seq5120-mesh-8x4]` | PASS | **0.995178** | 146.86s |

2 passed / 0 failed.

The MoE case was expected to fail (softmax reference vs sigmoid device gate) and did not — the
divergence costs only ~0.005 PCC. See F0: the gap is real but sits below this comparison's detection
floor, so the green result must not be read as "routing is correct".
The `dense` case is architecturally synthetic (`first_k_dense_replace = 0`, so no real layer is
dense) and needs a test-local `NUM_DENSE_LAYERS = 1` shim; it is a block-plumbing check.

## Perf metrics (Tracy op profiling)

No rebuild was needed: `build_Release` already has `ENABLE_TRACY=ON` in its `CMakeCache.txt`
(`cmake/project_options.cmake:7` defaults it ON — `build_metal.sh` only offers `--disable-profiler`,
so you opt *out*, not in). Any doc saying `--enable-profiler` / `-DENABLE_TRACY=ON` is stale.

Command (`mistral4_bringup/test_logs/run_tracy.sh`):
```
python -m tracy -p -r -v --op-support-count 100000 -m pytest \
  "models/demos/deepseek_v3_d_p/tests/test_mla.py::test_mistral4_mla[blackhole-mistral4-sequential-check_pcc-seq5k-max_sl-random-line-8x4]" -v -s
```
**Gotcha:** a quoted `-k "a and b"` does NOT survive Tracy's re-exec — it splits on spaces and pytest
fails with `file or directory not found: and`. Pass a full node ID.
`--op-support-count` must be raised well above its 1000 default or ops are dropped.

Artifacts: `mistral4_bringup/perf_analysis/mistral4_mla_seq5k_ops.csv` (1088 device-op rows,
32 devices) and `..._analysis.txt`.

### One MLA layer forward, seq 5120, 8x4 (SP=8 / TP=4)

Per-device kernel-duration sum: **max 1.984 ms** (device 16), min 1.063 ms (device 27) —
a 1.87x spread across the mesh. Latency is the slowest device, never the sum
(`tracy_guide_docs/README.md:736-746`).

| operation | max per-device | share | rows |
|---|---|---|---|
| `RingJointSDPADeviceOperation` | 1.006 ms | **50.7%** | 32 |
| `ReduceScatterMinimalAsyncDeviceOperation` | 0.342 ms | 17.2% | 64 |
| `HighBwAllGatherDeviceOperation` | 0.253 ms | 12.8% | 64 |
| `MatmulDeviceOperation` | 0.227 ms | 11.5% | 192 |
| `LayerNormDeviceOperation` | 0.025 ms | 1.3% | 64 |
| `ConcatDeviceOperation` | 0.023 ms | 1.1% | 64 |
| `TilizeDeviceOperation` | 0.022 ms | 1.1% | 128 |
| `RotaryEmbeddingLlamaDeviceOperation` | 0.022 ms | 1.1% | 64 |
| `SliceDeviceOperation` | 0.016 ms | 0.8% | 128 |
| `NlpCreateHeadsDeviceOperation` | 0.016 ms | 0.8% | 32 |
| `NLPConcatHeadsDeviceOperation` | 0.016 ms | 0.8% | 32 |
| remainder (typecast, reduce, KV update, generic) | 0.022 ms | 1.0% | 224 |

**Two things worth acting on:**
1. **Ring SDPA is half the layer.** The single longest op is `RingJointSDPADeviceOperation` at
   1.006 ms on device 16, and the 8 slowest individual ops are all this op on devices 8–23 —
   i.e. the middle SP ranks, which is where a ring's gather depth peaks.
2. **Collectives are ~30% of the layer** (reduce-scatter 17.2% + high-bw all-gather 12.8%),
   against only 11.5% in matmul. For a dense-MLA layer at SP=8/TP=4 that is the ratio to attack
   first — and it is consistent with the two TP all-reduces in `_q_a_latent`
   (`mla.py:983,998`) plus the SP gather inside ring attention.

Profiling overhead is large: 231 s under Tracy vs 34 s unprofiled for the same single case. Expected
— do not read wall-clock from a profiled run.

### pcc/test_ttnn_moe.py::test_mistral4_moe (NEW) -> log: moe_pcc_mistral4.log

`[blackhole-mistral4-mesh-8x4-mistral4-5k-pcc]` — **PASS**, 98.35s call.
Random weights (pretrained MoE is blocked by `packed_expert_checkpoint = True`).
seq 640/chip = 5120 total, 128 experts top-4, 4 experts/chip, `GateComputeMode.DEVICE_FP32`.

| check | PCC | threshold | margin |
|---|---|---|---|
| `shared_output` | 0.999761 | 0.997 | +0.0028 |
| `routed_output` | 0.974983 | 0.960 | +0.0150 |
| `final_output` | 0.994665 | 0.982 | +0.0127 |
| **`reference_output`** (softmax reference vs sigmoid device) | **0.972469** | **0.971** | **+0.0015** |

Stage timings: torch_forward 3.6s, tt_moe_creation 10.3s, tt_forward 27.1s, pcc_validation 1.6s.

The `reference_output` margin is the number that matters: **+0.0015**, i.e. essentially noise. The
reference here uses the model's real **softmax** router while the device runs **sigmoid**, and this
comparison still passes. A different seed could flip it either way. See F0 — the sign of this margin
says nothing about whether the routing rule is correct.

Note this test is *pessimistic* relative to the true device-vs-model gap: `create_gate_weights`
emits an `e_score_correction_bias` (σ = 0.01) that the device's noaux_tc selection consumes
(`tt_moe_gate_prefill.py:794`) but Mistral's router does not have. Measured on the fixture tensors,
that bias alone costs more (Σ(Δw)² = 0.046) than the softmax-vs-sigmoid difference (0.023), and
changes the 4th-ranked expert on ~40% of tokens (top-4 overlap 3.598/4 with bias, 4.000/4 without).
`test_mistral4_prefill_block` zeroes the bias, which is why its MoE PCC is a cleaner 0.995178.

### Same run through `tt-perf-report` (signposted MLA window, devices merged)

```
tt-perf-report mistral4_mla_seq5k_ops.csv --start-signpost MLA_START --end-signpost MLA_END \
  --no-color --csv mistral4-mla-detailed.csv --summary-file mistral4-mla-by-op > mistral4-mla-report.txt
```
**Both signposts are mandatory.** The test emits `MLA_START` / `MLA_END`; without the explicit flags
the tool takes the *last* signpost (`MLA_END`) and reports `No device operations found` — a silently
empty report, not an error. (`--summary-file` also writes a `.png` alongside the `.csv`; the `.txt`
is a shell redirect, not a tool feature.)

Artifacts in `mistral4_bringup/perf_analysis/`: `mistral4-mla-by-op.csv`, `mistral4-mla-by-op.png`,
`mistral4-mla-detailed.csv`, `mistral4-mla-report.txt`.

| share | op | device time | count | category | weighted mean FLOPs |
|---|---|---|---|---|---|
| **60.15%** | `RingJointSDPADeviceOperation` | 1005.65 µs | 1 | Other | — |
| 13.70% | `MatmulDeviceOperation` | 229.07 µs | 6 | Compute | **15.55%** (min 5.52 / max 27.21) |
| 11.05% | `ReduceScatterMinimalAsyncDeviceOperation` | 184.72 µs | 2 | Other | — |
| 7.08% | `HighBwAllGatherDeviceOperation` | 118.40 µs | 2 | Other | — |
| 1.51% | `LayerNormDeviceOperation` | 25.26 µs | 2 | Compute | — |
| 1.38% | `ConcatDeviceOperation` | 23.09 µs | 2 | TM | — |
| 1.31% | `RotaryEmbeddingLlamaDeviceOperation` | 21.86 µs | 2 | Compute | — |
| 1.02% | `SliceDeviceOperation` | 17.01 µs | 4 | TM | — |
| 0.96% | `NlpCreateHeadsDeviceOperation` | 16.05 µs | 1 | TM | — |
| 0.96% | `NLPConcatHeadsDeviceOperation` | 16.00 µs | 1 | TM | — |
| 0.40% | `FastReduceNCDeviceOperation` | 6.65 µs | 1 | Compute | — |
| 0.29% | `UpdateKVCacheOperation` | 4.84 µs | 1 | Other | — |
| 0.19% | `TypecastDeviceOperation` | 3.18 µs | 1 | TM | — |

**Overall DRAM roofline: 2.9% (15 GB/s).**

Reading of the MLA layer at seq 5120, SP=8/TP=4:
- **One `ring_joint_scaled_dot_product_attention` call is 60% of the layer.** It is a single op and
  the single biggest lever.
- **Collectives are 18.1%** (reduce-scatter 11.05 + high-bw all-gather 7.08) — more than matmul.
- **Matmul is only 13.7% of time at ~15.6% weighted FLOP utilisation**, spread across 6 ops
  (min 5.5%, max 27.2%). There is a lot of headroom in the GEMMs.
- At **2.9% of DRAM roofline** this layer is neither bandwidth- nor FLOP-bound; it is bound by the
  SDPA kernel and the fabric.

Two caveats on the numbers. (1) `tt-perf-report` **merges** devices (max for compute, mean for
AllGather/ReduceScatter), while `analyze_tracy_csv.py` reports the slowest device — hence 1.984 ms
there vs these merged figures. The tools disagree by design; do not cross-compare. (2) It also emits
`Unclassified operation` warnings for `ReduceScatterMinimalAsyncDeviceOperation`,
`HighBwAllGatherDeviceOperation` and `UpdateKVCacheOperation`, so ~71.5% of this layer lands in
"Other" and the by-category view is not meaningful for this model until those are added to
`OPERATION_CATEGORIES`.

### test_prefill_transformer.py::test_mistral4_prefill_transformer (NEW) -> log: prefill_transformer_mistral4_2L.log

Own body (not `run_model`, which dead-ends for mistral4 on *both* weight paths — the random branch
needs `reference_model_cls`, unset; the pretrained branch needs it too and `packed_expert_checkpoint`
gives attention only). Chains `mistral4_decoder_layer_reference` per layer, then rms_norm + lm_head,
producing the same `["embed", "layer_0", ..., "norm", "lm_head"]` label set the shared host-reference
branch builds, fed to the file's existing `_compare_intermediate_pcc` unchanged. Threshold 0.99,
unmodified. Random weights, `e_score_correction_bias` zeroed on every layer.

| case | result | call time |
|---|---|---|
| `2_layers ... smoke-random` (no host reference) | **PASS** | 91.87s |
| `2_layers ... pcc-random` | **FAIL** | 167.12s |

Per-stage PCC for the failing case (threshold 0.99):

| stage | PCC | Δ from previous |
|---|---|---|
| `embed` | **1.000000** | — |
| `layer_0` | 0.975813 | −0.024187 |
| `layer_1` | 0.942922 | −0.032891 |
| `norm` | 0.942987 | +0.000065 |
| `lm_head` | 0.928376 | −0.014611 |

**`embed` at exactly 1.000000 is the control that makes this diagnostic rather than suspicious**: the
embedding, the SP/TP sharding, the snapshot labelling and the comparator are all exact. Every bit of
error is introduced inside the decoder layers, and `norm` tracking `layer_1` to 5 decimal places
confirms the final norm adds nothing — it is purely inherited.

**The error compounds with depth**, and the marginal loss grows (−0.0242 then −0.0329). This is the
same softmax-vs-sigmoid routing gap as F0, but observed over depth instead of in one layer:

| depth | PCC |
|---|---|
| 1 layer (block test, `torch.randn` input) | 0.995178 |
| 1 layer (transformer, real embedding input) | 0.975813 |
| 2 layers | 0.942922 |

This is the answer to the question the block test could not settle. A single layer's 0.995 looked
benign; at 2 layers it already misses a 0.99 gate, and the trend is worse-than-linear. This model has
**36** layers.

**Not fixed, and deliberately so.** The fix is a `softmax` `score_func` in the device op
(`moe_grouped_topk.cpp` currently `TT_THROW`s on anything but `sigmoid`/`sqrtsoftplus`) plus a
matching host path and `SCORE_FUNC = "softmax"` on `Mistral4Small119BConfig`. That is a C++ kernel
change to an op shared by DeepSeek / Kimi / GLM — a major fix, not a test fix, so per instruction it
is reported rather than attempted. The threshold was not lowered, the reference was not switched to
sigmoid, and nothing was xfailed: this failure is the most valuable output of the whole exercise.

### 5-layer run — the accumulation confirmed  -> log: prefill_transformer_mistral4_5L.log

`5_layers ... pcc-random`: **FAIL**, 256.75s.

| stage | PCC | Δ per layer |
|---|---|---|
| `embed` | **1.000000** | — |
| `layer_0` | 0.975813 | −0.024187 |
| `layer_1` | 0.942922 | −0.032891 |
| `layer_2` | 0.906779 | −0.036143 |
| `layer_3` | 0.870295 | −0.036484 |
| `layer_4` | 0.834688 | −0.035607 |
| `norm` | 0.834414 | −0.000274 |
| `lm_head` | 0.769880 | −0.064534 |

Three things make this conclusive rather than suggestive:

1. **`layer_0` and `layer_1` are bit-identical to the 2-layer run** (0.975813 and 0.942922 to six
   decimals). The degradation is deterministic and reproducible, not run-to-run noise.
2. **`embed` is exactly 1.000000** — the harness, sharding and comparator are exact, so every bit of
   loss is introduced inside the decoder layers.
3. **The marginal loss converges to a constant ≈ −0.0355 PCC per layer** (−0.0361, −0.0365, −0.0356
   for layers 2, 3, 4). The first two layers are the transient; after that it is a steady linear
   bleed.

At a steady ≈0.0355 PCC per layer over a **36-layer** model, the output of the full stack is nowhere
near usable — the trend leaves the meaningful-correlation range within roughly the first third of the
network. (Stated as a trend, not an extrapolated number: PCC does not decay linearly all the way
down.) The `lm_head` drop is steeper than the per-layer rate because the logits projection amplifies
the accumulated hidden-state error across the 131072-wide vocabulary.

**This is the single most important result in this exercise.** The softmax-vs-sigmoid router
substitution is not a rounding-level nuisance that a threshold can absorb — it is a per-layer bias
that accumulates linearly with depth. The one-layer block test's 0.995 was misleading precisely
because one layer is where the effect is smallest.

---

# Re-run on `ssalice/mistral4-tests` (clean rebuild of `kmabee/prefill-shared-fixes`)

Logs: `mistral4_bringup/test_logs/on_mistral4_tests/`. Runner: `run_wt.sh`.

Built from scratch in a worktree (`build_metal.sh --clean` then `--enable-ccache`; ~4 min actual
compile, 0 errors, `ENABLE_TRACY=ON` preserved). This re-run was **required**, not belt-and-braces:
the branch needs a newer `_ttnn.so` (`ttnn.RoutedExpertActivation.SituGlu`, absent from the old
build) and the cherry-pick moved the block/transformer tests onto `torus_xy_device_params`, which
also sets `reliability_mode = RELAXED_INIT`.

**Trap worth recording:** with a *symlinked* `python_env`, `PYTHONPATH=$PWD` alone silently loads the
**main repo's** older `_ttnn.so` — the importable package is `<root>/ttnn/ttnn`, so `<root>` yields
only a namespace candidate and `PathFinder` continues to the shared env's `ttnn-custom.pth`. The
three-entry form `$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools` is mandatory, and
`run_wt.sh` asserts `'sf-trial' in ttnn._ttnn.__file__` before running anything.

## Result: 11 passed, 1 failed — same pass/fail pattern as the original branch

| test | old branch | new branch | verdict |
|---|---|---|---|
| `test_mistral4_kv_cache_table` | 1 PASS | **1 PASS** | same |
| `test_mistral4_mla -k 8x4` | 4 PASS | **4 PASS** | same |
| `test_mistral4_moe` | 1 PASS | **1 PASS** | same |
| `test_mistral4_prefill_block` | 2 PASS | **2 PASS** | same |
| `test_mla_chunked_prefill` (3 sub-8192) | 3 PASS | **3 PASS** | same |
| `test_mistral4_prefill_transformer` 2L | 1 PASS / 1 FAIL | **1 PASS / 1 FAIL** | same |

### Numbers side by side

| metric | old branch | new branch | Δ |
|---|---|---|---|
| MLA out PCC (4 cases) | 0.9981613, 0.9993314, 0.9980571, 0.9986129 | **bit-identical** | 0 |
| block dense | 0.9998777 | 0.9998774 | −3e−7 |
| block moe | 0.9951784 | 0.9951924 | +1.4e−5 |
| MoE `reference_output` | 0.972469 | 0.972497 | +2.8e−5 (still +0.0015 over threshold) |
| MoE `routed_output` | 0.974983 | 0.975551 | +5.7e−4 |
| chunked full-measured (3) | 0.9992858, 0.9992854, 0.9993017 | 0.9992884, 0.9992878, 0.9993045 | ~+2.6e−6 |
| transformer `layer_0` | 0.975813 | 0.976044 | +2.3e−4 |
| transformer `layer_1` | 0.942922 | 0.943361 | +4.4e−4 |
| transformer `lm_head` | 0.928376 | 0.942780 | **+0.0144** |

Everything except `lm_head` reproduces to 4+ decimals, so `RELAXED_INIT` and the 324 upstream commits
are numerically neutral for this model. `lm_head` improved by 0.014 — an upstream change in the
lm-head path, not something this work did; the stage still fails its 0.99 gate and the per-layer
accumulation is unchanged.

**The MoE accumulation reproduces exactly**: `embed` 1.000000, then −0.024 and −0.033 for the first
two layers. So the conclusion in F0 holds on the deliverable branch, on a fresh build, under
`RELAXED_INIT`.

### Op unit tests on the new branch

| test | selector | result | log |
|---|---|---|---|
| `op_unit_tests/test_ttnn_dispatch_combine.py` | `-k "mistral4 and 8x4"` | **4 passed** | `op_dispatch_combine.log` |
| `op_unit_tests/test_prefill_dispatch.py` | `-k "mistral4-perf_no_pcc and 8x4"` | **4 passed**, 12 skipped | `op_dispatch.log` |

(Case counts differ from the old branch because upstream added `fp8_scaled_in` / `fp8_out` axes, so
the same selector slices differently. No failures.)

## Final tally on `ssalice/mistral4-tests`

**19 mistral4 cases passed, 1 failed** — the failure being
`test_mistral4_prefill_transformer[... pcc-random ...]`, which is the softmax-router accumulation
(F0) and is the intended signal, not a defect in the wiring.
