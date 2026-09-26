# Known issues

Recurring traps, each of which has cost real time. A symptom that matches one of
these almost always *is* it — check here before diagnosing from scratch.

## Symptom index

| Symptom | Cause | Fix |
|---|---|---|
| Device hangs hard inside `ttnn.group_norm`; needs `tt-smi -glx_reset` | Non-uniform multicast groups from a hand-rolled core grid | Grid from `ttnn.determine_expected_group_norm_dram_grid_size(...)` — see below |
| Allocation failure with a byte count far above L1 (e.g. 5467008 B vs 1572864 B) | `num_out_blocks=-1` under-chunks at large spatial extents | Tune `num_out_blocks` per shape; more blocks = smaller per-iteration CBs |
| conv1d/2d/3d fails allocation with a fresh-looking L1 error | Missing `l1_small_size` in the device fixture | `{"l1_small_size": 32768}` (common) or `65536`; add `trace_region_size` if tracing |
| Conv3d blocking-fallback warnings after a dtype change; the gain vanishes | `utils/conv3d.py` keeps separate fp32 and default tables | Re-sweep blockings after **any** compute-dtype change; grep the log to confirm the tuned entry hits |
| A sweep hangs deterministically at certain SDPA chunk sizes | Configuration-specific, not sweep-specific — reproduces one-config-per-process | Sweep at the attention's **real** shape, one config per process, hard timeout; record unreached values |
| Output is plausible but wrong after enabling trace, no error | Buffer reuse — see "Trace" below | Re-gate quality e2e against the untraced baseline |
| `AssertionError: Device data missing: Op <id>`, or only a few ops captured | Tracy's ~1000-op-per-device buffer overflowed | Shrink the profiled scope — `../tt-dit-benchmark-profile/tracy-capture.md` |
| All device durations read zero | `TT_METAL_DEVICE_PROFILER`, `TT_METAL_WATCHER`, `TT_METAL_DPRINT_CORES` conflict — all use device SRAM | Set only one; also unset `TTNN_CONFIG_PATH` |
| Host times inflated, numbers unstable run to run | First iteration populates the program cache | Iterate ≥ 2×, measure the second |
| Profile says data movement dominates; a run of `Tilize`/`Untilize` sits at the head of the capture | Weight upload + activation prep inside the measured window | Signpost the model call only, or slice with `--id-range` — `../tt-dit-benchmark-profile/reading-profiles.md` |
| A change looks like a regression, later turns out to be a win | Whole-model wall clock has ~3× variance | Measure the **op**, under the profiler |
| PCC passes but the output has seams or flicker | A parallelism bug that whole-tensor PCC hides | `../tt-dit-add-model/testing-and-accuracy.md` § Artifact rubric |

## GroupNorm

The largest single source of hangs in this tree. Four separate traps.

| Trap | Detail |
|---|---|
| **Core grid** | A grid satisfying the divisibility rules (`Ht % nvr == 0`) can still produce non-uniform multicast groups and deadlock. `vae_mochi.ResBlock._valid_norm_grid` searches by those rules and has picked a hanging grid where the pinned API picks a working one. Always use `ttnn.determine_expected_group_norm_dram_grid_size(...)` |
| **Circular buffers** | `num_out_blocks=-1` under-chunks at large spatial extents. Keep a small per-shape table; `-1` is fine for small ones |
| **Precision floor** | The kernel is bf16-only, so in an otherwise-fp32 model every norm is a bf16 island — ~PCC 0.9999 and ~3e-2 relative max error per norm, compounding. **Not a bug**: measure the floor, record it, set bars above it |
| **Layout round-trip** | `Untilize → GroupNorm → Tilize` has measured over half of warm encoder device time on a video VAE. `../tt-dit-performance/optimization-levers.md` § 3 |

**Per-frame statistics on 5D tensors.** Use
`layers/normalization.py::GroupNorm3D` with `T` as the batch axis — reshape
`(1,T,H,W,C) → (T,1,H,W,C)`. Its `dims=3` pooling over `(C_group, T'=1, H, W)`
degenerates to exactly the per-frame `(C_group, H, W)` statistics. Input prep is
`tilize_with_zero_padding(reshape(x, (B,1,T*H*W,C)), use_multicore=True)` —
**not** `to_layout(TILE)` on the 5D tensor. The plain 2D `GroupNorm` gets none of
this right; `GroupNorm3D` gets all of it.

## Trace: silently wrong output, not a crash

Trace is the one optimization whose failure mode is **bad output with no error**.
Everything else either works or raises. Budget correctness attention accordingly.

| Hazard (documented on `utils/tracing.py::Tracer`) | Consequence |
|---|---|
| Tensors allocated **after** capture may be overwritten during replay | Corruption lands in an unrelated tensor, so it reads as a model bug |
| The same output tensor objects are returned every call, overwritten in place | Holding a reference across calls gives aliased data, not two results |
| `trace_region_size` is DRAM taken from weights and activations | Oversizing pushes an allocation failure elsewhere. In-tree: `90112` (one small region) to `500_000_000` (full pipeline) |

**Re-gate quality end to end after enabling trace.** A component PCC check runs
the untraced path and cannot catch a replay reading a clobbered buffer. Gate each
region behind its own env flag (`LTX_TRACED`, `LTX_VOC_TRACE`, `LTX_VAE_TRACE`)
so you can bisect which region broke.

## Python-side traps

| Trap | Fix |
|---|---|
| Comparing an fp32 tensor to a Python float literal (`0.7`) that is not fp32-representable | Compare fp32 tensors to fp32 tensors |
| `pytest.raises` rejected by the `prefer-expect-error` pre-commit hook | Use the `expect_error` fixture from `conftest.py` |
| `pre-commit` aborts with "not found" | Put `python_env/bin` on `PATH` before `git commit` |
| autoflake strips an import the module no longer uses but a test reads through it | Import it from its real home |
| tt_dit's plain `RMSNorm` defaults to `bias=True` | Pass `bias=False` for weight-only norms; `DistributedRMSNorm` asserts `not bias` |
| `Module` is an ABC with `forward` abstract | A parameter-owning container still needs a `forward` to instantiate |
