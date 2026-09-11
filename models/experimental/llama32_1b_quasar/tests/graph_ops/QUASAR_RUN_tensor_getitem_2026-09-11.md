# Quasar run report — `ttnn.Tensor.__getitem__` (llama32_1b_quasar graph op) — 2026-09-11

Uncommitted review artifact; delete before merge. Branch `vsureshTT/quasar_uplift_round_2` @ `23268d08f1b`, host lib
`build_Release/lib/_ttnn.so` built 2026-09-11 18:08 (unchanged during the craq-sim runs; the RTL case 00 row notes a later rebuild by another agent).
Test: `models/experimental/llama32_1b_quasar/tests/graph_ops/test_tensor_getitem.py` (3 captured cases, 108 model calls).

## Status: **GREEN on both targets** — 3/3 craq-sim PASS and 3/3 ZEBU RTL PASS, all PCC 1.0. **No source change was needed**; no
op in the dispatch chain hit a Quasar symptom, so nothing was ported or `ARCH_QUASAR`-guarded (recipe §2: already-M2 → no-op).

## 1. Dispatch chain (per case)

`ttnn.Tensor.__getitem__` is a Python composite (`ttnn/ttnn/operations/core.py:41`). For every captured case the key is a
tuple of 4 `slice` objects (no integer indices, no Ellipsis, no step), so:

| step | what runs | device op? | Quasar state |
|---|---|---|---|
| 1 | `core.__getitem__`: normalize slices → `slice_start/end/step` (`step=[1,1,1,1]`) | no | — |
| 2 | `ttnn.slice(t, start, end, step)` (`slice.cpp:102`, device tensor path) | — | — |
| 2a | no-op check `no_step && starts_zero && ends_max` → **false** for all 3 cases (an end is < shape) | | |
| 2b | `rm_only = layout!=TILE \|\| !no_step \|\| 1-D \|\| begins not tile-aligned` → **false** (TILE, begins all 0) → **no `to_layout(ROW_MAJOR)`** | | |
| 2c | `ttnn::prim::slice(...)` → `SliceDeviceOperation::select_program_factory`: `use_tensor_args=false`, TILE, no step → **`SliceTileProgramFactory`** | **yes — the only device op** | Metal 2.0, Quasar-GREEN (`slice/QUASAR_UPLIFT_REPORT.md`) |
| 2d | `ttnn::experimental::view(res, actual_shape, final_padded_shape)` | no (host-side metadata view) | — |
| 2e | `fill_implicit_tile_padding` — gated on `pad_value.has_value()`; not passed by `__getitem__` → **skipped** | no | — |
| 2f | `ret_adjustment`: `to_memory_config(same DRAM interleaved mc)` + `to_layout(TILE→TILE)` → **both no-ops** | no | — |
| 3 | squeeze loop: `singled_out_dims` empty → **no `ttnn.squeeze`** | no | — |

Per-case detail (all: input TILE, DRAM interleaved, single device):

| case | input → output | slice | prim::slice args (padded ends) | notes |
|---|---|---|---|---|
| 00 `8x1024x64_bf8` | `[1,8,1024,64]` → `[1,8,512,64]` | dim 2 `[:512]` | begins `[0,0,0,0]`, ends `[1,8,512,64]` | tile-aligned; **BFLOAT8_B as captured** (see dtype note) |
| 01 `32x1x64_bf16` | `[1,32,1,64]` → `[1,1,1,64]` | dim 1 `[:1]` | begins `[0,0,0,0]`, ends `[1,1,32,64]` (dim −2 padded 1→32) | output logical `[1,1,1,64]`, padded `[1,1,32,64]`; `dim_needs_fill(-2)` false because input dim is also 1 |
| 02 `8192x64_bf16` | `[1,1,8192,64]` → `[1,1,1024,64]` | dim 2 `[0:1024]` | begins `[0,0,0,0]`, ends `[1,1,1024,64]` | tile-aligned |

Run-time evidence that nothing else was dispatched: the per-run JIT cache
(`/localdev/vsuresh/tt-metal-cache-qsr-getitem/*/kernels/`) contains exactly
`reader_unary_unpad_dims_interleaved_start_id` and `writer_unary_interleaved_start_id` (the two `SliceTile` kernels,
`slice_program_factory_tile.cpp:175/205`) after all three cases; the watcher dump lists only those two kernels on the
DM cores. No `reshape`, `squeeze`, `to_layout`, `tilize`/`untilize`, or `copy_via_memmove` path is reached, so the known
misaligned-width reshape bug (tt-metal#56283) is **not on this op's path**.

## 2. What was changed to run it (test harness only — no repo source edits)

Temp copy `test_tmp_tensor_getitem_qsr.py` next to the original (deleted after the run; re-create from the recipe below):
1. **Case 00 `BFLOAT8_B → BFLOAT16`** (args + outs). Quasar rejects `Bfp8_b` DFBs in `ValidateProgramSpec`; this is the
   model-level dtype decision already recorded for other ops. Cases 01/02 are bf16 as captured.
2. **Host tilize**: `G.build_tensor` monkeypatched to `from_torch(layout=TILE)` on host + `ttnn.to_device(...)`. The
   generated harness's `from_torch(device=, layout=TILE)` runs an on-device `ttnn::to_layout` →
   `ttnn::tilize_with_val_padding`, which is fatal on Quasar (today: `ComputeGen1Config` on a Gen2 KernelSpec, see §6;
   on 09-10 it was still the legacy `DataMovementKernel is not supported on Quasar`). Helper-op blocker, not
   getitem/slice — same workaround as every earlier `test_tmp_*_qsr.py`.
3. `U.assert_pcc` wrapped to print the PCC on success (`QSR_PCC ...` lines), purely for this report.

Op-directory source edits: **none** (`git status` on `data_movement/slice` and `experimental/reshape` clean before and after).

## 3. craq-sim (Quasar functional simulator, 8x4 grid, 2 DRAM banks, slow dispatch, forced JIT)

| case | result | PCC (floor 0.999) | time | notes |
|---|---|---|---|---|
| 00 `8x1024x64_bf8` (run as bf16) | **PASS** | 1.0 | 1.05 s call | 8 JIT builds, 0 cache hits |
| 01 `32x1x64_bf16` | **PASS** | 1.0 | ~1 s call | ttnn Python op logging on: only `from_torch`, `to_device`, `Tensor.__getitem__`, `to_torch` |
| 02 `8192x64_bf16` | **PASS** | 1.0 | ~1 s call | |
| all 3, **LLK asserts + lightweight asserts + watcher (NoC sanitizer off)** | **PASS 3/3** | 1.0 / 1.0 / 1.0 | 12.4 s total | `watcher.log`: 0 faults/asserts; only the two slice kernels listed |

Known sim-only false failures checked: no `qsr_tile_counter_check_error` (craq-sim#355) and no all-zero blocks — neither
signature appears in any log (`sim_case00/01/02.log`, `sim_all_llkasserts_watcher.log`, `watcher_sim_all.log` in the
scratch dir `/tmp/claude-1211407858/-localdev-vsuresh-tt-metal/47c32300-1ca5-43ee-a917-81ae38742e1d/scratchpad/qsr_getitem/`).

## 4. RTL — ZEBU emulator `emu-quasar-1x3` (1 worker core)

Captured shapes run **unshrunk**: `SliceTile` splits tiles over whatever grid exists, so all three fit on one core.

| case | result | PCC (floor 0.999) | wall (incl. ~1 min boot) | notes |
|---|---|---|---|---|
| 01 `32x1x64_bf16` | **PASS** | 1.0 | 60 s | launcher log `emu_2026-09-11_18-59_.log` |
| 02 `8192x64_bf16` | **PASS** | 1.0 | 65 s | launcher log `emu_2026-09-11_19-06_.log` |
| 00 `8x1024x64_bf8` (run as bf16) | **PASS** | 1.0 | 106 s | launcher log `emu_2026-09-11_19-55_.log`; job queued ~45 min on the shared emu lock behind another agent's job; host lib `_ttnn.so` mtime at run time 19:55 (an exclusive `qsr_rebuild` by another agent landed while queued — `slice` sources were untouched, `git status` clean) |

No launch failures (`[ZTDB0349F]` / `test-start.log`) occurred; no recovery was needed.

## 5. Fixes applied

None. Nothing in the reached path (`SliceTileProgramFactory` + its 2 DM kernels, host-side `view`) fired a Quasar symptom.

## 6. Blockers / owner items

| item | severity | owner | notes |
|---|---|---|---|
| `Bfp8_b` rejected by `ValidateProgramSpec` on Quasar — case 00 as captured cannot run | model-level (not an op bug) | llama32_1b_quasar model owners | 64 of the 108 captured `__getitem__` calls are bf8 (the K/V `[1,8,1024,64]` slices). Decide bf16 for the Quasar variant, or a Quasar bf8 DFB format. |
| `from_torch(device=, layout=TILE)` → `ttnn::to_layout` → `ttnn::tilize_with_val_padding` fatal on Quasar: `TT_FATAL program_spec.cpp:907: KernelSpec 'compute' targets Gen2 (Quasar) but its ComputeHardwareConfig holds a ComputeGen1Config` | harness helper, pre-existing — **RED for its owner** | `data_movement/tilize_with_val_padding` owners | The op is Metal 2.0 now (the slice report of 09-10 still saw the legacy `DataMovementKernel` error), but all 3 factories hardcode `ComputeGen1Config compute_gen1{.enable_32_bit_dest = fp32_llk_acc}` (`factories/tilize_with_val_padding_{single_core,multi_core_default,multi_core_sharded}_program_factory.cpp:211/168/190`); needs a Gen2 `hw_config` variant per `gen2_hardware_configs.md`. Confirmed by running the checked-in test (`sim_as_checked_in_case01.log`). Worked around in the temp copy by host tilize; `graph_case.build_tensor` unchanged. |

No RED for `__getitem__` or for `slice`.

## 7. Reproduce

```bash
# temp copy: cp test_tensor_getitem.py test_tmp_tensor_getitem_qsr.py; in it: "BFLOAT8_B" -> "BFLOAT16" (case 00),
# rename the test fn, and after `_OP = ...` add the G.build_tensor host-tilize monkeypatch (identical to test_tmp_untilize_qsr.py).

# craq-sim (one case per job, foreground)
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env.sh getitem
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_tensor_getitem_qsr.py -k 00_8x1024x64 -v -s
#   ... -k 01_32x1x64 / -k 02_8192x64
# asserts + watcher pass:
TT_METAL_WATCHER=5 TT_METAL_WATCHER_DUMP_ALL=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 TT_METAL_LLK_ASSERTS=1 TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS=1 \
  qsr_test timeout 2400 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_tensor_getitem_qsr.py -k test_tmp_tensor_getitem_qsr -v -s

# ZEBU emulator (one case per job, foreground, exclusive emu lock)
cd /localdev/vsuresh/tt-metal && source /localdev/vsuresh/qsr-sim/env_emu.sh 1x3
qsr_emu timeout 2700 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tmp_tensor_getitem_qsr.py -k 01_32x1x64 -v -s
#   ... -k 02_8192x64 / -k 00_8x1024x64 ; then mv emu_*_*.log <scratch>

# as checked in (fails before slice runs, in from_torch's on-device to_layout/tilize_with_val_padding — ComputeGen1Config fatal):
qsr_test timeout 1800 ./python_env/bin/python -m pytest models/experimental/llama32_1b_quasar/tests/graph_ops/test_tensor_getitem.py -k 01_32x1x64 -v -s
```
