# Source-manifest patch plan — applied at source freeze

The parent authorized these manifest-only changes at the 2026-09-15 12:07
numerical-source freeze. They are now applied locally; no files were uploaded
by the manifest agent. The parent owns coordinated source sync and subsequent
current-source qualification. Paths
below are repository-relative. This is a bounded manifest of selected project
headers, numerical dispatch interfaces and experiment dependencies, **not** a
complete compiler/firmware/SFPI/generated-header closure. Historical records
retain their original pins; do not backfill them and claim historical coverage.

## Applied changes and actual checks

Seven producer files changed, only inside `source_files` / `source_hashes`:
native qualification, three original HiFi2/native-storage drivers,
V-transposed, and the two centered-V drivers. Combined and mean-error drivers
inherit repaired lists without source edits. The later HiFi2 BF16/LUT control
also inherits the repaired qualification manifest.

AST comparisons against captured pre-edit source passed for all seven edited
files after excluding only their manifest helper; the complete remaining AST
was identical. Combined and mean-error files were byte-for-byte unchanged.
All seven edited Python files passed `py_compile`. Standard-library-only
evaluation checked all required A/B/C/reference groups and path existence in
14 helper configurations, with these deduplicated counts:

| Helper | Pinned files |
|---|---:|
| Native qualification | 67 |
| Original HiFi2 native / native storage | 68 each |
| HiFi2 macro-LUT / later HiFi2 BF16 macro-LUT | 71 each |
| V-transposed, MAIN or FAST | 59 each |
| Centered V4, MAIN or FAST | 62 each |
| Centered V8, MAIN or FAST | 66 each |
| Combined FAST recipe | 79 |
| Mean-error, either supported KV format | 68 each |

The reciprocal8 driver already covered group A and native-exp dependencies,
with no mean-only dependencies, so needed no manifest repair. Its original
helper was separately checked for both destinations. During this audit the
parent authorized a separate final-scaling experiment owned by the Sage
agent; reciprocal8 numerical/dispatch changes are deliberately excluded from
these seven-file manifest-only assertions. The manifest agent did not edit
that driver or its kernels. No device jobs or C++/JIT builds were run here.

## A. Shared attention headers

Add these explicit paths, deduplicated with existing entries:

| Path | Actual inclusion / reason |
|---|---|
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_streaming_qktv.hpp` | All selected frozen/private streaming headers include it; PV geometry helpers. |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/q_chunk_remapping.hpp` | Selected frozen `compute_common.hpp` includes it; query assignment helpers. |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chunked_prefill_utils.hpp` | Selected common/streaming headers include it; chunk bounds/indexing. |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_geometry.hpp` | Direct streaming include; geometry/template definitions are compiled even for non-windowed cases. |
| `ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp` | Direct streaming include; same qualification as preceding row. |
| `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` | Selected frozen common includes it; destination helper definitions. |
| `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp` | Both fullchip and V-transposed chain readers directly include it. Already pinned by native qualification. |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | Both chain readers and ordinary fullchip reader directly include it; column identity generation. |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl` | Included at line 89 of the preceding header; implementation, not an unrelated mean-only dependency. |
| `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h` | Direct include in both compute wrappers. |
| `tt_metal/hw/inc/api/compute/experimental/matmul_custom.h` | Direct selected streaming include; custom matmul init/execute dispatch. Already in V-transposed and inherited combined manifests. |
| `tt_metal/hw/inc/api/compute/experimental/sdpa_sub_custom.h` | Direct selected streaming include; score-minus-row-max dispatch. |

This intentionally stops at these experiment-facing API boundaries, except
architecture files already explicitly pinned. It does not imply all other
Metal API/LLK dependencies are frozen. Existing version/checkout information
must continue to accompany results.

## B. Native exponential dispatch

`native_exp_qualification.source_hashes()` should also explicitly include:

```
ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/chain_link.hpp
tt_metal/hw/inc/api/compute/eltwise_unary/exp.h
tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_exp.h
```

The first is selected by the chain reader; the other two define native exp
initialization/replay used by the selected frozen common and native wrapper.
The three HiFi2/storage drivers already add chain_link and the LLK exp header,
but native qualification itself does not. V-transposed already pins all three.
Centered-V drivers select BF16 native exp through their frozen common and
should add the two exp paths as well (chain_link is already present).

## C. Mean-only dependencies: do not add to V-transposed/combined for attention

The center/mean-error modes genuinely call the generic H-axis device mean.
`native_exp_qualification` also supports this through its centered-K cases.
For those manifest unions add these paths where missing:

```
ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp
ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl
ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp
ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce.cpp
ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp
ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp
```

Selection evidence: `reduce_op_multi_core_h_program_factory.cpp` lines
449–452, 516–519, 592–597 select these three kernels for tiled interleaved
non-negating H reductions. `reduce.cpp:14` includes the compute helper header;
its implementation includes `reduce_helpers_common.hpp`. This applies to
both BF16-FPU and optional FP32-SFPU means in the supported layout, not RM,
sharded, negative-min, W-only or HW-only kernels.

For native qualification, also add the principal mean dispatch files currently
missing from its union (centered-V manifests already include them):

```
ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions.cpp
ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_h_program_factory.cpp
```

Both centered-V manifests already pin `reduce_helpers_compute.inl` and
`reduce_helpers_dataflow.inl`; retain those and add the missing headers/kernels.
Mean-error inherits this coverage through its BASE4/BASE8 manifest union.
V-transposed and combined do not call generic mean; their missing compute
reduction helpers are **not** an omission for those tested paths. Combined's
Hadamard matrix multiplication is not a generic mean reduction.

## Concrete edits at freeze

1. In `native_exp_qualification.source_hashes()`, add groups A/B and group C
   for its supported centered-K cases, using explicit paths. Keep its existing
   reference, codec, frozen common/SFPU and current kernel pins.
2. In `hifi2_native_fullchip.py`, `hifi2_lut_fullchip.py` and
   `native_storage_fullchip.py`, **remove** the
   `operations/transformer/sdpa.rglob("reduce_helpers*.inl")` expression.
   It currently matches zero files. Inherit explicit group A from qualification;
   do not replace it with another broad glob. Their no-centering CLIs do not
   independently require group C, though the inherited qualification manifest
   remains a documented superset covering the parent's suite.
3. Preserve the LUT driver's already-correct explicit additions:
   `experiments/sdpa-l2/bfp4-lofi-v2/exp_lut.hpp`, `exp_lut_macro.hpp`, and
   `exp_lut_macro_streaming/compute.cpp`. It does not execute that folder's
   reader/writer wrappers: it uses `fullchip/reader*.cpp` and `fullchip/writer.cpp`,
   which are already covered. No unused wrapper pins are necessary.
4. Add missing group A entries in `Vtransposed_fullchip.source_files()`.
   Its native/grid7/transpose kernels, PV-transpose header, frozen selected
   SFPU, low-level transpose/matmul interfaces, reference and quantizer helpers
   are already pinned. `combined_recipe_fullchip.source_files()` inherits this
   fix; its actual Hadamard and adaptive primitive lists are already included.
   Do not add mean kernels to this inheritance branch.
5. Add missing group A/B/C entries in both centered-V manifest helpers.
   Standalone `value_centered_fullchip.py` additionally needs these imported
   reference/oracle files, already present in BASE8 and thus in mean-error:

   ```
   tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py
   experiments/sdpa-l2/bfp4-lofi-v1/probe.py
   experiments/sdpa-l2/bfp4-lofi-v1/numerics.py
   experiments/sdpa-l2/frontier-accuracy-v1/run.py
   ```

6. No extra direct `value_mean_error_fullchip.source_files()` change is needed
   after its BASE4/BASE8 unions are repaired. It already includes its own driver,
   both imported wrappers, original reference/oracles, effective-V truncation
   primitive and the existing typecast/add/subtract/slice dispatch files.
   Do not claim this principal-op list exhausts all dynamically selected TTNN
   binary/typecast implementation dependencies.

Before a freeze commit, evaluate all helpers without device use, assert every
listed path exists and every selected addition is present, and keep a snapshot
of the manifest itself. Rerun at least native HiFi2, LUT HiFi2, LoFi storage,
V-axis/combined, and mean-error smoke with unchanged numerical settings.
Manifest additions alone cannot validate historical unpinned source state.

## Clock interpretation

`hardware-active-20260915-112401.json` reports AICLK `1306` MHz (raw `0x51a`),
configured max `1350` MHz (`0x546`). This is one active telemetry sample, not
a clock trace covering every benchmark. Report measured wall-time TFLOP/s
unchanged. Do not label all runs as sustained 1350 MHz, and do not globally
rescale their measured throughput. If quoting utilization at this observed
clock, explicitly scale the corresponding architectural peak by
`1306/1350 = 0.9674074`; utilization versus that peak is `1350/1306 = 1.03369`
times utilization versus a 1350 MHz peak. This is conditional clock accounting,
not evidence that a particular earlier run operated at 1306 MHz.
