# Precise gate unpack-mode width repair — pending hardware

The prior `precise_gate_repair_api_retry_jobs.json` failed JIT before a numerical comparison because its nonempty unpack override vectors contained 32 entries. This retry changes only the host vector width to 64; it does not change the kernels, selected FP32 operands, arithmetic, memory layouts, shapes, gates, or probe expectations. No devices were opened and no jobs were launched during preparation.

## Source diagnosis

- `tt_metal/jit_build/hlk_desc.hpp:43` initializes `buf_dataformat_arr(max_dfbs, Invalid)`: allocation spans the architecture's complete CB namespace, not merely the CB indices used by a particular kernel.
- `tt_metal/jit_build/genfiles.cpp:955`–`963` passes that vector and the explicit unpack override vector into format generation. `tt_metal/jit_build/data_format.cpp:184`–`192` requires a nonempty override vector to have at least as many entries as the format vector. The documented `>=` deliberately permits a host maximum allocation on a smaller architecture.
- `tt_metal/api/tt-metalium/circular_buffer_constants.h:31` defines the host maximum as 64; `tt_metal/hw/inc/api/circular_buffer_constants.h:23`–`26` selects 32 for Wormhole and 64 otherwise. The observed 64-entry JIT requirement is consistent with this source. It does not imply adding or allocating 64 CBs in each program.
- Preserve writable-field assignment `compute_config.unpack_to_dest_mode = unpack_modes`; this binding does not accept the field as a constructor keyword. Packed helper retains only CB23 in `UnpackToDestFp32`; conv helpers retain only CB10,11,12,13. All new entries 32–63 are `Default`.

## Immutable new modules

Under `tests/`:

- `multichip_flat_ordered_packed_precise_wide_modes.py`, `_candidate.py`, `_probe.py`.
- `multichip_conv_gates_precise_wide_modes.py`, `_candidate.py`, `_probe.py`.
- `multichip_conv_gates_matched_wide_modes_probe.py`: exact matched-control probe with the new helper import. It preserves production-convolution bitwise comparison and same-gates bitwise state check, as well as the .995 global/per-head gates.
- `multichip_fast_zero_conv_gates_wide_modes.py`, `_candidate.py`, `_probe.py`.
- `multichip_fast_zero_conv_gates_matched_wide_modes_probe.py`: matched production control plus wide-mode conv and existing fast-zero recurrent helper.

The fast-zero recurrent helper has no custom unpack-mode vector and is reused unchanged. All helper `KERNEL_DIR` paths still reference their original kernel directories. No C++ source clone is needed. The two historical precise candidates retain their original inheritance; the fast-zero cumulative candidate retains its current-production inheritance. This repair does not silently combine families or change controls.

## Parent-owned serialized commands

The eight-entry retry list preserves every nonrouting field of `precise_gate_repair_api_retry_jobs.json`: two direct Watcher probes, B6/B32 16-step state+benchmark cases, and B32 4095-prefix/1024-step answer cases. Labels and output files end in `_wide_modes`, with no previously recorded command overwritten.

```bash
PYTHONPATH=. python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_stage5_jobs models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/precise_gate_repair_wide_modes_jobs.json
```

For separately queued matched-control checks, use these modules with `--output <fresh-path>` and the usual Watcher environment (`TT_METAL_WATCHER=10`, `TT_METAL_WATCHER_NOINLINE=1`, `TT_METAL_WATCHER_APPEND=1`), through the serialized runner:

```text
models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_conv_gates_matched_wide_modes_probe
models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fast_zero_conv_gates_matched_wide_modes_probe
```

Fast-zero cumulative runs use `models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_fast_zero_conv_gates_wide_modes_candidate` with the existing cumulative harness arguments. They are intentionally separate from the eight precise repair jobs.

## Host validation and limits

`host_verify.py` reproduces AST, exact clone/source hashes, actual bound descriptor construction, all CLI help checks, and eight-job/reference validation. Device-opening functions fail closed during checks. `host_validation.json` records results; `source_manifest.json` identifies preserved originals and kernels. The only helper text difference is `* 32` to `* 64` in `unpack_modes`; candidate/probe changes are import routing only.

Host descriptor acceptance is not a device JIT, Watcher, numerical, or latency result. Both prior precision hypotheses remain pending the new hardware runs. If JIT passes but a numerical gate fails, that is a distinct arithmetic/ordering question; these files do not relax a quality gate to mask it.
