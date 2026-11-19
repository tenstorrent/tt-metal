# Precise-gate API adaptation

The original eight precise-gate jobs stopped at `ComputeConfigDescriptor(..., unpack_to_dest_mode=...)`. This was a host binding error before the candidate kernel ran, so it is not numerical or performance evidence against fusion.

This checkout's nanobind constructor accepts only fidelity/approximation/DST/packing flags (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:641–658`), while `unpack_to_dest_mode` is a separately writable `.def_rw` field (`:684–687`). Both helpers now construct the config with supported arguments, assign `compute_config.unpack_to_dest_mode = unpack_modes`, and pass that object to `KernelDescriptor`. `tt_metal/impl/program/program.cpp:585` forwards the descriptor field to the real compute config.

No kernel arithmetic, CB layout, quality gate, or selected unpack mode changed. Packed CB23 still uses UnpackToDestFp32; conv-gates CB10/11/12/13 still use it. All other CB modes remain Default. HiFi4, FP32 destination, and exact math are unchanged.

Before-edit helper snapshots, SHA256 before/after values, and minimal source diffs are preserved here. All original logs/results and the original job JSON remain untouched. New `doc/optimized_multichip_decoder/precise_gate_repair_api_retry_jobs.json` contains the same eight jobs with fresh `_api_retry` labels and output paths.

CPU-only validation imported the live TTNN binding, executed the actual configuration-assignment AST statements extracted from each edited helper, verified all 32 modes, constructed a KernelDescriptor, and verified its copied config preserves each mode and arithmetic flag. Both helpers pass Python compilation. Detailed results are in `cpu_binding_validation.json`. No mesh/device was opened, kernels were not compiled or executed, and hardware acceptance remains pending the root runner.
