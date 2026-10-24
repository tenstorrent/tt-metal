# Fused-kernel fidelity attribution

The fused matmul/reduce-scatter operation launches a LoFi matmul kernel and a separate HiFi4 reduction kernel. `tools/tracy/process_ops_logs.py` overwrites the scalar `MATH FIDELITY` field for each compute kernel, so the last reduction kernel supplies the row's scalar value.

The runtime audits retain that scalar value, inspect the captured `matmul_struct.compute_kernel_config`, and resolve each captured compute-kernel hash to its exact compiled `chlkc_descriptors.h`. The archived headers and SHA-256 receipts prove the matmul compiled with `MathFidelity(0)` (LoFi); they are stronger evidence than a Python policy label. No profiler value is silently rewritten.

Source chain: `matmul_reduce_scatter_async_device_operation.cpp` forwards the captured matmul compute config unchanged; the CCL reduction receives its own configuration. `process_ops_logs.py:1505–1508` flattens multiple kernel fidelities into one scalar. The enum is defined in `tt_metal/api/tt-metalium/base_types.hpp`.

`initial_prefill_receipt.json` corresponds to the first K4-column final-default capture; later final runtime audit JSONs carry the descriptors associated with their own captured kernel hashes. Changing column geometry does not imply reusing stale row-kernel evidence: every final fused row is independently matched by hash.
