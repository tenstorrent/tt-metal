Softmax program cache review

Status: Reviewed — no program cache issues found.

Reviewed files
- `device/multi_core/softmax_op_multi_core.cpp` (interleaved and sharded variants)

Findings
- Interleaved: override updates input, optional mask, and output addresses; recomputes per-core args and updates CB total sizes based on current shapes. Ordering aligns with creation indices.
- Sharded: dynamic CBs bind to current input/output and optional sharded mask buffers; override updates mask address in reader args per core when mask present.
- Program selection and hashing include relevant attributes (scale, causal, layouts, compute configs) via op infra; compile-time args capture tile sizes and grid parameters.

Recommendation
- No changes required.
