## Program cache review: experimental/ccl/llama_reduce_scatter_matmul

OP reviewed
- `ttnn::operations::experimental::ccl::Matmul_RS` (new infra; mesh program cache; fused reduce-scatter + matmul).

Creation/override summary
- Builds reduce-scatter program via `LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing` and a matmul program; stores shared vars for both.
- Override walks mesh programs and:
  - Calls RS per-program override to refresh CBs and semaphore/addresses.
  - Calls matmul override helper to refresh tensor buffer addresses and CB bindings.

Findings
- Override fully refreshes runtime-only addresses for both fused stages; sequencing is correct.

Conclusion
- No program cache issues found. No failing cache test required.
