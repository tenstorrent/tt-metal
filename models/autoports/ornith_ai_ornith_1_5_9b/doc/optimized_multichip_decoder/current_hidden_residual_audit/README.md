# Current hidden-residual / persistent collective family — pending device evidence

The remaining gap in `../finalization_gap_audit.md` was that the historical B32 hidden-residual family preallocated RS buffers but used allocation-producing norm-stat and activation AGs. `tests/multichip_current_hidden_residual_candidate.py` closes that experimental coverage gap in source. It inherits the current production decoder, including the full-grid CCL manager, current paged attention, partial RoPE, pooled TILE convolution history, and post-loop recurrence. It imports no historical candidate implementation. No hardware was run and no production or existing queued files were edited.

## Exact boundary and lifetime contracts

The stack's global residual is BF16 TILE `[1,1,B,4096]`, column-sharded over TP4, so each rank holds `[1,1,B,1024]` in L1 interleaved storage. `--layout sharded` is mandatory. Both layer entry and exit have that same local width and storage. The stack composes shards only for host checks. It does not restore replicated residuals after either output projection, at decoder boundaries, or between layers.

Each decoder executes: local norm statistics → stats AG → local normalized residual → activation AG (separate or fused into column projection) → current attention → row-projection RS → local residual add → the same norm/MLP pattern → local residual add. The last add consumes the RS output and returns width1024. There is no inter-layer collective; the next layer's local normalization is the first operation on the residual.

The two norm roles (`input_layernorm`, `post_attention_layernorm`) own distinct stats and activation banks. The attention-output and MLP-down projections own separate RS banks. Each layer owns its own bank objects; layers share only the setup's current CCL manager/semaphores. An activation gather can never overwrite a stats buffer or the other norm's activation, and one layer cannot overwrite another layer's live output bank. The serial command queue and retained CCL semaphore/barrier arguments order repeated calls. Trace captures these fixed addresses. Concurrent execution of one candidate instance is outside this existing serial-stack contract.

| Persistent role, per layer | Logical shape per rank | Storage / dtype |
|---|---|---|
| Stats AG, 2 distinct banks | `[1,1,B,128]` | L1 interleaved BF16 |
| Separate activation AG, 2 distinct banks | `[1,1,B,4096]` | L1 interleaved BF16, or BF8 in explicit BF8 policies |
| Separate RS staging, 2 distinct banks | `[2,1,32,4096]` | L1 interleaved communication dtype |
| Separate RS output, 2 distinct banks | `[1,1,B,1024]` | L1 interleaved communication dtype |
| Fused AG staging, 2 distinct banks | `[1,1,32,4096]` | DRAM interleaved BF16 |
| Fused matmul-RS output, 2 distinct banks | `[1,1,32,1024]` | DRAM interleaved BF16 |

All nontile batches are internally padded before fused kernels and sliced to B afterward. Separate kernels use TILE physical padding with logical B. The candidate is setup-specialized to a configured B in 1..32; public inputs do not need tile-aligned B. Inherited prefill, including non-aligned sequence tails, is unchanged. The copied dispatch preserves production's exact full-pool test `(3,1,B,2048)` TILE history plus recurrent batch count, and its per-request fallback for private, incomplete, permuted, or subset pools. B1 uses that current fallback.

## Policies and comparisons

- `separate`: current hidden residual with separately allocated AG and RS; matched control for persistence.
- `persistent_all`: identical arithmetic/collective placement with all stats AG, activation AG, RS staging and RS output tensors setup-owned.
- `gather`: persistent stats, fused AG+column matmul, and persistent separate RS. Its MLP uses the existing packed gate/up weight with fused SwiGLU.
- `scatter`: persistent separate stats/activation AG and preallocated fused matmul-RS outputs.
- `fused_all`: both fusions, with persistent stats, fused AG staging and fused RS outputs.
- `bf8` / `bf8_persistent_all`: separate collectives with BF8 normalized activations and RS payload, BF16 norm stats and residual. Weight and recurrence precision remain inherited.
- `mixed_fused_all`: fused-all with BF8 **row-projection inputs**, BF16 activation AG and BF16 fused-RS output. This is explicitly an activation precision experiment; the fused-RS API does not expose a BF8 wire/output dtype switch.

The fused MLP changes floating-point operation boundaries relative to separate BF16 SiLU/multiply. Fused matmul-RS accumulates cross-device adds in FP32. These are candidate numerical differences, not gate changes. The harness still applies its existing prefill/decode/state/replicated-state gates. Short, long, and Watcher jobs must pass before any performance claim.

## Source proof and API adaptation

- Production `tt/multichip_decoder.py:25`–`38` initializes CCL semaphores on the full compute grid. The inherited constructor uses the provided shared `collectives.ccl` at line157; this candidate does not import the old fixed-grid manager.
- Production dispatch at `tt/multichip_decoder.py:1225` checks distinct state regions and the current TILE pool shape. Only the residual assertion, input storage and bank index reset change in the copied method. Production attention/pooled recurrence/prefill implementations are inherited unchanged.
- `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:490`–`493` emits one tile-wide RMS stat per rank, hence width32 locally and width128 after TP4 AG. Stats banks use BF16 regardless of activation policy.
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/all_gather_async_nanobind.cpp:290` exposes `persistent_output_tensor`. The device operation validates persistent shape/layout and returns the caller buffer (`device/all_gather_async_device_operation.cpp:61`, `:283`). The candidate supplies exact logical shape and dtype rather than relying on the validator's permissive gather-width inequality.
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.cpp:79`–`89` assigns persistent slots `[staging, output]`; `:103`–`105` considers both sufficient on the non-contiguous Linear path. Its device operation at `device/reduce_scatter_minimal_async_op_device_operation.cpp:228` doubles padded dimension0 for Linear topology, explaining `[2,1,32,4096]`. No Ring-only contiguous-staging helper is called.
- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/all_gather_minimal_matmul_async_nanobind.cpp:241` exposes `persistent_output_buffer`. Device validation at `device/all_gather_minimal_matmul_async_device_operation.cpp:79`–`88` requires enough padded rows and gathered K. Candidate explicitly pads to32 and uses K4096. The public op strips the internal gathered activation before returning chunked matmul results (`:689`–`694`), so `[0]` selects the actual projection result.
- **The current fused-scatter API supports persistence:** `ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/matmul_reduce_scatter_prefill_nanobind.cpp:93`, `:115` exposes `output_tensor`. Its device operation checks exact logical shape, dtype, layout, memory/spec and rejects input/weight aliasing (`device/matmul_reduce_scatter_prefill_device_operation.cpp:74`–`107`); `:255` returns it directly. Candidate allocates distinct BF16 DRAM `[1,1,32,1024]` banks and slices after the fused call. There is no API blocker here.
- Fused-scatter input supports BF16/BF8, while output is BF16 (`matmul_reduce_scatter_prefill_nanobind.cpp:81`, `:99`). Its program factory forces FP32 DEST and full sync (`device/matmul_reduce_scatter_prefill_program_factory.cpp:674`–`680`). Candidate preserves role fidelity and explicitly disables approximation.

The original batched-family implementation imported a frozen pre-integration base and expected history batch in dimension0. Neither assumption is carried forward.

## Memory and context implications

For B1..32, TILE physical rows remain32. BF16 `persistent_all` reserves 1,720,320 bytes/chip/layer (1,179,648 for RS, 524,288 for activation AG, 16,384 for stats). BF8 persistence reserves 921,600 bytes/chip/layer using 1088-byte BF8 tiles; stats stay BF16. `fused_all` reserves 16,384 bytes/chip/layer in L1 and 655,360 in DRAM. These are candidate bank payloads, not claims of reduced traffic or improved roofline.

The existing stack harness also constructs its shared replicated-AR workspace even for sharded families; that unused 2MiB/chip allocation remains common to these comparisons. No candidate frees it or silently changes the harness. This preparation does not change KV/cache/history shapes, max context, or `context_contract.json`. If a promoted persistent family changes feasible max context, the required physical maximum-context validation and context-contract update still apply. Layer-owned banks are suitable for this two-layer discriminating experiment; sharing a smaller role-safe workspace across a full stack would be a separate integration design requiring live-range evidence.

## Prepared jobs and host checks

`../current_hidden_residual_jobs.json` has nine focused jobs: separate/persistent-all/fused-all × B32 short16, B32 4095-prefix/1024-step answer, and B1 short16. All use state checks and traced warmed benchmarking. `../current_hidden_residual_adaptations_jobs.json` adds B32 gather-only, scatter-only, BF8-persistent, and B6 persistent tail coverage. `../current_hidden_residual_watcher_jobs.json` contains three separate B32 Watcher jobs without benchmarking.

Parent-owned serialized launch, not executed during preparation:

```bash
PYTHONPATH=. python_env/bin/python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.run_stage5_jobs models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/current_hidden_residual_jobs.json
```

Use the analogous adaptation and Watcher lists after the relevant numerical cases pass. The runner snapshots sources, respects STOP and the shared device lock, and refuses existing labels.

`host_verify.py` checked 512 mocked shape/ownership cases (8 policies × B1..32 × both layer kinds), distinct banks within/across layers, exact output widths, fused padding/slicing, stats dtype, and all persistent operand shapes. It compared the dispatch AST against production after normalizing the documented boundary edits, verified inherited attention/prefill method identity, checked all16 job references, and passed CLI help. Actual bound compute configs are constructed; tensor operations are mocked. `host_validation.json` is CPU/source evidence only, not JIT, device, PCC, watcher or performance evidence. A first device/API failure must still be adapted and retried under AutoFix; this source preparation is not an optimization rejection.
