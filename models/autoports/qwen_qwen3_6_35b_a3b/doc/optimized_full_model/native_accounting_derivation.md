# Native reduced-profile accounting derivation

This closes the distinct reduced-layer extrapolation requested by `optimize/references/perf-accounting.md`. It reconstructs layer boundaries from the final native operation order and frozen source, then uses each rank’s own measured kernel work. It does not replace the clean whole-model wall measurement or the queued-layer engineering budget.

## Reconstructed boundaries and validation

`final/profile_reduced/ops.csv` was streamed once between `PERF_TOKEN_OUT` and `PERF_TOKEN_OUT_END`. For every rank, model trace 0 has 16 replay sessions 5–20 with exactly 199 device operations each; sampling trace 1 has sessions 7–22 with exactly 31 each. The operation name, native program ID and first two input shapes at every ordinal are identical across the 16 replays within each rank/trace. The accompanying JSON retains these ordered signatures, first-replay CSV line anchors and averaged per-ordinal durations.

| Trace | Inclusive 1-based operation ordinals | Count | Ownership evidence |
|---|---:|---:|---|
| Model | 1–9 | 9 | Token slice/embedding/tilize/hidden all-gather, then position cast and two RoPE embedding/tilize sequences; `tt/model.py` calls these before its layer loop. |
| Model | 10–114 | 105 | Linear layer 0: ordered input norm begins with square/product; packed 3136-wide input scan, convolution, recurrent slab, output scan, router/experts, final all-reduce and residual add. |
| Model | 115–189 | 75 | Full-attention layer 3: sharded input LayerNorm, 2560-wide Q/K/V/gate projection, RoPE, paged KV updates, SDPA, output projection, router/experts, final all-reduce and residual add. |
| Model | 190–199 | 10 | Terminal sharded-to-interleaved, FP32 norm and BF16 head path, position increment, logits padding/copy. |
| Sampling | 1–31 | 31 | Separate canonical sampler trace, including candidate exchange, sampling, token merge and seed control. |

The boundary is source-backed rather than a guess based on program hash: `tt/model.py::decode` invokes the two layer calls sequentially before `terminal`; `tt/multichip_decoder.py::decode_batch` delegates a singleton directly to `_forward`, which ends with the MoE all-reduce/residual add. Ordinal 114 is that linear final add and ordinal 115 is the full-layer input norm. Ordinal 189 is the full final add and ordinal 190 begins `terminal`’s explicit DRAM conversion. `tt/generator.py::_decode_into_sampler` appends the pad/copy after `model.decode` has advanced position. No per-layer signposts exist, so these remain reconstructed ownership boundaries; no per-layer timestamp span is claimed.

All native program IDs exist in `native_program_metadata.json`. Summing ordered rows reproduces both the model/sampler totals in `component_summary.json` and every rank/component/program total in `token_out_op_groups.csv` (relative tolerance 1e-12, absolute 1e-8 µs). `final_profile_attribution.csv` independently labels the corresponding head, packed scan and shared expert groups; its grouped repeated operations alone cannot locate a layer boundary.

## Same-profile per-kind extrapolation

All entries below are milliseconds of summed **operation work**, averaged over the same 16 token-out replays, not whole-window critical-path durations. The final column is `entry + 30 × linear + 10 × full + terminal/advance/handoff + sampler`. Choose the maximum complete same-rank result; do not add maxima from different ranks.

| Rank | Entry/RoPE | One linear | One full | Terminal/advance/handoff | Sampler | Extrapolated all-40 operation work |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.048121625 | 0.565144375 | 0.282453562 | 0.373409563 | 0.431209063 | 20.631607125 |
| 1 | 0.049105625 | 0.562810937 | 0.282562313 | 0.371389063 | 0.436702500 | 20.567148437 |
| 2 | 0.048836500 | 0.561920500 | 0.283022125 | 0.373005125 | 0.436510562 | 20.546188438 |
| 3 | 0.043397750 | 0.563862938 | 0.282812438 | 0.373045063 | 0.432043375 | 20.592498688 |

The maximum is rank 0: **20.631607125 ms/token extrapolated native operation work**, versus **21.41686824405214 ms/token** clean full-40 token-out wall time in `final/performance.json`. The arithmetic difference is **0.7852611190521408 ms** (3.666554% of clean wall time). This supplies the requested extrapolation and numerical difference; it is **not measured host overhead, a full-model device critical path, or a mathematical lower bound**.

The reduced capture fences each token with the profiler enabled at positions 128–143. The clean all-40 run queues 127 tokens with synchronization only at loop boundaries at positions 128–254. Repeating layer 0/3 also assumes representative weights, expert routes, initial layouts and cache costs; inter-layer transitions can add conversions absent from an embedding-fed layer 0. Per-operation timing sums omit inter-operation gaps and do not reconstruct overlap. These differences prevent attributing the subtraction solely to host dispatch. In particular, this extrapolation must not be used to hide a gap by inflating a claimed lower bound, or to infer avoidable work from a mismatched control.

## Same-run reduced useful-byte roofline

The recorded token-out positions are 128 through 143 inclusive (`initial_position=128`, `final_position=144`, 16 steps). The arithmetic evaluates the existing CPU-only `tests/multichip_functional.py::byte_model` for layer 0 and layer 3 at each position and averages their useful bytes. The terminal terms are identical to the established whole-model byte model. No padding, intermediates, repeated reads, sampler padding or collective traffic is added.

| Useful bytes per rank/token | Mean bytes |
|---|---:|
| Linear layer 0 | 26536096 |
| Full layer 3 | 17577600 |
| lm_head | 135086080 |
| embedding_row | 1024 |
| norm_weight | 8192 |
| Total | 179208992 |

`lm_head = 2048 × 62080 × 17/16`, `embedding_row = 512 × 2`, and `norm_weight = 2048 × 4`. With 512 GB/s per rank, or equivalently four times both bytes and bandwidth for the mesh:

- Useful-byte ideal: **0.3500175625 ms/token** (`179208992 / 512e6`).
- Same-window native device operation work: **1.7032948125 ms/token**, the maximum complete model+sampler rank total (rank 2).
- Same-run fenced host wall: **2.015464375290321 ms/token**, excluding profiler dumps but including each step’s fence.
- Ideal/fenced-wall fraction: **17.366596%**; ideal/native-work fraction: **20.549441%**.
- Fenced-wall minus same-rank native work: **0.31216956279032115 ms/token**. This contains inter-operation/dispatch/synchronization effects not resolved by operation sums; it is not independently measured CPU time.

The reduced terminal-heavy workload’s roofline fraction is not the all-40 model’s useful-bandwidth fraction. Both the reduced triplet and the all-40 extrapolation are retained separately in `native_accounting_derivation.json`, together with all ordered native rows and validation flags. The clean all-40 useful-byte gate and queued-layer budget remain separate evidence.

## Reproduction scope

CPU-only extraction: stream `csv.DictReader(ops.csv)` through the exact token-out signposts; group rows by `(DEVICE ID, METAL TRACE ID, METAL TRACE REPLAY SESSION ID)`; require 16 equal-signature sequences per rank/trace; average `DEVICE KERNEL DURATION [ns] / 1000` per ordinal; sum the inclusive intervals above; compare grouped sums against the existing compact CSV/JSON. Load `byte_model` with `runpy.run_path` (its module imports no TTNN), evaluate `range(initial_position, final_position)`, and add the three terminal terms. No hardware, TTNN import, source change, archive or new artifact hashing was used. This document is accounting evidence, not a stage-review verdict.
