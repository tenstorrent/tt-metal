# GPT-OSS 20B fused decoder

Stage 02 provides the graph-fused, single-Blackhole decoder in
`tt/fused_decoder.py`. `FusedDecoder` inherits construction, validation, and
weight loading from `FunctionalDecoder`, but implements both runtime forwards
and MoE itself. The fused-path test rejects calls to either functional forward,
so the delivered suite cannot pass through a functional fallback.

## Runtime and context contract

The functional decoder's public signatures, BF16/TILE/DRAM paged K/V caches,
and dense 1x1 mesh semantics are preserved:

| Path | Input/output | Cache | Configured bounds | Device-tested coverage |
| --- | --- | --- | --- | --- |
| Prefill | `[1, B, S, 2880]` | `[B, 8, 128, 64]` K and V | `2 <= S <= 128` | B=1 at S=3, 17, 33; B=2 at S=3 |
| Decode | `[1, B, 1, 2880]` | same paged cache, updated at `current_pos` | `0 <= current_pos < 128` | B=1 through positions 3–18; B=2 at position 3 |

Non-aligned lengths are accepted directly; TILE padding and logical slicing
stay internal. Position, RoPE, and cache-index views are cached lazily after
their first use, so warmed calls and trace replay do not dispatch host-side
position work. This also prevents a TTNN program-cache alias observed when a
changing scalar RoPE index reused the first compiled value.

`doc/context_contract.json` records the memory-layout change. Persistent split
gate/up copies add 1,062,051,840 bytes to the functional decoder's
1,646,373,824 bytes of static BF16 weights, for 2,708,425,664 bytes total. The
configured 128-entry cache remains allocated and the tested support boundary
advances from S=17 to S=33; no advertised Stage 01 capability is reduced. The
Hugging Face target of 131072 remains physically impossible for the inherited
dense-mask design because its BF16 causal mask alone is 34,359,738,368 bytes,
larger than the allocator-reported 34,178,731,008 bytes before weights, input,
cache, outputs, or temporaries. Lengths 34–131071 were not capacity-probed and
are not claimed infeasible.

## Delivered topology

Prefill replaces the functional manual GQA sequence—KV repetition, transpose,
QK matmul, scale, mask add, sink concat, softmax, sink removal, value repeat,
and value matmul—with one sink-aware causal
`scaled_dot_product_attention`. It attends to current logical K/V while
`fill_cache` preserves the cache side effect. The sink is pre-divided by the
attention scale once because TTNN SDPA scales its sink input together with QK.

Decode uses causal `scaled_dot_product_attention_decode` with cached position
metadata. This removes the explicit mask slice/repeat and Q DRAM copy. The
concat-head output remains width-sharded through the O projection, and a
reshape view replaces the prior permutation. One 0.58 us DRAM-interleaved to
height-sharded conversion is required because the GQA decode-SDPA operation
rejects sharded output.

The exact GPT-OSS MoE retains the FP32 router, stable top-4 softmax/scatter,
clipped gate and up branches, `up + 1`, 1.703125 sigmoid coefficient, and dense
expert accumulation. Biases are passed through `linear`; eligible elementwise
results reuse buffers; sigmoid is folded into its multiply. Multi-token
prefill uses construction-time split gate/up weights, eliminating two large
runtime strided extracts. Single-token decode uses the measured-faster wide
projection plus extracts.

## Correctness

All values below are direct Blackhole-to-Hugging-Face PCC; the functional
acceptance bar is 0.99.

| Coverage | Output PCC | K PCC | V PCC |
| --- | ---: | ---: | ---: |
| Synthetic prefill S=3 | 0.999885 | 0.999945 | 0.999951 |
| Synthetic prefill S=17 | 0.999849 | 0.999946 | 0.999950 |
| Synthetic prefill S=33 | 0.999840 | 0.999945 | 0.999950 |
| Batch-2 prefill S=3 | 0.999886 | 0.999944 | 0.999954 |
| Batch-2 decode position 3 | 0.999814 | 0.999945 | 0.999946 |
| Real layer 12, sliding prefill | 0.993041 | 0.999948 | 0.999952 |
| Real layer 12, sliding decode | 0.999298 | 0.999946 | 0.999949 |
| Real layer 13, full prefill | 0.995429 | 0.999951 | 0.999952 |
| Real layer 13, full decode | 0.999350 | 0.999958 | 0.999954 |
| Traced layer 12 decode at position 17 | 0.999800 | 0.999946 | 0.999949 |

The functional layer-12 prefill PCC was 0.999193. Sink-aware Flash SDPA changes
the BF16 reduction order, producing a -0.006152 delta while remaining above
the same 0.99 acceptance bar. Layer-12 decode remains 0.999298. Layer 13 covers
the other meaningful `full_attention` kind; at the configured cache extent,
both layer kinds intentionally see the same window.

The suite also compares real nonzero-weight split-prefill MoE against the wide
control at S=3, 17, and 33; every output has PCC 1.0. Two complete
prefill/decode runs are bitwise equal. Repeated paged decode advances positions
3–18 with output PCC 0.999712–0.999816, checks the complete initialized cache
prefix each step, and proves the unused suffix is unchanged. Ten trace replays
at position 17 are bitwise equal for output and the complete K/V tensors.

## Performance

Measurements use device 0 of a local P300c endpoint, the P150 1x1 mesh
descriptor, synchronized warmed calls, 20 prefill iterations, and 200 decode
trace replays. The final same-process gate is:

| Path | Functional baseline | Final fused | Improvement |
| --- | ---: | ---: | ---: |
| Warmed prefill mean | 8.206 ms | 7.158 ms | 12.77% |
| Warmed prefill minimum | 7.969 ms | 7.087 ms | 11.07% |
| Traced warmed decode mean | 5.988 ms | 5.891 ms | 1.62% |
| Traced warmed decode minimum | 5.974 ms | 5.883 ms | 1.52% |

The final `tt-perf-report` captures match this topology:

| Path | Summed device time | Ops | Main conclusions |
| --- | ---: | ---: | --- |
| Prefill | 7.028 ms | 52 | 6 matmuls (85.70%), 1 SDPA, no explicit layout conversion |
| Decode | 5.856 ms | 56 | 5 matmuls (80.46%), 1 decode SDPA, 2 paged updates, one required 0.58 us interleaved-to-sharded conversion |

The runtime contains no Torch conversion, host fallback, collective,
`tilize`/`untilize`, or explicit `reshard` call. Reported tilize/untilize
kernels are internal TTNN format work around exact router/MoE operations. The
wide single-token topology retains those internal kernels because the exact
split alternative regresses traced decode by about 0.90 ms.

Retained artifacts:

- `logs/final_suite.log`, `logs/performance_gate.log`, and
  `logs/watcher_correctness.log`
- `logs/candidate_*.log` and `tests/fused_decoder_candidates.py`
- `perf/prefill_tt_perf_report.csv`, `perf/prefill_tt_perf_summary.csv`, and PNG
- `perf/decode_tt_perf_report.csv`, `perf/decode_tt_perf_summary.csv`, and PNG
- `perf/reports/reports/fused_prefill_final/2026_07_22_17_56_07/ops_perf_results_fused_prefill_final_2026_07_22_17_56_07.csv`
- `perf/reports/reports/fused_decode_final/2026_07_22_17_56_21/ops_perf_results_fused_decode_final_2026_07_22_17_56_21.csv`

## Complete fusion-pattern audit

| Skill pattern | Assessment, experiment, and disposition |
| --- | --- |
| Activation, softmax, RMSNorm | Dedicated softmax and RMSNorm already used. Generic SiLU/SwiGLU cannot express GPT-OSS's two clamps, `up + 1`, and coefficient; the exact sigmoid is instead folded into multiply. |
| Prefill SDPA | Manual attention collapsed into sink-aware SDPA; real PCC passes and warmed prefill improves, so retained. |
| Split/create/concat heads | Dedicated prefill split-QKV/concat-heads and decode create/concat-heads forms retained. Decode concat stays sharded into O projection. |
| RoPE | Dedicated rotary embedding retained; warmed token views replace runtime slice dispatch. Removing prefill logical slices fails non-aligned K/V shape validation. |
| TopK and router | Dedicated `topk` retained. Existing fused MoE-gate modules do not preserve this dense 1x1 FP32 stable routing contract, so FP32 linear/softmax/top-k/scatter remains. |
| Residual + RMSNorm | Tried structurally; the residual sum is consumed again by the later residual. The fused form would require recomputation/copy and saves no dispatch, so rejected. |
| MoE sparse/dedicated experts | Packed and split BF16 sparse matmul, BF8 weights, and `in0_block_w` 1/2 were tested. Real decode PCC was 0.605508–0.702890, below 0.99. `moe_compute`/`TTMoEDecode` require packed low-precision and per-device expert contracts not equivalent to this dense 1x1 translation. |
| Paged cache update | Dedicated `fill_cache`/`paged_update_cache` retained and watcher-tested. Prefill SDPA is intentionally separate because cache fill is an observable output side effect. |
| Shared-LHS matmul | Wide packed gate/up is the shared-LHS rewrite. It is faster for decode; construction-time split operands are faster for prefill. Both exact candidates were PCC-tested. QKV is already packed. |
| Structural permute/reshape | Decode permute/reshape collapsed to a reshape view. Other head transformations are already dedicated operations. Spatial-mean and RepVGG rewrites do not occur. |
| Matmul bias/activation/transpose/slice | Biases use `linear`; eligible activation is folded. QK transpose/scale moved into SDPA. Pushing gate/up slices into split operands is retained for prefill but rejected for decode on measured latency. |
| Other op merging | No convolution, pooling, batch norm, max-subtracted softmax, scaled-sum, or reducible keepdim adjacency occurs. No multi-device collective exists in this stage. |
| Data movement | Removed decode mask slice/repeat, Q-to-DRAM, concat-to-DRAM, permutation, and per-call position slicing. GQA SDPA rejects sharded output, leaving the single measured 0.58 us conversion as required. |

After every applicable rewrite, the graph was rescanned and checked against
the final per-op reports. No remaining Stage 02 decoder fusion has both an
equivalent TTNN contract and a measured performance advantage.
