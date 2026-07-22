# Qwen3-32B fused decoder

Stage 02 provides the graph-fused single-Blackhole decoder layer in
`tt/fused_decoder.py`. `FusedDecoder` inherits construction, validation, and
weight loading from `FunctionalDecoder`, but owns both runtime forwards. The
static test rejects functional-forward fallback and host/Torch conversion in
the measured path.

This stage does not contain optimized-decoder precision or program tuning,
multichip execution, a full model, generation, or vLLM integration.

## Runtime and context contract

The functional contract is unchanged: batch 32, BF16 weights/activations,
BF16 TILE/DRAM K/V caches, a 1x1 mesh, device input/output
`[1, 32, S, 5120]`, prefill cache writes at `[0:S]`, and one-token decode at
`current_pos`. Prefill accepts every logical `1 <= S <= max_cache_len`; TILE
padding and logical slicing remain internal. Device tests cover non-aligned
lengths 3, 17, and 33.

`doc/context_contract.json` is intentionally unchanged. No public dtype or
cache layout changed, and an isolated fused
batch-32 prefill at sequence 4,096 passed with output shape
`[1, 32, 4096, 5120]`. The adjacent 4,097 DRAM limit remains the Stage 01
physical boundary. Warmed RoPE/index slice outputs live in DRAM, but each
cache is deliberately single-entry: changing length or position deallocates
the prior tensors. At S=4,096 the two BF16 RoPE views are bounded to about
2 MiB, and a full 128-position allocation-stability stress remains flat.
Temporary K/V update grids consume L1.

## Delivered graph

The retained changes are:

- fold SiLU into the consuming binary-NG gate/up multiply;
- replace two paged K/V update dispatches with one
  `paged_fused_update_cache` on equal-sized, non-overlapping grids;
- cache only the current prefill-length and decode-position RoPE/index views,
  evicting the previous DRAM tensors, so warmed calls and traces avoid
  metadata work without request-to-request growth;
- replace the pre-create-head decode permute/L1-copy with a reshape view;
- reuse projection and MLP output tensors for the two residual adds;
- retain the measured-fast concat-to-interleaved/slice/transpose sequence.

That final concat sequence looks less minimal than direct slice-to-DRAM plus a
reshape view, but the current reproducible runner measures 81.82055 ms direct
versus 81.76745 ms retained over 500 traced replays. It is therefore required for
the fastest Blackhole path, not an unexamined layout copy.

### Full measured operation sequence

| Path | Operation(s) | Input/result placement or movement |
| --- | --- | --- |
| Prefill | input RMSNorm; packed QKV matmul; split Q/K/V heads | DRAM interleaved throughout |
| Prefill | Q RMSNorm; K RMSNorm; two rotate-half RoPE ops | DRAM interleaved; logical S slice after RoPE |
| Prefill | 32 K slices + 32 V slices; 32 K fills + 32 V fills | DRAM cache side effect; the unpaged public cache has no page table for batch fill |
| Prefill | causal SDPA; concatenate heads; O matmul; residual add | DRAM interleaved; add reuses O output |
| Prefill | post RMSNorm; gate matmul; up matmul; fused `SILU(gate) * up`; down matmul; residual add | DRAM interleaved; final add reuses down output |
| Decode | input RMSNorm; packed QKV matmul; reshape view; dedicated create-heads | DRAM to API-required height-sharded L1 heads |
| Decode | V grid relocation; Q/K to DRAM; Q/K RMSNorm; two rotate-half RoPE ops; K to update grid | one required height-to-height internal reshard, two sharded-to-interleaved moves, then one interleaved-to-sharded move |
| Decode | fused paged K/V update; decode SDPA | one cache-update op; SDPA emits DRAM |
| Decode | SDPA output to height-sharded; dedicated concat heads; width-sharded result to DRAM; logical slice; transpose | one interleaved-to-sharded and one sharded-to-interleaved move; retained measured-fast transpose |
| Decode | O matmul; residual add; post RMSNorm; gate/up matmuls; fused gate multiply; down matmul; residual add | DRAM interleaved; both adds reuse producer buffers |

The final source has no Torch/NumPy conversion, host fallback, tilize,
untilize, explicit `ttnn.reshard`, or collective. `tt-perf-report` sees one
0.89 us internal reshard when V is relocated from the create-head grid to the
non-overlapping grid required by fused K/V update; removing it makes the fused
cache op contract invalid. The remaining decode movement is likewise imposed
by create/concat-head, cache-update, and decode-SDPA contracts or by the
measured-faster O-projection input sequence.

## Correctness

All 64 Qwen3 layers have the same dense decoder kind. Layer 32 is the
acceptance representative; layers 0 and 63 add first/middle/last coverage.

| Coverage | Output PCC | K PCC | V PCC |
| --- | ---: | ---: | ---: |
| Synthetic prefill S=3 | 0.999320 | 0.999905 | 0.999874 |
| Synthetic prefill S=17 | 0.999506 | 0.999903 | 0.999863 |
| Synthetic prefill S=33 | 0.999558 | 0.999902 | 0.999867 |
| Synthetic decode position 17 | 0.999573 | 0.999901 | 0.999863 |
| Real layer 0 prefill | 0.999947 | 0.999900 | 0.999865 |
| Real layer 0 decode | 0.999946 | 0.999892 | 0.999884 |
| Real layer 32 prefill | 0.998876 | 0.999901 | 0.999862 |
| Real layer 32 decode | 0.998605 | 0.999901 | 0.999882 |
| Real layer 63 prefill | 0.989861 | 0.999902 | 0.999865 |
| Real layer 63 decode | 0.993443 | 0.999903 | 0.999856 |
| Traced synthetic decode position 20 | 0.999618 | full cache checked | full cache checked |

Layers 0 and 32 exceed the 0.995 output bar. Layer 63 exposes an existing
functional-baseline limitation for the selected random hidden state: both
functional and fused output PCC are exactly 0.989861 prefill and 0.993443
decode, while fused-to-functional PCC is 1.0. Thus graph fusion introduces no
material numerical delta; every first/middle/last fused result is bitwise
equal to its functional counterpart.

Two complete S=17 prefill/decode runs are bitwise deterministic. Repeated
decode advances positions 17-19 and compares the full initialized cache
prefix each step. Ten trace replays at position 20 are bitwise equal for the
output and entire K/V cache tensors. The same test passes with the watcher and
`throw_exception_on_fallback=true`.

The allocation-stability test warms all prefill tile classes through S=127
and every decode position 0-127, then repeats them. Allocated DRAM remains
exactly flat at 976,015,360 bytes across repeated prefill cycles and
980,276,736 bytes across decode positions, proving bounded view lifecycle.

## Performance

Measurements are real Qwen3-32B layer-32 weights on one Blackhole p300c,
batch 32, BF16, prefill S=17. Prefill is an 11-sample warmed median; decode is
the mean of 500 nonblocking trace replays followed by one synchronization.

| Path | Functional | Final fused | Improvement | Functional PCC | Fused PCC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Warmed prefill | 83.3613 ms | 83.0103 ms | 0.421% | 0.998839 | 0.998839 |
| Traced warmed decode | 82.1000 ms | 81.7882 ms | 0.380% | 0.998694 | 0.998694 |

The final topology also beats every correct candidate combination. The key
traced-decode results are: 81.7674 ms final fused, 81.8206 ms reproducible
direct concat view, 82.0975 ms standalone SiLU,
82.5056 ms packed gate/up, 81.7902 ms final concat plus two cache updates,
81.7928 ms final concat plus uncached metadata, and 81.8257 ms adapted
sharded Q/K norm. Differences across tables reflect independent measurement
runs; candidate comparisons are same-process within each JSON.

The signpost-delimited `tt-perf-report` captures are:

| Path | Device ops | Device time + gaps | Main conclusions |
| --- | ---: | ---: | --- |
| Prefill | 147 | 82.850 ms | 5 matmuls are 97.59%; 66 slices + 64 cache writes implement the observable batch-32 unpaged cache fill |
| Traced decode | 26 | 81.776 ms | 5 matmuls are 98.66%; one fused cache update; one decode SDPA; six required layout/grid moves |

## Complete graph-fusing pattern audit

| Skill pattern | Assessment, experiment, and disposition |
| --- | --- |
| Activation, softmax, RMSNorm | SDPA softmax and all RMSNorm sites already use dedicated ops. Standalone `silu` was folded into the consuming multiply; same PCC, with final decode 81.7677 versus 82.0975 ms, so retained. |
| Distributed RMSNorm and fused collectives | Not present on this 1x1 single-device stage; owned by a later multichip stage. |
| Prefill/decode SDPA | Functional graph already uses dedicated causal prefill SDPA and decode SDPA; no primitive QK/scale/softmax/V subgraph remains. |
| Split/create/concat heads | Dedicated prefill split/concat and decode create/concat forms remain. The pre-create-head permute/copy was replaced by a view. The checked-in `direct_concat_view` candidate is PCC-identical but measures 81.82055 ms versus 81.76745 ms final; the measured-faster materialization/transpose remains. |
| RoPE | Generic rotate-half `rotary_embedding` already replaces primitive math. `rotary_embedding_llama_fused_qk` is interleaved-pair RoPE and requires disjoint sharded Q/K grids; Qwen3 uses canonical half-split RoPE. Adopting it would require weight/head/cache reordering and public cache conversion, so it is not numerically contract-equivalent. Warmed rotary views remove repeated metadata dispatch. |
| TopK, MoE, conv, pooling, BatchNorm | Absent from this dense transformer layer. RepVGG, spatial-mean, conv/bias/activation/scale, padding, and MoE rewrites are not applicable. |
| Residual add + RMSNorm | Assessed at both residual boundaries. TTNN's residual RMSNorm returns the norm result only, but Qwen must retain the summed residual for the later attention/MLP add. Recomputing or copying it saves no operation, so not applicable. |
| Paged cache update | Two decode updates were replaced by `paged_fused_update_cache`, using disjoint 32-core K/V grids. A 500-replay interaction control measured 81.7700 ms fused versus 81.7902 ms separate; watcher/NoC evidence is clean, so retained. |
| Prefill cache fill | `fill_cache` is already dedicated. `paged_fill_cache` requires a page table that the public unpaged functional contract does not provide; synthesizing one would change the API and add work. The 64 fills and source slices remain required observable side effects. |
| Shared-LHS matmul | Q/K/V is already one packed projection. A packed gate/up projection plus two slices was PCC-correct but slower: prefill 83.7077 versus 82.9617 ms and decode 82.5056 versus 81.7677 ms. Separate gate/up plus fused binary activation is retained. |
| Permute/reshape/peer structural rewrites | Decode input singleton-axis permutation collapsed to a reshape view. All remaining structural transforms are dedicated head ops or the measured-fast post-concat sequence. No reducible permute-reshape-permute chain remains. |
| Sharded Q/K normalization | Legal block-sharded RMSNorm and height-sharded RoPE were adapted and intermediate PCC was about 0.999993. Direct sharded Q into default decode SDPA silently changed head/batch interpretation; materializing Q to DRAM restored 0.998694 end-to-end PCC. At 81.8257 ms it loses to the final 81.7700 ms path, so rejected. |
| Bias, matmul activation, transpose, and slice merging | Projections have no bias. The eligible activation is folded into binary multiply per the BF16 caveat. QK transpose/scale are already inside SDPA. No projection-output slice can be pushed into an operand without changing the required model width; packed gate/up was the applicable measured experiment. |
| Stable softmax, reduction/reshape, scaled-sum | No spelled-out softmax, max-subtract, sum/scale, or reducible keepdim sequence exists after dedicated SDPA/RMSNorm use. |
| Data movement and host fallback | Source audit plus the final reports find no host conversion, tilize/untilize, explicit reshard, or collective. Every reported decode move is API-required or measured faster; all removable metadata and pre-create-head movement is gone. |

After each retained or rejected candidate, the graph and per-op reports were
rescanned. No remaining Stage 02 single-device decoder fusion has both an
equivalent TTNN contract and a measured performance advantage.

## Artifacts

- `results/before_after.json`: final PCC and 11/500-iteration performance gate
- `results/candidates_concat_final.json`, `candidates_mlp_final.json`,
  `candidates_decode_confirm.json`, and
  `candidates_final_interactions.json`: retained/rejected candidates
- `results/sharded_qk_norm_*.json`: failed boundary probes and corrected retry
- `perf/prefill_ops_perf_results.csv` and `decode_ops_perf_results.csv`: raw
  Tracy device-op captures
- `perf/*_tt_perf_report.csv`, `perf/*_tt_perf_summary.csv`, and PNG summaries
- `logs/final_suite.log`, `logs/context_4096.log`,
  `logs/performance_gate.log`, `logs/watcher_correctness.log`, and
  `logs/watcher_device.log.gz` (lossless gzip of the raw device log)

The exact commands, device procedure, artifact paths, and review checkpoints
are recorded in `work_log.md`.
