# Stage 07 Multichip and Optimization Checklist

This crosswalk separates Stage 07 full-path work from decoder policy already selected and closed in Stages 04–05. Stage 07 does not reopen a broad decoder datatype frontier; it verifies that the measured complete path still executes the selected policy and optimizes the terminal/orchestration overhead that was newly visible at full-model scope.

## Mesh and tensor contract

| Tensor/boundary | Global logical shape or role | Per-device/within-device contract | Padding or movement |
| --- | --- | --- | --- |
| Token embedding weight | `[262144, 5376]` | TP4 hidden-column shard `[262144, 1344]` | BF16 DRAM; embedding output is restored to the decoder's replicated BF16 residual contract on device. |
| Inter-layer residual | batch/tile rows by hidden 5,376 | Replicated BF16 DRAM on all four devices | No host or inter-layer layout boundary; Stage 05's fastest stack-compatible contract is preserved. |
| Decoder weights | 60 layers: 50 sliding, 10 full | TP4 local attention heads and column/row-parallel projections | Attention BFP8/LoFi, MLP BFP4/LoFi, exact Stage 05 program configs. Detailed shapes are in `../optimized_multichip_decoder/README.md`. |
| KV cache | 50 physical-1,024 and 10 physical-262,144 layers | Local TP heads/device, BFP8 paged storage | Page block 64; BF16 decode updates and BFP8 prefill fill; explicit page tables. |
| Final norm input/output | hidden 5,376 | Replicated BF16 residual, width-sharded working norm | Arbitrary logical M is retained. |
| LM-head weight | `[5376, 262144]` tied values | TP-local `[5376, 65536]`, then eight `[5376, 8192]` splits/device | Each split width-sharded over eight DRAM views as `[5376, 1024]` per view. No padded vocabulary IDs are introduced. |
| LM-head input | one decode tile by hidden 5,376 | Four logical width shards, 1,344 hidden columns/shard | 168 K tiles / 4 = 42 tiles/shard; selected `in0_block_w=2`. |
| Sampler-ready logits | one logical row by vocabulary 262,144 | TP-local BF16 `[1,1,32,65536]` shard/device | Softcap and sampling remain device-side; no full-vocabulary gather. |
| Greedy winner | one token/fixed slot | Eight local-winner cores/device, then small TP4 pair exchange/reduce | Explicit lowest-global-token tie rule; output aliases persistent `tt_out_tok`. |

Target mesh is four P150b devices in a one-dimensional TP4 mesh with Linear `FABRIC_1D`. This is the existing correct model topology and the strongest same-model/same-hardware baseline. Stage 07 improves within-device terminal sharding; it does not compare against a single-chip 31B model that cannot hold the same production contract.

## Operation-topology audit

| Sequence/bottleneck | Candidate or lower-movement family | Dtype/fidelity/layout constraint | Action and evidence |
| --- | --- | --- | --- |
| Embedding -> decoder residual | Keep hidden TP shard, distributed norm, or restore replicated residual | Decoder stack consumes the Stage 05 replicated BF16 boundary | Retained the measured stack-compatible boundary; no new all-gather/reshard or host boundary. |
| 60 decoder layers | Reopen residual/CCL/fused-projection families | Must preserve selected attention BFP8/LoFi, MLP BFP4/LoFi, phase CCL policy, and cache | Retained Stage 05 winner and its rejection ledger. Runtime rows verify the intended dtypes/fidelity. |
| Final norm -> interleaved LM head | DRAM-sharded LM head with width-sharded input | BF16 weights/logits and HiFi2 required by inherited accuracy policy | Implemented. Reduced steady improves 303.502 -> 339.164 t/s/u. |
| Common `LMHead1D` reuse | Use `models/common/modules/lm_head/lm_head_1d.py` directly | Gemma uses tied `embed_tokens.weight`, a selective loader, model-owned TT tensors, and arbitrary full-logit M; it has no TTTv1 `args.dram_matmul_config`/grid/cache contract | Reviewed and reproduced its execution family locally: split DRAM-sharded weights, width-sharded input, linear, sharded-to-interleaved, concat. Model-local code preserves tied-value loading, explicit deallocation, TP-local ordering, and 1–32-row tiling. |
| One 65,536-column local projection | Unsplit DRAM-sharded projection | Same BF16/HiFi2 | Rejected by exact 11,674,368-byte static-CB > 1,572,864-byte L1 failure. |
| LM-head split/input geometry | 4,096/8,192/16,384; four or eight input shards | Same BF16/HiFi2 policy and eight physical DRAM views | 8,192/four-shard/block-2 wins after throughput and qualitative A/B; 16,384 has exact L1 clash. See `candidate_results.csv`. |
| LM-head K block | selected four-shard geometry has 42 K tiles/shard; legal larger divisors 3, 6, 7, 14, 21, 42 | Same BF16/HiFi2 dtype/fidelity and split family | Block 3 is slower and changes aligned greedy; 6 clashes with live L1; 7/14/21/42 exceed total L1. Block 2 is the only correct passing winner. |
| Split outputs -> sampler logits | Keep sharded output through sampler versus interleaved local concat | Sampler requires one ordered TP-local vocabulary shard; no global gather | Local interleaved concat retained. Sharded-to-interleaved conversion is 2.08% of reduced device work; sampler consumes local shards directly. |
| Generic common TopK / force argmax | Common `Sampling1D`, partitioned TopK, force-argmax | Must be exactly greedy at BF16 ties and preserve fixed slots/trace ownership | Stage 06 evidence rejects incorrect/10.625-ms paths; exact custom sampler is 298.93 us. Non-greedy common top-k/top-p compatibility remains available. |
| Token/position/RoPE/page refresh | Host writes/readbacks versus persistent device tensors | Serving contract requires changed-only tables and reset-safe ownership | Persistent `tt_out_tok`, device increments, stable tables, and nonblocking split replay retained; final counters are in `token_out_no_readback.json`. |
| Multi-row readiness logits | Dynamic-M sharded matmul versus logical tile decomposition | Selected DRAM-sharded factory requires `per_core_M=1` tile | `$autofix` proves normalize-once 1–32-row tiling; 33-row hardware and full 249-row readiness pass. |

## `$optimize` closure

- [x] Complete token-out decode is traced with zero per-token host feedback, synchronization, or full-logit readback.
- [x] Decode working activations remain width-sharded in L1 where selected by Stage 05; residual/cache boundaries remain the measured stack-compatible layouts.
- [x] Prefill retains the established DRAM-interleaved/2D-program decoder policy; the new terminal accepts arbitrary logical M.
- [x] Operation-topology audit is recorded above, including layout conversions and terminal/sampling alternatives.
- [x] Decoder topology families, residual layouts, fused CCL candidates, activation/CCL precision, and persistent buffers retain the Stage 05 coherent-family measurements and rejection ledger.
- [x] Strongest available baseline and every material Stage 07 LM-head geometry are compared in `candidate_results.csv`.
- [x] Final default repeats the selected candidate within 0.2% (339.164 versus 339.823 t/s/u).
- [x] Profiler rows verify BF16/HiFi2 LM head, BFP8/LoFi attention, and BFP4/LoFi MLP.
- [x] Important decoder and terminal ops use explicit memory, program, and compute-kernel configs.
- [x] Every larger legal four-shard `in0_block_w` value (3, 6, 7, 14, 21, 42), the earlier eight-shard alternatives, and material split widths 4,096, 8,192, and 16,384 are measured or have exact L1 blockers under the final BF16/HiFi2 policy.
- [x] Decoder attention/MLP precision, fidelity, packing, core grids, SDPA, and dominant projection geometries remain the real-weight Stage 05 selections; Stage 07 did not silently replace them.
- [x] The selected terminal uses all eight physical DRAM views, four logical input shards, and clean tile divisibility.
- [x] Avoidable full-vocabulary gather, generic TopK, force-argmax, host argmax, and full-logit readback are absent.
- [x] Persistent CCL and sampler outputs remain preallocated; no trace-capture allocation is inserted in steady replay.
- [x] Reduced precision was not applied to the inherited BF16 terminal policy; broad Pareto selection remains owned by `$datatype-sweep`.
- [x] Reduced full-path `tt-perf-report` includes embedding, representative sliding/full layers, final norm, LM head, logits, exact sampling, feedback, cache/page-table state, and the production trace path, as required instead of profiling all 60 layers.
- [x] Full 60-layer end-to-end timing and correctness use the production generator; `perf_summary.json` reconciles lower bound, device representative evidence, and end-to-end timing.
- [x] Batch-1 is the latency target; batch-two mixed prompts/fixed slots/inactive rows remain hardware-tested. Full-context batch three is the physical capacity upper bound, not a claimed tested mode.

## `$multichip` closure

- [x] Target hardware, `[1,4]` mesh, TP4 weight/activation strategy, and terminal within-device sharding are explicit.
- [x] `../context_contract.json` preserves 262,144 context and unchanged weight/KV capacity while adding Stage 07 placement evidence.
- [x] Valid non-aligned sequence support passes 33-row hardware and exact 249-row readiness evidence.
- [x] Tensor shapes, mesh shards, within-device shards, and internal tile handling are recorded above.
- [x] Full-stack AIME24 prefill/teacher-forcing validates the target mesh against the pinned HF top-k reference.
- [x] BFP8 paged KV, mixed prompts, changed-only tables, fixed slots, and inactive rows pass the focused full-model suite.
- [x] Warmed split trace replay and exact greedy feedback pass repeatedly with persistent inputs.
- [x] Runtime fallback audit is clean.
- [x] Scoped worker watcher passes; full ETH instrumentation has an exact pre-execution configuration-buffer blocker and is not misclassified as model failure.
- [x] Before/after same-mesh latency and reduced full-path profiler communication/DRAM/compute/data-movement conclusions are recorded. A single-chip 31B production baseline is not physically comparable to this required TP4 contract.
