# AutoDebug Report

## Starting Evidence

- This is the fresh, source-only investigation requested by the AutoFix loop. The repo-local runner was attempted from this stage directory and exited before investigation with `autodebug.sh: codex executable not found`. The runner selects Codex by default and fails its `command -v codex` preflight at `.agents/scripts/autodebug.sh:131-134`; this independently forked investigation is the fallback. The earlier failure is consistent with the invoking shell not having the Codex executable on `PATH`.
- Starting review: `stage_review.md` says `more-work-needed`. Each claim below is treated as a hypothesis, not a runtime fact.
- Inspected: `tt/optimized_decoder.py`, optimized/functional/capacity tests, `context_contract.json`, stage README/work log/perf report, both timing JSONs, the retained final Tracy CSV and a read-only `tt-perf-report` rendering, and `shard_advise/{report.json,report.txt,final_ir.mlir}`. Required `autodebug`, `autofix`, `optimize`, and `shard-advise` skill contracts were read.
- The worktree is dirty and the optimized decoder, its tests, and stage directory are untracked; no implementation or test was modified. No TT hardware, server, profiler capture, vLLM, or full-model path was run.
- Retained final timing/profile evidence predates the current implementation and test: timing JSON `14:30:25`, Tracy CSV `14:31:01`, and source/test `14:38:26` UTC. Stage-local final regression, watcher, 8192-pass, and 16384-OOM console logs are absent.

## Headline Findings

### 1. Final evidence is not tied to the final source, and candidate evidence is overwritten

**Hypothesis.** The documented final gates may have passed an earlier tree, while most candidate rejections cannot be independently reconstructed because the harness overwrites a single result file.

**Source evidence.** The source/test mtimes postdate both final artifacts. `_write_result` writes one requested path (`tests/test_optimized_decoder.py:38-44`), and the candidate test always writes `initial_candidates.json` (`:527-530`). The two saved JSONs contain only `functional_bf16` and only `optimized_mlp_bfp4_lofi_40c`, although the work log lists more than twenty rows. The stage contains no compact ordinary-regression, watcher, or capacity logs and no manifest binding artifacts to source hashes.

**Prediction.** A clean rerun could differ in PCC, latency, operation rows, or capacity behavior; even if it agrees, there is currently no artifact proving agreement or preserving the rejected-candidate matrix.

**Focused experiment.** After all code fixes, calculate SHA-256 for `optimized_decoder.py`, `test_optimized_decoder.py`, and `context_contract.json`; run the ordinary suite, one immutable row per material candidate, selected final timing, final Tracy/`tt-perf-report`, 8192 and expected-16384 capacity probes, and a separate watcher run without changing those files. Save stdout/stderr and a manifest containing hashes, git identity/status, command, relevant environment, UTC time, exit status, and produced artifact paths. Reject the lifecycle hypothesis only if every gate shares the same hashes and the structured matrix reproduces the documented decisions.

**Likely smallest fix.** Test/artifact harness only: write a run-id directory plus append-only JSONL/CSV rows instead of one `initial_candidates.json`; emit a final manifest and compact logs (`ordinary.log`, `watcher.log`, `capacity_8192.log`, `capacity_16384.log`, timing JSON, compact perf TXT/CSV). Include exact watcher-signature grep and OOM text.

**Uncertainty.** Mtime ordering proves a provenance gap, not that the later source is wrong. The unsaved runs described in prose may have occurred exactly as stated.

### 2. Real checkpoint weights are used with synthetic activations; the available checkpoint cannot generate a layer-32 prompt activation

**Hypothesis.** Synthetic activation distribution, rather than target-model behavior, may be the only reason BFP8-attention LoFi/BFP4 policies were rejected.

**Source evidence.** Real-weight correctness, candidate timing, and profile tests all create prefill/decode inputs with `torch.randn` (`test_optimized_decoder.py:212-217,233-237,263-271,548-557`). No activation artifact or capture runner exists. The local Qwen3-32B snapshot contains only `model-00009-of-00017.safetensors` plus the index. The index maps layer 32 (and layer 31) to shard 9, but embeddings and layer 0 map to missing shard 1; layers 0-30 therefore cannot be evaluated to produce the input of layer 32. No alternate complete snapshot or recorded Qwen3-32B activation was found locally. Prompt tokenization assets are also absent from this snapshot.

**Prediction.** The existing random-input PCC rows will reproduce only synthetic stress behavior. A prompt-derived layer-32 input is impossible from the currently available files alone, and lower-precision ordering on real activations remains unknown.

**Focused experiment.** First perform a file-only prerequisite check for embedding plus layers 0-31 shards, config/tokenizer, and record their hashes. Once present, use a bounded HF-reference capture utility—not a TT full-model implementation, generator, or vLLM path—to stream embeddings and layers 0-31, hook the input of layer 32, and save CPU BF16 prefill `[1,32,17,5120]` plus several cache-consuming decode inputs at advancing positions. Record prompt/token IDs, checkpoint revision, positions, dtype/shape, and hashes. Feed the same saved tensors and reference caches through the packed final topology for the selected policy, attention LoFi, BFP4 attention, and all-BFP4; record prefill PCC, each advancing decode PCC/cache-use result, and traced latency. A one-shot sequential HF prefix is the minimum semantic producer of a genuine layer-32 activation and does not add a repo TT full-model or serving implementation.

**Likely smallest fix.** Add a small offline activation-capture/loader harness and immutable activation metadata under the test-evidence boundary. With the present snapshot the exact blocker is missing checkpoint shards 1-8 (at minimum all tensors for embeddings and layers 0-30) and prompt tokenizer assets; alternatively accept a hash-identified externally recorded layer-32 activation set.

**Uncertainty.** Real activations may confirm the current BFP8/HiFi2 attention choice. One prompt is insufficient for distribution tails; use a small fixed prompt/position set, but do not broaden this decoder-layer repair into generation or full-model bringup.

### 3. The `shard_advisor` mode is a partial projection/residual approximation, not the authoritative legal layout chain

**Hypothesis.** The saved 1.771 ms “literal advisor” result, even if accurate, does not reject all legal recommendations in `final_ir.mlir`.

**Source evidence.** Current advisor mode changes main residual/norm/projection configs (`optimized_decoder.py:546-589`), but decode unconditionally converts Q and K to DRAM and emits Q/K RMSNorm and RoPE to DRAM (`:1013-1033`). In contrast, `final_ir.mlir:45-58` uses separate L1 block-sharded Q/K norm layouts, L1 height-sharded RoPE inputs/outputs, and one-core L1 cos/sin rows. The IR also preserves the original L1-interleaved reshaped residual through the first residual add (`:38-41,68-70`) and applies an L1-interleaved reshape plus final DRAM output revert (`:79-83`), whereas current mode converts the residual to the 80-core layout before the first norm and converts to DRAM before reshape. No raw advisor timing/PCC row is retained.

**Prediction.** A candidate implementing the full legal Q/K norm, RoPE, residual, and revert chain will have a different op graph and may have different PCC/latency from 1.771 ms.

**Focused experiment.** Build a declarative, advisor-only candidate that maps every `final_ir` op/result layout and inserted `to_memory_config`/`to_layout` in order. Before runtime, dump a table of IR line/op, requested layout/program config/revert, materialized TTNN object, and any adaptation. Then run one real-activation advancing-decode PCC and traced profile. Compare its op sequence to `final_ir`; label it literal only if every legal entry is present.

**Likely smallest fix.** Candidate-mode code in `OptimizedDecoder`, keeping the production default unchanged: separate the preserved input residual layout from the projection residual layout; add Q block 10x4 and K block 8x4 norm configs; move normalized Q/K and cos/sin to the IR height-sharded layouts; materialize all MLP transitions and the final output revert.

The one exact minimal exception is `nlp_concat_heads_decode`: authoritative IR line 66 is itself marked `validation_unfixable` because it feeds DRAM/interleaved input, while current TTNN requires HEIGHT_SHARDED input (`nlp_concat_heads_decode_device_operation.cpp:43-59`). Its `memory_config` parameter is ignored and the op returns a fixed L1 width-sharded output (`:72-110,125-149`), so the IR's DRAM/interleaved concat result cannot be requested literally. The honest candidate must record this exception, insert the legal HEIGHT_SHARDED input conversion, accept the fixed output, and make the smallest explicit conversion to the next advised layout. Generic rotary embedding does support non-width sharded input/output when shard width equals padded width (`rotary_embedding_device_operation.cpp:60-96`), so RoPE is not blocked by source contract.

**Uncertainty.** Advisor/branch TTNN version skew may expose further runtime validation failures. Such failures must be retained as exact minimal repros rather than silently dropping the corresponding entry.

### 4. Dominant DRAM-sharded geometry is only partially swept, and output subblocks are internal rather than user-configurable

**Hypothesis.** The saved work-log matrix does not establish an independent, precision-locked per-role geometry optimum; “40 cores” describes storage-shard geometry, not a directly exposed DRAM-sharded compute-grid knob.

**Source evidence.** `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` exposes only `in0_block_w`, `per_core_M`, `per_core_N`, and optional fused activation (`matmul_program_config_types.hpp:71-76`; `matmul_nanobind.cpp:546-600`). It exposes no compute grid, output block, or output subblock. The factory chooses the device compute grid/DRAM readers and derives subblocks with `get_matmul_subblock_params`, then may pad subblock width (`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:91-170`). Thus `tt-perf-report`'s “No output subblock size found” is first an observability/API limitation, not proof of a 1x1 subblock.

The final CSV attributes show, per replay: QKV `in0=4, per_core_M=1, per_core_N=8` at about 183 us; O `8,1,5` at about 145-146 us; gate/up each `4,1,20` at about 233 us; down `20,1,4` at about 222-223 us. Their activation/output storage grids are respectively 40, 32, 40, and 40 cores from current source geometry. All are marked `SLOW` at roughly 54-58% reference. Raw CSV `CORE COUNT` (100/80), rendered-report `Cores` (12), storage-core count (40/32), and the program factory's derived worker set are different quantities and must not be collapsed into one “core count.” Current global `decode_in0_block_w_limit` cannot independently sweep roles: at the final storage grids, legal shard-local divisors are QKV/gate/up `{1,2,4}`, O `{1,2,4,8}`, and down `{1,2,4,5,10,20}`. Changing storage grids can unlock other divisors. Candidate constructors vary gate/up target cores while down defaults to 40, and retained artifacts contain none of those rows.

**Prediction.** A structured dump will show that several named cap candidates leave some roles unchanged, that the output subblock is factory-derived, and that at least some legal role-specific storage-grid/block combinations were not independently measured under the final dtype/fidelity.

**Focused experiment.** Add a source-derived config serializer and, for each role, record logical/padded shape, dtype/fidelity, weight layout and eight-bank shard, activation/output shard grid and shard shape in tiles, program class, `in0_block_w`, `per_core_M/N`, raw and normalized profiler core fields, internally derived/observed output subblock, row time, whole-layer time, PCC, and decision. First run a no-hardware enumeration of all legal common-divisor storage grids and shard-local block widths. Then measure one role at a time with final precision and recorded activations; include the strongest saved/reproduced baseline. Instrument the program descriptor or add a compact debug field to expose the factory-selected subblock before considering a new tuning knob.

**Likely smallest fix.** Evidence/harness plus per-role constructor knobs. Do not add public subblock fields initially: the current program class cannot express them, so the smallest exact blocker is the TTNN DRAM-sharded program-config API/factory boundary. If observing the derived choice reveals a material missed option, the next intervention boundary is that lower-level program config/factory, not model-side invented fields.

**Uncertainty.** Source predicts factory-derived `1x8` for QKV and typically `1x5` for the other N widths with eight active DRAM banks, but active-bank/padding decisions are runtime-derived; this must be observed rather than reported as a measured fact. The older profile also cannot prove current-source geometry.

### 5. Per-position tensors grow with context and fixed-position trace replay does not test generation semantics

**Hypothesis.** Each new position permanently retains device buffers, while the trace tests replay the same input and position and repeatedly overwrite one cache row.

**Source evidence.** Two dictionaries are initialized at `optimized_decoder.py:451-452`; every unseen position retains two sliced RoPE tensors (`:637-654`) and one repeated index tensor (`:656-662`) with no eviction/deallocation. A BF16 RoPE row is physically four 32x32 tiles, 8192 bytes; two rows plus 32 int32 indices are approximately 16,512 bytes per position per layer before allocator overhead. At 8192 positions this is about 135,266,304 bytes per layer and 8,657,043,456 bytes across 64 independent layer objects. These persistent bytes are absent from `context_contract.json`. The eager synthetic test advances four positions, but trace helpers capture position 17 once and replay unchanged buffers (`test_optimized_decoder.py:77-100,504-514,578-590`).

**Prediction.** Device allocation/owned-tensor count grows monotonically with unique eager positions; a single captured trace cannot produce correct position-advancing outputs merely by replaying it because its cos/sin/index addresses and contents remain for position 17.

**Focused experiment.** Before/after object and allocator accounting over many unique positions should show linear growth today. For semantics, capture once after prefill, then for positions 17-20 update both hidden input and position state before each replay, compare each output and newly written K/V row with HF, verify the next step consumes prior rows, and assert buffer addresses/count remain constant. The current test should be retained as fixed-input determinism only, not advancing-decode evidence.

**Likely smallest fix.** Add an explicit bounded `DecodeTraceState` owned per trace/in-flight slot: fixed-address cos row, sin row, and batch index buffers; `prepare_position` updates their contents in place before replay; `decode_forward` consumes that state instead of indexing persistent dictionaries. Use a bounded ring only when concurrent in-flight replays require it. Eager calls may create/deallocate temporaries, or reuse one state. Account `slots * 16,512` bytes (plus allocator alignment) in the context contract. A future device-side gather/increment can remove host preparation, but is not required to prove decoder-layer trace semantics.

**Uncertainty.** Exact allocator consumption may exceed tile payload bytes. Whether in-place host/device copies are trace-safe must be verified against fixed-address replay; if not, use the smallest preallocated device-side position-prep trace or indexed gather rather than returning to per-position ownership.

### 6. The documented accounting omits the theoretical roofline and mixes independent runs

**Hypothesis.** Observed bandwidth/utilization was mislabeled as theoretical roofline closure.

**Source evidence.** `perf_report.md` gives 1.357 ms device work and 1.367 ms wall time from an explicitly independent run, but no byte/peak-bandwidth calculation. For the saved final policy, physical 32x32 tile sizes are 1088 bytes for BFP8 and 576 bytes for BFP4 (repo data-format sources/tests). The five weights therefore total 321,454,080 bytes: QKV 55,705,600; O 44,564,480; gate/up/down 73,728,000 each. At `current_pos=17`, the SDPA dynamic window rounds 18 positions to one 32-token chunk (`rt_args_common.hpp:95-107`); two BFP8 caches for batch 32, 8 KV heads, width 128 add 2,228,224 bytes. The installed `tt-perf-report` Blackhole reference is 512 GB/s (`perf_report.py:316-323`). Thus the mandated weights-plus-KV lower bound for this batch-32 layer step is 323,682,304 / 512e9 = 0.632192 ms, before other DRAM traffic. This calculation describes the saved workload, not the newer unprofiled source.

**Prediction.** A coherent final rerun will report a theoretical lower bound near that value at position 17, with device time above it; the gap will include achieved DRAM efficiency plus non-weight/cache operations. At later context positions the KV term grows by the kernel's rounded dynamic window.

**Focused experiment.** In one final source-hashed invocation, emit a traffic ledger (physical tile bytes, tensor tile counts, position, rounded KV read window, device count, peak decimal GB/s), a signposted device window, and warmed wall timing for the same trace/workload. Label the unit “one batch-32 decoder-layer step,” not ambiguously “one user token.” Reconcile theoretical lower bound, device time, and wall time; separately list activation/cache-update/layout traffic excluded from the mandated lower bound.

**Likely smallest fix.** Reporting/harness only: generate a small structured roofline JSON beside the final timing/profile artifacts, sourced from runtime shapes/dtypes and the architecture reference, and remove the “complete” claim until a same-invocation final run exists.

**Uncertainty.** The exact full-path DRAM byte count is higher than the weights-plus-KV lower bound and depends on op traffic; 0.632192 ms must not be presented as a predicted measured latency. Peak 512 GB/s is a theoretical tool reference, not sustained bandwidth.

## Recommended Experiment Order

1. Fix evidence retention first: immutable run rows, source hashes, config serializer, compact logs, and a final manifest. Otherwise every following result remains unauditable.
2. Resolve the activation prerequisite. With the current files, record the missing-shard/tokenizer blocker; when provisioned, capture and hash a small layer-32 prefill plus advancing-decode activation set using only the HF reference prefix.
3. Implement the complete legal advisor candidate and its explicit concat-head exception. Run focused real-activation PCC/advancing-cache verification before timing it.
4. Replace the unbounded position dictionaries with bounded fixed-address trace state; prove one-capture, advancing-input/position/cache behavior and constant ownership.
5. Enumerate and serialize DRAM-sharded geometry, expose factory-derived subblocks, then run precision-locked per-role measurements. Keep only individually evidenced changes.
6. Select the final default, then rerun all gates on unchanged hashes: ordinary, capacity, watcher, timing, Tracy/`tt-perf-report`, candidate comparison, and one-invocation roofline/device/wall accounting.

## Scope/Constraints

- Source-only diagnosis. No runtime fact is asserted unless present in retained artifacts; calculations are labeled as source/artifact-derived predictions or lower bounds.
- No implementation/test changes were made. Only this `AUTODEBUG.md` was written.
- No TT device command, server, profiler capture, reset, vLLM, multichip, generator, or repo full-model implementation was started.
- The activation proposal is a bounded offline HF-reference prefix capture solely to produce decoder-layer test inputs. It must not grow into TT full-model or serving work. It is currently blocked by missing local checkpoint prefix shards/tokenizer assets.
- Advisor recommendations seed one candidate; they do not replace the authoritative DRAM-sharded search. The concat-head entry is the one source-proven advisor IR exception; any additional exception needs an exact retained error/minimal repro.
