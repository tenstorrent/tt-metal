# Qwen3.6-27B Performance Optimization Plan

Status: Active
Last updated: 2026-08-27
Target: Qwen/Qwen3.6-27B on Blackhole P150 TP4/TP8
Branch baseline: `hous/qwen-perf-optimization` at `02051e16947`

## 1. Purpose

This is the authoritative plan for the next Qwen3.6-27B performance pass. It
supersedes completed and rejected attention-tuning experiments recorded in
`../RESULTS.md`.

The next pass is not another SDPA parameter sweep. Its primary objective is to
build the Gated DeltaNet (GDN) state and recurrent-kernel infrastructure needed
for:

1. faster single-token decode;
2. checkpoint-trained multi-token prediction (MTP) speculative decoding;
3. automatic prefix caching (APC) for hybrid attention/GDN state; and
4. lower-overhead continuous-batching request admission.

The first implementation milestone is a fused recurrent GDN decode/verify
operation supporting `T=1..3`, together with device-side GDN state copy,
snapshot, restore, and commit primitives.

## 2. Current measured baseline

The previous optimization pass was successful and is the baseline for all new
comparisons:

| Metric | Before | Current | Improvement |
|---|---:|---:|---:|
| Production gated-attention chain | 392.7 us | 233.1 us | 40.6% lower |
| 128K TTFT | 27.283 s | 23.845 s | 12.6% lower |
| 128K decode throughput | 23.740 tok/s | 25.330 tok/s | 6.70% higher |

At 25.33 tok/s, a normal decode step is approximately 39.5 ms/token. New
decode work must be compared against this production configuration, including
BF16 Q, BF8 K/V, the fused SDPA concat writer, and the 24-core decode
reduction.

`../RESULTS.md` remains the result and rejection ledger. Do not delete it or
replace measured results with projections.

## 3. Architecture evidence and identified gaps

The [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B) and
checkpoint config establish the following:

- 64 decoder layers: 48 GDN layers and 16 full-attention layers.
- Hidden size 5120 and vocabulary size 248,320.
- GDN uses 16 key heads, 48 value heads, and 128-dimensional key/value heads.
- The checkpoint was trained with MTP.
- `mtp_num_hidden_layers=1` and `mtp_use_dedicated_embeddings=false`.
- The recommended vLLM configuration uses the Qwen MTP method with two
  speculative tokens.
- Native context length is 262,144 tokens.

The local implementation has these concrete gaps:

- Both normal and FP8 weight loaders discard all 15 `mtp.*` tensors.
- There is no MTP module, draft loop, multi-token target verifier, acceptance
  path, or speculative GDN-state commit path.
- GDN decode uses a Python composition of generic TTNN operations rather than
  a dedicated recurrent decode operation.
- Qwen advertises `supports_prefix_caching=False`; its recurrent and
  convolution states are not represented in the prefix cache.
- Continuous-batching prefill downloads every GDN layer state to the host and
  uploads it again into the selected decode slot.
- The multimodal wrapper constructs the vision tower even for text-only
  serving.

The following suspected gaps were checked and are not runtime defects:

- Runtime layer dispatch reads the Hugging Face `layer_types` list correctly;
  the stale eight-layer `FULL_ATTENTION_LAYERS` constant is unused.
- Vocab-sharded LM-head output is already passed directly to on-device
  sampling, avoiding a full-logit all-gather in that path.
- Long-prefill GDN already uses the custom phased chunk kernel, flat QKV,
  in-kernel Q/K normalization, and fused collective projections.

## 4. Execution order

### Milestone 0: Lock references and measurement contracts

Goal: make every later optimization falsifiable and prevent correctness or
measurement drift.

Work:

- [ ] Add a device-free reference test for one recurrent GDN step and for
      sequential `T=2` and `T=3` steps.
- [ ] Capture the current TTNN graph/op count for one GDN decode layer.
- [ ] Add state comparisons for recurrent state, all four convolution taps,
      and output tensors.
- [ ] Define paired silicon measurements for isolated GDN latency and full
      model decode at ISL 128, 4K, 16K, and 128K.
- [ ] Preserve the current end-to-end baseline and sampling configuration.

Exit criteria:

- The reference independently detects output and state corruption.
- The benchmark reports warmup separately, uses paired A/B runs, and records
  raw repetitions rather than only a best result.

### Milestone 1: Device-side GDN state primitives

Goal: remove serving-path host round-trips and establish the state semantics
needed by speculation and APC.

Required operations:

- [ ] Copy a B=1 prefill scratch state directly into an arbitrary live decode
      slot without disturbing other slots.
- [ ] Snapshot a slot or active-prefix state entirely on device.
- [ ] Restore a snapshot to a slot entirely on device.
- [ ] Select and commit one of several tentative recurrent states.
- [ ] Apply slot remapping using the same state API.

Initial integration target:

- Replace the `ttnn.to_torch` / `ttnn.from_torch` round-trip in
  `Qwen36Model.prefill_paged_slots`.

Constraints:

- Recurrent state remains FP32. Do not trade state precision for bandwidth
  without long-decode model-level PCC evidence.
- Trace-visible buffers must retain stable addresses.
- Operations must preserve inactive batch rows.

Exit criteria:

- Continuous-batching admission performs no GDN state D2H/H2D transfer.
- Full prefill followed by decode produces the same logits and states as the
  existing path for B=1, B=8, and B=32.
- Slot remap and slot reuse tests pass with live neighboring requests.

### Milestone 2: Fused recurrent GDN decode operation

Goal: replace the generic recurrence chain with one purpose-built device
operation and make multi-token verification efficient.

Operation contract:

- Inputs: Q, K, V, beta, decay gate, FP32 recurrent state, scale, and active
  batch width.
- Outputs: GDN output and updated state, with optional tentative state
  checkpoints for `T>1`.
- Shapes: support the Qwen TP4/TP8 head geometry and `T=1..3` initially.
- Numerics: Q/K normalization and recurrence accumulation remain high
  precision.

Kernel design:

- [ ] Assign one or more value heads to each participating core.
- [ ] Perform Q/K normalization and key-to-value head expansion in-kernel.
- [ ] Load each 128x128 FP32 head state once per invocation.
- [ ] Fuse decay, `k @ state`, delta construction, beta scaling, rank-one
      update, and `q @ state`.
- [ ] Write the committed or tentative state without a separate full-state
      copy operation.
- [ ] Keep the token loop inside the kernel for `T=2..3`, so state stays local
      across verification positions.
- [ ] Reuse existing chunk-GDN utilities where they fit, but do not route
      `T<=3` through the large prefill algorithm merely to avoid a new op.

Follow-up fusion boundary, only after the recurrent op wins:

- Fuse the four-tap depthwise convolution, Q/K/V slicing, beta/decay
  preparation, or gated RMSNorm only when profiling shows that the expanded
  boundary improves end-to-end latency.

Exit criteria:

- `T=1` output and state match the current implementation at the established
  PCC thresholds.
- `T=2` and `T=3` match sequential reference execution for every token.
- Graph capture shows one recurrent device operation in place of the generic
  recurrence chain.
- Isolated kernel latency improves materially and full-model decode improves
  beyond run-to-run noise on silicon.
- No regression at B=1, B=8, or B=32.

### Milestone 3: Checkpoint-trained MTP speculative decoding

Goal: use the model's trained MTP head to increase accepted tokens per target
verification pass.

Model implementation:

- [ ] Stop filtering `mtp.*` weights and add explicit mapping tests for all 15
      checkpoint tensors.
- [ ] Expose the base model hidden state at the MTP seam before final norm and
      LM-head projection.
- [ ] Implement the embedding-side and hidden-side pre-FC RMSNorms.
- [ ] Concatenate the current token embedding and base hidden state, then apply
      `mtp.fc` from 10240 to 5120.
- [ ] Implement the checkpoint's one full-attention MTP decoder layer, final
      MTP norm, and shared LM head.
- [ ] Reuse the base token embedding and LM-head weights; the config says MTP
      does not use dedicated embeddings.

Serving implementation:

- [ ] Start with two speculative tokens, matching the model-card vLLM
      recommendation.
- [ ] Run the target model over the current token plus two drafts in one
      verification pass.
- [ ] Implement acceptance/rejection and sampling semantics on device where
      practical.
- [ ] Commit the correct GDN state for every acceptance length.
- [ ] Treat unaccepted attention KV writes as tentative through logical cache
      length or explicit commit semantics.
- [ ] Keep ordinary `T=1` decode as a feature-flagged fallback.

Measurements:

- Draft latency per speculative step.
- Target verification latency for `T=1`, `T=2`, and `T=3`.
- Acceptance-length histogram by workload.
- Accepted tokens per target pass.
- Effective end-to-end tok/s, not raw drafted tok/s.
- P50/P95 latency for short and long generations.

Workload set:

- Coding completion and repository-agent prompts.
- Tool-use/reasoning prompts with thinking enabled.
- Conversational prompts.
- Low-temperature/greedy and model-card sampling defaults.

Exit criteria:

- MTP logits and draft progression pass reference comparisons.
- All acceptance lengths commit the same target state as sequential decode.
- Effective decode throughput beats `T=1` on representative workloads without
  correctness or sampling regressions.
- MTP disables itself or falls back cleanly when its measured acceptance does
  not cover its overhead.

Generic external-draft speculative decoding is out of scope until the trained
MTP path has been evaluated. The checkpoint-provided head has lower memory and
integration cost and is the model vendor's recommended path.

### Milestone 4: Hybrid automatic prefix caching

Goal: reuse repeated prompt prefixes across both full-attention KV state and
GDN recurrent/convolution state.

State budget:

- On TP4, one GDN layer/device needs approximately 786,432 bytes of FP32
  recurrent state plus 20,480 bytes of BF16 convolution state.
- Across 48 GDN layers this is approximately 36.9 MiB/device per complete
  prefix checkpoint.
- Fine-grained GDN checkpoints would therefore consume excessive memory.

Design:

- [ ] Introduce a hybrid prefix entry containing attention page references,
      one complete GDN checkpoint, token boundary, hash/key, and ownership.
- [ ] Begin with sparse 2048-token-aligned anchors and an explicit LRU memory
      budget.
- [ ] Restore the nearest usable anchor and prefill only the suffix.
- [ ] Preserve absolute positions and full attention page tables during suffix
      prefill.
- [ ] Replace the new-sequence GDN reset with restore semantics on a cache hit.
- [ ] Add metrics for hit length, restored bytes, suffix tokens, eviction, and
      TTFT saved.

Correctness cases:

- Full uninterrupted prefill versus restore-at-2K plus suffix.
- Prefix hit followed by an exact boundary token.
- Partial final block and page-table remapping.
- Cache eviction and slot reuse.
- Concurrent cache hits into different decode slots.
- MTP and APC enabled separately first; combined operation is a later gate.

Exit criteria:

- Cache-hit logits, recurrent state, convolution state, and attention KV match
  uninterrupted prefill.
- Cache misses retain current performance within noise.
- Repeated-prefix TTFT improves in proportion to avoided prefix work, with
  bounded checkpoint memory.

APC is a repeated-prefix optimization. It is not expected to improve cold
single-request decode throughput.

### Milestone 5: Remaining long-context prefill work

These items follow the decode/MTP foundation unless profiling promotes them:

1. **GDN direct output layout**
   - Extend the chunk-GDN writer to emit token-major flattened output.
   - Remove `nlp_concat_heads` from all 48 GDN prefill layers.

2. **In-kernel valid-length masking**
   - Pass scalar/per-row valid lengths to the chunk operation.
   - Predicate padded tokens in-kernel instead of uploading host masks and
     applying five separate multiplies.

3. **Tiled causal depthwise-four convolution**
   - Consume QKV and carry in tiled layout.
   - Emit tiled convolution output and updated carry without the current
     concat and tile/row-major round-trips.

4. **Full-attention split-K prefill**
   - Parallelize long K ranges and reduce partial online-softmax state.
   - Enable only beyond a measured sequence-length threshold.
   - Treat this as a new algorithm, distinct from the rejected Q-chunk/core
     scheduling changes.

Each candidate must be isolated behind a flag until it passes operation-level
accuracy, model-level accuracy, and paired silicon A/B measurements.

### Milestone 6: Text-only serving mode

Goal: avoid multimodal startup and memory overhead for text-only deployments.

- [ ] Add a `language_model_only` initialization option.
- [ ] Skip vision reference loading, TT vision weight construction,
      multimodal registration/warmup, and trace buffers when enabled.
- [ ] Preserve the current multimodal default unless the deployment explicitly
      selects text-only mode.
- [ ] Measure startup time, peak host/device memory, available KV capacity, and
      maximum concurrency.

This is a capacity and startup optimization, not a direct token-latency claim.

## 5. Completed and rejected work: do not repeat

The following are not active plan items. They were already implemented,
measured, or rejected. Reopen one only with a new technical hypothesis that
directly addresses the recorded failure mode.

Completed and retained:

- BF8 paged K/V with BF16 Q.
- SDPA direct concat-heads output.
- Active-core reader/writer barrier correction.
- Dtype-dependent K chunk and output placement.
- Decode reduction cap of 24 cores.
- GDN decode L1 placement cleanup.
- GDN duplicated scalar-multiply removal.
- Fused GDN chunk-prefill phased path and native depthwise prefill convolution.

Rejected:

- BF4 K/V.
- Full BF8 Q/K/V.
- Q/K chunk size 256.
- Q chunk sizes 64 or 32.
- One Q chunk per core.
- More than 24 decode reduction cores.
- BF8 gate tensor with an added cast.
- Gate-multiply output moved to L1.
- Dtype-specific K/V reader barrier split.
- Approximate sigmoid gating.
- Exact sigmoid/gate fusion into the SDPA epilogue.

Important distinction: full-attention prefill split-K has not been attempted.
It is not equivalent to the rejected Q-chunk scheduling experiments.

## 6. Correctness and performance policy

### Correctness

- Preserve FP32 GDN recurrent state unless long-decode evidence justifies a
  change.
- Compare hidden/output tensors and persistent states, not only first-token
  logits.
- Teacher-force identical tokens when comparing continuation paths.
- Cover eager and traced execution, short/tail/long prefill, TP4/TP8 where
  applicable, and B=1/B=8/B=32.
- Any dtype or reduction-order change requires model-level PCC coverage.

### Performance

- Use the current production default as the control.
- Warm programs and traces before timing.
- Run paired A/B repetitions and report mean, spread, and raw samples.
- Report isolated kernel time and end-to-end impact; neither substitutes for
  the other.
- Record failed and neutral experiments in `../RESULTS.md` before reverting.
- A feature is not a win if it merely moves latency outside the measured span.

### Hardware authorization

Hardware is shared. A new silicon run requires fresh explicit authorization
for that run. Before accessing devices:

1. run `../hw-preflight.sh`;
2. follow its reported device/lock requirements;
3. use the shared lock required by the local harness; and
4. never pin `TT_VISIBLE_DEVICES` on this host.

Device-free reference tests, graph inspection, compilation, and simulator work
may proceed without silicon authorization.

## 7. Decision gates and stopping rules

- Do not proceed from a kernel microbenchmark to model integration without a
  state-aware correctness test.
- Do not enable MTP by default until acceptance-adjusted end-to-end throughput
  is positive across representative workloads.
- Do not enable APC by default until cache misses are neutral and checkpoint
  memory is bounded.
- Revert a candidate that loses consistently on silicon even when simulator or
  operation-count proxies predict a win.
- If a milestone misses its performance gate, preserve the reusable
  correctness/state infrastructure and record the failed performance
  hypothesis in `../RESULTS.md`.

## 8. Immediate next task

Start Milestones 0-2 as one bounded implementation sequence:

1. Add sequential `T=1..3` recurrent reference and state tests.
2. Add device-side scratch-to-slot state copy and snapshot/restore primitives.
3. Implement a fused `T=1` recurrent GDN operation.
4. Prove `T=1` parity and benchmark it.
5. Extend the winning operation to `T=2..3` with tentative-state outputs.
6. Only then integrate the checkpoint-trained MTP head.

The first code change should target the recurrent operation and its reference
tests, not another gated-attention scheduling or dtype experiment.
