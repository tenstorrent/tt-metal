# AutoDebug: nondeterministic batch-1 decode trace replay

## Symptom and inspected scope

- Failing check: `test_decode_trace_replay_is_deterministic[mesh_device0-1]`.
- Reported behavior: eager prefill/decode agrees with the HF oracle at PCC
  0.99997, while the tensor returned during capture is not bitwise equal to
  each of three replays.
- Inspected:
  `tt/functional_decoder.py`,
  `tests/test_functional_decoder.py`,
  `models/tt_transformers/tt/generator.py`, and trace tests under
  `models/demos/gemma4`, `models/demos/vision`, and
  `models/demos/deepseek_v3`.
- This was source inspection only. No implementation or hardware state was
  changed.

## Ranked hypotheses

### H1 (highest): the test confuses numerical repeatability with bitwise equality

Evidence: the only assertion is `torch.equal(captured, value)`. Decode contains
large BF16 matmuls and paged decode SDPA; the acceptance contract elsewhere is
PCC-based. The test does not print mismatch count, maximum error, PCC, NaNs, or
whether eager executions are themselves bit-identical. A one-ULP scheduling
difference therefore has the same failure signature as stale/corrupt trace
state.

Prediction: capture/replay tensors are finite, have very high mutual PCC, and
differ only at a small number of BF16 elements; repeated eager calls may show
the same strict-equality failure.

Smallest experiment: retain the existing harness and print, for capture versus
each replay and replay versus replay, `torch.equal`, mismatch count, max/mean
absolute difference, finiteness, and PCC. Then run three synchronized eager
forwards with fresh identical caches and compare with the same metrics. If
eager is also non-bitwise but PCC is stable, change the determinism gate to an
explicit tight numerical tolerance/PCC plus repeated-input HF PCC, rather than
claiming trace corruption.

### H2: the batch-1 tile padding contains unwritten data that enters a decode kernel

Evidence: batch 1 is physically tile-padded to 32. The path uses
`nlp_create_qkv_heads_decode`, paged update/SDPA, `nlp_concat_heads_decode`, and
only then slices `[0:batch]`. Batch 32 has no inactive batch lanes. Several of
these decode-specific kernels operate on physical tiles/shards; an unwritten
inactive lane is a direct source of replay-to-replay bit changes even when the
logical eager result has excellent PCC.

Prediction: batch 32 is stable while batch 1 is not; padding the logical input
to an explicitly zeroed 32-row workload (with invalid/inactive positions
handled by the supported contract) removes the variation, or the first
variable boundary is one of QKV-head creation, SDPA, or head concat.

Smallest experiment: first run the existing parameterized test separately for
batch 1 and 32 with the H1 metrics. If only batch 1 varies, capture progressively
larger subgraphs and compare their retained outputs: QKV head creation; after
RoPE; after paged SDPA; after head concat; final decoder. Inspect logical and
padded shapes/memory configs at each boundary. This localizes the first
operation exposing uninitialized padding without changing precision.

### H3: the in-place KV-cache update/read sequence is not replay-idempotent at position 0

Evidence: a trace replay performs two in-place
`paged_update_cache` calls and immediately reads both caches in paged SDPA.
The test compares capture (the first traced write) with later writes to the
same physical cache and position, but never snapshots cache rows. Thus the
observed output difference may originate at the mutable cache boundary rather
than in trace output handling. The nearby DeepSeek trace test validates paged
update execution but does not establish this decoder's update-then-read
determinism.

Prediction: the physical K/V row addressed by page-table slot 0 changes across
replays, or a trace containing cache update alone is unstable. If cache rows
remain identical, this hypothesis is refuted.

Smallest experiment: after capture and each replay, copy only the addressed
physical K/V cache row to host and compare it to the projected/rotated K and V.
Also capture an update-only trace with the same tensors and compare that row
after three replays. Repeat once at position 33 with a prefilled cache to rule
out the empty-cache/position-zero special case.

### H4: allocator/lifetime setup leaves capture buffers vulnerable to aliasing

Evidence: `compile_output` is retained through capture and never deallocated.
Repo trace harnesses commonly synchronize and deallocate warmup output before
capture; some explicitly reproduce allocator state and verify buffer
addresses. This test records no buffer addresses and therefore cannot exclude
an output/scratch alias or a capture allocation pattern different from the
intended steady state.

Prediction: deallocating `compile_output` after synchronization before capture,
or using a separately allocated stable trace input/output setup matching the
production generator pattern, makes replay stable; alternatively an address
ledger reveals an overlap.

Smallest experiment: log buffer addresses for input, caches, page table,
positions, captured output, and retained intermediates if exposed. A/B only
`compile_output.deallocate(True)` after synchronization and before capture.
Do not keep this as a fix unless the address evidence or repeatable A/B result
verifies the hypothesis.

### H5: strict capture-versus-replay comparison includes a one-time capture-state effect

Evidence: capture executes the graph while recording it. The harness compares
that execution directly to replay, but has no unmeasured warm replay. Production
examples commonly capture, execute one warm replay, and then measure/use later
replays. Stateful cache writes make capture-versus-steady-replay a weaker
equivalence check than replay-versus-replay.

Prediction: replay 1/2/3 are mutually stable although the captured execution
differs, or only replay 1 differs.

Smallest experiment: report the full pairwise matrix
`capture,R1,R2,R3` using the H1 metrics. If all replays agree with each other
but not capture, investigate capture-time cache/allocator state; do not label
the trace generally nondeterministic.

## Recommended experiment order

1. Add metric diagnostics and the pairwise matrix (H1/H5); no graph changes.
2. Compare batch 1 with batch 32 and bisect retained subgraph outputs (H2).
3. Snapshot the exact addressed cache rows and run update-only trace (H3).
4. Run the allocator-lifetime A/B and address ledger (H4).

The current evidence is insufficient to select a fix. In particular, raising
precision or changing compute-kernel configuration is not supported: eager
HF-vs-TTNN PCC is already 0.99997, and no precision-localized failure has been
shown.
