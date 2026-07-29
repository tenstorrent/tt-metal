# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Select the faster authentic dense-expert precision policy, or produce an
  equivalent authentic failure.
  Evidence:
  `STAGE_REVIEW_2.md` required BFP4 to be selected if equivalent authentic
  coverage passed. The new precision test forces the dense-expert path with
  `dense_expert_batch_threshold=1` and varies its real checkpoint weight dtype
  at `tests/test_optimized_decoder.py:419-449`. Both eight-row matrices pass:
  BFP8 is 0.999103-0.999989 and BFP4 is 0.997234-0.999941 for layers 1/4,
  prefill/decode, and batches 1/32
  (`artifacts/authentic_bfp{8,4}_matrix_clean.{xml,log}`). Matched candidate
  timing also shows BFP4 winning on every dense-expert workload, including
  batch-32 decode at 2.2145 ms versus 3.3911 ms and batch-1 prefill at
  3.7623 ms versus 4.7528 ms
  (`candidates/real_validated_bfp4_expert_*.json` versus
  `candidates/matched_bfp8_expert_*.json`). Nevertheless, the shipped defaults
  remain BFP8 at `tt/optimized_decoder.py:40-41`. The work log explicitly says
  the only veto is the synthetic b32/s1 and b2/s33 matrix at PCC
  0.981633/0.981610 (`work_log.md:287-295`,
  `artifacts/synthetic_bfp4_dense_expert_rejection.{xml,log}`).
  Why this matters:
  This directly violates OPT-012 and the stage-review more-work standard:
  synthetic/random weights and activations cannot veto a faster
  lower-precision policy after equivalent real-target weights and propagated
  target activations pass. It also does the opposite of the explicit second
  review instruction.
  Required next step:
  Select BFP4/LoFi for the dense-expert gate/up/down fields while retaining
  independently justified BFP8 for the sparse active-expert path, then rerun
  the final authentic, functional, trace, watcher, b1/b32 latency, and current
  profiler gates. If the two synthetic conditions represent a real model
  risk, reproduce either condition with real checkpoint weights and
  propagated target activations; only an equivalent authentic failure can
  justify retaining BFP8.

- P1: The selected batch-1 MoE prefill is still dense all-expert, not the
  claimed active-expert path.
  Evidence:
  The default threshold is 32 tokens (`tt/optimized_decoder.py:79`), and
  `_sparse_moe()` routes every `total_tokens >= 32` to
  `_dense_expert_moe_chunk()` (`tt/optimized_decoder.py:1116-1141`). Thus the
  final sequence-128 batch-1 prefill rows are dense. The current profiler
  confirms `RepeatDeviceOperation` followed by three ordinary
  `b={128}` expert matmuls in
  `tracy/selected/moe_rope_prefill_b1/filtered.csv`; there are no sparse
  expert matmuls in that measured prefill path. The purported branch-proof
  test does not exercise the selected default: it overrides the threshold to
  `1 << 30` and also changes intermediate placement at
  `tests/test_optimized_decoder.py:643-705`. No warmed active-expert prefill
  candidate is saved. This contradicts `README.md:110-113` ("Batch 1 uses
  active-expert sparse matmuls") and the completed checklist claim at
  `work_log.md:260-264`.
  Why this matters:
  The optimize contract treats dense all-expert execution as a debug baseline,
  not an optimized MoE result. AutoFix investigated batch-32 decode only; its
  conclusion that batch 1 remains active-expert is true for decode but not for
  the reported prefill workload. The first-review request to measure
  representative non-aligned active-expert prefill therefore remains open.
  Required next step:
  Measure the already branch-proven active-expert path under the same warmed
  batch-1 prefill harness at representative aligned and non-aligned lengths,
  integrate it if correct and faster, and rerun final evidence. If it cannot
  satisfy the functional no-regression bar, adapt/AutoFix the prefill-specific
  sparse topology and preserve an exact blocker before retaining dense
  execution. Correct the README and performance accounting to name the path
  actually measured.

- P1: The legal DRAM-sharded expert probe does not earn rejection of the
  DRAM-sharded family.
  Evidence:
  `test_dram_sharded_expert_projection_candidate` runs one synthetic
  eight-bank group with fixed gate/up `in0_block_w=4` and down
  `in0_block_w=2`; it is not connected into the decoder or target expert
  weights (`tests/test_optimized_decoder.py:529-608`). Its reported "lower
  bound" then multiplies a particular unfused serial implementation by 16,
  including separate SiLU, multiply, an immediate
  `sharded_to_interleaved(..., DRAM)`, reduction, and an add for every group
  (`tests/test_optimized_decoder.py:619-634`). The program explicitly leaves
  `fused_activation=None` at line 587, does not compare packed gate/up, does
  not carry the sharded result through the actual down/routing/reduction
  chain, and does not sweep geometry. The earlier projection-only measurements
  were 2.438 ms BFP8 and 1.999 ms BFP4, both below the selected complete
  batch-32 layer, so adding one chosen set of removable boundary operations
  does not establish a lower bound. The BFP4 projection PCC is also accepted
  with a lowered 0.99 test threshold and measures 0.99384, while the authentic
  full-output BFP4 path passes the normal 0.995 bar.
  Why this matters:
  The adapted probe proves that the batched DRAM-sharded op can express an
  eight-expert group; it is not a minimal repro proving the family impossible.
  OPT-003/004/010/011/014 require the legal family to be measured as a
  compatible full chain, including fused/packed alternatives and avoiding an
  immediate restore to the old DRAM-interleaved contract. The second review
  therefore is not closed by this extrapolation.
  Required next step:
  Integrate the legal bank-grouped kernel into an actual dense-expert
  gate/up-activation-down-routing-reduction candidate. Cross BFP4/LoFi and
  BFP8/LoFi with legal block widths/geometries, packed versus split gate/up,
  fused SiLU where supported, and a sharded carry through the next consumer.
  Measure correctness with real target weights/activations and traced
  whole-layer b32 latency. Reject only the measured full family or preserve an
  exact blocker encountered after these adaptations.

- P1: Required profiler evidence predates the final correctness repairs.
  Evidence:
  Every selected raw/filtered profile is from the July 28 23:09-23:12 runs
  under `tracy/selected/`. Commit `0085f30d237` subsequently changed the
  decode RoPE core ordering and the non-aligned packed QKV/O path. The old
  batch-32 decode layout was not merely slower: the authentic investigation
  found it corrupting lanes 8-31, and the corrected current path is only
  evidenced by JUnit/logs and wall-latency JSON. No advice-enabled
  `tt-perf-report` was collected from the corrected code in
  `0085f30d237`/`c7e024e8faa`.
  Why this matters:
  Required dtype/topology/device-time rows currently describe a path later
  proven incorrect. Final wall timing reproduces the repaired default, but it
  cannot verify current op topology, dtype propagation, device time, or the
  wall/device/roofline reconciliation required by the optimize skill.
  Required next step:
  After resolving the policy/topology findings above, collect new separate
  advice-enabled prefill and traced-decode reports from the final correct
  default for representative layer kinds at b1 and b32. Recompute accounting
  from those same runs and use those rows, not the pre-repair profiles, as the
  final evidence.

- P2: The optimized prefill path has no advertised-context preservation
  evidence.
  Evidence:
  `doc/context_contract.json` records 500,000-token prefill evidence only
  under `functional_decoder/`. The sole optimized capacity artifact,
  `doc/optimized_decoder/context500000_decode_b32.json`, covers dense decode
  and cache allocation. Although `tests/optimized_decoder_capacity.py`
  supports `--mode prefill`, no optimized maximum-context or near-maximum
  non-aligned prefill result exists. This is material because the optimized
  prefill implementation replaced the functional matmul programs and the
  second-review investigation found non-finite output in one large-M 10x10
  packed QKV/O contract before adding a special 512-row compatibility branch.
  Why this matters:
  The decoder advertises the inherited 500,000-token public prefill contract.
  A decode-only cache allocation does not prove that the changed optimized
  prefill path preserves that capability, especially after a real large-M
  correctness defect was found.
  Required next step:
  Preserve machine-readable optimized-path prefill evidence at the advertised
  aligned and near-limit non-aligned lengths for the representative layer
  kinds (at least the same b1 capability covered by the functional stage), or
  record and resolve a concrete physical blocker. Do not reduce the context
  contract without the capacity proof required by the optimize skill.

## Other Concerns

- The exact stdout PCC and DRAM-candidate timing logs used for the
  second-review closure are present in the live worktree but are not all
  tracked at `c7e024e8faa`; the tracked one-line JUnit XML proves test
  outcomes but not the exact numbers quoted in the README. Preserve compact
  machine-readable metrics with the remediation reruns so a clean checkout
  is independently auditable.
- `tt-perf-report` advice still marks the selected dense MLP rows `SLOW` and
  the dense expert rows at `in0_block_w=1`. Earlier geometry evidence is
  useful, but the current final-policy rerun must explicitly reconcile this
  advice after the precision and DRAM-sharded findings are resolved.

## Hard-Check Gaps

- No current profiler artifact covers the post-review2 correct path; this is
  required work above, not merely a preferred evidence format.
- No selected-default warmed latency exists for active-expert batch-1
  prefill; the available path test changes the policy.
- No optimized-path advertised-context prefill artifact exists.
- No final-default performance/watcher/profiler suite exists with the
  authentic-passing dense BFP4 policy.

## Anomaly Ledger

- Observed anomaly:
  Dense BFP4 passes the complete authentic matrix and is faster, but BFP8 is
  selected.
  Evidence:
  `authentic_bfp4_matrix_clean.{xml,log}`, matched candidate JSON, default
  policy fields, and `work_log.md:287-295`.
  Affected path:
  Dense-expert MoE prefill and batch-32 decode.
  Control or comparison:
  Equivalent eight-row BFP8 matrix and matched BFP8 timing.
  Likely subsystem:
  Precision-policy selection/evidence interpretation.
  Investigation performed:
  Authentic layer 1/4 prefill and cache-consuming traced decode at b1/b32,
  plus a separate synthetic stress matrix.
  Resolution:
  more-work-needed; the synthetic-only veto is invalid.

- Observed anomaly:
  Documentation says batch 1 is active-expert, while selected prefill is
  dense all-expert.
  Evidence:
  Default threshold and branch source, final sequence-128 policy JSON, and
  dense `b={128}` profiler rows.
  Affected path:
  Layer 1/4 batch-1 prefill at 32 or more logical tokens.
  Control or comparison:
  The forced-threshold sequence-33 active-expert correctness test.
  Likely subsystem:
  Phase/workload routing policy.
  Investigation performed:
  Source-path derivation and profiler-row inspection.
  Resolution:
  more-work-needed; the forced test is not selected-path performance.

- Observed anomaly:
  A measured serial bank-group estimate is labeled a DRAM-sharded lower
  bound.
  Evidence:
  `tests/test_optimized_decoder.py:529-634` and
  `dram_sharded_expert_candidate_full_lower_bound.log`.
  Affected path:
  Batch-32 dense expert gate/up/down.
  Control or comparison:
  Selected complete BFP8 layer and isolated projection-only measurements.
  Likely subsystem:
  Expert layout/topology and performance accounting.
  Investigation performed:
  Legal eight-bank BFP8/BFP4 projection probe with synthetic tensors.
  Resolution:
  more-work-needed; the actual compatible full chain was not measured.

- Observed anomaly:
  Initial authentic b32 matrices produced non-finite prefill and corrupted
  decode lanes.
  Evidence:
  `authentic_bfp{8,4}_matrix.log`, attention/RoPE probe logs, and the diff from
  `03b1b0078f1` to `0085f30d237`.
  Affected path:
  Non-aligned multi-user prefill and b32 RoPE decode.
  Control or comparison:
  Attention-only probes, exact row-wise core order, and the clean eight-row
  matrices after repair.
  Likely subsystem:
  Large-M QKV/O program selection and decode sharded-core ordering.
  Investigation performed:
  Attention isolation, chunk-size probes, unrotated Q/K controls, cache
  controls, and corrected reruns.
  Resolution:
  fixed for correctness; current 30-test normal/watcher suites and both clean
  authentic matrices pass. Performance-profile closure remains
  more-work-needed because all selected profiles predate the fixes.

- Observed anomaly:
  Watcher-only final suite emitted platform motherboard warnings.
  Evidence:
  `artifacts/final_after_review2_watcher.log`.
  Affected path:
  Platform discovery only.
  Control or comparison:
  `watcher/final_after_review2/generated/watcher/watcher.log` has no
  fatal/assert/invalid-NoC/CB-bounds/overflow/sanitizer/timeout/hang/tripped or
  kernel-error signature, and all 30 tests pass.
  Likely subsystem:
  Host platform metadata.
  Investigation performed:
  Case-insensitive signature scan of the 1,092-line watcher log and JUnit
  parse.
  Resolution:
  controlled; unrelated to model correctness.

## Scope Inspected

- Goal/skill paths:
  Original functional checkpoint `78dbd88bec7`, optimized checkpoints
  `03b1b0078f1`, `0085f30d237`, and `c7e024e8faa`;
  `.agents/skills/optimize/SKILL.md`;
  `.agents/skills/stage-review/SKILL.md`;
  prior `STAGE_REVIEW_1.md` and `STAGE_REVIEW_2.md`.
- Artifact paths:
  `README.md`, `work_log.md`, `AUTODEBUG.md`, `AUTOFIX.md`,
  `AUTOTRIAGE.md`, all candidate JSON, all selected raw/filtered/summary
  profiler trees, correctness/watcher JUnit and logs, authentic BFP8/BFP4
  matrices, synthetic rejection evidence, DRAM-sharded probe evidence, twelve
  final after-review2 timing JSON files, watcher output, and context contract
  artifacts.
- Code paths:
  `tt/optimized_decoder.py`,
  `tests/test_optimized_decoder.py`,
  `tests/optimized_decoder_perf.py`, and
  `tests/optimized_decoder_capacity.py`, compared with the functional
  checkpoint and implementation.
- Commands run:
  Read-only `git status/log/show/diff`, `find`, `rg`, `sed`, `nl`, `stat`,
  CSV/JSON/XML parsing and consistency scripts, AST parsing, profiler-row
  aggregation, and watcher signature scans. No hardware was opened, no tests
  or servers were launched, and no implementation/test/evidence file was
  modified.

## Residual Risk

- The current repaired BFP8 default has strong correctness evidence: the
  direct optimized suite passes 30/30 normally and under watcher, both clean
  authentic matrices cover all eight requested rows, paged/non-aligned/trace
  checks pass, the 12 current wall-latency rows preserve b1 improvements and
  avoid b32 regression, and the 500,000-token b32 decode/cache probe passes.
- Those strengths do not close the stage because the selected expert precision
  contradicts the real-target result, a reported primary prefill topology is
  not the selected path, the DRAM-sharded family was not measured end to end,
  and required final profiler/context evidence is stale or absent.
- The live worktree was otherwise clean for this model at review start; the
  unrelated untracked `tt_metal/third_party/tt-cluster-descriptors/` directory
  was not inspected or modified.
