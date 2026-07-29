# Stage Review

Verdict: more-work-needed

Reviewed commit: `c1b26703d8536c909b2c25e0c6512fa93c5d2051`

## Required Work

- P1: Complete mandatory OPT-015 after the external shard-advisor environment
  is installed.
  Evidence:
  The optimize skill requires `$shard-advise` to run this pass on the rewritten
  dense attention+MLP block and requires retained `report.json` and
  authoritative `final_ir.mlir`, application of the emitted layouts/programs
  as first candidates, and measured comparison with the selected path. At the
  reviewed commit, `doc/optimized_decoder/shard_advise/` contains only
  `bootstrap.txt`; neither required output exists and no compiler-emitted
  candidate has been applied or measured.

  This is an externally blocked prerequisite, not a discovered decoder-code
  defect. `tests/advise_optimized_decoder.py` is a model-owned capture target
  for dense layer 0 at batch 32 or batch 1. `AUTOFIX.md` records the exact
  fresh-process bootstrap command and its exit-1 result. The retained output
  says `ttnn-advise` is absent, and read-only inspection confirms that neither
  `ttnn-advise` nor `ttnn_jit` is installed in the visible environment. The
  visible tt-mlir checkout is branch
  `mvasiljevic/5738-distributed-rmsnorm-rulebook` at `21c1b3bc4a81`, not the
  shard-advisor setup's required
  `mvasiljevic/shard-advisor-dram-sharding` commit `618cd4e75d`.
  The shard-advise setup contract explicitly classifies building the pinned
  environment as one-time operator setup and says not to build tt-mlir inside
  the model experiment. AutoFix therefore exhausted the in-scope action
  without fabricating advice.

  Why this matters:
  OPT-015 is explicitly mandatory. The extensive manual DRAM-sharded and
  residual-layout sweeps do not substitute for the required compiler seed.
  Without `final_ir.mlir`, there is no defensible way to invent the per-op
  input layouts, output layouts, reverts, or 1-D multicast programs.

  Required next step:
  Have an operator install the pinned advisor environment with
  `TTMLIR_ENABLE_OPMODEL=ON` and `TTMLIR_ENABLE_TTNN_JIT=ON`. Run the prepared
  capture in separate fresh processes at batch 1 and batch 32. Retain
  `report.json`, `final_ir.mlir`, `report.txt`, and `pipeline.log`; apply the
  emitted dense-path choices as candidates, including authoritative boundary
  reverts and register-budget subblock clamping; compare PCC and traced warmed
  decode against the selected DRAM-sharded default; then rerun affected
  correctness, performance, profiler, watcher, context, and stage-review
  gates.

- P2: Complete the local checkpoint SHA ledger.
  Evidence:
  `work_log.md` lists model-isolated checkpoints only through
  `a53eebe040d`, while the reviewed stage range also contains
  `770f70051f9` (`Record isolated North Mini stage history`),
  `74c95ddaf4f` (`Point North Mini docs at final review suite`), and
  `c1b26703d85` (`Fix North Mini final prefill geometry`). The range is
  locally committed and model-scoped, and there is no evidence of a push, but
  the user's explicit “log SHAs” gate is not complete at this HEAD.

  Why this matters:
  The final stage handoff must identify the exact committed implementation and
  evidence state. An incomplete ledger makes the recorded checkpoint
  ambiguous even though Git history itself is clean.

  Required next step:
  After the remaining remediation and final review, append every stage-owned
  checkpoint SHA, including the final review/evidence checkpoint, to
  `work_log.md`. Commit the stage-owned documentation locally and do not push.

## Other Concerns

- The live worktree contains untracked raw Tracy `.logs/` directories and
  `tracy_profile_log_host.tracy` files under the two review-5 profile
  directories, plus unrelated
  `tt_metal/third_party/tt-cluster-descriptors/`. None is part of the reviewed
  commit. The committed evidence already contains compact compressed raw CSV,
  filtered tables, human reports, runtime JSON, and summaries, so the omitted
  large profiler working files are not an evidence gap. Keep them out of the
  eventual stage commit and preserve the unrelated directory.
- `git diff --check 78dbd88bec7..c1b26703d85` reports CRLF/trailing whitespace
  in generated profiler CSVs. This is artifact hygiene, not a decoder
  correctness or performance issue.

## Hard-Check Gaps

- `shard_advise/report.json` and `shard_advise/final_ir.mlir` do not exist;
  consequently there is no compiler-seeded batch-1/batch-32 candidate or
  measured keep/reject result.
- The work-log SHA list does not yet identify all commits in the exact reviewed
  stage range.
- Per the review contract, no TT device, model test, profiler, watcher, server,
  reset, or vLLM command was run during this independent review.

## Review-5 Closure Validation

The batch-32 BFP4/LoFi dense-expert prefill finding from
`STAGE_REVIEW_5.md` is fixed.

- The implementation now has prefill-specific
  `MatmulMultiCoreReuseMultiCastProgramConfig` fields. Defaults select packed
  gate/up on a 10x8 grid with `in0_block_w=8`, `per_core_M/N=4/5`, and
  subblock `1x5`, plus a 10x8 down program with `in0_block_w=6`,
  `per_core_M/N=4/7`, and subblock `1x7`. `prefill_packed_dense_experts=True`
  and `packed_dense_experts=False` keep the selection prefill-only; decode
  still uses the separately swept split/automatic family.
- The runtime dispatch passes `phase` into `_dense_expert_moe_chunk`, selects
  the phase-specific programs there, and uses packed weights only for prefill.
  The selected optimized methods remain overridden; the source audit forbids
  functional math fallback and runtime Torch conversions.
- The retained same-policy sweep is coherent:

  | Candidate | Layer / phase | Mean | Decision |
  |---|---|---:|---|
  | previous selected automatic split | layer 1 prefill | 139.959 ms | replaced |
  | one-BFP4-residency control | layer 1 prefill | 140.177 ms | refutes residency as latency cause |
  | split 64/80, `in0_block_w=8/6` | layer 1 prefill | 100.909 ms | slower |
  | split 88/88, `in0_block_w=8/6` | layer 1 prefill | 107.555 ms | slower |
  | packed 80/80, `in0_block_w=8/6` | layer 1 prefill | 96.844 ms | selected |
  | packed 80/80 control | layer 4 prefill | 96.644 ms | selected |
  | plain promoted default | layer 1 / 4 prefill | 96.750 / 96.440 ms | reproduced |
  | plain promoted default | layer 1 / 4 traced decode | 2.214 / 2.219 ms | no regression |

  Candidate and final-default JSON agree on BFP4 expert weights, LoFi expert
  compute, batch 32, sequence 128, three warmups, twenty samples, and the
  selected geometry. The profiler rows independently show BF16 x BFP4 LoFi
  packed/down matmuls on 80 cores with `in0_block_w=8/6` and subblocks
  `1x5/1x7`.
- The stale 117.903-ms artifact is no longer treated as a best-correct
  candidate. It has only three samples, no PCC, no source revision, and was
  produced before the committed correction of the batch-32 QKV construction.
  The final-code one-residency control remains about 140 ms, while legal
  prefill-specific geometry and packing produce a reproducible sub-97-ms
  winner.
- Authentic target-weight sequence-33 batch-32 prefill passes at PCC
  0.99923857 / 0.99993403 for layers 1 / 4. The final combined suite records
  54 tests, 0 failures, 0 errors, and 16 opt-in candidate skips: 38 selected
  and static-path checks pass.
- The selected packed prefill rows pass 2/2 in a separate
  `TT_METAL_WATCHER=10` run. The committed 2,170-line watcher log has no
  fatal, invalid-NoC, CB-bounds, overflow, sanitizer, timeout, hang, tripped,
  assertion, or kernel-error signature.
- Fresh review-5 profiler evidence is separate from watcher evidence. It
  records 231 / 229 device ops, zero host ops, 95.720 / 94.711 ms device time,
  and 97.439 / 96.458 ms wall time for layer 1 / 4. The only
  tilize/untilize rows are the small, documented internals of TTNN scatter's
  row-major routing-mask contract; there is no Torch, `from_torch`,
  `to_torch`, or host fallback in the measured window.
- The additional packed family is accounted as 226,492,416 bytes. Final
  layer-1 and layer-4 batch-32 construction plus traced decode at context
  500,000 and position 499,999 are finite with the 32.768-GB BF16 cache.
  `doc/context_contract.json` retains the advertised 500,000 context with no
  capability reduction and accurately leaves completion `in-progress`.

## Anomaly Ledger

- Observed anomaly:
  The mandatory compiler seed is still absent after AutoFix.
  Evidence:
  `shard_advise/bootstrap.txt`, `AUTOFIX.md`, absence of `report.json` and
  `final_ir.mlir`, current PATH/package search, and the visible tt-mlir
  checkout's branch/commit.
  Affected path:
  Dense attention+MLP L1 layout skeleton and 1-D multicast comparison at
  batches 1 and 32.
  Control or comparison:
  The selected manual 16-core residual chain and DRAM-sharded matmul sweeps.
  Likely subsystem:
  External machine/operator toolchain setup, not the model runtime.
  Investigation performed:
  Read the shard-advise skill and setup recipe, inspected the prepared capture
  target and exact bootstrap evidence, searched the visible toolchain, and
  verified the visible tt-mlir revision.
  Resolution:
  more-work-needed; blocked before capture on operator-installed tooling.

- Observed anomaly:
  The prior retained 117.903-ms prefill result appeared faster than the old
  final default.
  Evidence:
  `dense_expert_chunk1024_prefill_b32.json`,
  `review5_single_bfp4_residency.json`, all
  `candidates/review5_prefill/*.json`, and the final selected profiles.
  Affected path:
  Batch-32 sequence-128 MoE prefill, layers 1 and 4.
  Control or comparison:
  A final-code one-residency control, three legal same-policy geometry
  families, authentic PCC, plain-default reproduction, and fresh profiles.
  Likely subsystem:
  Stale pre-commit batch-32 attention construction plus untuned M=1024 expert
  geometry.
  Investigation performed:
  Compared provenance and policy fields, verified phase-specific runtime
  wiring, parsed every review-5 candidate, inspected JUnit, and checked the
  profiler's dtype/fidelity/program rows.
  Resolution:
  fixed. The final default is both faster than the stale number and supported
  by correctness and profiler evidence.

- Observed anomaly:
  Packed prefill adds 216 MiB of persistent expert weights, and full-attention
  layer 4 is much slower than sliding-attention layer 1 at the advertised
  decode position.
  Evidence:
  `PREFILL_GEOMETRY_AUTOFIX.md`,
  `context500000_decode_b32_layer{1,4}_review5.json`, and
  `doc/context_contract.json`.
  Affected path:
  Batch-32 context-500,000 construction and decode.
  Control or comparison:
  Both MoE layer kinds allocate the same 32.768-GB cache and final resident
  expert set; layer 1 uses its 4,096-token sliding window while layer 4 attends
  the full context.
  Likely subsystem:
  Expected attention-window cost and persistent-memory capacity.
  Investigation performed:
  Recomputed the cache and expert-byte accounting, checked both JSON results
  for finite output and traced replay, and traced the layer-kind contracts.
  Resolution:
  controlled. Both pass; no advertised-capability reduction is needed.

- Observed anomaly:
  The live tree contains raw, untracked profiler working files after the
  compact profile artifacts were committed.
  Evidence:
  `git status --short` and `git ls-files` for both review-5 profile roots.
  Affected path:
  Repository hygiene only.
  Control or comparison:
  Committed gzip raw CSV, filtered CSV, human report, runtime JSON, summary
  CSV, and PNG for each selected row.
  Likely subsystem:
  Tracy working-directory output.
  Investigation performed:
  Distinguished tracked compact evidence from untracked raw working files and
  audited the exact commit range separately.
  Resolution:
  controlled; do not add the raw working files to the stage checkpoint.

## Scope Inspected

- Goal/skill paths:
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/optimize/SKILL.md`,
  `.agents/skills/tt-device-usage/SKILL.md`,
  `.agents/skills/shard-advise/{SKILL.md,SETUP.md}`, functional checkpoint
  `78dbd88bec7`, and reviewed HEAD `c1b26703d85`.
- Artifact paths:
  `README.md`, complete `work_log.md`, `STAGE_REVIEW_5.md`, `AUTOFIX.md`,
  `PREFILL_GEOMETRY_AUTOFIX.md`, context contract and capacity JSON, all
  review-5 candidate JSON, three review-5 JUnit files, both fresh selected
  profiler trees, the focused watcher log, and prior final runtime/profile
  evidence needed for comparison.
- Code paths:
  Complete `tt/optimized_decoder.py`,
  `tests/test_optimized_decoder.py`,
  `tests/test_optimized_decoder_prefill_geometry.py`,
  `tests/optimized_decoder_perf.py`,
  `tests/optimized_decoder_capacity.py`, and
  `tests/advise_optimized_decoder.py`.
- History:
  All eight commits in `78dbd88bec7..c1b26703d85`. Every changed path is
  under `models/autoports/coherelabs_north_mini_code_1_0/`; no `.agents`,
  multichip, full-model, vLLM, or unrelated implementation path is in the
  reviewed range.
- Commands run:
  Read-only `git status/log/show/diff/ls-files/check-ignore`, `rg`, `sed`,
  `nl`, `find`, JSON/XML/gzip parsing, JUnit and candidate aggregation,
  profiler-row/advice inspection, watcher-signature scanning, and tt-mlir
  revision/tool availability checks. No hardware-facing command was run.

## Residual Risk

- Apart from OPT-015, the delivered optimized decoder evidence is strong and
  internally consistent: selected paths are branch-proven optimized methods,
  functional fallback is forbidden, representative layer kinds and batches
  pass, arbitrary logical lengths and paged cache semantics are exercised,
  trace replay is deterministic, batch-1 remains the primary winning decode
  target, batch 32 does not regress, final dtype/fidelity reaches profiler
  rows, and watcher/profile runs are separated.
- The review-5 code change adds a packed weight allocation at construction but
  does not change batch-1 MoE execution: packed use is prefill-only at the
  dense serving threshold, and the new allocation occurs after the selected
  sparse/dense split weights. The unchanged batch-1 runtime rows therefore
  remain applicable; worst-case batch-32 capacity was explicitly revalidated.
- A compiler-seeded candidate could still reveal a better dense residual
  skeleton or a profitable 1-D multicast row. That uncertainty is precisely
  why OPT-015 remains a gate and why this review cannot return `clean-pass`.
