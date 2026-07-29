# Stage Review

Verdict: more-work-needed

Reviewed commit: `8bd1bf9e318b8f255591b01e63b612b0672ed62f`

## Required Work

- P1: Complete mandatory OPT-015 after the pinned shard-advisor environment
  is installed externally.

  Evidence:
  The optimize skill requires `$shard-advise` to run during this pass on the
  rewritten dense attention+MLP block. It requires retained `report.json` and
  authoritative `final_ir.mlir`, application of the emitted L1
  layouts/program configs as first candidates, and measured comparison with
  the selected DRAM-sharded path. At the reviewed commit,
  `doc/optimized_decoder/shard_advise/` still contains only
  `bootstrap.txt`; the two mandatory outputs and a compiler-emitted candidate
  comparison do not exist.

  This is an external prerequisite, not a North-Mini decoder defect.
  `tests/advise_optimized_decoder.py` is a prepared model-owned capture target
  for dense layer 0 at batch 32 and supports a separate fresh batch-1
  capture. The retained bootstrap result says `ttnn-advise` is absent.
  Read-only inspection again finds no `ttnn-advise` executable and no
  importable `ttnn_jit` package. The visible tt-mlir checkout is branch
  `mvasiljevic/5738-distributed-rmsnorm-rulebook` at
  `21c1b3bc4a81cba1642c170fb08ef0b048040a8a`, not the setup contract's pinned
  `mvasiljevic/shard-advisor-dram-sharding` commit `618cd4e75d`. The
  shard-advise setup instructions classify building that pinned environment
  as one-time operator setup and explicitly prohibit building it inside a
  model experiment. `AUTOFIX.md` records that AutoFix exhausted the in-scope
  bootstrap/capture work without inventing advice.

  Why this matters:
  OPT-015 is explicitly mandatory. Manual L1/DRAM-sharded sweeps cannot
  substitute for the compiler seed, and without `final_ir.mlir` there is no
  defensible way to infer the advisor's per-op input/output layouts, boundary
  reverts, and 1-D multicast programs.

  Required next step:
  An operator must install the pinned advisor build with
  `TTMLIR_ENABLE_OPMODEL=ON` and `TTMLIR_ENABLE_TTNN_JIT=ON`. Then run the
  prepared capture in separate fresh processes at batch 1 and batch 32,
  retain `report.json`, `final_ir.mlir`, `report.txt`, and `pipeline.log`,
  apply the emitted dense-path choices as first candidates, and compare PCC
  plus traced warmed decode with the selected DRAM-sharded default. Rerun the
  affected correctness, performance, profiler, watcher, context, and
  independent-review gates after deciding which compiler candidates to keep.

## Closed Findings

- `STAGE_REVIEW_6.md` P2 is closed.

  The work log now records the formerly omitted model-isolated checkpoints
  `770f70051f9`, `74c95ddaf4f`, `c1b26703d85`, and the committed review-6
  evidence checkpoint `1774f50bf8c`; its earlier ledger already records
  `f77d4e00940`, `2a9f76b6e29`, `c709816ba57`, `a9219e58d4e`, and
  `a53eebe040d`. Commit `8bd1bf9e318` is itself the local ledger repair, so it
  cannot contain its own content-derived SHA without creating an
  uncloseable self-reference; this review anchors that exact HEAD. The branch
  has no configured upstream in the inspected branch listing, and the work
  log records that no commit was pushed.

- The review-5 batch-32 BFP4/LoFi prefill finding remains closed.

  The optimized source, affected tests, review-5 candidates, JUnit artifacts,
  selected Tracy evidence, and focused watcher log are unchanged from
  `c1b26703d85` through the reviewed HEAD. Defaults still select packed
  gate/up prefill on a 10x8 grid with `in0_block_w=8`,
  `per_core_M/N=4/5`, and subblock `1x5`, plus a 10x8 down projection with
  `in0_block_w=6`, `per_core_M/N=4/7`, and subblock `1x7`.
  `prefill_packed_dense_experts=True` and `packed_dense_experts=False`
  preserve phase separation: batch-32 prefill uses the selected packed
  topology while decode remains on its separately swept split/automatic
  path.

  The retained final-default runs reproduce 96.750/96.440 ms for layer-1/4
  batch-32 prefill and 2.214/2.219 ms for traced decode. The final combined
  JUnit has 54 cases, zero failures/errors, 38 selected/static passes, and 16
  opt-in DRAM-candidate skips. The authentic packed-prefill gate and the
  separate `TT_METAL_WATCHER=10` gate each pass both layer kinds. The
  2,170-line focused watcher log has no fatal, invalid-NoC, CB-bounds,
  overflow, sanitizer, timeout, hang, tripped, assertion, or kernel-error
  signature.

  Fresh selected profiles remain consistent with the promoted default:
  layer-1/4 profile wall times are 97.439/96.458 ms, device times are
  95.720/94.711 ms, and the dominant expert rows are BF16 x BFP4 with LoFi
  compute on 80 cores. Their advice tables show the selected gate/up
  `in0_block_w=8`, subblock `1x5`, and down `in0_block_w=6`, subblock `1x7`
  as good. The measured windows contain zero host ops and no Torch,
  `from_torch`, `to_torch`, or host fallback.

- Exact stage scope and context capability remain correct.

  All ten commits in `78dbd88bec7..8bd1bf9e318` descend from the functional
  checkpoint. Every changed path is under
  `models/autoports/coherelabs_north_mini_code_1_0/` and is an
  optimized-decoder implementation, test, or documentation artifact. The
  range contains no `.agents`, functional-decoder implementation,
  multichip, full-model, generator, LM-head, sampling, or vLLM change.

  `doc/context_contract.json` parses, every referenced artifact exists, and
  it accurately remains `in-progress` with OPT-015 as its sole pending gate.
  It preserves the advertised 500,000-token capability with no reduction.
  Final packed-weight capacity evidence covers both MoE layer kinds at
  batch 32 and context 500,000 with a 32.768-GB BF16 KV cache; construction
  and traced replay are finite at 3.307/132.660 ms. The contract retains
  non-aligned prefill, MoE chunk-boundary, sliding-window-boundary, paged
  cache, and batch 1/4/32 coverage.

## Other Concerns

- None beyond the external OPT-015 prerequisite. The only live untracked
  worktree item is
  `tt_metal/third_party/tt-cluster-descriptors/`, which is unrelated and not
  part of the reviewed commit.

## Hard-Check Gaps

- `shard_advise/report.json` and `shard_advise/final_ir.mlir` are absent, so
  no compiler-seeded batch-1/batch-32 candidate has been applied or measured.
- Per the review contract, no TT device, test, profiler, watcher, server,
  reset, or vLLM command was run during this independent rereview.

## Anomaly Ledger

- Observed anomaly:
  The mandatory compiler seed is still absent after AutoFix.
  Evidence:
  `shard_advise/bootstrap.txt`, `AUTOFIX.md`, absence of `report.json` and
  `final_ir.mlir`, current CLI/package availability, and the visible tt-mlir
  branch/revision.
  Affected path:
  Dense attention+MLP L1 layout skeleton and 1-D multicast comparison at
  batches 1 and 32.
  Control or comparison:
  The selected manual residual-layout and DRAM-sharded matmul sweeps.
  Likely subsystem:
  External machine/operator toolchain setup.
  Investigation performed:
  Read the optimize, shard-advise, and setup contracts; inspected the prepared
  capture and retained failure; checked the visible executable, Python
  package, and tt-mlir revision without running capture or opening hardware.
  Resolution:
  more-work-needed; blocked before model capture on a missing external
  prerequisite, not on a model-code failure.

- Observed anomaly:
  Review 6 found an incomplete checkpoint ledger.
  Evidence:
  The `8bd1bf9e318` work-log diff and the complete ten-commit history from
  functional checkpoint `78dbd88bec7`.
  Affected path:
  Stage handoff provenance only.
  Control or comparison:
  Every pre-ledger stage-owned commit is now named in the work log, and this
  report identifies the exact ledger commit under review.
  Likely subsystem:
  Documentation bookkeeping.
  Investigation performed:
  Compared the work-log SHA entries with every commit in the reviewed range
  and audited every changed path.
  Resolution:
  fixed.

- Observed anomaly:
  The old 117.903-ms prefill artifact appeared faster than the prior selected
  default.
  Evidence:
  The stale artifact's missing provenance, the final-code one-residency
  control at 140.177 ms, the legal same-policy geometry matrix, final-default
  reproduction, authentic selected-path tests, and fresh profiles.
  Affected path:
  Batch-32 sequence-128 MoE prefill, layers 1 and 4.
  Control or comparison:
  Split 64/80 and 88/88 candidates versus selected packed 80/80 under
  BFP4/LoFi, plus unchanged traced decode.
  Likely subsystem:
  Stale pre-commit batch-32 QKV construction and untuned expert geometry.
  Investigation performed:
  Rechecked code phase dispatch, candidate policy fields, JUnit gates,
  profiler dtype/fidelity/program rows, and unchanged-artifact history.
  Resolution:
  fixed; the promoted default is reproducibly faster and preserves
  correctness and decode performance.

## Scope Inspected

- Goal/skill paths:
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/optimize/SKILL.md`,
  `.agents/skills/tt-device-usage/SKILL.md`,
  `.agents/skills/shard-advise/{SKILL.md,SETUP.md}`, functional checkpoint
  `78dbd88bec7`, and reviewed HEAD `8bd1bf9e318`.
- Artifact paths:
  `README.md`, complete `work_log.md`, `STAGE_REVIEW_5.md`,
  `STAGE_REVIEW_6.md`, `AUTOFIX.md`, `PREFILL_GEOMETRY_AUTOFIX.md`,
  `doc/context_contract.json`, all review-5 final/candidate JSON, review-5
  JUnit artifacts, both selected review-5 profile trees, focused watcher log,
  and final context-capacity JSON.
- Code paths:
  `tt/optimized_decoder.py`,
  `tests/test_optimized_decoder.py`,
  `tests/test_optimized_decoder_prefill_geometry.py`,
  `tests/optimized_decoder_perf.py`,
  `tests/optimized_decoder_capacity.py`, and
  `tests/advise_optimized_decoder.py`.
- History:
  All ten commits in `78dbd88bec7..8bd1bf9e318`; the focused review-5
  implementation/evidence paths are unchanged in
  `c1b26703d85..8bd1bf9e318`.
- Commands run:
  Read-only `git status/log/show/diff/branch/rev-list`, `rg`, `sed`, `find`,
  `sha256sum`, JSON/JUnit parsing, candidate aggregation, profiler/advice
  inspection, watcher-signature scanning, artifact-reference validation,
  executable/package discovery, and tt-mlir revision inspection. No
  hardware-facing command was run.

## Residual Risk

- No code or evidence defect remains from the review-5 prefill closure, and
  no new model defect was found in this focused rereview.
- A compiler seed could still expose a better dense residual skeleton or
  profitable 1-D multicast program. That unresolved optimization uncertainty
  is exactly the mandatory OPT-015 gate. Until the external advisor is
  installed and its candidates are measured, the user requirement for a
  clean stage-review pass is not met.
