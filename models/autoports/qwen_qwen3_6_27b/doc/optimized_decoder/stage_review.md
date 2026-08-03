# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- The 169/169 combined static tests pass, but both TTNN-importing pytest
  processes emit nanobind reference-leak diagnostics during interpreter
  shutdown. The diagnostics do not change either process's zero exit status
  and do not contradict any optimized-decoder result.

## Hard-Check Gaps

- Full-attention long prefill retains output/cache population evidence at
  S=32769 and S=192511 rather than numerical PCC across two query chunks.
  Source inspection verifies page-aligned K/V fill, global page-table use,
  `chunk_start_idx`, tail padding/slicing, and output concatenation. The
  retained S=192511 capacity result reaches the inherited single-pass contract;
  this is residual risk rather than missing required stage evidence.
- Watcher result JSON proves successful ten-step B32 completion with
  `watcher_enabled=true` and all nine replay PCC checks passing, but the full
  watcher consoles are not retained.
- Raw Tracy reports were removed after filtering. All 78 retained
  `profile_run.json` records have zero Tracy and `tt-perf-report` exit status,
  and each has nonempty `perf.csv`, `summary.csv`, and `summary.png`.

## Anomaly Ledger

- Observed anomaly:
  Two structured subblock-contract paths omitted the live `device/` directory.

  Evidence:
  `program_contracts.json` now cites
  `device/config/matmul_program_config_types.hpp:71-76` and
  `device/config/matmul_program_config.cpp:946-982`. Both paths and ranges
  exist. The first range exposes exactly `in0_block_w`, `per_core_M`,
  `per_core_N`, and `fused_activation`; the second contains
  `get_matmul_subblock_params` and its internal subblock selector.

  Affected path:
  Exact provenance for the DRAM-sharded matmul output-subblock contract.

  Control or comparison:
  The nanobind citation exposes the same four constructor fields, the factory
  citation calls the internal selector and widens its result, and the static
  negative-API/exact-lowering tests pass.

  Likely subsystem:
  Structured evidence documentation.

  Investigation performed:
  Parsed the JSON; resolved all seven source ranges plus the named static test;
  checked bounds and expected tokens in live source. The eight contract
  citations all resolve. Across the four hand-written stage documents, 71
  additional code references resolve; together with the seven structured
  source ranges, all 78 source citations resolve with valid bounds.

  Resolution:
  fixed.

- Observed anomaly:
  The initial optimized full-attention result was visibly wrong on official
  weights, while the diagonal synthetic fixture passed.

  Evidence:
  Retained A/B evidence isolates a width-sharded QKV-head consumer error
  (`V`-head PCC -0.020282 before the L1-interleaved boundary and 0.999837
  after) and the HF per-head q/gate packing convention. The durable final
  official-weight artifacts report HF PCC 0.997612 at B1 and 0.998095 at B32.

  Affected path:
  Packed full-attention Q/K/V/gate projection and head creation.

  Control or comparison:
  Direct HF layer, packed projection/split probes, and the functional-stage
  control.

  Likely subsystem:
  Weight packing and QKV helper layout.

  Investigation performed:
  Inspected `AUTODEBUG.md`, `AUTOFIX.md`, the real-weight runner, current
  packing code, narrow interleaved boundary, and final real/traced artifacts.

  Resolution:
  fixed.

- Observed anomaly:
  Earlier linear-attention signoff had not crossed the selected BFP4/LoFi
  projection policy with material geometry.

  Evidence:
  The final matrix has paired B1/B32 rows for packed-input widths
  1/4/5/10/20, output widths 1/2/3/4/6/8/12/24, three cumulative width-5
  crosses, and a four-core control. Widths 10/20 and the four-core family
  retain exact L1/CB failures. Every passing contender has compact profiler
  evidence.

  Affected path:
  Linear packed-input and output DRAM-sharded projections.

  Control or comparison:
  The BFP4/LoFi width-2/3 precision baseline and all legal isolated/cumulative
  geometry contenders at both batches.

  Likely subsystem:
  DRAM-sharded matmul block geometry.

  Investigation performed:
  Re-derived policy inheritance from source, checked the candidate pairs and
  evidence links, and read final profiler rows. They show
  `LoFi BF16 x BFP4 => BF16`, DRAM sharding, and selected widths 5 and 12.
  The selected contender reports 1.521726/15.890707 ms device time; the final
  default reproduces 1.521271/15.893577 ms.

  Resolution:
  fixed.

## Scope Inspected

- Goal/skill paths:
  `.agents/prompts/model_bringup_multigoal/02-optimized-decoder.txt`;
  `.agents/skills/{optimize,tt-device-usage,stage-review}/SKILL.md`.
- Repository state:
  branch `skillexp-work-qwen36`; HEAD
  `ebfec4116792e85ec6eb6cf722ceab62903cffc9`; live dirty worktree. Review
  remained read-only except for this report.
- Artifact paths:
  complete `doc/optimized_decoder/{README.md,work_log.md,AUTODEBUG.md,
  AUTOFIX.md}` and `doc/context_contract.json`; all 165 candidate JSON files;
  all 114 candidate-matrix rows and 214 matrix evidence links;
  `program_contracts.json`; `signoff_manifest.md`; all 78 profile provenance
  records and all 100 retained compact profile directories; final official
  PCC, transition, trace, determinism, watcher, and capacity artifacts.
- Code paths:
  complete `tt/optimized_decoder.py`; relevant functional loader, cache,
  attention, affine-scan, prefill, and decode paths; all optimized-stage tests
  and evidence runners; cited matmul config, binding, factory, validator, and
  selector source.
- Commands run:
  branch/HEAD/status inspection; read-only `rg`, `find`, `sed`, `nl`, `jq`,
  JSON/CSV analysis, matrix-reference resolution, source-citation resolution,
  profiler dtype/fidelity/geometry and device-time accounting; 165 optimized
  static tests plus four inherited functional/context tests (169/169 passed);
  `py_compile`; Black check; scoped `git diff --check`; parser `--help`.
  No TT device, reset, watcher, profiler, server, or other hardware-facing
  command was run.

## Residual Risk

- Official-weight linear transition/decode evidence is B1; B32 linear
  correctness uses synthetic full-shape inputs. Full attention has
  official-weight B1 and B32 evidence.
- Linear prefill intentionally calls the proven functional affine-scan method
  on the optimized object. It constructs no `FunctionalDecoder` fallback and
  the final prefill profiles include the selected BFP8 physical-state
  conversions.
- The final defaults reproduce the selected correct candidates, beat the
  functional B1 baseline, improve B32, preserve the 262144 context contract,
  pass non-aligned prefill, paged/stateful transitions, trace replay,
  determinism, fallback audit, and watcher stress, and match the retained
  runtime dtype/fidelity/program rows.
- Per the stage workflow, the stage owner should now include this clean review
  in the isolated local checkpoint commit and record that SHA; the review
  itself does not push or create the checkpoint.
