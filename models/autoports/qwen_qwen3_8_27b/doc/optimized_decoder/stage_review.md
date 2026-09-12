# Stage Review

Verdict: clean-pass

Independent review of stage 3, optimized-decoder, for Qwen/Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, starting from fused checkpoint
`ad43d1388fd610fb14356b428593bc806a403fcd`. The worktree was live during review.
Final reviewed decoder SHA256:
`4797849af321c854084a998f4d5ce4bd6c36dd442a6886efef581391efcffd26`.

## Required Work

None. Findings raised during this review were resolved and their replacement
evidence was inspected before this verdict. Local stage commits and recording
their SHAs remain the stage owner's post-review handoff; no push is authorized.

## Other Concerns

- The final decoder is independent of the functional and fused implementations.
  Its setup boundary uploads weights/constants; forward helpers use TTNN and
  shape orchestration. The tests reject either older constructor and guard
  Torch operations and host tensor conversions during forward execution.
- The final `selected_stress` artifacts contain 68 cases across 12 pytest groups,
  with zero failures, errors, or skips. Every case records exactly
  `final_policy.json`, real checkpoint weights, and recorded HF inputs. Coverage
  includes both layer kinds, batches 1/2/3/8/16/32, nonaligned lengths through
  4097, continuation, per-user PCC, deterministic replay, changed inputs and
  positions, sequential state-consuming replay, and full-attention page swaps.
  Minimum prefill/decode PCC is 0.9995203/0.9993832; minimum changed-input PCC is
  0.9992942. All exceed 0.995.
- Full-capacity output parity is 0.9997951 linear and 0.9996872 full at 262144
  tokens. Last-valid-position traced decode parity is 0.9998990/0.9999097, with
  bitwise repeats. The context contract retains 262144 and records batch-32
  coverage separately. The last source change affects only batch-one full
  decode with fewer than 16 mapped pages; it cannot change either capacity
  probe's large-context branch or the linear branch.
- Final warmed prefill/traced decode medians are 2.260554/0.822062 ms linear
  and 1.783167/0.661752 ms full, from `verified_benchmark_l{0,3}.json`.
  Same-harness full default/control/default decode is
  0.661752/0.663265/0.661923 ms. The final default therefore beats the reproduced
  strongest prior correct control. The README appropriately distinguishes
  these small differences from timing spread and uses final reproduced values.
- Both final decode profiles show the intended BF16 activation/output,
  BFP4 weight, LoFi policy for all five dominant projections. Raw attributes
  confirm DRAM-sharded weights, L1-sharded inputs/outputs, three readers per bank,
  `per_core_M=1`, and role block widths 2/4/2/2/17. The 300-row projection search
  uses recorded real HF projection inputs and crosses BFP4/LoFi with material
  core/shard/reader/block geometries, including wider shards and large legal
  divisors. Whole-decoder evidence, rather than its quantized arithmetic
  controls, decides precision acceptance.
- The operation-topology audit and measured packed/separate projection,
  residual-carry, sharded-norm, DRAM-matmul, explicit-2D, minimal-matmul,
  L1-prefill, and trace-wait alternatives are present. Initial API/allocation
  failures were followed by adapted legal attempts or exact native-contract
  evidence. The selected DRAM-sharded matmul cannot fuse untilization; native
  validation restricts that feature to Mcast1D, whose adapted controls were
  rejected earlier with PCC evidence.
- Same-run accounting reconciles independently: linear kernel/gap/host time is
  0.788721/0.037125/0.849521 ms; final full is
  0.625718/0.036491/0.685662 ms. The read rooflines are explicitly lower bounds,
  and normal benchmark medians are kept separate from instrumented host times.
  Watcher and profiler runs are separate; inspected watcher logs end with normal
  detach and report no disabled features. All final checklist items have evidence.

## Hard-Check Gaps

No missing stage gate was found. The following are the documented limits of the
evidence, rather than claims of additional coverage:

- Direct HF comparison reaches 4097 tokens. The 262143/262144 checks compare
  complete optimized and frozen fused outputs using periodically repeated
  recorded inputs; they are not a fresh full-context HF oracle.
- Full context and batch 32 are separate coverage points, not their complete
  cross-product. This is a single-device decoder stage; stacked-model accuracy,
  generation, multichip behavior, and serving are outside its contract.
- The reviewer did not execute hardware tests or import TTNN. Hardware findings
  above were re-derived from the stage owner's logs, JSON, XML, raw operation
  rows, and source. Independent local checks were read-only and host-only.

## Anomaly Ledger

- Observed anomaly: synthetic BFP4 prefill PCC was 0.993451.
  Evidence: `bfp4_mlp_l3.log`, `real_bfp4_mlp_l3.json`, final stress artifacts.
  Affected path: projection precision selection.
  Control or comparison: real checkpoint weights with recorded HF layer inputs.
  Likely subsystem: synthetic input distribution and quantization sensitivity.
  Investigation performed: recorded actual HF inputs; checked both kinds,
  continuation, cache-consuming replay, and per-user outputs at final precision.
  Resolution: controlled; the synthetic result did not veto the real-weight win.

- Observed anomaly: three-reader down matmul lacked output storage.
  Evidence: `AUTODEBUG_readers.md`, `readers3_storage_fixed.json`, raw final rows.
  Affected path: padded DRAM-bank reader output assignment.
  Control or comparison: one/two-reader controls and the original three-reader
  policy after changing its storage calculation.
  Likely subsystem: program geometry derived from logical rather than stored N.
  Investigation performed: native factory inspection and focused repair using
  bank-shard width; later role sweeps and final stress exercise three readers.
  Resolution: fixed.

- Observed anomaly: public batch rows failed packed width-shard validation.
  Evidence: `AUTODEBUG_batch.md`, `AUTOFIX_batch.md`, `batch3_fixed_l0.json`.
  Affected path: one-token residual/norm/MLP and prefill tails.
  Control or comparison: original B3 failure, B1 controls, final B2/B3/B32 cases.
  Likely subsystem: physical `[B,1,H]` versus `[1,1,B,H]` row packing.
  Investigation performed: native layout/reshape review; retained packed rows
  through the residual/MLP and interleaved before public B>1 unpacking.
  Resolution: fixed, including the optional carry-output branch guard.

- Observed anomaly: B32 linear decode collided with GDN L1 circular buffers.
  Evidence: `stress_initial_failure.log`, `AUTODEBUG_batch32_l1.md`,
  `batch32_attention_boundary.json`, final stress and watcher B32 results.
  Affected path: expanded public attention projection and native delta prep.
  Control or comparison: broad public-DRAM control followed by attention-only
  placement and B8/B16/B32 coverage.
  Likely subsystem: physical row expansion and live L1 intermediates.
  Investigation performed: allocation/shape inspection and narrowed placement.
  Resolution: fixed; batch-one L1 behavior is preserved.

- Observed anomaly: continuation alignment failed, and some larger SDPA blocks
  could read beyond logical page-table capacity despite passing masked PCC.
  Evidence: `AUTODEBUG_continuation.md`, initial failure log, final stress and
  watcher long-tail/selected cases.
  Affected path: full-attention paged prefill and decode.
  Control or comparison: minimal disjoint shuffled mappings, short and long
  tails, changed positions/page tables, and full-capacity probes.
  Likely subsystem: native SDPA start alignment and rounded read bounds.
  Investigation performed: native factory/reader inspection; adapt Q/K blocks
  to absolute alignment and mapped capacity. Independent arithmetic review
  checked 183271 representative start/end/capacity cases without target imports.
  Resolution: fixed; unsafe intermediate candidates are excluded explicitly.

- Observed anomaly: oversized profiling passed device work but failed enrichment
  with missing operation 422915.
  Evidence: `oversized_profile_l0.log`, bounded final reports, `tracy/raw_archive.json`.
  Affected path: profiler collection/postprocessing, not model correctness.
  Control or comparison: bounded signposted prefill and single traced-decode windows.
  Likely subsystem: excessive profiling volume.
  Investigation performed: retained failure/raw evidence and recollected bounded
  windows successfully; inspected archive destinations exist.
  Resolution: controlled; no performance claim uses the failed enrichment.

- Observed anomaly: report core counts understate DRAM reader workers, and
  minimal-matmul advice says no program config was supplied.
  Evidence: final raw attributes, `final_matmul_rows.csv`, explicit source configs.
  Affected path: profiler advice and utilization interpretation.
  Control or comparison: native three-readers/eight-banks configuration and
  explicit `MinimalMatmulConfig` plus measured configuration/L1 alternatives.
  Likely subsystem: report field recognition and core-count estimation.
  Investigation performed: compared raw attributes with source and table rows.
  Resolution: controlled; limitations are documented rather than treated as
  measured utilization above physical peak or untried configuration advice.

- Observed anomaly: the first wide-grid final candidate lost about 1 us to the
  strongest prior control; blocking trace replay did not consistently close it.
  Evidence: `comparison_matrix.json`, blocking cases, grid sweeps, `verified_matrix.json`.
  Affected path: short full-attention traced decode.
  Control or comparison: precision-locked grids, long-context controls, then
  final default/control/default runs.
  Likely subsystem: SDPA sequence parallelism/reduction overhead at short context.
  Investigation performed: selected 8x2 only for B1 and fewer than 16 mapped
  pages; retained wide grids where smaller grids lose or cannot cover batch.
  Resolution: fixed; final default wins the reproduced control and passes the
  full stress rerun, selected watcher, and final profile.

- Observed anomaly: empty policies in `selected_matrix` measured historical BASE
  rather than the intended default, giving approximately 1.124/0.959 ms.
  Evidence: saved effective policies in `selected_benchmark_*`, repaired sweep
  source, `verified_benchmark_*`, and `verified_repeat_l3.json`.
  Affected path: selection benchmark orchestration, not decoder implementation.
  Control or comparison: default constructor path used by pytest/profile/watcher.
  Likely subsystem: implicit merge of empty experiment policy with BASE.
  Investigation performed: reviewer identified the mismatch; stage owner added
  explicit `default_policy` selection and reran default/control/default. Reviewer
  independently checked exact policy equality and final source hashes.
  Resolution: fixed; misleading old runs remain classified historical evidence.

## Scope Inspected

- Goal/skill paths: stage owner's supplied optimized-decoder user contract;
  `.agents/skills/stage-review/SKILL.md` and `.agents/skills/optimize/SKILL.md`.
- Artifact paths: this directory's README, work log, optimization checklist,
  candidate/projection CSVs, policy/performance/matmul summaries, activation
  provenance, initial and final stress XML/JSON/logs, watcher JSON/logs, context
  JSON/logs, AutoDebug/AutoFix reports, benchmark/control/grid matrices and
  results, source snapshots, and Tracy reports/raw rows/archive manifest;
  `../context_contract.json` and relevant frozen fused artifacts.
- Code paths: `../../tt/optimized_decoder.py`; optimized runner/context/pytest,
  projection tuner, activation recorder, sweep/summary scripts and shell
  wrappers; frozen fused/functional source comparison; native matmul untilize
  validation and SDPA grid/reader/factory contracts where needed.
- Commands run: `rg`, `cat`, `sed`, `diff`, `git status`, `git rev-parse`,
  branch/diff inspection, and small standard-library Python scripts for AST,
  SHA256, JSON/XML/CSV, medians, profiler sums, archive existence, and static
  capacity arithmetic. No TTNN import, target execution, device use/reset,
  server, build, long test, or implementation edit was performed by the reviewer.

## Residual Risk

The tuned defaults are supported on the measured Blackhole 11x10/eight-bank
device and target shapes. Real-weight decoder PCC does not establish a future
full-model accuracy frontier. Caller-owned cache state and trace inputs must
still follow the documented restore/refresh contract. Sub-percent candidate
differences are subject to the recorded timing spread. These limits are
disclosed and do not leave a required optimized-decoder item unresolved.
