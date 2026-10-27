Historical initial review; superseded by [final local review](GROUPED_FINAL_REVIEW.md).

# Stage Review

Verdict: more-work-needed

Independent, read-only source/evidence review, 2026-10-27 14:42 UTC. Scope: stage07 optimized-full-model **grouped recurrent decode candidate, before integration**, Qwen/Qwen3.8-27B, all64 layers, TP4 QB2. This is not a whole-stage or release signoff. The candidate remains disabled in production.

Reviewed worktree: `/home/mvasiljev/build/tt-metal-qwen38-swe`, branch `mvasiljevic/qwen38-swe-opt`, HEAD `893cee41061cd763525f3c7634f423b12cba292b`, base `ed3804e8afd3475a1fb74054a346d662cc8b4844`, with an uncommitted prototype. Evidence root below means `/home/mvasiljev/build/qwen38-swe/grouped-recurrence/`. This is a live worktree; later changes and running tests are outside this snapshot.

No confirmed head-indexing, arithmetic, in-place-state race, or circular-buffer lifetime defect was found in the inspected decode-only kernels. Existing tracked runs provide useful primitive and real-weight-layer evidence. The concrete missing checks below must close before promotion; pending full-model, profiling, serving and scored results are not waived.

## Required Work

- **P1: Prove inactive nonzero state survives runtime mask changes.**
  Evidence: `tests/probe_grouped_decoder.py:99-102` allocates zero initial state and constructs one fixed mask. Its inactive-state assertion at lines137-138 checks that inactive rows remain zero. `decoder-b32-masked-tracked.json` and `decoder-b32-inactive-tracked.json` pass this narrower contract. `tt/multichip_decoder.py:631-637,708-714` explicitly preserves the original convolution and FP32 recurrent state, so preservation of arbitrary prior state is part of the affected path.
  Why this matters: resetting, rounding or replacing an inactive row with zero would pass the current all-inactive test. Serving deactivates and later reuses rows containing real prior carry; their FP32 low bits matter.
  Required next step: one focused real-weight replay with distinct nonzero per-row convolution/recurrent state, including nontrivial FP32 low bits. Refresh the same bound mask tensor between active, partially inactive, all-inactive and reactivated steps. Compare split/grouped output and both states on every chip, and compare inactive rows bitwise to their own immediately preceding state. Use `TT_METAL_TRACE_ALLOC_TRACKING=1` and `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, preallocate outputs/inputs, and retain both traces. This can extend the existing small probe; it does not require a new broad sweep.

- **P2: Cover automatic selection of three heads per core before exposing the helper for every supported batch.**
  Evidence: `tt/grouped_recurrence.py:42-49` automatically chooses `ceil(B*12/cores)`. On the110-core grid, B1-9 select1, B10-18 select2, B19-27 select3, and B28-32 select4. The retained primitive acceptance rows are B16/17 at2 and B31/32 at4. The test candidate defaults to an explicit1 and the primitive probe supplies an explicit count, so those runs do not exercise the production helper's automatic decision.
  Why this matters: the new helper admits an untested execution family if integrated for arbitrary B1..32. Three sequential heads also exercise another reuse count of every per-core CB.
  Required next step: add one strict-tracked advancing-state boundary case, preferably B19 through the automatic helper, with exact split/grouped output/state equality on all chips. Alternatively keep unvalidated batches on the existing split implementation until covered. B16/17/31/32 are useful split-tail cases, but `B*12` is divisible by every allowed count1..4, so they do not exercise a partially occupied final core; no artificial partial-core sweep is required for this Qwen contract.

- **P1: Validate the final wired path and changed full-model traces before enabling the candidate.**
  Evidence: production contains no import/call of `tt/grouped_recurrence.py`; the probes call `tests/grouped_recurrence_candidate.py`. The decoder probe's saved generated `_delta` AST matches the current method with exactly the decode `kernel_batch` assignment replaced by `batch`. At inspection, `full-b16-baseline.json` has `status=running`, tracking0, no generation records and no completed baseline/candidate comparison. No completed grouped full-model artifact is present. `tests/benchmark_grouped_model.py` as inspected records token outputs but does not itself compare split/grouped logits or recurrent states.
  Why this matters: primitive equality and one layer's equality do not establish the final generator's recurrent-state ownership, sampled-token feedback, mixed-mask trace selection or late-allocation safety. An incomplete clean-timing baseline is neither a failure nor acceptance evidence.
  Required next step: after wiring a clean optional path, complete matched all64-layer B16/B32 split/grouped comparisons on the final source, using the same precision, fixture, sampling and physical chunk geometry. Check generated tokens/output and recurrent/convolution-state continuation, including a partial batch and a reset/resume boundary. Run representative repeated full-model replay with tracking1/program-cache-skip0 and no broad allocation exemption; retain existing trace IDs/capture counts across the mixed work. Separately collect diagnostics-disabled timing. Reuse unchanged primitive/kernel evidence rather than rerunning every historical case.

- **P2: Close the remaining performance and integration gates using the actual target path.**
  Evidence: existing layer timings below are strict-tracked measurements with real layer0 weights and synthetic hidden inputs. The two ongoing scored pilots use the earlier serving configuration, not grouped source. No new reduced profile, watcher run, integrated serving measurement or grouped scored SWE result is in the reviewed evidence set. The owner identified the initial full-model fixture's default physical prefill4096/128 buckets versus SWE physical256/eight32-row buckets; the initial run had not generated or scored anything.
  Why this matters: recurrence/layer speedups cannot be multiplied by48 to claim a model or SWE gain. Startup from an unnecessarily large preparation family also distorts the intended workload comparison.
  Required next step: align the comparison fixture to physical256 while retaining logical4096 and native262144 capability; report setup separately. Obtain a minimal warmed reduced non-serving profile of the final path and a separate watcher run, then complete final-model timing and the shared three-prompt qualitative suite with a matched control. Before serving promotion, verify stale-input/mask refresh, restored recurrent plus convolution carry, mixed cached/chunked prefills, slot reuse and trace reuse under the existing serving contract. Before any SWE100/500 performance or accuracy forecast, collect a matched scored grouped pilot with task IDs, sampling, deadlines, outcomes and normal endings held/comparable. These are explicit open gates, not requests for another unrelated optimization matrix.

## Other Concerns

- **Source/evidence identity needs a final boundary.** All three grouped kernel hashes and `tt/multichip_decoder.py` match the saved tracked-final primitive and decoder artifacts. The formatted Python candidate/harness hashes do not match those saved runs. The saved decoder transformation is AST-equivalent to the current intended transformation, and inspection found no arithmetic change in the kernels. Preserve these narrower valid results, but record the final implementation identity and run affected representative checks on the final imported helper; old hashes alone cannot accept later integration. Hashes are currently collected in `finally`, so avoid editing source while a measured process is active or also snapshot identity at startup.
- **The new helper is a duplicate of the test implementation.** Host AST comparison shows their recurrence functions differ only in the automatic count/default and kernel-directory resolution. Prefer one implementation once integrated, with the experiment selecting it explicitly. This prevents later fixes or tests from drifting between copies.
- **Runtime provenance differs from CI.** The README records local compiled TTNN from retained checkout `3fbf3dbf1e88` with Python source from this worktree, versus the pinned CI image/source. The local comparisons remain useful within that regime. Record the resolved production runtime when integration is tested; do not imply these are already measurements of that exact CI stack.
- **Capability preservation is plausible from source, not a new native-context proof.** The change leaves full-attention KV, recurrent state dimensions/dtype, chunked prefill, context admission and sampler policy unchanged. `doc/context_contract.json` retains inherited262144 evidence. This review found no new cap and does not require repeating unaffected native-context sweeps merely for this decode scheduling change. Representative long-prefix continuation and actual serving reuse remain open target-workload checks.

## Hard-Check Gaps

- No independent defect was found in the generated-source harness: it changes exactly one decode split marker, binds the new recurrence in the generated function's globals, patches the batch-specific decoder view, and leaves the prefill chunk branch unchanged. Full-model construction through the patched `model_module.MultichipDecoder` creates the subclass and its copied views; the final production wiring still needs its own run.
- Reviewed primitive results, all with tracking1/program-cache-skip0, three advancing comparisons plus100 additional updates, exact outputs/states on all four chips:

  | Artifact | Split µs | Grouped µs | Speedup |
  |---|---:|---:|---:|
  | `b16-h2-tracked-final.json` |555.795|306.409|1.814×|
  | `b17-h2-tracked-final.json` |569.512|322.721|1.765×|
  | `b31-h4-tracked-final.json` |1057.856|502.437|2.105×|
  | `b32-h4-tracked-final.json` |1116.248|530.832|2.103×|

- Reviewed layer0 results, all tracking1/program-cache-skip0, with exact output/conv/recurrent comparisons on every chip at steps1,2,3 and after timing:

  | Artifact | Baseline µs | Grouped µs | Speedup |
  |---|---:|---:|---:|
  | `decoder-b16-tracked.json` |1248.592|929.187|1.344×|
  | `decoder-b32-tracked.json` |2001.417|1281.268|1.562×|
  | `decoder-b32-masked-tracked.json` |2482.569|1902.364|1.305×|
  | `decoder-b32-inactive-tracked.json` |2483.293|1901.615|1.306×|

- These are traced host wall times with tracking overhead, not clean kernel timings, full-model speedups, HF accuracy or scored SWE improvements. Full-model/profile/watcher/serving/scored gates listed above are still open.
- Retained sampler comparisons keep the default unchanged. They do not supply grouped recurrence correctness or scoring evidence. CI pilots40516531644 and40518335859 are not accepted grouped results.

## Anomaly Ledger

- Observed anomaly: the initial B32 tracked primitive probe failed before replay with a live output allocation that another trace could corrupt.
  Evidence: `b32-h4-tracked.json` retains the tracker RuntimeError; it is a failed artifact.
  Affected path: probe ownership of trace-created output across alternating split/grouped replays.
  Control or comparison: `b32-h4-tracked-stable.json` and the four `*tracked-final.json` runs.
  Likely subsystem: test-harness output lifetime, not grouped recurrence arithmetic.
  Investigation performed: source now allocates both output buffers and state buffers before either capture, copies transient results into those buffers, warms both variants before capture, alternates both traces and compares outputs/state.
  Resolution: **fixed for the retained primitive/layer probe pattern**. Corrected runs use tracking1/program-cache-skip0 with no corruptible exemption. This does not waive full-model lifetime validation.

- Observed anomaly: full-model baseline setup selected a larger physical prefill preparation family than the SWE path and had no output/score yet.
  Evidence: the inspected benchmark omitted `prefill_chunk_size`; generator default is4096 and its bucket construction spans32..4096. `full-b16-baseline.json` was still running with zero records; the log was in `PREPARE_PREFILL` work.
  Affected path: full-model comparison fixture startup and workload comparability.
  Control or comparison: actual SWE block256 path; owner reports planned physical256 alignment without reducing logical4096/native262144.
  Likely subsystem: benchmark configuration.
  Investigation performed: read generator defaults/bucket construction and the live result/log; owner is preserving and replacing the owned warmup.
  Resolution: **more-work-needed**. Do not count the replaced warmup as a generated, timed or scored comparison result.

## Scope Inspected

- Goal/skill paths: supplied grouped stage07 contract; installed `stage-review/SKILL.md`, relevant `optimize/SKILL.md` requirements, `tt-enable-tracing/SKILL.md`, `model-bringup/SKILL.md` and `model-bringup/references/target-workload.md`; checkout `tech_reports/AdvancedPerformanceOptimizationsForModels/TraceCorrectness.md`. No applicable ancestor `AGENTS.md` was found in the explicitly checked paths.
- Artifact paths: `doc/swe_optimization/{README.md,GROUPED_RECURRENCE.md}`, `doc/workload.md`, inherited `doc/context_contract.json`, `doc/qualitative_prompts.json`; the raw JSON rows above, retained failed B32 JSON, corresponding local logs and incomplete `full-b16-baseline.json`/`grouped-full-b16-baseline.log`.
- Code paths: `tests/{grouped_recurrence_candidate.py,probe_grouped_recurrence.py,probe_grouped_decoder.py,benchmark_grouped_model.py}`; `tt/{grouped_recurrence.py,multichip_decoder.py,fp32_recurrence.py,optimized_decoder.py,model.py,generator.py,precision.py}` as relevant; grouped reader/compute/writer kernels; original FP32 compute and upstream reader/writer kernels.
- Commands run: read-only `cat`, `sed`, `nl`, `rg`, `diff`, `git status/log/branch`, UTC clock, and small host-only Python3 JSON/hash/AST/index-arithmetic analysis. No model, server, TTNN import/device open, hardware test, reset, CI dispatch, source edit or other checkout mutation was performed. Only this review report was written.
- Source snapshot hashes: candidate `c8291591b033ab85bd3a4f42702feebd35639ea0a9bc8f9b35065e8fc6840334`; optional helper `d31c90018156d033317c644fa299d2b3e889654e214068b3654893ee615ade17`; decoder probe `b33db39622ab462b431aeee4ddffa7c11b78c21a9d87daeb2e8b0a59f59c4c02`; benchmark `0265ace242cb083f7243295dd8c1fe65b5695c97283c112a35f47f1f45026722`.

## Residual Risk

- Source inspection supports the per-head ownership invariant: each core receives consecutive disjoint heads; reader and writer agree on head-major output/state addresses; final-only in-place writes target the same head only after its full initial state was read. Reader prefetch of another head is disjoint from the prior head's write.
- Compute math is unchanged relative to `tt/kernels/fp32_recurrence.cpp`; grouped changes add the outer head loop/count and final carried-state pop. With T1, cb_s2 is consumed at the end of each head; all matrix/input/intermediate CBs are consumed in the existing loop, and the writer drains the separate cb_state copy. Per-core CB allocation remains663552 bytes. No new persistent state is introduced by the helper.
- Strict-tracked exact primitive/layer comparisons materially reduce indexing and numerical-regression risk, but fixed-input zero-state masked tests leave the concrete reuse gap above. The auto3 family, final integrated full-model path, current-runtime watcher/profile, representative long-context cached continuation and scored workload effect remain unverified.
- The practical SWE100/500 decision must use measured useful task completion and accuracy. No full-suite forecast or release-readiness claim follows from this review.
