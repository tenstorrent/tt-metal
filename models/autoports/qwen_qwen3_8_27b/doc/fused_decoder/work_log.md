# Fused decoder work log

Stage 2, Qwen/Qwen3.8-27B, revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
Started from clean branch mvasiljevic/qwen38-full-bringup at 41bdb276313.
Scope: fused_decoder.py, tests, and documentation. No later stage or native edits.
Status: completed; independent stage_review_2.md returned clean-pass.

## Startup

Read local and installed graph-fusing, tt-device-usage, stage-review, AutoFix,
installed model-bringup startup, and local tracing guidance. Installed enabled
AutoDebug 0.1.5 and model-bringup 0.1.4 verified in the run Codex config.
Environment exports are validated by the installed environment.py in
`tests/run_fusion_experiment.sh`. Existing python_env is used; no install/build.
Four Blackhole p300c devices enumerate; a 1x1 mesh opens and closes with 11x10
worker grid. See device_list.log and mesh_smoke.log. No reset or process kill.
Hardware commands are serial. Watcher and profiler will run separately.

## Measurement contract

The baseline and all candidates use real layer 0/3 weights, BF16 weights and
activations, FP32 recurrent state, BF16 paged KV, and unchanged HiFi4 projection
configuration. `run_fused_decoder.py` records five warmed prefill wall times and
30 warmed trace replay times, with one device synchronization per sample.
State restoration/readback is outside timing. Every trace sample must equal the
first restored-state output bitwise. HF PCC and changed-input/page-table checks
remain >=0.995. Control tensors in control/ are local comparison boundaries,
never runtime inputs. Exact experiments are appended to commands.log by the
shell wrapper. Candidate source snapshots preserve each graph being assessed.

Initial baseline full command: the same wrapper arguments as baseline_linear,
with name baseline_full and layer 3 (executed directly before wrapper creation).
Both use --baseline --benchmark --length 128 --compare-dir <stage>/control.
Baseline medians: full prefill/decode 3.5543/2.4438 ms; linear 4.8483/3.3508 ms.

## Candidate observations

- Dedicated partial RoPE (`rope_full`): fused/functional prefill/decode PCC
  0.99996156/0.99997342; HF 0.99892974/0.99824727; median 3.5323/2.4192 ms.
  Per-user application preserves different RoPE buffers; decode token_index=0
  consumes the caller's already-selected row. Remaining nonrotary channels pass
  through. Kept provisionally, pending boundary tests.
- Flat scan (`flat_scan_linear`): passed PCC, but prefill 4.9489 ms was slower.
  Source audit found obsolete head-split and reverse reshape. Adapted the
  producer to stay flat (`flat_scan_linear_adapted`): prefill 4.2622 ms.
  Native prep fuses L2 normalization, scale and GQA mapping. Zero padded g/beta
  make internal tail transitions identity; public logical lengths are retained.
- Dedicated four-tap convolution: initial `conv_linear` failed on Python config
  namespace. Tests and __init__.py show the config is exported at top-level
  ttnn; corrected and retry is `conv_linear_adapted`. This is not yet a rejection.

## Continued fusion experiments

- `conv_linear_adapted`: real-weight fused/functional PCC 0.99996370/0.99997103;
  prefill/decode 3.8084/3.1682 ms. The four-tap native op preserves staged BF16
  product/add rounding and SiLU. Input tails are internally padded to 32, then
  only logical rows update history. Channel block 256 was the first config.
- `scan_decode_linear`: use the same flat native scan for one real token plus
  neutral padded transitions. Decode 2.8324 ms, direct PCC 0.99997175.
- `concat_full`: dedicated prefill concatenation helps prefill (3.4000 ms),
  but slows decode (2.4815 ms). Decode now directly flattens SDPA's head order;
  the redundant transpose pair is removed.
- `packed_full`: host reorder interleaved Q/gate weights and pack Q,K,V,gate;
  prefill/decode 2.9278/2.3874 ms. All projected channels remain consumed.
- `split_full`: dedicated prefill QKV/head creation reduces prefill to 2.7525
  ms, but costs extra padded normalization work in decode (2.4610 ms).
  `decode_heads_full` uses the decode-specific head creator and direct SDPA
  flatten: 2.7544/2.3262 ms. Q/K normalization stays before partial RoPE.
- `packed_mlp_full`: gate/up packing 2.8492/2.3459 ms; slower than separate.
  `activation_full` retains packing with SiLU/sigmoid folded into multiply:
  2.7976/2.3377 ms. `activation_separate_full` uses the same activation folding
  with separate projections: 2.7010/2.3203 ms. Separate projections win.
- `sharded_norm_full` fails the RMSNorm prohibition on HEIGHT_SHARDED.
  Retrying with equivalent BLOCK_SHARDED (`sharded_norm_full_adapted`) passes
  with unchanged PCC, 2.6884/2.3132 ms. V stays sharded through cache update.
- `packed_linear`: pack QKV,Z,B,A with tile padding for 48-wide A/B. Passed
  direct PCC, 3.6432/2.7920 ms. Padding columns do not become model heads.
- `gated_norm_linear`: sigmoid-gated RMSNorm plus Z multiply implements SiLU
  gating. Initial prefill/decode 3.3363/2.8031 ms. Native scan supports
  `output_head_major=True`; using it removes the head-layout round trip
  (`head_major_linear`), 3.2114/2.7697 ms. Feeding FP32 scan output directly
  into that fused norm (`norm_fp32_linear`) passes and gives 3.1879/2.7668 ms.
- `fused_cache_full`: dedicated dual paged update uses disjoint K/V core sets;
  K is placed directly on its final update cores. Passed, 2.7119/2.3066 ms.
- `row_history_linear`: retain BF16 history as ROW_MAJOR, matching its only
  consumer. Eliminates repeated history untilize/tilize; 3.0722/2.5108 ms.
  `history_slice_linear` removes full history concat for >=3 real tokens:
  3.0283/2.5097 ms. Short prefixes concatenate only the retained old rows.
- `softplus_linear`: fold softplus into the decay multiply; 3.0695/2.5036 ms.
  Prefill sample variation is retained in raw arrays; final measurements will
  report the reproduced full path rather than combining candidate minima.

Snapshots in candidates/ and logs/JSON preserve rejected/adapted graphs.
No rewrite is selected solely because it reduces op count.

## Pre-fix graph refinements and validation

- Conv channel blocks 64/128/256/512 were compared with unchanged dtype and
  compute policy. Traced decode medians: 2.5266/2.5084/2.5036/2.5111 ms.
  Retained 256. This tunes the selected fused op, not a later decoder stage.
- Explicit matmul SiLU epilogue with the current 11x10 grid passes
  (`matmul_activation_full`), 2.7249/2.3056 ms. The default matmul activation
  argument without an explicit grid would append a unary op in matmul.cpp;
  the selected explicit-grid path really fuses the activation.
- Specialized decode concat first failed because paged GQA SDPA rejects sharded
  output. Added the required output conversion; next probe exposed the op's
  padded batch dimension. Sliced its logical batch and reran successfully
  (`decode_concat_full_padded`): 2.7079/2.3122 ms. Direct flatten is faster.
- Larger-batch block-shard norm failed at batch 2 (`boundaries_full_initial`).
  Preserve the batch-1 block-sharded fast path; larger batches normalize
  interleaved Q/K, retaining all other fused decode operations. The entire
  batch-2 full-attention sweep passes (`boundaries_full`).
- `l1_rope_full`: keep the batch-1 partial-RoPE intermediate in interleaved L1
  instead of DRAM. Passed, 2.7175/2.3013 ms.
- `padded_scan_linear`: retain convolution Q/K/V's existing padded rows; only
  beta and log-decay need neutral zero tails. Passed including lengths
  1,2,3,31,32,33,127,128,129,257 and continuation (`padded_scan_boundaries`).
  Decode falls to 2.4603 ms.
- `logical_view_linear`: expose the logical output length with an unchanged
  physical tile shape before the row-independent projection. Removes a tail
  slice/relayout; 2.4562 ms decode. Entire affected boundary sweep passes.
- `native_rope_full`: keep [1,B,H,D] from decode head creation through partial
  RoPE and paged SDPA. Avoid converting to prefill heads and back. Decode
  2.2600 ms; affected full-attention boundary/continuation/routing sweep passes.
- Final cleanup moves the linear norm weight's rank conversion to setup,
  removes the obsolete decode switch from the common native scan, and uses
  the same native RoPE layout for a one-token prefill. Physical-shape-preserving
  views hide only padded trailing rows. `synthetic_pytest.log/xml` proves the
  delivered fused tests pass (2 tests, each with 11 lengths/continuation/replay
  cases). The test patches FunctionalDecoder construction to raise on fallback.
- Watcher runs use `TT_METAL_WATCHER=10` and separate log roots
  `watcher_l{0,3}_b{1,32}`. All four real-weight 257-token continuation runs
  pass, including restored-state bitwise replay, changed inputs/positions and
  full-attention page-table rerouting. `watcher_summary.json` records paths and
  metrics; all four watcher logs have no fatal/invalid/overflow/out-of-bounds/
  error/corrupt/sanitize matches. No recovery/reset was needed.
- `tests/profile_fused_decoder.sh` collects fresh baseline and fused Tracy runs,
  serially and without watcher. All four runner correctness checks pass.
  Human tables and filtered CSVs are in `tracy/{baseline,fused}_l{0,3}/`.
  `performance.json` retains real device rows, including dtype/fidelity and
  movement. Tracy's mixed-column CSV parsing warning is a host parser warning;
  all expected signposts/device rows were generated.

## Pre-fix full-context checks

`run_fused_context.py` calls each public prefill API on complete logical inputs
4097,262143,262144,31 with the same seed, real weights and page-table generation
as the frozen functional-stage runner. It compares every output element, then
checks traced decode and repeated replay at every in-contract next position.
It does not split the public request to hide allocation limits.

The functional-stage HF correlations are reused explicitly, not reported as
new HF runs. For centered normalized output vectors, correlations obey
`r(new,HF) >= r(new,old)*r(old,HF) - sqrt((1-r(new,old)^2)*(1-r(old,HF)^2))`.
Thus measured on-device equivalence plus the frozen real HF evidence provides
a conservative numerical lower bound, asserted >=0.995. The raw correlations,
bounds, baseline-artifact hashes and both source hashes are saved in context
JSON. The exact baseline source, weights, software environment and seeded
inputs are unchanged. Small/medium cases also retain direct HF comparisons.

Commands (substitute layer 0/3 and linear/full):

```bash
PYTHONPATH=. timeout -k 10 7200 python_env/bin/python \
 models/autoports/qwen_qwen3_8_27b/tests/run_fused_context.py \
 --snapshot /home/mvasiljevic/hf-cache/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
 --layer 0 --baseline models/autoports/qwen_qwen3_8_27b/doc/functional_decoder/linear_context.json \
 --output models/autoports/qwen_qwen3_8_27b/doc/fused_decoder/linear_context.json
```

Both context runs completed with all gates passed. Full attention at 262144
tokens has a conservative HF prefill PCC bound 0.99606463; decode at the final
valid context has bound 0.99743386. Linear 262143-token prefill bound is
0.99799147. Independent review findings are recorded below.

## Final review corrections and last layout candidate

Initial independent review `stage_review_1.md` returned more-work-needed for
final-source benchmarks, uneven batch-3 grouping, the last untilize assessment,
and completion of handoff docs. It accepted both complete context runs,
current-source profiler and watcher evidence, and found no source defect.

The untilize projection experiment split QKV from the other linear projections
to meet the 1D kernel's FP32 destination constraints (per-core output block at
most four tiles; the original packed projection needs at least five). Whole
128-token prefill also cannot fit the output block constraint without further
splitting M. This experiment therefore targets batch-1 decode only.

- `untilize_projection_linear`: `ttnn.linear` returns TILE despite the config
  flag; native source forces the operation attribute false. Conv rejects it.
- `untilize_matmul_linear` and `untilize_matmul_layout_linear`: adapted to
  `ttnn.matmul`; actual output still reports TILE, including before slicing.
  The diagnostic preserves the config with untilize_out=1 and actual layout.
- `split_projection_linear`: explicit conversion produces a legal graph but
  fails fused/functional decode PCC (0.856359). This is rejected correctness
  evidence, never a selected result or a waived accuracy threshold.
- `untilize_output_linear`: supply a setup-allocated ROW_MAJOR output tensor;
  valid execution, but decode PCC 0.848129. Increasing the per-core N/subblock
  width from 3 to 4 (`untilize_output_n4_linear`) gives PCC 0.815826.
- `split_projection_control_linear`: same split and N=3 with untilize_out=False
  and explicit conversion passes PCC 0.9999932 and deterministic replay.
  Median prefill/decode is 3.04609/2.45809 ms. This isolates the failing
  untilize-output path from packing, scan and state handling. It does not beat
  the selected 2.45619 ms precursor and still needs the conversion.

The available untilize-output entry point is not a correct replacement in this
runtime. The native matmul tests also mark an FP32 cross-block untilize precision
case xfail (issue #49836), but that is not claimed as the diagnosis of this
experiment. Current source advertises a row-major matmul output contract that
the live entry point did not satisfy; no native C++ or environment change was
made to work around it. All adaptations and the passing control are retained
under candidates/ and their corresponding logs. No final runtime edit resulted.

Clock-settling warnings and Tracy CSV mixed-column warnings were inspected
independently in stage review. The device traces, signposts and CSV sums are
complete, all four watcher logs are clean, and no recovery/reset occurred.
Wall samples retain their natural variation; selected comparisons use medians
from complete runs, without combining favorable individual samples.

## AutoFix: additional batch-3 boundary

The new real-weight batch-3 sweep passed lengths 1,2,3,31,33,129, but length257
changed-input replay PCC was 0.99362123, below 0.995. Its prefill, continuation,
normal eager/traced decode and bitwise replay all passed. A same-case functional
control fails changed-input PCC at 0.99300992, so this is not evidence of a new
grouping or trace regression by itself. The expanded delivered synthetic suite
passes all three cases (linear B2/B3, full B2), each at 11 lengths.

AutoFix was invoked and a fresh xhigh source-only AutoDebug agent is diagnosing
the shared discrepancy. Exact failing commands, logs and successful partial
cases are retained; no tolerance was relaxed and no runtime edit made.

## Pre-fix delivery measurements and portable evidence

`delivered_linear_benchmark` and `delivered_full_benchmark` run the final default
source (a362a3e1be350b1fdd2edfa50aaf724727bcd2d37822883c673632bad758ff04), not
a candidate subclass. Medians are linear prefill/decode 3.06045450/2.45555537 ms,
full 2.68278364/2.25890474 ms. Those improve the functional baseline by
36.9%/26.7% and 24.5%/7.6%. Earlier `final_*_benchmark` names are historical
candidate records and are superseded; they are not used for final claims.

Candidate sources are committed as `candidates.zip`, with per-file SHA256 and
sizes in `candidate_sources.json`. The original extracted paths remain local;
`python3 -m zipfile -e candidates.zip .` restores them from this doc directory.
Raw op CSVs are included losslessly as `tracy/*/ops.csv.gz` (decompress with
`gzip -dk .../ops.csv.gz`); all filtered report CSVs and human tables are directly
included. Large raw Tracy captures and Torch setup/control tensors remain local
and are recorded by hash. No weights or temporary Python caches are committed.

The eight tables were produced with `python_env/bin/tt-perf-report`, one command
per variant/layer/mode; for example:

```bash
python_env/bin/tt-perf-report \
 models/autoports/qwen_qwen3_8_27b/doc/fused_decoder/tracy/fused_l0/ops.csv \
 --start-signpost PERF_DECODE --end-signpost PERF_DECODE_END \
 --csv models/autoports/qwen_qwen3_8_27b/doc/fused_decoder/tracy/fused_l0/decode_perf_report.csv --no-advice
```

The human table invocation uses the same window with
`--no-summary --no-advice --no-color` and redirects stdout to the matching `.txt`.
Replace DECODE with PREFILL for prefill and select baseline/fused, l0/l3.

## Proven normalization correction

AutoFix refuted the leading cache-rounding hypothesis: HF BF16 cache versus
its pre-copy FP32 recurrence changes output PCC by only about 0.0000025.
Eager and trace outputs and both state buffers are bitwise equal in all state
swap probes. The error is concentrated in user 0, so uneven 2+1 grouping is
not the cause. FP32 folded norm weights alone worsen PCC and were discarded.

The isolated native RMSNorm compute-config control restores the original
unmodified-HF gate from 0.99362123 to 0.99988180 (per-user minimum0.99950445).
It also improves prefill from 0.99894450 to 0.99996527. The implementation now
passes its existing HiFi4/exact/FP32-destination compute config to linear
attention's common RMSNorm calls. BF16 weights/activations/history and FP32
recurrence are unchanged. The original seven-length batch-3 real-weight sweep
passes after this minimal correction.

Applying this configuration to full attention was tested separately and
rejected: it already passed the complete correctness/context contract, while
traced decode rose from about2.259 to2.297 ms. The selected condition applies
only to linear_attention, preserving the faster correct full-attention graph.
Earlier linear timings with default norm are now pre-fix candidates, not valid
winners of the expanded correctness gate. `AUTOFIX.md` records the controlled
experiments and final precision-policy comparison.

`tests/validate_fused_decoder.sh` is the serial final validation command: it
runs the expanded delivered pytest suite, four watcher cases, independent
baseline/fused profiling, and both full-context comparisons. Its `verified_*`,
`watcher_final_*` and `tracy_final/` artifacts supersede corresponding pre-fix
evidence. The older evidence is retained as history rather than overwritten.

## Current-source final evidence

The sole retained AutoFix edit gives final runtime SHA256
`3f324562925d42fb0c2997672f86fe68926a21869d3f2d351d1f5c3ba9792035`.
Sections explicitly marked pre-fix above describe historical source a362...;
all final claims use corrected/verified/watcher_final/tracy_final artifacts.

Final B1 warmed medians (`corrected_*_benchmark`): linear prefill/decode
3.07446346/2.49408465 ms, full 2.75435578/2.26043211 ms. Three paired linear
runs preserve 90 samples for each packed/split alternate: pooled medians
2.49281060/2.49233423 ms, with winner reversal across runs. That 0.019% difference
is not a consistent speed advantage; retain the packed implementation and do
not claim it decisively beats the tied alternate. The substantial baseline
reductions remain 36.6%/25.6% linear and22.5%/7.5% full (prefill/decode).
`candidate_metrics.csv` now includes all corrected runs and labels the invalid
pre-fix normalization policy rather than treating those times as valid winners.

Final profiler kernel sums (prefill/decode): linear baseline4600.646/3011.400 µs,
fused2865.165/2466.776 µs; full baseline3400.790/2375.970 µs,
fused2590.584/2223.064 µs. Counts remain88/100 ->32/38 and52/57 ->29/32.
`performance_final.json` and `tracy_final/*/*_perf_report.{csv,txt}` preserve
native dtypes, fidelity, movement and gaps. These kernel sums exclude gaps.

Final synthetic suite:3 tests pass, 33 boundary cases total. Four final watcher
logs are clean and independently hash-checked; `watcher_final_summary.json`
records them. Linear full-context passes all four lengths with worst prefill
HF correlation lower bound0.99615411 (262143); exact-limit bound0.99623798.
The full-attention final-source repeat also passed all four lengths: minimum
prefill bound0.99601026 (262143), exact-limit bound0.99606463, and final valid
decode bound0.99743386. Its graph remains unchanged.

The changed-token control intentionally remains documented: corrected-vs-frozen
functional PCC0.99181628 because frozen functional fails stock HF, while the
corrected result passes stock HF at0.99988180. This is a localized inherited
normalization error repaired by AutoFix, not a replacement oracle or relaxed
threshold. `probe_batch3_equivalence.json` retains the comparison.

Artifact packaging preserves raw execution and watcher logs losslessly in
`logs_execution.zip` and `logs_watcher.zip`, with their original paths inside.
`python3 -m zipfile -e <archive> .` restores either from this doc directory.
CSV report line endings and text-table trailing whitespace are normalized only
for repository formatting; raw op CSV bytes remain in `ops.csv.gz`.
Generated inspector auxiliary dumps are local tooling data and are excluded.

Final serial validation finished successfully at2026-09-11 21:19:57 UTC with
clean mesh teardown. Command: `bash models/autoports/qwen_qwen3_8_27b/tests/validate_fused_decoder.sh`.
No hangs/resets/recovery occurred. Checks also run: Black --check with py312,
Python source syntax compilation, bash -n on all three shell runners,
repository scripts/check_file_size.py, and git diff --cached --check. No C++
build is required for this Python/docs-only stage. The pre-commit/isort/autoflake executables were absent from PATH, but the
repository commit hook supplies cached hook environments. It ran the full
hook suite and fixed one import-order issue in run_fused_context.py; that
formatting-only change was retained before retry. No dependencies were
installed by this stage.

## Stage acceptance

Independent fresh-context xhigh review `stage_review_2.md` returned **clean-pass**
on2026-09-11, with no required work. It independently verified all final context
rows and source hashes, stock-HF gates, archived original tensors/logs, profiler
rows and totals, watcher logs, candidate archives, and the packed/split timing
tie. All findings from stage_review_1.md are resolved; AutoFix succeeded.

Only fused_decoder.py, stage tests and docs/context metadata are changed. No
optimized-decoder, multichip, full-model or serving work was started. The local
stage checkpoint SHA is recorded below; the following provenance commit records
that immutable checkpoint. No push is authorized or performed.

## Local checkpoint provenance

- Starting functional-stage/provenance commit: `41bdb276313`.
- Fused-stage checkpoint: `21f030fc4376d99f18fa9d41e1b0f3a71f8feb65`.
- All repository commit hooks passed on the checkpoint retry, including Black,
  autoflake, isort, whitespace/EOF, file-size and applicable repository checks.
- The next local provenance commit contains this SHA record only. Its identity
  is available in `git log`; a commit cannot include its own content hash.
- No push was performed. Working scope remains the fused decoder, tests and docs.
