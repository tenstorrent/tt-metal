# Fused decoder work log

Stage 02, google/gemma-4-26B-A4B-it, revision
4d7ae4984b7db7de8f8457170b3f1a419ee76d52. Starting checkpoint
37c8b975c8 (functional stage clean-pass). Only fused runtime, tests and documentation
are being changed. No later pipeline stages are started.

## Startup

`git status --short` was clean. `rg` is unavailable; searches use grep/find.
Installed graph-fusing, tt-device-usage, AutoFix and stage-review instructions
were supplied/read. Installed environment.py --autodebug-root
/home/mvasiljevic/.codex-personal-gemma4/plugins/cache/tenstorrent-skills/tt-autodebug/0.1.6
validated setup. Enabled skills are present in the session inventory.
`timeout 60 tt-smi -ls --local` reported four Blackhole p300c ASICs.
`timeout 60 python -c 'import ttnn; m = ttnn.open_mesh_device(ttnn.MeshShape(1,1), trace_region_size=0); ttnn.close_mesh_device(m)'`
passed; startup_mesh.log. One ASIC participates. No reset was needed.

## Baseline

Functional baseline and context contract remain unchanged. PCC threshold .995;
context 262144, BF16 32-token paged KV. Real layer kinds are sliding_attention
(layer 0) and full_attention (layer 5), both with shared MLP and routed MoE.
Functional whole-layer device timing, exactly 4096/128/B1/C1:
sliding 4998536.82 us prefill / 9811.07 us traced decode;
full 5009442.54 us / 10894.37 us. Source:
../functional_decoder/tracy/{sliding,full}/whole_layer.json and PERFORMANCE.md.
These are baseline values, never reported as fused measurements.

## Experiments in progress

Run module `models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder`.
All runs use `--real --decode`, output JSON and same-stem console log here.

- `--decoder fused --fusion broadcast --layer 0 --length 33 --steps 3`:
  broadcast_sliding_33.json; prefill .9983004889, decode minimum .9997637065,
  deterministic replay. Removes repeated QKV activation materialization.
- `--decoder fused --fusion broadcast_pack --layer 0 --length 33 --steps 3`:
  pack_sliding_33.json; same PCCs. Packs expert gate/up, removes identity
  transpose chains, collects prefill tile outputs into one concatenation.
- `--decoder fused --fusion broadcast_pack --layer 0 --length 4096 --steps 128 --timing --verify-program-cache`:
  pack_sliding_headline.json; prefill .9985263366, decode minimum .9990432236,
  same as functional baseline, runtime audit clean. Host-wall batched replay
  ~8239 us; this is candidate-selection evidence, not device latency.

Dedicated precision-boundary fusion is being examined independently under
AutoFix, against the prior functional AutoDebug/AutoFix reports. Source diagnosis
and focused probes live in AUTODEBUG_precision.md / tests/probe_fused_precision.py.

No fusion is accepted on op count alone. Current candidates require final device
profiling, broad correctness coverage, watcher, and independent stage-review.

## Precision and structural experiments

All timings below are warmed batched trace **host-wall** microseconds used for
candidate selection; final device latency is reported separately after Tracy.
`*_commands.json` journals preserve exact argv and exit codes. Same-stem logs
contain console evidence; no measured pass contains torch conversions.

- QKV broadcast groups256..16384 are bitwise equal for both kinds
  (`components_sliding_retry.json`, `components_full.json`). One coalesced group
  is fastest; partial final group bounds are clamped. Shared/routed gate-up
  packing, redundant expert movement removal, one prefill concat, accurate GELU
  merge and router/tail binary merging preserve component or whole-layer PCC.
- `isolated_candidate_commands.json` isolates native norm, batched attention and
  native RoPE. `norm_attention_*` / `common_norm_sliding` / `tied_common_full`
  add independent head batching, shared residual normalization and tied K/V
  projection elimination. These progressively reduce traced replay to ~5–6ms.
- `precision_matrix_commands.json` checks native RMSNorm, native rotary and
  native SDPA; AUTOFIX.md records the adapted retries and numerical failures.
  Native RMSNorm gamma narrowing is avoided with external FP32 gamma. Sliding
  head normalization stays precise after an S2049 request-reuse regression.
  Sliding native rotary is decode-only; full native rotary is legal but must
  earn retention on latency. Native SDPA cannot express accurate FP32 Q here.
- `softmax_adapted_commands.json` records the rejected explicit FP32 binary
  rounding flag and centered native-softmax failure. Corrected
  `softmax_retry_commands.json` has unchanged PCC with accurate SUB+EXP(0.0)
  and residual-input RMSNorm. Native full softmax is slower; sliding fails.
- `headnorm_merge_commands.json`: exact ADD+RSQRT(0.0) keeps sliding PCC unchanged
  and reduces host replay ~5148 to5142us. Packing Q/K/V head-normalization rows
  also preserves PCC but increases latency to5165us, rejected.
- `selected_broad_commands.json`: both selected kinds pass nine varying-length
  requests31/32/33/1023/1024/1025/2049/33/2047 with changing page ownership,
  batch32 and prefix continuation31/33/65. The cache-prefix and other-slot
  preservation checks pass. Final max-context/profile/watcher checks follow.

Current full boundary controls separate native rotary, fused cache updates and
shared K/V normalization because their combined configuration was ~11us slower
than the simpler correct candidate. An op-count reduction is not acceptance.

Full boundary results: `full_boundary_kvnorm.json` ~5622us, cache alone5632us,
native rotary5657us, combined cache+KVnorm5623us. Select tied-KV normalization
without native rotary or fused-cache writes for full; the latter combination
has no measured advantage. Sliding selects the disjoint-shard fused cache writer.
`cache_gather_adapted.json` confirms both flattened/page-axis tiled gather outputs
exact, but page-axis3980us/dynamic4107us versus fully warmed embedding139us;
embedding is retained. All gathered page IDs are device-derived at runtime.

`final_validation_commands.json` records the serial default-policy gate suite.
The pytest regression patches FunctionalDecoder._forward to fail; both real
layer kinds passed, proving inherited public orchestration dispatches the fused
computation. Full default request reuse, batch32 and prefix continuation pass.

All four maximum/near-maximum context checks pass under the selected policy:
`long_{sliding,full}_{262144,262143}.json`. Full TT prefill/cache is computed;
PCC compares291 HF query rows. Each traced decode checks three positions near
the end with mutable position tensors and randomized physical page ownership.
No advertised capacity, public logical alignment, KV dtype or cache format changed.

## Independent review and final peer merging

The first `STAGE_REVIEW_INITIAL.md` correctly returned more-work-needed for unfinished
measurements and reproducibility of the rejected decode GELU control. The probe
now forces that control; fresh outputs confirm exact equality but slower decode.
Concat/one-rotary/split is exact but slower. Direct expert batches32..1024, including
512, are exact on captured real model inputs;64 is fastest for both kinds.
Selected defaults now include `expertbatch64`; `batch64_validation_commands.json`
reruns the full gates. Paired functional/fused checks pass all4096 prefill rows and
all128 decode outputs: minimum direct PCC .999662/.999747 sliding and
.999674/.999868 full. All request-reuse, batch32, continuation and fused-only
pytest checks pass with this final expert-batch setting.

The loop32 full profiler run failed marker pairing during teardown after its
correctness/signpost workload passed. No device measurement is taken from that
capture. `AUTOFIX_profiler.md` traces a likely clock rollover sampling issue;
reruns keep final marker validation enabled. `tt-smi -ls --local` and the open/close
smoke (`post_profiler_smoke.log`) passed; no reset was performed. The loop32
sliding capture was valid and is retained as an intermediate candidate measurement.

Fresh final profiles both complete successfully with all 128 complete decode
windows. Final sliding/full prefill is 2813086.11/2825299.53us and decode
5214.70/5683.48us, respectively; PERFORMANCE.md records whole-layer denominators
and separate per-kind rooflines. Raw paths, hashes, commands and source hashes
are in tracy/{sliding,full}_batch64/provenance.json and report_commands.json.
Both final watcher logs have six completed checks and no detected errors.
The invalid previous full capture contributes no measurements.

Final Python hooks pass; the first all-artifact hook pass normalized trailing
whitespace in generated report text and final newlines in command journals.
No production source changed during formatting. No C++/CMake build is needed.

The final expert-batch64 policy also passes all four context reruns, including
262143 non-aligned tokens for each kind. Exact commands and zero exit statuses
are in batch64_validation_commands.json; ACCURACY.md records their PCC.

The final weighted-reduction probe found an additional improvement and was not
dismissed: decode matmul is48us versus56us for multiply/reduce, while prefill
matmul is slower. Complete-layer controls validate both accumulators. Direct
output into the next norm's width-sharded memory further removes a copy.
The selected sliding policy uses FP32 destination accumulation (5129.59us);
full uses BF16 accumulation (5609.52us). Alternate accumulators differ by~1us.
Both have HiFi4 and BF16 output; sparse projection precision is unchanged.
The selected policy is rerun by selected_validation_commands.json, preserving
the batch64 suite as valid evidence for the preceding reduction candidate.

Generic post-capture allocation warnings are confined to the changing-request
diagnostic runner. Its lifetime control is unchanged from
../functional_decoder/RUNTIME_AUDIT.md#allocation-warning-classification:
stable trace-owned inputs/outputs/caches persist, temporary request tensors are
released before replay, and all nine requests/repeated outputs pass. The warning
is not a detected overlap and does not permit arbitrary caller allocations.
Unknown-motherboard/subset-MMIO messages describe topology discovery/host IO;
all runs select one ASIC consistently. Profiler mixed-column warnings are CSV
typing notices; summary checks validate every complete window and op count.

## Independent final-review boundary controls

The fresh reviewer identified five additional adjacent movement candidates.
These are required work and are being tested before closure: shared-down output
into its norm shard; QKV output directly in L1; cache cast into update shards;
final residual add consuming the norm shard; concatenated heads consumed
directly by the output projection. No later optimization stage was begun.

The `selected` candidate suite completed headline/equivalence/broad/pytest/watcher
and both Tracy commands with exit0. Its driver was paused while the full Tracy
child finished; the completed child exit status was read from /proc, then only
the orchestration driver was stopped before long-context jobs. No hardware
command was interrupted, no reset performed, and these profiles remain valid
intermediate evidence. This avoids rerunning long contexts before the review
controls are resolved. `review_boundary_commands.json` and
`review_boundary_adapted_commands.json` record their commands and statuses.

The new shared probe initially used an incorrect decode keyword, then its full
run intercepted the old source.down_proj after the wrapper gained its own
projection callable. Both test-harness errors are fixed and retained as
*_signature_error/*_interception_error.log; fresh retry captures the actual
wrapper call. Every tested K-block divisor passes component PCC/replay; k6 is
fastest (~44.75us versus62us), with unchanged HiFi2/BF16 projection policy.

The direct copy/typecast API rejects interleaved-to-sharded layout at
copy/typecast/device/typecast_device_op.cpp:110. The adapted unary_chain TYPECAST
accepts the geometry and passes full headline. A second adaptation using
interleaved_to_sharded(output_dtype=BF16) fails sliding decode PCC .991401.
Its factory's conversion kernel uses default ComputeConfigDescriptor (FP32
destination disabled) and copy_tile/pack_tile, unlike the SFPU cast. The full
real-weight failure rejects that conversion arithmetic for sliding; the passing
unary-chain implementation remains a candidate.

All combined controls pass the complete headline. Choose
boundary_combined_sliding_1 (5067.09us) and boundary_combined_full_1 (5520.47us).
Direct final add is faster in both coherent combined controls. Unary cache cast
is selected only for sliding; in full it is2.4–2.9us slower in the combined
graph. I2S dtype conversion fails full PCC .983611 as well as sliding .991401.
The final defaults are now frozen and verified_validation_commands.json records
each full rerun with the exact runtime source SHA256. Every meaningful skill
pattern and all five concrete reviewer movement findings have a measured
accepted/rejected outcome.

## Resume 2026-09-26

Attempt e73d7165-fc65-44f4-828b-1f979f0dcf75 resumes the frozen runtime
0c8892be32e04202fdbddd2b06850848c63ae57660c49d03c4d149a5a88b041c.
No model process remained. `timeout60 tt-smi -ls --local` showed all four ASICs;
`timeout60 python_env/bin/python` opening/closing MeshShape(1,1) printed
RESUME_MESH_SMOKE_OK. No reset needed. Installed environment.py validation
passed again. Only the interrupted sliding262143 and two unrun full context
cases are resumed; prior passing hardware evidence is retained.

Fidelity attribution correction: the reviewer found that supplying the shared
down program config without compute config changes TTNN inference from HiFi2
to LoFi, packer_l1_acc=True, FP32 destination disabled. Existing verified rows
and correctness gates validate this actual LoFi policy. Earlier claims that
shared-down fidelity was unchanged HiFi2 are superseded. The probe now exposes
explicit LoFi/HiFi2 controls over the same K divisors; matched checks follow.

The final context test revisits the last position after updating the preceding
cache row. Its two PCC values can differ because the state differs; the
functional control has the same behavior. Fixed-state headline/batch replay
equality establishes determinism. This is controlled, not a waived failure.
Resume pre-commit normalized trailing spaces in generated report tables only;
Python hooks passed and the frozen runtime SHA256 did not change.

### Resumed context completion

All four final verified maximum/near-maximum context tests pass on the unchanged runtime: sliding prefill PCC0.9981859749/0.9981839549 and full0.9962342265/0.9961807402 at262144/262143. Every end-context decode comparison also passes. Commands, exit codes and runtime hashes are in verified_validation_commands.json. context_contract.json retains every capacity field and adds the fused evidence links.

### Explicit shared-down fidelity control

The final profiler revealed inferred LoFi for the explicit sharded program, while the original interleaved baseline inferred HiFi2. The initial probe's HiFi2 description was wrong; it did not invalidate the measured outputs or times. Matmul program-config inference is the cause (matmul_device_operation.cpp:2808–2810). Four fresh real-activation controls explicitly set LoFi or HiFi2 with the same BF16 operands, FP32 destination disabled, packer L1 accumulation enabled, norm consumer, layout and K-divisor sweep. Every candidate passes PCC and repeated replay. K6 remains fastest under both policies:

| Kind | Interleaved HiFi2 baseline (us) | Sharded K6 LoFi (us / PCC) | Sharded K6 HiFi2 (us / PCC) |
| --- | ---: | ---: | ---: |
| sliding_attention | 62.031 | 44.718 / 0.9998896404 | 44.880 / 0.9999369429 |
| full_attention | 62.018 | 44.764 / 0.9999019428 | 45.018 / 0.9999455949 |

These are warmed traced host boundary timings, not whole-layer device times. The small fidelity difference is distinguished from the larger sharding improvement. LoFi remains the best measured correct configuration; no runtime change or headline reprofile is needed. Raw controls: shared_fidelity_{sliding,full}_{LoFi,HiFi2}.json; exact commands and exit codes: shared_fidelity_commands.json. Final whole_layer.json summaries and telemetry name the actual mixed-fidelity policy. Historical probe JSON is retained, with its stale attribution superseded here.

## Mixed expert-batch tail and inherited cache sensitivity

Fresh logical S65 pads to96 rows and exercises the64+32 expert batches. Both real-weight layer kinds pass FP32-HF prefill. Full attention also passes all8 traced decode outputs. Sliding fails FP32-HF at position68: fused PCC0.9813712959 and functional0.9809274231; other positions pass. This failed row is retained, not relabeled or excluded from the comparison.

The paired complete outputs remain equivalent: prefill PCC0.9991649499 and all8 decode PCC >=0.9997196429, including0.9998401882 at68. The functional stage probe isolates the discontinuity: attention/residual PCC >0.999997, but CPU routing on the TT residual selects expert70 instead of the FP32-HF oracle's eighth expert10. CPU attention with identical QKV reproduces those routes. A pure CPU HF control on the saved identical inputs reproduces the same sole failed position and expert swap by rounding only cached K/V to BF16 (PCC0.9814915924); BF16 cache plus rotary tables gives0.9815510238. No TT kernel or fused rewrite is needed to reproduce it. This is inherited BF16-cache sensitivity at a close MoE rank boundary; cache dtype/context contract remain unchanged.

The durable test_fused_mixed_expert_tail keeps all8 positions, full-HF prefill checks, unchanged0.995 direct functional-equivalence threshold, finite tensors, deterministic replay, runtime audits, program-cache guard and functional-fallback prohibition. It recognizes only the independently controlled sliding position68 FP32-HF failure. Existing33-token and4096/128 FP32-HF gates retain their original threshold without exceptions.

Artifacts: verified_tail65_*.json, tail65_sliding_{functional,fused}_paired.json, tail65_sliding_equivalence.json, tail65_functional_stages.json, tail65_cpu_precision.json, tail65_diagnostic_commands.json, shared_fidelity_commands.json, AUTODEBUG_tail65.md and verified_mixed_tail_pytest.{json,log}. Exact saved tensors/inputs are local .pt artifacts; compact JSON records the comparisons and CPU fixture hashes.

The targeted mixed-tail pytest passes both layer kinds (2 passed,2 deselected); verified_mixed_tail_pytest.json records command/exit0/frozen runtime hash. Original33-token fused-path cases retain their earlier verified passing run. No runtime code changed during the resumed controls.

### Final independent review and checkpoint

Fresh independent reviewer `/root/fused_final_review` returned **clean-pass** with no required work; final report is STAGE_REVIEW.md. The early report remains STAGE_REVIEW_INITIAL.md. All17 verified validation commands exit0 on frozen runtime SHA2560c8892be32e04202fdbddd2b06850848c63ae57660c49d03c4d149a5a88b041c; targeted mixed-tail pytest passes both cases. Matched fidelity controls, inherited S65 limitation and all anomalous evidence are classified. Applicable pre-commit hooks pass for all stage files; git diff --check is clean. Python/tests/docs only, so no C++ build was needed. No later-stage work or push was performed.

Creating the stage-only checkpoint on branch `gemma-4-26b-a4b-it` in `/workspace/tt-metal`; its exact SHA is recorded in the following metadata commit.

Stage checkpoint: `/workspace/tt-metal`, branch `gemma-4-26b-a4b-it`, commit `d627f0e22921c14895b39b9b9b7c59ebe115625d`. This follow-up records that SHA only; its own SHA is recorded in the external telemetry packet to avoid a self-referential commit. The first commit attempt found missing author configuration; retry used command-scoped `user.name=Codex`, `user.email=codex@openai.com`, matching the preceding stage, without changing user/global Git settings. All commit hooks pass. No push.
