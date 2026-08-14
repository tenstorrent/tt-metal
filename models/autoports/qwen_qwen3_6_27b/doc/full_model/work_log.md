# Qwen3.6-27B full-model work log

## 2026-08-13 — construction and reduced probe

Skills in use: `$full-model`, `$tt-device-usage`, `$tt-enable-tracing`,
`$multichip`, `$optimize`, `$qualitative-check`; `$autofix` is active for the
first hard sampler failure. No vLLM files or serving adapter work are in scope.

Starting branch `mvasiljevic/fmf/qwen-qwen3-6-27b`, HEAD `1fbeedfb654`.
Unrelated pre-existing untracked Tracy, triage, Falcon, and third-party paths
were preserved. Decoder inventory confirmed the selected TP4 contract and its
clean-pass artifact `doc/optimized_multichip_decoder/STAGE_REREVIEW.md`.

Hardware health was serialized:

```text
timeout 60 tt-smi -ls --local                 # four Blackhole p300c visible
open MeshShape(1,4), FABRIC_1D_RING, close    # MESH_SMOKE_OK
```

Implemented `tt/model.py` and `tt/generator.py`. The checkpoint reader opens
only the safetensors shard for each requested tensor. Full-model boundaries are
device embedding, the unchanged decoder stack, final RMSNorm, a padded TP4
vocab-sharded BFP8 LM head, and common split sampling. Cache and page-table
ownership are explicit. Host sampling is exposed only as a named compatibility
mode; optimized autoregressive token feedback is device-owned.

Reduced-probe command (four real prefix layers includes linear layers 0–2 and
full-attention layer 3, real terminal tensors, S5):

```text
build_generator(..., num_layers=4, max_context=128, batch=1)
generator.prefill_forward([[1,2,3,4,5]], prompt_lens=[5])
```

Result: `REDUCED_PREFILL_OK`, logits `[1,1,248320]`, all finite. The first
token-out attempt failed in common `ttnn.sampling`: gathered values had one
logical row while device-offset indices followed the sampler's 32-row tile
contract. Evidence is `logs/reduced_split_trace.log`. AutoFix is testing the
prediction that padding the decode terminal input/logits to 32 physical rows,
without changing logical batch, restores equal candidate shapes.

Fresh reference command (local snapshot pins revision exactly):

```text
python -m models.common.readiness_check.generate \
  --hf-model /huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --prompt-source aime24 --chat-template --gen-len 100 --top-k 100 \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/readiness_aime24_chat.refpt
```

The command completed. `reference_metadata.json` records the exact revision and
prompt contract; the reference has prompt `[1,161]`, continuation `[1,100]`,
and HF top-k `[100,100]`.

## Full-stack prefill accuracy

The first complete all-layer device run loaded all 64 official-weight layers
and passed logical S5 through embedding, the mixed linear/full stack, terminal
norm, and TP4 vocab-sharded LM head. Evidence: `logs/full64_s5_prefill.log`;
finite logits `[1,1,248320]`, measured post-load S5 TTFT 399.5 ms.

The standard readiness prefill runner then used the fresh AIME24 reference at
logical length 261 (161 prompt + 100 HF tokens), with `max_context=512` only as
the construction-time cache allocation for this accuracy workload—not as an
advertised context reduction:

```text
run_prefill_check(..., build_kwargs={"max_context":512,"batch":1})
```

Result in `logs/run_prefill_check.log`: top-1 90/100, top-5 100/100,
top-100 100/100. The terminal mapping and full-stack prefill accuracy clear the
required bar. Full traced teacher forcing is the active next gate.

This was the contemporaneous status of the initial prefill run; the final
teacher-forcing result is recorded below.

## Full-stack traced teacher forcing

The compatibility mode was made explicitly traced rather than eager. It binds
stable device token/position inputs, snapshots every persistent layer cache
before exact decode warmup, restores those tensors so warmup is not an extra
state transition, captures the all-layer logits path, and refreshes teacher
tokens/positions outside replay. This host boundary is intentional and confined
to compatibility tests; the measured optimized token-out path uses separate
model/sampler traces and device feedback.

Standard `run_teacher_forcing` on all 100 AIME24 tokens passed top-1 98/100,
top-5 100/100, and top-100 100/100. The runner reported TTFT 16.595 s (includes
first compatibility trace setup), traced decode 6.262 t/s/u, and 3.086 e2e
t/s/u. Evidence: `logs/run_teacher_forcing.log`.

## Final correctness, trace, qualitative, and capacity evidence

The terminal now uses device-local 30,976/31,104-column DRAM-sharded chunks
and 32-row sequence tiles. Final prefill is top-1 92/100, top-5 100/100,
top-100 100/100 (`logs/run_prefill_check_split_lm_head_v2.log`). Final teacher
forcing is top-1 97/100, top-5/top-100 100/100, TTFT 7663.13 ms, and 6.96
traced decode t/s/u (`logs/run_teacher_forcing_final.log`). Teacher forcing is
the explicit host-sampling compatibility path, not optimized token-out.

The standard 100-token HF and TT completions are in `autoregressive_final/`.
Both are coherent, English, non-repetitive, and set up the same distance/rate
equations; neither reaches the final answer inside 100 tokens. TT differs
stylistically after three exact tokens but has no wrong-language drift or
pathological early divergence. The machine checker reports zero adjacent
duplication, trigram-loop fraction 0.0526, and no degeneracy
(`artifacts/degenerate_output.json`).

AutoFix repaired common-sampler 32-row logits and the B2 full-attention gate's
multiply-boundary padding. Final artifacts include
`logs/autofix_split_trace_stable_alloc.log`,
`logs/autofix_mixed_b2_gate_pad.log`, and
`logs/reduced_trace_state_page_table_final.log`. The last records unchanged
page-table replays 2/refreshes 0, then replays 3/refreshes 1 after a real table
change, position `[8,3]` for active/inactive rows, and zero host token/position
refreshes.

The full-model physical probe reserves 13,460,453,888 weight bytes/device,
including embedding, norm, and LM head. B1 C262,144 passes. B32 C72,192 passes
and adjacent C72,256 fails (`artifacts/capacity/`). Public single-pass prompt
extent is 192,511, the inherited largest physical prefill pass; B1 cache/decode
context remains 262,144. No host or layer-stack chunking fallback is claimed.

Static verification: `py_compile` passes and 171 inherited decoder contract
tests pass in 38.63 seconds. The runtime audit is
`artifacts/runtime_fallback_audit.txt`.

## Stage-review AutoFix remediation

The review's context contradiction was verified. Public prompt capacity is now
consistently 192,511 in code, `doc/context_contract.json`, and README, backed by
the measured 192,511 pass / 194,559 contiguous-allocation failure. The separate
B1 decode-cache/absolute-position extent remains 262,144 and is no longer
described as public prompt capacity.

The generator now exposes public `setup_token_out_decode` and
`token_out_decode_step` methods. They accept explicit cache, page table,
positions, fixed-slot active mask, and common `SamplingParams`; replay returns a
device token by default and never returns logits. Greedy remains the measured
force-argmax mode, while the same public boundary can select common top-k/top-p.
Static contract tests are in `tests/test_full_model_public_contract.py`.

The earlier layer-0 mixed-prompt log was mislabeled as full-wrapper evidence;
README now says exactly what it covers. A reduced real-wrapper B2 probe with
embedding, layers 0--3 (both layer kinds), terminal, public sampler trace,
mixed S65/S63 prompts and an inactive slot is
`tests/full_model_mixed_slots.py`.

The qualitative metadata names the shared prompt source and prompt-correct
HF/TT runner. The serialized hardware command completed:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/full_model_qualitative.py \
  --max-new-tokens 50 \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/artifacts/qualitative/shared_suite.json
```

Manual review of all six cases found coherent, prompt-relevant English and no
repetition, language drift, prompt echo, control-token leakage, cross-request
leakage, or suspicious early divergence. Matched HF controls have the same
thinking-first style. See `artifacts/qualitative/verdict.md`.

Required B2 public-boundary evidence command:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/full_model_mixed_slots.py \
  |& tee models/autoports/qwen_qwen3_6_27b/doc/full_model/logs/full_model_mixed_slots.log
```

The initial run found the LM-head's B1-only physical shard-height assumption.
AutoFix proved the retained DRAM-sharded kernel only supports one 32-row tile;
the terminal now device-slices each slot, runs the unchanged projection, and
device-concatenates batch outputs. Final B2 mixed S65/S63 passes with public
temperature 0.8/top-k 5/top-p 0.9 sampling, token output `[12,220]`, and
positions `[66,63]` (inactive row unchanged). Evidence:
`logs/full_model_mixed_slots_lm_head_fix_v4.log`. B1 greedy regression also
passes two replays with zero host state refreshes.

Required reduced profiler evidence must be a separate non-Watcher run. The
current profiler harness accepts `--num-layers 4` and now renders the S128
prompt through the tokenizer chat template. Capture it with the checkout's
standard Tracy wrapper, preserve the generated device operations CSV under
`doc/full_model/artifacts/profile_reduced/`, then run both human and CSV forms:

```bash
python -m tracy -r -p \
  -m models.autoports.qwen_qwen3_6_27b.tests.full_model_perf \
  --prompt models/autoports/qwen_qwen3_6_27b/doc/full_model/aime24_chat_prompt.txt \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/artifacts/profile_reduced/perf.json \
  --prompt-tokens 128 --decode-tokens 16 --num-layers 4
tt-perf-report <device_ops_csv> \
  > models/autoports/qwen_qwen3_6_27b/doc/full_model/artifacts/profile_reduced/tt_perf_report.txt
tt-perf-report --csv models/autoports/qwen_qwen3_6_27b/doc/full_model/artifacts/profile_reduced/tt_perf_report.csv \
  <device_ops_csv>
```

The profile must classify LM head, all-gather, argmax/common sampler, trace
replay, and calculate `48 * linear_layer_ms + 16 * full_attention_layer_ms`
from the selected-policy decoder evidence before the stage can close.

Still unresolved without hardware evidence: shared-suite output review,
non-greedy public token-out replay, the B2 full-wrapper probe, reduced profiler
and sampler-dominance comparison, Watcher/runtime-integrity, and a correct
per-slot cache reset/reuse implementation. Whole-model `reset()` is safe; no
partial reset is claimed or fabricated because paged KV blocks and recurrent
row state require an exact on-device clearing probe.

Final synchronized `tests/full_model_perf.py` used S128 and 128 canonical
split-trace replays: TTFT 4404.73 ms, trace setup 0.794 s, token-out 17.1085
t/s/u. Model trace `MeshTraceId(0)` and the common sampler trace were both
live; counters were 128 replays, zero host token/position/page-table refreshes,
and 129 sampled-token readbacks (`artifacts/full_model_perf_b1.json`). A B1
C262,144 physical capacity rerun with `--trace-snapshot` also passes.

## Canonical sampler dominance repair

The first valid named reduced `tt-perf-report` is under
`artifacts/profile_reduced_v4/`. In the one-linear/one-full plus real terminal
replay, the original force-argmax contract spent 1318.959 us in ArgMax,
830.644 us in full-vocabulary async all-gather, and 92.786 us untilizing 32
padded rows. The comparable common candidate-gather path under
`artifacts/profile_candidate_greedy/` was worse: 9697.361 us in fixed top-32
and 13.274 ms wall time versus 5.533 ms force-argmax.

The selected common force-argmax path now slices logits to active fixed slots
before sampling, retains semantic global greedy all-gather, argmaxes only real
rows, and pads/copies only the tiny sampled-token result to the persistent
32-slot feedback tensor. Exact reduced tokens remain `[220,220,220]` with zero
host state refreshes. The final reduced profile is
`artifacts/profile_active_row_greedy/`: wall 4.264 ms, named device ops
3401.823 us; sampler all-gather 829.754 us (24.4%), while LM-head projections
total 1197.908 us (35.2%). ArgMax is no longer a dominant row. Thus sampler
ops do not dominate the selected token-out contract.

Inherited selected-policy B1 layer medians are 0.900798 ms linear and 0.595597
ms full attention. The decoder-stack lower bound is therefore
`48*0.900798 + 16*0.595597 = 52.7679 ms/token`, before terminal and sampling.

Final 64-layer S128/128 measurement after active-row greedy repair is TTFT
4419.88 ms and token-out 17.5168 t/s/u (57.088 ms/token), with 128 replays,
zero token/position/page-table host refreshes, and both traces live. Evidence:
`artifacts/full_model_perf_active_row_b1.json` and its log.

## 2026-08-13 AutoFix: inactive KV and per-slot reuse — incomplete

Implemented a kernel-supported inactive decode proposal using
`paged_update_cache`'s `INT32 -1` skip index while preserving real RoPE/SDPA
positions. Added selective linear-state reset, decode-before-refill rejection,
inactive cache-fill page-table entries, contiguous host uploads, and focused
B2 probes. Static public-contract tests pass, but no device cache assertion
completed.

Official-weight probes stalled during model setup in
`FDMeshCommandQueue::finish_nolock`: during tensor upload and, after the first
reset, during global-semaphore setup. `tt-triage` could not capture cores due
to an installed UMD `noc_read(..., memoryview)` API mismatch.

Two safe recovery sequences are exhausted. The second bounded list/reset/list,
mesh open/close smoke, and standalone fabric/global-semaphore smoke passed,
but the unchanged one-layer B2 repro again stalled in a mesh-buffer write
before decode. Post-failure board telemetry remained healthy. Full evidence is
in `doc/full_model/triage/AUTOFIX_INACTIVE_KV_RESET.md`.

Incomplete gates: exact inactive paged-KV preservation, active-KV mutation,
selective slot reset/refill with live-peer preservation, and rereview. The code
hypothesis is unproven and not refuted. Operator host reboot is required before
another hardware attempt. This result does not authorize stage completion or
a commit.

### Resumed source audit and lifecycle repair

A fresh source-only AutoFix audit is recorded in
`triage/CACHE_FIX_SOURCE_AUDIT.md`. It verified both kernel sentinel contracts:
`paged_update_cache` skips an update index of INT32 `-1`, and
`paged_fill_cache` consumes but does not write page-table blocks set to `-1`.
It also found an independent request-lifecycle bug: nonblocking token-out replay
could be followed immediately by trace release/reset without a completion
fence. The generator now synchronizes the mesh before releasing model/sampler
traces or their persistent snapshots, and whole-cache reset synchronizes before
returning. The cache update predicate is explicitly typecast to INT32 before
the INT32 `where`, removing the untested BF16-predicate/INT32-result variant.

`py_compile`, the two public-contract tests, and `git diff --check` pass after
these repairs. Hardware proof remains pending because the host boot identity is
unchanged (`uptime -s` is still `2026-06-09 23:29:58`) after the previously
exhausted two-reset recovery. The unchanged focused TP4 test must be the first
device command after an operator host reboot.

### Long-prefill request-lifetime repair

The source-only audit in `triage/LONG_PREFILL_SOURCE_AUDIT.md` confirms that
default public prefill selects one terminal hidden row per slot on device
before LM-head projection; it no longer allocates sequence-by-vocabulary logits
at long context. It found a separate request metadata leak: at S192,511 the
generator uploads 15,040 mask/selector tensors and every decoder retained those
lists after return; layers also retained a temporary inactive-fill page-table
handle after its deallocation.

The model now exposes an ordered `clear_prefill_request_state()` boundary. The
generator completes logits readback, fences queued consumers, clears all layer
aliases, and explicitly deallocates generator-owned token, position, mask,
selector, and temporary page-table tensors. A source regression exercises the
alias cleanup. Public-contract tests are now 3/3 pass, with clean compilation
and diff checks. The all-64-layer S192,511 public prefill plus subsequent decode
remains a mandatory post-reboot hardware gate; source analysis and capacity
placeholders do not substitute for it.

The required separate Watcher attempt is
`logs/full_model_watcher_reduced_final.log`. It fails before model construction:
Watcher instrumentation makes ACTIVE_ETH fabric firmware 27,920 bytes, above
the hardware kernel-config buffer limit 25,600. No single-chip/no-fabric
fallback was used. The final wrapper is covered by overflow-free named device
profiling and clean traced execution; inherited decoder Watcher evidence covers
the unchanged layer kernels.

### 2026-08-13 resumed inactive-slot hardware proof

Four p300c devices enumerated and a fresh `MeshShape(1,4)`
`FABRIC_1D_RING` open/close returned `MESH_SMOKE_OK`.  The official-weight B2
one-layer control passed for both `active=[1,1]` and `active=[1,0]`; the latter
proved inactive K/V bitwise unchanged on every rank and active K/V changed.
Evidence is `/tmp/qwen_full_attention_b2_all_active.log` and
`logs/full_attention_inactive_kv_final.log`.

The reduced real full-wrapper lifecycle then passed mixed S65/S63 prefill,
public top-k=5/top-p=0.9 sampling, traced feedback, inactive K/V exactness,
selective linear-state reset, decode rejection before refill, and slot refill
and reuse without disturbing the live peer:

```text
FULL_MODEL_MIXED_SLOTS_OK [198, 220] [68, 63]
INACTIVE_KV_EXACT RESET_REUSE_OK
```

Evidence: `logs/full_model_mixed_slots_reset_final.log`.  This closes the first
P1 finding in `STAGE_REREVIEW.md`; maximum-length public prefill and fresh
post-sampler qualitative evidence remain active gates.

### Greedy feedback regression and repair

Fresh post-active-row autoregressive evidence exposed a real regression: TT
returned newline token 198 for all 100 steps and the degeneracy checker marked
the completion near-empty.  Focused overwrite probes proved the sliced-logits
force-argmax trace did not update persistent `tt_out_tok`; the reduced profile's
natural repeated token had hidden the stale feedback.

Restoring the common sampler's shape-exact 32-row logits/32-slot output contract
changes a deliberately seeded token 123 to sampled token 220 after one replay
(`logs/feedback_overwrite_shape_exact_probe.log`).  The all-layer 8-token AIME
control now produces changing, coherent TT tokens
`[198,760,3377,16561,364,279,2702,854]`, decoded as “The problem asks for the
total time ...”; see `autoregressive_feedback_shape_exact_smoke/`.

This closes the correctness regression but reopens terminal performance work:
the previously measured 32-row force-argmax path is material.  Completion still
requires a faster semantically greedy contract that demonstrably overwrites
feedback, plus fresh 100-token/shared-suite evidence on that final contract.

### Active-row argmax root cause and repair

Source inspection identified why the prior tile-logit slice went stale.  The
multi-core argmax derives its inner row extent from the padded input shape but
its outer work from logical volume.  Slicing the tile tensor to B1 therefore
left a physical 32-row extent and produced zero outer work after untilization.
The selected repair crops only after full-vocabulary all-gather and untilize,
so argmax receives a materialized row-major B1 tensor while still writing into
the canonical 32-slot persistent feedback buffer.

The reduced overwrite probe now passes (`123 -> 220`) and reaches 228.38
t/s/u.  A complete all-64-layer S128 run reports TTFT 4.041 s and 17.43 traced
t/s/u over eight replays, with zero host token, position, or page-table
refreshes (`profile_active_rm_crop_full_smoke.json`).  Its deliberately
truncated prompt makes the emitted control tokens unsuitable as qualitative
evidence.  AutoFix reviewed the repair and required stricter exact-argmax and
B2 mapping checks; those were added to `tests/full_model_perf.py` and
`tests/greedy_sampler_active_rows.py`.  Fresh final qualitative evidence and
the isolated B2 sampler gate remain required.

The strengthened gates subsequently passed.  The isolated traced B2 sampler
returned distinct exact tokens `[17,118]` on every rank
(`logs/greedy_sampler_active_rows_b2_final.log`).  The model-integrated reduced
gate matched sampled token 220 to the gathered full-vocabulary torch argmax
220 while overwriting seed 123 (`logs/exact_argmax_reduced.log`).  The aligned
row-major writer may modify padding words outside the configured fixed-slot
prefix; those words are neither returned nor consumed by model decode.

The final AIME24 run generated TT output fresh on the repaired sampler while
reusing the existing HF completion only after proving an exact revision,
161-token prompt-ID, and 100-token generation-budget match.  Both completions
coherently restate the nine-kilometer setup and begin enumerating the two
scenarios.  TT diverges after token 3 but remains relevant English with no
repetition or wrong-language drift.  The standard checker reports zero
adjacent duplication, 5.26% trigram-loop fraction, and no degeneracy.  Exact
artifacts are under `autoregressive_active_rm_final/` and the command log is
`logs/autoregressive_active_rm_final.log`.

### Final full-context, profiler, and qualitative closure

The all-64-layer public wrapper passed non-aligned S192,511 prefill followed by
one decode after the full-stack streaming repair. Both results were finite and
all request-owned metadata aliases were cleared. Exact artifact:
`artifacts/full_model_long_prefill_s192511_final.json`; command log:
`logs/full_model_long_prefill_s192511_final.log`. The run uses six ordered
32,768-token-or-smaller chunks through the unchanged TP4 stack and preserves
the selected dtype/fidelity/cache/CCL and replicated residual policies.

The final repaired-sampler reduced Tracy run passed its seeded semantic-greedy
probe (`123 -> exact global argmax 220`) and measured 230.166 t/s/u over 16
replays with zero token, position, or page-table host refreshes. Tracy captured
raw device data but its host/device correlation postprocessor asserted on a
missing device op after capture. Direct analysis of the preserved raw CSV gives
median critical paths of 3.605 ms for model trace 0 and 1.024 ms for sampler
trace 1; sampler share is 22.1% of their combined device time and argmax is
59.6 us. Artifacts are under `artifacts/profile_active_rm_crop_final/`; the
postprocessor assertion is in `logs/profile_active_rm_crop_final.log`.

The exact-revision matched shared qualitative suite was rerun on final code:

```bash
python models/autoports/qwen_qwen3_6_27b/tests/full_model_qualitative.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/full_model/artifacts/full_model_qualitative_final.json \
  --max-new-tokens 50
```

All six TT outputs are coherent English, prompt-relevant, and non-repetitive,
with no wrong-language drift or control-token leakage. Three match HF for all
50 tokens; the remaining first divergences are at tokens 1, 4, and 28 and stay
semantically appropriate. Log: `logs/full_model_qualitative_final.log`.

### Stage-review remediation and semantic-greedy closure

Forced-streaming overlap at S65 now compares the ordinary and streaming paths
on the same reduced official-weight model. A linear-only run is exact across
terminal logits, subsequent decode, and eight cache/state tensors. The `[0,3]`
linear+full-attention run reports prefill PCC 0.999896, decode PCC 0.999980,
minimum PCC 0.999968 across 20 cache/state tensors, and identical greedy token.
Artifacts: `artifacts/streaming_overlap_linear_s65_final.json` and
`artifacts/streaming_overlap_linear_full_s65_final.json`.

The apparent all-layer semantic-greedy failures were a test-oracle defect.
Trace capture records the graph but does not execute it, while the probe read
`_trace_logits` immediately after capture. Executing and synchronizing the
captured model trace before reading its logits makes the reduced populated
probe exact (`expected_global_argmax=sampled=220`, seed overwrite true) in
`artifacts/semantic_probe_populated_reduced.json`.

An off-by-default gather diagnostic then compared host-composed input shards,
the captured post-all-gather tensor on every rank, and an eager gather. All
four ranks had identical top-eight values/indices and the same argmax 248046;
captured and eager sampling also returned 248046. Evidence:
`artifacts/full_model_gather_debug.json` and
`triage/SAMPLER_GATHER_AUTODEBUG.md`. The speculative top-k=1 alternative was
therefore refuted and removed.

The corrected all-64-layer S128 gate passed 128 trace replays with exact
semantic greedy (`expected=sampled=248046`), feedback overwrite, zero host
token/position/page-table refreshes, TTFT 4.036888 s, and 17.466620 t/s/u.
Artifact: `artifacts/full_model_perf_active_rm_final_128.json`; log:
`logs/full_model_perf_active_rm_final_128.log`.

### Maximum-context execution gate

The first S262,144 attempt was terminated externally after about 56 minutes
without a Python/TT exception or artifact. Cgroup OOM, OOM-kill, and memory
limit counters were zero, RSS was 4.19 GiB with 109 GiB host memory available,
and all devices were healthy. The proven S192,511 run took 107m34s, so linear
scaling predicts roughly 146 minutes at S262,144; the early termination is an
execution-session lifetime failure, not physical-capacity evidence. AutoFix
added unbuffered elapsed/peak-RSS heartbeats. The original process had in fact
survived its detached parent shell and completed after about 144 minutes. The
all-64-layer S262,144 result has finite `[1,1,248320]` terminal logits, greedy
token 248046, and complete request-state cleanup. Exact artifact:
`artifacts/full_model_long_prefill_s262144_final.json`; command log:
`logs/full_model_long_prefill_s262144_final.log`. Decode is deliberately skipped
at the exact maximum because the next position, 262,144, is outside the context.

### Final-review lifecycle and qualitative remediation

Fresh stage review found six generator-owned persistent trace inputs were
nulled but not deallocated. AutoFix added synchronized, identity-deduplicated
deallocation after all model/compatibility/sampler traces release. The
three-cycle representative TP4 regression reports byte-identical post-warmup
allocated/free/largest-contiguous DRAM values and all aliases cleared in
`artifacts/full_model_trace_lifecycle.json`.

The shared suite was extended to 200 tokens with the same exact revision,
tokenizer, chat template, prompts, and greedy settings. All six HF and TT
outputs remain coherent and task-specific with no repetition, wrong-language
drift, prompt echo, or control-token leakage. Both controls remain inside this
checkpoint's long visible reasoning preamble, while matching task facts and
plans (including the same correct French translation). Artifact:
`artifacts/full_model_qualitative_200_final.json`.

The leading chat controls in `full_model_perf_active_rm_final_128.json` are not
qualitative output: that benchmark deliberately truncates the 161-token
reference prompt to S128 inside its chat-template controls before timing.

### Independent final rereview

A fresh read-only `$stage-review` inspected implementation ownership, context,
accuracy, semantic-greedy tracing, allocator lifecycle, profiler, and every
HF/TT qualitative output. Verdict: `clean-pass`; required work: none.

Stage implementation and evidence commit: `42ff45ede5f` (local only; not
pushed). The following documentation-only commit records this handoff SHA.

### 2026-08-14 runner-side full-model gate remediation

The independent stage-6 checker reproduced with exit 1 because its recursive
artifact discovery treated two intentionally preserved negative probes as
current completion evidence. Both probes predate and directly motivated the
sampler-feedback repair documented above:

- `autoregressive_feedback_fix_smoke_v2` emitted newline token 198 for all
  eight steps before the shape-exact feedback repair;
- `autoregressive_post_sampler` emitted newline token 198 for all 100 steps
  during the regressed active-row force-argmax experiment.

The underlying runtime defect remains fixed and is contradicted by the later
canonical evidence: `autoregressive_feedback_shape_exact_smoke` changes token
at every step, while `autoregressive_active_rm_final` generates 100 tokens of
coherent English with zero adjacent duplication and no near-empty finding.
To preserve the failed probes as causal evidence without presenting them as
current `run_autoregressive` completion claims, their metadata files are now
named `autoregressive_meta.failed_probe.json`. The sibling logs, token IDs,
and decoded text remain intact. The authoritative runner check subsequently
scans only the healthy canonical artifacts.

A fresh independent `$stage-review` then inspected the implementation, final
outputs, accuracy, context, trace lifecycle, performance, profiler, fallback,
Watcher limitation, and failed-probe classification. Its persisted verdict in
`STAGE_REREVIEW.md` is `clean-pass` with no required work.
