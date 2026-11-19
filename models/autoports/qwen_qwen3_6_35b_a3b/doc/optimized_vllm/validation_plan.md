# B4 candidate validation plan

CPU-only follow-up inspection on 2026-11-17. No implementation/test edits,
device commands or server requests were made by this auditor. Paths below are
relative to the model root. This plan does not claim any proposed check passed.

## Candidate and invariant boundary

The proposed adapter declares prepared physical prefixes2/4/32 instead of2/32.
Model math, sampling widths32/64, max sequences32, context262144, trace
reservation805306368bytes, cache geometry, async scheduling and primary
benchmark configuration stay fixed. A bucket is selected by highest active
physical slot plus one. Three low slots should useB4; a single slot>=4 still
needsB32. There should be five retained traces, with no request-time capture,
retirement or program-cache growth.

The minimum acceptance has three distinct layers: reduced exactness and
lifetime, all40 source-matched correctness, and real plugin mixed traffic.
One cannot substitute for the others. Performance runs disable tracking and
detailed audit; tracked validation enables tracking1/skip0 before TTNN import.
No profiling is needed or permitted.

## Current reduced discriminator

`tests/run_vllm_bucket_probe.py --four-row-bucket` now constructs2/4/32 and
compares each candidate trajectory against retained fixed32 on the same
generator. It uses real layers0/3, context1024, capacity32, all-chip cache
snapshots, complete active-vocabulary logits, tokens, positions and seeds.
Greedy and seeded top-k8 schedules cover2→4→32→4→2, physical owners0/1/3/17,
an inactive hole, unchanged-table resident steps with unreadable token/position
sentinels, two immutable pending outputs, and exact dormant-state preservation.
It checks constant program entries, captures, trace IDs and zero workload
releases. This is already a strong focused discriminator.

Precise small additions/corrections recommended before final evidence:

- Update the module docstring, which still claims only2/32, and restore
  `generator.decode_buckets = buckets` at completion rather than hardcoded2/32.
  The latter does not release traces today, but obscures the tested final state.
- Exercise every B4 row, especially slot2, and all four active rows. Current
  owners0/1/3/17 skip slot2 and never have all0/1/2/3 active. A short case with
  prepared slot2 state plus `(0,1,2,3)` then `(2,)` checks the new row boundary.
  Preserve at least one separately prepared high-slot owner for B32 return.
- Changed page mapping and true state permutation are explicitly absent from
  this script. Add those to the existing adapter/stale-input probe instead of
  pretending host-authoritative layout changes are a remap test. Use a live
  required logical page, not an unused padded page; compare to fixed32 with
  the same physical mapping. Assert page-only change does not refresh tokens
  or positions. Keep unchanged-table counter assertions.
- Add one native64 sampled case (top-k40 or64) for the new model bucket plus
  a return to greedy; the existing seeded top-k8 case only exercises native32.
  The canonical sampler kernel itself is unchanged, so a complete new sampler
  component sweep is unnecessary. Retain the existing dormant-wide selector
  regression and ensure it remains valid with B4.
- Reuse the existing same-environment allocation-tracker positive control;
  keep its concrete artifact alongside the new clean run. Every workload must
  end with a model replay so late allocations cannot escape validation.

No timing from synchronized cache snapshots should become a speedup claim.
The existing negative limitations in the JSON correctly separate this probe
from automatic allocator growth and actual async overlap performance.

## All40 logits and served output

`tests/run_vllm_logit_determinism.py` currently hardcodes2/32 at construction,
expected trace keys, normal selection, and cleanup restoration. Add the same
optional bucket list/four-row switch consistently. With2/4/32, compare the
three existing fixtures129/130/131 in forward/reverse row orders against forced
B32 and soloB2, retaining exact full-vocabulary comparisons and fixed32
reference. Add an all-four-row case if reduced coverage alone is insufficient
for the changed full-attention projection shape; do not silently report the
three-row case as full physical-row coverage.

The HTTP mode sends logprob requests and compares12 responses against the
standalone compact reference. This is explicit host-logit compatibility
evidence, not on-device sampling performance. Its source-identity gate hashes
all `tt/*.py`; regenerate the standalone artifact after the adapter change,
do not edit old source hashes to make them match. The HTTP concurrent3 case
should now executeB4; confirm that from actual decode-submitted events, since
simultaneous client submission alone does not guarantee scheduling overlap.

Preserve shared qualitative outputs with checkpoint-default chat formatting,
and fresh full sampling coverage. The reduced exactness check cannot prove
all40 text quality or the real plugin's route decisions.

## Mixed-shape workload and summary changes

`tests/run_vllm_lifecycle.py` currently runs a solo129, a3-request overlap
(130 already decoding plus a long prompt and131), a reordered3-request burst,
sampled130 and greedy129 again. Its existing configured long length517 is
too short to guarantee scheduler chunking with budget2048; use the existing
stage9 long-prompt override above2048, then assert actual nonzero continuation
offsets, final tail and intervening decode events. Keep unseen lengths129/130/131.

With new buckets this workload will naturally exerciseB2/B4, **not B32**.
Keep the3-request phase for direct B4 mixed-latency evidence. Add a short
5-request phase with sufficiently long outputs to have five simultaneously
resident sequences, then a repeated solo or3-request phase. Use distinct
prompt histories and change arrival order. This creates a real2→4→32→4/2
opportunity without increasing `max_num_seqs`. Record the actual active slots
and selected trace; do not declareB32 coverage from concurrency intent.

Precise summary changes in `tests/summarize_vllm_lifecycle.py`:

1. Replace hardcoded `both_decode_buckets_executed` over2/fixed with
   `all_prepared_model_buckets_executed`, deriving expected keys from capture
   signatures/model traces (exclude sampling keys). Assert replay delta>0 for
   model_2,model_4,model_fixed in this expanded workload.
2. Replace `len(bucket_transitions)>=3` as the sole return proof. Require an
   earlier signature to recur after a different one, plus explicit coverage
   of transitions entering/leavingB4. Three distinct one-way transitions are
   not necessarily a return.
3. Keep `one_model_replay_per_submission`; additionally derive expected
   physical bucket from `active_slots` and capture `physical_rows`, and compare
   the observed key. Report rows and bucket per submission.
4. Add completion-length equality and nonempty observations to the summary.
   The request script already checks requested count, but summary currently
   accepts HTTP200 alone and several `all(...)` expressions can pass empty
   collections. Require decodes, submitted events, known repeated prompts and
   both sampling widths when top-k40/64 is requested.
5. Preserve baseline traces captured before workload, unchanged total capture
   count5, zero workload release, one capture per actual signature and stable
   IDs. Existing code compares `len(captures)` from the entire server log;
   summarize before teardown or filter teardown separately as already intended.
6. Keep page-only preservation, allocator-growth events, two pending outputs,
   concurrent decode between actual continuation chunks, latency percentiles
   and completed-request throughput. Require returned native32 after the
   native64 request from sampling replay deltas/event sequence, not only that
   both widths ran sometime.

One tracking-enabled lifecycle run proves allocation/lifetime. Run the same
client workload with tracking/audit performance overhead disabled for the
mixed-traffic TTFT/ITL result; event evidence can come from a separate scoped
audit run. Fixed128 benchmarks do not substitute for first-encounter mixed
latency. Preserve baseline/after request payloads exactly when comparing
mixed-traffic speed; if expanding the baseline workload is necessary, measure
both shapes against the matching source/config rather than mix workloads.

## Page growth and async comparison

`tests/run_vllm_growth_control.py` already sends three distinct129/130/131-token
prompts with160/112/80 generated tokens, repeats reverse admission, and compares
greedy tokens to a non-overlapped control. Those lengths cross multiple page32
boundaries and now exerciseB4. Update its descriptive config string from2/32
to the actual bucket list. Keep the fixtures and sampling unchanged so a new
2/4/32 overlapped run can compare with the trustworthy existing2/32
non-overlapped control **if** the new all40 B4-versus-fixed32 exact check has
passed. Label the old control's source/buckets explicitly; changing the
description must not relabel old execution as new.

The helper itself does not inspect allocator or token-position inputs. Pair it
with server events proving new required logical pages, active slots, unchanged
token/position refresh counters on page-only updates and changed table copies.
If new B4 execution is absent in the actual log, that run does not cover it.
An unexplained difference requires the focused current-source non-overlapped
rerun and debugging; do not waive it as scheduling or rounding noise.

## Sampling and checkpoint thinking defaults

Authoritative scope is `doc/vllm_integration/owner_decisions_20261117.md`:
the owner explicitly rejected direct-answer server default and accepted only
the recorded constrained-thinking corpus/recipe/seeds/cap limitation for
stage9. `tests/run_vllm_server.sh` now omits default-template overrides and
preserves checkpoint thinking-on. Keep that setting for every stage10 server.
Do not restore the old thinking-off launch flag or change generation config
to make tests easier. The old68-pass full-suite record remains historically
thinking-off and cannot become fresh default-on evidence by copying files.

Stage9 affected validation ran these three original canonical tests against
the default-on server with an explicit per-request `enable_thinking=false`:
`test_host_only_params.py::TestHostOnlyParameters::test_bad_words` and the two
tests in `test_structured_output_dp1.py`. Its model-owned pytest helper
`tests/checkpoint_default_test_requests.py` modifies **every chat request in
any test where it is loaded**, not just those three node IDs. Therefore do not
load it blindly across the full suite and call the result unchanged defaults.

Recommended fresh sampling evidence:

- Run the shared runner's full canonical suite against the final default-on
  server first, or run the known three separately from all other collected
  nodes if avoiding an already understood short-answer/template mismatch.
  Preserve exact collection, command, counts and default request provenance.
- For the three affected short-answer cases, reuse the existing narrowly
  invoked helper with `QWEN_CHAT_TEST_REQUEST_LOG` and retain all101 request/
  response records if its capacity workloads are unchanged. Assertions,
  caps, seeds, constraints and concurrency stay canonical. This tests explicit
  request mode, which is a supported API feature; it does not test omitted
  thinking mode. The other canonical cases should run without that helper.
- If a single full-suite invocation is needed, first scope a **model-owned**
  helper by exact node IDs so it adds the existing request field only to these
  three cases. Log each override. Do not edit shared tests or plugin source,
  weaken assertions, add broad skips, or silently apply overrides to all chat.
  This remains “full canonical test selection with three explicit-request-mode
  cases,” not “unmodified default-on full canonical suite.”
- Separately rerun `check_vllm_checkpoint_default` (omitted/on/off rendering and
  token comparisons), default qualitative outputs and affected P3 checks on
  the same final server. Scan route counts/OOV/errors per job; explicit host
  compatibility tests legitimately differ from device-only performance runs.

The current user asks sampling tests to pass. A failure outside the known
short-answer template assumptions, any crash/logprob/OOV bug, or B4-dependent
wrong output is required work. Stage9 owner acceptance is not a blanket waiver
for new failures or new quality scopes. Present the exact composite evidence
and its provenance to the fresh stage reviewer; never declare an unchanged
full-default pass if the measured invocation used targeted request overrides.

For reproducibility, the accepted targeted command shape is:

```bash
QWEN_CHAT_TEST_REQUEST_LOG=<persistent-path>/explicit_requests.jsonl \
PYTEST_ADDOPTS='-p vllm' timeout 1800 python -m pytest -v \
  -p models.autoports.qwen_qwen3_6_35b_a3b.tests.checkpoint_default_test_requests \
  ../vllm-tt-plugin/tests/tt/test_host_only_params.py::TestHostOnlyParameters::test_bad_words \
  ../vllm-tt-plugin/tests/tt/test_structured_output_dp1.py \
  --tt-server-url=http://localhost:8000 \
  --tt-model-name=Qwen/Qwen3.6-35B-A3B --tt-max-num-seqs=32 \
  --junitxml=<persistent-path>/explicit_sampling.xml
```

This is a proposed command, not an execution record. The full suite's canonical
command comes from `readiness_check.run_vllm_server._run_plugin_sampling_tests`;
it invokes the standalone plugin `tests/tt` directory with the same three
`--tt-*` options. Keep the runner's raw stdout and exit status even when
supplemental explicit-mode evidence resolves a scoped assumption.
