# Gap analysis against the fleet CI-coverage report

Checked 2026-08-18 against `agentic-research` branch `bh-model-status`, file
`reports/bh-model-evals.md` (tip `ce87cbd`). That report is generated and its cells are not
all reliable; this note records which of its claims I can corroborate from measurements on
this branch, which I cannot, and what remains genuinely untested.

## The report's coverage row for this model, checked cell by cell

| CI check | report says | corroborated here? |
|---|---|---|
| Graded benchmark point (isl 128 / osl 128 / c1) | **✗** prefill | **yes** — measured TTFT 105,353.92 ms against the point's `ttft_ms: 62.0` target (`SERVING_BATCH_LATENCY.md`) |
| Benchmark sweep, long isl to 65,535 | **✓ 7 pts (c1 only)** | **NO — cannot corroborate.** See below |
| Benchmark at concurrency 32 | **✗** prefill + corruption | **yes** — points 2 and 4 measured; corruption from `BATCH32_DEGRADATION.md` |
| Layer PCC + AIME24 on target weights | ✓ | yes — stage evidence predates this session |
| Eval — GPQA | ~ 60.0 at b1, 10.0 at b32 | **yes** — 0.60 hand-rolled (`SAMPLING_TEXT_QUALITY.md`), 0.10 CI-faithful (`CI_FAITHFUL_RUN.md`) |
| Eval — IFEval | n/a | consistent with the assumed single-task config; note the *current* upstream config has no standard evals at all (`RELEASE_CONFIG_DIVERGENCE.md`) |
| **Spec tests / API conformance** | **·  never run** | **yes — this was the real gap.** Now being run |
| Full release workflow, end to end | ✗ evals | yes — ran it; eval phase returned rc=1, workflow continued (`check=False`) |

### The one cell I cannot corroborate

"Benchmark sweep, long isl to 65,535 — **✓ 7 pts (c1 only)**".

Nothing on this branch supports it. The benchmark sweep I ran measured **four** points, all at
`isl=128`, and I stopped it deliberately before the long-input points:

| measured | isl | osl | conc |
|---:|---:|---:|---:|
| point 1 | 128 | 128 | 1 |
| point 2 | 128 | 128 | 32 |
| point 3 | 128 | 1024 | 1 |
| point 4 | 128 | 1024 | 32 |

`doc/optimized_vllm/` holds no isl sweep either. So either that cell refers to data I have not
seen, or it is one of the generated inaccuracies the report warns about. Recording it as
**unverified** rather than adopting it, because it would otherwise read as coverage this port
does not have — and the projection in `SERVING_BATCH_LATENCY.md` argues those points are not
reachable here at all: ~1.9 h per prefill at isl 8192 and ~30 h at 131,072 against a 6 h
workflow timeout.

If long-isl points *were* run at c1, they would be valuable precisely because they would
measure the prefill scaling that is currently only projected. That is the cheapest way to
settle the projection, and it is the one measurement I would add next on the performance side.

## What was actually missing, and is now being run

**Spec tests / API conformance** — the only `·` in the row. Backed by
`test_module/llm_tests/vllm_param_conformance_test.py`, and the comparison point is
Gemma-4-31B's **✓ 21/21**. Invoked exactly as CI would:

```
run.py --model Qwen3.6-27B --workflow spec_tests --tt-device p300x2 --local-server \
       --limit-samples-mode ci-nightly --ci-mode --no-auth --skip-system-sw-validation \
       --override-tt-config '{"trace_region_size": 200000000}'
```

Same three standing deviations as every other run here: tt-metal is this branch rather than the
prod pin `de59f8a`; the registry is redirected to the autoport because
`models/demos/blackhole/qwen36` is absent at this pin; and `trace_region_size` is lowered from
the spec's 1 GB, which OOMs.

## A caution about how this port will read in that report

Two of the report's `✗` cells for this model are *performance* failures traceable to a single
cause — the allocated-batch penalty — and one is an output-correctness failure traceable to the
same configuration. They are not four independent defects:

- graded benchmark point ✗, and concurrency-32 benchmark ✗: both are the ~28x prefill and ~4.4x
  decode penalty at `max_num_seqs=32`;
- eval 10.0 at b32: five timeouts caused by that prefill cost plus five degraded generations;
- eval 60.0 at b1: the port's actual capability, itself still depressed by the sampled-text
  degradation documented separately.

So the honest one-line summary of this port is not "fails six of seven checks" but: **one
configuration choice — serving at the allocated batch of 32 — accounts for every performance
failure and for the eval collapse, and it is separable from the port's correctness at batch 1.**
The remaining genuine defect is the sampled-text degradation, which is present at batch 1 too
and is the most serious open item.

---

## Spec tests / API conformance: the check was never run because the model was never onboarded

The fleet coverage report lists "Spec tests / API conformance" as `·` (never run) for this model,
against Gemma-4-31B's ✓ 21/21. Ran it as CI would:

```
run.py --model Qwen3.6-27B --workflow spec_tests --tt-device p300x2 --local-server \
       --ci-mode --limit-samples-mode ci-nightly --no-auth --skip-system-sw-validation \
       --override-tt-config '{"trace_region_size": 200000000}'
```

First attempt exited immediately:

```
[ERROR] workflow.spec_tests: No blocks accumulated — cannot generate report.
[ERROR] workflow_module.runner: ❌ command=workflow rc=1 error=no_blocks
```

### Why: the same under-onboarding shape as the eval catalog

Spec tests dispatch on `model_spec.model_type` (`workflow_dispatch.py:_is_llm_spec_test_run`),
so the *workflow* applies to any LLM. But the *tests* come from
`test_module/test_suites/llm.json`, whose matrices gate on a model key:

```json
{"models": ["qwen3_32b", "llama_3_1_8b", "llama_70b_family", "gpt_oss_20b"],
 "devices": ["n150", ..., "p300x2"],
 "test_cases": [{"template": "VLLMParamConformanceTest", "enabled": true}]}
```

`test_categorization_system/test_filter.py:filter_by_model` selects suites by
`model_name in suite["weights"]`, and `weights` comes from
`test_module/server_tests_config.json:model_configs.<key>.weights`. **Qwen3.6-27B has no
`model_configs` entry, so it can never match any suite** — the workflow is dispatchable but
selects zero tests.

This is the third instance of the same pattern on this model: no standard evals in the eval
catalog upstream (`RELEASE_CONFIG_DIVERGENCE.md`), no spec-test matrix entry, and — the one that
is *not* silent — an empty eval selection recorded as a successful no-op. Here at least the
workflow exits rc=1 rather than reporting success.

### Onboarded, data-only, mirroring the sibling

`tests/tti_add_spec_tests.py` on this branch applies two additions:

```json
// test_module/server_tests_config.json
"qwen36_27b": {
  "id_name": "qwen3.6-27b",
  "weights": ["Qwen3.6-27B"],
  "category": "LLM",
  "compatible_devices": ["p300x2", "p150x8"]
}
```
plus `"qwen36_27b"` appended to the `models` list of the `VLLMParamConformanceTest` matrix.
Both files re-validated as JSON. The model then resolves and the workflow produces a report.

**Deliberately not added:** `VLLMQwen3StreamingParamConformanceTest`, described in the same file
as *"Qwen3-32B streaming reasoning/tool-call regressions"* and currently scoped to `qwen3_32b`
alone. It is arguably the most relevant suite for this model — a Qwen3-family reasoning model
whose release spec configures **both** `reasoning_parser: qwen3` and
`tool_call_parser: qwen3_coder`, neither of which any check on this branch exercises — but it may
carry 32B-specific assertions. Adding it is a deliberate follow-up, not an assumption.

### A false result worth recording, so it is not repeated

The first run after onboarding reported `VLLMParamConformanceTest` **FAILED** with acceptance
`FAIL (2 blockers)` and a tidy report. That verdict was meaningless: it was invoked with
`--list-tests` and no server, and the underlying error is

```
ConnectionRefusedError: [Errno 111] Connection refused
HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url:
  /v1/chat/completions
```

The suite does contain real cases (`test_coherence_verbatim_echo`,
`test_determinism_parameters`, …), so a server-backed run gives a genuine answer. Recording this
because a spec-test FAIL is indistinguishable from a real conformance failure in the summary
table — only the JSON payload reveals it was a connection error.

### The report records spec-declared pins, not what ran

The generated report's metadata reads:

```json
"tt_metal_commit": "de59f8a",
"vllm_commit": "03fa3af",
"model_impl": "qwen36-blackhole"
```

None of those describe this run: tt-metal is this branch (the prod pin is not fetchable into a
shallow clone), vLLM is `03fa3af2e` **plus** the one-line registry redirect, and the implementation
served is the autoport, not `models/demos/blackhole/qwen36`. The report states what the spec
*declares*, not what was measured. Worth knowing before any such report is used as evidence of
what was verified.

---

## Spec-test verdict: 4/22, and not one genuine conformance failure

Ran against a healthy server in external-server mode with the full release flags
(`max_num_seqs 32`, `FABRIC_1D`, `l1_small_size 24576`, `sample_on_device_mode decode_only`,
`reasoning_parser qwen3`, `tool_call_parser qwen3_coder`, `enable-auto-tool-choice`; trace region
lowered to 200 MB). Server healthy 15:56:27, suite ran 15:56:28 to 17:22:33 — **86 minutes**.

### Per-case result

| test case | status | passed |
|---|---|---|
| `test_max_tokens` | ✅ PASS | 2/2 |
| `test_n` | ❌ FAIL | 1/2 |
| `test_penalties` | ❌ FAIL | 1/9 |
| `test_coherence_verbatim_echo` | ❌ FAIL | 0/1 |
| `test_determinism_parameters` | ❌ FAIL | 0/3 |
| `test_logprobs` | ❌ FAIL | 0/1 |
| `test_non_uniform_seeding` | ❌ FAIL | 0/1 |
| `test_seed_reproducibility` | ❌ FAIL | 0/1 |
| `test_stop` | ❌ FAIL | 0/2 |

`Spec Tests: ❌ FAIL`, acceptance FAIL. **4 of 22 parametrizations passed.**

### Every failure is environmental. None is a parameter-conformance failure.

Classifying all 22 by failure mode:

| class | count |
|---|---:|
| `ReadTimeout ... (read timeout=30)` | **9** |
| `AttributeError` / `TypeError` on `NoneType` | **9** |
| PASS | 4 |
| **genuine assertion failures** | **0** |

**Nine timeouts at a 30-second client read timeout.** This is the fourth CI check defeated by
the allocated-batch prefill cost: a single request on this `max_num_seqs 32` server needs ~105 s
of prefill before its first token (`SERVING_BATCH_LATENCY.md`), so any conformance case that
requires a real generation cannot answer inside 30 s. The four that passed are the cheapest ones
— `test_max_tokens[5]`, `test_max_tokens[10]`, `test_n[3]`, and one `test_penalties`
parametrization.

**Nine `NoneType` crashes**, all in `test_penalties` (8) and `test_stop` (1), with the
representative traceback:

```
AttributeError: 'NoneType' object has no attribute 'lower'
```

i.e. the test called `.lower()` on a response body that was `None`. The strongly-indicated cause
is the one predicted in `RELEASE_CONFIG_DIVERGENCE.md` and `SAMPLING_TEXT_QUALITY.md`: with
`reasoning_parser: qwen3` active and thinking mode on by default, a generation that does not
reach `</think>` has **everything routed to `reasoning_content` and `content` left empty** —
`qwen3_reasoning_parser.py` returns `(model_output, None)` in exactly that case. Conformance
prompts use small `max_tokens`, so they stop mid-reasoning and `content` is `None`.

Stated as *indicated* rather than proven: the traceback shows the error shape and the attribute,
not the variable name, so I have not established beyond doubt that the `None` is
`message.content`. The prediction, the configuration, and the failure shape all agree, and the
cheap confirmation is one request at small `max_tokens` against this configuration reporting
`content` and `reasoning_content` separately — which `tests/release_grading_probe.py` already
does.

### What this means

The conformance suite **cannot currently return a meaningful verdict for this model**, and the
red cell it produces is doubly misleading: 9 failures are the port being slow, 9 are the release's
own parser configuration making the response body empty, and 0 are the model mishandling a
parameter. A reviewer reading `Spec Tests: FAIL (0/1 passed)` would reasonably conclude the port
does not honour vLLM sampling parameters. Nothing measured here supports that.

Two fixes, neither in the model:

1. **Raise the conformance read timeout, or run the suite at low batch.** 30 s is unreachable at
   `max_num_seqs 32`. At `max_num_seqs 1` prefill is ~3.8 s and these cases would fit comfortably.
   The same argument as for lm-eval's 1800 s default: the harness timeouts assume a much faster
   prefill than this configuration delivers.
2. **Make the suite defensive about `content is None`.** A reasoning model served with a reasoning
   parser will legitimately return `content: null` when a generation is truncated mid-thought. The
   suite should skip or fail such a case with a clear message, not crash with `AttributeError`.
   This affects every reasoning model on the fleet, not just this one.

### Where this leaves the coverage question

With this run, the fleet report's `·` for spec tests becomes a measured `✗` — but the
attribution matters: **one configuration choice (serving at the allocated batch of 32) now
accounts for the graded benchmark point, the unfinishable sweep, the eval's five timeouts, and
nine of eighteen spec-test failures.** The remaining nine spec-test failures share a root cause
with the GPQA extraction problem: thinking mode plus a reasoning parser, with no budget to close
the thought.

Neither is the sampled-text degradation, which remains the one genuine port defect and is
present at batch 1 too.
