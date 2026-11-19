# Final request compatibility audit

CPU-only source inspection, 2026-11-16; no TTNN imports, devices, server requests,
or implementation edits. This is a compatibility forecast, not passing P2/P3
or sampling evidence. Plugin source is
`/home/ttuser/qwen38-flash/vllm-tt-plugin` at
`042a94863ad27d88207bcf71abdf67122c6f4c27`. Installed upstream source is
`/home/ttuser/qwen38-flash/vllm-env/lib/python3.12/site-packages/vllm`.
The readiness package root is
`/home/ttuser/qwen38-flash/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.24`.
Paths below are relative to those named roots, except autoport paths.

## P2 routing and acceptance

The unmodified `runtime/readiness_check/eval_request_profiles.json` sends
`lm_eval_reasoning` with temperature 1, top-p .95, top-k 40 and seed 42.
`tt/generator.py:48-58` declares `max_device_top_k=32`. Therefore this profile
**must use the plugin host sampler** with the present generator. Setting
`sample_on_device_mode=all` requests device sampling when supported; it does
not disable fallback. Plugin `src/vllm_tt_plugin/model_runner.py:3983-4082`
returns false for stochastic top-k above the model ceiling, penalties when
unsupported, host-only controls, structured output without device grammar,
and all logprobs on this four-device mesh. The decision applies to the entire
submitted step, so a top-k40 companion can move seeded top-k20 requests to the
host mid-request.

| Profile | Effective route with checkpoint defaults | Reason |
| --- | --- | --- |
| plain | device | temperature 0 |
| lm_eval_reasoning | host | explicit top-k40 exceeds32 |
| lm_eval_non_reasoning | device | greedy; seed42 stays device under policy auto |
| model_card | device | temperature1/top-p.95/top-k20 |
| agentic | device | omitted top-k inherits checkpoint top-k20 |

Do not silently replace top-k40 with32, declare a wider ceiling without a
correct wider canonical sampler, or change the installed profile. The selected
checkpoint's `generation_config.json:9-11` needs no operator override.

P2's actual gates are request acceptance and measured TPOT ratio, not a
standalone assertion that every profile used the device. See
`run_harness_requests.py:236-313`: `accepted_lm_eval_reasoning` is critical and
not waivable; the ratio at `load_users` is critical and defaults to <=1.15;
the 1-user ratio is advisory unless it is the selected load width. Defaults
are users1,32, 128 output tokens, seed policy auto. The probe caps the harness
output budget to its requested probe length; the32768+1024 context-fit check
is advisory and fits the262144 contract. If physical capacity requires smaller
concurrency, supply and document `--users 1,N --load-users N`, without calling
that32-user coverage.

The top-k40 host route is a concrete risk for the critical ratio, not proof of
a failure before measurement. The skill's `references/eval-readiness-requests.md`
permits a recorded host cost/profile constraint; after the prescribed two-hour
fix time box it allows a supported limitation for a waivable ratio. Acceptance
of `lm_eval_reasoning` cannot be waived. The separate user requirement to avoid
host fallback in performance still applies: headline and CI greedy performance
must retain device sampling and be audited separately.

## Optional host compatibility and full sampling suite

No extra standalone `HostSampler` should be added to the adapter. For a host
route the plugin omits `sampling_params`; the adapter sets
`sample_on_device=False`, and generator output is full host logits `[B,1,V]`.
Plugin `model_runner.py:4340-4444` owns `SamplingMetadata`, penalties, seeded
generators, logprobs and the host sampler. Existing boundaries are
`tt/generator_vllm.py:108-154` and generator `read_decode_output` /
`process_decode_output_host`. This compatibility route is explicitly selected
by the requested unsupported feature, with capability declarations documenting
its boundary; it is optional per request while ordinary requests stay device.
Do not globally set `seeded_sampling_policy=host` to repair determinism.

**No missing device-logprobs translation is required on TP4.**
Plugin `model_runner.py:4062-4080` sends even logprobs0 to host unless the
device count per DP rank is8 or32. The generator must return logits without
trying `set_sampling_params` with `enable_log_probs=True`; the current adapter
already follows that rule. The device-logprob tuple sanity check at4476 is
unreachable for a correctly configured TP4 logprob request. Verify actual
logprob values and completeness through the full suite; do not fabricate
device metadata. `tests/tt/test_logprobs.py:12-81` covers0/1/3/5/10 alternatives;
the all-vocabulary case can skip for an explicitly reported server cap.

`run_vllm_server.py:433-459` selects the entire plugin `tests/tt/` directory for
`--sampling-profile full`. Device/host routing transitions remain a concrete
seeded-isolation risk; capture admissions and routes when mixed tests fail.
An intentional host route for penalties/logprobs/top-k40 is not permission to
classify correctness or missing-logprob failures as reproducibility-only.

**Logging defect found by a pure CPU check:** current adapter JSON lines use
`"sampling_mode": "host"` or `"sampling_mode": "device"`. P2's regexes at
`run_harness_requests.py:79-80` allow at most three non-word characters after
`mode`; normal JSON has four (`": "`). Both counters therefore report zero
for the current events. Reproduced by extracting those regex literals with
Python AST and applying them to `json.dumps` events: host0/device0 for both.
Add a plain adapter line such as `sampling_mode=host` / `sampling_mode=device`,
or compact JSON separators, and retain per-step event accounting. Do not edit
the installed probe or equate its present zero with zero fallback. Intermediate
prefill chunks deliberately request logits without a final sample; label them
separately when interpreting host-mode forward counts as sampled-step counts.

## P3 parsers and template

Use the checkpoint template and these additional server flags:

```text
--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

`qwen3_coder` is an alias for the same tool parser, but `qwen3_xml` describes
this checkpoint's syntax directly. Registry entries:
`reasoning/__init__.py:143-146`, `tool_parsers/__init__.py:185-192`.
`parser/qwen3.py:1-16,39-49,204-256` implements `<think>...</think>` plus
`<tool_call><function=...><parameter=...>` and reads `enable_thinking`.
The checkpoint at
`/home/ttuser/qwen38-flash/hf/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/812d96e10a0dd82b69d749e68693698cf6c36daa/chat_template.jinja`
uses exactly those markers at68,118-145,169-172.

The template supports `enable_thinking=false` and defaults to thinking on with
`reasoning_effort=xhigh`; allowed efforts are xhigh/medium/low (lines46-55).
P3 may explicitly use
`--thinking-on-kwargs '{"enable_thinking":true,"reasoning_effort":"low"}'`
and `--thinking-off-kwargs '{"enable_thinking":false}'` to exercise the genuine
switch with bounded thinking; record those kwargs and keep matching template
controls. This changes probe request settings, not served defaults. Do not
claim a truncated unfinished thought as a completed answer.

No preemptive template patch or custom parser is indicated. The template
handles text content parts, initializes missing historical reasoning to an
empty string (lines111-119), and serializes nonstring tool arguments as JSON
(137-142). The installed parser's schema coercion
(`parser/engine/parser_engine.py:244-295,366-405`) preserves string-only fields
while converting nested object/boolean/array parameters, matching P3 c04's
JSON-looking-string and nested-options checks (`run_chat_conformance.py:236-297`).

Run all P3 cases on the final all-layer server. c01/c02 are nonwaivable;
tool calls/results, reasoning history, continuation, stream parsing and4K
system-prompt acceptance require real responses. c02/c11 require reasoning
in `reasoning`/`reasoning_content` with no tags leaked into content. Current
source inspection supplies no runtime verdict for any of these.
