# AutoDebug: intended top-64 accuracy sampling

Status: device-policy mismatch verified by the parent host-only probe; request-specific host fallback is supported by inspected source and its routing check, but no API validation or benchmark has run.

## Contract and cause

Accuracy requests require the pinned generation policy temperature=1, top_p=0.95, top_k=64. `models/common/sampling/generator.py:607-612` clamps every top_k above 32 to 32. The autoport constructs its common sampler with max_top_k=32 (`tt/generator.py:80`) and its parameter validation invokes that formatter (`tt/generator.py:128-129`). Raising or bypassing the formatter alone therefore does not establish top-64 device support. Parent evidence: `run/sampling_policy_probe.json`.

## Smallest supported candidate

Keep Stage 10 `sample_on_device_mode="all"`, selected precision, TP4 hardware, full model/context and profile slot configurations. Enable existing adapter compatibility permission `GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1` at launch. Send accuracy chat requests with `temperature:1.0`, `top_p:0.95`, `top_k:64`, `logprobs:true`, `top_logprobs:0`, alongside the established generation budget and native template policy. Do not add logprobs to greedy performance requests.

This route uses these inspected files:

- `/workspace/tt-metal/vllm/vllm/entrypoints/openai/chat_completion/protocol.py:190,424-426,490,495`: top_k is a request field carried into SamplingParams; logprobs=true creates non-null logprobs (zero when top_logprobs=0).
- `/workspace/tt-metal/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:2641-2687`: TP4 logprobs requests force host sampling in both phases. Top_k=64 by itself does **not** select this fallback.
- Same runner `:2715-2729` and plugin `async_decode.py:588-601`: only device-sampling calls supply adapter sampling_params. Host calls use its default None.
- Autoport `tt/generator_vllm.py:29,127-139,182-185,213-214,239-251`: the explicit environment permission allows None, switches/release traces as necessary, and returns full host logits for vLLM sampling. This is distinct from setting standalone generator.host_sampling=True, whose validation at `tt/generator.py:132-140` allows only greedy unpenalized tests. The adapter host path does not call that sampler configuration.
- Plugin runner `:3035-3062,3132-3155`: host branch builds vLLM SamplingMetadata from top_k/top_p/temperature and invokes host_sampler. vLLM `v1/sample/ops/topk_topp_sampler.py:243-280` applies requested top-k and top-p without the common TT formatter cap.

Host-only discriminating check executed the AST-extracted actual `check_perform_device_sampling` with mocked TP4 metadata. With mode all and absent logprobs, both phases select device sampling. With max_num_logprobs=0 or 1, both select host sampling. Assertions passed; results: `run/host_sampling_route_probe.json`. This check verifies routing only, not logits, full integration, API transport or sampled distribution.

## Transport and remaining validation

Installed benchmark runtime `runtime/benchmark_stage/evaluate.py:22,62,114-125` subclasses upstream LocalChatCompletion and passes `generation` to simple_evaluate as gen_kwargs. It overrides no payload builder; raw response capture is at lines 95-111. No installed lm_eval implementation is available to inspect or execute, so arbitrary gen_kwargs forwarding of top_k/logprobs is **not verified**. Record an actual outgoing structured chat body and confirm these fields reach the API; if the pinned upstream client filters them, add the narrow backend payload override at that boundary and test it against a local recording HTTP stub before any server work. Do not infer transmission merely from summary generation_overrides.

A live configuration must demonstrate that accuracy uses host logits with exact top-64 and that greedy performance remains on the Stage 10 device sampler. Preserve full model/precision identity and raw request/response evidence. Report host fallback explicitly as accuracy protocol behavior. This candidate requires no shared sampler change, no silent top-k cap, and no model/precision alteration. API and device execution remain unverified; no final intended-policy validation claim is warranted.

## Source identity correction

The first routing probe inspected a different checkout under `/home/container_app_user/vllm-tt-plugin`; it is retained as `run/host_sampling_route_probe_other_checkout.json` and is not target-serving evidence. The parent identified `/workspace/tt-metal/vllm/plugins/vllm-tt-plugin` as the target plugin checkout. The source references above and `run/host_sampling_route_probe.json` now use that exact checkout; the host-only AST probe was rerun there with all six assertions passing. Target plugin commit: `7f72b1c6e905f5137fe3377f2e7b42738d3f271d`. Target `src/vllm_tt_plugin/model_runner.py` SHA256: `068d956521f8ee055b120dc1d49751d993e8c68fcf13dd16e563a57631caf442`. This correction proves source identity for the static check, not the actual imported module of a running server.
