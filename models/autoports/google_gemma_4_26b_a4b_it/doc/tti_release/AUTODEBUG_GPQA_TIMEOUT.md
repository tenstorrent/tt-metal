# AutoDebug: Gemma 4 Stage 11 GPQA 30-minute timeout

## Headline finding

The GPQA failure is a client-side `lm-eval` total HTTP timeout, not evidence of a server crash, context-limit rejection, or invalid generation budget.

The Gemma 4 Stage 11 `meta_gpqa_cot` task requests `max_gen_toks=32768` but supplies no `model_kwargs.timeout`. The installed `lm_eval.models.api_models.TemplateAPI` therefore uses its default `timeout=1800`, converts it with `self.timeout = int(timeout)`, and constructs the concurrent aiohttp session with `ClientTimeout(total=self.timeout)`. All ten requests are submitted concurrently. Five completed; the five still active at the common 1800-second boundary raised `TimeoutError` together and were replaced by inference-error sentinels.

The minimal repair is to add a task-local `model_kwargs={"timeout": 7200}` to this Gemma 4 `meta_gpqa_cot` `EvalTask`. This changes only the client's per-request total HTTP allowance. It must not reduce the device `max_context=262144`, the command builder's resulting `max_length=262144`, or `gen_kwargs.max_gen_toks=32768`.

## Direct observations

At checkout HEAD `e3ea566df954eb611b5e4139a107f8e3ad6d4428`:

- The release log enters `generate_until` at `04:03:56` and reports the failures at `04:33:56`, exactly 30 minutes later.
- Five of ten requests completed before that boundary. At the boundary, five identical `Exception:TimeoutError(), undefined, retrying.` messages appear together, followed by `5/10 request batch(es) failed`.
- The emitted command contains:
  - `max_length=262144`
  - `max_retries=1`
  - `--gen_kwargs stream=false,max_gen_toks=32768,seed=42`
  - no `timeout=...` in `--model_args`
- The final `10 prompt(s) failed` message is downstream accounting: failed batches are converted to `__INFERENCE_ERROR__` sentinels, and the CLI rejects the resulting samples. It is not evidence that ten independent server requests failed.
- The reported server behavior remained healthy and continued generating. That is consistent with aiohttp abandoning client waits at its total deadline; the client deadline does not prove cancellation of server-side generation.

## Causal chain in source

1. `reference_config/evals/eval_config.py` defines the Stage 11 Gemma 4 26B `meta_gpqa_cot` task with `stream=false` and `max_gen_toks=32 * 1024`, but its otherwise-empty default `model_kwargs` does not override the HTTP timeout.
2. `llm_module/eval_command.py` serializes `task.model_kwargs` directly into lm-eval `--model_args`, while independently adding the device-authoritative `max_length`. The observed command proves the empty timeout configuration reached the process unchanged.
3. The installed `lm_eval/models/api_models.py` declares `TemplateAPI(..., timeout: int = 1800, ...)`, assigns `self.timeout = int(timeout)`, and creates `ClientSession(..., timeout=ClientTimeout(total=self.timeout))` in the concurrent request path.
4. With `num_concurrent=32`, the ten GPQA requests begin in the same concurrent batch. Any request not fully returned and decoded inside 1800 seconds raises aiohttp `TimeoutError`.
5. `max_retries=1` means the tenacity wrapper allows one total attempt, despite the generic log text saying “retrying.” The five exceptions are gathered, converted to per-sample inference sentinels, and ultimately make the eval command return 1.

The synchronized cutoff, exact 1800-second elapsed time, omitted command-line timeout, and installed lm-eval default form a complete causal chain. A server-side failure is neither required nor supported by this evidence.

## Why the long request is valid

The two relevant limits are separate:

- `max_length=262144` is the model/device context contract passed as a model argument.
- `max_gen_toks=32768` is the allowed output budget passed as a generation argument.
- `timeout` is only elapsed wall-clock time allowed for the HTTP request.

Raising `timeout` does not admit extra tokens or change either token limit. A 32K-token reasoning budget can legitimately take longer than 30 minutes on this serving path. Existing long-reasoning task configurations in the same file already use task-local `model_kwargs={"timeout": 7200}`, establishing both the supported plumbing and the local precedent.

## Minimal fix boundary

Change only the Gemma 4 26B Stage 11 `meta_gpqa_cot` task configuration:

```python
model_kwargs={
    # Per-request HTTP timeout; lm-eval defaults to 1800 seconds, which is
    # shorter than valid 32K-token generations on this serving path.
    "timeout": 7200,
},
```

Keep all of the following unchanged:

- device/server maximum context: `262144`
- command-line lm-eval `max_length`: `262144`
- task `max_gen_toks`: `32768`
- non-streaming behavior: `stream=false`
- global lm-eval defaults and unrelated tasks

The task-local boundary is preferable to changing lm-eval or globally raising all eval timeouts: only this demonstrated long-running workload needs the larger allowance. `7200` seconds matches established long-reasoning configurations and comfortably permits requests beyond 30 minutes without making the timeout unbounded.

## Focused experiment (no hardware/server required)

Add a command-builder unit test for the exact Gemma task/spec combination and assert that the generated argv contains all three independent contracts:

```text
--model_args ...timeout=7200...max_length=262144...
--gen_kwargs stream=false,max_gen_toks=32768,...
```

Also assert there is exactly one `timeout` and one `max_length`, so task kwargs cannot shadow the device-authoritative context value.

For direct behavioral proof of lm-eval ownership, use a local aiohttp test double that delays a valid chat-completions JSON response. Scale the timings down: instantiate the API model once with `timeout=1` and confirm a 1.2-second response becomes `TimeoutError`/`__INFERENCE_ERROR__`; instantiate it with `timeout=2` and confirm the identical response succeeds. Monkeypatch tokenizer loading or use the non-tokenized path so the test needs neither model weights nor hardware. This falsifies server-health and token-limit explanations while exercising the exact `ClientTimeout(total=...)` boundary.

## Verification after the configuration change

Run the command-builder unit test first. In the next real Stage 11 release run, verify the logged command includes `timeout=7200`, retains `max_length=262144` and `max_gen_toks=32768`, and no longer cuts all outstanding requests off at 1800 seconds. A request that exceeds 7200 seconds may still time out by design; this fix permits valid requests longer than 30 minutes rather than disabling failure bounds.

## Residual uncertainty

No hardware or live server was used. The source and timestamp evidence identifies why lm-eval stopped waiting, but it does not establish how long every GPQA generation will take or whether a separate server-side issue could affect a future run. The healthy, still-generating server observation makes such an issue unnecessary to explain this failure.
