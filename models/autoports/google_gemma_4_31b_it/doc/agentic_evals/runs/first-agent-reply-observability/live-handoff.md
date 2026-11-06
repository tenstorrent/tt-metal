# Coordinator-only live handoff

Prepared private scripts under `bringup/artifacts/first-agent-reply-observability/`; no launch occurred. Root-approved forecast0.06h based on prior0.05905h; box0.12h. Existing420s wrapper remains below432s box, but do not let an unexpected active request be killed to force the time box: coordinator tracks progress and retains safe ownership/cleanup. No device reset. TTab2730 source/image remains unchanged.

Use a fresh arm directory, e.g. `bringup/artifacts/first-agent-reply-observability/live`. Coordinator creates/registers its device-queue job and checks owners. From checkout root, launcher (in its own managed session):

```bash
mkdir -p bringup/artifacts/first-agent-reply-observability/live
bash bringup/artifacts/first-agent-reply-observability/launch-arm.sh \
  bringup/artifacts/first-agent-reply-observability/live 1 \
  > bringup/artifacts/first-agent-reply-observability/live/launch.log 2>&1
```

Then the coordinator runs client in its own session:

```bash
python_env/bin/python bringup/artifacts/first-agent-reply-observability/client.py \
  bringup/artifacts/first-agent-reply-observability/live \
  > bringup/artifacts/first-agent-reply-observability/live/client.log 2>&1
```

The private launcher copies the baseline launch unchanged except `VLLM_DEBUG_LOG_API_SERVER_RESPONSE=true` plus stock `--enable-log-requests --max-log-len 0`. Client imports the existing live_pair module and runs its exact three bodies/seeds, staggeredC2/max256/384 including existing ignore_eos setting. It then sends four simultaneous simple synthetic n1/nonstreaming thinking requests, max32, seeds5150–5153; this separately authorized reduced cap is **only nonscored logging diagnosis**, never a release/eval value. It saves all native request/response bodies privately and requires idle metrics before completion. Three-call baseline is `bringup/artifacts/phase-archive-idle-race/candidate`.

After client exits successfully, run one compact audit (destination must not already exist):

```bash
python_env/bin/python bringup/artifacts/first-agent-reply-observability/audit.py \
  bringup/artifacts/first-agent-reply-observability/live \
  bringup/artifacts/first-agent-reply-observability/live/audit.json
```

It binds each native client response ID to stock Received request and full JSON response-body log, requires exact body equality, compares all three baseline request/semantic/native-token results, reports paired request-window wall times and actual log growth. It does not emit raw bodies or reasoning. An empty-visible probe count0 leaves the native empty-path claim unconfirmed; host synthetic middleware check already proves that branch and its limitation remains explicit. Re-read actual source hashes/startup identity and final phase integrity for final confirmation. Audit failure does not authorize hard shutdown or altered sampling.

For clean shutdown only after metrics show running0/waiting0:

```bash
python_env/bin/python bringup/artifacts/first-agent-reply-observability/idle-close.py \
  bringup/artifacts/first-agent-reply-observability/live
```

This private copy preserves the existing owned-readiness PID/idle/SIGTERM/device-owner checks, but removes the old hardcoded queue update. Coordinator closes the correct new queue job itself. No script launches another server or dispatches CI. Host syntax checks (py_compile, bash -n) passed.

## Config publication proposal and growth forecast

Model-only merge: `env_vars.VLLM_DEBUG_LOG_API_SERVER_RESPONSE="true"`, `device_model_spec.vllm_args.enable_log_requests=true`, `device_model_spec.vllm_args.max_log_len=0`. Preserve all existing keys. It changes no TT/plugin/image-installed source or dependencies, so source-compatible with ab2730 image reuse, subject to normal immutable provenance proof. No model entry was edited in this preparation. Frozen dry/release must both use the accepted new TTI config.

Final overhead will compare matched call-window14.044s approximately (derive exact from baseline records) against live, not the baseline summary's196.514s that includes readiness wait. One sequential pair is descriptive, not a causal logging cost measurement. Report actual log bytes and serialized response characters/tokens. Full response logger ignores max-log-len; reasoning can be large.

Before live data, a deliberately marked **guess** of64serialized bytes per generated token plus8KiB metadata gives approximately8.13MB per126976-token response and650.8MB for80 such responses. If native prompt-ID arrays are requested, add about7bytes×262144prompt IDs≈1.84MB per worst-context response (~146.8MB/80). Ordinary lm-eval does not request those arrays, but full reasoning remains. Workflow duplication of the server log could roughly double retained/logged volume to1.3–1.6GB in this extreme scenario, not a bound. Actual tokenizer escaped-character ratios and workflow truncation may differ; substitute measured byte/token ratio from live controls with the caveat that short English controls do not predict long multilingual reasoning. No release projection should assume this logger is free or bounded by the phase-export cap.

Earlier RESULT.json/MD are preserved. AUTOFIX.md already records the host-supported candidate and explicitly supersedes their broad no-config-fix conclusion; final once-only `confirmation.json` and confirmation Markdown will record the actual live result after root returns measurements. No completed dry or initial-agent pre-score window follows merely from this local diagnostic.
