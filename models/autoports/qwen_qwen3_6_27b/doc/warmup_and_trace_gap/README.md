# Warmup and trace capture in the vLLM serving path

Facts first, each with how it was observed. Interpretation and suggestions are
in the last section and are labelled as such.

## Facts

### F1. Both vLLM warmup hooks are no-ops

`tt/generator_vllm.py`:

```python
def warmup_model_prefill(self, *args, **kwargs):
    del args, kwargs

def warmup_model_decode(self, *args, **kwargs):
    # Trace capture is stateful and is performed from the first real
    # scheduler batch so its page table and active slots are exact.
    del args, kwargs
```

Decode carries a stated reason. Prefill has no comment and no body.

*Observed:* read the file.

### F2. They were added by stage 09, whose own commit says the stage is incomplete

`dced79f19f4`, 2026-08-14, "qwen3.6-27b stage 09: preserve work; seeded sampling
is a plumbing gap, not a device-capability gap (STAGE NOT COMPLETE)".

*Observed:* `git log -S"def warmup_model_prefill" -- tt/generator_vllm.py`.

### F3. The spec tells the server not to warm the model, because the model claims to

`workflows/model_specs/dev/llm.yaml` sets `has_builtin_warmup: true` for this
model. In `vllm-tt-metal/src/run_vllm_api_server.py:594`:

```python
if not disable_trace_capture and model_spec_json.get("has_builtin_warmup", False):
    disable_trace_capture = True
    logger.info("Model has builtin warmup (has_builtin_warmup=True), "
                "skipping background trace capture")
```

So the server's background trace capture is skipped, and F1 means nothing
replaces it.

*Observed:* read both files.

### F4. The plugin calls warmup in two phases and documents a correctness hazard

`vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py` ~2500-2540 calls
`warmup_model_prefill(enable_trace=False, ...)` then `(enable_trace=True, ...)`,
with this comment:

```
#   2. Prefill warmup must cover all supported sequence lengths.
#      If a new sequence length appears during inference, its
#      first compilation will allocate new kernel cache entries
#      (including reshape caches) that can corrupt active traces.
# See: https://github.com/tenstorrent/tt-metal/commit/5043de3df5
```

It also resets `already_warmed_up_prefill` between phases; this class has no
such attribute.

*Observed:* read the plugin source.

### F5. Shipped tt-metal models declare a coverage set; the default is one length

`models/common/models/qwen3_32b/executor.py:136`:

```python
prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_supported_seq_lens", (128,))
```

`models/common/llm_runtime/warmup.py` expands that into a plan of cases over
batch size x sequence length x cached tokens x sampling path, and walks every
case twice: once into `self._eager` (compile), once into `_trace_registered`
(capture). It orders `topk` before `argmax` so the shared trace artifact owns the
K/P/T buffer, and primes all four tile-start variants for
`batch=1, seqlen=128`.

So the convention is neither exhaustive nor single-shot: a *declared* set,
defaulting to `(128,)`.

*Observed:* read both files.

### F6. Traces live in device DRAM, and this model's budget is 200 MB

Traces occupy the on-device trace region sized by `trace_region_size`, which
this model's spec sets to 200000000. Nothing persists across restarts: the
startup banner reports `enable_model_cache=false`. `doc/ci_dispatch_qb2` records
that `trace_region_size: 1073741824` kills the engine during startup because the
value is subtracted per bank (8.6 GB/device).

*Observed:* spec, ttnn startup banner in server logs, and the earlier measured
OOM recorded in that doc.

### F7. The skills require traced decode and do not mention prefill warmup

Three skills state the decode requirement:

- `tt-enable-tracing/SKILL.md`: "For the readiness harness, teacher-forcing
  decode must always use traced decode. Do not treat an eager teacher-forcing
  pass as acceptable evidence..."
- `full-model/SKILL.md`: "The final decode path uses traced TTNN execution...
  Eager decode is useful only while debugging and is not completion evidence."
- `vllm-integration/SKILL.md`: "The serving decode pass must drive the
  generator's traced decode path, not an eager-only fallback."

Searching `.agents/skills/` for `warmup_model_prefill`,
`trace_prefill_supported_seq_lens`, or prefill warmup coverage returns nothing.

*Observed:* `grep -rn` over `.agents/skills/`.

### F8. No trace capture occurs in a real serving run, and TPOT matches eager decode

A local vLLM run at the CI benchmark point (ISL 128, OSL 128, concurrency 32, 32
requests, `max_num_seqs=32`) produced **zero** log lines matching
`begin_trace_capture|capture_trace|trace captured|traced decode|end_trace`.

Measured decode step times for this model, from `regression_active_row.log`:
multichip batch-32 traced decode is **0.628 ms**. Serving TPOT measured in the
same configuration and in CI is **~244 ms/token**, i.e. **389x** the traced
figure. The recorded pre-existing baseline was 61.33 ms.

*Observed:* grep of `readiness_vllm/server.log`; TPOT from `vllm_result.json`
and from CI run 34136971571; traced figures from the regression log.

Absence of those log lines is not by itself proof that no trace was captured --
the code may not log it. The 389x ratio is the stronger evidence.

### F9. The CI failure occurred on the first scheduler step of the concurrency-32 point

CI benchmarks run 34136971571: the concurrency-1 point passed (8/8, mean TTFT
1890.98 ms). The concurrency-32 point failed with
`TT_THROW ... TIMEOUT: device timeout, potential hang detected, the device is
unrecoverable`. The scheduler dump at failure:

```
SchedulerStats(num_running_reqs=32, num_waiting_reqs=0, step_counter=0,
               kv_cache_usage=0.0024)
num_scheduled_tokens={...: 128, ...}      # all 32 requests
scheduled_new_reqs=[NewRequestData(prompt_token_ids_len=128, ...), ...]
num_computed_tokens=0
```

`run_vllm_api_server` sets `TT_METAL_OPERATION_TIMEOUT_SECONDS = "5.0"`
unconditionally unless `DISABLE_METAL_OP_TIMEOUT=1`, in which case it deletes
the variable. There is no path to a larger value from the spec.

*Observed:* CI job log, and `set_metal_timeout_env_vars()` in the server source.

### F10. Triage found the device healthy and located no hung op

Same run, `check_arc.py` table: all four devices, up time 0:16:29.900000, clock
800 MHz, heartbeats ~9.8589/s. `check_broken_components.py` reported, for many
ethernet and functional-worker cores, "Was halted by triage but is no longer
halted - core was broken during triage." Zero `Out of Memory` occurrences in the
run.

*Observed:* triage output tee'd into the CI job log.

### F11. Locally, the same point is slow rather than hung

The local run above has no `TT_METAL_OPERATION_TIMEOUT_SECONDS` set. At 30
minutes elapsed the EngineCore process was `STAT=Rl` with 45:09 of CPU time over
29:49 elapsed (~150% CPU), no `TT_FATAL`, no OOM, and no result yet. Expected
work at the measured TPOT is 128 steps x ~244 ms ~= 31 s.

*Observed:* `ps -o pid,etimes,time,stat`, and the run's own logs.

### F12. Each benchmark sweep point runs once, and compilation lands inside TTFT

`llm_module/runner.py` executes one `self.driver.run(cfg, ...)` per sweep point,
with `inter_run_sleep_s` between points; there is no repeat loop. TTFT is
measured client-side by `vllm bench serve` from request send to first token, so
any first-request compilation is included in the reported TTFT. Program caching
is per process, so a shape compiles once per server lifetime.

*Observed:* read `runner.py`; `enable_model_cache=false` from the startup banner.

### F13. Prefill shape varies because chunked prefill is disabled for this arch

`platform.py` disables chunked prefill for `model_type=qwen3_5`, so prefill runs
whole requests and its shape varies with the number of active rows and the
sequence length, rather than being one fixed chunk shape. Measured active-row
counts per prefill call: concurrency 8 gave `active=1` then `active=7`;
concurrency 32 gave `active=1` then `active=31`; the CI concurrency-32 step
scheduled all 32 as new requests at once.

*Observed:* the recorded `platform.py` behaviour in
`doc/vllm_rerun_on_new_base`, and `QWEN36_PREFILL_LOG_K=1` output.

## Interpretation and suggestions -- not established fact

Everything below is opinion. It is separated deliberately because none of it was
measured.

**S1. The likely reason prefill warmup was never filled in.** F7 shows the
skills specify traced *decode* in three places and never mention prefill warmup
coverage. `doc/prefill_general_optimizations` records the same asymmetry twice
already (decode 225 mentions vs prefill 30; core-grid guidance scoped to
"decode-time consumers"). I read F1/F2 as a third instance of that bias rather
than a misreading of guidance: the stage was told to make decode traced, prefill
warmup was in nobody's checklist, and the stub survived. This is an inference
about cause, not an observation.

**S2. F8 appears to contradict a skills requirement.** `vllm-integration`
requires the serving decode pass to drive the traced path and not an eager
fallback. The 389x TPOT ratio suggests serving decode is eager. I have not
instrumented the decode path to confirm which branch executes, so this is
suspicion, not proof. Confirming it is a small piece of work and, in my view,
worth more than any further prefill optimisation: 244 ms/token dominates
end-to-end latency by ~17x over TTFT at the graded point.

**S3. What I would implement, in order.** (a) Determine whether serving decode
is traced, by instrumenting the branch rather than inferring from timing.
(b) If it is eager, make trace capture actually happen -- the deferral in F1's
comment is defensible in intent but nothing appears to complete it later.
(c) Give `warmup_model_prefill` a declared coverage set in the style of F5, sized
to the 200 MB trace budget in F6, so compilation leaves the measured TTFT.

**S4. On the CI timeout.** F9-F11 together suggest the 5 s watchdog fired on a
long no-dispatch-progress window rather than a hang, since the device was healthy
and the same point makes progress locally without a watchdog. I would not
disable the watchdog: it is the only thing that turned this into a fast, legible
failure with a triage report. Making the threshold configurable and setting it
generously seems better than deleting it, but that requires the small
tt-inference-server change described in `doc/ci_dispatch_qb2`. Note this is a
judgement about the right knob, and F11 leaves open the possibility that the
workload is simply far slower than expected at concurrency 32 -- in which case
the watchdog is reporting something real and the correct fix is S3, not the
timeout.

**S5. Warming everything is not the answer.** F5 and F6 together say the
convention is a small declared set, and the trace region is 200 MB against 27B of
weights. Choosing coverage is a budget decision, which is my reading of why F1
was not a two-line fix.

## How the shipped demo prefills, and what that says about the hang

### F14. The demo for this same architecture also prefills one request at a time

`models/demos/blackhole/qwen36/tt/qwen36_vllm.py` is the hand-written demo for
`Qwen3_5ForConditionalGeneration` -- the same architecture this autoport serves,
behind the same vLLM plugin. For the multi-device multi-slot case it takes
`_prefill_forward_tp_batched`, whose docstring reads:

> TP batched (max_num_seqs>1) prefill: prefill each request in this step into
> its decode slot. vLLM prefills new requests while other slots decode, so each
> user's B=1 state is written into row empty_slots[u] of the batched GDN buffers
> without disturbing the live rows (model-owned, via prefill_paged_slots).

and whose body is a per-request list, not a padded batch:

```python
token_ids_list = [tokens[u : u + 1, : plens[u]].to(torch.int32) for u in range(N)]
host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens)
logits = torch.cat([hl.reshape(1, 1, -1) for hl in host_logits], dim=0)
```

So "batched" there means *batched decode slots*, not a batched prefill tensor.
Every request is prefilled at B=1 into its own slot.

*Observed:* read the file.

### F15. Two things the demo does that this autoport does not

- **`valid_lens=plens`** -- each request is prefilled to its own real length
  (`tokens[u:u+1, :plens[u]]`). This autoport pads every request in a step to
  the step's maximum, so a 128-token request scheduled alongside a 4096-token
  one computes 4096.
- **`_remap_gdn_slots(slot_remap)` in decode** -- when vLLM condenses slots, the
  demo reindexes its per-slot GDN recurrent/conv state, because "GDN state is
  model-internal" and the plugin only remaps its own buffers. This autoport has
  `remap_decode_slots`, which appears to be the same idea; whether it covers the
  same state is unverified.

*Observed:* read the file, and compared against
`tt/generator_vllm.py::prefill_forward` / `decode_forward`.

### Interpretation -- not established fact

**S6. This reframes the per-request fix.** I committed
`QWEN36_PREFILL_PER_REQUEST` as a fix for a hang, with serialisation presented
as its cost. F14 suggests that framing is wrong: per-request prefill is what the
reference implementation for this architecture does, for a stated reason (GDN
state is per-slot and must be written without disturbing live rows). On that
reading the padded multi-row prefill was the anomaly, and its TTFT is not a
compromise but simply what this architecture costs.

I have **not** established that our multi-row path hangs *because* of GDN state
writes. Eight variables were eliminated by experiment (see the commit for
`71384bee159`) and the mechanism is still unknown. F14 is a strong hint about
where to look, not a diagnosis.

**S7. The remaining prefill gap is `valid_lens`.** Our per-request loop still
pads each call to the step's maximum length. At the graded ISL 128 point every
prompt is 128 so it costs nothing, but a mixed-length eval batch would pay for
the longest prompt on every request. Worth fixing before reading any
mixed-length eval timing.
