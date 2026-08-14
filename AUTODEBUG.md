# AutoDebug: Mistral Small 24B vLLM punctuation anomaly

## Headline finding

The reported `learning,,,` completion is **not reproduced on the production serving path** and the available evidence refutes a general stale-token, stale-position, page-table, slot-remap, or async device-sampling defect.

The anomaly was preserved only by the earlier compatibility-enabled final qualitative artifact. The current host-compatibility-disabled production suite is clean, as are all focused production reruns. The most specific remaining hypothesis is therefore a compatibility-only transition between a host-sampled stochastic request and the following device-sampled greedy request, with a close-logit BFP4 branch as a lower-confidence alternative. Neither hypothesis is established as a production bug.

No implementation fix is recommended from this evidence.

## Direct observations

- The live server uses `trace_mode=all` and `sample_on_device_mode=all`; `readiness_vllm/server.log:15,85` records the configuration.
- `TT_DEVICE_SAMPLING_AUDIT` records `perform_device_sampling=True` for both prefill and decode on supported stochastic and greedy production requests (`readiness_vllm/server.log:119-120,154-155`). Mixed eight-request waves also remain on device (`server.log:124-145`).
- With host compatibility disabled, all of the following were clean:
  - one supported stochastic request at temperature `0.7`, top-k `32`, 128 output tokens;
  - the exact prompt twice through `httpx` and twice through the OpenAI client;
  - two repetitions of `prompt 0 greedy -> prompt 0 stochastic -> prompt 1 greedy`;
  - eight concurrent identical prompt-1 greedy requests;
  - the full six-prompt readiness qualitative stage.
- The current production result, candidate result, and optimized-vLLM final result are byte-identical and clean:

  ```text
  a4a2338f026e0baefcd69a40d41b51fc8889bcb1645fa6d878a5ac7a3c07f3f9
  ```

  This is the SHA-256 of each current `vllm_qualitative_outputs.json`.
- The earlier compatibility-enabled final artifact contained `In unsupervised learning,,, you have to...`; this is the artifact reviewed in `doc/optimized_vllm/stage_review.md:13,37`. It has since been replaced by the clean production artifact, so the old bytes are not available for further static inspection in the worktree.
- With Mistral's fixed tokenizer regex, `,,,` is two tokens, `[64704 (',,'), 1044 (',')]`, not three repetitions of one comma token. The symptom is therefore not evidence of a one-token feedback loop by itself.
- No focused production request changed behavior between `httpx` and the OpenAI client.

## Source audit

### Readiness request ordering

`models/common/readiness_check/run_vllm_server.py:416-471` runs each prompt serially as greedy and then stochastic before moving to the next prompt. Prompt 1 is therefore preceded by prompt 0's stochastic request.

### Compatibility routing creates a distinct transition

`vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:2505-2610` routes Mistral requests with nonzero temperature to host sampling only when `MISTRAL_SMALL_24B_VLLM_HOST_SAMPLING_COMPAT=1`. Greedy requests remain eligible for device sampling. Consequently, the compatibility suite exercises:

```text
device greedy -> host stochastic -> device greedy
```

whereas the production suite exercises device sampling throughout. This is the principal execution-path difference that matches the artifact split.

The adapter explicitly handles this boundary. `tt/generator.py:964-969` merges device-authoritative token/position state and releases device traces when switching to host sampling. A following device request enters `_ensure_decode_traces()` and captures or refreshes the traced path (`tt/generator.py:775-797,940-960`). This code makes the transition a legitimate audit target, but static inspection alone does not prove it produced the old punctuation.

### Persistent traced inputs

`tt/generator.py:630-659` refreshes persistent token and position tensors when requested and refreshes the persistent page table only when its host snapshot changes. During steady async decode, `reset_batch=False` deliberately retains the device-authoritative sampled token and advanced position while refreshing scheduler-owned page-table state (`tt/generator.py:948-960`). Layout resets merge device state through a full slot permutation and force newly prefetched destination slots to use host state (`tt/generator.py:575-627`).

The current focused sequential, mixed, and concurrent production results do not support a generic defect in this logic. In particular, changed scheduler layout and slot reuse did not reproduce the anomaly.

### Async boundary

The production adapter returns device tensors by default from `decode_forward()` (`tt/generator_vllm.py:217-263`). The plugin submits decode without reading, schedules `read_decode_output(..., async_read=True)`, and synchronizes recorded events only in finalization (`vllm_tt_plugin/async_decode.py:550-710`). Model and sampling traces replay nonblocking (`tt/generator.py:829-835`). Nothing in this static path explains a compatibility-artifact-only punctuation branch.

## Ranked hypotheses

1. **Compatibility-only host/device transition** — plausible, not proven. The anomalous artifact followed a host-sampled stochastic request because compatibility was enabled; every host-compatibility-disabled control stayed on device and was clean. If a bug exists, the boundary that releases a device trace for host sampling and recaptures it for the next greedy request is the narrowest matching scope.
2. **Close-logit numerical branch** — possible. BFP4/LoFi model execution can select a nearby token when top logits are close. The two-token `',,' + ','` encoding is consistent with ordinary autoregressive choices and does not specifically implicate stale feedback. Repeated clean production controls make this a low-severity qualitative variance unless the compatibility sequence reproduces it reliably.
3. **Stale saved artifact/provenance** — possible but unproven. The anomalous bytes now exist only as prior review evidence, while all three current artifacts have the same clean SHA. This can explain the discrepancy between the stage review and current files, but not the original generation itself.

## Refuted or substantially weakened hypotheses

- **Generic persistent token/current-position/page-table corruption:** refuted by the clean exact-sequence, mixed-batch, concurrent, and full-suite production controls.
- **Request/physical-slot attribution failure:** weakened by eight concurrent identical clean outputs and mixed waves with audit-confirmed device routing.
- **Device stochastic sampling is falling back to host:** refuted for the production controls by `TT_DEVICE_SAMPLING_AUDIT perform_device_sampling=True` at prefill and decode.
- **OpenAI client request formatting causes the anomaly:** refuted by byte-identical clean `httpx` and OpenAI-client controls.
- **A repeated comma token is stuck in device feedback:** refuted by tokenizer evidence that the text is two different token IDs.

## Focused follow-up experiments

No further experiment is required to clear the host-compatibility-disabled production path. If compatibility-mode quality must also be diagnosed, use a separately authorized compatibility-enabled server and preserve the audit log:

1. Repeat exactly `prompt 0 greedy -> prompt 0 stochastic -> prompt 1 greedy` and compare it with `prompt 1 greedy` alone. Record response token IDs and `perform_device_sampling` for each prefill/decode phase.
2. Repeat the same sequence with the middle request greedy. This removes the host/device transition without changing prompt order.
3. Repeat with the compatibility switch enabled but omit the middle stochastic request. This separates environment configuration from transition behavior.
4. If only the transition case fails, add a unit-level transition test around device decode -> host decode -> device decode that asserts trace release/recapture, authoritative token/current-position merge, slot remap, and changed/unchanged page-table copies. Do not alter the production steady-state trace path based on the current evidence.
5. If all variants occasionally choose the punctuation tokens, collect the top few logits for the first divergent step in a diagnostic-only run and compare their margin. Treat a close top-2 margin as numerical qualitative variance rather than scheduler corruption.

## AutoDebug runner limitation

The required `.agents/scripts/autodebug.sh --agent codex ...` fresh-context run was executed first. Its Codex sandbox could not initialize even for `pwd` because Bubblewrap failed with:

```text
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

Its three read-only subagents encountered the same failure, and its patch tool could not write the report. This report was therefore completed afterward, as required by the skill's follow-up step, using read-only source/log/artifact inspection in the parent environment plus the focused runtime evidence supplied by the main agent. No request was sent to the live server, no server process was stopped, no hardware operation was performed, and no implementation code was edited during this AutoDebug task.
