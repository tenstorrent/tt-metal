# Resumed-prefill penalty history: contract investigation

Date: 2026-11-13. Read-only AutoFix hypothesis investigation; the initial investigation wrote only this report. A subsequent parent-requested follow-up added the host probe, its JSON evidence, and the proposed plugin-change record. No device access, server launch, implementation change, reset, or package change occurred.

## Verdict

**Correct presence/frequency penalties for arbitrary device-sampled resumed prefill cannot be implemented entirely inside the Ornith adapter/generator using the pinned documented interface.** The plugin knows the original prompt/output split but does not convey it on prefill. Existing lifecycle calls cannot reconstruct it unambiguously, even if the model retains every prior call and sampled token indefinitely. This is an information deficit, not proof that any particular new capability is necessary.

A normal first prefill and an uninterrupted decode are distinguishable and implementable with existing mechanisms. The defect is specifically the first sampled token of a replay that includes previously generated output. A later decode history reset cannot repair a token already sampled incorrectly.

The canonical generator's current prefill reset is consequently incorrect for that replay: it treats all replayed tokens as prompt and clears output history. Model-side history support is necessary once authoritative information is available; adding it alone cannot close the serving gap.

## Inspected sources and ownership

`P` is `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin`, verified HEAD `042a94863ad27d88207bcf71abdf67122c6f4c27`. `M` is `models/autoports/ornith_ai_ornith_1_5_9b` in the current tt-metal tree. The adapter was being repaired concurrently by the parent; this conclusion depends on the canonical generator and pinned plugin, not on an adapter snapshot remaining unchanged.

- `P/docs/MODEL_EXECUTION.md:38-55`: prefill `tokens` contains history; `prompt_lens` is the exclusive scheduled end, not original prompt length. Resume may replay generated tokens.
- `P/src/vllm_tt_plugin/model_runner.py:2394-2480`: replay and original prompt are distinguished internally using original prompt counts, computed counts, and resumed request IDs. Prefill tensors contain history through the scheduled end.
- `P/src/vllm_tt_plugin/model_runner.py:2679-2694`: separate prompt/output histories are built for decode only.
- `P/src/vllm_tt_plugin/model_runner.py:4094-4148`: `submit_prefill` forwards tokens, positions, tables, slots, optional modality/stage payload and sampling parameters. It forwards neither original prompt lengths, request IDs, resumed status, nor separate histories.
- `M/tt/generator.py:474-479`: before device sampling, `_reset_sampling_state` receives the complete prefix as `prompt_tokens`, without `output_tokens`.
- `M/tt/sampling_state.py:54-57,355-382`: absent history normalizes to empty; reset replaces the selected output counts/mask with zeros.
- `models/common/sampling/tt_penalties.py:58-69`: presence uses output membership; frequency uses output occurrence counts. Repetition uses the union of prompt and output membership and is not the discriminating penalty here.
- `P/docs/MODEL_LIFECYCLE.md:45-68` and `P/src/vllm_tt_plugin/model_runner.py:962-1023`: finish and preemption both call the same `release_request(slot)`, with no reason or identity argument. Resume rebuilds state through prefill.

This is both a local generator assumption and a shared serving information gap. The plugin already has authoritative history, so a repair should derive it there instead of guessing inside the model. Classifying it as a missing model sampling feature alone is incorrect: the model's existing `reset_history` already supports separate histories.

## Discriminating host experiment and information-flow proof

Executed with `python_env/bin/python` in this checkout. The harness AST-extracted the actual pinned `TTModelRunner.submit_prefill` and `_release_model_request` methods, supplied simple host-only recorder objects and a sampling dataclass, and AST-extracted the actual `normalize_histories`. It imported no torch, TTNN, model, plugin runtime, or server. No harness file was added.

Construct these two legal histories with identical model-visible calls:

1. Both worlds initially prefill prompt `[7,8]` into slot 0 and return sampled token `9` with the same sampling parameters.
2. In world F the request finishes or is cancelled; in world R it is preempted. Both issue `release_request(0)`.
3. In world F a **new** request arrives with prompt `[7,8,9]`. In world R the **same old** request resumes with original prompt `[7,8]` and generated history `[9]`.
4. Both prefill from position 0 to end 3 into slot 0, with identical sampling parameters and page table. Freed physical pages may be reused, so equal page addresses are legal. Prefix caching is disabled for Ornith; zero is a valid resume start.

No request ID, release reason, original prompt boundary, or resumed bit distinguishes these worlds. They can share the entire prior model-visible transcript, not just the final call. Keeping old state, matching tokens, or remembering the first prompt length cannot decide whether the new prefill belongs to the old request or a new request. This also defeats unbounded content-keyed history caches.

For the next-token logits `logit[9]=2.0`, `logit[10]=1.5`, set temperature 0, repetition penalty 1, presence penalty 0.75, and frequency penalty 0.5:

| World | Correct output history | Adjusted score for token 9 | Greedy token |
|---|---|---:|---:|
| F: fresh request | `[]` | 2.0 | 9 |
| R: resumed request | `[9]` | 0.75 | 10 |

Identical inputs and internal model-observable history require different correct outputs. Therefore no model-only algorithm over that interface can be correct in both worlds. The numbers illustrate the standard penalty equations; they are not an assertion about actual Ornith logits for those token IDs. The information deficit exists independently of that particular numerical example.

The host harness also deliberately populated internal `model_input.prompt_tokens` and `output_tokens` differently in the two worlds. The extracted `submit_prefill` still emitted identical kwargs, demonstrating that adding history merely to `TTModelInput` would not deliver it to the adapter.

Observed output:

```text
PASS: extracted pinned submit_prefill and _release_model_request expose identical complete call transcripts for fresh versus resumed histories
PASS: even populated internal model_input prompt_tokens/output_tokens are omitted by submit_prefill
PASS: required greedy sample differs: fresh=9, resumed=10; repetition_penalty=1, presence=0.75, frequency=0.5
PASS: extracted normalize_histories(None) produces empty output history
No torch, ttnn, plugin runtime, model, or server imports; no device calls.
```

This is a host information-flow proof, not a hardware or full-runner execution test. Source inspection establishes that finish and preemption share the extracted release call and that the two prefill forms are allowed.

## Existing mechanisms checked

| Mechanism | Why it does not supply the missing information |
|---|---|
| `tokens`, `start_pos`, `prompt_lens` | They describe causal replay extent. Zero-start replay is allowed; a prompt with those exact tokens is equally valid. Chunk boundaries depend on scheduling, not the prompt/output boundary. |
| `empty_slots`, `slot_remap`, `note_state_slots_moved`, `release_request` | They track current physical ownership. Release is identical for finish/preemption and drops ownership before resume. A slot is not a persistent request identity. |
| Page-table/KV identity, per-layer tables | Allocations and physical addresses can be reused. KV encodes causal token context, not whether a token originated in the prompt or output. Opaque model-owned KV objects contain no scheduler request metadata by contract. |
| Sampler masks/counts and retained host history | Work for uninterrupted ownership. Reusing released state misattributes a fresh request in world F; discarding it loses world R. The proof also rules out content-addressed recovery. |
| Decode `prompt_tokens`/`output_tokens` and version-1 reload commands | Authoritative when supplied; `async_decode.py:1382-1409` forwards histories only through decode. No such call occurs before the resumed-prefill sample. A later correction is too late. |
| Sampling seed/parameters | Not unique identities. Multiple requests may share every parameter, including seed. Unspecified seed is legal. Original prompt length is not a sampling field. |
| M-RoPE deltas | Text deltas are zero in both worlds. They are output positional metadata, not original prompt boundaries. |
| `supports_stage_inputs` / `stage_inputs` | Existing optional vLLM-Omni request payload, not automatic scheduler metadata. `MODEL_LIFECYCLE.md` explicitly says it has no request ID; ordinary text requests may supply no payload. Requiring callers to invent a new identity or boundary payload would change the supported request contract and cannot repair arbitrary existing text requests. |
| Multimodal metadata | Ordinary text requests carry none; repurposing these fields for scheduler data changes their meaning. |
| `supports_device_penalties`, `supports_sample_on_device`, `max_device_top_k` | These control routing/admission, not history delivery. Turning penalties off or declaring them unsupported would bypass a feature already required and implemented, not repair resume. |
| `sample_on_device_mode=decode_only`, `seeded_sampling_policy=host`, host-only controls | Existing routing can avoid device prefill, but does not satisfy device-sampled resume. Moreover, the pinned host-prefill sampler also uses empty output history, as described below. |
| `always_compat_sampling` | Selecting the compatibility sampler cannot restore history omitted from the constructed sampling metadata. |
| `supports_stage_outputs` / block-output admission | Their no-preemption restrictions belong to different genuine execution contracts. Declaring them solely to avoid this failure would misrepresent the model. |
| Disabling prefix caching / chunked prefill | Neither eliminates scheduler preemption replay. Prefix caching is already disabled here. |
| KV sizing and concurrency limits | Reserving enough memory can avoid ordinary memory-pressure preemption in a restricted launch. It does not implement resumed-prefill semantics, and forced reset/preemption paths remain relevant. No universal correctness claim follows. |
| Registration (`TT_MODEL_CLASS_OVERRIDES`, `EXTRA_MODELS_DIR`), trace/warmup/fabric/layout options | Select the adapter or execution resources; do not add request history. Reviewed `MODEL_SETUP.md`, capability inventory, and README configuration/registration sections. |
| Runner inspection, stack introspection, monkeypatching, global side channels from the adapter | Not the documented model interface, and effectively modify shared plugin behavior. They are not model-only implementations of this contract. |

## Separate host-path finding

Routing resumed prefill to the host is not sufficient in this pinned plugin. `model_runner.py:4371-4396` uses separate output history only when `is_decode`, otherwise constructs empty output history; it likewise labels all prefill input tokens through the chunk end as prompt. Lines 4418-4437 pass these values into `SamplingMetadata` and the host sampler. Thus the same presence/frequency omission exists in the host-prefill path. This is source-confirmed but was not executed through the actual host sampler in this investigation.

Correcting that history construction is a plugin-owned semantic repair that can potentially reuse existing internal payload fields without new public options. It must be validated separately and does not by itself repair device-sampled prefill. Do not present host routing as a verified workaround.

## Interface and authorization boundary

Reviewed the exact stage policy at `/home/ttuser/ornith-9b/codex_home/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.23/skills/vllm-integration/references/plugin-changes.md` and `P/AGENTS.md`.

- Model-local use of the existing history reset API, accurate slot remapping, and adapter translation are existing implementation work; no new shared interface is required for those changes.
- Correcting plugin-internal host history construction can be a plugin defect repair without new capabilities, environment variables, configuration keys, or hooks. The policy still requires its justification record and regressions before implementation.
- Delivering original prompt lengths or separate histories on prefill changes the model execution boundary. An opt-in new capability, hook, marker, or version meaning requires **specific human authorization**. Unconditionally adding kwargs can break existing models and changes behavior for models that have not opted in; the policy also explicitly requires authorization for such a default change. Signature probing or an architecture-specific branch does not evade this boundary.
- The proof establishes that additional authoritative information or changed plugin orchestration is necessary for this device-prefill behavior. It does **not** establish that a new capability is the only design. Reusing existing calls in different plugin orchestration would need its own semantic/compatibility analysis; none was implemented or claimed correct here.
- A possible minimal information item is original prompt length per submitted row, since full replay tokens already arrive. Separate prompt/output histories are more explicit and match the existing decode vocabulary. Neither proposal has been implemented or authorized by this investigation. Request IDs alone are not necessary if the split is delivered directly.

If a shared extension is chosen, prepare a `doc/vllm_integration/plugin_changes.md` entry with status `proposed`, exact API semantics and affected callers/models, the two-world reproducer, alternatives above, pending exact authorization, and planned host/device regressions before seeking authorization. The parent owns that decision and goal status; this delegated investigation creates neither a proposal implementation nor a goal-status change.

Required later validation includes a real resumed request with nonzero presence/frequency penalties, a fresh request whose prompt equals a released request's full history, chunked replay crossing the original prompt boundary, row/slot reuse and remap, plus host and device paths. No serving readiness, quality, performance, or hardware pass is claimed.

## Persisted follow-up artifacts

The parent-requested executable proof is now `tests/vllm_resume_contract_probe.py`; [resume_contract_probe.json](resume_contract_probe.json) records its successful host run against the pinned SHA. [plugin_changes.md](plugin_changes.md) is the concrete proposed opt-in interface and host correction, with authorization pending. No plugin or model implementation was edited.
