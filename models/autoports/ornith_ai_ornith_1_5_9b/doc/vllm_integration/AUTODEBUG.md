# AutoDebug: Ornith Stage 9 first-request adapter failure

Investigation date: 2026-11-13. Inspection only: no device access, server runs,
resets, package changes, or source edits. This report is the only file written.

## Diagnosis

**The observed failure is an adapter sampling-schema mismatch.** The plugin's
`TTSamplingParams` reaches the canonical generator unchanged; its type lacks
`log_probs_enable_temperature`, which the canonical slot-state merger requires.
The common formatter preserves the input dataclass type, so formatting cannot
repair this mismatch.

Three further source-confirmed adapter mismatches affect subsequent execution
or the alternative host-sampling route:

1. Qwen3.5 M-RoPE prefill requires `(output, rope_deltas)` even with
   `--language-model-only`; the adapter returns a bare output.
2. M-RoPE decode supplies `rope_deltas_all_users`, including when its value is
   `None`; forwarding it into the canonical generator raises an unknown-keyword
   error.
3. Host-sampled prefill already returns a torch logits tensor, but the adapter
   reads `.logits` and `.row_slots` from it. This also breaks intermediate
   chunked prefill, which the plugin deliberately routes through the host path.

These are Python interface defects. The smallest repair boundary is
`tt/generator_vllm.py`; no plugin modification, new capability, or numerical
model change is needed to address these four defects. Correcting them does not
by itself establish full serving correctness or Stage 9 readiness.

## Scope and revision consistency

Paths below use these prefixes:

- `M`: `models/autoports/ornith_ai_ornith_1_5_9b` in this tt-metal checkout.
- `P`: `/home/ttuser/ornith-9b/dependencies/vllm-tt-plugin`.
- `E`: `/home/ttuser/ornith-9b/tt-metal/python_env/lib/python3.12/site-packages`.

Verified locally:

| Component | Inspected state |
|---|---|
| tt-metal HEAD | `d1c644f35289ee46bfa4a23598aa607d5ea96522` |
| Plugin HEAD | `042a94863ad27d88207bcf71abdf67122c6f4c27`; plugin working tree clean |
| Installed vLLM distribution | `0.31.0+empty`, read from its installed `METADATA` |
| Adapter | Current untracked `M/tt/generator_vllm.py`; no changes made |
| Baseline | Canonical `M/tt/generator.py`; Stage 8 pass is requester context, not rerun here |

`stack_provenance.json` agrees with the revisions. Diagnostic3 stack frames
match the inspected adapter, generator, sampling-state, and pinned plugin line
numbers. Diagnostic2 predates the adapter's empty-multimodal-argument filter;
its traceback is evidence of that earlier boundary only.

## Direct observations versus conclusions

| Direct observation | What it establishes |
|---|---|
| `diagnostic3/server.log:9-23` records vLLM 0.31.0, the diagnostic adapter override, `language_model_only=True`, `max_num_seqs=1`, block size 32, context 262144, and `trace_mode=decode_only`. | This is the reduced two-layer serving diagnostic, not a full-model quality run. |
| `diagnostic3/server.log:173` records one device-sampled prefill: start 0, exclusive end 129, slot 0. | The failure is on a new request's first prefill, not an intermediate chunk or decode. |
| `diagnostic3/server.log:216-227` enters the plugin's tuple-prefill branch and ends at `sampling_state.py:424` with the missing attribute. | Both the active M-RoPE return contract and the immediate schema exception are observed. The tuple unpack has not yet completed. |
| `diagnostic3/request.json` records HTTP 500; shutdown counters at log line 228 show zero decode-model replays, sampling replays, and generator readbacks. | No successful sampled response or decode is shown. Zero replay counters do **not** prove that prefill did not execute. |
| `diagnostic2/server.log:217-224` reports the five empty multimodal kwargs as unknown. Current adapter lines 149-158 consume their empty values. | The earlier keyword failure was exposed first; diagnostic3 progresses past it to another interface mismatch. |

The interpretation supported by these observations is incomplete translation at
the serving adapter boundary. There is no evidence here that a low-level TTNN
operation, the selected datatype policy, or a model numerical mismatch caused
this Python exception.

## 1. Observed root cause: incompatible sampling dataclasses

The exact chain is:

1. `P/src/vllm_tt_plugin/model_input.py:68-83` defines nine-field
   `TTSamplingParams`, without `log_probs_enable_temperature`.
2. Plugin `model_runner.py:4112-4126` converts its tensors into per-row lists,
   maps unspecified seed sentinels to `None`, and constructs that same plugin
   dataclass for prefill.
3. `M/tt/generator_vllm.py:169-178` passes it unchanged to the canonical generator.
4. `models/common/sampling/generator.py:63-82` defines the canonical ten-field
   `SamplingParams`, including `log_probs_enable_temperature=False`.
   `OrnithGenerator.__init__` initializes its authoritative sampling metadata
   with this canonical type (`M/tt/generator.py:133-135`).
5. `format_sampling_params` tolerates the missing field while formatting, but
   includes it only if the input dataclass declares it and returns
   `replace(sampling_params, **kwargs)` (`models/common/sampling/generator.py:810-817,873-883`).
   Thus the incoming update remains a `TTSamplingParams` throughout
   `M/tt/sampling_state.py:73-99` and `M/tt/generator.py:464-473`.
6. `SamplingSlotState.reset_parameters` obtains a ten-field canonical previous
   value and a nine-field plugin update, iterates `fields(previous)`, and reads
   `getattr(update, field.name)[row]` (`M/tt/sampling_state.py:418-425`). The
   missing tenth field raises the exact logged `AttributeError`.

This explains why validation and formatting can succeed before the merge fails,
and why the standalone baseline does not exercise this failure: native callers
use the canonical dataclass. Merely adding a formatter call at the adapter is
insufficient; it preserves the foreign type. Attaching an ad hoc attribute is
also insufficient because subsequent dataclass `replace` calls reconstruct it.

**Minimum translation:** when `sampling_params` is present, construct
`models.common.sampling.generator.SamplingParams` in the adapter, copying all
nine plugin fields and supplying `log_probs_enable_temperature=False`. Apply
the same conversion before both canonical prefill and canonical decode. This
normalizes the type at the integration boundary instead of changing the common
slot merger or pinned plugin.

Use the class in `models/common/sampling/generator.py`, not the legacy class
in `models/common/sampling/sampling_params.py`, which also lacks the extra field.

Preserve the raw per-row temperatures, top-k/top-p, all penalties, seeds,
`enable_log_probs`, and `num_logprobs`. Do not pre-invert temperatures, clamp
top-k, substitute an unspecified seed with zero, or turn logprob requests off.
The core formatter already transforms temperature; `_raw_parameters` deliberately
preserves raw temperature for later merges. Plugin `num_logprobs=-2` means no
logprobs, whereas zero requests the sampled-token logprob
(`P/src/vllm_tt_plugin/model_input.py:86-105`). Zero is not a disabled flag.

At TP4, plugin `model_runner.py:4076-4080` routes logprob requests to host
sampling; lines 4033-4040 likewise route unsupported stochastic top-k to host.
The extra field's canonical default is sufficient here because device logprobs
are rejected by `M/tt/generator.py:230-231` and `sampling_state.py:427-429`.
This makes a functioning host path necessary even with sampling mode `all`.

The exception occurs **after** `helper.forward` at `M/tt/generator.py:452`.
That helper performs actual prefill work (`prefill_contract.py:379-424`), so
cache state and queued device work may already exist when the host merge fails.
The schema diagnosis does not rely on treating the request as side-effect-free.

## 2. Next first-request blocker: missing M-RoPE tuple

The flag `--language-model-only` disables modality inputs; it does not remove
the architecture's M-RoPE contract:

- Installed `E/vllm/transformers_utils/config.py:689-726` detects `mrope_section`
  in the HF config or its text config. `M/tests/target_config.json:75-84`
  declares `[11,11,10]`.
- Installed `E/vllm/config/multimodal.py:124-126,515-520` makes modality limits
  zero for language-model-only mode; it does not change that detection.
- Plugin `model_runner.py:353-357` sets `request_specific_rope` from
  `model_config.uses_mrope`. Its `submit_prefill` unconditionally expects
  `(tt_out, rope_deltas)` and calls `rope_deltas[i].item()` when that flag is
  true (`:4141-4147`). The diagnostic3 stack directly confirms this branch.
- Current adapter returns the bare sampled tensor at
  `M/tt/generator_vllm.py:180-181`, or bare host logits at line 185.

Once the schema failure is repaired and prefill returns, the diagnostic's
single-row sampled result has shape `[1,1]`. Unpacking that tensor into two
values raises `ValueError: not enough values to unpack (expected 2, got 1)`.
This is a predicted follow-on failure, not a second exception observed in the
existing log.

**Minimum translation:** retain whether the supplied HF config uses M-RoPE
(using the same installed detection as the plugin), and return
`(output, torch.zeros((B,1), dtype=torch.long, device="cpu"))` for every prefill
on that path. This applies to host and device sampling, including intermediate
chunks. `B` is the submitted prompt row count, not physical slot capacity.

Zero deltas are supported by the reference text-only semantics: installed
`E/transformers/models/qwen3_5/modeling_qwen3_5.py:1912-1916` repeats text
positions across all three rotary channels and explicitly creates zero deltas.
Ornith already indexes absolute text positions in prefill and decode
(`M/tt/multichip_decoder.py:1046-1048,1229-1239`). This translation requires no
new rotary state or vision implementation. The sibling Qwen3-VL adapter uses
the tuple boundary at `models/demos/qwen3_vl/tt/generator_vllm.py:365-369`.

Row correspondence was checked: the ordinary front-packed path removes
unscheduled requests and condenses `input_batch` before preparing inputs
(`P/.../model_runner.py:1026-1092,2337-2344,2881`;
`P/.../input_batch.py:680-682`). Thus the loop at line 4144 covers the same `B`
rows in the same order. `empty_slots` must not index the returned delta vector.
Lane-DP's different resident-row behavior is not a counterexample: the plugin
rejects request-specific RoPE there (`model_runner.py:2921-2929`).

## 3. Next decode blocker: forwarded RoPE metadata is not a core command

Plugin `P/src/vllm_tt_plugin/async_decode.py:1430-1451` passes
`rope_deltas_all_users` whenever `request_specific_rope` is true. It supplies a
list in request order after a layout change/new request, and **still supplies
the keyword with value `None` otherwise**.

`M/tt/generator_vllm.py:187-195` forwards it unchanged. In canonical decode it
is not an explicit parameter (`generator.py:863-881`), so it reaches
`DecodeCommands.from_kwargs`. That parser only permits the six declared
command fields and rejects unknown keys (`decode_contract.py:64-96`). Both
`rope_deltas_all_users=None` and `[0]` therefore raise:

```text
TypeError: Unknown decode command keyword(s): rope_deltas_all_users
```

**Minimum translation:** consume this metadata at the adapter. Accept `None`
and all-zero text deltas, rejecting nonzero values because the model implements
absolute text positions. Do not apply a zero offset again or invent persistent
M-RoPE state. The sibling Qwen3-VL adapter explicitly consumes this keyword
(`models/demos/qwen3_vl/tt/generator_vllm.py:371-379`).

Preserve `slot_remap`, `enable_trace`, `read_from_device`, histories, and the
four version-1 reload commands. The adapter already advertises
`decode_input_update_contract=1`, so the plugin sends those four commands rather
than legacy `reset_batch` (`async_decode.py:1395-1412`). They match the core's
command names; they are not keywords to strip. No evidence supports adding a
generic unknown-keyword filter. Speculation, per-layer multi-group tables, and
deferred grammar options require their own support if enabled; the current
ordinary single-group request does not establish those preconditions.

## 4. Independent blocker: host prefill mistakes a tensor for PrefillBatch

`M/tt/generator_vllm.py:184-185` does:

```python
logits = self.model.host_logits(result.logits)[0, 0]
return torch.stack([logits[slot] for slot in result.row_slots]).unsqueeze(1).clone()
```

However, when `sampling_params is None`, `M/tt/generator.py:453-454` returns
the helper's host result directly. `PrefillContract.forward` stores each host
result in input-row order and concatenates them
(`M/tt/prefill_contract.py:395-424,446-450`). For the default last-logit serving
call this is already a torch tensor `[B,1,V]`. `PrefillBatch` is only returned
when `sample_on_device=True` (`:436-445`); the canonical generator consumes
that internally before returning sampled IDs.

Therefore current host prefill raises `AttributeError: 'Tensor' object has no
attribute 'logits'`, before any tuple-return issue can be reached on that path.

This is required functionality, not an optional fallback: plugin
`model_runner.py:2665-2669` forces a prefill containing any intermediate chunk
onto host sampling so device RNG does not advance for discarded tokens. Host
routing also covers TP4 logprobs and unsupported device sampling controls.
The observed 129-token request fits the 1024-token chunk budget, so it does not
exercise this independent defect.

**Minimum translation:** return/validate the canonical host logits tensor
directly in its existing input-row order, then apply the M-RoPE tuple wrapper
when needed. Do not download it again or gather host rows by `empty_slots`.

## Minimum adapter-owned work and failure order

| Boundary | Required adapter behavior | Evidence/confidence |
|---|---|---|
| Empty multimodal kwargs | Consume the five known keys only when absent/all-`None`; reject actual modalities. | Already implemented, matches diagnostic2-to-3 progression. |
| Sampling input, prefill and decode | Convert plugin dataclass to common `SamplingParams`, retaining all nine fields plus the canonical extra default. | Logged root cause; CPU schema check reproduced and removed it. |
| Host prefill output | Preserve canonical `[B,1,V]` tensor and row order. | Source-confirmed; isolated adapter-method check reproduces `.logits` exception. |
| M-RoPE prefill output | Wrap either sampling output with CPU zero deltas for the submitted `B` text rows. | Active branch in log; single-row unpack failure reproduced on CPU. |
| M-RoPE decode input | Consume/validate `rope_deltas_all_users`; retain explicit reload/remap controls. | Source-confirmed strict-parser failure reproduced for zero and `None`. |

For the diagnostic's device route the order is: empty-kwargs filter (now
passes), schema merge (currently fails), tuple unpack (next predicted failure),
then decode's metadata parsing. For a host/intermediate prefill, the erroneous
`.logits` access is reached instead of the schema merge. The baseline model
implementation avoids these plugin-specific boundaries, so the reported Stage 8
pass is consistent with the Stage 9 failures.

## Other Potential Issues and configuration boundaries

These findings do not explain the logged first-request exception and must not
be presented as additional observed failures.

- **Required trace configuration:** plugin worker defaults to `trace_mode=all`
  (`P/src/vllm_tt_plugin/worker.py:258-266`). This adapter rejects traced
  prefill (`M/tt/generator_vllm.py:110-113`), while canonical decode rejects
  untraced execution (`generator.py:890-891`). The existing configuration must
  retain `tt.trace_mode=decode_only`, as diagnostic3 already does.
- **Capture before live requests:** disabling existing model warmup bypasses
  `model_runner.warmup_model` (`P/.../worker.py:526-531`). Canonical first decode
  can then call `capture` after loading live inputs (`M/tt/generator.py:929-957`);
  capture invokes model and sampler bodies with no local snapshot/restore
  (`:694-724`). Default-enabled warmup already captures before requests. The
  disabled-warmup path needs a focused state-preservation check or adapter
  setup enforcement; no resulting hardware corruption is claimed here.
- **Seed stream continuity on transitions:** the plugin resets device sampling
  state after prefill/layout or mode transitions
  (`P/.../async_decode.py:409-411,457-483`), but supplies prompt/output histories
  only when penalties need them (`P/.../model_runner.py:2679-2694`). Ornith
  reconstructs its seed as the initial hashed device seed plus `len(output)`
  (`M/tt/generator.py:289-305`); missing histories normalize to empty
  (`sampling_state.py:54-57`). Thus a seeded stochastic request without
  penalties can restart its stream after prefill or a later layout transition.
  The schema conversion must preserve seeds, but does not solve this separate
  ownership mismatch. Validate continuity against an uninterrupted seeded run;
  this does not predict failure for every greedy or unseeded request.
- **Resumed-prefill penalty history:** the canonical device-prefill path
  classifies the entire prefix through `prompt_lens` as prompt history and
  supplies no output history (`M/tt/generator.py:474-476`). Plugin execution
  docs explicitly allow resumed prefill to replay generated tokens
  (`P/docs/MODEL_EXECUTION.md:54-55`). Empty output histories clear output
  counts/masks (`M/tt/sampling_state.py:376-381`), which presence/frequency
  penalties consume (`models/common/sampling/tt_penalties.py:58-69`). If that
  route is used, those penalties need separate validation; schema translation
  alone cannot recover the missing prompt/output boundary. This is conditional
  on resume/preemption, absent from diagnostic3.

No additional immediate contradiction was found in the inspected ordinary
decode output/readback contract. `M/tt/generator.py:1017-1075` queues owning
host transfers, records an event, returns `(handle, [event])` for async reads,
waits before host conversion, produces `[B,1]` tokens or `[B,1,V]` logits,
preserves inactive-row sentinels, and retires the submission. Adapter
`generator_vllm.py:197-204` delegates those hooks compatibly. This is a source
contract check, not a hardware or concurrency pass.

## Checks performed and remaining validation

All executable checks used `python_env/bin/python` on the host. AST extraction
executed only selected pure definitions or adapter methods with fake generator
results; no model, plugin, vLLM, or TT module was imported by these harnesses.
No test files were created.

- Extracted the actual plugin/common sampling dataclasses, formatter, and raw
  parameter expansion. The original merge reproduced the exact missing-field
  `AttributeError`; conversion to the common dataclass made the same merge
  succeed. Checked preserved row values, untouched slots, raw temperatures,
  penalties, `None`/integer seeds, and disabled-logprob sentinel. Canonical
  validation still rejected unsupported stochastic top-k and device logprobs.
- Extracted actual adapter `prefill_forward` with a fake canonical host result:
  `[1,1,8]` CPU tensor reproduced `AttributeError: 'Tensor' object has no
  attribute 'logits'`.
- Extracted actual adapter device-prefill method with fake `[[7]]` result:
  the plugin's two-value unpack reproduced the predicted `ValueError`.
- Extracted actual `DecodeCommands`: both `None` and `[0]` RoPE metadata
  reproduced its unknown-keyword `TypeError`. Removing that adapter metadata
  allowed the four version-1 commands plus a CPU tensor slot remap to parse.

After the first draft, findings were checked again against callers, callees,
and failure ordering. In particular, prefill execution was not conflated with
decode-replay counters; the apparent request-row/delta-vector mismatch was
ruled out for the supported path; and state-transition/configuration concerns
were retained only in the secondary section above.

Suggested later validation, outside this inspection-only task: host contract
checks for both sampling routes and all output wrappers, then a fresh reduced
server request producing multiple tokens; long/mixed chunked prefill; logprobs
including zero; unsupported-device top-k host routing; penalties and seeded
transitions; slot reuse/remap and request release; resumed prefill; and finally
the complete model's Stage 9 quality/performance gates. No hardware result,
successful serving request, or stage pass is claimed by this report.

## 2026-11-13 — mixed unknown-to-known prefill horizon

Independent read-only source audit by resume_contract reproduced a failing
transition using the actual extracted contract/runtime validators: A original4096
ends512 alongside resumed B original129/end130. The conservative adapter makes
the whole batch horizon unknown, so A caches a512 profile. Next A alone ends1024
and receives known horizon4096. The helper checks reconstruction only when
request_lens is None, so the runtime rejects incompatible cached profiles. A
non-aligned unknown-to-known transition can instead lack its required tail
snapshot even with compatible profile buckets. Both are rejected before device
writes. Existing mocks did not check these real runtime invariants.

The earliest inconsistent transition is the helper's restriction of compatibility
checking to unknown destination horizons. Keep runtime assertions, and apply the
existing reconstruction checks to continuation transitions regardless of whether
the new horizon is known. Explicit caller permission to omit history retains its
existing negative-control semantics. A focused regression must use the real
profile and boundary validation, not only a fake accepting runtime.
