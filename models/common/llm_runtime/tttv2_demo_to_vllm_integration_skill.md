---
name: integrate-tttv2-demo-model-with-vllm
description: Audit and integrate a TTTv2 text model that currently runs only through demo code into the TT vLLM plugin. Use when an agent must determine which provider, model, executor, generator, policy, KV-cache, data-parallel, registration, serving-geometry, and three-tier hardware contracts already exist; identify exactly what is missing; implement only the missing model-owned pieces; register the exact HF model; and qualify it with server smoke, benchmark, and deterministic quality evidence.
---

# Integrate a TTTv2 demo model with vLLM

Bring one TTTv2 text model from direct-demo execution to validated vLLM serving.
Treat demo success as tensor-model evidence only, not proof of the executor,
lifecycle, façade, policy, KV, DP, registration, or serving contracts. Audit
first, publish the gap report, then implement the smallest missing layer.

## Eligibility gate

Use this skill only for a model that already has a working TTTv2 demo
implementation. Do not use it to evaluate models that are merely demo-only,
planned TTTv2 ports, or non-TTTv2 packages.

Before doing any contract audit, prove that the demo path instantiates a TTTv2
tensor model. If that cannot be shown from local code and existing demo
evidence, stop and report `NOT_TTTV2_TARGET`.

## Authoritative local references

Read `models/common/models/llama3_8b/README.md` and its
`{hf_adaptor,model,executor,generator}.py` as the package/lifecycle reference.
Read `models/common/tests/llm_runtime/test_executor_integration.py` for exact
facade contracts and `models/common/llm_runtime/vllm_adapter.py` for external
normalization and late KV binding. Read R1-R6 and the test contract in
`tttv2_sibling_model_vllm_registration_plan.md`, plus
`/localdev/gwang/vllm_duo/vllm/tttv2_vllm_integration_policy_design.md`.
Search the lessons-learned and model-specific-fixes documents by heading and
target architecture. Use the hardware runner, validator, and one completed
expectations JSON as the executable evidence-schema reference.

## Hard boundaries

- Limit vLLM product changes to the exact `HF_MODEL` mapping and necessary
  family-version branch in `plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`.
  Preserve TTTv1 defaults and invalid-version errors. Do not modify `loader.py`,
  vLLM core/execution/tests, CI, packaging, or tt-inference-server. Do not add
  a generic dispatcher, default generator, or fallback.
- Keep special treatment inside the target model package. If a reproduced
  three-tier failure appears to require `models/common/llm_runtime/`,
  `models/common/modules/`, or `VLLMAdapter`, write the reduction and proposed
  shared fix to `tttv2_sibling_model_runtime_migration_lessons_learned.md`,
  stop, and obtain explicit approval before editing shared code.
- Record every model-local production exception in
  `tttv2_sibling_model_model_specific_fixes.md` with evidence, before/after
  delta, possible shared owner, promotion test, and removal criterion.
- Use `/localdev/gwang/vllm_duo/tt-metal-too/python_env`. Use
  `HF_HOME=/proj_sw/user_dev/huggingface` and offline snapshots. Do not
  download an existing model again.
- Serialize all TT hardware work. The physical host is an eight-device T3K;
  select N150/N300 subsets on it when the demo geometry requires them, but do
  not run subset or full-T3K jobs concurrently.
- Treat Tier 0/1/2 as the complete hardware acceptance set. The models-side
  host contract test is a required readiness gate before hardware acceptance;
  it is not a substitute for Tier 0/1/2. Do not run or repair vLLM test suites.

## Start with a read-only characterization

Define the target facts before judging any contract:

If the package supports multiple checkpoints, select one exact HF model first;
do not infer one implementation or serving matrix for the whole family.

```text
model package:
demo entry point and exact working command:
HF model ID and pinned revision:
HF architecture and expected TT architecture:
tokenizer snapshot and required files:
supported physical mesh shapes:
TP per lane and supported DP counts:
per-lane maximum batch size:
maximum context length:
paged-attention block size:
supported trace modes and sequence buckets:
device-sampling capability and supported sampling modes:
fabric per geometry:
weight-cache root and topology keys:
direct-demo accuracy/performance evidence:
```

Derive these facts from the demo's model-parameter map, configuration builder,
checkpoint adapter, and accepted demo command. Do not invent vLLM rows that
the demo does not support. Distinguish these concepts:

- physical host: the eight-device T3K machine;
- selected mesh: the devices assigned to one row;
- TP per lane: devices used by one model replica;
- DP: number of independent lanes;
- global admission: `DP * per-lane max_num_seqs`, never `TP * max_num_seqs`.

Use read-only searches such as:

```bash
MODEL_DIR=models/common/models/<model>
rg -n "from_pretrained|DEFAULT_HF_REVISION|mesh|cluster_shape|tt_data_parallel|max_batch|max_context|block_size|trace|prefill|sampling|fabric|cache" "$MODEL_DIR" models/common/tests/demos
rg -n "<HF model ID>|<generator class>" /localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
```

## Use explicit audit statuses

Assign exactly one status to every contract:

- `SATISFIED`: implementation and current evidence both exist.
- `IMPLEMENTED_UNVERIFIED`: code appears conforming but lacks execution evidence.
- `PARTIAL`: some required surface exists and the missing pieces are named.
- `MISSING`: the required layer or behavior does not exist.
- `INCORRECT`: code exists but violates the contract.
- `UNSUPPORTED_GEOMETRY`: the demo/model cannot support the proposed row.
- `BLOCKED_SHARED_CHANGE`: the apparent owner is shared code; stop for approval.

Never report `SATISFIED` from symbol presence alone.

## Contract C1: provider adapter and model product

Require the provider layer, normally `hf_adaptor.py`, to:

1. resolve the exact HF model and pinned revision;
2. load config and tokenizer from the pinned offline snapshot;
3. derive architecture and runtime metadata;
4. load and translate the state dict, including fused-weight splitting and
   layout conversion where required;
5. construct the TTTv2 tensor-model configuration and model;
6. expose generation defaults and cache-location policy; and
7. return one model product containing the tensor model, tokenizer, runtime
   config, and metadata needed by the executor.

Check:

```bash
rg -n "def from_pretrained|AutoConfig|AutoTokenizer|DEFAULT_HF_REVISION|state_dict|model_cache|runtime_config" "$MODEL_DIR"
```

Trace the working demo from its entry point to model construction. Confirm
that no demo-global object, CLI parser, or mutable environment side effect is
required after `from_pretrained` returns.

Report as missing when the demo constructs weights and runtime state directly
inside the demo, lacks a reusable product object, silently follows `main`
instead of a required pinned revision, or derives a relative weight-cache path
from the caller's current directory without an explicit recorded policy.

## Contract C2: tensor model and runtime hooks

Require the tensor model to expose:

- prefill and decode graph entry points;
- embedding and rotary/input preparation hooks required by the common runtime;
- paged-attention metadata and transactional KV bind/unbind support;
- prefill post-processing for logits and, when supported, sampled tokens;
- decode output gathering and position/state updates;
- `iter_executor_named_modules()` or the equivalent module-validation surface;
- static sampling configuration when device sampling is supported; and
- deterministic cleanup-safe behavior when the cache is not yet allocated.

Check the target against the closest completed sibling and the llama3_8b
reference:

```bash
rg -n "def (prefill_forward|decode_forward|set_kv_cache|iter_executor_named_modules|post_process_prefill_output|process_output)" "$MODEL_DIR"
rg -n "PagedAttention|Sampling1D|LMHead|vocab_size|padded_vocab" "$MODEL_DIR/model.py"
```

Do not equate a demo's one-shot forward method with this contract. Identify
which input staging, KV ownership, output normalization, or sampling behavior
still lives in demo orchestration and must move behind model-owned hooks.

## Contract C3: model-owned executor and lifecycle

Require `executor.py` to compose the model with common runtime owners:

```text
model
├── PagedKVCacheManager
├── OutputReader
├── PrefillRuntime
├── DecodeRuntime
├── ProgramCompiler
├── EagerExecutor
├── optional TraceCompiler
├── optional TracedExecutor over the same EagerExecutor
└── WarmupCoordinator
```

Require exact executor call surfaces matching
`test_executor_integration.py`: `compile_prefill`, `compile_decode`,
`prefill_forward`, `decode_forward`, `read_decode_output`,
`process_decode_output_host`, `can_trace_prefill`, `warmup_model_prefill`,
`warmup_model_decode`, `configure_paged_kv_cache`, `allocate_kv_cache`, and
`cleanup`.

Check:

```bash
rg -n "class .*Executor|def (compile_prefill|compile_decode|prefill_forward|decode_forward|read_decode_output|process_decode_output_host|can_trace_prefill|warmup_model_prefill|warmup_model_decode|configure_paged_kv_cache|allocate_kv_cache|cleanup)" "$MODEL_DIR/executor.py"
```

Verify ownership, not just signatures:

- Construct one eager executor and let trace execution wrap that exact eager
  executor.
- Keep eager execution available before and after trace activation.
- Make cleanup ordered, retryable, idempotent, and exception-safe.
- Release externally leased decode outputs before tearing down runtime and
  trace owners.
- Treat KV tensors passed by vLLM as borrowed resources.
- Recompute prefill, decode, warmup, and page-table layouts after late physical
  KV capacity is resolved and before allocation/warmup.

Mark the contract `PARTIAL` if an executor exists but still calls demo helpers
for lifecycle, owns a second execution path, uses traced execution as an eager
fallback, allocates KV tensors before vLLM supplies physical geometry, or lacks
asynchronous decode-output handling.

## Contract C4: vLLM-facing generator façade

Require `generator.py` to own normalization and dispatch but no TT tensors.
For DP1 target one executor; for DP>1 target `LaneGroupExecutor` over one
executor per submesh.

Require `model_capabilities`, `requires_prefill_trace_warmup`, plugin-facing
properties (`model`, `model_args`, `mesh_device`, `cache_path`, and
`already_warmed_up_prefill`), `get_max_tokens_all_users`, and
`initialize_vllm_model` with the established loader arguments. Require
`allocate_kv_cache` to support model-owned allocation and the legacy
three-argument late-binding call. Normalize through `VLLMAdapter`, then
delegate compilation, forwards, warmup, asynchronous output, and cleanup.

Check:

```bash
rg -n "class .*Generator|model_capabilities|requires_prefill_trace_warmup|initialize_vllm_model|get_max_tokens_all_users|allocate_kv_cache|VLLMAdapter|LaneGroupExecutor|def (compile_prefill|compile_decode|prefill_forward|decode_forward|read_decode_output|process_decode_output_host|warmup_model_prefill|warmup_model_decode|cleanup)" "$MODEL_DIR/generator.py"
```

Once the facade exists, add its binding to the models-side
`test_executor_integration.py` and run that file with `python_env`. Treat the
result as structural support evidence, not an acceptance tier.

Inspect `initialize_vllm_model` directly. For vLLM construction, require the
maximum supported static ceiling:

```text
trace_mode="all"
device_sampling_enabled=True
```

Keep direct-demo config defaults independently controllable; changing the
vLLM classmethod default must not force all demos to enable tracing or device
sampling.

Mark C4 `MISSING` when only demo builders exist. Mark it `INCORRECT` when the
generator owns TT resources, drops dynamic inputs, returns compact DP output,
uses a default/fallback model, or requires loader-specific TTTv2 arguments.

## Host contract readiness gate

After C1-C7 appear implemented and before writing or running the hardware
expectations, add the target model to:

```text
models/common/tests/llm_runtime/test_executor_integration.py
```

This is the readiness gate for the model-owned vLLM contract. It checks the
executor/generator surface, ownership composition, late KV binding, trace/eager
dispatch policy, construction defaults, DP lane grouping, capacity reporting,
and cleanup behavior without using TT hardware.

Run it with the repo environment:

```bash
PYTHONPATH=/localdev/gwang/vllm_duo/tt-metal-too \
  /localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/pytest -q \
  /localdev/gwang/vllm_duo/tt-metal-too/models/common/tests/llm_runtime/test_executor_integration.py
```

If the target has no binding in this test, classify C3/C4/C6/C7 as
`IMPLEMENTED_UNVERIFIED` at best. If the binding cannot be written without
demo-only globals, real TT tensors, loader-specific arguments, or special-case
test hacks, classify the violated contract as `INCORRECT` or `PARTIAL` and fix
the model-owned layer before hardware work.

Do not modify vLLM tests for this readiness gate. This is a tt-metal models
contract test only.

## Contract C5: dynamic trace and sampling policy

Enforce static configuration as a ceiling and dynamic inputs as selectors:

| Static capability | Dynamic input | Required result |
| --- | --- | --- |
| trace mode enabled | `enable_trace=True` | replay eligible captured trace |
| trace mode enabled | `enable_trace=False` | execute eagerly |
| trace mode disabled | `enable_trace=True` | clear error |
| device sampling enabled | `can_sample_on_device=True` | warm sampling variants |
| device sampling enabled | `can_sample_on_device=False` | skip sampling warmup for that phase |
| device sampling disabled | `can_sample_on_device=True` | clear error |
| device sampling enabled | `sampling_params` present | return sampled tokens |
| device sampling enabled | no `sampling_params` | return logits for host fallback |
| device sampling disabled | `sampling_params` present | clear error |

Verify all four warmup combinations of `enable_trace` and
`can_sample_on_device`. In particular, accept eager sampling warmup
(`False, True`) before trace capture.

Require eager calls to bypass graph-key construction and lookup. Require
missing-key rejection only when a traced, trace-eligible request needs an
uncaptured trace. Do not let `trace_mode="all"` force a decode-only warmup to
capture prefill traces.

## Contract C6: paged-KV late binding

Require construction to establish a safe maximum KV ceiling without allocating
physical tensors. When vLLM later calls:

```text
generator.allocate_kv_cache(kv_cache_shape, dtype, num_layers)
```

require this order:

```text
VLLMAdapter.resolve_legacy_kv_cache_config
→ target.configure_paged_kv_cache
→ refresh page-table/prefill/decode/warmup layouts
→ target.allocate_kv_cache
→ bind per-layer K/V tensors
```

Check the model's block size, local KV heads, head dimension, layer count,
dtype, maximum blocks, and physical blocks. Reject partial three-argument
input atomically. Do not hard-code demo KV geometry into vLLM allocation.

## Contract C7: DP and output geometry

Require one complete model/executor per DP lane. Use `LaneGroupExecutor` for
submesh creation, global-slot routing, lane-local slot translation, decode
splitting, output aggregation, warmup, cache configuration, and cleanup.

Require sampled-token host output to preserve global slot shape:

```text
DP * per-lane batch capacity
```

Use signed `torch.int64` host tokens. Do not accept compact active-user output
as a fallback. Keep physical LM-head/tile row geometry distinct from logical
sampling users; normalize at the runtime ownership boundary before sampling.

Classify impossible TP/DP layouts as `UNSUPPORTED_GEOMETRY`, supported layouts
with missing lane composition as `MISSING`, and shape failures on supported
layouts as `INCORRECT`.

## Contract C8: strict plugin registration

Verify the HF config's `architectures` entry and the TT-prefixed architecture
that vLLM registers. Add one exact `HF_MODEL` → generator import string under
the architecture family's explicit `tt_transformers_v2` branch in:

```text
/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
```

Require missing or unknown `HF_MODEL` to fail during startup with the supported
set named. Preserve the family variable's unset behavior and all TTTv1 paths.

Before and after registration, enforce the vLLM allowlist:

```bash
cd /localdev/gwang/vllm_duo/vllm
git status --short
git diff --name-only
git ls-files --others --exclude-standard
```

The integration diff may contain only
`plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`. Do not add or modify
vLLM tests.

## Contract C9: serving geometry and expectations

Create schema-v2 `tttv2_<model>_vllm_expectations.json`. Freeze the exact model,
architecture, generator, canonical row IDs, primary decode-only smoke row, and
every demo-supported platform/mesh/TP/DP geometry. Record allowed trace modes,
trace/compile counts, per-rank/global capacity, model length, trace-region
size, fabric, async/device-sampling/prefix-caching policy, model/tokenizer
revision, verified offline snapshot files, explicit weight-cache policy, and
quality budget/semantic groups.

Use both `decode_only` and `all` only where prefill traces are supported. A
device/TP lane with no prefill trace buckets gets a decode-only row; requesting
`all` there is an operator error, not a useful failure test.

Use `max_model_len = min(model context ceiling, 32768)` unless the demo proves
a lower limit. Never exceed per-lane batch or KV capacity. Treat trace-region
size as configurable execution policy, not model geometry; increase and rerun
when an otherwise supported trace does not fit.

Validate expectations and commands without touching hardware:

```bash
./tttv2_vllm_hardware_gate_runner.sh \
  --tier benchmark \
  --expectations tttv2_<model>_vllm_expectations.json \
  --artifact-root /tmp/tttv2_<model>_dry_run \
  --dry-run
```

## Contract C10: three-tier hardware acceptance

Run only after C1-C9 are implemented or explicitly classified.

### Tier 0: server smoke

Run the declared primary decode-only row. Require reset, healthy server, one
temperature-zero completion, exact completion budget, coherent on-topic text,
clean shutdown, successful reset-after, zero forbidden scanner hits, and an
empty final process inventory.

Use Tier 0 for the development loop. Do not repeatedly run full matrices after
every edit.

### Tier 1: benchmark matrix

Run every canonical geometry and trace mode into a fresh root. Require every
row to complete 320/320 requests with zero failures, exact trace/program
counts, clean lifecycle, zero scanner hits, and empty process inventories.
Record duration, request throughput, output and total token throughput, TTFT,
TPOT, and comparable decode-only/all deltas.

### Tier 2: deterministic quality matrix

Run the same rows with the fixed rain/road prompt, temperature zero,
`ignore_eos=true`, and the declared token budget. Require exact token count,
finish `length`, all semantic groups, clean lifecycle, and human acceptance of
every applicable decode-only/all pair. Compare meaning, not token identity;
reject material contradiction, corruption, or trace-mode semantic divergence.

Run each tier with:

```bash
./tttv2_vllm_hardware_gate_runner.sh --tier <smoke|benchmark|quality> \
  --expectations tttv2_<model>_vllm_expectations.json \
  --artifact-root <fresh-tier-root>
```

After human review of `quality_review.json`, rerun the quality command against
the same root with `--validate-only`.

Do not accept runner completion alone. Require the validator's final exit zero.

## Diagnose failures by boundary

Use the earliest primary failure, not a downstream symptom:

| Failure point | Investigate first |
| --- | --- |
| before import | plugin family branch, exact HF mapping, architecture name |
| checkpoint/tokenizer startup | revision, snapshot files, geometry, offline cache |
| KV allocation | late-binding order, block size, heads, dtype, physical capacity |
| warmup | eager-before-trace order, program coverage, trace plan, sampling variants |
| first async request with empty deque | preserve as secondary; use a temporary no-async diagnostic to expose the model exception, then restore canonical async mode |
| missing program after trace activation | eager program coverage for the actual Q/row/batch signature |
| uncaptured trace key | trace eligibility, supported-plan registration, final capture timing |
| row-end/tensor-row mismatch | physical LM-head rows versus logical sampling users; check the sampling boundary contract |
| partial Tier 1 failures | admission groups, preemption/resume, compiled batch signatures, KV budget |
| trace-region overflow | increase the configurable region and rerun the row |
| firmware/reset failure | classify as hardware lifecycle evidence; reset and rerun, do not blame model correctness |
| Tier 2 pair divergence | exact inputs/revisions first, then trace/eager output handling and sampling determinism |

Keep failed and diagnostic artifacts immutable. Do not promote partial-run
throughput as a performance baseline. Use `null` for metrics that do not exist.

## Produce the pre-implementation gap report

Before editing, create this table in the task report or model W0 document:

| Contract | Status | Evidence found | Exactly missing or incorrect | Smallest owner | Verification needed |
| --- | --- | --- | --- | --- | --- |
| C1 provider/product |  |  |  |  |  |
| C2 model/runtime hooks |  |  |  |  |  |
| C3 executor/lifecycle |  |  |  |  |  |
| C4 generator façade |  |  |  |  |  |
| C5 trace/sampling policy |  |  |  |  |  |
| C6 late KV binding |  |  |  |  |  |
| C7 DP/output geometry |  |  |  |  |  |
| C8 plugin registration |  |  |  |  |  |
| C9 serving geometry |  |  |  |  |  |
| C10 three-tier evidence |  |  |  |  |  |

For every non-satisfied row, state a concrete missing symbol, behavior,
configuration fact, evidence row, or approval. Avoid vague findings such as
“needs vLLM support.”

End W0 with one maturity classification:

```text
DEMO_ONLY
MODEL_RUNTIME_READY
EXECUTOR_READY
GENERATOR_READY
REGISTERED_UNQUALIFIED
FULLY_QUALIFIED
BLOCKED_SHARED_CHANGE
```

## Implement in dependency order

Implement only missing contracts, in this order:

1. C1 provider/product.
2. C2 model hooks.
3. C3 executor and lifecycle.
4. C5 and C6 runtime policy/KV behavior inside the model-owned executor.
5. C4 generator façade and C7 lane composition.
6. C9 expectations from proven demo geometry.
7. C8 exact registration.
8. C10 Tier 0, then Tier 1, then Tier 2.

After each failure, update the gap report and record whether the evidence
changed the owner or merely revealed the next missing contract. Do not broaden
the design to solve hypothetical future models.

## Completion output

Do not declare completion until the handoff includes the final C1-C10 table,
exact registration/revision facts, accepted geometry, Tier 0 response and
lifecycle, every Tier 1 row's throughput/latency/trace measurements and
deltas, and every Tier 2 row's token/semantic/pair result. Include functional
and performance before/after deltas where comparable; otherwise explain a
`null` delta. List model-local fixes, shared-adoption candidates, unsupported
geometries, scope proof that vLLM changed only `platform.py`, and artifact
roots/validator outcomes as provenance rather than substitutes for results.

Treat a plan, import, healthy server, or passing smoke alone as incomplete. Full
qualification requires every declared row and zero unresolved contract gaps.
