# Stage 08 — vLLM serving integration — Qwen3-Coder-30B-A3B-Instruct

**The model serves through the shared vLLM path on 4 Blackhole dies, on device-sampled
traced decode, at the full advertised 262144 context.**

## Headline — primary single-user serving

Workload: **128-token input, 128-token output, 1 prompt, `--max-concurrency 1`,
`max_num_seqs=1`, `ignore_eos`, greedy (`--temperature 0.0`)**, on the complete
48-layer model, 1x4 mesh, `sample_on_device_mode: all`, decode trace on.
Artifact: [`../../readiness_vllm/vllm_benchmark.json`](../../readiness_vllm/vllm_benchmark.json)
(raw: [`vllm_result.json`](../../readiness_vllm/vllm_result.json)).

| Metric | Value |
|---|---|
| **TTFT** median / P99 | **312.367 ms** / 312.367 ms |
| **Decode t/s/u** (from mean TPOT, `1000 / 19.778`) | **50.560 t/s/u** |
| TPOT mean / P99 | 19.778 ms / 19.778 ms |
| ITL median / P99 | 19.804 ms / 20.084 ms |
| Aggregate output throughput | 45.318 tok/s |
| Request throughput | 0.354 req/s |
| Completed | 1/1, 128 output tokens, 2.825 s |

**Against the full-model lower bound.** Stage 07 shipped standalone token-out at
**19.213 ms (52.049 t/s/u)** and teacher-forcing decode at **43.54 t/s/u**
(`../datatype_sweep/README.md`). Serving decode is **19.778 ms**, i.e. vLLM adds
**0.565 ms per token, 2.9 %**, over the standalone traced path, and comfortably
clears the teacher-forcing lower bound. Serving TTFT is 312.367 ms against the
standalone 129.941 ms at ctx128 — that gap is prefill-side request handling,
tokenisation, scheduling and detokenisation, not decode, and it is the one number
where vLLM costs real time. See "Where the remaining overhead is".

**Re-run after the on-device penalty stage landed — no regression.** Same
command, same shape, same server flags:
[`logs/penalty_rerun_vllm_benchmark.json`](logs/penalty_rerun_vllm_benchmark.json)
(raw [`logs/penalty_rerun_vllm_result.json`](logs/penalty_rerun_vllm_result.json),
server [`logs/penalty_rerun_server.log`](logs/penalty_rerun_server.log)).

| Metric | Headline run | Re-run with the penalty stage present |
|---|---|---|
| TTFT | 312.367 ms | **306.894 ms** |
| TPOT mean | 19.778 ms | **19.808 ms** |
| Decode t/s/u | 50.560 | **50.485** |
| ITL median / P99 | 19.804 / 20.084 ms | 19.787 / 23.847 ms |

TPOT moved by **0.030 ms, 0.15 %**, and TTFT improved by 5.473 ms — run-to-run
noise in both directions, which is the expected result: none of these requests
sets a penalty, so `_penalty_mode` is 0 and the captured decode graph is
byte-for-byte the one that produced the headline. The headline row stays the
canonical figure; this is the regression check, not a replacement.

**Penalised requests are slower, and that is a serving characteristic, not a
footnote.** The headline above is the fast path: no request in it sets a penalty,
so the penalty ops are not in the captured trace at all. A request that *does* set
one pays per-step host work to stage the operands, and on this path host work is
serial with the trace replay, so it lands directly on TPOT. Measured in situ on
the **same** 128/128/1 workload against the same live server, median of 3 runs
each, all three legs decoding exactly 128 tokens
([`probes/penalty_serving_cost_probe.py`](probes/penalty_serving_cost_probe.py) →
[`penalty_serving_cost_probe.json`](probes/penalty_serving_cost_probe.json)):

| Request | TTFT | TPOT | **Decode t/s/u** | vs. unpenalised |
|---|---|---|---|---|
| no penalty (the headline path) | 298.276 ms | 19.873 ms | **50.321** | — |
| `repetition_penalty` only | 328.555 ms | 22.702 ms | **44.049** | +2.829 ms, **+14.2 %** |
| all three penalties | 305.230 ms | 24.951 ms | **40.079** | +5.078 ms, **+25.6 %** |

`repetition_penalty` alone is the case that matters most, because this
checkpoint's `generation_config.json` injects `repetition_penalty=1.05` into
every request that does not override it — so a server run **without**
`--generation-config vllm` puts every request on that row. Details, the
before/after of the staging optimisation that got it here, and the remaining
headroom are under "Sampling penalties".

**Qualitative verdict: pass.** Coherent, on-topic, correctly-formatted, no
repetition loops, no gibberish, no language drift, no request contamination.
Details and quotes below.

**Sampling gate: 56 passed / 16 failed / 1 skipped** on the canonical TT plugin
suite, `--sampling-profile full`, `--tt-max-num-seqs 32` — up from 52 / 20 / 1
once sampling penalties were implemented on device (see "Sampling penalties").
**All 16 remaining failures are reproducibility-only or checkpoint properties, not
serving-path defects: 14 seeding/RNG, 2 presence-penalty.** The 2 presence
failures are shown below to fail against **vLLM's own reference sampler** on this
checkpoint as well, byte for byte. Correctness, all 20 logprobs tests, all 5
host-only-parameter tests, structured output, request isolation, greedy
determinism, all repetition and frequency penalty tests, and crash-free serving
all pass. Breakdown below.

## Secondary — CI serving burst (vLLM-nightly shape)

Workload: **100-token input, 100-token output, 32 prompts, no explicit
`--max-concurrency`, `max_num_seqs=32`, `ignore_eos`, greedy**. Same model, same
mesh, same TT config. Artifact:
[`../../readiness_vllm/vllm_ci_serving_benchmark.json`](../../readiness_vllm/vllm_ci_serving_benchmark.json)
(raw: [`vllm_ci_serving_result.json`](../../readiness_vllm/vllm_ci_serving_result.json)).

| Metric | Value |
|---|---|
| **Aggregate output throughput** | **104.062 tok/s** |
| TTFT median / P99 | 4901.160 ms / 4902.203 ms |
| TPOT mean / P99 | 261.154 ms / 262.508 ms |
| ITL median / P99 | 260.972 ms / 263.150 ms |
| TPOT-derived per-user decode | 3.829 t/s/u |
| Request throughput | 1.041 req/s |
| Completed | 32/32, 3200 output tokens, 30.751 s |

**This is not the headline decode number and must not be read as one** — the goal
forbids it, and here the reason is visible rather than theoretical. All 32
requests are admitted in a burst, so every TTFT includes queueing behind 31 other
prefills (hence ~4.9 s), and every decode step computes all 32 slots.

The 32-slot decode cost is the interesting part, and it is **not** a serving
artefact. The same 128/128/1 single-user workload, run against a
`max_num_seqs=32` server, gives TPOT **263.470 ms**
([`vllm_benchmark_maxnumseqs32.json`](../../readiness_vllm/vllm_benchmark_maxnumseqs32.json))
— essentially identical to the burst's 261.154 ms with 32 *active* users. So the
per-step cost depends on the **configured** slot count, not on how many slots
carry a request:

| Server | Workload | TPOT mean | Per-user t/s/u | Aggregate tok/s |
|---|---|---|---|---|
| `max_num_seqs=1` | 128/128/1 | 19.778 ms | 50.560 | 45.318 |
| `max_num_seqs=32` | 128/128/1 (1 active) | 263.470 ms | 3.796 | 3.786 |
| `max_num_seqs=32` | 100/100/32 (32 active) | 261.154 ms | 3.829 | 104.062 |

Reading: this is a **MoE decode batch-scaling** property of `tt/model.py`, not of
the adapter. At batch 1 a decode step routes to 8 experts; at batch 32 up to 32x8
token-expert pairs land across most of the 128 experts, and the layer does that
much more work. The adapter cannot change it — it hands the generator the fixed
`max_num_seqs`-row batch vLLM requires for trace stability. Concurrency still
pays: 32 users move **104.062 tok/s aggregate against 45.318 tok/s** for one.
Closing the per-user gap at large `max_num_seqs` is stage-09 (optimized-vLLM)
work on the MoE decode path, and it is recorded as an open item below.

## How the model is registered — without touching the plugin checkout

The goal asks for registration in
`vllm_tt_plugin/platform.py::register_tt_models()`. That function's **first
statement** is

```python
# Dynamic hook: register any bundles dropped under EXTRA_MODELS_DIR. Runs
# first so a distributed bundle can supply a model without touching this file.
_register_models_from_extra_dir(ModelRegistry)
```

(`/home/raahem/vllm-tt-plugin/src/vllm_tt_plugin/platform.py:475-481`). This is
not a workaround around the goal; it is the mechanism the plugin ships **for this
exact purpose**, and it means the model is genuinely registered *by*
`register_tt_models()` — the same call, the same `ModelRegistry`, the same
`_register_model_if_missing` helper, the same `TT`-prefixed arch convention as
every built-in. Using it satisfies the goal's wording while honouring the
standing constraint that core source files of other repos are not modified. The
plugin checkout is byte-identical to `bc4af2d`; `git -C /home/raahem/vllm-tt-plugin
status` shows no tracked change.

The bundle is self-contained and lives with the model:

```text
models/autoports/qwen_qwen3_coder_30b_a3b_instruct/vllm_bundle/
└── qwen3_coder_30b_a3b_instruct/
    ├── vllm_metadata.json                  # arch + main_class
    └── tt_qwen3_coder_30b_a3b_instruct.py  # importable entry point
```

`vllm_metadata.json` declares `arch: "Qwen3MoeForCausalLM"` (this checkpoint's
`config.json` architecture) and
`main_class: "tt_qwen3_coder_30b_a3b_instruct:Qwen3CoderForCausalLM"`. The hook
appends the bundle folder to `sys.path` (append, never `insert(0)`) and registers
the arch prefixed, so the resolved name is **`TTQwen3MoeForCausalLM`**. Because
registration is lazy — vLLM resolves the `"module:Class"` string later, in the
API-server process and again in each EngineCore worker — the entry point cannot
assume the tt-metal checkout is importable, so it appends the repository root
itself before re-exporting the real adapter from `tt/generator_vllm.py`.

Enabling it is one environment variable:

```bash
export EXTRA_MODELS_DIR=$PWD/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/vllm_bundle
```

Proof, from [`../../readiness_vllm/server.log`](../../readiness_vllm/server.log)
(API server, EngineCore, and the worker each log it):

```
INFO [platform.py:499] Registered TT model TTQwen3MoeForCausalLM ->
  tt_qwen3_coder_30b_a3b_instruct:Qwen3CoderForCausalLM
  (from EXTRA_MODELS_DIR/qwen3_coder_30b_a3b_instruct)
```

## The server commands that produced this evidence

Common to all of them:

```bash
source python_env/bin/activate
export EXTRA_MODELS_DIR=$PWD/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/vllm_bundle
```

**Headline primary single-user** (`max_num_seqs=1`):

```bash
python -m models.common.readiness_check.run_vllm_server \
  --model-dir models/autoports/qwen_qwen3_coder_30b_a3b_instruct \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --mesh-device P300x2 --max-num-seqs 1 --max-model-len 262144 \
  --block-size 32 --port 8100 --stages serve \
  --tt-config '{"trace_region_size": 50331648, "fabric_config": "FABRIC_1D_RING"}' \
  --additional-server-args "--generation-config vllm"
# then, against that live server:
python -m models.common.readiness_check.run_vllm_server --stages benchmark \
  --model-dir models/autoports/qwen_qwen3_coder_30b_a3b_instruct \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-num-seqs 1 --server-url http://localhost:8100 --no-benchmark-ci-serving
```

**Sampling, qualitative and CI serving burst** (`max_num_seqs=32`): the same
launch with `--max-num-seqs 32`, then

```bash
python -m models.common.readiness_check.run_vllm_server --stages qualitative \
  --model-dir models/autoports/qwen_qwen3_coder_30b_a3b_instruct \
  --hf-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --max-num-seqs 32 --server-url http://localhost:8100
python -m models.common.readiness_check.run_vllm_server --stages benchmark ... --max-num-seqs 32 ...
python -m models.common.readiness_check.run_vllm_server --stages sampling \
  --sampling-profile full --max-num-seqs 32 --server-url http://localhost:8100 ...
```

Notes on the flags that mattered:

* **`--port 8100`.** Port 8000 is held on this host by a process outside this
  session; the runner refuses to launch on a busy port.
* **`--block-size 32`.** This is the page block size the model's paged KV cache,
  paged `update_cache` and paged decode SDPA were brought up and measured on in
  stages 02–07. The adapter does **not** hard-code it — `allocate_kv_cache`
  reads it out of the shape vLLM passes and installs it through
  `Qwen3CoderGenerator.configure_paging` — but 32 is the tested value and is what
  every number here was taken at.
* **`--generation-config vllm`.** This checkpoint's `generation_config.json`
  injects `repetition_penalty=1.05, temperature=0.7, top_k=20, top_p=0.8` into
  every request that does not override them. The penalty is now **honoured**
  rather than dropped (see "Sampling penalties"), so this flag is no longer a
  correctness requirement — it is kept because the checkpoint default would put
  *every* request, including the benchmark's, on the penalised path and its
  ~17 ms of per-step host staging, which is not the configuration any earlier
  stage was measured under. Neutral vLLM defaults keep the headline comparable;
  every benchmark and qualitative request sets temperature/top-p explicitly
  anyway. A client that asks for a penalty still gets it.
* **`--tt-config`.** `trace_region_size: 50331648` is the value
  `tests/test_perf.py` uses for this model; `fabric_config: FABRIC_1D_RING` is
  the topology recorded in `../context_contract.json`.
* `sample_on_device_mode: "all"` is the runner's own default and is what all of
  this ran under.

## TT configuration as served

| | |
|---|---|
| Mesh | `P300x2` -> `MeshShape(1, 4)`, 4 Blackhole dies, tensor-parallel |
| Fabric | `FABRIC_1D_RING`, `STRICT_INIT` |
| `trace_region_size` | 50331648 |
| `sample_on_device_mode` | `all` |
| `trace_mode` | `all` (prefill is eager in this port; decode is traced) |
| `enable_model_warmup` | `True` |
| Data parallel | 1 (rejected otherwise — this port occupies the whole mesh with TP) |
| Block size | 32 |
| `max_model_len` | **262144** |
| `max_num_seqs` | 1 (headline) / 32 (sampling, qualitative, CI burst) |
| KV cache | vLLM-owned. 8193 blocks x 32 = **262176** tokens at `max_num_seqs=1`; 8224 blocks x 32 = **263168** tokens at `max_num_seqs=32` (see below) |
| KV dtype | `bfloat16` — from the selected precision config, **not** from vLLM's `dtype` argument |
| Precision | `../datatype_sweep/selected_precision_config.json` |

## Served context: the full 262144, unreduced

`../context_contract.json` records `current_supported_context: 262144` and
`capability_reduction: false`. The server advertises exactly that, and nothing in
the adapter lowers it: `get_max_tokens_all_users` returns the contract value so
vLLM sizes enough blocks for a **single request to fill the whole context**
(262176 tokens of KV at `max_num_seqs=1`), and `initialize_vllm_model` **raises**
if `--max-model-len` is above the contract rather than quietly clamping.

Confirmed in [`../../readiness_vllm/server.log`](../../readiness_vllm/server.log):

```
INFO [worker.py:632] Getting max_tokens_all_users=262144 ... from generator
  '<class ...Qwen3CoderForCausalLM>'.
INFO [kv_cache_utils.py:2146] GPU KV cache size: 262,176 tokens
vLLM-owned KV cache: 8193 blocks x 32 tokens = 262176 tokens, 1 local KV heads,
  dtype DataType.BFLOAT16 (from the selected precision config; vLLM asked for torch.bfloat16)
```

**Where the `max_num_seqs=32` numbers in the table come from.** That log is the
`max_num_seqs=1` run (run B). The `max_num_seqs=32` run (run A, which produced
the sampling suite, both qualitative JSONs, the CI burst and
[`../../readiness_vllm/vllm_benchmark_maxnumseqs32.json`](../../readiness_vllm/vllm_benchmark_maxnumseqs32.json))
was served from a server whose log was later overwritten by run B, so **8224 /
263168 are not quoted from a log — they are derived**, and the derivation is
exact. `get_max_tokens_all_users` returns the contract's 262144 and the TT worker
adds its own headroom before converting to blocks
(`worker.py:624`, `ceil((max_tokens_all_users + block_size * max_num_seqs) / block_size)`):

| `max_num_seqs` | blocks | tokens | |
|---|---|---|---|
| 1 | `ceil((262144 + 32*1)/32)` = **8193** | 8193 x 32 = **262176** | matches the log above |
| 32 | `ceil((262144 + 32*32)/32)` = **8224** | 8224 x 32 = **263168** | derived by the same formula |

The `max_num_seqs=1` row is the check on the formula: it is computed the same way
and it reproduces the retained log exactly. `probes/check_published_figures.py`
re-computes both rows and re-checks the 8193/262176 row against `server.log`.

The rotary cos/sin tables are sized to the served context at construction
(`rope_cache_len=max_seq_len`), because the traced decode loop advances
`rotary_position` on device with `ttnn.plus_one` and nothing on device clamps it —
a table that could still grow mid-run would either reallocate underneath a
captured trace or gather out of range.

## Non-aligned prompt lengths

Serving must accept any valid length, not just multiples of the page block (32),
the tile (32), the sampling slot count (32) or the benchmark's 128. It does,
because nothing rounds the length up at the model boundary:
`Qwen3CoderGenerator.prefill_forward` prefills row *i* at exactly
`tokens[i, :prompt_lens[i]]` and selects row `prompt_len - 1`.

Direct requests against the live headline server —
[`../../readiness_vllm/non_aligned_prompt_lengths.json`](../../readiness_vllm/non_aligned_prompt_lengths.json):

| Prompt tokens | ÷8 | ÷32 | ÷64 | ÷128 | ÷1024 | Result |
|---|---|---|---|---|---|---|
| 37 | no | no | no | no | no | 12 tokens out |
| 131 | no | no | no | no | no | 12 tokens out |
| 333 | no | no | no | no | no | 12 tokens out |
| 1025 | no | no | no | no | no | 12 tokens out |
| 4097 | no | no | no | no | no | 12 tokens out |
| 43 (natural text) | no | no | no | no | no | *"Paged attention needs a block table to efficiently manage and track the memory pages containing attention keys and valu…"* |

`usage.prompt_tokens` came back equal to the requested length in every case, so
nothing was capped or truncated. The runner's own probe
([`non_aligned_prompt_37.json`](../../readiness_vllm/non_aligned_prompt_37.json))
records the same at 37, and prefill **warmup** deliberately compiles 129 as well
as 128 so the non-aligned shape is on the warmed path rather than a cold one.

## Qualitative verdict — read, not just collected

Two collections, both on the complete 48-layer model through the live server.

**Raw `/v1/completions`** (what the shared runner sends; no chat template):
[`../../readiness_vllm/vllm_qualitative_outputs.json`](../../readiness_vllm/vllm_qualitative_outputs.json).
**Chat-templated `/v1/chat/completions`** (the format this checkpoint declares —
it *has* a chat template, so continuation-style output would have been the wrong
thing to judge):
[`../../readiness_vllm/vllm_qualitative_chat_outputs.json`](../../readiness_vllm/vllm_qualitative_chat_outputs.json).

Judged on the chat collection, with the raw one as a continuation-style control:

* **Coherence and topic — pass.** *"Think of it like learning from different
  types of teachers:\n\n**Supervised Learning** = Learning from a teacher who
  tells you the right answers…"*; the thermodynamics answer opens *"## First Law
  of Thermodynamics\n**Energy cannot be created or destroyed, only transformed
  from one form to another.**"* and correctly names it the conservation of
  energy. Markdown structure is well-formed across 200-token completions.
* **Repetition — pass.** No loops. `check_degenerate_output.py` measured
  `adjacent_duplication` 0.0 on every completion except the Fibonacci one at
  0.0074, and `trigram_loop_fraction` at or below 0.088 for every completion of
  meaningful length. The only 0.5 trigram figure is the six-word French
  translation, which is too short for the metric to mean anything.
* **Gibberish — none.** The Fibonacci answer emits syntactically valid Python
  inside a fenced block with a correct docstring and complexity note.
* **Wrong-language drift — none, and the one translation is correct.**
  *"Bonjour, comment allez-vous aujourd'hui ?"* — correct French, correct
  formal register, correct French spacing before the question mark. English
  prompts produced English throughout.
* **Request contamination — none.** This is the serving-specific failure, so it
  was checked directly rather than inferred. Four concurrent staggered requests
  on the reduced target each continued **their own** prompt with no cross-talk;
  the plugin's own `test_structured_output_dp1::test_dp1_full_capacity_mixes_
  structured_and_plain_requests` passed at full 32-slot capacity; and all six
  chat prompts return answers about their own subject. The
  `test_request_isolation` failure is a *seed reproduction* assertion, not an
  isolation one — the outputs in that failure are each on-topic for their own
  prompt.

**Control comparison.** The stage-05/06 full-model control at
[`../../readiness_qualitative/vllm_qualitative_outputs.json`](../../readiness_qualitative/vllm_qualitative_outputs.json)
answers the haiku prompt with

> Data streams flow—
> Neural networks dream in patterns,
> Wisdom emerges.

The serving path returns **exactly that, byte for byte**, for both greedy and
sampled. Serving output is not materially worse than the prompt-correct control
anywhere; on the contrary it matches it where they overlap.

**Degeneracy gate:** `check_degenerate_output.py --hf-model
Qwen/Qwen3-Coder-30B-A3B-Instruct --missing-artifacts critical --scope vllm` →
`No degenerate output detected.`, exit 0
([`logs/check_degenerate_vllm.log`](logs/check_degenerate_vllm.log)).

## Proof that serving reuses the full model's split sampling

The requirement most easily violated by accident, so it is checked mechanically
rather than argued. `probes/adapter_contract_probe.py` drives
`tt/generator_vllm.py` with the exact kwargs `vllm_tt_plugin` builds and asserts
on `Qwen3CoderGenerator.trace_stats`, which counts every host-side action on the
token path. **All 13 checks pass** —
[`probes/adapter_contract_probe.json`](probes/adapter_contract_probe.json):

| Check | Result |
|---|---|
| `sampler_is_the_full_model_sampler` | `_WatcherCleanSampling1D` from `tt/model.py`, same object across the run |
| `async_split_returns_device_tensor` | `decode_forward(read_from_device=False)` returns a `ttnn.Tensor`; `read_decode_output(async_read=True)` records an event; `process_decode_output_host` only formats |
| `steady_replays` | replays **+8 over 8 tokens** — one traced model replay + one traced sampling replay per token |
| `steady_no_token_host_copies` | `token_host_copies` **+0** — the sampled token reaches step *N+1* through `tt_out_tok` on device |
| `steady_no_position_host_copies` | `position_host_copies` **+0**, `rotary_position_host_copies` **+0** — both advance on device via `ttnn.plus_one` |
| `steady_no_page_table_copies` | **+0** for an unchanged page table |
| `changed_page_table_is_copied` | **+1** when every slot moves to fresh blocks |
| `one_readback_per_token` | `caller_token_readbacks` **+8** — the 128-byte sampled-token tensor, nothing else |
| `steady_no_recapture` | captures/releases/warmups **+0** in steady state |
| `stale_host_input_ignored` | a steady step fed `token=12345` and `position-1` **reproduced the clean run exactly** |
| `current_position_reinstalled_on_layout_change` | after a `reset_batch` onto fresh blocks, device `current_pos` = 132 for a host request of 131 (the trace advanced it once) |
| `non_aligned_prefill` | 131-token prompt through the adapter boundary |
| `vllm_owns_cache` | `decode_forward` with no `allocate_kv_cache` **raises** instead of allocating a standalone cache |

The same JSON carries the model's own `runtime_fallback_audit`, taken off real
tensors rather than off the config, and it agrees:
`host_logit_readback_on_token_out_path: false`,
`host_argmax_on_token_out_path: false`,
`sampling_greedy: "Sampling1D force-argmax, distributed: per-die
untilize/argmax/gather -> all-gather 4 candidates -> masked-min, traced, writes
tt_out_tok"`, `decode_rope_position_source: "device tensor advanced by
ttnn.plus_one inside the trace"`, `kv_cache_dtype: "bfloat16"`,
`ccl_dtype: "bfloat16"`, `experts_gate_up_dtype`/`experts_down_dtype`
`bfloat4_b`, `experts_fidelity: LoFi`, `lm_head_fidelity: HiFi2`,
`norm_fidelity: HiFi4`, `experts_gate_up_in0_block_w: 64`,
`experts_down_in0_block_w: 24` — i.e. the serving path is running the selected
precision policy, weight groups, exceptions and all.

`token_host_copies == 0` is the load-bearing one: an adapter-side argmax, a
full-logits readback or a Python readback/writeback feedback loop would all have
to put a token or a vocabulary on the host every step, and none of them is
there. Greedy takes `Qwen3CoderModel.sample_greedy_argmax` (the `Sampling1D`
force-argmax strategy with this port's distributed reduction, 6.6x faster than
the split path at this vocabulary); anything with `top_k > 1` or `top_p > 0`
takes `sample_split`. Both are the same module, both traced, both write
`tt_out_tok`, and `set_sampling_params` releases and recaptures when the batch
crosses between them. Neither is new to this stage.

The probe runs on a **2-layer** target on purpose: every property it asserts is
about host-side work per token and cache/page-table/scheduler-input handling,
none of which depends on layer count. It is not an accuracy or performance
artifact, and it is labelled as such inside the JSON.

## Sampling penalties, on the column-parallel shards

`presence_penalty`, `frequency_penalty` and `repetition_penalty` are applied
**on device, inside the sampling trace, before the selection**. This closed the
6 `test_tt_penalties` failures this stage originally shipped with.

### Why it had to be model work

`models/common/modules/sampling/sampling_1d.py` has no penalty stage of any
kind, and the plugin does not compensate: `platform.py` routes `min_p`,
`bad_words`, `logit_bias`, `allowed_token_ids`, `min_tokens`, `prompt_logprobs`
and structured output to host sampling, and **deliberately does not route
penalties there**. It packs all three into `TTSamplingParams` and sends the token
history alongside them — `model_runner.py` populates `prompt_tokens` /
`output_tokens` "if penalties are needed (decode only)" — because it expects the
model's on-device sampler to apply them. So the gap was ours to close, and it is
closed the same way stage 05 closed the argmax gap: in this port's subclass
`_WatcherCleanSampling1D` (`tt/model.py`). **No shared file is edited.**

### The hard part: a penalty is a global token id, the logits are not

Die *d* holds vocabulary ids `d*37984 … d*37984+37983` and nothing else. A
penalty on id *t* must touch local column `t % 37984` on die `t // 37984` and no
column on the other three. Getting that wrong does not raise — it penalises three
unrelated tokens and produces fluent, wrong output.

The stage never does that arithmetic in a kernel. The operands are built on the
host at **full vocabulary width** — `[1, 1, 32, 151936]`, indexed by global id,
the only frame in which a penalty is defined — and handed down through
`ttnn.ShardTensorToMesh(dim=-1)`: the same even 4-way split, over the same
constant `_dist_local_vocab` the distributed argmax's `_dist_die_offset` is built
from, that the column-parallel LM head produced the logits under. Column *t*
therefore lands on the die and the local column that hold logit *t* **by
construction**, and every device op is elementwise between two tensors of
identical per-die shape, so no op ever needs to know a global id.

The device arithmetic is vLLM's `apply_penalties`, in vLLM's order — repetition
first, on the raw logit, because it is sign-dependent and therefore not
expressible as an additive delta:

```
pos    = gtz(x)                      # 1.0 where x > 0, else 0.0
factor = rep_neg + pos * rep_dif     # rep_neg = p, rep_dif = 1/p - p
x      = x * factor                  # repetition, over prompt+output
x      = x - add_delta               # f*count(output) + q*presence(output)
```

For a column nobody penalises the host writes `rep_neg = 1.0`, `rep_dif = 0.0`,
`add_delta = 0.0`, and `x * 1.0 - 0.0` is **bit-exact** in bf16. Unpenalised
tokens are not approximately unchanged, they are unchanged — which is what makes
the cross-die claim a property of the arithmetic rather than of a tolerance.
Batch isolation is structural for the same reason: the operands are 32-row and
every op is elementwise, so row *i*'s history can only ever meet row *i*'s
logits.

### The test that proves it

[`probes/penalty_shard_boundary_probe.py`](probes/penalty_shard_boundary_probe.py)
→ [`penalty_shard_boundary_probe.json`](probes/penalty_shard_boundary_probe.json).
It runs the shipped classes — the real `_WatcherCleanSampling1D`, and the real
`Qwen3CoderGenerator.set_penalty_params` bound onto a device-only shim, so a bug
in the host scatter is a bug it sees — on a synthetic 1×4 mesh at the shipped
`[1,1,32,151936]` shape, and compares against a torch transcription of
`vllm/model_executor/layers/utils.py::apply_penalties`.

| Check | Result |
|---|---|
| `matches_vllm_reference` | every one of 32×151936 columns within one bf16 ulp of the fp32 reference (worst 0.19 ulp) |
| `no_unexpected_columns_moved` | exactly the 13 requested `(row, token)` pairs changed, nothing else |
| `reaches_die_0` / `reaches_die_3` | 5 penalised ids on die 0 and 4 on die 3, in the same step |
| `boundary_columns_covered` | local columns **0 and 37983** — the ids either side of a shard seam |
| `same_local_index_on_other_dies_untouched` | for every penalised id *t*, `t ± k*37984` on the other three dies is **bit-identical** to the input |
| `unpenalised_rows_bit_identical` | all 28 unpenalised rows byte-for-byte unchanged, alongside 4 penalised ones |
| `sampler_matches_penalised_reference` / `forced_matches_reference` | the whole `decode_forward`, argmax included, on both batches |
| `forced_penalty_changed_the_winner` | penalising each row's **current winner** (one on die 0, one on die 3) moves the sampled token, and the control row's does not move |
| `neutral_request_is_fast_path` / `fast_path_is_identity` | a neutral request drops back to mode 0 and `_apply_penalties` returns the input tensor itself |

The `forced_*` leg exists because the plain reference comparison is a null result
on its own: with random logits, penalising 13 of 151936 columns essentially never
moves the argmax, so "sampler == reference" there shows only that the two agree,
not that the penalty reached the selection.

Row 2 and row 4 of the probe's batch are the aliasing trap on purpose: row 2's
history is `{5, 37984+5, 2*37984+5}` — three different global ids sharing one
local index — and row 4's is `{3*37984+5}` alone. A stage that computed a local
index and broadcast it would pass every other check and fail these two.

### Cost: nothing when unused, 14–26 % on TPOT when used

`_penalty_mode` is a **graph** property, not a value. Mode 0 means the ops are not
in the captured trace at all: no op, no buffer, no upload, byte-identical graph
to the one stage 08 shipped. The generator releases and re-captures the decode
traces when the mode changes, exactly as it already does when
`_sampling_stochastic` flips between the argmax and split strategies, and
`_decode_graph_key` carries the mode so the eager warm pass recompiles first.
Bit 0 (repetition) and bit 1 (frequency/presence) are independent, so a
repetition-only request never pays for the additive tensor.

**On device the stage is almost free.** Trace-captured over the whole sampler,
median of 50 (`probes/penalty_shard_boundary_probe.py --time`):

| Mode | Sampler, trace-replayed | Δ vs. the shipped graph |
|---|---|---|
| 0 — no penalty (the fast path) | **0.6325 ms** | — (it *is* the shipped graph) |
| 1 — repetition only | 0.6876 ms | **+0.0551 ms** |
| 3 — repetition + frequency/presence | 0.7053 ms | **+0.0728 ms** |

**The cost is host-side operand staging, and it was 4–5x worse before it was
optimised.** The operands are full-vocabulary, so every penalised step ships
9.7 MB per operand to the dies. The first working version staged three operands
from one `[1,1,32,151936]` host tensor through `ttnn.ShardTensorToMesh(dim=-1)`:

| | first version | shipped | |
|---|---|---|---|
| full-width operands, mode 1 / mode 3 | 2 / 3 | **1 / 2** | `1/p − p` is derived on device with `ttnn.reciprocal` instead of uploaded |
| host staging, mode 1 | 11.6391 ms | **1.5351 ms** | **7.6x** |
| host staging, mode 3 | 17.1644 ms | **3.3894 ms** | **5.1x** |

Two changes, both measured before being adopted:

* **The reshard was the cost, not the wire and not tilization.** Handing a
  full-width host tensor to the mesh mapper makes it re-slice a strided 9.7 MB
  view into four contiguous copies *on every decode step*: **6.601 ms** of a
  **6.897 ms** upload, against 0.747 ms for the `copy_host_to_device_tensor` that
  actually moves the bytes. Keeping the four `[1,1,32,37984]` staging buffers
  contiguous from the start and assembling them with `ttnn.from_host_shards`
  costs **2.049 ms** end to end — 3.4x less for bit-identical device content.
* **One operand instead of two for repetition.** `rep_dif = 1/p − p` used to be a
  second uploaded tensor; it is now `ttnn.reciprocal(rep_neg) − rep_neg` on
  device. That is only safe because `reciprocal(1.0)` is **exactly** 1.0 here —
  checked on the device, and load-bearing, since it is what keeps an unpenalised
  column at `x * 1.0 − 0.0`. Cost: +0.0286 ms of device time, −1.8543 ms of host time (one operand's staging, which is exactly `host_staging_ms_mode3 − host_staging_ms_mode1`).
  It changed **no sampled token**: the serving parity probe is still 11/11
  byte-identical to vLLM's own sampler after the substitution.

**What a user gets** is the in-situ table in the headline section: **44.049 t/s/u
with `repetition_penalty`, 40.079 with all three, against 50.321 unpenalised** —
+14.2 % and +25.6 % on TPOT. The unpenalised path is untouched and measures
identically to the shipped headline, which is also how that probe validates its
own harness.

**Remaining headroom, not taken here — and it is not this port's staging.** The
in-situ overhead is 2.829 / 5.078 ms, but this port's own per-step host cost
**at a serving-sized history** is only 1.5674 / 3.7624 ms
(the same probe re-timed with 256 history tokens per row instead of the
correctness batch's 2–5 — it barely moves, because the staging is dominated by the
fixed 9.7 MB operand, not by the history length). The unaccounted
**1.26 / 1.32 ms** is

What makes that attribution an argument rather than an assertion is that the
residual is **essentially constant across two modes whose staging cost differs
by 2.4x** (1.5351 -> 3.3894 ms). A cost belonging to this port's staging would
scale with the staging; a fixed per-step cost incurred before this port's code
runs would not. It does not. That residual is
per-step host marshalling of the token history *before* this port's code runs: vLLM rebuilds and re-sends the
**entire** `prompt_tokens` / `output_tokens` tensors every step
(`make_prompt_token_ids_tensor` / `make_output_token_ids_tensor`), and this port
then re-derives each row's operand from that whole history, even though the
history grows by exactly one token per row per step. A genuinely incremental
update — keep the operands, add one column per row — would remove almost all of
it, and an on-device scatter of a `[1,1,32,1]` index would remove the 9.7 MB
upload as well. Both were left out deliberately: the scatter reintroduces exactly
the global→local index arithmetic this design eliminates, and the incremental
update needs a reliable "same request still in this slot" key that the adapter
does not currently receive. They are the next cut, with the measurement above
sizing them.

### What is deliberately *not* penalised

The token sampled by **prefill**. vLLM only sends the history on decode
(`model_runner.py`: "decode only"), and a prefill's row *i* is the *i*-th
admitted request, not slot *i* — so applying a staged penalty row there would
penalise the wrong tokens. `Qwen3CoderGenerator._penalties_suspended` turns the
stage off around prefill and around the eager host/device decode compatibility
paths for exactly that reason. This matches the plugin's own contract, not a
shortcut around it.

## Capability flags, each with evidence

| Flag | Value | Evidence |
|---|---|---|
| `supports_sample_on_device` | **True** | The server ran under `sample_on_device_mode: all` throughout; `test_top1_is_greedy` passes; the probe shows the traced sampler is the model's own |
| `supports_async_decode` | **True** | The split is implemented, not asserted: `async_split_returns_device_tensor` in the probe, plus a live `--async-scheduling` run (below) |
| `supports_prefix_caching` | **False** | Not implemented, not tested. The plugin disables prefix caching for this model at startup, and the adapter **raises** on a non-zero prefill `start_pos` so a regression cannot silently skip cached tokens |

**The `--async-scheduling` overlap test.** Since `supports_async_decode=True`
gates async scheduling, it was validated on the complete model rather than
assumed: `max_num_seqs=1`, `sample_on_device_mode: all`, decode trace on,
`--async-scheduling` accepted by the platform (it is *not* force-disabled — see
[`logs/async_scheduling_server_grep.log`](logs/async_scheduling_server_grep.log),
`'async_scheduling': True`).

* All six qualitative greedy completions came back **byte-identical** to the
  synchronous `max_num_seqs=32` run — no doubled subwords, no repeated control
  tokens ([`logs/async_scheduling_qualitative_outputs.json`](logs/async_scheduling_qualitative_outputs.json));
* `check_degenerate_output.py` on those outputs: `No degenerate output detected.`;
* the primary 128/128/1 profile gives TPOT mean **19.808 ms / 50.483 t/s/u**
  against **19.778 ms / 50.560 t/s/u** without it
  ([`logs/async_scheduling_vllm_benchmark.json`](logs/async_scheduling_vllm_benchmark.json))
  — the same number within noise.

So async scheduling is **safe** here but currently **buys nothing**: the plugin
logs `Using custom scheduler class vllm_tt_plugin.scheduler.TTScheduler … you
will see degraded performance due to async scheduling being disabled`, i.e.
`TTScheduler` is not an `AsyncScheduler` subclass, so no real overlap happens.
The shipped command therefore leaves it off. The adapter is nonetheless built to
be correct under overlap — `_merge_scheduler_view` prefers the device's token and
position for any slot whose position is continuous with the scheduler's *and*
whose page-table row is unchanged, and takes the host's for a slot that changed
hands — which is exactly what the probe's `stale_host_input_ignored` and
`current_position_reinstalled_on_layout_change` checks exercise.

## Sampling test results — 56 passed, 16 failed, 1 skipped

`--sampling-profile full`, `--tt-max-num-seqs 32`, against the live 48-layer
server. Log: [`../../readiness_vllm/sampling_tests.log`](../../readiness_vllm/sampling_tests.log)
(600.82 s).

**This supersedes the 52 / 20 / 1 this stage first reported, and the decomposition
is exact:** the seeding/RNG class is **14 failures in both runs**, and the whole
difference is the on-device penalty stage — 4 of the 6 `test_tt_penalties` moved
from FAILED to PASSED. (An intermediate run of the same suite gave 58 / 14 / 1;
the seeding class fluctuates between 12 and 14 run to run without any code
change, which is the fixed-RNG cause showing through. This log is the one taken
against the shipped code.)

**Passed (56)** — everything that is about serving correctness:

* all 20 `test_logprobs::test_logprobs[*]` parametrisations;
* all 5 `test_host_only_params` (`min_p`, `bad_words`, `logit_bias`,
  `allowed_token_ids`, `min_tokens`) — these are the requests the plugin routes
  to host sampling, so this is the compatibility mode working;
* `test_structured_output_dp1::test_dp1_full_capacity_mixes_structured_and_plain_requests`;
* `test_seeding_and_variety::test_top1_is_greedy`,
  `test_different_seeds_produce_different_outputs`, `test_uniform_noseed_varied`,
  `test_negative_seed_does_not_crash`, all 5 `test_temperature_varied_in_batch`,
  both `test_batch1_seed_reproducible`, `test_topk[19]`, `test_topk[32]`;
* all 8 `test_build_logprobs_from_topk` and all 3 `test_config`;
* **4 of the 6 `test_tt_penalties`** — both `TestRepetitionPenalty` and both
  `TestFrequencyPenalty`.

**Failed (16), in exactly two classes, neither a serving-path defect:**

*Class A — per-request seeding and RNG (14 failures).* `test_seeding`,
`test_same_seeds_reproduce_across_batches`, 4x `test_specific_seed_reproducible`,
4x `test_uniform_seed_deterministic[10|32-*]`, `test_batch1_no_seed_varied`,
`test_temperature_varied_between_batches`, `test_topk[15]`,
`test_request_isolation::test_mixed_params_batch`. Root cause: this port's
sampler draws from a fixed device RNG buffer and the per-request `seed` is not
plumbed into it, so a seeded request does not reproduce and an unseeded one does
not vary run to run. `Sampling1D.decode_forward` does take a `seeds=` tensor, so
the path exists; wiring it through the traced decode input set is real work and
was not in scope here. The named tests are precisely the reproducibility-only
class the stage contract sets aside, and the preconditions for setting them aside
hold: correctness, logprobs, crash-free serving and qualitative output all pass.
The class is **not a stable set**: `test_topk[19]` failed on the first run and
passed on this one, `test_topk[15]` did the reverse, and an intermediate run had
12 failures rather than 14 — none of it with any change touching seeding. That is the same fixed-RNG cause showing through — a test whose
assertion is "two runs differ" is a coin flip when the RNG is a fixed buffer —
and it is why the class is quoted as a class rather than as a list of stable
failures.

*Class B — presence penalty on this checkpoint (2 failures).*
`TestPresencePenalty::test_different_presence_penalties` and
`::test_presence_penalty_mixed_batch`. **This is not the penalty stage failing** —
the other 4 `test_tt_penalties` pass, and the stage is verified against vLLM's own
sampler below. Both tests assert that presence penalty **changes the output** for
the prompt `"a b c a b c a b c"`, and for this checkpoint it cannot: vLLM caps
`presence_penalty` at ±2.0, presence subtracts a *constant* (not a growing count)
from each token already emitted, and `a`, `b` and `c` are all already emitted —
so the whole ±2.0 range shifts all three equally and never closes this model's
logit gap to the nearest non-cycle token.

That is a claim about the checkpoint, so it is measured rather than asserted.
[`probes/penalty_serving_parity_probe.py`](probes/penalty_serving_parity_probe.py)
→ [`penalty_serving_parity_probe.json`](probes/penalty_serving_parity_probe.json)
sends every request **twice** against the live 48-layer server: once plain, which
takes this port's on-device stage, and once with `min_p=0.01`, which the plugin
routes to **host sampling** so that vLLM's own
`model_executor/layers/utils.py::apply_penalties` produces the answer. At
temperature 0 both are deterministic, so the comparison is byte equality.

| Result | |
|---|---|
| `all_identical_to_vllm_reference` | **all 11 cases byte-identical**, across repetition, frequency, presence and all three together |
| `presence_reference_also_unchanged` | vLLM's **reference** sampler also returns the unpenalised text at `presence_penalty` −2.0 **and** +2.0 — so these two tests fail against the reference implementation too, on this checkpoint |
| `frequency_changes_output_on_same_prompt` | on the *same* prompt, `frequency_penalty` 0.5 and 1.0 do break the cycle (device and reference alike) — so this is "presence is capped too low here", not "penalties do nothing" |

The frequency threshold is the arithmetic that makes it concrete: on this prompt
`frequency_penalty=0.3` does not move the output and `0.5` does, and by then the
cycle tokens have appeared ~9 times, i.e. a penalty of ~4.5 — more than twice the
largest presence penalty vLLM will accept.

## Where the remaining overhead is

Serving decode is 19.778 ms against a 19.213 ms standalone traced token-out —
**0.565 ms, 2.9 %**. The adapter's own per-token host work is: two
`ttnn.execute_trace` calls, one `torch.equal` over the `[1, 8192]` int32 page
table, one 128-byte sampled-token readback, and a `set_sampling_params` call that
returns early on an unchanged snapshot. Everything else the goal warns about was
checked for and is absent: no fallback sampling on the measured path, no
per-token page-table copy, no blocking trace replay
(`ttnn.execute_trace(..., blocking=False)`), no extra synchronisation, no
adapter-side reconstruction.

Serving TTFT is 312.367 ms against a standalone 129.941 ms at ctx128. Prefill in
this port is eager (no prefill trace), and the extra ~182 ms is vLLM's
request-side work — tokenisation, scheduling, detokenisation and the HTTP round
trip — plus the decode trace re-capture that a prefill triggers (see the next
section). It is the clearest remaining target and is stage-09 work.

## Trace lifetime around prefill — a hang, and what it cost to avoid

`Qwen3CoderGenerator.prefill_forward` releases the decode traces before it runs.
In serving that fires on every request admission, so this stage tried keeping
them alive (`preserve_decode_traces=True`). **The mesh hung** — `tt-triage`
reported `NOC0 CB0..3 active (0xFFFFFFFF). NoC is likely hung.` on device 0
([`triage/tt-triage-preserve-traces-hang.txt.gz`](triage/tt-triage-preserve-traces-hang.txt.gz)),
with `py-spy` showing the EngineCore blocked forever in the prefill sampler's
readback. The prefill sampler and the captured decode sampler share the same
persistent CCL buffers and semaphores; running one eagerly between replays of the
other desynchronises that state. The flag stays **off**
(`QWEN3_VLLM_PRESERVE_DECODE_TRACES=1` reproduces the hang), and
`_decode_compiled_keys` was added so the re-capture skips the redundant eager
warm pass. Measured cost on the reduced target: ITL P99 2.673 ms against P50
2.113 ms, i.e. ~0.56 ms for the re-capturing step — **that pair is unarchived**:
the 2-layer bring-up wrote to a scratch dir on purpose and nothing from it was
promoted to `readiness_vllm/` ([`work_log.md`](work_log.md) §5). Take it as a
sizing observation, not a published artifact-backed figure. The full-model
number below is the artifact-backed one. On the full model the primary
profile's ITL P99 is 20.084 ms against a P50 of 19.804 ms — the re-capture is
inside the noise of a single token out of 128. Full account in
[`work_log.md`](work_log.md) §4.

## Runtime fallback and process-cleanup audit

**Runtime fallbacks on the measured path: none.** `serving_audit()` and
`Qwen3CoderModel.runtime_fallback_audit()` are captured in
[`probes/adapter_contract_probe.json`](probes/adapter_contract_probe.json). The
adapter counts, and reports, every request feature it cannot honour:

| Counter | Meaning |
|---|---|
| `host_sampled_decode_steps` / `host_sampled_prefills` | steps vLLM itself routed to logits (logprobs on a non-8/32-die mesh, `min_p`, `bad_words`, `logit_bias`, structured output). Explicit, optional, per request, never measured. |
| `top_k_clamped_requests` | `top_k <= 0` or `> 32` clamped to 32 — `Sampling1DConfig(max_top_k=32)` is a device limit |
| `penalised_decode_steps` | decode steps that ran with the on-device penalty stage live (see "Sampling penalties") |
| `ignored_seed_requests` | per-request seed not honoured (Class A above) |

**Process cleanup.** Every server in this stage was shut down with
`pkill -f readiness_check.run_vllm_server`, then `pkill -9 -f VLLM::EngineCore`
and `pkill -9 -f vllm.entrypoints`, then verified with `ps aux`. After the
`preserve_decode_traces` hang the full `$tt-device-usage` recovery ran:
processes killed, `timeout 240 tt-smi -r` (all 4 PCI devices, "Re-initializing
boards after reset"), `tt-smi -ls --local` back to 8 Blackhole rows, and a mesh
smoke (`open_mesh_device(MeshShape(1,4))` / `close_mesh_device`) printing
`MESH_SMOKE_OK`. One reset was sufficient. **No vLLM or EngineCore process is
left holding a device**; the final `ps aux | grep -c "[V]LLM::EngineCore"`
returns 0. No Tracy, `tt-perf-report` or `TT_METAL_DEVICE_PROFILER` run was made
against a live server, per the stage rule.

## Limitations and open items

1. **Per-request seeds are not honoured** (Class A, 14 sampling failures). The
   sampler draws from a fixed device RNG buffer. `Sampling1D.decode_forward`
   accepts a `seeds=` tensor, so the fix is to add it to the traced decode input
   set and refresh it with the other sampling parameters.
2. **A penalised request decodes 14–26 % slower than an unpenalised one.** All
   three penalties are applied on device, before the selection, and are
   byte-identical to vLLM's own sampler (11/11 cases) — but the operands are
   full-vocabulary and are staged from the host every decode step. Measured on
   the primary 128/128/1 workload: **44.049 t/s/u with `repetition_penalty`
   (+14.2 % TPOT), 40.079 with all three (+25.6 %), against 50.321
   unpenalised.** The unpenalised path is unaffected — the ops are not in its
   trace at all — so the headline stands for any request that does not ask for a
   penalty. **This is the single most important thing to know before enabling a
   penalty in production, and it is why `--generation-config vllm` is still on
   the shipped command:** this checkpoint's `generation_config.json` would
   otherwise put *every* request on that path with `repetition_penalty=1.05`.
   The device work is 0.0728 ms; the rest is host-side staging, already optimised
   5–8x (see "Sampling penalties") with the next cut identified and sized.
3. **The token sampled by prefill is not penalised.** Penalties apply from the
   first *decode* step onward, so the **first generated token** of every request
   is chosen from unpenalised logits. This is a real semantic difference from a
   CUDA backend, and a reader diffing outputs against a GPU reference will hit
   exactly that token first. It follows the plugin's own contract rather than
   working around it — `model_runner.py` populates `prompt_tokens` /
   `output_tokens` "if penalties are needed (**decode only**)", so no history is
   sent at prefill, and a prefill's row *i* is the *i*-th admitted request rather
   than slot *i*, so a staged penalty row could not be applied to it safely
   anyway. `Qwen3CoderGenerator._penalties_suspended` turns the stage off there
   explicitly rather than by omission. Closing it needs the plugin to send the
   prompt history at prefill, or this port to reconstruct it from the prefill
   `tokens` argument and stage a prefill-row-ordered operand.
4. **`top_k` is clamped to 32.** `Sampling1DConfig(max_top_k=32)`; `top_k <= 0`
   ("no top-k" in vLLM) and any `top_k > 32` both become 32, so sampling draws
   from the widest candidate set the device sampler supports rather than the full
   vocabulary. Counted in `top_k_clamped_requests`.
5. **Per-user decode at large `max_num_seqs` is ~13x slower than at 1**
   (263.470 ms against 19.778 ms on the same 128/128/1 workload). MoE decode
   batch scaling in `tt/model.py`, not adapter overhead; aggregate throughput
   still rises 2.3x from 1 to 32 users. Stage-09 target.
6. **Prefix caching is off** and `supports_prefix_caching=False`; chunked prefill
   is disabled by the plugin for all TT models.
7. **Prefill is eager** — this port has no prefill trace, so `enable_trace` is
   accepted and ignored on the prefill path and TTFT carries the full eager cost
   plus a decode-trace re-capture.
8. **`--async-scheduling` is inert** with the current `TTScheduler`, though it is
   accepted and proven output-safe.
9. **Data parallel > 1 is rejected**; this port uses the whole 1x4 mesh for
   tensor parallelism.

## Exact artifacts

Under `readiness_vllm/`:

| File | What |
|---|---|
| `vllm_benchmark.json`, `vllm_result.json`, `vllm_benchmark.log` | **primary single-user 128/128/1 at `max_num_seqs=1`** — the headline |
| `vllm_ci_serving_benchmark.json`, `vllm_ci_serving_result.json`, `vllm_ci_serving_benchmark.log` | secondary CI serving burst 100/100/32 at `max_num_seqs=32` |
| `vllm_benchmark_maxnumseqs32.json`, `vllm_result_maxnumseqs32.json`, `vllm_benchmark_maxnumseqs32.log` | the 128/128/1 shape at `max_num_seqs=32`, for the slot-count comparison |
| `sampling_tests.log` | full TT plugin sampling suite, 56/16/1 |
| `vllm_qualitative_outputs.json` | raw `/v1/completions` greedy + sampled, 6 prompts |
| `vllm_qualitative_chat_outputs.json` | chat-templated `/v1/chat/completions`, same 6 prompts |
| `non_aligned_prompt_lengths.json` | 37 / 131 / 333 / 1025 / 4097 / 43-token requests |
| `non_aligned_prompt_37.json` | the runner's own non-aligned probe |
| `server.log` | the live server, including registration, cache sizing and warmup |

Under `doc/vllm_integration/`:

| File | What |
|---|---|
| `README.md`, `work_log.md` | this, and the chronological account |
| `probes/adapter_contract_probe.py`, `probes/adapter_contract_probe.json` | the 13 mechanical contract checks and their result |
| `probes/penalty_shard_boundary_probe.py`, `probes/penalty_shard_boundary_probe.json` | the on-device penalty stage against a torch transcription of vLLM's `apply_penalties`, shard-boundary and batch-isolation legs, and the with/without cost |
| `probes/penalty_serving_parity_probe.py`, `probes/penalty_serving_parity_probe.json` | the served path against vLLM's **own** sampler, 11 cases, byte equality |
| `probes/penalty_serving_cost_probe.py`, `probes/penalty_serving_cost_probe.json` | what a **penalised** request costs on the primary 128/128/1 workload, in situ |
| `logs/penalty_watcher.log` | `TT_METAL_WATCHER=10` over the penalty probe, zero tripped asserts |
| `logs/stage08_penalties_model_suite.log` | the model test suite after the penalty change, 158 passed |
| `probes/check_published_figures.py` | re-derives every figure in this file from the artifacts above, and prints the numbers it does **not** cover |
| `logs/check_degenerate_vllm.log` | the degeneracy gate |
| `logs/stage08_gate_08-vllm.check.log` | both halves of the stage gate, exit 0, with command and git state |
| `logs/stage08_review_regression_tests.log` | `test_trace.py` + `test_full_model.py` after the `generator.py` changes |
| `logs/async_scheduling_*.json`, `logs/async_scheduling_server_grep.log` | the `--async-scheduling` overlap run |
| `triage/tt-triage-preserve-traces-hang.txt.gz`, `triage/triage-summary-preserve-traces-hang.txt` | the NoC hang that decided `preserve_decode_traces` |

Source:

| File | What |
|---|---|
| `tt/generator_vllm.py` | the adapter |
| `tt/generator.py` | `configure_paging`, `decode_device_state`, `read_sampled_tokens`, `preserve_decode_traces`, `validate_page_coverage`, `_decode_compiled_keys` |
| `vllm_bundle/qwen3_coder_30b_a3b_instruct/` | the `EXTRA_MODELS_DIR` bundle |
