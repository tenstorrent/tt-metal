# Model request behavior

These integration tests execute real, full-layer models through tt-metal's
factories and generators, without vLLM or HTTP. A scripted driver controls
admission, physical slots, histories, completion, and KV-page reuse. No dummy
weights, substitute stochastic sampler, output tolerance, or automatic retry
stands in for a model run.

## Models and CI hardware

The new jobs use the same enabled SKUs and per-SKU tiers as each model's existing
e2e entry in `tests/pipeline_reorg/models_e2e_tests.yaml`. A CPU regression checks
that mapping and the matching workflow selectors. Request behavior runs in the
existing tier 1 or tier 2 workflows, with its own allowance in each tier's time
budget. The new jobs remain `release_ready: false`.
The 45/60/90-minute job limits are provisioning estimates; Gemma 26B on Wormhole
uses a conservative 120 minutes based on local eager/traced runtime. These are
provisional until representative runs on each physical CI SKU establish the
policy's measured runtime plus approximately 15% allowance.

| Backend option | Checkpoint | CI SKUs | Physical request slots |
| --- | --- | --- | --- |
| `galaxy-llama70b` | `meta-llama/Llama-3.3-70B-Instruct` | `wh_galaxy_perf` | 32 |
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct` | `wh_n150`, `bh_p150`, `wh_llmbox_perf`, `bh_quietbox_2` | 32 |
| `gemma-4-26b-a4b` | `google/gemma-4-26B-A4B-it` | `wh_llmbox_perf`, `bh_quietbox_2` | 32 |
| `qwen3.6-27b` | `Qwen/Qwen3.6-27B` | `bh_quietbox_2` | 32 |
| `qwen3.6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | `bh_quietbox_2` | 32 |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | `wh_galaxy_perf`, `bh_quietbox_2`, `bh_galaxy` | 128 on WH; 1 on BH |

Gemma's disabled Blackhole Galaxy entry remains disabled. Qwen remains on
Blackhole. GPT-OSS follows its demo's batch restriction: throughput experts and
multiple users on Wormhole Galaxy; one user on Blackhole. Multi-request
isolation cases explicitly skip when the configured capacity is insufficient.
Penalty and top-p sensitivity still exercise all eight prompts/draws in waves
on single-user configurations.

### CI runtime accounting

`verify_time_budget.py` sums declared timeouts across **all** pipeline YAMLs by
`(team, budget_type + tier, SKU)`. Each behavior job includes both execution modes,
model construction and warmup. The additions below reserve the job limits, not
measured accelerator usage; tier 3 receives no additional jobs or budget.

| Models budget bucket | SKU | Prior minutes | New job contributions (minutes) | Total minutes |
| --- | --- | ---: | --- | ---: |
| `e2e_tier1` | `wh_n150` | 28 | Llama 8B: 45 | 73 |
| `e2e_tier1` | `bh_p150` | 27 | Llama 8B: 45 | 72 |
| `e2e_tier1` | `wh_galaxy_perf` | 215 | Llama 70B: 60; GPT-OSS: 90 | 365 |
| `e2e_tier1` | `bh_quietbox_2` | 434 | Gemma 26B: 90; Qwen 27B: 90; Qwen 35B: 90; GPT-OSS: 90 | 794 |
| `e2e_tier1` | `bh_galaxy` | 78 | GPT-OSS: 90 | 168 |
| `e2e_tier2` | `wh_llmbox_perf` | 270 | Llama 8B: 45; Gemma 26B: 120 | 435 |
| `e2e_tier2` | `bh_quietbox_2` | 186 | Llama 8B: 45 | 231 |

Available local measurements (2026-09-22) are 283.5 seconds for Llama 8B's
full eager/traced sweep on a one-chip Wormhole Galaxy submesh, and approximately
43.3 minutes eager plus 47.3 minutes traced for Gemma 26B on an eight-chip Galaxy
submesh. These establish functional coverage and a provisioning starting point;
they are not timings from physical N150 or T3K CI runners. GPT-OSS's Galaxy runs
required a partial rerun after an intermittent failure and do not establish a
clean full-sweep timing. No local Blackhole/Qwen timings are available.

Per-SKU measurements are still pending in
[the PR validation run](https://github.com/tenstorrent/tt-metal/actions/runs/35842732749).
After successful representative runs, record the model/SKU, cache conditions,
both-mode wall time, run link and margin here; size each timeout to approximately
`ceil(measured_minutes * 1.15)` and adjust its pooled bucket by the same delta.
Do not use failed/partial runs or submesh timings to mark this requirement met.

## Running

Use a built checkout and its Python environment. Set the normal HF and TT weight
cache variables; `HF_MODEL` can identify a local checkpoint. The five new backends
default to the checkpoint IDs above. The original Galaxy backend requires an
explicit `HF_MODEL`.

```bash
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/path/to/writable/tt-weight-cache
mkdir -p generated/test_reports/model_behavior
python -m pytest --confcutdir=tests/model_behavior tests/model_behavior \
  --model-behavior-backend=llama3.1-8b \
  --model-behavior-execution=both \
  --basetemp=generated/test_reports/model_behavior/requests \
  --junitxml=generated/test_reports/model_behavior/results.xml
```

The SKU is inferred only when the architecture and total device count match a
supported CI configuration. An explicit `--model-behavior-sku=wh_n150` or
`--model-behavior-sku=wh_llmbox_perf` allows a supported smaller configuration
on a Wormhole Galaxy for local testing. This does not add a Galaxy CI leg for
that model, and a submesh run does not certify a different physical machine.
Wrong architectures and unsupported model/SKU combinations fail setup.

Gemma uses the production factory's linear `FABRIC_1D` configuration. For a
local eight-chip Gemma run on Wormhole Galaxy, the adapter opens the 32-chip
parent mesh so fabric neighbors are initialized, then gives the model a `1x8`
submesh. Reserve the whole Galaxy for that run; model weights and requests still
use only the eight-chip submesh. Native eight-chip CI machines open directly.

For Gemma, install `models/demos/gemma4/requirements.txt` first. For GPT-OSS,
`--model-behavior-skip-model-load` selects its existing factory's prebuilt TT
weight-cache path, matching CI's read-only NAS setup. Use it only with a complete
cache. Trace reservations come from `models/model_trace_region_sizes.yaml`.
`TT_METAL_CACHE` controls the separate writable kernel cache.

Use `eager` or `traced` to run one mode. The model and mesh are recreated between
modes. Without an explicit backend, hardware scenarios skip. CPU checks need
pytest, pytest-timeout, torch, and PyYAML, but no TT runtime or weights:

```bash
python -m pytest --confcutdir=tests/model_behavior tests/model_behavior/unit
```

## Coverage and implementation boundaries

- Mixed requests replay exactly in sparse and full batches, with different
  top-k, top-p, temperatures, seeds, and penalties.
- Stochastic requests must vary for identical prompts. Explicit seeds replay;
  changed seeds affect output; unseeded admissions vary. Seeds 0 and -1 are used.
- Identical-seed equality is checked only when the actual sampler declares that
  contract. Other models retain their production duplicate-seed salting, and
  that equality case explicitly skips. Distinct-seed permutation remains tested.
- Top-k=1 matches greedy generation. Restrictive top-p changes output.
- Repetition, presence, and frequency penalties each affect at least one of
  eight prompts, against an exactly replayed control. Neighbor cases check that
  another request's penalties cannot change the unpenalized target.
- Mid-generation admission must preserve the survivor. Slot reuse must preserve
  a target after an unrelated seeded, penalized request owned its pages.
- Sampled-token logprobs must match CPU `log_softmax` of the exact device logits
  consumed by the sampler, including padded vocabulary, within 0.05. Tests cover
  greedy/stochastic sampling, full batches, and mixed reporting flags. Seeded
  tokens must remain unchanged when reporting is disabled. Both sampled-value
  outputs and GPT-OSS's token-indexed top-k result format are handled.
- Supported prefix-resume paths must recall an exact passphrase in short, long
  (>6000-token), and shared-cache scenarios. A different owner overwrites the
  baseline KV data before the resumed run. The observer verifies actual nonzero
  execution offsets, rejecting silent fallback to whole-prompt prefill.

| Family | First prefill token | Decode sampling/logprobs | External prefix resumption |
| --- | --- | --- | --- |
| Llama 70B, Llama 8B, Gemma 4 | TT sampler + logprob oracle | TT sampler + logprob oracle | Tested |
| Qwen 3.6 | Demo-style host argmax; **no stochastic or logprob coverage** | TT sampling; logprobs currently unsupported on its 4-chip CI SKU | Explicitly skipped: persistent GDN state cannot resume this API's prefix |
| GPT-OSS | TT sampler + logprob oracle | TT sampler + logprob oracle | Explicitly skipped: model currently accepts chunk arguments without forwarding them to attention |

The production logprob calculator currently supports only 8- or 32-device
meshes, with at least two vocabulary shards. Logprob cases explicitly skip on
1- and 4-device configurations (including Qwen's current CI SKU); they do not
accept the generator's disabled-logprob placeholder as a valid value. The
adapter reads this capability from the actual calculator. The table's logprob
coverage therefore applies only on a supported mesh.

The new shared-generator adapters submit prefill requests sequentially, with
normal physical-slot assignment. This keeps each sampler invocation observable
for the oracle, including traced model prefill; it does **not** test true batched
prefill. Decode uses the full configured slot layout. Qwen uses the production
model-owned traced prefill chunks in both execution modes; `eager`/`traced`
selects its decode mode. Models may also select eager prefill for lengths their
normal trace policy does not support.
Gemma uses eager prefill for nonzero cached-prefix continuations and input
sequence lengths above 4096, including in traced mode; decode remains traced.

Every slot owns stable, disjoint KV pages within its cache domain. Block zero is
reserved for padding. Four boundary slots (one for batch size one) support 8192
tokens; other slots support 1024, or 2048 for Qwen's masked prefill buckets.
Gemma 26B on Wormhole reserves 1 GiB for its full prefill trace sweep plus
decode/sampling traces; the four prefill traces alone total about 658 MB.
GPT-OSS's four Wormhole row groups have separate device caches: physical block
IDs can repeat across those domains, never between requests in the same domain.
The original Galaxy adapter retains its 8x4 layout and allocation scheme.

GPT-OSS prompts use the tokenizer's `continue_final_message` option with an empty
assistant final response. Its normal generation prompt ends before the channel
header, making the first sampled token a common formatting delimiter. Completing
that header in the prompt lets the unchanged prefill diversity check observe
answer content. This suite therefore tests GPT-OSS final-response continuation,
not its selection or generation of an analysis channel.

Token budgets drive completion, even after EOS. Blocking readback keeps scheduler
and asynchronous execution outside this suite. Layout changes pass complete
surviving histories. Paged adapters require the generator's public
`release_request(slot)` hook at completion, as serving does; tests do not reset private KV or
sampler state between requests. Traced decode must create a model trace. Unseeded decode also requires
a sampler trace unless the model explicitly disables that path in production
(e.g. Gemma's trace policy); stale explicit request seeds are still rejected.

## Reports and remaining coverage

Each scenario writes `requests.json` and records its path as a JUnit property.
Reports contain prompts, sampling parameters, slot/page mappings, token IDs,
positions, supported phases, and requested/observed prefix offsets. Logprob
records identify both the sampled token and its CPU reference. An equality
failure identifies the first divergent token and whether it was prefill or
decode. CI's normal test-report artifact includes these files.

`models/common/tests/test_tt_sampling.py` retains synthetic-logit sampling and
penalty checks; `models/common/tests/test_sampling.py` retains the seed and
logprob unit tests. These scenarios cover the real model/generator boundary.
CPU regressions inject ignored parameters, slot-dependent seeds, neighbor leaks,
misaligned logprobs, missing decode observations, and ignored prefix offsets.

Serving-only behavior remains in `vllm-tt-plugin/tests/tt`: HTTP/API formatting,
all-vocabulary/top-logprob response formatting, host logit processors, structured
outputs, scheduler timing, and chunk-budget decisions. Passing these tests does
not claim coverage of those contracts, batched prefill, or unavailable SKUs.
