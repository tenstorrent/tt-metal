# Running the tt-shield release flow locally (QB2 p300x2)

Date: 2026-08-18 UTC. Branch: tt-metal `mvasiljevic/fast-models-fast/gemma4-31b`,
tt-inference-server `mvasiljevic/fast-models-fast/gemma4-31b-minimal`.

Why this document: the `release` workflow is what the nightly cron actually
dispatches (`on-nightly.yml` sets `workflow: "release"`), and running it locally
found four defects that no unit-level test in this repo caught. Two of them would
have shipped silently.

## Command

```bash
unset LD_LIBRARY_PATH TT_METAL_RUNTIME_ROOT EXTRA_MODELS_DIR
export TT_METAL_HOME=/home/mvasiljevic/tt-metal
export PYTHONPATH=/home/mvasiljevic/vllm    # vLLM is a source checkout locally
cd /home/mvasiljevic/tt-inference-server
python run.py --model gemma-4-31B --workflow release --tt-device p300x2 \
  --impl gemma4-31b-autoport --local-server --ci-mode \
  --tt-metal-home /home/mvasiljevic/tt-metal \
  --vllm-dir /home/mvasiljevic/vllm \
  --tt-metal-python-venv-dir /home/mvasiljevic/tt-metal/python_env
```

`release` = evals + benchmarks. Budget ~70 min: ~16 min server bring-up, ~50 min
`--ci-mode` evals, ~20 min for the 17-point sweep.

### Host prerequisites

- **Reset the devices between serving runs.** A crashed EngineCore leaves the
  fabric dirty and the next `ttnn.open_mesh_device` fails with
  `Timed out while waiting for active ethernet core 29-25`. `tt-smi -r` then a
  mesh open/close proof.
- **No `vllm` directory inside `TT_METAL_HOME`.** `run_local_server.py` puts
  `TT_METAL_HOME` first on `PYTHONPATH` and expects vLLM pip-installed. A
  `tt-metal/vllm` symlink pointing at the vLLM *repo root* (no `__init__.py`)
  makes `vllm` resolve as an empty namespace package:
  `ImportError: cannot import name 'ModelRegistry' from 'vllm' (unknown location)`.
- **`tt-inference-server/tt-metal` symlink.** The specs reach into tt-metal with
  cwd-relative `../../tt-metal/...` paths (the convention the existing
  `TT_MESH_GRAPH_DESC_PATH` entries use). In the container TTI's root *is*
  `/home/container_app_user`, alongside `tt-metal`; the symlink mirrors that
  locally. Keep it out of git.

## Zero code changes to tt-inference-server or vLLM

The model lives in tt-metal. The TTI footprint is registration and reference
**data** only -- no server, launcher, or workflow-engine code:

| File | Kind |
| --- | --- |
| `workflows/model_spec.py` | `ImplSpec` registration (+12 lines) |
| `workflows/model_specs/{dev,prod}/llm.yaml` | device spec |
| `reference_config/evals/eval_config.py` | eval tasks |
| `reference_config/benchmarking/.../model_performance_reference.json` | perf targets |
| `.github/workflows/models-ci-config.json` | CI registration |
| `tests/test_gemma4_31b_autoport_spec.py` | spec guard test |

vLLM needs nothing: the plugin's `_register_models_from_extra_dir()` runs before
its built-in architecture map, so a bundle under `EXTRA_MODELS_DIR` selects the
autoport. Verified in the server log:

```text
setting env var: EXTRA_MODELS_DIR=../../tt-metal/models/autoports/vllm_bundles
Registered TT model TTGemma4ForConditionalGeneration ->
  models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM
  (from EXTRA_MODELS_DIR/gemma4_31b_a...)
```

Two values that previously needed entrypoint code are now spec data:
`EXTRA_MODELS_DIR` and `chat-template`, both cwd-relative.

## Results

### Evals (`--ci-mode`)

| Task | Samples | Score |
| --- | ---: | ---: |
| `mmlu_generative` 5-shot | 2127 (15% subset) | **78.37%** +/- 0.85 |
| `gpqa_diamond_generative_n_shot` 5-shot | 40 (20% subset) | **37.50%** flexible-extract |

MMLU by category: social sciences 86.1, other 81.9, humanities 75.0, STEM 72.4.
78.4% is decisive end-to-end evidence -- a numerically broken port lands near the
25% random floor. Two independent runs produced **bit-identical** scores
(0.7837 / 0.3750 twice), so the serving path is deterministic.

GPQA `strict-match` is 0.0 and `flexible-extract` 37.5: a base checkpoint does not
emit the literal `Answer: (C)` format, which is why the config scores
`flexible-extract`.

**No published reference exists for either task.** `google/gemma-4-31B` reports
`model-index: null` and Google publishes instruction-tuned numbers only, so both
`published_score` and `gpu_reference_score` are None and
`compute_accuracy_check()` returns `NA`. Same-task comparators that do exist:

| Task | Comparator | Score |
| --- | --- | ---: |
| `mmlu_generative` | openai/gpt-oss-20b (published) | 80.4 |
| `mmlu_generative` | openai/gpt-oss-120b (published) | 85.9 |
| `gpqa_diamond_generative_n_shot` | Falcon3-7B-Instruct (H100 ref) | 43.43 |
| `gpqa_diamond_generative_n_shot` | Qwen2.5-72B-Instruct (H100 ref) | 42.93 |
| `gpqa_diamond_generative_n_shot` | Qwen2.5-7B-Instruct (H100 ref) | 33.80 |

### What Google actually publishes

Checked the model cards directly (2026-08-18). Both
`huggingface.co/google/gemma-4-31B` and `...-31B-it` carry the same table:

| Benchmark | Gemma 4 31B |
| --- | ---: |
| GPQA Diamond | 84.3% |
| MMLU **Pro** | 85.2% |
| AIME 2026 (no tools) | 89.2% |

Both cards state: *"Evaluation results marked in the table are for
instruction-tuned models."* So there is still **no published base-checkpoint
figure**, which is why this entry keeps `published_score=None`. Two further
caveats:

- The published number is **MMLU Pro**, not plain MMLU, so it is not comparable
  to `mmlu_generative` at all.
- 84.3 **is** Gemma's own GPQA Diamond value. tt-inference-server's
  `gemma-4-31B-it` entry has the right number with a misattributed
  `published_score_ref` (it cites `huggingface.co/Qwen/Qwen3.6-27B`). Do not
  compare our score against it directly regardless: it is the instruct model on
  `r1_gpqa_diamond` with thinking enabled, and TTI's own H100 measurement for that
  configuration is 83.33.

Placing our result in that chain, each step a plausible drop:

| Configuration | GPQA |
| --- | ---: |
| Published, `-it`, thinking/CoT | 84.3 |
| TTI H100 measured, `-it`, `r1_gpqa_diamond` + thinking | 83.33 |
| `-it` on the n-shot variant (TTI note: costs ~30 points) | ~53 |
| **This port: base, n-shot, 40 samples** | **37.5** |

Note some secondary aggregators (e.g. datalearner.com) describe 84.3/85.2 as
*base* scores; that contradicts the primary card text, which is what this document
follows.

### Would these scores pass a graded CI check?

Searched for published references (2026-08-18). Results:

| Benchmark | Published reference | Applies to |
| --- | --- | --- |
| GPQA Diamond | **84.3%**, Google's own via the HF blog; independently confirmed by benchlm.ai citing that source | **instruction-tuned only** |
| IFEval | **none exists.** Absent from Google's card; benchlm.ai lists Instruction Following as "Not measured, 0 benchmarks" |  |

A web-search summary asserted "IFEval 95.0"; that number appears in none of the
sources it cited and should not be used.

With no `gpu_reference_score`, `compute_accuracy_check` falls back to
`accuracy >= published_score * (1 - tolerance)`, tolerance 0.05. So if the
instruction-tuned references were populated:

| | GPQA |
| --- | ---: |
| Pass bar (84.3 * 0.95) | **80.09** |
| This port, base, n-shot | **37.50** |
| The *instruct* model on the same n-shot variant | ~53 |

It would fail by a wide margin -- and so would the instruct model, because 84.3 is
only valid paired with `r1_gpqa_diamond` plus thinking mode, which this base
checkpoint cannot run (it needs non-greedy sampling and a thought channel the
tokenizer does not define). There is therefore no honest way to grade this
model's GPQA against the published figure.

For IFEval there is nothing to grade against at all, so the check returns `NA`.
Stage 11's recorded 25.18 cannot be compared to anything; against a plausible
instruct-level bar of ~90 it would be ~28%. That is the quantitative form of the
pipeline finding: `$tti-release` mandates `meta_ifeval` as a hard gate, and for a
base checkpoint no reference is obtainable, because nobody publishes
instruction-following scores for a model that was never instruction-tuned.

Practically: this entry keeps both references `None`, so accuracy reports `NA`,
and at `EXPERIMENTAL` `evals_enforced` is `False` so it would not gate even if it
failed. The lane passes because accuracy is both ungraded and unenforced --
promotion to `FUNCTIONAL` would require a same-task reference that does not exist.

### Benchmarks: 17/17 sweep points, zero failed requests

Concurrency points (the meaningful ones):

| isl | conc | Median TTFT | Median TPOT | Output tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 32 | 101 ms | 33.2 ms | 389.8 |
| 1024 | 32 | 312 ms | 34.4 ms | 223.4 |
| 2048 | 32 | 610 ms | 34.5 ms | 148.7 |
| 4096 | 26 | 1.14 s | 34.5 ms | 86.0 |
| 8192 | 13 | 7.25 s | 35.1 ms | 42.0 |
| 16384 | 6 | 11.6 s | 35.3 ms | 20.1 |
| 32768 | 3 | 32.0 s | 36.1 ms | 8.7 |
| 65536 | 1 | 34.7 s | 74.1 ms | 2.1 |

**The `max_concurrency=1` points are warmup-dominated, not steady state.** They
run n=4/2/1 requests, so the first request's trace capture (~2.4 s) sits inside
the median: isl=128 conc=1 reads TTFT 2470 ms where isl=128 conc=32 reads 101 ms
for the same prompt length. TPOT is also *better* at concurrency (33-36 ms) than
at conc=1 (62-84 ms), which is backwards for batching and the same artifact. The
33-36 ms figure matches the measured per-layer decode floor (27.98 ms + overhead).

Only sweep point 1 has perf targets; the other 16 log `NA (ungraded)`.

## What a green release asserts

`enforce_acceptance_criteria` only fails on tiers in
`ModelStatusTypes.required_target_tiers`. This model is `EXPERIMENTAL`, whose list
is empty, and `evals_enforced` derives from that same list -- so **neither perf
targets nor eval accuracy gate the result**. Separately, all 381 entries in
`model_performance_reference.json` use the `theoretical` tier, which appears in no
status's required list, so perf gates nothing for any model on this platform. A
green `release` means "every workflow executed and completed".
