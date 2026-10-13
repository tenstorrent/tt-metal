# pplx-decider-v1-27b on one Blackhole p150a

[`perplexity-ai/pplx-decider-v1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b) (revision
`b01a5cbaca5391f73bd55103d4f27e8982cd5e60`) is a 255-way decision classifier. It runs a Qwen3.5 hybrid
backbone (48 Gated DeltaNet + 16 gated full-attention layers) over a prompt, reads the last token's
hidden state through a 5120x255 `readout` and returns a softmax over the options. It does prefill only:
no decode, no KV cache across requests. This directory runs text and image requests on one p150a, batch 1.

| what | result |
|---|---|
| Decisions vs HF bf16 | 25/25 text rows, 8/8 image rows |
| Accuracy (200 examples each) | WinoGrande 87.0 %, FinancialPhraseBank 84.0 %, Belebele 96.5 % (card: 83.3 / 84.18 / 94.0 %) |
| Request latency, burst (ms) | 115 / 315 / 564 / 1227 / 2882 at buckets 128 / 1024 / 2048 / 4096 / 8192 |
| Request latency, sustained (ms) | 114 / 414 / 821 / 1679 / 3510 |
| Image request, 1024 bucket | 336-354 ms burst |
| Mixed throughput | 3.10 decisions/s (buckets 128 + 1024) |

All numbers are measured. The full report is in [`doc/benchmark/REPORT.md`](doc/benchmark/REPORT.md).

## Setup

You need a source build of tt-metal with its `python_env`, one Blackhole p150a, about 50 GB of disk
for the HF snapshot, 27 GB for the device weight cache, 2 GB for the goldens, and about 20 GB of free
host RAM. Run every command in this README from the tt-metal root.

```bash
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_HOME=/path/to/hf_cache                       # where the snapshot lives (HF default: ~/.cache/huggingface)
hf download perplexity-ai/pplx-decider-v1-27b --revision b01a5cbaca5391f73bd55103d4f27e8982cd5e60
export PPLX_DECIDER_ARTIFACTS=/path/to/artifacts       # optional; default <tt-metal>/generated/pplx_decider
```

The code finds the snapshot in the HF cache at that revision (`local_files_only=True`); set
`HF_MODEL=<dir or repo id>` (or pass `model_path`) to use another checkpoint. Goldens, logs and test
results go under `$PPLX_DECIDER_ARTIFACTS`; the device weight cache goes next to the weights (see
[Where the weight cache goes](#where-the-weight-cache-goes)). To print every resolved location, with the
model path, the weight cache and where each came from (CPU only):

```bash
python -m models.demos.pplx_decider_v1_27b.tt.paths
```

The first model load converts the weights from the snapshot into the weight cache (27 GB, about 160 s).
Later loads read the cache: about 37 s with a cold page cache, about 5 s with a warm one.

## How to run the model

The demo runs the examples of the snapshot's `inference.py` and prints JSON in the app's
`Decider.predict` format. It does not need the goldens.

```bash
python models/demos/pplx_decider_v1_27b/demo/demo.py                     # text: urgency (yes/no) + routing (choice)
python models/demos/pplx_decider_v1_27b/demo/demo.py --image photo.png   # image: "What is the dominant color?"
python models/demos/pplx_decider_v1_27b/demo/demo.py --state "Refund my order" \
  --question '{"type": "noul", "instructions": "Is this a refund request?"}'
python models/demos/pplx_decider_v1_27b/demo/demo.py --golden-row v02_count_circles   # image golden row vs HF (needs the image golden)
python models/demos/pplx_decider_v1_27b/demo/demo.py --model-path /path/to/checkpoint  # a local dir or an HF repo id
```

In Python, with the same call shape as `models/demos/gemma4`:

```python
import ttnn
from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider

device = ttnn.open_device(device_id=0, l1_small_size=24576)
decider = TTDecider.from_pretrained(device, model_path=None, max_seq_len=8192, vision=True)
print(decider.predict("My Stripe integration keeps failing.", {"type": "noul", "instructions": "Is it urgent?"}))
question = {"type": "choice", "instructions": "What is the dominant color?",
            "criteria": {"red": "Red", "green": "Green", "blue": "Blue", "other": "Another color"}}
print(decider.predict("Look at the supplied image.", question, images=["photo.png"]))
ttnn.close_device(device)
```

- `model_path`: a local checkpoint directory or an HF repo id. `None` means `$HF_MODEL`, else
  `$PPLX_DECIDER_SNAPSHOT` (legacy), else the HF cache at the pinned revision. A repo id is looked up in the
  HF cache only; if it is missing, the error gives the `hf download` command.
- `max_seq_len`: the largest prefill bucket, one of 128, 1024, 2048, 4096 or 8192. Longer prompts are
  rejected, never truncated. Batch is always 1.
- `vision=False` skips the vision tower: text only, 0.96 GiB less DRAM.

`images` takes what the app's `open_image` takes: file paths, `data:image/...` URLs or PIL images. Each
image adds 64 to 256 prompt tokens, which count toward the `max_seq_len` limit.

### Where the weight cache goes

The device weight cache (27 GB) is located from the model path, as in `models/demos/gemma4`:

1. `$TT_CACHE_PATH`, if set;
2. else `$PPLX_DECIDER_WEIGHT_CACHE` (legacy), if set;
3. else the model directory itself, next to the weights. For the HF-cache snapshot that is
   `$HF_HOME/hub/models--perplexity-ai--pplx-decider-v1-27b/snapshots/<revision>/`.

The cache is the `tt_weight_cache/<key>/` directory under that root. `<key>` is the revision for an HF-cache
snapshot, also when you reach the snapshot through a symlink (the layout is checked on the resolved path).
Any other directory gets `local-` plus a hash of its `config.json` and `model.safetensors.index.json` and
the sizes of `readout.safetensors` and the shards, so two checkpoints never share a cache and a plain `cp`
of a checkpoint keeps its key. The load log names the model path and the cache directory, and says whether
the cache was a full hit.

To keep the cache somewhere else, for example on another disk, move the directory and point
`TT_CACHE_PATH` at the new root:

```bash
export TT_CACHE_PATH=/data/tt_cache
mkdir -p $TT_CACHE_PATH/tt_weight_cache
mv $HF_HOME/hub/models--perplexity-ai--pplx-decider-v1-27b/snapshots/<revision>/tt_weight_cache/<revision> \
   $TT_CACHE_PATH/tt_weight_cache/
```

## Generate the HF reference goldens (CPU, once)

The tests compare against HF bf16/fp32 outputs that are computed on CPU, one layer at a time, from
the snapshot. The 54 GB model is never loaded at once. A test whose golden is missing fails with the
command that generates it.

| golden | command | time | peak RAM | disk |
|---|---|---:|---:|---:|
| layer inputs (PCC tests) | `python -m models.demos.pplx_decider_v1_27b.reference.hf_reference --seq-len 8192 --layers 0 3 61 63 --through-final` | ~10 min | not recorded | 0.8 GB |
| 25-row text decisions (e2e) | `python -m models.demos.pplx_decider_v1_27b.reference.hf_decision_golden` | ~32 min | 5.4 GB | 0.6 GB |
| vision tower, 8 images | `python -m models.demos.pplx_decider_v1_27b.reference.hf_vision_reference` | ~13 s | 2.3 GB | 0.4 GB |
| 8-row image decisions | `python -m models.demos.pplx_decider_v1_27b.reference.hf_image_decision_golden` | ~3 min | 4.0 GB | 30 MB |

Run the vision golden before the image golden: it draws the 8 PNGs that the image rows use. Outputs go
to `$PPLX_DECIDER_ARTIFACTS/goldens/{S8192,decisions,vision,vision/e2e}`.

## Correctness tests

Times are wall clock with the weight cache built and in the page cache.

| what it checks | file | command | time |
|---|---|---|---:|
| every module and both layer kinds vs HF (PCC >= 0.995, 5 buckets) | `tests/pcc/` | `pytest models/demos/pplx_decider_v1_27b/tests/pcc -q` | ~11 min |
| one layer kind at one length | `tests/pcc/test_decoder_layer.py` | `pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_decoder_layer.py -q -k "L3 and S2048"` | ~20 s |
| text decisions vs HF (>= 24/25, logit PCC >= 0.99), padding, determinism, on-device audit | `tests/e2e/test_model.py` | `pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_model.py -q` | ~50 s |
| image decisions vs HF (>= 7/8), splice, 3D positions, `TTDecider` images | `tests/e2e/test_image_model.py` | `pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_image_model.py -q` | ~20 s |
| vision tower: host tables, per-block and tower PCC, key mask, audit, latency, DRAM | `tests/vision/` | `pytest models/demos/pplx_decider_v1_27b/tests/vision -q` | ~25 s |
| 3D mRoPE vs HF (CPU, bit exact) and layer 3 on an image row | `tests/pcc/test_mrope.py` | `pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_mrope.py -q` | ~12 s |
| a prefill pass issues no host round trips or torch ops | `tests/pcc/test_runtime_audit.py` | `pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_runtime_audit.py -q` | ~10 s |
| default path loads the dtypes and fidelities of the precision JSON | `tests/e2e/test_precision_config.py` | `pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_precision_config.py -q -s` | ~5 s |
| model path and weight cache resolution order, cache keys (CPU, no device) | `tests/test_paths.py` | `pytest models/demos/pplx_decider_v1_27b/tests/test_paths.py -q` | ~1 s |

Tables from the logs: `python models/demos/pplx_decider_v1_27b/tests/pcc_report.py` (PCC suite),
`python models/demos/pplx_decider_v1_27b/tests/vision/vision_pcc_report.py` (vision) and
`python models/demos/pplx_decider_v1_27b/tests/e2e/report_layer_trace.py` (per-layer trace from the
last `test_model.py` run). To check that a change leaves text outputs bit-identical, compare two
`e2e_outputs.safetensors` dumps:

```bash
python -m models.demos.pplx_decider_v1_27b.tests.e2e.compare_outputs BEFORE.safetensors AFTER.safetensors
```

## Performance

All perf tests are warmed (compile and program cache excluded) and report medians. "Burst" means each
pass runs after 10 s idle; "sustained" means back to back. The device is slower when sustained.

| what it measures | file | command | time |
|---|---|---|---:|
| wall clock per layer kind and module (layer, mixer, MLP, embedding, head), 5 buckets | `tests/perf/test_prefill_perf.py` | `pytest models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py -q -s -k "test_layer_and_module_perf or test_embedding_and_head_perf"` | ~15 s |
| wall clock of a full request per bucket, burst and sustained; device forward, tokenize, upload | `tests/perf/test_model_perf.py` | `pytest models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py -q -s -k "test_model_perf and not traced and not profile"` | ~12 min |
| traced vs eager full forward (128 / 1024 / 2048) | `tests/perf/test_model_perf.py` | `pytest models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py -q -s -k test_traced_forward` | ~35 s |
| image request per golden row, with a phase breakdown | `tests/perf/test_image_perf.py` | `pytest models/demos/pplx_decider_v1_27b/tests/perf/test_image_perf.py -q -s` | ~17 min |
| vision tower per patch bucket; DRAM next to the text model | `tests/vision/test_vision_perf.py` | `pytest models/demos/pplx_decider_v1_27b/tests/vision/test_vision_perf.py -q -s` | <25 s |
| benchmark accuracy, mixed throughput, image latency | `benchmark/run_benchmark.py` | see [Benchmark](#benchmark-stage-11) | ~8 min |

The long runs are long on purpose: burst mode idles 10 s before every timed pass.

Print the per-layer table from its log: `python models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py`.
Results land in `$PPLX_DECIDER_ARTIFACTS/perf/prefill_perf.jsonl`, `e2e/model_perf.json`,
`image_e2e/image_perf.json` and `vision/vision_perf.json`.

### Tracy device profile and tt-perf-report

Two tests exist only to be profiled. Each runs warm-up passes, then one pass between the
`PREFILL_START` / `PREFILL_END` signposts.

```bash
# one decoder layer: L0_linear or L3_full, at S128, S2048 or S8192
python -m tracy -r -p -v --no-web-server -o generated/pplx_tracy/layer -m pytest \
  "models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py::test_profile_layer[L3_full-S2048-device_params0]"
# the full 64-layer model at the 2048 bucket (~1.7k ops; raise the profiler program count)
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=4000 python -m tracy -r -p -v --no-web-server -o generated/pplx_tracy/model \
  -m pytest "models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py::test_profile_model"
```

The op CSV lands in `<-o dir>/reports/<timestamp>/ops_perf_results_<timestamp>.csv`. Summarize the
signposted window:

```bash
tt-perf-report generated/pplx_tracy/layer/reports/<timestamp>/ops_perf_results_<timestamp>.csv \
  --start-signpost PREFILL_START --end-signpost PREFILL_END --no-advice --csv layer_ops.csv
```

Drop `--no-advice` to get optimization hints. Earlier reports are in `doc/*/perf/`.

## Benchmark (stage 11)

WinoGrande, FinancialPhraseBank and Belebele: a frozen, seeded subset of 200 examples each, run as
`choice` decisions through `TTDecider.predict`. The dataset step downloads the datasets into the HF
cache and needs network access.

```bash
B=$PPLX_DECIDER_ARTIFACTS/benchmark
python -m models.demos.pplx_decider_v1_27b.benchmark.datasets --out $B/subsets                  # CPU
PPLX_DECIDER_STAGE6_DIR=$B/perf_buckets pytest models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py::test_model_perf -q -s
python -m models.demos.pplx_decider_v1_27b.benchmark.run_benchmark --subsets $B/subsets --out $B/run
python -m models.demos.pplx_decider_v1_27b.benchmark.report --stage11 $B --doc models/demos/pplx_decider_v1_27b/doc/benchmark
```

`run_benchmark` needs the image golden (it times two golden image rows). `report` rewrites
`doc/benchmark/results.json` and `predictions.csv`. `REPORT.md` is written by hand from those files.

## Environment variables

| variable | default | used for |
|---|---|---|
| `HF_HOME` | HF default | HF cache with the snapshot (and the benchmark datasets) |
| `HF_MODEL` | HF cache at the pinned revision | model path (local directory or HF repo id) when `model_path` / `--model-path` is not given |
| `TT_CACHE_PATH` | the model directory | root of the device weight cache (27 GB), `<root>/tt_weight_cache/<key>/` |
| `PPLX_DECIDER_SNAPSHOT` | unset | legacy: model path, used after `HF_MODEL` |
| `PPLX_DECIDER_WEIGHT_CACHE` | unset | legacy: weight cache root, used after `TT_CACHE_PATH` |
| `PPLX_DECIDER_ARTIFACTS` | `<tt-metal>/generated/pplx_decider` | root of every default below |
| `PPLX_DECIDER_GOLDEN_DIR` | `$ARTIFACTS/goldens` | HF layer inputs (`S8192/`, `S16384/`) |
| `PPLX_DECIDER_DECISION_GOLDEN` | `$ARTIFACTS/goldens/decisions` | 25-row text decision golden |
| `PPLX_DECIDER_VISION_GOLDEN` | `$ARTIFACTS/goldens/vision` | vision tower golden and the 8 PNGs |
| `PPLX_DECIDER_IMAGE_GOLDEN` | `$PPLX_DECIDER_VISION_GOLDEN/e2e` | 8-row image decision golden |
| `PPLX_DECIDER_PCC_LOG` | `$ARTIFACTS/logs/pcc_results.jsonl` | PCC records of the PCC tests |
| `PPLX_DECIDER_PROBE_LOG` | `$ARTIFACTS/logs/context_probe.jsonl` | context probe (`tests/probe`, S=16384) |
| `PPLX_DECIDER_PERF_LOG` | `$ARTIFACTS/perf/prefill_perf.jsonl` | per-layer perf records |
| `PPLX_DECIDER_STAGE6_DIR` | `$ARTIFACTS/e2e` | text e2e results, `model_perf.json`, precision check |
| `PPLX_DECIDER_STAGE12B_DIR` | `$ARTIFACTS/image_e2e` | `image_perf.json`; fallback for the next row |
| `PPLX_DECIDER_IMAGE_E2E_DIR` | `$PPLX_DECIDER_STAGE12B_DIR` | image e2e results |
| `PPLX_DECIDER_STAGE12A_DIR` | `$ARTIFACTS/vision` | vision test results |
| `PPLX_DECIDER_VISION_PCC_LOG` | `$PPLX_DECIDER_STAGE12A_DIR/pcc_vision.jsonl` | vision PCC records |
| `PPLX_DECIDER_PRECISION_CONFIG` | `doc/datatype_sweep/selected_precision_config.json` | precision policy JSON; candidates in `doc/datatype_sweep/candidates/` |

All locations are defined in [`tt/paths.py`](tt/paths.py). Set a result directory per run when you do
not want to overwrite an earlier run's JSON.

## Code layout

```
tt/                  TTNN model
  model.py           PplxDeciderModel: weight cache, bucket padding, image input prep, on-device splice
  decoder.py         one decoder layer (both kinds); attention.py, gated_deltanet.py, mlp.py, norm.py
  rope.py            1D tables, 3D mRoPE (get_rope_index port), per-request rotary
  embedding.py, readout.py, head.py   embedding; final norm -> readout -> mask -> / T -> softmax
  weight_adapter.py  HF -> TT weight transforms; optimizations.py: dtypes and op configs (one place)
  paths.py           every filesystem location, model path and weight cache resolution (tests/test_paths.py, CPU)
  vision/            vision tower: patch embed, 27 blocks, merger (tower.py)
demo/                decider.py (TTDecider, the app's Decider.predict) and demo.py (CLI)
reference/           CPU HF goldens, built one layer at a time; prompt sets (decision_prompts.py, image_decision_prompts.py)
tests/pcc/           per-module and per-layer PCC, mRoPE, prefill contract, runtime audit
tests/e2e/           full-model decision gates (text, image), precision config, compare_outputs.py
tests/vision/        vision tower tests; tests/perf/: latency; tests/probe/: context probe beyond 8192
benchmark/           stage-11 datasets, runner and report
doc/                 per-stage READMEs, work logs and perf reports
```

## Precision and limits

- Text model: precision config C0 (`doc/datatype_sweep/selected_precision_config.json`): BFP8_B
  weights for every projection and the readout, HiFi2 matmuls with fp32 accumulation, BF16
  activations, fp32 DeltaNet state and fp32 head. The vision tower runs in BF16. BFP4 MLP weights
  were 17 % faster at 2048 but failed the per-row logit-PCC bar; see the datatype sweep.
- Batch 1 only. Prompts are right-padded to buckets 128 / 1024 / 2048 / 4096 / 8192; prompts over
  8192 tokens are rejected, as in the app.
- Tested with one image per request. Several images per request go through the same code but are
  untested.
- Weights and state take 27.3 GiB of the 31.8 GiB DRAM; with the vision tower, 3.6 GiB stay free.

## Stage reports

| stage | report |
|---|---|
| 1, functional layers | [`doc/functional_decoder/README.md`](doc/functional_decoder/README.md) |
| 2, graph fusing | [`doc/fused_decoder/README.md`](doc/fused_decoder/README.md) |
| 3A/3B, optimized decoder | [`doc/optimized_decoder/README.md`](doc/optimized_decoder/README.md) |
| 6, full model and demo | [`doc/full_model/README.md`](doc/full_model/README.md) |
| 8, datatype sweep | [`doc/datatype_sweep/README.md`](doc/datatype_sweep/README.md) |
| 11, benchmark | [`doc/benchmark/REPORT.md`](doc/benchmark/REPORT.md) |
| 12A/12B, vision tower and image decisions | [`doc/vision/README.md`](doc/vision/README.md) |
| context contract | [`doc/context_contract.json`](doc/context_contract.json) |
