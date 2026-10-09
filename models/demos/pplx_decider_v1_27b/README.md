# pplx-decider-v1-27b on one Blackhole p150a (TTNN, prefill only)

[`perplexity-ai/pplx-decider-v1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b)
(revision `b01a5cbaca5391f73bd55103d4f27e8982cd5e60`) is a 255-way decision classifier on a
Qwen3.5 hybrid text backbone: 64 decoder layers (48 Gated DeltaNet `linear_attention` layers and
16 gated `full_attention` layers, full at index 3 mod 4), then the final RMSNorm, the last token's
hidden state, a 5120x255 `readout`, a mask by option count, `/ temperature` and a softmax. It has no
decode step, no KV cache across requests and no LM head.

Status: stage 6 (full model, decision head, demo). All 64 layers, the embedding and the decision
head run on one p150a; 27.30 GiB of weights and state stay in device DRAM. On the 25-row HF bf16
decision golden, TT makes the same decision on **25/25** rows. Readout-logit PCC is >= 0.998 on
every row. Warmed request latency at batch 1 runs from 148 ms (128 bucket) to 3.5 s (8192 bucket)
for an isolated request, and is 1.25-1.36x higher under sustained back-to-back load. Details:
[`doc/full_model/README.md`](doc/full_model/README.md).

Stage 8 (datatype sweep, short) kept the all-BFP8 HiFi2 policy (person decision) and stage 6's numbers stand.
The policy is now `doc/datatype_sweep/selected_precision_config.json`, which the default construction path
reads. A BFP4 MLP (C1/C2) agrees on 25/25 rows but fails the per-row logit-PCC bar on one row (0.98895 < 0.99).
BFP4 with LoFi would be 17 % faster at the 2048 bucket and would free 8 GiB of DRAM. Details:
[`doc/datatype_sweep/README.md`](doc/datatype_sweep/README.md).

Stage 12A (vision tower, text model unchanged): the Qwen3.5 ViT runs on the same p150a in BF16. It
covers the patch embed, the learned position embedding, 27 blocks and the 2x2 merger, and outputs
5120-wide image features. Results on 8 golden images (256-1024 patches), measured against the HF
BF16 golden:

- Teacher forced, every module and every block has PCC >= 0.999976 (bar 0.995).
- The whole tower, fed only the pixels, has PCC >= 0.998654 (bar 0.99).
- Padded keys are masked. An image of 936 patches in the 1024 bucket scores 0.999758.

The tower takes 15.5-28.6 ms per image. Its weights take 0.964 GiB of DRAM, which leaves 3.57 GiB
free next to the full text model. Splicing the features into the text and the 3D mRoPE are stage
12B. Details: [`doc/vision/README.md`](doc/vision/README.md).

Stage 1 (functional layers): every module and both decoder-layer kinds match the HF reference with
real weights (PCC >= 0.995) at the prefill buckets 128, 1024, 2048, 4096 and 8192. The app pads a
prompt on the right to the next bucket. The classifier reads the last real token. Details are in
[`doc/functional_decoder/README.md`](doc/functional_decoder/README.md).

Stage 2 (graph fusing, prefill) changed the modules in place:

- fused SwiGLU MLP matmul;
- separate q|k|v and gate matmuls;
- one full-width fused RoPE op on pre-permuted q/k head dims;
- GDN in-projection as qkv plus z|b|a.

Layers are 3-8 % faster at S >= 1024, and S=128 is about 12 % slower. Details are in
[`doc/fused_decoder/README.md`](doc/fused_decoder/README.md).

## Layout

| path | content |
|---|---|
| `tt/` | TTNN modules: `embedding.py`, `norm.py`, `rope.py`, `attention.py` (gated full attention), `gated_deltanet.py`, `mlp.py`, `decoder.py` (both layer kinds), `readout.py`, `head.py` (final norm -> readout -> mask -> / T -> softmax), `model.py` (the full model, weight cache, bucket padding); `weight_adapter.py` (HF -> TT weight transforms), `optimizations.py` (the one place for dtypes and op configs), `model_config.py`. |
| `tt/vision/` | Vision tower (stage 12A): `config.py` (vision args, patch buckets), `weights.py` (`visual.*` -> TT weights, strict keys, head-dim / intermediate padding), `inputs.py` (host pos-embed and rotary tables, as HF), `patch_embed.py`, `layernorm.py`, `attention.py` (masked windowed SDPA), `mlp.py`, `block.py`, `merger.py`, `tower.py` (`PplxVisionTower`: `prepare_inputs` + device-only `forward`). |
| `demo/` | `decider.py` (`TTDecider.predict(state, question)`, the app's `Decider.predict`), `demo.py` (the snapshot `inference.py` example). |
| `reference/hf_reference.py` | Layer-streamed HF golden generator. Builds one HF module at a time from the snapshot; never loads the 54 GB model. |
| `reference/decision_prompts.py`, `reference/hf_decision_golden.py`, `reference/hf_demo_reference.py` | The 25-row decision prompt set, its layer-streamed HF bf16 golden, and HF answers for the demo rows. |
| `tests/e2e/` | Full-model decision-agreement gate vs the HF golden, layer-wise trace, bucket padding, determinism, fallback audit. |
| `tests/perf/test_model_perf.py` | Full-model latency per bucket (burst and sustained), trace replay, profiler target. |
| `tests/pcc/` | Real-weight PCC tests per module and per layer kind, the chunked-prefill contract test and the runtime fallback audit. |
| `tests/perf/test_prefill_perf.py` | Warmed prefill latency per layer kind and module; device-profiler capture target. |
| `tests/vision/` | Vision tower vs the HF BF16 golden: host tables and adapter (CPU), per-module and per-block PCC (teacher forced), tower end to end, key mask, GELU variant, fallback audit, determinism, latency, DRAM next to the text model. |
| `doc/vision/` | Stage-12A README and work log (PCC table, perf, DRAM). |
| `tests/probe/test_context_probe.py` | Context probe beyond the app limit (S=16384) with DRAM numbers. |
| `doc/context_contract.json` | Context contract (HF-advertised, app limit, tested). |
| `doc/functional_decoder/` | Stage-1 README, work log and `perf/` (tt-perf-report tables and CSVs). |
| `doc/fused_decoder/` | Stage-2 (graph fusing) README, work log and `perf/` (tt-perf-report at S=128/2048/8192). |
| `doc/full_model/` | Stage-6 README (latency, agreement table, layer trace), work log, `perf/` (tt-perf-report summary of one 2048 forward). |

## Setup

Requirements: a source build of tt-metal with its `python_env`, one Blackhole p150a, the HF snapshot
in the local HF cache, about 2 GB of free disk for the goldens plus 27 GB for the full-model weight
cache, and about 20 GB of free host RAM.

```bash
cd /path/to/tt-metal
source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export HF_HOME=/local/ttuser/gtobar/hf          # HF cache that holds the snapshot
```

Environment variables read by this model directory:

| variable | default | purpose |
|---|---|---|
| `HF_HOME` | HF default | HF cache; the snapshot is resolved with `local_files_only=True`. |
| `PPLX_DECIDER_SNAPSHOT` | resolved from `HF_HOME` | Absolute snapshot directory, overrides the cache lookup. |
| `PPLX_DECIDER_GOLDEN_DIR` | `/local/ttuser/gtobar/artifacts/pplx_decider/goldens` | HF layer inputs streamed from the snapshot. |
| `PPLX_DECIDER_PCC_LOG` | `/local/ttuser/gtobar/artifacts/pplx_decider/logs/pcc_results.jsonl` | PCC records appended by every test. |
| `PPLX_DECIDER_PERF_LOG` | `/local/ttuser/gtobar/artifacts/pplx_decider/perf/prefill_perf.jsonl` | Perf records. |
| `PPLX_DECIDER_PROBE_LOG` | `/local/ttuser/gtobar/artifacts/pplx_decider/logs/context_probe.jsonl` | Context-probe records (`tests/probe`). |
| `PPLX_DECIDER_WEIGHT_CACHE` | `/local/ttuser/gtobar/artifacts/pplx_decider/weight_cache` | ttnn disk weight cache of the full model (27 GB; `<revision>/weights/` below it, shared by all policies; file names carry the dtype). |
| `PPLX_DECIDER_PRECISION_CONFIG` | `doc/datatype_sweep/selected_precision_config.json` | Precision policy JSON read by `PrecisionPolicy.default()`; set it to a file in `doc/datatype_sweep/candidates/` to run another policy. |
| `PPLX_DECIDER_DECISION_GOLDEN` | `/local/ttuser/gtobar/artifacts/pplx_decider/goldens/decisions` | Stage-6 decision golden (`reference/hf_decision_golden.py`). |
| `PPLX_DECIDER_STAGE6_DIR` | `/local/ttuser/gtobar/artifacts/pplx_decider/stage6` | JSON results of the e2e and full-model perf tests. |

Generate the goldens once (CPU only, about 10 minutes, about 800 MB):

```bash
python -m models.demos.pplx_decider_v1_27b.reference.hf_reference \
  --seq-len 8192 --layers 0 3 61 63 --through-final
```

The stage-6 decision golden (25 rows, CPU, about 0.5 h, peak RSS about 5 GB):

```bash
python -m models.demos.pplx_decider_v1_27b.reference.hf_decision_golden
```

## Run the demo

The first run converts the weights layer by layer from the snapshot into the disk weight cache
(about 2.7 minutes, 27 GB). Later loads read the cache: about 37 s with a cold page cache, and 4 s
when the files are still in the page cache.

```bash
python models/demos/pplx_decider_v1_27b/demo/demo.py           # snapshot inference.py example, prints JSON
python models/demos/pplx_decider_v1_27b/demo/demo.py --state "Refund my order" \
  --question '{"type": "noul", "instructions": "Is this a refund request?"}'
```

In Python:

```python
import ttnn
from models.demos.pplx_decider_v1_27b.demo.decider import TTDecider

device = ttnn.open_device(device_id=0, l1_small_size=24576)
decider = TTDecider.from_pretrained(device)
print(decider.predict("My Stripe integration keeps failing.", {"type": "noul", "instructions": "Is it urgent?"}))
```

## Run the end-to-end test and the full-model perf

```bash
pytest models/demos/pplx_decider_v1_27b/tests/e2e/test_model.py -q -s     # 25-row agreement gate + checks, ~2 min
python models/demos/pplx_decider_v1_27b/tests/e2e/report_layer_trace.py   # doc/full_model/layer_trace.{md,png}
pytest models/demos/pplx_decider_v1_27b/tests/perf/test_model_perf.py -q -s -k "test_model_perf and not traced and not profile"
```

## Run the PCC tests

```bash
pytest models/demos/pplx_decider_v1_27b/tests/pcc -q                     # all cases, about 11 minutes
pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_decoder_layer.py -q -k "L3 and S4096"
python models/demos/pplx_decider_v1_27b/tests/pcc_report.py               # PCC table from the log
```

## Run the vision tower tests

```bash
pytest models/demos/pplx_decider_v1_27b/tests/vision -q -k "not test_dram_with_text_model"  # about 20 s
pytest models/demos/pplx_decider_v1_27b/tests/vision/test_vision_perf.py -q -s -k test_dram_with_text_model
python models/demos/pplx_decider_v1_27b/tests/vision/vision_pcc_report.py                  # PCC table from the log
```

## Run the perf measurements

```bash
pytest models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py -q -s \
  -k "test_layer_and_module_perf or test_embedding_and_head_perf"
python models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py  # median table

# Device profiler: one warmed layer between PREFILL_START / PREFILL_END signposts
python -m tracy -r -p -v --no-web-server -o /tmp/pplx_tracy -m pytest \
  models/demos/pplx_decider_v1_27b/tests/perf/test_prefill_perf.py -k "test_profile_layer and L3_full and S2048"
tt-perf-report /tmp/pplx_tracy/reports/<timestamp>/ops_perf_results_<timestamp>.csv \
  --start-signpost PREFILL_START --end-signpost PREFILL_END --no-advice
```
