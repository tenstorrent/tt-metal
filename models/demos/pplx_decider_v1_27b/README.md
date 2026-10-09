# pplx-decider-v1-27b on one Blackhole p150a (TTNN, prefill only)

[`perplexity-ai/pplx-decider-v1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b)
(revision `b01a5cbaca5391f73bd55103d4f27e8982cd5e60`) is a 255-way decision classifier on a
Qwen3.5 hybrid text backbone: 64 decoder layers (48 Gated DeltaNet `linear_attention` layers and
16 gated `full_attention` layers, full at index 3 mod 4), then the final RMSNorm, the last token's
hidden state, a 5120x255 `readout`, a mask by option count, `/ temperature` and a softmax. It has no
decode step, no KV cache across requests and no LM head.

Status: stage 1 (functional layers). Every module and both decoder-layer kinds run on device and
match the HF reference with real weights (PCC >= 0.995) at the prefill buckets 128, 1024, 2048,
4096 and 8192. The app pads a prompt on the right to the next bucket. The classifier reads the last
real token. The 64-layer model,
the decision head and the demo are later stages. Stage-1 details, numbers and the self-check are in
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
| `tt/` | TTNN modules: `embedding.py`, `norm.py`, `rope.py`, `attention.py` (gated full attention), `gated_deltanet.py`, `mlp.py`, `decoder.py` (both layer kinds), `readout.py`; `weight_adapter.py` (HF -> TT weight transforms), `optimizations.py` (the one place for dtypes and op configs), `model_config.py`. |
| `reference/hf_reference.py` | Layer-streamed HF golden generator. Builds one HF module at a time from the snapshot; never loads the 54 GB model. |
| `tests/pcc/` | Real-weight PCC tests per module and per layer kind, the chunked-prefill contract test and the runtime fallback audit. |
| `tests/perf/test_prefill_perf.py` | Warmed prefill latency per layer kind and module; device-profiler capture target. |
| `tests/probe/test_context_probe.py` | Context probe beyond the app limit (S=16384) with DRAM numbers. |
| `doc/context_contract.json` | Context contract (HF-advertised, app limit, tested). |
| `doc/functional_decoder/` | Stage-1 README, work log and `perf/` (tt-perf-report tables and CSVs). |
| `doc/fused_decoder/` | Stage-2 (graph fusing) README, work log and `perf/` (tt-perf-report at S=128/2048/8192). |

## Setup

Requirements: a source build of tt-metal with its `python_env`, one Blackhole p150a, the HF snapshot
in the local HF cache, about 2 GB of free disk for the goldens, and about 20 GB of free host RAM.

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

Generate the goldens once (CPU only, about 10 minutes, about 800 MB):

```bash
python -m models.demos.pplx_decider_v1_27b.reference.hf_reference \
  --seq-len 8192 --layers 0 3 61 63 --through-final
```

## Run the PCC tests

```bash
pytest models/demos/pplx_decider_v1_27b/tests/pcc -q                     # all cases, about 11 minutes
pytest models/demos/pplx_decider_v1_27b/tests/pcc/test_decoder_layer.py -q -k "L3 and S4096"
python models/demos/pplx_decider_v1_27b/tests/pcc_report.py               # PCC table from the log
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
