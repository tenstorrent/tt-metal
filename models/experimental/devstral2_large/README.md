# Devstral 2 123B Instruct (experimental)

## Platforms

Blackhole Loudbox (1×8 mesh).

## Introduction

This folder contains an experimental Tenstorrent (`ttnn`) port of **Mistral [Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)** (Ministral3 text stack). PCC tests compare subgraphs and the full model against HuggingFace references.

**Maximum context length:** The HF checkpoint advertises very long context (YaRN / RoPE tables up to 256K positions). On Blackhole Loudbox (1×8), the working KV budget (`max_seq_len`) is what you can actually run. **~150K tokens** is a rough theoretical upper bound on this mesh given per-chip DRAM; but have **only validated end-to-end generation at 32K** (`DEVSTRAL2_MIN_MAX_SEQ_LEN` default in `text_demo.py`).

**Traced prefill/decode** with **2CQ** decode staging is on by default (`DEVSTRAL2_TRACE_PREFILL=1`, `DEVSTRAL2_DECODE_TRACE_2CQ=1`). Set either to `0` to disable.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Python packages (install into your tt-metal environment):

  ```sh
  pip install git+https://github.com/huggingface/transformers
  ```



## How to run (PCC tests)

From repo root:

**All PCC tests** (building blocks + full model; long run, loads Devstral weights):

```sh
pytest models/experimental/devstral2_large/tests/ -k pcc
```

Single file, e.g. attention only:

```sh
pytest models/experimental/devstral2_large/tests/test_ministralattn.py -k pcc
```

## Demos

Run from repo root on Blackhole Loudbox.

| Demo | Script | Description | Command |
|------|--------|-------------|---------|
| Text LM | `demo/text_demo.py` | Text-only TT prefill/decode + LM head. Override prompt with `DEVSTRAL2_PROMPT`. | `pytest models/experimental/devstral2_large/demo/text_demo.py` |
| Interactive agent | `demo/tt_demo_agent.py` | Multi-turn coding REPL on TT. | `python models/experimental/devstral2_large/demo/tt_demo_agent.py --mesh-device T3K` |

## Environment setup (for performance tests and demos)

```sh
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH="${TT_METAL_HOME}"
export BH_ARCH_YAML=tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml
```

## Performance

### Running the tests

**End-to-end (88-layer, full model) — reports TTFT, prefill tok/s, and decode tok/s/user:**

```sh
pytest models/experimental/devstral2_large/tests/perf/test_e2e_performant.py -k L88
```

**Single-layer device perf — validates trace pipeline on a 1-layer model:**

```sh
pytest models/experimental/devstral2_large/tests/perf/test_perf.py
```

### Results

| Test | System | Mesh | Prompt tokens | Decode iters | TTFT (ms) | Decode tok/s/user |
|:-----|:-------|:-----|-------------:|-------------:|----------:|------------------:|
| E2E L88 (2CQ Traced) | BH Loudbox | 1×8 | 128 | 32 | 102.2 | 14.06 |



## Resources

| Path | Purpose |
|------|---------|
| **`demo/`** | Runnable demos: `text_demo.py` (TT text LM), `tt_demo_agent.py` (interactive TT agent), `decode_trace_2cq.py` (2CQ decode-trace helpers). |
| **`tt/`** | TT layer implementations (Ministral3 building blocks). |
| **`tt/tt_ministral3_model.py`** | Top-level model: embed → decoder layers → RMSNorm. |
| **`tt/tt_ministral3_decoder_layer.py`** | Decoder layer (attention + MLP + norms). |
| **`tt/weight_loading.py`** | Host → device FP8 → bf16 weight dequant and upload (shard-by-shard, disk-cached). |
| **`tests/`** | Per-op PCC tests (attention, MLP, norms, RoPE, decoder layer, full model). |
| **`tests/perf/`** | Performance tests: e2e throughput (`test_e2e_performant.py`) and single-layer device perf (`test_perf.py`). |
| **`reference/`** | Pure PyTorch / HF reference inference script (`devstral2_123b_inference.py`). |

## Model and limits

| Item | Value |
|------|--------|
| HF `max_position_embeddings` | 262,144 (model-native RoPE horizon) |
| Practical `max_seq_len` (TT KV cache) | Sized per run: `prompt + max_new_tokens`, floored by `DEVSTRAL2_MIN_MAX_SEQ_LEN` (default **32,768**) |
| Theoretical context on 1×8 BH | **~150K** tokens (KV/DRAM bound) |
| Tested context (text demo) | **32K** |

### Limitations

- **Context length:** Long context is limited by per-chip KV and RoPE allocation, not by the HF config alone. **~150K tokens** may be reachable in theory on a 1×8 Blackhole mesh, but this port has **only been tested through 32K** (`DEVSTRAL2_MIN_MAX_SEQ_LEN=32768`). Longer prompts need a larger `max_seq_len` / `--max-seq-len` and sufficient device memory.
