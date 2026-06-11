# Devstral-2-123B-Instruct (`experimental`)

Package path: `models/experimental/devstral2_123B_instruct/`

## Platforms

Blackhole Loudbox (1×8 mesh).

## Introduction

This folder contains an experimental Tenstorrent (`ttnn`) port of **Mistral [Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)** (Ministral3 text stack). PCC tests compare subgraphs and the full model against HuggingFace references.

**Full-model PCC:** End-to-end parity for the whole TT stack (`TtMinistral3Model`, all **88** decoder layers) is measured in `tests/test_ministral3_full_model.py` — **0.99 PCC**.

**Maximum context length:** The HF checkpoint advertises very long context (YaRN / RoPE tables up to 256K positions). On Blackhole Loudbox (1×8), the working KV budget (`max_seq_len`) is what you can actually run. **Up to 96K tokens** has been verified end-to-end on this mesh (`DEVSTRAL2_MIN_MAX_SEQ_LEN` / `--max-seq-len` default **98304** in `text_demo.py` and `tt_demo_agent.py`).

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
pytest models/experimental/devstral2_123B_instruct/tests/ -k pcc
```

Single file, e.g. attention only:

```sh
pytest models/experimental/devstral2_123B_instruct/tests/test_ministralattn.py -k pcc
```

## Demos

Run from repo root on Blackhole Loudbox.

| Demo | Script | Description | Command |
|------|--------|-------------|---------|
| Text LM | `demo/text_demo.py` | Text-only TT prefill/decode + LM head. Override prompt with `DEVSTRAL2_PROMPT`. | `pytest models/experimental/devstral2_123B_instruct/demo/text_demo.py` |
| Interactive agent | `demo/tt_demo_agent.py` | Multi-turn coding REPL on TT. | `python models/experimental/devstral2_123B_instruct/demo/tt_demo_agent.py --mesh-device T3K` |

## Environment setup (for performance tests and demos)

```sh
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH="${TT_METAL_HOME}"
export BH_ARCH_YAML=tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml
```

## Performance

### Running the tests

**End-to-end (88-layer, full model) — reports TTFT, prefill tok/s, steady-state decode tok/s/user, and end-to-end decode tok/s/user:**

```sh
pytest models/experimental/devstral2_123B_instruct/tests/perf/test_e2e_performant.py -k L88
```

**Single-layer wall-clock perf — validates trace pipeline on a 1-layer model:**

```sh
pytest models/experimental/devstral2_123B_instruct/tests/perf/test_perf.py
```

**Single-layer device perf (prefill + decode) — Tracy capture for 1-layer partial weights (prefill 128 tokens, decode 1 token at position 128):**

```sh
pytest models/experimental/devstral2_123B_instruct/tests/perf/test_device_perf_single_layer_prefill_decode.py \
    -v -m models_device_performance_bare_metal
```

This invokes `run_device_perf`, which runs the profile workload in `tests/perf/test_profile_single_layer_prefill_decode.py`. Each measured iteration profiles **both** prefill and decode inside the `start`/`stop` signpost window (one warmup iteration runs outside signposts).

After the run, analyze the ops CSV under `generated/profiler/devstral2_123B_instruct_L1_prefill_decode/reports/<timestamp>/`:

```sh
tt-perf-report generated/profiler/devstral2_123B_instruct_L1_prefill_decode/reports/<timestamp>/ops_perf_results_*.csv \
    --start-signpost start --end-signpost stop
```

`--start-signpost` and `--end-signpost` are required; the default `tt-perf-report` mode only anchors on the last signpost and shows no device ops.

Optional: capture Tracy directly without the device-perf dumper:

```sh
python -m tracy -p -v -r --dump-device-data-mid-run \
    pytest models/experimental/devstral2_123B_instruct/tests/perf/test_profile_single_layer_prefill_decode.py \
    ::test_profile_single_layer_prefill_decode -v
```

### Results

Measured with ``pytest models/experimental/devstral2_123B_instruct/tests/perf/test_e2e_performant.py -k L88``
(2CQ decode trace on, ``DEVSTRAL2_DECODE_TRACE_2CQ=1``). Prefill prompt is synthetic zeros (128 tokens).

| Test | System | Mesh | Prompt tokens | Decode iters | TTFT (ms) | Prefill tok/s | Steady-state tok/s/user | End-to-end tok/s/user |
|:-----|:-------|:-----|-------------:|-------------:|----------:|--------------:|------------------------:|----------------------:|
| E2E L88 (2CQ traced) | BH Loudbox | 1×8 | 128 | 32 | 102.2 | 1262 | 14.02 | 8.81 |

**Metric definitions** :

- **TTFT** — one prefill trace replay after capture (time to first decode logits).
- **Prefill tok/s** — ``prompt_len / prefill_trace_replay_time`` (compile pass excluded).
- **Steady-state tok/s/user** — ``decode_iters / decode_trace_replay_total`` (compile + capture excluded).
- **End-to-end tok/s/user** — ``decode_iters / (TTFT + decode compile + decode capture + decode replays)``.



## Resources

| Path | Purpose |
|------|---------|
| **`demo/`** | Runnable demos: `text_demo.py` (TT text LM), `tt_demo_agent.py` (interactive TT agent), `decode_trace_2cq.py` (2CQ decode-trace helpers). |
| **`tt/`** | TT layer implementations (Ministral3 building blocks). |
| **`tt/tt_ministral3_model.py`** | Top-level model: embed → decoder layers → RMSNorm. |
| **`tt/tt_ministral3_decoder_layer.py`** | Decoder layer (attention + MLP + norms). |
| **`tt/weight_loading.py`** | Host → device FP8 → bf16 weight dequant and upload (shard-by-shard, disk-cached). |
| **`tests/`** | Per-op PCC tests (attention, MLP, norms, RoPE, decoder layer, full model). |
| **`tests/perf/`** | Performance tests: e2e throughput (`test_e2e_performant.py`), single-layer wall-clock perf (`test_perf.py`), and single-layer prefill+decode device perf (`test_device_perf_single_layer_prefill_decode.py`, `test_profile_single_layer_prefill_decode.py`). |
| **`reference/`** | Pure PyTorch / HF reference inference script (`devstral2_123b_inference.py`). |

## Model and limits

| Item | Value |
|------|--------|
| HF `max_position_embeddings` | 262,144 (model-native RoPE horizon) |
| Practical `max_seq_len` (TT KV cache) | Sized per run: `prompt + max_new_tokens`, floored by `DEVSTRAL2_MIN_MAX_SEQ_LEN` (default **98,304**) |
| Verified context on BH Loudbox (1×8) | **Up to 96K** tokens (end-to-end text generation) |
| Full-model PCC (88 layers) | **0.99** (`tests/test_ministral3_full_model.py`) |

### Limitations

- **Context length:** Long context is limited by per-chip KV and RoPE allocation, not by the HF config alone. On Blackhole Loudbox (1×8), **up to 96K tokens** has been verified for end-to-end generation. Defaults use `max_seq_len=98304` (`DEVSTRAL2_MIN_MAX_SEQ_LEN` / `--max-seq-len`). Prompts longer than the configured budget need a larger cap and sufficient device memory.
