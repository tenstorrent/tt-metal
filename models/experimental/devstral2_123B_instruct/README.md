# Devstral-2-123B-Instruct (`experimental`)

Package path: `models/experimental/devstral2_123B_instruct/`

## Platforms

Blackhole Loudbox (1×8 mesh).

## Introduction

Experimental Tenstorrent (`ttnn`) port of **Mistral [Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)** (Ministral3 text stack).

**Context length:** Up to **262144** tokens (YaRN / RoPE). Default ``max_seq_len`` is **262144**
(`DEVSTRAL2_MIN_MAX_SEQ_LEN` / `--max-seq-len` in `text_demo.py` and `tt_demo_agent.py`).

**Traced prefill/decode** with **2CQ** decode staging is on by default
(`DEVSTRAL2_TRACE_PREFILL=1`, `DEVSTRAL2_DECODE_TRACE_2CQ=1`). Set either to `0` to disable.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Python packages:

  ```sh
  pip install git+https://github.com/huggingface/transformers
  ```

- **Model weights:** [Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512) (**gated**). Accept the Hub license, then `huggingface-cli login` or `export HF_TOKEN=<token>`. First run downloads required safetensor shards (FineGrained FP8 → bf16 on host) and builds a tiled TT cache under `generated/ttnn/devstral2_123B_instruct/weight_cache/`. Set `DEVSTRAL2_HF_LOCAL_ONLY=1` when weights are already cached (Loudbox CI uses `/mnt/MLPerf/huggingface`).

## Environment setup

From the tt-metal repo root (Blackhole Loudbox):

```sh
source python_env/bin/activate
export TT_METAL_HOME=$(pwd)
export PYTHONPATH="${TT_METAL_HOME}"
export BH_ARCH_YAML=tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml
```

## Demos

| Demo | Script | Command |
|------|--------|---------|
| Text LM | `demo/text_demo.py` | `pytest models/experimental/devstral2_123B_instruct/demo/text_demo.py` |
| Interactive agent | `demo/tt_demo_agent.py` | `python models/experimental/devstral2_123B_instruct/demo/tt_demo_agent.py --mesh-device T3K --workspace-root /path/to/repo` |

Override the text prompt with `DEVSTRAL2_PROMPT`. The agent demo is interactive/local only; Blackhole CI runs `text_demo.py`.

### Interactive agent (`tt_demo_agent.py`)

Multi-turn coding REPL on TT with **workspace-only** file tools. Inference is on-device; tool
execution is Python on the host. Useful flags: `--num-layers`, `--max-seq-len`, `--max-context-tokens`,
`--verbose`. Type `quit` / `exit` to stop; `/clear` resets chat.

| Tool | What it does |
|------|----------------|
| `read_file` | Read a slice of a text file (offset/limit). |
| `write_file` | Create or overwrite (or append) a file under the workspace. |
| `search_replace` | Find/replace in a file. |
| `grep` | Regex search over files under a path (in-process; no shell). |
| `inspect_codebase` | List file paths under a directory (skips `.git`). |
| `load_skill` | Read a markdown/text “skill” file from the workspace. |
| `todo` | In-memory task list for the session. |
| `ask_user_question` | Prompt you in the terminal for clarification. |

Shell, web fetch, and web search are **not** implemented. The model may write scripts to the
workspace but the demo does not execute them. Use a dedicated `--workspace-root`, not your home
directory or a repo with secrets.

## End-to-end full-model tests

Full 88-layer comparisons vs HuggingFace at varying prefill lengths (powers of two **32 … 262144**).
Both use **128-token chunked prefill**, Tale of Two Cities tokens (tiled as needed), and a shared
on-disk weight cache at **`seq_262144`** (`DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN`).

**CI evaluates short prefill lengths only** (`-k sanity`). Long prefills take hours per point on BH
Loudbox and are available via `-k sweep` for extended validation.

| Test | `-k sanity` | `-k sweep` |
|------|-------------|------------|
| Logit PCC | 32, 128 | 32 … 262144 (14 points) |
| Token match | 32, 64, 128 | 32 … 262144 (14 points) |

| Workflow | Command |
|----------|---------|
| ``(Blackhole) e2e tests`` → ``devstral2-123b-instruct`` | ``test_model_logit_pcc.py -k sanity`` then ``test_model_token_match.py -k sanity`` |
| ``(Blackhole) Demo tests`` → ``devstral2-123b-instruct`` | ``demo/text_demo.py`` |

Job: ``bh-lb-devstral2-123b-e2e-sanity`` (120 min timeout).

### Logit PCC

**Files:** `tests/test_model_logit_pcc.py`, `tests/logit_pcc_common.py`

Compares **raw logits** (full vocab). After each check, both HF and TT advance with the **HF greedy**
token (temperature=0 argmax). Threshold: PCC ≥ **0.90** on last-prefill logits and **10** decode
steps. HF reference runs on CPU with disk offload when no CUDA GPU is present.

```sh
pytest models/experimental/devstral2_123B_instruct/tests/test_model_logit_pcc.py -k sanity -v
pytest models/experimental/devstral2_123B_instruct/tests/test_model_logit_pcc.py -k sweep -v
```

### Token match (teacher-forced)

**File:** `tests/test_model_token_match.py`

Top-1 / top-5 token accuracy vs HuggingFace over **500** teacher-forced eval tokens after each
prefill. Thresholds: top-1 ≥ **96%**, top-5 ≥ **99%** (`DEVSTRAL2_MIN_TOP1_ACC`,
`DEVSTRAL2_MIN_TOP5_ACC`).

```sh
pytest models/experimental/devstral2_123B_instruct/tests/test_model_token_match.py -k sanity -v
pytest models/experimental/devstral2_123B_instruct/tests/test_model_token_match.py -k sweep -v
```

Sweep outputs: `tests/teacher_forced_sweep_outputs/` (`references/*.refpt`, `results/prefill_*.json`).


#### Results

Blackhole Loudbox (1×8). **500** eval tokens per prefill.

| Prefill len | Top-1 | Top-5 | Top-1 mismatches | Pass |
|------------:|------:|------:|-----------------:|:----:|
| 32 | 97.80% | 100% | 11 / 500 | ✓ |
| 64 | 98.20% | 100% | 9 / 500 | ✓ |
| 128 | 98.80% | 100% | 6 / 500 | ✓ |
| 256 | 99.80% | 100% | 1 / 500 | ✓ |
| 512 | 100.00% | 100% | 0 / 500 | ✓ |
| 1024 | 99.40% | 100% | 3 / 500 | ✓ |

## PCC tests

Building-block and layer-level comparisons vs HuggingFace.

### Decoder layer (prefill + decode)

Layer-0 `TtDecoderLayer` vs HF `Ministral3DecoderLayer` with **pretrained layer-0 weights**.
Inputs are random hidden states (not token IDs). Shared helpers: `tests/decoder_pcc_common.py`.
Weight cache: `generated/ttnn/devstral2_123B_instruct/weight_cache/…/layers_88/seq_262144/`.

| Setting | Value |
|---------|--------|
| Layer | 0 only |
| Batch size | 1 |
| PCC threshold | ≥ 0.99 |
| TT `max_seq_len` / KV budget | `262144` (shared weight cache; not swept) |

**Decode** — `tests/test_decoder.py`: 10 decode steps at positions 0 … 9.

```sh
pytest models/experimental/devstral2_123B_instruct/tests/test_decoder.py -v
```

**Prefill** — `tests/test_decoder_prefill.py`: sweeps input seq length **32 … 262144** with
128-token chunked prefill; TT layer rebuilds per length, weight cache is reused.

| Test | Input lengths |
|------|---------------|
| `test_decoder_prefill_pcc_sanity` | 32, 128 |
| `test_decoder_prefill_pcc_sweep` | 32 … 262144 (14 points) |

```sh
pytest models/experimental/devstral2_123B_instruct/tests/test_decoder_prefill.py -k sanity -v
pytest models/experimental/devstral2_123B_instruct/tests/test_decoder_prefill.py -k sweep -v
```

### Other PCC tests

```sh
pytest models/experimental/devstral2_123B_instruct/tests/ -k pcc
pytest models/experimental/devstral2_123B_instruct/tests/test_ministralattn.py -k pcc
```

**Full-model hidden-state PCC:** `tests/test_ministral3_full_model.py` — **0.99 PCC** on backbone
hidden states after 128-token prefill + one decode step (no `lm_head`).

## Performance

**End-to-end (88-layer)** — TTFT, prefill tok/s, steady-state and end-to-end decode tok/s/user:

```sh
pytest models/experimental/devstral2_123B_instruct/tests/perf/test_e2e_performant.py -k L88
```

**Single-layer wall-clock perf:**

```sh
pytest models/experimental/devstral2_123B_instruct/tests/perf/test_perf.py
```

**Single-layer device perf (Tracy, prefill 128 + decode 1):**

```sh
pytest models/experimental/devstral2_123B_instruct/tests/perf/test_device_perf_single_layer_prefill_decode.py \
    -v -m models_device_performance_bare_metal
```

Analyze ops CSV with ``tt-perf-report`` using ``--start-signpost start --end-signpost stop``.

### Results

``test_e2e_performant.py -k L88``, 2CQ decode trace on, 128-token prefill, 32 decode iters:

| Test | System | Mesh | TTFT (ms) | Prefill tok/s | Steady-state tok/s/user | End-to-end tok/s/user |
|:-----|:-------|:-----|----------:|--------------:|------------------------:|----------------------:|
| E2E L88 (2CQ traced) | BH Loudbox | 1×8 | 102.2 | 1262 | 14.02 | 8.81 |

- **TTFT** — prefill trace replay after capture.
- **Prefill tok/s** — ``prompt_len / prefill_trace_replay_time`` (compile excluded).
- **Steady-state tok/s/user** — ``decode_iters / decode_trace_replay_total``.
- **End-to-end tok/s/user** — includes TTFT and decode compile/capture.

## Repository layout


```
devstral2_123B_instruct/
├── README.md
├── demo/
│   ├── decode_trace_2cq.py       # 2CQ decode-trace helpers
│   ├── on_device_sampling.py     # On-device greedy sampling
│   ├── text_demo.py              # Text LM demo (CI)
│   └── tt_demo_agent.py          # Interactive coding agent REPL
├── reference/
│   ├── devstral2_123b_inference.py
│   └── hf_reference_loader.py    # Shared HF load path
├── tests/
│   ├── _devstral_weights.py      # HF download / dequant helpers
│   ├── decoder_pcc_common.py
│   ├── logit_pcc_common.py
│   ├── model_test_helpers.py
│   ├── test_decoder.py
│   ├── test_decoder_prefill.py
│   ├── test_ministral3_full_model.py
│   ├── test_ministralattn.py
│   ├── test_ministralmlp.py
│   ├── test_ministralrmsnorm.py
│   ├── test_ministral_rotaryemb.py
│   ├── test_model_logit_pcc.py
│   ├── test_model_token_match.py
│   └── perf/
│       ├── test_e2e_performant.py
│       ├── test_perf.py
│       ├── test_device_perf_single_layer_prefill_decode.py
│       └── test_profile_single_layer_prefill_decode.py
└── tt/
    ├── model_args.py
    ├── weight_loading.py         # FP8 → bf16 upload + disk cache
    ├── tt_ministral3_model.py    # Top-level model
    ├── tt_ministral3_decoder_layer.py
    ├── tt_ministralattn.py
    ├── tt_ministralmlp.py
    ├── tt_ministralrmsnorm.py
    ├── tt_ministral_rotary_emb.py
    ├── ccl_helpers.py
    └── mem_config.py
```

## Model and limits

| Item | Value |
|------|--------|
| HF `max_position_embeddings` | 262,144 |
| **`max_batch_size`** | **1** (single-user only) |
| Default `max_seq_len` (TT KV) | **262144** (`DEVSTRAL2_MIN_MAX_SEQ_LEN`) |
| Hidden-state PCC (88 layers) | **0.99** |
| Logit PCC (88 layers) | **0.90** |
| Token match | top-1 ≥ **96%**, top-5 ≥ **99%** |

On-device sampling pads decode logits to a **32-row tile** internally — only row 0 is the active user.
Multi-user / `batch_size > 1` is not validated on this mesh configuration.
