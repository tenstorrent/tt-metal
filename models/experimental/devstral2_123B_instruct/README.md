# Devstral-2-123B-Instruct (`experimental`)

Package path: `models/experimental/devstral2_123B_instruct/`

## Platforms

Blackhole Loudbox (1×8 mesh).

## Introduction

This folder contains an experimental Tenstorrent (`ttnn`) port of **Mistral [Devstral-2-123B-Instruct-2512](https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512)** (Ministral3 text stack). PCC tests compare subgraphs and the full model against HuggingFace references (see [PCC tests](#pcc-tests)).

**Maximum context length:** The HF checkpoint advertises very long context (YaRN / RoPE tables up to 262144 positions). On Blackhole Loudbox (1×8), the working KV budget (`max_seq_len`) is what you can actually run. **Up to 96K tokens** has been verified end-to-end on this mesh. The KV floor default is **262144** (`DEVSTRAL2_MIN_MAX_SEQ_LEN` / `--max-seq-len` in `text_demo.py` and `tt_demo_agent.py`).

**Traced prefill/decode** with **2CQ** decode staging is on by default (`DEVSTRAL2_TRACE_PREFILL=1`, `DEVSTRAL2_DECODE_TRACE_2CQ=1`). Set either to `0` to disable.

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- Python packages (install into your tt-metal environment):

  ```sh
  pip install git+https://github.com/huggingface/transformers
  ```



## PCC tests

### Decoder layer (prefill + decode)

Layer-0 PCC tests compare `TtDecoderLayer` against HuggingFace `Ministral3DecoderLayer` using **pretrained layer-0 weights** from the Hub checkpoint. Inputs are **random hidden states** (not token IDs), matching the tt-transformers decoder block test pattern.

Shared helpers live in `tests/decoder_pcc_common.py`. Both tests reuse the on-disk weight cache at **`seq_262144`** (same as `text_demo.py` / teacher-forced sweep), under:

`generated/ttnn/devstral2_123B_instruct/weight_cache/…/layers_88/seq_262144/`

Override the cache key with `DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN` (default `262144`).

| Setting | Value |
|---------|--------|
| Layer | 0 only |
| Batch size | 1 |
| PCC threshold | ≥ 0.99 |
| TT `max_seq_len` / KV budget | `262144` (shared weight cache; not swept) |
| Weights | Pretrained Hub checkpoint (FP8 → bf16); HF and TT share the same `state_dict` |

**Decode** — `tests/test_decoder.py`

- One token per step, random `[1, 1, hidden_size]` activations (`hidden_size=12288`).
- **10** decode steps at positions **0 … 9** (RoPE at increasing positions, paged KV read/write).
- No seq-length sweep (input shape is fixed; KV budget comes from the shared 256K cache).

```sh
pytest models/experimental/devstral2_123B_instruct/tests/test_decoder.py -v
```

**Prefill** — `tests/test_decoder_prefill.py`

- Sweeps **input** seq length (powers of two **32 … 262144**); TT layer init and weight cache stay at **seq_262144**.
- **128-token chunked prefill** on TT and HF (`kv_block_size=128`): random activations per chunk, HF via incremental `DynamicCache` forwards. Host memory stays **O(chunk)**, not O(seq_len).

| Test | Input lengths | Purpose |
|------|---------------|---------|
| `test_decoder_prefill_pcc_sanity` | 32, 128 | CI gate (`-k sanity`) |
| `test_decoder_prefill_pcc_sweep` | 32 … 262144 (14 points) | Full sweep (`-k sweep`); one mesh session per test, fresh TT KV per length, shared weight cache |

```sh
# CI / quick gate
pytest models/experimental/devstral2_123B_instruct/tests/test_decoder_prefill.py -k sanity -v

# Full seq-length sweep
pytest models/experimental/devstral2_123B_instruct/tests/test_decoder_prefill.py -k sweep -v
```

**Host memory:** Prefill PCC runs HF and TT in **128-token chunks** with per-chunk PCC ≥ 0.99, so host RAM stays bounded on long sweep points (32K … 262144). Each seq length rebuilds the TT layer (KV starts empty) but reuses the **`seq_262144`** on-disk weight cache.

### Other PCC tests

Building-block and full-model PCC (from repo root):

```sh
# All tests whose names match "pcc"
pytest models/experimental/devstral2_123B_instruct/tests/ -k pcc

# Example: attention only
pytest models/experimental/devstral2_123B_instruct/tests/test_ministralattn.py -k pcc
```

**Full-model hidden-state PCC:** One decode step after 128-token prefill in `tests/test_ministral3_full_model.py` — **0.99 PCC** on backbone hidden states (no `lm_head`).

See [Full-model logit PCC](#full-model-logit-pcc) for the 88-layer logit comparison test.

## Demos

Run from repo root on Blackhole Loudbox.

| Demo | Script | Description | Command |
|------|--------|-------------|---------|
| Text LM | `demo/text_demo.py` | Text-only TT prefill/decode + LM head. Override prompt with `DEVSTRAL2_PROMPT`. | `pytest models/experimental/devstral2_123B_instruct/demo/text_demo.py` |
| Interactive agent | `demo/tt_demo_agent.py` | Multi-turn coding REPL on TT with **workspace-only** file tools (see below). | `python models/experimental/devstral2_123B_instruct/demo/tt_demo_agent.py --mesh-device T3K` |

### Interactive agent demo (`tt_demo_agent.py`)

Local REPL that runs Devstral-2-123B on the TT mesh and lets the model call **tools** across
multiple turns (KV prefix cache for the system/tool rules, traced decode, optional 2CQ). Inference
is on-device; tool execution is ordinary Python on the host.

**Run** (point `--workspace-root` at a directory you are willing to let the model read/write):

```sh
python models/experimental/devstral2_123B_instruct/demo/tt_demo_agent.py \
  --mesh-device T3K \
  --workspace-root /path/to/your/repo
```

Useful flags: `--num-layers` (partial model for bring-up), `--max-seq-len`, `--max-context-tokens`,
`--verbose` (KV/prefill/decode logging). Type `quit` / `exit` to stop; `/clear` resets chat and
todo state.

**Not in CI:** Blackhole pipelines run `text_demo.py` only. The agent is for interactive/local use.

#### Tools the agent can use

All file paths are resolved under `--workspace-root`; paths that escape that directory are rejected.

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

Rough equivalents: `inspect_codebase` ≈ `ls`, `read_file` ≈ `cat`, `grep` ≈ ripgrep — but only
inside the workspace and without spawning a shell.

#### What the agent cannot do

These tools are **intentionally not implemented** (calls return `Unknown tool`):

| Removed capability | Examples that do **not** work |
|--------------------|-------------------------------|
| **Shell / terminal** | `ls`, `ps`, `top`, `python script.py`, `git`, `make`, `curl` |
| **Delegated shell** | `delegate_task` |
| **Web fetch** | Download arbitrary URLs |
| **Web search** | DuckDuckGo or other search APIs |

The model may still **`write_file`** a script into the workspace, but the demo **does not execute**
it. Run builds/tests yourself outside the agent.

#### Why tools are restricted

Earlier prototypes exposed `terminal` / `bash` and network tools. That gives the model **full
user-level command execution** and **arbitrary HTTP egress** whenever it emits a tool call — without
per-command approval. Risks include:

- **Prompt injection** in repo content (“run this shell command…”).
- **Mistakes** (`rm -rf`, bad `git` commands) with your normal OS permissions (sudo not required).
- **Secret access** — anything readable by your user (`~/.ssh`, `.env`, tokens in the environment).
- **Data exfiltration** — upload files or hit internal/metadata URLs via `curl` / `web_fetch`.

File-only tools narrow the blast radius to a directory you choose (`--workspace-root`). That is
enough for coding assistance (read/edit/search the tree) while avoiding “model as remote shell”.
This matches the hardened Devstral-Small agent demo pattern (shell disabled; no web tools on the
123B path).

**Operational guidance:** use a dedicated clone or scratch directory as `--workspace-root`, not your
home directory or a repo with secrets. The agent demo is experimental, not a production coding agent.

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



## Full-model logit PCC

End-to-end **logit PCC** for the full 88-layer `TtMinistral3ForCausalLM` vs HuggingFace
`AutoModelForCausalLM` (tt-transformers `test_model.py` pattern). Compares **raw logits**
(full vocab) — not token IDs. After each logits check, decode advances with the **HF greedy**
token (temperature=0 argmax) on both HF and TT.

**Test file:** `tests/test_model_logit_pcc.py`
**Shared helpers:** `tests/logit_pcc_common.py`

| Test | Prefill lengths | Purpose |
|------|-----------------|--------|
| `test_model_logit_pcc_sanity` | 32, 128 | CI gate (`-k sanity`) |
| `test_model_logit_pcc_sweep` | 32 … 262144 (14 points) | Full sweep (`-k sweep`) |

**Per sweep point:**

- Tale of Two Cities tokens (tiled when a longer stream is needed).
- **128-token chunked prefill** on HF and TT; HF uses incremental `DynamicCache` (O(chunk) host memory).
- PCC ≥ **0.99** on last-prefill logits and **10** decode steps (`DECODE_GENERATION_LENGTH`).
- TT model built once per mesh; tiled weights load from the shared on-disk cache at
  **`seq_262144`** (override with `DEVSTRAL2_WEIGHT_CACHE_SEQ_LEN`). Runtime
  `model_max_seq_len` is fixed for the whole sweep (worst-case prefill + decode budget).

**Run:**

```sh
# CI / quick gate (32, 128)
pytest models/experimental/devstral2_123B_instruct/tests/test_model_logit_pcc.py -k sanity -v

# Full 14-point sweep
pytest models/experimental/devstral2_123B_instruct/tests/test_model_logit_pcc.py -k sweep -v
```

**Timeout:** Sanity (2 points) ≈ **30 min** on BH Loudbox; full sweep default pytest cap **12 h**
(override with `DEVSTRAL2_LOGIT_PCC_SWEEP_TIMEOUT_SEC`; observed ~10 h wall with warm weight cache).

**HF reference:** Loads the full checkpoint once per sweep (same path as `test_ministral3_full_model.py`).
Runs on **CPU with disk offload** when no CUDA GPU is present (`device_map=cpu`, `offload_folder`).
Override with `DEVSTRAL2_HF_DEVICE` / `DEVSTRAL2_HF_DEVICE_MAP` / `DEVSTRAL2_HF_OFFLOAD_FOLDER`.
Requires sufficient host DRAM/disk offload space and `HF_TOKEN` when gated.

**vs token match:** Logit PCC checks numerical agreement on logits (PCC). Token match
(`test_model_token_match.py`) checks top-1 / top-5 token accuracy over **500** teacher-forced
eval tokens per prefill length.

## Model token match (teacher-forced)

End-to-end **teacher-forced** top-1 / top-5 token accuracy vs HuggingFace over a range of prefill
context lengths. Eval uses a fixed **500 tokens** after each prefill (same as tt-transformers CI
`max_generated_tokens=500` in `simple_text_demo.py`; that test uses a 1024-token `.refpt` with a
512-token prefill and stops decode at 500 steps).

**Test file:** `tests/test_model_token_match.py`

| Test | Prefill lengths | Purpose |
|------|-----------------|--------|
| `test_devstral2_teacher_forced_accuracy_sanity` | 32, 64, 128 | CI gate (`-k sanity`) before full sweep is enabled |
| `test_devstral2_teacher_forced_accuracy_sweep` | 32 … 262144 (14 points) | Full sweep (`-k sweep`) |

**Default sweep:** powers of two from 32 through 262144. Tale of Two Cities is tiled when a longer stream is needed.

**Outputs:**

`tests/teacher_forced_sweep_outputs/`

| Path | Contents |
|------|----------|
| `references/prefill_{N}_total_{M}.refpt` | Cached HF reference for that prefill + 500 eval tokens |
| `results/prefill_{N}.json` | Per-length top-1/top-5 accuracy, predictions, pass/fail |
| `results/sweep_summary_{run_id}.json` | Full sweep rollup |

**Run:**

```sh
# CI / quick gate (32, 64, 128)
pytest models/experimental/devstral2_123B_instruct/tests/test_model_token_match.py -k sanity -v

# Full 14-point sweep
pytest models/experimental/devstral2_123B_instruct/tests/test_model_token_match.py -k sweep -v
```

**CI (BH Loudbox):**

| Workflow | Test | Purpose |
|----------|------|---------|
| ``(Blackhole) e2e tests`` → ``devstral2-123b-instruct`` | ``test_model_logit_pcc.py -k sanity`` | Logit PCC gate (prefill 32/128) |
| ``(Blackhole) e2e tests`` → ``devstral2-123b-instruct`` | ``test_model_token_match.py -k sanity`` | Token match gate (prefill 32/64/128) |
| ``(Blackhole) Demo tests`` → ``devstral2-123b-instruct`` | ``demo/text_demo.py`` | End-to-end text generation smoke |

Both sanity tests run sequentially in the ``bh-lb-devstral2-123b-e2e-sanity`` e2e job (90 min timeout).

Switch e2e to ``-k sweep`` when the full seq-length teacher-forced run is ready.
Requires Devstral-2-123B weights in the MLPerf HF cache mount (`/mnt/MLPerf/huggingface` on Loudbox).

**Timeout:** Each test has its own pytest timeout budgeted for its prefill list. Full sweep
(14 points, 32 … 262144) is calibrated at ≈ **71 hours** cold on BH Loudbox; sanity (3 points) ≈ **45 min**.
Re-runs with cached ``.refpt`` files finish much sooner.

**Thresholds:** top-1 ≥ 96%, top-5 ≥ 99% (override with `DEVSTRAL2_MIN_TOP1_ACC` /
`DEVSTRAL2_MIN_TOP5_ACC`).

### Results

Teacher-forced sweep on Blackhole Loudbox (1×8), run `20260617T162901Z`
(``pytest models/experimental/devstral2_123B_instruct/tests/test_model_token_match.py -k sweep``).
Each row evaluates **500** tokens after the given prefill length.

| Prefill len | Top-1 | Top-5 | Top-1 mismatches | Pass |
|------------:|------:|------:|-----------------:|:----:|
| 32 | 97.80% | 100% | 11 / 500 | ✓ |
| 64 | 98.20% | 100% | 9 / 500 | ✓ |
| 128 | 98.80% | 100% | 6 / 500 | ✓ |
| 256 | 99.80% | 100% | 1 / 500 | ✓ |
| 512 | 100.00% | 100% | 0 / 500 | ✓ |
| 1024 | 99.40% | 100% | 3 / 500 | ✓ |

Longer prefill lengths (2048 … 262144) each take on the order of several hours to
run on BH Loudbox.

## Resources

| Path | Purpose |
|------|---------|
| **`demo/`** | Runnable demos: `text_demo.py` (TT text LM), `tt_demo_agent.py` (interactive TT agent), `decode_trace_2cq.py` (2CQ decode-trace helpers). |
| **`tt/`** | TT layer implementations (Ministral3 building blocks). |
| **`tt/tt_ministral3_model.py`** | Top-level model: embed → decoder layers → RMSNorm. |
| **`tt/tt_ministral3_decoder_layer.py`** | Decoder layer (attention + MLP + norms). |
| **`tt/weight_loading.py`** | Host → device FP8 → bf16 weight dequant and upload (shard-by-shard, disk-cached). |
| **`tests/`** | PCC and unit tests: decoder layer prefill/decode (`test_decoder_prefill.py`, `test_decoder.py`), per-op blocks (attention, MLP, norms, RoPE), full model hidden-state PCC (`test_ministral3_full_model.py`), teacher-forced logit PCC (`test_model_logit_pcc.py`), teacher-forced token match (`test_model_token_match.py`). |
| **`tests/perf/`** | Performance tests: e2e throughput (`test_e2e_performant.py`), single-layer wall-clock perf (`test_perf.py`), and single-layer prefill+decode device perf (`test_device_perf_single_layer_prefill_decode.py`, `test_profile_single_layer_prefill_decode.py`). |
| **`reference/`** | HF reference loader (`hf_reference_loader.py`) and pure PyTorch / HF inference script (`devstral2_123b_inference.py`). |

## Model and limits

| Item | Value |
|------|--------|
| HF `max_position_embeddings` | 262,144 (model-native RoPE horizon) |
| **`max_batch_size` (TT inference)** | **1** — single-user only; all demos, perf tests, and PCC tests use `max_batch_size=1` |
| Practical `max_seq_len` (TT KV cache) | Sized per run: `prompt + max_new_tokens`, floored by `DEVSTRAL2_MIN_MAX_SEQ_LEN` (default **262144**) |
| Verified context on BH Loudbox (1×8) | **Up to 96K** tokens (end-to-end text generation) |
| Full-model hidden-state PCC (88 layers) | **0.99** (`tests/test_ministral3_full_model.py`) |
| Full-model logit PCC (teacher-forced, 88 layers) | **0.99** (`tests/test_model_logit_pcc.py`) |

### Limitations

- **Batch size:** Only **batch size 1** is supported and tested (`Devstral2Args.max_batch_size=1` in `tt/model_args.py`). Demos (`text_demo.py`, `tt_demo_agent.py`), perf tests, PCC tests, and the teacher-forced sweep all run a single sequence at a time. Multi-user / `batch_size > 1` is not validated on this 123B mesh configuration. On-device sampling pads decode logits to a **32-row tile** internally (`demo/on_device_sampling.py`) for `TTSampling` — that is not multi-batch serving; only row 0 is the active user.
- **Context length:** Long context is limited by per-chip KV and RoPE allocation, not by the HF config alone. On Blackhole Loudbox (1×8), **up to 96K tokens** has been verified for end-to-end generation. Defaults use `max_seq_len=262144` (`DEVSTRAL2_MIN_MAX_SEQ_LEN` / `--max-seq-len`). Prompts longer than the configured budget need a larger cap and sufficient device memory.
