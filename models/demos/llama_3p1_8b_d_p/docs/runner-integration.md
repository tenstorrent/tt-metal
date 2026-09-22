<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# Llama prefill runner: SC1 acceptance

The shared runner owns the mesh, H2D input service, cache, address-table publication,
layer acknowledgements and shutdown. The Llama adapter provides the model and its
cache layout. The shared producer sends two distinct prompts, waits for completion,
and reads the cache through the exported table from a separate process.

The initial acceptance configuration is one Blackhole Galaxy, SP=4, TP=8,
32 layers, two slots, 2,048 tokens per slot and 1,024-token compute chunks.
K and V use `bfloat8_b`. The runtime uses eager execution and sends all 32 layer
acknowledgements after a chunk completes and device synchronization succeeds.

## Setup and independent reference

Use a built tt-metal environment on an allocated Blackhole Galaxy. The Python
bindings, libraries and kernels must match the checkout. Start in the repository
root. Point `PROMPT_FILE` at a local book excerpt containing at least 4,094 tokens.
The generator uses different consecutive passages for the two slots.

```bash
export TT_METAL_HOME="$PWD"
export PYTHONPATH="$PWD:$PWD/ttnn"
export PREFILL_HF_MODEL=/mnt/models/meta-llama/Llama-3.1-8B-Instruct
export PREFILL_MODEL=llama_3p1_8b
export GOLDEN_DIR=/path/to/new/llama-golden-2k
export PROMPT_FILE=/path/to/book.txt

python models/demos/llama_3p1_8b_d_p/scripts/generate_prefill_trace.py \
  --checkpoint "$PREFILL_HF_MODEL" --prompt-file "$PROMPT_FILE" \
  --output-dir "$GOLDEN_DIR" --seq-len 2048 --threads 8

export PREFILL_PRODUCER_SLOT_TRACES="$GOLDEN_DIR/slot0,$GOLDEN_DIR/slot1"
```

The CPU reference uses Hugging Face FP32 weights and SDPA. It saves post-RoPE K
and V from all 32 layers, plus exact token IDs and reference timing. It does not
read device output. Each trace contains:

```text
metadata.json                         # token_ids, dimensions, frame, timing
kv_cache/layer_0.safetensors          # key_cache_layer_0, value_cache_layer_0
...
kv_cache/layer_31.safetensors
```

Each tensor has shape `[1, 8, 2048, 128]`. Keys use HF half-split rotary coordinates.
The producer converts K to the adjacent-pair device coordinates. V needs no
coordinate conversion.

## Required checks

Run these commands sequentially: only one process group may own the Galaxy.

| Check | What a pass establishes |
|---|---|
| `test_prefill_runtime.py` | Cache ownership, request forwarding, completion order and failure handling |
| `test_kv_cache_table.py` | Every synthetic K/V page agrees with independent live tensor placement; protobuf preserves addresses and owners; real layer-produced K/V is readable through the table |
| Shared `test_producer_runner_pcc[llama31_2k_two_slots]` | Real H2D requests, two interleaved slots, all 32 layers, published table/device map, golden KV PCC and graceful shutdown |
| `run_multirank_pcc.sh llama31 sc1` | The same producer/runner contract works under the standard tt-run/MPI launcher and passes the per-rank verdict gate |

```bash
python -m pytest -q models/demos/llama_3p1_8b_d_p/tests/test_prefill_runtime.py

python -m pytest -s -q models/demos/llama_3p1_8b_d_p/tests/test_kv_cache_table.py

python -m pytest -s -q \
  'models/demos/common/prefill/tests/test_producer_runner_e2e.py::test_producer_runner_pcc[llama31_2k_two_slots]'

# CI normally supplies /etc/ttop/hostfile. For an allocated standalone host:
export TTRUN_DIR="$PWD/generated/llama-sc1-hosts"
mkdir -p "$TTRUN_DIR"
hostname > "$TTRUN_DIR/hostfile"
export PREFILL_SUMMARIES="$PWD/generated/llama-sc1-results"
bash models/demos/common/prefill/runners/ci/run_multirank_pcc.sh llama31 sc1
```

The launcher retains the table, per-rank verdicts and logs under the reported
evidence directory. Require a zero exit status and comparisons for both slots.
A skipped test, empty comparison, missing layer, missing owner or missing verdict
is not acceptance. The initial golden PCC threshold is 0.99 for each layer's K
and V; the exact table test also detects byte placement errors independent of PCC.

These tests use the upstream TTNN address-table and readback APIs. They do not
require a modified tt-d-gen checkout. The shared runner's `PREFILL_MOCK_MIGRATION=1`
mode publishes source metadata without attaching a native migration manager.
A pass establishes the prefill integration and source-table contract. A separate
native transfer/decode integration test remains necessary for a deployed system.

## KV-chunk table contract

| Field | Value / meaning |
|---|---|
| Config order | `k_h0` … `k_h7`, then `v_h0` … `v_h7` |
| Layer | Global index 0–31 |
| Slot | 0 or 1; separate cache storage |
| Sequence coordinate | Absolute token position in that slot |
| Chunk | 32 tokens × 128 values for one KV head |
| Bytes per chunk | 4 BFP8 tiles × 1,088 bytes = 4,352 bytes, including exponents |
| Chunk owner | One fabric node; TP column equals KV-head index |
| SP placement | Consecutive 256-token sections of each 1,024-token compute chunk go to SP rows 0–3 |
| Address | Encoded DRAM bank and byte offset, valid while the runner owns the cache allocation |
| K coordinates | Adjacent Meta rotary pairs after Llama-3.1 RoPE |
| V coordinates | Unrotated head values |

The receiver must also agree on model/checkpoint, dtype/tile format, rotary frame,
layer/head identities, slot ownership, valid token length, completion signals and
cache lifetime. A matching table alone cannot establish those semantic contracts.

## Larger-capacity follow-up

After the 2K acceptance passes, list separate 4K, 8K, 16K, 32K and 64K runs.
Each must fill both slots to its capacity, read boundary pages through the table,
confirm slot isolation, and finish with clean shutdown. Keep the 1K compute chunk.
Reuse the proven small host/runtime regressions instead of rerunning a large CPU
edge-case matrix for every length. Full all-layer golden KV comparisons at each
larger length are a separate, optional accuracy cost.
