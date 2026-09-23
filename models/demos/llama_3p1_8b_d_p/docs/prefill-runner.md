<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Common prefill runner: Llama-3.1-8B

Run the common prefill runner and producer on one Blackhole Galaxy (SC1).
The adapter uses all 32 layers on an SP4/TP8 mesh. The acceptance case uses a
BFP8 KV cache, two independent slots, 2048 tokens per slot, and 1024-token
prefill chunks.

## What the tests prove

| Check | Evidence |
| --- | --- |
| Runtime contracts | The engine owns the input/cache. Compile warmup leaves no logical user prefix. All 32 layer acknowledgements follow device synchronization. Failed work cannot acknowledge success. |
| `test_kv_cache_table.py` | Every one of 65,536 synthetic cache pages maps to the expected slot, layer, head and token position. Serialization preserves addresses and ownership. Real QKV/RoPE writes are readable through the table. |
| `test_producer_runner_pcc[llama31_2k_two_slots]` | The producer sends two different book prompts through the common runner. Both slots' K and V for all 32 layers meet PCC ≥ 0.99 against independent FP32 Hugging Face traces. The completion counter must total 128, and the runner shuts down cleanly. |
| `run_multirank_pcc.sh llama31 sc1` | The standard launcher discovers the Galaxy, publishes the address table, checks the populated rank verdict for both slots, and verifies shutdown. |

The SC1 tests validate the prefill source and table readback in a separate
producer process. They do not transfer the cache to a second Galaxy or validate
a decode consumer. Those need a compatible receiver and an integration test.

## Run the complete acceptance stage

Use a built tt-metal checkout, its matching installed TTNN wheel, the normal
MPI/tt-run host configuration, and an allocated Blackhole Galaxy. Run from the
repository root. The checkpoint must contain its config, tokenizer, safetensors
index and all weight shards.

```bash
export TT_METAL_HOME="$PWD"
export PREFILL_HF_MODEL=/mnt/models/meta-llama/Llama-3.1-8B-Instruct
export LLAMA31_8B_CHECKPOINT="$PREFILL_HF_MODEL"
export PREFILL_SUMMARIES="$PWD/generated/llama31-runner"
bash models/demos/common/prefill/runners/ci/run_llama31_sc1_acceptance.sh
```

The stage runs the focused host checks, generates two independent 2K reference
traces from *Pride and Prejudice*, then runs the table, direct producer/runner,
and standard-launcher tests sequentially. It returns nonzero on failure.
The two runner checks reuse the same newly generated reference traces.

For a local installation, set `TTRUN_DIR` to the directory containing its MPI
`hostfile`; the default is `/etc/ttop`. `PREFILL_TCP_INTERFACE` selects the TCP
interface used by the standard launcher; the default is `ens5f0np0`.

In **Blaze Models Prefill tests**, select `llama31_prefill_runner`. This selects
one SC1 allocation for the complete acceptance stage.

## Address-table contract

The table contains 16 configurations, ordered `k_h0` through `k_h7`, then
`v_h0` through `v_h7`. Each describes 32 layers, two slots and 2048 positions.
A page contains 32 tokens × 128 head elements: four BFP8 tiles, each 1088 bytes,
for a total of 4352 bytes. K is stored after RoPE in Meta-interleaved order.

Each TP column owns one KV head. Each SP row owns a 256-token stripe within
each 1024-token input chunk. Each device's local cache has shape
`[64, 1, 512, 128]`; its first dimension is `slot * 32 + layer`.
The table resolves logical `(config, slot, layer, token position)` to a live
NoC address and owning fabric device. Addresses belong to the current process
and allocation; a receiver must use the freshly published table.

The runtime waits for the whole chunk to finish before acknowledging its
layers. Acknowledgements identify the worker's chunk request, not the user ID.
The producer checks an aggregate completion counter. That counter does not
carry layer/request identities; a duplicate and a missing event could cancel
in its total. Runtime contract tests separately check the callback identities
and ordering.

The receiver must also agree on configuration names/order, dtype and tile
format, K ordering, slot identity, valid token range, completion protocol and
allocation lifetime. A matching table alone does not establish receiver
compatibility.

## Follow-up capacity gates

Repeat the focused runner acceptance at 4K, 8K, 16K, 32K and 64K after the 2K
case passes. Each capacity needs matching allocation, manifest, chunk count
and references. The current SC1 acceptance command is deliberately a fixed
2K gate; changing only an environment variable does not validate larger
capacities.
