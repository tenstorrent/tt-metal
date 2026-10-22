<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Common prefill runner: Llama-3.1-8B

Run the common prefill runner and producer on one Blackhole Galaxy (SC1).
The adapter uses all 32 layers on an SP4/TP8 mesh. Acceptance uses a BFP8 KV cache
with two allocated slots, one active producer slot, and 1024-token prefill chunks.
The local default is 2048 tokens; CI selects 32768 tokens.

## What the tests prove

| Check | Evidence |
| --- | --- |
| Runtime contracts | The engine owns the input/cache. Compile warmup leaves no logical user prefix. All 32 layer acknowledgements follow device synchronization. Failed work cannot acknowledge success. |
| `test_kv_cache_table.py` | Every one of 65,536 synthetic cache pages maps to the expected slot, layer, head and token position. Serialization preserves addresses and ownership. Real QKV/RoPE writes are readable through the table. |
| `test_producer_runner_pcc[llama31]` | The producer sends the saved token IDs through the common runner. K and V for all 32 layers meet PCC ≥ 0.99 against the saved reference. Uses the common completion drain before readback and the existing fixture teardown. |
| `run_multirank_pcc.sh llama31 sc1` | The standard launcher discovers the Galaxy, publishes the address table, checks the rank verdict for the configured active slots, and uses the shared shutdown path. |

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
bash models/demos/llama_3p1_8b_d_p/scripts/ci/run_prefill_acceptance.sh
```

The stage runs the focused host checks, prepares the reference, then runs the
table and standard-launcher tests sequentially. By default it reuses
`/mnt/models/llama-3.1-8b-prefill-cache/golden/llama31_8b_kv_2048_32L`.
`PREFILL_TRACE_DIR` overrides that path; point it at the directory containing
`metadata.json`, not its `kv_cache` subdirectory. Saved traces may be BF16 or FP32.
The reader honors `rope_frame: meta` and HF's `key_rotary_frame: hf_half_split`.

A longer trace supplies the requested token/KV prefix. Missing files or
insufficient tokens/cache rows trigger independent HF generation from *Pride
and Prejudice*. Incompatible tensor shapes or key-frame metadata fail explicitly.
New traces are saved separately under `PREFILL_GOLDEN_CACHE_DIR` (default:
`$PREFILL_SUMMARIES/llama31_golden`), and `trace_paths.json` records them for reuse.
Set this cache directory to persistent writable storage to retain generated
fallbacks between jobs. Shared input goldens are never overwritten. The prompt
file must contain enough text when generating longer references.

The reuse policy is shared in `common/prefill/runners/trace_utils.py` and is also
used by the common prompt-driven producer/runner test. Other models supply their
own reference generator and, where needed, cache-format validation.

The direct pytest entry point remains available for local debugging:

```bash
export PREFILL_MODEL=llama_3p1_8b
export PREFILL_TRACE_DIR=/path/to/golden
python3 -m pytest -v --tt-arch blackhole \
  'models/demos/common/prefill/tests/test_producer_runner_e2e.py::test_producer_runner_pcc[llama31]'
```

To exercise both allocated slots, set `PREFILL_PRODUCER_NUM_USERS=2`. One trace
can be reused for both; use `PREFILL_PRODUCER_SLOT_TRACES=/path/A,/path/B` with
different prompts to detect swapped slot identities. The synthetic table tests
continue to check both allocated slots even when the model accuracy test uses one.

The direct fixture uses the scenario in `tests/utils.py`; the standard launcher
sources the model's `scripts/ci/runner_config.sh`. Both derive shape defaults from
the model manifest and use the same reference validator. The direct test starts
standalone child processes; the standard launcher also exercises tt-run discovery
and MPI placement. Both use the common completion and teardown behavior.

For a local installation, set `TTRUN_DIR` to the directory containing its MPI
`hostfile`; the default is `/etc/ttop`. `PREFILL_TCP_INTERFACE` selects the TCP
interface used by the standard launcher; the default is `ens5f0np0`.
The standard launcher prints verdicts and log summaries, then removes its temporary
run directory using the shared cleanup path.

In **Blaze Models Prefill tests**, select `llama31_prefill_runner`. This selects
one SC1 allocation for the complete acceptance stage at 32K. To match that CI case locally,
add these settings before running the acceptance script:

```bash
export PREFILL_MAX_SEQ_LEN=32768
export PREFILL_PRODUCER_NUM_USERS=1
export PREFILL_PRODUCER_SLOT_TRACES=/mnt/models/llama-3.1-8b-prefill-cache/golden/llama31_8b_kv_131072_32L
```

This reads the first 32K token IDs and cache rows from the saved 128K reference.
Reference validation runs on the MPI worker, where the model's Python dependencies are installed.

## Address-table contract

The table contains 16 configurations, ordered `k_h0` through `k_h7`, then
`v_h0` through `v_h7`. Each describes 32 layers, two slots and the configured token capacity.
A page contains 32 tokens × 128 head elements: four BFP8 tiles, each 1088 bytes,
for a total of 4352 bytes. K is stored after RoPE in Meta-interleaved order.

Each TP column owns one KV head. Each SP row owns a 256-token stripe within
each 1024-token input chunk. At the default 2K capacity, each device's local cache has shape
`[64, 1, 512, 128]`; its first dimension is `slot * 32 + layer`.
The table resolves logical `(config, slot, layer, token position)` to a live
NoC address and owning fabric device. Addresses belong to the current process
and allocation; a receiver must use the freshly published table.

The runtime waits for the whole chunk to finish before acknowledging its
layers. Acknowledgements identify the worker's chunk request, not the user ID.
The producer uses the existing common aggregate completion drain before readback.
The common drain logs and returns a partial count on timeout; this PR does not add
a separate Llama completion verdict or change that shared policy. The counter does
not expose layer/request identities. Runtime contract tests separately check the
callback identities and ordering.

The receiver must also agree on configuration names/order, dtype and tile
format, K ordering, slot identity, valid token range, completion protocol and
allocation lifetime. A matching table alone does not establish receiver
compatibility.

## Follow-up capacity gates

Repeat the focused runner acceptance at 4K, 8K, 16K, 32K and 64K after the 2K
case passes. Set `PREFILL_MAX_SEQ_LEN` to select the acceptance capacity; the shared fixture
derives the chunk count and the recipe reuses a sufficient reference or generates one. The table's
synthetic address test remains the focused 2K case. The standard launcher passed
all-layer K/V PCC at 2K, 4K, 8K, 16K, 32K and 64K with one active slot, using
prefixes of the saved 128K reference. Twelve consecutive 4K runs also passed
without resets between runs. These checks use the unchanged PCC threshold of
0.99. Two allocated slots remain a model constraint.

## Real loopback migration remains a separate gate

The tests above cover Gate 1 of the common `PREFILL_MIGRATION_TESTING.md` guide.
Gate 2 requires the real endpoint/workers and `migration_driver` with destination
byte and golden verification. It has not been run on this branch. One source and
one destination can use a proposed slot 0 -> 1 test; two independent source/destination
pairs require four slots and a model-capacity change. Prefill completion counters
are not evidence of completed migration operations.
