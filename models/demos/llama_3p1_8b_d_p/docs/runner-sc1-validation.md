<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# SC1 runner acceptance results

The shared Llama prefill runner and producer passed the 2K/two-slot acceptance
checks on one Blackhole Galaxy on 2026-09-22. Reproduce them with the
[runner guide](runner-integration.md).

## Configuration

- Llama-3.1-8B-Instruct, all 32 layers, SP=4 and TP=8.
- Two independent slots, each with 2,048 tokens from a different book passage.
- 1,024-token compute chunks; four chunks across the two requests.
- BFP8 K/V, 16 head/cache configurations and 4,352-byte address-table pages.
- Eager execution, host completion acknowledgements after synchronization.
- Independent FP32 Hugging Face K/V reference for every layer and both slots.

## Passed checks

| Check | Result | Evidence established |
|---|---|---|
| Compact host regressions | 29 passed; 50.21 s | GQA readback, runtime forwarding/completion/failure handling and shared summary compatibility |
| `tests/test_kv_cache_table.py` | 3 passed; 127.35 s | Exact readback of all 65,536 synthetic pages; protobuf addresses/owners; real QKV/RoPE writer readback |
| `test_producer_runner_pcc[llama31_2k_two_slots]` | 1 passed; 218.20 s | Real H2D requests, two complete 2K slots, all-layer table-based golden comparison and clean shutdown |
| `run_multirank_pcc.sh llama31 sc1` | Exit 0; 1/1 rank verdict passed | Standard tt-run/MPI launch, published table, both slots checked and successful producer/runner exits |

The times above are test durations, including setup and verification. They are
not model performance measurements. The direct test used an existing kernel
cache; an earlier cold run also passed in 418.70 s.

The standard launcher's retained verdict records these minima across all layers
and both slots:

| Cache | Minimum PCC | Required minimum | Result |
|---|---:|---:|---|
| K | 0.9998040037 | 0.99 | Pass |
| V | 0.9992389318 | 0.99 | Pass |

Both 2K golden traces were generated independently on CPU. Model loading took
4.95 s; the two forward passes took 41.54 s and 38.54 s. These are reference
forward times, not device times or total file-generation times.

## What this proves

The runner accepts requests, fills separate caches, publishes valid addresses,
signals completion and shuts down. A separate producer process can use that table
to retrieve K/V that agrees with the independent reference. Exact synthetic page
checks complement PCC by detecting wrong placement even when values correlate.

This SC1 test uses the shared runner's mock migration mode. It validates the
prefill source contract without requiring a modified tt-d-gen checkout. Native
transfer to another Galaxy and a decode consumer are separate integration gates.

The [4K–64K follow-up list](runner-capacity-plan.md) is ready. Those larger
shared-runner cases have not run on this branch. The model numerical and
performance suites were preserved; their earlier results remain separate from
this runner acceptance.
