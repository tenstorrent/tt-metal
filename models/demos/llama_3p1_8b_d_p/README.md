<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# Llama-3.1-8B disaggregated prefill

Llama-3.1-8B-Instruct prefill runs on one Blackhole Galaxy with SP=4 and TP=8.
The shared prefill runner owns input transport, cache lifetime, table publication,
completion acknowledgements and shutdown. The model uses 32 layers, two slots,
1,024-token compute chunks and a BFP8 K/V cache.

Related issue: [tt-blaze#4138](https://github.com/tenstorrent/tt-blaze/issues/4138).

## Run and verify

- [SC1 runner guide and cache-table contract](docs/runner-integration.md): independent
  golden traces, live table readback, shared runner/producer PCC and tt-run commands.
- [Model numerical validation](docs/validation-2k.md): saved full-model and component evidence.
- [Model performance](docs/performance-prefill.md): saved eager full-model measurements.
- [Current integration milestones](ROADMAP.md).
- [Longer-context scope](LONG_CONTEXT_PLAN.md).

The initial shared-runner acceptance case is 2K per slot. A model performance result
at a larger length does not establish shared-runner acceptance at that length.

## Source map

| Path | Purpose |
|---|---|
| `tt/` | Attention, MLP, decoder layers and full prefill model |
| `tt/runners/adapters/llama_3p1_8b.py` | Import-light adapter registered with common prefill |
| `tt/tt_prefill_runtime.py` | Eager chunk runtime with borrowed cache and post-sync acknowledgements |
| `tt/runners/kv_chunk_table.py` | Source cache addresses, owners and protobuf export |
| `tests/unit/`, `tests/full_model/` | Numerical model checks |
| `tests/test_kv_cache_table.py` | Independent live table readback and protobuf checks |
| `tests/test_prefill_runtime.py` | Compact runtime and adapter regressions |

The model implements Llama-3.1 RoPE, RMSNorm, full causal grouped-query attention
and bias-free dense SiLU SwiGLU. Cache keys use adjacent Meta rotary pairs.
