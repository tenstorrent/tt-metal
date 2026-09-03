<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 08 — Disaggregated-prefill integration

_(pending — written in P10.)_

**Inputs already gathered** (`02_SURVEY.md` rows 28-29, all citations verified): base classes at
`models/demos/common/prefill/adapter.py:104` (`PrefillModelAdapter`), `:46` (`PrefillRunParams`),
`:277` (`ADAPTER_PATHS`); template at
`models/demos/gpt_oss_d_p/tt/runners/adapters/gpt_oss.py:41` with its five runner defaults at
`:45-49` (delete `default_gate_mode` at `:50` — MoE-router-only).

**The trap, confirmed:** the producer's KV read-back is *not* adapter-dispatched. It branches on
`ADAPTER.name` in `models/demos/common/prefill/runners/prefill_producer.py:503`, with
`:507-508` → minimax, `:509-510` → gpt-oss, and `:511` falling through to the **MLA** reader
(`_read_slot_kv_and_check_pcc_mla` at `:685`). The MLA reader is wrong for Llama's plain packed K/V,
so without an added branch `G-MOCK-MIG` would silently check the wrong bytes.
`_read_slot_kv_and_check_pcc_gpt_oss` (`:534`) *is* the plain packed-K/V block-cyclic GQA reader —
add `llama31_8b_d_p` to that branch, which is legitimate because P2 committed to keeping gpt-oss's
`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32` and shard shape
(`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27`, `:87`). That edit touches **shared code** —
it needs a `DEC` and an entry here, and it should generalise the name check rather than duplicate
the function.

## Migration coverage: what Gate 1 proves, and what Gate 2 would have added

This bring-up runs the engine's **Gate 1** (mock migration) and does **not** run **Gate 2** (loopback
migration). Gate 2 needs binaries from the private `tt-llm-engine` repo, which is not available in this
environment; the full reasoning and the exact missing products are in `DEC-070`, and the one residual
gap is `R-040`. It is recorded as **out-of-scope by decision**, not as a blocked gate — the distinction
matters, because "blocked" would imply this package has untested surface that it does not.

### Covered here

| Property | By |
|---|---|
| Prefill writes correct KV for a slot | `G-MOCK-MIG` — device-less `read_dram_umd` read-back, PCC vs the golden trace |
| This package's `build_kv_chunk_table` is correct (one rank's layer slice) | `G-MOCK-MIG` — the read-back resolves every chunk through our published table, so a wrong address fails it |
| The device map is published where every device-less reader expects it | `G-MOCK-MIG` |
| The adapter satisfies the engine's contract | `G-ADAPTER` |
| Request-mode serving: H2D push, chunk schedule, per-layer ack drain | `G-REQUEST` |

### NOT covered, and what each omission means

| Not exercised | What it means in practice |
|---|---|
| The real DRAM -> transport -> DRAM copy | Whether the *engine's* byte copy lands. Model-agnostic: its default `dst-bytes` mode decodes nothing and asserts only dst == src. No Llama-specific property rides on it. |
| `MigrationLayerClient` attach, `WORKER_READY` handshake, cross-endpoint pairing | Engine/worker-side orchestration. Our runtime's only obligation is `set_layer_ack_channel`, and the ack channel itself *is* driven in request mode by `G-REQUEST`; what is untested is migration-triggered scheduling on top of it. |
| **The multi-rank merged KV-chunk table** | **The one real gap in our own code** (`R-040`). Gate 1 is single-rank only by design, so `kv_migration_base_address` / `kv_migration_stages` are implemented to the documented contract but never executed. First pipelined multi-rank run must treat this as unproven. |
| Destination-slot read-back (`dst-bytes` / `dst-golden`) | Verification of a copy we never make. |

### If you later need it

Get the migration component built (products and constraints: `DEC-070`), then run Gate 2 on a 2- or
4-rank binding to close `R-040`. Two cautions from the engine doc that apply the moment anyone does:
loopback verification is **loopback only** — cross-endpoint prefill -> decode is skipped with a warning
because the destination lives in the decode galaxy's address space; and with a single shared prompt
every source slot is byte-identical, so a copy landing in the **wrong** destination is indistinguishable
from a correct one unless `PREFILL_PRODUCER_SLOT_TRACES` gives each slot its own prompt.
