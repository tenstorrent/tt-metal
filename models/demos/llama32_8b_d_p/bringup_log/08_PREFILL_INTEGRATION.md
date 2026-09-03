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
add `llama32_8b_d_p` to that branch, which is legitimate because P2 committed to keeping gpt-oss's
`NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32` and shard shape
(`models/demos/gpt_oss_d_p/tt/attention/kv_cache.py:27`, `:87`). That edit touches **shared code** —
it needs a `DEC` and an entry here, and it should generalise the name check rather than duplicate
the function.
