# Decode handoff contracts and joint validation

Source review, 2026-09-18. No device execution or implementation change. Scope: prefill-to-decode integration through 64K; 128K is deferred.

This handoff is part of the local fixture/result delivery requested by the user; no public push is pending. The immutable public links below identify existing source contracts. Proposed decoder integration tests are not accepted device results. See the [migration test guide](migration-prefill-tests.md) for the recorded prefill-only coverage and current capacity status.

## 1. What to share

Share these published files, not an address dump from a previous allocation:

- [Llama table exporter](https://github.com/tenstorrent/tt-metal/blob/7258d9a314b7f84e826574411a58889cbdacfc84/models/demos/llama_3p1_8b_d_p/tt/runners/kv_chunk_table.py)
- [Logical-to-physical layout](https://github.com/tenstorrent/tt-metal/blob/7258d9a314b7f84e826574411a58889cbdacfc84/models/demos/llama_3p1_8b_d_p/tt/runners/kv_layout.py)
- [Shared protobuf schema](https://github.com/tenstorrent/tt-metal/blob/7258d9a314b7f84e826574411a58889cbdacfc84/tt_metal/impl/internal/disaggregation/protobuf/kv_chunk_address_table.proto)
- [Native bridge command/lifetime contract](https://github.com/tenstorrent/tt-d-gen/blob/6eeb1a9c4a86e4b4e1795665bd7a504b216e504f/docs/native_prefill_source_bridge.md)

These immutable links are on the published `divanovic/llama31-8b-disagg` and `divanovic/native-prefill-source-bridge-publication` histories. The later portable fixture commits are not claimed public here.

The source and destination each generate a table from their own live allocation. Their physical addresses, device groups and fabric-node identities differ. They must agree on the semantic identity of each page. Decode needs its own correct table exporter if its physical layout differs; the prefill SP4/TP8 layout must not be assumed to describe decode memory.

| Contract | Current prefill value / required agreement |
|---|---|
| Logical key | Config, slot, layer, absolute token position. Config order is `k_h0`…`k_h7`, then `v_h0`…`v_h7`; 32 layers, two slots. |
| Page representation | BFP8_B tiled DRAM; 32 tokens × head dimension 128 per packed page; 4,352 bytes. Current source uses SP4/TP8. |
| Address representation | `noc_addr`, byte size, device-group index, fabric-node-to-host map. Source exporter packs the bank index in the upper 32 bits and the DRAM byte address (allocation base plus bank-local offset) in the lower 32 bits. |
| Serialization | `configs` is authoritative; schema supports unrolled entries and strided runs plus format versions. Use compatible import/export APIs, not an entries-only assumption. |
| K/V meaning | K is already RoPE-rotated; V is raw. Preserve KV-head and layer identities. |
| Validity | A logical `[from,to)` interval differs from the whole 32-token pages copied. Copied padding does not extend valid length. `tokens_transferred` in this client reports exclusive endpoint `to`, not `to-from`. |
| Request binding | Same prompt token IDs and generation UUID; correct source/destination slots and endpoint. Slot numbers need not match. A worker chunk request ID is not a user-request ID. |
| Lifetime | Publish readiness only after writes are complete. Completion, retirement and cancellation must use the correct generation. Keep buffers alive while native work can access them; the tested owners require both exact managers stopped before releasing caches on shutdown. |

## 2. Other agreements, especially the first token

Pin checkpoint revision/weight conversion, tokenizer revision and special-token/chat formatting. Agree capacity, token IDs, 32 query heads / 8 KV heads / head dimension 128, and sampling settings. The [model configuration](https://github.com/tenstorrent/tt-metal/blob/7258d9a314b7f84e826574411a58889cbdacfc84/models/demos/llama_3p1_8b_d_p/reference/llama_3p1_8b_config.py) and [RoPE implementation](https://github.com/tenstorrent/tt-metal/blob/7258d9a314b7f84e826574411a58889cbdacfc84/models/demos/llama_3p1_8b_d_p/tt/rope.py) specify Meta adjacent-pair coordinates and Llama3 scaling: theta 500,000; factor 8; low/high frequency factors 1/4; original context 8,192. A decode implementation using HF half-split coordinates needs the corresponding conversion; byte equality alone cannot fix a convention mismatch.

The [prefill runtime](https://github.com/tenstorrent/tt-metal/blob/7258d9a314b7f84e826574411a58889cbdacfc84/models/demos/llama_3p1_8b_d_p/tt/tt_prefill_runtime.py#L100-L117) is KV-only: it skips the LM head, synchronizes, discards output, then acknowledges layers.

**The generic engine already handles first-token generation without a logits handoff.** For prompt length L and engine reuse-block size B, it requests a migrated prefix ending at `floor((L-1)/B)*B`, leaving at least the final prompt token to evaluate. After landing, decode forwards the remaining prompt suffix; the forward at position L−1 produces the first generated token. That generated token is subsequently forwarded at position L. See [inbound admission](https://github.com/tenstorrent/tt-d-gen/blob/6eeb1a9c4a86e4b4e1795665bd7a504b216e504f/engine/src/runtime/backend_runtime.cpp#L419-L480), [prefix cap](https://github.com/tenstorrent/tt-d-gen/blob/6eeb1a9c4a86e4b4e1795665bd7a504b216e504f/engine/include/engine/control/prefix_indexer.hpp#L69-L75) and [decode writer](https://github.com/tenstorrent/tt-d-gen/blob/6eeb1a9c4a86e4b4e1795665bd7a504b216e504f/engine/src/runtime/decode_writer.cpp#L76-L133). B is a configured reuse unit, distinct from the 32-token migration page and 1,024-token prefill compute chunk; joint configuration must preserve compatible alignment. Conditional example: B=32, L=2048 migrates `[0,2016)` and decode evaluates positions 2016…2047. Total cache capacity must cover input plus generated-token KV. A 2K prompt in a 2K cache is a prefill/migration fixture; writing a generated token at position 2048 requires a larger cache. This control contract exists; the real Llama decode adapter still needs joint validation. The passive full-prefix byte fixture does not demonstrate decoder consumption.

Decisions for the decode developer: confirm the destination byte/layout and RoPE conventions; adopt the existing suffix-evaluation/first-token contract and agree B; bind the same token sequence and UUID; agree when transfer, retirement, cancellation and safe reuse are acknowledged. Reverse decode-to-prefill migration is not implemented in the current standalone scope.

## 3. Prioritized joint tests

1. **Real consumption:** retain exact landing-byte checks, then compare decode using migrated KV against the same decode implementation with locally produced KV. Check the first few logits/steps with agreed numerical tolerances and deterministic sampling. Cross-implementation logits may differ slightly; near-tied logits should not force identical tokens on every prompt. Add an independent reference comparison where existing numerical evidence supports it.
2. **Isolation and position:** two distinct prompts in crossed slots; assert the correct UUID/slot/head/layer mapping and the first output position. Compare migrated and local paths using identical prefix boundaries.
3. **One composed edge sequence:** a partial tile, tile-aligned continuation, retained prefix and untouched suffix, then a distinct third request after retirement. Check padding separately from valid tokens and preserve the other slot.
4. **Failure and lifetime:** delay readiness, cancel after real first-chunk completion, drain while buffers remain owned, then restart with fresh identities. Verify no stale completion or old KV is consumed. Bytes-in-flight cancellation and abrupt peer loss remain separate limits unless actually exercised.

Existing engine [paired host tests](https://github.com/tenstorrent/tt-d-gen/blob/6eeb1a9c4a86e4b4e1795665bd7a504b216e504f/engine/tests/test_kv_disagg_paired.cpp) cover the generic control composition with mock compute. They do not replace a real decoder test.

## 4. Capacity strategy

Run the focused 4K, 8K, 16K, 32K and 64K migration ladder once as its acceptance coverage. For C, the prepared owner computes two actual prompts of lengths C and C−32, then crosses selected ranges `[C−1024,C)` and `[C−1056,C−32)`. It compares 32,768 packed pages / 142,606,336 bytes across all 16 configurations and 32 layers. It also checks adjacent/untouched samples, source preservation, full table loading and process-identity-bound RSS/HWM. The high-end ranges are selected transfer coverage, not an entire-prefix transfer or exhaustive untouched-region scan.

Compose that with the [accepted 2K full-prefix native transfer](https://github.com/tenstorrent/tt-metal/blob/7258d9a314b7f84e826574411a58889cbdacfc84/models/demos/llama_3p1_8b_d_p/docs/migration-native-2k.md), [independent full-table address evidence](https://github.com/tenstorrent/tt-metal/blob/7258d9a314b7f84e826574411a58889cbdacfc84/models/demos/llama_3p1_8b_d_p/docs/migration-prefill-address-evidence.md), runtime/range tests and existing ≤64K full-model execution/performance/output checks. Detailed all-layer numerical KV validation is anchored at 2K, not proven across the whole capacity ladder.

This is sufficient targeted evidence for the stated prefill migration capacity risks without repeating a complete HF/PCC/performance sweep at every capacity. It does not establish long-context numerical equivalence of a new decoder. After a localized change, repeat its affected gate and a small baseline; repeat the full relevant ladder when table geometry, capacity accounting or shared lifetime behavior changes. Keep 128K deferred.
