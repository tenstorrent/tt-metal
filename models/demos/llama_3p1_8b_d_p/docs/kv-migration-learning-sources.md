<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# KV migration learning guide: source and evidence manifest

## Scope

This manifest supports `status/kv-migration-learning.html` and `notes/kv-migration-learning.md`.

The review used source inspection and existing recorded tests. It ran no model or device command.

## Revision pins

| Repository or document | Revision or path | Use |
|---|---|---|
| tt-metal | `4cf42fb0b9a56d11540d6099965f1a9bc1e2fc11` | Llama modules, cache geometry, and tests |
| tt-d-gen pinned baseline | `10b66f8a4c3285158da6ffb1d611d5b6d6349a63` | Original task baseline |
| tt-d-gen inspected main | `9a8c53135896ff8047b27f298ed855d2a0a7495e` | Current runtime, legacy client, native manager, and adapter |
| Native client PR head | `7ee35d8d46899f5111470ea98e5daf81bd4bb607` | Proposed `KvmClient` engine wiring |
| tt-blaze pinned decode | `adda6bcdb276a7023d02b3d9eb25dce9b783c862` | Llama TP2 cache allocation and update kernel |
| Runtime contract note | `notes/runtime-wire-completion-contract.md` | Llama runtime obligations |
| Full-model preparation note | `notes/full-model-runtime-preparation.md` | Current and planned model scope |
| Backend review note | `notes/2026-09-15-backend-revisit.md` | Native KVM option review |

The tt-d-gen main clone was shallow and grafted. An ancestry result cannot establish PR merge state.

Authenticated GitHub API access was unavailable. The guide does not claim the present PR state.

## Primary source anchors

### Request planning and first-token ownership

| Claim | File and anchor |
|---|---|
| Keep one block below prompt end | `engine/include/engine/control/prefix_indexer.hpp:69-75` at `9a8c531` |
| Destination requests the capped prefix | `engine/src/runtime/backend_runtime.cpp:419-479` at `9a8c531` |
| Successful inbound queues decode-side prompt work | `engine/src/runtime/backend_runtime.cpp:846-875` at `9a8c531` |
| Non-last tail forwards discard logits | `engine/src/runtime/decode_writer.cpp:76-129` at `9a8c531` |
| Final prompt forward returns first output token | `engine/src/runtime/decode_writer.cpp:97-129` at `9a8c531` |
| Prefill source plans all prompt chunks | `engine/src/runtime/backend_runtime.cpp:788-796` at `9a8c531` |
| Prefill writer sends the valid tail range | `engine/src/runtime/prefill_writer.cpp:51-102` at `9a8c531` |

For 1,033 tokens and a 32-token block, the destination cap is 1,024.

The source still computes valid ranges `[0,1024)` and `[1024,1033)`.

Pinned links:

- [Reusable prefix cap](https://github.com/tenstorrent/tt-d-gen/blob/9a8c53135896ff8047b27f298ed855d2a0a7495e/engine/include/engine/control/prefix_indexer.hpp#L69-L75)
- [Destination request plan](https://github.com/tenstorrent/tt-d-gen/blob/9a8c53135896ff8047b27f298ed855d2a0a7495e/engine/src/runtime/backend_runtime.cpp#L419-L479)
- [Complete source prefill plan](https://github.com/tenstorrent/tt-d-gen/blob/9a8c53135896ff8047b27f298ed855d2a0a7495e/engine/src/runtime/backend_runtime.cpp#L788-L796)
- [Tail replay and first output token](https://github.com/tenstorrent/tt-d-gen/blob/9a8c53135896ff8047b27f298ed855d2a0a7495e/engine/src/runtime/decode_writer.cpp#L76-L129)

### Source Llama cache

| Claim | File and anchor |
|---|---|
| 32 layers, 32 Q heads, eight K/V heads, width 128 | `models/demos/llama_3p1_8b_d_p/reference/llama_3p1_8b_config.py` |
| SP4, TP8, two slots, 2,048 positions | `models/demos/llama_3p1_8b_d_p/tt/kv_cache.py` |
| Bank grid uses `get_num_dram_banks` | `_cache_memory_config` in `tt/kv_cache.py` |
| Page shape `[1,1,32,128]` | `_cache_memory_config` and `_validate_cache_tensor` |
| Post-RoPE K and raw V writes | `write_kv_chunk` and its module documentation |
| SP owner formula | `_owned_positions` in `tests/unit/test_kv_cache.py` |
| Local row formula | `_cache_rows_by_sp` in `tests/unit/test_kv_cache.py` |
| BF8_B page is `4 × 1088 = 4352` bytes | `BF8_PAGE_BYTES` and allocation test |
| Meta RoPE frame requirement | `tt/model_config.py` module documentation |

The Llama-specific table builder does not exist at this revision. Table details remain a required contract.

Pinned source links:

- [Prefill mesh, slots, layers, and local shape](https://github.com/tenstorrent/tt-metal/blob/4cf42fb0b9a56d11540d6099965f1a9bc1e2fc11/models/demos/llama_3p1_8b_d_p/tt/kv_cache.py#L15-L28)
- [Prefill NdShard bank grid and allocation](https://github.com/tenstorrent/tt-metal/blob/4cf42fb0b9a56d11540d6099965f1a9bc1e2fc11/models/demos/llama_3p1_8b_d_p/tt/kv_cache.py#L77-L123)
- [Prefill slot and layer update call](https://github.com/tenstorrent/tt-metal/blob/4cf42fb0b9a56d11540d6099965f1a9bc1e2fc11/models/demos/llama_3p1_8b_d_p/tt/kv_cache.py#L228-L239)
- [Source SP ownership and local row order](https://github.com/tenstorrent/tt-metal/blob/4cf42fb0b9a56d11540d6099965f1a9bc1e2fc11/models/demos/llama_3p1_8b_d_p/tests/unit/test_kv_cache.py#L80-L97)
- [Source slot-layer plane formula](https://github.com/tenstorrent/tt-metal/blob/4cf42fb0b9a56d11540d6099965f1a9bc1e2fc11/ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp#L125)

### Layer-ready prefill path

| Claim | File and anchor |
|---|---|
| H2D metadata carries slot and valid range | `engine/include/engine/pipeline/pipeline_types.hpp` |
| Host reshuffle occurs before H2D | `engine/src/pipeline/prefill_pipeline.cpp` |
| Reader maps layer acknowledgements to ranges | `engine/src/runtime/prefill_reader.cpp:46-170` |
| Source sends one migration layer per acknowledgement | `engine/src/runtime/prefill_reader.cpp:66-84` |

The common runner completion sink does not itself synchronize the device.

The future Llama runtime must wait for relevant device work before publishing each layer acknowledgement.

### Legacy and native engine clients

| Claim | File and anchor |
|---|---|
| `migration` uses vendored legacy layer | `engine/src/kv_manager_clients/migration_kv_manager_client.cpp:16-19,154-163` at `9a8c531` |
| Current main links legacy adapter | `engine/CMakeLists.txt:26-27,55` at `9a8c531` |
| Proposed `kvm` uses `EngineAdapter` | `engine/src/kv_manager_clients/kvm_client.cpp:15-18,63-87` at `7ee35d8` |
| Proposed native registry key is `kvm` | `engine/src/kv_manager_clients/kvm_client.cpp:578` at `7ee35d8` |
| Proposed build links `kvm::engine_adapter` | `engine/CMakeLists.txt:41-43,77-90` at `7ee35d8` |
| Proposed commands carry both slots and ranges | `kvm_client.cpp:245-271` at `7ee35d8` |
| Proposed landing follows settled outcomes | `kvm_client.cpp:274-318` at `7ee35d8` |
| Proposed cancellation retains command handles | `kvm_client.cpp:394-415` at `7ee35d8` |
| Proposed peer loss and inbound timeout terminate waits | `kvm_client.cpp:444-458,543-563` at `7ee35d8` |

Pinned PR links:

- [Native client command and settled landing](https://github.com/tenstorrent/tt-d-gen/blob/7ee35d8d46899f5111470ea98e5daf81bd4bb607/engine/src/kv_manager_clients/kvm_client.cpp#L245-L318)
- [Native client cancellation handles](https://github.com/tenstorrent/tt-d-gen/blob/7ee35d8d46899f5111470ea98e5daf81bd4bb607/engine/src/kv_manager_clients/kvm_client.cpp#L394-L415)
- [Native client registry key](https://github.com/tenstorrent/tt-d-gen/blob/7ee35d8d46899f5111470ea98e5daf81bd4bb607/engine/src/kv_manager_clients/kvm_client.cpp#L578)

### Native manager tables and data path

| Claim | File and anchor |
|---|---|
| Adapter imports all table configs | `kv_manager/src/control_plane/maps/kv_chunk_address_table_adapter.cpp:145-190` |
| Config names and IDs must match | `kv_manager/src/control_plane/maps/kv_table_manager.cpp:143-183` |
| Lookup returns table address and size | `kv_chunk_address_table_adapter.cpp:216-227` |
| Read index deduplicates and write index groups replicas | `kv_manager/include/data_plane/kv_chunk_index.hpp:59-96` |
| Device session waits for zero in-flight commands | `kv_manager/src/data_plane/device_write_session.cpp:69-77` |
| Drain credit follows device settlement | `kv_manager/src/data_plane/drain_session.cpp:28-59` |
| Successful drain response follows session completion | `kv_manager/src/data_plane/data_plane.cpp:463-536` |
| DMK uses a final NoC write barrier | `kv_manager/device_io/kernel/dmk.cpp:117-179` |
| DMK publishes done after the barrier | `kv_manager/device_io/kernel/dmk.cpp:254-266` |
| Host retires from device done count | `kv_manager/device_io/src/dmk_link.cpp:100-120` |
| Control plane waits for remote drain success | `kv_manager/src/control_plane/control_plane.cpp:1422-1453,2413-2433` |
| Terminal success ACK follows migration completion | `kv_manager/src/control_plane/control_plane.cpp:2503-2583` |

These sources define the ordering contract. This guide did not validate that ordering on five Galaxies.

Pinned barrier links:

- [Device mover write barrier](https://github.com/tenstorrent/tt-d-gen/blob/9a8c53135896ff8047b27f298ed855d2a0a7495e/kv_manager/device_io/kernel/dmk.cpp#L117-L179)
- [Device done counter after the barrier](https://github.com/tenstorrent/tt-d-gen/blob/9a8c53135896ff8047b27f298ed855d2a0a7495e/kv_manager/device_io/kernel/dmk.cpp#L254-L266)
- [Writer settlement condition](https://github.com/tenstorrent/tt-d-gen/blob/9a8c53135896ff8047b27f298ed855d2a0a7495e/kv_manager/src/data_plane/device_write_session.cpp#L69-L77)
- [Drain credit after settlement](https://github.com/tenstorrent/tt-d-gen/blob/9a8c53135896ff8047b27f298ed855d2a0a7495e/kv_manager/src/data_plane/drain_session.cpp#L28-L59)

### Decode layout and address-table boundary

| Claim | File and anchor |
|---|---|
| Decode shards K/V heads on tensor dimension one | `tests/blaze/fused_ops/llama31_decoder_layer_stage/harness.py:188-216` at `adda6bc` |
| Decode allocates full sequence caches with four K/V heads per TP device | `harness.py:306-375` at `adda6bc` |
| Pinned Llama constants use TP2, one slot, and 2,048 positions | `blaze/models/llama_3p1_8b/llama_3p1_8b_blaze_config.py:150-168,495-498,541` |
| Update kernel converts position to tile row | `blaze/ops/gqa_kv_cache_update/kernels/op.hpp:171-195` |
| Generic kernel has separate paged and single-user formulas | `op.hpp:201-235` at `adda6bc` |
| Migration spec must follow allocation and update kernel | `docs/kv_migration_spec.md:117-153` at `adda6bc` |
| Runner gathers real bases and topology | `docs/kv_migration_spec.md:239-242` at `adda6bc` |
| Pinned Llama entry exposes no migration spec | `blaze/models/llama_3p1_8b/entry.py:20-67` at `adda6bc` |

Pinned Blaze links:

- [Llama cache head sharding and allocation](https://github.com/tenstorrent/tt-blaze/blob/adda6bcdb276a7023d02b3d9eb25dce9b783c862/tests/blaze/fused_ops/llama31_decoder_layer_stage/harness.py#L188-L216)
- [Llama cache bank geometry](https://github.com/tenstorrent/tt-blaze/blob/adda6bcdb276a7023d02b3d9eb25dce9b783c862/tests/blaze/fused_ops/llama31_decoder_layer_stage/harness.py#L306-L375)
- [TP2 and single-slot constants](https://github.com/tenstorrent/tt-blaze/blob/adda6bcdb276a7023d02b3d9eb25dce9b783c862/blaze/models/llama_3p1_8b/llama_3p1_8b_blaze_config.py#L150-L168)
- [Decoder 1x2 mesh and K/V TP2](https://github.com/tenstorrent/tt-blaze/blob/adda6bcdb276a7023d02b3d9eb25dce9b783c862/blaze/models/llama_3p1_8b/llama_3p1_8b_blaze_config.py#L488-L500)
- [Configured slot count](https://github.com/tenstorrent/tt-blaze/blob/adda6bcdb276a7023d02b3d9eb25dce9b783c862/blaze/models/llama_3p1_8b/llama_3p1_8b_blaze_config.py#L541)
- [KV update position and slot formulas](https://github.com/tenstorrent/tt-blaze/blob/adda6bcdb276a7023d02b3d9eb25dce9b783c862/blaze/ops/gqa_kv_cache_update/kernels/op.hpp#L171-L235)
- [Migration layout requirements](https://github.com/tenstorrent/tt-blaze/blob/adda6bcdb276a7023d02b3d9eb25dce9b783c862/docs/kv_migration_spec.md#L117-L153)
- [Llama entry surface](https://github.com/tenstorrent/tt-blaze/blob/adda6bcdb276a7023d02b3d9eb25dce9b783c862/blaze/models/llama_3p1_8b/entry.py#L20-L67)

The head mapping follows contiguous sharding of the K/V head dimension across two TP devices.

The pinned Llama allocation has one slot. It cannot establish a destination stride for example slot 1.

No source supports a concrete destination NoC address for the planned two-slot deployment yet.

## Evidence status

### Published Llama module evidence through `4cf42fb`

- QKV projection tests passed.
- Indexed RoPE tests passed.
- KV cache tests passed.
- MLP tests passed.
- RMSNorm tests passed.
These results were published through tt-metal revision `4cf42fb`.

### Uncommitted Task 6 attention evidence

- Task 026 passed 20 real-weight cases across ten intervals and two cache dtypes.
- Task 026 passed pre-O heads and the post-O projection at its tested inputs.
- Task 027 repeated-token pre-O heads fail.
- Task 027 worst NL2 is 0.12608 for BF16 and 0.09887 for BF8_B.
- Task 027 post-O output passes.
- Task 028 exact-input pre-O residual remains NL2 0.10081.
- Task 029 stock FP32 passes the independent SOURCE-HF pre-O gate on every valid chip.
- Task 029 minimum PCC is 0.99989028, and maximum NL2 is 0.01513274.
- Ring versus exact remains about 0.10081 NL2. Stock BF16 remains about 0.10029 NL2.
- The Task 029 BF16 error cosine is 0.9965. A production FP32 path is not implemented.
- Task 030 supported-FP32 prototype passed one repeated-token continuation for both cache dtypes.
- Task 030 used interval `[1024,1537)`, recorded `actual/verified = 0`, and closed cleanly at 11:11:21.
- Task 031 passed 12 exact boundary cases across six intervals and two cache dtypes.
- Task 031 also passed all six original Task 027 real-weight source-head and output-projection cases.
- Task 031 BF16 worst head PCC and NL2 are 0.9997846133 and 0.0209190180.
- Task 031 BF16 post-O minimum PCC and maximum NL2 are 0.9999621894 and 0.0104071685.
- Task 031 BF8_B worst head PCC and NL2 are 0.9996511653 and 0.0327059192.
- Task 031 BF8_B post-O minimum PCC and maximum NL2 are 0.9999156025 and 0.0152525783.
- All 1,760 per-chip scalar metrics were finite.
- Exact cache gather, order, masks, and padded-zero checks passed on all 32 chips.
- Task 031 recorded `actual/verified = 0` and closed cleanly at 11:29:34.830.
- The candidate uses selected packed-cache gather, reorder, explicit masking, and supported stock FP32 SDPA.
- Production adoption, reuse, cache immutability, and validation tests remain pending.
- Task 032 raw synthetic hash characterization completed against the original limits.
- BF16 source hashes pass 3 of 10 cases. BF8_B source hashes pass 3 of 10 cases.
- BF8_B exact-cache hashes pass 7 of 10 cases.
- Worst cache-relative NL2 is about 9.3% at `[224,257)` for both cache dtypes.
- Raw cancellation-heavy hash accuracy remains unresolved. Diagnosis continues.
- Task 033 stock causal control recorded `actual = 0`, `verified = 0`, and one passing test.
- All 2,800 Task 033 metric scalars were finite. The device closed cleanly at 11:46:34.296 UTC.
- Explicit-local-mask and stock-standard-causal paths share the same worst chip 8, row 256.
- Both paths have PCC 0.9957571199646015 and NL2 0.0929282984724009 on that row.
- Both paths have expected RMS 0.0024138343012115623 and error RMS 0.0002243135144059073.
- Both paths have maximum absolute error 0.000705384649336338.
- This parity shows that the local mask did not introduce this row's error.
- It does not identify one underlying operation or explain every raw hash failure.
- A ten-interval, two-dtype stock-causal parity test is planned.
- Its prechosen per-chip gates are PCC 0.9999 and NL2 1%.
- Production candidate and permanent validation work are authorized. Task 6 remains unaccepted.
- Permanent isolation, reuse, and real-weight tests are authorized and pending.
- Evidence anchor: `evidence/task-6-attention/attempt-033-224-stock-causal-control/root-verification.txt`.
- Original hash source failures remain documented characterization evidence.
- The old ring path still fails.
- The BF16 gates remain PCC 0.999 and NL2 0.03. The BF8_B gates remain PCC 0.995 and NL2 0.05.
- No threshold was relaxed.
- Task 6 remains unaccepted.

The real-weight prototype gates pass. Stock causal parity bounds one failure, while raw hash diagnosis continues.

### Completed common runtime evidence

- The exact reshuffle helper passed 191 host cases.
- Cases covered 64 aligned starts and tail variants.
- Native manager and adapter unit tests exist at the inspected source revisions.
- The native client branch has two-process tests with mock device I/O and TCP.

### Not yet proven

- Complete Llama decoder blocks
- Full Llama prefill model
- Llama runner registration and device completion waits
- Llama prefill and SC4 address tables
- Native client integration in the deployed engine build
- Real cross-endpoint native KVM movement for Llama
- Five-Galaxy semantic equivalence
- Cancellation and retry behavior under all hardware interleavings

## Arithmetic checks

```text
BF8_B transfer chunk = 4 tiles × 1,088 bytes = 4,352 bytes
Example migrated prefix = 32 pages × 16 configs × 32 layers × 4,352 bytes
                        = 71,303,168 bytes = 68 MiB
Full logical cache      = 64 pages × 16 configs × 32 layers × 4,352 bytes
                        = 142,606,336 bytes = 136 MiB
Two logical slots       = 272 MiB
```

The full cache calculation is a capacity calculation. The current tail-replay rule migrates at most 2,016 prompt positions.

## Language review

The guide applies selected rules from ASD-STE100 Issue 9, dated 2025-01-15.

Primary language sources:

- <https://www.asd-ste100.org/>
- <https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf>
- <https://www.asd-ste100.org/about_STE.html>

The automated scan checks sentence length heuristically. It excludes tables, code blocks, headings, and link targets.

The manual review checks active voice, stable terminology, one topic per paragraph, and defined project terms.

The scan does not prove formal ASD-STE100 certification.
