<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

> Status snapshot: 17 September 2026. See the [live test page](http://127.0.0.1:8768/migration-prefill-tests.html) for updates.

# Prefill migration: the tests that define completion

This page covers **Llama-3.1-8B-Instruct prefill and native tt-d-gen migration from its cache**. It does not cover a decoder. No SC4 test is part of this plan.

**Current conclusion:** the 2K model, the prefill runtime, table export and a local packed-cache copy have passed their recorded checks. **Native tt-d-gen transfer has not passed yet.** A local TTNN copy does not test the native transport.

The page uses short sentences and defines technical terms. “Passed” always refers to the stated test scope and saved evidence.

## 1. The boundary we will test

```text
Two different prompts
        |
        v
Host-to-device input + slot/start/end metadata
        |
        v
One Galaxy: SP=4, TP=8, all 32 Llama layers
        |
        +---- live K/V buffers ---- address table + chip ownership map
        |
        +---- wait for writes ---- layer acknowledgements
                                      |
                                      v
                          Native tt-d-gen KV manager
                                      |
                                      v
                          Passive destination buffers
                                      |
                                      v
                       Independent byte verification
                              TEST STOPS HERE
```

The destination is a test receiver. It allocates memory and checks bytes. It loads no decoder weights and generates no tokens. Two assigned Galaxies can test the native transfer path without an SC4 cluster. The native manager may call the destination a “decode” role in its configuration; that role name does not make the receiver a decoder model.

The source and receiver use their own live allocations and tables. They must not use invented addresses or a fake hostname. The current native manager requires different hosts for its prefill and destination table roles. This is why a two-host transport test replaces the older same-host recipe.

### Terms

| Term | Meaning here |
|---|---|
| KV cache | The key and value tensors that prefill writes for attention. |
| Slot | One request's private region of that cache. We use two slots. |
| Compute chunk | Up to 1,024 real prompt tokens processed in one call. |
| Transfer block | 32 token positions for one K or V head, one layer and one slot. |
| Config | A named stream in the table: `k_h0` through `k_h7`, then `v_h0` through `v_h7`. |
| Address table | The map from config, layer, position and slot to a chip, bank and address. |
| Acknowledgement | A message that says a layer's cache writes for a compute chunk are complete. It does not say that transfer finished. |
| PCC | A numerical similarity measure used for model values. A byte copy needs exact equality, not a PCC threshold. |
| Drain | Wait for accepted transfer work to finish before releasing resources. |

## 2. When are we done?

There are two completion points. Neither requires decode testing.

**A. Prefill migration works at 2K.** All required rows M01–M12 below have passing evidence for their full stated scope. The real native manager transfers both distinct source slots. Independent checks verify every requested byte, all 16 configs and all 32 layers. Tests also cover partial chunks, continuation, slot reuse and failure handling. M14 publishes the tested implementation and reproducible commands.

**B. Prefill migration supports all requested capacities.** A is complete, and M13 passes at **4K, 8K, 16K, 32K, 64K and 128K**. Each capacity gets a real prefill/transfer test. Host address arithmetic alone cannot establish support.

The existing 2K numerical evidence is the model-correctness anchor. We will not add larger-context golden KV comparisons. At larger capacities, we check execution, valid ranges, source placement and exact transferred bytes. The separate performance effort records full-model wall time, throughput, per-chunk time and book next-token observations.

No finite test suite guarantees every future request or deployment. These gates establish a precise claim: **the tested prefill producer supplies the correct named cache regions, makes them readable only after writes complete, and preserves their bytes through the tested native transfer path.** They do not establish decoder layout compatibility or a generated response.

## 3. Required test map

These are coverage areas, not prescribed test functions or 14 separate device runs. Existing model tests help identify use cases and failure modes. We choose the simplest reliable Llama tests, combine overlapping checks and reuse accepted evidence when it still applies. One well-designed run can cover several IDs. Model-specific reference cases that do not apply to Llama need an explanation, not a copied test.

Each ID links to its detailed checks. Live badges show the recorded status. A partial badge is not a pass for the whole row.

| ID | Test | What must be true | Status |
|---|---|---|---|
| [M01](#m01) | Model and run contract | The selected model, precision, slots and geometry agree before allocation. | Partial |
| [M02](#m02) | Table structure and saved-table round trip | Every logical block has the right config, owner and address after export/import. | Passed at 2K |
| [M03](#m03) | Independent live table readback | Table reads select the intended live tensor data, not merely a self-consistent wrong address. | Partial |
| [M04](#m04) | Real source values at 2K | The runner/table path carries the already validated Llama cache values. | Partial |
| [M05](#m05) | Input, partial chunks and continuation | Valid tokens reach the correct absolute positions; padding cannot become resident prompt data. | Partial |
| [M06](#m06) | Layer readiness | Exactly 32 acknowledgements follow completed writes for each valid chunk. | Partial |
| [M07](#m07) | Native source registration and peer readiness | The real manager accepts this table and live device map, and waits for the correct peer. | Pending |
| [M08](#m08) | Native transfer and exact destination bytes | Every requested destination block equals the saved source block. | Pending |
| [M09](#m09) | Incremental transfer ranges | Only completed, selected ranges move; later chunks do not damage earlier data. | Pending |
| [M10](#m10) | Two-slot isolation and reuse | Distinct prompts remain distinct during interleaving, remapping and reuse. | Partial |
| [M11](#m11) | Failure and verifier sensitivity | Errors cannot produce a false success, duplicate readiness or an empty comparison pass. | Partial |
| [M12](#m12) | Buffer lifetime, drain and shutdown | Source memory remains valid until transfer ends; teardown leaves no owned work active. | Partial |
| [M13](#m13) | Capacity ladder through 128K | The same sender contract works at every requested capacity. | Pending |
| [M14](#m14) | Reproduction and publication | Tests, commands, source versions, results and logical commits agree. | Partial |

### M01

**Test: model and run contract.** Check checkpoint/config identity, Llama3 RoPE settings, all 32 layers, SP4/TP8, two slots, 1,024-token chunks and BFP8_B cache. Head width is 128. Each 32-token transfer block contains four packed tiles: **4 × 1,088 = 4,352 bytes**. Reject unsupported geometry, capacity, slot count, tracing and DFlash before device work. Bind the fixture, source revision and native binary versions in the run record.

Existing adapter/runtime host tests cover configuration forwarding and early rejection. A complete native run still needs the identity checks at its process boundaries. Llama uses ordinary dense attention and SwiGLU; GPT-OSS MoE, attention sinks and sliding-window rules do not belong here.

### M02

**Test: address table and saved-table round trip.** Cover all 16 ordered configs, all 32 layers, both slots and all 32 chips. Check SP row changes, DRAM bank changes, slot/layer strides, first block and last block. Compare against an independent bank walk. Export the live table, import it again and require identical lookups and ownership.

At 2K, the full allocation has **16 × 32 × 2 × (2,048 / 32) = 65,536 entries**. Live 2K serialization and ownership checks passed. Host boundary checks extend through 128K. The longer-capacity device checks remain in M13.

### M03

**Test: independent live table readback.** Write values that distinguish config, head, layer, slot and position. Read each region through the exported table. Separately read the live tensor with an independent placement calculation. Decode BFP8 blocks for exact value comparison between those two views. Check raw block sizes and count every expected comparison.

This is the same logical test as GPT-OSS `test_gpt_oss_kv_chunk_table_readback`. Comparing two reads that both use the same wrong table is insufficient. The existing Llama host bank walk and packed-copy evidence are useful prerequisites; they do not close this independent live-placement gate by themselves.

### M04

**Test: real source values at 2K.** Reuse the accepted 2K model evidence and frozen inputs. Link table-visible cache data from the real runner to an independent cache view. Apply the documented 2K numerical policy when comparing with the reference. Report any raw FP32 intermediate differences under that policy; do not rewrite them as exact equality.

The direct 32-layer model and all-layer KV checks passed in the accepted 2K suite. The missing link is the complete runner → exported table → independent reader path. This test checks that the right data is selected before transport. It is separate from M08, which checks whether transport changes bytes. No larger-context golden matrix is required.

### M05

**Test: input, partial chunks and continuation.** Use the real host-to-device service. Check its 12-byte metadata: slot, absolute start and exclusive end. The input tensor is borrowed; the runtime must not upload, reorder or free it again. The engine already puts tokens in device order.

Cover valid lengths around 32, 256 and 1,024-token boundaries. Examples are 31/32/33, 255/256/257 and 1,023/1,024/1,025. Use legal tile-aligned starts such as 32, 224, 256 and 992, with end positions inside capacity. Reject unaligned starts and out-of-range metadata. Check valid rows and expected padding separately inside a partially written tile. Compare whole untouched blocks byte for byte. A 32-token block that contains new valid rows is expected to change.

Continuation starts from the prior length rounded **down** to a 32-token boundary. It replays the short tail using absolute positions. At a nonzero start, SP token order must still be correct. The current live runtime pass used aligned full chunks; it does not prove all these edge cases.

### M06

**Test: layer readiness.** Warmup emits zero readiness messages. Each accepted chunk emits exactly one acknowledgement for each global layer 0–31. Request IDs identify the actual chunk and increase across slots. Verify acknowledgement counts and order on the real channel.

The current implementation waits for the **whole compute chunk** before emitting its 32 acknowledgements. This is a functional design with less opportunity to overlap transfer and compute. Four live chunks produced 128 acknowledgements. Host tests cover duplicate IDs and forward/synchronization/sink failures. An integration check must still prove that delayed writes cannot make ranges transferable early.

### M07

**Test: native manager source registration and peer readiness.** Start the actual tt-d-gen manager with the exported source table, correct host identity and live physical device map. Attach its client and passive receiver. Wait for real readiness before accepting a transfer. A missing peer or mismatched config must fail within a recorded timeout and must not dispatch an invalid device read.

Drive the real source client through `register_source → enqueue_layer → finish_burst`. The Llama acknowledgements must identify the right request, slot and absolute range. A burst is a group of transfer commands. It completes only after the group is sealed and every required command has a final outcome. Exercise backpressure: if the command queue is full, pending layers must resume in order. A standalone EngineAdapter copy does not establish this source bridge.

This gate is pending. Loading a table through TTNN is not the same as registering it with the native manager. The tested Metal build must remain unchanged while native dependencies are built. Record the manager binary, DMK binary, transport version, selected device resources and kernel-driver identity in the eventual run evidence.

### M08

**Test: real native transfer.** Save the source bytes before transfer. Give the passive destination a different initial pattern. Transfer through the native manager, wait for actual completion, then compare every requested block with the saved source. Check source bytes again. Include BFP8 exponent bytes, mantissa bytes and packed padding bytes within each transferred block.

Exercise both slots, all configs and all layers. Include a crossed mapping, source 0 → receiver 1 and source 1 → receiver 0. Selected ranges have an explicit expected block count. Require zero missing or skipped blocks. Keep destination regions outside the transfer unchanged.

The existing local TTNN copy matched **65,536 pages / 285,212,672 bytes** and preserved the source. It does not pass this native-transfer gate. “Command completed” is also insufficient: the independent byte check must pass.

### M09

**Test: incremental ranges.** Transfer a ready prefix. Produce the next chunk. Transfer the added range. Check both the old prefix and the new data. Exercise a short final chunk and a continuation at a non-chunk-aligned, tile-aligned position. A delayed acknowledgement must prevent premature transfer. Retry behavior must follow the native protocol, with no duplicate success accounting.

Transfer selection uses complete 32-token blocks and explicit valid/resident lengths. Allocated capacity is not the same as valid prompt length. The test must never treat unwritten capacity or a partially valid final block as a completed prompt prefix.

### M10

**Test: two-slot isolation and reuse.** Use two different prompts and different recognizable test patterns. Interleave their chunks in deterministic and seeded schedules. Check ordinary mapping and crossed destination slots. After all reads and transfers for one request finish, reuse its source slot for a third distinct prompt. The other slot must stay correct.

The live readiness test already interleaved two slots. The packed-copy test established distinct source slots. Native remapping and reuse remain pending. Repeating the same prompt in both slots cannot detect cross-wiring.

### M11

**Test: visible failure and a sensitive verifier.** At the appropriate host or test boundary, inject a no-op copy, a short read, one wrong block, wrong config/slot identity and source-plus-destination corruption. The saved pre-transfer snapshot must catch joint corruption. Require a nonzero result for any missing comparison, skipped required config/layer, timeout or failed device close.

Also test write, synchronization and acknowledgement-sink exceptions. A partly failed runtime must reject later requests until it is restarted. Test delayed/missing native peer and completion errors with bounded waits. Do not test malformed device addresses by issuing unsafe reads; reject them before dispatch or use a host stub. Existing host failure tests and local-copy verifier tests pass, but native integration failures remain pending.

Reuse the native client's tests for one failed outcome, a response after timeout, duplicate success, cancellation and a stale source announcement. Verify that each request has one final outcome. A cancelled request must retain its source pin until outstanding commands have drained.

### M12

**Test: lifetime and shutdown.** Hold a transfer in progress in a controlled test. Confirm that its source allocation cannot be freed or reused early. Finish or cancel it through the documented native path, drain accepted work, stop the manager and only then release the model-owned device resources. Record actual process exits and all 32 device closes.

Check a fresh restart with run-specific service names and a newly exported table. Old acknowledgements, tables or completion files must not certify the new run. Clean close passed for the completed runtime/local-copy runs. Native in-flight drain and restart remain untested.

### M13

**Test: capacity ladder.** Run the prefill-side contract at 2K, 4K, 8K, 16K, 32K, 64K and 128K. At each size, exercise valid data near the beginning, SP/bank/chunk transitions and the end. Transfer the complete selected valid prefix for both slots. Verify all its requested blocks across all configs and layers. Include a shorter valid prefix inside a larger allocation.

Long-context comparison should stream bounded blocks or use a saved snapshot, rather than retain all source and destination tensors in host RAM. Preserve pre-transfer source evidence so a joint corruption cannot pass. Review memory, storage and lease time before each run. No full golden reference run is added. Current model execution at 4K/8K is a prerequisite result, not a migration pass.

### M14

**Test: reproducible publication.** Each required gate records its command, fixture, source/native hashes, expected coverage, observed coverage, exact exits and cleanup result. Each test method gets a short comment explaining its purpose. Keep host, device-placement and native-transfer tests distinguishable.

Push logical commits only with their relevant passing checks. Reuse accepted 2K numerical/performance evidence when execution is unchanged. Do not call a sampled layer/config check a full pass. The runtime and 25 host tests are published; the remaining native test implementation and results are not yet complete.

## 4. One request story, with the tests attached

This is a **planned test story**, not a claim that native transfer has already passed.

### Step 1 — Prepare two private work areas

Alice's prompt has 1,500 tokens. Bob's different prompt has 1,100 tokens. Each gets a 2,048-position source slot. Both use the same frozen checkpoint and tokenizer. The runner allocates BFP8 K and V buffers and publishes a table for all 16 configs. **M01–M04** check identity, addresses, saved-table import and source values.

### Step 2 — Send the first chunks

The producer sends Alice's first 1,024 tokens to slot 0. It sends Bob's first 1,024 tokens to slot 1. Each call carries its own slot/start/end metadata. The engine arranges tokens for the four SP rows. The runtime uses that input directly. **M05** checks token order and ownership. **M10** checks that Alice and Bob cannot overwrite each other.

### Step 3 — Make the first prefixes safe to read

All 32 layers write K and V. The runtime waits for those writes to finish. It then emits 32 layer acknowledgements for the chunk. The manager must not treat a submitted computation as a completed cache write. **M06** checks this order. **M09** checks that transfer waits for the required readiness.

### Step 4 — Send short final chunks

Alice's next call describes `[1024,1500)`. Bob's describes `[1024,1100)`. The remaining input buffer positions are padding. Padding is not a new user token. Tests inspect valid positions and protected regions around the write. **M05** catches wrong bounds or a runtime that ignores the real end.

### Step 5 — Select complete transfer blocks

For this sender test, select Alice's complete blocks `[0,1472)` and Bob's `[0,1088)`. These are **46 and 34 blocks per config/layer**. The final 28 and 12 prompt positions stay outside this transfer selection. This is explicit test policy; it does not test a decoder's tail-replay policy.

There are **(46 + 34) × 16 × 32 = 40,960 blocks** to verify. The full allocation contains 65,536 blocks, but only the selected 40,960 blocks may count toward this transfer result. **M02, M08 and M09** check that distinction.

### Step 6 — Move the bytes through the real manager

The source table and peer are registered with native tt-d-gen. The passive receiver has separate allocations filled with different data. The test saves the source blocks, requests transfer and waits for native completion. **M07** checks registration and readiness. **M08** checks actual transport and completion.

### Step 7 — Prove what arrived

The verifier reads the passive destination using its own table. It compares all 40,960 blocks with the saved source. It checks source preservation, unselected destination regions and crossed slot mappings. A completion message alone cannot satisfy this step. **M08, M10 and M11** catch dropped writes, wrong destinations and false passes.

### Step 8 — Continue and reuse safely

A separate continuation case extends Alice's prompt. It starts from the prior length rounded down to a tile boundary and replays the short tail. A separate reuse case replaces Alice with a third prompt only after her transfer is finished. Bob remains unchanged. **M05, M09, M10 and M12** check absolute positions, residency, isolation and lifetime.

Recomputed BF16/BFP8 values need the stated numerical comparison policy. The shared prefill guide does not promise bit-identical recomputation after tail replay. In contrast, copying one saved packed block must be byte-identical.

### Step 9 — Make failures observable, then close

Controlled tests delay a peer, fail a write, corrupt a block and interrupt an acknowledgement sequence. Each must produce the specified failure, rather than a success marker. Successful transfers are drained before shutdown. **M11 and M12** establish those failure and cleanup behaviors.

### Step 10 — Repeat the contract at larger capacities

Use the same logical tests with each requested capacity. Do not infer 128K support from a 2K address calculation. **M13** supplies the device evidence. **M14** publishes the tested revision and complete coverage report. The prefill-side task is then complete for those configurations.

## 5. How this follows other models

The comparison below is a source review. It is not a claim that we reran these other models or verified their present deployment status.

| Existing reference | Logical check we reuse | Llama adaptation |
|---|---|---|
| GPT-OSS `test_gpt_oss_kv_chunk_table_smoke` | Check every table entry and config. | M02: 32 layers, 8 K heads + 8 V heads, width 128, SP4/TP8. |
| GPT-OSS `test_gpt_oss_kv_chunk_table_protobuf_roundtrip` | Export/import must preserve every lookup. | M02: keep native shared table format and real ownership. |
| GPT-OSS `test_gpt_oss_kv_chunk_table_readback` | Compare table reads against an independent live tensor view. | M03: Llama's own slot/layer/bank layout, all required entries. |
| Kimi, GLM and Mistral prefill table tests | Read real or marked caches through their tables; check distinct cache configs and stage placement. | M02–M04: cover all 16 Llama configs. Do not add MLA, sparse index caches or multi-rank layers. |
| Shared prefill Gate 1 | Real runner input, layer acknowledgements and source KV validation before transport. | M04–M06: preserve the accepted 2K numerical policy and cover partial/continued input. |
| Shared prefill Gate 2 `dst-bytes` | Compare destination bytes after real migration completion. | M07–M09: native tt-d-gen with passive buffers; do not substitute the legacy launcher or local TTNN copy. |
| Shared producer scenarios | Deterministic interleaving, seeded schedules and slot reuse. | M10: use the two supported Llama slots with genuinely different prompts. |
| Shared continuation documentation | Tile-aligned replay with absolute positions. | M05/M09: add explicit automated boundary coverage. The shared guide itself identifies a dedicated host-test gap. |
| Shared destination-check warnings | A subset, skipped host or identical prompt can hide an error. | M08/M11: exact coverage counts, no required skips, distinct data and saved source snapshots. |

We keep the **logical tests**, not every reference parameter. GPT-OSS's example 0.82 source-PCC setting, 36 layers, head width 64 and legacy queue names are not Llama acceptance criteria. The GPT-OSS mock manifest also points both slots at one trace; we use distinct inputs to expose cross-wiring, as the shared guide recommends.

The older Gate 2 recipe uses tt-llm-engine processes. The byte-equality requirement remains useful. Our required implementation uses **native tt-d-gen**, and the evidence must identify that backend. A native source registration test and a native TRANSFER test remain necessary even if the legacy flow passed elsewhere.

### Native tests to reuse

These are existing test definitions in the inspected tt-d-gen revision. Their presence does not mean they have passed in this workspace. We will reuse them for host/protocol coverage and connect the Llama source to the real manager for integration coverage.

| Native suite | Relevant checks | Limitation of the existing suite |
|---|---|---|
| `test_kv_disagg_source.cpp` | `AdmitRegistersTheSource`, `QueuedLayersFlushOnArm`, `PrefillDoneWaitsForBurst`, `BackpressureDefersLayers`, `PartialDrainResumesInOrder`, `AbortReleasesPinViaDeferredClose`, `StaleAnnounceIsDropped` | Recording or mock components; it does not read a live Llama cache. |
| `test_kvm_client.cpp` | `EachLayerIsOneCommandOverItsOwnRange`, `TheBurstCompletesOnlyAfterTheLastOutcome`, `AnAckedBurstStillWaitsForItsSeal`, `OneFailedOutcomeFailsTheWholeBurst`, `CancelCompletesOnlyOnceItsCommandsDrain` | Real client protocol with a stand-in manager; it does not establish device transfer. |
| `engine_adapter_test.cpp` | Readiness, backpressure, timeout, late response, duplicate success and response identity. | Protocol behavior, not model values or physical cache placement. |
| `control_plane_table_loading_test.cpp` | Host selection, missing device, empty index, bank overflow and invalid command ranges. | Table/control validation needs a separate live ownership test. |
| `data_plane_test.cpp` | Multiple configs, transfer/drain, stream credits and interruptible shutdown. | Uses mock I/O or transport. |
| Device migration and Mooncake examples | Real transport and device I/O mechanics. | Their fixed DRAM test locations must be replaced with independently allocated live buffers. They are not Llama acceptance evidence. |

## 6. Evidence and source links

- [Published runtime and 25 host tests](https://github.com/tenstorrent/tt-metal/commit/8d05143750e8df9197203fd7b5afce31e1097e5a).
- [Accepted 2K numerical validation](https://github.com/tenstorrent/tt-metal/blob/f3f704a266b362201771d4ecbdbb3060a025a5a2/models/demos/llama_3p1_8b_d_p/docs/validation-2k.md).
- [Live readiness proof](http://127.0.0.1:8768/evidence/task-10-native-migration/readiness-003-verifier-correction-001/root-verification.json).
- [Local packed-copy proof](http://127.0.0.1:8768/evidence/task-10-native-migration/packed-copy-launch-preparation-001/run/root-verification.json).
- [Published full-model performance and book observations](https://github.com/tenstorrent/tt-metal/blob/f3f704a266b362201771d4ecbdbb3060a025a5a2/models/demos/llama_3p1_8b_d_p/docs/performance-prefill.md).
- [GPT-OSS table tests](https://github.com/tenstorrent/tt-metal/blob/f3f704a266b362201771d4ecbdbb3060a025a5a2/models/demos/gpt_oss_d_p/tests/test_kv_cache_table.py).
- [GPT-OSS mock producer manifest](https://github.com/tenstorrent/tt-metal/blob/f3f704a266b362201771d4ecbdbb3060a025a5a2/models/demos/gpt_oss_d_p/tt/runners/manifests/gpt_oss_producer_mock_migration.yaml).
- [GPT-OSS loopback producer manifest](https://github.com/tenstorrent/tt-metal/blob/f3f704a266b362201771d4ecbdbb3060a025a5a2/models/demos/gpt_oss_d_p/tt/runners/manifests/gpt_oss_producer_loopback_migration.yaml).
- [Kimi/GLM/Mistral table tests](https://github.com/tenstorrent/tt-metal/blob/f3f704a266b362201771d4ecbdbb3060a025a5a2/models/demos/deepseek_v3_d_p/tests/test_kv_cache_table.py).
- [Shared prefill migration gates](https://github.com/tenstorrent/tt-metal/blob/f3f704a266b362201771d4ecbdbb3060a025a5a2/models/demos/common/prefill/docs/PREFILL_MIGRATION_TESTING.md).
- [Shared producer/runner scenarios](https://github.com/tenstorrent/tt-metal/blob/f3f704a266b362201771d4ecbdbb3060a025a5a2/models/demos/common/prefill/tests/test_producer_runner_e2e.py).
- [Native source lifecycle tests](https://github.com/tenstorrent/tt-d-gen/blob/7ee35d8d46899f5111470ea98e5daf81bd4bb607/engine/tests/test_kv_disagg_source.cpp).
- [Native client protocol tests](https://github.com/tenstorrent/tt-d-gen/blob/7ee35d8d46899f5111470ea98e5daf81bd4bb607/engine/tests/test_kvm_client.cpp).
- [Native adapter response tests](https://github.com/tenstorrent/tt-d-gen/blob/7ee35d8d46899f5111470ea98e5daf81bd4bb607/kv_manager/tests/unit/engine_adapter/engine_adapter_test.cpp).
- [Native table-loading tests](https://github.com/tenstorrent/tt-d-gen/blob/7ee35d8d46899f5111470ea98e5daf81bd4bb607/kv_manager/tests/unit/control_plane/control_plane_table_loading_test.cpp).
- [Native data-plane unit tests](https://github.com/tenstorrent/tt-d-gen/blob/7ee35d8d46899f5111470ea98e5daf81bd4bb607/kv_manager/tests/unit/data_plane/data_plane_test.cpp).

The status file records the last review time. The page refreshes its status badges every 15 seconds. It does not turn a preparation result into a device or native-transfer pass.
