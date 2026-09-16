<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Llama 3.1 8B prefill and native KV migration

## Purpose

This guide explains one disaggregated Llama 3.1 8B request from input text to streamed output.

> **A test-only attention prototype passes real-weight stress gates.** Stock causal control matches one raw failure. Production attention remains unaccepted.

It centers on the native `tt-d-gen` KV Manager, called native KVM in this guide.

The target uses five Blackhole Galaxies. One Galaxy runs prefill. Four Galaxies form one SC4 decode system.

The model is Llama 3.1 8B Instruct. The target deployment has two request slots and 2,048 total token positions.

This page separates verified code, completed tests, proposed wiring, and required integration tests.

The prototype evidence does not change production model code. It does not accept Task 6.

Open the [sources and evidence manifest](kv-migration-learning-sources.md) or the [review report](kv-migration-learning-report.md).

## Read the status labels first

| Label | Meaning |
|---|---|
| **Verified source** | The cited source implements the stated behavior. |
| **Passed test** | A recorded test passed on the named revision and scope. |
| **Proposed** | The design is selected, but the complete path is not implemented. |
| **Integration gate** | The team must verify this contract before production use. |

Source inspection cannot prove hardware behavior. A host test cannot prove cross-endpoint visibility or model quality.

## System map

The frontend sends the same request identity to the prefill and decode engines.

The prefill Galaxy computes prompt activations and K/V cache entries. It does not generate the user-visible response here.

Native KVM moves selected K/V cache pages. It uses serialized address tables for both cache layouts.

The SC4 decode system replays the prompt tail. It then produces one output token per decode step.

The KVM code does not understand Llama tensor layouts. The address tables provide every physical location.

### Five-Galaxy data path

1. The frontend tokenizes the request.
2. The prefill engine assigns a source slot.
3. The decode engine assigns a destination slot.
4. The prefill Galaxy computes all prompt tokens.
5. Each source layer reports readiness after its K/V writes finish.
6. Native KVM moves the requested K/V prefix.
7. The destination engine waits for a landing report.
8. The decode system replays the prompt tail.
9. The decode system streams generated tokens.

## Worked example: 1,033 prompt tokens and 16 output tokens

Assume the tokenizer produces exactly 1,033 tokens. This token count is an example input to the mechanics.

The shortened prompt text below is illustrative. It does not claim a specific tokenizer result.

> “Explain why a migrated KV cache must preserve RoPE position and head identity.”

Assume the source gets slot 0. Assume the destination gets slot 1. These slot numbers are example assignments.

The request allows 16 new tokens. The final position is below the 2,048-position deployment limit.

### Phase A: admission and planning

The frontend gives both engines one request UUID. The engines use this UUID for rendezvous.

The prefill engine stores all 1,033 token IDs. It plans two 1,024-token prefill chunks.

The first prefill chunk covers logical positions `[0, 1024)`. The second prefill chunk covers `[1024, 1033)` plus padding.

The decode engine computes a reusable prefix cap. The cap is the last 32-token boundary below the prompt length.

For 1,033 prompt tokens, the reusable prefix cap is 1,024 tokens.

The decode engine therefore requests migration for `[0, 1024)`. It retains nine prompt tokens for local replay.

### Phase B: prefill compute

The source runs 32 transformer layers for both prefill chunks.

The 1,024-token prefill chunk gives each SP row 256 tokens.

The nine-token tail uses a padded physical buffer. Its valid range remains `[1024, 1033)`.

For global position `g`, the source SP owner is `(g % 1024) // 256`.

The local cache row is `(g // 1024) * 256 + (g % 256)`.

Positions 1,024 through 1,032 therefore belong to SP row 0. They use local rows 256 through 264.

Each layer writes post-RoPE K and raw V. Q remains a temporary attention activation.

The runner must publish one layer-ready acknowledgement after device completion for each source layer.

The current common runner has this acknowledgement channel. The future Llama runtime must add the required device wait.

### Phase C: native KVM transfer

The destination announces UUID, destination slot 1, and range `[0, 1024)`.

The source pairs this announcement with source slot 0. It clips every queued layer range to the requested span.

The native client submits one migration command per ready layer. Each command names source and destination slots.

The command also names one layer range and one token range. The KVM resolves both layouts through tables.

The transfer contains 32 KV transfer chunks per config. Each transfer chunk covers 32 token positions.

There are 16 configs. The required order is `k_h0` through `k_h7`, then `v_h0` through `v_h7`.

There are 32 layers. Each BF8_B transfer chunk contains four physical tiles and uses 4,352 bytes.

The logical payload for this example is `32 × 16 × 32 × 4,352 = 71,303,168` bytes.

That value is exactly 68 MiB. It excludes retries, protocol records, replication, staging, and scratch space.

The destination device writer retires a command after its final NoC write barrier.

The host observes the device kernel completion counter after that barrier. The writer then reports all channels settled.

The drain path writes its credit only after those device writes settle. KVM success follows successful drain completion.

The source engine then sends a landing report to the destination engine. The destination can admit the migrated prefix.

This ordering is source-verified. The five-Galaxy Llama path has not yet validated it on hardware.

### Phase D: prompt-tail replay and first output token

The destination starts with positions `[0, 1024)` resident in slot 1.

It forwards prompt positions 1,024 through 1,032 through the decode model.

The first eight tail forwards build K/V and discard their intermediate logits.

The forward at position 1,032 is the final prompt forward. Its logits produce the first generated token at position 1,033.

The current engine does not consume prefill logits for this handoff. Tail replay does not overwrite migrated positions below 1,024.

The Llama device codec and SC4 runtime must preserve this engine contract. That work remains an integration gate.

### Phase E: streaming response

The decoder feeds each generated token back as the next input token.

Each feedback step writes that token's K/V at its generated position. The next result advances the response.

After 16 outputs, the request ends because of its generation limit, an EOS token, or another configured stop.

An illustrative response could be: “RoPE binds each K vector to its absolute position and head frame.”

This sentence is an example only. No current end-to-end test produced it.

## Three sizes that must stay distinct

| Term | Size | Purpose |
|---|---:|---|
| **Prefill chunk** | 1,024 tokens | One host-to-device prefill work unit. |
| **KV transfer chunk** | 32 tokens | One K or V head page moved by KVM. |
| **Physical tile** | 32 by 32 values | TT tile storage and compute unit. |

Do not use the word “chunk” alone in reviews. Name the specific unit.

A 128-wide K or V head uses four physical tiles for 32 tokens.

One BF8_B physical tile uses 1,088 bytes. One BF8_B KV transfer chunk therefore uses 4,352 bytes.

The 32-token divisor does not describe a DRAM bank count.

The cache allocator obtains the actual bank grid from `get_num_dram_banks(mesh_device)`.

## Why only K and V migrate

Llama attention has 32 query heads and eight K/V heads. Four query heads share one K/V head.

This design is grouped-query attention, or GQA. It reduces cached state and transfer volume.

Q is needed only during the current attention calculation. Future decode steps never read an old Q value.

K and V are the attention memory for prior positions. Every later token reads them.

Weights already exist on both model systems. Moving weights per request would be wasteful.

Intermediate activations are also transient. Decode reconstructs its needed activations from tokens, weights, and K/V.

## Source cache geometry

The prefill mesh shape is 4 by 8. SP uses four rows, and TP uses eight columns.

TP assigns one K/V head to each column. SP assigns token groups to rows.

Each K or V cache has local shape `[64, 1, 512, 128]` on each device.

The leading dimension packs two users times 32 layers. The local sequence length is `2048 / 4 = 512`.

The cache uses tiled DRAM. Its NdShard page shape is `[1, 1, 32, 128]`.

Pages use row-major orientation and round-robin one-dimensional distribution across the real DRAM bank grid.

The source cache writes post-RoPE K. It writes V without RoPE.

### RoPE frame rule

Hugging Face and Meta use different Q/K element arrangements around RoPE.

The prefill weights must convert Q and K to the Meta adjacent-pair frame.

Blaze decode uses that Meta frame. A wrong frame can copy identical bytes and still produce wrong attention.

Byte equality therefore proves transport fidelity only. It cannot prove RoPE, head, config, layer, slot, or position identity.

## Address tables and the layout-opaque mover

Each endpoint publishes a serialized `KvChunkAddressTable`. It also publishes a device map.

The prefill table describes source locations. The decode table describes destination locations.

The native manager imports every named config. It checks that config IDs are unique and contiguous.

It also requires matching name-to-ID assignments across prefill and decode tables.

The manager builds immutable read and write indexes. The data plane queries those indexes during movement.

The read index deduplicates replicas. The write index returns all local replica destinations.

Each lookup returns a device group, encoded DRAM address, byte size, and 32-token position bounds.

The mover does not calculate Llama strides. Table producers own all layout arithmetic.

This separation permits different source and destination layouts. It also makes wrong tables a high-risk failure source.

### Required Llama table contract

The planned Llama tables use 16 named configs. Names and IDs must match on both endpoints.

The planned order is K heads 0 through 7, followed by V heads 0 through 7.

Every config describes two slots, 32 layers, 2,048 positions, 32-token pages, and 4,352-byte BF8_B pages.

The prefill table must follow the SP4 and TP8 source packing. The decode table must follow the SC4 cache packing.

The Llama-specific table exporters and cross-layout migration test remain proposed.

Do not publish a concrete NoC address until both table builders exist and a lookup test verifies the address.

## Source-to-destination layout map

Source and destination tables share each config identity. They do not share physical placement.

A config identity selects K or V and one global K/V head. Physical placement selects devices, banks, slots, and byte offsets.

Native KVM reads both table entries and copies their bytes. The native data plane should require no Llama-specific change.

The Llama table builders must translate each logical coordinate into the allocation that its endpoint actually uses.

| Coordinate | Prefill source | Decode destination |
|---|---|---|
| K/V head | TP8 gives one global K/V head to each TP column. | TP2 gives four global K/V heads to each TP shard. |
| Token position | SP4 distributes 256-position blocks within each 1,024-position group. | Each TP shard keeps the full 2,048-position sequence axis. |
| Slot | Two slots use user-major layer planes. | The pinned Llama allocation has one slot. Two-slot placement remains pending. |
| Layer | Slot and layer select one packed source plane. | Each decoder stage owns its live cache tensors. |
| DRAM placement | NdShard uses the device's real DRAM bank grid. | The live allocator selects banks per head, bank order, and base addresses. |

The decode mesh mapper shards tensor dimension one. That dimension is the K/V head dimension.

Eight K/V heads cross two TP shards. Therefore, each destination TP shard receives four contiguous global heads.

For global head `h`, the destination TP shard is `h // 4`. Its local head index is `h % 4`.

The destination keeps logical positions in one full sequence axis. Its update kernel starts with `tile_row = position // 32`.

The kernel then maps that tile row into the allocated bank layout. The migration table builder must use the same mapping.

### Logical walk: head 5, position 1,024, layer 13

This is a separate address example. Position 1,024 is outside the worked example's migrated range. It requires a longer migrated prefix.

This walk uses K config `k_h5`, which has config ID 5. Config `v_h5` repeats the walk with config ID 13.

1. The request selects source slot 0, destination slot 1, layer 13, and position 1,024.
2. Source TP column 5 owns global K/V head 5.
3. Source SP row `(1024 % 1024) // 256` is row 0.
4. The source local cache row is `(1024 // 1024) * 256 + 1024 % 256`, which is 256.
5. Source slot 0 and layer 13 select packed plane `0 * 32 + 13`, which is plane 13.
6. The source table must return the real location for config 5, plane 13, row 256.
7. Destination TP shard `5 // 4` is shard 1. Head 5 becomes local head `5 % 4`, which is 1.
8. Destination position 1,024 starts at tile row `1024 // 32`, which is 32.
9. Layer 13 selects that decoder stage's cache tensor and gathered base address.
10. Destination slot 1 has no verified stride in the pinned Llama allocation.
11. A future two-slot allocation and its update kernel must define the slot 1 placement.
12. The destination table must return that verified placement before migration can use this walk.

This walk stops before a NoC address. It does not invent a bank, base address, layer stride, or slot stride.

The pinned Blaze configuration sets `BATCH_SIZE=1` and `n_slots=1`. Its Llama entry also lacks `kv_migration_spec`.

The generic update kernel contains a paged multi-slot formula. That formula becomes relevant only when the Llama allocation uses it.

The actual table must derive two-slot strides from live Blaze tensors. It must match the update kernel and bank order.

Config identity stays stable across endpoints. Physical placement can change without changing the native KVM data plane.

## Native KVM control and completion sequence

Native KVM has a control plane, a data plane, device I/O, and an engine adapter.

The engine adapter sends commands over ZMQ. The native manager returns one terminal outcome per accepted command.

The data plane reads source pages, transfers batches, drains destination pages, and returns credits.

The device movement kernel copies data through four L1 bounce slots. It issues a final NoC write barrier per command.

The device kernel publishes `doneCount` after this barrier. The host link retires work from that completion count.

`DeviceWriteSession` settles only after every planned operation is issued and every device channel has zero in-flight work.

`DrainSession` finishes after device settlement and credit writeback. The control plane then returns command success.

The proposed native `KvmClient` waits for all command outcomes. It then reports the landing to the destination engine.

The landing report is the destination engine's admission signal. A command submission or source ACK is insufficient.

### Current implementation boundary

Pinned tt-d-gen main `10b66f8` contains native KVM and `EngineAdapter`.

Current inspected main `9a8c531` also contains the native manager and adapter.

Current main registers `MigrationKvManagerClient` under `migration`. That client uses the vendored legacy migration layer.

PR head `7ee35d8` adds `KvmClient`. It registers the native client under `kvm` and links `kvm::engine_adapter`.

The present PR state was not available through authenticated API access during this review.

Do not infer merge state from these revisions. Use the deployed build's registry and CMake output.

## Slot lifecycle, cancellation, failure, and retry

### Normal source lifecycle

The prefill engine registers the UUID before source K/V exists. This prevents a destination announcement from racing registration.

The source slot stays pinned while the burst is open. Each layer-ready acknowledgement can queue one layer transfer.

The source seals the burst after its final prefill chunk retires. It releases the pin after terminal completion.

### Normal destination lifecycle

The destination enters `AWAITING_MIGRATION`. It cannot decode from the slot yet.

The destination receives a landing report after the source observes all native command outcomes.

Successful landing marks the migrated prefix resident. The runtime then queues prompt-tail replay.

### Cancellation

The proposed native client sends a failed landing for a cancelled source transfer with an armed destination.

It releases the UUID immediately. It retains submitted command handles until their outcomes drain.

An inbound cancellation drops the local binding. The runtime owns the destination request terminal event.

UUID release does not make the source slot reusable. Submitted native writes can still reference its cache buffers.

Integration must prove that the source pin and buffers remain valid until every submitted command reaches a terminal outcome.

Treat this proof as a required lifecycle audit before native serving acceptance.

### Timeout and peer loss

The engine adapter maps command timeouts and manager failures to terminal outcomes.

The native client has an inbound timeout for announcements or landings that never arrive.

Peer loss fails related transfers. The runtime can then release, reject, or locally recompute according to request state.

### Retry rule

Do not retry blindly into the same destination slot. A failed transfer can leave a partial destination range.

First invalidate or overwrite the complete requested range. Then issue a new transfer with a new correlation identity.

This retry policy is a required integration decision. Current tests do not cover every cancellation and retry interleaving.

## Capacity and payload arithmetic

One 2,048-position K or V head contains 64 KV transfer chunks.

The complete logical cache payload for one request is `64 × 16 × 32 × 4,352` bytes.

That product is 142,606,336 bytes, or exactly 136 MiB.

Two slots contain 272 MiB of aggregate logical K/V payload across the prefill mesh.

These totals exclude replicas, temporary tensors, allocator padding, address tables, staging slabs, and scratch buffers.

They also do not equal per-chip storage. TP and SP distribute the payload across the source mesh.

They do not equal network traffic. Retries, headers, credits, and transport behavior can change network bytes.

A 2,048-token prompt does not migrate all 64 pages under the current tail-replay rule.

Its reusable cap is 2,016 tokens, or 63 pages. The last prompt block remains for decode-side replay.

## Read numerical test results

PCC compares correlation between actual and reference values. A high PCC can hide scale error.

NL2 is `norm(actual - reference) / norm(reference)`. Lower values indicate smaller relative error.

An all-zero reference makes this ratio undefined. Use exact or absolute-error checks for zero references.

Pre-O is the attention head output before the output projection. Post-O is the result after that projection.

The output projection can reduce visible error. Check both pre-O and post-O results.

## Test evidence by phase

| Phase or risk | Evidence | What it proves | What it does not prove |
|---|---|---|---|
| Model constants | Config cross-check against Llama dimensions | 32 layers, 32 Q heads, eight K/V heads, head width 128 | Correct weights or runtime behavior |
| QKV projection | Real-weight module tests | Projection values and TP head placement at tested cases | Full attention or migration |
| Indexed RoPE | Module tests against Hugging Face | Position selection and Meta frame at tested positions | Transport config identity |
| Cache allocation | Galaxy tests in both BF16 and BF8_B | Shape, zero state, bank-grid use, 32 by 128 page geometry | Decode cache geometry |
| Cache writes | Boundary and tail cases | SP ownership, local row, slot/layer packing, zero padding | Native transfer or decode consumption |
| QKV, RoPE, cache composition | Real-weight layer-zero tests | Post-RoPE K and raw V reach correct source cache positions | All layers or cross-endpoint movement |
| MLP and RMSNorm | Module tests | Isolated module numerics at tested shapes | Complete decoder block |
| Attention task 026 | Uncommitted Task 6 run | Twenty real-weight interval cases passed pre-O heads and post-O projection | Repeated-token stability or task acceptance |
| Attention task 027 | Uncommitted repeated-token run | Post-O output passes | Pre-O heads fail: BF16 worst NL2 0.12608; BF8 worst NL2 0.09887 |
| Attention task 028 | Uncommitted exact-input residual run | Reproduces a focused input condition | Pre-O residual remains NL2 0.10081 |
| Attention task 029 | Uncommitted stock-FP32 isolation | Independent SOURCE-HF pre-O gate passes every valid chip | Production FP32 path is absent; ring and stock BF16 still show about 0.10 NL2 |
| Attention task 030 | Test-only supported-FP32 prototype on `[1024,1537)` | PASS for one repeated-token continuation and both cache dtypes | One interval only; production attention is unchanged |
| Attention task 031 | Test-only FP32 boundary stress | PASS for 12 exact boundary cases and all six original task 027 real-weight cases | Production adoption, reuse, immutability, and validation |
| Attention task 032 | Raw cancellation-heavy synthetic hash characterization | Completed against the original limits | Candidate misses some source and exact-cache hash limits; diagnosis continues |
| Attention task 033 | Stock standard FP32 causal control at `[224,257)` | Matches the explicit-local-mask path on the worst row | One shared row does not explain every raw hash failure |
| Prefill reshuffle | 191 host cases for exact helper | Host token ordering for aligned starts and tails | H2D transport or device writes |
| Layer-ready channel | Common runner source and host contracts | Required metadata and 32 layer notifications | Correct Llama device synchronization |
| Native table import | KVM unit and source checks | Multi-config import, ID checks, read/write index construction | Correct Llama-specific table content |
| Native engine adapter | Adapter tests and source | Command backpressure, timeout, matching terminal outcome | Llama serving integration |
| Native client branch | Two-process tests with mock device I/O and TCP | Rendezvous and command lifecycle in tested host scenarios | Real DMK, RDMA, SC4, or five Galaxies |
| DMK byte tests | Native device-I/O test surface | Device mover can verify bytes in its own scope | Correct RoPE frame or table semantics |
| Five-Galaxy request | Pending | Nothing yet | End-to-end correctness, latency, failure recovery |

Published QKV, cache, RoPE, MLP, and RMSNorm results are through tt-metal revision `4cf42fb`.

Attention tasks 026 through 033 are uncommitted Task 6 evidence. Task 6 remains unaccepted.

Tasks 027 and 028 record historical failures. The old ring path still fails.

Tasks 030 and 031 validate a test-only candidate. They do not change production attention.

### Verified test-only attention prototype

The candidate gathers selected packed-cache values. It reorders them, applies an explicit mask, and uses supported stock FP32 SDPA.

Attempt 030 passed one repeated-token continuation for BF16 and BF8_B caches. Its interval was `[1024,1537)`.

It recorded `actual/verified = 0`. It closed cleanly at 11:11:21.

Attempt 031 passed 12 exact boundary cases. It covered six intervals with two cache dtypes.

The intervals were `[0,1)`, `[0,33)`, `[224,257)`, `[1024,1537)`, `[1056,2048)`, and `[2016,2048)`.

It also passed all six original Task 027 repeated-token source-head and output-projection cases.

| Cache dtype | Worst head PCC | Worst head NL2 | Minimum post-O PCC | Maximum post-O NL2 |
|---|---:|---:|---:|---:|
| BF16 | 0.9997846133 | 0.0209190180 | 0.9999621894 | 0.0104071685 |
| BF8_B | 0.9996511653 | 0.0327059192 | 0.9999156025 | 0.0152525783 |

The unchanged BF16 gates are PCC 0.999 and NL2 0.03. The BF8_B gates are PCC 0.995 and NL2 0.05.

All 1,760 per-chip scalar metrics were finite.

Exact cache gather, order, masks, and padded-zero checks passed on all 32 chips.

Attempt 031 recorded `actual/verified = 0`. It closed cleanly at 11:29:34.830.

No threshold was relaxed.

Task 032 completed the raw synthetic hash characterization. The candidate still misses the original hash limits.

BF16 source hashes pass 3 of 10 cases. BF8_B source hashes pass 3 of 10 cases.

BF8_B exact-cache hashes pass 7 of 10 cases.

The worst cache-relative NL2 is about 9.3% for interval `[224,257)` in both cache dtypes.

These cancellation-heavy synthetic checks remain unresolved. Diagnosis continues.

Task 033 compared the explicit local mask with stock standard FP32 causal attention.

Both paths have the same worst result on chip 8, row 256.

| Metric | Both paths |
|---|---:|
| PCC | 0.9957571199646015 |
| NL2 | 0.0929282984724009 |
| Expected RMS | 0.0024138343012115623 |
| Error RMS | 0.0002243135144059073 |
| Maximum absolute error | 0.000705384649336338 |

This parity shows that the local mask did not introduce this row's error.

The result bounds this row to arithmetic shared with stock causal FP32. It does not identify one underlying operation.

It also does not explain every raw hash failure.

Attempt 033 recorded `actual = 0` and `verified = 0`. One test passed.

All 2,800 reported metric scalars were finite. The device closed cleanly at 11:46:34.296 UTC.

A broader stock-causal parity test is planned for ten intervals and two cache dtypes.

Its prechosen per-chip gates are PCC 0.9999 and NL2 1%.

Production candidate work is authorized. Permanent validation, isolation, reuse, and real-weight tests are authorized and required.

No active allocation is implied. This guide ran no device commands.

Task 6 closure, decoder assembly, full model, runtime wiring, table publication, and migration remain pending.

## Required test plan for the worked example

### Gate 1: device-free request plan

1. Create a 1,033-token request with a fixed UUID.
2. Assert two source prefill chunks.
3. Assert source valid ranges `[0,1024)` and `[1024,1033)`.
4. Assert destination reusable cap 1,024.
5. Assert destination migration range `[0,1024)`.

This gate catches an off-by-one error before hardware runs.

### Gate 2: source cache correctness

1. Run both source prefill chunks through all 32 layers.
2. Read selected K and V pages from both slots.
3. Compare post-RoPE K and raw V with the reference.
4. Verify SP owner and local row at positions 1,023, 1,024, and 1,032.

This gate proves source content. It does not prove transport.

### Gate 3: table parity

1. Export all 16 prefill configs.
2. Export all 16 decode configs.
3. Assert matching config names and IDs.
4. Assert every lookup uses 32 tokens and 4,352 bytes.
5. Check both slots, every layer, and every boundary page.
6. Compare table lookups with actual allocated buffers.

This gate catches wrong head, layer, slot, position, bank, and stride mapping.

### Gate 4: native movement

1. Fill source pages with identity-coded patterns.
2. Fill destination pages with a distinct sentinel.
3. Migrate `[0,1024)` from source slot 0 to destination slot 1.
4. Wait for the destination landing report.
5. Read every migrated destination page.
6. Confirm positions `[1024,2048)` still contain the sentinel.

This gate proves cross-layout byte placement for the requested range.

It does not prove the values have the correct model meaning.

### Gate 5: semantic handoff

1. Use one real 1,033-token prompt.
2. Run disaggregated prefill and native migration.
3. Replay the nine-token tail on SC4.
4. Capture the first generated token at position 1,033.
5. Compare logits and tokens with a non-disaggregated baseline.
6. Continue for 16 generated tokens.

This gate tests the exact first-token convention and later decode continuity.

### Gate 6: failure paths

1. Cancel before any layer-ready acknowledgement.
2. Cancel after some native commands finish.
3. Drop the rendezvous connection before landing.
4. Force one command timeout.
5. Confirm both slot managers reach a terminal state.
6. Confirm no UUID, source pin, or destination wait remains.

Run retry tests only after the team defines destination invalidation behavior.

## Advantages

Prefill and decode can scale independently. Long prompts do not occupy decode compute during full prompt processing.

Layer-ready transfer can overlap migration with later prefill work. This overlap can reduce handoff latency.

Address tables let source and destination use different cache layouts. The mover remains model-agnostic.

GQA reduces transfer payload because eight K/V heads serve 32 query heads.

Block-aligned migration keeps the final prompt block on decode. This makes the first-token rule explicit.

## Limitations and subtle risks

The supported-FP32 candidate remains test-only. Production adoption, reuse, cache immutability, and validation tests remain pending.

Task 032 raw synthetic hash accuracy remains unresolved. Diagnosis continues.

Task 033 explains one mask question only. It does not close the broader raw hash diagnosis.

The old ring attention path still fails. Task 6 remains unaccepted.

The current Llama runtime and table exporters are incomplete. Native cross-endpoint migration is therefore unproven.

The current main serving client named `migration` uses the legacy migration layer. Its name does not imply native KVM.

The native manager accepts layout tables as truth. A consistent wrong table can move bytes without an obvious transport error.

Layer-ready acknowledgement requires a real device completion wait. A host enqueue is not sufficient.

KVM command success is only one part of destination admission. The engine also needs the landing rendezvous record.

Cancellation cannot cancel an already submitted device or RDMA operation immediately. Cleanup must drain terminal outcomes.

The current native client branch tests use mock device I/O. They do not cover all hardware timing interleavings.

Two slots limit concurrency. A stuck source pin or destination wait can block new requests quickly.

The 2,048-position limit includes prompt and generated tokens. Admission must reject any request that exceeds this combined limit.

## Debugging checklist

| Symptom | First check | Likely contract |
|---|---|---|
| Fluent but wrong output | Compare Meta RoPE frame and config order | Semantic identity |
| Correct first prefill chunk, wrong tail | Check SP owner and local row formulas | Source packing |
| Destination wait never ends | Inspect landing rendezvous and inbound timeout | Control completion |
| Source slot never releases | Inspect command outcomes and source pin count | Source lifecycle |
| Wrong head only | Compare config name-to-ID mapping | Table parity |
| Wrong layer only | Compare packed slot/layer plane and command layer range | Layer identity |
| Every 32-token boundary fails | Check transfer page size and DRAM page mapping | Page geometry |
| Byte copy passes, logits fail | Check RoPE frame and decode tail replay | Model meaning |
| Retry corrupts later tokens | Check destination invalidation before retry | Retry policy |

## Questions and exercises

1. Why does a 1,033-token prompt migrate 1,024 tokens under the current engine rule?
2. Which component owns the mapping from position 1,024 to a physical DRAM address?
3. Why can a correct byte comparison still hide a wrong RoPE frame?
4. Which signal allows the destination engine to leave `AWAITING_MIGRATION`?
5. Why does native command completion wait for the device channel in-flight count to reach zero?
6. What changes in the payload calculation for BF16 cache pages?
7. Which test would catch a swapped `k_h3` and `v_h3` config ID?
8. Why must a retry invalidate the complete destination range first?
9. Where does global head 5 reside after TP8-to-TP2 remapping?
10. Why can the guide calculate tile row 32 but not a destination NoC address?

## Glossary

| Term | Definition |
|---|---|
| **Acknowledgement** | A control signal that confirms a defined completion point. |
| **Config identity** | A stable config name and ID for one K or V global head. |
| **Decode** | Autoregressive work that produces user-visible output tokens. |
| **Device map** | A file that maps fabric nodes to local devices and hosts. |
| **DMK** | The native data movement kernel that copies bytes through NoC. |
| **GQA** | Grouped-query attention, where several Q heads share one K/V head. |
| **KV cache** | Stored K and V vectors for prior token positions. |
| **KV transfer chunk** | One 32-token page for one K or V head. |
| **Landing report** | The source engine's message that completes destination inbound wait. |
| **Native KVM** | The tt-d-gen native KV Manager control and data planes. |
| **NL2** | Relative L2 error between actual and reference values. Lower is better. |
| **NoC** | The on-chip network used for device memory movement. |
| **PCC** | Pearson correlation between actual and reference values. Higher is better. |
| **Physical tile** | A 32 by 32 tensor storage unit. |
| **Physical placement** | The device, bank, base address, and byte offset for stored data. |
| **Post-O** | Attention output after the output projection. |
| **Prefill** | Prompt processing that creates KV cache entries. |
| **Prefill chunk** | One 1,024-token source work unit in this deployment. |
| **Pre-O** | Attention head output before the output projection. |
| **RoPE** | Rotary position embedding applied to Q and K. |
| **SC4** | The four-Galaxy decode system used by this deployment. |
| **Slot** | One fixed cache region assigned to a request. |
| **SP** | Sequence parallelism across four prefill mesh rows. |
| **TP** | Tensor parallelism that divides heads across devices. Prefill uses eight columns, and decode uses two shards. |

## Language and review method

This guide follows selected ASD-STE100 Issue 9 writing rules. It uses short sentences, active voice, and defined technical terms.

The review uses the official [ASD-STE100 site](https://www.asd-ste100.org/), [Issue 9 PDF](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf), and [official explanation](https://www.asd-ste100.org/about_STE.html).

The automated sentence scan is only a screening tool. It does not certify formal ASD-STE100 compliance.

See the [source manifest](kv-migration-learning-sources.md) for revisions, anchors, evidence limits, and the language-check artifact.
