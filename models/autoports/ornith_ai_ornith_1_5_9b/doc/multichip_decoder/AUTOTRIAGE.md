# AUTOTRIAGE

## Diagnosis

The hang is a **single-reader DRAM-sharded matmul activation-geometry contract violation**: local K is 96 tiles, but the sweep allocates 64 activation storage cores with two tiles each. The multicast factory truncates `num_blocks_per_shard` to `96 / 64 = 1`; its kernel consequently decodes a 64-entry sender-coordinate table as a 96-entry table. At the first block, all seven receiver readiness increments go to logical core `(0,4)` instead of the waiting sender `(0,0)`. The captured semaphore values are exactly **7 at the wrong core and 0 at the intended core on all four devices**. No activation block is published.

The actual stopped configuration is **`mlp.down_proj`, cores=64, block=1, readers=1**, not cores=32/readers=8. Both the original requester attribution and `AUTOTRIAGE_INPUT.md` misidentify the last completed row. This is an unsupported geometry accepted by incomplete shared-op validation; a model-local input-layout restriction can avoid it without changing shared kernels or precision.

## Triage Evidence

Evidence paths below are relative to this report's directory unless stated otherwise.

1. **Running operation and predecessor.** `triage_geometry/tt-triage.txt:249–260` identifies op **309, `MatmulDeviceOperation`**, on devices 0, 1, 2, 3, with 264 participating cores (66 per chip). Input A is BF16 `[1,1,1,3072]`, width-sharded L1; B is BFLOAT8_B `[3072,4096]`, width-sharded DRAM. Op 308 is `InterleavedToShardedDeviceOperation`, converting the same activation from interleaved DRAM. The mesh table's displayed 2x2 arrangement does not establish a different application mesh; the test explicitly opens 1x4.

2. **The completed-case record corrects the failing-case attribution.** `geometry_0.json` contains 132 completed rows: 36 each for packed input, gate, and up, followed by 24 for down. Its last rows are:

   | Role | Activation cores | Block | Readers | PCC | Median ms |
   |---|---:|---:|---:|---:|---:|
   | down | 32 | 1 | 4 | 0.99999994 | 0.032337 |
   | down | 32 | 1 | 8 | 0.99999994 | 0.033443 |
   | down | 32 | 3 | 1 | 0.99999994 | 0.049510 |
   | down | 32 | 3 | 2 | 0.99999994 | 0.033169 |
   | down | 32 | 3 | 4 | 0.99999994 | 0.034457 |
   | down | 32 | 3 | 8 | 0.99999994 | 0.045972 |

   The next iteration in `tests/multichip_geometry.py:44–79` is cores=64, block=1, readers=1. Each saved row follows correctness checking, trace capture, 120 timed replays, synchronization, and trace release. The final log warning at `geometry_0.log:292`, 19:25:30.348 UTC, also requests 64 output shards of `[32,64]`, rather than the 32-core configuration's `[32,128]` shards. Neither `ROLE_DONE mlp.down_proj` nor `GEOMETRY_PASS` appears.

3. **Earliest producer stop.** On every chip, NCRISC kernel **394**, `reader_bmm_tile_layout_in0_sender_dram_sharded`, on logical `(0,0)` is in `sender_sem.wait(...)`, source line **277**. Five other storage/compute cores wait for activation VALID at line **336**; two compute-only cores, `(0,9)` and `(6,9)`, wait at line **354**. Another 58 storage-only cores wait at line **209**. These are the same logical sites on all devices despite differing physical coordinates.

4. **CB and semaphore values.** `dump_circular_buffers` starts at line 1664; `dump_semaphores` at 2897. On all four devices:

   | Resource | Direct observation |
   |---|---|
   | CB0, activation | 2 BF16 pages; pushed=0, popped=0 on every listed matmul core |
   | CB1, weight, eight compute cores | 48 BFP8 pages; pushed=32, popped=0, occupancy=32 |
   | CB2, output | 16 BF16 pages; pushed=0, popped=0 |
   | CB3, partials | 16 FP32 pages; pushed=0, popped=0 |
   | Semaphore 0, `0xab40`, logical `(0,0)` | 0, although the sender needs 7 |
   | Semaphore 0, `0xab40`, logical `(0,4)` | **7**, initial value 0 |
   | Semaphore 1, `0xab50`, eight compute cores | 0: activation INVALID |
   | Semaphore 2, `0xab60` | 1, its initial value; this is unused legacy state, not a missing credit |

   The misplaced count occurs at triage lines **3001, 3205, 3409, 3613**. The intended sender's zero occurs at **2899, 3103, 3307, 3511**. `(0,4)` is itself a storage-only sender waiting for eight credits for a later block; the seven incorrect first-block credits cannot satisfy it either.

5. **Downstream compute/read sites.** All eight active BRISCs per device are in weight-reader `reserve_back` at line **178**. All eight TRISC0s wait for CB0 at compute line **418**. TRISC1 at `tile_regs_commit` line **514** and TRISC2 at `pack_block` line **531** are stalled behind that missing unpack input. There is no captured LLK assertion or `ebreak`; watcher, LLK asserts, and lightweight asserts were disabled in this run.

6. **Host snapshot, read without attaching a debugger.** `/proc/2111328` identifies the original `timeout 1800` supervisor, sleeping in `sigsuspend`. `/proc/2111329` identifies the requested Python module, with its main thread in `futex_do_wait`, no tracer, and 78 threads. This supports a waiting host but does not provide a Python/C++ stack or prove the exact synchronization call. The op window, empty CB0, and saved-case boundary identify the device-side blocker independently.

7. **Triage artifacts and limits.** `triage-summary.txt` labels every script `pass`, but the detailed `check_noc_status.py` section explicitly says `fail`; the summary cannot be read as an application-health verdict. The NoC mismatches concern Ethernet **ERISC1** software counters at zero versus nonzero hardware counters; examples have nonposted requests and acknowledgements both 484, or both 335. They do not establish an outstanding worker transaction. `check_broken_components.py` says cores were halted by triage and subsequently were no longer halted, including unrelated/idle cores. Those are capture-time halt-state anomalies, not proof of preexisting silicon failure. Binary-integrity, core-magic, L1, and DRAM checks report no corresponding failure. Ethernet links/heartbeats are up, although historical CRC/resend/drop counters are nonzero; a single snapshot cannot attribute those counts to this hang.

## Source Evidence

Source references in this section are relative to the checkout root.

**Model and sweep create excess activation storage.** `models/autoports/ornith_ai_ornith_1_5_9b/tests/multichip_geometry.py:44–60` calculates `kw = ceil(k / (32 * cores))`, uses an 8-by-`cores/8` grid, and allocates `[32,kw*32]` activation shards. With k=3072 and cores=64:

```text
Kt = 3072 / 32 = 96
S = 64 activation storage cores
q = ceil(96 / 64) = 2 tiles per storage shard
storage capacity = S*q = 128 tiles, versus 96 logical/padded K tiles
```

Only the first 48 row-major storage shards contain K data. Allocating capacity for 128 tiles does not change either operand's K dimension to 128. The block filter requires `kw % block == 0` and `Kt % block == 0`; it never verifies the storage-core coverage contract.

The same layout construction is present in current `tt/multichip_decoder.py:197–218` and in the imported snapshot, `doc/multichip_decoder/initial_multichip_decoder.py:181–202`. Neither contains an exact-coverage guard or role-specific input-core fallback. The original-input note explains that production collective code changed after import; the sweep's failing operation calls `ttnn.linear` directly, so those collective changes do not explain it.

**The selected implementation changes with readers.** `matmul/device/utilities/matmul_utilities.cpp:470–491` selects the alternate pipeline only when `use_multicore_pipeline` is true and readers >=2, together with its tensor/operation eligibility checks. The sweep sets that flag to `readers > 1`. `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:1843–1876` dispatches accordingly. Four/eight readers on these full-tile, no-bias calls use the alternate pipeline, which reads activations directly from their shards and accepts storage capacity >= K (`:1412–1431`). One reader uses the semaphore-gated multicast implementation. The three captured kernel names belong to that implementation. The helper's `[1,3]` reader limit belongs to the older assignment path and is not evidence that the passing four/eight-reader alternate-pipeline cases were invalid.

**The host/kernel coordinate-table contract is violated.** In `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`:

- `:253` sets `num_blocks = Kt / block = 96`.
- `:297–310` recognizes the two-tile shard but chooses `shards_per_block=1` for block width 1.
- `:382–388` checks coverage only when a block spans multiple shards. On the current branch it sets `num_blocks_per_shard = num_blocks / S = 1`, with integer truncation.
- `:599–645` sorts the actual 64 storage cores row-major and supplies varargs `[x[0..63], y[0..63]]`. The vararg count is 128.

In `reader_bmm_tile_layout_in0_sender_dram_sharded.cpp:99–131`, the kernel reconstructs:

```text
num_storage_cores = num_blocks / num_blocks_per_shard = 96
sender_noc_y_base = num_storage_cores = 96       # actual host y-base is 64
sender_core_of_block(0) = 0
```

At the first block, the seven non-source compute cores execute the atomic increment at `:332–333` or `:352`. They read vararg 0 for X, correctly `x[0]`, but vararg 96 for Y, which is actually **`y[32]`**. Storage index 32 is logical `(0,4)` on the 8-wide grid. Thus each atomic targets `(x[0], y[32], 0xab40)`, exactly the semaphore containing seven credits in the capture. This is an intra-device NOC1 atomic to a valid but wrong core, so successful delivery does not require a NoC integrity failure. The intended sender at `(0,0)` cannot reach line 303's multicast or publish CB0.

This is more than a wrong Y-base: one-block-per-shard scheduling would assign only 64 producer blocks against 96 consumer blocks, map block 1 to the next shard rather than the second tile of shard 0, and eventually index beyond the actual sender-coordinate list. Fixing only the coordinate offset would leave the data and loop-count contract broken.

**Compiled artifacts confirm these are the running arguments.** `generated/inspector/kernels.yaml` associates IDs 394/395/396 with program **242**; `programs_log.yaml:5106–5145` records its compilation at approximately 19:25:30.349–.580 UTC. Under:

```text
/home/ttuser/ornith-9b/tt-metal-cache/tt-metal-cache2778076107680524023/kernels/
```

the existing generated headers contain:

| Kernel directory / `kernel_args_generated.h` | Relevant values |
|---|---|
| `reader_bmm_tile_layout_in0_sender_dram_sharded/18251237559923876113/` | `in0_block_w=1`, `num_blocks=96`, `num_blocks_per_shard=1`, `shards_per_block=1`, `in0_shard_width_bytes=4096`, `in0_mcast_num_dests=8`, `in0_mcast_num_cores=80` |
| `reader_bmm_tile_layout_in1_sender_dram_sharded/6326850593012264857/` | `workers_per_bank=1`, `in1_block_num_tiles=16`, `in1_buffer_blocks=3`, `in1_page_size=8704`, `in1_num_pages=2` |
| `bmm_large_block_zm_fused_bias_activation_metal2/5927651034915080090/` | `num_blocks_inner_dim=96`, `in0_block_num_tiles=1`, `in1_block_num_tiles=16`, `last_subblock_w_valid=4` |

The matching binding header assigns sender/receiver semaphores to IDs 0/1. These headers were only read. The multicast rectangle contains 80 logical cores, whereas 66 cores have storage or compute roles: 64 storage + 8 compute - 6 overlapping. Its rectangle size is not the receiver-ready count; only the eight compute cores supply credits. For block 0, the source is one of those eight, so exactly seven remote credits are required.

**Producer/consumer ledger for this call:**

| Resource | Producer and expected production | Consumer / protocol | Captured consequence |
|---|---|---|---|
| Resident activation shards | Op 308 supplies row-major shards, two K tiles per core, 48 populated cores | NCRISC reads shard data; no resident-shard FIFO credits | Storage shape and sender scheduling disagree before first multicast |
| Sender semaphore 0 | Each other compute core adds 1 after reserving CB0: 7 increments for a compute sender, 8 for a storage-only sender | Current source waits for exact count, then resets to 0 | All 7 first-block increments reach `(0,4)`; `(0,0)` stays 0 |
| Receiver semaphore 1 | Source multicasts VALID after activation data; receiver resets INVALID before each request | NCRISC waits VALID, then pushes CB0 | Eight compute cores remain INVALID at first block |
| CB0 / in0 | NCRISC should push one tile for each of 96 blocks on every compute core | Compute waits/pops one activation tile per K block | Zero pushed; first compute input absent |
| CB1 / in1 and TRIDs | BRISC reads 16 tiles per block, 96 blocks = 1536 tiles per compute core; 3 buffer slots/tags, rotating 1→2→3 | Compute consumes 16 tiles per block after CB0 arrives | Two blocks pushed (32/48 pages); reserve of another 32 pages cannot succeed with no pops. Stop is CB capacity, not a TRID barrier |
| CB3 / partials | Compute accumulates K-block results using a 16-page FP32 partial buffer | Compute reloads/accumulates and eventually emits final output | No partials published; math/pack pipeline cannot advance without unpack |
| CB2 / output | Compute should emit 16 BF16 tiles per compute core after K reduction | BRISC waits and writes them into output shards | Zero pushed; output writes have not reached their completion phase |
| Semaphore 2 | Initialized to VALID | No kernel binds it in this factory (`:579–587`) | Its value 1 supplies no readiness credit |

**Why the guards miss it.** `matmul_device_operation.cpp:1749–1763` checks K/block divisibility and shard-width/block compatibility. Its output-spec branch at `:3110–3115` checks `Kt % shard_width == 0`; 96 % 2 passes. Neither checks `storage_core_count * shard_width == Kt` for this multicast branch. The factory's existing exact-coverage check applies only to `shards_per_block > 1`, so it also passes this case.

**Passing contrast.** At 32 activation cores, down projection has three tiles per shard: `32*3=96`, and block 1 gives `num_blocks_per_shard=3`, reconstructed coordinate count 32, and the correct Y-base 32. Block 3 gives one block per shard and remains consistent. At 16 cores, `16*6=96` is also exact. Earlier K=4096 projections have 128 tiles, so 64 cores with two tiles each are exact as well. These distinctions explain why the sweep fails only upon reaching the single-reader 64-core down-projection case; neither BFP8 weights nor LoFi arithmetic explains the address/count mismatch.

## Downstream Effects

The dependency chain is: **excess activation-core geometry → truncated shard/block schedule → readiness atomics sent to the wrong core → first activation multicast never issued → CB0 empty → unpack waits → math/pack wait → weight-reader CB backpressure → host completion waits**. All four chips receive the same invalid geometry, so identical failure on each is expected without cross-chip synchronization.

This isolated sweep performs no CCL operation in the tested forward call. Fabric routers run their persistent polling/credit routines, but the evidence identifies no payload send-slot, route, or connection failure preceding the local matmul producer. Dispatch waits, Ethernet counter mismatches, and capture-time broken-core messages do not explain the exact seven-credit displacement and should not displace it as the diagnosis. The run has not reached mesh close/teardown in the source control flow.

The output-memory warnings are separate: `matmul_device_operation.cpp:3124–3132` builds output shards in row-major order over the device's full compute grid and `:293–299` explicitly warns that it uses the computed configuration. This occurs in passing cases too. The final 64-shard warning helps identify the iteration but does not account for missing activation credits. The model should respect the returned output memory configuration rather than infer that its requested 8-wide output rectangle was retained.

## Proposed Fix

**Within authorized model/test scope:**

1. In `tt/multichip_decoder.py`, plan activation storage independently of output storage. For calls using the multicast implementation, require exact activation coverage: `input_core_count * input_shard_width_tiles == Kt`, with the existing compatible block-width rules. Use the already passing **32-core `[32,96]` activation layout for down projection** when a requested 64-core layout would overcover K. The existing default readers=4, cores=32 already works; retain it. A 48-core `[32,64]` activation layout is algebraically exact too, but has no passing evidence here and is not the first repair candidate. Do not indiscriminately force exact N coverage on the output layout, which follows a different contract.

2. In `tests/multichip_geometry.py`, apply the same input planner or reject/record the unsupported single-reader case **before matmul dispatch**. With the current test's layout definition, skip `(K=3072, cores=64, readers=1)` rather than hang before the later reader candidates. Keep reader labels and actual activation/output core counts separate if a requested layout is adapted. Log each candidate and phase before execution and retain results incrementally. Do not globally remove four/eight-reader cases: their qualifying alternate pipeline is distinct and already passes at 32 cores.

3. Apply the guard to every role that can reach the multicast path, not just this weight name. For example, attention output K=1024 has 32 K tiles and cannot be represented exactly by 64 one-tile activation shards on this path. Check pipeline eligibility as well as the requested reader count if future callers add bias, fused activation, or other features that cause fallback.

These changes preserve the real BFLOAT8_B weights, BF16 activations, LoFi MLP compute, TP4 partitioning, and collective choices. Simply increasing the requested shard allocation or changing only block width to 2 does not repair the 64-core/K96 multicast contract. If preserving a 64-core input is mandatory, either select a validated qualifying multi-reader pipeline, or explicitly pad **both activation K and each corresponding weight K** from 3072 to 4096 with zero tails before conversion, yielding Kt=128 and exact `64*2` coverage. The latter requires model-local setup/decode changes and extra work/traffic; it is a secondary option, not an already tested result.

**Shared-op limitation, outside repair scope:** the multicast factory currently cannot correctly schedule this overcovered activation grid. Upstream should at least reject it with an exact-coverage validation error before dispatch. Full support would require consistent actual storage-coordinate counts, per-shard valid block ranges, and correct roles for padding-only storage cores, so no sender waits for a block that consumers never request. Updating only `sender_noc_y_base` or replacing one integer division is insufficient. No shared-source edit is proposed for this stage.

**Smallest future discriminating check:** after the live owner's existing run has ended and a separate hardware check is authorized, run a targeted down-projection control with the same weights, precision, block=1, readers=1, and requested 64-core output, changing only activation storage to 32 cores of `[32,96]`. Check completion and PCC against the same reference, followed by trace/replay correctness. This isolates the input geometry while holding output geometry fixed. Existing cores32/block1/readers1 is already a completed correctness/timing control. Verify the model planner/test rejects the original 64-core single-reader input without dispatch; do not deliberately reproduce its hang. The 48-core and padded-K options need their own validation if selected.

## Uncertainty

- Confidence is high in the first-block diagnosis: compiled arguments, source indexing, stop sites, and the four identical misplaced semaphore counts agree. Live runtime varargs were not reread from devices; their layout is derived from the factory and corroborated by the exact observed target `(0,4)`.
- The root-level `AUTOTRIAGE_INPUT.md` is absent; the supplied context was found at `doc/multichip_decoder/AUTOTRIAGE_INPUT.md`. Its 132-row count is correct, but its claimed last row and the matching work-log entry are stale/incorrect relative to the actual JSON and compiled arguments.
- No device command, target execution, debugger attach, process signal, reset, or new triage capture was performed. Host observations used read-only `/proc` files. The only investigation output written is this report; implementation and tests were not edited.
- No hardware repair validation was performed. In particular, completion/PCC/performance of an adapted input with the original 64-core output, 48-core storage, padded K, or later 64-core multi-reader candidates remains unverified. The sweep stops before recording those later cases. A full decoder check must follow any eventual model change.
- The detailed capture is not an atomic machine-wide snapshot and reports halt-state anomalies during collection. It cannot prove every historical Ethernet counter benign or eliminate independent hardware issues. Those uncertainties do not explain the source-predicted seven credits at the wrong core.
- Generated inspector/cache files are supplementary local artifacts and may later be replaced or evicted. Their exact kernel IDs, paths, and relevant argument values are recorded above to retain the evidence distinction from the source snapshot.
