# AutoDebug: tiled convolution history versus warmed serial prefill

## Finding

**The supplied failure is real, but this source-only audit did not establish a numerical root cause or a justified corrective patch to `read_linear_state` / `write_linear_state`.** In particular, the cache-user/request-axis mapping is correct, invalid input rows are excluded by the current kernels, and the current program-cache callback refreshes `update_idx` correctly.

The earliest demonstrable discrepancy is in the comparison harness: it changes collective ownership as well as history storage, and its execution/completion schedule differs from the passing stack test. It checks only the final repetition, so the evidence does not identify the first bad request, repetition, tensor, or operation. The most useful next experiment is a second production RM decoder in the same separately constructed instance position as TILE, followed by checks that distinguish queued execution, output retention, and state restoration.

There is also a **separate, concrete KV-cache program-key defect**: `compute_kernel_config` is omitted even though it controls compiled formats and buffer sizes. The reported candidate always uses the same default configuration, so this is **not an established explanation of this failure**. Details and a minimal independent fix are below.

Production was not changed. This investigation ran no model, hardware, reset, or reproduction command. Only this report was written.

## Evidence and revision discipline

Repository HEAD at inspection: `b8d8d43d8e3deb6fc772000adbb6f160cf81e863`. The checkout has existing modifications and untracked model evidence. The recorded failing command's Python hashes were checked against the archived snapshots, not assumed to describe the current working files.

For line references below:

- **C** is the [exact candidate snapshot](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/source_snapshots/multichip_tiled_history_candidate_463d38122e7d.py.txt), SHA256 `463d38122e7db734ddd0b1e883fa0e65aefdda5dd6f79bba1cd182ee4edb95f3`.
- **D** is the [exact production decoder snapshot](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/source_snapshots/multichip_decoder_da8f4bd47467.py.txt), SHA256 `da8f4bd474675698af4e162ddbc0afb9e9df17467a7a41480ec17b8913715d8f`.
- **H** is [the prefill harness](models/autoports/ornith_ai_ornith_1_5_9b/tests/multichip_history_prefill.py), SHA256 `53777fe9c2088d65974a03c4724a938e6bb07117dab9741efcc2232703c3e0c0`, matching the recorded command and archived snapshot.
- **S** is [the passing stack harness](models/autoports/ornith_ai_ornith_1_5_9b/tests/multichip_tiled_history_stack.py), SHA256 `f7b9cb2cfa0ab7be32f9ed4a5270e2ffe46b747092a8596047d2bb4867e65b39`, also matching the recorded commands.

The candidate working file has a later import change; production has later full-attention changes. Neither was spliced into the failure's causal chain. Current tracked KV-cache C++ is unchanged relative to HEAD, but the command archive does not hash the loaded C++ library or these kernel/helper sources. Its exact historical binary correspondence remains unverified. The log's JIT build-cache hits are not evidence that runtime arguments were stale.

Direct observations from [the command](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/history_prefill_batch32_command.json), [log](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/history_prefill_batch32.log), and [partial result](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/history_prefill_batch32.json):

| Case | Output PCC | Contribution PCC | Rank agreement | Result |
|---|---:|---:|---:|---|
| B32, S32, private | 0.99999994 | 0.99999994 | 0.99999994 | Pass |
| B32, S32, pooled RM | 0.99999994 | 0.99999994 | 0.99999994 | Pass |
| B32, S32, pooled TILE | 0.91366011 | 0.90486729 | 0.93253678 | Fail |

The harness defaults to lengths `32,33,1024,1025`, but stops at the first assertion. There is no result here for the latter three lengths or for warmed B6 prefill. The failed output is from repetition 17 of 18, with three warmups and 15 timed repetitions; earlier repetitions were not scored.

Both passing chunked-stack commands name the same C/D hashes as the failing prefill command:

| Batch | Minimum prefill PCC | Minimum decode PCC, 16 steps | Conv-state PCC | Recurrent-state PCC |
|---|---:|---:|---:|---:|
| 32 | 0.99990076 | 0.99960572 | 0.99999702 | 0.99999100 |
| 6 | 0.99989927 | 0.99982810 | 1.00000000 | 0.99999499 |

Sources: [B32](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/tiled_history_chunked_32.json), [B6](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/tiled_history_chunked_6.json). Those tests also run serial prefills, so it would be wrong to say the updater/read adapter was never tested. Their steady-state pooled decode bypasses those adapters, however, and their prefill completion schedule differs substantially.

## Earliest differences in the harness

### Collective ownership is not controlled

H:32–33 constructs two decoder instances without `collectives=...`. D:136–147 consequently creates separate CCL managers and workspaces. H:39 then compares private and pooled RM using **the same first decoder**, before testing TILE using **the second decoder**. The passing stack explicitly shares a `MultichipCollectives` instance across layers (S:174–195).

Thus the observed contrast is not solely RM versus TILE: the tiled arm also first exercises a second instance's collective resources after the first instance has warmed same-shaped operations. Compared with the passing shared-workspace stack, the extra workspace changes live L1 reservations and allocation placement, even though S32 prefill does not use its decode receive buffers. Both workspaces are constructed before any measured arm, so this is not evidence of a new L1 reservation appearing exactly when TILE starts.

This is a concrete experimental confound, **not proof of a CCL bug**. The inspected C++ does refresh caller-owned semaphores:

- `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp:719–731,817–828` refreshes barrier/sem0/sem1 in reader and writer; tensor addresses use declared buffer bindings.
- `ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_program.cpp:497–523,2064–2082` refreshes the linear reduce-scatter common arguments, including tensor and semaphore addresses.
- `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:703–734` applies bindings and invokes these overrides on cache hits.

The simple story that the second decoder necessarily retains the first decoder's semaphore addresses is contradicted by this source.

### Completion, retention, and scoring are different

S:207–220 downloads/checks each request's prefill immediately, imposing a completion boundary before the next request. H:46–59 restores state, synchronizes, queues all 32 prefills while retaining their returned tensors, then synchronizes once. It discards repetitions 0–16 and downloads only repetition 17.

The reset copies finish before timing begins (H:47–49), and all queued work finishes before final downloads (H:53,59). Merely returning from each `prefill_forward` does not establish that its device work completed before the next request was queued.

There is no demonstrated output alias in the Python path: D:623 returns a new residual add, and D:930 stores the prefill output in DRAM. For S32, D:387's S=1 branch is skipped, so these reductions do not directly return the reusable decode `ar_buffers`. A lifetime/reuse or asynchronous hazard remains a hypothesis, not a finding.

## Read adapter: no demonstrated invalid-row promotion

C:21–27 intends the identity `local[0,0,tap,channel] = pool[tap,0,request,channel]`.

| Stage | Logical shape | Padded shape |
|---|---|---|
| Persistent TILE pool | `[3,1,B,2048]` | `[3,1,32,2048]` |
| Selected request, TILE | `[3,1,1,2048]` | `[3,1,32,2048]` |
| Permuted TILE | `[1,1,3,2048]` | `[1,1,32,2048]` |
| Returned RM | `[1,1,3,2048]` | `[1,1,3,2048]` |

Request zero takes the tile-aligned slice path. It can carry the other requests' bytes in its **invalid physical rows**, but restores logical height one. Nonzero requests have unaligned height starts, so slice converts the full pool to RM, slices it, then restores the original TILE layout. Relevant source: `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:185–194,224–235,378–382,385–446`.

The `(2,1,0,3)` permute selects the interleaved tiled row-invariant factory and allocates a separate output (`.../permute/device/permute_device_operation.cpp:36–45,169–175`). Its writer uses logical source height one, places the three taps into output rows 0–2, and separately zero-fills output rows 3–31 (`.../permute/device/kernels/dataflow/writer_permute_interleaved_tiled_row_invariant.cpp:120–125,193–353`). The final `to_layout` unpads using logical endpoints (`ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:169–182`).

Consequently, neither B32's other requests nor B6's padded request rows are shown to leak into valid local history. B32 nonzero requests can additionally select a codegen untilize route for the full-pool conversion; B6 requires unpadding. The exact runtime route is not recorded. The inspected planner includes its selected buffering tier in its hash; no concrete stale-plan error was found.

D:689–703 reshapes the compact result to RM `[1,3,2048]` and passes it as history and predecessor carry. That is valid for local convolution; this TP mesh's default SP axis has size one. The reader consumes the caller's history at the segment start (`.../experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/reader_qkv_causal_conv1d_silu.cpp:149–168`). No missing history transpose or predecessor distinction was established.

## Write adapter and lowered cache contract

C:30–40 converts local history to TILE, splits channels into eight 256-wide blocks, updates each block, concatenates the updated blocks, and copies into the persistent pool. Crucially, **cache users are taps; cache sequence positions are requests**. The input should remain `[1,1,3,256]`; transposing it to `[3,1,1,256]` would violate the cache API.

For both B32 and B6, every call lowers to:

| Quantity | Value |
|---|---|
| Cache logical / padded | `[3,1,B,256]` / `[3,1,32,256]` |
| Input logical / padded | `[1,1,3,256]` / `[1,1,32,256]` |
| `Bcache`, `Binput`, `Wt` | `3`, `32`, `8` |
| Granularity / batched heads / cores | `1` / `1` / `1` |
| Cache tiles / tiles per tap | `24` / `8` |
| `update_idx` | Request row `r`, not tap index |
| Replacement byte width / destination row offset | `512` / `r * 512` |

Validation agrees with these shapes (`ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_device_operation.cpp:45–58,130–152`). Factory geometry uses `Bcache=3` for the actual user count (`.../update_cache_multi_core_program_factory.cpp:158–178,455–477`). `kernels/update_cache_tile_work.hpp:11–17` produces three user iterations, not 32.

The exact queue ledger is balanced:

1. Reader produces eight input tiles once, then three cache blocks of eight tiles, at tile IDs `0..7`, `8..15`, `16..23`.
2. Compute untilizes that input once; for each tap it untilizes eight cache tiles and retilizes the edited block.
3. Writer reads only input rows 0,1,2, overwrites row `r` of each tap, consumes three output blocks, then drains the input block. Rows 3–31 of the input are never used as cache users.

Sources: `kernels/dataflow/reader_update_cache_interleaved_start_id.cpp:47–83`, `kernels/compute/update_cache.cpp:30–40`, `kernels/dataflow/writer_update_cache_interleaved_start_id.cpp:48–101`.

For `r=31`, the written extent within each eight-tile BF16 block is `31*512+512 = 16384` bytes, exactly its capacity. For B6 the highest logical request is `r=5`; padding remains untouched. No out-of-bounds access was shown for either case.

### Cache hits and Blackhole helpers

The operation hash excludes `update_idx`, but the cache-hit callback recomputes the cache tile start, row byte offset, input offset, and current tensor bindings (`update_cache_multi_core_program_factory.cpp:502–537`). The framework applies these named runtime arguments as partial updates (`tt_metal/impl/metal2_host_api/program_run_args.cpp:1604–1623,1721–1756`). With `r<32`, tile start stays zero while offset changes from 0 through 15872. A frozen first-request offset is not supported by the current source.

The BF16 width-eight Blackhole path selects fast untilize and fast tilize, each split into `4+4` tiles. Formats are BF16 throughout under the default compute configuration. The helper lifecycle performs unpack/pack reconfiguration and init/uninit; the math/pack reset and remap helpers explicitly wait for pending work. The two aliased 16-tile intermediate FIFOs alternate two eight-tile slots. The writer completes its row copy before publishing the edited alias. No specific fast-path, format-state, or alias-handshake violation was established. This is a static audit, not runtime proof of the kernels. Supporting implementations include `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.inl:168–194,277–280`, `tilize_helpers.inl:87–140,196–204`, and `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:70–82,152–170`.

### The earlier L1 failure is accounted for separately

Current default-BF16 buffer backing, counting the aliased intermediates once:

| Allocation | Width2048 | Width256 |
|---|---:|---:|
| Cache-input FIFO | 262144 B | 32768 B |
| Update-input FIFO | 262144 B | 32768 B |
| Aliased intermediate pair | 262144 B | 32768 B |
| Untilized update input | 262144 B | 32768 B |
| Output (`Binput * Wt` tiles) | 4194304 B | 524288 B |
| **Total** | **5242880 B = 5 MiB** | **655360 B = 640 KiB** |

Factory source: `update_cache_multi_core_program_factory.cpp:174–269`. The original [B32 L1 log](models/autoports/ornith_ai_ornith_1_5_9b/doc/optimized_multichip_decoder/tiled_history_32.log) reports an end address of 5355456 B versus L1 size 1572864 B. The five-MiB backing plus lower reserved memory is consistent with that failure; the comment's “5.36MiB” is not the exact allocation formula. Chunking reduces the backing eightfold. It does not prove or explain the subsequent correctness failure.

## Reset, aliasing, and rank agreement

H:43–44 deduplicates shared tensor handles by Python identity and clones their buffers. `clone` allocates a new tensor (`ttnn/cpp/ttnn/operations/data_movement/clone/device/clone_device_operation.cpp:65–68`). H:47–49 copies the snapshots back and synchronizes before each repetition. C:40 copies into the existing convolution pool rather than rebinding individual states. Recurrent pooling uses a sharded input and `slice_write` to preserve the destination allocation (D:499–510). No obvious missing buffer, duplicate reset, or snapshot alias was found.

S32 has no padded sequence-token tail: D:691 and D:721's `n % 32` branches are skipped. Candidate pooled-decode switches such as `fused_pad`, `fused_conv`, and head-wave geometry are not on this prefill path.

Rank disagreement constrains the explanation: an isolated wrong convolution calculation with otherwise correct memory and collectives should still produce replicated final residuals. However, it does **not** clear candidate state writes. The attention result is reduced at D:784, then C:49–50 performs pooled state writes, then D:930 invokes `finish`. An invalid write could corrupt the already reduced intermediate differently on each rank. No such invalid write was proven, and the log provides no failing request/rank correspondence beyond minima.

## Minimal interventions and discriminating experiments

### Correct the comparison before proposing a numerical fix

The smallest concrete harness correction is to share the same setup-owned collective workspace, as the passing stack already does:

```python
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import (
    MultichipDecoder, MultichipCollectives,
)

collectives = MultichipCollectives(mesh)
layers = [
    cls.from_state_dict(
        sd, hf_config=config(), layer_idx=0, mesh_device=mesh,
        collectives=collectives,
    )
    for cls in (MultichipDecoder, TiledHistoryCandidate)
]
assert layers[0].ccl is layers[1].ccl
```

This removes an uncontrolled variable; it is **not a proven correction for PCC .91366**. Sharing also changes allocation placement, so success alone would not identify stale semaphore bindings. Keep production RM unchanged. There is insufficient evidence to recommend changing request indices, transposing cache input, clearing padding, or adding blanket synchronization as a production fix.

### First: falsify the claim that TILE is necessary

Use the archived C/D/H versions and the original separate-workspace setup. Replace the second decoder with a second **production** decoder and run its pooled RM state in the same third-arm position. Compare:

- Second-instance RM fails: TILE history is unnecessary to trigger the failure; investigate cross-instance warmed execution/placement first.
- Second-instance RM passes, TILE alone in a fresh mesh fails: focus on the tiled adapter under queued execution.
- TILE alone passes, but fails after private/RM: preserve cross-instance cache/collective/allocator history as a required trigger.

Start with B32/S32. Record every request and every repetition starting at zero, rather than interpreting the existing repeat-17 minimum as the first divergence.

### Second: separate execution ordering from returned-output retention

Run independent diagnostic arms, keeping the original no-per-request-sync arm:

| Arm | Change | Meaning if it alone restores correctness |
|---|---|---|
| Asynchronous snapshot | Clone each returned output immediately on device; synchronize only after all 32 calls | Suggests output lifetime/reuse or allocation-sensitive behavior; cloning also perturbs placement/timing |
| Completion boundary | Synchronize after each request while retaining all original outputs | Establishes an ordering/timing dependence; does not identify the responsible operation |
| Early/late output comparison | Download output 0 after request 0, then again after requests 1–31 | A change proves later work modifies a live retained result; account for the synchronization perturbation |

Capture output and pooled-state buffer addresses, per-request/per-rank scores, repetition numbers, and program-cache entry counts outside timing. Check snapshots and destination zero state immediately after restoration in a separate diagnostic run. Reading state after every request introduces synchronization and can mask the original trigger; do not use that as the only check.

### Third: isolate the adapter without model arithmetic

Use both B32 and B6. Populate three taps with exact BF16 sentinels varying by tap, request, channel block, and rank. Queue writes to requests `0,31,7,1,30,12,0,31` for B32, and an in-range shuffled equivalent for B6. Include overwrite, zero reset, and 18 reset/write rounds. Queue checks/snapshots without downloads until the end; compare the full pool independently on the host, including untouched requests and physical padding where possible. A test that downloads after every write does not reproduce the failing completion schedule.

Compare read extraction against a direct full-pool download, rather than validating only write/read round trips through the same adapter. A useful read-only control is:

```python
pool_rm = ttnn.to_layout(state.conv, ttnn.ROW_MAJOR_LAYOUT)
selected_rm = pool_rm[:, :, row : row + 1, :]
local_conv = ttnn.permute(selected_rm, (2, 1, 0, 3))
```

This removes row zero's tiled slice path and the tiled permute/retilize steps. It is a diagnostic implementation, not a justified fix. If the queued sentinel test passes but the model fails, snapshot the attention result immediately before candidate state writes and after them; preserve both on device until the end. That directly tests whether writeback damages a live intermediate before `finish`.

Only after B32/S32 is understood should the original remaining lengths `33,1024,1025` and warmed B6 be treated as regression cases. Their failure or success is currently unknown.

## Other potential issues: real compute-config cache-key defect

`UpdateKVCacheOperation::compute_program_hash` hashes only operation type and the two tensor specs (`ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_device_operation.cpp:178–181`). The framework's collision-resolution key does not rescue the omitted fields: operations with a custom hash receive only the operation-type prefix (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1320–1333`). `compute_kernel_config` controls `fp32_dest_acc_en`, the intermediate format and tile byte sizes, FIFO allocations, and compiled hardware configuration (`update_cache_multi_core_program_factory.cpp:150–156,219–269,356–413`). The hit callback changes `Wbytes` using the new configuration but cannot rebuild those compiled resources (`:513–537`).

Two otherwise identical BF16 calls using `fp32_dest_acc_en=False` and then `True` can therefore reuse a program compiled with 512-byte rows while the new writer offset/copy width assumes 1024-byte rows. That is a concrete key/compiled-contract inconsistency.

**Minimal independent fix:** include the normalized effective `compute_kernel_config` in the program hash; continue excluding runtime-only `update_idx` / `batch_offset` and refreshing them through the callback. A focused follow-up should alternate configurations in both orders with the program cache enabled, verify untouched rows, and compare against fresh-cache calls.

**Why secondary:** C:38 passes no compute config; `kv_cache.cpp:30–38` normalizes the same default each time, with FP32 destination accumulation disabled. Neither this harness nor its preceding private/RM arms supplies an alternating configuration to ordinary `update_cache`. The necessary collision precondition is not evidenced. Fixing this generic defect cannot honestly be presented as the established repair for the supplied failure.

## Report review

The first-draft claims were checked against the exact archived Python and current lowered implementations. Padding leakage, a missing transpose, frozen `update_idx`, duplicate snapshot clearing, direct decode-buffer aliasing, stale second-instance semaphore bindings, and an obvious Blackhole fast-helper handshake error did not survive as supported headline diagnoses. The compute-config defect remains separate because its trigger is absent from the recorded path. The numerical failure remains unresolved pending the controlled experiments above; no hardware-dependent claim is reported as proven.
