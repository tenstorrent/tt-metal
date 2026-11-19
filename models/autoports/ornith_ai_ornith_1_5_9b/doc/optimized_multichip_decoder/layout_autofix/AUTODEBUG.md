# AutoDebug: the decoder plans requested shards, but matmul produces different shards

Inspection date: 2026-11-12. Checkout HEAD: `b8d8d43d8e3deb6fc772000adbb6f160cf81e863`.

**Leading finding:** this is a caller/automatic-output-layout mismatch that causes avoidable tensor movement. The automatic DRAM-sharded matmul output planner ignores the requested shard grid and derives storage from tiled M/N and the full device grid. The decoder subsequently converts that actual output to its separately planned consumer layouts. The 48-core result for N3072 is correct for `per_core_N=2`; it is not evidence of lost columns or 48 compute workers. No numerical matmul bug is established by the supplied evidence.

**Smallest repair supported by inspection:** keep the existing 8×4 residual/norm/all-reduce output layout; let gate/up and full-attention packed-input matmuls read that layout directly; let down-projection read the actual 48-core gate/up layout; pass row-projection output directly to the existing all-reduce, which already maps different input/output grids. Preserve this resident output between layers. This needs caller changes, not a matmul kernel rewrite. Runtime correctness and speed of these changes remain unverified; no source code was edited and no devices were run.

## Scope, observations, and provenance

- Focus: `models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py`, its grouped batch candidate, DRAM-sharded matmul, and immediate norm/binary/collective consumers. No full-model or vLLM investigation.
- This investigation left source and existing stage-five artifacts untouched; its only write was this report. Checks used source reads, hashing, AST comparison, JSON/log parsing, and standalone integer arithmetic; none imported TTNN or opened devices. Other work changed production/test files in the shared workspace during report review; historical and live claims are separated below.
- The recorded `batched_candidate_32.log` has 960 layout-mismatch warnings and ends with `STACK_BATCH_PASS 32` (line 981). Its JSON records batch32, replicated residuals, shared collective workspace, layers `[0,3]`, four checked trace replays, 64 prefill comparisons, 128 decode comparisons, and minimum decode PCC **0.9997884035**. The recorded median stack time is **9.061573 ms**; this is historical evidence, not a fresh benchmark.
- Candidate SHA256 is `1152504b1ccca6bc45a488b191b5c11cc1c85ea0e961bf1b6d03c587c8bc7692`; the decoder at investigation entry had SHA256 `bc8386dc4e5c8f6fb18f24addca003e9cd37efe2b57341e6be9b06948aef88fc`. Both match the JSON. The unchanged `tests/multichip_stage4_reference.py` and `doc/optimized_multichip_decoder/baseline/multichip_decoder.py.txt` retain that decoder hash. This matters because `BatchedCandidate` inherits `Candidate`, which imports the stage-four snapshot rather than the production module (`tests/multichip_optimization.py:8`).
- **Concurrent production update:** by draft review, `tt/multichip_decoder.py` had SHA256 `8dbed3eeecdffdb62ad5ec03f25dbd2628625cbe8f331c9733433417f0d8a95e`. It introduces `decode_linear`, `decode_reduce`, `decode_norm`, and `decode_batch_packed`; its `decode_forward` now returns resident output. This is not the source hash recorded by the supplied pass. In this report, **DEC references intentionally name the unchanged stage-four snapshot**, which is the exact inspected baseline; **LIVE** names the newer production file as observed at this hash. Matmul/AR source was unchanged during review.
- The harness waits for completion: in `tests/multichip_stack.py`, the four-step checked-replay loop uses `execute_trace(..., blocking=True)` and downloads every output; the benchmark loop brackets nonblocking replays with synchronization. The pass is stronger than merely reaching the end of a host enqueue function. It does **not** prove zero movement: the tested path explicitly contains conversions, and the recorded allocation-tracking setting is `0`. Harness function/loop references are used where concurrent edits shifted its line numbers.

Direct observations are the warnings, recorded pass/results, matching file hashes, and code below. The repair and performance implications are deductions from those contracts. No speedup or new hardware pass is claimed.

## Source reference key

All paths are relative to this checkout. References such as `MM:3124` mean the indicated file and line, not a log interpretation.

| Key | Source |
|---|---|
| DEC | `models/autoports/ornith_ai_ornith_1_5_9b/tests/multichip_stage4_reference.py` (recorded baseline; initially identical to production) |
| LIVE | `models/autoports/ornith_ai_ornith_1_5_9b/tt/multichip_decoder.py` (concurrent update, hash above) |
| BATCH | `models/autoports/ornith_ai_ornith_1_5_9b/tests/multichip_batched_candidate.py` |
| MM | `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp` |
| CFG | `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config_types.hpp` |
| MF | `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp` |
| MK | `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_dram_sharded_multicore.cpp` |
| ARV | `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.cpp` |
| ARF | `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_factory.cpp` |
| LN | `ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp` |
| BIN | `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/binary_ng_utils.cpp` |

## Exact calculation: requested cores are not produced cores

`DEC:250-284` constructs rectangular width-sharded configurations:

```text
requested output cores C = 32 or 64
ow = ceil(N_elements / (32*C))
requested output = 8 × (C/8) cores, shard [32, ow*32], ROW_MAJOR, L1
program = DRAMSharded(in0_block_w=block, per_core_M=1, per_core_N=ow,
                      num_workers_per_dram_bank=readers,
                      use_multicore_pipeline=(readers>1))
```

`exact_k_shards` separately selects input storage that exactly covers K (`DEC:20-32,271-273`). It does not force the output grid. Projected weights are width-sharded across DRAM banks with whole K in each bank (`DEC:235-248`).

Without a supplied output tensor, the DRAM-sharded branch instead computes (`MM:3098-3140`):

```text
Mt = flattened padded activation rows / 32
Kt = padded K / 32
Nt = padded weight N / 32
num_blocks_y = ceil(Mt / per_core_M)
num_blocks_x = ceil(Nt / per_core_N)
num_output_cores = num_blocks_x * num_blocks_y
grid = num_cores_to_corerangeset(num_output_cores,
                               device.compute_with_storage_grid_size(), row_wise=True)
shard_shape = [per_core_M*32, per_core_N*32]
```

The requested **layout class and buffer type** survive; its grid and shard geometry are reconstructed. `num_cores_to_corerangeset` fills full rows and then a partial row (`tt_metal/common/work_split.cpp:107-124,147-149`). On this device, row width is 11, not 8. The warning merely reports the difference (`MM:293-299`); creation allocates the computed spec (`MM:3312-3316`). It performs no automatic conversion back to the requested grid.

The grouped input is `[1,1,32,K]`: 32 requests become **one M tile row**, not 32 independent matmul batches with different weights (`BATCH:70-95`). The same M-tile geometry applies to logical row counts 1–32. Validation requires all padded M rows in each width shard and `Mt == per_core_M` (`MM:1701-1718`).

Concrete default TP4 geometry, derived from actual weight construction (`DEC:192-233`) and role selection (`:262-282`):

| Role | K→N elements | Input storage, cores × shard width | Requested output cores | `per_core_N` | Actual output cores / shard |
|---|---:|---:|---:|---:|---:|
| Linear-attention packed input | 4096→3136 | 32×128 | 32 | 4 | 25 / `[32,128]` |
| Full-attention packed input | 4096→2560 | 64×64 | 64 | 2 | 40 / `[32,64]` |
| Gate and up, each | 4096→3072 | 64×64 | 64 | 2 | 48 / `[32,64]` |
| Attention row projection, either kind | 1024→4096 | 32×32 | 64 | 2 | 64 / `[32,64]` |
| MLP down projection | 3072→4096 | 48×64 | 64 | 2 | 64 / `[32,64]` |

For gate/up: `Nt=3072/32=96`, so `ceil(96/2)=48`. Forty-eight `[32,64]` shards cover all 3072 columns exactly. The decoder's nominal 64-core request has excess capacity, not a requirement that all 64 cores own logical columns.

Computed row-major coordinate sets on an 11-column device:

| Count | Core ranges, inclusive | Warning occurrences in supplied log |
|---:|---|---:|
| 25 | `(0,0)..(10,1)` plus `(0,2)..(2,2)` | 3 |
| 40 | `(0,0)..(10,2)` plus `(0,3)..(6,3)` | 189 |
| 48 | `(0,0)..(10,3)` plus `(0,4)..(3,4)` | 384 |
| 64 | `(0,0)..(10,4)` plus `(0,5)..(8,5)` | 384 |

These counts/shapes match the warnings, including 25-core `[32,128]` output. Frequencies mix warmup, per-request calls, and grouped calls; the log lacks tensor identities, so frequency is not used to invent a precise per-call mapping or latency attribution.

## Lowered path and why this causes movement

The selected calls have BF16 activations/output, BFP8 weights, untransposed 32×32 tiles, no bias or fused matmul activation, tile-aligned K, L1 width-sharded A/output, and readers 2 or 4 (`DEC:76-79,143-150,262-282`; `BATCH:19-26`). They qualify for the multicore pipeline predicate in `device/utilities/matmul_utilities.cpp:470-491`. The SiLU is attached to the later multiplication, not the matmul.

The factory chooses bank-adjacent compute workers independently (`MF:1371-1399`). It separately enumerates the **actual input and output storage grids**, converts their coordinates to device worker coordinates, and derives shard widths from those tensors (`MF:1401-1431`). The stream kernel receives `DSMC_XC`, `DSMC_OC`, `SW`, `OW`, and output capacity (`MF:1580-1598`). The writer selects `kOutCores[n/OW]` and within-shard offset `n%OW`, then completes an async-write barrier before releasing its output buffer (`MK:374-401`). Thus output-storage core count is not compute-worker count, and the produced layout is backed by the writer's coordinate contract.

The subsequent copies are real. `to_memory_config` returns the same tensor only for an equal config with no explicit output buffer (`operations/core/to_memory_config/to_memory_config_op.cpp:260-265`). A changed sharded config invokes reshard or a sharded/interleaved fallback (`:283-336`); sharded-to-interleaved is also an actual operation (`:354-356`).

| Boundary | Current source behavior | Why avoidable / constraint |
|---|---|---|
| Norm → gate and up | Norm emits 8×4 `[32,128]`; each linear converts to 8×8 `[32,64]` (`DEC:449-458,426-434`; `BATCH:19-26,49-58`) | Both hold K4096 exactly. Pipeline can read the existing 32 shards. Separate gate/up calls can each enqueue a conversion; no source-level sharing is present. |
| Norm → full-attention packed input | Converts the same 8×4 normalized stream to 64 input shards | Same K4096 contract; existing normalized shards suffice. Linear-attention packed input already requests the compatible 32×128 input. |
| Gate/up → SiLU-multiply → down | Multiply receives matching actual 48-core tensors; down converts result to its planned **8×6** 48-core grid | Both use `[32,64]`, but actual flattened coordinates differ. Down can read the producer coordinates directly. |
| Row projection → all-reduce | Converts actual 64×`[32,64]` to 8×4 `[32,128]` before AR (`DEC:355-368`; `BATCH:33-46`) | Direct AR can read the actual grid and produce the requested 8×4 grid itself. |
| Baseline layer return → next layer | Recorded baseline `decode_forward` forces DRAM interleaving (`DEC:712`), and next layer imports to `ar_mem` (`:705`) | Historical boundary movement, already avoided by grouped inner calls and now removed from newer production `decode_forward` (`LIVE:729`). Do not report this as still present in the newer packed path. |

Packed attention is a separate boundary: `BATCH:27` explicitly converts packed output to L1 interleaved for slicing/head/conv consumers; `DEC:435-439` used DRAM interleaved; newer `LIVE:451-455` uses L1 interleaved. Attention's local output also needs preparation for its row projection. The diagnosis does not claim those conversions disappear from changing residual/MLP layouts.

The central mismatch survives the concurrent refactor: role layout setup remains unchanged (`LIVE:251-285`), `decode_linear` still converts to each planned `im` (`LIVE:443-449`), and `decode_reduce` still converts to `ar_mem` before AR (`LIVE:372-387`). Its packed path calls those helpers (`LIVE:791-805`). Thus the source-supported MLP/pre-AR repair below applies to both snapshots without treating historical return behavior as current.

## Recommended bounded repair: consume the produced layout

Apply within the existing native-local replicated decode/grouped path: 1–32 logical rows, **one physical M tile row**, row-major width shards, compatible 32×32 BF16 tile/page layout for the AR input and output. Preserve other paths' contracts. The linear page-stream AR mapping below does not establish arbitrary multirow width repartitioning.

1. **Keep `ar_mem`, RMSNorm, residual adds, AR buffers, and global semaphores unchanged.** This preserves the existing layer boundary: 32 shards of `[32,128]`, BF16, 8×4, row-major.
2. **Use the normalized stream directly as A for K4096 projections.** Choose `ar_mem` as their input layout, or use the actual normalized tensor after checking its compatible L1/width/row-major/K contract. This covers gate, up, and full-attention packed input. Do not globally replace layouts for unrelated attention-local tensors.
3. **Retain gate/up's actual output through multiplication and down-projection.** Both producers have identical computed configs; with output inheriting that config, same-shape L1 binary uses native sharding (`BIN:922-959`). Down's K is exactly `48*64=3072`, so consume its actual config instead of rebuilding 8×6 coordinates. Prefer the producer's actual `memory_config()` to a second hand-maintained grid calculation, with assertions on the intended producer contract.
4. **Remove the pre-AR conversion only for compatible row-projection outputs.** Call the two-tensor direct AR overload with the original matmul result, existing buffer/semaphore, and `memory_config=self.ar_mem`. Keep dtype, topology, mesh axis, link count, and ping-pong order unchanged.
5. **Preserve the native resident return between layers.** `BATCH:95-98` already does this; `tests/multichip_stack.py`'s `forward` packs once, calls each layer's `decode_batch_packed`, and converts only after the entire stack. New production already returns resident output (`LIVE:729,781-805`); retain that behavior. Its list compatibility wrapper still converts through DRAM (`LIVE:807-816`), and `BATCH:100-104` still exports in its single-request wrapper. Selecting either exporting wrapper per layer restores movement; use the packed inner path for the stack and export only at its outer boundary.
6. **Make output metadata truthful after eliminating the copies.** Retain automatic output placement and consume its actual spec; use matching expected specs for assertions if desired. Replacing `om` with computed geometry or omitting a shard spec only removes warnings unless steps 2–5 are also done. Do not count that metadata-only change as the optimization.

The matmul validator requires A/output to share buffer type and layout class, not core coordinates (`MM:1576-1589,1684-1700`). The multicore reader enumerates arbitrary actual storage coordinates and checks `input_shard_width_tiles * input_core_count >= Kt` (`MF:1412-1431`). Proposed A coverage is exact: K4096 is 32×4 tiles; K3072 is 48×2 tiles. Neither requires a padded fake K or a nondivisible shard count. The proposed K4096 shard width also remains compatible with the existing block width 2 if a supported single-reader fallback is exercised.

### Why direct all-reduce does not require a preliminary reshard

The overload with `buffer_tensor` goes directly to `prim::all_reduce_async`, without the composite all-gather/reduce-scatter wrapper (`operations/experimental/ccl/all_reduce_async/all_reduce_async.cpp:541-572`). Its validation requires width-sharded input/buffer/output; buffer grid must contain the **output** grid; buffer shard volume must cover ring-size × output shard volume (`ARV:47-74`). Input/output grid equality is not required.

`ARF:301-302` separately enumerates input and output cores; `:322-385` assigns output page counts to input shard spans; `:544-575` builds separate actual input/output coordinate arrays. The reader accesses input coords (`device/kernels/dataflow/worker_reader.cpp:49-67`); the writer packs pages to output coords and stops at output-shard and packet boundaries (`worker_writer.cpp:119-163`). This is existing data redistribution inside the collective, not a claim of zero NoC traffic.

Concrete TP4 batch32 ledger with two links:

```text
hidden4096 = 128 BF16 tile pages; each page is 2048 bytes
matmul input-to-AR: 64 cores × 2 pages = 128 pages
AR output:         32 cores × 4 pages = 128 pages
each link:         16 output cores = 64 pages = 32 input cores
input split:       link0 [0,32), link1 [32,64); both start at tile offset 0
reader:            reserve/read/barrier/push at most 2 pages per input core
writer:            wait/write/flush/pop up to 4 output pages, packet bounded
scratch per link:  ceil(32/2)*4 = 64 pages, enough for both granularities
receive buffer:    [32,512] = 16 pages/core = 4 devices × 4 output pages
```

Sources: `ARF:203-228,269-284,322-385`, reader/writer above. Semaphore writes/increments address output coordinates (`worker_writer.cpp:128,169-170`); receivers wait and reset there (`reduction_receiver.cpp:23-36`). Existing semaphores cover 8×8 (`models/demos/gpt_oss/tt/ccl.py:29-32`) and buffers cover 8×4 (`DEC:56-72`), so retaining the 8×4 output needs no workspace changes even though some input coordinates have x=8–10. Queue order still must preserve producer completion, collective use, and shared-buffer reuse. The existing serialized same-command-queue stack should remain the test boundary; host return alone is not completion evidence.

## Compatible alternatives and API traps

### A. Preserve chosen rectangular storage with `optional_output_tensor`

This is a real existing API, unlike adding a compute grid to the DRAM config. The Python keyword is **`optional_output_tensor`** (`operations/matmul/matmul_nanobind.cpp:923,946`). Allocate a tensor with the exact logical result shape/dtype/tile/device and desired valid L1 width-sharded config, pass that tensor plus matching `memory_config`, and keep its lifetime separate from simultaneously live activations.

`MM:2973-2974` returns the supplied tensor's spec before automatic output-grid computation; `:3302-3310` reuses the tensor. Validation checks true logical result volume/width/rows, dtype, and matching requested memory config (`MM:228-255,3431-3452`). This bypass is intentional behavior to exploit carefully, not proof that every arbitrary storage geometry is safe.

For the qualifying multicore pipeline, actual output shape/grid determine `OW` and `DSMC_OC` (`MF:1413-1425,1582-1592`). The factory passes no `per_core_N` into the multicore implementation (`MF:1854-1871`); the writer uses the actual `OW` (`MK:382-396`). Thus a row projection can write directly into 8×4 `[32,128]`, or gate/up can write into a deliberately selected exact 8×6 `[32,64]` grid for down. The matmul already writes compute results to storage cores; this selects their destination rather than adding a post-matmul reshard.

Merely preserving the original gate/up **8×8** `[32,64]` metadata would still have only 48 shards containing logical data. TensorSpec permits excess core capacity (`tt_metal/impl/tensor/spec/tensor_spec.cpp:62-73`), and the writer covers only the produced column range. The other coordinates are unused cores, not additional logical output; do not feed their contents to a reduction. Prefer exact-capacity storage for a new plan.

**Fallback restriction:** the single-reader path uses configured `per_core_N` as output storage stride (`MF:1890-1892,695-696,774-855`). If physical shard width differs from `per_core_N*32`, do not permit fallback. Either keep them equal, or guard the exact multicore eligibility predicate. Bias, fused matmul activation, untilize, tile changes, and reader count can change eligibility. For example, set `per_core_N=4` with `[32,128]` if the chosen plan must also work through supported fallback; do not rely on a pipeline-only width mismatch.

Allocate fresh outputs, or explicitly plan persistent ownership/lifetimes for warmup/capture/replay. A single shared output buffer for gate, up, residual, or live adjacent operations could overwrite inputs; nothing in this report authorizes that aliasing. Existing `test_matmul_dram_sharded_multirow.py:266-285` exercises preallocated output/trace pressure, but does not establish the exact new rectangular layout choices here.

This alternative can remove copies, but consuming automatic layouts plus existing AR remapping is the smaller caller intervention and needs no new output-buffer ownership scheme.

### B. Carry computed64 through AR, residuals, and the next norm

Also source-compatible in this checkout, but broader. Nonrectangular prefix grids are supported for nondistributed sharded norm when every core has the whole height and the set fills its bounding box in shard order (`LN:245-275`). For the hidden64-core result, use bbox extent `(11,6)`, shard `[32,64]`, `block_h=1`, `block_w=2`, and a dividing `subblock_w`, e.g. 2 (`LN:379-457`). The existing norm config `(8,4), block_w=4` is incompatible and cannot simply be reused.

AR output/workspace and semaphore coverage must then include columns 8–10, and all residual operands/output configs must follow the same layout. Existing 8×4 buffers and 8×8 semaphores do not cover it. A computed64 residual still does not make 48-core gate output and the next K3072 layout automatically identical; plan those separately.

Ordinary RMSNorm is **not** a fused reshard just because another output memory config is accepted: write-back is enabled only for `POST_ALL_GATHER` (`normalization/layernorm/device/layernorm_op_multi_core_sharded.cpp:112-116`), while ordinary output buffers are bound on the input compute cores (`sharded_layernorm_factory_helpers.cpp:734-742,1039-1042`). Keep ordinary norm's input/output sharding consistent. Binary ops with different grids can take a TensorAccessor path rather than native local execution (`BIN:940-955`); zero extra movement requires matching residual operands and output.

### C. Other program configs are not interchangeable grid knobs

- `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` has no `compute_with_storage_grid_size` or `allowed_worker_cores` field (`CFG:75-90`). Passing a generic `core_grid` alongside an explicit program config is rejected (`MM:3424`). Changing reader count does not select output-storage coordinates.
- The BatchedDRAM config does preserve caller output specs (`MM:3141-3164`), but it is a different batch-per-weight, height-sharded operation (`MM:1774-1822`). Its factory requires A/output cores and ordering to match optimal DRAM bank reader assignment (`factory/matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp:94-124`). Grouped batch32 here shares one weight matrix and places requests in M; it is not that API's batch dimension.
- `MatmulMultiCoreReuseMultiCast1DProgramConfig` exposes grid/allowed-worker controls, but ordinary multicast and `gather_in0` have different placement contracts (`CFG:40-72`). A 1D gather path with direct DRAM-sharded weights is not a generic arbitrary-grid escape: its factory derives bank ownership from placement/order (`factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:2092-2126,2597-2764`). Without prefetch pipes/global CB, `num_global_cb_receivers` must equal 1 (`MM:2349-2354`). Switching to interleaved weights or prefetch pipes is a larger algorithm/setup change and requires its own validation.
- A C++ change to honor provided output shards could be reasonable for this qualifying multicore branch, but must validate height, capacity, orientation, grid extent, and fallback stride. Changing only `compute_output_specs` globally, or silencing the mismatch warning, would not repair existing consumer conversions and could break single-reader assumptions. No C++ change is necessary for the recommended bounded caller repair.

## Discriminating follow-up tests — proposed, not run

1. **Inspect geometry at the real producer/consumer boundaries.** For both layer kinds and grouped rows 1, 2, 31, 32, record logical/padded shape, dtype/tile, buffer/layout class, shard width, ordered core coordinates, and actual buffer identity after norm, each matmul, multiply, AR, residual, and layer return. Assert all gate/up/multiply/down-A configs match, and norm/gate-A/up-A configs match. Compare the exact current requested-vs-produced counts in the table before testing a repair. This falsifies any assumption that the warning itself performed a reshard.
2. **Isolate direct AR's mapping change.** Hold original row matmul output, workspace, topology, links, semaphore order, dtype, and AR output fixed. Compare old `AR(to_memory_config(x,ar_mem),...)` with `AR(x,...)`. Use rank-, row-, tile-, and column-distinct data so accidental permutation, wrong rank, stale input, or duplicated tiles cannot hide behind PCC. Check all four ranks against the expected sum; wait explicitly after eager calls. Exercise the actual 64×2-page → 32×4-page mapping with the configured link count. Existing `tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py:119-145,195-207,323-334` already describes differing input/output grids, but does not prove this exact new case passes.
3. **Isolate A-layout reuse.** Compare original and changed gate/up/full-packed inputs on the same quantized weights, compute config, and output dtype. Then compare native gate/up-multiply-down with and without the old down-input reshard. Check meaningful columns against the unchanged reference, preserving numerical gates. A failure before AR separates the matmul/binary/layout change from collective mapping.
4. **Discriminate output override from metadata-only requests.** For representative K/N, compare (a) requested rectangle with no optional tensor, (b) `optional_output_tensor` rectangle, and (c) auto output consumed directly. Assert actual output config and compare numerical data. Cover N3072's 48 meaningful shards, N3136's last partial output shard, and N4096. For the override route, test width matched to `per_core_N`, plus a separately guarded pipeline-only different width; test rejected/unsupported fallback conditions explicitly. Seed excess capacity with sentinels if testing overprovisioned grids, and verify consumers use logical data only.
5. **Prove no inter-layer movement at the operation level.** Run the two-layer grouped stack `[0,3]` with outputs left resident until the outer boundary. Record graph/profiler operations and verify there is no `Reshard`, sharded/interleaved conversion, or copy between the preceding final add and the next input norm; verify no standalone conversion at norm→gate/up, multiply→down, or row projection→AR. Confirm binary native-sharded dispatch rather than inferring it from the absence of explicit Python conversions. Attention packing/state operations remain separately attributable. Warning count alone is not an acceptance metric.
6. **Recheck evolving-state trace and workspace reuse.** Reuse `tests/multichip_stack.py`'s different starts/shuffled page tables and four changing-input checked replays. Exercise both attention kinds and consecutive layers sharing AR buffers. Test fresh-address/cache-hit calls and blocking trace replay; assert output values only after completion. If optional matmul outputs are used, retain simultaneous gate/up/residual lifetimes and test back-to-back captured uses. Add tracked allocation replay as a separate check; the supplied pass used tracking off. Ensure the selected class is the repaired production decoder rather than the untouched stage-four candidate base.
7. **Measure only after correctness and operation elimination are established.** Keep precision and role compute configs fixed, synchronize the measured interval, and compare warmed same-stack latency plus movement operations/bytes. The source proves avoidable operations exist, not their exact contribution to the recorded 9.06 ms or a guaranteed net speedup from different NoC source placement.

## Other Potential Issues / excluded explanations

- **Historical grouped-integration gap is now addressed in source, not proven by the historical pass:** the recorded baseline's ordinary `linear` selects decode only for `shape[2]==1` (`DEC:426`); the candidate overrides it under `_grouped`. New production adds a dedicated packed path and helper calls (`LIVE:781-805`). Its correctness is outside what the old batch32 JSON proves; no claim that current production lacks a packed API remains.
- **Snapshot inheritance is now an active provenance distinction:** production changed while `BatchedCandidate`'s stage-four base did not. The historical pass still validates the old candidate path, not the concurrent production changes. Test the implementation actually used and refresh source provenance without silently rewriting the numerical reference.
- **No established lost work, hang, precision bug, or semaphore bug:** the core arithmetic covers logical widths, the factory uses actual tensor coordinates, and the existing synchronized stack passes. The warning is diagnostic metadata divergence; subsequent explicit conversions explain the performance symptom without assuming corruption.
- **Fewer storage cores are not an error to suppress:** N3072/pN2 naturally needs48, N2560/pN2 needs40, and N3136/pN4 needs25. Forcing64 without a consumer plan adds capacity and can change padding behavior, not useful logical work.
- **No blanket claim that nonrectangular norm grids are unsupported:** the current source explicitly supports prefix grids under the conditions above. Conversely, blindly requesting a different ordinary-norm output grid is not proven safe and is unnecessary for the recommended repair.

## Draft claim review

The first draft was rechecked locally and by an independent xhigh investigator against caller, output planner, selected factory, kernel coordinate tables, AR page splitting, norm validation, binary native dispatch, and the actual synchronized test harness. In particular, initial tempting explanations were rejected: 48 cores do not imply dropped columns; `memory_config` alone does not preserve the grid; `optional_output_tensor` really does preserve supplied storage for this factory; direct AR does not require matching input/output grids; and this checkout's norm can accept prefix grids. Review also caught concurrent source changes: historical per-layer export/grouped-integration findings were moved out of current headline claims, while the remaining helper conversions were reverified in the newer source. The AR recommendation was explicitly bounded to one physical M tile and compatible BF16 page layout. Remaining uncertainty is explicit: none of the proposed changes or their performance was tested on hardware in this inspection-only session.
