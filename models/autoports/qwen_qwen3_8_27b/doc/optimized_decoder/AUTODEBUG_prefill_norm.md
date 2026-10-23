# AutoDebug: prefill RMSNorm gathers statistics from unused cores

**Leading diagnosis:** the sharded normalization implementation accepts a nonrectangular width-shard grid, but its secondary reduction workers enumerate peers by wrapping around the grid's **bounding rectangle**, not its active cores. With the recorded 11-wide device grid, 80 and 40 shards both leave holes. The 512-token prefill activates 16 reduction workers; workers 1–15 read those holes and omit valid statistics. Batch-one decode activates only worker 0, whose traversal is correct. This is a concrete source-level correctness defect that explains the reported prefill/decode and sharded/interleaved contrasts.

Confidence is **high in the peer-enumeration defect and its applicability to the stated configuration**. It is the leading explanation for the failed runs, but actual NaN production remains unverified: the logs show `PREFILL_PCC nan`, without intermediate finite-value checks. Reading unwritten statistics can produce nonfinite or otherwise incorrect results; it does not guarantee NaNs for every previous SRAM contents.

Investigation was inspection-only: no device access, target execution, resets, or source changes. Only this report and its requested copy were written. The existing `doc/optimized_decoder/AUTODEBUG.md` was preserved.

## Evidence and revision boundary

Inspected checkout: `22612a5fc90813932f3212bc3d8d842fa9142706`.

The stable [candidate5.py](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/candidate5.py:231) has SHA256 `bfc28ef63c963c5940d429f296a634e4f93c0a382a374c6c511ebe346eae81ba`. The [runner](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_optimized_decoder.py:229) has SHA256 `2bf0f04c49a82ecf90c29dd9e3949608c1958f0372fa93ad57edc0ec7f20e138`. Both match the saved prefinal passing controls. Failed runs stopped before producing result/provenance JSON, so their source identity comes from the requester-specified snapshot and experiment context, not a failed-run hash. The older movement decode controls use a different model-source hash and are supporting geometry controls, not exact-revision reproductions. Saved Python provenance also does not independently identify the loaded native library or JIT-kernel source revision: the defect is proven in the prepared source, while exact execution of that source in the recorded runs remains a provenance limitation.

Direct observations:

| Saved artifacts under `doc/optimized_decoder/` | Observation |
| --- | --- |
| `prefinal_prefillnorm_l0.log`, `prefinal_prefillnorm_l3.log` | `PREFILL_PCC nan`; warm-prefill PCC assertion fails. Policy enables prefill sharding with default 80 cores. |
| `prefinal_prefillnorm40_l0.log`, `prefinal_prefillnorm40_l3.log` | Same failure with 40 prefill cores. The additional layer-3 log was also inspected. |
| `prefinal_control_l0.json`, `prefinal_control_l3.json` | Interleaved-prefill PCC `0.9998514124756404` / `0.9998702998915044`, batch 1, sequence 4096. |
| `movement_norm80_l0.json`, `movement_norm40_l0.json` | Decode PCC `0.9998240758635726`; prefill sharding disabled. |
| `movement_norm80_l3.json`, `movement_norm40_l3.json` | Decode PCC `0.9995510900224003`; prefill sharding disabled. |

Each failed prefinal policy differs from the corresponding passing policy only by `prefill_sharded_norm: true`, plus `prefill_norm_cores: 40` for the 40-core cases. The [sweep definition](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/sweep_prefill_final.py:6) agrees. The earlier [device smoke record](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/functional_decoder/work_log.md:21) explicitly records an actual compute grid of 11×10. That is an artifact observation, not a fresh hardware query.

The runner computes PCC using FP64 `torch.corrcoef` ([lines 23–27](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_optimized_decoder.py:23)). A NaN PCC alone does not distinguish nonfinite elements from zero variance. Initial prefill is read back before PCC at lines 229–236. The timing rerun resets request state, synchronizes, enqueues prefill, synchronizes again, and then fails the readback assertion at lines 250–259. Thus the failure is observed after a completion boundary; it is not merely a host-side enqueue succeeding. Failed-run decode was never reached.

## Exact model and operation contract

The [prefill loop](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/candidate5.py:431) uses 512-token chunks and calls `_norm` before either attention implementation. `_finish` calls it again on the residual sum before the MLP ([lines 252–256](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/candidate5.py:252)). Both layer types therefore exercise this common path; the 128/256-wide head norms do not satisfy its width-5120 branch.

`from_state_dict` constructs `1 + input_layernorm.weight` and `1 + post_attention_layernorm.weight` in host FP32, then uploads tiled BF16 DRAM gamma with logical width 5120 ([lines 69–95](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/candidate5.py:69)). The compute configuration is HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True`, `packer_l1_acc=True` ([lines 61–64](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/candidate5.py:61)). Epsilon is the model's `rms_norm_eps=1e-6`.

| Effective value | Prefill, 80 cores | Prefill, 40 cores | Batch-one decode, 80 / 40 cores |
| --- | --- | --- | --- |
| Input to `_norm` | `[1,512,5120]`, tiled BF16 | Same | `[1,1,5120]`, tiled BF16 |
| Reshaped logical input | `[1,1,512,5120]` | Same | `[1,1,1,5120]` |
| Physical matrix height | 512 | 512 | 32 |
| L1 width-shard shape | `[512,64]` | `[512,128]` | `[32,64]` / `[32,128]` |
| Program `block_h`, `block_w`, `subblock_w` | `16,2,1` | `16,4,1` | `1,2,1` / `1,4,1` |
| Row-major active grid | Seven full rows of 11, then 3 cores | Three full rows of 11, then 7 cores | Same respective grid |
| Bounding rectangle | 11×8 = 88 cores | 11×4 = 44 cores | Same respective rectangle |
| Output | Separate BF16 tensor, same shard specification; converted to interleaved DRAM and reshaped | Same | May retain sharding under `carry_residual` |

These are legal, exactly divisible width shards: `Kt=5120/32=160`, hence 2 or 4 width tiles per core. No column mask is needed. Rank-one tiled gamma is intentional and also used by [the RMSNorm tests](/home/mvasiljev/tt-metal/tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py:403).

The [RMSNorm wrapper](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/rmsnorm/rmsnorm.cpp:67) forwards the explicit configuration and tensors to `prim::layer_norm` with `RMSNORM`; it does not swap operands, convert gamma, or alter this geometry. [Factory selection](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:19) uses `input.is_sharded()`. Validation explicitly accepts nonrectangular grids whose active cores are a traversal prefix and whose shards contain the entire height ([lines 245–273](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_device_operation.cpp:245)). Program/shard dimensions also satisfy the checks at lines 392–457. This is a callee defect within an accepted contract, not an unsupported model shape.

`use_welford` and `legacy_reduction` default to false. The selected implementation consists of `reader_mcast_sender_unary_sharded_ln.cpp`, `reader_mcast_receiver_unary_sharded_ln.cpp`, `writer_unary_sharded_ln.cpp`, and `compute/layernorm_sharded.cpp`, with `RMSNORM` and `FUSE_GAMMA` enabled; no residual pre-add or beta. [Kernel selection](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp:477) and [format selection](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp:108) establish BF16 input/gamma/output and FP32 statistics/intermediates. Interleaved prefill selects a different factory and does not perform this cross-shard gather.

## Root cause: active-core count and coordinate wrap disagree

The earliest inconsistent conversion is the construction of receiver peer coordinates:

1. [GridParams](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp:169) sets `mcast_1d=true`, `num_blocks=active_core_count`, and disables two-stage reduction for a nonrectangular grid. Thus `num_blocks` is 80/40, while `num_x*num_y` is 88/44.
2. [WorkerDistribution](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp:197) assigns `ceil(block_h/num_blocks)=1` tile row per reduction worker: 16 workers for prefill, one for decode.
3. [Receiver runtime arguments](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp:1802) start each secondary worker's traversal at its own grid-relative coordinates. Worker 1 starts at `(1,0)`; worker 15 at `(4,1)`.
4. The receiver invokes [compute_single_stage_noc_addrs](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/layernorm_dataflow_utils.h:122) with 80/40 peers. Its increments wrap x at `num_x` and y at `num_y`, hence cycle through 88/44 positions. The exact loop is lines 131–154.
5. [The receiver reads every resulting address](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp:169), without an active-core predicate. Row offset is `worker_index * 4096` bytes for FP32 statistics; the offset remains appropriate, but some cores are wrong.

Numbering positions in row-major order (`index=y*11+x`), a host-only transcription of that loop produced:

| Active cores | Reducer / tile row | Invalid positions read | Valid positions omitted |
| --- | --- | --- | --- |
| 80 | 0 | None | None |
| 80 | 1 | 80 = `(3,7)` | 0 |
| 80 | 15 | 80–87 | 7–14 |
| 40 | 0 | None | None |
| 40 | 1 | 40 = `(7,3)` | 0 |
| 40 | 15 | 40–43 | 11–14 |

All prefill reducers 1–15 have invalid peers; only reducer 0 is correct. Decode has no secondary reduction worker, so all its peer sets are valid. This is not a harmless permutation: the membership changes. On a complete rectangle a rotated traversal would still contain every shard, which explains why ordinary rectangular tests do not expose it.

### Concrete buffer and completion ledger

For 80-core prefill, each core contains 32 BF16 input tiles; for 40-core prefill, 64. Local squaring produces 32/64 FP32 tiles, then reduction produces 16 FP32 partial-statistics tiles per active core. The square/reduce loops are at [compute lines 401–467](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp:401).

| Phase | 80-core prefill | 40-core prefill | Decode contrast |
| --- | --- | --- | --- |
| Partial statistics | Each active core produces 16 tiles; reader waits for 16 | Same | One tile per core |
| Gather for each assigned tile row | Reserve/read/push 80 external tiles, each 4096 bytes | Reserve/read/push 40 external tiles | Only sender performs gather; every peer is active |
| Global compute | Wait/pop one external tile 80 times, produce one statistic and one reciprocal-root tile | 40 times, same result counts | One correct result tile |
| Final exchange | Sender gathers 16 result tiles, broadcasts 16 tile rows; each active core consumes all 16 | Same | Gather/broadcast/consume one tile |

The external buffer capacities are exactly 80/40 tiles, and local statistics capacities are 16 tiles ([buffer sizes](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp:510)). The numerical gather counts are balanced; the defect is which producers supply those tiles. Receiver reads complete at a NoC barrier before publishing the external tiles ([receiver lines 176–192](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_receiver_unary_sharded_ln.cpp:176)). Its startup semaphore accounts for actual active producers; it does not establish valid statistics on hole cores.

Unused cores have buffers/semaphores so rectangular multicast can land safely, but their kernel bodies are compiled out with `IDLE_CORE` ([idle-kernel construction](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/sharded_layernorm_factory_helpers.cpp:1417)). They do not produce partial statistics. Reading their L1 is therefore not a valid contribution, even if a physical read completes successfully.

Global compute sums the gathered values, adds epsilon, and applies reciprocal square root ([compute lines 469–533](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp:469)). Arbitrary hole contents can violate the nonnegative finite mean-square invariant. A negative sum after epsilon yields an internal FP32 NaN in the [default reciprocal-square-root mode](/home/mvasiljev/tt-metal/tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h:29); the [Blackhole implementation](/home/mvasiljev/tt-metal/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sqrt.h:79) explicitly assigns quiet NaN for this case. Stale NaNs can also contaminate the reduction. Subsequent format conversions and downstream operations determine the final nonfinite representation. Zero-filled holes would instead underestimate statistics by omitting real shards. Neither possibility requires FP32 overflow from the actual input.

The sender gathers those computed normalization factors and multicasts them to every width shard ([sender lines 238–299](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/dataflow/reader_mcast_sender_unary_sharded_ln.cpp:238)). The compute kernel consumes one factor tile per input tile row ([lines 541–585](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp:541)). Thus rows 32–511 of an isolated 512-row norm are exposed to incorrect factors, while rows 0–31 avoid this particular gather defect. This predicts exposure, not that every exposed element must be nonfinite. Attention, recurrence, and the second normalization can propagate corruption beyond that pattern in the final decoder output.

## Minimal candidate fixes

1. **Already evidenced workaround:** keep `prefill_sharded_norm` disabled. The corresponding saved controls pass. Decode sharding can remain enabled.

2. **Small model-only candidate:** explicitly create a complete rectangular shard grid: 10×8 for 80 cores or 10×4 for 40. Use the same rectangle in both `ShardSpec` and `LayerNormShardedMultiCoreProgramConfig.compute_with_storage_grid_size`; keep shard dimensions, input precision, and `block_h/block_w` unchanged. Changing only the program-grid field does not repair this bug: the factory derives its peer grid from the tensor's shard specification. These rectangles select two-stage reduction: `ceil(16/10)=2` rows per worker, eight first-stage workers per physical core row, and 64/32 total worker instances. This avoids holes but requires hardware correctness/performance validation; it is not a demonstrated pass in the named artifacts.

3. **Small library correction for the accepted nonrectangular contract:** in receiver runtime-argument construction, use canonical gather origin `(0,0)` when `mcast_1d && !use_two_stage_reduce && !grid_is_rectangular`, instead of the secondary worker's coordinate. Validation already requires active cores to form the orientation-ordered prefix of the bounding rectangle. Every receiver will then enumerate precisely that prefix while retaining its own tile-row byte offset. Preserve the existing traversal for other configurations. RMSNorm sums do not require a worker-specific peer rotation. A more general implementation can pass explicit active-peer coordinates, but arbitrary-hole support would also need changes to the existing validator and worker assignment.

The proposed factory change can be scoped immediately before `AddRuntimeArgsForNode` in the all-to-all receiver branch:

```cpp
if (ctx.grid.mcast_1d && !ctx.grid.use_two_stage_reduce && !ctx.grid.grid_is_rectangular) {
    start_x = 0;
    start_y = 0;
}
```

This is a proposed change, not an applied patch. Preserve `all_to_all_tile_offset_bytes`: every worker still owns its distinct output tile row.

The canonical-origin correction removes the first wrong mapping; clearing hole SRAM, changing gamma layout, relaxing PCC, or changing accumulator precision does not restore the missing shard contributions. A modulo-active-count traversal would also fix membership for this uniform-width RMSNorm, but [shared Welford consumers attach weights to peer order](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/normalization/layernorm/device/kernels/compute/layernorm_sharded_welford.cpp:109), so a common-helper change needs an ordering audit. Canonical ordering avoids extending this patch into that separate issue.

Suggested regression coverage, not executed here: exact 512×5120 cases at 80/40 cores, batch-one decode at the same counts, heights 32/64/512, rectangular versus nonrectangular grids, and explicit HiFi4/FP32 settings. Use nonuniform inputs and gamma; assert finite output and per-row error before PCC. A host-level peer-set check should require every reducer to enumerate each active shard exactly once and no hole. For a shared-factory fix, also cover offset grids, column-major traversal with multiple tile rows, and uneven logical widths in Welford LayerNorm to check peer ordering. The existing [nonrectangular RMSNorm test](/home/mvasiljev/tt-metal/tests/ttnn/unit_tests/operations/fused/test_rms_norm_sharded.py:376) fixes `h=32`, as does [the analogous LayerNorm test](/home/mvasiljev/tt-metal/tests/ttnn/unit_tests/operations/fused/test_layer_norm_sharded.py:657); both miss secondary reducers.

## Claim review and other potential issues

| Candidate | Review result |
| --- | --- |
| Invalid width/grid/program shape | Rejected for the reported cases: legal prefix grids, complete height per shard, exact width division, and matching program/shard dimensions. |
| Tiled gamma instead of row-major gamma | Rejected as a headline cause: supported reader path, matching width, and matching test usage; independent of the prefill tile-row count. |
| FP32/HiFi4 format or pack-state error | No concrete contradiction found in the selected BF16-to-FP32 square/reduce, normalization broadcast, or gamma/output transitions. FP32 normalization explicitly reconfigures both inputs at compute lines 541–542. An architecture-specific issue is not needed to explain the coordinate defect. |
| Unbalanced circular-buffer counts / later teardown failure | Counts instantiated above are coherent. The peer error changes data while permitting all reads and handshakes to complete. Logs fail at readback PCC, then close the device. |
| Core-count reduction from 80 to 40 should solve it | Refuted: both counts leave holes in their 11-wide bounding rectangles, and both activate 16 reducers. |
| Every bad run proves NaNs originate in the first norm | Unproven. No intermediate finite checks exist; the claimed causal chain is a code-supported diagnosis, not a captured first-nonfinite trace. |
| Hardcoded program-grid width 11 | A portability concern on another device grid. It agrees with the recorded grid here, so it is not the observed cause. |

After drafting, the headline was rechecked against the receiver argument builder, coordinate loop, validation, idle producers, result broadcast, and passing/failing worker counts. A Python-standard-library-only transcription of the coordinate loop confirmed invalid peers for all 15 secondary prefill workers in both configurations and none for batch-one decode. No TTNN import or hardware execution was used. The loop was also checked with canonical origin: every worker reads exactly the active set. Runtime confirmation still needs a norm-only finite-value probe or a corrected-grid/kernel run by the parent investigator.
