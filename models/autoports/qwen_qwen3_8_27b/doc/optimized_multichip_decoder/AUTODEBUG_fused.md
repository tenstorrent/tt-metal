# AutoDebug: Stage05 fused AG/MM and MM/RS with native DRAM-sharded weights

Inspection date: 2026-10-23 UTC; final sweep evidence checked at 22:29 UTC. Checkout HEAD: `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`, with pre-existing model edits and concurrently arriving evidence. No hardware execution, reset, process termination, implementation edit, or child investigation was performed. Only this report was written.

## Findings

**The native WIDTH_SHARDED DRAM path is selected, but its composition with these fused operations is broken. It is incorrect to say these fused kernels categorically require interleaved weights or have no DRAM-sharded reader.** There are three concrete defects/contracts:

1. **Leading AG hang diagnosis: a two-word runtime-argument parsing mismatch.** The host appends nine AG fusion arguments before the DRAM descriptor. The weight reader's `MatmulOpReceiver(false, ...)` consumes only seven. The reader consequently interprets semaphore IDs as DRAM VC/shard count and walks beyond its actual bank descriptors. This directly reaches the DRAM reader branch seen in the captures and can issue malformed reads/writes.
2. **A second AG correctness defect remains after repairing that parser:** the WIDTH_SHARDED DRAM reader ignores the reordered K tile ID produced by the AG fusion receiver. Activations follow ring arrival order; weights follow monotonically increasing K. Fixing argument consumption alone is insufficient.
3. **Leading RS numerical-failure diagnosis: `out_block_w=10` contradicts a DRAM descriptor built for `per_core_N=20`.** The reader reserves/pushes 10-wide blocks but copies 20 weight tiles per row, with a 10-tile destination row stride. It overlaps rows, writes beyond reserved space and eventually beyond the CB allocation, and repeats the wrong columns for the second output-width block. Both RS K-block multipliers have this defect.

These are source-proven contradictions. The precise corrupted addresses and final output values were not captured, so their complete silicon-level consequences remain unverified. The parser defect is the strongest explanation of the observed AG stop sites; the RS width mismatch is the strongest explanation of completed-but-invalid RS results.

**Smallest model adaptation:** set `fused_dram_sharded_weights=false` (or omit it), preserving the one-time `to_memory_config(weight, ttnn.DRAM_MEMORY_CONFIG)` conversion already present. To reproduce an evidenced passing AG configuration, also use the existing `fused_refine` 1D AG settings. For an RS-only native-weight experiment, changing the hardcoded `out_block_w` from 10 to `per_core_N` (20 here) removes the demonstrated width contradiction without changing weights; this is source-consistent, not hardware-validated.

## Observations, separated from interpretation

Evidence directory: [optimized_multichip_decoder](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder).

| Evidence | Direct observation |
|---|---|
| `fused_sharded_both_agk10_l0.log`, `fused_sharded_both_agk20_l0.log` | Single-chip baseline completes; four-chip prefill reaches `PREFILL_DONE`; no multichip `DECODE_DONE`. Sweep records exit `-15`. |
| `triage_fused_sharded_both_agk10_l0/tt-triage.txt` | Across four devices, 28 nonzero-column BRISCs are in `reader_bmm_tile_layout_in1_sender_writer_padding.cpp:389`, inside DRAM bank address/read-state setup. Their 28 unpack TRISCs wait for input 1 at compute line 340. Four column-zero BRISCs wait to reserve input 1 at reader line 374; column-zero input-0 senders wait for multicast receivers at line 380. |
| `triage_fused_sharded_both_agk20_l0/tt-triage.txt` | Same 28 nonzero-column DRAM-reader sites. Column-zero mailbox decoding reports invalid launch pointers, including `971899910 >= 8`, and possible corruption. This is a diagnostic, not a captured proof of which write corrupted it. |
| Layer-3 `triage_fused_sharded_both_agk10_l3` and `...agk20_l3` | Both captures have the same 28/4 reader/waiter pattern as layer-0 K10. Both sweep entries record exit `-15`. |
| `fused_sharded_rs_k1_l0.json`, `fused_sharded_rs_k2_l0.json` | Each contains all 128 decode scores, all `NaN`. Prefill PCC is `0.9999584868043154`; convolution/recurrent state PCCs are approximately `0.9998994` / `0.9997669`. Logs reach multichip `DECODE_DONE`, then fail the final finite-PCC assertion with exit 1. |
| `fused_sharded_rs_k1_l3.json`, `fused_sharded_rs_k2_l3.json` | Both finish 128 decode steps and fail the final numerical gate with exit 1. K1 decode PCC range is `0.00667294–0.01561778`; K2 is `0.00670584–0.01564470`. Every score is finite and below 0.995, including step zero. Prefill PCC is `0.9999760876`; key/value cache PCCs are approximately `0.9998951` / `0.9998743`. |
| `fused_refine_results.json` | All eight recorded combinations of layer 0/3, RS/both, RS multiplier 1/2 exit 0. Minimum recorded prefill/decode PCC is at least `0.999943`. Decode medians are roughly 595–598 µs for layer 0 and 506–527 µs for layer 3. |

The completed native sweep therefore covers all eight named cases: four AG/both hangs and four RS numerical failures, across both decoder types and both K choices. The RS observations are **not hangs**. A `NaN` correlation also does **not** by itself prove tensor elements are NaN: zero variance or nonfinite intermediate arithmetic can produce it. No raw RS outputs were available to classify that further. Passing replay equality checks do not establish numerical correctness.

The evidence does not supply per-core runtime argument dumps, CB counters, or a tensor/operation-ID mapping sufficient to prove whether a captured AG is `mix` or `gateup`. `mix` is the first expected fused AG in decode; both roles have the same parser and K-order defects. The diagnosis does not require assigning the captured kernel to one role.

## Actual call chain and geometry

The important source anchors are:

- [Candidate preparation and dispatch](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/multichip_fused_candidate.py:8): line 14 retains native weights only under `fused_dram_sharded_weights`; lines 30–32 convert activations to interleaved memory and reshape; lines 34–42 choose 2D/1D config; lines 47–59 call the fused APIs.
- [Weight construction](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/multichip_decoder.py:53): model TP shards are separately uploaded as WIDTH_SHARDED DRAM tensors with ROW_MAJOR shard orientation. N is padded to `32 * dram_grid.x * readers`; `readers=2` here. This padding does not itself configure the fused reader to use two workers per bank.
- [Ordinary decode matmul](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:238): the non-fused path explicitly uses `MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(num_workers_per_dram_bank=2)`, with width-sharded L1 activations/output. The fused candidate bypasses that program family.
- [AG wrapper](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/all_gather_matmul_async.cpp:34) and [attribute construction](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp:172): weight remains operand B; AG output is operand A; no implicit weight conversion or operand swap repairs the native layout.
- [AG factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:77): 2D configs use the legacy `Program&` CCL-fusion 2D helper; 1D configs use the 1D helper.
- [RS factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.cpp:65): builds an RS descriptor and appends the descriptor overload of the same CCL-fusion 2D matmul helper. Both overloads bind the same legacy weight-reader file, not its `_metal2.cpp` sibling.

The device is Blackhole according to captured stacks. Its [architecture descriptor](/home/mvasiljev/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml:11) has eight DRAM channels. For the supplied policies, 32×32 tiles, and batch 1, the concrete per-device geometry is:

| Role | Local K | Local logical N before padding | Physical N | N tiles | DRAM shard width / 2D per-core N, tiles |
|---|---:|---:|---:|---:|---:|
| Layer-0 `mix` | 5120 | 4160 | 4608 | 144 | 18 |
| Layer-3 `mix` | 5120 | 3584 | 3584 | 112 | 14 |
| `gateup` | 5120 | 8704 | 8704 | 272 | 34 |
| `out`, both layer types | 1536 | 5120 | 5120 | 160 | 20 |
| `down` | 4352 | 5120 | 5120 | 160 | 20 |

AG receives local A `[1,1,1,1280]`, padded M=32; AG output is `[1,1,1,5120]`, Kt=160. Native candidates use 2D grid `(8,1)`, `per_core_M=1`, `in0_block_w=10` or 20, subblock `(1,2)`, and default `out_block_w=per_core_N`. The [nanobind constructor](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp:168) owns that default. Thus there are 16 or 8 K blocks and one output-width block per worker. CCL offset `(0,6)` does not overlap the matmul row.

RS uses the same `(8,1)` grid, `per_core_M=1`, `per_core_N=20`, subblock `(1,5)`, **explicit `out_block_w=10`**, and K blocks 6/12 for `out`, 17/34 for `down`. It consequently has two output-width blocks. A is interleaved BF16, B native WIDTH_SHARDED DRAM BFP4, MM output interleaved BF16. The persisted intermediate/output buffers have widths 5120/1280. Compute is LoFi, FP32 destination accumulation disabled, packer L1 accumulation enabled.

Both APIs receive ring topology, four-chip mesh `(1,4)`, two links, no bias, and default false transposes. The selected policies put activations, MM outputs and CCL buffers in interleaved L1. Mesh sharding of the residual is distinct from within-device weight sharding and within-device interleaved activation storage.

### What is actually supported by the API surface?

| Question | Code-level answer |
|---|---|
| Can these APIs accept the ordinary `...DRAMShardedProgramConfig` and its `num_workers_per_dram_bank`? | No. [AG validation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp:54) admits 1D/2D Multicast; [RS validation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_device_operation.cpp:37) admits only 2D Multicast. Their factory visitors likewise do not dispatch the dedicated DRAMSharded family. |
| Does 2D fusion contain a native DRAM weight reader anyway? | Yes. [2D factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp:650) sets `IN1_DRAM_WIDTH_SHARDED`; lines 545–548 append compile-time shard strides, and lines 1357–1422 append bank descriptors. Descriptor equivalents are at 2410–2413 and 3254–3319. |
| Are all DRAM-sharded combinations therefore legal? | No. The AG descriptor parser/K-order composition and the RS multi-width-block composition violate the selected reader's contract. Acceptance by validation is insufficient. |
| Must every experimental fused matmul use interleaved B? | No such general claim follows. Interleaved B is the evidenced working adaptation for these candidates. Other CCL/minimal-matmul families are different APIs and were not established as drop-in replacements. |

## AG defect 1: nine host arguments, seven consumed

[Host serializer](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp:473) always appends:

```text
num_transfers, ring_size, ring_index, slice_width,
output_page_offset, last_output_page_offset, clockwise,
signal_semaphore_0, signal_semaphore_1
```

The [weight reader](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:151) constructs `MatmulOpReceiver(false, rt_args_idx, ...)`. In its [constructor](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp:186), the first seven arguments are always consumed, but **both remaining increments are inside `if (wait_for_op_signal)`**. There is no `else` that skips them.

The next [reader statements](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:177) use the now-wrong cursor for `vc`, `num_dram_shards_to_read`, byte offset and interleaved `(stride_bytes, bank_id)` pairs. This is not a compile-time TensorAccessor offset issue: the error is in the runtime stream before that DRAM descriptor.

For no bias and interleaved output, the common prefix occupies indices 0–20. The host AG prefix is indices 21–29, so DRAM metadata starts at 30. The weight reader instead starts parsing DRAM metadata at 28.

The four [multicast semaphore allocations](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp:359) use IDs 0–3; [fusion semaphore allocation](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp:378) then uses IDs 4,5 on the input-0 sender core. [Semaphore allocation](/home/mvasiljev/tt-metal/tt_metal/impl/program/program.cpp:2720) chooses the first unused ID. These are source-derived values, not read-back runtime values.

Concrete layer-0 `mix`, worker column j=1:

| Runtime word | Host meaning/value | Native reader's mistaken meaning |
|---:|---|---|
| 28 | AG semaphore ID 4 | `vc=4` |
| 29 | AG semaphore ID 5 | `num_dram_shards_to_read=5` |
| 30 | Actual VC 2 | `dram_tensor_start_offset=2` bytes |
| 31 | Actual shard count 1 | First stride = 1 byte |
| 32 | Actual DRAM start offset 0 | First bank = 0 |
| 33 | Actual width = `18*576=10368` bytes | Second stride = 10368 bytes |
| 34 | Actual bank 1 | Second bank = 1 |
| 35 onward | Outside this actual argument list | More stride/bank pairs demanded by the erroneous count 5 |

The BFP4 32×32 tile size is [576 bytes](/home/mvasiljev/tt-metal/tt_metal/api/tt-metalium/tt_backend_api_types.hpp:127). The first decoded stride gives `1/576=0` tiles, but the reader still advances its destination offset by one byte. Subsequent reads then use malformed alignment and later nonexistent descriptor words. Do not assume the values beyond the argument list: the capture did not record them. The faulty parser nevertheless proves the descriptor boundary violation without those values.

This precisely selects the source branch sampled at line 389 (`set_async_read_state` on a parsed DRAM bank). Malformed read destinations can also corrupt local state, consistent with the K20 mailbox warning, though no capture proves the exact corrupting transaction. Changing K10 to K20 does not alter the nine-versus-seven prefix, so it cannot repair this defect. Layer-3 widths likewise change payload sizes, not the parsing error.

Interleaved B avoids the trailing DRAM parse, so leaving two unused words at the end is harmless for this particular issue. The passing `fused_refine` AG runs additionally use 1D AG; their grid/config differences mean they are not a strict one-variable layout control.

### Queue and wait ledger

For layer-0 `mix`, per worker:

| Quantity / participant | K10 | K20 | Contract |
|---|---:|---:|---|
| K iterations | 16 | 8 | Complete Kt=160 |
| Input-0 reserve/push/compute pop per iteration | 10 | 20 | Input-0 sender multicasts to seven receivers |
| Input-1 reserve/push/compute pop per iteration | 180 | 360 | 18 output tiles × K-block |
| Input-1 CB capacity | 360 | 720 | Two K blocks |
| Intended DRAM bank descriptors per worker | 1 | 1 | Native reader wrongly decodes 5 |

The [reader pushes](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:534) after reads/barrier; [compute waits](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp:339) for both operands and pops both at lines 493–494. If input 1 does not arrive, input 0 cannot drain; its receivers then cannot reserve the next block, leaving the sender waiting for seven receivers. That explains the observed broad input-0/compute waits as downstream effects. Snapshot stacks do not establish the exact iteration or buffer occupancy.

## AG defect 2: the native reader discards AG's K permutation

The [AG factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp:54) supplies a local slice width of 40 tiles and weight slice stride `40*Nt`. [MatmulOpReceiver](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp:229) updates the current K block's tile ID in ring arrival order. The [input-0 sender](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp:264) uses this updated tile ID for actual activation reads.

The weight reader also calls that update at lines 361–363. Its interleaved branch uses the resulting ID at lines 466–484. Its WIDTH_SHARDED DRAM branch instead initializes a separate byte cursor to zero at line 357 and advances it by `per_core_N_storage * in0_block_w * tile_bytes` at line 422. It never converts the updated tile ID into a bank-local K-row offset.

For ring size 4, the consumed source slices derived from the actual helper are:

| Device ring rank | Activation slice order | Native DRAM weight slice order after a parser-only fix |
|---:|---|---|
| 0 | 0, 3, 1, 2 | 0, 1, 2, 3 |
| 1 | 1, 0, 2, 3 | 0, 1, 2, 3 |
| 2 | 2, 1, 3, 0 | 0, 1, 2, 3 |
| 3 | 3, 2, 0, 1 | 0, 1, 2, 3 |

Rank 1, first K10 block, should pair A's K tiles 40–49 with B's rows 40–49; the native reader fetches B rows 0–9. For the layer-0 mix shard, the missing initial source offset is `40*18*576=414720` bytes. Rank 0 matches only the first slice and diverges at K block 4 (K10) or block 2 (K20). There is no compensating weight permutation in `prepare_fused`, which only retains/converts and reshapes the original weight.

This is not merely a mathematically harmless change in reduction order: A and B use different permutations. It explains why a parser fix must be accompanied by correct K addressing. It is secondary to the parser for the current hang because the supplied run never establishes correct DRAM descriptor execution.

## RS defect: per-core descriptor width exceeds per-block width

RS uses `OpSignaler`, whose [runtime parser](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp:42) consumes the complete matching [host stream](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/ccl/ccl_op_fusion.cpp:434). The AG two-word bug does not apply to RS.

Instead, the [descriptor factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp:2021) sets `in1_block_w=out_block_w=10`, `num_blocks_w_dim=20/10=2`, and CB block size `10*in0_block_w`. But the [DRAM descriptors](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp:3265) are built using **per_core_N=20**, one full 20-tile bank shard per worker. A clamp to `per_core_N` already exists; adding that same clamp again would not fix this problem.

At [reader lines 397–416](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:395), the descriptor selects 20 tiles per source row while destination rows advance by only `in1_block_w=10` tiles. For K-block height H, destination tile positions are `10*m+w`, with `m=0..H-1`, `w=0..19`. The footprint is therefore `10H+10` tiles; only `10H` were reserved. The comment at lines 354–356 explicitly assumes only one output-width block, which this candidate violates.

| Role / multiplier | Kt | H | K iterations per width block | Reserved/pushed/popped per iteration | CB capacity | First footprint | Second footprint relative to CB base |
|---|---:|---:|---:|---:|---:|---|---|
| out / 1 | 48 | 6 | 8 | 60 | 120 | [0,70) | [60,130) |
| out / 2 | 48 | 12 | 4 | 120 | 240 | [0,130) | [120,250) |
| down / 1 | 136 | 17 | 8 | 170 | 340 | [0,180) | [170,350) |
| down / 2 | 136 | 34 | 4 | 340 | 680 | [0,350) | [340,690) |

All quantities above are BFP4 tiles. The second K iteration writes **10 tiles / 5760 bytes past the CB allocation**, even when the reservation succeeds and the prior block has drained. Within each block, the upper half of a copied row overlaps the next row's lower half. The second output-width loop also resets the DRAM K cursor but never applies its output-column tile-ID advancement to the DRAM descriptor; it reads the same columns again instead of the worker's upper 10 columns.

The nominal queue counters themselves remain balanced: reader reserve/push and compute wait/pop all use `10H`. The corruption comes from the larger actual write footprint, so successful completion is possible. The [RS signal](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp:771) follows the writer's output loops; completing the collective does not validate what MM wrote.

Passing interleaved RS uses the **same** `out_block_w=10`, but the interleaved reader loops over exactly ten tiles, increments its tile ID for each width block, and stays inside reserved space. In RS mode the AG-only policy deltas are inactive, leaving weight layout as the relevant effective candidate difference. The parent decoder hashes differ between runs, so this is not a perfectly revision-identical experiment; the local source contradiction does not depend on assuming it is. Both layer types use the same `out` and `down` matrix dimensions, so the footprint calculation covers all four native RS failures. It predicts broken output from the first decode, as observed. It does not predict the exact layer-0 NaN-PCC versus layer-3 finite-PCC distinction without the missing values and allocation/address mapping.

## Smallest adaptations and repair boundaries

### Model-side adaptation, no fused kernel change

1. **For an evidenced working fallback:** omit/set false `fused_dram_sharded_weights` and use `fused_refine` settings, including `fused_ag_1d=true` for AG. Existing `prepare_fused` already converts B once outside the timed decode path, preserving BFP4 dtype and logical TP partitioning. RS can retain its 10-wide output blocks with interleaved B. This adds weight storage and selects a different read path; no speedup is claimed.
2. **Minimum layout-only intervention for the failing 2D AG path:** set `fused_dram_sharded_weights=false`. This removes both native-reader defects from that call. The exact unchanged-2D/layout-only run is not present in the provided results; the existing passing AG evidence uses 1D too.
3. **Minimum native-weight RS-only intervention:** use `out_block_w=nt=20`, preserving `per_core_N=20`, grid `(8,1)`, subblock width 5 and K-block multiplier. This makes each worker execute one width block and matches the host descriptor, source row stride, destination row stride, and CB reservation. Because source already signals RS after all MM output loops, this does not introduce a new requirement to signal after each half-block. Capacity/performance and correctness still need a fresh hardware check. It does not repair AG.
4. **Keep the native optimized decode readers:** retain separate CCL and ordinary matmul using the existing dedicated DRAMSharded config. Passing that config to either fused API is not an available adaptation: it is rejected before the requested reader family can be selected.

### Upstream fixes needed to make the native AG/RS combinations reliable

- **AG stream parsing:** consume or skip both serialized semaphore words even when the weight reader does not wait for AG signals; alternatively make the host/reader stream contract consistently omit them for B. Do not simply change B's receiver to `wait_for_op_signal=true`: AG signal receiver cores are the input-0 sender cores, not every weight-reader core.
- **AG addressing:** derive the bank-local K byte offset from the fused receiver's current logical K row, with the actual shard row stride. Keep ordinary non-fused sequential behavior intact. A parser-only repair is incomplete.
- **RS width blocking:** either reject `WIDTH_SHARDED DRAM B && out_block_w != per_core_N`, or construct per-output-block bank descriptors and use the correct `bw` source offset and bounded destination footprint. A model-level `out_block_w=per_core_N` avoids needing the more general reader repair.
- **Validation:** reject unsupported fusion/layout combinations until implemented. AG currently skips ordinary matmul validation altogether when `persistent_output_buffer=None` ([lines 33–45](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp:33)). RS invokes matmul validation, but its [WIDTH_SHARDED checks](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:1798) do not enforce this native-reader width equality. Divisibility (`20 % 10 == 0`) is insufficient.

No implementation changes above were made in this investigation.

## Other Potential Issues and rejected explanations

- **Hardcoded AG transfer count / watcher assertion:** factory line 58 sets `num_transfers=4`, independent of four-device ring size. `MatmulOpReceiver` computes eight tensor slices and asserts `num_tensor_slices * num_blocks_per_slice == num_blocks` at line 220. Here that is `8*4 != 16` for K10, or `8*2 != 8` for K20. This is a separate concrete assertion-contract problem for watcher-enabled runs. The inspected failing stacks are past initialization, and interleaved/1D fusion shares the helper, so it is not the leading explanation of the native-only DRAM-reader stall. Outside that assertion, the inspected update loop is bounded by actual K blocks, yielding the four-slice orders above. A future watcher test must account for this independent issue.
- **`allowed_worker_cores` warnings:** the helper supplies the requested rectangular grid when absent. That warning alone does not explain the layout-specific failures; the derived grid has eight workers and no overlap with CCL offset `(0,6)`.
- **Not BFP4 rounding or alignment by itself:** 576 bytes is divisible by Blackhole's 64-byte DRAM alignment. BFP4 also appears in passing interleaved runs. Malformed runtime offsets or overlong writes are concrete failures before numerical tolerances become relevant.
- **Not the old unclamped DRAM-shard stride bug:** both CCL-fusion builders already clamp a partial shard contribution to `per_core_N`. That does not constrain it to the smaller `out_block_w` used by RS.
- **No demonstrated fabric or teardown root cause:** reader stalls and malformed source contracts precede downstream waits. Do not interpret queued host warnings for later fused operations as proof those operations completed.
- **Test gap:** existing [AG test setup](/home/mvasiljev/tt-metal/tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py:59) and [RS test setup](/home/mvasiljev/tt-metal/tests/ttnn/unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py:47) default weights to their supplied CCL memory config; the named tests do not establish this native DRAM-width-sharded combination. The [Blackhole RS cache regression](/home/mvasiljev/tt-metal/tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_matmul_reduce_scatter_cache_bh.py:51) explicitly uploads interleaved weights while testing half-width output blocks. That passing coverage cannot certify the native-width reader.

## Asynchronous boundaries and confidence limits

The [harness](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:157) reads prefill output, executes blocking prefill trace replays, and reads back again before `PREFILL_DONE`. The current harness then calls eager decode and reads it at line 207, before decode trace capture. Its timed decode loop synchronizes, executes the trace with `blocking=True`, and reads output every step (lines 222–229). It closes the mesh in `finally` before calculating and asserting final PCC (lines 298–310).

Thus the RS result files plus `DECODE_DONE` and final numerical assertion establish completed execution, not just successful enqueue. AG's downstream factory warnings can be emitted while device work is queued; they cannot demote the captured weight-reader failure. Because the harness source changed after the earlier runs, the exact current line-207 boundary is not asserted as the historical AG stop location. The final layer-3 RS files do match the inspected harness and decoder hashes.

Provenance matters:

- The inspected candidate SHA256 is `f23bc640de958c64a95e94091270b4b8dacd238101ee67f64683b79ee2b32dbd`, matching the passing `fused_refine` and completed native RS JSON files.
- Refine and layer-0 native RS JSON files record harness SHA256 `881fee05...`, which differs from the current harness `96395001...`. Both final layer-3 native RS files match `96395001...`.
- Passing refine runs record parent decoder `a1dc704b...`; layer-0 native RS runs record `74790910...`; the inspected parent and both layer-3 native RS runs are `5235d110...`. Do not treat the full experiments as revision-identical merely because candidate policies differ in only a few fields.
- C++ files used for this diagnosis have no working-tree modifications reported by `git status`; the captures' source locations and named reader branches agree with the inspected tree. The evidence does not include a complete compiled-binary provenance manifest.

## Review and inspection checks

After drafting, each headline claim was checked against both host setup and its actual kernel consumer, including the RS descriptor overload rather than assuming AG and RS share the same builder entry point. Cheap local Python arithmetic checks, without importing TTNN, reconstructed:

- all five physical weight shapes and shard widths from the model's padding rule;
- the AG nine-word prefix / seven-word parse and concrete column 0, 1 and 7 descriptors;
- four-device activation K permutations versus native sequential B reads;
- all four RS role/multiplier footprints, each exceeding its second CB block by 10 tiles, and the bounded footprint after setting width 20;
- completed-result PCC counts, passing/failing policy differences, provenance hashes and per-kernel triage fanout.

The source analysis establishes illegal runtime parsing, mismatched K addressing, and out-of-bounds RS write footprints. It does not establish exact silicon outputs after any proposed change. The most decisive future controls are (a) the same 2D AG config with only B converted to interleaved memory, (b) isolated native RS with only `out_block_w=20`, and (c) native AG after both parser and K-offset fixes. Those are follow-up experiments, not results claimed here. The arriving layer-3 RS evidence was incorporated during report review; it retains the same relevant policies and confirms completed but severely incorrect decode for both multipliers.
