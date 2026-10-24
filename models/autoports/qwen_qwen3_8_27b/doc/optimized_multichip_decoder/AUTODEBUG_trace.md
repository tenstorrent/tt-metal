# AutoDebug: Stage05 mixed-stack decode trace preparation

## Finding

**The runner warms a different output-liveness condition from its second capture. The strongest source-backed explanation is an allocator-dependent native untilize cache signature, not changing projection reader counts or persistent state layout.** In the linear decoder's convolution-history conversion, `UntilizeWithUnpaddingParams.enough_space_height` is recomputed from live L1 capacity and participates in the program hash. For this particular shape, that Boolean can change even though both values select the same block factory and, near the threshold, the same block geometry. A new workload still needs its binaries loaded before capture.

This mechanism is supported by the exact call path and by the B1/output-residency and stack-order differences. **The artifacts do not establish which operation missed, whether capture 1 or capture 2 failed, or whether the capacity threshold actually crossed.** Those are unresolved runtime facts. This report therefore identifies a definite preparation discrepancy and a leading, falsifiable causal explanation; it does not claim an observed hash value or a proven hardware fix.

The smallest preparation change to try is one additional eager decode **while retaining the first eager output, before either retained decode trace is captured**. Complete both eager calls, release their temporary outputs, restore request state, and then capture. Preserve program-cache allocation tracking and the existing `traced2`-only acknowledgment. Warming a new variant between the two captures would introduce a different allocation-lifetime hazard.

Investigation was inspection-only. No hardware commands, reproduction, reset, process termination, source changes, or acknowledgment changes were performed. Only this report was written.

Historical artifact links below resolve to the preserved original failing run, not the later corrected acceptance run.

## Evidence and revision

Checkout HEAD: `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`, with pre-existing working-tree changes. SHA256 values recomputed during inspection:

| File under `models/autoports/qwen_qwen3_8_27b` | SHA256 |
|---|---|
| `tt/multichip_decoder.py` | `1894e0e0360ceca2a989c87229aa238e5917c26012dc2e275d2141ad72bd9ab8` |
| `tt/optimized_decoder.py` | `ed3f7efd9f0e743a0863f264c7c966e9af0677b588d7e90f015a656db368fadf` |
| `tests/run_multichip_decoder.py` | `297df724af26a73452f87fafd69681076b3e8cf1fb30be2b966c976b168c40fb` |

All three passing stack JSONs record those same hashes. The supplied `1894e0e...` names the multichip decoder, not the single-chip optimized decoder; these are not conflicting revisions. The failing log has no completed provenance JSON. Its attribution to the active v2 run comes from the requester and neighboring suite evidence; it does not independently checksum the loaded Python/C++ code.

Direct observations:

- [Failing log, lines 29–30](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/trace_failure_original/stack_b1_l3.log:29) contains `baseline PREFILL_DONE` followed by `Cannot load new binaries during trace capture`. There is no operation name, program hash, traceback, or marker separating the two decode captures.
- The [runner](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:93) opens the one-chip baseline first with fabric disabled. The four-chip arm runs afterward. This failure is before TP4 execution, not evidence of a TP4 CCL fault.
- The [suite record](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/trace_failure_original/stack_validation.json:1) contains the three reverse-order passes and the failing B1/layer3 command, including S129, 16 steps, `--stack --async-stress`. The failure's recorded exit code is `-15`; the fatal preceded watchdog termination. The [watchdog code](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/validate_optimized_multichip.py:73) preserves triage before terminating a timed-out process.
- The preserved [triage artifact](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/triage_stack_b1_l3/tt-triage.txt:1) reports dispatch-related stacks and `dump_running_operations.py: pass`. It supplies no missing operation/hash or failing model-kernel producer. It is not a basis for assigning a fabric or device-kernel deadlock.

| Recorded case | Stack order | Result | Minimum decode PCC against baseline |
|---|---|---|---:|
| [B1/layer0](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/trace_failure_original/stack_b1_l0.json:1) | linear → full | Both arms complete; two captures; final blocking/nonblocking stress output exactly equal | 0.9999425908 |
| [B8/layer0](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/trace_failure_original/stack_b8_l0.json:1) | linear → full | Same | 0.9999395371 |
| [B32/layer0](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/trace_failure_original/stack_b32_l0.json:1) | linear → full | Same | 0.9999470923 |
| B1/layer3 | full → linear | Baseline capture-time binary-loading fatal | No completed comparison |

The suite [defines six cases](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/validate_optimized_multichip.py:51), covering both orders at B1/B8/B32. The inspected suite record has no result for full → linear B8/B32; do not label them passing or failing. The stress assertion compares the final `traced2` output after each 32-replay arm, not every intermediate stress output.

## Preparation and completion ledger

The sequence is explicit in [runner lines 168–240](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:168):

| Phase | Retained decode output on entry | Device execution / boundary |
|---|---|---|
| Prefill validation | None | Eager read, capture, blocking replay, exact output comparison; prefill trace released |
| Decode setup | None | State backup, inputs, page table and all token-update sources allocated |
| Only eager decode, line 218 | None | `forward(); read(eager)` completes output; `del eager`; restore state |
| Capture 1, lines 219–221 | None | Host constructs commands; returned `traced` stays allocated |
| Capture 2, lines 223–225 | `traced` | Same Python forward while one more output remains live |
| Replay validation | `traced`, `traced2` | State restored; token loop synchronizes and uses blocking trace execution/read |
| Async stress | Both outputs | 32 blocking alternating replays; restore; 32 nonblocking replays on the same CQ; synchronize; compare final `traced2` |

Capture itself does not execute the model commands: [FDMeshCommandQueue's capture branch](/home/mvasiljev/tt-metal/tt_metal/distributed/fd_mesh_command_queue.cpp:439) records a trace node and returns. Consequently, not restoring *between* captures does not advance the device recurrent state by two tokens. Host allocation and cached-program mutations still occur.

The exact fatal is a **workload binary-residency guard**. [MeshWorkloadImpl::load_binaries](/home/mvasiljev/tt-metal/tt_metal/distributed/mesh_workload.cpp:171) sees an empty `program_binary_status_`, nonzero binary data and an active capture, and throws before allocating/loading its DRAM binary buffer. A new TTNN cache entry normally produces this condition. The guard alone is not proof of a changed hash: the same function also documents lazy initialization of an incompletely initialized cached workload. No such incomplete initialization is evidenced in this runner.

This is separate from allocation-tracker validation, which runs [before replay](/home/mvasiljev/tt-metal/ttnn/ttnn/__init__.py:137) and reports live corruptible allocations. Broadening acknowledgments cannot fix this earlier capture-time binary guard.

## Leading source chain: native untilize's capacity Boolean

### The actual operands and lowering

For B1/S129/16 steps, capacity rounds to 256 tokens; each decode forward still has sequence length **one**, not 129 or 16. The linear decoder's [`_delta`](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:366) updates its three-row convolution history with:

```python
ttnn.concat([
    state.conv[:, 1:, :],
    ttnn.to_layout(projected[..., :10240], ttnn.ROW_MAJOR_LAYOUT),
], dim=1)
```

Its conversion input is BF16, TILE, interleaved L1, logical `[1,1,10240]`, padded `[1,32,10240]`. `_linear` explicitly moves the mix output to L1 interleaved under the default policy ([lines 254–258](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:254)); the slice preserves that memory configuration. The conversion inherits L1 for its compact row-major output ([to_layout_op.cpp:94](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:94)). Because the logical height is not tile aligned, it directly calls **native `untilize_with_unpadding`** ([line 181](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/core/to_layout/to_layout_op.cpp:181)). The public untilize codegen router is not on this call path.

Current slicing does not silently untilize because an *end* is unaligned: the RM condition depends on nonaligned starts/strides/rank-one input ([slice.cpp:235](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:235)). Thus `cq[:, :1]` remains tiled, and the projected slices starting at 0, 10240, 16384 and 16448 do not introduce a different implicit untilize merely because some widths are 48. `state.conv` is already row-major.

### The precise signature field

The [native wrapper](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/untilize_with_unpadding.cpp:168) computes a pending L1-output reservation, calls `is_enough_space`, and passes `enough_space_height` to the primitive. It is a field in [UntilizeWithUnpaddingParams](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp:11). There is no custom hash for this operation; [default device-op hashing](/home/mvasiljev/tt-metal/ttnn/api/ttnn/device_operation.hpp:72) hashes operation type, **all operation attributes**, and tensor arguments. Therefore changing this Boolean changes the hash inputs even if the user-visible tensor specs are identical.

The capacity function is:

```text
C = min(max(lowest_live_compute_L1_address - CB_base, 0), ordinary_L1_bank_capacity)
enough_space_height = max(C - pending_output_reservation, 0) > estimated_row_CBs
```

See [program_cache_l1.hpp:19](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/core/program_cache_l1.hpp:19) and [common.cpp:614](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:614). It is a **lowest-address capacity**, not total free bytes or total allocated tensor bytes.

For this exact conversion:

| Quantity | Calculation | Bytes |
|---|---|---:|
| Estimated input row CB | 320 BF16 tiles × 2048 | 655,360 |
| Estimated output row CB | 320 BF16 tiles × 2048 | 655,360 |
| Combined row CB estimate | Sum | 1,310,720 |
| Pending compact RM output reservation | One whole 10240-element BF16 row on its busiest bank | 20,480 |
| Required current capacity for `true` | Strictly greater than sum | **1,331,200** |
| Ordinary L1 capacity in passing artifacts | 110 banks, per-bank capacity | 1,461,120 |

The output reservation is not `20480/110`: [common.cpp:590–611](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:590) budgets whole pages on the busiest bank, then allocator granularity. The row-major page is one complete row here. Relative to the reported capacity, an effective occupied span of **129,920 bytes or more** makes the Boolean false.

This check occurs **before** output allocation and reserves that upcoming output. The block factory later samples capacity **after** output creation; it must not subtract the reservation again. The source already handles this timing distinction. A missing output reservation is not the finding.

### A hash change need not change the reader or actual block geometry

For a 320-by-1 tile input on 110 cores:

- `enough_space_height=false` immediately selects `UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory`.
- `true` enters the wide-row heuristic: row splitting would use one core, whereas the width/height split uses a block width of three tiles and 107 cores. It therefore selects the **same** factory.

These branches are at [device operation lines 88–115](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation.cpp:88); the numerical split follows [compute_ncores_wh](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/core/work_split/work_split_tilize.hpp:97).

The actual [block planner](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/common/common.cpp:998) has its own live-capacity limit, so identical factory names do not prove identical geometry in general. **Near this Boolean's threshold**, however, its tile-pair limit is approximately 320, far above the selected width three. The plan remains 106 cores with three tiles each plus one core with two tiles: `106*3 + 2 = 320`. Each core handles one tile row; the unpadding writer retains one logical row, producing `106*192 + 128 = 20480` bytes in total.

The selected sources are `reader_unary_interleaved_wh_multicore_metal2.cpp`, `untilize_wh_metal2.cpp`, and `writer_unary_stick_layout_wh_multicore_metal2.cpp`, bound by the [block factory](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:166). Its full/cliff buffer pairs hold three/two tiles respectively; the factory checks compute block width against buffer capacity ([line 249](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.cpp:249)). No reader/compute queue mismatch is established here. The 1.31 MB estimate is for the rejected/irrelevant whole-row strategy, not the actual three-tile block CB allocation.

Thus two otherwise identical calls on opposite sides of this threshold can generate distinct workload cache entries for equivalent programs. [MeshWorkload binary residency](/home/mvasiljev/tt-metal/tt_metal/distributed/mesh_workload.cpp:171) is per workload; previously compiling identical kernel source elsewhere does not satisfy that new workload's binary-loading requirement.

## Why output liveness and stack order matter

At B1, the baseline carries its final residual in tiled BF16 width-sharded L1. The down projection sets `per_core_M=1`, `per_core_N=ceil(5120/(32*68))=3`. C++ [output-spec construction](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2834) therefore gives 54 storage cores (`ceil(160/3)`), each with a `[32,96]` shard = **6,144 bytes**. The 68 activation-storage cores and DRAM reader count are different quantities. [`_finish`](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:361) preserves this memory configuration in its final add.

This geometry is corroborated by [passing B1 memory data](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/trace_failure_original/stack_b1_l0.json:71): zero L1 after weights and 12,288 reported bytes per bank after retaining two outputs. B8/B32 instead move the final projection to DRAM under the B1-only carry condition; their baseline after-trace L1 allocation is zero.

| Condition at the linear history conversion | Linear → full B1 | Full → linear B1 | Linear → full B8/B32 baseline |
|---|---|---|---|
| Linear decoder's incoming residual | Runner's DRAM input | Prior full decoder's L1 output | DRAM |
| First capture's retained final output during capture 2 | Additional L1 output | Additional L1 output | DRAM output |
| Persistent state placement | DRAM, unchanged | DRAM, unchanged | DRAM, unchanged |
| Consequence for candidate | Lower starting L1 pressure | Extra live intermediate residual plus retained trace output | No extra retained L1 final output |

The prior full output stays alive as the stack's `x` and linear decoder's `hidden_states` while `_delta` executes ([stack loop](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:27), [decode_forward](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:515)). The first capture's final output is another independent allocation during capture 2.

This predicts the **direction** of the recorded contrast. It does not prove a 6,144-byte subtraction at the selector: allocator placement/holes determine the lowest live address. A small retained tensor can pin a much lower address, and aggregate memory totals cannot reconstruct that placement. Neither failure-time capacity nor either Boolean value is logged. If the Boolean remains identical in eager/capture1/capture2, this particular causal attribution is falsified and the first different operation signature must be examined instead.

## Persistent state and other proposed signatures

The persistent-state alternative does not survive source inspection:

| State | B1 physical/logical contract | Update behavior |
|---|---|---|
| Conv | `[1,3,10240]`, BF16, row-major, interleaved DRAM | `ttnn.copy(next_conv, state.conv)` |
| Recurrent | `[1,48,128,128]`, FP32, TILE, interleaved DRAM | Default in-place recurrence; reshape aliases state; conditional copy-back if needed |
| K/V | Each `[8,4,32,256]`, BFP8, TILE, interleaved DRAM | Paged fill/update into the existing cache tensors |

These are [setup allocations](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:201), never rebound during forwards. Conv and recurrent updates are at [lines 390–443](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:390); K/V updates at [lines 471–475](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:471). Runner restoration copies values into the same tensors. For B1 the full recurrent batch slice is a no-op, and [`fp32_recurrence`](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tt/optimized_decoder.py:602) reuses the reshaped initial state for `final`.

Other candidates were tested against the same source/sequence standard:

| Candidate | Source fact and necessary precondition | Assessment / smallest implied intervention |
|---|---|---|
| Native untilize capacity Boolean | Reachable exact conversion; live capacity must cross 1,331,200 bytes before output allocation | Leading candidate; warm the retained-output condition before captures. Actual crossing unmeasured. |
| Automatic projection matmul geometry/readers | Would require an automatically generated config | Ruled out for the baseline projection path. Every decode projection passes an explicit DRAM-sharded config; [get_program_config](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config.cpp:1022) returns it immediately. No reader-policy change is justified. |
| Persistent state becomes sharded/replaced | Would require a state-field rebind or layout-changing update | Not found; source uses copy/in-place updates. No state conversion fix is justified. |
| Generic recurrence hashes buffer addresses | Would require runtime argument values in its hash | Ruled out: addresses are runtime args; [generic-op hashing](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:53) hashes their sizes, not values, and ignores tensor args in its custom hash. Interleaved accessor compile args encode configuration/page size, not addresses. |
| Codegen untilize tier changes | Would require entering the codegen primitive | Not this conversion: `to_layout` directly selects native unpadding. Do not attribute its failure to `UntilizeCodegenDeviceOperation.chosen.tier`. |
| Different interleaved RMSNorm plan | Some norm plans do depend on live capacity | Possible in general, but hidden norms here use explicit sharded configs, and head norms have only 128/256 columns. No observed plan change or matching capacity boundary was found. Secondary only. |
| Capture corrupts host program state / lazy binary initialization | Capture updates cached host objects before suppressed device dispatch | Real lifecycle behavior, but no source violation or incomplete initial enqueue identified for this sequence. Secondary to the reachable capacity-dependent signature. |
| TP4 CCL fault or allocation-tracker rejection | Requires TP4 execution or replay validation | Inconsistent with the recorded stop phase and error origin. |

Metal capture is not side-effect-free on the host: [cache-hit handling](/home/mvasiljev/tt-metal/ttnn/api/ttnn/device_operation.hpp:291) rebinds cached descriptors/runtime arguments before the CQ capture branch. Trace construction snapshots command data. This is distinct from actual graph `NO_DISPATCH`, whose cache guards appear at lines 302 and 354. Those facts were checked rather than assuming suppressed device execution also suppresses all host mutation; no related bug was demonstrated.

## Smallest safe preparation change

At [runner line 218](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/run_multichip_decoder.py:218), retain the first eager result through an additional eager call for `--async-stress`. Conceptual replacement, **not applied**:

```python
warm0 = forward()
read(warm0)
restore()
if args.async_stress:
    warm1 = forward()  # warm0 remains live, like traced during capture 2
    read(warm1)
    del warm1
del warm0
restore()
ttnn.synchronize_device(mesh)
# Existing capture 1, capture 2, traced2-only acknowledgment, and checks follow.
```

This keeps timed trace contents and decoder policy unchanged. The added eager call can initialize its persistent program/helper allocations after the released prefill trace and **before** the two retained decode traces. Completion before releasing warmup buffers and entering capture is explicit. State restoration makes the warmup independent of extra eager state updates.

One additional eager call is the smallest targeted intervention, **not a source-only guarantee of convergence**. Warmup itself can initialize persistent helper allocations or leave different allocator holes. When implementing, verify both zero-held-output and one-held-output passes reach stable effective signatures/program residency before capturing; if necessary repeat that bounded preparation sequence, rather than warming during a live trace interval. Cache-count stability is a useful check, but identify the differing operation/flag to establish this diagnosis and do not assume counts alone prove every placement is safe.

Do **not** insert the added eager call between captures. A completed trace is [registered with the tracker](/home/mvasiljev/tt-metal/tt_metal/distributed/mesh_device.cpp:1655); subsequent persistent program/helper allocations can occupy addresses that trace 1 used for intermediates. Later replay may overwrite them. The tracker [deliberately includes those allocations](/home/mvasiljev/tt-metal/tt_metal/impl/allocator/trace_allocation_tracker.cpp:183). Retain `TT_METAL_TRACE_ALLOC_TRACKING=1`, `TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`, and only the current `acknowledge_corruptible(traced2)` call. Removing the guard, skipping program allocations, or acknowledging additional buffers is not a preparation fix.

A more invasive alternative already has an in-tree preparation model: [probe_mixed_decoder_traces.py:42](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/probe_mixed_decoder_traces.py:42) copies each temporary result into preallocated DRAM output sinks and [warms all contexts before capture](/home/mvasiljev/tt-metal/models/autoports/qwen_qwen3_8_27b/tests/probe_mixed_decoder_traces.py:60). Such sinks remove the newly retained L1 output difference and can eliminate output acknowledgments, but add conversion/copy work to the measured trace. Prefer the smaller preparation change first for this acceptance runner.

## Follow-up needed to close the runtime attribution

For a future authorized run, without altering tracking or acknowledgment scope:

1. Mark eager start/end and each capture start/end. Record the first operation needing binary initialization, its effective attributes/hash, and factory. Existing C++ operation-hash logging is compiled out in some build configurations ([device_operation.hpp:145](/home/mvasiljev/tt-metal/ttnn/api/ttnn/device_operation.hpp:145)); an environment variable alone may not expose it.
2. At the history conversion record logical/padded shapes, input/output memory configuration, lowest occupied compute-L1 address, CB base, ordinary bank capacity, pending output reservation and `enough_space_height` for all three forwards. Actual `traced` placement is necessary to prove the liveness-to-capacity link.
3. Test the extra eager retained-output preparation before both captures, then require the original exact prefill/restored replay/changed-input/alternating async checks. Confirm no fresh program/helper allocations occur after capture 1 beyond the existing allowed second output.
4. Complete both stack orders for B1/B8/B32. Passing B8/B32/layer0 does not establish the missing reverse-order cases. Preserve all failing artifacts and do not reinterpret a corrected baseline run as full TP4 acceptance until the TP4 arm also completes.

No numeric failed hash or changed reader kernel can be recovered from the provided log. If instrumentation instead shows capture 1 failing, investigate allocations/selector initialization between eager execution and capture 1; the source-backed sensitivity still matters, but the specific retained-`traced` explanation would not apply to that failure.

## Other Potential Issues and review outcome

- The capacity Boolean is redundant in the cache key for this shape near the identified boundary. A future library change could normalize keys to effective, safe program plans, but simply removing capacity-dependent attributes globally would be unsafe: other shapes genuinely select different factories/CB sizes. That is broader than the smallest runner preparation fix.
- The block planner documents that live L1 affects its miss-time split and should not be recomputed for cache hits ([common.hpp:147](/home/mvasiljev/tt-metal/ttnn/cpp/ttnn/operations/data_movement/common/common.hpp:147)). This was considered; the local three-tile plan is far below the capacity limit at the relevant Boolean boundary. No block-buffer corruption is needed to explain a fresh workload during capture.
- The comment that all programs were prepared before capture 2 is not established by one eager pass under different liveness. The passing runs prove their own signatures were ready, not a general invariant for every stack order.
- Final review explicitly rejected automatic matmul geometry, state-layout mutation, generic-op address hashing, and codegen-tier attribution as headline causes for this path. It also rejected identifying capture 2 from the generic fatal alone.

Validation performed: source/call-path review, SHA256 comparison against all three passing artifacts, JSON extraction of the complete recorded stack matrix, and a hardware-free Python arithmetic check of the 320-tile history shape, 1,331,200-byte capacity cutoff, 107-core block split, output-byte count, and 54-core/6,144-byte final-output shard. Illustrative capacity values on opposite sides of the cutoff were calculations, **not measurements from the failed run**. The first draft was independently reviewed against source for false positives, state/shape claims, trace semantics and fix safety; its broken source link was corrected. No model or TTNN device code was executed.
