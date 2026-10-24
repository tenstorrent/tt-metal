# AutoDebug: Qwen TP4 prefill column matmul

## Finding

**The native DRAM-width-sharded B reader in the 2D multicast matmul path does not support the multiple N blocks admitted by these candidates. Its host descriptors cover `per_core_N=14` tiles, but its reader, multicast, and compute buffers are sized for `out_block_w=7`. This causes overlapping row writes, writes outside the B circular buffer, and reuse of the wrong columns for the second N block.**

This is a source-proven contract violation and the leading diagnosis for `mix_k80` and `mix_k160`. It matches the passing/failing boundary: all named passing mix candidates use `out_block_w == per_core_N == 14`; both failures introduce two N blocks per core. The exact mechanism behind the final eager-versus-replay equality failure remains unmeasured: buffer corruption can depend on placement/timing, and the assertion also rejects matching NaNs. The logs do not distinguish those manifestations.

The earliest inconsistent composition combines `DramWidthShardWalker(..., per_core_N, ...)` in `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp:988` with a reader whose row pitch and reservation use `out_block_w` (same factory:419–434), without slicing the descriptor for each N block. The walker's 14-column worker-ownership calculation is itself correct. The earlier admission gap is that `validate_matmul_mcast2d_config` allows this combination without enforcing the native reader's single-N-block contract.

**Smallest containment:** for this native B-reader path, require `out_block_w == per_core_N`, and retain a measured passing K-block geometry such as K40/H1/W14/sub1x2. Supporting W7 or W2 correctly requires block-aware bank descriptors/addressing, not a subblock restriction or a larger buffer alone.

## Scope and evidence quality

- Inspection only; no hardware operations, reproductions, resets, or source edits. Only this report was written. Cheap Python checks modeled descriptor/address arithmetic without importing TTNN.
- Checkout HEAD: `ad9ed5d3c46f649ae4a0fa0804e899e7999d8180`. The working tree contains ongoing model changes; they were preserved.
- Inspected `prefill_column_role0_candidates.json`, `prefill_column_role0_results.json`, the K80/K160 failure logs, passing receipts, model code, Python/C++ dispatch, factory validation, the selected Metal2 dataflow/compute kernels, and Blackhole compute support through a separate investigation.
- The K40 receipt's hashes match the current harness, adapter, optimized/multichip decoders, workspace, and fused prefill/decode files. K20/K32/K40 use adapter v2. Failed runs do not emit a completed receipt; their logs identify the current harness/adapter call sites, but do not independently establish the binary/source hash of every C++ component. The legacy dataflow kernel retains the same descriptor/row-pitch assumption, so the finding is not dependent solely on the Metal2 wrapper.
- The AutoDebug launcher was not invoked. Three inspection-only subagents independently audited native descriptors, compute/subblocks, and trace/lifetimes; their claims were checked against source before inclusion.

### Direct observations

`models/autoports/qwen_qwen3_8_27b/doc/optimized_multichip_decoder/prefill_column_role0_results.json` records successful control and K5/8/10/16/20/32/40 runs, and exit code 1 for K80/K160. Both failure logs reach `multichip EAGER_PREFILL_DONE`, then fail `run_multichip_decoder.py:186` with `prefill replay mismatch`. They contain no intermediate mix outputs, mismatch coordinates, nonfinite counts, or buffer placement dump. Successful baseline execution is a separate single-device run and does not validate the failing TP4 geometry.

### Interpretation, explicitly limited

The native-reader violation explains why narrowing N blocks is unsafe and why all reported W14 contrasts avoid it. It provides a direct corruption mechanism before downstream attention, CCL, and residual operations. It does **not** by itself establish which neighboring allocation was overwritten, whether finite values differed across replay, or whether the first bad full-model value was observed at the mix projection. A deterministic wrong-column result alone would not prove eager/replay disagreement.

## Actual lowered operation and repro matrix

The effective mix operation is `A[1,4096,5120] @ B[5120,4608]` per TP rank, with 32×32 tiles. Logical output width is 4160; 4608 is the padded physical width used by the matmul.

- `multichip_decoder.py:137–146` builds local mix widths `[512,512,1536,1536,32,32]`, totaling 4160. Its `put` at :95–113 pads using eight DRAM banks and `weight_pad_readers=2`: `ceil(4160/512)*512 = 4608`, with `[5120,576]` per bank, or 160×18 tiles.
- `multichip_decoder.py:387–389` gathers the 1280-channel residual to 5120 channels before the mix linear. Its role-specific prefill gather and `optimized_decoder.py:231–236` retain/cast BF8 activations. Weights are BF4 (`DEFAULT_POLICY`, :22–26); output is BF16 DRAM interleaved (:287–288). `readers_final_padded_l0.json:3–11` independently records K5120/N4608/logical4160/BF4/LoFi.
- Explicit config: grid 11×10, `per_core_M=ceil(128/10)=13`, `per_core_N=ceil(144/11)=14`, no transposition, `fuse_batch=False`, no bias or fused mix activation. `math_approx_mode=False`, `fp32_dest_acc_en=False`, requested `packer_l1_acc=True` (`optimized_decoder.py:237–239,267–288`).
- The adapter overrides only H/W/sub-W inside the synchronous `_base_linear` call and restores the constructor in `finally` (`run_prefill_column_candidate.py:14–33`). The C++ config is already a concrete value when the call returns.
- Explicit config is retained (`device/config/matmul_program_config.cpp:1022–1023`). `matmul.cpp:268–303` does not swap these operands or manually transpose them. `MatmulDeviceOperation` selects `MatmulMultiCoreReuseMcast2DProgramFactory` (:2508–2509), **not** the dedicated DRAM-sharded decode factory.
- That factory defines `IN1_DRAM_WIDTH_SHARDED` (:735–743), selects `reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp` (:1452), and `bmm_large_block_zm_fused_bias_activation_metal2.cpp` (:1772–1776).

In the table, Kblock is `in0_block_w` in tiles; the full contracted dimension remains 160 tiles / 5120 elements.

| Mix case | Kblock | Output block H×W | Subblock | K iterations | N blocks/core | Recorded result |
|---|---:|---|---|---:|---:|---|
| control | 4 | 13×14 | 1×2 | 40 | 1 | pass |
| K5/8/10/16 | 5/8/10/16 | 13×14 | 1×2 | 32/20/16/10 | 1 | pass |
| K20/32/40 | 20/32/40 | 1×14 | 1×2 | 8/5/4 | 1 | pass |
| K80 | 80 | 1×7 | 1×7 | 2 | 2 | equality assertion |
| K160 | 160 | 1×7 | 1×7 | 1 | 2 | equality assertion |

Python explicitly chooses sub-W2 for per-core N14 (:270–271). Omitted output blocks default to per-core M/N (`matmul_nanobind.cpp:368–369`). Thus K, output subdivision, and subblock width change together in the failing observations. No evidence supports blaming K alone.

## Descriptor/consumer contradiction

The relevant factory is `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp` (abbreviated F below); the selected B reader is `device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding_metal2.cpp` (R); compute is `device/kernels/compute/bmm_large_block_zm_fused_bias_activation_metal2.cpp` (C).

1. F:419–434 sets B row width to `out_block_w`, N-loop count to `per_core_N/out_block_w`, and one B buffer block to `out_block_w*Kblock` tiles. B is double-buffered only when `batch*K_iterations > 1`; the buffering depth is 2 (`utilities/matmul_utilities.hpp:23`).
2. F:630 derives storage width as `ceil(144/8)=18` tiles, which matches the actual shard width here. F:988 and :1201–1205 construct/advance one bank descriptor per sender for **14** columns. `DramWidthShardWalker::advance` (:229–291) splits that width across banks; it has no per-output-N-block descriptor.
3. F:1491–1496 passes B width 7, B height Kblock, block tiles `7*Kblock`, two N blocks, and 13 M blocks. F:1562–1565 passes DRAM row pitch 18 tiles and K-block stride `18*Kblock` tiles.
4. R:419 reserves `7*Kblock`. R:428–460 reads the complete descriptor's 14 tiles per K row while advancing the **destination row by only seven tiles**. The direct `CoreLocalMem` writes do not wrap through the circular-buffer address calculation.
5. R:396–400 resets its DRAM K offset for each `bw`; the native branch ignores the N-block tile cursor. R:799 advances that cursor, but only other reader branches use it. Every `bw` therefore reuses the original banks/start columns. The source comment itself assumes one N block (:397–399).
6. R:543–546 multicasts only the nominal block bytes and :578 pushes only the nominal tile count. C:403–404 waits for the same count and :574–575 pops it. Handshake counts can remain balanced despite wrong data and memory writes.

### Concrete bank assignments

Each entry is `(bank, starting column within bank, tile count)`. These follow the walker's arithmetic; a local static model checked them.

| Sender x | First global N tile | Descriptor spans |
|---:|---:|---|
| 0 | 0 | (0,0,14) |
| 1 | 14 | (0,14,4), (1,0,10) |
| 2 | 28 | (1,10,8), (2,0,6) |
| 3 | 42 | (2,6,12), (3,0,2) |
| 4 | 56 | (3,2,14) |
| 5 | 70 | (3,16,2), (4,0,12) |
| 6 | 84 | (4,12,6), (5,0,8) |
| 7 | 98 | (5,8,10), (6,0,4) |
| 8 | 112 | (6,4,14) |
| 9 | 126 | (7,0,14) |
| 10 | 140 | (7,14,4) |

For sender 0 and W7, K-row 0 writes destination tiles 0…13, row 1 writes 7…20, and so on. The final occupied extent is `7*Kblock+7`, not `7*Kblock`. For sender 1, overlapping writes also originate from different banks: destination tile 7 receives row 1/local-column 0 from bank 0 and row 0/local-column 7 from bank 1. No assumed completion order is needed to prove the contradiction: two different source tiles target one destination.

On `bw=1`, sender 0 should consume global N tiles 7…13; it again starts at tile 0. The correct sender-1 block split would be `(bank0,offset14,width4)+(bank1,offset0,width3)` for `bw=0`, then `(bank1,offset3,width7)` for `bw=1`. The current descriptors are unchanged between those blocks.

Sender 10 is a separate tail: only four physical columns exist, and its second W7 block is entirely outside physical N. It does not have the same 14-column overrun; it does have the missing N-window handling. All its output is beyond logical N4160 and is sliced away. The headline proof uses full-width senders 0–9, including useful columns, and does not rely on this padding tail.

### Instantiated queue and address ledger

Counts below are for a full-width B sender, one M/N block, batch 1. R reserves, pushes and multicasts one nominal B block; compute waits and pops that same block. Output/partials hold one H×W BF16 block.

| Configuration | B block tiles | B allocation tiles | K iterations | Maximum exclusive write offset relative to current block | Buffer violation |
|---|---:|---:|---:|---:|---|
| K40/W14 | 560 | 1120 | 4 | 560 | none from this mismatch |
| K80/W7 | 560 | 1120 | 2 | 567 | first slot exceeds reservation; second slot writes through allocation offset 1126 |
| K160/W7 | 1120 | 1120 | 1 | 1127 | first slot writes through allocation offset 1126 |
| proposed K80/W2 | 160 | 320 | 2 | 172 | 12 tiles beyond each nominal slot |
| proposed K160/W2 | 320 | 320 | 1 | 332 | 12 tiles beyond allocation |

BF4 32×32 tiles occupy 576 bytes (`tt_metal/api/tt-metalium/tt_backend_api_types.hpp:126–127`). The W7 overrun is 7×576 = **4032 bytes**; W2 would overrun by 12×576 = **6912 bytes** at the end slot. K80/W7 and K160/W7 both allocate 1120 B tiles (645120 bytes), but get there through double versus single buffering. A larger reservation alone would still leave overlapping row layout and wrong N-block source selection.

Allocation rounding does not absorb this violation: Metal2 lowering retains entry size/count (`tt_metal/impl/metal2_host_api/program_spec/construction/resource/dfb.cpp:220–222`), `total_size()` is their exact product (`tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp:134`), and allocation marks that exact size (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3032`). Starting-address alignment (:3019) does not enlarge the declared buffer; 645120 is already a multiple of 64. The physical victim beyond this buffer still requires runtime placement evidence.

W14 has matching descriptor width, destination pitch, and one N iteration. Reducing H to 1 in K20/32/40 adds M iterations; R resets K offset for each M block, which is appropriate and explains why those cases need not fail.

## Validation, compute, and lifetime adjudication

**Validation gap.** `matmul_device_operation.cpp:544–586` validates K and block/subblock divisibility; both failing candidates satisfy those checks. `validate_matmul_mcast2d_config` (:1711 onward, B checks :1798–1895) accepts DRAM width-sharded B without checking `out_block_w == per_core_N`. Its non-DRAM B restrictions do not repair this path. A sibling implementation already documents the native reader restriction in `models/autoports/qwen_qwen3_8_27b/tt/fused_decode.py:4–5` and uses equal widths at :34–36. This comment corroborates, rather than substitutes for, the address proof above.

**Sub1×7 is not established as a bug.** The supported C++ subblock choices explicitly include `{1,7}` (`matmul_program_config.cpp:18–26`). Blackhole math iterates the configured destination columns (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_matmul.h:806–814`), unpack issues the configured column count (`llk_unpack_AB_matmul.h:421` in that directory), and packing loops over individual tiles (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h:154–159`). Inspection found no even-width requirement or out-of-range compute index for seven tiles. For W7/sub1×7, each M/N group has one subblock and a seven-tile output/partials ring. K80 has two K iterations with partial spill/reload; K160 has one and no spill. F:374 enables packer accumulation only with multiple K blocks, so both failures do not share a required partial-reload condition. This does not certify every numeric fast path; it makes an unsupported-width theory much weaker than the direct reader violation.

**Completion/lifetime sequence.** Harness :168–170 executes eager prefill, completes host readback, and releases the eager output. :172 restores the independent empty-state snapshot; :173–175 captures with persistent input/state objects; :179–182 restores state, explicitly synchronizes, then executes each replay with `blocking=True`. :186 reads the retained capture output. Snapshot clones are independent (`optimized_decoder.py:218–222`). Thus `EAGER_PREFILL_DONE` includes host output readback, and the failed comparison follows completed replay/readback; it is not merely an enqueue-time print.

Restoring the Python config constructor does not invalidate the constructed C++ config or captured commands. No missing restore or premature release was demonstrated. The explicit program config also prevents a free-L1 heuristic from silently choosing a different mix geometry. The investigation did not prove all generic cache internals bug-free; there is no sequence-specific cache contradiction needed to reach the native-reader violation.

## Proposed repair and discriminating adaptations

### Immediate narrow repair

Add a native-B-specific check in the 2D-config validator, conceptually:

```cpp
if (B.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
    B.buffer()->buffer_type() == BufferType::DRAM) {
    TT_FATAL(config.out_block_w == config.per_core_N,
             "Native DRAM-width-sharded B requires out_block_w == per_core_N");
}
```

The check belongs under the 2D config path, which is validated on a program cache miss (:2541,2665–2676); it must not reject interleaved B or unrelated factories. A matching factory assertion would protect internal entry points. At the candidate/model boundary, reject the unsupported W7/W2 native combination before capture and retain W14 with a measured fitting K block. Simply restoring W14 while retaining K80/K160 doubles B-block storage and may exceed L1; their resource fit is not established.

### If multiple N blocks with native weights are required

Make the native descriptor/reader describe the actual `(sender, bw)` window:

```text
n_start = sender_N_index * per_core_N + bw * out_block_w
valid_width = min(out_block_w, max(0, physical_N_tiles - n_start))
bank = n_start / shard_width_tiles
bank_offset = n_start % shard_width_tiles
```

Split that window across banks; use its offsets for source addresses and its local destination offsets with row pitch `out_block_w`. Preserve the bank's full 18-tile physical row stride and K-block advance. Clip or initialize padding without reading beyond physical N. This can be implemented with per-block descriptors or by clipping the existing per-core descriptor in the kernel. Validate the sums and maximum destination extent for every block. Inspect the analogous fused-bias reader if sharing this support, though this mix repro has no bias.

**Do not change only the walker's `per_core_N` argument from 14 to 7.** It advances once per sender; that change would assign sender 1 columns 7…13 instead of 14…27 and still would not select `bw=1` correctly. Do not merely enlarge the B buffer, relax equality, or forbid 1×7 while allowing other narrow N blocks.

### Smallest useful controls for the parent sweep (not run here)

| Adaptation | What it isolates | Prediction from this diagnosis |
|---|---|---|
| Same K80/K160, W7/sub1×7, **interleaved B with identical padded shape and quantized values** | B addressing while preserving compute geometry | Removes the native descriptor violation, but retains a distinct generic-reader N-tail overread described below; not a fully clean control |
| Native W2/sub1×2, same per-core N14 | Removes odd subblock width but introduces seven N blocks | **Not a repair**: remains source-invalid, with a larger 12-tile overrun; a pass would not establish safety |
| Interleaved W2/sub1×2 versus interleaved W7/sub1×7, with the N tail fixed or padded as below | Subblock/compute behavior after removing native corruption | Useful if the interleaved W7 control still fails |
| Native K40/H1/W14/sub1×7 versus passing sub1×2 | Changes only subblock width, retaining one N block and measured B capacity | Clean test of 1×7 independent of native narrow-block corruption |
| Native W7/sub1×1 | Holds reader geometry while removing 1×7 | Same descriptor violation remains; optional falsification, not a safe candidate |

For the interleaved control, convert/cache the mix weight during setup, outside eager/capture, preserving all **4608** physical columns, BF4 bytes/values, and the rank-local partition. Keep role and logical-width metadata associated with the replacement (`weight_roles`, `logical_widths` are keyed by tensor identity). Rebuilding only 4160 columns would change per-core N and invalidate the contrast. Reuse the same replacement for eager and capture; do not insert a conversion inside just one execution. Prefer isolating the mix matmul initially so decode's dedicated DRAM-sharded path does not receive an incompatible replacement weight.

The interleaved branch actually iterates only `in1_block_w` tiles per row and consumes the advancing N-block cursor (R:502–527,799). This removes the identified full-width native descriptor bug while retaining 1×7. **However, its tail predicate assumes the last valid N block is the last nominal N block.** At sender 10 with W7, R:514 admits columns 140…146 for `bw=0` and 147…150 for `bw=1`, although only 140…143 exist. At the final K row this reads beyond physical B. The generic bounds check should instead use each sender's valid N extent for each `bw`, not just `bw < num_blocks_w_dim-1`.

For a cleaner additional control without a kernel change, make a second **interleaved-only** weight padded to 4928 columns / 154 tiles: all 11 workers then have exactly 14 physical columns, while grid11×10, per-core N14, K, W7/sub1×7, precision, and the 4160 logical columns remain unchanged. This adds ten zero tiles and changes the B row stride, so record it as a separate padding control, not the requested same-shape control. Use the same padding for the interleaved W2 comparison. Alternatively, correct the generic N-tail read predicate before relying on the exact same-shape control. No success is asserted for either adaptation.

Collect finiteness counts and first mismatching `(rank, token, channel)` coordinates for the mix result before slicing/downstream attention, plus full output. Compare eager A/B, eager-to-capture/replay, and repeated eager executions after equal state resets. Keep raw maximum-error diagnostics and exact finite-value equality; do not mask NaNs or treat tolerance relaxation as repair. This distinguishes wrong but repeatable data from replay-sensitive corruption. Physical neighbor placement is needed before naming the specific overwritten buffer.

### Verification expected with a repair

Add targeted regression coverage at the intervention boundary: reject native narrow-block configs if guarded, or compare repaired native W7/W2 against same-shape interleaved and host references if support is implemented. Include a bank-crossing sender, K80 double buffering, K160 single buffering, and the final four-column physical tail. Use repeated allocation/cache/trace execution to expose state dependence, and verify logical output and nonfinite counts.

Existing `test_matmul_2d_multiple_output_blocks_per_core` covers multiple N blocks, but uploads B with `DRAM_MEMORY_CONFIG` (`tests/ttnn/unit_tests/operations/matmul/test_matmul.py:741–747`). It therefore exercises the interleaved reader and does not validate this native-width-sharded descriptor contract. Tests of sharded A/output likewise do not imply sharded-B coverage.

## Other Potential Issues and unresolved dimensions

| Candidate | Direct code contradiction | Match to observations | Remaining precondition / falsifier | Disposition |
|---|---|---|---|---|
| Native per-core descriptor consumed as per-N-block data | Yes: 14-column spans, 7-column pitch/reservation, no `bw` source shift | Exactly separates named W14 passes and W7 failures; affects both K80 and K160 | Runtime C++ provenance and exact overwritten neighbor/NaN-vs-finite manifestation unmeasured; same-shape interleaved control is decisive follow-up | Leading diagnosis |
| Unsupported sub1×7 | None found; explicitly listed as supported, compute counts fit | Sub-W changes with the failures, but is confounded with N subdivision | Safe W14/sub1×7 or interleaved W7 control | Secondary, unproven |
| Constructor restoration/cache/lifetime bug | None demonstrated in adapter/harness sequence | Shared v2 adapter passes K20/32/40; persistent inputs/state and completion boundaries are coherent | Would need a concrete cached-field mutation or lifetime violation; no such evidence | Secondary, unproven |
| Packer partial-reload bug | None established on this exact compute path | K160 has no partial-reload phase, while K80 does | Cannot independently cover both without another cause | Secondary |
| Identical nonfinite outputs rejected by `torch.equal` | Assertion permits this explanation; outputs not saved | Compatible with completion plus comparison failure | Finiteness dump | Manifestation uncertainty, not a separate demonstrated root cause |
| Incorrect storage shard-width derivation | F derives `ceil(N/banks)` instead of consulting shard shape | Here both equal 18 tiles exactly | Different padding/shard geometry would be needed | Not causal for this repro |

**Additional source issues, outside the headline causal path:**

- The interleaved reader's last-worker N-tail overread is demonstrated above. It matters to the proposed control; the reported native-width-sharded path executes a different branch. Tail outputs are discarded, so it does not explain useful-output corruption in the first ten senders. The 2D validator (:897–934) lacks the corresponding restriction present in the 1D mcast-in1 validator (:854–866). Keep it separate from the native descriptor violation.
- On assertion failure, the harness skips explicit `pre_tid` release (:173,186–187); its `finally` releases only `tid`/`tid2` before closing the mesh (:330–333). This occurs after the observed comparison failure and is not its cause; no persistent leak beyond mesh closure is asserted.
- The adapter's `finally` rewrites provenance/description in any existing output JSON even after a failed run (:39–46). That can relabel a stale receipt. No K80/K160 output JSON existed at inspection, so this does not explain the present failure evidence.

## Draft review

The first draft was rechecked against the actual dispatch, explicit Python subblock choice, factory descriptor construction, both K-buffering modes, native and interleaved reader branches, compute wait/pop counts, padding tail, and harness synchronization. Independent reviews confirmed the native mismatch and lifecycle/matrix claims. Review also found the generic interleaved N-tail overread, so the same-shape control is explicitly qualified and a separate tail-free padding control is provided. No claim that passing cases already exercise 1×7 was retained: they explicitly use 1×2. No exact overwritten allocation, hardware reproduction, finite mismatch, or successful proposed workaround is claimed. W2 is classified as a discriminating but still invalid native configuration, not as a repair. The report's primary finding survives these checks without requiring a trace/cache hypothesis.
