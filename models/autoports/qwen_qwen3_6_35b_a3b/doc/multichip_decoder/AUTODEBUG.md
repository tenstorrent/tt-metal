# AutoDebug: multichip reduction shapes and TP4 linear audit

## Findings

**The logged indexed-decode failure is a caller-side logical/padded shape mismatch after `fast_reduce_nc`.** The reduction returns logical `[1,1,32,2048]`, but the multichip caller requests a one-shape reshape to `[1,1,2048]`. The logical volumes are 65,536 and 2,048. This accounts for the exact `reshape_common.cpp:58` assertion. The optimized baseline already handles this boundary with explicit logical and padded shapes.

**The original TP4 linear implementation contains two additional, independently established blockers:** its scan projections repeat the same 32-row-to-one-row reshape error, and its recurrent reduction promotes eight logical heads to 32. The projection error would stop the first linear prefill before the recurrent error is reached. These are latent source findings, not explanations of the supplied layer-3 full-attention log.

No additional head ownership, QKV packing, or persistent-state update bug was established. Several uses of physical 32-row tiles are intentional; changing every `32` to `8` would break valid custom-kernel contracts.

Inspection only: no implementation edits, device execution, resets, or hardware reproduction were performed by this investigation. Three independent xhigh subagents inspected reduction lowering, linear attention, and the validation runner. Only this report was written.

## Evidence and revision boundary

The diagnosis below refers to the source present at investigation start. SHA256 checks matched all four files against `smoke_l3_s32_v3.json.provenance.json`:

| File, relative to the model directory | Original SHA256 |
|---|---|
| `tt/multichip_decoder.py` | `e35e0541d07ce6b604ffb365484a01f4331bf342dbf666031bf7efd9afacd9ce` |
| `tests/run_multichip_decoder.py` | `b4f063e51232046c5c682006c14c576f249a0e9cdf5eb00d01f425f5d0ceeb7a` |
| `tt/optimized_decoder.py` | `99e2a2fa1d4cb78f4dd0eb95079cc8288e8f7e0ebd92e6ffbc44ffb3be05cb00` |
| `tt/fused_decoder.py` | `d8106611f65edccdf069baab44acc9c0088baa709090b017a6a6df75f0a7bb87` |

Recorded git HEAD is `2901d8a506198dc69a9e99714db3a9c590b6ff95`. Multichip implementation/tests are untracked in that checkout, so the hashes are necessary to identify the failing implementation. Model-source line references below use this original version unless explicitly identified otherwise. Native-source references describe the prepared checkout; the provenance does not separately fingerprint the loaded native library.

Concurrent AutoFix edits became visible during the investigation. A later inspection found explicit logical/padded reshapes at both the indexed-MoE and linear-projection reduction boundaries, plus an eight-head crop in `_recurrent_sum`. Those changes address the three source mismatches described here. They were not made by the investigator and have no runtime validation from this investigation. Line numbers after the edited regions may have shifted. The report preserves the original causal analysis rather than combining the old failure with new caller behavior.

### Direct observations

- The provenance command selects `--layer 3 --length 32 --decode-steps 2`, one prefill repeat, and a saved baseline. No batch override is present; the runner default is batch 1 (`run_multichip_decoder.py:388`). The checked config identifies layer 3 as full attention and layers 0–2 as linear attention.
- `logs/smoke_l3_s32_v3.log:21-28` records decoder load, a prefill dispatch, `PREFILL_DONE`, and then a fatal reshape error.
- The traceback (`log:34-58`) reaches runner `warm = decode()` at line 185, `decode_batch`, and `_moe`'s reduction-result reshape at original multichip line 959. It then enters `fused_decoder.reshape` and fails `infer_dims_for_reshape`'s volume assertion.
- Trace capture starts at runner line 190, after the failing eager warmup. The supplied failure is therefore before trace capture.

### Interpretations supported by source

The failing path is indexed MoE at batch 1; the 32-token prefill uses grouped MoE instead. The exact tensor shapes below follow from the matching source and native output-spec contract. The log does not contain intermediate shape dumps or a completed mesh accuracy result. No numerical-equivalence or post-fix trace claim follows from `PREFILL_DONE` alone.

## 1. Observed failure: indexed MoE loses the explicit padding contract

Original call chain:

```text
runner.decode -> MultichipDecoder.decode_batch
 -> _attention_residual for each row
 -> _normalized_input -> _moe(tokens=1)
 -> indexed gate/up -> SiLU/multiply -> indexed down
 -> routing multiply -> fast_reduce_nc(dims=[1])
 -> fused reshape(total, (1,1,2048)) -> volume assertion
```

Per-chip shapes from original `multichip_decoder.py:922-964`:

| Tensor | Logical | Padded |
|---|---|---|
| Normalized residual | `[1,1,2048]` | `[1,32,2048]` |
| Sparse gate/up result | `[1,8,1,256]` | `[1,8,32,256]` |
| Activated local intermediate | `[1,8,1,128]` | `[1,8,32,128]` |
| Sparse down / weighted contributions | `[1,8,1,2048]` | `[1,8,32,2048]` |
| Unpreallocated reduction output | **`[1,1,32,2048]`** | `[1,1,32,2048]` |
| Requested reshape | `[1,1,2048]` | implicit |

The earliest mismatch is the caller's assumption that reduction preserves the unreduced logical row extent of one:

1. `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.cpp:121-137` copies `input.padded_shape()`, sets the reduced axis to one, and constructs `TensorSpec(output_shape, layout)`. Without `output=`, the padded spatial axes become logical output axes.
2. `fast_reduce_nc/fast_reduce_nc.cpp:43-56` returns that result without cropping. This is a local reduction on each rank, distinct from the later mesh collective.
3. `fused_decoder.py:14-16` calls `ttnn.reshape(x, shape)` before `to_layout`. The tuple overload enters `reshape_view/reshape.cpp:898-913`, which calls `infer_dims_for_reshape`. `reshape_common.cpp:16,57-58` compares the old logical volume against the requested volume and rejects 65,536 versus 2,048. This is exactly the overload and assertion in the log. The trailing `to_layout` never executes.

The lowered kernel supports the shape derivation. `fast_reduce_nc_program_factory.cpp:48-65,107-112` partitions padded tile planes. For this case, `Wt=64`, `Ht=1`, `inner_tile_size=64`, and `reduce_tile_size=512`. The reader (`device/kernels/reader_reduce_nc.cpp:49-74`) reads eight corresponding expert planes at stride 64 tiles. The compute kernel (`device/kernels/reduce_nc.cpp:42-76`) sums corresponding tile lanes and packs whole tiles. It does not compact row zero into a physically one-row tensor. Padding rows do not mix into valid row zero through this reduction.

### Smallest intervention

Restore the baseline's explicit interpretation at this boundary:

```python
total = ttnn.experimental.fast_reduce_nc(
    weighted, dims=[1], memory_config=DRAM, compute_kernel_config=self.compute
)
routed = ttnn.reshape(total, (1, 1, 2048), (1, 32, 2048))
```

This is the existing pattern in `optimized_decoder.py:162-166` and `:295-298`. The explicit overload is bound at `reshape_nanobind.cpp:117-136` and directly dispatches via `reshape.cpp:866-884`; its tile-view handling permits this logical metadata adjustment while retaining physical tile storage (`:778-787`). A logical crop or correctly shaped preallocated reduction output can also preserve the boundary contract.

The generic reshape validator is reporting the metadata it received correctly. Relaxing its volume check, globally changing the shared `reshape` helper, or changing the reducer's global output contract is unnecessary for this decoder-scoped repair.

### Why prefill gets farther

For the supplied length 32, `_moe` takes `_grouped_moe` (`multichip_decoder.py:965-966`), bypassing the indexed reduction. Its `post_combine_reduce` (`:1033-1041`) derives output from the logical combine shape by removing the expert dimension (`deepseek_prefill/post_combine_reduce/device/post_combine_reduce_device_operation.cpp:149-172`). Its `[1,1,32,2048] -> [1,32,2048]` reshape preserves logical volume. This explains the passing-versus-failing control-flow boundary without invoking a CCL, convolution, or trace defect.

## 2. First-linear blocker: scan projection repeats the same volume error

Original `LinearTP4Mixin.linear`, `multichip_decoder.py:640-649`, calls `DirectCoefficientStreamingScan`, reduces dimension zero without `output=`, and tuple-reshapes the result to `[1,1,width]`.

The scan allocates logical `[blocks,1,1,width]`, TILE, FP32 (`fused_decoder.py:2801-2809`). Its physical shape is `[blocks,1,32,width]`. The native scan writer (`STREAMING_SCAN_WRITER`, `fused_decoder.py:2560`) zeroes a whole tile and copies the terminal scan row into row zero. Its one-logical-row / 32-physical-row representation is intentional.

The unpreallocated reduction exposes all 32 rows as logical:

| Projection | Local K / width | Blocks for decode | Reduction logical output | Requested logical volume |
|---|---|---|---|---|
| `attention_packed` | `2048 / 1600` | 4 at K-block 512 | `[1,1,32,1600]`, volume 51,200 | 1,600 |
| `output_only_packed` | `2048 / 1536` | 4 at K-block 512 | `[1,1,32,1536]`, volume 49,152 | 1,536 |
| Ordered `linear_attn.out_proj.weight` | `1024 / 2048` | 2 at K-block 512 | `[1,1,32,2048]`, volume 65,536 | 2,048 |

For multi-token prefill, `linear` iterates over rows and recursively uses K-block 1024 (`multichip_decoder.py:622-632`), so the input projections have two blocks instead of four. Their output padding and erroneous factor of 32 are unchanged. The first `attention_packed` call in `_linear_attention` (`:397-399`) therefore fails for both multi-token prefill and singleton decode before convolution or recurrent state update. This is a necessary source-level repair before first linear device use, but it did not run in the supplied full-attention smoke.

The baseline prepares an FP32 TILE L1 output `[1,1,1,width]` and passes `output=result` for these reductions (`fused_decoder.py:763-777,828-842`; the nonterminal input-projection branch also does so at `:871-885`). The reducer returns a preallocated tensor's own spec (`fast_reduce_nc_device_operation.cpp:117-118`). Restoring this pattern is one minimal intervention. The explicit logical/padded reshape `[1,1,width] / [1,32,width]`, subsequently observed in the shared tree, also addresses the mismatch without changing physical storage.

## 3. Next-linear blocker: recurrence turns eight logical heads into 32

Original `_recurrent_sum`, `multichip_decoder.py:346-352`, adapts the baseline's head count but omits an output logical-head correction:

| Stage | Logical shape | Physical padding relevant to heads |
|---|---|---|
| Products | `[1,8,128,128]` | heads are an outer axis; eight is exact |
| Permute | `[1,128,8,128]` | penultimate head axis pads to 32 |
| Chunked key reduction input | `[8,16,8,128]` | `[8,16,32,128]` |
| `fast_reduce_nc(dims=[0,1])` | `[1,1,32,128]` | logical heads have become 32 |
| Original return permute | `[1,32,1,128]` | expected `[1,8,1,128]` |

The wrapper sorts dimensions and reduces 1 then 0 (`fast_reduce_nc.cpp:37-56`). These axes together cover the 128 key features as 8×16; that grouping is correct. The problem is the unreduced head axis, which is spatial during the reduction and inherited from padded shape.

At `multichip_decoder.py:378-379`, `memory` consequently has `[1,32,1,128]` while `v` has `[1,8,1,128]`. Neither 8 nor 32 is a singleton, so subtraction cannot broadcast. The binary validation enforces precisely this invariant in `eltwise/binary_ng/device/binary_ng_device_operation.cpp:440-456`. The second `_recurrent_sum` call for the output has the same logical-head problem.

`fused_decoder.py:1130-1136` uses 32 real heads, so its unpreallocated reduction's logical and padded head extents coincide. Merely replacing the baseline's head constants with eight fails to preserve that invariant.

Smallest intervention: crop the reduced tensor to `total[:, :, :8, :]` before the return permute, or provide final reduction output logical `[1,1,8,128]` with physical `[1,1,32,128]`. The crop subsequently observed in the shared tree matches this requirement. Retain eight FP32 state heads; do not expand persistent state or `v` to 32 logical heads to make broadcasting succeed.

This affects decode and one-token linear prefill that takes `_recurrent_step`. Multi-token linear prefill uses `Chunk64OutputGraph` instead, so this finding is not a claim that its prefill recurrence fails for the same reason. The projection blocker masks this later failure until repaired.

## TP4 packing and state audit

The following contracts were checked against sibling paths and custom-kernel setup. These are source-level consistency findings, not device correctness results.

| Boundary | Checked contract and result |
|---|---|
| Head ownership | Rank `r` owns key heads `4r..4r+3` and value heads `8r..8r+7`. Loader slices Q/K by 512, V/Z by 1024, a/b by eight (`multichip_decoder.py:670-691`). Each local key head pairs with two adjacent value heads. The rank boundaries preserve that grouping. |
| a/b projection padding | Local state projection is `[K512,V1024,a32,b32]`, width 1600. Only the first eight columns of each scalar region are logical channels. The slices `1536:1544` and `1568:1576` agree with the loader's separate 24-column pads (`:404-406,673-683`). Treating this as a compact width 1552 or putting b immediately after eight a columns would be wrong. |
| QKV/tap order | Runtime concatenation gives `[Q512,K512,V1024]`; taps are packed in exactly the same rank-local order (`:403,685-697`). Initial zero convolution state can be sharded directly; subsequent updates preserve the local packed order. |
| Pair-swapped scan weights | Weight rows are pair-swapped at load time, matching the scan's activation pairing (`:666,682,684,690-691`). Chip boundaries are multiples of two and of the selected K-block sizes. No rank-local pair crosses a chip boundary. |
| Decode convolution sizes | Local QKV/taps are 2048 channels = 64 width tiles. Setup partitions exactly 64 tiles; debug writer stride changes 256→64 (`:60-61,170-265`). `DECODE_CONV_READER` reads the three old RM rows before in-place carry writes, shifts rows 1/2, and appends current QKV (`fused_decoder.py:3086`). The compute loop consumes four taps and one final accumulator per assigned tile; the writer consumes one output per tile. No new count mismatch was found. |
| Decay lookup | Table is sharded to eight head rows. Both head loops in the reused reader are changed from 32 to eight; input logical width is eight and the FP32 output remains logically eight wide (`:59,273-341,715-719`; `fused_decoder.py:3298`). A full padded tile is deliberate. |
| Prefill ordering | Q/K repeat twice along the head axis. Q/K/V are passed as `[1,8,S,128]`, g/beta as `[1,8,S,1]`. `Chunk64OutputGraph` returns `[B,S,H,Dv]` (`:486-495`; `fused_decoder.py:2315-2372`), matching the later flattening to token-major `[1,S*8,128]`. No head/time permutation discrepancy was found. |
| Persistent recurrent state | Each request allocates FP32 local `[1,8,128,128]`, sharded on head dimension. Decode supplies the original state tensor as `output_state`; prefill copies its final state back (`:473-493,583-601`). Public-boundary BF16 rounding is inherited; internal multi-token chunks keep FP32 carry. No new ownership or rounding-polarity defect was established. |
| Weighted norm physical rows | `TP4WeightedNormTail` intentionally pads eight valid rows to 32 because the reused kernel processes four full-width tiles (`:64-74`; `fused_decoder.py:2380-2458`). It returns physical/logical 32-row carriers for the following custom reader. These are temporary carriers, not 32 recurrent heads. |
| Local SiLU head concat | Eight workers each read one valid weighted head row and four Z tiles. Reader page `head*4+tile`, heads 0–7, covers exactly local Z1024 / 32 tiles (`:85-163`; `fused_decoder.py:2883`). Output width shards are eight ×128. The class docstring's references to Z4096/32 shards are stale; executable sizes are 1024/eight shards. |
| Linear output reduction | Prefill native output projection and ordered decode output return FP32 rank partials. `_reduce` preserves input dtype through the collective and casts to BF16 afterward (`:618-621,648-649,870-884`). Source inspection of the CCL wrapper shows dtype-aware FP32 compute-config resolution; no forced per-rank BF16 cast was found on this attention-output path. Numerical accumulation order still needs measurement. |
| Indexed/shared TP weights | Indexed gate/up local packing is `[gate128,up128]`, paired with down K128 (`:775-782,953-958`). Shared scalar is replicated and gates each local down partial before summation (`:798-812,902-908`), which is algebraically consistent. No packing contradiction was found. |
| Runner state assembly | `run_multichip_decoder.py:292-297` reconstructs conv as global `[Q2048,K2048,V4096]` and concatenates recurrent states along contiguous head ownership. Full KV assembly checks duplicated rank pairs and compares written logical cells (`:281-291`). These mappings match the loader. |

## Other potential issues and validation gaps

These do not explain the logged reshape failure and are not promoted to additional hardware root causes.

1. **Per-row output acceptance can be masked.** The runner records prefill and decode per-row PCC (`run_multichip_decoder.py:360-364`), but `passed` checks only aggregate output PCC plus state PCC (`:373-376`). An isolated output-row failure can be diluted by other rows or tokens. Gate the already recorded per-row output scores as well; their presence in JSON alone does not enforce the threshold.
2. **Linear batch decode changes router arithmetic.** Original `decode_batch` calls `_moe` without `ordered_router=True` (`multichip_decoder.py:1140`), while cached `_forward` passes it (`:1089-1090`). The baseline linear cached-decode path enables ordered routing (`fused_decoder.py:1805`) and `linear` selects a separate scan branch for that flag (`:688-709`). The multichip router also selects from FP32 logits with native `topk`, whereas the baseline singleton linear path uses BF16 logits plus `inline_selector` (`optimized_decoder.py:237-250`). These are real numeric-path differences, but no captured logits/IDs prove a wrong route in the supplied smoke. With IDs and weights paired, a top-k order difference preserves the sum in exact arithmetic; finite-precision accumulation can still differ. Preserve this distinction when investigating later PCC failures.
3. **Trace evidence is incomplete.** Stable inputs, positions, snapshots, and restores appear coherently ordered (`runner:91-107,173-207`), and measured/replayed execution has explicit synchronization or blocking execution. However, eager equality covers only replay step zero (`:210`). `changing_inputs` (`:246`) checks host inputs, not whether later replay outputs follow those changed inputs. A stale-input trace can be deterministic. Compare every replay step against an eager changing-input/state control.
4. **No live-trace resumed-prefill case.** The runner performs all prefill before capture and never calls prefill while the same decode trace remains live. It therefore does not validate that lifecycle or continuation-state interaction. This is a coverage gap, not evidence of an existing lifetime violation.
5. **Numerical TP equivalence remains open.** Distributed norm, TP output summation, indexed-TP versus EP accumulation, router precision, and linear scan/recurrence numerics need paired outputs and per-head state checks. Neither aggregate correlation nor the source-consistent packing ledger proves them correct.

## Case coverage and proposed AutoFix checks

| Case | What this investigation establishes |
|---|---|
| Supplied full-attention layer 3, prefill32, batch1 decode | Exact observed reshape cause; grouped-prefill versus indexed-decode contrast explained. |
| Full-attention single-token MoE elsewhere | Same indexed reduction boundary is vulnerable in the original source, including singleton prefill. Not independently executed here. |
| Multi-row batched MoE | Uses grouped MoE, so the indexed reshape diagnosis is not a claim that this branch fails. |
| Linear prefill of any positive length | Original first scan projection has the factor-32 logical-volume error. Longer prefill reaches it through per-row projection recursion. |
| Linear decode / singleton prefill after projection repair | Original recurrent reduction has the eight-versus-32 head mismatch. |
| Multi-token linear chunk64 recurrence | Packing/state flow is source-consistent; no corresponding head-promotion failure proved in that separate recurrence. |

Follow-up checks belong to AutoFix/device validation, not to this inspection:

- Inspect all three reduction boundaries with explicit assertions for both logical and padded shapes. For linear projections cover widths 1600, 1536, 2048 and K-blocks 512/1024. For recurrent sums use nonzero, distinct data in all eight heads so a head-order or padding error cannot hide behind zero state.
- Re-run the exact layer-3 batch1 smoke after the indexed-MoE repair. Verify completion through host reads, accuracy comparison, capture, blocking replay, and state checks; getting past the reshape alone is insufficient.
- Before a broad linear sweep, validate one linear layer with prefill lengths 1, 32, 33, 64, and 65, followed by changing-input multi-step decode. Include more than one batch row and distinct state per row, then check each output row and each reconstructed state.
- Check resumed nonaligned prefill and same-trace replay against eager continuation, including zeroing/restoring state between controlled runs. A failed call can already have enqueued cache/convolution updates before a later host assertion; do not treat its state as an untouched initial state.

## Report review and limits

The headline findings were rechecked against the original matching hashes, exact traceback, reducer wrapper, output-spec builder, program-factory dimensions, tile reader/compute behavior, reshape overloads, baseline preparation paths, and binary broadcast validation. A standard-library host calculation confirmed the concrete volumes and two-stage recurrence shape above. The four inspected Python files parsed successfully at investigation start.

The leading diagnosis predicts the exact assertion and why the supplied grouped prefill avoids it. The two linear findings each have a direct source-contract contradiction and are explicitly separated from the reported full-attention execution. Packing, cache rounding, custom-kernel tile size, trace lifetime, and numeric-route concerns were challenged against their producer/consumer invariants; unsupported failure stories remain secondary or were cleared. Concurrent source fixes were inspected for the relevant shape changes only. Hardware behavior, PCC, performance, and the complete revised implementation remain unverified by AutoDebug.

## B32 profiler source-location overflow (CPU-only isolated audit)

The broad B32 correctness fixture exceeds Tracy's signed-16-bit source-location
capacity during 32 serial prefills, before final signposted measurements. Both
uncached and cached op metadata attempts fail. Exact evidence is preserved at
`failures/profiler_b32_no_cache/` and `failures/profiler_b32_cached/`.
Independent CPU source audit (`linear_mesh_audit`) found:

- `ttnn/api/tools/profiler/op_profiler_serialize.hpp:339–389` caches JSON by
  program hash; this does not control Tracy's source-location count.
- `tt_metal/third_party/tracy/server/TracyWorker.cpp:3496–3503,3885–3897`
  implements the static/dynamic source-location limit. The failing message
  does not identify which family exhausted it.
- `tools/tracy/__init__.py:53–87` only pauses Python tracing hooks; native
  TTNN instrumentation stays active. `TTNN_OP_PROFILER` permanently latches
  enabled in `op_profiler_serialize.hpp:252–258`.
- `build/CMakeCache.txt` has `TRACY_ON_DEMAND:BOOL=OFF`; the wrapper starts
  capture before the workload. Delayed capture is not a supported drop-in fix.
- Device-zone pushing was already disabled by the effective guard in
  `tt_metal/impl/profiler/profiler.cpp:2664–2670`.

Recommended discriminating repair: prepare and export exact post-prefill states
in an unprofiled process; profile one representative S32 prefill and the real
B32 decode using those states. Keep full correctness/stress as separate evidence.
