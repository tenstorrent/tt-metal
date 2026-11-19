# Production integration draft: ordered recurrence, convolution, partial RoPE

This is a source-only draft, not a hardware-validated default. Nothing in `tt/` or `tests/` was changed by this audit. `proposed/tt/` contains the complete proposed decoder, four helper modules, and twelve kernel/header files. `integration.patch` targets the real model-local `tt/` paths; `source_manifest.json` records the source hashes. The patch passed `git apply --check`; all proposed Python files parsed, and none imports this model's tests. `build_draft.py` rebuilds these review files from the current sources and writes only inside this directory.

The draft retains the production CCL grid repair and current attention pool/table/SDPA logic. It excludes the postnorm candidate, whose repair is still in progress. Existing candidate evidence remains candidate evidence: the resulting production default needs its own correctness, trace, watcher, context, and performance measurements.

## Inheritance that must not be lost

The effective chain is `FlatOrderedConvCandidate -> FlatOrderedCandidate -> FlatRecurrentCandidate -> KernelFusionCandidate -> DirectHistoryCandidate -> TiledHistoryCandidate -> multichip_fusion_reference.MultichipDecoder`. `KernelFusionCandidate` also assigns `PartialRopeCandidate.full_attention_decode_batch` directly. Therefore the best convolution candidate already includes tiled pooled history, direct history row writes, and partial Q/K norm/RoPE. It does **not** inherit the current production decoder directly.

Relevant sources are `tests/multichip_flat_ordered_conv_candidate.py:11–38`, `multichip_kernel_fusion_candidate.py:11–14`, `multichip_history_row_write.py:22–26`, and `multichip_tiled_history_candidate.py:8–51,136–186`. Do not transplant the whole frozen reference or copy only the most-derived method. The draft composes the required changes into current production instead.

## Exact production changes

Line numbers below refer to the original production source recorded in the manifest.

| Method/site | Proposed change | Required preservation |
|---|---|---|
| Module imports; class attributes near line 96 | Import four model-local helpers; `decode_recurrent_wave=104` | Existing precision policy, CCL manager, projection geometry, and residual layout |
| `allocate_batch_state`, line 458 | Allocate pooled linear convolution history as TILE BF16 `[3,1,B,2048]` | `B=1` continues through `allocate_state`; recurrent remains FP32 `[B,8,128,128]`; full-attention allocation is unchanged |
| New `read_linear_state` before line 515 | Extract one pooled request, permute `[tap,1,1,C]` to `[1,1,tap,C]`, return ROW_MAJOR convolution history | Private tensors pass through; recurrent extraction remains axis 0 |
| `write_linear_state`, line 515 | Dispatch pooled convolution writes through `history_row_write` | Private copy and recurrent sharded `slice_write` remain unchanged; never replace backing pools |
| `linear_attention`, line 687 | Build a compact local state using the new read helper | Existing private prefill/decode implementation, logical-length history maintenance, and both writebacks |
| `pooled_linear_attention_decode_batch`, line 958 | Use `tiled_conv` then `flat_ordered`, wave 104; omit `pad_value` on gate reshapes | BF16 beta, FP32 g/state, post-recurrence BF16 norm and SiLU gate stay as in passing candidate; no postnorm fusion |
| `full_attention_decode_batch`, line 1096 | Replace only separate Q/K interleaving, norm and permutation/RoPE with `partial_norm_rope` in native shards | Current position handling, pool offsets, heterogeneous tables, SDPA alignment/chunk/grid, scatter-back, gate, and concat-head behavior |
| `decode_batch_packed`, line 1251 | Require shared pools, rows exactly `0..B-1`, TILE history shape `[3,1,B,2048]`, and recurrent pool size B | Duplicate-region rejection; incomplete/reordered/mixed pools use existing per-request path; independent private states retain their existing vectorized path |

Private `allocate_state`, `decode_forward`, `prefill_forward`, `prefill_batch`, `linear_attention_decode_batch`, `full_attention`, and `decode_finish` are not replaced. In particular, the existing B1 and non-aligned sequence implementations stay intact.

## Dependencies to promote

| Test source | Proposed production destination | Kernel files |
|---|---|---|
| `tests/multichip_flat_ordered.py` | `tt/flat_ordered.py` | `tt/kernels/flat_ordered/{reader.cpp,writer.cpp,compute.cpp,norm_stages.hpp}` |
| `tests/multichip_tiled_conv.py` | `tt/tiled_conv.py` | `tt/kernels/tiled_conv/{reader.cpp,writer.cpp,compute.cpp}` |
| Helper only from `tests/multichip_history_row_write.py` | `tt/history_row_write.py` | `tt/kernels/history_row_write/write.cpp` |
| `tests/multichip_partial_norm_rope.py` | `tt/partial_norm_rope.py` | `tt/kernels/partial_norm_rope/{reader.cpp,writer.cpp,compute.cpp,norm_rope_stages.hpp}` |

Kernel contents are copied unchanged. Relative `Path(__file__).parent / kernels/...` lookup moves naturally with the wrappers. `flat_ordered/compute.cpp:28` retains its repository include of `ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/kernels/compute/chunk_gdn_math.hpp`; no test file is needed for this dependency. The helper extraction removes candidate classes and CLI entrypoints. Raw recurrence, unordered flat recurrence, and postnorm wrappers/kernels are not needed.

## State and padding constraints

Private convolution state remains ROW_MAJOR BF16 `[1,1,3,2048]`. A pooled B>1 state shares TILE BF16 `[3,1,B,2048]`; `state_row` indexes axis 2. Recurrent state stays FP32 `[B,8,128,128]`, with the request on axis 0. These layouts must be documented together: slicing both state tensors on axis 0 silently misaddresses convolution history.

The direct writer handles one request without reading or modifying neighboring requests. Its 16 cores own 128 channels each; it copies three compact history rows with 32-byte tile-face writes (`kernels/history_row_write/write.cpp:12–29`). Sequential queued writes, restart of one request, and mixed/private pools therefore use the same ownership rule.

`tiled_conv` owns complete physical tiles, advancing all three history tap planes in place (`kernels/tiled_conv/writer.cpp:15–23`). It is safe only for a complete ordered pool. Incomplete prefixes, subsets, permutations, or states spanning pools must not use it, even when all inputs fit one physical batch tile. The draft's predicate rejects all these cases; the per-request path extracts compact state and uses the direct row writer. A single member of a larger pool uses that same path.

The no-gate-padding option is specific to `flat_ordered`: its reader selects `batch=head/8`, copies only the active gate scalar, and zeros the scratch rows/columns before computation (`kernels/flat_ordered/reader.cpp:12,52–63`). It does not consume the unspecified physical padding of the input gate tensors. Do not remove gate padding from the older chunk-GDN path by analogy. Likewise, do not merge in raw recurrence arithmetic: the ordered implementation deliberately preserves the chunk-order FP32 computation.

Partial RoPE normalizes all 256 channels and rotates only the first 64 NeoX channels, packing BF16 between norm and rotation. It updates Q/K in place in their native user shards and leaves V untouched. Its enumeration of the input core ranges must retain request order. Current SDPA compute ordering and Q rectangle-prefix sharding remain untouched; the B2/B3 concat-head special case also remains intact.

## Memory/context contract update required at integration

Update `doc/context_contract.json`'s `optimized_multichip_decoder.conv_history_layout` (currently line 127) and add physical pooled history sizing. Per device and linear layer:

* Private B1 history: `3*2048*2 = 12,288` bytes, unchanged.
* Old pooled history: `12,288*B` bytes.
* New pooled history for every B2..32: `3*32*2048*2 = 393,216` bytes because the request dimension pads to 32.
* Increase: `12,288*(32-B)` bytes; largest at B2 is 368,640 bytes. B32 is unchanged.
* Recurrent pool: `524,288*B` bytes, unchanged. KV dtype/layout/context and setup projection/rope storage are unchanged.

The draft must not claim a tested context reduction or improvement. Keep the current advertised context pending rerun; establish available memory and retain the largest feasible contract. Reallocate old pooled states and recapture traces after this layout transition—there is no in-place migration of existing ROW_MAJOR pools. Public logical input lengths remain arbitrary; allocation context alignment remains the existing allocation API contract.

## Required integration verification

1. Apply-check against the current production source hash, promote helpers/kernels, and inspect the method-only diff. Re-run fallback audit: runtime imports must resolve entirely through `tt/`, not test subclasses or monkeypatches.
2. Exercise private B1 for both layer kinds and B2..32 complete pools, particularly B2,3,6,13,14,16,31,32. B13 fills one 104-head recurrence wave; B14 crosses the wave boundary. Vary token values and per-request positions across eager and trace calls.
3. Run production `multichip_pool_contracts --layer 0` and `--layer 3` with its unchanged gates. It already understands TILE history in host state comparisons (lines 157–160), covers subsets/permutations/mixed capacities, checks untouched requests bitwise, and retains multiple live traces. Compare repeated resets/re-prefills and changed input tensors; preserve distinct backing addresses and state-region checks.
4. Repeat candidate boundary and long-answer cases through the production entrypoint. Existing references include `flat_ordered_conv_nopad_boundary_{6,32}_raw.json`, `flat_ordered_conv_nopad_extended_answer1024_32.json`, and partial-RoPE B6/B32 artifacts. Reusing a reference is appropriate; reusing a candidate's timing as the final default timing is not.
5. Cover non-aligned prefill/continuation lengths 1,2,3,31,32,33,1023,1024,1025 and max-context terminal windows. Include starts immediately before/after KV page and SDPA chunk boundaries, randomized page mappings, and heterogeneous private/pool capacities. No public alignment restriction may be introduced.
6. Collect watcher-clean runs separately from profiler runs. Test multiple decoder instances and shared/separate CCL workspaces, using the production full-grid semaphore repair. Changing state/input addresses and interleaved live traces must exercise generic-op cache rebinding, in-place recurrent writes and direct history row writes.
7. Measure warmed prefill and traced warmed decode on the final default TP4 path, then collect operation tables/CSV/provenance and compare with the accepted baseline. Update context and residual-layout documentation, rerun stage review, and commit only after the parent completes those gates.

No hardware execution or performance claims were made by this audit.
