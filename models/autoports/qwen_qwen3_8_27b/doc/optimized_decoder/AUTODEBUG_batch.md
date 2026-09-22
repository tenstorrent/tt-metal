# AutoDebug: optimized decoder batch row packing

Status: source diagnosis confirmed against the saved exception and native
validation code; proposed repair still requires the main agent's device tests.
This investigator ran no target code or hardware operations and made no
implementation changes.

## Evidence

- Stage 3, Qwen/Qwen3.8-27B, starting HEAD
  `ad43d1388fd610fb14356b428593bc806a403fcd`, with untracked optimized source,
  tests, and evidence already present.
- Failing command from `commands.log`:

  ```bash
  models/autoports/qwen_qwen3_8_27b/tests/run_optimization_experiment.sh batch3_probe --layer 0 --length 257 --batch 3 --continuation --activations /home/mvasiljevic/qwen38-full-rerun/tt-metal-cache/optimized_decoder_activations --policy-file models/autoports/qwen_qwen3_8_27b/doc/optimized_decoder/selected_candidate.json
  ```

- `batch3_probe.log` reaches `TT_PREFILL_BEGIN`, logs two nonrectangular
  reshape fallbacks, then raises `Shard height 32 must match physical height
  96 for width sharded`. The actual saved traceback is `_finish`, line 321,
  `down = ttnn.to_memory_config(down, memory)`, rather than `_norm`.
- Saved implementation SHA256:
  `874baf217892c867e60f0ec2eb7bbb8978d916bcee05f71942aa7c4d7218f5eb`.
  Its immutable copy is in `sources/<hash>.py.txt`. Decoder line references below
  use that copy. The live file also has independent prefill-2D policy changes;
  those do not change this failing one-token branch.
- Selected policy uses DRAM-sharded BFLOAT4_B/LoFi projections, BF16
  activations/output, sharded norms, carried residuals, residual/gate/up
  grids of 80 cores, and separate gate/up projections. The existing readers
  diagnosis addresses a different defect and is not the cause of this error.

## Root cause and native contract

A logical user is being moved between an outer dimension and the tiled row
dimension. Equal logical volume does not make that operation a metadata view:

| B | Public `[B,1,H]` padded height | Packed `[1,1,B,H]` padded height |
| --- | ---: | ---: |
| 1 | 32 | 32 |
| 2 | 64 | 32 |
| 3 | 96 | 32 |
| 32 | 1024 | 32 |

H is tile-aligned. Public rows occupy one tile-height slice per user; packed
users occupy consecutive rows of a single tile for B <= 32. Increasing the
shard height to `B*32` at the failing call would mask the representation
mistake and expand residual storage instead of retaining the intended packed
matmul path.

Native evidence, all relative to repository root:

1. `tt_metal/impl/tensor/spec/layout/tensor_layout.cpp:341` computes physical
   height by walking dimensions from the inside outward and applying tile
   alignment before multiplying outer batch dimensions. Its padded-shape
   implementation at line 391 gives the table above.
2. `tt_metal/impl/tensor/spec/tensor_spec.cpp:49` requires width-shard physical
   height to equal tensor physical height. The constructor asserts this at
   line 161, producing the saved exception.
3. `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:613`
   permits a tiled metadata view when the final width matches and the
   penultimate dimension either matches or both old/new dimensions are
   tile-aligned. `[1,1,B,H] -> [B,1,H]` is not such a view for B > 1,
   including B=32 because the destination inner height is 1.
4. That file's general sharded reshape path at lines 441-494 recomputes the
   shard configuration. Its helper at lines 119-129 falls back to interleaved
   on nonrectangular grids, while lines 177-185 can keep a rectangular width
   grid with a newly derived height. Therefore a returned tensor may remain
   sharded with height `B*32`, or become interleaved; neither outcome retains
   the original packed-height contract. The two warnings are compatible
   with gate/up output conversions, but identifying their exact producers
   needs instrumentation rather than inference from the warning alone.
5. `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard/device/reshard_device_operation.cpp:309`
   constructs its output using the unchanged input logical shape plus the
   requested memory config. Consequently reshard cannot repack public
   `[3,1,H]` data into a width-shard config with height32.

This explains both why batch1 passes and why changing only the final reshard
would leave sibling conversions and excess work unresolved. `_norm` is another
instance of the same faulty assumption, although this build's automatic shard
recomputation allowed it to advance in the saved run.

## Minimal coherent branch changes

Use a private one-token representation `[1,1,B,W]` only inside the optimized
residual/MLP path. Retain public/helper `[B,1,W]` at attention boundaries. Keep
the selected projection dtypes, compute configs, reader counts, and DRAM
matmul program. Do not select FunctionalDecoder or a generic matmul for B > 1.

Packing/unpacking must use logical B, never padded height. For rank3 one-token
input B is `shape[0]`; for the private rank4 input B is `shape[-2]`. Distinguish
these cases explicitly: current `shape[0]` on packed data would silently treat
every batch as 1. Conversion to interleaved must precede a physical reshape;
converting after it does not avoid the erroneous sharded reshape.

1. **`_linear`, lines 199-229.** Accept public one-token input and the private
   packed rank4 input. Derive B as above; construct activation shard height
   `ceil(B/32)*32` and `per_core_M=ceil(B/32)`. An already-packed tensor needs
   no public round trip before `to_memory_config`. For a public sharded input
   with B > 1, interleave first, then pack, then shard. With
   `keep_sharded=True`, retain the packed projection output for B > 1 instead
   of reshaping it to `[B,1,N]`. With `keep_sharded=False`, explicitly
   interleave the packed result before restoring `[B,1,N]`, even if the input
   was packed. That default public result preserves existing attention and
   optional packed-MLP slicing contracts. Preserve batch1's current return
   shape/zero-cost views and the separate multi-token prefill paths.
2. **`_norm`, lines 273-287.** Support the private packed rank4 shape for
   layernorms; do not reinterpret its outer `1` as B. Use the same packed
   shape through RMSNorm. For an input already packed, return packed. For
   public input with B > 1, interleave output before restoring its public
   shape, even when `carry_residual=True`; this is the input-normalization
   boundary before `_delta`, `_full_decode`, or `_full_prefill`. Batch1 keeps
   its existing sharded return. If supporting B beyond32, norm shard height
   and `block_h` must both derive from `ceil(B/32)`; the requested B2/B3/B32
   range uses height32/block_h1. Head norms are separate and unchanged.
3. **`_finish`, lines 296-322.** For B > 1 keep `h = add(packed_x,
   packed_attention)` packed; remove its public reshape before the
   post-attention norm. Feed packed h to `_norm`, then feed packed n to
   gate/up projections. Their packed outputs can be multiplied directly.
   Keep down packed through its conversion to the residual memory config
   and through `add(h, down)`. Interleave the final result, then reshape once
   to the saved public `[B,1,H]`. The final add must use the retained packed
   h, not a public alias. A B > 1 branch permits retaining the original batch1
   implementation and the current non-carried/multi-token branch verbatim.
4. **Optional MLP branches.** `packed_mlp` currently slices rank3 output with
   `packed[:,:,:width]`; do not let private rank4 escape into those slices.
   The default `_linear(..., keep_sharded=False)` public result described
   above preserves them, and down can repack their public product.
   Alternatively make both slices explicitly target the last axis before
   retaining packed shape. `minimal_mlp` may receive packed n and return
   packed interleaved product: its native validator accepts rank>=2 and takes
   M/K from the last two dimensions
   (`experimental/minimal_matmul/device/minimal_matmul_device_operation.cpp:65`).
   It still needs the existing conversion to DRAM before minimal matmul and
   the rank-aware down projection. Neither optional policy is selected here.
5. **Decode/public return, lines 608-618.** `_norm` must return rank3 here so
   attention helper batch extraction remains valid. `_finish` must restore
   rank3 before `decode_forward` returns. The existing higher-batch Q/K head
   norm interleaving in `_full_decode` is appropriate; do not remove it.
6. **Prefill tails, lines 620-657.** The final one-token chunk of length257
   enters exactly the same optimized branch. Unaligned full-attention
   continuation also calls `_full_decode` one token at a time until page
   alignment. Ensure every `finished` chunk is public rank3 before appending
   or concatenating on time axis1. Existing DRAM conversion before multi-chunk
   concatenation then remains valid. Single-chunk length1 must also return
   public rank3: relying only on the `length > count` conversion misses it.

The other reshape sites in `_qkv`, `_full_decode`, and `_delta` consume
interleaved/helper shapes if the default `_linear` and `_norm` contracts above
are preserved. `_delta` line605 deliberately hides padded time rows while
retaining its physical padded shape; it does not move batch into time and
should retain that behavior. Its following one-token output projection must
perform the normal logical packing before the optimized DRAM matmul.

## Focused verify/refute experiments for the main agent

These are proposed tests, not completed results.

1. For B in {1,2,3,32}, upload distinct BF16 values for every user and feature.
   Capture logical/padded shape, dtype, layout, shard grid/shape at packed
   norm output, gate/up output, product, down before residual reshard, final
   packed add, and public output. Verify pack/unpack preserves every logical
   row exactly; inactive tile rows must never become logical users. For
   B2/B3/B32 every width-sharded MLP/residual tensor should have physical
   height32. Compare norm to a per-user reference and compare optimized
   projections to the same quantized TT weight with the same fidelity.
2. Assert tensor shapes explicitly before every host PCC comparison:
   prefill `[B,S,5120]`, decode `[B,1,5120]`, and each per-user state shape.
   The existing `pcc` flattens inputs and can miss a rank or axis contract
   regression with equal volume. Use per-user PCC as well as aggregate PCC
   so a bad row cannot be hidden by the larger batch. Keep the existing
   >=0.995 parity gate.
3. Rerun the exact original command under a new evidence name. It covers
   B3/layer0/257, one-token tail, continuation, eager decode, trace replay,
   repeated replay, refreshed inputs, and sequential traced decode.
4. Run the same selected policy for both layers0 and3, B2 and B3 with
   lengths `1,129,257` and continuation. Run B32 with lengths `1,33,129`
   and continuation. Length1 catches the public return without concat;
   lengths129/257 catch chunk tails; length33 continuation starts at16 and
   exercises full-attention page alignment. Preserve page-table row-swap
   and unused-page checks for full attention. Add a distinct-input batch
   permutation check for linear attention's conv/recurrent state.
5. Rerun batch1 selected-candidate layer0/layer3 correctness and its existing
   trace benchmark. No higher-batch performance improvement is claimed
   without measurements. Compare the native program arguments and dtype
   ledger to ensure the repair still selects optimized sharded projections.

If a follow-up error moves into the attention or cache kernels after these
representation fixes, record it separately and localize there. The saved
exception proves an illegal residual reshard, not a numerical or cache bug.
