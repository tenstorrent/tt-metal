# Migration: eltwise_where_sfpu_row_col_bcast.cpp -> compute_kernel_lib::eltwise_chain

## Scope
- Single kernel: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels_ng/compute/eltwise_where_sfpu_row_col_bcast.cpp`
- Mirrors the structure used by the already-migrated peer
  `eltwise_where_sfpu_row_bcast.cpp` (stride-3 DEST scratch chain for Where).

## Structural changes
- `process_tile(...)` is now `template <uint32_t num_tiles_per_cycle>` so the
  CB ids resolve to `constexpr` constants needed by the chain element template
  args. The outer `complete_iterations` / `remaining_iterations` `tile_freq`
  driver in `kernel_main` is unchanged.
- The post-`unary_bcast` stage that used to be hand-rolled (acquire, copy_tile
  to dst slot j*3, fill / copy true / false, BINARY_SFPU_OP, pack) is replaced
  by a 3-stage `compute_kernel_lib::eltwise_chain`:
  - `BlockCopyTileStride3Cond<cb_left, BlockSize>`: copies condition tiles to
    DEST slots `j*3`.
  - `LocalWhereStage<cb_right, BlockSize>`: per-tile copies the true/false
    tensor into slot `j*3+1` or `j*3+2` (depending on `WHERE_TTS`/`WHERE_TST`),
    fills the missing slot with `scalar_val`/`scalar_value`, runs the
    `BINARY_SFPU_OP` over the stride-3 dst triple.
  - `BlockPackTileStride3<cb_out, BlockSize>`: packs DEST slot `j*3` -> cb_out
    slot j.
  - `compute_kernel_lib::eltwise_chain(1u, CondLoad{}, mid, PackStage{});`
- The pre-stage `unary_bcast<BroadcastType::ROW>` to `cb_llk_post` is preserved
  byte-identical (it produces the row-broadcasted "other" input on top of which
  the inner Where chain runs). The bcast direction enum is unchanged from
  the original (BroadcastType::ROW for the inner stage).
- CB lifecycle (wait_front/reserve_back/push_back/pop_front) stays on the outer
  scope; chain elements use `NoWaitNoPop` policies, identical to the peer.

## Trait additions vs peer
The peer file `eltwise_where_sfpu_row_bcast.cpp` does NOT compile on this branch
in its current form: the chain pipeline now requires every element to expose
`static constexpr uint32_t lane_width`, otherwise `chain_lane_width_v<Chain>`
fails to instantiate (see `eltwise_chain.inl:824`). The same blocker affects
`eltwise_where_no_bcast.cpp` and presumably the other stride-3 Where kernels.

To keep my migrated kernel compilable, I added
`static constexpr uint32_t lane_width = 3;` to each of the three chain element
structs (matching the stride used by `BINARY_SFPU_OP` over `j*3, j*3+1, j*3+2`).
This is a strictly additive trait; behavior is unchanged because
`AutoBlock::Off` is the default and `block_size` is forced to 1, so the lane
width is consulted only by `chain_lane_width_v` for the constexpr trait
evaluation.

I did NOT touch the peer or `eltwise_where_no_bcast.cpp` (out of scope).

## Tests
Targeted pytest hitting the migrated kernel directly:
- `tests/ttnn/nightly/unit_tests/operations/eltwise/test_ternary_bcast.py::test_ttnn_where_row_col_mixed_bcast_tts` (2 cases) -> PASS
- `tests/ttnn/nightly/unit_tests/operations/eltwise/test_ternary_bcast.py::test_ttnn_where_row_col_mixed_bcast_tst` (2 cases) -> PASS

These exercise both `ROW_A_COL_B` and `ROW_B_COL_A` x `WHERE_TTS`/`WHERE_TST`.
Max diff was 0 in all cases (the migrated kernel is bit-exact vs the original
stride-3 hand-rolled body, which is expected since the chain is just a
restructuring of the same LLK calls).

Note: `tests/ttnn/unit_tests/operations/eltwise/test_where.py::test_ttnn_where`
fails on the very first parameter (no_bcast, 32x32) with a `lane_width` not a
member compile error in `eltwise_where_no_bcast.cpp` -- pre-existing branch
breakage, independent of this migration. Verified by stashing the migration
and rerunning the same test (same error). Not addressed here per scope
discipline.

## Final
migrated. test: PASS. notes: 4/4 row+col where pytest cases pass; peer +
no_bcast where kernels are independently broken on this branch (missing
`lane_width` trait) and are out of scope.
