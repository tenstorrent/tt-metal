# Step 3 — layernorm readers

Replace the hand-rolled full+partial scaler emission in the three layernorm readers with the
`calculate_and_prepare_partial_reduce_scalers` helper.

## What the workaround did

These readers already implemented the exact mechanism the helper now provides — two `calculate_and_
prepare_reduce_scaler` calls into the same CB, the second with a reduced fill count:

```cpp
constexpr uint32_t partial_last_tile_cols = W % tt::constants::TILE_WIDTH;

calculate_and_prepare_reduce_scaler<dfb_scaler, SUM, REDUCE_ROW, SUM_AND_MAX_REDUCE_FACTOR>();

if constexpr (partial_last_tile_cols > 0) {
    calculate_and_prepare_reduce_scaler<
        dfb_scaler, SUM, REDUCE_ROW, SUM_AND_MAX_REDUCE_FACTOR>(partial_last_tile_cols);
}
```

Note the asymmetry that made this worth consolidating: when `W` *is* tile-aligned the CB gets one
tile, when it isn't it gets two — and the knowledge that "tile 1 is the partial one" lived implicitly
in the ordering of two separate call sites, with the matching selection rule written independently in
`numeric.h`.

## What it is now

```cpp
if constexpr (partial_last_tile_cols > 0) {
    calculate_and_prepare_partial_reduce_scalers<
        dfb_scaler, SUM, REDUCE_ROW, partial_last_tile_cols, SUM_AND_MAX_REDUCE_FACTOR>();
} else {
    calculate_and_prepare_reduce_scaler<dfb_scaler, SUM, REDUCE_ROW, SUM_AND_MAX_REDUCE_FACTOR>();
}
```

Byte-for-byte identical CB contents in both branches — this is a pure refactor. The pairing is now
expressed by one named call instead of an ordering convention across two.

`partial_last_tile_cols` was already `constexpr` in all three readers (`W` / `W_logical` are
compile-time), so the template-parameter form of the helper applies directly; no runtime overload
needed here.

## Files changed

| File | Change |
|---|---|
| `.../dataflow/reader_unary_interleaved_ln.cpp` | dual emission → one helper call |
| `.../dataflow/reader_unary_interleaved_ln_large_tensor.cpp` | same (`W_logical`) |
| `.../dataflow/reader_unary_interleaved_ln_rm_gb.cpp` | same |

+33 / −20 lines (the growth is comment; the call sites shrink).

## What was deliberately NOT changed

The consumer, `normalization/kernel_util/compute/numeric.h`, still calls `reduce_tile` directly and
still carries its own copy of the selection rule:

```cpp
const auto scaler_tile_idx = block.to_global(j) == num_tiles - 1 && last_tile_partial ? 1 : 0;
```

That is `ReducePartialScaler::last_tile_at(1)` written by hand. Unifying it is Step 4, and is a much
larger change than this one: `numeric.h` is its own reduce framework (block policies, epilogues,
multi-CB accumulation) rather than a single call site. Keeping the reader swap separate means Step 3
is verifiable on its own as a no-op refactor.

## Test coverage check

The Step 2 lesson — a green suite that does not exercise the changed path — was applied here **before**
running anything. `test_layer_norm.py` parametrises `w` over 24, 42, 127, 519, 31, 487, 3821 and shape
pairs including (19, 2865), (1001, 4083), (32, 2592). Many are ragged (`42 % 32 = 10`,
`127 % 32 = 31`, `24 % 32 = 24`), so the partial path is genuinely covered here — unlike softmax.

## Test results

| Suite | Result |
|---|---|
| `test_layer_norm.py` | **265 passed** in 391.99s |

No failures. For reference, `test_layer_norm.py` + `test_softmax.py` together ran **638 passed,
1 skipped** at the helper commit before any migration; this step re-ran the layernorm half, which is
the part that touches these readers.
