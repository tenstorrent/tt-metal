# Step 11 — Phase 4: `moreh_{sum,mean}_w` and the generic MIN kernels

**Outcome: no code change.** Both Phase 4 items are blocked, each on a different missing piece. Both
verdicts are backed by something checked rather than assumed.

## 11a — `moreh_sum_w` / `moreh_mean_w`: wrong scaler layout, and a property worth keeping

These kernels do not reduce with `reduce_tile` at all. The W reduce is a **matmul against a scaler
column vector**:

```cpp
matmul_init(cb_input, cb_scaler, false);
matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx);   // out[r][0] += sum_c in[r][c] * scaler[c][0]
```

so the ragged tail could in principle be handled by zeroing the scaler's rows `>= mask_w` instead of
masking the input tile's columns. Structurally this is the *good* per-call form (one matmul per tile,
accumulating in DST), not the accumulate-then-reduce form step 8 rules out. Two things block it anyway.

### The existing helpers produce the wrong tile layout

`generate_mm_scaler` (`ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp`) writes the scaler into
**column 0 of every row**, in faces 0 and 2:

```cpp
for (int i = 0; i < 128; i += 8)   { ptr[i] = single_packed_scalar; }   // face 0, col 0 of rows 0..15
for (int i = 256; i < 384; i += 8) { ptr[i] = single_packed_scalar; }   // face 2, col 0 of rows 16..31
```

`prepare_reduce_scaler` / `prepare_partial_reduce_scalers` instead produce a **row-0 fill per face**
(`fill_each_face_row0`), which is what the reduce LLK wants and is a different tile entirely. A partial
matmul scaler would need a new generator — "fill column 0 for the first `partial_positions` rows" — not a
reuse of the partial-scaler helpers. That is a new dataflow helper, i.e. a feature, not a migration.

### The current mask is demonstrably robust to poisoned padding; a partial scaler would not obviously be

Measured on device (`W = 95`, so 2 full tiles plus 31 valid columns), sweeping the padding value with
`ttnn.fill_implicit_tile_padding`:

| padding | `moreh_sum_w` | `moreh_mean_w` |
|---|---|---|
| untouched | finite, rel err 0.0072 | finite, rel err 0.0100 |
| `0` | finite, rel err 0.0067 | finite, rel err 0.0111 |
| `1000` | finite, rel err 0.0054 | finite, rel err 0.0092 |
| `+inf` | finite, rel err 0.0070 | finite, rel err 0.0109 |
| `NaN` | finite, rel err 0.0109 | finite, rel err 0.0114 |

(errors are ordinary bf16 accumulation error and do not vary with the padding value)

So `mask_tile` does **not** turn `inf`/`NaN` padding into `NaN` output — it is not behaving like a plain
multiply by zero. A partial matmul scaler would instead compute `garbage * 0` inside the FPU, which is
exactly where `inf * 0 = NaN` would appear. Migrating would trade a *measured* robustness property for
an unverified one, to save one mask CB, one scratch CB and one accumulator CB in two kernels.

**Verdict:** revisit only together with a `generate_partial_mm_scaler` helper, and only after re-running
the table above against the new implementation. The mask stays until then.

## 11b — `reduction/generic/reduce_{h,hw,w}_neg.cpp`: the helper cannot hold N accumulators

These are the generic reduction op's MIN kernels (negate → reduce → negate). Step 5 listed them as a
migration target for goal 2 ("no kernels calling `reduce_tile` directly"). The FPU branch reduces a
chunk of `ntiles` columns, each into **its own DST register**, and carries `ntiles` running accumulators
across the H loop through a CB:

```cpp
for (uint32_t ht = 0; ht < Ht; ++ht) {
    ...
    if (ht > 0) { for (i < ntiles) copy_tile(dfb_acc, i, i); }      // reload ntiles accumulators
    reduce_init<REDUCE_OP, REDUCE_DIM>(dfb_ineg, dfb_scaler, dfb_acc);
    for (uint32_t i = 0; i < ntiles; ++i) { reduce_tile<...>(dfb_ineg, dfb_scaler, i, 0, i); }
    ... pack ntiles tiles back to dfb_acc
}
```

`compute_kernel_lib::reduce`'s `Accumulate` reloads exactly **one** tile:

```cpp
copy_tile(accumulate.config.cb_accumulator, 0, accumulate.config.dst_index);
```

so it cannot express `ntiles` parallel accumulators. The only way to fit the current helper is one
`reduce()` call per column, which throws away the chunking these kernels are built around — a
pipelining regression on the generic reduction op, which is far more widely used than anything else in
this migration, and this migration has no perf mandate to spend.

**Verdict:** blocked on a multi-slot accumulator (`Accumulate` reloading `n` tiles into `n` DST
registers). That is the same class of gap as step-4's block-wise input policy: a helper feature, worth
scoping as one.

## Where this leaves goal 2

"No kernels calling `reduce_tile` directly" now has a complete accounting:

| Caller | Status |
|---|---|
| `topk_router_gpt` | **done** (step 7) |
| `deepseek_prefill` ×2 | blocked — multi-slot DST + two outputs per acquire; runtime CB ids in a shared framework (step 7e) |
| `reduce_{h,hw,w}_neg` | blocked — multi-slot accumulator (this step) |
| `sdpa/compute_common.hpp` | expressible, but perf-critical; needs an SDPA perf run |
| `numeric.h` + `layernorm_large_tensor`, `layernorm_sharded*`, `layernorm_distributed`, `rms_allgather` | out of scope by request (normalization family) |

So the reachable part of goal 2 is finished, and every remaining caller has a named blocker.
