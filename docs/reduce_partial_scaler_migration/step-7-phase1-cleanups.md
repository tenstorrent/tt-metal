# Step 7 — Phase 1: close out Steps 1, 2 and 6

Four small items that Steps 1–6 knowingly left open, plus one latent hang found while doing them.

## 7a — The `softmax_small_w` regression (Step 6's open item)

Step 6 measured a real ~2–3% regression on `softmax_small_w`, on **both** the aligned and the ragged
variant, and named the likely cause: the migrated kernel paid for shape-genericity unconditionally.
The reader always emitted two max-scaler tiles and the compute kernel always selected tile 1 for the
last tile, even when the shape was tile-aligned and tile 1 was byte-identical to tile 0.

The fix makes the decision per shape instead of per build-configuration:

```cpp
// compute (moreh_softmax_{h,w}.cpp)
constexpr uint32_t mask_w = get_compile_time_arg_val(2);   // valid cols in the last W tile
constexpr bool do_partial_w = mask_w < TILE_W;
constexpr uint32_t num_max_scaler_tiles = do_partial_w ? 2 : 1;
constexpr auto max_partial_scaler = do_partial_w ? ReducePartialScaler::last_tile_at(1)
                                                : ReducePartialScaler::none();
```

```cpp
// reader (reader_moreh_softmax_{h,w}.cpp)
if (mask_w < tt::constants::TILE_WIDTH) {
    calculate_and_prepare_partial_reduce_scalers<cb_max_scaler, MAX, REDUCE_ROW>(mask_w);
} else {
    calculate_and_prepare_reduce_scaler<cb_max_scaler, MAX, REDUCE_ROW>();
}
```

Step 6 proposed plumbing `mask_w` to compute as a *runtime* arg. It went in as a **compile-time** arg
instead: `Wt` (and `Ht`) is already a compile-time arg for these kernels, so they are rebuilt per shape
regardless — nothing is gained by deferring the decision to runtime, and a compile-time value lets the
whole partial-scaler path constant-fold away on aligned shapes. The reader keeps its runtime `mask_w`,
because adding a compile-time arg there would shift the `TensorAccessorArgs<1>` base index.

Host side, `mask_h`/`mask_w` moved out of the per-core runtime-arg loop (it never varied per core) so
the CB sizing can use it: the max-scaler CB is now `do_partial ? 2 : 1` tiles.

The reader/compute predicate has to agree exactly, or the compute kernel waits for a tile the reader
never emits and the program hangs. Both sides compare against the same `TILE_WIDTH`/`TILE_HEIGHT`
constant (32) rather than against the tensor's own tile dims.

## 7b — A hang in the `ttnn` general softmax, introduced by Step 2

`ttnn::prim::SOFTMAX_KERNEL_PATH_GENERAL` points at
`ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels`. The `ttnn.softmax` "general" path
**shares the moreh softmax kernels**, so Step 2's change to `reader_moreh_softmax_{h,w}.cpp` and
`moreh_softmax_{h,w}.cpp` silently applied to two more factories that were not updated with it:

- `normalization/softmax/device/softmax_program_factory_general_w_small.cpp`
- `normalization/softmax/device/softmax_program_factory_general_h_small.cpp`

Both size `c_2` (max scaler) at exactly one tile. After Step 2 the shared reader emitted **two** into
it and the shared compute kernel waited for **two**. A one-page CB cannot satisfy either side: the
reader blocks in `reserve_back` for the second tile, the compute kernel blocks in `wait_front(2)`, and
the program deadlocks. Reachable from plain `ttnn.softmax` — `select_program_factory` picks
`GeneralWSmall` for any last-dim softmax that fits in L1 (and is not the rank-4 attention path), and
`GeneralHSmall` for `dim == rank - 2`.

It went unnoticed because Step 2 ran only `test_moreh_softmax.py`. The suite that covers this path is
`tests/ttnn/unit_tests/operations/fused/test_softmax.py::test_softmax`, which parametrises
`h ∈ {24, 32, 64} × w ∈ {42, 32, 64} × dim ∈ {-1, -2, -3, 0, 1, 2}` on rank-3 tensors — ragged on both
axes, and it poisons the padding with `-42` via `fill_implicit_tile_padding` first. It would have
failed (as a timeout) immediately.

Fixed as part of 7a: both general factories now size `c_2` from the same ragged predicate and pass
`mask_h`/`mask_w` to the compute kernel. `is_softmax_general_{w,h}_small_available` — the L1-fit
estimate that gates these factories — was updated to match, and while there it was corrected to count
the sum-scaler tile it had always omitted. It can now only be more conservative than before, so a
borderline shape may fall back from `*Small` to `*Large`; both are correct paths.

**Lesson, and it is the same one as Step 2's:** these kernels are shared. Changing a kernel under
`moreh/` is not necessarily a moreh-only change — grep for the kernel *path* (not just the op) before
assuming the blast radius.

## 7c — Dead runtime args removed (Step 1's open item)

Step 1 left `mask_h` being passed as a runtime arg by `moreh_sum_h` and `moreh_mean_h` after the
readers stopped reading it, on the grounds that removing it meant touching the per-core emission loops.
Removed now, along with the comments that explained its presence: one arg from
`moreh_sum_h_program_factory.cpp`'s `emplace_runtime_args`, and for `moreh_mean_h` both the
`AddRuntimeArgsForNode` entry and the `"mask_h"` name in the kernel's `runtime_arg_schema`.
The compile-time `PARTIAL_H` define / `partial_h` named arg that replaced it are untouched.

## 7d — `topk_router_gpt` off direct `reduce_tile` (Step 5's clean candidate)

`experimental/topk_router_gpt/device/kernels/compute.cpp`, both single-tile reduces in the
collector's softmax phase:

```cpp
compute_kernel_lib::reduce<
    PoolType::MAX, ReduceDim::REDUCE_ROW, cb_softmax_tmp_id, cb_bcast_scaler_id, cb_reduce_scalar_id,
    ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::single());
```

`WaitUpfrontNoPop` is the policy that matches what the kernel did by hand: wait for the input tile but
leave it in the CB, because the next step reads it again. As Step 5 predicted, the surrounding
`wait_front` / `reserve_back` / `push_back` came out with the `reduce_tile` line — the helper owns them.
The SUM reduce's `recip_tile` became the `post_reduce_op` lambda, and the operand-swapped
`reconfig_data_format(scaler, input)` that SUM+REDUCE_ROW needs is now the helper's business.

One behaviour change: the helper always issues its reconfig pair (default
`ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT`), so the MAX reduce now gets an unpacker reconfig and a
`pack_reconfig_data_format` it did not have before. That is the safe direction, not a correctness
change, but it is not free — if this op is ever profiled and the reduces show up, `reconfig_mode` is the
knob.

## 7e — Both `deepseek_prefill` sites are NOT clean conversions

Step 5 listed these as "Likely clean (single tile)". Reading them says otherwise; **neither is
converted**, and Step 5's verdict is wrong on both.

**`per_token_cast_to_fp8/.../compute_per_token_cast_to_fp8.cpp` — structurally blocked.** The
`reduce_tile` is not a single-tile reduce at all: it reduces `block_wt` tiles into `block_wt` *separate*
DST slots, folds those slots together with `binary_max_tile` (SFPU), clamps, multiplies by `1/448`,
copies to a second slot, takes a reciprocal, and packs **two different output CBs** — all inside one
`tile_regs_acquire`. `reduce<>` owns its DST lifecycle and packs one output per call, so it cannot
express "reduce into N slots, then keep computing on them, then produce two outputs". This is the same
blocker class as `numeric.h` (see step-4), not a per-tile detail.

**`moe_grouped_topk/.../moe_gate_common_compute.hpp` — blocked by convention.** The `reduce_tile` in
`normalize_scores` *is* a clean single-tile no-pop reduce. But every function in this shared framework
takes its CB ids as **runtime** `const uint32_t` parameters, and `reduce<>` takes them as template
parameters. The callers do pass `constexpr` values, so `normalize_scores` could be templated on its CB
ids — but that leaves one function in the file with a different signature convention than the other
seven, which is exactly the "two mechanisms in one small framework" outcome step-4 rejected for
`numeric.h`. If this file is migrated it should be templated as a unit, and that is a refactor of the
framework rather than a reduce migration.

Net effect on the end goal: two more files move from "pending" to "blocked, with a reason", and the
Step 5 recommended order loses its items 1 (done) and 2 (not viable).

## Files changed

| File | Change |
|---|---|
| `moreh_softmax/device/kernels/moreh_softmax_{h,w}.cpp` | `mask_{h,w}` CT arg; conditional scaler-tile count and partial scaler |
| `moreh_softmax/device/kernels/reader_moreh_softmax_{h,w}.cpp` | emit the pair only when ragged |
| `moreh_softmax/device/softmax_{h,w}_small/*.cpp` | hoist `mask_{h,w}`; CB `c_2` sized `1 or 2`; pass it to compute |
| `normalization/softmax/device/softmax_program_factory_general_{h,w}_small.cpp` | same, fixing the hang |
| `normalization/softmax/device/softmax_device_operation.cpp` | L1-fit estimate counts the scaler pair + the sum scaler |
| `moreh_sum/device/moreh_sum_h_program_factory.cpp`, `.../reader_moreh_sum_h.cpp` | drop dead `mask_h` RT arg |
| `moreh_mean/device/moreh_mean_h_program_factory.cpp`, `.../kernels/reader_moreh_mean_h.cpp` | same, incl. `runtime_arg_schema` |
| `experimental/topk_router_gpt/device/kernels/compute.cpp` | two `reduce_tile` blocks → `reduce<>` |

## Verification

Coverage was checked before running anything, per the Step 2 lesson:

| Suite | Why it covers this change |
|---|---|
| `tests/ttnn/unit_tests/operations/fused/test_softmax.py::test_softmax` | the general small-W/small-H path from 7b; ragged `h=24`, `w=42` with poisoned padding |
| `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_softmax.py` | `test_softmax_non_tile_aligned` (ragged, from Step 2) + the aligned shapes, which now take the new single-tile branch |
| `test_moreh_sum.py`, `test_moreh_mean.py` | runtime-arg removal; `319 % 32 = 31` still drives the partial path |
| `tests/ttnn/nightly/unit_tests/operations/experimental/test_topk_router_gpt.py` | 7d |
| `tests/ttnn/unit_tests/operations/toy_reduce_partial/` | helper canary (no `kernel_lib` change here, so expected green) |

### Test results

| Suite | Result |
|---|---|
| `tests/ttnn/unit_tests/operations/fused/test_softmax.py::test_softmax` | **108 passed** in 33.67s |
| `test_moreh_softmax.py` | **115 passed, 32 skipped** (incl. Step 8's 4 new cases) |
| `test_moreh_logsoftmax.py` | **100 passed, 32 skipped** |
| `test_moreh_softmin.py` | **92 passed, 32 skipped** |
| `test_moreh_sum.py` | **229 passed, 155 skipped** |
| `test_moreh_mean.py` | **76 passed, 72 skipped** |
| `test_topk_router_gpt.py` | **12 passed** |
| `toy_reduce_partial/` | **36 passed** |

### The 7b hang, confirmed by accident

While setting up the perf A/B below, the "pre-fix" emulation was applied to the moreh factories but
**not** to the two general ones — i.e. exactly the state Step 2 left the tree in: a shared reader
emitting two scaler tiles into a one-tile CB. The bench deadlocked, ran past its 40-minute timeout, and
left the card wedged (`Timed out waiting for ETH heartbeat ... Stuck at 0xaabb1146`); it needed
`tt-smi -r 0` to recover. So the hang in 7b is not an inference — it reproduces, it is a hard hang, and
it takes a board reset to clear.

### Perf: not resolvable in this environment

The 7a fix removes one scaler-tile fill per launch on tile-aligned shapes and constant-folds the
partial-scaler path away entirely. An A/B was run in a single build (conditional emission vs. an
always-emit-the-pair emulation of the pre-fix behaviour, patched consistently across all four
factories):

| case | always-pair | conditional | delta |
|---|---:|---:|---:|
| `softmax_small_w.aligned_512` | 141.41 | 152.25 | +7.66% |
| `softmax_small_w.ragged_511` | 155.16 | 141.37 | −8.89% |
| `softmax_small_h.aligned_512` | 203.70 | 200.39 | −1.62% |
| `ttnn_softmax_general_w_small.aligned_512` | 113.90 | 114.07 | +0.16% |

**These numbers are noise, and they say so themselves:** the change only *removes* work in the aligned
case and does nothing at all in the ragged case, yet aligned reads +7.7% and ragged −8.9%. The
run-to-run spread here is ±8%, against Step 6's reported 0.1–0.5%. Two likely causes: this build has the
Tracy profiler enabled (`build_metal.sh` default; Step 6's numbers are also **not** comparable in
absolute terms for the same reason — everything here is 10–20% slower), and the host has 8 cores, so
host dispatch is plausibly the bottleneck rather than device time.

**Conclusion: the `softmax_small_w` regression is not confirmed fixed by measurement.** The mechanism is
sound and the correctness suites are green, but anyone wanting the number should re-measure with
`./build_metal.sh --disable-profiler` on both this branch and `main`, in the same configuration.
