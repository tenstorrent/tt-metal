# Quasar Matmul Functional And Perf Sweep Strategy

## Goal

Use one shared source of truth for Quasar matmul test combinations while keeping functional coverage broad and perf compile time manageable.

The functional test can afford a wider runtime sweep because dimensions are marked with `runtime(...)` and therefore do not normally create one ELF per dimension. Perf is commonly run with `--speed-of-light`; in that mode runtime parameters are promoted to compile-time constants, so every swept value can become another build.

## Current Functional Axes

`tests/python_tests/quasar/test_matmul_quasar.py` currently sweeps:

- formats from `MATMUL_FORMAT`
- math fidelity: `LoFi` only for `Int8`, otherwise `LoFi`, `HiFi2`, `HiFi3`, `HiFi4`
- destination accumulation and sync: `Int8` uses `DestAccumulation.Yes`; other formats use both accumulation modes and both sync modes
- dimensions where output tile count satisfies `M * N <= max_tiles`
- implied math format: MX inputs use `Yes`; non-MX inputs use both `No` and `Yes`
- `MxFp4_2x_A` and `MxFp4_2x_B` register-format hints for `MxFp4` input on Quasar
- direct indexing only when a 2x register-format hint is present
- `Transpose.No`

This is appropriate for functional coverage. It exercises smaller destination occupancies, both implied and explicit non-MX math-format configuration, both destination sync modes, and the Quasar-only MxFp4 2x paths.

## Recommended Shared Helpers

Create shared helper functions for the sweep policy rather than duplicating the comprehensions in functional and perf tests.

Suggested shape:

```python
def matmul_dimensions(dest_acc, dest_sync, *, exact_dest_fill=False, kt_dims=kt_dims):
    max_tiles = DEST_SYNC_TILE_LIMITS[dest_sync] // (
        2 if dest_acc == DestAccumulation.Yes else 1
    )
    return [
        ([mt_dim * TILE_DIM, kt_dim * TILE_DIM], [kt_dim * TILE_DIM, nt_dim * TILE_DIM])
        for mt_dim in range(1, max_tiles + 1)
        if not exact_dest_fill or max_tiles % mt_dim == 0
        for nt_dim in (
            [max_tiles // mt_dim]
            if exact_dest_fill
            else range(1, max_tiles // mt_dim + 1)
        )
        for kt_dim in kt_dims
    ]
```

Use `exact_dest_fill=False` for functional and `exact_dest_fill=True` for perf.

For implied math format:

```python
def matmul_implied_math_formats(format, *, perf=False):
    if perf:
        return [ImpliedMathFormat.Yes]
    if format.input_format.is_mx_format():
        return [ImpliedMathFormat.Yes]
    return [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
```

This keeps the policy explicit while letting both tests call the same helper.

## Assessment Of Proposed Perf Policy

### 1. Keep All Formats

This gives the strongest broad performance matrix, but it is also the largest compile-time driver under `--speed-of-light`. If this perf test is intended for a nightly or manual full sweep, keeping all formats is reasonable. If it is intended for a regular CI perf signal, consider a smaller representative format set.

Recommended split:

- full perf sweep: all formats, manual or nightly
- smoke perf sweep: representative formats only, for quick regression signal

Useful representatives include:

- `Float16 -> Float16`
- `Float16_b -> Float16_b`
- one or two mixed float output conversions
- `MxFp8R`, `MxFp8P`
- `MxFp4` with both `MxFp4_2x_A` and `MxFp4_2x_B`
- one MX integer format
- `Int8 -> Int32`

### 2. Keep All Math Fidelities

This makes sense for math performance, especially for matmul. The compile cost is large because fidelity is a template parameter, but reducing it would hide real math-path differences.

If compile time becomes too high, prefer keeping all fidelities in a focused `MATH_ISOLATE` perf sweep and using fewer fidelities for broad L1-to-L1 coverage.

### 3. Keep All Dest Accumulation

Keeping all destination accumulation modes is reasonable. `dest_acc` changes destination capacity and accumulator mode, so it can affect both coverage and performance.

The dimension helper should derive `max_tiles` from the selected destination accumulation mode so perf fills the actual destination capacity for each case.

### 4. Perf Uses `SyncHalf`

For perf, use only `DestSync.Half` / `DestSync::SyncHalf`. `SyncFull` is not relevant for the perf signal you want and would multiply the speed-of-light build matrix without adding useful coverage. Functional should keep both sync modes for correctness coverage.

The dimension helper should still take destination sync as an input because functional uses both modes and because `max_tiles` is derived from the selected sync mode.

### 5. Perf Dimensions Use `M * N == max_tiles`

This is the best first reduction. It keeps boundary coverage for destination capacity and removes low-occupancy cases that multiply compile count in speed-of-light mode.

Exact-fill output shapes are:

- `max_tiles = 4`: `(1,4)`, `(2,2)`, `(4,1)`
- `max_tiles = 8`: `(1,8)`, `(2,4)`, `(4,2)`, `(8,1)`
- `max_tiles = 16`: `(1,16)`, `(2,8)`, `(4,4)`, `(8,2)`, `(16,1)`

Each shape still expands over selected K tile counts.

Consider whether perf needs larger K values than functional. Functional currently uses `kt_dims = [1, 2, 4]`. Perf may benefit from a long-K case such as `8` or `32` to measure steady-state math throughput. Keep this policy-driven, for example `functional_kt_dims = [1, 2, 4]` and `perf_kt_dims = [1, 4, 8, 32]`.

### 6. Perf Uses `ImpliedMathFormat.Yes`

This is a good reduction. For non-MX float formats, `ImpliedMathFormat.No` versus `Yes` mainly changes ALU format setup. The steady-state matmul instruction path is the same. Functional should keep both for non-MX correctness coverage; perf can use `Yes` to avoid doubling the non-MX build matrix.

MX formats should continue to use `Yes`.

### 7. Keep Register Format Hint

Keep this for MxFp4 performance coverage. The 2x register-format hints select Quasar-specific source register formats and can affect the matmul path.

### 8. Keep Enable Direct Indexing

Keeping both direct-indexing values is useful for MxFp4 2x performance because it selects a different matmul instruction variant. It does multiply the MxFp4 perf matrix. If compile time becomes too high, measure both direct-indexing modes in a focused MxFp4-only perf test and use one default mode in the broad sweep.

### 9. Keep Transpose

`Transpose.No` is already the only current Quasar matmul case. Keep it as a shared axis for consistency, but it does not add compile pressure while it has one value.

## Speed-Of-Light Source Requirement

If a perf test reuses the Quasar functional C++ source, make the source compatible with `--speed-of-light`. In SOL mode the generated `RuntimeParams` struct is empty, while runtime parameters become top-level `constexpr` values in `params.h`.

Use the pattern already present in existing perf kernels:

```cpp
#ifndef SPEED_OF_LIGHT
const std::uint32_t CT_DIM = params.CT_DIM;
const std::uint32_t RT_DIM = params.RT_DIM;
const std::uint32_t KT_DIM = params.KT_DIM;
const Operand& buffer_A = params.buffer_A;
const Operand& buffer_B = params.buffer_B;
const Operand& buffer_Res = params.buffer_Res;
#endif
```

Then use `CT_DIM`, `RT_DIM`, `KT_DIM`, `buffer_A`, `buffer_B`, and `buffer_Res` directly in the kernel body. In non-SOL they are locals loaded from `params`; in SOL they resolve to generated compile-time values.

## Suggested Final Policy

For functional:

- keep the current broad format, fidelity, destination, implied-format, register-hint, direct-indexing, and dimension policies
- use shared helpers with `exact_dest_fill=False`
- keep dimensions wrapped in `runtime(...)`

For full perf:

- use the same format, fidelity, destination accumulation, register-hint, direct-indexing, and transpose axes
- use only `DestSync.Half`
- use shared helpers with `exact_dest_fill=True`
- use `ImpliedMathFormat.Yes`
- consider a perf-specific K set that includes one long-K case
- run as manual or nightly because all remaining axes are still template-heavy under SOL

For smoke perf:

- use the same helper functions, but pass a representative format subset
- keep exact destination fill and `ImpliedMathFormat.Yes`
- keep both direct-indexing modes only for a small MxFp4-focused subset

This gives one implementation for combination generation while making the coverage policy explicit per test type.

## Sweep Counts

These counts assume the current `MATMUL_FORMAT` definition in `test_matmul_quasar.py`:

- `8 * 8 = 64` input/output combinations from `input_output_formats([...])`
- `1` explicit `Int8 -> Int32` combination
- `65` total format combinations

The format groups used below are:

- regular non-MX inputs: `Float16` and `Float16_b` inputs across 8 outputs, so `2 * 8 = 16`
- MX inputs excluding `MxFp4`: `MxFp8R`, `MxFp8P`, `MxInt8`, `MxInt4`, `MxInt2` across 8 outputs, so `5 * 8 = 40`
- `MxFp4` input across 8 outputs, so `8`
- `Int8 -> Int32`, so `1`

Dimension counts with `kt_dims = [1, 2, 4]`:

| Policy | Dest mode | `max_tiles` | `(M, N)` shapes | K values | Dimension count |
| --- | --- | ---: | ---: | ---: | ---: |
| Functional, `M * N <= max_tiles` | `SyncHalf`, `DestAccumulation.Yes` | 4 | 8 | 3 | 24 |
| Functional, `M * N <= max_tiles` | `SyncHalf`, `DestAccumulation.No` | 8 | 20 | 3 | 60 |
| Functional, `M * N <= max_tiles` | `SyncFull`, `DestAccumulation.Yes` | 8 | 20 | 3 | 60 |
| Functional, `M * N <= max_tiles` | `SyncFull`, `DestAccumulation.No` | 16 | 50 | 3 | 150 |
| Perf, `M * N == max_tiles` | `SyncHalf`, `DestAccumulation.Yes` | 4 | 3 | 3 | 9 |
| Perf, `M * N == max_tiles` | `SyncHalf`, `DestAccumulation.No` | 8 | 4 | 3 | 12 |

For non-`Int8` functional cases, the dimension count across all destination modes is:

```text
24 + 60 + 60 + 150 = 294
```

For non-`Int8` perf cases, using only `SyncHalf`, the dimension count across all destination accumulation modes is:

```text
9 + 12 = 21
```

For `Int8 -> Int32`, the current functional policy only uses `DestAccumulation.Yes`, so the functional dimension count is:

```text
24 + 60 = 84
```

The perf policy uses `SyncHalf` only, so the `Int8 -> Int32` perf dimension count is:

```text
9
```

Parameterized test counts:

| Group | Format count | Functional count | Perf count |
| --- | ---: | ---: | ---: |
| Regular non-MX inputs | 16 | `16 * 4 fidelities * 294 dims * 2 implied modes = 37,632` | `16 * 4 fidelities * 21 dims = 1,344` |
| MX inputs excluding `MxFp4` | 40 | `40 * 4 fidelities * 294 dims = 47,040` | `40 * 4 fidelities * 21 dims = 3,360` |
| `MxFp4` input | 8 | `8 * 4 fidelities * 294 dims * 4 hint/direct-indexing modes = 37,632` | `8 * 4 fidelities * 21 dims * 4 hint/direct-indexing modes = 2,688` |
| `Int8 -> Int32` | 1 | `1 * 1 fidelity * 84 dims * 2 implied modes = 168` | `1 * 1 fidelity * 9 dims = 9` |
| Total | 65 | `122,472` | `7,401` |

The functional number is the pytest parameterized case count before any skips. It is not the same as compile count, because dimensions are runtime parameters and are excluded from the compile key in normal non-SOL functional runs.

For perf, `PerfConfig` also adds `PERF_RUN_TYPE` as a template parameter for every selected run type. If the perf test uses the common five run types:

- `L1_TO_L1`
- `UNPACK_ISOLATE`
- `MATH_ISOLATE`
- `PACK_ISOLATE`
- `L1_CONGESTION`

then the perf run-type variants are:

```text
7,401 parameterized perf cases * 5 run types = 37,005 perf run-type variants
```

If a perf-specific K set is changed, scale the dimension-dependent counts by:

```text
len(perf_kt_dims) / 3
```
