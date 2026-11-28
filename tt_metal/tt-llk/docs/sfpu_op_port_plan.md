# Porting an SFPU op to the new API

## 0. Rules for every port

1. **No perf regression** on anything the legacy API supported. The new code must be identical or strictly better at the instruction level, and hardware perf must confirm it.
2. **Supported behaviour is preserved.** Everything the legacy API supported must give the same results and touch the same Dest rows:
   The port may add behaviour legacy didn't support or fail clearly where legacy silently did the wrong thing. Whatever's added must be reachable only through combinations legacy didn't support, tested against its own golden, and listed under "New behaviour".
3. **The legacy compute API keeps working unchanged.** ttnn builds many calls as host-side strings (`unary_op_utils.cpp`, `binary_ng_utils.cpp`).
4. **Removing Old implementations**: The `calculate_*` implementations can and should be deleted if they are unused after the refactor. If they are still used the in the implementation of another Compute API, if it makes sense, that compute API should also be ported to the new Operation implementation. If another implementation depends on the implementation, raise it as an issue.
5. **Port first, optimize later.** The port only changes structure; every scheduling change gets its own commit and measurement.
6. **Don't touch core abstractions** Changes go into the op class, the compute API and `llk::sfpu::Operation`. Don't modify the core abstractions (`shim.h`, `eltwise.h`, `traverse.h` etc,).
7. **Minimize number of files that Compute implementation spans** Ideally, the implementation should be Compute API in one file, calling Operation in other.

## 1. Classify the op

| Kind | How to recognise it | How to port | Example |
|---|---|---|---|
| **A: plain SFPI loop** | `for (d < ITERATIONS) { v = dst_reg[..]; …; dst_reg[..] = r; dst_reg++; }` | Rewrite on `eltwise_*`; the loop body becomes the functor | Mask, BinopScalar |
| **B: hand-scheduled per face** | `TTI_*`, `SFPLOADMACRO`, replay, `SFPCONFIG` macro setup | Wrap the existing per-face function unchanged in `walk_faces<shape>` | Typecast |
| **C: whole tile or custom walk** | Runs once per tile (`VectorMode::None`, `ITERATIONS == 32`) | Whole-tile entry, no face walk | exp, 32 iterations |

Classify each template combination that's used; one op can be several kinds.

**Inventory before writing code:**

- **Call sites:** direct, templated, and host strings (`rg '"op_tile'`). These fix which signatures stay. Include related entry points in the same family (`rsub_tile`, `rsub_unary_int32_tile`).
- **Combinations in use:** approx, fp32 Dest, flags, iterations, `VectorMode`, formats.
- **Dest layout per format:** Wormhole's default int32 layout is sign-magnitude, Blackhole's is two's complement. Note what legacy reads and writes (`.mode<DataLayout::I32>()`).
- **State set by init:** constants, LOADMACRO and replay setup, config register, counters.
- **Loop unrolling:** the face loop's unroll (`_llk_math_eltwise_sfpu_apply_vector_mode_`: `#pragma GCC unroll 0`) and the CKernel's per-vector `#pragma GCC unroll`, for each combination in use.
- **Shared per-vector helpers:** these stay and become functor bodies.
- **Wormhole/Blackhole differences.** Quasar branches stay legacy.
- **Tests:** the functional test, `perf_*.py`, and the harness call site.

## 2. Target design

### Files

| What | Where |
|---|---|
| Op class (`llk::sfpu::<Op>`) | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/sfpu/macro/<op>.h`, one per architecture, identical except for a contained architecture difference (e.g. `sfpu/macro/mask.h`, `sfpu/macro/binop_scalar.h`) |
| Compute API | the op's existing header under `tt_metal/hw/inc/api/compute/`, which includes `sfpu/macro/<op>.h` and calls `Op::init` / `Op::execute` |
| Base class | `llk::sfpu::Operation` in `tt_metal/tt-llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/sfpu/operation.h` |
| Micro-operations | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/sfpu/micro/<name>.h` |

Small parts of an implementation that other ops reuse (e.g. bf16 RNE rounding) go into `sfpu/micro/` as micro-operations. A micro-operation is usually a function over a single Dest vector (`vFloat` / `vInt` in, same out) that the op class calls from its functor. It has no init, no face walk and no Dest indexing of its own. Keep anything used by one op inside that op's class.

### Op class

```cpp
class Op : public llk::sfpu::Operation {
private:

    template <::DataFormat format>
    using vector_type = std::conditional_t<is_integral(format), sfpi::vInt, sfpi::vFloat>;

    static constexpr sfpi::DataLayout layout(::DataFormat format) {
        return is_integral(format) ? sfpi::DataLayout::I32 : sfpi::DataLayout::Default;
    }

    template <Mode>
    static constexpr bool always_false = false;

public:
    static void init() {
        SAN_HOOK(unsupported());
        reset_sfpu_config();
        // + op-specific state only (constants, LOADMACRO/replay setup)
    }

    template <Mode mode, ::DataFormat format, ckernel::DstSync sync, ckernel::TensorShape shape,
              PaddingAgnostic padding = PaddingAgnostic::Yes>
    [[gnu::always_inline]] static void execute(std::uint32_t output, std::uint32_t input…) {
        // static_asserts for unsupported combinations; Operation's shape and Dest index checks
        eltwise_unary<shape, padding>(output, input,
            [captures](const DestRegister& out, const DestRegister& in) __attribute__((always_inline)) {
                const vType x = in.mode<layout(format)>();
                vType result;
                if constexpr (mode == …) { … }
                else { static_assert(always_false<mode>, "Op: unhandled Mode"); }
                out.mode<layout(format)>() = result;
            });
    }

    static void uninit() {}
};
```

| | |
|---|---|
| Derive from `llk::sfpu::Operation` and use its checks (`assert_dest_index_valid`, `assert_dest_range_valid`, `assert_distinct`, `assert_shape_walkable`, `assert_walk_agnostic`) | one set of checks, compiled out in perf builds |
| `init()` calls `reset_sfpu_config()`, plus only what the op itself needs | ops such as TopK leave the SFPU control register changed; generic inits are documented to clear it |
| **No counter reset in `init()`** | `calculation_end` resets the Dest counter at the end of every execute |
| **Force inlining with `[[gnu::always_inline]]`.** Use `__attribute__((always_inline))` only on lambdas, where C++17 has no place for the attribute; the compute API keeps `ALWI` | `execute` was compiled out of line without it; the compiler rejects the bracket form on a lambda |
| Functor parameters are `const DestRegister&`, not `const auto&` | `in.mode<…>()` needs no `.template` |
| Compile-time helpers are `static constexpr` functions (`layout(format)`); type helpers are aliases (`vector_type<format>`) | |
| The Dest layout helper is named `layout`: `I32` for Int32, default otherwise | Int32 stays two's complement on Wormhole; Mask keeps the mask itself in the default layout, as legacy compared it |
| One functor for all modes and formats; an exhaustive `if constexpr` chain ending in `static_assert(always_false<mode>)` | a missing mode fails to compile |
| Unsupported combinations fail with a `static_assert` message | |
| `shape` has no default; the compute API always passes it. `padding` is the last template parameter of both `execute` and the compute API, defaults to the op's setting, and is forwarded unchanged. Modes and formats are template parameters; tile indices are runtime arguments | the kernel author can override padding at the call site |
| Constants and scalars captured once per tile; per-architecture constants in a struct that's empty when unused | Wormhole RSUB rounding constants |
| Sanitizer: `SAN_HOOK(unsupported())`, no model | |
| **Loop unrolling matches legacy.** Set `EltwiseOptions` (or `walk_faces`' `UnrollFaces`) so the face and vector loops unroll as the original did. Change unrolling only as a separate optimization commit (§5) | unrolling changes code size and instruction count; the defaults (face 0, vector 8) match the common legacy kernel but not every one |
| Out of place; in place is the same index | |

### Code style

- **One algorithm, generalized over the data.** When the algorithm is fundamentally the same for every format, write it once over `vector_type<format>` (`vFloat`/`vInt`) and `layout(format)` (the Dest `DataLayout`). Don't specialize it per format with `if`/`switch` branches or separate functors that repeat the same steps.
- **Fork only when the algorithm differs.** A different sequence of operations, a hand-scheduled kernel, or a different walk gets its own path. Different value types, layouts or operators on the same steps do not.
- **One `eltwise` call per algorithm.** Modes that share the algorithm differ only in the operation inside the functor, selected by an exhaustive `if constexpr` chain that ends in `static_assert(always_false<mode>)`.
- **Keep architecture differences small and contained.** Put a Wormhole/Blackhole difference in one helper (e.g. the rounding-constant struct, empty where unused), not in a second copy of the algorithm.
- **Compile-time choices are helpers, not branches in the body.** Type helpers are aliases (`vector_type<format>`); value helpers are `static constexpr` functions (`layout(format)`, `scalar_value<format>(scalar)`). Use the same names in every op class.
- **Typed functor parameters.** `const DestRegister&`, so loads and stores read `in.mode<layout(format)>()` with no `.template`.
- **Work done once per tile is captured, not repeated per row:** scalars, constants, rounding masks.
- **No indirection layers.** The compute API calls `Op::execute` directly; don't add per-op wrapper helpers between them.

```cpp
// Preferred: one functor, the format picks the type and the layout.
eltwise_unary<shape, padding>(output, input, [value](const DestRegister& out, const DestRegister& in) __attribute__((always_inline)) {
    const vType x = in.mode<layout(format)>();
    vType result;
    if constexpr (mode == Mode::Add) { result = x + value; }
    else if constexpr (mode == Mode::Sub) { result = x - value; }
    else { static_assert(always_false<mode>, "Op: unhandled Mode"); }
    out.mode<layout(format)>() = result;
});

// Avoid: the same algorithm duplicated per format.
if constexpr (is_integral(format)) {
    eltwise_unary<shape, padding>(output, input, [s](const auto& out, const auto& in) { /* int add/sub */ });
} else {
    eltwise_unary<shape, padding>(output, input, [s](vFloat x) { /* float add/sub */ });
}
```

### Compute API

- **New entry point, required shape.** Every new compute API function takes at least a `TensorShape` and a `PaddingAgnostic`, and passes both through to `Op::execute`:

  ```cpp
  template <
      TensorShape shape,                                                      // required, no default
      /* op template parameters: mode…, format = … */
      llk::sfpu::PaddingAgnostic padding = llk::sfpu::PaddingAgnostic::Yes>  // last; the op's default
  ALWI void op_tile(uint32_t output, uint32_t input…) {
      MATH((llk::sfpu::Op::execute<…, DST_SYNC_MODE, shape, padding>(output, input…)));
  }
  ALWI void op_tile_init() { MATH((llk::sfpu::Op::init())); }
  ```

  - `shape` comes first, has no default, and is passed by value, so a kernel can write it inline: `op_tile<make_tensor_shape(16, 16, 1, 1), …>`.
  - `padding` comes last and defaults to the op's own setting: `Yes` for lane-wise ops, `No` for padding-sensitive ones. A kernel overrides it explicitly.
  - Output and input are both Dest indices, so the call works out of place.
  - Template parameter names and order are the same in the compute API and in `Op::execute`. Legacy entry points keep the default padding.
- **Legacy entry points call `Op::execute` directly.** Runtime `VectorMode` goes through `detail::with_legacy_vector_mode_shape(vector_mode, [&](auto shape) __attribute__((always_inline)) { Op::execute<…, decltype(shape)::value>(…); })` in `api/compute/legacy_vector_mode.h` (RC, R, C → their faces; any other mode → face 0). Fixed-RC entry points pass `DEFAULT_TENSOR_SHAPE`.
- **Quasar keeps its legacy bodies** under `#ifdef ARCH_QUASAR`.
- **The op header is included on every TRISC** when the signature names its types. Fix any include those headers were missing.
- **`[[deprecated]]` only after ttnn has moved off the legacy entry.**

### Comments

- Keep them very light: only constraints, invariants, or why a choice was made. Follow the tt-llk `code-comments` skill.
- Doxygen describes the contract only: operands, valid values, the init/execute pairing. No registers, instructions or algorithm.

## 3. Commits

1. `<Op> as an SFPU op class, <op>_tile through it`: the class, the new and legacy compute API, and include fixes.
2. `<Op> tests through the op class`: the harness and tests drive the class. Perf numbers count from here.
3. Optional `<Op>: <optimization>`, one idea per commit.
4. `[WH]` / `[BH] Remove the legacy <op> kernel`.

Changes to `Operation`, or to an earlier op class, amend the commit that introduced them.

## 4. Verification

| Gate | Hardware machine | Simulator only |
|---|---|---|
| 0: compile checks | yes | yes |
| 1: correctness | yes | partly (no `SFPLOADMACRO`) |
| 2: tt-llk perf | yes | no: hand off or CI |
| 3: system perf | yes | no: hand off or CI |

### Gate 0 (any machine with the tt-llk test environment)

- **LLK-level code comparison:** legacy macro against `Op::execute` for every legacy combination and `VectorMode` (including `None` + 32 iterations where the harness uses it), both architectures, asserts off. Count Tensix instructions executed per tile:
  - replays expanded
  - loop trip counts from the loop's counter and step
  - instructions the RISC-V core builds at runtime included
  - a tail call, an out-of-line copy, or no matching cases is a failure

  | Pass criterion | |
  |---|---|
  | kinds B, C | Tensix stream identical |
  | kind A | per tile, total instructions ≤, `sfpnop` ≤, constant loads ≤ |
  | `init` | instructions ≤ legacy |

- **Compute-API-level code comparison:** each legacy entry point against its original `SFPU_UNARY_CALL` expansion, same criteria.
- **Compute API on every TRISC:** unpack, math and pack, fp32 on and off, both architectures, using the JIT include list plus the descriptor arrays from `fake_jit_prelude.h`. The math object has no out-of-line lambda or `execute`.
- **Rejection tests:** one per `static_assert`, each with its own message.
- **Sanitizer build:** `-DLLK_SAN_ENABLE -DENABLE_LLK_ASSERT` compiles.
- **Pre-commit:** clang-format and codespell.

### Gate 1

- After commit 2, run the functional tests. Hardware: `pytest -q -n 8 test_<op>.py`. Simulator: `TT_METAL_DISABLE_SFPLOADMACRO=1 pytest -q --run-simulator …`.
- Compile perf-only variants: `CHIP_ARCH=<arch> pytest --compile-producer`.
- Sweep legacy combinations against the existing golden, and new ones against their own.
- Run chained tests when init touches replay, macro or constant state.
- Note any combination without a functional test.

### Gate 2 (hardware)

`.claude/scripts/perf_compare_commits.sh <arch> perf_<op> --baseline <before series> --current <commit>`, 2% / 30 cycles. Any regression on a legacy combination blocks. Report INIT and TILE_LOOP separately.

### Gate 3 (hardware)

ttnn device kernel duration for every op using the API, including fused users, then model perf CI and the ttnn unit tests.

### PR evidence

The code-comparison table, pass counts per architecture, perf reports, ops measured, "New behaviour", and where each gate ran.

## 5. Optimization (after the port)

| Lever | When | Example |
|---|---|---|
| hoist constants into the capture | rebuilt per row or per face | Blackhole RSUB rounding constants |
| init deduplication | repeats the common setup | `exp_init` plus `SFPU_UNARY_INIT_FN` |
| change `unroll_faces` / `unroll_vectors` from legacy | measured gain at acceptable code size | |

## 6. The five ops

| Op | Kind | Status |
|---|---|---|
| `binop_with_scalar_tile_init`, `*_unary_tile`, `rsub_tile`, `*_unary_tile_int32`, `rsub_unary_int32_tile` | A | **ported**, example branch `llk/sfpu/binop-scalar` (`5ad1cf383dc`, `fa89ab7b8a3`). Gates 0 and 1 done; Gates 2 and 3 pending; int32 and R/C/None have no functional test. Quasar keeps the legacy bodies. `add/sub_unary_tile_uint32` (new on main) still call the legacy kernel, which blocks its removal (rule 4) until they are ported too |
| `mul_binary_tile` | A | next: `SfpuBinary<Mul>`; Wormhole fp32 must match the two-row path |
| `recip_tile` | Wormhole A, Blackhole B | Wormhole on `eltwise`; Blackhole wraps the `SFPLOADMACRO` kernels |
| `exp_tile` | B (+ A, C) | wrap everything; last |

## 7. Context: SFPU refactor proposal

The parts of `docs/docs/modules/ROOT/pages/sfpu/refactor.adoc` a port relies on.

### Old and new call chain

| | |
|---|---|
| Old | Compute API → `SFPU_UNARY_CALL` macro → `_llk_math_eltwise_*_sfpu_params_` → per-face CKernel |
| New | Compute API → op class (`Op::init` / `Op::execute` / `Op::uninit`) |

The legacy CKernel ran once per face, inside the `_params_` wrapper's face loop. `Op::execute` runs once per call and owns the walk over the region, the calculation start and end, the asserts and the sanitizer hook. Each ported op drops its `SfpuType` entry.

### Operation helpers

`llk::sfpu::Operation` (`sfpu/operation.h`), all `protected static`:

| Helper | Kind | Use |
|---|---|---|
| `reset_sfpu_config()` | init | Reset the SFPU control register to its default |
| `assert_dest_index_valid<Sync>(index)` | `LLK_ASSERT` | `index` is inside the Dest acquired for `Sync` |
| `assert_dest_range_valid<Sync>(first, tiles)` | `LLK_ASSERT` | `tiles` tiles from `first` are inside Dest |
| `assert_distinct(a, b)` | `LLK_ASSERT` | Two Dest tiles differ, for ops that can't alias operands |
| `assert_shape_walkable<Shape>()` | `static_assert` | Faces of 1..16 x 1..16 in a grid of 1..2 x 1..2 |
| `assert_walk_agnostic<Shape, Agnostic>()` | `static_assert` | A padding-sensitive op doesn't walk a shape that touches padding |

### TensorShape

`TensorShape {face_r_dim, face_c_dim, num_faces_r_dim, num_faces_c_dim}` replaces the runtime `VectorMode` and the per-face `ITERATIONS`. The walk derives the vectors per face from `face_r_dim`. Op classes and the compute API take it as a by-value template parameter.

| Legacy `VectorMode` | Shape (faces high x wide x face height x width) |
|---|---|
| `RC` | 2x2x16x16 |
| `R` | 1x2x16x16 |
| `C` | 2x1x16x16 |
| anything else | `LEGACY_SINGLE_CALL_SHAPE`, 1x1x16x16 (`api/compute/legacy_vector_mode.h`) |

Each op states the shapes it supports with `static_assert`. An op that walks Dest itself, as `VectorMode::None` kernels did, simply doesn't call a walk helper.

<a id="padding-agnostic"></a>
### PaddingAgnostic

```cpp
/// Whether a call may compute on Dest lanes outside the region it is given.
enum class PaddingAgnostic { Yes, No };
```

The walk touches padding when `face_r_dim` isn't a multiple of 4 (the last 4-row band), or when `face_c_dim < 16`, since every vector interleaves the face's columns.

- `Yes`: the op is lane-wise; any walkable shape works.
- `No`: computing on padding would be wrong (e.g. a reduction); `assert_walk_agnostic` rejects such shapes at compile time.

It's set per op, as the default of the last template parameter of `execute` and the compute API (§2). A kernel can override it at the call site after making sure the partial face meets the op's constraints. The eltwise helpers take it with no default.

### Eltwise API

`sfpu/eltwise.h`, in `llk::sfpu`:

```cpp
template <TensorShape Shape, PaddingAgnostic Agnostic, EltwiseOptions::Struct Options = EltwiseOptions().create(), typename Functor>
void eltwise_unary(std::uint32_t output, std::uint32_t input, Functor&& functor);
// eltwise_binary(output, left, right, functor), eltwise_ternary(output, first, second, third, functor),
// and the N-ary eltwise<Shape, Agnostic, Options>(output, functor, inputs...) they forward to.
```

The functor takes one of two forms:

- **Direct store:** `(const DestRegister& out, const DestRegister& in…)` loads and stores itself, for example with `in.mode<layout(format)>()`. The op classes use this form (§2).
- **Deferred store:** `(in…) -> value` returns the result, and eltwise stores it.

`EltwiseOptions().unroll_faces(f).unroll_vectors(v).create()` is passed by value:

| Option | Default | Controls |
|---|---|---|
| `unroll_faces(f)` | `0` | `#pragma GCC unroll f` on the outer loop over the shape's faces. `0` keeps it rolled, which is what legacy's face loop does; `f` copies the face body `f` times. |
| `unroll_vectors(v)` | `8` | `#pragma GCC unroll v` on the inner loop over the vectors of one face, the legacy CKernel's `ITERATIONS` loop. `8` fully unrolls a 16-row face; `0` keeps it rolled. |

Eltwise handles the calculation start and end, pointing Dest at the output tile, the walk over the shape's vectors, Dest advancing on the output store (`ADDR_MOD_6`, no `dst_reg++`), and in-place and out-of-place calls.

- **Load timing.** A concrete parameter type such as `vFloat` loads every input when the functor is called. `DestRegister` or `auto` loads where the value is first used.
- **Dest addressing.** `dst_reg[0]` is the output tile, not Dest tile 0. Legacy code that addressed operands with `index * dst_tile_size` offsets has to be rewritten.

### Walk APIs

For kind B (§1), `sfpu/traverse.h`:

```cpp
template <const TensorShape& Shape, std::size_t UnrollFaces = 0, typename FaceFunctor>
void walk_faces(std::uint32_t output_index, FaceFunctor&& functor);  // functor once per face
template <std::size_t Vectors> void step_vectors();
template <std::size_t Faces> void step_faces();
// constexpr: face_count, vector_count, is_shape_walkable, is_walk_agnostic
```

`walk_faces` owns the calculation scope and moves Dest to the next face after each call. The functor may leave Dest anywhere inside the face, but must not touch the carriage-return counter. A per-face CKernel ports unchanged as `walk_faces<shape>(dst, [] { calculate_x<…>(); })`.

`eltwise` walks with less overhead, because it never leaves Dest mid-face.

`walk_faces` takes `Shape` by reference, so a by-value `shape` is forwarded as-is.
