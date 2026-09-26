# `ttsl::move_only_function` implementation evaluation (#57444)

Standalone harness comparing **zoo** and **fu2** against a **`std::function`** baseline, to pick the
backing implementation for `ttsl::move_only_function`.

It is deliberately outside the tt-metal build: it needs only google-benchmark and the two candidate
headers, so it configures in seconds and can be built with each supported compiler in turn.
Building tt-metal itself twice to cover gcc and clang is not practical.

## Contenders

All three are pinned to whatever inline capacity `std::function` has in the current configuration,
so no side gets a bigger buffer than the baseline. That capacity varies by standard library; see
`candidates.hpp`, and run `sbo_probe` to check it.

| Name | Type |
| --- | --- |
| `StdFn` | `std::function<void()>` |
| `ZooFn` | `zoo::Function<zoo::AnyContainer<zoo::Policy<void*[N], Destroy, Move, RTTI>>, void()>` — capacity is in **pointers** |
| `Fu2Fn` | `fu2::function_base<true, false, fu2::capacity_fixed<kInlineBytes>, true, false, void()>` — `unique_function` with the capacity pinned rather than defaulted |

### Why this zoo spelling

`zoo::VTableFunction` is the more obvious alias, but `AnyContainer` provides no `operator bool` and
no `has_value()`, so it cannot stand in for `std::move_only_function`. Both live on `zoo::Function`.

The `RTTI` affordance is what makes `operator bool` trustworthy: without it the fallback compares
the vtable's destroy pointer against `Destroy::noOp`, and for a trivially destructible target — a
captureless lambda, a function pointer — those can be merged by identical code folding (MSVC
`/OPT:ICF`, `lld`/`gold --icf=all`), so an engaged function reports itself empty. With the
affordance, `operator bool` routes through `type()`, which cannot collapse.

It costs nothing per object: the affordance adds one pointer to the per-type static vtable, not to
the instance. It does emit `typeid` for every erased callable, which is binary size. It does not
cost `-fno-rtti` compatibility, because zoo does not compile under `-fno-rtti` regardless — the
`typeid` tokens in `VTablePolicy.h` and `FunctionPolicy.h` are rejected at parse time by both gcc
and clang even with a plain `Destroy, Move` policy.

## What each scenario measures

| Benchmark | Issue item | Measures |
| --- | --- | --- |
| `BM_ConstructInvokeDestroy<*, SmallCapture>` | 1 | Full lifecycle, capture fits inline (16 B) |
| `BM_ConstructInvokeDestroy<*, BoundaryCapture>` | 1/2 | 24 B capture: inline under libc++, heap under libstdc++ |
| `BM_ConstructInvokeDestroy<*, LargeCapture>` | 2 | Same, capture forced to the heap (64 B) |
| `BM_MoveOnlyCapture_Std` / `BM_MoveOnlyCapture<*>` | 3 | `std::unique_ptr` capture. `std::function` cannot hold one, so its baseline uses the `shared_ptr` wrapper the codebase uses today — the very workaround this issue exists to remove |
| `BM_MoveConstruct<*, *>`, `BM_MoveAssign<*, *>` | 4 | Move construction and move assignment, both capture sizes |
| `BM_QueueThroughput<*, SmallCapture>/1024` | 5 | Enqueue 1024 then drain, mirroring `ThreadPool::enqueue(std::function<void()>&&)` |
| `codegen_tu.cpp` | 6 | One object file per candidate, same TU, for `.o` size comparison |
| `compile_time_tu.cpp` | 7 | 300 instantiations across 3 signatures per candidate, for compile-time cost |

Every benchmark reports **`allocs/iter`** from a counting global `operator new`. That is what proves
a "small" case really stayed inline and a "large" case really hit the heap, instead of inferring it
from capture size. A candidate reporting ~0 ns would mean the compiler elided the work — a harness
bug, not a result.

## Running

Everything is driven by `reproduce.py`, which configures and builds each configuration, runs all
seven items, and prints the tables below as markdown:

```bash
./reproduce.py                  # everything, 5 repetitions
./reproduce.py --repetitions 3  # faster, noisier
./reproduce.py --only gcc       # a single configuration
./reproduce.py --skip-build     # reuse existing build directories
```

It aborts if `sbo_probe` reports that the pinned inline capacity no longer matches
`std::function`, since every number below depends on that holding.

Needs `g++-12`, `clang++-20`, libc++ headers, `cmake` and `size` on PATH. `--only` narrows the set
if you have fewer. Raw google-benchmark output is not checked in; re-run the script for it.

## Results

Median ns over 5 repetitions; coefficient of variation was 0.2-2.5% throughout, so the gaps below
are real. Full output in `results/`. Posted to #57444.

**The motivating case — `BM_MoveOnlyCapture`.** `std::function` cannot hold a `unique_ptr` capture,
so its baseline is the `shared_ptr` wrapper the codebase uses today:

| config | std (shared_ptr) | zoo | fu2 |
| --- | --- | --- | --- |
| gcc-12 / libstdc++ | 42.69 (3 allocs) | **13.43** | 14.45 |
| clang-20 / libstdc++ | 43.29 | **14.00** | 15.27 |
| clang-20 / libc++ | 45.64 | **12.72** | 20.11 |

About 3x faster, 3 allocations down to 1. This is the premise of the issue, confirmed.

**Queue throughput, 1024 jobs** — the `ThreadPool::enqueue` shape. zoo wins every configuration:

| config | std | zoo | fu2 |
| --- | --- | --- | --- |
| gcc-12 / libstdc++ | 5519 | **5074** | 6766 |
| clang-20 / libstdc++ | 6633 | **5610** | 6363 |
| clang-20 / libc++ | 7068 | **5697** | 14875 |

**Construct/invoke/destroy, small inline capture** — 2.3-3.2 ns for all three, except fu2 under
libc++ at 12.58 ns. See the fu2 note below.

**Compile time**, 300 instantiations across 3 signatures, best of 3:

| config | std | zoo | fu2 |
| --- | --- | --- | --- |
| gcc-12 / libstdc++ | 3.67s | 3.58s (0.98x) | **11.21s (3.05x)** |
| clang-20 / libstdc++ | 2.79s | 2.53s (0.91x) | **6.23s (2.24x)** |
| clang-20 / libc++ | 7.46s | **2.88s (0.39x)** | 7.85s (1.05x) |

**Object size**, same TU per candidate, `.text` bytes: zoo smallest everywhere (3111 / 2114 / 2108),
fu2 largest (8520 / 3823 / 7953), std in between (5229 / 3256 / 3633).

### fu2 is ~5x slower under libc++, and it is structural

fu2 calls `std::align` on every construction (`function2.hpp:493`):

```cpp
return type(std::align(alignof(T), sizeof(T), inplace, from_capacity));
```

libstdc++ defines `std::align` inline in `bits/align.h`, so the optimiser constant-folds it away —
a minimal construct/invoke/destroy compiles to 2 instructions with no calls. libc++ declares it
`_LIBCPP_EXPORTED_FROM_ABI` (`__memory/align.h:21`), an out-of-line symbol in the shared library.
The compiler cannot see the body, so it emits a real call; the same function becomes 52 instructions
with 8 calls, including `std::__1::align`, `_Unwind_Resume` and two `__clang_call_terminate`.

Ruled out as explanations: capacity (fu2 is ~2.5 ns at capacity 16/24/32 under libstdc++ and ~12 ns
at all three under libc++) and allocation (`allocs/iter=0` throughout).

It is not tunable — no fu2 template parameter avoids that call — it hits every construction, which
is the enqueue path, and libc++ is a supported tt-metal toolchain. zoo does its own alignment
arithmetic and is unaffected.

### Caveats

- `std::move_only_function` is unavailable in both toolchains, so there is no direct measurement
  against the eventual replacement.
- `BoundaryCapture` (24 B) is inline for all three under libc++, so those rows show inline cost
  rather than a heap path.
- The 2-instruction libstdc++ figure above is from a minimal TU that permits more elision than the
  benchmark, which holds the work live with `DoNotOptimize`. It explains the mechanism; the 2.3 vs
  12.6 ns from the matrix is the number to quote.

## Non-performance findings

Two differences the timings will not surface, both to weigh alongside them.

### `-fno-rtti`

**zoo does not compile under `-fno-rtti`; fu2 does.** This is not about the `RTTI` affordance —
zoo fails with a plain `Destroy, Move` policy too, because the `typeid` tokens in `VTablePolicy.h`
(lines 123, 148) and `FunctionPolicy.h` (line 190) are rejected at parse time. Verified on both
g++-12 and clang++-20; fu2 builds clean under `-fno-rtti` on both.

Nothing in tt-metal that would consume the alias is affected today. The only `-fno-rtti` in the
tree is device-side — `tt_metal/jit_build/build.cpp:263` for JIT-compiled kernels, plus LLK test
tooling — and kernels do not include `tt_stl/`. `ThreadPool`, the first planned consumer, is
host-side, and the host build sets no such flag.

The residual risk is downstream. `tt_stl` is a public API library (`FILE_SET api TYPE HEADERS`), so
`ttsl/move_only_function.hpp` ships to consumers; anyone building with `-fno-rtti` who includes it
would get a parse error inside a vendored header. That is bounded — it only affects translation
units that include it — but it is a constraint fu2 does not impose.

Caveat: this comes from grepping for the usual spellings of the flag, so a compiler wrapper or an
externally supplied `CMAKE_CXX_FLAGS` could still introduce it, and it says nothing about how
consumers outside this repo build.

### `noexcept` move, and what it costs

`std::move_only_function`'s move constructor is unconditionally `noexcept`, and `std::function`'s is
too, so the alias must not regress that. fu2 controls it with `HasStrongExceptGuarantee`, and its
own `unique_function` default is `false` — which would make our alias the only contender with a
throwing move. It is set to **`true`** here.

Measured:

| Type | `noexcept` move | Accepts a throwing-move capture |
| --- | --- | --- |
| `std::function` (baseline) | yes | yes |
| `std::move_only_function` (target) | yes, mandated | yes — heap-allocates them |
| zoo `Function` + RTTI | yes | yes |
| fu2, `HasStrongExceptGuarantee=false` | **no** | yes |
| fu2, `HasStrongExceptGuarantee=true` | yes | **no**, `static_assert` |

`true` costs a compile-time rejection of callables whose move can throw. That is rare in practice —
`unique_ptr`, `shared_ptr`, `string` and `vector` all move `noexcept` — and it fails loudly at the
call site rather than silently. The alternative loses a guarantee that call sites will bake in and
that the eventual `std::move_only_function` provides, so `false` would leave behind defensive code
that migration does not automatically unwind.

**A caveat on how zoo gets both columns.** `Move::VTableEntry` is declared
`void (*mp)(void*, void*) noexcept` while zoo still accepts throwing-move targets, so a target whose
move actually throws terminates. `std::move_only_function` earns the same guarantee honestly by
heap-allocating such callables. fu2 with `true` is the honest form of the promise — it refuses the
target rather than accepting it under a `noexcept` it cannot keep. Worth weighing: zoo's behaviour
here is a latent `std::terminate`, not a compile error.

### Empty-call behaviour

All contenders are configured to **throw**, matching `std::function`. Note this is one place the
alias deliberately does not match its eventual replacement: calling an empty `std::move_only_function`
is UB, whereas these throw.

fu2 can abort instead, via its `IsThrowing` template parameter. zoo cannot — `Executor::DefaultExecutor`
is a `constexpr static`, not a policy affordance. Keeping both throwing is what makes the comparison
symmetric; it also keeps behaviour identical to the `std::function` call sites being migrated, so
anything catching `bad_function_call` keeps working.

**Neither choice touches the hot path.** Both libraries put the empty case in the erased slot rather
than branching per call: zoo's `operator()` is an unconditional `executor_(args..., this)`, where an
empty object simply holds a throwing `DefaultExecutor`, and fu2's vtable holds `empty_invoker::invoke`.
There is no `if (empty)` in a normal invocation, so there is no call-time cost to buy back.

Nor is there much size in it. Measured on the same TU with fu2 at `IsThrowing` true vs false:

| Compiler | `.o` | `.text` | `.eh_frame` + `.gcc_except_table` |
| --- | --- | --- | --- |
| g++-12 | 99,656 → 99,352 (−304) | 12,298 → 12,261 | 2,660 → **2,660** |
| clang++-20 | 54,096 → 53,752 (−344) | 5,659 → 5,618 | 1,928 → **1,928** |

The exception-handling sections are byte-identical, because `throw` appears once per signature in
`empty_invoker`, not once per erased callable. Dropping it removes that single thunk and the
`bad_function_call` RTTI, ~0.3-0.6% of the object. So abort-vs-throw is a behaviour decision, not a
size optimisation.

### Release tagging

`fu2` publishes release tags (latest **4.2.5**). `zoo` publishes **no tags at all**, only branches,
so it can only be pinned by commit SHA (`7fb5eed...`).
