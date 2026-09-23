# `ttsl::move_only_function` implementation evaluation (#57444)

Standalone harness comparing **zoo** and **fu2** against a **`std::function`** baseline, to pick the
backing implementation for `ttsl::move_only_function`.

It is deliberately outside the tt-metal build: it needs only google-benchmark and the two candidate
headers, so it configures in seconds and can be built with each supported compiler in turn.
Building tt-metal itself twice to cover gcc and clang is not practical.

## Contenders

All three are pinned to **whatever inline capacity `std::function` actually has in the current
configuration**, so no side gets a bigger buffer than the baseline.

That capacity belongs to the **standard library, not the compiler** — clang with libstdc++ behaves
exactly like gcc. Measured by `sbo_probe`:

| Configuration | Buffer | `sizeof(std::function<void()>)` | Buffer on LP64 |
| --- | --- | --- | --- |
| libstdc++ (g++-12, clang++-20 default) | 2 pointers | 32 B | **16 B** |
| libc++ (clang++-20 `-stdlib=libc++`) | 3 pointers | 48 B | **24 B** |

Both libraries size the buffer in **pointers**, so the byte figures are target dependent — 16 and 24
on LP64, 8 and 12 on a 32-bit target. Tenstorrent ships only 64-bit hosts, so that is all this
harness is built and validated on; `inline_capacity.hpp` just keeps the pointer-relative form rather
than baking in an unstated assumption.

It cannot be computed from `sizeof` either: libstdc++ has 2 pointers of overhead and libc++ has 3,
so `sizeof - 2 * sizeof(void*)` is right for the first and overstates the second by 8 bytes. The
constants are therefore measured, and `sbo_probe` **exits non-zero** if a standard library, or a
target, ever moves the boundary — a silent mismatch would skew every number in the table.

| Name | Type |
| --- | --- |
| `StdFn` | `std::function<void()>` |
| `ZooFn` | `zoo::VTableFunction<kInlinePointers, void()>` — move-only policy (`Destroy, Move, CallableViaVTable`); capacity is in **pointers** |
| `Fu2Fn` | `fu2::function_base<true, false, fu2::capacity_fixed<kInlineBytes>, true, false, void()>` — `unique_function` with the capacity pinned rather than defaulted |

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

Three configurations, because the inline buffer differs by standard library:

```bash
cmake -S . -B out-gcc    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-12
cmake -S . -B out-clang  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-20
cmake -S . -B out-libcxx -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=clang++-20 \
                         -DCMAKE_CXX_FLAGS=-stdlib=libc++
for d in out-gcc out-clang out-libcxx; do cmake --build $d -j; done

# run this first: it fails if the pinned capacity no longer matches std::function
for d in out-gcc out-clang out-libcxx; do ./$d/sbo_probe || echo "MISMATCH in $d"; done

./out-clang/bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
./out-gcc/bench   --benchmark_repetitions=5 --benchmark_report_aggregates_only=true

# item 6
size out-clang/CMakeFiles/codegen_{std,zoo,fu2}.dir/codegen_tu.cpp.o

# item 7
for c in std zoo fu2; do
  /usr/bin/time -f "$c %e" cmake --build out-clang --target compile_time_$c --clean-first >/dev/null
done
```

## Results

Pending — to be filled in and posted to #57444.

## Packaging note

`fu2` publishes release tags (latest **4.2.5**). `zoo` publishes **no tags at all**, only branches,
so it can only be pinned by commit SHA (`7fb5eed...`). That is an adoption input alongside the
performance numbers, not just a packaging detail.
