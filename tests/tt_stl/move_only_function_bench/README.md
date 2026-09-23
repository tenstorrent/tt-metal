# `ttsl::move_only_function` implementation evaluation (#57444)

Standalone harness comparing **zoo** and **fu2** against a **`std::function`** baseline, to pick the
backing implementation for `ttsl::move_only_function`.

It is deliberately outside the tt-metal build: it needs only google-benchmark and the two candidate
headers, so it configures in seconds and can be built with each supported compiler in turn.
Building tt-metal itself twice to cover gcc and clang is not practical.

## Contenders

All three are pinned to whatever inline capacity `std::function` has in the current configuration,
so no side gets a bigger buffer than the baseline. That capacity varies by standard library; see
`inline_capacity.hpp`, and run `sbo_probe` to check it.

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
