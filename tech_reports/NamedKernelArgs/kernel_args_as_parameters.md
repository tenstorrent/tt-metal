# Kernel Arguments as Function & Template Parameters

## Args in Metal 2.0 today

Metal 2.0 gives kernel arguments **names**:

- **Host registers** them by kind — compile-time, per-core runtime, common runtime.
- **JIT emits** a per-kernel `kernel_args_generated.h` declaring an `args::<name>` accessor for
  every registered argument (e.g. `args::start_tile_id`), typed by its kind.
- **Kernel reads** them via a single overloaded `get_arg(args::<name>)`, covering all three
  kinds: **CTA** (compile-time, `constexpr`), **RTA** (per-core runtime), **CRTA** (common runtime).

What's left is boilerplate: the kernel still hand-writes `kernel_main()` and one `get_arg` call
per argument, by name.

## Proposal

Remove the hand-written part. The user writes a plain typed function whose **parameters are the
arguments**; the `kernel_main()` that fetches them is generated.

## C++ syntax carries the argument kinds

We let ordinary C++ express the one distinction that matters — compile-time vs runtime:

- **template parameters → CTAs** — true constant expressions, usable in `if constexpr`, array
  bounds, etc.
- **function parameters → runtime args** — RTA vs CRTA is decided entirely by the host schema;
  the kernel body can't tell and doesn't need to.

```cpp
// NEW: the whole kernel.
template <uint32_t Ht, uint32_t Wt, uint32_t untilize>   // CTAs (compile-time)
TT_KERNEL void my_kernel(uint32_t start_tile_id, uint32_t num_tiles, uint32_t start_row,  // RTAs
                         uint32_t scaler) {                                                // CRTA
    // Ht/Wt/untilize are compile-time constants; start_tile_id/... are runtime values.
}
```

```cpp
// LEGACY: the same kernel, with every argument fetched positionally by hand.
void kernel_main() {
    constexpr uint32_t Ht       = get_compile_time_arg_val(0);
    constexpr uint32_t Wt       = get_compile_time_arg_val(1);
    constexpr uint32_t untilize = get_compile_time_arg_val(2);
    uint32_t start_tile_id = get_arg_val<uint32_t>(0);
    uint32_t num_tiles     = get_arg_val<uint32_t>(1);
    uint32_t start_row     = get_arg_val<uint32_t>(2);
    uint32_t scaler        = get_common_arg_val<uint32_t>(0);
}
```

## Implementation

1. **Tag.** The entry is marked `TT_KERNEL`, defined in `experimental/kernel_args.h` (so the
   kernel never defines it). In Phase 1 it expands to `FORCE_INLINE`, so the user entry folds
   into the generated `kernel_main()` with no call indirection.
2. **Parse.** Before JIT compile, `genfiles` scans the source for the lone `TT_KERNEL` token and
   extracts the entry's **name**, its **template parameter names** (CTAs), and its **function
   parameter names** (RTAs/CRTAs).
3. **Shim.** From that signature, `genfiles` generates `kernel_main()` reusing the Metal 2.0
   `get_arg` infrastructure — each name becomes `get_arg(args::<name>)`, CTAs in the angle
   brackets (constexpr), runtime args in the parentheses. It is emitted after the `args::`
   header and the user source, so every symbol is already in scope:

   ```cpp
   void kernel_main() {
       my_kernel<get_arg(args::Ht), get_arg(args::Wt), get_arg(args::untilize)>(
           get_arg(args::start_tile_id), get_arg(args::num_tiles), get_arg(args::start_row),
           get_arg(args::scaler));
   }
   ```

**On manual parsing.** Phase 1 uses a small hand-rolled tokenizer (strip
comments/strings/preprocessor → find the marker → match the `template<…>` and `(…)` lists →
split on top-level commas → take each parameter's trailing identifier as its name). It is a
name extractor, not a full C++ parser — which is why parameters are restricted to **`uint32_t`**
this phase. It is robust and cheap: parsing a synthetic **8.49 MB** source (thousands of
functions, comment/string/preprocessor noise, and decoy `TT_KERNEL` mentions, with one real
entry buried in the middle) takes **~37 ms (~227 MB/s)** — negligible beside a kernel's JIT
compile, and real kernels are a few KB.

## Next phase

Widen the type surface behind a whitelist — most notably **`uint64_t`** and **`std::array`** —
by extending the `get_arg`/accessor layer to multi-word reads. Arbitrary type spellings break
the tokenizer's "trailing identifier is the name" heuristic, so depending on the complexity we
move parsing onto a real C++ frontend — **libclang / Clang AST tooling** (e.g.
`clang -ast-dump=json` or LibTooling) — kept behind the same parse interface so the shim
generator is unchanged.
