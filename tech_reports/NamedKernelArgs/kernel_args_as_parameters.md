# Kernel Arguments as Function and Template Parameters

**Status:** Design proposal
**Scope:** Device-side kernel authoring ergonomics for runtime/common/compile-time args in tt-metal
**Builds on:** Metal 2.0 named-argument infrastructure (`experimental/kernel_args.h`, per-kernel `kernel_args_generated.h`)

## 1. Motivation

Today a kernel reads its arguments positionally, tracking L1 indices by hand:

```cpp
void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t dst_addr  = get_arg_val<uint32_t>(2);
    // ...
}
```

Indices are manual, meaningless, and easy to desync from the host's `SetRuntimeArgs`
call. Insert an argument in the middle and every downstream index shifts silently.

Metal 2.0 already removes the raw indices by giving arguments **names**. The host
registers ordered arg names; the JIT emits a per-kernel `kernel_args_generated.h`
with one accessor constant per name, and the kernel reads by name:

```cpp
void kernel_main() {
    auto src_addr  = get_arg(args::src_addr);
    auto num_tiles = get_arg(args::num_tiles);
    auto dst_addr  = get_arg(args::dst_addr);
}
```

This proposal takes the **last step**: let the user write the kernel as an ordinary
typed function whose parameters *are* the arguments. The user never writes
`get_arg`, never writes `kernel_main`, never sees L1 at all.

```cpp
template <uint32_t z>                                     // CTAs  (compile-time)
TT_KERNEL void my_awesome_kernel(uint32_t x, uint32_t y) { // RTAs / CRTAs (runtime)
    // Look ma, real kernel arguments!
    if constexpr (z) {
        // x + y, etc.
    }
}
```

Two parameter lists, two axes:

- **Template parameters are compile-time args (CTAs).** They are genuine constant
  expressions, so they can drive `if constexpr`, array bounds, other template arguments —
  anything that must be known at compile time. This is the one distinction the *source*
  has to express, because it changes what the compiler may specialize away.
- **Function parameters are runtime args.** Whether each is delivered per-core (RTA) or
  shared across cores (CRTA) is a pure **host-schema** choice with **zero** source impact —
  `get_arg` returns the value either way (Section 2). The kernel body cannot tell, and
  shouldn't have to.

On the host, the user supplies values by name — a name→value map (or zipped key/value
vectors) for runtime args, and the named compile-time args for the CTAs:

```cpp
auto k = CreateKernel(program, "my_awesome_kernel.cpp", core,
    ComputeConfig{
        .named_compile_args = {{"z", 1}},                 // CTA  → template arg
        .runtime_arg_names  = {"x", "y"},                 // RTA/CRTA → fn params
    });

SetRuntimeArgs(program, k, core, {
    {"x", 5},
    {"y", 7},
});
```

## 2. Key insight: one accessor, two binding sites

The Metal 2.0 accessor layer (`tt_metal/hw/inc/experimental/kernel_args.h`) defines
three accessor kinds and one overloaded `get_arg`:

```cpp
template <typename T> struct RtaArg  { uint32_t byte_offset; };  // per-core runtime arg
template <typename T> struct CrtaArg { uint32_t byte_offset; };  // common runtime arg
template <typename T> struct CtaVal  { T value; };               // compile-time constant

template <typename T> FORCE_INLINE T get_arg(RtaArg<T> a);            // reads per-core L1
template <typename T> FORCE_INLINE T get_arg(CrtaArg<T> a);           // reads common L1
template <typename T> FORCE_INLINE constexpr T get_arg(CtaVal<T> a);  // returns the constant
```

Two properties of this layer make the whole feature fall out:

**(a) RTA and CRTA are indistinguishable to the caller.** Both `get_arg(RtaArg<T>)` and
`get_arg(CrtaArg<T>)` return `T` (today `uint32_t`). Which one is selected depends only on
the *type* of the generated `args::<name>` constant, which is decided by the **host
schema**, not the kernel source. So a runtime parameter `x` is fetched as
`get_arg(args::x)` whether the host registered `x` as per-core or common — the call site
is identical. Promoting `x` from RTA to CRTA is a one-line host change with **zero** kernel
edits. This is why function parameters don't need to encode RTA-vs-CRTA.

**(b) The CTA accessor is `constexpr`.** `get_arg(CtaVal<T>)` is a constant expression, so
`get_arg(args::z)` can be used as a **template argument**. That is what lets compile-time
args bind to *template* parameters:

```cpp
my_awesome_kernel<get_arg(args::z)>(get_arg(args::x), get_arg(args::y));
//               └ CTA: constexpr, template arg ┘ └ RTA/CRTA: runtime fn args ┘
```

The single generated call site above is the entire device-side mechanism. Compile-time
args go in the angle brackets (constant expressions), runtime args go in the parentheses,
and the RTA/CRTA choice for the parenthesized ones is invisible. We are not inventing a
new fetch path — we generate a thin `kernel_main()` that instantiates and calls the user
function through accessors that already exist.

## 3. Architecture

No changes to firmware or to the host L1 write path. The firmware
(`tt_metal/hw/firmware/src/tt-1xx/brisck.cc:80`, and the ncrisc/trisc equivalents) keeps
calling a bare `kernel_main()`. We add exactly two things:

1. **Host:** a name→value runtime-arg API that reconciles against the registered schema
   (CTAs already flow through the existing `named_compile_args` path).
2. **JIT codegen:** generate a `kernel_main()` shim that fetches each arg by name,
   instantiates the user template with the CTAs, and calls it with the runtime args.

```
 user kernel.cpp                     host program
 ┌────────────────────────────┐      ┌─────────────────────────────────────┐
 │ template <uint32_t z>       │      │ CreateKernel(...,                    │
 │ TT_KERNEL void my_kernel(   │      │   .named_compile_args = {{"z",1}},   │
 │     uint32_t x,             │      │   .runtime_arg_names  = {"x","y"})   │
 │     uint32_t y) {…}         │      │ SetRuntimeArgs(prog,k,core,          │
 └──────────────┬─────────────┘      │   {{"x",5},{"y",7}})                 │
                │ parsed at JIT       └───────────────┬─────────────────────┘
                │ (template + fn params)              │ ordered names + values
                ▼                                      ▼  (existing dispatch path)
        ┌─────────────────────────────────────────────────────────┐
        │ genfiles.cpp                                              │
        │  • write_kernel_args_generated_header()  (exists)         │
        │      namespace args {                                     │
        │        CtaVal<uint32_t> z{1};                             │
        │        RtaArg<uint32_t> x{0}; RtaArg<uint32_t> y{4}; }    │
        │  • write_kernel_main_shim()              (NEW)            │
        │      void kernel_main() {                                 │
        │        my_kernel<get_arg(args::z)>(    // CTA → tmpl arg   │
        │            get_arg(args::x),           // RTA/CRTA → fn    │
        │            get_arg(args::y));                             │
        │      }                                                    │
        └───────────────────────────┬─────────────────────────────┘
                                     ▼
                  kernel_includes.hpp  =  generated headers
                                       +  user source
                                       +  generated kernel_main() shim
                                     ▼
                       brisck.cc calls kernel_main()  (unchanged)
```

## 4. Device-side authoring contract

### 4.1 The marker

The entry function carries a `[[tt::kernel_main]]` attribute, surfaced to users as the
`TT_KERNEL` macro — both compiler-benign and a reliable textual anchor for the parser:

```cpp
// in a public kernel header
#define TT_KERNEL [[tt::kernel_main]]
```

Spelled at the use site as `TT_KERNEL void my_awesome_kernel(...)`, it expands to
`[[tt::kernel_main]] void my_awesome_kernel(...)`. `[[tt::kernel_main]]` is a
vendor-namespaced attribute; the riscv toolchain ignores unknown attributes (we build
kernels with `-Wno-attributes`, or register it). It carries no semantics for the compiler —
it exists so codegen can locate the entry unambiguously without heuristics about "the one
free function."

Exactly one marked entry per kernel translation unit. Zero → fall back to the classic
hand-written `void kernel_main()` (full backward compatibility). Two → JIT error.

### 4.2 Two parameter lists; both bind by name

The entry has up to two parameter lists, mapping to the two axes from Section 2:

| Source position | Arg kind | How it reaches the kernel |
| --- | --- | --- |
| **template** non-type params (`template <uint32_t z>`) | **CTA** (compile-time) | template argument: `my_kernel<get_arg(args::z)>(…)` (constexpr) |
| **function** params (`(uint32_t x, uint32_t y)`) | **RTA or CRTA** (runtime; host decides) | function argument: `…(get_arg(args::x), get_arg(args::y))` |

Both lists **bind by name**: each parameter binds to the host arg whose name equals the
parameter name. Order within a list is irrelevant to correctness; the shim emits
`get_arg(args::<paramname>)` per parameter in declared order, and `args::<paramname>` is
keyed by name. Consequences:

- Reordering parameters within a list is safe.
- A parameter whose name was never registered on the host produces a compile error
  (`args::<name>` does not exist) — caught at JIT time, with a clear generated-file
  location. A host-side pre-check (Section 5.3) gives a friendlier message.
- A template param's name should be a registered **CTA** (`named_compile_args`); a function
  param's name should be a registered **runtime** arg (`runtime_arg_names` /
  `common_runtime_arg_names`). One crossing is caught for free by the type system: putting
  a **runtime** arg in a template slot fails to compile, because `get_arg(args::<rta>)` is
  not `constexpr` and can't be a template argument. The reverse — a **CTA** name used as a
  function param — would silently compile (the constant is just passed at runtime), so the
  host-side check in Section 5.3 flags it instead.
- Parameter *types* are `uint32_t` today (Section 8). RTA-vs-CRTA is never spelled in the
  source — only compile-time-vs-runtime is (template vs function list).

### 4.3 What the user does *not* write

No `get_arg`, no `args::`, no `kernel_main`, no L1 addresses, no indices. The
`kernel_args_generated.h` and the shim are generated and live only in the JIT build dir.

## 5. Host-side API

### 5.1 Declaring the schema (exists, lightly extended)

All three kinds are declared at `CreateKernel` time on the existing kernel spec
(`tt_metal/api/tt-metalium/kernel_types.hpp`):

- **CTAs:** `named_compile_args` (`unordered_map<string,uint32_t>`) — already drives the
  `CtaVal` constants in `kernel_args_generated.h`. These are the template parameters.
- **Runtime args:** ordered `runtime_arg_names` / `common_runtime_arg_names`. These are the
  function parameters; the name's presence in one list vs. the other is what makes it an
  RTA or a CRTA — invisibly to the kernel.

No change required to *declare* names; the only new host surface is *setting runtime
values by name* (5.2).

### 5.2 Setting values by name (new overload)

Add overloads alongside the existing positional `SetRuntimeArgs`
(`tt_metal/api/tt-metalium/host_api.hpp`):

```cpp
// name → value; reconciled against the kernel's registered runtime_arg_names.
void SetRuntimeArgs(
    const Program&, KernelHandle,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>&,
    const std::unordered_map<std::string, uint32_t>& named_args);

// zipped key/value vectors (the user's other requested form)
void SetRuntimeArgs(
    const Program&, KernelHandle,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>&,
    stl::Span<const std::string_view> names,
    stl::Span<const uint32_t> values);

void SetCommonRuntimeArgs(
    const Program&, KernelHandle,
    const std::unordered_map<std::string, uint32_t>& named_args);
```

Implementation reorders the map into the registered name order and then funnels into the
**existing** positional `set_runtime_args` path — i.e. the L1 layout, offsets
(`dispatch.cpp`), and `WriteRuntimeArgsToDevice` are entirely unchanged. This is pure
host-side name→index resolution.

### 5.3 Validation

At `SetRuntimeArgs` and again at program finalize:

- every registered name has a value (or a documented default policy);
- no unknown names supplied;
- (optional, strongest) every entry parameter name is a registered name **of the matching
  kind** — template params ↔ `named_compile_args`, function params ↔ runtime-arg names.
  This catches the one crossing the type system misses (a CTA name used as a function
  param, Section 4.2). It requires the parsed parameter lists to be available on the host;
  see Section 6.4 on threading parse results back.

## 6. JIT codegen changes

All changes are in `tt_metal/jit_build/genfiles.cpp`, adjacent to the existing
`write_kernel_args_generated_header()` (`genfiles.cpp:200`) and
`jit_build_genfiles_kernel_include()` (`genfiles.cpp:282`).

### 6.1 Parse the entry signature

We need, from the user source: the **function name**, the optional **template
non-type-parameter list** (CTAs), and the **function-parameter list** (runtime args) — each
as an ordered `(type, name)` list, anchored on the marker.

Anchor handling: the marker (`TT_KERNEL` → `[[tt::kernel_main]]`) sits between the optional
`template < … >` clause and the return type. So the parser scans **backward** from the
marker for an optional `template < … >` clause, and **forward** for `ret-type name
( params )` up to the matching `{`. Each list splits on top-level commas.

Two implementation options:

- **(A) Lightweight tokenizer (recommended for v1).** Kernels are small and the anchor is
  exact. Handles the realistic surface — `template <uint32_t z, uint32_t w>`, plain
  `uint32_t name` function params, trailing commas, comments, line breaks. No new build
  dependency. Reject, with a clear error, anything it cannot parse unambiguously (type
  template params, defaulted params, macros in the signature), pushing those authors to the
  classic `kernel_main()`.
- **(B) libclang parse (hardening / v2).** Robust against arbitrary C++. Adds a host-side
  libclang dependency to the JIT path and a parse pass over the (preprocessed) source.
  Worth it once we support non-`uint32_t` types and want exact type info.

Recommendation: ship (A), keep the parser behind a single function
(`parse_kernel_main_signature(source) -> {name, template_params, fn_params}`) so (B) can
replace it without touching the codegen.

### 6.2 Emit the shim

New `write_kernel_main_shim(out_dir, settings, parsed_sig)` emits, into a generated
header included **after** the user source:

```cpp
// AUTO-GENERATED — do not edit.
void kernel_main() {
    my_awesome_kernel<
        get_arg(args::z)        // one per template param (CTA), in declared order
    >(
        get_arg(args::x),       // one per function param (RTA/CRTA), in declared order
        get_arg(args::y));
}
```

`get_arg(args::<name>)` for every parameter — CTAs inside the `<…>`, runtime args inside
the `(…)`. If there are no template params, the `<…>` is omitted entirely (plain call).
Because the `args::*` constants already exist (from `write_kernel_args_generated_header`)
and `get_arg` is overloaded on their type — `constexpr` for the CTA template args, a runtime
L1 read for the function args — this is the whole shim.

### 6.3 Ordering in `kernel_includes.hpp`

Current `jit_build_genfiles_kernel_include()` builds:

```
kernel_bindings_generated.h
kernel_args_generated.h
<user source>
```

New order:

```
kernel_bindings_generated.h
kernel_args_generated.h
<user source>            // declares my_awesome_kernel + the [[tt::kernel_main]] marker
kernel_main_shim.h       // NEW: defines kernel_main() that calls my_awesome_kernel(...)
```

The shim must follow the user source so the entry function is already declared. The TRISC
path (`jit_build_genfiles_triscs_src`, `genfiles.cpp:305`) gets the same shim include in
each `chlkc_*.cpp` wrapper.

### 6.4 Threading parse results to the host (for 5.3)

The parser runs at genfiles time (host process). To enable the strongest host-side
validation, store the parsed parameter lists (template + function, with kinds) on the
kernel object when the source is first read, so `SetRuntimeArgs` / finalize can cross-check
name *and* kind. If we keep the parser purely in codegen for v1, we rely on the JIT compile
error from a missing `args::<name>` (and the constexpr-in-template-slot error) instead.

## 7. Backward compatibility

- Kernels with no `[[tt::kernel_main]]` and a hand-written `void kernel_main()` are
  untouched — no shim generated.
- Metal 1.0 indexed `get_arg_val` kernels are untouched.
- Metal 2.0 `get_arg(args::x)` kernels with a hand-written `kernel_main()` are untouched.
- The feature is purely additive codegen gated on the marker.

## 8. Limitations and roadmap

- **Types.** v1 is `uint32_t`-only, inherited from the `sizeof(T)==4` static_assert in
  `kernel_args.h`. The shim mechanism is type-agnostic; widening is a `get_arg`/accessor
  change (multi-word reads), not a shim change. Tracked as a follow-up. Template params are
  likewise non-type `uint32_t` (and trivially `bool`) only.
- **RTA-vs-CRTA stays host-side.** By design the function-param list never spells per-core
  vs. common; the host schema decides (Section 2), so promotion between them needs no
  kernel edit. The compile-time-vs-runtime split *is* spelled in the source (template vs
  function param), because moving an arg across it changes what the compiler can
  specialize — so that promotion is intentionally a source edit.
- **Parser surface.** Tokenizer rejects exotic signatures (type template params, defaults,
  macros in the signature); libclang (option B) lifts that limit.
- **Varargs.** Out of scope; the existing `get_vararg()` helpers
  (`genfiles.cpp:272`) remain available inside the body.

## 9. Worked examples (target end state)

### 9.1 Minimal — 1 CTA, 2 RTAs

Device — `reader.cpp` (`cb_id` is a CTA so the CB index is a compile-time constant;
`src_addr`/`num_tiles` are runtime):

```cpp
#include "dataflow_api.h"   // pulls get_arg + the TT_KERNEL marker macro

template <uint32_t cb_id>                          // CTA
TT_KERNEL void reader(uint32_t src_addr, uint32_t num_tiles) {   // runtime args
    const InterleavedAddrGenFast<true> s{.bank_base_address = src_addr, /* … */};
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_id, 1);                 // cb_id is a constant here
        noc_async_read_tile(i, s, get_write_ptr(cb_id));
        noc_async_read_barrier();
        cb_push_back(cb_id, 1);
    }
}
```

Host:

```cpp
auto k = CreateKernel(program, "reader.cpp", core,
    DataMovementConfig{
        .named_compile_args = {{"cb_id", tt::CBIndex::c_0}},   // CTA → template arg
        .runtime_arg_names  = {"src_addr", "num_tiles"},       // RTA/CRTA → fn params
    });

SetRuntimeArgs(program, k, core, {
    {"src_addr",  src_buffer.address()},
    {"num_tiles", 64},
});
```

Generated (in the JIT build dir, never seen by the user):

```cpp
// kernel_args_generated.h
namespace args {
  constexpr experimental::CtaVal<uint32_t> cb_id{0u};        // tt::CBIndex::c_0
  constexpr experimental::RtaArg<uint32_t> src_addr{0};
  constexpr experimental::RtaArg<uint32_t> num_tiles{4};
}
// kernel_main_shim.h
void kernel_main() {
    reader<get_arg(args::cb_id)>(get_arg(args::src_addr), get_arg(args::num_tiles));
}
```

### 9.2 All three kinds — 3 CTAs, 3 RTAs, 2 CRTAs

Device — `my_kernel.cpp` (what the user writes):

```cpp
#include "dataflow_api.h"   // pulls get_arg + the TT_KERNEL marker macro

template <uint32_t block_h, uint32_t block_w, uint32_t untilize>   // 3 CTAs
TT_KERNEL void my_kernel(
    uint32_t src_addr,        //  RTA  ┐
    uint32_t dst_addr,        //  RTA  │ 3 per-core runtime args
    uint32_t num_tiles,       //  RTA  ┘
    uint32_t scaler,          //  CRTA ┐ 2 common runtime args
    uint32_t sem_addr)        //  CRTA ┘
{
    // CTAs are constants — legal in compile-time contexts:
    constexpr uint32_t tiles_per_block = block_h * block_w;
    if constexpr (untilize) { /* specialized path compiled in or out */ }

    // RTAs and CRTAs are just runtime uint32_ts; the body cannot tell which is which:
    for (uint32_t i = 0; i < num_tiles; ++i) {
        /* use src_addr, dst_addr, scaler, sem_addr ... */
    }
}
```

Host (what sets the values):

```cpp
auto k = CreateKernel(program, "my_kernel.cpp", core,
    ComputeConfig{
        .named_compile_args       = {{"block_h", 4}, {"block_w", 2}, {"untilize", 1}}, // 3 CTAs
        .runtime_arg_names        = {"src_addr", "dst_addr", "num_tiles"},             // 3 RTAs
        .common_runtime_arg_names = {"scaler", "sem_addr"},                            // 2 CRTAs
    });

SetRuntimeArgs(program, k, core, {        // per-core values
    {"src_addr",  src.address()},
    {"dst_addr",  dst.address()},
    {"num_tiles", 64},
});

SetCommonRuntimeArgs(program, k, {        // one shared copy for all cores
    {"scaler",   0x3f800000},
    {"sem_addr", sem.address()},
});
```

Generated `kernel_args_generated.h` (never seen by the user):

```cpp
namespace args {
  // 3 CTAs — value baked in (the brace is the VALUE)
  constexpr experimental::CtaVal<uint32_t>  block_h{4u};
  constexpr experimental::CtaVal<uint32_t>  block_w{2u};
  constexpr experimental::CtaVal<uint32_t>  untilize{1u};

  // 3 RTAs — per-core L1 region (the brace is a BYTE OFFSET)
  constexpr experimental::RtaArg<uint32_t>  src_addr{0};
  constexpr experimental::RtaArg<uint32_t>  dst_addr{4};
  constexpr experimental::RtaArg<uint32_t>  num_tiles{8};

  // 2 CRTAs — separate common L1 region, so offsets restart at 0
  constexpr experimental::CrtaArg<uint32_t> scaler{0};
  constexpr experimental::CrtaArg<uint32_t> sem_addr{4};
}
```

Generated `kernel_main()` shim:

```cpp
void kernel_main() {
    my_kernel<get_arg(args::block_h), get_arg(args::block_w), get_arg(args::untilize)>(
        get_arg(args::src_addr),   // RtaArg  → L1 read, per-core region offset 0
        get_arg(args::dst_addr),   // RtaArg  → offset 4
        get_arg(args::num_tiles),  // RtaArg  → offset 8
        get_arg(args::scaler),     // CrtaArg → L1 read, common region offset 0
        get_arg(args::sem_addr));  // CrtaArg → common region offset 4
}
```

Three things this example makes concrete:

- **The CTA-vs-runtime split is the only thing in the signature** — 3 template params (CTAs)
  vs 5 function params. The shim puts the CTAs in `<…>` (they're `constexpr`) and the rest
  in `(…)`.
- **RTA vs CRTA is invisible in the kernel.** All five function args are plain `uint32_t`
  and all five shim calls are identical `get_arg(args::name)`. Only the *generated type*
  (`RtaArg` vs `CrtaArg`) differs, and that came purely from the host putting the name in
  `runtime_arg_names` vs `common_runtime_arg_names`.
- **RTA and CRTA offsets are independent.** Both start at `0` because per-core and common
  args live in separate L1 regions (`rta_offset` vs `crta_offset` in the launch message).

## 10. Implementation checklist (for the eventual PR)

1. `TT_KERNEL` marker macro (→ `[[tt::kernel_main]]`) + attribute tolerance in the kernel
   build flags (`-Wno-attributes` or register the attribute).
2. `parse_kernel_main_signature()` (tokenizer, option A) in `jit_build` — extracts function
   name, template non-type params (CTAs), and function params (runtime).
3. `write_kernel_main_shim()` in `genfiles.cpp`: emit `my_kernel<get_arg(args::ct)…>(
   get_arg(args::rt)…)`, omitting the `<…>` when there are no template params. Wire its
   include into `jit_build_genfiles_kernel_include()` and `jit_build_genfiles_triscs_src()`
   after the user source.
4. `SetRuntimeArgs` / `SetCommonRuntimeArgs` name→value overloads in
   `host_api.hpp` + `tt_metal.cpp`, resolving to the existing positional path.
5. Host-side validation (Section 5.3), including name↔kind matching.
6. Tests: a DM-kernel and a compute-kernel example (each with at least one CTA template
   param and runtime fn params), built and run on a Wormhole target; an RTA→CRTA promotion
   test (host-only change, **zero** kernel edits); and a negative test that a runtime arg
   in a template slot fails to compile.
7. Docs: promote this design to a user-facing how-to once landed.

## Appendix: source anchors (as of this investigation)

| Concept | File | Symbol |
| --- | --- | --- |
| Metal 2.0 accessors + `get_arg` | `tt_metal/hw/inc/experimental/kernel_args.h` | `RtaArg`/`CrtaArg`/`CtaVal`, `get_arg` |
| Named-arg header codegen | `tt_metal/jit_build/genfiles.cpp:200` | `write_kernel_args_generated_header` |
| Kernel-include assembly | `tt_metal/jit_build/genfiles.cpp:282` | `jit_build_genfiles_kernel_include` |
| TRISC wrapper assembly | `tt_metal/jit_build/genfiles.cpp:305` | `jit_build_genfiles_triscs_src` |
| Firmware entry (bare call) | `tt_metal/hw/firmware/src/tt-1xx/brisck.cc:80` | `_start` → `kernel_main()` |
| Device fetch primitives | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | `get_arg_val`/`get_arg_addr` |
| Per-RISC L1 base setup | `tt_metal/hw/inc/internal/firmware_common.h` | `rta_l1_base`/`crta_l1_base` |
| Host RTA API | `tt_metal/api/tt-metalium/host_api.hpp` | `SetRuntimeArgs`/`GetRuntimeArgs` |
| Host RTA storage | `tt_metal/impl/kernels/kernel.cpp` | `set_runtime_args` |
| L1 write path | `tt_metal/impl/host_api/tt_metal.cpp:1016` | `WriteRuntimeArgsToDevice` |
| RTA offset config | `tt_metal/impl/program/dispatch.cpp:142` | `configure_rta_offsets_for_kernel_groups` |
| Arg-kind schema fields | `tt_metal/api/tt-metalium/kernel_types.hpp` | `named_compile_args`, `runtime_arg_names` |
