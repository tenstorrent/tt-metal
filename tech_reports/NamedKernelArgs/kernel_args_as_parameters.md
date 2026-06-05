# Kernel Arguments as Function and Template Parameters

**Status:** Design proposal
**Scope:** Device-side kernel authoring ergonomics for compile-time / runtime / common-runtime args
**Builds on:** Metal 2.0 named-argument infrastructure (`experimental/kernel_args.h`, per-kernel `kernel_args_generated.h`)

## 1. What we have today (Metal 2.0)

Metal 1.0 reads args positionally — `get_arg_val<uint32_t>(0)` — with hand-tracked indices
that silently desync from the host's `SetRuntimeArgs`. Metal 2.0 fixed the indices by
giving args **names**. Three pieces already exist in tree:

**Accessor layer** (`tt_metal/hw/inc/experimental/kernel_args.h`) — three accessor structs
and one overloaded `get_arg`:

```cpp
template <typename T> struct RtaArg  { uint32_t byte_offset; };  // per-core runtime arg
template <typename T> struct CrtaArg { uint32_t byte_offset; };  // common runtime arg
template <typename T> struct CtaVal  { T value; };               // compile-time constant

template <typename T> FORCE_INLINE T get_arg(RtaArg<T> a);            // reads per-core L1
template <typename T> FORCE_INLINE T get_arg(CrtaArg<T> a);           // reads common L1
template <typename T> FORCE_INLINE constexpr T get_arg(CtaVal<T> a);  // returns the constant
```

**Per-kernel codegen** — `write_kernel_args_generated_header()` (`genfiles.cpp:200`) emits
a `kernel_args_generated.h` with one `args::<name>` constant per declared name, its *type*
fixing the arg kind and its *brace* holding a baked value (CTA) or a byte offset (RTA/CRTA).

**Host schema** — `named_compile_args` (CTAs) and ordered `runtime_arg_names` /
`common_runtime_arg_names` on the kernel spec (`kernel_types.hpp`).

So a kernel reads by name:

```cpp
void kernel_main() {
    auto src_addr  = get_arg(args::src_addr);
    auto num_tiles = get_arg(args::num_tiles);
}
```

Two properties of this layer are load-bearing for the proposal:

- **(a) RTA and CRTA are indistinguishable to the caller.** Both `get_arg` runtime overloads
  return `T`; which is selected depends only on the *type* of `args::<name>`, decided by the
  host schema. Promoting an arg from per-core to common is a host-only change — the call
  site is identical.
- **(b) The CTA accessor is `constexpr`.** `get_arg(args::z)` is a constant expression, so it
  can be used as a **template argument**.

**The remaining gap:** the user still hand-writes `void kernel_main()` and calls `get_arg`
once per arg.

## 2. Proposal

Let the user write the entry as an ordinary typed function whose **parameters are the
args**. No `get_arg`, no `kernel_main`, no L1, no indices.

- **Template parameters → CTAs.** Real constant expressions (property *b*), usable in
  `if constexpr`, array bounds, nested template args. This is the one distinction the
  *source* must express, because it changes what the compiler can specialize.
- **Function parameters → runtime args.** RTA vs CRTA is a pure host-schema choice with
  zero source impact (property *a*); the body can't tell and shouldn't have to.

The entry is tagged with a `TT_KERNEL` marker (`#define TT_KERNEL [[tt::kernel_main]]`). At
JIT time we parse its signature and generate a `kernel_main()` shim that fetches every arg
by name and calls the user function — CTAs in the angle brackets, runtime args in the
parens:

```cpp
my_kernel<get_arg(args::z)>(get_arg(args::x), get_arg(args::y));
//       └ CTA: constexpr ┘ └ RTA/CRTA: runtime L1 reads ┘
```

This is the entire device-side mechanism. **No firmware change** (`brisck.cc` still calls a
bare `kernel_main()`), **no host write-path change** — we reuse the accessors and codegen
that already exist. The two additions are a name→value host API and the shim generator.

### Worked example — 3 CTAs, 3 RTAs, 2 CRTAs

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

Generated `kernel_main()` shim (never seen by the user):

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

Three things this makes concrete:

- **The CTA-vs-runtime split is the only thing in the signature** — 3 template params vs 5
  function params. The shim puts the CTAs in `<…>` (constexpr) and the rest in `(…)`.
- **RTA vs CRTA is invisible in the kernel.** All five function args are plain `uint32_t`
  and all five shim calls are identical `get_arg(args::name)`. Only the *generated type*
  (`RtaArg` vs `CrtaArg`) differs — set purely by which host list the name is in.
- **RTA and CRTA offsets are independent.** Both start at `0` because per-core and common
  args live in separate L1 regions (`rta_offset` vs `crta_offset` in the launch message).

## 3. Pros and cons

**Pros**

- **Zero boilerplate.** Kernels read like ordinary functions; no `get_arg`, `args::`,
  `kernel_main`, indices, or L1 addresses anywhere in user code.
- **Cheap to build.** Reuses the existing Metal 2.0 accessors + `args::` codegen; no
  firmware change and no host L1 write-path change. The net new code is a signature parser,
  a shim emitter, and host name→value overloads.
- **Host-only RTA↔CRTA promotion.** Moving a runtime arg between per-core and common is a
  one-line host edit with zero kernel changes (property *a*).
- **Real compile-time args.** CTAs are genuine `constexpr` template params — usable for
  `if constexpr` specialization, array sizes, and further template arguments.
- **Mistakes are caught by the compiler, at the user's source.** A runtime arg placed in a
  template slot fails to compile (a non-`constexpr` value can't be a template argument). A
  CTA misused as a *function* param still errors the moment the body uses its
  compile-time-ness (`if constexpr (z)`, `int buf[z]`, `foo<z>()`) — pointed at the user's
  own line. Names are self-documenting and order-independent within a list.
- **Purely additive.** Gated on the `TT_KERNEL` marker; Metal 1.0, Metal 2.0, and
  hand-written `kernel_main()` kernels are untouched.

**Cons / costs**

- **New JIT parse step.** We must extract the entry signature. A lightweight tokenizer
  suffices while args are `uint32_t`-only, but has a limited surface (rejects type template
  params, defaulted params, macros in the signature); richer types (Phase 2) force a real
  parser, which adds either a clang dependency or a GCC plugin (§5).
- **Compile-time-vs-runtime is a source decision.** Moving an arg across that line (CTA ↔
  runtime) *is* a kernel edit — unavoidable, since it changes what the compiler can
  specialize, but it breaks the "host-only" promotion story for that one axis.
- **`uint32_t`-only initially**, inherited from the `sizeof(T)==4` assert in `kernel_args.h`
  (template params likewise non-type `uint32_t`/`bool`).
- **More indirection to debug.** Compile errors can point at the generated shim /
  `kernel_args_generated.h` rather than the user file; needs decent diagnostics and marker
  discipline (exactly one `TT_KERNEL` per TU).

## 4. Implementation details

All codegen lives in `tt_metal/jit_build/genfiles.cpp`, next to the existing
`write_kernel_args_generated_header()` and `jit_build_genfiles_kernel_include()`.

**Marker.** `#define TT_KERNEL [[tt::kernel_main]]` in a public kernel header. Used as
`TT_KERNEL void my_kernel(...)`. The attribute is a vendor-namespaced no-op for the compiler
(build with `-Wno-attributes` or register it); it exists purely as a reliable parse anchor.
Exactly one per TU — zero falls back to a hand-written `kernel_main()`, two is a JIT error.

**Signature parser** — `parse_kernel_main_signature(source) -> {name, template_params,
fn_params}`. The marker sits between the optional `template <…>` clause and the return type,
so the parser scans backward for the template clause and forward for `ret-type name
( params )` up to the matching `{`, splitting each list on top-level commas. Ship a
**tokenizer** for Phase 1 (no new dependency; the `uint32_t`-only surface keeps it
tractable), kept behind this one function so a real-parser implementation can replace it in
Phase 2 — see §5 for the GCC-driven options (`clang -ast-dump=json`/libclang as a separate
parse, or a GCC plugin in the existing compile) once richer types break the tokenizer.

**Shim emitter** — `write_kernel_main_shim()` emits `kernel_main()` calling
`my_kernel<get_arg(args::ct)…>(get_arg(args::rt)…)`, one `get_arg(args::<name>)` per param
in declared order, omitting the `<…>` entirely when there are no template params. Because
the `args::*` constants already exist and `get_arg` is overloaded on their type, that's the
whole shim.

**Include ordering** — append the shim *after* the user source in `kernel_includes.hpp`
(the entry must be declared first): `kernel_bindings_generated.h` → `kernel_args_generated.h`
→ user source → `kernel_main_shim.h`. The TRISC path
(`jit_build_genfiles_triscs_src`, `genfiles.cpp:305`) gets the same shim include in each
`chlkc_*.cpp` wrapper.

**Host name→value API** — add overloads alongside positional `SetRuntimeArgs`
(`host_api.hpp`):

```cpp
void SetRuntimeArgs(const Program&, KernelHandle,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>&,
    const std::unordered_map<std::string, uint32_t>& named_args);
void SetRuntimeArgs(const Program&, KernelHandle,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet>&,
    stl::Span<const std::string_view> names, stl::Span<const uint32_t> values);
void SetCommonRuntimeArgs(const Program&, KernelHandle,
    const std::unordered_map<std::string, uint32_t>& named_args);
```

These reorder the map into the registered name order and funnel into the **existing**
positional `set_runtime_args` path — L1 layout, offsets (`dispatch.cpp`), and
`WriteRuntimeArgsToDevice` are unchanged. Pure host-side name→index resolution.

**Validation.** At set-time / finalize: every registered name has a value, and no unknown
names are supplied. Beyond that, the compiler already does the heavy lifting — a missing
`args::<name>` or a runtime arg in a template slot both fail the JIT compile at the user's
source — so v1 needs no parameter-kind cross-check on the host. (An optional lint flagging a
CTA name used as a function param could be added later, but it only catches a benign,
correct-valued case and needs the parsed parameter lists threaded back; low priority.)

## 5. Phased rollout

- **Phase 0 — done.** Metal 2.0 named accessors (`kernel_args.h`), `args::` codegen, host
  named-arg schema. Kernels still hand-write `kernel_main()` + `get_arg`.
- **Phase 1 — all three kinds, `uint32_t`.** `TT_KERNEL` marker + tokenizer for *both* the
  `template <…>` clause (CTAs) and the function parameter list (RTA/CRTA); shim generation
  that instantiates the template and calls with the runtime args; host name→value
  `SetRuntimeArgs` / `SetCommonRuntimeArgs` overloads. `uint32_t`-only, which keeps the
  tokenizer viable (the parameter surface is just `uint32_t name`). Proves the whole
  mechanism end-to-end on silicon. Deliverable: the §2 example (CTAs + RTAs + CRTAs), an
  RTA→CRTA host-only promotion test, and a negative test that a runtime arg in a template
  slot fails to compile.
- **Phase 2 — types beyond `uint32_t`.** Lift the `sizeof(T)==4` restriction in
  `kernel_args.h` (multi-word POD reads in `get_arg`/accessors) and allow richer parameter
  types in the signature. This is what forces a **real parser**: arbitrary type spellings
  (`float`, structs, qualified/templated types) are past what the tokenizer can safely
  handle. Given the kernels build with sfpi **GCC** (not clang), the realistic options are a
  separate `clang -ast-dump=json` / libclang parse (a new clang dependency; expand the marker
  to `[[clang::annotate("tt_kernel_main")]]` so it survives into the AST) or a GCC plugin
  that emits the signature during the existing compile. Also: improve diagnostics so errors
  point back at the user file, and promote this design to a user-facing how-to.

## Appendix: source anchors

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
| L1 write path | `tt_metal/impl/host_api/tt_metal.cpp:1016` | `WriteRuntimeArgsToDevice` |
| RTA offset config | `tt_metal/impl/program/dispatch.cpp:142` | `configure_rta_offsets_for_kernel_groups` |
| Arg-kind schema fields | `tt_metal/api/tt-metalium/kernel_types.hpp` | `named_compile_args`, `runtime_arg_names` |
