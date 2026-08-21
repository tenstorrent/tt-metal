# Metal 2.0 — Explicit (named) kernel args and recovering `if constexpr`: analysis

**Author:** Claude (Metal 2.0 recipe development), with Gaurav.

**Purpose:** Work out, and record with evidence, exactly *what* the explicit
kernel-argument syntax from [PR #46623](https://github.com/tenstorrent/tt-metal/pull/46623)
changes, *why* the basic Metal 2.0 port is forced to spell conditional resources with `#ifdef`
instead of `if constexpr`, and *which* of those `#ifdef`s the new syntax can actually retire (and
which it deliberately leaves). This is the checked-in reasoning behind the post-port style recipe
[`ai/post_port/style/named_kernel_args.md`](../ai/post_port/style/named_kernel_args.md); the
recipe is the procedure, this is the findings it rests on. `softmax` is the worked example
throughout.

> **Decaying snapshot.** File paths, line numbers, and the "not yet implemented" claims below
> (esp. unbound DFB accessors) are true as of 2026-08-18 on this branch. Re-verify against current
> code before relying on any specific line cite.

> **Revised 2026-08-20 after the first real run.** This analysis originally recommended converting the
> token-gated control flow to `if constexpr` behind minimal *fallback-alias islands* (Finding B). A
> softmax port exercised that and it worked numerically — but on review the recipe **rejected** it: a
> fallback alias trades a hard compile error for a silent wrong-id, a bad bargain in a
> behaviour-preserving pass. **The recipe now leaves every conditional-resource `#ifdef` exactly as
> the basic port wrote it, and converts only (a) the kernel signature and (b) *pure* compile-time
> flags** (those whose branches name only always-present symbols). The run also established that a
> kernel with **any conditionally-registered arg is a hard stop** (Finding D). The sections below are
> updated to that model; Finding B is kept as the reason the shortcut was *available and declined*,
> not as a recommendation.

---

## 1. The feature in one paragraph

Before PR #46623, a Metal 2.0 kernel hand-wrote `void kernel_main()` and pulled every argument
by name through a single overloaded accessor: `constexpr auto x = get_arg(args::x)` for a
compile-time arg (CTA), `auto y = get_arg(args::y)` for a runtime (RTA) or common-runtime (CRTA)
arg. PR #46623 removes the boilerplate: you write **one plain function whose parameters are the
arguments**, and the JIT generates the `kernel_main()` shim that fetches them and calls you.

```cpp
// NEW: the whole kernel. CTAs are template params; RTAs/CRTAs are function params.
template <uint32_t Ht, uint32_t Wt, uint32_t untilize>            // CTAs (compile-time)
TT_KERNEL void my_reader(uint32_t start_tile_id, uint32_t scaler) {  // RTAs / CRTAs (runtime)
    // Ht/Wt/untilize are true constant expressions; start_tile_id/scaler are runtime values.
}
```

The generated shim, emitted after the user source and the `args::` header, is simply:

```cpp
void kernel_main() {
    my_reader<get_arg(args::Ht), get_arg(args::Wt), get_arg(args::untilize)>(
        get_arg(args::start_tile_id), get_arg(args::scaler));
}
```

Key facts, all load-bearing for the recipe:

- **`TT_KERNEL`** is `static FORCE_INLINE`, defined in
  `tt_metal/hw/inc/experimental/kernel_args.h`. The user never defines `kernel_main()`; the shim
  does, and folds the entry into it.
- **CTAs ride the template parameter list; RTAs/CRTAs ride the function parameter list.** The
  kernel cannot tell RTA from CRTA and does not need to — that split is entirely host-side, which
  is exactly what makes moving an arg between RTA and CRTA a host-only dispatch tweak.
- **Phase-1 parser restrictions** (a hand-rolled tokenizer, not a C++ frontend, in
  `tt_metal/jit_build/kernel_signature_parser.cpp`):
  every template and function parameter must be spelled **`uint32_t`** (or `std::uint32_t`). No
  `typename` template params, no other scalar types, no defaulted params, non-`void` return
  throws. `uint64_t` / `std::array` are the next phase. **So a config flag becomes a `uint32_t`
  NTTP valued 0/1 — not a `bool`.**
- **Only available with Metal 2.0**, and (as of this branch) both DM and compute kernels are
  supported.

Real in-tree examples using the syntax today (not `experimental/quasar/`, which is off-limits):
`ttnn/cpp/ttnn/operations/experimental/kda/sigmoid_gated_rms_norm/device/kernels/` (reader,
writer, compute) and the framework tests
`tests/tt_metal/tt_metal/test_kernels/dataflow/tt_kernel_named_args_*.cpp`.

---

## 2. Why the basic port is forced onto `#ifdef`

The basic Metal 2.0 port recipe **mandates** `#ifdef` (never `if constexpr`) for any
conditionally-bound resource. The reason is stated at
`ai/shared/migration_guide.md:804` and `ai/shared/port_patterns.md:194`, and it is precise:

> Metal 2.0's `dfb::<name>` namespace is **generated from the actual host bindings** — `dfb::cb_scaled`
> exists only when the host actually binds it — and `if constexpr` in **non-template** `kernel_main`
> still performs name lookup on the discarded branch, so `if constexpr (false) { … dfb::cb_scaled … }`
> fails to compile at parse time.

The generator confirms the premise: `dfb::`, `sem::`, `tensor::`, and `args::` tokens are emitted
**per binding** in `genfiles.cpp` (`namespace dfb { constexpr DFBBindingToken <name>{id}; }`, only
for names in `dfb_entries`). Omit the host binding on a code path and the token does not exist in
that build — it is a genuine *missing name*, not a wrong value.

The migration guide phrases the constraint as a property of *non-template* `kernel_main`, which
correctly implies the question PR #46623 raises: **does making the kernel body a template rescue
`if constexpr` over a missing token?**

---

## 3. Empirical findings (the decisive part)

I compiled the minimal cases with `g++ -std=c++17`. These are the facts the recipe is built on.

**Finding A — a genuinely missing token fails in *both* a template and a non-template body.**

```cpp
namespace dfb { constexpr int cb_out = 5; }   // cb_fusion intentionally absent

template <bool FUSE>
int writer_tmpl() {
    if constexpr (FUSE) { return dfb::cb_fusion; }  // discarded when FUSE=false
    else                { return dfb::cb_out; }
}
int main() { return writer_tmpl<false>(); }
```

→ **`error: 'cb_fusion' is not a member of 'dfb'`**, identical to the non-template version.
`dfb::cb_fusion` is a *non-dependent qualified-id*; two-phase lookup binds it at template
**definition**, and `if constexpr`'s "discarded statement is not *instantiated*" rule does not
suppress that lookup. **Templating the kernel does not, by itself, let you name a token that the
host did not emit.** (Moving the reference *inside* the discarded branch does not help either —
same error.)

**Finding B — with the name *always declared*, a well-formed use under `if constexpr` compiles in
both bodies.** Give the alias a fallback so the *name* always exists, and only its *value* is
conditional:

```cpp
#ifdef FUSED
constexpr auto dfb_fused_attn = dfb::fused_attn;   // real token when the host bound it
#else
constexpr uint32_t dfb_fused_attn = 0;             // fallback; never actually used
#endif
...
if constexpr (fused) { DataflowBuffer a(dfb_fused_attn); a.wait_front(1); }
```

This compiles feature-off in a non-template `kernel_main` **and** in a template body:
`DataflowBuffer(0)` is well-formed, so the discarded branch type-checks (non-template) / is not
instantiated (template). **This is the shortcut the recipe declines** (revision note above, and §7):
it works, but the `= 0` is a silent wrong id the moment a use escapes the guard, so the recipe keeps
the `#ifdef` instead of shrinking it.

**Finding C — a template body additionally protects a discarded branch that is ill-formed only
*on instantiation*** (a dependent construct: a member a type only conditionally has, an
`int arr[N]` with `N==0`, a `static_assert` on the template param). A non-template `if constexpr`
type-checks both branches fully and cannot express that; a template does not instantiate the
discarded one. This is a real but secondary benefit; softmax's branches are mostly not of this
shape. (It is also the reason the declined fallback shortcut is *extra* unsafe in a no-CTA,
non-template entry: there the `= 0` fallback branch is fully type-checked even when unreached.)

**Finding D — a kernel with any *conditionally-registered* arg cannot be converted at all.** Not a
C++ fact but a JIT-validator one, found on softmax's readers. `validate_signature_against_schema` →
`check_name_sets` (`tt_metal/jit_build/kernel_signature_parser.cpp`) requires the kernel's parameter
set to **exactly equal** the *per-build* registered args — every parameter has a registered arg, and
every registered arg is a parameter. A signature is one fixed source, so if the factory registers a
different arg set per build (softmax's readers register the mask/causal args only when `has_mask` /
causal), **no single signature satisfies every build**: omit the args and the mask build fails
(`registered runtime argument(s) not taken as a function parameter`); `#ifdef`-gate them in the
signature and the no-mask build fails, because the signature parser reads *raw, unpreprocessed*
source and extracts the union. Such a kernel stays legacy `kernel_main()`. **Conditional *tokens* do
not block conversion — only conditional *args* do.**

### What this means

| Claim | Verdict |
|---|---|
| Template kernel lets `if constexpr` name a **host-omitted** token | **False** (Finding A) |
| A fallback alias makes the missing name resolve, so `if constexpr`-over-*uses* compiles | **True** (Finding B) — but **declined** (silent wrong-id; §7) |
| A kernel with a conditionally-registered *arg* can be converted | **False** (Finding D) — hard stop |
| The token-existence `#ifdef` can be deleted today | **False** — needs unbound accessors (§8) |

So the honest headline is: **the new syntax converts the kernel *signature*, and lets a pure
compile-time flag use `if constexpr` — but a `#ifdef` guarding a conditionally-bound token is left in
place, and a kernel with a conditional *arg* does not convert at all.** The recipe does **not** try to
minimize the token-existence `#ifdef` via fallbacks; it leaves it.

---

## 4. The two categories of `#ifdef` in a ported op

Every `#ifdef` in a basic-ported kernel is one of two kinds, and they have different fates:

1. **Control-flow / value gate** — the guarded code names only always-present symbols (LLK calls,
   arithmetic, loops, always-bound DFBs). It was spelled `#ifdef` only because its *condition*
   (`FUSED_SCALE_MASK`, `NUMERIC_STABLE`, …) arrived as a host `#define` rather than a constant.
   → **Convertible to `if constexpr`.**

2. **Conditional-resource gate** — the guarded code names a `dfb::`, `tensor::`, `sem::`, or
   `args::` token that the host binds/registers only on this path, so the name is absent otherwise.
   → **Left as `#ifdef`, untouched.** (A conditional *arg* is stronger: it makes the whole kernel a
   hard stop — Finding D.)

A single flag usually guards *both* kinds (e.g. `FUSED_SCALE_MASK` selects an algorithm **and**
gates the fused DFBs). When it does, **the conditional-resource kind wins and the whole flag stays on
`#ifdef`.** Only a flag that is *purely* value-gating everywhere it appears in the kernel is
converted — so on softmax, where all four flags gate a conditional resource, none convert.

---

## 5. Softmax pattern catalog (evidence)

Concrete `#ifdef` sites in the ported attention factory
(`device/kernels/attention/compute/softmax.cpp`, `.../dataflow/reader_unary_interleaved_sm.cpp`),
with the reframed disposition. The four host flags are emitted as `compiler_options.defines` in
`softmax_program_factory_attention_optimized.cpp:311-329`.

| Flag / site | Why it stays / stops | Disposition |
|---|---|---|
| `#if FUSED_SCALE_MASK` → `dfb::fused_scale/fused_attn/scale_mask` aliases + the whole fused block (`softmax.cpp:131-139, 163-260, 386-407`) | `dfb::fused_*` bound only when `has_mask` (`factory:341-345, 433-438`) | **stays `#ifdef`** |
| `#ifdef NUMERIC_STABLE` → `dfb::max` alias + `calc_numeric_stable`/`exp` blocks (`softmax.cpp:146-147, 195-197, 240-242, 301-312`) | `dfb::max` self-loop bound only when numeric-stable (`factory:440-442`) | **stays `#ifdef`** |
| `#if defined(NUMERIC_STABLE) && (FUSED‖MASK_PADDED)` → `dfb::x` alias (`softmax.cpp:148-153`) | `dfb::x` bound only in that combo; the basic port already has a natural `dfb_x = dfb_exps` on the else path (`softmax.cpp:156`) | **stays `#ifdef`** (leave the natural fallback as-is) |
| `#ifdef MASK_PADDED_DATA` → `dfb::mask_padded` alias + pad-mask loop (`softmax.cpp:140-143, 268-337`) | **`dfb::mask_padded` is *unconditionally* bound** (writer PRODUCER + compute CONSUMER regardless), so that token always exists — but the same `MASK_PADDED_DATA` define **co-gates `dfb::x`** (`softmax.cpp:148`), which is conditional | **stays `#ifdef`** |
| `#ifdef CAUSAL_MASK` → `add_init` vs `add_bcast_rows_init`, loop shape (`softmax.cpp:199-221, 244-258`) | pure control-flow in the compute, but nested inside `FUSED_SCALE_MASK`; in the reader `CAUSAL_MASK` gates conditional args | **stays `#ifdef`** |
| reader `#if FUSED_SCALE_MASK` → `get_arg(args::Ht/start_ht/start_mask_id/pre_scale)` (`reader:35-37, 58`) | **conditional args** — registered only when `has_mask` (`factory:341-350`) | **hard stop** — reader does not convert (Finding D) |
| reader `#if CAUSAL_MASK` → `get_arg(args::num_tiles_causal_mask/mask_start_ht/mask_offset)` (`reader:46-48`) | **conditional args** — registered only when causal (`factory:351-354, 364-369`) | **hard stop** — reader does not convert (Finding D) |

**Takeaway.** Every one of softmax's four flags gates a conditional `dfb::` token (or, for
`MASK_PADDED_DATA`, co-gates one), so **every `#ifdef` stays exactly as the basic port wrote it.**
The compute kernels and the writer convert for their **signature only**; the two readers do not
convert at all, because they register the mask/causal args conditionally (Finding D). Net: the diff
is the signatures, and nothing else.

---

## 6. The transformation model

Behaviour-preserving, minimal host churn. Two things change per convertible kernel; a third is
deliberately left alone.

**1. Kernel signature (the primary change).** Convert each entry to the PR #46623 shape:
- Every value the basic port read as `constexpr auto n = get_arg(args::n)` (an **unconditional**
  CTA) → a `uint32_t` **template parameter** named `n`.
- Every value read as `auto n = get_arg(args::n)` (an **unconditional** RTA/CRTA) → a `uint32_t`
  **function parameter** named `n`.
- **The parameter name must equal the registered arg name** — the shim emits
  `get_arg(args::<paramname>)`. A rename silently reaches the wrong arg or fails to compile.
- Delete the hand-written `void kernel_main()`; add `template<…> TT_KERNEL void <entry>(…)` (a
  no-CTA kernel is a bare, non-template `TT_KERNEL void <entry>(…)`).
- **If *any* CTA/RTA/CRTA is registered conditionally, the kernel is a hard stop** — do not convert
  the signature at all (Finding D).

**2. Pure-value flags (the only `#ifdef` → `if constexpr` conversion).** A flag *every* branch of
which names only always-present symbols → promote to a `uint32_t` NTTP, move it from `defines` to
`compile_time_args`, and use `if constexpr (FLAG)`. If the flag gates a conditional resource
anywhere in the kernel, it is **not** this case.

**3. Conditional-resource `#ifdef`s — left untouched.** A `#ifdef` guarding a conditionally-bound
`dfb::`/`tensor::`/`sem::` token, or reading a conditional arg, stays exactly as the basic port wrote
it. No fallback aliases, no derived `constexpr bool`, no promotion; the `#define` stays too. This is
the reversal of this analysis's original draft (see the revision note) — the shortcut in Finding B is
available but declined (§7).

**Net host diff for softmax:** none. `defines`, conditional bindings, and CTA/RTA registration are
all unchanged; the arg *names* simply now double as parameter names on the converted kernels. The
work is entirely kernel-side, and the readers are not touched.

---

## 7. Why the fallback shortcut is declined

- **Finding B works, but a fallback alias trades a compile error for a silent wrong value.** With
  the alias absent (the basic port's `#ifdef`), using the buffer off-path fails to *compile* — a
  loud, safe error. With a `= 0` fallback the name resolves, so an off-path use silently reads DFB
  id 0. A template body defuses most of this (the false branch is not instantiated, so the fallback
  is unreachable in emitted code), but not all: a use placed *outside* every `if constexpr` guard
  compiles against the `= 0`, and in a no-CTA **non-template** entry the discarded branch is fully
  type-checked, so even a guarded template-argument use can be instantiated against the fallback
  (Finding C). For a pass whose entire value is a diff a reviewer can trust, converting a loud
  compile error into a latent wrong-id is the wrong trade — so the recipe keeps the `#ifdef`.
- **`volatile` and format metadata** are not in play for softmax's conditional buffers, but the
  general caution from the sync-free-DFB pass applies if a conditional buffer carries either.
- **The parts that *are* done are behaviour-preserving.** CTAs-as-defines and CTAs-as-template-params
  are *both* compile-time; the generated device code, and thus numerics and performance, are
  identical. The shim is the only added indirection and it is `FORCE_INLINE`d away. That is what
  makes the op's own tests a valid before/after check.

---

## 8. The residual `#ifdef`, and the future that removes it

The token-existence `#ifdef` survives because a `dfb::`/`args::` token that the host did not emit is
an undeclared name, and no C++ construct can reference an undeclared non-dependent name. The
framework fix is tracked as
[issue #52179 — "Add unbound DFB accessors for single-source optional DFB kernels"](https://github.com/tenstorrent/tt-metal/issues/52179):
if `genfiles` always emits `dfb::<name>` (bound or as an *unbound* sentinel), the name always
resolves, and `if constexpr(flag)` alone suffices — no `#ifdef`. As of this branch that is **not
implemented** (no `unbound` emission path in `genfiles.cpp`; nothing in `dataflow_buffer.h`). Until
it lands, leaving the `#ifdef` is the correct, honest end-state, and the recipe says so rather than
inventing a workaround (fallback aliases, always-binding, `.id` sentinels smuggled through CTAs).

The same reasoning extends to **optional tensors** (`tensor::`) and **semaphores** (`sem::`); neither
has an unbound accessor today either, so both keep their `#ifdef`.

---

## 9. Open questions for the recipe maintainers

- **A flag that is pure-value in one kernel but conditional-resource in another.** The recipe
  converts a flag only when it is pure-value *everywhere it appears in that kernel*, so this is
  decided per kernel: the flag can be a CTA on the kernel where it is pure-value and stay a `#define`
  on the kernel where it gates a resource. Workable, but a mixed op would carry the same flag in two
  forms — worth a consistency convention if it shows up.
- **CRTA vs RTA is invisible to the kernel** — the signature cannot express, and need not, which a
  param is. The recipe reminds the porter to preserve the host's existing RTA/CRTA split and never
  infer it from the kernel.
- **Varargs** (`get_vararg`) have no signature spelling (PR #46623 §"grossness"); a variadic kernel
  keeps the legacy `get_arg`/`get_vararg` reads and is only partially convertible. softmax has none,
  but the recipe flags it as a stop condition.
- **Unbound accessors (#52179)** are the single change that would let the conditional-resource
  `#ifdef`s become `if constexpr` — the biggest lever on this pass, and out of the porter's hands.
