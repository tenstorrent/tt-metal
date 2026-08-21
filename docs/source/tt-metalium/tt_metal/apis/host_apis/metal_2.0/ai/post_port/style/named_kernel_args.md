# Post-Port Fix — Explicit (named) kernel args, and recover `if constexpr` from `#ifdef`

> **Procedure:** [`pass_procedure.md`](../pass_procedure.md). Read it first — its baseline,
> scope, verify, report, and stop rules all bind here unchanged. **This pass is larger than the
> others and overrides one thing:** its [Step 2 (Survey)](#step-2--study-and-plan) becomes a
> per-kernel *study + written plan*, and a [confirmation stop](#step-3--confirm-with-the-invoker)
> is inserted before Apply. The procedure explicitly permits a recipe to override a step; this is
> that. Everything else in `pass_procedure.md` holds.
>
> **Behaviour-preserving?** Yes. CTAs delivered as `#define`s or as template parameters are *both*
> compile-time; the generated device code, numerics, and performance are identical. The only added
> indirection is the JIT-generated `kernel_main()` shim, which is `FORCE_INLINE`d away. The op's own
> tests are therefore a valid before/after check.
>
> **Target:** Gen1 (Wormhole / Blackhole) ops already ported to Metal 2.0 whose kernels still use the
> hand-written `void kernel_main()` + `get_arg(args::…)` form. The headline change is the
> explicit-args signature; `#ifdef` → `if constexpr` is a narrow secondary that applies only to a
> pure compile-time flag, so most `#ifdef`s are left as they are.
>
> **Ordering & batching.** This pass has a [confirmation stop](#step-3--confirm-with-the-invoker),
> so — unlike the small style passes — it is **best run on its own with the invoker reachable**,
> not dropped into an unattended batch. If it *is* run alongside
> [`sync_free_dfbs.md`](sync_free_dfbs.md), run **that one first**: it removes DFBs, so there are
> fewer conditional buffers whose `#ifdef`s you must leave in place here.
>
> **Deep background (optional):** the *why* behind every claim here — the C++ name-lookup rule,
> the compile experiments, the softmax catalog — is in
> [`analyses/2026-08-18_named_kernel_args_and_constexpr.md`](../../../analyses/2026-08-18_named_kernel_args_and_constexpr.md).
> You do not need it to run the pass; reach for it when a site argues with the recipe.

---

## What this fix is

Two changes, in priority order:

1. **Adopt the explicit kernel-argument syntax** from
   [PR #46623](https://github.com/tenstorrent/tt-metal/pull/46623) — the **primary** change, and the
   one that applies to nearly every kernel. The kernel stops hand-writing `void kernel_main()` and
   pulling each argument by name; instead its **arguments become its parameters** — compile-time args
   (CTAs) as `uint32_t` **template parameters**, runtime and common-runtime args (RTAs/CRTAs) as
   `uint32_t` **function parameters** — and the JIT generates the `kernel_main()` that fetches them.

2. **Recover `if constexpr` from `#ifdef` — narrowly.** *Only* where a `#ifdef FLAG` gates code that
   names **only always-present symbols**, promote `FLAG` to an NTTP CTA and write `if constexpr`.
   **A `#ifdef` that guards a conditionally-bound resource — a `dfb::`/`tensor::`/`sem::` token, or a
   conditional arg — is left exactly as the basic port wrote it.** Do not convert it, and do not add
   fallback aliases to make it look convertible. On a typical op most flags fall in this second
   bucket, so this half often does nothing and the signature change (1) is the whole deliverable.

**Why it matters.** The ported corpus is what the next porter, the next op author, and the
customer reading a reference implementation all learn Metal 2.0 *from*. A kernel whose arguments
are positional `get_arg` calls and whose structure is a thicket of `#ifdef` teaches the old model
in new syntax. This is also a **Quasar precondition**: the template-kernel form is where Gen2 is
going, so an op converted here is an op that does not have to be converted again, more expensively,
later.

**What it is *not*.** It does not change which resources are bound, what the kernel computes, or
how work is split. If you find yourself wanting to change any of those to make the syntax fit,
stop — see [When to stop](#when-to-stop).

---

## Prime directive — behavior preservation is paramount

This requirement outranks everything else in this recipe. **The device program built from the
converted kernels must be the one `main` builds today** — identical numerics, L1 footprint,
dispatch, and performance. Your kernel edits are *syntactic sugar*: the same compile-time values and
the same control flow, spelled in the new syntax. Nothing more.

Three things are therefore **out of scope, always** — even when tempting, and even when correct:

- **A change to "get it working."** If the clean conversion won't compile or won't pass, the answer
  is never a workaround — a demoted arg, an always-bound buffer, a hand-tweaked loop, a
  fallback-alias island. Those *are* the failure mode, not the fix. Stop.
- **A bug fix.** If the kernel looks wrong — a real defect, a latent race, a guard that is subtly
  off — **do not fix it here.** A fix changes behavior, which this pass is defined not to do, and
  bundling it makes the one thing this pass guarantees (no behavior change) unverifiable. It belongs
  in a **separate PR, landed *before* this pass runs.** Report it and stop; do not carry it.
- **Any functional / semantic change** to make the syntax fit — moving arithmetic host↔device,
  changing what is bound, altering the RTA/CRTA split, "tidying" the dispatch.

**When the minimal change gets tricky, or the clean version would be ugly, stop and warn — that is a
success, not a failure.** Make the warning **compact and explicit**: what you were converting, the
exact site (`file:line`), a short snippet of the code that does not fit, and one sentence on why a
syntactic-only change cannot cover it. A reader would far rather have that than a diff that quietly
did something clever to avoid stopping. This is the discipline the base port runs on — a grounded
stop is a complete deliverable, on the same tier as a finished conversion.

---

## The one thing this syntax does *not* fix — read before you plan

The seductive error on this pass is to believe that a template kernel lets `if constexpr` replace
*every* `#ifdef`. It does not, and the boundary is sharp.

Metal 2.0's `dfb::<name>`, `tensor::<name>`, `sem::<name>`, and `args::<name>` tokens are generated
**per host binding**. On a code path where the host does not bind a resource, its token **does not
exist as a name** in that build. And a missing non-dependent name is caught at *parse* time —
**even inside a discarded `if constexpr` branch, even in a template kernel** (verified; see the
analysis). Templating the body does not rescue it.

So sort every `#ifdef` you meet into exactly two kinds, and treat them very differently:

- **Pure value gate** — the guarded code names only symbols that exist in *every* build (LLK calls,
  arithmetic, loops, unconditionally-bound resources), and the condition is a plain compile-time
  flag. It was `#ifdef` only because that flag arrived as a `#define`. → **Convertible:** promote the
  flag to an NTTP CTA and use `if constexpr` ([Rule 3](#rule-3--flags)).
- **Conditional-resource gate** — the guarded code names a `dfb::`/`tensor::`/`sem::`/`args::` token
  the host provides only on this path (or reads a conditional arg). → **Leave it as `#ifdef`,
  untouched** ([Rule 4](#rule-4--conditionally-bound-resources-leave-the-ifdef)). It cannot become
  `if constexpr`, because the token is a missing *name* off-path and `if constexpr` still name-checks
  its dead branch. The framework fix that *will* let these convert — unbound accessors,
  [issue #52179](https://github.com/tenstorrent/tt-metal/issues/52179) — is not in the tree yet.

**When a single flag guards both kinds, the conditional-resource kind wins: leave the whole flag on
`#ifdef`.** Do not split one flag into an NTTP for some branches and a `#define` for others, and do
not reach for a fallback-alias trick (a `dfb::x = 0` dummy, or aliasing an unrelated buffer) to make
the resource branch *look* convertible. That trades a hard compile error for a silent wrong-id bug,
and it spends the one thing this pass is worth — a reviewer being able to trust the diff. In practice
most flags on a real op guard a conditional resource somewhere, so the honest, common outcome is:
**convert the signature, leave the `#ifdef`s.**

---

## Procedure

Steps 0, 1, 4-verify, and 5-report are `pass_procedure.md`'s, unchanged. Steps 2-3 are this
recipe's override.

### Step 0 — Confirm the inputs

Per [`pass_procedure.md` Step 0](../pass_procedure.md#step-0--confirm-the-inputs): the **op**, the
**fix** (this file), and the **sentinel set** (your invoker supplies it; do not assemble one
yourself). Venv active.

**Scope unit.** Like a Metal 2.0 port, the natural unit is **one program factory (or descriptor)
together with the kernel entry points it binds** — not necessarily the whole op at once, and it is
also the **unit of commit** ([Step 4](#step-4--apply)). A large multi-factory op is fine to convert
factory-by-factory, each a complete, shippable sub-pass. Convert one factory's
kernels fully, commit it, report the rest as remaining, and stop or continue as context allows. Do
not interleave two factories in one working tree.

**The op must already be ported to Metal 2.0.** If a factory still calls `CreateCircularBuffer` /
`SetRuntimeArgs`, it is unported — stop and report, per Step 0. This pass is a style fixup *on top
of* a working port, not a port.

### Step 1 — Establish the green baseline

Per [`pass_procedure.md` Step 1](../pass_procedure.md#step-1--establish-the-green-baseline). Build
with `./build_metal.sh --build-all` in the background, read the log, run the sentinels from the
repo root. Every sentinel green before you touch anything, or stop and report.

### Step 2 — Study and plan

*(This replaces Survey.)* The unit of this pass is a **kernel signature and its conditional
structure**, not a uniform find-and-replace site, so inventory before you touch code and write it
down. For **each kernel** in the factory you are converting, record in `METAL2_STYLE_PLAN.md`:

1. **The argument census.** Every value the kernel reads, tagged:
   - `constexpr auto n = get_arg(args::n)` → **CTA** → destined for the template parameter list.
   - `auto n = get_arg(args::n)` → **RTA or CRTA** → destined for the function parameter list.
   - Take the **unconditional vs conditional** split from the **factory's** registration (a
     `compile_time_args` / `runtime_arg_names` entry behind a host `if`), not from the kernel. **If
     *any* arg is conditionally registered, the kernel is a hard stop — not convertible.** See
     [Rule 5](#rule-5--conditional-args-are-a-hard-stop).
   - Confirm each is a bare `uint32_t` / `std::uint32_t`. A **non-`uint32_t` registered arg** can be
     neither expressed in the signature nor omitted from it (Rule 5), so that kernel is a **hard
     stop**. A **vararg** (`get_vararg` / `get_common_vararg` / `get_compile_time_vararg`) is
     different — it is not a named arg, so the named args still convert and the vararg reads stay
     manual, making the kernel **partially convertible** (see [When to stop](#when-to-stop)).
2. **The flag census.** Every `#define` the factory emits for this kernel
   (`compiler_options.defines`) and, for each, whether it gates **only always-present code** or
   **touches a conditional resource** (a conditionally-bound `dfb::`/`tensor::`/`sem::` token, or a
   conditional arg). Cross-check against the factory: which bindings and which registered args are
   conditional.
3. **The `#ifdef` disposition.** For each `#ifdef` block, its kind per [the boundary
   above](#the-one-thing-this-syntax-does-not-fix--read-before-you-plan): a **pure value gate**
   (planned end-state: `if constexpr` on an NTTP CTA) or a **conditional-resource gate** (planned
   end-state: *left untouched*). Decide per **flag**: if a flag guards a conditional resource in even
   one block, mark the whole flag "leave on `#ifdef`".
4. **Host touches.** For a pure-value flag you convert, the one host edit is moving it from
   `compiler_options.defines` to `compile_time_args`. A flag that gates any conditional resource stays
   a `#define`, untouched. List the exact factory edits; for most ops there will be **none** beyond
   the signature's arg names matching the registered names — conditional bindings, defines, and arg
   registration are all unchanged.

Also confirm the **shared-kernel** status of each entry point: a kernel this op's *other*
(unconverted) factories also bind cannot be converted in place — it forks, per
[`pass_procedure.md`](../pass_procedure.md) scope rules and the port recipe's shared-kernel
caution. Note it; do not edit a shared kernel to suit one factory.

A kernel already in the explicit-args form with no convertible `#ifdef` is a legitimate
**no-op** entry — record it and move on.

### Step 3 — Confirm with the invoker

Stop. Present, and get explicit sign-off on:

- **The sentinel set** — that it covers the factory/factories you are converting, and that no
  critical test is missing. A missed sentinel turns this whole pass into a silent false-green.
- **The scope** — which factory or factories this run converts, and which are deferred.
- **The `METAL2_STYLE_PLAN.md`** — in particular any flag you propose to promote to a CTA (a host
  change), any kernel you found a **hard stop** (conditional args, or a non-`uint32_t` registered
  arg), and any kernel only *partially* convertible (varargs, shared-kernel fork). These are the
  decisions worth a human glance before you spend a build.

This is the checkpoint that keeps a large structural pass from running off a wrong assumption. It
is cheap for the invoker (a read) and it is where a scope or test-coverage error gets caught before
it costs anything.

### Step 4 — Apply

**Work one program factory at a time, and commit it as a unit.** The factory — with the kernel entry
points it binds — is the pass's unit of work *and* of commit. Convert all of that factory's
convertible kernels, leave the hard-stop ones legacy, verify against the factory's sentinels
([Step 5](#step-5--re-verify)), then **commit the whole factory conversion as one commit** before
moving to the next factory. Do not interleave two factories, and do not commit half a factory — the
per-factory commit is the reviewable, revertible unit that [Step 5](#step-5--re-verify) leans on.
Give it a message that carries the outcome, e.g.:

```
metal2(named-args): <op>/<factory> — <n> kernels converted, <m> left legacy

Converted:  <kernel list>
Legacy:     <kernel list>  (conditional args / vararg / shared)
```

**Always rebuild before you test — but let the test, not the build, be the gate.** The standing rule:
any time you change a C++ file (a kernel *or* factory host code), run `./build_metal.sh --build-all`
from the repo root before running any test. Know *why* the test is still the real gate: device kernels
are JIT-compiled from source at *test* time, so the build never compiles them — a signature/schema
mismatch ([Rule 5](#rule-5--conditional-args-are-a-hard-stop)) surfaces only when a test JIT-compiles
the kernel (the error names the offending entry, so converting a factory's kernels together still
localizes the failure). So: `--build-all`, then run the factory's sentinels — in the background,
reading the log, one pytest at a time.

Change nothing outside the factory and its kernels; the three scope limits of
[`pass_procedure.md` Step 3](../pass_procedure.md#step-3--apply) hold in full, as does the ban on
`experimental/quasar/` as evidence.

### Step 5 — Re-verify

Per [`pass_procedure.md` Step 4](../pass_procedure.md#step-4--re-verify). Rebuild with
`./build_metal.sh --build-all`, run the **same** sentinel set, every one green. On a regression:
**revert, do not patch forward** — the change is
behaviour-preserving, so a red test means the transformation was misapplied (a renamed arg reaching
the wrong slot is the classic culprit — [Rule 2](#rule-2--the-arg-name-contract)), not that it
needs tuning.

**Revert by committing the attempt, then a separate revert commit — never by discarding the
working tree.** A red sentinel does not always mean the conversion is unsalvageable; the root cause
may be fixable once understood. So preserve the attempt in history where a follow-up can lift it back
out, instead of throwing it away and forcing someone to re-derive it:

1. **Commit the failing factory conversion as-is**, labelled so its state is unmistakable — e.g.
   `WIP: <op>/<factory> named-args conversion — REGRESSES <sentinel>; reverted in next commit`. This
   captures the whole factory attempt as one reviewable unit.
2. **Revert it in the very next commit** — `git revert <that-commit>` — so the tree is green again and
   the attempt-and-its-revert sit adjacent in the log.

The pass's end state is still green (the revert is on top). But the attempt is now recoverable: a
follow-up that finds the fix runs `git revert <the-revert-commit>` (or cherry-picks the attempt) to
bring the conversion back, then fixes it forward — no re-typing. Record the regressing sentinel and
your hypothesis in the report, and carry on to the next factory. **Do not** leave the failing commit
as the tip, and **do not** fold the fix into the attempt here — a fix is behaviour-changing work for a
separate change ([Prime directive](#prime-directive--behavior-preservation-is-paramount)).

### Step 6 — Report

Because this pass commits per factory ([Step 4](#step-4--apply)), it follows `pass_procedure.md`'s
**committed** report shape rather than the leave-it-uncommitted single-pass default: write
`METAL2_POSTPORT_REPORT.md` in the op directory and commit it (with `METAL2_STYLE_PLAN.md`) alongside
the factory commits. Beyond the standard outcome/sites/verification, record **per factory**: kernels
converted vs left legacy (and why — conditional args / vararg / shared), any flag promoted to a CTA,
and which `#ifdef`s were left in place (conditional-resource gates).

---

## Transformation rules

### Rule 1 — The signature

Delete `void kernel_main()`. Declare the entry as a `TT_KERNEL` template: **CTAs in the template
list, RTAs/CRTAs in the function list, all `uint32_t`.** Include
`#include "experimental/kernel_args.h"` (it defines `TT_KERNEL`); it is usually already there.

```cpp
// before
void kernel_main() {
    constexpr std::uint32_t in0_t = get_arg(args::in0_t);   // CTA
    const std::uint32_t NCHt = get_arg(args::num_rows);     // RTA
    const std::uint32_t Wt   = get_arg(args::Wt);           // RTA
    // ... body ...
}

// after
template <std::uint32_t in0_t>                              // CTAs → template params
TT_KERNEL void compute(std::uint32_t num_rows, std::uint32_t Wt) {  // RTAs/CRTAs → fn params
    const std::uint32_t NCHt = num_rows;                    // keep local names if the body uses them
    // ... body unchanged ...
}
```

The entry may have **any name** (`compute`, `reader`, `writer`). It must be preceded by
`TT_KERNEL` and return `void`. Do **not** write your own `kernel_main()` — the shim is generated
and a hand-written one collides with it.

**List every registered CTA, even one the body never reads.** Because `check_name_sets` is two-sided
([Rule 5](#rule-5--conditional-args-are-a-hard-stop)), a CTA the host registered must appear in the
template head whether or not the body uses it — it becomes a harmless *dead* template parameter. Do
not drop it to tidy the signature; omitting a registered arg fails the two-sided schema check at test
time. (A `get_arg` port can quietly read-and-drop an unused arg; the signature form makes the dead
arg explicit — that is a feature, not a reason to prune it.)

**A kernel with no CTAs has no template head** — it is a plain `TT_KERNEL void entry(…)`:

```cpp
// zero CTAs → non-template TT_KERNEL entry (the parser accepts a bare function)
TT_KERNEL void compute(std::uint32_t num_rows, std::uint32_t num_tiles, std::uint32_t block) { /* ... */ }
```

Don't invent a dummy template parameter to force the template form — a bare `TT_KERNEL void entry(…)`
is exactly what the parser expects when the kernel has no CTAs.

### Rule 2 — The arg-name contract

The generated shim emits `entry<get_arg(args::P)…>(get_arg(args::Q)…)` using your **parameter
names**. So a parameter name must **exactly equal the arg name the host registered**
(`compile_time_args` keys; `runtime_arg_schema.runtime_arg_names`). A mismatch either fails to
compile (`args::wrongname` undeclared) or, worse, silently binds a different arg. Where the legacy
body used a *different local name* than the registered arg (e.g. `NCHt` for `args::num_rows`),
name the **parameter** after the arg and re-bind the local, as above.

Also: **a name in the template list must be a host CTA, and a name in the function list must be a
host RTA/CRTA.** `get_arg` resolves by how the host registered the name; a CTA placed in the
function list (or vice-versa) is a compile error. The kernel cannot and need not distinguish RTA
from CRTA — **preserve the host's existing split; never infer it from the kernel.**

### Rule 3 — Flags

A flag is convertible **only if every branch it gates — anywhere in this kernel — names only
always-present symbols.** Then, and only then:

- Promote it to a `uint32_t` NTTP template parameter, move it from `compiler_options.defines` to
  `compile_time_args` on the host, and branch with `if constexpr (FLAG)`.

```cpp
// before                                      // after
#ifdef UNTILIZE                                template <std::uint32_t UNTILIZE, /* ... */>
    /* uses only always-present names */       TT_KERNEL void compute(/* ... */) {
#else                                              if constexpr (UNTILIZE) { /* same names */ }
    /* ditto */                                    else                    { /* ditto */ }
#endif                                         }
```

**If the flag gates *any* conditionally-bound resource — anywhere in the kernel — leave every one of
its `#ifdef`s exactly as the basic port wrote them, and leave the `#define` in place.** Do not derive
a `constexpr bool`, do not convert its other branches, do not promote it. A flag half-expressed as an
NTTP and half as a `#define`, or a block that flips between `if constexpr` and `#ifdef`, is more
confusing than the preprocessor it replaced — see
[Rule 4](#rule-4--conditionally-bound-resources-leave-the-ifdef).

Expect this second case to be the common one: when a flag gates a conditional resource anywhere in the
kernel it stays on `#ifdef`, and the signature change ([Rule 1](#rule-1--the-signature)) is the real
deliverable. The worked example — an op where *every* flag fell here — is in the
[field notes](#field-notes-softmax-2026-08-18).

### Rule 4 — Conditionally-bound resources: leave the `#ifdef`

A `dfb::`/`tensor::`/`sem::` token the host binds only on some path is a **missing name** off-path.
There is no sound way to make it an `if constexpr`, so **the basic port's `#ifdef` is the end-state —
keep it verbatim.** Concretely, leave a block like this exactly as you found it:

```cpp
#ifdef FUSED_SCALE_MASK
constexpr auto dfb_fused_attn = dfb::fused_attn;   // token exists only on the fused path
// ... fused algorithm, using dfb_fused_attn ...
#endif
```

**Do not** rewrite it into a fallback-alias island (`#else constexpr uint32_t dfb_fused_attn = 0;`)
plus an `if constexpr (fused)` body. That compiles, but it replaces a hard compile error (touching the
token off-path) with a silent wrong-id read the moment a use escapes the guard — a bad trade in a pass
whose whole value is a diff a reviewer can trust. `if constexpr` cannot gate a missing name — that is
settled ([the boundary above](#the-one-thing-this-syntax-does-not-fix--read-before-you-plan)) — and the
real fix is the framework's, not the porter's (unbound accessors,
[issue #52179](https://github.com/tenstorrent/tt-metal/issues/52179)).

Two clarifications, because they are easy to trip on:

- **A pre-existing natural fallback is not yours to change either.** The basic port may already write
  `#ifdef NUMERIC_STABLE … #else constexpr auto dfb_x = dfb_exps; #endif`, aliasing a real,
  always-bound buffer. That is the basic port's code — leave it as it is. (This is *not* the
  fallback-alias trick forbidden above: you are not adding it to enable an `if constexpr`; it was
  already there, and you keep the `#ifdef` around it.)
- **An *unconditionally*-bound token never needed a token-existence `#ifdef`.** If a `#ifdef FLAG` gates code that
  reaches an always-bound token, the token is not the reason for the fence — the flag is — and whether
  that block converts is decided by [Rule 3](#rule-3--flags) (does *every* name in the branch always
  exist?), not here. A flag can guard an always-bound token's *use* yet still stay `#ifdef` because the
  same flag co-gates a *conditional* token elsewhere in the kernel.

### Rule 5 — Conditional args are a hard stop

**A kernel whose factory registers *any* CTA / RTA / CRTA conditionally cannot be converted.** Leave
it as legacy `void kernel_main()` and report it. This is a stop, not a workaround-able case — there
is no "keep the conditional reads manual" escape.

*Why:* the JIT validates the parsed signature against
the **per-build** registered args and demands **exact two-sided equality** — every parameter must
have a registered arg, *and* every registered arg must be a parameter
(`validate_signature_against_schema` → `check_name_sets`, in
`tt_metal/jit_build/kernel_signature_parser.cpp`). A signature is one fixed source; if the factory
registers a different arg set per build, no single signature can equal all of them:

- **Omit the conditional args from the signature** → the build that *does* register them fails:
  `TT_KERNEL entry '<entry>': ... registered runtime argument(s) not taken as a function parameter:
  <the conditionally-registered args>`.
- **`#ifdef`-gate the extra params inside the signature** → fails the *other* way, because the
  signature parser reads **raw, unpreprocessed** source: it blanks only the `#ifdef`/`#endif`
  directive lines and keeps the parameters between them, so it extracts the *union* and the build
  that does *not* register them reports `function parameter(s) with no matching registered runtime
  argument`.

The only way to convert such a kernel is to make the factory register the args **unconditionally** —
a host **dispatch change** (extra runtime args on paths that don't use them), which is a semantic
change out of scope for this behaviour-preserving pass, and mildly wasteful. Do not do it here;
report the kernel as blocked on it.

> **Conditional *args* ≠ conditional *tokens*.** A conditionally-**bound DFB/tensor** (a `dfb::`
> token absent off-path) does **not** block the conversion: its `#ifdef` simply stays
> ([Rule 4](#rule-4--conditionally-bound-resources-leave-the-ifdef)) and the kernel's *signature*
> still converts. A conditionally-**registered arg** is the case that blocks the whole kernel.
> (A kernel with conditional tokens but unconditional args converts, signature only; a kernel with
> any conditional arg stops.) Get the split from the
> **factory's** `compile_time_args` / `runtime_arg_names` registration (look for `push_back` /
> `insert` behind a host `if`), never from the top of `kernel_main`.

### Rule 6 — `#ifdef` → `if constexpr` decision

For each flag, one question decides it: **does any branch it gates — anywhere in the kernel — name a
conditionally-bound token or a conditional arg?**

- **No** — every name is always present → promote the flag to an NTTP and convert its blocks to
  `if constexpr (FLAG)` ([Rule 3](#rule-3--flags)).
- **Yes** — leave every one of that flag's `#ifdef`s exactly as they are
  ([Rule 4](#rule-4--conditionally-bound-resources-leave-the-ifdef)).

Decide per **flag**, not per block, so the kernel keeps one gating style per flag rather than a mix.
Because most flags on a real op touch a conditional resource somewhere, the answer is usually
"leave it," and the signature conversion ([Rule 1](#rule-1--the-signature)) is the real win.

### Rule 7 — Host side stays minimal

The point of keeping conditional-resource flags as `#define`s is that the factory barely changes. For
the common case (every flag touches a conditional resource), **CTA/RTA registration and conditional
bindings are unchanged** — the arg *names* simply now double as parameter names. The only host edit is
promoting a **pure-value** flag to a CTA, if you found one; otherwise nothing. Do not reorder or
re-kind args to "tidy" the signature — that is a semantic change to the dispatch, out of scope for
this pass.

### Rule 8 — Comments: preserve

Per the basic recipe's hard-won rule: **a previous porter deleted blocks of load-bearing comments
while adding `#ifdef`s — do not repeat that in reverse while removing them.** When an `#ifdef`
becomes an `if constexpr`, the comment explaining *why* the branch exists still applies; carry it.
Only a comment your change made *untrue* (one that describes the preprocessor mechanics you just
replaced) is edited, and it is reported as a repair per
[`pass_procedure.md` Step 3](../pass_procedure.md#step-3--apply).

---

## Pitfalls and mitigations

| Pitfall | Symptom | Mitigation |
|---|---|---|
| A registered arg is non-`uint32_t` | the parser rejects the param, and it can't be omitted either | **Hard stop.** The signature can neither express it (`uint32_t`/`std::uint32_t` only, [Rule 1](#rule-1--the-signature)) nor leave it out (`check_name_sets` exact equality, [Rule 5](#rule-5--conditional-args-are-a-hard-stop)). Leave the kernel legacy. (Named args are all `uint32_t` today, so this is rare.) |
| Vararg kernel | no signature spelling exists | `get_vararg` args have no parameter form. Convert the named args; leave vararg reads as-is; note "partial" in the report. (Tested on permute — see [field notes](#field-notes-permute-2026-08-20).) |
| Parameter renamed off the registered arg | undeclared `args::…`, or a silent wrong value | [Rule 2](#rule-2--the-arg-name-contract): parameter name ≡ registered arg name; re-bind the legacy local instead. |
| Kernel has **any** conditionally-registered CTA/RTA/CRTA | JIT `check_name_sets` throws at *test* time (registered arg not a param, or param not registered) | [Rule 5](#rule-5--conditional-args-are-a-hard-stop): the kernel is a **hard stop**, not convertible. Detect from the factory in Step 2. |
| Tried to `#ifdef`-gate signature params | fails off-path: `function parameter(s) with no matching registered runtime argument` | The signature parser reads **raw** source and extracts the *union* of params; no `#ifdef`-varying signature is possible. |
| Trusting a green build to mean the kernel is fine | it isn't — kernels JIT-compile at test time, so a signature/schema mismatch surfaces only when the test runs | Always `./build_metal.sh --build-all` before tests ([Step 4](#step-4--apply)), then treat the **test run** as the validation, not the green build. |
| A clean conversion is silent in the test log | a *passing* test echoes nothing about the kernels it JIT-compiled, so the log can't confirm the converted kernel ran | Compile command lines appear only on warning/error; to prove the kernel recompiled, check for a fresh `.o` in the JIT cache (mtime within the run). Silence ≠ not exercised. |
| Converted a conditional-resource `#ifdef` (fallback-alias island) | silent wrong-id if a use escapes the guard; an unreviewable "clever" diff | [Rule 4](#rule-4--conditionally-bound-resources-leave-the-ifdef): leave conditional-resource `#ifdef`s untouched — convert the signature only. |
| Promoting a flag that gates a conditional resource | loses the `#define` the `#ifdef` still needs, and there's nothing to convert to anyway | Promote *only* pure-value flags (every gated name always present); leave the rest as `#define` + `#ifdef` ([Rule 3](#rule-3--flags)). |
| Hand-written `kernel_main()` left in | collides with the generated shim | Delete it; the `TT_KERNEL` entry is the whole kernel. |
| CRTA inferred from the kernel | wrong dispatch kind | The kernel can't see RTA vs CRTA; preserve the host split ([Rule 2](#rule-2--the-arg-name-contract)). |
| Shared entry point converted in place | breaks the op's other (unported) factories | Fork per the shared-kernel caution; never edit a shared kernel to suit one factory. |
| Always-binding a conditional DFB to dodge the `#ifdef` | wasted L1; the anti-pattern the basic recipe forbids | Keep the binding conditional; keep the `#ifdef`. The real fix is unbound accessors (#52179), not here. |

---

## When to stop

Per [`pass_procedure.md` — When the fix doesn't fit](../pass_procedure.md#when-the-fix-doesnt-fit),
and per the [Prime directive](#prime-directive--behavior-preservation-is-paramount) above. A grounded
stop is a complete, success-tier outcome — **warn the invoker compactly and explicitly: the site
(`file:line`), a short snippet of what does not fit, and one sentence on why the minimal change
cannot cover it.** Specifically here:

- **The factory registers any arg conditionally** — the whole kernel is unconvertible
  ([Rule 5](#rule-5--conditional-args-are-a-hard-stop)). Leave it legacy, report it, move on. This
  is the most common stop, and it is a per-*kernel* stop, not a per-*factory* one: convert the
  kernels whose args are all unconditional and leave the rest legacy.
- **A registered arg is non-`uint32_t`** — the signature can neither express it nor omit it
  ([Rule 5](#rule-5--conditional-args-are-a-hard-stop)), so the whole kernel is a **hard stop**.
  (Rare: named args are all `uint32_t` today.)
- A kernel reaches arguments through **varargs** (`get_vararg`), which have no signature spelling —
  convert the named args, leave the vararg reads manual, and report the kernel as **partially
  convertible**.
- The transformation would require touching a **shared kernel** in place, or any file **outside
  the op directory**.
- Anything that pushes you toward changing *what the kernel computes or binds* rather than *how it
  spells its arguments and gates its branches*. This pass changes spelling and gating; if the
  behaviour has to move for the conversion to work, the kernel is doing something this recipe did
  not anticipate — report it in full, because that is what improves the recipe.

---

## Field notes (softmax, 2026-08-18)

First real run: the attention-optimized **interleaved** factory. It converted the writer + both
compute kernels and left the two readers legacy (conditional args — [Rule 5](#rule-5--conditional-args-are-a-hard-stop)),
green across every flag combination. **That run also converted the token-gated control flow to
`if constexpr` using fallback-alias islands — the technique
[Rule 4](#rule-4--conditionally-bound-resources-leave-the-ifdef) now forbids.** The islands were
numerically sound (the tests passed), but the trade — a silent wrong-id where a hard compile error
used to be, on an op where nearly every flag touches a conditional token — was judged not worth it.
Under this recipe, that factory's compute kernels convert for their **signature only**; the
`FUSED_SCALE_MASK` / `NUMERIC_STABLE` / `MASK_PADDED_DATA` / `CAUSAL_MASK` `#ifdef`s stay. A mixed
factory (converted kernels beside legacy `kernel_main` kernels) is valid — each kernel is parsed
independently.

What the run established that still holds:

- **Conditional args are a hard stop** ([Rule 5](#rule-5--conditional-args-are-a-hard-stop)). The
  readers register the mask/causal args conditionally, so no fixed signature satisfies the JIT's
  `check_name_sets` on both the mask and no-mask paths.
- **A no-CTA kernel is a non-template `TT_KERNEL`** — a bare `TT_KERNEL void entry(…)`
  ([Rule 1](#rule-1--the-signature)). This is also why the fallback-alias trick was *extra* unsafe in
  the large compute: outside a template, `if constexpr` fully type-checks the dead branch — one more
  reason [Rule 4](#rule-4--conditionally-bound-resources-leave-the-ifdef) drops it.
- **A green `build_metal.sh` does not validate kernels** — they JIT-compile at test time, so only a
  test run catches a kernel error (you still `--build-all` before testing; [Step 4](#step-4--apply)).
- **An unconditionally-bound DFB never needed a `#ifdef`** — softmax's `mask_padded` is bound
  unconditionally even though its `MASK_PADDED_DATA` define is conditional (the define co-gates the
  conditional `dfb::x`).
- **`Rule 2` name-matching is enforced twice** — by the shim codegen *and* by `check_name_sets`;
  `std::uint32_t` and `uint32_t` both parse.

Append later runs below this line.

## Field notes (permute, 2026-08-20)

Second run, and the first that **tested** the vararg path: the tiled interleaved factory
(`permute_tiled_program_factory.cpp`). Baseline **1605 passed → post-port 1605 passed**
(`test_permute.py`), the converted kernels confirmed freshly JIT-compiled. Then rolled back — it was
an experiment to ground this recipe, not a landed conversion.

What it established (now tested, not just argued):

- **Varargs "partial" is correct.** The invariant/generic readers keep their `get_vararg(i …)` reads
  verbatim while their named CTAs → template params and RTAs → fn params; it builds and passes. The
  JIT ignores varargs because `check_name_sets` compares only the two *named* sets
  (template ↔ `compile_time_args`, params ↔ `runtime_arg_names ∪ crta`), varargs are carried as a
  *count* (`advanced_options.num_runtime_varargs`) and never as a name, and the signature parser reads
  the signature and never the body — so the `get_vararg(...)` calls are invisible to it. This is the
  concrete vararg example softmax lacked.
- **The conditional-padding writers are Rule 5 hard stops.** Both tiled writers `push_back` their
  `start/end_padding_tile_idx` RTAs behind a host `if (needs_*_padding)`, so the registered set differs
  per build and no single signature satisfies the two-sided check. Left legacy —
  [Rule 5](#rule-5--conditional-args-are-a-hard-stop) confirmed on a second op.
- **One flag, two kinds, decided independently — the per-flag rule holds.** In the generic reader,
  `needs_x_padding` is a **CTA** gating only arithmetic → promoted to an NTTP + `if constexpr` (pure
  value, [Rule 3](#rule-3--flags)); `NEEDS_Y_PADDING` is a **`#define`** gating the conditional
  `dfb::cb_pad` → `#ifdef` left verbatim ([Rule 4](#rule-4--conditionally-bound-resources-leave-the-ifdef)).
  Two different flags, resolved separately — exactly the per-flag discipline of
  [Rule 6](#rule-6--ifdef--if-constexpr-decision).

Three things this recipe did not warn about, worth carrying:

- **A "factory" may be several factory *methods* in one file, and some kernels it binds are donor
  kernels *outside* the op directory.** `permute_tiled_program_factory.cpp` holds three methods, and
  two of the tiled kernels are shared from `transpose/` and `eltwise/unary/`. Those out-of-dir donors
  are untouchable by the scope rule ([Step 4](#step-4--apply)) and the shared-kernel caution — convert
  the in-dir kernels, leave the donors. Expect a factory's bound-kernel list to include kernels you
  must *not* edit.
- **An unused named CTA still must be listed as a template param** — the invariant reader registers
  `page_size`/`num_tiles` but never reads them, yet the two-sided `check_name_sets` forbids dropping
  them; they become harmless "dead" parameters ([Rule 1](#rule-1--the-signature)).
- **A clean conversion is *silent* in the test log.** JIT compile command lines are echoed only on
  warning/error, so a passing conversion prints nothing about its kernels even though they ran.
  Proving the converted kernel was exercised means checking for a fresh `.o` in the JIT cache (mtime
  within the run), not grepping the pytest log — don't misread the silence as "not compiled."

<!-- FINDINGS: append pitfalls/patterns from later conversions below this line. -->
