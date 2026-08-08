# Post-Port Fix — Add the Gen2 hardware configs

> **Procedure:** [`pass_procedure.md`](../pass_procedure.md). Read it first; it is the *how*. This
> file is the *what*, and it **overrides Step 4** — see [What your verification can and cannot
> see](#what-your-verification-can-and-cannot-see).
>
> **Behaviour-preserving?** On Gen1, yes — the Gen1 path must come out of this pass textually
> unchanged. On Gen2, no: this pass is what gives the op a Gen2 configuration it did not have.

---

## What this fix is

A `KernelSpec`'s `hw_config` is a variant that holds **exactly one generation's** config. The
Metal 2.0 port targeted Gen1 and was explicitly told not to author Gen2 configs, so a ported op's
custom configs hold a `DataMovementGen1Config` or a `ComputeGen1Config` and nothing else.

This pass adds the missing alternative, selected on the device's architecture at runtime.

**Why it matters.** A kernel whose `hw_config` carries only a Gen1 config cannot run on Quasar at
all. This is not a polish pass — it is the difference between an op that exists on the next
generation and one that does not, and it is far cheaper to do now, with the Gen1 values in front of
you, than during a Quasar bring-up where someone has to reconstruct the intent.

## What your verification can and cannot see

This is the one place this fix departs from the standard procedure, and you should hold it clearly
rather than assume the usual safety net is under you.

- **Your build does check the Gen2 config.** The branch you add is ordinary compiled code, so a
  wrong field name, a wrong type, a field that doesn't exist on the Gen2 struct — all of that fails
  the build like anything else.
- **Your sentinels do check Gen1.** They run on Gen1 hardware and exercise the Gen1 branch, so
  they will catch a structural break in the path you must not disturb.
- **Nothing checks the Gen2 *values*.** That branch is never taken on the bench you are running on.
  A Gen2 config that compiles and is wrong looks exactly like one that compiles and is right.

So the correctness of this pass does not come from measurement. It comes from the field
dispositions below being **exhaustive and mechanical** — every field has a prescribed outcome, and
there is no field on which you are asked to exercise judgement. If you find yourself deciding
rather than transcribing, that is the signal to stop.

## Step 2 — Survey

Find every `hw_config` in the op's program factory (or factories), **and every place a field is set
on one after the fact**. That second half is the one that is easy to miss and the one that matters
most:

```bash
grep -rn "hw_config\|to_compute_hardware_config\|Gen1Config\|std::get<\|std::get_if<\|holds_alternative" <op-dir>
```

Two of those terms close specific escapes, and dropping them costs you sites. `Gen1Config` matches
shape 4's own definition directly, catching a hand-written config built into a local and passed
onward through a parameter not named `hw_config` — do not rely on the assignment line being spelled
the way you expect. `std::get_if<` catches the shape-3 variant described below; note that
`std::get<` does **not** match it.

Each result is one of four shapes.

**1 — DM config from an arch-agnostic helper.** `ttnn::create_reader_datamovement_config(arch)` or
`…create_writer_datamovement_config(arch)`. These select the right alternative internally and set
everything there is to set. **No work; do not touch them.**

**2 — Compute config from `to_compute_hardware_config(arch, …)`, with nothing set afterwards.**
Also complete. **No work.**

**3 — Compute config from the helper, with fields then set on the returned variant.** This is a
site, it is the most common one, and it is the most severe. The helper is arch-agnostic but
deliberately **does not set `unpack_modes`** (nor `bfp_pack_precision_mode`), leaving both to the
factory — so a factory that needs either reaches into the returned variant to set it:

```cpp
std::get<ComputeGen1Config>(compute_hw_config).unpack_modes = { … };
```

On Gen1 that is correct. On Quasar the helper returned the **Gen2** alternative, so this `std::get`
throws `std::bad_variant_access`. The op does not merely run slowly on Quasar — it does not run.

Some ops have already noticed and guard it, funnelling the access through a wrapper that
`TT_FATAL`s on `std::holds_alternative` with a "this op is Gen1-only" message. **A guarded site is
still a site**, and still needs converting; the guard only replaces a crash with a legible refusal.

**The `std::get_if` variant is the same site with a quieter failure**, and it is easy to wave past
because it looks like it already handles both generations:

```cpp
if (auto* gen1 = std::get_if<ComputeGen1Config>(&compute_hw)) { gen1->unpack_modes = { … }; }
```

On Quasar that pointer is null, the block does not run, and nothing throws — so the field the
factory meant to set is simply never set, and the op proceeds with whatever the helper's default
was. That is worse than the `std::get` form to find and no better to have: a crash names itself,
whereas this surfaces later as wrong numerics, a validation failure far from the cause, or a silent
pessimization. **Convert it the same way**, and note in your report that it was the `get_if` form,
since its absence of a Quasar crash is why nobody has noticed it yet.

**4 — A `DataMovementGen1Config` or `ComputeGen1Config` written by hand.** Needs a Gen2 branch. The
compute case here is the op that avoided the helper deliberately — usually because the helper would
have forwarded a resolved field the legacy op never applied.

Shapes 3 and 4 are the sites. An op built entirely on shapes 1 and 2 is a legitimate zero-site
pass.

## Step 3 — Apply

> **Reaching for a worked example?** The field dispositions below are the specification — prefer
> them to any op you find. `ttnn/cpp/ttnn/operations/experimental/quasar/` is **out of bounds** and
> is not evidence of anything, however authoritative it looks; see [the
> procedure](../pass_procedure.md#step-3--apply).

### Shape 3: helper-built compute config with fields set afterwards

The helper already picked the right alternative. What is Gen1-only is the **`std::get`** that
follows it — and the fix is to stop naming a generation at all.

Every field common to both generations has an accessor in `compute_hardware_config.hpp` that
resolves the alternative for you. Swap the `std::get` for it and the site is done:

```cpp
auto compute_hw = ttnn::to_compute_hardware_config(arch, compute_kernel_config);

// TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
m2::unpack_modes(compute_hw) = {      // exactly the entries that were there before
    {INPUT_DFB, UnpackMode::UnpackToSrc},
    …
};
```

The accessors return references, so in-place mutation converts the same way —
`std::get<m2::ComputeGen1Config>(cfg).unpack_modes.emplace(dfb, mode)` becomes
`m2::unpack_modes(cfg).emplace(dfb, mode)`.

There is **no `arch` branch and no arch parameter** in this shape. The helper already used `arch`
to choose the alternative; the accessor reaches whichever one it chose. If you find yourself
introducing an architecture check here, you have the wrong transformation.

The available accessors are `fpu_math_fidelity`, `sfpu_precision_mode`, `enable_32_bit_dest`,
`double_buffer_dest`, and `unpack_modes` — the five fields common to both generations. A
generation-specific field has none by design: `bfp_pack_precision_mode` exists only on Gen1 and
`enable_2x_src_register` only on Gen2, so reaching either one *should* make you name the
generation. **If these accessors are not in your checkout, stop and report** — the recipe depends
on them and hand-rolling a substitute is how the original problem got here.

**The entries are copied unchanged.** Same buffers, same modes, same order. The Quasar-optimal
answer is very likely different, and that is #52269's business, not yours.

**`bfp_pack_precision_mode` is the exception.** It exists only on Gen1, so it keeps its
`std::get<m2::ComputeGen1Config>` — but that `std::get` still throws on Quasar, so it has to become
conditional rather than unconditional. It is rare; if you find one, leave it as the only
arch-guarded statement in the shape and report it as a site the accessors could not simplify.

**If the op guards the access** behind a `TT_FATAL`-on-`holds_alternative` wrapper, converting every
site through the accessors makes that guard assert something untrue — it claims the op is Gen1-only
when it no longer is. Convert **every** call site first, then remove the wrapper. If any site can't
be converted, leave the guard alone, convert nothing, and report — a half-converted op carrying a
stale "Gen1-only" guard is worse than either end state.

#### What else shape 3's conversion takes with it

The conversion removes exactly the variant handling its own transformation makes unnecessary, and
no more. These are in scope, because leaving them *is* the bug:

- **Every `std::get<ComputeGen1Config>` that reaches a common field** — whether it writes or merely
  reads. A read throws on Quasar exactly as a write does.
- **A guard whose claim the pass falsifies**, per above.
- **The parameter type of a helper that takes a `ComputeGen1Config&`**, widened to
  `ComputeHardwareConfig&` so it works on either generation. Change the signature and leave the
  helper otherwise alone — do not delete an abstraction the op's author chose just because its body
  got shorter, and do not restyle it.

For what stays untouched, see [What not to tidy, in any shape](#what-not-to-tidy-in-any-shape) —
that list governs shape 4 as well, so it lives after both shapes rather than here.

### Shape 4, data movement: default-construct

`DataMovementGen2Config` has **no field in common** with the Gen1 config. Gen1 carries `processor`,
`noc`, `noc_mode` — a placement onto specific RISC cores and NOCs. Gen2 has no such concept, so
there is nothing to map and nothing to decide: every custom DM config, however elaborate its Gen1
side, gets a **default-constructed** `DataMovementGen2Config{}`. This is what the arch-agnostic
helper does for the default reader and writer too, so you are matching existing behaviour, not
inventing it.

**Do not set `disable_dfb_implicit_sync_for_all`, or its per-buffer sibling
`disable_dfb_implicit_sync_for`.** Both are opt-outs from implicit sync, which should be **on** by
default on Quasar. Ops have disabled it in places to work around a temporary bug; that is a
workaround, and this pass must not turn it into policy by propagating it. If you encounter one set
`true` elsewhere in the op, it is not a template — leave it, and note it in your report.

### Shape 4, compute: copy the corresponding fields

`ComputeGen2Config` shares four fields with Gen1, drops one, and adds one. The dispositions are
complete — this table covers every field on both structs:

| `ComputeGen1Config` field | Gen2 disposition |
|---|---|
| `fpu_math_fidelity` | copy verbatim |
| `sfpu_precision_mode` | copy verbatim |
| `enable_32_bit_dest` | copy verbatim |
| `double_buffer_dest` | copy verbatim |
| `bfp_pack_precision_mode` | **drop** — no Gen2 equivalent; Gen2 replaces BFP formats with MXFP |
| `unpack_modes` | copy verbatim, **and add the marker below** |
| *(Gen2-only)* `enable_2x_src_register` | **leave at its default. Never set it.** |

Copy only the fields the Gen1 config actually sets. An unset field is already at a default that
matches, so writing it out explicitly adds noise without adding meaning.

`enable_2x_src_register` deserves its emphasis: it is an interim MXFP4-only setting that produces
**garbage math results** for any instruction outside the matmul and column-reduce families. It is
not yours to enable, and nothing about this pass calls for it.

### The `unpack_modes` marker

`unpack_modes` is the one field whose *optimum inverts between generations*. On Gen1, `UnpackToSrc`
is both the default and the fastest option, so a Gen1 config either omits the table or fills it
with Gen1-optimal choices. On Gen2 there is no penalty for unpacking straight to Dest, so
`UnpackToDest` is preferred for any SFPU-consumed buffer.

Copy the Gen1 values verbatim anyway — a faithful copy is correct and safe. But it is very likely
*slower than it needs to be on Quasar*, and that pessimization has **no code footprint**: the
Gen1-faithful answer is often an absent table entry, indistinguishable from a considered one.
Nothing fails, nothing looks wrong, and no reviewer sees anything.

Deciding it properly means knowing which buffers are SFPU-consumed and how that interacts with
`enable_32_bit_dest` — analysis this pass does not equip you for, and correctly not your call. So
leave a marker for the humans who will do it:

```cpp
// TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
```

**Put it wherever this pass causes Quasar's `unpack_modes` to take a Gen1-derived value** — beside
the accessor assignment in shape 3, or on the Gen2 config you build in shape 4 *compute* — whether
or not you believe this op has any SFPU-consumed buffers. That belief is exactly the judgement we
just said you are not equipped to make, and a marker that is sometimes present and sometimes not is
ambiguous between "considered, not applicable" and "nobody looked." Unconditional means its absence
is meaningful.

**Unconditional within that scope, which is compute only.** `unpack_modes` is a compute field;
`DataMovementGen2Config` has no such field, and nothing in shape 4 *data movement* can cause Quasar
to take a Gen1-derived value for it. So a pass whose sites are all shape-4-DM carries no marker
anywhere, and that absence is not a signal — there was nothing for it to mark. Say so in your
report rather than leaving a future reader to infer it, and do not add a marker to a
`DataMovementGen2Config{}` to be safe: a marker on a config that has no `unpack_modes` field points
at nothing and devalues the ones that do.

### Getting `arch`, and where the branch goes

**Shape 4 only** — shape 3 needs no architecture check at all. Get the architecture the way the
op's factories already do — `operation_attributes.mesh_device->arch()` or `device->arch()`. Ops
already on the DM helper demonstrate the idiom.

Hoist it to a `const auto arch` local only if **the code you add** reads it more than once. The
count is over your own additions, not over the factory: an op may already call `device->arch()`
several times, and rewriting those call sites is out of scope. Hoisting a local that sits directly
above surviving `device->arch()` calls reads as half-finished and invites exactly the cleanup the
procedure forbids.

**One branch can cover several configs.** Where a factory has more than one custom DM config — a
reader and a writer, typically — put them in a single `if`, rather than giving each its own branch
and its own hoisted local. That is usually the difference between needing a local and not.

**Prefer the form that leaves the existing Gen1 initializer textually untouched:**

```cpp
m2::ComputeHardwareConfig compute_hw = compute_gen1;   // the existing, unmodified Gen1 config
if (arch == tt::ARCH::QUASAR) {
    // TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
    compute_hw = m2::ComputeGen2Config{ /* fields per the table */ };
}
```

If the Gen1 config is currently written inline inside a `KernelSpec`'s designated initializers,
hoist it to a local first — **copying the initializer verbatim, not retyping it** — and then add
the branch. Hoisting moves lines; retyping invites a transcription error into the one path your
tests actually cover.

When writing the new braced initializer, end the last field with a trailing comma. Without it,
clang-format aligns the whole list to the opening brace instead of block-indenting it, and the
result is unreadable and churns the diff.

### What not to tidy, in any shape

Reaching a config through a variant is ugly, and this pass makes some of that ugliness go away. That
is not a licence to remove the rest. **This applies to whichever shape you are working on** — shape
4 included, where the temptation arrives precisely because you already have the config open. Note
these in your report and leave the code as it is:

- **`std::get<m2::ComputeHardwareConfig>(kernel_spec.hw_config)` — the *outer* variant.** It looks
  like the same grossness and it isn't. That variant distinguishes a compute config from a data
  movement one, so a mismatch is a programming error rather than something the architecture
  decides, and it does not throw on Quasar. Leave it.
- **`bfp_pack_precision_mode`**, which is Gen1-only and must keep naming its generation.
- Any other variant awkwardness you notice on the way past.

## The Gen1 path must come out unchanged

Review your own diff before you finish. **Every line of every Gen1 config must be untouched** —
same fields, same values, same order. Your sentinels cannot fully police this: a config change here
is a performance or precision shift, not a failure, so a passing test set is not evidence that the
Gen1 side survived.

One specific trap: **do not reroute a custom config through the arch-agnostic TTNN helper** in
order to get the Gen2 branch "for free." The helper's defaults are the high-performance ones, so
every field the custom config had set — and that you would no longer be passing — silently flips.
That is a real Gen1 behaviour change, invisible to your tests, disguised as a simplification. At
least one op in the tree carries a comment from a previous porter explaining that the helper is
*deliberately* not used for exactly this reason.

## When to stop

Per [When the fix doesn't fit](../pass_procedure.md#when-the-fix-doesnt-fit). Specifically here:

- A Gen1 compute config sets something the table above does not cover — the table is meant to be
  exhaustive, so a gap in it is a recipe defect worth reporting rather than a judgement call for
  you to resolve.
- The Gen1 config is computed rather than written literally — assembled through branches or helper
  functions such that "copy the fields" has no single site to copy from.
- You find yourself weighing whether a buffer is SFPU-consumed, or whether some Gen2 value would be
  better than the Gen1 one. That is #52269's job, not this pass's. Add the marker and move on.
