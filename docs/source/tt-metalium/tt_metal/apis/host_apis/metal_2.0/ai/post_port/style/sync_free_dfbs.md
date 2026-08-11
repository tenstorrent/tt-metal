# Post-Port Fix — Convert sync-free DFBs to their proper shapes

> **Procedure:** [`pass_procedure.md`](../pass_procedure.md). Read it first; it is the *how*. This
> file is the *what*, and it uses the procedure's steps unchanged.
>
> **Behaviour-preserving?** Yes. Results, numerics and observable behaviour are identical. L1
> allocation *order* shifts — scratchpads are allocated alongside DFBs, from the same region — but
> nothing functional depends on that ordering.
>
> **Target:** Gen1 (Wormhole / Blackhole) ops already ported to Metal 2.0.

---

## What this fix is

The legacy API gave you one on-core buffer construct — the CircularBuffer — so op authors used it
for everything that needed L1, whether or not it had anything to do with a FIFO. Those uses came
through the Metal 2.0 port unchanged, as DataflowBuffers, because the port is a mechanical CB→DFB
swap and correctly changes nothing else.

A DFB is a **software FIFO for handing data between a producer and a consumer**. When a buffer
never uses any of that machinery, calling it a DFB is a misdescription that the reader has to see
through. It is really one of two things, and Metal 2.0 has a construct for each:

- **A view onto memory the op already owns** → `LocalTensorAccessor`.
- **A private scratch region a kernel scribbles in** → `Scratchpad`.

This pass finds the misdescribed ones and gives them their right shape.

**Why it matters.** The primary reason is that the ported corpus is what everyone downstream learns
Metal 2.0 from, and a buffer declared as a FIFO that is not one teaches the API wrong. There is a
concrete secondary benefit: every DFB removed frees a DFB id, and the id budget is genuinely tight
— a recently discovered hardware bug constrains the number of available DFBs further still. For a
DM self-loop the conversion is also a Quasar prerequisite, since that shape does not survive to
Gen2 — but treat that as a bonus rather than the reason.

## The criterion

### The property

This pass turns on a semantic question: **does anything, anywhere, synchronize through this
buffer?** If nothing does, the buffer is not a FIFO and declaring it one misdescribes it. If
something does, converting it deletes that synchronization.

On Gen1, a DFB's synchronization lives in a pair of credit counters, and six methods touch them:

```
reserve_back   push_back   wait_front   pop_front         // block on the credits, or post them
pages_reservable_at_back   pages_available_at_front       // read them without blocking
```

**The second row is the one that gets missed.** Neither name contains any of the first four as a
substring, so a grep written from the first row alone does not see them — and they are not
hypothetical: ported matmul readers spin on `dfb_in1.pages_reservable_at_back(...)` today. Those two
do not block by themselves, but a kernel polling one is reading exactly the state a `wait_front`
would have blocked on, and that is synchronization however it is spelled.

Nothing else counts here. Implicit sync — NOC transfers exchanging credits without any FIFO call —
is supported only on Gen2 DFBs, and this pass runs on Gen1 ops recently migrated to Metal 2.0, so
every buffer you meet synchronizes explicitly or not at all.

So the property is: **no kernel that binds this DFB ever causes any of those six calls on it.**
*Causes*, not *contains* — that distinction is way 4 below. It is a property of the DFB, holding
across every binding kernel and every configuration the spec is built under; it is not per kernel
and not per file.

### The test

Gather evidence for the property by grepping each binding kernel:

```bash
grep -nE "reserve_back|push_back|wait_front|pop_front|pages_reservable|pages_available" <each binding kernel>
```

This is a good first move and it is **not** the property. It is a syntactic search standing in for a
semantic question, and the gap between the two is where every mistake in this pass has come from.
Use the grep to *find* synchronization; never read a clean result as proof there is none. The four
known divergences are below, and there is no reason to think that list is complete.

### When you cannot establish absence, it is not a site

The two ways to be wrong here cost very different amounts, so let your uncertainty fall on the cheap
side deliberately.

- Wrongly calling a DFB **synchronized** costs you a site. This pass is explicitly allowed to find
  zero sites, and the site stays there for whoever looks next.
- Wrongly calling a DFB **sync-free** deletes real synchronization, silently, in a way your
  sentinels are not guaranteed to catch.

The bar is therefore not *"I found no evidence of synchronization"* but *"I am confident there is
none."* If you are not, leave the DFB as it is and say so in your report. That is a complete and
correct outcome, and a useful one — it tells the people maintaining this recipe where the test ran
out, which is how the list below got written in the first place.

### Four ways the test diverges from the property

Each of these is a real pattern, and the first two appear *side by side in a single file*
(`moreh_fold`'s `reader_fold_rm.cpp`), which is worth reading before you start.

**1. A self-loop is not the same as sync-free.** A DFB bound `PRODUCER` *and* `CONSUMER` by one
kernel is a self-loop — that is a statement about endpoints, not about synchronization. A kernel
can absolutely run a full FIFO against itself: reserve space, fill it, push, wait, read, pop, as a
bounded staging buffer. `moreh_fold` binds its input DFB exactly that way and it calls every one of
the blocking four. **Not a site.** Self-loops are a good place to *look*, never a verdict.

**2. Neither pointer getter tells you anything — in either direction.** A sync-free DFB still needs
its base address, and since nothing ever advances either pointer, `get_read_ptr()` and
`get_write_ptr()` both return the base for the buffer's entire lifetime. So a sync-free kernel may
reasonably call either one, or both, and which it picked is arbitrary. Meanwhile a fully
synchronized DFB calls them too, between its reserve and its push, because that is ordinary FIFO
usage. The getters appear on both sides of the line and carry no information about it. **Do not
read a pattern into which getter a kernel used.** Only the absence of the six credit methods
decides.

The base address does not always arrive through a getter, either. A helper taking the DFB's id —
`get_pointer_to_cb_data<T>(dfb::recip, 0)`, as `layernorm_pre_all_gather_welford` uses — hands back
a pointer without either getter appearing. It carries exactly as little information as they do:
it is a base-address grab, not a synchronization signal.

**3. A DFB is only sync-free if it is sync-free *everywhere*.** Check every kernel that binds it.
A buffer that looks like an untouched scratch region in the kernel you are reading may be waited on
by a second kernel you have not opened. Enumerate the binding kernels from the host spec, not from
the kernel you happen to have in front of you.

*Everywhere* also spans **configurations**, not just kernels. One `DataflowBufferSpec` can be
sync-free scratch under one sharding and a genuine FIFO under another — conv2d's `ACT_TILIZED` is
the canonical example, sync-free when height-sharded and a real FIFO when block- or width-sharded.
So a factory that builds its bindings or its kernel set behind a branch has to be answered per
branch, and a DFB that is sync-free on only some paths is **not a site**. If the branch is chosen
host-side from something you cannot pin down, that is a case for [the safe
default](#when-you-cannot-establish-absence-it-is-not-a-site).

**4. A grep for those methods can come back clean on a synchronized DFB.** The calls do not have
to appear in the kernel you are reading — a helper can make them on its behalf, and then the buffer
reads as untouched. Two real shapes:

- **An RAII guard.** `integral_image`'s `device/kernels/common.hpp` defines `ReadCBGuard` and
  `WriteCBGuard`, whose constructor and destructor call `wait_front`/`pop_front` and
  `reserve_back`/`push_back`. A kernel that says `ReadCBGuard g(dfb::acc, n);` contains none of the
  four names, and the calls inside the guard are made on a `DataflowBuffer(cb)` temporary, so there
  is no named handle to attribute them to either.
- **A shared kernel library, outside the op entirely.** `moreh_dot`'s compute kernel calls
  `compute_kernel_lib::reduce<…>` with the synchronization chosen as a *template argument* —
  `ReduceInputPolicy::WaitAndPopPerTile` — and the FIFO calls live in
  `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl`. That file is not under the op directory, so
  even a recursive grep of the whole op finds nothing.

This is the one item on this list that is **unsafe** rather than merely wrong. The other three make
you miss a site; this one makes you convert a buffer whose synchronization is real. A `Scratchpad`
has no synchronization semantics at all, so the conversion does not preserve that synchronization —
it deletes it. On a compute kernel, where Unpack / Math / Pack run on different physical cores, the
result is a race, and a race that happens to pass your sentinels looks exactly like a clean pass.

**Absence of a grep hit is only evidence when nothing opaque touched the buffer.** If the DFB's
handle — or the id it was constructed from — is ever passed to a function, a constructor, or a
template argument, follow it before concluding anything.

## Step 2 — Survey

Work from the host spec outwards; it is the only place the complete picture exists.

1. **List every `DataflowBufferSpec`** in the op's program factory (or factories).
2. **For each, list every `DFBBinding` that names it**, and note the kernel each binding sits on.
   Include conditional bindings — a DFB bound inside an `if` still counts.
3. **For each DFB, open every binding kernel** and search for the six credit methods *on that DFB's
   handle*:

   ```bash
   grep -nE "reserve_back|push_back|wait_front|pop_front|pages_reservable|pages_available" <each binding kernel>
   ```

   Any hit on that handle → synchronized, not a site.

4. **A clean grep is not yet a verdict — first follow the handle.** Per
   [way 4](#four-ways-the-test-diverges-from-the-property), the calls may be made by a helper on the
   kernel's behalf.
   Before you may conclude sync-free, check whether the DFB's handle, or the id it was built from,
   is passed anywhere: into a function, a constructor (including an RAII guard), or a template
   argument. If it is, open that callee and apply the same test there — following it out of the op
   directory if it leads there, as `ttnn/cpp/ttnn/kernel_lib/` will.

   No hits on that handle in any binding kernel, **and** nothing opaque taking it → **sync-free**.

5. **For each sync-free DFB, read `borrowed_from` on its spec.** That field decides its end-state
   and is the whole fork:

   | `borrowed_from` | What it is | Becomes |
   |---|---|---|
   | set (names a `TensorParamName`) | a view onto memory the op already owns | **`LocalTensorAccessor`** |
   | unset | a private scratch region | **`Scratchpad`** |

An op with no sync-free DFBs is a legitimate zero-site pass, and most ops with a heavy FIFO
pipeline will be exactly that.

> **If the op also has a fake-FIFO DM self-loop, run this pass first and [that
> one](../semantic/dm_self_loop_dfbs.md) after.** Both passes end at a `Scratchpad` and one op can
> hold a site for each, so they meet in the same file. This one is a style pass and small; that one
> is a semantic pass that rewrites control flow, and it leans on the criterion you are learning here.
> They never contend for the same buffer — a DFB either calls the FIFO machinery or it does not — so
> this is about doing the cheap, safe one while it is still cheap, not a correctness constraint.

## Step 3 — Apply

> **Reaching for a worked example?** This recipe cites real ones — `moreh_fold` for the criterion,
> `layernorm_pre_all_gather_welford` for the borrowed case. `ttnn/cpp/ttnn/operations/experimental/quasar/`
> is **out of bounds** and is not evidence of anything, however authoritative it looks; see [the
> procedure](../pass_procedure.md#step-3--apply).

**Don't pattern-match on how the old code wrapped the address.** The examples below show the address
being grabbed into a `CoreLocalMem<T>`, which is the tidy Device 2.0 form, but ported kernels carry
at least four shapes and all of them collapse to the same replacement:

```cpp
CoreLocalMem<uint16_t> lut(dfb.get_read_ptr());                            // Device 2.0 wrapper
auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());  // raw cast
uint32_t addr = dfb.get_write_ptr();                                       // bare address, passed on
auto* q = get_pointer_to_cb_data<uint16_t>(dfb::lut, 0);                   // helper, takes the id
```

What identifies the code you are replacing is that **an address is taken from the DFB and memory is
then reached through it directly**. The wrapper — or its absence — is incidental. Expect to meet all
three, and note that the rawest form tends to sit on the most obviously-scratch buffers.

### Borrowed → `LocalTensorAccessor`

The DFB was never a buffer; it was a window onto a tensor. Give the kernel the tensor directly.

**Host.** Delete the `DataflowBufferSpec` and every `DFBBinding` naming it. The tensor it borrowed
from — the `TensorParamName` in `borrowed_from` — must be bound to each kernel that used it, as a
`TensorBinding` with an accessor name. If such a binding already exists on that kernel, reuse it;
do not add a second one for the same tensor.

**Kernel.** Replace the `DataflowBuffer` local and its pointer arithmetic with an accessor built
from the tensor binding token, indexing directly:

```cpp
// before
DataflowBuffer lut_dfb(dfb::lut);
CoreLocalMem<uint16_t> lut(lut_dfb.get_read_ptr());
… lut[i] …

// after
LocalTensorAccessor<uint16_t> lut(tensor::lut);
… lut[i] …
```

`T` is the element type the kernel actually reads, chosen by you. Element access is **not**
bounds-checked, exactly as the raw-pointer form wasn't — this is not a safety regression, but do
not present it as a safety improvement in your report either.

**If the before-code was `volatile`, `T` is `volatile` too.** The qualifier is part of the element
type, not decoration on the old cast: dropping it lets the compiler hoist or elide a load the old
code forced. Do not assume a NOC read barrier covers you — `invalidate_l1_cache()` is `asm("fence")`
on Blackhole with **no memory clobber**, and empty on Wormhole, so it is not a compiler barrier.
Both forms pass your sentinels today, which is exactly why this has to be read off the old code
rather than decided.

### Regular-backed → `Scratchpad`

**First, check the node rule.** The invariant is simple: **one scratchpad instance serves exactly one
kernel instance.** On any given node the mapping is 1:1 — that is what a scratchpad *is*, private
working memory belonging to one kernel, with no synchronization to share it through.

A `ScratchpadSpec`, though, spans every node its bound kernels run on. So two `KernelSpec`s may bind
the same spec provided their node sets are **disjoint**: each node still gets its own instance
serving its own single kernel, and you have merely saved yourself declaring two identical specs.
That allowance is host-side convenience; it does not loosen the 1:1 rule.

So the question is just: **does any single node have two kernels touching this DFB?**

- **No** — one binding kernel (every self-loop, and the common case), or several over disjoint node
  sets → convert.
- **Yes** — it cannot become a scratchpad. **Stop and raise a feature request**, see [When you can't
  convert a shared scratch DFB](#when-you-cant-convert-a-shared-scratch-dfb). Do not convert it and
  do not work around it.

**Host.** Delete the `DataflowBufferSpec` and its `DFBBinding`s. Add a `ScratchpadSpec` to the
ProgramSpec and a `ScratchpadBinding` to each kernel that used it:

```cpp
// ProgramSpec
ScratchpadSpec{
    .unique_id = SCRATCH,
    .size_per_node = <the bytes the DFB reserved per node>,
},

// KernelSpec
.scratchpad_bindings = {
    ScratchpadBinding{.scratchpad_spec_name = SCRATCH, .accessor_name = "scratch"},
},
```

**`data_format_metadata` has no counterpart, so establish that nothing uses it before dropping it.**
A `ScratchpadSpec` carries only `unique_id` and `size_per_node`. The field exists on a DFB for the
LLKs, which reach tiles through the FIFO protocol, so on a buffer that qualifies for this pass it is
*expected* to be inert — but that is a statement about what the field is for, not a guarantee about
the op in front of you, and nothing stops an op consulting it some other way.

So check rather than assume, and be particular about it here: unlike the DM self-loop pass, this one
can convert a buffer bound by a **compute** kernel, which is where LLK use is plausible in the first
place. Walk every use of the DFB's handle or id across the binding kernels — if each one is a raw
address grab or a NOC operand, nothing consults the declared format and the field drops. A comment
beside the spec saying the format is inert is a pointer to what the author intended, not a substitute
for looking.

**If anything does consult it, stop and report** rather than improvising a way to carry it. That
combination is not currently expressible, and a real op needing it is exactly the evidence the API
owner wants; a workaround in your diff buries it instead. Either way, do not carry it through as a
compile-time arg — that is out of scope for this pass.

`size_per_node` is the same number of bytes the DFB reserved on each node — carry it across, do not
recompute it from tile counts. A DFB sized as *entries × entry size* becomes that product.

If the DFB was **conditionally bound**, the scratchpad is conditionally bound the same way, and the
kernel-side `#ifdef` guarding it stays exactly as it is.

**Kernel.** Construct from the binding token and index it:

```cpp
// before
DataflowBuffer scratch_dfb(dfb::scratch);
… .addr = scratch_dfb.get_write_ptr() …

// after
Scratchpad<uint8_t> scratch(scratch::scratch);
… .addr = scratch.get_base_address() …
```

`Scratchpad` is a template over the element type the kernel views the region as, and that type
**cannot be deduced** from the binding token — write it explicitly. Use `uint8_t` when the kernel
only ever hands the region's address to something else; use the actual element type when it indexes
into it.

Pick the accessor by what the destination wants:

- **`get_base_address()`** returns a `uint32_t` L1 address — the form NOC transfers and LLK
  configuration consume, and the direct replacement for `get_write_ptr()` / `get_read_ptr()`.
- **`operator[]`** for element access. Unlike the raw-pointer form it replaces, it is bounds-checked
  against the region's extent, so prefer it wherever the kernel is actually reading or writing
  elements.

Do not take the address of an element (`&scratch[0]`) to feed an address-consuming API — that is a
`T*`, not the `uint32_t` those APIs take, and it will not compile.

**A note on compute kernels.** A scratchpad has no synchronization semantics at all, and in a
compute kernel the Unpack / Math / Pack stages run on different physical RISC-V cores. Converting a
*genuinely* sync-free DFB removes no synchronization, so the conversion is safe by construction —
but if that reasoning feels strained for the site in front of you, that is a sign the DFB was not
sync-free after all. Re-check the criterion rather than proceeding.

## When you can't convert a shared scratch DFB

A regular-backed sync-free DFB touched by two or more kernels **on the same node** has no
scratchpad equivalent today. Stop, leave the DFB exactly as it is, and report it as a **feature
request**, not merely as a note — the Metal 2.0 host-API owner wants to know when this shape occurs
in real ops, and this pass is the only thing that will ever see it.

Write it so your invoker can forward it as a standalone ticket:

> **Feature request — Scratchpad shared by kernels on the same node.**
> Op / factory: …
> The DFB `<name>` is sync-free and regular-backed, so it is a scratchpad by nature, but it is
> bound by `<kernels>`, which share nodes — and a `ScratchpadSpec` may be bound by multiple
> `KernelSpec`s only over disjoint node sets. What the kernels use it for: …
> Left as a DFB; no workaround applied.

Report it under both the pass's [Outcome](../pass_procedure.md#step-5--report) and its own heading,
so it does not get filed away with the ordinary "noticed, not done" observations.

## When to stop

Per [When the fix doesn't fit](../pass_procedure.md#when-the-fix-doesnt-fit). Specifically here:

- The shared-node scratch case above.
- The borrowed tensor is not reachable as a `TensorBinding` on a kernel that needs it — for
  instance because binding it would exceed a limit, or the tensor is not a program input at all.
- A DFB that is sync-free in every kernel but whose size or layout you cannot account for, so you
  cannot carry `size_per_node` across without guessing.
- Anything that makes you want to change *how* the kernel accesses the memory rather than *what it
  calls the memory*. This pass renames a construct; it does not restructure access patterns. If the
  access has to change for the conversion to work, the DFB was doing something this recipe has not
  anticipated, and that is worth reporting in full.
