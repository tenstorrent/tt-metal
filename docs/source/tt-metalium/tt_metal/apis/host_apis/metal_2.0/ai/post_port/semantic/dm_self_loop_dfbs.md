# Post-Port Fix — Replace fake-FIFO DM self-loops with a scratchpad

> **Procedure:** [`pass_procedure.md`](../pass_procedure.md). Read it first; it is the *how*. This
> file is the *what*, and it uses the procedure's steps unchanged.
>
> **This is a semantic pass** — see [what that
> means](../pass_procedure.md#style-passes-and-semantic-passes). It replaces the FIFO's own address
> arithmetic with arithmetic you write, and removes the credit bookkeeping the FIFO was doing on the
> way. Your sentinels are a real check on the numerics and a weak one on everything else, which is
> why this recipe stops as often as it does. Take those stops literally.
>
> **Behaviour-preserving?** Yes, by intent. The converted kernel performs the same reads and writes,
> in the same order, to and from the same remote addresses. Its own L1 region may land elsewhere — a
> scratchpad is allocated alongside DFBs out of the same region, so the allocation order shifts — and
> nothing functional depends on that.
>
> **Target:** Gen1 (Wormhole / Blackhole) ops already ported to Metal 2.0. **Data movement kernels
> only** — see [Why data movement only](#why-data-movement-only).

---

## What this fix is

A **DM self-loop** is a DataflowBuffer that one data movement kernel binds as *both* `PRODUCER` and
`CONSUMER`. The kernel fills it and drains it, and nobody else touches it.

A DFB is a FIFO for handing data between two parties. Here there is only one party, running
single-threaded on a single RISC-V core, so **none of the FIFO machinery can synchronize anything** —
there is no second actor to wait for, and a wait that could actually block would be a deadlock. What
the four FIFO calls are doing instead is keeping track of an address for the author: a loop index,
wearing a FIFO costume.

The memory is conceptually a scratchpad. This pass makes it one, and moves the address bookkeeping
into plain local variables where a reader can see it.

**Why it matters, and why this one is worth doing properly.** This shape does not survive to Gen2:
Quasar does not support DM self-loop DFBs. Its credit machinery is complex and the hardware
resources it draws on are scarce, and a DM self-loop has no legitimate use case to spend them on —
so the shape is simply unsupported. (Compute self-loops *are* supported on Quasar, with the same
kernel binding both endpoints; the restriction is specific to data movement.) That leaves whoever
ports the op two options, and the wrong one is
much cheaper — bind the unused endpoint to some unrelated kernel and the legalizer stops complaining.
That costs a DFB id the op does not need (and the id budget is genuinely tight — a recently
discovered hardware bug constrains it further), and it leaves behind a buffer whose declared
structure is a fiction, which is worse than what it replaced. **The point of this recipe is that the
correct fix is written down, so nobody has to invent it under deadline.**

## Why data movement only

A **compute** self-loop is legitimate and stays exactly as it is. A compute kernel's Unpack, Math
and Pack stages run on *different physical RISC-V cores*, so a FIFO between them is doing real
synchronization work. The argument above — single thread, nothing to synchronize — applies only to
data movement.

Do not generalize this pass to compute kernels. Self-loop DFBs are supported there — on Gen2 as
well as Gen1 — and are not a problem to be solved.

## Step 2 — Survey

What you are looking for is a DFB **instance** whose producer and consumer are the same **DM kernel
instance**. Both live on one node: a `DataflowBufferSpec` and a `KernelSpec` each materialize once per
node, so the question is always asked about a single node's worth of resources.

At the spec level that resolves to a simple test: **every** `KernelSpec` binding the DFB binds it as
*both* `PRODUCER` and `CONSUMER`.

1. **Find the self-loop bindings.** In the program factory, look for two `DFBBinding`s naming the
   same `dfb_spec_name` on the same DM `KernelSpec`, one `PRODUCER` and one `CONSUMER`.
2. **Then check every *other* kernel binding that same spec.** Do not stop at the first self-loop,
   and do not disqualify the spec merely because a second kernel binds it. Node-exclusive
   multi-binding is legal, so one `DataflowBufferSpec` may be bound by several `KernelSpec`s over
   disjoint node sets:
   - **Every binder takes both roles** → every instance is its own self-loop. All of them are sites,
     and they convert together (see [Apply](#step-3--apply)).
   - **Any binder takes only one role** → that is a genuine cross-kernel handoff, the FIFO is doing
     real work, and the spec is not a site. Leave it.

   You do not need to reason about per-node variation: the DFB implementation requires a spec's bound
   kernel endpoints to sit on the same RISC core across nodes, so the binding structure is uniform
   and a spec-level answer is a per-instance answer.
3. **Confirm every binding kernel is data movement**, not compute. See above.
4. **Account for every use of the DFB, against a closed list.** This is the step the pass turns on,
   and it works the other way round from a search: rather than hunting for something disqualifying,
   you enumerate *everything* the binding kernels do with this buffer and check each use against the
   list below. What is not on the list is not covered.

   | Use | what the transformation does with it |
   |---|---|
   | `reserve_back` / `push_back` / `wait_front` / `pop_front` | translated — this is the fake FIFO |
   | `get_write_ptr()` / `get_read_ptr()` | become the indices |
   | operand of `noc.async_read` / `noc.async_write`, or their `_with_state` forms | the `Scratchpad` substitutes directly |

   Start from the handle rather than from method names, so that a kernel binding several buffers
   cannot lend one buffer's hits to another:

   ```bash
   grep -nE "<the dfb's local>|dfb::<its accessor name>" <each binding kernel>
   ```

   Then read every hit and attribute it. Follow the handle — and the id it was built from — into any
   function, constructor, or template argument it is passed to, out of the op directory if that is
   where it leads.

   **Anything not on the list is a stop.** Not a smaller conversion and not a judgement call: report
   the use and move on. The list is short because it is exactly what the transformation below covers,
   and a use outside it needs a decision this pass was not given. Four that really occur:

   - **Any other method on the handle** — `get_entry_size()`, `get_total_num_entries()`,
     `get_total_size_bytes()`, `get_stride_size()`, `evil_set_write_ptr` / `evil_set_read_ptr`.
     `get_entry_size()` is the common one by a wide margin: it is how most ported kernels spell the
     entry size, and it is a method on the very object this pass deletes. A `Scratchpad` has no
     equivalent, so the value would have to be reconstructed from somewhere — which is exactly the
     guess this pass declines to make.
   - **`pages_reservable_at_back` / `pages_available_at_front`.** These read the credit counters
     without blocking, and the translation deletes the credit posts along with the calls that made
     them — sound only while nothing reads them. Note that neither name contains any of the four
     FIFO calls as a substring, so a grep written from memory misses them.
   - **`noc.async_write_zeros`, or the DFB as a multicast destination.** Both reject a `Scratchpad`
     outright — `async_write_zeros` accepts `CircularBuffer` or `DataflowBuffer` only, and
     `noc_traits_t<Scratchpad<T>>::dst_addr_mcast` is a hard `static_assert(false)`. The one-token
     substitution in [Step 3](#step-3--apply) does not reach these, and the failure is a compile
     error only *after* you have written the whole conversion.
   - **A FIFO call made through anything other than a `DataflowBuffer` declared in this kernel** — a
     helper taking the DFB or its id, an RAII guard such as `integral_image`'s `ReadCBGuard`, a
     template argument as in `ttnn/cpp/ttnn/kernel_lib/`. The translation rewrites calls sitting
     beside the handle, with `wr` and `rd` in that scope; it has nothing to say about a call made
     inside a callee.

   **Every use on the list, and at least one of the four FIFO calls → this is a site.**

   **Every use on the list, but none of the four FIFO calls → not a site *for this pass*.** A
   self-loop that never calls the FIFO machinery belongs to the [sync-free pass](../style/sync_free_dfbs.md),
   which is a smaller and safer change. Note it in your report for that pass and **carry on surveying
   the rest** — this is a per-DFB verdict, not a reason to stop. An op can easily hold one of each.

   Getting *that* verdict wrong costs more than a missed site, which is why the helper case above is
   a stop rather than a hand-off: a helper-synchronized buffer looks exactly like one with no FIFO
   calls, and the sync-free pass converts to a `Scratchpad`, which has no synchronization semantics
   at all. The two passes fail into each other here, so neither may guess.

5. **Confirm the DFB is not built on borrowed memory.** Check `borrowed_from` on its
   `DataflowBufferSpec`: it must be **unset**.

   A `ScratchpadSpec` is a *fresh private allocation*. A borrowed DFB is a *window onto memory the op
   already owns* — a tensor it allocated or was handed. Swapping one for the other does not move the
   data; it points the kernel at different memory entirely, and nothing complains. Everything
   compiles, the kernel runs, and whatever the op expected to find in that tensor is not there.

   **If `borrowed_from` is set, stop and report.** The index translation would be unchanged, but the
   destination cannot be a scratchpad — it would have to be a `LocalTensorAccessor` over the borrowed
   tensor, and fake-FIFO bookkeeping over borrowed memory is a combination nothing in this suite has
   examined. Report it as a site this recipe does not cover, with what the buffer borrows from and
   what the kernel does with it; that is more useful than a guess.

6. **Confirm the DFB's size is not overridden at runtime.** In the same factory, read the
   `ProgramRunArgs` construction and look for a `dfb_run_overrides` entry naming this DFB.

   `DataflowBufferSpec`'s `entry_size` and `num_entries` are *declared* sizes; `ProgramRunArgs` can
   override both per execution, and the header says so beside the fields. A `ScratchpadSpec` carries
   only `size_per_node`, fixed when the spec is built, and there is no scratchpad counterpart in
   `ProgramRunArgs` at all — so the size reaches the kernel as a constant, and `stage.size()`, the
   wrap point and the bounds check all come from the declared value.

   **If an override names this DFB, stop and report.** Everything the translation computes would be
   built on a number the op does not actually run with.

An op with no fake-FIFO DM self-loops is a legitimate zero-site pass.

> **Run the [sync-free pass](../style/sync_free_dfbs.md) before this one on the same op.** The two
> are siblings — both end at a `Scratchpad` — and one op can hold a site for each, so they will meet
> in the same file and often the same kernel prologue. Take the safe one first: sync-free is a style
> pass whose worst outcome is a missed site, and it teaches both the DFB-to-scratchpad substitution
> and the helper-hidden-call failure mode that step 4 above turns on. Arriving here having already
> run it makes that cross-reference a callback rather than a detour.
>
> They never contend for the same buffer: a DFB either calls the FIFO machinery or it does not. So
> the order is about which diff you would rather write second, not a correctness constraint.

## Step 3 — Apply

> **Reaching for a worked example?** This recipe cites real ones — `indexed_fill` carries a
> complete conversion of this shape. `ttnn/cpp/ttnn/operations/experimental/quasar/` is **out of
> bounds** and is not evidence of anything, however authoritative it looks; see [the
> procedure](../pass_procedure.md#step-3--apply).

### How to think about it

**A `Scratchpad<T>` is a C array.** `Scratchpad<uint32_t> stage(scratch::stage)` is `uint32_t
stage[N]`: one chunk of allocated memory, indexed from zero, and every access goes through
`stage[i]`. That is the whole model, and it is not exotic.

Two things follow, and they are what the rest of this section is:

- **The FIFO's pointers become array indices.** The intro described the four calls as a loop index
  in a FIFO costume; here you take the costume off. Where the old code asked the FIFO "where is the
  write pointer," the new code already knows — it is an index it has been keeping itself.
- **You never manufacture a view at an offset.** You would not take `&arr[k]` in C and build a
  second array object out of it, and the same applies here: no `CoreLocalMem` rebuilt from an
  address, no second `Scratchpad`. The scratchpad *is* the allocation, which is what makes its
  bounds check mean anything; a fabricated view has no allocation behind it.

### The translation

Replay the FIFO's state machine in two local indices. Each call maps individually — you do **not**
need to work out what the surrounding loop is doing, and you should not try to:

```cpp
uint32_t wr = 0;   // write index, in elements of T
uint32_t rd = 0;   // read index, in elements of T
```

| FIFO call | becomes |
|---|---|
| `reserve_back(n)` | *nothing* |
| `wait_front(n)` | *nothing* |
| `get_write_ptr()` | the index `wr` **as it stands where the old call was** — the kernel accesses `stage[wr + …]` |
| `get_read_ptr()` | the index `rd`, likewise at the point of the old call |
| `push_back(n)` | `wr += n * entry_elems`, then wrap — see [The wrap](#the-wrap) |
| `pop_front(n)` | `rd += n * entry_elems`, then wrap — likewise |

The two waits drop because a sole single-threaded toucher cannot usefully block on itself: either the
space or the data is already there, or the original program would have hung. They are *pure* waits —
neither one moves a pointer, both only spin on the buffer's credit counters — so deleting them
removes a stall and no state. Everything else reproduces exactly what the FIFO was computing, which
is why this is *equivalent by construction* rather than equivalent by argument.

**Substitute the index's value where the old call sat, not the variable as it reads later.** The two
getter rows are the ones this bites. A kernel that snapshots the address into a local —
`uint32_t addr = dfb.get_write_ptr();` — and goes on using `addr` *after* a `push_back` is still
referring to the pointer's value from before the advance. Replacing `addr` with a bare `wr` there
silently retargets every access to the next slot, which was never written. Where the old code
captured once and never re-derived, the index it captured is frozen at that value; the later updates
are what [Cleanup](#cleanup) item 1 is about.

**You need a stride only if an index is read after it advances.** Settle that first, because if no
index is, everything from here to the end of this subsection is moot. Ask: does any access to the
buffer use `get_write_ptr()` or `get_read_ptr()` at a point *after* a `push_back` or `pop_front` has
moved that pointer? If none does, then every read of that index sees `0` — the advance is dead before
you write it, so you do not write it, and there is no stride and no wrap. The calls simply disappear
and the kernel indexes the scratchpad from zero.

That is a question about the kernel, not about the spec. A single-entry buffer reaches it
automatically; so does a multi-entry buffer whose kernel captures the pointer before its only
`push_back`. It runs the other way too — `num_entries = 2` is a common default even where nothing
overlaps — so neither a small entry count nor a large one answers it. Read what the kernel does with
the pointer.

**Where an index *is* read after it advances**, the stride is the only extra quantity, and it is in
elements:

```cpp
constexpr uint32_t entry_elems = entry_size / sizeof(T);   // elements per former FIFO entry
```

**The stride is whatever expression the `DataflowBufferSpec`'s `entry_size` field held** — read it
before you delete the spec, and carry that expression forward. It is the only place the stride is
stated exactly.

**Do not assume the size the kernel passes to its NOC transfers is the same number.** It very often
is not: a factory routinely aligns the entry up (`aligned_page_size = round_up(page_size, …)`) while
the kernel transfers the *unaligned* size into it. Reuse a kernel-side argument only after checking
it is the same expression as the spec's `entry_size` — and be wary of a near-miss neighbour, since an
op may carry several aligned variants of one size that agree on the architecture you are looking at
and diverge on another. Getting this wrong gives a stride that is quietly a few bytes off.

**If the kernel has no value equal to the spec's `entry_size`, stop.** Do not add a compile-time arg
to supply one. A CTA is baked into the generated header when the program is built, so it stays
correct only if the program is rebuilt whenever the stride changes — and that depends on the op's
program-cache key. Most ops put shape in that key; some deliberately do not, and Metal 2.0 supports
both. Working out which one you are looking at is not this pass's job.

Reusing a constant the kernel *already* has is a different matter and is fine: the op already depends
on that value being right for this invocation, so you introduce no new dependency on the cache key.
Adding one is where this pass would have to start guessing, so it stops there instead. Report the
buffer, the spec's `entry_size` expression, and what the kernel does have.

**The division must be exact.** If `entry_size % sizeof(T) != 0` then `T` is wrong for this buffer —
the kernel is viewing it as something the entries do not align to — and every index built on the
stride will be off. **Stop**; see [When to stop](#when-to-stop).

#### The wrap

The bound is `stage.size()`, in **elements**; `stage.size_in_bytes()` is the byte-indexed equivalent.
Match whichever unit your indices are in. The scratchpad already knows it either way, so
`num_entries` never needs reconstructing kernel-side.

**The wrap is not a modulo.** Write what the FIFO wrote:

```cpp
wr += n * entry_elems;
ASSERT(wr <= stage.size());              // the FIFO asserts exactly this
if (wr == stage.size()) { wr = 0; }
```

That is `push_back`'s own body on a Gen1 DM kernel, minus its credit post. `pop_front` is the same
three lines on `rd`.

**Why testing for equality is enough.** A DFB requires the pushes in one trip around the buffer to
sum to *exactly* its size. The individual pushes need not divide it — on a twelve-entry buffer
`push_back(5)` then `push_back(7)` is correct and `push_back(7)` twice is not — so the pointer lands
on the end exactly and never past it. That is why the FIFO compares for equality, and why a variable
`n` needs nothing extra here. A kernel that broke the rule was already broken as a DFB; it is not a
case your translation has to survive.

**Why not `%`.** It is not what the code you are replacing did, and this translation's whole claim is
that it replays that code. It costs a division where the original cost a compare. And it *conceals* a
mistake: derive `entry_elems` wrongly and a modulo folds the bad index quietly back into range, so
the op computes on the wrong entries and nothing anywhere objects — whereas the form above runs the
index off the end, in reach of the assertion and of `operator[]`'s bounds check. Both of those are
`ASSERT`s, compiled in only under watcher or lightweight kernel asserts, so what this buys you is a
failure that is *detectable*, not one that is guaranteed. It is still the side to be on.

> **Keep `wr` and `rd` separate while translating, however alike they look.** A loop that pushes once
> and pops once per pass, with the same `n`, invites collapsing them into one index — and that is
> wrong whenever the buffer is *read between the push and the pop*, because there the write index has
> advanced and the read index has not. Folded, the read lands on the wrong entry: numerically wrong,
> and on a two-entry buffer it may still pass a test, because the other slot holds recent data.
> Cleanup item 2 permits the fold only when nothing touches the buffer in between; do not anticipate
> it here.

Do the translation faithfully first, then clean up. Trying to shortcut to the tidy end state while
translating is how an off-by-one gets in.

### Only those six calls change

Leave everything else in the kernel exactly as it is. In particular, **do not touch NOC barriers** —
`noc_async_read_barrier()` and friends stay where they are. `push_back` was never doing barrier duty,
and a barrier that looks redundant beside a FIFO call you just deleted is still the thing making an
async transfer safe.

### Host side

Same shape as any DFB-to-scratchpad conversion:

- Delete the `DataflowBufferSpec` and **all** of its `DFBBinding`s — both roles, on every kernel that
  bound it.
- Add a `ScratchpadSpec` to the `ProgramSpec`, registered on the spec (`spec.scratchpads`). Its
  `size_per_node` is the DFB's whole allocation — `entry_size * num_entries`. There is no total-size
  field on `DataflowBufferSpec` to copy, so this one *is* computed; write the product from the two
  fields rather than a literal.
- Add a `ScratchpadBinding` to **each** kernel that bound it, naming the accessor that kernel will
  use. Where several kernels shared the spec they keep sharing it: a `ScratchpadSpec` may be bound by
  multiple `KernelSpec`s over disjoint node sets, which is exactly the configuration the DFB was in.
- **`data_format_metadata` has no counterpart, so establish that nothing uses it before dropping
  it.** A `ScratchpadSpec` carries only `unique_id` and `size_per_node`. The field exists on a DFB
  for the LLKs, which reach tiles through the FIFO protocol, so on a buffer this pass converts it is
  *expected* to be inert — but that is a statement about what the field is for, not a guarantee
  about the op in front of you, and nothing stops an op consulting it some other way.

  So check rather than assume. Walk every use of the DFB's handle or id across the binding kernels:
  if each one is a raw address grab or a NOC operand, nothing consults the declared format and the
  field drops. Some ops say as much in a comment beside the spec — useful as a pointer to what the
  author intended, not as a substitute for looking.

  **If anything does consult it, stop and report.** That combination is not currently expressible,
  and a real op needing it is exactly the evidence the API owner wants; a workaround in your diff
  buries it instead. Either way, do not carry it through as a compile-time arg — that is out of
  scope for this pass.

**A multi-bound spec converts as a unit.** If two kernels each self-looped on it, convert both in the
same pass. Half a conversion — one kernel on the scratchpad, the other still binding a DFB that no
longer exists — does not build, and splitting it across two passes gains nothing.
- If the DFB was conditionally bound, the scratchpad is conditionally bound the same way — **and the
  `spec.scratchpads` registration carries that same guard.** A `ScratchpadSpec` declared but bound by
  no kernel is rejected at program creation with a `TT_FATAL`, not at compile time, so an unguarded
  registration builds cleanly and then fails on whichever configuration skips the binding — possibly
  not one your sentinels run. The kernel-side `#ifdef` guarding it stays as it is.

Two mechanical details that will otherwise cost you a build:

- **`KernelSpec` fields are written in declaration order**, and designated initializers are compiled
  with `-Werror=reorder-init-list`, so the new entries go in their declared positions rather than
  appended. **Read the order off `kernel_spec.hpp`.** `dfb_bindings` precedes `scratchpad_bindings`,
  but that is a subset: the struct carries other fields before, between and after them, and it grows.
  Do not treat those two as the list — they are only the ones you are editing.
- **The spec-name constant changes type**, from `DFBSpecName` to `ScratchpadSpecName` — different
  `StrongType`s, so the declaration must be edited regardless. While you are on that line, rename it
  off any `cb`/`dfb` wording: a scratchpad named `input_cb` is exactly the thing these passes exist
  to stop.

### Kernel side, the handle

`Scratchpad` is a template over the element type the kernel views the region as; it cannot be deduced
from the binding token, so write it out.

Which accessor replaces the old pointer depends on what the old code *did* with it.

**Where the buffer was a NOC operand**, hand the NOC the scratchpad itself. A `DataflowBuffer` is
passed to `Noc` operations as an object, not as an address, and `Scratchpad` works the same way — so
this is a one-token substitution, with the former FIFO pointer becoming the transfer's byte offset:

```cpp
// before
DataflowBuffer stage_dfb(dfb::stage);
noc.async_read(src, stage_dfb, size, src_args, {.offset_bytes = 0});

// after
Scratchpad<uint32_t> stage(scratch::stage);
noc.async_read(src, stage, size, src_args, {.offset_bytes = wr * sizeof(uint32_t)});
```

**Why the offset picks up `wr`:** a `DataflowBuffer` used as a NOC operand resolves to its *current
FIFO pointer* plus the given offset — `get_write_ptr() + offset_bytes` as a destination,
`get_read_ptr() + offset_bytes` as a source. That moving pointer is the thing your index now
replaces, which is why a `{.offset_bytes = 0}` on the old call becomes
`{.offset_bytes = wr * sizeof(T)}` on the new one rather than staying zero. A `Scratchpad` operand
resolves from its base, so the index has to appear explicitly — and the field is named
`offset_bytes` for a reason, so an element-indexed `wr` scales by `sizeof(T)` on the way in. (With
byte-indexed `wr`, as when `T` varies across branches, it goes in unscaled.)

This holds for `async_read`, `async_write` and their `_with_state` forms, which all resolve an operand
the same way. It does **not** extend to `async_write_zeros`, nor to a multicast destination — both
reject a `Scratchpad` outright, which is why [survey step 4](#step-2--survey) stops on them rather
than letting you discover it as a compile error at the end. Within the forms it does cover, do **not**
extract an address and pass that instead; a bare address is not a NOC operand, and reaching for
`local_mem()` here is working around a substitution that already exists.

**Where an address was genuinely needed** — an endpoint struct's `.addr` field, LLK configuration —
use `get_base_address()`:

```cpp
… .addr = stage.get_base_address() + wr * sizeof(uint32_t) …
```

Both forms want *bytes*, and `wr` is an index in elements of `T`, so it scales by `sizeof(T)` —
not by the entry stride, which `wr` already carries. In the single-entry case `wr` is `0` and both
expressions lose their offset entirely.

**Choosing `T`.** Read it off the code you are replacing — the old wrapper or cast already names how
the kernel views this memory:

| what the before-code does | `T` |
|---|---|
| `CoreLocalMem<uint32_t> m(dfb.get_write_ptr())` | `uint32_t` |
| `CoreLocalMem<volatile uint32_t> m(dfb.get_write_ptr())` | `volatile uint32_t` |
| `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr())` | `volatile uint32_t` |
| hands the address to a NOC call, never accesses elements | `uint8_t` — the index is then a byte offset, and `entry_elems == entry_size` |

**Carry `volatile` across if the before-code had it** — it is part of `T`, not decoration. The
qualifier is the caller's to choose (`core_local_mem.h` documents both forms), and dropping it lets
the compiler hoist or elide a load that the old code forced. Do not assume the NOC read barrier
protects you: `invalidate_l1_cache()` is `asm("fence")` on Blackhole with **no memory clobber**, and
on Wormhole its body is empty — it is not a compiler barrier. Nothing else in the converted kernel
writes that region as far as the compiler can see, so a non-`volatile` `T` is free to be optimized
across the transfer. This is invisible to your sentinels: both forms pass today, and the one that
breaks does so on some later compiler or optimization level.

If the kernel's element type depends on a dtype branch — `uint16_t` on one path, `float` on another —
construct the scratchpad on each branch with that branch's `T`. The binding token carries no type, so
each branch gets a correctly-typed array over the same allocation.

**When `T` varies across branches, keep `wr` and `rd` in bytes instead of elements**, and convert at
the point of access:

```cpp
uint32_t wr = 0, rd = 0;                       // byte offsets into the region
…
wr += n * entry_size;                                 // stride and bound both in bytes
ASSERT(wr <= stage.size_in_bytes());
if (wr == stage.size_in_bytes()) { wr = 0; }
…
value = stage_u16[rd / sizeof(uint16_t) + i];         // convert where the type is known
```

Bytes are the one representation every branch agrees on, and the indices usually have to live outside
the `#ifdef`s anyway — the FIFO calls they replace are typically in the common path even when the
element accesses are not. Do this only when the type genuinely varies; with a single `T`, element
indices read better and need no conversion at all.

The NOC calls are usually in that common path too, so declare **one more scratchpad outside the
branches** for them to use. Its `T` is free — a NOC operand resolves by address and `size_in_bytes()`
is type-independent — so use `uint8_t` and let the byte indices read naturally against it. Take the
typed views inside the branches, once each at the top rather than rebuilt per iteration: the
scratchpad base does not move, so there is nothing to re-anchor.

**Where the kernel read or wrote elements through the old address**, index the scratchpad directly.
`Scratchpad<T>` *is* the typed handle to that memory, so a `CoreLocalMem` built on top of it — or a
raw cast serving the same purpose — is a second wrapper around the same allocation and goes away:

```cpp
// before
DataflowBuffer stage_dfb(dfb::stage);
CoreLocalMem<uint32_t> stage_mem(stage_dfb.get_write_ptr());
… stage_mem[i] = v …

// after
Scratchpad<uint32_t> stage(scratch::stage);
… stage[i] = v …
```

`operator[]` is bounds-checked against the region's extent, which neither the raw pointer nor the
`CoreLocalMem` form was, so this is the one place the conversion buys a little safety for free. Keep
the claim narrow, in your report as well as in your head: the check validates *the element you are
subscripting*, not a range. `&stage[i]` handed to something that then reads a page from it is
checked at `i` and nowhere else.

Note there is no `wr` in that example, and that is not a simplification. The write pointer sits at
its initialized position — the base of the allocation — until something advances it, so a view taken
once at the top of the kernel is a view of the whole array. Indices only enter where the kernel
re-reads `get_write_ptr()` / `get_read_ptr()` *after* a `push_back` or `pop_front` has moved the
pointer, and there the access is `stage[wr + i]`, still through `operator[]` and still
bounds-checked.

`local_mem()` hands back the underlying `CoreLocalMem<T>` **at the base**, for the few operations
`operator[]` does not cover. Do not offset it: `local_mem() + k` is an unbounded handle with no
allocation behind it, which is the manufactured view this recipe is removing, not a way to express
one.

**If the old address doubled as a flag, you have to re-express the flag.** A kernel that stages
conditionally often declares the pointer up front, leaves it null when it does not stage, and later
tests the pointer itself as the predicate:

```cpp
volatile tt_l1_ptr int* addr_ptr = nullptr;
if (batch_id_size > 0) { … addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(l1_write_addr); }
…
if (addr_ptr) { … }        // means "did we stage anything?", not "is this pointer valid?"
```

A `Scratchpad` cannot be null — it is bound for the program's lifetime whether or not the kernel put
anything in it — so `if (addr_ptr)` has no direct translation and **must** change. This is a
required repair rather than optional cleanup, and it is [work your own change
made necessary](../pass_procedure.md#step-3--apply), so report it as such.

Substitute the condition the pointer was *standing in for*, which is almost always sitting right
there as the guard that set it — `if (batch_id_size > 0)` above. Recover it by reading, not by
inventing a nearer-looking test: a predicate that merely agrees on the inputs you thought about is
the kind of error nothing downstream will catch. If you cannot identify what the nullness meant,
stop — see [When to stop](#when-to-stop).

If you know what it meant but the condition is not usable where the predicate sits — the value is
out of scope there, or recomputing it would be awkward — carry the answer in an explicit `bool`,
set where the pointer used to be:

```cpp
bool staged = false;
if (batch_id_size > 0) { … staged = true; }
…
if (staged) { … }
```

**Do not encode the flag as a sentinel index** (`-1`, `UINT32_MAX`). The pointer was only ever a
flag by accident — a pointer happened to be what was in hand — and an index carrying a sentinel
repeats that double duty in fresh code, where it has no excuse. It also forces the index signed or
magic-valued, and the index is the thing your [wrap](#the-wrap) runs on — where a sentinel is out of
range by construction and trips the assertion. Often there is no index to overload anyway: once [Cleanup](#cleanup) item 1 has run on a buffer whose update was dead, none
remains.

Note this is a *runtime* condition and a different thing from a **conditionally bound** DFB, which
is a compile-time affair and keeps its `#ifdef` untouched (see [Host side](#host-side)).

### Cleanup

After the translation is in and the tests are green, the following are permitted, and **nothing
else**. Every item is safe by local inspection — that is why the list is this short.

**These are readability nice-to-haves, and correctness outranks every one of them.** A cleanup you
are not certain of is not worth its own risk: the pass has already delivered its value once the
translation is in, and a simplification that turns out to be wrong costs far more than the tidiness
was worth. **If you are not sure, describe it in your report and leave the code alone.** That is a
success-tier outcome — someone who knows the op can act on the observation, and nothing was
gambled to get it.

Do not lean on your sentinels to license one of these. Coverage varies a great deal across ops, and
some of these buffers are exercised thinly or not at all by the tests you are running; a green run
is much weaker evidence for a cleanup than it is for the translation, which is equivalent by
construction. The whole list assumes you have satisfied yourself *locally*, by reading, and the
tests are only a backstop.

1. **The index's value is never read after the update** → the update is dead, so delete it along
   with the stride and the wrap. Accesses become `stage[i]` and any address is just
   `get_base_address()`. This should normally have been settled during [the
   translation](#the-translation), which asks the same question before deriving a stride; this item
   is the backstop for a case you only see once the code is in front of you.

   The question is **observability, not entry count.** A single-entry buffer reaches this
   automatically, since both indices stay `0` — but so does a multi-entry buffer whose kernel
   captures the pointer *before* the only `push_back` and never re-reads it afterwards. Conversely a
   two-entry buffer that does re-read the pointer needs its rotation kept. Read what the kernel does
   with the index; do not infer it from `num_entries`. (If the translation was done faithfully, a
   dead update is usually already visible as a variable nothing consumes.)
2. **`rd` and `wr` provably hold the same value at every point they are used** → fold them into one
   index.

   **The dangerous near-miss:** a loop that pushes once and pops once per pass, with the same `n`,
   looks like a textbook match — and is not one if it *reads the buffer between the two*. In
   `push` → read → `pop`, the write index has advanced at the read and the read index has not, so the
   two genuinely differ exactly where it matters. Folding them makes the read hit the wrong entry:
   numerically wrong, and on a two-entry buffer it may well still pass a test, because the other slot
   holds recent data. **If anything touches the buffer between the push and the pop, do not fold.**
3. **Rename the leftover locals** off `cb_` / `dfb_` prefixes, which now describe nothing. Name each
   one for the buffer — `stage`, `ids` — as you would anyway.
4. **Delete a local the translation left unused.** A `uint32_t addr = dfb.get_write_ptr();` whose only
   consumer was the FIFO becomes a variable holding an index nobody reads. Check it really has no
   other use, then remove it rather than leaving the translation's scaffolding behind.

**Do not collapse a rotation that is still observed onto a single slot**, however plainly the buffer
looks emptied on each pass. This is precisely the case item 1 does *not* reach: there the index's
updated value is never read, so removing it changes nothing; here the kernel does read it after it
advances, so the rotation is live code and the question becomes whether it is load-bearing. That
depends on how many NOC transfers are outstanding against the buffer at once, and there is more than
one way to have several in flight:
transaction-ID-scoped barriers, or simply issuing a batch of reads before one blanket barrier. A
rotation doing that work looks identical to one that is pure bookkeeping — and pinning the wrong one
lets an in-flight transfer overwrite data still being read. No error, no failing test, wrong
results. If you believe a rotation is vestigial, **say so in your report and leave it in the code**;
a surviving wrap is a correct outcome, not an unfinished job.

Anything you want to change beyond that list goes in your report, not in the diff. This recipe
produces more incidental untidiness than most, which makes it the easiest one to over-clean; the list
is closed on purpose.

## When to stop

Per [When the fix doesn't fit](../pass_procedure.md#when-the-fix-doesnt-fit). Specifically here:

**The general rule is [survey step 4](#step-2--survey): the kernel does something with this DFB that
is not on the covered list.** The first two entries below are that rule; the rest are conditions it
does not reach.

- **Any use of the DFB outside the covered list** — another method on the handle (`get_entry_size()`
  most often), `pages_reservable_at_back` / `pages_available_at_front`, `noc.async_write_zeros`, the
  DFB as a multicast destination, or a FIFO call reached through a helper, an RAII guard or a
  template argument rather than sitting beside the handle. Survey step 4 says why each is outside
  what the transformation covers.
- **`evil_set_write_ptr` / `evil_set_read_ptr` on the same DFB.** Also off the list, but worth its
  own line because it fails differently: these move the FIFO pointers outside the four calls, so a
  local replay does not capture the buffer's state and the translation comes out silently wrong
  rather than unbuildable.
- **The DFB is built on borrowed memory** (`borrowed_from` set) — it is a view onto a tensor the op
  owns, and a scratchpad is not. See survey step 5.
- **The DFB's size is overridden at runtime** — a `dfb_run_overrides` entry names it, and a
  scratchpad's size is fixed when its spec is built. See survey step 6.
- **An index is read after it advances and no kernel-side value equals the spec's `entry_size`**, so
  the stride cannot be written without inventing one. Do not add a compile-time arg to supply it —
  see [the stride](#the-translation). (Where no index is read after advancing, there is no stride to
  find and this does not apply.)
- **`entry_size` is not a whole number of `T`** — `entry_size % sizeof(T) != 0` means `T` is wrong
  for this buffer, and every index built on the stride will be off.
- **The buffer's entries are not uniform** — a variable-size `reserve_back` is fine, but a buffer
  whose entry stride is not a single constant has no `entry_elems` to compute.
- **The kernel reads or writes past what the FIFO handed it**, so the FIFO's entry accounting was
  never describing the real access pattern.

**And whatever you do, do not make the Gen2 legalizer error go away by binding the unused endpoint to
another kernel.** That is the workaround this recipe exists to prevent: it silences the error, wastes
a DFB id the op does not need, and leaves a buffer claiming a structure it does not have — a worse
state than the one you started in, and much harder for the next reader to unpick. If you cannot do
the conversion, stop and report; leaving the self-loop in place is the correct outcome.
