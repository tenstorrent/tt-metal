# Post-Port Fix — Replace fake-FIFO DM self-loops with a scratchpad

> **Procedure:** [`pass_procedure.md`](../pass_procedure.md). Read it first; it is the *how*. This
> file is the *what*, and it uses the procedure's steps unchanged.
>
> **Behaviour-preserving?** Yes. The converted kernel performs the same reads and writes, at the same
> addresses, in the same order. Your sentinel tests are a real check here.
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
4. **Confirm the FIFO calls are used** — in each binding kernel:

   ```bash
   grep -nE "<the dfb's local>\.(reserve_back|push_back|wait_front|pop_front)" <the binding kernel>
   ```

   **Grep for calls on *that DFB's handle*, not for the method names loose in the file.** A kernel
   binding several buffers will show plenty of hits belonging to the others, and a bare name-grep
   then reads as "this one uses the FIFO" for a buffer that never touches it. Attribute every hit to
   its receiver.

   **A clean grep is not yet a verdict.** The calls need not appear in the kernel at all — a helper
   can make them on the kernel's behalf — and this grep is stricter than most, because it requires
   the `<handle>.<method>` spelling to appear literally. Under an RAII guard such as
   `integral_image`'s `ReadCBGuard`, or a shared library like
   `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl` where the synchronization is selected by a
   template argument, no such expression exists in the kernel anywhere. Before concluding, check
   whether the DFB's handle — or the id it was built from — is passed into any function, constructor,
   or template argument, and follow it if it is, out of the op directory if that is where it leads.
   The [sync-free pass](sync_free_dfbs.md) documents this failure mode in full.

   **No hits on that DFB, and nothing opaque taking it → not a site for this pass.** A self-loop that
   never calls the FIFO machinery is handled by the [sync-free pass](sync_free_dfbs.md) instead,
   which is a smaller and safer change. Exclude that DFB from your site list, note it in your report
   for the other pass, and **carry on surveying the rest** — this is a per-DFB verdict, not a reason
   to stop the pass. An op can easily hold one of each.

   Getting this one wrong costs more than a missed site. A helper-synchronized buffer that reads as
   having no FIFO calls is exactly what this step hands to the sync-free pass — and that pass
   converts to a `Scratchpad`, which has no synchronization semantics at all. The two passes fail
   into each other here, so the check has to hold in this one.

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

An op with no fake-FIFO DM self-loops is a legitimate zero-site pass.

> **Run this pass before the [sync-free pass](sync_free_dfbs.md) on the same op.** The two are
> siblings — both end at a `Scratchpad` — and one op can hold a site for each, so they will meet in
> the same file and often the same kernel prologue. Order them this way because this pass is the one
> that rewrites control flow: doing it against an unmodified kernel keeps its diff readable, and the
> sync-free pass afterwards is a small, local change that reads cleanly on top. The reverse order
> works but buries the riskier diff underneath the cheaper one.
>
> They never contend for the same buffer: a DFB either calls the FIFO machinery or it does not.

## Step 3 — Apply

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
| `get_write_ptr()` | the index `wr` — the kernel accesses `stage[wr + …]` |
| `get_read_ptr()` | the index `rd` |
| `push_back(n)` | `wr = (wr + n * entry_elems) % stage.size()` |
| `pop_front(n)` | `rd = (rd + n * entry_elems) % stage.size()` |

The two waits drop because a sole single-threaded toucher cannot usefully block on itself: either the
space or the data is already there, or the original program would have hung. Everything else
reproduces exactly what the FIFO was computing, which is why this is *equivalent by construction*
rather than equivalent by argument.

**In the single-entry case — one entry, filled and drained each pass — none of this survives.** Both
indices stay `0`, so there is no index, no stride, and no modulo: the calls simply disappear and the
kernel indexes the scratchpad from zero.

Do not *expect* that case, though. A rotating buffer is just as likely — `num_entries = 2` is a
common default even where nothing overlaps — so read the spec rather than assuming the easy shape.

**Where the buffer genuinely rotates**, the stride is the only extra quantity, and it is in elements:

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

If the kernel has no argument equal to the spec's `entry_size`, add one as a compile-time arg:
`KernelSpec::compile_time_args` (a top-level field, *not* part of `compiler_options`), read
kernel-side as `constexpr uint32_t entry_size = get_arg(args::<name>);`. A CTA is a constant in a
generated header and costs nothing, so prefer adding one over reusing an argument you had to reason
about. A stride that varies with input shape is still fine as a CTA — shape participates in the
program-cache key, so a different shape rebuilds the kernel rather than reusing a stale constant.

The wrap bound is `stage.size()`, in **elements**; `stage.size_in_bytes()` is the byte-indexed
equivalent. Match whichever unit your indices are in.

The wrap bound is `stage.size()`, which the scratchpad already knows — `num_entries` never needs
reconstructing kernel-side.

**The division must be exact.** If `entry_size % sizeof(T) != 0` then `T` is wrong for this buffer —
the kernel is viewing it as something the entries do not align to — and every index built on the
stride will be off. **Stop**; see [When to stop](#when-to-stop).

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

**A multi-bound spec converts as a unit.** If two kernels each self-looped on it, convert both in the
same pass. Half a conversion — one kernel on the scratchpad, the other still binding a DFB that no
longer exists — does not build, and splitting it across two passes gains nothing.
- If the DFB was conditionally bound, the scratchpad is conditionally bound the same way, and the
  kernel-side `#ifdef` guarding it stays as it is.

Two mechanical details that will otherwise cost you a build:

- **`KernelSpec` fields are written in declaration order**, and designated initializers are compiled
  with `-Werror=reorder-init-list`, so the new entries go in their declared positions rather than
  appended. **Read the order off `kernel_spec.hpp`** rather than trusting a list here — it grows.
  At the time of writing: `dfb_bindings` → `semaphore_bindings` → `scratchpad_bindings` →
  `tensor_bindings` → `compile_time_args` → `runtime_arg_schema`.
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
replaces, which is why a `{.offset_bytes = 0}` on the old call becomes `{.offset_bytes = wr}` on the
new one rather than staying zero. A `Scratchpad` operand resolves from its base, so the index has to
appear explicitly.

This holds for every `Noc` operation — `async_read`, `async_write`, their `_with_state` forms,
`async_write_zeros` — because they all resolve an operand the same way. Do **not** extract an address
and pass that instead; a bare address is not a NOC operand, and reaching for `local_mem()` here is
working around a substitution that already exists.

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
| `reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr())` | `uint32_t` |
| hands the address to a NOC call, never accesses elements | `uint8_t` — the index is then a byte offset, and `entry_elems == entry_size` |

If the kernel's element type depends on a dtype branch — `uint16_t` on one path, `float` on another —
construct the scratchpad on each branch with that branch's `T`. The binding token carries no type, so
each branch gets a correctly-typed array over the same allocation.

**When `T` varies across branches, keep `wr` and `rd` in bytes instead of elements**, and convert at
the point of access:

```cpp
uint32_t wr = 0, rd = 0;                       // byte offsets into the region
…
wr = (wr + n * entry_size) % stage.size_in_bytes();   // stride and bound both in bytes
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
`CoreLocalMem` form was, so this is the one place the conversion buys a little safety for free.

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

### Cleanup

After the translation is in and the tests are green, the following are permitted, and **nothing
else**. Every item is safe by local inspection — that is why the list is this short.

1. **The buffer had a single entry** → both indices are always `0`, so delete them, the stride, and
   the modulo. Accesses become `stage[i]` and any address is just `get_base_address()`. (If the
   translation was done faithfully this is usually already how it came out.)
2. **`rd` and `wr` provably hold the same value at every point they are used** → fold them into one
   index.

   **The dangerous near-miss:** a loop that pushes once and pops once per pass, with the same `n`,
   looks like a textbook match — and is not one if it *reads the buffer between the two*. In
   `push` → read → `pop`, the write index has advanced at the read and the read index has not, so the
   two genuinely differ exactly where it matters. Folding them makes the read hit the wrong entry:
   numerically wrong, and on a two-entry buffer it may well still pass a test, because the other slot
   holds recent data. **If anything touches the buffer between the push and the pop, do not fold.**
3. **Rename the leftover locals** off `cb_` / `dfb_` prefixes, which now describe nothing.
4. **Delete a local the translation left unused.** A `uint32_t addr = dfb.get_write_ptr();` whose only
   consumer was the FIFO becomes a variable holding an index nobody reads. Check it really has no
   other use, then remove it rather than leaving the translation's scaffolding behind.

**Do not collapse a multi-entry rotation onto a single slot**, however plainly the buffer looks
emptied on each pass. Whether the rotation is load-bearing depends on how many NOC transfers are
outstanding against the buffer at once, and there is more than one way to have several in flight:
transaction-ID-scoped barriers, or simply issuing a batch of reads before one blanket barrier. A
rotation doing that work looks identical to one that is pure bookkeeping — and pinning the wrong one
lets an in-flight transfer overwrite data still being read. No error, no failing test, wrong
results. If you believe a rotation is vestigial, **say so in your report and leave it in the code**;
a surviving modulo is a correct outcome, not an unfinished job.

Anything you want to change beyond that list goes in your report, not in the diff. This recipe
produces more incidental untidiness than most, which makes it the easiest one to over-clean; the list
is closed on purpose.

## When to stop

Per [When the fix doesn't fit](../pass_procedure.md#when-the-fix-doesnt-fit). Specifically here:

- **The DFB is built on borrowed memory** (`borrowed_from` set) — it is a view onto a tensor the op
  owns, and a scratchpad is not. See survey step 5.
- **The kernel does `evil_*` pointer surgery on the same DFB.** `evil_get_*` / `evil_set_*` move the
  FIFO pointers outside the four calls, so a local replay does not capture the buffer's state and the
  translation would be silently wrong.
- **A rotating buffer whose `entry_size` is not available kernel-side and cannot be added as a
  compile-time arg**, so the stride cannot be written at all. (Single-entry buffers need no stride
  and never reach this.)
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
