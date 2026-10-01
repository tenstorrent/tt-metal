# TensorAccessor on the Quasar address generator: what the compiler could take over

Status: proposal (Phase C2). Library side implemented and tested on emu-quasar-2x3; compiler side not started.
Code: `tensor_accessor_addrgen.h` (walkers), `overlay/addrgen_state.hpp` (save/restore),
`api/tensor/transfer_noc_addr.h` (the hook the NoC APIs call).

## 1. What runs today, without compiler help

Every NoC transfer whose endpoint is a TensorAccessor (directly, through `pages()` / `shard_pages()`, `PageView`,
`ShardView` or `AbstractTensorAccessorWrapper`) asks `transfer_noc_addr()` for its address. On Quasar with ATT,
that address comes from one of the DM core's two address generators:

| Step, per transfer | Cost today |
|---|---|
| Find the walk for this accessor and direction: key compare over 4 records | ~4 compares, run time |
| Pick the hardware slot (generator and side are instruction immediates) | 1 switch into one of 4 copies of each RoCC sequence |
| Walk not resident (a 3rd tensor in one direction evicted it): save the evictee's registers, restore this walk's | 11 `rd_reg` + fences, ~10 `wr_reg` |
| Requested page behind the walk, or past what the programming covers: **seek** | recompute the address in software, ~10 `wr_reg` |
| Requested page ahead of the walk: **skip** in hardware | 1 extra pop |
| Otherwise | 1 pop (plus the learned stride as its pop amount) |

Measured on the TensorAccessor suite (16–32 page tensors): 1 seek per tensor for interleaved, HEIGHT, WIDTH, ND
round-robin; 1 seek per shard band for blocked layouts; strided walks (every Nth page) cost 1 pop per page once the
stride repeats; 3 tensors on one core cost 3 seeks + 1 restore per transfer instead of 1 seek per transfer.

Slots: each address generator has a source and a destination side, so a DM core has four walk slots. Reads walk on
source sides and writes on destination sides, so reads and writes never evict each other; only more than two walks in
the same direction need save/restore.

What the library cannot remove at run time: the walk lookup, the slot switch, and per-transfer save/restore when more
than two walks in one direction are live. Those are decisions the compiler can make once, statically.

## 2. The asks

### Ask 1: treat the address-generator registers as compiler-visible state

Today every `__builtin_riscv_ttrocc_addrgen_*` is an opaque side effect, so the compiler can't reorder, merge, or
drop any of them. Proposal: model each generator's registers as named state (like CSRs), with these effects:

| Builtin | Reads | Writes |
|---|---|---|
| `wr_reg(G, r, v)` | — | register r of G |
| `rd_reg(G, r)` | register r of G | — |
| `pop_x_src(G, n)` / `pop_x_dest(G, n)` | all source (dest) loop state of G | the source (dest) position registers of G: BANK_CURRENT, INNER_ADDRESS, OUTER_ADDRESS, face base |
| `reset(G)` | — | all of G |
| `push_*` | as pop | as pop, plus the paired command buffer |

Generators 0 and 1 are independent, and so are a generator's two sides (they share only the MISC register); none of
them aliases memory. With that, the compiler can:
- drop a `wr_reg` whose value the register already holds (re-programming the same banking on every seek);
- hoist loop-invariant writes out of loops;
- keep a generator's state across calls it can see into.

### Ask 2: static walk allocation

When a region (a loop nest, or a whole kernel) has at most two live read walks and two live write walks, give each
walk a slot (generator + side, by direction) at compile time. The walker's key lookup and address-generator branch then disappear. The generator id becomes the
immediate the RoCC instruction needs anyway.

What the compiler needs to know, per walk:
- **Which tensor.** The accessor's binding id, a compile-time constant in its type
  (`DistributionSpec<..., BindingId>`, from Paul's op_to_op change). Accessors built from the same binding share a
  walk. Accessors without a binding id, and `AbstractTensorAccessorWrapper` (type-erased), stay on the run-time
  walker.
- **Which kind and direction.** Page-id walk, shard-base walk (ShardView), or in-shard walk (`shard_pages()`), read
  or write. These are different walks even for the same tensor.

With more than two live walks, the compiler would place `save_position_addrgen` / `restore_addrgen`
(`overlay/addrgen_state.hpp`) at region boundaries, where register allocation would place spills, instead of the
library's per-transfer LRU.

### Ask 3: program walks outside loops

For a loop whose page id is affine in the induction variable (`page = a*i + b`), program the walk once before the
loop for page `b`, with pop amount `a`, and emit only the pop inside the loop. This is what the library's
seek + learned stride converge to after two iterations; the compiler can do it from the first iteration and drop
the per-iteration "is this the page I expected?" check.

Needs: the accessor is not modified in the loop (it is `const` in practice), the loop's page ids stay inside what
one programming covers (the library's `run_end`; for interleaved: always; for sharded: until the next shard band),
and no other walk evicts it inside the loop (Ask 2).

Where a loop crosses shard bands (blocked and ND layouts), the reprogramming at each band stays inside the loop.
The library already computes where a band ends (`plan_cross_bank`); the compiler would split the loop there or
keep the check.

### Not asked for

- Choosing the recipe (interleaved banking, cross-bank BANK_MIDDLE, single-bank runs). It depends on the tensor's
  shape and bank map, which are run-time values in general. The library keeps that.
- Pushing addresses straight into the command buffer (`push_*`). That's a separate library step, and Ask 1's
  effects would already cover it.

## 3. Hardware and toolchain findings the implementation must respect

1. **The count-less pop builtins pass a pop amount of 0.** `__builtin_riscv_ttrocc_addrgen_pop_src(G)` and
   `pop_dest(G)` encode `xs1` set with `rs1 = x0`. On emu-quasar-2x3 that doesn't advance by one: the next address
   jumped by an arbitrary amount. The old ROCC macro passed 1. `addrgen_api.hpp` now implements
   `pop_src_addrgen<G>()` / `pop_dest_addrgen<G>()` as `pop_x_*(G, 1)`. Either the builtin or the spec's
   "0 behaves like 1" needs fixing (toolchain / Vuk).
2. **Back-to-back `rd_reg` instructions hang the address generator** (emu-quasar-2x3). The same reads spaced apart
   return correct values; a `fence` after each read avoids the hang. Any compiler-scheduled spill code must keep
   reads separated (or the hardware team should confirm whether this is an emulator artifact).
3. **Thread-local storage and the stack share 8 KB per DM core,** and the DFB and CB interfaces already take
   ~6.6 KB of it. Walker state is ~400 bytes (guarded by a `static_assert`), and a three-tensor sharded kernel
   peaks at ~690 bytes of stack with ~340 to spare. Two things cost us real stack: per-accessor-type copies of the
   seek code inlined into one kernel frame (each tensor binding is its own type), which made the frame ~750 bytes
   until the seeks were made real calls; and struct copies on the spill path. A compiler-placed spill scheme has to
   budget for both.
4. **The bank shift is shared by both sides of a generator.** One 6-bit `bank_offset` in MISC serves the source and
   destination side, while the bank order has a field per side. Walks on different sides (and any walk restored into
   any slot) therefore must agree on the endpoint shift. It's 26 for both the DRAM and worker windows on both
   current maps, so L1 reads with DRAM writes work (tested); a map with differing shifts would not. A compile-time
   check fails the build on such a map. Hardware suggestion: give `bank_offset` a field per side, like the order.
5. **Loop semantics** (checked by `AddrgenLoopProbe`): the inner loop starts at its programmed value and wraps to
   0; the outer loop keeps its programmed start as a base; `BANK_CURRENT` is relative to `BANK_BASE`; a pop amount
   of N advances through N addresses of the loop nest, including across wraps. Saving and restoring the registers
   continues a walk exactly, in either generator, including one that was just running a different walk.

## 4. Open questions

- Do Paul's binding ids stay unique across all tensors a kernel binds (they're per-binding CRTA offsets today)?
  Ask 2 relies on it.
- Is the `rd_reg` hazard (finding 2) real hardware behaviour? It sets the cost of any spill scheme.
- Should Ask 3 be a compiler transform, or a library API (`for (auto page : acc.pages())` already carries the
  affine structure) that the compiler only has to not break?
