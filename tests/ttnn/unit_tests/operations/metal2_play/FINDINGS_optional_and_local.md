# Findings — optional DFBs, local L1, TT_KERNEL/NTTP, validator behaviour

Probes: `me/play_spec.py`, `me/test_play.py`, `me/kernels/`. **16/16 green** on Blackhole P150b.
Captured error text: `VIOLATION_MESSAGES.txt`.

---

## BLOCKED

### 1. A DFB cannot be a NoC *unicast destination* — so DFB→DFB is not expressible
The obvious spelling of a local L1→L1 copy:
```cpp
noc.async_write(dfb_out, dfb_stage, tile_bytes, {.offset_bytes = 0}, {.offset_bytes = 0});
```
fails to compile:
```
dataflow_buffer.h:392: error: static assertion failed:
    DataflowBuffer without mcast range can only be used as L1 destination
note: '(Noc::AddressType::NOC == Noc::AddressType::LOCAL_L1)' evaluates to false
```
`Noc::async_write` resolves its destination as `AddressType::NOC` (`noc.h:357`), but
`noc_traits_t<DataflowBuffer>::dst_addr` only accepts `LOCAL_L1`. A DFB *is* a legal
`async_write` **source**, and *is* a legal **multicast** destination (`dst_addr_mcast`
asserts `NOC`) — it is only the unicast-destination slot that has no path.

**Verdict: BLOCKED — the one topology you cannot name is the local one.** Multicasting a
tile to 64 cores is a one-liner; handing it to your own next buffer is not.

The message is also actively misleading: it says a DFB "can only be used as L1 destination"
at the exact moment it is being rejected *as* a destination. It should say the DFB is not a
valid unicast destination and name the two routes below.

---

## UGLY

### 2. Local L1 copy: two working routes, both leaving the model
Since route (A) is blocked, the alternatives:

**Route B — NoC loopback to my own (x,y).** Works, correct.
```cpp
UnicastEndpoint self;
noc.async_write(dfb_out, self, tile_bytes, {.offset_bytes = 0},
                {.noc_x = my_x[0], .noc_y = my_y[0], .addr = dfb_stage.get_write_ptr()});
```
You must peek the destination DFB's raw write pointer to build the endpoint — the named
binding gets you nothing on the destination side. It also pays a full NoC round trip for a
copy that never leaves the core.

**Route C — no NoC at all.** Works, correct, and is the one I'd actually ship.
```cpp
CoreLocalMem<uint32_t> src(dfb_out.get_read_ptr());
CoreLocalMem<uint32_t> dst(dfb_stage.get_write_ptr());
for (uint32_t w = 0; w < tile_bytes / sizeof(uint32_t); ++w) dst[w] = src[w];
```
`CoreLocalMem<T>` (`api/core_local_mem.h`) is the sanctioned typed view over a raw local
address — `scratchpad.h:144` names it explicitly as the escape for "arbitrary address/size".
It is `DEBUG_SANITIZE_L1_ADDR`-checked on `operator[]`, so it is not a bare pointer.

**Verdict: UGLY, not broken.** Both routes are legitimate and use whitelisted `get_*_ptr`
peeks. But "copy L1 to L1" — the cheapest data movement on the chip — is the one operation
with no named-binding expression, and both answers start by dropping to an address.

### 3. A conditional binding has to be threaded through six places in the host builder
For one optional DFB (`out2`) the Python spec code needs a conditional at:
1. the `dataflow_buffers` list, 2. the `tensor_parameters` list, 3. the compute kernel's
`dfb_bindings`, 4. the writer's `dfb_bindings`, 5. the writer's `tensor_bindings`,
6. the `defines` dict — **on every kernel that names the token**, not just one.
Plus the `generic_op` io list and its name→index mapping. Nine edit sites for one flag.

Miss any one and you get a different error at a different stage. There is no
"optional resource" grouping — nothing ties the DFB, its tensor parameter, its bindings and
its define together as one conditional unit.

**Verdict: UGLY, and the single biggest authoring hazard I hit.** For our agents this is
exactly the kind of scattered invariant they get wrong. A tiny host-side helper that takes
`(condition, dfb_spec, [(kernel, role, name)], define)` and fans it out would remove the
whole class.

### 4. `TT_KERNEL` accepts only `uint32_t` — no `bool`, no enum
```
TT_KERNEL template parameter 'touch_optional' has unsupported type 'bool'
    (only uint32_t is supported in Phase 1)
```
Every flag becomes `uint32_t` and every test becomes `if constexpr (flag != 0)`.
**Verdict: UGLY but self-announcing.** Excellent error message; costs one line of noise.

---

## BROKE THE MODEL

Nothing. I did not have to smuggle an address through a runtime arg, fake an endpoint role,
or extract `.id` anywhere. Every route I needed had a sanctioned spelling, even when ugly.
The `get_*_ptr` peeks in §2 are explicitly whitelisted, not a defeat.

---

## WIN

### 5. Conditional binding really does save the L1 — measured
| config | max `out2` depth that still builds |
|---|---|
| `out2` bound (`#ifdef` fused build) | **707 entries = 1414 KB** |
| `out2` not bound (unfused build) | unbounded — the DFB does not exist |

Worker L1 unreserved = 1,532,032 B (1496 KB). The bound DFB is charged essentially exactly
(1414 KB + the 6 live tiles + ~70 KB overhead ≈ the budget).

**So "just always bind it and gate on a CTA" is not a free simplification** — it costs the
full buffer. Both gating styles work and produce identical numbers; only the L1 differs.
This settles the guidance: for an L1-tight op the conditional binding is mandatory, and
`/memory-budget-metal` should say so with this number.

### 6. A DFB declared but bound by nobody is rejected
```
program_spec.cpp:402: DFB 'never_bound' is defined but not bound by any kernel
```
I declared a 200 MB DFB no kernel touched; the spec refused rather than silently allocating
or silently ignoring. Dead buffer declarations — a classic slow L1 leak under
`ProgramDescriptor`, where a stale `CBDescriptor` just sits there — are now impossible.

### 7. The endpoint validator catches half-declared topologies, precisely
```
program_spec.cpp:394: DFB 'out' has no consumer
program_spec.cpp:394: DFB 'stage' has no consumer      # self-loop with only the PRODUCER half
```
Both fire at spec-build time, before any kernel compiles, and name the buffer. Under
`ProgramDescriptor` the equivalent mistake is a **hang** you triage from a callstack.
**This is the single biggest win on the list**: an entire class of CB-topology bug moved
from runtime hang to host-side error with the buffer's name in it.

### 8. One error message teaches the fix inline
```
Kernel 'reader' has two PRODUCER bindings to DFB 'in_a' under different accessor names.
Within a kernel a DFB may be bound at most once per role (the only multi-binding form is
the self-loop pair: one PRODUCER + one CONSUMER). To refer to one buffer by multiple names
in kernel code, alias the handle (constexpr auto x = dfb::y) instead of adding a second
binding.
```
Rule, rationale, and the correct alternative in one message. This is the standard the rest
of the diagnostics should be held to.

### 9. Self-loop DFBs work, including at depth 1
The writer bound PRODUCER+CONSUMER on one staging DFB under a single accessor name, correct
at `num_entries=2` and at `num_entries=1` (reserve/push then wait/pop in the same iteration).
Confirms the upstream claim that a self-loop is the sanctioned Gen1 shape, and it is the
mechanism that makes local staging expressible at all.

### 10. `TT_KERNEL` NTTP CTAs give real compile-time branching, on a DM kernel, from Python
```cpp
template <uint32_t tiles_per_iter, uint32_t touch_optional>
TT_KERNEL void read_pairs(uint32_t num_tiles, uint32_t start_id) { ... }
```
`if constexpr (tiles_per_iter == 1)` selects a genuinely different loop body; both
instantiations correct. No `kernel_main()`, no `get_arg` calls, no positional anything —
the arguments *are* the parameters. This is the syntax we should author in.

### 11. Signature ↔ schema drift is caught by name, at build time
```
TT_KERNEL entry 'read_pairs': template parameters do not match the registered
compile-time arguments.
  template parameter(s) with no matching registered compile-time argument: touch_optional
```
Host and kernel can no longer disagree about the argument list. Under positional CTAs this
is a silent wrong-value bug.

---

## The `#ifdef` question, settled in the real JIT

Issue #52179's thread claims merged PR #46623 (NTTP CTAs) removes the need for `#ifdef` on
conditionally-bound resources. **It does not.** With a `TT_KERNEL` NTTP template, naming an
unbound token inside a discarded `if constexpr` branch:
```
reader_ttk.cpp:27:37: error: 'out2' is not a member of 'dfb';
    did you mean 'tt::out2'? [-Wtemplate-body]
```
The `[-Wtemplate-body]` tag is the confirmation: this is two-phase lookup resolving a
**non-dependent** name at template *definition* time. `if constexpr` suppresses
instantiation, never lookup. NTTPs fix conditional *code*; they cannot conjure a token the
JIT never emitted, because whether `dfb::out2` exists was decided when genfiles wrote
`kernel_bindings_generated.h`.

**Footgun found while proving it:** the compiler's suggestion is real. `tt::out2` is a
legacy CB-index enum (`kernel_structs.h:155`, `out2 = 18`). A kernel with `using namespace tt;`
that names an unbound `out2` would silently compile against **CB index 18** instead of
failing. Worth a rule: never `using namespace tt;` in a Metal 2.0 kernel.

## What I'd want from the API
1. A DFB as a legal unicast destination — or an explicit `dfb_local_copy(src, dst, bytes)`.
2. A host-side "optional resource" bundle so one flag doesn't fan out to nine edit sites.
3. The unbound-token error to name the fix (`#ifdef`, or a host binding you forgot) the way
   the two-accessor-names error does.
4. `bool` template parameters in `TT_KERNEL`.

---

# Round 2 — the `split_work_to_cores` two-group shape (`me/test_groups.py`, 3/3 green)

This is the shape **every** op we generate uses, so it mattered more than anything else here.

### 12. WIN — two core groups with different per-group CTA works, and the CTA stays a CTA
200 tiles over a 130-core grid → `g1 = 90 cores × 2 tiles`, `g2 = 20 cores × 1 tile`.
Two `KernelSpec`s of **one source** (`compute_grouped.cpp`) differing only in
`compile_time_args={"tiles_per_core": n}`, each in its own `WorkUnitSpec` over disjoint
nodes. Reader and writer are listed in **both** work units — a kernel in two WUs composes
exactly as documented, and its effective node set is the union.

`tiles_per_core` stays a genuine compile-time constant, so the loop still unrolls. The
upstream "demoting per-group CTA to RTA" anti-pattern is avoidable from Python, with no
tricks. **Our whole op corpus's work-split idiom ports 1:1.**

### 13. WIN — overlapping placement is caught at spec build, with the best message in the API
Making g2's work unit cover g1's nodes (two computes on one node):
```
program_spec.cpp:1406: Local DFB 'in_a' is malformed at node 0-0: 1 producer instance(s)
('reader') and 2 consumer instance(s) ('compute_g1', 'compute_g2'). A local DFB lives in
shared SRAM on each node, so every node it is instantiated on must run exactly one producer
and one consumer kernel instance. Multiple same-role kernel instances land on this node —
their placements overlap; give each disjoint nodes.
```
Names the buffer, the **node coordinate**, the offending instances *by kernel id*, the
hardware reason, and the fix. This is the standard to hold the rest to.

### 14. Runtime varargs work from Python
`ttnn.KernelAdvancedOptions(num_runtime_varargs=2)` on the schema plus
`ttnn.AdvancedKernelRunArgs(runtime_varargs={core: [v0, v1]})` on the run args, read with
`get_vararg(0)` / `get_vararg(1)`. Correct on device. (I used it for two scalars that should
have been named args — deliberately the wrong choice — purely to prove the plumbing.)
**Verdict: WIN (it works), but nothing stops the wrong choice.** Nothing warns that a
distinct field read once belongs in `runtime_arg_names`; a mechanical author will reach for
varargs because it needs no schema entry. That's a rule for our reference, not an API gap.

### 15. Measured: every distinct CTA combination is a new program-cache entry
| run | program cache entries |
|---|---|
| baseline | 2 |
| 200 tiles → CTA (2, 1) | 2 |
| 300 tiles → CTA (3, 2) | 3 |
| 401 tiles → CTA (4, 3) | 4 |
| 3 reps of an already-seen shape | 4 (**delta 0**) |

A CTA's *value* is written into `kernel_args_generated.h`, so it changes the kernel hash and
forces a rebuild; an RTA's does not. Re-running a known shape adds nothing — the cache-hit
tensor-binding refresh works as advertised.

**Consequence for us:** `tiles_per_core` as a CTA buys loop unrolling and costs one JIT
compile *per distinct shape*. Given that compile is 50-80% of our eval wall-clock, that is a
real trade, not a free win — and it is invisible in the source, because the kernel reads CTAs
and RTAs through the identical `get_arg(args::name)`. Our reference must name it: **a value
that varies per shape belongs in an RTA unless the unroll is measured to pay for the rebuild.**
