# Metal 2.0 ProgramSpec — compute kernels, DFB memory model, precision config

Box: Blackhole **p150a** (Gen1), branch `mstaletovic/agent_eval`.
Probes: `compute/test_compute_probes.py` (34 passed, 1 xfailed), specs in `compute/specs.py`,
kernels in `compute/kernels/`. Raw run logs in `compute/agent_logs/run_final.log`.
Every probe is a real on-device run gated on numerical correctness, except the ones whose
*point* is a host-validator or JIT-compiler rejection.

---

## WIN

### W1 — `dfb::` tokens survive every template position the helper library uses
`compute_kernel_lib::reduce<>` takes its three DFB ids as `std::uint32_t` **non-type template
parameters** (params 3/4/5). Passing `dfb::name` there works — the implicit
`constexpr operator uint32_t()` is a legal user-defined conversion inside a *converted constant
expression*. Same for the dataflow side (`dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb_id, ...>`).

```cpp
compute_kernel_lib::reduce<
    ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW,
    dfb::in_tiles, dfb::scaler, dfb::out_tiles,      // <-- DFBBindingToken in NTTP position
    compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, 1));
```

It also survives **two levels** of constexpr indirection — the eltwise chain API takes a
*class-type* NTTP built by a constexpr factory that takes the id:

```cpp
square<input(dfb::in_tiles), output(dfb::out_tiles)>(IterationShape::grid(Ht, Wt).block_size(1));
```

And plain function-argument position on the raw LLK-facing API:
`matmul_block_init(dfb::in0, dfb::in1, 0, 1, 1, 1); matmul_block(dfb::in0, dfb::in1, 0, 0, 0, false, 1, 1, 1);`

**Verdict: no conversion gap found anywhere. Helpers are drop-in; no `.id` extraction needed.**
(`A`, `B`, `I2`)

### W2 — the "wrong CB index / wrong CB format" bug class is gone
`copy_compute.cpp` is **byte-identical** for bf16→bf16, bf16→fp32, and fp32→fp32. The packer
output format, the tile size and the unpack format all come from the *named* DFB spec, and
`compute_kernel_hw_startup(dfb::a, dfb::b)` cannot be given a mismatched (index, format) pair
because there is no index to get wrong. bf16-in / fp32-out is bit-exact with zero kernel change. (`M1`)

### W3 — the missing-`unpack_modes` trap is now a host error with the fix in the message
```
Compute kernel 'compute' consumes FP32 DFB 'in_tiles' with enable_32_bit_dest=true, but provides
no unpack_modes entry for this DFB. This configuration requires an explicit choice between
UnpackMode::UnpackToSrc and UnpackMode::UnpackToDest.
```
Python spelling (get this right — it is a plain dict keyed by DFB name):
```python
ttnn.ComputeGen1Config(
    enable_32_bit_dest=True,
    unpack_modes={"in_tiles": ttnn.UnpackMode.UnpackToDest},   # or UnpackToSrc
)
```
Measured on a pure fp32 tile copy:

| config | max rel err | bit-exact |
|---|---|---|
| `enable_32_bit_dest=True`, `UnpackToDest` | 0.0 | yes |
| `enable_32_bit_dest=True`, `UnpackToSrc`  | 9.3e-4 | no |
| `enable_32_bit_dest=False`, no entry      | 7.7e-3 | no |

So the demand is real and the two answers really differ. (`E1`–`E4`)

### W4 — `TT_KERNEL` on a compute kernel: CTAs as template params, and the diagnostics are excellent
```cpp
template <uint32_t do_scale, uint32_t scale_bits>   // CTAs
TT_KERNEL void ttk_compute(uint32_t num_tiles) {    // RTA
    ...
    if constexpr (do_scale != 0) { mul_unary_tile(0, scale_bits); }   // real compile-time branch
}
```
Host side is just `compile_time_args={"do_scale": 1, "scale_bits": f32_bits(4.0)}`. Both branches
verified numerically. No `get_compile_time_arg_val(N)` index bookkeeping, no `#define`, and
`if constexpr` on a CTA gives genuine dead-code elimination. Rename one CTA and you get:
```
TT_KERNEL entry 'ttk_compute': template parameters do not match the registered compile-time arguments.
  template parameter(s) with no matching registered compile-time argument: scale_bits
  registered compile-time argument(s) not taken as a template parameter: scale_bitz
```
**This is the best error message in the whole surface.** (`H`, `H2`)

### W5 — `alias_with` really does save L1, and the legality messages are precise
24 same-size dead DFBs of 96 KB each on one core:
- not aliased → `Statically allocated dataflow buffers on core range [0-0 - 0-0] grow to 2478976 B which is beyond max L1 size of 1572864 B`
- aliased into one clique → **fits and runs correctly.**

All four ways of getting the declaration wrong are caught on the host with the rule spelled out:

| what I broke | message |
|---|---|
| one member drops the back-edge | `DFBs 'in_tiles' and 'out_tiles' do not declare the same alias group. Every DFB in an alias group must list every other member in its alias_with field.` |
| different total size | `Aliased DFBs 'in_tiles' and 'out_tiles' have different total sizes (4096 vs 6144 bytes). Aliased DFBs must have the same total size (entry_size * num_entries).` |
| self-reference | `DFB 'in_tiles' lists itself in alias_with` |
| unknown name | `DFB 'in_tiles' lists unknown alias 'nope' in alias_with` |

(`D2`, `D3`)

### W6 — the self-loop DFB is real and it is the clean answer to "packer writes the output, nobody drains it"
One compute kernel bound as **both** PRODUCER and CONSUMER of `resident_out`, under **one**
accessor name, with `borrowed_from` pointing at the L1-resident output tensor. No writer kernel
at all; the compute kernel recycles its own credits. Output verified bit-exact.
```python
dfb_bindings=[
    ttnn.consumer_of("in_tiles", "in_tiles"),
    ttnn.producer_of("resident_out", "resident_out"),
    ttnn.consumer_of("resident_out", "resident_out"),   # same accessor name
]
```
```cpp
DataflowBuffer resident(dfb::resident_out);   // one object, both roles
```
The validator explicitly sanctions exactly this shape and rejects the near-miss (two bindings of
the *same* role under different names) with a message that names the correct alternative:
`... To refer to one buffer by multiple names in kernel code, alias the handle (constexpr auto x = dfb::y) instead of adding a second binding.` (`G`, `G2`)

### W7 — a compute kernel CAN reach resident tensor memory, via `LocalTensorAccessor`
The upstream line "a compute kernel cannot bind a TensorAccessor" is only half the story: the
**tensor binding itself is legal on a compute kernel**, and `LocalTensorAccessor<T>` is NoC-free
so it compiles on TRISC. This is the working way for a compute kernel to read a device-resident
runtime scalar (e.g. an SFPU multiplier computed by a previous op) without a CB:
```cpp
const LocalTensorAccessor<uint32_t> scale(tensor::scale);   // L1-resident TensorParameter
const uint32_t scale_bits = scale[0];
...
mul_unary_tile(0, scale_bits);
```
Verified numerically against `in * 2.5`. The DRAM case is a hard `static_assert` in the ctor, and
the host rejects a DRAM `borrowed_from` with
`DFB 'in_tiles' borrows memory from TensorParameter 'src', but its TensorSpec is not L1-resident`. (`F2`, `F4`)

### W8 — `ScratchpadSpec` works on a compute kernel and is exactly "private L1 that is not a FIFO"
It is undocumented outside `tt_metal/hw/inc/api/scratchpad.h` + `scratchpad_spec.hpp`, but it is
complete and it works on TRISC. Program-scope, allocated from the same L1 region as DFBs, one
instance per node, raw and uninitialized, `operator[]` bounds-checked under watcher, size known at
compile time from the binding token.
```python
scratchpads=[ttnn.ScratchpadSpec(unique_id="scale_table", size_per_node=16)]
# on the kernel:
scratchpad_bindings=[ttnn.ScratchpadBinding("scale_table", "scale_table")]
```
```cpp
Scratchpad<uint32_t> table(scratch::scale_table);
for (uint32_t i = 0; i < table.size(); ++i) { table[i] = ...; }   // random access, no credits
mul_unary_tile(0, table[i]);
```
Verified: a compute kernel builds a per-tile scale table in its scratchpad and applies it
(`out[i] == in[i] * (i+1)`). This is the right tool for a compute kernel that needs a small LUT,
a spill slot, or a persistent counter — things a CB cannot express at all. It is also accounted in
the same L1 budget (a 2 MB request produces the same `grow to 2216832 B which is beyond max L1 size
of 1572864 B` throw), so it is not free memory. (`C`, `C5`)

---

## UGLY

### U1 — the compute-kernel scratchpad is ONE region shared by three RISC-V cores, and nothing says so at the call site
`kernel_main()` is compiled three times (UNPACK / MATH / PACK) and **all three builds receive the
same binding token, hence the same L1 address**. Proven on device (`C4`): the UNPACK thread stamps
a sentinel at kernel entry, the MATH thread reads it back after `copy_tile` (where MATH
synchronizes with UNPACK, so the read is ordered) and scales by 3.0 only if it sees it — output is
exactly `in * 3.0`.

The consequence is that the natural-looking table fill in `C`:
```cpp
for (uint32_t i = 0; i < table.size(); ++i) { table[i] = bits(i + 1); }
```
actually executes **three times over the same L1**, concurrently. It is correct there only because
the writes are value-identical and idempotent. Anything stateful — a counter, a queue index, a
scratch spill — needs an explicit `UNPACK()`/`MATH()`/`PACK()` guard, and there is **no barrier**
to order one thread's write against another thread's read except the implicit
`tile_regs_acquire/commit/wait` pipeline handshake. The header's `CAUTION` block says this; the
Python API surface (`ScratchpadSpec(unique_id, size_per_node)`) says nothing.

**What I wanted to write:** a per-thread scratchpad (`ScratchpadSpec(..., per_thread=True)`), or at
minimum a `Scratchpad<T>` whose `operator[]` refuses to compile outside a thread guard.

### U2 — a compute kernel building a `TensorAccessor` fails as a 40-error header cascade
This is the documented-illegal case, so *that* it fails is right. But the diagnostic is:
```
error: 'NOC_INDEX' was not declared in this scope; did you mean 'NOC_MODE'?
error: ambiguating new declaration of 'uintptr_t get_common_arg_addr(int)'
error: expected primary-expression at end of input [-Wtemplate-body]     (x ~35, across
       pages_address_iterator.h / shard_pages_address_iterator.h / dspec.h)
```
40 `error:` lines, none of which mention TRISC, NoC availability, or `LocalTensorAccessor`.
`#include "api/tensor/tensor_accessor.h"` simply does not survive `-DCOMPILE_FOR_TRISC`.
Compare `local_tensor_accessor.h`, which *does* carry a real `static_assert` for its DRAM misuse.

**What I wanted:** `#if defined(COMPILE_FOR_TRISC) #error "TensorAccessor needs the NoC; a compute
kernel must use LocalTensorAccessor (api/tensor/local_tensor_accessor.h)." #endif` at the top of
`tensor_accessor.h`. One line. (`F1`)

### U3 — `alias_with` gives you a shared address space and *zero* ordering, and the rules that are enforced are not the rules that keep you safe
The four enforced rules (clique, equal total size, equal node coverage, consistent `borrowed_from`)
are all *static shape* rules. None of them has anything to do with the actual hazard, which is
**lifetime**: two aliased DFBs keep independent credit counters, so `reserve_back` on B succeeds
while A still holds live data in the same bytes.

Concretely: my aliased in/out copy (`D1`) is correct **only** because
`tile_regs_commit()/tile_regs_wait()` orders the MATH-side unpack read of `in_tiles` before the
PACK-side write of `out_tiles`, and because reader→writer never laps. Grow it and it silently
corrupts: with `entries = N` for both, the reader can reserve `in_tiles` slot *j* (credit freed by
compute's `pop_front`) while the writer has not yet read `out_tiles` slot *j* — the same L1 bytes.
Nothing on the host or the device notices.

**What I wanted:** the alias declaration to require a *phase* (`alias_with=[...], phase=0/1`) that
the framework can turn into a real handshake, or at least a doc line at the Python binding saying
"you own the ordering; the enforced rules do not give you any."

### U4 — upstream says aliased DFBs must have the same bound kernels; the implementation only checks the same *nodes*
`D4` aliases `in_tiles` (reader → compute) with `pad0` (reader → writer). Different consumer
kernels, same core. **Accepted, runs.** The relevant validator rule
(`program_spec.cpp:1632-1642`, "Rule 3: same node coverage") is worded as node coverage, not kernel identity. Either the docs or the
validator is wrong; I would rather the docs relax, since same-node/different-kernel aliasing is
genuinely useful (that is how you alias a reader staging buffer against a writer drain buffer).

### U5 — `PoolType` / `ReduceDim` are unqualified in a compute kernel but not in a dataflow kernel
The exact same helper-library call has to be spelled differently on the two sides:
```cpp
// dataflow kernel: MUST qualify
dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
```
Unqualified gives `error: 'PoolType' was not declared in this scope` followed by
`parse error in template argument list` and `template argument 2 is invalid` — three errors for one
missing namespace, and none of them says "add `ckernel::`" (gcc's `note: suggested alternatives:
'ckernel::PoolType'` does, buried). Compute kernels get `ckernel` in scope for free, so a
copy-paste from a compute kernel to a reader breaks. `reduce_helpers_dataflow.hpp` does
`using ckernel::PoolType;` *inside* `namespace dataflow_kernel_lib`, so it is reachable but only
under that qualification. (`A`)

### U6 — the Python bindings hand you a list of references; rebinding a slot is silently lost
```python
spec.dataflow_buffers[0].num_entries = 99                     # WORKS (mutates the C++ spec)
spec.dataflow_buffers[0] = ttnn.DataflowBufferSpec(...)       # SILENTLY DISCARDED
```
Confirmed via `ttnn.compute_program_spec_hash` before/after: the field write changes the hash, the
slot write does not, and neither raises. `def_rw` on a `std::vector<BoundClass>` casts to a *new*
Python list whose elements are references into the C++ vector — so half of the obvious Python
idioms work and half are no-ops. This cost me one wrong "DID NOT RAISE" result before I noticed.
Worse, some fields are read-only (`data_format_metadata` is `def_prop_ro`), so the only way to
change a DFB's format after construction is to rebuild the whole `ProgramSpec`.

**What I wanted:** either a real mutable-sequence proxy, or `def_prop_ro` on the vectors so that
slot assignment raises. (`J`)

### U7 — `validate_program_args` is **off by default**, and toggling it mid-session is not reliable
`ttnn/api/ttnn/config.hpp:31`: `bool validate_program_args = false;  // Off by default; CI turns it on.`
Every good error message in this document — W3, W5, W6, `F4`, `E5`, `K` — is gated on that flag.
Run without it and an fp32 + `enable_32_bit_dest=True` + no-`unpack_modes` spec **just runs**,
silently taking `UnpackToSrc` (measured 9.2e-4 rel err on a copy). Reproduced in isolation (`L`).

There is also an inconsistency I could not fully explain and am reporting as observed, not
diagnosed: in a *full-file* run, setting `ttnn.CONFIG.validate_program_args = False` mid-session
(flag readback confirms `False` immediately before the call) still hits
`ValidateProgramSpec` — the same test passes the other way when run alone with `-k test_L`.
So the flag is not purely "read at program create" in every path.

---

## BROKE THE MODEL

### B1 — a borrowed-memory DFB needs a producer kernel that produces nothing
`borrowed_from` is the clean, zero-copy way to point a DFB at an L1-resident tensor. But the
validator requires **≥1 PRODUCER and ≥1 CONSUMER for every DFB**, and for already-resident data
there is no producer. So I had to write this kernel, and put it in the spec, and give it a
work-unit slot on the core, purely to hand out credits:
```cpp
void kernel_main() {
    DataflowBuffer in(dfb::in_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        in.reserve_back(1);
        // No write. The bytes are already there.
        in.push_back(1);
    }
}
```
The ProgramSpec now asserts a producer/consumer dataflow edge that does not exist, and burns a DM
RISC on it. (`F3`)

**What I wanted to write:** a third endpoint role, e.g.
`ttnn.resident_source_of("in_tiles", "in_tiles")` on the *consumer* kernel, or a
`DataflowBufferSpec(..., borrowed_from="src", prefilled=True)` flag that pre-credits the DFB at
program start and exempts it from the producer requirement. The self-loop pair (`W6`) is the
sanctioned dodge on the *output* side; there is no equivalent on the input side.

### B2 — `interm_buf` on `matmul_block` almost forces a phantom DFB
`compute_kernel_lib::matmul_block` takes four buffers and the fourth (`interm_buf`) is unused when
`num_k_blocks == 1`. The helper's contract says to pass `out_buf` as the placeholder, which is
fine — but only because the helper documents it. The general shape ("this API needs a buffer
argument it will not touch") collides head-on with the validator's every-DFB-needs-both-roles rule:
had I passed a distinct scratch DFB I would have had to invent a fake producer *and* a fake
consumer for it, exactly like B1. Worth watching as more helpers get ported.

### B3 — no cross-kernel private L1 on one core; the "shared scratch" pattern has to become a DFB
Two kernels on the same node cannot share a scratchpad:
```
ScratchpadSpec 'scale_table' is bound by 2 kernel instances on node 0-0 ('reader', 'compute').
A scratchpad is private node-local L1; multiple kernels may bind the same scratchpad only on
disjoint nodes ... Sharing one node's scratchpad across kernels is not yet supported
```
So "the reader computes a table once, the compute kernel reads it" cannot be expressed as private
memory — it has to be a 1-entry DFB, which drags in credits, a `push_back`/`wait_front` handshake,
and a `data_format`. The message says "not yet supported", so this looks like a planned
`AdvancedOption`; it is the single most-wanted missing piece for reader↔compute cooperation.
(`C3`)

---

## BLOCKED

### N1 — a compute kernel cannot bind a semaphore, full stop
```
KernelSpec 'compute' has semaphore bindings. Semaphore bindings are not supported for compute kernels.
```
(`program_spec.cpp:1088`, comment: "There's no use case for ever wanting this, so best just forbid
it.") Any compute-side cross-core coordination has to be routed through a DM kernel. Not a
regression from ProgramDescriptor, but it is a hard wall, and combined with B3 it means a compute
kernel's only channels to the rest of the core are DFBs, CTAs/RTAs, scratchpad (private), and
`LocalTensorAccessor` (raw L1). (`K`)

### N2 — nothing validates a DFB's declared `data_format` against the actual bytes
Declaring `data_format=ttnn.float32` on a DFB whose tensor is bf16 (`entry_size` left at the bf16
tile size) is accepted by the host and produces silent garbage: **max abs diff 4.9** on a copy that
should be exact. The DFB format is a user assertion, and it is the one format-mismatch bug class
Metal 2.0 does *not* close — W2 removes the (index, format) pairing error but not the
(DFB, tensor) agreement error. There is no `TensorParameter`-derived format helper in the Python
surface; `entry_size`/`data_format` are hand-computed integers next to a `TensorParameter` that
already knows both. (`M2`)

---

## Side finding (not Metal 2.0)

`ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.{hpp,inl}` **does not compile at all** on this
branch, from any host model:
```
matmul_block_helpers.inl:184: error: 'mm_block_init_short' was not declared in this scope
matmul_block_helpers.inl:337: error: 'mm_block_init_short_with_dt' was not declared in this scope
```
`api/compute/matmul.h` now exposes `matmul_init`, `matmul_tiles`, `matmul_block_init`,
`matmul_block` — there is no `_short` variant and no `mm_*` alias anywhere in `tt_metal/`
(only under `models/demos/deepseek_v3_b1/kernel_includes/`). The helper is stale against the
renamed compute API. Recorded as `test_I_matmul_block_helper_with_dfb_objects`
(`xfail(strict=True)`); the raw-API equivalent (`I2`) passes, so this is a helper-library problem,
not a DFB/Metal 2.0 one.

---

## What I'd want from the API

1. **A resident/prefilled DFB endpoint role** so `borrowed_from` inputs stop needing a no-op
   producer kernel (B1). Highest-value item on this list.
2. **Same-node scratchpad sharing** (the validator already says "not yet supported") so
   reader↔compute cooperation does not have to become a FIFO (B3).
3. **A phase or handshake on `alias_with`**, or an explicit "you own the ordering" contract at the
   Python binding — the four enforced rules protect nothing that actually bites (U3).
4. **`#error` in `tensor_accessor.h` under `COMPILE_FOR_TRISC`**, pointing at
   `LocalTensorAccessor`. Turns 40 errors into 1 (U2).
5. **Derive `entry_size` / `data_format` from a `TensorParameter`** —
   `ttnn.DataflowBufferSpec.for_tensor("in_tiles", tp="src", num_entries=2)` — closing the last
   format-mismatch bug class (N2).
6. **Make `validate_program_args` default on**, or make the run-args path fail loudly when it is
   off. Right now the entire validator is opt-in and CI-only (U7).
7. **A per-thread scratchpad**, or a compile-time guard on `Scratchpad::operator[]` in compute
   kernels (U1).
8. **Real mutable sequences (or read-only ones) for `ProgramSpec` vectors** in the nanobind layer;
   silent no-op slot assignment is a trap (U6).
9. **`using ckernel::PoolType/ReduceDim` at global scope in the dataflow kernel prelude**, so the
   same helper call reads identically from a reader and from a compute kernel (U5).
