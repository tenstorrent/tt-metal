# 09 — Debugging

*There is no segfault, no exception, and no stack trace. Here's what to do
instead.*

---

## The three failure modes

Almost everything that goes wrong is one of:

1. **It hangs** — the program stops and never finishes.
2. **The numbers are wrong** — it completes, the output is garbage or subtly off.
3. **It won't compile** — the easy one, and the only one with a decent error
   message.

Each has a distinct set of causes. Identify which you're in first.

---

## 1. It hangs

**Cause: essentially always circular-buffer accounting.**

Two processors are waiting for each other, and neither will move (chapter 01,
section 6). Nothing crashes.

The dojo sets a 30-second timeout so this reports as an error rather than
freezing your terminal:

```
1 tile     FAIL  device hang (deadlock)
```

The device recovers on its own; just fix the bug and run again.

### The checklist

Work through these in order:

| Check | Failure looks like |
|---|---|
| Every `cb_reserve_back` has a matching `cb_push_back` | Hangs after a few iterations as the buffer fills |
| Every `cb_wait_front` has a matching `cb_pop_front` | Hangs after a few iterations |
| The **counts** match: reserving 1 and pushing 2 | Hangs once the miscount accumulates |
| You never wait for more pages than the CB holds | Hangs immediately, first iteration |
| All kernels agree on how many items they process | Hangs at the end, after most work is done |
| All four `tile_regs_*` calls are present each iteration | Hangs immediately in a compute kernel |

### The mechanical test

For each circular buffer, count over the whole run:

- total pages pushed by the producer
- total pages popped by the consumer

If those differ, that's your bug. This is more reliable than reading the code
looking for something wrong, because the error is usually in the *arithmetic*,
not the structure.

### Watcher

For hangs you can't find by inspection, tt-metal's watcher reports which core is
stuck and where:

```bash
TT_METAL_WATCHER=1 ./dojo test 05
```

It also catches NoC address violations and CB overflow, and names the offending
core and kernel. Turn it on the moment something hangs and you don't immediately
see why. It slows things down, so turn it off again afterwards.

---

## 2. The numbers are wrong

Read the grader output first — it tells you a lot about *which* kind of wrong.

```
FAIL  pcc=0.007616 max_abs_err=4.093 mismatched=100.00%
      first mismatch at (0, 0, 0, 0): got 0.0402832, expected -0.111816
```

### Interpreting PCC

**PCC** is how well your output correlates with the reference, from -1 to 1.

| PCC | Meaning | Look at |
|---|---|---|
| ~0 (or negative) | Output is unrelated to the answer | Is anything being computed at all? Missing barrier? Kernel not running? |
| 0.5 – 0.99 | Structurally right, substantially wrong | Indexing bug affecting many elements; wrong operand order |
| 0.999+ but tolerance fails | Nearly right | Precision issue, or a small region wrong |
| Exactly 1.0 but tolerance fails | Impossible — investigate the harness | |

**`mismatched=100%` with PCC near 0** usually means you're reading uninitialised
memory — the output was never written, or was written before the data arrived.

**`mismatched=100%` with PCC near 1** means a systematic offset or scale — often
the right computation on the wrong data.

**A small `mismatched` percentage** means an indexing bug affecting part of the
output. The reported first-mismatch coordinate tells you which part; if it's at
the start of a tile boundary, suspect your tile index arithmetic.

### The usual suspects

| Symptom | Cause |
|---|---|
| All zeros, or stale/random values | Missing `noc_async_read_barrier()` before using data |
| Output partly correct, partly stale | Missing `noc_async_write_barrier()` before `cb_pop_front` — the page was recycled mid-write |
| First tile right, rest are copies of it | `get_write_ptr` called inside the fill loop instead of once before it |
| First item of a block right, rest wrong | CB-relative index hard-coded to 0 instead of `t` |
| Matmul: plausible but wrong matrix | `SrcOrder::Regular` where `Reverse` was needed |
| Matmul: wrong shape of wrongness | Transposed page index — `k * Nt + n` vs `n * Kt + k` |
| Results drift as the reduction gets deeper | Packing inside the K loop, rounding every partial sum |
| Works at size 1, wrong at larger sizes | Loop bound or offset bug that only shows once there's more than one iteration |
| Only some cores' output is wrong | Runtime args wrong for those cores; or kernel placed on cores with no args |

### Shrink the case

The dojo's cases go from 1 tile upward for exactly this reason:

```bash
./dojo test 06 --case "1x1x1"
```

If the smallest case passes and a bigger one fails, the bug is in your
*iteration* — offsets, block handling, loop bounds — not in the core
computation. That halves the search space immediately.

### Compare against the reference

```bash
./dojo solution 06
```

Prints the reference implementation. Diff it against yours mentally, or:

```bash
diff exercises/06_matmul/kernels/compute.cpp exercises/06_matmul/solution/compute.cpp
```

Reading the solution isn't cheating if you then understand *why* the difference
matters. That's the whole point of having it.

---

## 3. Printing from a kernel

`DPRINT` is printf for kernels. It genuinely works, and sometimes it's the only
way to see inside.

```cpp
#include "api/debug/dprint.h"

DPRINT << "tile " << i << " addr " << addr << ENDL();
```

Run with the cores you want output from:

```bash
TT_METAL_DPRINT_CORES=0,0 ./dojo test 01
```

Caveats worth knowing:

- **It's slow**, and it perturbs timing. Never leave it in while benchmarking.
- **Enable one or two cores**, not all 64, unless you enjoy reading 64
  interleaved streams.
- Variants like `DPRINT_DATA0` restrict output to a specific processor, which
  helps when a compute kernel prints three times (once per TRISC build).

### Reading L1 directly

Circular buffers are just addressable memory, so you can inspect them:

```cpp
volatile tt_l1_ptr uint16_t* p = (volatile tt_l1_ptr uint16_t*)get_read_ptr(cb);
DPRINT << "first value: " << p[0] << ENDL();
```

The RISC-V cores are slow at this, so it's a debugging tool only — never a way
to compute things. But for answering "did the data actually arrive?" it's
decisive.

---

## 4. Compile errors

The one case with real error messages. Two notes:

**Your editor's errors are not the compiler's errors.** clangd doesn't know
tt-metal's kernel include paths, so it will flag `'api/dataflow/dataflow_api.h'
file not found` and every API call as undeclared, in every kernel file. This is
expected and harmless. The real compile happens at run time under a different
toolchain.

If you want working IDE support, tt-metal can generate a kernel
`compile_commands.json`:

```bash
./build_metal.sh --enable-fake-kernels-target
```

**Real kernel compile errors appear in the `dojo test` output**, in among the
JIT compiler's noise. Look for `error:` specifically — the profiler build also
emits a lot of `note: #pragma message` lines that are not problems.

---

## A general approach

1. **Which failure mode?** Hang, wrong numbers, or won't compile.
2. **Smallest failing case.** `--case "1 tile"`. If small passes and large
   fails, it's iteration logic.
3. **Check the mechanical things first.** Barriers, CB counts, index bases.
   These account for the large majority of bugs and cost nothing to verify.
4. **Only then reason about the algorithm.**
5. **When stuck, `DPRINT` one core** and confirm what's actually in the buffer,
   rather than what you believe is in it.

The recurring theme: on this hardware you get no help from the runtime, so
verify assumptions explicitly instead of inferring them from behaviour.

---

**Back to:** [the theory index](../THEORY.md) · [the main README](../README.md)
