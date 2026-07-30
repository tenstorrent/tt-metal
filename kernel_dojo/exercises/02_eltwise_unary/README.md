# 02 — Element-wise unary: your first compute kernel

**Goal:** compute `exp(x)` on the device. Same data movement as lesson 01, plus
a third kernel that actually does maths.

> **Background:** [`theory 06 — Compute`](../../theory/06-compute.md) covers
> everything in this lesson in more depth, including the full list of SFPU
> operations and why the init calls exist.

---

## Theory

### The compute pipeline is three processors deep

You write *one* compute kernel source file, but it is compiled **three times**,
once for each of TRISC0, TRISC1, TRISC2. Macros select which parts of the code
survive in each build:

```
       CB in L1
          │
      ┌───▼────┐
      │ UNPACK │  TRISC0 — moves tiles from L1 into SrcA/SrcB registers
      └───┬────┘
      ┌───▼────┐
      │  MATH  │  TRISC1 — runs FPU/SFPU, result lands in DST registers
      └───┬────┘
      ┌───▼────┐
      │  PACK  │  TRISC2 — moves DST back out to a CB in L1
      └───┬────┘
          ▼
       CB in L1
```

When you call `copy_tile(cb, 0, 0)`, the unpack build of your kernel issues an
unpack instruction and the math build issues a datacopy; the pack build skips it
entirely. You don't manage this — but knowing it explains why the API looks the
way it does, and why the three threads need explicit synchronization.

### DST registers

The math unit cannot write to L1. It writes to **DST**, a small register file,
and the packer is the only route from DST back to memory.

DST holds 16 tiles physically, but in the default mode math and pack work on
opposite halves so they can overlap — leaving **8 usable**. (Turning on 32-bit
accumulation halves that again.) Every block size in this course is capped at 8
for this reason.

DST is shared between the math and pack threads, so access is handshaked:

```cpp
tile_regs_acquire();   // MATH: wait until DST is mine
   ... produce into DST ...
tile_regs_commit();    // MATH: I'm done, packer may proceed

tile_regs_wait();      // PACK: wait for math to commit
   pack_tile(0, cb_out);
tile_regs_release();   // PACK: DST is free again
```

Both halves must be present, in that order. Miss one and you hang.

### FPU vs SFPU

Two different engines sit behind the compute API:

- **FPU** (matrix unit) — the fast one. Does tile-granular matmul and
  element-wise add/sub/mul between two *CBs*. This is where the TFLOP/s are.
- **SFPU** (vector unit) — a 32-lane SIMD unit for everything transcendental:
  `exp`, `sqrt`, `recip`, `gelu`, comparisons. It operates **in place on DST**,
  not on CBs.

That distinction drives the shape of this kernel. `exp` is an SFPU op, so you
cannot read straight from a CB — you must first get the tile *into* DST with
`copy_tile`, then apply `exp_tile` to it there:

```cpp
copy_tile(cb_in, 0, 0);   // CB tile 0 → DST tile 0   (uses unpacker + FPU datacopy)
exp_tile(0);              // DST tile 0 = exp(DST tile 0)   (SFPU, in place)
pack_tile(0, cb_out);     // DST tile 0 → CB
```

### Init functions

Before the first op you must configure the hardware for the data formats and
operation you're about to run:

```cpp
init_sfpu(cb_in, cb_out);   // one-time: unpacker/packer formats, DST sync mode
exp_tile_init();            // per-op: program the SFPU for exponential
```

`init_sfpu` does MMIO writes to configuration registers and is **only safe at
the top of the kernel**, before any other compute call. The `*_tile_init()`
calls are cheap and are needed once per *kind* of SFPU op you use — call
`exp_tile_init()` once before the loop, not inside it.

---

## Your task

Write **`kernels/compute.cpp`**. The reader and writer are already done for you
(they're the lesson-01 solution, reading into CB 0 and writing out of CB 16).

For each of `n_tiles` tiles: take a tile from CB 0, compute `exp` of it, put the
result in CB 16.

### What the host gives you

| | |
|---|---|
| compile-time arg 0 | input CB index (0) |
| compile-time arg 1 | output CB index (16) |
| runtime arg 0 | number of tiles |

> CB indices are conventional, not magic: 0–7 tend to be inputs, 16–23 outputs.
> The hardware treats all 32 alike.

### API you need

```cpp
init_sfpu(icb, ocb);           // once, at the top
exp_tile_init();               // once, before the loop

tile_regs_acquire();  tile_regs_commit();     // math side
tile_regs_wait();     tile_regs_release();    // pack side

copy_tile(cb, tile_idx_in_cb, dst_idx);
exp_tile(dst_idx);
pack_tile(dst_idx, cb);

cb_wait_front / cb_pop_front / cb_reserve_back / cb_push_back   // as in lesson 01
```

### Run it

```bash
./dojo test 02
```

---

## Hints

<details>
<summary>Loop structure</summary>

```
wait for an input tile
acquire DST
  copy the tile into DST slot 0
  apply exp to DST slot 0
commit DST
reserve an output page
wait for DST (pack side)
  pack DST slot 0 into the output CB
release DST
push the output page
pop the input tile
```

</details>

<details>
<summary>It hangs immediately</summary>

You probably have `tile_regs_acquire()` without `tile_regs_commit()`, or
`tile_regs_wait()` without `tile_regs_release()`. All four are required every
iteration.

</details>

<details>
<summary>Results are wrong but not garbage</summary>

If you forgot `exp_tile_init()`, the SFPU runs whatever it was last programmed
for. If you called it *inside* the loop it still works, just slower.

Note the tolerance: `exp` on bfloat16 through the SFPU is approximate, so the
grader checks correlation and a relative tolerance rather than exact equality.

</details>

---

## Going further

Swap `exp_tile` for `sqrt_tile` (on positive inputs), `gelu_tile`, or
`recip_tile` — each needs its own `*_tile_init()`. The full list is in
`tt_metal/hw/inc/api/compute/eltwise_unary/`.
