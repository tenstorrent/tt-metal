# 06 — Compute

*The math engines, the registers they write to, and the synchronisation that
keeps three processors from tripping over each other.*

---

## The three-stage pipeline

Recall from chapter 02: one compute kernel source file, compiled three times,
for three processors that form a pipeline.

```
   CB in L1
      │
  ┌───▼────┐
  │ UNPACK │  TRISC0 — L1 → SrcA/SrcB registers
  └───┬────┘
  ┌───▼────┐
  │  MATH  │  TRISC1 — FPU or SFPU → DST registers
  └───┬────┘
  ┌───▼────┐
  │  PACK  │  TRISC2 — DST → L1
  └───┬────┘
      ▼
   CB in L1
```

You write one sequence of calls; each build picks up its own part. That's
automatic. What is *not* automatic is the handshaking between the math and pack
stages, because they share the output registers.

---

## DST: where results land

The math engines cannot write to L1. They write to **DST**, a small register
file, and the packer is the only route from DST back to memory.

DST holds **16 tiles** physically. But in the default mode, math and pack work
on opposite halves so they can overlap — which leaves you **8 usable tiles**.

| Mode | Usable DST tiles |
|---|---|
| default (half-sync), bfloat16 | **8** |
| `fp32_dest_acc_en` (32-bit accumulate) | 4 |
| `dst_full_sync_en` | 16, but math and pack no longer overlap |

Every block size in this course is capped at 8 for this reason.

### The handshake

Math and pack are separate processors sharing DST, so access is explicitly
coordinated by four calls:

```cpp
tile_regs_acquire();     // MATH: claim DST. Also ZEROES it.
    ... produce results into DST ...
tile_regs_commit();      // MATH: done, packer may proceed

tile_regs_wait();        // PACK: wait for math's commit
    pack_tile(0, cb_out);
tile_regs_release();     // PACK: DST is free again
```

**All four are required, every iteration.** Miss one and you deadlock: the math
thread waits for a release that never comes, or the pack thread waits for a
commit that never comes.

Two details worth internalising:

- **`tile_regs_acquire()` zeroes DST.** That's what makes accumulation work
  without an explicit clear — see matmul below.
- **The pair split across threads is deliberate.** `acquire`/`commit` are
  math-side; `wait`/`release` are pack-side. They read like four steps of one
  sequence, but they're really two two-step protocols meeting in the middle.

The handshake costs real time. Doing it once per tile is wasteful when you could
do it once per block of 8 — that's part of what lesson 05 measures.

---

## Two engines: FPU and SFPU

|  | **FPU** (matrix engine) | **SFPU** (vector engine) |
|---|---|---|
| Reads from | two CBs, directly | DST only |
| Writes to | DST | DST (in place) |
| Operations | matmul, add/sub/mul, reduce, transpose | exp, sqrt, recip, gelu, comparisons, ... |
| Speed | very fast — this is where the FLOP/s are | slower, but general |

The split matters because it changes the shape of your kernel.

### FPU: one call

The FPU reads its two operands straight out of circular buffers:

```cpp
add_tiles(cb_a, cb_b, 0, 0, 0);   // CB tile 0 + CB tile 0 → DST tile 0
```

The unpacker feeds SrcA from `cb_a` and SrcB from `cb_b`, the FPU adds, DST gets
the result. Nothing else needed.

### SFPU: two calls

The SFPU only operates on DST, so you must get the data there first:

```cpp
copy_tile(cb_in, 0, 0);   // CB tile 0 → DST tile 0
exp_tile(0);              // DST tile 0 = exp(DST tile 0), in place
```

That extra `copy_tile` is not optional and not a formality — the vector unit
physically cannot address a circular buffer.

---

## Initialisation

Before your first operation you must configure the hardware: data formats for
the unpacker and packer, DST sync mode, and which operation the engine should
perform.

There are two levels.

### One-time hardware setup

Exactly one of these, as the **first compute call in the kernel**:

| Op family | Call |
|---|---|
| Unary / SFPU | `init_sfpu(icb, ocb)` |
| Binary FPU | `binary_op_init_common(icb0, icb1, ocb)` |
| Matmul | `compute_kernel_hw_startup<SrcOrder::Reverse>(icb0, icb1, ocb)` |

These write to configuration registers over MMIO. They are slow, and they are
**only safe when the execution units are idle** — i.e. at the very top of the
kernel. Calling one in the middle of a running kernel races with in-flight work
and produces bugs that are extremely hard to localise.

### Per-operation setup

Cheap. Tells the engine which operation to perform:

```cpp
exp_tile_init();               // SFPU: program for exponential
add_tiles_init(cb_a, cb_b);    // FPU: program for add
matmul_init(cb_a, cb_b);       // FPU: program for matmul
```

Call once before the loop, not inside it. If you switch operations mid-kernel
(say, add then multiply), call the corresponding `*_init` again at the switch.

---

## Matmul, and its two traps

```cpp
matmul_tiles(cb_a, cb_b, tile_a, tile_b, dst);   // DST[dst] += A_tile @ B_tile
```

### Trap 1: it accumulates

That `+=` is the whole reason matmul is easy to write. `C[m][n]` is the sum over
`k` of `A[m][k] @ B[k][n]`, which is just a loop into the same DST slot:

```cpp
tile_regs_acquire();                          // DST[0] = 0
for (uint32_t kt = 0; kt < Kt; kt++) {
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    matmul_tiles(cb_a, cb_b, 0, 0, 0);        // DST[0] += A @ B
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
}
tile_regs_commit();
// pack ONCE, here — outside the k loop
```

**Do not pack inside the k loop.** Packing writes to a bfloat16 circular buffer,
which rounds to 7 mantissa bits. Doing that on every step of a 64-deep reduction
destroys the result. Keeping the accumulation in DST keeps it at the FPU's
internal precision.

### Trap 2: `SrcOrder::Reverse`

Matmul is the one operation where the operands map to the source registers
backwards: `in0` goes to **SrcB** and `in1` to **SrcA**. The hardware
configuration has to be told:

```cpp
compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
matmul_init(cb_a, cb_b);
```

Use the default `SrcOrder::Regular` and you get **wrong numbers with no error**.
Two lines, and they cost people an afternoon.

Note also that matmul does *not* use `binary_op_init_common`, even though it
takes two inputs.

---

## Math fidelity

The FPU multiplies mantissas in slices, one per pass. More passes, more
precision, proportionally more time.

| Mode | Passes | Relative speed | Retains |
|---|---|---|---|
| `LoFi` | 1 | 4× | ~5 mantissa bits |
| `HiFi2` | 2 | 2× | all 8 bits of a bfloat16 input |
| `HiFi3` | 3 | 1.33× | more |
| `HiFi4` | 4 | 1× | full bf16 × bf16 |

Set on the host side, in the compute kernel's configuration.

For **bfloat16 inputs, `HiFi2` is usually the right answer**: it captures the
entire input mantissa, so `HiFi4` is mostly buying precision the inputs never
had, at twice the cost.

**But** — and this is the lesson 07/08 punchline — this only matters if the FPU
is what you're waiting on. In this course's matmuls it never is, and switching
`HiFi4` → `LoFi` changes the runtime by about 1%. Always check before tuning it.

---

## Working in blocks

Two things scale with block size:

**Amortise the DST handshake.** One `acquire`/`commit`/`wait`/`release` per
block of 8 rather than per tile:

```cpp
cb_wait_front(cb_a, 8);
cb_wait_front(cb_b, 8);
cb_reserve_back(cb_out, 8);

tile_regs_acquire();
for (uint32_t t = 0; t < 8; t++) {
    add_tiles(cb_a, cb_b, t, t, t);   // CB index t → DST slot t
}
tile_regs_commit();

tile_regs_wait();
for (uint32_t t = 0; t < 8; t++) {
    pack_tile(t, cb_out);
}
tile_regs_release();

cb_push_back(cb_out, 8);
cb_pop_front(cb_a, 8);
cb_pop_front(cb_b, 8);
```

**Note the indices.** With 8 tiles visible in the window, the CB-relative index
is `t`, not `0` (chapter 04). And the DST slot is `t`, which is why 8 is the
ceiling.

**Enable reuse.** If you hold several tiles resident and use each against many
others, you cut the traffic needed to feed the engine. That's lessons 07 and 08,
and it's the difference between 0.1 and 13 TFLOP/s in this course.

---

## Where to find the rest

Every compute API is documented in the headers, with argument tables:

```
tt_metal/hw/inc/api/compute/
├── eltwise_binary.h              add/sub/mul between two CBs
├── eltwise_unary/                exp, sqrt, recip, gelu, ... (one file each)
├── matmul.h                      matmul_tiles, matmul_block
├── reduce.h                      sum/max along rows, columns, or both
├── tile_move_copy.h              copy_tile, copy_block
├── transpose_wh.h                transpose within a tile
├── tilize.h / untilize.h         layout conversion
└── compute_kernel_hw_startup.h   the startup call
```

They're readable and they're the authoritative reference. `eltwise_unary/` in
particular is just a long list of SFPU operations, each with a `*_tile()` and a
`*_tile_init()`.

---

**Next:** [07 — Many cores](07-multi-core.md) — parallelism and how cores
cooperate.
