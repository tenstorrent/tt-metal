# 02 — The chip

*What's physically there, and what each piece is for.*

---

## The grid

A Wormhole chip is a rectangular grid of small units connected by a network.
Most of the units are **Tensix cores** — the things that compute. The rest are
memory controllers, Ethernet ports, and PCIe.

```
        ┌────┬────┬────┬────┬────┬────┬────┬────┐
DRAM ───┤ T  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │
        ├────┼────┼────┼────┼────┼────┼────┼────┤
DRAM ───┤ T  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │
        ├────┼────┼────┼────┼────┼────┼────┼────┤
        │ .. │ .. │ .. │ .. │ .. │ .. │ .. │ .. │   8 x 8 = 64 Tensix cores
        ├────┼────┼────┼────┼────┼────┼────┼────┤
DRAM ───┤ T  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │
        └────┴────┴────┴────┴────┴────┴────┴────┘
              all connected by the NoC
```

Key numbers for the card in this machine (Wormhole n150/n300):

| | |
|---|---|
| Usable Tensix cores | **8 × 8 = 64** |
| L1 memory per core | **1464 KB** (~1.4 MB) |
| Total on-chip SRAM | ~91 MB |
| DRAM | **12 GB** across 6 channels |
| Practical DRAM bandwidth | **~195 GB/s** (measured, this access pattern) |
| Clock | ~1 GHz |

`./dojo doctor` prints the grid size for your actual device.

### Cores are addressed by coordinate

A core is identified by `(x, y)`, so `(0,0)` is one corner and `(7,7)` the
opposite one. When you launch a kernel you give a **set of cores** to run it on,
and every core in that set runs the same compiled binary.

---

## Inside one Tensix core

This is the part that surprises people. A Tensix core is not one processor.

```
┌───────────────────────────────────────────────────────────┐
│  Five RISC-V processors, each running its own program:    │
│                                                            │
│   BRISC     NCRISC   │   TRISC0    TRISC1    TRISC2       │
│   ───────────────    │   ────────────────────────────     │
│    data movement     │        compute control             │
│                      │   (unpack)   (math)    (pack)      │
├───────────────────────────────────────────────────────────┤
│  The engines they drive:                                   │
│                                                            │
│      FPU  (matrix engine)      SFPU  (vector engine)      │
│                                                            │
│      SrcA / SrcB  ──────▶  DST registers                  │
├───────────────────────────────────────────────────────────┤
│                 L1 SRAM — 1464 KB                          │
│      (shared by all five processors, directly addressable) │
├───────────────────────────────────────────────────────────┤
│            NoC0 port          NoC1 port                    │
└───────────────────────────────────────────────────────────┘
```

### The five RISC-V processors

They are small, slow, in-order CPUs. **They are not there to do arithmetic.**
Their job is to issue commands to the hardware that does the real work.

- **BRISC** and **NCRISC** — the two *data movement* processors. They program
  the NoC to move bytes. By convention BRISC runs the "writer" kernel and NCRISC
  the "reader", but nothing enforces that; they're interchangeable.

- **TRISC0, TRISC1, TRISC2** — the three *compute* processors. They drive the
  math engines in a three-stage pipeline:
  - **TRISC0 → the unpacker**: moves tiles from L1 into the math engine's input
    registers
  - **TRISC1 → the math unit**: runs FPU/SFPU operations
  - **TRISC2 → the packer**: moves results from the output registers back to L1

If you write a loop doing scalar arithmetic in a data movement kernel, it runs
on a small in-order core with no vector unit. It will be slow. Keep arithmetic
in the compute kernel.

### Why five programs?

Because each processor can then run ahead independently. That's the pipeline
from chapter 01: while the math engine works on tile 4, NCRISC is fetching tile
5 and BRISC is writing out tile 3. They only synchronise where they have to —
at the buffers between them.

If it were one program doing all three jobs in sequence, everything would
serialise and you'd lose most of the chip's performance.

### One source file, three compilations

Here's the piece of magic worth knowing early: you write **one** compute kernel
file, and it is compiled **three times** — once for TRISC0, once for TRISC1,
once for TRISC2.

Macros inside the API headers select which parts survive in each build. When you
call `add_tiles(...)`:

- the *unpack* build emits instructions to feed the input registers
- the *math* build emits the actual add
- the *pack* build emits nothing for that call

You never write these separately. But this explains why the compute API looks
the way it does, and why the three threads need explicit synchronisation
around the result registers (chapter 06).

---

## Memory: three levels, no cache

| | Size | Speed | Who can reach it |
|---|---|---|---|
| **DST registers** | 16 tiles (8 usable) | immediate | math + pack only |
| **L1 SRAM** | 1464 KB per core | ~tens of cycles | everything on that core |
| **DRAM** | 12 GB total | ~hundreds of cycles | anything, via the NoC |

Say it once more, because it governs everything: **there is no cache.** Data
does not move between these levels on its own. A kernel must explicitly issue
every transfer.

### L1 is not a CPU's L1

Despite the name, this isn't a cache. It's a plain block of addressable memory
that belongs to one core. You place things in it deliberately. Nothing is
evicted behind your back, and nothing appears in it because you happened to
touch an address.

Every core has its own. Core (0,0) cannot read core (3,5)'s L1 with a normal
load instruction — it has to send a NoC transaction, the same as it would to
reach DRAM.

### DRAM is spread across banks

The 12 GB is split across 6 channels. A large buffer is **interleaved**: page 0
goes in bank 0, page 1 in bank 1, and so on, cycling round.

This is a bandwidth trick. If every core hammered a single bank, that bank would
be the bottleneck and the other five would idle. Interleaving spreads a
sequential access pattern evenly.

The consequence for you is that "which bank and what offset holds page 37" is
not obvious arithmetic. The `TensorAccessor` type (chapter 05) works it out.

---

## The NoC

The **network on chip** carries data between cores, and between cores and DRAM.
It's a 2-D mesh: data travels in hops between neighbouring grid positions.

There are actually **two** networks, NOC0 and NOC1, running in opposite
directions around the grid. Each data movement processor defaults to a different
one. That's deliberate — a reader pulling data in and a writer pushing data out
don't compete for the same wires.

Two practical consequences:

- **Distance costs a little.** Reading from a DRAM controller on the far side of
  the chip takes more hops than a near one. Usually a second-order effect, but
  it's why sharding data near the cores that use it can matter.
- **Bandwidth is shared.** The ~195 GB/s figure is for the whole chip. 64 cores
  each want a slice of it, which is exactly why the scaling curve in lesson 04
  flattens.

---

## What the host does

Your Python program runs on the CPU and talks to the chip over PCIe. It:

1. allocates buffers in device DRAM
2. copies input data across
3. builds a **program**: which kernels run on which cores, what buffers exist,
   what parameters each core gets
4. launches it
5. waits, then copies results back

Steps 3–4 have real overhead — packaging commands and sending them over PCIe
takes tens of microseconds. For small kernels this can exceed the kernel's own
runtime, which is why performance is measured on-device (chapter 08).

---

## Putting it together

The shape of essentially every kernel you'll write in this course:

```
        DRAM
          │  reader kernel (NCRISC) issues NoC reads
          ▼
   ┌─────────────┐
   │  CB in L1   │   a queue of tiles
   └─────────────┘
          │  unpacker (TRISC0)
          ▼
      SrcA/SrcB  →  FPU or SFPU (TRISC1)  →  DST registers
          │  packer (TRISC2)
          ▼
   ┌─────────────┐
   │  CB in L1   │
   └─────────────┘
          │  writer kernel (BRISC) issues NoC writes
          ▼
        DRAM
```

Three kernels you write, five processors, two queues, and a pipeline that
ideally never stalls.

---

**Next:** [03 — Tiles and numbers](03-tiles-and-numbers.md) — what the data
actually looks like.
