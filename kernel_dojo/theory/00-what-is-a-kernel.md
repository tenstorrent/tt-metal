# 00 — What is a kernel, and why is this hard?

*Assumes you can read C++ and know what a CPU does. Nothing else.*

---

## The one-sentence version

A **kernel** is a small program that runs *on an accelerator chip* instead of on
your computer's CPU. You write it in C++, but it runs on completely different
hardware, with completely different rules.

That's it. The rest of this chapter is about why those rules are so unfamiliar.

---

## Why accelerators exist

Suppose you want to add two lists of a million numbers. On a CPU:

```c
for (int i = 0; i < 1000000; i++) {
    c[i] = a[i] + b[i];
}
```

A modern CPU core runs at ~4 GHz and can do maybe 4–8 of these additions per
clock cycle using vector instructions. Call it 30 billion additions a second.
That sounds like a lot until you're training a neural network, where a single
layer might need trillions.

So people built chips that do less, but far more of it at once. A Tenstorrent
Wormhole chip has **64 independent compute cores**, and each one has a matrix
engine that can multiply two 32×32 matrices as a single operation. That's 32,768
multiply-adds per instruction, times 64 cores.

The catch is that all the machinery a CPU uses to make programming *pleasant* —
caches, out-of-order execution, branch prediction, virtual memory — costs
transistors. Accelerators spend those transistors on math units instead, and
hand the resulting complexity to you.

**A kernel is the code you write to manage that complexity.**

---

## The three things a CPU does for you that an accelerator doesn't

### 1. It moves data for you

On a CPU, when you write `a[i]`, the hardware:

- checks whether that memory is in L1 cache (a few cycles away)
- if not, checks L2, then L3
- if not, fetches it from DRAM (hundreds of cycles away) and *automatically*
  caches it for next time

You never think about this. There is a whole invisible system making memory
look uniform and fast.

**An accelerator has no cache.** It has fast local memory (SRAM) and slow bulk
memory (DRAM), and *nothing moves between them unless your program explicitly
says so*. If you want to add two numbers that live in DRAM, you must:

1. issue a command to copy them into local memory,
2. wait for that copy to finish,
3. do the addition,
4. issue a command to copy the result back.

Steps 1, 2 and 4 are usually **more code and more of your attention** than step
3. This surprises everyone. Most of this course is about data movement, and
that is not an accident — it's a fair reflection of where the difficulty is.

### 2. It hides parallelism from you

A CPU core executes your instructions in order (as far as you can tell). If you
want to use 8 CPU cores, you spawn threads and the OS schedules them.

On an accelerator you write the program for one core, and then explicitly say
"run this on cores (0,0) through (7,7)". All 64 run *the same compiled code* at
the same time. Making them do *different* work is something you arrange by
passing each one different parameters.

There is no OS, no scheduler, no preemption. A core runs your kernel from start
to finish and stops.

### 3. It protects you from yourself

If a CPU program reads a bad pointer, you get a segfault and a stack trace.

If a kernel reads a bad address, you get **wrong numbers**, or a **hang**, or
occasionally silent corruption of a completely unrelated buffer. There is no
memory protection. There is no exception. The hardware does exactly what you
told it.

This is why the debugging habits in this course lean so heavily on
*verification*: run it, compare against a known-good answer, and don't trust
code that hasn't been checked against a reference.

---

## What "a kernel runs on a core" actually means

Here's the shape of every kernel you'll write:

```cpp
void kernel_main() {
    // read some parameters that the host program passed in
    uint32_t how_many_tiles = get_arg_val<uint32_t>(0);

    for (uint32_t i = 0; i < how_many_tiles; i++) {
        // ... do one unit of work ...
    }
}
```

No `main()`, no arguments, no return value, no printf (well — almost, see
chapter 09). It gets parameters through a side channel, does its work, and
exits.

The program that *starts* kernels is called the **host program**. It runs on
your CPU, in this course it's Python, and it's responsible for:

- allocating buffers in the accelerator's DRAM
- copying input data into them
- deciding which cores run which kernels
- passing each core its parameters
- launching everything
- copying results back

In the dojo, the host side is written for you (it's in each exercise's
`task.py`) so you can concentrate on the kernels. But it's worth knowing that
split exists: **host code decides what happens; kernel code makes it happen.**

---

## Two kinds of kernel

Tenstorrent hardware splits kernels by *job*, which is unusual and worth
flagging early.

Inside one Tensix core there are several small processors. Some are good at
issuing memory transfers; others drive the math engines. You write **separate
programs for each**, and they run concurrently on the same core, passing data to
each other through local memory.

A typical setup is three kernels running at once on every core:

| Kernel | Job |
|---|---|
| **reader** | pull data from DRAM into local memory |
| **compute** | do the maths on it |
| **writer** | push results from local memory back to DRAM |

They form an assembly line. The reader is fetching item 5 while compute works on
item 4 and the writer is shipping item 3. Getting that assembly line to flow
smoothly is what chapter 01 is about, and it's the single biggest source of
performance in practice.

If you've written CUDA, this is the biggest structural difference: there is no
single kernel function that does everything. The work is split across
cooperating programs by hardware role.

---

## What you're actually optimising

When people say a kernel is "fast", they almost never mean the arithmetic is
fast. The arithmetic is nearly always the easy part — the math engines are
enormously capable and mostly sit idle.

What they mean is that the kernel manages to **keep the expensive hardware
busy**: data arrives before the math unit needs it, results leave before the
buffer fills up, and no part of the assembly line spends its time waiting for
another part.

That framing — *what is this waiting on?* — is the thread running through the
whole course.

---

## Vocabulary you'll meet

Defined properly later, but so nothing is a surprise:

| Term | Rough meaning |
|---|---|
| **host** | your CPU, running the Python that sets everything up |
| **device** | the accelerator chip |
| **DRAM** | the chip's large, slow memory (12 GB on this card) |
| **L1** | a core's small, fast local memory (1464 KB) — *not* a CPU-style cache |
| **NoC** | "network on chip", the wiring that carries data between cores and DRAM |
| **tile** | a 32×32 block of numbers; the hardware's native unit of data |
| **circular buffer (CB)** | a queue in L1 that kernels use to hand tiles to each other |
| **DST** | the register file where the math engine writes results |
| **kernel** | a program that runs on one processor inside one core |

---

**Next:** [01 — Latency, bandwidth, and keeping hardware busy](01-latency-and-throughput.md)
covers the performance concepts — pipelining, batching, double buffering,
bottlenecks — from first principles. It is the most useful chapter in this
directory and it is not Tenstorrent-specific.
