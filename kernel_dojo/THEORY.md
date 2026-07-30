# Theory

A from-scratch course on Tensix kernel programming. Assumes you can read C++ and
know roughly what a CPU does — nothing else. Terms like *pipelining*, *double
buffering*, *in flight* and *memory-bound* are defined where they first appear
rather than assumed.

Read with `./dojo theory <n>`, or open the files directly.

---

## Foundations

These two are not Tenstorrent-specific. If you've never written code for an
accelerator, start here; if the exercises stop making sense, come back here.

| | | |
|---|---|---|
| **00** | [What is a kernel, and why is this hard?](theory/00-what-is-a-kernel.md) | Accelerators vs CPUs. No cache, no scheduler, no memory protection. Host vs device. Why there are three kernels per core. |
| **01** | [Latency, bandwidth, and keeping hardware busy](theory/01-latency-and-throughput.md) | **The most useful chapter here.** Latency vs bandwidth. In-flight requests. Batching. Pipelining. Double buffering. Producers, consumers, deadlock. Arithmetic intensity, memory-bound vs compute-bound. Why parallelism stops helping. |

## The hardware

| | | |
|---|---|---|
| **02** | [The chip](theory/02-the-chip.md) | The core grid, the five RISC-V processors inside one Tensix core, L1 and DRAM, the NoC. What the host does. |
| **03** | [Tiles and numbers](theory/03-tiles-and-numbers.md) | The 32×32 tile. Tile layout and the page-index formula. bfloat16 and why nothing is bit-exact. How correctness is graded. |

## Writing kernels

| | | |
|---|---|---|
| **04** | [Circular buffers](theory/04-circular-buffers.md) | The queues between kernels. The four operations, the three things that catch everyone, and how to diagnose a hang. |
| **05** | [Data movement](theory/05-data-movement.md) | Async reads and writes, barriers, `TensorAccessor`, compile-time vs runtime args. The reader/writer pattern. |
| **06** | [Compute](theory/06-compute.md) | Unpack/math/pack. DST registers and the handshake. FPU vs SFPU. Init calls. Matmul and its two traps. Math fidelity. |
| **07** | [Many cores](theory/07-multi-core.md) | Work splitting, per-core arguments, load imbalance. Semaphores, multicast, sharding. |

## Making it fast, and fixing it

| | | |
|---|---|---|
| **08** | [Performance](theory/08-performance.md) | How to measure honestly. Every optimisation lever with measured numbers from this hardware — including the ones that turn out not to work. |
| **09** | [Debugging](theory/09-debugging.md) | Hangs, wrong numbers, `DPRINT`, watcher. What each PCC value tells you. |

## Appendix

Not part of the linear course — reference material for when you need the detail.

| | | |
|---|---|---|
| **10** | [`TensorAccessor` in depth](theory/10-tensor-accessor.md) | Why it's two objects, what the compile-time words contain, the real interleaved address formula, the sharded path, and runtime-configurable shapes. |

---

## Suggested order

If you're new to this: read **00** and **01** before touching an exercise. Read
**02–04** before lesson 01, **06** before lesson 02, and **08** before lesson 04.

If you have GPU or HPC experience: skim **01** for the vocabulary this course
uses, then go straight to **02** and start the exercises, using **04–06** as
reference when an API is unfamiliar.

Each exercise's own README recaps what it needs, so you can also just start at
lesson 01 and come here when something is unexplained.

---

## Beyond this

In the main repo:

| | |
|---|---|
| `tt_metal/programming_examples/` | the same ideas as standalone C++ programs, including the multicast matmul that lesson 08 points at |
| `tt_metal/hw/inc/api/compute/` | every compute API, documented in the headers with argument tables |
| `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | every data movement API |
| `METALIUM_GUIDE.md` | the official architecture guide |
| `tech_reports/` | deep dives on matmul optimisation, data formats, the NoC |
| `tt_metal/tt-llk/` | the layer beneath the compute API, when you need to know why something behaves as it does |
