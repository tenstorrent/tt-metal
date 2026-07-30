# 01 — Latency, bandwidth, and keeping hardware busy

*No hardware background assumed. Nothing here is Tenstorrent-specific — these
ideas apply to GPUs, CPUs, disk I/O and network code equally. They are the
foundation for every performance decision in the rest of the course.*

---

## 1. Latency and bandwidth are different things

This is the single most important distinction in performance work, and it trips
up almost everyone at first.

- **Latency** is *how long one thing takes*, start to finish.
- **Bandwidth** (or throughput) is *how much you can get done per second*.

They are not the same, and improving one does not improve the other.

### The delivery truck

A truck carrying 20 tonnes of hard drives from London to Edinburgh takes 8
hours. Its **latency** is 8 hours — nothing arrives sooner than that. But its
**bandwidth** is enormous: 20 tonnes of drives is maybe 400 petabytes, over 8
hours, which beats any internet connection on the planet.

Meanwhile a fibre link has millisecond latency and comparatively tiny bandwidth.

Which is "faster"? Depends entirely on what you're doing. Neither number alone
tells you.

### Why it matters for kernels

Reading one value from DRAM on an accelerator has:

- **latency** of several hundred clock cycles (call it ~0.5 microseconds
  including all the queuing)
- **bandwidth** of hundreds of gigabytes per second across the whole chip

So DRAM is *slow to respond* but *capable of enormous volume*. Those two facts
pull in opposite directions, and the entire art of writing fast kernels is
exploiting the second while hiding the first.

Here's the trap. If you write the obvious loop:

```
read one value  →  wait for it  →  use it  →  read the next  →  wait  →  ...
```

you pay the full latency every single time, and you never get anywhere near the
bandwidth. You're using a 20-tonne truck to deliver one hard drive at a time.

Real measurement from lesson 04 of this course: one core doing exactly that
achieves **9.7 GB/s**. The chip is capable of about **195 GB/s**. Same hardware,
same data — 20× off, purely from how the requests were issued.

---

## 2. Hiding latency: get many things in flight

**"In flight"** means *requested but not yet arrived*. A request you've issued
and haven't waited for yet is in flight.

The fix for latency is not to make each request faster — you can't. It's to have
many requests outstanding simultaneously, so their waiting **overlaps**.

### Serial: latency paid 4 times

```
request A  ├────wait────┤ A arrives
                        request B  ├────wait────┤ B arrives
                                                request C  ├────wait────┤
total: 3 × latency
```

### Overlapped: latency paid roughly once

```
request A  ├────wait────┤ A arrives
request B   ├────wait────┤ B arrives
request C    ├────wait────┤ C arrives
total: ~1 × latency
```

The requests didn't get faster. You just stopped waiting for each one before
starting the next.

This is why the hardware provides **asynchronous** operations:

```cpp
noc_async_read_page(0, src, dst0);   // returns immediately, data NOT here yet
noc_async_read_page(1, src, dst1);   // returns immediately
noc_async_read_page(2, src, dst2);   // returns immediately
noc_async_read_barrier();            // NOW wait — for all three at once
```

`noc_async_read_page` only *asks* for the data. It returns before anything
arrives. The **barrier** is where you actually wait.

> **Barrier**: a point in the program that blocks until previously-issued
> asynchronous work has completed. Think of it as "wait for everything I asked
> for."

Issuing three reads then one barrier costs about the same wall-clock time as
issuing one read and one barrier. That's the whole trick.

### The corresponding hazard

Because the data isn't there when the call returns, **using it before the
barrier gives you garbage** — whatever happened to be in that memory before.
No error, no crash. Just wrong numbers, or numbers that are right on Tuesday
and wrong on Wednesday.

Every async operation needs a matching wait before you touch the result. This
is the number one source of "it works sometimes" bugs.

---

## 3. Batching

**Batching** just means *grouping several items and handling them together
instead of one at a time*.

You batch to amortise a fixed cost. If every trip to the shop takes 20 minutes
regardless of what you buy, you don't make one trip per item.

In kernels the fixed cost being amortised is usually one of:

- **the barrier wait** — one wait for 8 reads instead of 8 waits for 8 reads
  (this is section 2, expressed as a code pattern)
- **synchronisation between processors** — handshakes between the parts of the
  chip cost time; do one per group of 8 items rather than one per item
- **instruction issue overhead** — the setup around an operation, paid once per
  batch

Concretely, a loop that goes from "one tile at a time" to "eight tiles at a
time" in this course is **2.6× faster** with identical hardware and identical
arithmetic (lesson 05).

The costs of batching are real though:

- **Memory.** You need somewhere to hold a whole batch.
- **Latency of the first result.** Batch 8 and nothing is ready until 8 are
  done. Usually irrelevant in a kernel, sometimes crucial in a server.
- **Awkward remainders.** 100 items in batches of 8 leaves 4 over, and you need
  code to handle them. (This course sizes its problems to divide evenly so you
  can ignore that.)

---

## 4. Pipelining

**Pipelining** is overlapping *different kinds of work* so that specialised
hardware doesn't sit idle.

### The laundry example

You have 3 loads of washing. Washing takes 30 min, drying 30 min, folding 30
min.

**Sequentially** — do everything for load 1, then load 2, then load 3:

```
load 1: WASH DRY FOLD
load 2:                WASH DRY FOLD
load 3:                               WASH DRY FOLD
                                                    = 270 min
```

The dryer sits unused two-thirds of the time.

**Pipelined** — as soon as load 1 leaves the washer, load 2 goes in:

```
load 1: WASH DRY  FOLD
load 2:      WASH DRY  FOLD
load 3:           WASH DRY  FOLD
                                = 150 min
```

Same machines, same work, 1.8× faster. And with many loads it approaches 3×,
because all three machines are busy nearly all the time.

### In a kernel

The reader / compute / writer split from chapter 00 is exactly this:

```
tile 1: READ COMPUTE WRITE
tile 2:      READ    COMPUTE WRITE
tile 3:              READ    COMPUTE WRITE
```

Three different pieces of hardware — the memory system, the math engine, the
memory system again — each busy while the others work.

**A pipeline runs at the speed of its slowest stage.** If reading takes 100ns
and computing takes 10ns, you get one result per 100ns no matter how fast the
math is. Making the math twice as fast changes nothing. That slowest stage is
called the **bottleneck**, and finding it is section 7.

---

## 5. Double buffering

Now, the thing that makes pipelining actually work.

Look again at the laundry pipeline. Load 2 goes into the washer while load 1 is
in the dryer. Fine — they're different machines. But what if there were only one
laundry *basket* between them, and a load had to sit in it during the handover?

Load 1 is in the basket waiting to be dried. Load 2 finishes washing and needs
the basket. It can't have it. The washer stalls.

**One basket serialises the whole pipeline no matter how many machines you
have.** You need two.

### The general shape

A **buffer** is a piece of memory where a producer leaves data for a consumer.

With **one buffer**:

```
producer: FILL ....wait.... FILL ....wait....
consumer: ....wait.... DRAIN ....wait.... DRAIN
```

They take turns. Only one is ever working. All the pipelining is lost.

With **two buffers** ("double buffering"):

```
producer: FILL[A] FILL[B] FILL[A] FILL[B]
consumer: ....... DRAIN[A] DRAIN[B] DRAIN[A]
```

The producer fills buffer B while the consumer drains buffer A, then they swap.
Both work continuously.

> **Double buffering**: using two buffers alternately, so a producer can be
> filling one while a consumer empties the other. Also called *ping-pong
> buffering*.

Nothing stops you using three, or eight. More buffers tolerate more variation in
timing — if the producer is sometimes slow, a deeper buffer means the consumer
has more banked up to chew through before it starves. The cost is memory, which
on a chip with 1464 KB per core is a real constraint.

In this course, buffer depth is a number you pass when creating a queue
(`n_pages=2` means double buffering), and lesson 05 makes you feel the
difference.

---

## 6. Producers, consumers, and deadlock

Two pieces of hardware sharing a buffer need rules, or the producer will
overwrite data the consumer hasn't read yet.

The rules are enforced by four operations. In this course the buffer is called a
**circular buffer** (chapter 04 explains the "circular" part) and the operations
are:

| Producer side | Consumer side |
|---|---|
| `cb_reserve_back(cb, n)` — *wait for n free slots* | `cb_wait_front(cb, n)` — *wait for n filled slots* |
| `cb_push_back(cb, n)` — *"n slots are now full"* | `cb_pop_front(cb, n)` — *"n slots are now free"* |

The two `wait`-flavoured calls **block**: they stop the processor until the
condition is true. That blocking is what keeps the two sides in step. It's also
what can hang your program.

### Deadlock

> **Deadlock**: two or more parties each waiting for something only another can
> provide, so none of them ever proceeds. Nothing crashes; everything just
> stops, forever.

The classic kernel deadlock:

1. The producer fills every slot in the buffer and calls `cb_reserve_back` for
   one more.
2. There are no free slots, so it blocks — waiting for the consumer to pop.
3. The consumer is meanwhile blocked in `cb_wait_front` waiting for a *different*
   buffer that nobody is filling.
4. Neither will ever move.

Almost all deadlocks in this course come from **miscounting**:

- reserving 1 but pushing 2
- waiting for 4 tiles when only 3 will ever be sent
- forgetting to pop, so the buffer fills and never drains
- two kernels disagreeing about how many items they're processing
- waiting for more slots than the buffer physically has (this one can *never*
  succeed)

The symptom is always the same: the program stops and produces nothing. The dojo
sets a 30-second timeout so this shows up as an error rather than a freeze, but
you should recognise the shape.

**Rule of thumb:** every `reserve` needs a matching `push`, every `wait` needs a
matching `pop`, and the counts must agree across all the kernels sharing the
buffer.

---

## 7. Finding the bottleneck

A pipeline runs at the speed of its slowest stage. Optimising anything else
achieves literally nothing. So before changing code, work out what you're
waiting on.

### Memory-bound vs compute-bound

Two questions:

- How many **bytes** does this move?
- How many **arithmetic operations** does it do?

The ratio is called **arithmetic intensity** — operations per byte.

> **FLOP**: one floating-point operation (one add, or one multiply). "FLOP/s" is
> floating-point operations per second. A multiply-add counts as 2.

Every chip has two ceilings: a maximum bandwidth (bytes/second) and a maximum
compute rate (FLOP/s). Which one you hit first depends on your arithmetic
intensity.

- **Memory-bound**: low intensity. You're waiting for data. The math units are
  idle. *Making the math faster does nothing.* Move fewer bytes instead.
- **Compute-bound**: high intensity. Data arrives faster than you can process
  it. *Moving fewer bytes does nothing.* Do less work, or do it more cheaply.

Worked example — adding two arrays. Per output number: read 2 values, write 1
(6 bytes at 2 bytes each), perform 1 addition. That's **0.17 operations per
byte**. Hopelessly memory-bound. Any chip on earth is bottlenecked on memory for
this, and no amount of clever arithmetic will help.

Worked example — multiplying two large matrices. Every element of the input gets
used many times over, so the intensity scales with the matrix size and can reach
hundreds of operations per byte. This is why matmul is the operation
accelerators are designed around, and why lessons 06–08 spend their time on it.

### How to tell, in practice

Measure, then look at which number is near its ceiling:

- If **GB/s is pinned at the hardware maximum** and time scales exactly with
  bytes moved → memory-bound.
- If **GB/s is well below maximum** and reducing bytes doesn't help → something
  else is the limit.
- If the compute rate is near the chip's peak FLOP/s → compute-bound.

And the decisive test: **change one thing and see if the time moves.** In lesson
07 of this course, reducing the math work by 4× changes the runtime by 1%. That
single measurement proves the math was never the bottleneck, and saves you from
optimising it further.

### Amdahl's law, informally

If a stage takes 20% of your time, making it infinitely fast gives you at most a
25% speedup. Optimisation effort should go where the time actually is, and the
only way to know where that is, is to measure.

---

## 8. Parallelism, and why it stops helping

Splitting work across more cores is the most obvious optimisation and the first
one you'll do (lesson 04). Two things limit it.

### Shared resources saturate

64 cores all reading from the same DRAM will, at some point, ask for more
bandwidth than DRAM can supply. Past that point extra cores don't help — and can
actively hurt, because they add contention for a resource that's already full.

Measured in lesson 04 of this course:

| cores | bandwidth | speedup |
|---|---|---|
| 1 | 9.7 GB/s | 1.0× |
| 2 | 19.4 GB/s | 2.0× |
| 8 | 74 GB/s | 7.6× |
| 32 | 195 GB/s | 20× |
| 64 | 181 GB/s | **18.6×** ← worse than 32 |

Perfect scaling up to 8, saturation by 32, and *regression* at 64. "Use all the
cores" is not automatically correct.

### The slowest worker sets the pace

If the work doesn't divide evenly, everyone waits for whoever got the most.
65 items across 64 workers takes as long as 128 items across 64 workers: one
worker does 2 while 63 do 1 and then idle.

This is **load imbalance**, and it's why problem sizes and grid shapes get
chosen to divide cleanly.

---

## 9. Measuring honestly

Three habits worth adopting now.

**Measure the device, not the host.** Launching work on an accelerator has
overhead — packaging commands, sending them over PCIe, waiting for completion.
For a kernel taking 70 microseconds, the host might observe 168. If you optimise
against the host number you'll be measuring the launch machinery. (The dojo
reports both, so you can see the gap.)

**Warm up first.** The first run of a kernel compiles it, which takes seconds.
Timing that tells you about the compiler. Discard the first few runs.

**Change one thing.** If you improve the buffer depth and the core count at the
same time and it gets 3× faster, you've learned nothing about either. Every
benchmark in this course sweeps exactly one variable.

And treat small differences as noise. A 3% change between runs is normal; if
your optimisation produced 3%, you haven't measured an improvement yet.

---

## Summary

| Idea | One line |
|---|---|
| Latency | How long one operation takes |
| Bandwidth | How much you can move per second |
| In flight | Requested but not yet arrived |
| Async + barrier | Issue many requests, wait once, overlap the latency |
| Batching | Group items to amortise a fixed per-group cost |
| Pipelining | Overlap different *stages* so all hardware stays busy |
| Double buffering | Two buffers so producer and consumer never take turns |
| Deadlock | Everyone waiting for everyone; miscounted buffer operations |
| Arithmetic intensity | Operations per byte; decides which ceiling you hit |
| Memory-bound | Waiting on data — move fewer bytes |
| Compute-bound | Waiting on maths — do cheaper maths |
| Bottleneck | The slowest stage; the only thing worth optimising |

---

**Next:** [02 — The chip](02-the-chip.md) — what a Tensix core actually contains,
and how these ideas map onto real hardware.
