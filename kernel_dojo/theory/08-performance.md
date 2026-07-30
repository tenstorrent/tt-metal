# 08 — Performance

*How to measure, what the levers are, and how to tell which one to pull. Every
number here was measured on the Wormhole card in this machine.*

Prerequisite: [chapter 01, sections 7–9](01-latency-and-throughput.md) —
bottlenecks, arithmetic intensity, honest measurement.

---

## First: what are you waiting on?

Before changing any code, work out which resource is saturated. Everything else
is guessing.

The two ceilings on this chip:

| Resource | Practical ceiling |
|---|---|
| DRAM bandwidth | **~195 GB/s** (measured, interleaved tile access) |
| FPU throughput | tens of TFLOP/s, depends on fidelity and how you issue |

And the diagnostic:

- **GB/s pinned at ~195, time scales exactly with bytes moved** → memory-bound.
  Move fewer bytes. Nothing else will help.
- **GB/s well below 195, and cutting bytes doesn't help** → you've left the
  memory-bound regime; something else is now the limit.
- **TFLOP/s near peak** → compute-bound. Do cheaper maths.

The decisive experiment is always: **change one thing by a large factor and see
if the time moves.** If reducing the math work 4× changes the runtime by 1%, the
math was never the bottleneck — and you've just saved yourself from optimising
it further. (That is a real result from lesson 07, not a hypothetical.)

---

## Measuring

### Use device time, not host time

Launching work has overhead: packaging commands, sending over PCIe, waiting for
completion. Measured in lesson 04 at 64 cores: the device does the work in
**70 µs**, the host observes **168 µs**. More than half the host number is
launch machinery.

Optimise against the host figure and you'll be tuning the dispatch path.

The dojo reports both, so you can see the gap:

```
time/iter        69.61 us   [device]
  host/iter     167.65 us   (dispatch + sync overhead)
```

### Warm up

The first run of a kernel **compiles it** — seconds, not microseconds. The dojo
discards 3 warm-up runs, then times 20.

### Change one variable

Every benchmark in this course sweeps exactly one thing: core count, or block
size, or fidelity. If you change two and it gets faster, you've learned nothing
about either.

### Treat small differences as noise

Run-to-run variation is a few percent. The dojo prints the spread:

```
spread   63.87 .. 65.35 us  over 20 runs
```

If your optimisation produced a 3% improvement, you have not yet measured an
improvement.

### Doing it yourself

Outside the dojo, the same machinery:

```bash
TT_METAL_DEVICE_PROFILER=1 \
TT_METAL_PROFILER_MID_RUN_DUMP=1 \
TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
python your_script.py
```

```python
ttnn.ReadDeviceProfiler(device)
summary = ttnn._ttnn.profiler.get_latest_kernel_duration_summary()
# {chip_id: KernelDurationSummary(count, min_ns, max_ns, avg_ns, histogram)}
```

These variables must be set **before the process opens a device** — the
profiling instrumentation is compiled into the dispatch path at startup.

For full timeline captures, including per-NoC-transaction events, tt-metal
integrates **Tracy**; the tooling is in `tt_metal/tools/profiler/`.

---

## The levers, with measured results

Roughly in order of how much they typically buy.

### 1. More cores — until saturation

*Lesson 04, element-wise add, 2048 tiles:*

| cores | time | bandwidth |
|---|---|---|
| 1 | 1293 µs | 9.7 GB/s |
| 2 | 647 µs | 19.4 GB/s |
| 8 | 170 µs | 74 GB/s |
| 32 | **64.6 µs** | **195 GB/s** |
| 64 | 69.6 µs | 181 GB/s |

Linear to 8, saturated by 32, regressing at 64. Find the knee; don't assume the
full grid is right.

### 2. Reuse — stop re-reading the same bytes

Usually the largest single win on a real operation, because it attacks the
traffic itself rather than how efficiently you move it.

*Lesson 08, matmul, 16 cores, varying how many output rows share each read of B:*

| block | time | traffic | bandwidth |
|---|---|---|---|
| 1 | 394 µs | 74 MiB | 197 GB/s |
| 2 | 222 µs | 42 MiB | 198 GB/s |
| 4 | 165 µs | 26 MiB | 165 GB/s |
| 8 | 160 µs | 18 MiB | 118 GB/s |

Read this carefully — it's the most instructive table in the course.

At blocks 1→2, bandwidth is **pinned at ~197 GB/s** and time tracks traffic
exactly. Textbook memory-bound: halve the bytes, halve the time.

At 4→8, traffic drops another 30% and time improves by **3%**. Bandwidth has
fallen to 118 GB/s, well off the ceiling. You are no longer memory-bound, and
further traffic reduction buys nothing.

### 3. Batched NoC transactions

Overlap latency instead of paying it serially (chapter 05).

*Lesson 05, 8 cores, varying tiles per barrier:*

| tiles/batch | time | bandwidth |
|---|---|---|
| 1 | 170 µs | 74 GB/s |
| 2 | 109 µs | 115 GB/s |
| 4 | 75.8 µs | 166 GB/s |
| 8 | **64.6 µs** | **195 GB/s** |

**2.6× on identical hardware and identical arithmetic.** And note that 8 cores
batched reaches the same 64.6 µs as 32 cores unbatched — efficiency substituting
for four times the hardware.

### 4. Deeper circular buffers

A CB with one page serialises producer and consumer completely. Two allows
overlap; `2 × block` is the usual choice when working in blocks. Costs L1.

### 5. Blocked DST usage

Amortise the math/pack handshake across a block instead of per tile
(chapter 06). Folded into the lesson 05 numbers above.

### 6. Math fidelity — *if* you're compute-bound

Up to 4× on the math, one line of host code. But check first:

*Lesson 08, 16 cores, block 8:*

| fidelity | time |
|---|---|
| HiFi4 | 159.8 µs |
| HiFi2 | 156.1 µs |
| LoFi | 155.9 µs |

A 4× reduction in math passes buys **2%**. In this course's matmuls, the FPU is
never the bottleneck — so this lever, which sounds like the most powerful one
available, does nothing.

That's the lesson, not a disappointment. It's why "what am I waiting on?" comes
before "what can I make faster?"

### 7. Cheaper data formats

`bfp8_b` halves the bytes per tile versus bfloat16 (chapter 03). On a
memory-bound kernel that approaches a 2× win, and it makes `LoFi` accurate
enough to use. Out of scope here, first thing to try next.

### 8. Multicast

When many cores need the same operand, broadcast it instead of each core reading
it from DRAM (chapter 07). Cuts that operand's traffic by the number of
receivers.

---

## What the numbers in the dojo mean

```
time/iter        64.60 us   [device]      ← mean device kernel duration
  spread   64.11 .. 65.23 us  over 20 runs ← run-to-run variation; judge noise
  host/iter    107.02 us                   ← includes dispatch; the gap is overhead
cores                 8
bandwidth        194.79 GB/s  (12.0 MiB moved)
throughput        13.442 TFLOP/s (2.15 GFLOP)
```

- **bandwidth** = bytes the kernel moves ÷ device time. Compare against
  ~195 GB/s.
- **throughput** = floating-point operations ÷ device time. A multiply-add
  counts as 2 FLOPs.
- **bytes moved** is computed from the *access pattern*, not the tensor size —
  so a kernel that reads B fifty times counts all fifty. Watching that number
  fall as you add reuse is the point.

---

## A workflow

1. **Get it correct first.** A fast wrong kernel is worthless, and the dojo
   refuses to benchmark one.
2. **Measure the baseline.** Note device time, GB/s, TFLOP/s.
3. **Work out the regime.** Is bandwidth near 195? Is throughput near peak? Is
   neither — meaning you're latency-bound and not keeping either resource busy?
4. **Pick the matching lever.** Memory-bound → fewer bytes (reuse, cheaper
   format) or better overlap (batching, depth). Compute-bound → fidelity,
   cheaper formats. Latency-bound → batching, more cores.
5. **Change one thing. Re-measure.** Did the pinned number come unpinned?
6. **Repeat until the remaining levers are all in the wrong regime.**

Step 3 is the one people skip, and it is the one that decides whether steps 4–6
are worth anything.

---

**Next:** [09 — Debugging](09-debugging.md) — what to do when it's wrong or
stuck.
