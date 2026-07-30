# tt-metal kernel dojo

An interactive, exercise-driven course on writing Tensix kernels. You write real
kernels, a grader runs them on real hardware against a torch reference, and a
profiler tells you how fast they were.

```
./dojo theory        # the background reading
./dojo list          # the syllabus
./dojo info 01       # read the lesson
                     # ...edit exercises/01_tile_copy/kernels/*.cpp...
./dojo test 01       # grade it
./dojo bench 01      # measure it
```

**New to accelerator programming?** Start with `./dojo theory 00` and
`./dojo theory 01`. They assume no background beyond C++ and explain the
concepts the exercises rely on — latency vs bandwidth, pipelining, batching,
double buffering, deadlock, and how to tell what your kernel is waiting on.
Roughly 30 minutes, and the rest of the course is much easier afterwards.

---

## Setup

You need a built tt-metal in this repo and a Tenstorrent device visible to the
machine. From the repo root:

```bash
./build_metal.sh --enable-ccache --release
```

Then check the dojo can see everything:

```bash
./dojo doctor
```

That verifies `ttnn` imports, `TT_METAL_HOME` is set, the device opens, and
reports the compute grid size. The `./dojo` wrapper sets `TT_METAL_HOME`,
`PYTHONPATH`, and (for `bench`) the profiler environment variables, so you
should not need to export anything yourself.

You also need `torch` in the environment — it computes every golden result.

---

## The syllabus

| | Lesson | What it teaches |
|---|---|---|
| **01** | Tile copy | Circular buffers, NoC reads/writes, barriers, tiles |
| **02** | Element-wise unary | The compute pipeline, DST registers, SFPU |
| **03** | Element-wise binary | FPU ops, two operands, overlapping reads |
| **04** | Multi-core | Work splitting, per-core runtime args, DRAM bandwidth limits |
| **05** | Pipelining | Batched NoC transactions, blocked DST use, CB depth |
| **06** | Matmul | K-accumulation in DST, `SrcOrder::Reverse`, arithmetic intensity |
| **07** | Matmul at scale | Operand reuse, row parallelism, math fidelity |
| **08** | Output blocking | 2-D reuse, and finding out what's *actually* the bottleneck |

The last two lessons come with measured numbers in their READMEs, including a
couple of results that contradict the obvious guess — those are the ones worth
reading closely.

01–03 are about *correctness* — learning the programming model. 04–05 are about
*throughput* on a memory-bound op. 06–08 move to matmul, where the optimisation
calculus is different and the interesting question becomes which resource you
are actually waiting on.

Do them in order. Each lesson's provided kernels are the previous lesson's
solution, so skipping ahead means reading code you haven't written yet.

---

## How an exercise is laid out

```
exercises/01_tile_copy/
├── README.md      the lesson: theory, task, API reference, hints
├── skeleton/      pristine starting kernels — never edited
├── kernels/       YOUR working copy (created on first `info`/`test`)
├── solution/      reference implementation
└── task.py        host-side setup, test cases, golden, perf targets
```

You edit `kernels/`. Everything else is scaffolding.

Some exercises hand you completed kernels alongside the ones you write — a
lesson about compute kernels shouldn't make you rewrite a reader you already
built. The lesson README says which files are yours.

---

## Commands

| | |
|---|---|
| `./dojo theory` | list the theory chapters |
| `./dojo theory <n>` | read a chapter |
| `./dojo list` | the syllabus |
| `./dojo info <n>` | print a lesson (honours `$PAGER`) |
| `./dojo test <n>` | run all correctness cases |
| `./dojo test <n> --case "64 tiles"` | run one case |
| `./dojo test <n> --solution` | grade the reference instead of your code |
| `./dojo test <n> -v` | full traceback on failure |
| `./dojo bench <n>` | device-timed performance run |
| `./dojo solution <n>` | print the reference |
| `./dojo solution <n> --apply` | copy the reference into `kernels/` |
| `./dojo reset <n>` | restore the skeleton, discarding your work |
| `./dojo doctor` | environment check |

Exercises can be named `1`, `01`, `01_tile_copy`, or `matmul` — anything
unambiguous.

---

## How grading works

For each case the grader:

1. Generates inputs with a **fixed seed**, so a failure is always reproducible.
2. Computes the golden result on the CPU with torch.
3. Uploads the inputs, builds a `ttnn.ProgramDescriptor` pointing at *your*
   kernel files, and runs it via `ttnn.generic_op`.
4. Downloads the output and compares.

Your kernels are JIT-compiled by tt-metal on each run, so there is no build step
between editing and testing.

The comparison reports **PCC** (Pearson correlation) and an element-wise
tolerance. Both must pass. Exact equality is only required where it's
meaningful — a copy — because bfloat16 arithmetic legitimately rounds.

When a case fails you get the first mismatching element with its expected and
actual values, which is usually enough to identify an indexing bug immediately.

---

## How the performance numbers work

`./dojo bench` enables tt-metal's **device profiler**, which timestamps kernel
dispatches on the device itself. The reported `time/iter` is silicon time, not
host wall clock — for kernels this small the host number is mostly dispatch
overhead and would tell you nothing about your code.

Both are shown so you can see the difference.

Each benchmark:
- runs 3 warm-up iterations (the first compiles kernels and fills the program
  cache — that's seconds, not microseconds),
- then times 20 iterations and reports mean, min and max,
- **verifies correctness first** — a fast wrong kernel is not a result.

Where a workload is defined, you also get:

- **GB/s** — bytes across the NoC ÷ device time. The number to watch on
  memory-bound kernels (01–05).
- **TFLOP/s** — the number to watch on compute-bound kernels (06–07).

Several lessons sweep a parameter — core count, block size, math fidelity — so
`bench` prints a curve rather than a point. The curve is the lesson; a single
number would not teach you where the knee is.

> Timings vary a few percent run to run. Treat differences under ~5% as noise,
> and re-run before concluding anything from a small change.

---

## When things go wrong

**The test hangs.** Almost always a circular-buffer mismatch: a producer that
reserves without pushing, a consumer that waits for tiles nobody sends, or two
kernels that disagree on the tile count. Kill it with Ctrl-C and re-read your
`cb_*` calls in pairs.

**Output is zeros or garbage.** Usually a missing `noc_async_read_barrier()`
before you use data, or a missing `noc_async_write_barrier()` before you free
the page you're writing out of.

**Your editor shows errors in every kernel file.** Expected. Kernels compile
under tt-metal's JIT toolchain with its own include paths, not your host
toolchain, so clangd cannot resolve `api/dataflow/dataflow_api.h`. The dojo
compiles them fine. If it bothers you, tt-metal can generate a
`compile_commands.json` for kernels via `build_metal.sh
--enable-fake-kernels-target`.

**A kernel change seems to have no effect.** Check you edited `kernels/` and not
`skeleton/` or `solution/`.

---

## Where to go next

[`THEORY.md`](THEORY.md) indexes ten chapters covering the whole model from
first principles, including the material the exercises only gesture at:
sharding, multicast, semaphores, and the profiler.

In the main repo:

- `tt_metal/programming_examples/` — the same ideas as standalone C++ programs,
  including the multicast matmul that lesson 07 points at.
- `METALIUM_GUIDE.md` — the official architecture guide.
- `tt_metal/hw/inc/api/compute/` — every compute API, documented in the headers.
- `tt_metal/hw/inc/api/dataflow/dataflow_api.h` — every data movement API.
