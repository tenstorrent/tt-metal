# Reading a profile

## The warm window — establish it first

Every headline `tt-perf-report` prints is computed over whatever slice it was
given, and by default that is the entire file: construction, weight upload,
activation prep, first-iteration compilation, then the inference you care about.

**How to spot the prep block.** Weight upload and activation preparation are not
subtle in a profile — they show up as a **run of `TilizeWithValPadding` /
`Untilize` / layout ops at the head of the capture, before the first real compute
op**, usually with huge op-to-op gaps between them. Everything above the first
`Matmul`/`Conv3d`/`SDPA` of the first real layer is setup, not inference.

Either exclude it or don't measure it:

| Method | How |
|---|---|
| **Signpost only the model call** (best) | Put weight upload, input prep *and* the warm-up iteration **outside** the signposted region — `tracy-capture.md`. Then `--start-signpost` / `--end-signpost` |
| **`--id-range <n>-`** | Find the ID of the first real compute op and slice from there. Use when the capture is already taken and you can't re-instrument |
| **Tail in pandas** | `df.tail(N)`, `N` ≈ one warm iteration's op count. Check the gap distribution stabilizes as `N` shrinks |

State which method you used. A profile with an unspecified window is not
reproducible, and one that silently includes the prep block is worse than no
profile — it will tell you data movement dominates when it does not.

## The op-to-op gap — the trap that caught this tree

`tt-perf-report` prints:

```
These ops have a >6 us gap since the previous operation.
Running with tracing could save 47463439 us (97.1% of overall time)
```

**Never quote that without checking the `OP TO OP LATENCY [ns]` distribution.**
It is computed over the whole file, and weight upload produces individual gaps
of 650–880 ms. A handful drag the mean up four orders of magnitude while the
median stays under a microsecond.

Real numbers from one video-VAE encoder profile:

| window | device time | op-to-op gap | gap share |
|---|---|---|---|
| last 500 ops | 1115.6 ms | 9212.9 ms | 89.2% |
| last 400 ops | 1100.1 ms | 5171.5 ms | 82.5% |
| **last 300 ops** | **568.6 ms** | **110.0 ms** | **16.2%** |

Median gap 0.6 µs, mean 18425.9 µs. The share collapses as the window excludes
construction. Correct reading: **in steady state device time is the bottleneck,
not dispatch** — the opposite of the headline.

Acting on the headline sent an agent down a trace-capture path that was not on
the critical path, and it took a retraction amendment to undo. **Report median
and mean, always.** When they differ by orders of magnitude you have not found
your warm window yet.

## Ranking

Rank by `DEVICE FW DURATION [ns]` — first RISC firmware start to last RISC
firmware end. For a layer or block scope add a contribution column; an op is
*the* bottleneck iff its contribution exceeds the sum of all others.

| Column | Tells you |
|---|---|
| `OP CODE` | C++ class name |
| `OP TYPE` | `tt_dnn_device` ran on device; `python_fallback` / `tt_dnn_cpu` did not |
| `CORE COUNT` | Cores dispatched to. Compare against the grid the device reports (`mesh_device.compute_with_storage_grid_size()`), not the SoC nominal — harvesting varies. Low count on a heavy op = under-parallelized |
| `PARALLELIZATION STRATEGY` | Missing or fallback usually means suboptimal placement |
| `MATH FIDELITY` | LoFi / HiFi2 / HiFi3 / HiFi4. Over-specified fidelity is a common silent cost |
| `DEVICE FW DURATION [ns]` | Headline |
| `DEVICE KERNEL DURATION [ns]` | Compute-kernel window. `FW − KERNEL` ≈ setup + teardown + CB flush |
| `DEVICE BRISC/NCRISC KERNEL DURATION [ns]` | Reader (DM0) / writer (DM1) |
| `DEVICE TRISC0/1/2 KERNEL DURATION [ns]` | Unpack / Math / Pack |
| `OP TO OP LATENCY [ns]` | Dispatch gap — see above |
| `HOST DURATION [ns]` | Queue + host dispatch. Inflated on the first iteration |

## Per-RISC bound tags

The RISC with the largest kernel duration relative to its peers is the bound.

| Tag | Signal |
|---|---|
| `reader-bound` | BRISC ≳ TRISC1 — compute waits on tiles |
| `compute-bound` | TRISC1 dominates — fidelity or a real ceiling |
| `writer-bound` | NCRISC dominates — downstream pressure or small output CB |
| `under-parallelized` | Low `CORE COUNT`, high per-core duration |
| `NOC-stall` | BRISC and NCRISC high, TRISC low |
| `host-dominated` | Short `FW`, large `HOST DURATION` past iteration 1 — verify against the gap distribution first |

Record the tag; lever selection belongs to `tt-dit-performance`.

## Op-level bound class

`tt-perf-report` emits `DRAM %` / `FLOPs %` for matmuls and tags suspect rows
`SLOW`. Different question from the per-RISC tags — this classifies the op, not
which RISC blocks inside it.

The cutoffs below are **rules of thumb for picking a lever family, not measured
constants** — treat a value near a boundary as "inspect per-RISC", not as a
verdict.

| DRAM % | FLOPs % | Bound |
|---|---|---|
| < 40 | < 40 | **overhead / sync** — dispatch, barriers, undersized blocks |
| ≥ 60 | < 40 | **bandwidth** |
| < 40 | ≥ 60 | **compute** |
| — | ≥ 70 | **near peak** — stop tuning this op |

Confirm the class before proposing levers. Compute-bound levers on an
overhead-bound op is the most common way an optimization loop stalls.

**Overhead confirmation (same caveat — indicative bands):** `overhead_ratio = (FW − KERNEL) / FW`. Under 15%,
compute or bandwidth dominates. Over 40%, dispatch and sync dominate regardless
of FLOPs%.

## Peaks

Let `tt-perf-report` compute `FLOPs %` / `DRAM %` — it knows the per-arch peaks
and auto-detects on new reports (`--arch blackhole`). Do not hand-derive a
ceiling.

Two facts you do need:

- **HiFi2 peak is 2× HiFi4** (math cycles per tile halve); LoFi is another 2×.
- `FLOPs %` is against the *current* fidelity's peak, so HiFi4 → HiFi2 can show
  FLOPs% falling while throughput rises. **Compare absolute TFLOPs across a
  fidelity change, never FLOPs%.**

### What `CORE COUNT` should be compared against

Not the SoC descriptor. `blackhole_140_arch.yaml` lists 140 functional workers,
but that is the physical count — the usable compute-with-storage grid is **120**,
and on a Blackhole Galaxy tt_dit clamps matmuls further:

```python
# utils/matmul.py
_BH_GALAXY_MAX_CORE_GRID = (11, 10)   # 110 cores, power constraint at >= 32 devices
```

So on a Galaxy, an op on 110 cores is **fully parallel**, not under-parallelized.
Read `CORE COUNT` against `mesh_device.compute_with_storage_grid_size()` and, for
matmuls, against what `get_matmul_core_grid()` actually returns — otherwise every
matmul on a Galaxy looks under-parallelized when it is at the ceiling.

## TT-DiT-specific patterns

| In the profile | Means |
|---|---|
| `Untilize` + `TilizeWithValPadding` bracketing `GroupNorm` near the top | The layout round-trip. Measured 52.8% of warm encoder device time on a video VAE — larger than the norm itself |
| `Concat` / `Permute` / `Reshape` / `Transpose` individually cheap | Group them. 36% of one ViT layer was pure data movement, mostly head reshaping a fused head op removes |
| `BinaryNg` / `Unary` at HiFi4 | Elementwise paying for precision it does not need |
| `Conv3d` dominant in a video VAE, with blocking-fallback warnings in the log | The tuned table missed; you are on a conservative default |
| Two ops doing identical work at very different durations, or an op on 57 cores while neighbours get 120 | An anomaly, usually a wrong config. Cheap to fix — chase it |

## When numbers look wrong

| Symptom | Cause |
|---|---|
| All device durations zero | Env conflict — watcher / DPRINT / `TTNN_CONFIG_PATH` |
| Only a handful of ops, or `Device data missing: Op <id>` | Tracy buffer overflow (`tracy-capture.md`) |
| Wildly different across runs | Program-cache miss on the measured iteration, or another workload on the device |
| Gap share > 80% | Construction is still in the window. Shrink it |

**Never judge a per-op change by whole-model wall clock.** One recorded case
concluded an explicit SDPA config was a regression (0.652 s → 0.876 s per wave);
the same code spans 0.34–0.99 s/wave, so the conclusion was noise. Measure the
op, under the profiler.
