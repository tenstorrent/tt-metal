# Per-op power and energy breakdown of a transformer decoder block

Measures every op of a Llama-style decoder block separately and reports, per op, its dynamic
power, its energy contribution to one block pass, and its energy per FLOP.

**tt-ember itself is not modified.** `auto.py` drives any executable given as `--app-exe` and
parses its stdout, so `ttnn_ops_workload.py` is just a workload that happens to print the
summary rows `parser.py` expects. `preview_op_breakdown.py` only reads tt-ember's output.

Results already collected on Blackhole **p100a** are in `results/p100a/`. This document is
written so the same measurement can be reproduced on Wormhole **n150** and the two compared.

---

## 1. Prerequisites

### Exact versions the p100a results were produced with

Check these out rather than tracking `main`, so any difference between the two boards is the
board and not a four-month drift in the software. This is the information that was missing from
the earlier Wormhole runs and cost real time to reconstruct.

| component | pin |
|---|---|
| **tt-metal** | `e0c2360bb6ce706f801cd728b46e20b28f5b42d8` (upstream `main`, 2026-09-02) |
| **tt-umd** (submodule of the above) | `169f35494e4d7e7aabefb0da0da3c44500dddeb0` (2026-08-27) |
| tt-ember | this branch, off `ncvetkovic/fix-telemetry-tdp-regex` |

`tt-umd` is pinned by tt-metal's `.gitmodules`, so `git submodule update --init --recursive` on
the tt-metal commit above lands on it automatically. Verify with:

```bash
git -C tt_metal/third_party/umd rev-parse HEAD   # expect 169f35494e4d...
```

Host-side environment on the p100a machine, for reference. None of it is believed to be
load-bearing, but record yours when reporting so the two runs can be compared:

```
TT-KMD 2.9.0 · tt-smi 5.2.0 · tt_umd (host) 0.9.5 · pyluwen 0.8.5
firmware bundle 19.12.0 · Ubuntu 22.04.5 · venv python 3.10.19 · torch 2.11.0+cpu
```

### tt-metal

```bash
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git checkout e0c2360bb6ce706f801cd728b46e20b28f5b42d8
git submodule update --init --recursive
./build_metal.sh --build-type Release --enable-ccache
export TT_METAL_HOME=$PWD
```

Python bindings are on by default; this workload needs them. The build takes roughly 20 minutes
cold. Nothing beyond a stock build is required -- in particular the `high_power_matmul`
programming example and its `POWER_CASE` work, which live on other branches, are **not** used
here. This workload needs only `ttnn` and `build_Release/tools/umd/telemetry`.

### The tt-metal python venv

`ttnn` needs tt-metal's own dependency set; a bare venv with numpy will not import it. Build it
with the script tt-metal ships:

```bash
cd $TT_METAL_HOME
./create_venv.sh --env-dir /path/to/ttnn_venv
source /path/to/ttnn_venv/bin/activate
python3 -c "import ttnn; print(len(dir(ttnn)))"   # expect several hundred attributes
```

That venv also carries numpy/matplotlib, so it can serve as tt-ember's `--tt-venv-activate`
too. **Activate it before invoking `auto.py`**, because the workload script is launched by
`auto.py` with an inherited environment and resolves `python3` from `PATH`.

### tt-ember

Use this branch. It contains one fix beyond `main` that is **mandatory** against any current
tt-metal:

> Current `tt-umd` prints TDP as a value/limit pair, `TDP 16/150 W`. `parser.py` on `main`
> requires a bare value, so the TDP match fails; `parse_line()` needs all three of TDP/TDC/
> VCORE, so **every telemetry sample is discarded** and the run aborts with
> `No telemetry lines matched. Check log format.`

If you see that error, you are on a tt-ember without the fix.

---

## 2. Run it

```bash
cd /path/to/tt-ember
export TT_METAL_HOME=/path/to/tt-metal
source /path/to/ttnn_venv/bin/activate

tt-smi -r && sleep 15          # clean thermal/electrical baseline

python3 auto.py \
  --telemetry-exe "$TT_METAL_HOME"/build_Release/tools/umd/telemetry \
  --telemetry-freq 100 \
  --app-exe /path/to/tt-metal/tools/tt-ember/op_power_breakdown/ttnn_ops_workload.py \
  --parser-script ./parser.py \
  --tt-venv-activate /path/to/ttnn_venv/bin/activate \
  --tt-metal-root "$TT_METAL_HOME" \
  --output-root ./out_ops --subdir block_n150 \
  --slot-ms 1 --device-id 0 --trim-ms 1.0 \
  --app-args --seq 1024 --hidden 2048 --ffn 8192 --heads 16 \
             --target-call-ms 1.5 --max-batch 96 \
             --target-seconds 4.0 --pause-seconds 5.0
```

`--app-args` **must be last** — `auto.py` uses `argparse.REMAINDER` and will otherwise swallow
every following argument.

Runtime is roughly 3 minutes: 12 ops x (4 s window + 5 s idle gap) plus device init and kernel
JIT on the first run.

Then render the breakdown:

```bash
python3 /path/to/tt-metal/tools/tt-ember/op_power_breakdown/preview_op_breakdown.py ./out_ops/block_n150
```

That prints the table and writes two figures into `out_ops/block_n150/Figures/`:

| file | what it shows |
|---|---|
| `op_energy_breakdown.png` | energy per op for one block pass, split into dynamic and idle-floor share |
| `op_power_breakdown.png` | three panels over one op axis: pJ/FLOP (log scale), mJ/pass, dynamic W |

tt-ember's own 15 figures are written alongside. Most of them plot against *core count*, which
this workload repurposes as an op index, so only `telemetry_overview.png` is meaningful from
that set — it shows the 12 ops as plateaus separated by idle gaps, which is a good sanity check
that the run is well formed.

---

## 3. Wormhole n150 notes

The workload adapts itself, but two differences matter.

**Less DRAM.** n150 has 12 GB against p100a's 28 GB. Batch sizes are chosen at runtime and the
script halves the batch and retries on allocation failure, printing `# NOTE <op>: retrying at
batch=N`. A few of those lines are expected and fine. If an op ends up at `batch=1` and its
call is still well under 1 ms, see the accuracy note below. If the run fails outright, lower
`--max-batch` (try 32) or shrink the shape, e.g. `--seq 512 --hidden 2048 --ffn 4096`.

**Fewer cores.** n150 exposes an 8x8 compute grid against p100a's 11x10, so absolute power and
throughput will be lower. That is the thing being measured, not a problem.

**Keep the shape identical to the p100a run** if you want a like-for-like comparison —
`--seq 1024 --hidden 2048 --ffn 8192 --heads 16`. Only drop it if memory forces you to, and say
so when reporting.

---

## 4. Reading the results

From the p100a run, for orientation:

| | p100a |
|---|---|
| energy for one block pass | **234 mJ** (110 mJ dynamic + 124 mJ idle floor) |
| block latency | 2081 us |
| idle floor | ~60 W |
| dense projections (qkv/out/ffn_up/ffn_dn) | 0.8-0.9 pJ/FLOP, 60% of total energy |
| elementwise and norms | 50-270 pJ/FLOP, ~4% of total energy |

Three things worth checking on n150:

1. **The idle floor was 53% of block energy on p100a.** n150's floor is far lower in absolute
   terms (~28 W versus ~60 W), so this fraction should differ substantially. It is probably the
   single most interesting number to compare.
2. **`attn_av` was the anomaly on p100a** — third-largest total energy consumer (15%) despite
   modest dynamic power, because it is the slowest op in the block (470 us) at low utilisation.
   Worth seeing whether that reproduces.
3. **The pJ/FLOP panel is log-scaled**, because the values span 0.8 to 270 -- on a linear axis
   the matmul bars are about a pixel tall and read as missing. Every bar is labelled with its
   value. **pJ/FLOP for elementwise ops is 50-300x worse than for matmuls**, because they do ~1 FLOP
   per element while streaming the whole tensor. This is expected, not a defect, and those ops
   are a negligible share of energy — the two panels deliberately tell opposite stories.

---

## 5. Accuracy notes

Read these before treating any number as precise.

**Telemetry time resolution is ~103 us**, regardless of `--telemetry-freq` — the tool caps
around 9.7 kHz, so requesting 50 us gains nothing. An op call shorter than that is never
resolved, and its measured power is diluted by whatever host dispatch gap follows it. Measured
on p100a: a 41 us call reads about 24% low, and the effect vanishes once a call spans roughly
ten samples. This is exactly why the script batches each op to ~1.5 ms per call. **Check the
`# DONE` lines: any op well under ~1 ms per call is probably reading low.** On the p100a run
`rms_norm` and `residual_add` only reached ~700 us even at batch 32, so they may still be
slightly understated; they are under 1% of block energy, so it does not matter there.

**Current is quantised to 1 A**, but this is not a limitation: averaged over ~30,000 samples
per interval the standard error is about 0.002 A. Quantisation is not what limits accuracy.

**Baseline drift is what limits accuracy.** The idle floor climbs as the board warms — about
9 A across a p100a run. `parser.py` estimates it from the pauses either side of each interval,
which gives roughly 1 A of uncertainty per interval: under 1% on a 130 A matmul, but 5-8% on a
15 A elementwise op.

**FLOP counts are exact only for matmuls** (`2*M*N*K`). Elementwise and normalisation ops use
per-element conventions declared in `FLOPS_PER_ELEM` at the top of `ttnn_ops_workload.py`
(add/mul 1, silu 4, rms_norm 4, softmax 5). Hatched bars in the pJ/FLOP panel mark these. The
50-300x gap survives any reasonable choice of convention; the precise ratios do not.

**Idle-floor attribution.** Each op is charged the full idle floor for its duration, which
answers "what did this op cost in a serial block pass". It does **not** mean removing the op
would recover that energy — the floor is there regardless. Only the dynamic column is
recoverable.

**One measurement per op, single board.** No error bars. Re-run to gauge repeatability before
relying on any difference smaller than a few percent.

---

## 6. How it fits together

`parser.py`'s `RE_PROGRAM_ROW` only understands a compute grid, so the workload borrows those
fields:

* the `grid` field carries the op index as `<idx>x1`, making every op a distinct interval;
* the `cores` field carries the same index, so tt-ember's own figures separate the points;
* op names are printed as `# OP <idx> <name> iters=... flops_per_iter=... batch=...` lines,
  which `RE_PROGRAM_ROW` does not match and `parser.py` therefore ignores.

`preview_op_breakdown.py` reads those `# OP` lines back out of `summary.txt` and joins them to
`program_intervals.csv` on the index. That is the whole mechanism — no tt-ember change.

Each op is run repeatedly for ~4 s so the interval is long enough to measure, and separated by
a 5 s idle gap because `parser.py` derives the baseline from the pauses between intervals. Both
are configurable via `--target-seconds` and `--pause-seconds`.
