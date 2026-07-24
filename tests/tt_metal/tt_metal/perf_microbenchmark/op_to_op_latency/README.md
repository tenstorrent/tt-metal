# `op_to_op_latency` — back-to-back program latency & DRAM-BW benchmark

Measures per-core latency and DRAM bandwidth across a back-to-back program (op) boundary on
Wormhole: from when one program finishes writing to when the next starts reading/computing.

## Pipeline

On every Tensix core, the same `MeshWorkload` is enqueued `--num-programs N` times (FD or trace):

```
reader (NOC0) ──CB_in──▶ compute (3 TRISCs) ──CB_out──▶ writer (NOC1)
  interleaved DRAM        UNPACK / MATH / PACK          write + end barrier
```

Kernel markers in the device CSV (all kernels also emit `PROG_ID`; 0 = warmup, 1..N = timed):

| RISC | role | markers |
|---|---|---|
| TRISC_0 unpack | `TILE_IDX`=*i* at tile *i* start |
| TRISC_2 pack | `FINISH_LAST_PUSH` at kernel exit |
| NCRISC reader | `NCRISC_GO/DONE`; `READ_BEFORE/LAST_BARRIER` |
| BRISC writer | `BRISC_GO/DONE`; `WRITE_BEFORE/LAST_ISSUED/AFTER_BARRIER` |

## Build & run

```bash
export TT_METAL_HOME=$(pwd) TT_METAL_DEVICE_PROFILER=1
./build_metal.sh --build-tests            # first time
cmake --build build_RelWithDebInfo --target test_op_to_op_latency -j
BIN=build_RelWithDebInfo/test/tt_metal/perf_microbenchmark/op_to_op_latency/test_op_to_op_latency

$BIN --use-trace --num-programs 2 --compute-nops 1000 --use-device-profiler
```

**Steady-state recipe** (what the sweeps use): trace + warmup so timings stabilize, drop trace-start
transitions, per-trid double-buffered reader, output CB past the compute-bubble knee.

```bash
$BIN --num-active-cores 56 --num-pages-per-core 64 --compute-nops 256 \
  --input-cb-depth-tiles 16 --output-cb-depth-tiles 32 \
  --reader-dbuf-trid --reader-trid-in-flight 4 --reader-push-tiles 2 \
  --writer-end-barrier-mode 0 --num-programs 4 --use-trace --trace-warmup-replays 2 \
  --lean-compute --skip-output-validation --use-device-profiler
```

## Profilers — which to use

- **Realtime** (`--use-realtime-profiler` → `profile_log_device_rt.csv`): per-program dispatch
  `go`/`done`, **dump-free**. Use for **absolute op-to-op / wall-clock** numbers.
- **Device** (`--use-device-profiler` → `profile_log_device.csv`): rich per-core markers, but its
  per-program L1→DRAM dump (~3 µs) lands in the op-to-op gap and inflates it. Use for **per-config
  comparison, BW/skew, and per-core shape** — not absolute boundary timing.
- **Accumulate** (`TT_METAL_PROFILER_ACCUMULATE=1`, upstream "Profiler L1 buffer accumulate mode" #48506):
  keeps markers in L1 across programs to defer that dump. Evaluated but **not usable here** — it drops
  `PROG_ID`/the per-op report and asserts on this benchmark's custom markers. Not relied upon (no longer
  carried on this branch); noted only so you don't reach for it.

## Key CLI flags

| flag | default | meaning |
|---|---:|---|
| `--num-active-cores N` | 0 (full grid) | cap active cores (core-count sweeps) |
| `--num-pages-per-core N` | 4 | interleaved DRAM pages/core = per-op work knob |
| `--input-cb-depth-tiles N` / `--output-cb-depth-tiles N` | auto / 2 | CB depths |
| `--reader-push-tiles N` | 2 | reader push-chunk size |
| `--reader-dbuf-trid` / `--reader-trid-in-flight N` | off / 2 | reader mode 2 (per-trid double buffer) + reads-in-flight |
| `--reader-read-bytes N` | full page | read only N bytes but push a full page (output-bound isolation) |
| `--compute-nops N` | 0 | `TTI_NOP`s per tile (compute load) |
| `--lean-compute` | off | minimal passthrough compute (nops still apply) |
| `--num-programs N` / `--kernel-unroll K` | 8 / 1 | back-to-back enqueues / reps fused into one program (no mid-barrier) |
| `--use-trace` / `--trace-warmup-replays N` | off / 0 | capture-once-replay / untimed warmups |
| `--writer-end-barrier-mode {0,1,2}` | 0 | end sync: barrier / flush-only / none |
| `--read-only` | off | writer skips DRAM writes (isolate read BW) |
| `--cross-program-dram-offset` | off | each program uses a disjoint DRAM slice |
| `--write-progress-every N` / `--read-progress-every N` | 0 (off) | emit periodic `WRITE_PROG`/`READ_PROG` bytes-vs-time markers (guarded; instrumentation only) |
| `--core-offset` / `--core-list "x,y;.."` / `--log-core-map` | — | place/log the active core set (logical coords) |
| `--buffer-tune[-grid]` | off | sweep CB depths for peak DRAM BW, then run at the smallest within tolerance |

## Reader / writer modes

- **Reader**: 0 push-1 incremental (default) · 1 `--reader-batch-push` (reserve N, one barrier) ·
  2 `--reader-dbuf-trid` per-trid double buffer (highest per-core BW; needs
  `input_cb_depth ≥ 2 × trid_in_flight × page_size_tiles`).
- **Writer**: flushes once per CB-worth (never per page). End sync via `--writer-end-barrier-mode`:
  `0` barrier (DRAM ACK) · `1` flush (L1 drain only) · `2` none (next op may overlap in-flight writes).

## Sweep scripts

All driver/analysis/chart scripts live in `scripts/`. Each sweep writes per-config runs under
`generated/profiler/op_to_op_runs/…` and drives `decompose_latency_bw.py`; run from anywhere (they
self-locate the repo root) and re-run with `BUILD=0` if the binary is already built.

| script | purpose |
|---|---|
| `run_stage_a_sweep.sh` | Stage-A balanced-compute decomposition (layout × cores; tune CB, then latency) — the central sweep |
| `run_bw_vs_cores.sh` | aggregate BW + done-skew + starvation vs core count (`READ_ONLY=1` for read-only) |
| `run_bw_vs_trid_cores.sh` | read-only BW vs reads-in-flight (N) × core count |
| `run_skew_vs_nops.sh` | writer done-skew vs compute load (nops/tile) |
| `run_io_latency.sh` | two-sided I/O latency: read-fill vs N, and write-drain, at cores {8,56} |
| `run_noc_dependence.sh` | read-only BW vs spatial extent (rows/cols) × NoC |
| `run_read_stagger.sh` | induced per-core read-stagger → write de-correlation study |
| `run_unroll_vs_b2b.sh` | fused (`--kernel-unroll K`, no mid-barrier) vs back-to-back wall-clock |

## Analysis / chart scripts

| script | purpose |
|---|---|
| `export_op_to_op_profiler_csv.py` | main exporter: device (+ rt) CSV → per-transition gaps, summaries, timeline; `--min-prog-id N` drops trace-start transitions |
| `decompose_latency_bw.py` | per-sweep analyzer (BW, skew, latency components); driven by every sweep |
| `event_timeline.py` | per-op event timeline (used by stage-A / read-stagger sweeps) |
| `chart_rolloff.py` | robust BW-over-time / roll-off & fast-vs-slow allocation (`--anchor-first`) |
| `chart_txnsize.py` | `overall_bw_gbps` (total bytes / kernel envelope) + done-spread vs core count, one line per DRAM txn size (2K/4K/8K = `--page-size-tiles 1/2/4`); `--csv`/`--title`/`--note`/`--bw-label` (used for both read and write charts) |
| `chart_raw_graph.py` | render the device-program RAW dependency DAG (graphviz PNG+SVG) from a capture: green border = boundary relaxable, orange = unresolved (predecessor in-place), gray dashed = in-place op, black arrow = RAW dep; `--resolve-inplace`/`--add-edge P:C`/`--range START:END` (see RAW-hazard section) |
| `chart_timeline_k2.py` | K=2 back-to-back vs fused per-core timeline (also provides `per_core_invocations`) |
| `chart_active_cores.py` · `chart_core_heatmap.py` | active-cores + delivered BW over time · per-core finish heatmap |
| `chart_read_stagger.py` · `chart_timeline_bars.py` · `chart_unroll_vs_b2b.py` | figure generators for the corresponding sweeps |

## RAW-hazard analysis (cross-model barrier-relaxation opportunity)

`raw_hazard_analyzer.py` measures how often back-to-back **device programs** have **no** read-after-write
dependency — the legality signal for relaxing an op-boundary barrier. It parses a ttnn graph capture,
reduces it to device programs, matches producer→consumer on **buffer address** (tensor_id is re-wrapped
between ops and unreliable), and reports: adjacent-RAW / confirmed-reorderable / unresolved-in-place %,
distance-to-first-consumer, critical path (independence ceiling), WAW (true vs allocator-reuse), and
shard locality (cross-core reliable floor vs same-core upper bound). Has `--self-test`.

Capture a model's graph (random weights — only op *structure* matters, not values), then analyze:

```bash
PY=python_env/bin/python3
D=tests/tt_metal/tt_metal/perf_microbenchmark/op_to_op_latency/scripts
$PY $D/capture_mamba.py                        # -> /tmp/mamba_capture.json
$PY $D/raw_hazard_analyzer.py /tmp/mamba_capture.json
```

| capture script | model / architecture type | notes |
|---|---|---|
| `capture_decoder_layer.py` | Llama-style decoder layer (transformer decode) | fused SDPA |
| `capture_sdxl_unet.py` | SDXL UNet (diffusion, conv+attention) | needs slow dispatch |
| `capture_mamba.py` | Mamba-2.8b (state-space / selective scan) | needs slow dispatch; config is hardcoded for d_model=2560 (use the 2.8b variant) |
| `capture_whisper.py` | Whisper-base (encoder-decoder, cross-attention) | fast dispatch OK |

Visualize any capture with `chart_raw_graph.py <capture.json> --out <base>` (graphviz DAG, see chart
table). The most faithful Llama graph comes from the **real checkpointed** `TransformerBlock` (fused QKV,
paged KV, in-place RoPE/cache), not a hand-built layer: run `models/tt_transformers/tests/test_model.py`
`-k "performance and quick"` (`HF_MODEL=unsloth/Llama-3.2-1B-Instruct MESH_DEVICE=N150`, dummy weights)
with a monkeypatch plugin that wraps `TransformerBlock.forward` in graph capture.

**Two gotchas (each cost a debugging round):**

- Use **`RunMode.NORMAL`**, not `NO_DISPATCH` — `NO_DISPATCH` skips buffer allocation, so tensors carry
  no address and producer→consumer chaining collapses into falsely-low RAW (e.g. SDXL read a bogus 92%
  reorderable-adjacency vs the real 26% under NORMAL).
- Models whose ttnn configs hardcode a full **8×8** grid (SDXL, Mamba) abort config validation on this
  box's harvested **8×7** grid (`program_config grid (8x8) must be contained within device grid (8x7)`).
  Run with **`TT_METAL_SLOW_DISPATCH_MODE=1`** — slow dispatch frees the Tensix row that fast dispatch
  reserves for dispatch kernels, exposing the full 8×8. Capture only needs one forward, so slow is fine.

**Resolving dropped edges — `--resolve-inplace`, `--add-edge`, `--range`.** The capture drops dependency
edges wherever an op emits **no output tensor**: in-place RMW ops (RoPE, PagedFusedUpdateCache), **all
`NLPCreateHeads`**, and reshape/relayout **views** (buffer address lost). Consumers (especially SDPA) then
look like independent roots, so raw critical-path slack is badly inflated. Flags on `raw_hazard_analyzer.py`
(and `chart_raw_graph.py`):

- `--resolve-inplace` — treat a no-output op as read-modify-write (inputs ARE its outputs) **and** auto-infer
  create-heads→SDPA edges. Use it for any attention model; it only ADDS edges (can never fabricate slack).
  This is the trustworthy default. `chart_raw_graph.py` draws inferred edges dashed-orange.
- `--add-edge P:C` — inject a known-real edge the capture can't recover (e.g. a reshape view). Explicit and
  auditable; the Llama hand-verification uses `--add-edge 3:4 --add-edge 6:8` (view + RoPE→SDPA).
- `--range START:END` (chart only) — draw a legible slice of a huge graph, e.g. one UNet block.

**Findings (cross-model, `--resolve-inplace`).** `can't-hide` = critical-path / total = the latency floor
after ideal reordering (everything off the longest chain is hideable); `reorder payoff` = 1 − can't-hide.

| model | can't-hide | reorder payoff | basis |
|---|--:|--:|---|
| Llama-3.2-1B decode | ~90% | ~10% | hand-verified (`--add-edge 3:4 6:8`) |
| Whisper-base (enc-dec) | ~70% | ~30% | estimate (view-break dominated) |
| SDXL UNet (diffusion) | ~65% (floor 54%) | ~35% | estimate; floor rigorous w/ attention edges |
| Mamba-2.8b (SSM) | ~61% | ~39% | trustworthy (0 unresolved) |
| ResNet50 (CNN) | ~66% | ~34% | trustworthy (0 unresolved) |

Key lessons: (1) **all these architectures are majority-serial** (60–90% can't-hide) — earlier "wide open"
readings (UNet 74% hideable, Whisper 86%) were artifacts of dropped create-heads→SDPA and view edges;
(2) **never trust critical-path slack on an attention model without `--resolve-inplace`**; (3) **WAW is
negligible everywhere** (0 true — all allocator-reuse or in-place RMW), so a barrier relaxer only needs to
reason about RAW; (4) reorder payoff is real but the **minority** (10–40%). Residual uncertainty (view
false-roots) is bracketed by a program-order-reconnect estimate, calibrated on Llama (est 86% vs true 90%);
exact non-Llama numbers need the data-flow edges preserved at capture time (a ttnn-side change).

## Files

```
test_op_to_op_latency.cpp          host test binary
kernels/reader_interleaved.cpp     NCRISC reader (modes 0/1/2, cross-program offset, progress markers)
kernels/writer_interleaved.cpp     BRISC writer (flush + end-barrier mode 0/1/2/3-posted, unroll, markers)
kernels/compute_copy_with_nops.cpp TRISC copy + tunable NOPs
scripts/*.py                       captures, exporter, analyzers, chart + RAW-hazard generators (see tables)
scripts/run_*.sh                   sweep drivers (see table above)
```

## Kernel profiler notes

Do **not** use `DeviceZoneScopedMainN` in compute `kernel_main` (breaks `TRISC-KERNEL` pairing). Wrap
unpack/pack timestamps in `UNPACK(...)` / `PACK(...)` so each marker lands on the intended TRISC.
Kernels pick up `PROFILE_KERNEL` only when `TT_METAL_DEVICE_PROFILER=1` is set before the process
starts; the JIT cache dependency-checks profiler macros (no manual clear needed).
