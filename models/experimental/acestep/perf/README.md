# ACE-Step v1.5 — performance investigation & optimization design

This folder holds **performance measurement** for the ACE-Step TT pipeline (kept separate from
`tests/`, which is correctness/PCC only). It also records the design for how we will apply
optimizations **externally** — without hardcoding HW/arch-specific choices into the module files —
so the same modules run on Wormhole and Blackhole with different tuned configs.

> Status: **investigation** (no optimization applied yet). This documents how `tt_dit` solves the
> same three problems (perf measurement, trace capture, optimization config) and the design we will
> follow before re-activating autoresearch to actually optimize.

---

## 1. Measured sequence-length range (batch 1, p150)

From an empirical device sweep of a single DiT forward (see `bench_dit.py`):

| | latent T | T′ = T/2 (DiT seq) | duration | DiT step latency | regime |
|---|---|---|---|---|---|
| shortest | 2 | 1 | 0.08 s | ~40 ms | overhead-bound |
| **T′=128 target** | 256 | 128 | 10.24 s | ~42 ms | overhead-bound (edge) |
| — | 1024 | 512 | 40.96 s | ~118 ms | compute-bound |
| — | 4096 | 2048 | 163.8 s | ~468 ms | attention-bound |
| longest | 65536 | 32768 | ~44 min | ~30 s | RoPE ceiling |

**Hard ceiling:** T′ = 32768 = DiT `max_position_embeddings` (RoPE positions). No device-memory
wall below that on the p150; the limit is the position table, not L1/DRAM.

**Key finding for optimization:** latency is **flat (~40 ms) for T′ ≤ 128** — the device is
underfilled, so the step is dominated by **host dispatch / kernel-launch overhead**, not matmul
throughput. For the ≤10 s regime (T′ ≤ 128), the win is **trace capture** (remove the host path),
not compute tuning. Above T′ ≈ 512 it flips to compute/attention-bound, where sharding + matmul
program configs matter. The sliding window is 128, so **at exactly T′=128 sliding == full
attention** — optimizing only there hides the long-sequence sliding-window path.

### End-to-end stage breakdown (measured, `bench_pipeline.py`, 10.24 s / 30 steps)

| stage | mean | note |
|---|---|---|
| encode (prompt → context) | 0.03 s | negligible, runs once |
| **denoise (30 DiT steps)** | **1.29 s** | 43 ms/step, 23 steps/s — overhead-bound as predicted |
| **VAE decode** | **7.91 s** | **dominates — 6× the whole denoise loop** |
| total | 9.24 s | |

**Surprise finding that re-prioritizes optimization:** at the 10 s target the **VAE decode is the
bottleneck (86% of wall time), not the DiT**. The Oobleck VAE upsamples 256 latent frames → 491,520
stereo samples through 5 ConvTranspose1d stages (×1920 total), and it currently runs **fp32,
ROW_MAJOR, un-traced, and partly via `Conv1dViaConv3d`** — all expensive. So the first, highest-ROI
target is **the VAE**, not the denoise loop:
  1. VAE trace capture + on-device dtype (fp32→bf16 where PCC allows) — biggest single win.
  2. DiT denoise trace capture — removes the ~43 ms/step host path (secondary, ~1.3 s pool).
  3. Long-sequence (T′≥512) compute/attention tuning — only if long songs become a requirement.
The two `bench_*.py` scripts let autoresearch measure each independently. **Correctness gates
(PCC tests) still bound any dtype/precision change — e.g. VAE bf16 must hold its ≥0.97 gate.**

---

## 2. How `tt_dit` measures performance

Reference: `models/tt_dit/tests/models/sd35/test_performance_sd35.py` (+ flux1/wan/mochi/qwenimage).

- Reuses the shared harness `models/perf/benchmarking_utils.py`:
  - `BenchmarkProfiler()` — context-manager timer: `with profiler("denoising_step_3", iteration=i):`
    then `profiler.get_duration("denoising_step_3", i)`.
  - `BenchmarkData().save_partial_run_json(...)` — emits CI-consumable JSON.
- Pattern: **one warmup run** (captures the trace), then **N measured runs**; report mean/std/min/max
  per stage (encode / per-denoise-step / VAE / total) and derived throughput (steps/s).
- `device_params` fixture sets `trace_region_size` + `l1_small_size` (trace needs a DRAM trace region).
- Optional Tracy `Profiler` for op-level profiling when available.
- Expected metrics are **keyed by mesh shape** (`get_expected_metrics(mesh_device)` → per-shape dict)
  so the pass/fail threshold is HW-specific, not a single hardcoded number.

Our `bench_dit.py` / `bench_pipeline.py` follow this shape but single-device (p150, 1×1 mesh).

---

## 3. How `tt_dit` does trace capture

Reference: `models/tt_dit/utils/tracing.py` (`Tracer`, `traced_function`) + `pipelines/ltx`.

- **`Tracer(fn, device=, prep_run=, clone_prep_inputs=)`** wraps a fn whose inputs/outputs are all
  `ttnn.Tensor` | scalar | None (optionally nested). First call **captures**
  (`ttnn.begin_trace_capture` → run fn → `ttnn.end_trace_capture`), subsequent calls **replay**
  (`ttnn.execute_trace`) — replay copies changed inputs into the **same** device buffers in place and
  reuses the **same** output buffers. This is what removes the host path.
- Caveats baked into the abstraction: a trace **bakes absolute tensor addresses**, so every held
  input/const must be allocated **before** capture and never rebuilt; outputs are overwritten in place
  each replay.
- LTX pipeline pattern (`pipeline_ltx.py`): one `Tracer` per **fixed shape** (`trace_key` "s1"/"s2"),
  resident across `generate()`, freed by `release_traces()`. Per-trace **constants** (rope tables,
  masks, cross-attn PE) and **latent buffers** are pre-allocated once; the traced denoise step only
  updates `xt` + timestep. `capture_inputs` (full set, first call) vs `replay_inputs` (only what
  changes) — omitted kwargs reuse captured buffers.
- Ergonomic decorator `@traced_function(device=lambda self: self.mesh_device)` adds a `traced=True`
  call kwarg to any method; the SD35/LTX pipelines call `pipeline(..., traced=True)`.

**Implication for ACE-Step:** our `generate()` loop currently (a) rebuilds `FlowMatchStep`, rope
tables, sliding mask every call (fine — they're outside the loop) but (b) runs each DiT step through
the **host path** (torch timestep → new device tensors, `deallocate`/realloc of `xt`). To trace, we
must: pre-allocate rope/mask/const + `xt`/`context`/`enc` buffers once, make the per-step timestep a
device buffer we update in place, and wrap the single-step DiT forward in a `Tracer` keyed by T′.
Because we optimize for a **fixed T′=128**, a single resident trace is ideal.

---

## 4. How `tt_dit` handles optimization configuration (the non-hardcoding design)

`tt_dit` keeps optimization choices **out of the module forward code** in three layers:

### (a) Parallel/topology config — passed in, keyed by HW
- `models/tt_dit/parallel/config.py`: `DiTParallelConfig(cfg/sp/tp = ParallelFactor(factor, mesh_axis))`,
  `EncoderParallelConfig`, `VAEParallelConfig`, etc. — plain NamedTuples, `from_tuples(...)`.
- Modules **receive** a `parallel_config` argument; they never decide parallelism themselves.
- `pipeline...Config.default(mesh_shape=...)` looks up a **`_PRESETS` dict keyed by mesh shape**:
  ```python
  _PRESETS = {(2, 4): {"cfg": (2,1), "sp": (2,0), "tp": (2,1), "num_links": 1},
              (4, 8): {"cfg": (2,1), "sp": (4,0), "tp": (4,1), "num_links": 4}}
  ```
  → the tuned config for each device is data, selected at build time, not baked in the layer.

### (b) Matmul program configs — a tuning table keyed by (shape, core-grid)
- `models/tt_dit/utils/matmul.py`: `get_matmul_config(M, K, N, core_grid, default_block_size)` looks
  up `grid_88_configs[(M,K,N)]` etc. — **per-core-grid dicts of best blockings**, with a sane default
  + one-time warning on a miss. `get_matmul_core_grid(mesh_device)` clamps the grid per arch
  (e.g. Blackhole Galaxy power limit).
- So block sizes are **data keyed by (M,K,N, grid)**, tunable per HW without touching forward code.

### (c) Compute-kernel config — derived from `mesh_device.arch()`
- `models/tt_dit/layers/linear.py`: `ttnn.init_device_compute_kernel_config(mesh_device.arch(), ...)`
  — fp32-acc / packer-l1 / math-fidelity are derived from the **arch at build time**, not hardcoded
  per file; callers can still override per-call (`forward(..., compute_kernel_config=)`).

**Takeaway:** the pattern is **"config is data, keyed by (arch, mesh-shape, tensor-shape); modules
take it as an argument; a preset table selects it at build time."** Nothing HW-specific lives in a
module's forward.

---

## 5. Proposed ACE-Step design (to implement when autoresearch resumes)

Goal: apply optimizations **externally**, portable across Wormhole/Blackhole, no per-file hardcoding.

1. **`ExecConfig` (new, in `tt/exec_config.py`)** — a frozen dataclass carrying the *optimization*
   knobs (distinct from `AceStepModelConfig`, which is architecture dims):
   - `use_trace: bool`, `trace_seq_len: int | None` (fixed T′ to capture)
   - `compute_kernel_config` (or a factory taking `arch`), `weight_dtype`, `activation_dtype`
   - `matmul_program_configs` — optional shape→config table (reuse `tt_dit.utils.matmul`)
   - `parallel_config` — for multi-device later (p150 is 1×1 today; keep the seam)
   - `attn_impl` knobs (e.g. SDPA program config, sliding vs full at short T′)
2. **Preset table keyed by `(arch, mesh_shape)`** — `EXEC_PRESETS[(arch, shape)] → ExecConfig`, e.g.
   `("blackhole", (1,1))` and `("wormhole_b0", (1,1))` get different tuned configs. `ExecConfig.default(
   mesh_device)` resolves arch + shape → preset, exactly like `tt_dit`'s `_PRESETS`.
3. **Thread `exec_config` through builders** — `create_tt_pipeline(args, device, *, exec_config=None)`;
   `build_*` pass it into module configs. Modules read `exec_config` fields (default = today's
   behavior) instead of embedding constants. **No module hardcodes arch/shape choices.**
4. **Trace in `generate()`** — when `exec_config.use_trace`, pre-allocate consts + buffers once and
   wrap the single DiT step in a `Tracer` (reuse `tt_dit.utils.tracing.Tracer`) keyed by T′.
5. **Autoresearch optimizes the preset table, not the modules** — each experiment edits an
   `ExecConfig` preset (block sizes, dtypes, trace on/off) and measures `bench_pipeline.py`; the module
   code stays fixed. This is inherently non-overfitting: presets are per-HW data, and correctness PCC
   tests still gate every change.

### Why not hardcode
The model will ship **multiple optimization configs across HW (p150/other Blackhole) and arch
(Wormhole vs Blackhole)**. Baking block sizes / dtypes / trace flags into `attention.py` etc. would
force per-arch `if is_blackhole()` branches to multiply through every module. Keeping them in an
external `ExecConfig` preset table (data keyed by arch+shape+tensor-shape) means: one module
implementation, N tuned configs, selected at build time — the exact pattern `tt_dit` already uses.

---

## Scripts

- `bench_dit.py` — sweep a single DiT forward across T′; report latency + regime (the source of §1).
- `bench_pipeline.py` — end-to-end `generate_song` timing at a fixed length (encode / denoise-loop /
  VAE / total), reusing `models/perf/benchmarking_utils.BenchmarkProfiler`. Baseline before/after
  trace + config changes.

Run (needs the pipeline bundle + device):
```bash
export ACESTEP_PIPELINE_DIR=/path/to/acestep_pipeline
python models/experimental/acestep/perf/bench_dit.py
python models/experimental/acestep/perf/bench_pipeline.py
```
