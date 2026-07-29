---
name: matmul-tuning
description: >-
  Hand-tune ttnn.matmul / ttnn.linear program_config + memory_config on Tenstorrent
  (Blackhole/Wormhole) to maximize per-op throughput. Use when optimizing a matmul or
  linear op, deciding whether an op is data-movement-bound vs compute-bound, choosing
  per_core_M/N / in0_block_w / subblocks, picking L1 vs DRAM for act/out, or reading
  DEVICE KERNEL DURATION from a tracy ops_perf_results CSV to compute utilization%.
  Follows Iva's process: roofline first, maximize cores, then tune blocking, then move
  data to L1 if DM-bound. Report measured numbers only — no perf theories.
---

# Matmul / linear tuning (ttnn, Blackhole & Wormhole)

Hand-tune the `program_config` + `memory_config` of a single `ttnn.matmul` / `ttnn.linear`
so it runs as close to its roofline as possible. The core question this skill answers is:
**is this op compute-bound or data-movement (DM) bound?** — because that decides which knob
moves it. Everything is grounded in measurement; do not report perf theories (see Reporting).

Template test (copy it, don't reinvent): `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_mla_matmuls_glm_chunked.py`
Roofline/parse helper: `models/demos/deepseek_v3_d_p/utils/parse_ring_joint_perf.py`

---

## The workflow (do these in order)

1. **Nail the shapes.** Get M, K, N (and batch Z if the matmul is batched) *after* SP/TP
   sharding — i.e. the per-chip shape the op actually runs. Fix the input dtypes to what the
   real model uses (typically in0=BF16 activation, in1=BF8 weight; out per case). Tile counts:
   `M_t=M/32`, `K_t=K/32`, `N_t=N/32`.

2. **Roofline both resources and decide the bound** (see next section). This tells you whether
   program_config tuning can even help, and whether to reach for L1.

3. **Pick `per_core_M` / `per_core_N` FIRST — to maximize active cores.**
   `active_cores = ceil(M_t/per_core_M) × ceil(N_t/per_core_N)`, capped by the grid.
   Target **110** cores (11×10 grid); **≥90 fair, ≥80 okayish**. Cores are the biggest lever on
   a compute-bound op; a config that only lights up 40 cores leaves the machine idle.

4. **Then tune the blocking**: `in0_block_w`, `out_subblock_h`, `out_subblock_w`. Constraints:
   - `out_subblock_h` divides `per_core_M`
   - `out_subblock_w` divides `per_core_N`
   - `in0_block_w` divides `K_t`
   - `out_subblock_h × out_subblock_w ≤ 8` (dst register tile budget)
   Bigger `in0_block_w` = fewer K passes / better reuse; bigger subblocks = fewer dst reloads.

5. **Measure** (tracy) and compute utilization% against the compute roofline.

6. **If DM-bound, move data to L1.** Try `act` and/or `out` in `L1_MEMORY_CONFIG` instead of
   `DRAM_MEMORY_CONFIG`. This is the main win for DM-bound ops; program_config alone won't help
   them. Re-measure.

7. Iterate 3–6, keep the best config, record cores / us / util%.

---

## Determining the bound (the heart of it)

Compute two roofline times from **measured shapes only**, then compare to each other and to the
measured kernel time. All arithmetic — no fitting, no hypothesizing.

### A. Compute roofline
```
tiles         = M_t · K_t · N_t         (× Z if batched)
cycles        = tiles · FIDELITY_CYCLES
compute_ns    = cycles / active_cores / clock_ghz
```
- `FIDELITY_CYCLES = {HiFi4: 64, HiFi3: 48, HiFi2: 32, LoFi: 16}` (cycles per output tile per pass)
- `clock_ghz`: **Blackhole 1.35**, Wormhole_b0 1.0
- **No causal /2 for a plain matmul.** (The `/2` in `parse_ring_joint_perf.py` is SDPA-causal-specific.)

### B. Data-movement roofline
```
bytes_moved   = (in0_bytes + in1_bytes + out_bytes)     per dtype: BF16=2 B/elem, BF8_b≈1 B/elem
dm_ns         = bytes_moved / DRAM_BW
```
Use the elementwise bytes actually read/written on-chip. `in1` (weights) is often the big term
for skinny-M matmuls; the output is the big term when M and N are both large or the out dtype is
BF16. Compare the terms — the largest single mover is usually the story.

### C. The decision
- **`dm_ns > compute_ns` ⇒ DM-bound.** program_config won't move it. Fix = put act/out in **L1**,
  or drop the output dtype (BF16→BF8), or an op-level change (less bytes). Confirm by sweeping a
  few program_configs and observing the time is **config-invariant** (the tell-tale signature —
  see below).
- **`dm_ns < compute_ns` ⇒ compute-bound.** program_config tuning helps. Maximize cores, then
  blocking; util% climbs toward 100.
- **Utilization%** `= compute_ns / measured_kernel_ns × 100`. Low util + config-invariant time =
  DM- or core-count-bound, not compute-bound.

### Empirical tells (confirm the roofline with the sweep, don't just trust the math)
- **Config-invariant time** across several program_configs (different `in0_block_w`, subblocks,
  even 1D-vs-2D) ⇒ **DM-bound or core-count-floored**, not compute-bound. This is the single most
  reliable signal. Always sweep 3–6 variants before declaring an op "tuned".
- **Time drops when you add cores** (via smaller `per_core_M/N`) ⇒ was **core-count-bound**.
- **Time drops when you move act/out to L1** ⇒ was **DM-bound** on that tensor.
- **Tiny N (small N_t)** floors the core count (`ceil(N_t/per_core_N)` cols) ⇒ core-count-limited
  by geometry; nothing to tune, it's at the floor. e.g. N_t=4 → 40 cores max, N_t=1 → 10 cores.

---

## Program config selection

Three config types (constructor signatures are in the template test's `_mc2d` / `_reuse` / `_mc1d`):

| Type | ttnn class | When |
|------|-----------|------|
| **2D multicast** | `MatmulMultiCoreReuseMultiCastProgramConfig` | Default for non-batched (Z=1) matmuls. `transpose_mcast=False, fuse_batch=False`. |
| **Reuse (non-mcast)** | `MatmulMultiCoreReuseProgramConfig` | **Batched matmuls** (Z>1, both operands batched). Folds `Z·M_t` onto the grid. |
| **1D multicast** | `MatmulMultiCoreReuseMultiCast1DProgramConfig` | Narrow cases; `mcast_in0` picks the broadcast operand. |

**Batched-matmul gotcha (important):** for a true batched matmul (both operands carry the batch,
e.g. per-head `[1,H,M,K]×[1,H,K,N]`), the **1D multicast path serializes the batch onto ~5–20
cores** and is catastrophically slow. Use `MatmulMultiCoreReuseProgramConfig` (non-mcast), which
folds `Z·M_t` across the grid — e.g. `per_core_M=4` on `Z·M_t=320` → 80 cores.

Grid: use **11×10 = 110 cores**, not the full 12×10 — deliberately backed off for di/dt and
throttling headroom on Blackhole.

Compute kernel config used for tuning (matches the template):
`math_fidelity=HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True`.

---

## Run + profile

```bash
source python_env/bin/activate
export TT_METAL_CACHE=/localdev/ipotkonjak/tt-metal-cache   # REDIRECT off /home if that mount is full

# PCC correctness only:
python3 -m pytest <test_file>::test_glm_mla_mm -v

# tracy device perf -> ops_perf_results CSV:
python3 -m tracy -r -p -m "pytest <test_file>::test_glm_mla_mm"
```

**tracy invocation rules (each one has burned a full debug loop before):**
- The pytest command MUST be a **single quoted string** to `-m`. Bare args after `-m pytest` are
  silently dropped → "No device logs found".
- The test must call `ttnn.ReadDeviceProfiler(mesh_device)` (or `synchronize_device`) after the
  profiled region, or device logs never flush.
- **Do not** pass `-o <dir>`. Default `generated/profiler/` is where logs land and get read from.
  CSV → `generated/profiler/reports/<TS>/ops_perf_results_<TS>.csv`.
- Profiler needs the **correct firmware**. Wrong FW ⇒ tracy emits raw device timestamps (~4.6e12)
  instead of durations in `DEVICE KERNEL DURATION [ns]`.

**Environment gotchas:**
- `TT_METAL_CACHE` MUST point off any full mount, or the instrumented-kernel build fills the disk
  mid-run ("No space left on device").
- On the 32-chip BH Galaxy, `ulimit -u` defaults to **512** and is the root cause of both EAGAIN
  teardown crashes and ring-fabric LTO build failures (`posix_spawn: Operation not permitted`).
  Fix before any run: `ulimit -u 1048576` (no admin needed).
- Never run overlapping/parallel device tests — one at a time, foreground. Watch early output for a
  `CHIP_IN_USE` lock-wait warning (an orphaned process still holds the chip).

---

## Reading the CSV

- One row **per op per device** — typically **8 rows per op** on an 8-chip mesh. **Average**
  `DEVICE KERNEL DURATION [ns]` over the devices.
- **Rows are scrambled and same-shape variants collide.** Map each row to its matmul via the
  **`ATTRIBUTES`** column (contains the full `program_config`: `in0_block_w`, `per_core_M/N`,
  subblocks, config type) plus `INPUT_0_MEMORY` / `INPUT_0_DATATYPE` / `OUTPUT_0_MEMORY`. **Never**
  map by row order.
- Utilization%, computed from the compute roofline above:
  ```
  util% = tiles · FIDELITY_CYCLES / active_cores / clock_ghz / measured_ns · 100
  # e.g. HiFi2 on Blackhole: util% = tiles·32 / cores / 1.35 / measured_ns · 100
  ```
- To distinguish core-bound from DM-bound after the fact, pull `DEVICE KERNEL DURATION DM START` /
  `PER CORE MIN/MAX` columns, or just run another config and see if the time moves.
- Use a **trace / `trace_bar`** config (not launch-then-run) for the cleanest util numbers — it
  removes per-op launch skew that otherwise inflates the measured time.

---

## Reporting conventions

- **Measured numbers only. No theories.** State what the CSV/run shows and stop. If a cause is
  unknown, name the measurement that would settle it (another config, a specific column) rather
  than asserting a mechanism. Decompositions are fine *only* if they're arithmetic on measured
  values; flag anything extrapolated.
- Microseconds are **`us`**, milliseconds are **`ms`** (keep them distinct). Keep raw
  profiler columns in **`ns`**; convert to `us`/`ms` only in summaries.
- Per matmul, record: **program_config, act mem, out mem, out dtype, active cores, us, util%**, and
  the **bound** (compute / DM / core-count).

---

## Worked reference — GLM-5.2 MLA, 9 matmuls, chunked prefill (seq_local=640, BH loudbox)

Real results, as a sanity anchor for what "tuned" looks like (grid 11×10, HiFi2):

| matmul | shape (M×K×N, Z) | program_config | act→out | cores | us | util% | bound |
|--------|------------------|----------------|---------|-------|-----|-------|-------|
| o_proj | 640×4096×6144 | MC2D ib16 sub1×6 pc2×18 | DRAM→L1 | 110 | 135.4 | 78 | compute (won via ib8→16 + out-L1) |
| q_b_proj | 640×2048×4096 | MC2D ib8 sub1×6 pc2×12 | L1→L1 | 110 | 49.1 | 72 | compute (won via act/out-L1) |
| indexer.wq_b | == q_b_proj | (identical config) | L1→L1 | 110 | 49.1 | 72 | compute |
| wkv_b2 | 640×512×256, Z=16 | **Reuse** ib2 sub4×2 pc4×8 | L1→L1 | 80 | 51.7 | 23 | **DM-bound**, config-invariant |
| wkv_b1 | 640×192×512, Z=16 | **Reuse** ib6 sub2×4 pc4×16 | L1→L1 | 80 | 49.8 | 18 | **DM-bound**, config-invariant |
| q_a_proj | 640×1536×2048 | MC2D ib8 sub1×6 pc2×6 | DRAM→L1 | 110 | 22.7 | 58 | compute |
| kv_a_proj_with_mqa | 640×1536×576 | MC2D ib8 sub1×2 pc2×2 | L1→L1 | 90 | 11.2 | 41 | compute |
| indexer.wk | 640×1536×128 | MC2D ib8 sub1×1 pc2×1 | DRAM→DRAM | 40 | 13.9 | 16 | **core-floored** (N_t=4) |
| indexer.weights_proj | 640×1536×32 | MC2D ib8 sub1×1 pc2×1 | DRAM→DRAM | 10 | 10.2 | 22 | **core-floored** (N_t=1) |

Takeaways this run proved:
- **o_proj** and **q_b_proj** were the real compute wins (cores maxed, then L1 + bigger `in0_block_w`).
- **wkv_b1 / wkv_b2** are batched and **DM-bound**: config-invariant at ~50 us across 6 variants;
  core count floored at 80 (`Z·M_t=320`, `per_core_M=4`→80; smaller `per_core_M` didn't divide 320
  or overflowed the grid). The BF16 output bytes are the bottleneck — an op-level fix, not a config.
- **indexer.wk / weights_proj** are floored by tiny N — nothing to tune, they're at the geometry
  floor.

---

## Checklist for tuning one matmul

- [ ] Per-chip M, K, N, Z after sharding; tile counts; fixed input/output dtypes
- [ ] compute_ns and dm_ns computed from shapes → predicted bound
- [ ] per_core_M/N chosen to maximize cores (target 110)
- [ ] in0_block_w / subblocks obey the divisibility + `≤8 dst` rules
- [ ] batched? → `MatmulMultiCoreReuseProgramConfig`, NOT 1D multicast
- [ ] PCC ≥ 0.99 vs torch
- [ ] tracy CSV mapped via ATTRIBUTES, duration averaged over devices, util% computed
- [ ] swept 3–6 variants → confirmed compute-bound (time moves) or DM/core-bound (invariant)
- [ ] if DM-bound: tried act/out → L1, tried out dtype BF16→BF8
- [ ] recorded config / cores / us / util% / bound
