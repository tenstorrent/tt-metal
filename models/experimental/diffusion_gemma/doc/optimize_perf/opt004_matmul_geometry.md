# OPT-004 — matmul-geometry tuning of the 5 sparse-MoE matmuls

**Scope:** the five `ttnn.matmul` calls in `tt/sparse_moe.py` that were never given a `program_config`
(the Lever-A GO/NO-GO prototype passed only `memory_config` + `compute_kernel_config`). This is
rank 2 of `path_to_100tps.md`'s build plan (lever 1): *"Tune the 5 batched matmuls (OPT-004) …
expected 1.8–2.5× on the MoE (10.5 → ~4–6 ms)."*

**This is a device-free deliverable.** The configs + rationale below are derived from the TTNN op
contract (read from source, cited) and the real per-device shapes. The **empirical** winner among the
candidate geometries is resolved by the device sweep in `bench_opt004_matmul_geometry.py` — *written,
not run here* (the QB2 box is owned by another agent). Nothing about the default path changes until a
device run confirms PCC parity and a latency win: the tuned configs are **opt-in via
`DG_SPARSE_MOE_TUNED=1`** and the flag-off path is byte-identical to the current prototype.

**No `models/demos/gemma4/` edits.** All geometry lives in `tt/sparse_moe.py` (DiffusionGemma-local),
composing over the untouched backbone. `git diff main -- models/demos/gemma4/` is unaffected by this
change.

---

## The workload (real QB2 shapes)

QB2 = P150×4, mesh `(1,4)`, TP=4. Per **chip** compute grid = **13×10 = 130 Tensix cores**
(`tests/ttnn/nightly/.../conv/test_conv2d.py:5253` confirms `(13,10)` for P150). Config values:

| symbol | meaning | value | tiles (÷32) |
|---|---|---|---|
| `E` | experts | 128 | — |
| `top_k` | active experts/token | 8 | — |
| `H` | hidden | 2816 | **88** |
| `I` | moe_intermediate **per device** (704 → 176 → tile-padded 192; 704 pads to 768 total / 4) | 192 | **6** |
| `S` | canvas | 256 | **8** |
| `C` | capacity (tokens/expert) | 32 | **1** |
| `EC` | `E·C` | 4096 | **128** |

> The `path_to_100tps.md` roofline text quotes `I=96 (3 tiles)` — that is the **TP=8 Galaxy** case
> (`704/8=88 → pad 96`). On the **TP=4 QB2** target the padding math in
> `models/demos/gemma4/tt/experts/weights.py:71-92` gives `704/4=176 → pad 192 = 6 tiles`. All numbers
> below use the TP=4 value.

Expert weights load as **`bfloat8_b`** by default (`moe.py:26`, `weights.py:38`), so a bfp8 weight tile
is ~1088 B vs a bf16 tile's 2048 B. Activations/intermediates/outputs are bf16. The compute-kernel
config is the module default — **HiFi2, `fp32_dest_acc_en=False`, `packer_l1_acc=True`** — which gives a
**dest-register budget of 8 half-tiles**, so every `out_subblock_h·out_subblock_w ≤ 8`.

---

## The five matmuls

| # | call site | op form | M×K×N (tiles) | config class |
|---|---|---|---|---|
| 1 | `_batched_experts` gate | `gathered[1,E,C,H] @ gate_proj[1,E,H,I]` | 1 × 88 × 6 (batch E=128) | `MatmulMultiCoreReuse` |
| 2 | `_batched_experts` up | `gathered[1,E,C,H] @ up_proj[1,E,H,I]` | 1 × 88 × 6 (batch E=128) | `MatmulMultiCoreReuse` |
| 3 | `_batched_experts` down | `down_input[1,E,C,I] @ down_proj[1,E,I,H]` | 1 × 6 × 88 (batch E=128) | `MatmulMultiCoreReuse` |
| 4 | `sparse_experts_forward` gather | `disp^T[1,1,EC,S] @ hidden[1,1,S,H]` | 128 × 8 × 88 | `MatmulMultiCoreReuseMultiCast` (2D) |
| 5 | `sparse_experts_forward` combine | `comb[1,1,S,EC] @ down_flat[1,1,EC,H]` | 8 × 128 × 88 | `MatmulMultiCoreReuseMultiCast` (2D) |

**Why these configs and not what's there now.** A plain `ttnn.matmul` on a DRAM-interleaved 4D batched
tensor with no `core_grid` never auto-selects the batched *reuse* factory — it falls through
`create_simple_matmul_program_config` to a multicast or the naive `MatmulMultiCore` factory
(`matmul_program_config.cpp`, `create_simple_matmul_program_config`). The naive path does not block the
K-reduction for weight reuse and does not deliberately pack the 128 experts across the grid, which is
consistent with the measured **~46 GB/s effective** (415 MB bf16 / 8.9 ms) — ~18 % of the @256 GB/s
roofline (`path_to_100tps.md` §b). OPT-004 replaces the auto-selection with an explicit, blocked,
full-grid config.

---

## Batched experts (gate / up / down) — `MatmulMultiCoreReuseProgramConfig`

### The op contract that fixes the geometry

Read from `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp` and
`.../factory/matmul_multicore_reuse_optimized_program_factory.cpp`:

1. **`per_core_N` MUST equal `Nt`** — `TT_FATAL(N == per_core_N, ...)` (device op ~L1651). **The N
   dimension is never split across cores** in the reuse factory.
2. **Cores distribute the batch×M blocks.** `num_output_blocks_total = (B·Mt / per_core_M) · (Nt /
   per_core_N)` (factory L174). With `per_core_N == Nt` the second factor is 1, so it reduces to
   `(B·Mt)/per_core_M`.
3. **`per_core_M` can pack experts.** `batch_scale_factor = per_core_M/Mt` when `per_core_M > Mt`
   (factory L110-111): a core then loops `batch_scale_factor` experts internally.
4. **`split_work_to_cores(grid, units)`** uses **exactly `units` cores when `units < grid_cores`**
   (`tt_metal/common/work_split.cpp:345-351`), else all grid cores.

**Design choice: `per_core_M = Mt`.** Then `num_output_blocks_total = B·Mt/Mt = E = 128` — one block per
expert. `split_work_to_cores` therefore uses **`min(128, grid_cores)` cores**: on BH's 130-core grid
that is **128 cores → exactly one expert per core, zero cross-expert serialization** (the exact concern
`path_to_100tps.md` §b.3 flagged as "unmeasured — if 128 M=1-tile matmuls fundamentally serialize,
in-repo tuning plateaus"). On a 64-core WH grid it is 64 cores × 2 experts each. `per_core_N = Nt` is
forced. That leaves **`in0_block_w` (the K-block) as the only real geometry knob**: larger = fewer
K-passes and bigger contiguous DRAM reads, bounded by L1.

`in0_block_w` is chosen as the largest divisor of `Kt` keeping the double-buffered in1 (weight) CB
within a tile budget (`per_core_N · in0_block_w ≤ 176`, `_pick_in0_block_w`). Subblocks are the largest
`(h,w)` with `h|per_core_M`, `w|per_core_N`, `h·w ≤ 8` (`_pick_out_subblock`).

### Chosen configs (BH 13×10 grid, C=32)

| role | per_core_M | per_core_N | in0_block_w | out_subblock (h,w) | cores used | K-passes |
|---|---|---|---|---|---|---|
| gate/up (Mt1,Kt88,Nt6) | 1 | 6 (=Nt) | **22** (88/22=4) | (1,6) | 128 | 4 |
| down (Mt1,Kt6,Nt88) | 1 | 88 (=Nt) | **2** (6/2=3) | (1,8) | 128 | 3 |

`in0_block_w`: gate/up cap `176/6 = 29 → 22` (largest divisor of 88); down cap `176/88 = 2 → 2`
(largest divisor of 6). Both are well above the `in0_block_w ≤ 2` floor OPT-004 warns against (gate/up),
and down is K-tiny (6 tiles) so `in0_block_w=2` is 3 passes — the weight bank is read once regardless.

### L1 budget (BH, batched CB model from the factory L145-164)

`in0_CB = per_core_M_per_batch·in0_block_w·2` (bf16 act); `in1_CB = per_core_N·in0_block_w·2` (weight);
`out_CB = interm0_CB = per_core_M·per_core_N` (bf16). Tile = 2048 B (bf16), 1088 B (bfp8 weight).

| role | in0_CB | in1_CB (bfp8 / bf16) | out+interm | **total (bfp8 / bf16)** |
|---|---|---|---|---|
| gate/up | 44 t = 88 KB | 264 t = 280 / 528 KB | 12 t = 24 KB | **≈ 392 / 640 KB** |
| down | 4 t = 8 KB | 352 t = 373 / 704 KB | 176 t = 352 KB | **≈ 733 / 1064 KB** |

Both fit BH's ~1.4 MB usable L1 with margin (down is the tighter one because `per_core_N=88` is forced —
its `out_CB` alone is 88 tiles). The default bfp8 weights leave ~2× headroom; the `_IN1_BLOCK_TILE_BUDGET`
= 176 floor is sized for the bf16 worst case so the same config is safe if a fidelity pass switches
experts to bf16.

> **Why no L1-sharded outputs?** OPT-004 also recommends L1-sharded outputs "where beneficial." Here the
> intermediates are enormous (`gathered`/`down` = `128·32·2816` = 11.5 M elems ≈ 23 MB each), far beyond
> L1, so they must stay DRAM-interleaved. For these matmuls the win is purely program-config geometry
> (core grid + `in0_block_w`), which is exactly OPT-004's "sweep geometry, not only dtype."

---

## Gather / combine — `MatmulMultiCoreReuseMultiCastProgramConfig` (2D)

2D systolic: **M parallelized over grid.y, N over grid.x** (`llms.md:1497`, factory
`matmul_multicore_reuse_mcast_2d_program_factory.cpp:188-198`). `per_core_M = ceil(Mt/gy)`,
`per_core_N = ceil(Nt/gx)`; `ceil` is legal (2D pads the last block). `num_blocks_y = ceil(Mt/per_core_M)
≤ gy`, `num_blocks_x = ceil(Nt/per_core_N) ≤ gx`. `in0_block_w | Kt`. Subblock `h|per_core_M`,
`w|per_core_N`, `h·w ≤ 8`.

### Chosen configs (BH 13×10)

| role | Mt×Kt×Nt | per_core_M | per_core_N | in0_block_w | subblock | cores (y×x) |
|---|---|---|---|---|---|---|
| gather | 128 × 8 × 88 | ceil(128/10)=13 | ceil(88/13)=7 | **8** (=Kt, 1 pass) | (1,7) | 10 × 13 = 130 |
| combine | 8 × 128 × 88 | ceil(8/10)=1 | ceil(88/13)=7 | **16** (128/16=8) | (1,7) | 8 × 13 = 104 |

`in0_block_w`: gather `176/7=25 →` largest divisor of Kt=8 is **8** (single K-pass — the whole gather is
one pass); combine `176/7=25 →` largest divisor of Kt=128 ≤25 is **16** (OPT-004 note: `16` has beaten
`8` on material MLP rows in prior experiments). `per_core_M=13`/`per_core_N=7` are prime-ish, so subblocks
land at `(1,7)` (product 7 ≤ 8) — acceptable; the sweep also tries grids that yield composite
`per_core_N` (e.g. `gx=11 → per_core_N=8 → out_subblock_w=8`).

### L1 budget (2D CB model)

`in0_CB = per_core_M·in0_block_w·2`; `in1_CB = per_core_N·in0_block_w·2`; `out+interm = 2·per_core_M·per_core_N`.
Both operands bf16 (disp/hidden, comb/down_flat), so no bfp8 headroom — these are the real numbers.

| role | in0_CB | in1_CB | out+interm | **total** |
|---|---|---|---|---|
| gather | 208 t = 426 KB | 112 t = 229 KB | 182 t = 373 KB | **≈ 1.03 MB** |
| combine | 32 t = 64 KB | 224 t = 448 KB | 14 t = 28 KB | **≈ 540 KB** |

Both fit. (`combine` is narrow — `Mt=8` → `is_narrow_shape` true, `W/H=11 > 8` — so the op would auto-pick
a 1D config; the 2D config here is legal and the sweep compares it against a 1D `mcast_in0` candidate.)

---

## Expected impact & reconciliation

Current MoE (`path_to_100tps.md` §b, `bench_ondevice_dispatch.py`): **10.54 ms/layer**, decomposed as
batched experts **~8.9 ms** (of which ~1.62 ms is the bf16 weight roofline @256 GB/s and ~7.3 ms is
geometry inefficiency), gather+combine+all-reduce **~1.7 ms**, on-device dispatch build (topk/scatter/
cumsum — **not a matmul, out of OPT-004 scope**) **~1.87 ms**.

OPT-004 attacks the ~7.3 ms batched inefficiency and the ~1.7 ms gather/combine:

- **Batched experts 8.9 → ~4–5 ms** if forcing 128 parallel cores + `in0_block_w`=22/2 lifts effective
  BW from ~46 GB/s toward ~90–110 GB/s (2–2.4×). If the auto-config path was serializing experts
  (worst case), the win is larger; if it was already parallel, smaller. **Unknown until measured.**
- **Gather/combine 1.7 → ~1 ms** from full-grid 2D configs.
- Dispatch build (1.87 ms) and all-reduce untouched.

**Projected: 10.54 → ~6–7 ms/layer** (conservative) toward the **~5–6 ms** rank-2 target if the batched
matmul reaches the optimistic end. This is the single load-bearing uncertainty in the `path_to_100tps`
plan ("per_layer to ~5–6 ms in-repo is *likely*"), and OPT-004 is the lever that tests it.

**Correctness:** these are pure geometry changes — same dtype (bf16 act / bfp8 weight), same fidelity
(HiFi2). PCC of the tuned MoE output vs the untuned prototype must be **≈ 1.0** (only tile-order rounding
differences, if any). The bench asserts `PCC ≥ 0.9997` (the untuned-vs-dense bar) and, more strictly,
tuned-vs-untuned `PCC ≥ 0.9999`.

---

## How to verify (device-free authored; run on QB2 when free)

`bench_opt004_matmul_geometry.py` (this directory) — **write-only; do NOT run here.** It:

1. Builds the real 26B MoE layer-0 (`build_tt_model_from_checkpoint_dir`, mesh `(1,4)`), real router
   routing on a real normed hidden.
2. Prints the device compute grid and the 5 chosen configs (so the geometry is visible in the log).
3. For each matmul, times the **untuned** (`program_config=None`) vs **tuned** variant on the real
   shapes, and checks tuned-vs-untuned PCC ≈ 1.0.
4. Sweeps a handful of candidate geometries per matmul (grid variants, `in0_block_w` values, and a 1D
   candidate for `combine`) and reports the fastest **legal** one per role.
5. Times the whole `sparse_experts_forward` with `DG_SPARSE_MOE_TUNED` off vs on, and MoE-output PCC vs
   the dense path, so the layer-level win and correctness are one number each.

Markers: `RESULT_GRID`, `RESULT_MATMUL role=… untuned_ms=… tuned_ms=… pcc=…`,
`RESULT_SWEEP role=… best=… ms=…`, `RESULT_FULL_MOE untuned_ms=… tuned_ms=… speedup=… pcc=…`.

**Signoff bar (per the `optimize` skill / OPT-004):** tuned MoE PCC ≥ 0.9997 vs dense; every dominant
matmul row in `tt-perf-report` shows `in0_block_w > 2` (gate/up, gather, combine) and the expected core
grid; the layer time drops; and the final default run reproduces the winning candidate. If a candidate
FATALs on legality, the sweep records the exact assert; if the batched matmul plateaus above ~5–6 ms,
that is the documented trigger for the upstream fused-kernel lever (rank 8).
