# tt-blaze evaluation for GLM-4.7-Flash on Blackhole Galaxy

**Status:** one op measured and adoption-ready (`dram_streaming_matmul`, 2.45x on o_proj).
Everything else is blocked on layout assumptions in blaze, documented below with evidence.

**Hardware:** 32-chip Blackhole Galaxy, every chip 1x-harvested → **12x10 = 120 cores** per
device, 8 DRAM banks. This matters throughout: blaze's model layouts are written for an
**unharvested 13x10 = 130-core** Blackhole.

**Baseline this is measured against:** 33.2 ms/token, bs=1, ISL=128, traced sampling — see
the Blackhole Galaxy section of [README.md](README.md).

---

## 1. Why blaze at all, and how much is on the table

`blaze-pick-targets` ranks blaze's mechanisms, and #1 is *"overlap weight reads with compute
— DM1 streams from DRAM while TRISC computes"*. Its stated test is to double the weight
dtype width and see whether the step moves proportionally.

Run on **BH** (the WH number in that skill is not transferable):

| | ms/token | Δ |
|---|---:|---:|
| `DENSE_TT_DTYPE=bf8` (default) | 33.2 | — |
| `DENSE_TT_DTYPE=bf16` | 37.1 | **+3.9 ms (+11.7%)** |
| *WH reference for the same A/B* | *(50.1 base)* | *+9.8 ms (+19.6%)* |

**Exposed weight-read time is only ~3.9 ms of a 33.2 ms step**, so mechanism #1 is capped
near **12%** here. BH's 2x DRAM bandwidth (512 vs 258 GB/s) already collected most of what
WH left on the table.

The consequence for target selection: blaze's remaining leverage on BH is mostly *not*
weight streaming. It is keeping intermediates in L1 instead of round-tripping to DRAM,
overlapping phases across 120 cores, and the 1x32-tile advantage at bs=1 (below).

## 2. Measured results

Method per `blaze-vs-ttnn-bench`: same device, one golden, median of 5 iterations after 2
warmups, reading `DEVICE KERNEL DURATION`. Correctness gates the verdict before latency.

### o_proj — ADOPT (2.45x)

| impl | cores | rows | µs | PCC |
|---|---:|---:|---:|---:|
| ttnn DRAM-sharded matmul | 80 act / 64 out | 32 | 58.4 | 0.9999 |
| blaze `DRAMStreamingMatmul` | 8 | 1 | **23.8** | 0.9999 |

K=5120, N=2048, bf8 — the largest single weight in the decode step (11.1 MB/layer) and the
one the GlobalCB prefetcher targeted before the L1 budget killed it.

Read it with two asymmetries in mind, pulling opposite ways:

- **For blaze, legitimately.** ttnn computes 32 rows when only row 0 is real at bs=1, and
  cannot express otherwise here. blaze's 1x32 decode tile can. It also uses 8 cores against
  ttnn's 80, freeing ~72 cores — which is mechanism #3 (overlap phases on disjoint grids).
- **Against extrapolating.** At real batch, ttnn's 32 rows are all useful and blaze would
  need m=32, which is **numerically broken** (finding 1). This is a bs=1 decode win and does
  not currently extend to the batch / aggregate-TPS story.

**Cluster vs step, kept separate.** 58.4 → 23.8 µs saves 34.6 µs/call; o_proj runs once per
layer x 47 layers ≈ **1.6 ms/token, ~5% of the step** — an upper bound assuming it is all on
the critical path and translates perfectly. Consistent with the ~12% ceiling above. A
cluster win is not a step win: this model previously measured removing 23% of its ops for
**0.0 ms** under trace.

### rmsnorm — REGRESSION, do not adopt

| config | ttnn µs | blaze µs | speedup | ttnn PCC | blaze PCC | verdict |
|---|---:|---:|---:|---:|---:|---|
| LoFi / bf16 accumulate | 23.2 | 22.4 | 1.04x | 1.0000 | 0.9865 | REJECT (< 0.99 floor) |
| `fp32_dest_acc_en=True` | 23.3 | 25.2 | 0.92x | 1.0000 | 0.9999 | REGRESSION |

Blaze only wins while accumulating in bf16, which fails PCC; correct, it is 8% slower. This
is the expected result for a **MicroOp** measured alone — no fusion, no avoided DRAM
round-trip, still paying dispatch. Do not generalise it to FusedOps.

Note `fp32_dest_acc_en` must be set on the **program**, not `RMSNorm.emit` — emit deletes
its own copy (`blaze/ops/rmsnorm/op.py:143`, "the program's ComputeConfigDescriptor is
authoritative"). Easy silent precision loss.

## 3. What runs on a 12x10 (harvested) Galaxy today

| op | kind | status |
|---|---|---|
| `dram_streaming_matmul` | MicroOp | 89/89 tests pass, no grid guard |
| `rmsnorm`, `broadcast_rmsnorm` | MicroOp | pass |
| `qa_projection`, `q_branch`, `kv_branch` | Fused/Micro | pass with the grid guard lifted |
| `glm_moe_router` | FusedOp | **passes** with finding 5's remaps |
| `glm_routed_expert` | FusedOp | **passes** with finding 5's remaps |
| `pre_sdpa` / `post_sdpa` | FusedOp | pass for **kimi_k2**, fail for deepseek_v3 |
| `shared_expert`, `glm_moe` | FusedOp | blocked, finding 4 |

## 4. Findings

### F1 — `DRAMStreamingMatmul` is numerically wrong at m=32

At GLM's o_proj shape (5120x2048): **PCC 0.0074** at m=32; m=1 and m=8 pass. Reproduced with
blaze's own `_run_and_compare`, not a custom harness. The shipped tests only cover
m ∈ {1,4,8}, so m=32 is untested territory. **Worth filing upstream.**

### F2 — `ab_harness` cannot import its PCC helper

`blaze-vs-ttnn-bench/scripts/ab_harness.py:132` does
`from tests.blaze.utils.torch_golden import comp_pcc`. That can never resolve: `tt-blaze/tests`
has no `__init__.py` (namespace portion only), while `tt-blaze/tt-metal/tests` **does** and is
also on `PYTHONPATH` — a regular package terminates the namespace search and shadows it.
The harness is unusable out of the box. Fix upstream is adding `tests/__init__.py` and
`tests/blaze/__init__.py`; we worked around it by registering the module by path.

### F3 — MLA layout cannot express GLM-4.7-Flash's head count

`tests/blaze/backed/layout_plan.py:186` requires `n_heads_per_device % 8 == 0`
(`_QB_GRID_ROWS = 8`). GLM has **20 heads**, and no TP divisor works:

| tp | heads/device | %8 | %(8x3) |
|---|---|---|---|
| 1 | 20 | 4 ✗ | 20 ✗ |
| 2 | 10 | 2 ✗ | 10 ✗ |
| 4 | 5 | 5 ✗ | 5 ✗ |
| 5 | 4 | 4 ✗ | 4 ✗ |

Worse, GLM's nope:rope ratio is **192:64 = 3:1** (vs 2:1 for DSv3/Kimi), so
`qrope_grid_cols = heads/(8*3)` needs a multiple of **24**. DSv3's 64 and Kimi's 32 satisfy
both; 20 satisfies neither. Unblocking needs `_QB_GRID_ROWS` made config-driven **and**
`qb_per_core` decoupled from `qk_nope_head_dim` — both feed kernel compile-time args, and
the 8 is a deliberate invariant (`n_sdpa_cores = heads//8`, *"derive such that
heads_per_receiver = 8 (see PR #1063)"*).

### F4 — shared-expert gate/up split only balances at 130 cores

`blaze/weights/moe_grid_layout.py` hardcodes `NUM_SHARED_GATE_UP_MM_CORES = 64`,
`NUM_SHARED_DOWN_MM_CORES = 112`, and a fixed column pattern, with
`assert len(gate_coords) == len(up_coords) == 64`:

| grid | gate | up | 64/64? |
|---|---:|---:|---|
| 13x10 = 130 (unharvested) | 64 | 64 | OK |
| **12x10 = 120 (ours)** | 64 | **54** | fails |

The op requires matching `(k_parallel, n_parallel)`; observed error
`gate=(8,8) up=(8,7)`. 64/64 needs 128 cores against **118 usable** (120 − sender − phantom).
A balanced **56/56 = (8,7) is 112 cores and would fit**, so this is fixable upstream, not
impossible — but the pattern must become grid-aware *and* `preprocess_gate_up`'s placement
spec regenerated to match, since a mismatched layout "would feed gate-laid-out tiles to 'up'
cores, producing garbage matmul output."

`glm_moe` is blocked **solely** by this; everything else in that composition works.

### F5 — column-12 assumptions, and they are remappable

GLM-5's `BlazeConfig` pins `moe_router_gate_mm_cores` to x=12 and `sender_core` to (12,9) —
13-column assumptions. On 12x10 both fail with
`Circular buffer core range [12-N] ... exceeds device compute grid (12x10)`.

Remapping both to column 11 makes `glm_moe_router` (3 passed) and `glm_routed_expert`
(1 passed) work. Column 11 rows 0–7 are free; `GridConfig` puts sender at (11,9) and the
idle phantom at (11,8) — and would compute (11,9) itself on a 12-wide grid.

### F6 — a gate-core count mismatch HANGS the device, silently

Driving `glm_routed_expert` at GLM-4.7-Flash dims (64 experts) while leaving GLM-5's **8**
gate MM cores wired in hangs: 20+ minutes, zero output, against ~190 s for the GLM-5 shape.
`num_gate_mm_cores` must equal `num_experts // 32` (**2** for GLM). No error is raised — the
handshake is sized for one count while a different set of cores participates, and everyone
waits. Same class as blaze's own warning that a non-dividing ring "deadlocks rather than
raising."

Killed cleanly; **the Galaxy did not need a reset.** Use short timeouts when driving an op at
dims its config does not match.

`BlazeConfig` correctly refuses the half-fix: `sanity_check_model_config` asserts
`num_gate_mm_cores >= num_experts/32` (`blaze/models/config/blaze_config.py:378`), and
`__replace__` raises `_REPLACE_FORBIDDEN_MSG` — configs are meant to be **authored, not
patched**. Our `dataclasses.replace` only slipped through because Python 3.10 does not
dispatch to `__replace__` (3.13+ does).

### F7 — `requires_grid_size((13,10))` is conservative

The marker is a blanket precondition; the real constraints are per-op. With it stubbed,
`qa_projection`, `q_branch`, `kv_branch` and `pre_sdpa`/`post_sdpa` (kimi) all pass on 12x10.
DSv3 fails for a specific reason worth knowing: *"Expected number of shards 128 to be less
than or equal to total number of L1 banks 120"* — 128 heads vs 120 banks.

### F8 — transient TLB exhaustion

`glm5_moe_router` failed 3/3 with `tt_tlb_alloc failed ... error code -12` (ENOMEM), then
passed 3/3 on retry with the device idle. Not a real defect — retry before investigating.

### F9 — blaze requires its own tt-metal

Not C++20 as the README suggests: blaze's tt-metal also compiles kernels with `-std=c++17`,
but adds SFPI flags ours lacks — `-ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit`.
So benchmarks run in the tt-blaze tree and the model runs in ours; the two are not
interchangeable. Our tt-metal already has the named CT-arg infrastructure
(`tt_metal/hw/inc/api/compile_time_args.h:81`).

## 5. Reproducing

Environment (built at `tt-metal v0.75.0-dev20260715-120-gd1a81ac358a`, Release):

```bash
cd /home/ttuser/sdawle/tt-blaze && source env.sh && unset TT_MESH_GRAPH_DESC_PATH
export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
```

Bench and diagnostic scripts (untracked, in the tt-blaze checkout):

| file | what |
|---|---|
| `ab_oproj_v2_bench.py` | the 2.45x o_proj A/B (both sides PCC-verified) |
| `ab_rmsnorm_bench.py` | the rmsnorm A/B |
| `ab_m32_check.py` | reproduces F1 at m ∈ {1,8,32} |
| `no_grid_guard_plugin.py` | stubs `requires_grid_size` (F7) |
| `remap_col12_plugin.py` | column-12 → 11 remap (F5) + gate-core truncation (F6) |
| `blaze/models/glm4_flash/glm4_flash.model_config.json` | GLM-4.7-Flash in blaze's schema |

A copy of the model config is kept at [`blaze_eval/glm4_flash.model_config.json`](blaze_eval/glm4_flash.model_config.json)
so it survives a re-clone of tt-blaze.

## 6. Next steps

1. **Adopt `dram_streaming_matmul` for o_proj** — the only measured, correctness-gated win.
   Gate on F1 (keep m=1 for decode) and re-measure *step* time, not just cluster time.
2. **Author `GLM4_FLASH_BLAZE_CONFIG`** to finish the routed-expert A/B at our dims. The MoE
   core block is small: 2 gate cores at (11,0)–(11,1), sender (11,9), 8 DRAM banks unchanged,
   red2one (0,2), tp=1. Open question: whether `sanity_check_model_config` also demands
   non-empty shared-expert coords (F4 says those cannot balance on 12x10; `dflash/config.py`
   sets them empty, which suggests it tolerates that).
3. **File F1 and F2 upstream** — both are small, self-contained, and affect any blaze user.
4. **Raise F3 and F4 with the blaze team** — GLM-4.7-Flash is unlikely to be the only model
   with a non-8-divisible head count or a 3:1 nope:rope ratio, and F4 blocks every harvested
   Galaxy, not just this one.
