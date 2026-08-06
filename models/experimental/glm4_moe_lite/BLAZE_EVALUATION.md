# tt-blaze evaluation for GLM-4.7-Flash on Blackhole Galaxy

**Status:** The routed MoE path is now **numerically correct at GLM-4.7-Flash's dims**. F11's hang
is fixed (17.8 s against >700 s), and the PCC 0.0235 that replaced it is fixed too — traced to
uninitialised L1 being ranked as an expert (F14) and then to a golden that asserted an ordering
the kernel does not promise (F15). F3 is **resolved, not worked around**: GLM's 20 heads do fit,
and the Q-branch grid is now solved rather than assumed.

Against that, one new blocker and one hard number. **`DRAMStreamingMatmul` stalls under
multi-device mesh execution** (F17) — measured on the *stock, unmodified* op, so every fused op
built on it is blocked at the 32-device model boundary, and that is now the critical path rather
than any layout question. And at the real integration boundary — the model's native TILE DRAM
tensors in and out — the fused QKV-A cluster measures **0.53x, nearly 2x slower than ttnn**, against
4.76x on blaze-native tensors. Three clusters are correctness-gated in isolation and there is still
**zero measured end-to-end gain**. Read section 7 before planning adoption.

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

### Every decode matmul — all six ADOPT on kernel time

`DRAMStreamingMatmul` is numerically correct at **every** GLM-4.7-Flash decode matmul shape
(`glm47_all_shapes_check.py`, 6/6 pass at m=1 bf8), so the adoption surface is the whole dense
matmul set, not just o_proj. Full A/B (`ab_all_shapes_bench.py`), ttnn side mirroring
`dram_sharded_linear` exactly:

| shape | ttnn cores in/out | ttnn µs | blaze µs | speedup | calls/layer | x47 layers |
|---|---|---:|---:|---:|---:|---:|
| q_a_proj 2048x768 | 64 / 24 | 44.7 | **4.9** | **9.17x** | 1 | 1.87 ms |
| kv_a_proj 2048x576 | 64 / 24 | 44.7 | **5.0** | **9.03x** | 1 | 1.87 ms |
| q_b_proj 768x5120 | 24 / 80 | 20.3 | 11.6 | 1.75x | 1 | 0.41 ms |
| o_proj 5120x2048 | 80 / 64 | 58.4 | 23.8 | 2.45x | 1 | 1.62 ms |
| mlp_gate_up 2048x1536 | 64 / 48 | 45.0 | **8.2** | **5.50x** | 2 | 3.46 ms |
| mlp_down 1536x2048 | 48 / 64 | 34.4 | 8.1 | 4.25x | 1 | 1.24 ms |

All twelve PCCs are 0.9999. The largest wins are the *small-N* projections (q_a, kv_a), where
ttnn spreads a 768- or 576-wide output over 64 input cores and is overhead-dominated, while
blaze uses 8 and computes 1 row instead of 32.

**THE SUM IS NOT BELIEVABLE AS A STEP SAVING, AND THAT IS THE POINT.** Naively these total
**10.47 ms of a 33.2 ms step (31.5%)**. But doubling every dense weight's bytes costs only
**3.9 ms (11.7%)** — measured, section 1. A 10.5 ms saving therefore exceeds the model's entire
weight-bandwidth budget, which is arithmetically impossible if the win were weight-read time.

So most of this per-call delta is **not on the step's critical path**. Under trace replay the
command stream is pre-recorded and per-op cost pipelines — the same reason this model measured
removing 23% of its ops as **0.0 ms**. Read the table as a *ranking* of where blaze's kernels
are more efficient, not as a step-time forecast.

The implication for adoption order: q_a/kv_a/mlp_gate_up look most attractive per call, but the
only way to know what lands is a **step-level** measurement of the traced model, which needs
GLM running under blaze's tree (F9). That is the next real milestone, and no further cluster
A/B changes it.

### The step-level answer so far: NO measurable improvement

Two experiments, both in the shipping traced regime on the real model.

**1. ttnn with blaze's core layout — no gain.** blaze's matmul wins partly by pinning one worker
per DRAM bank (8) where ttnn spreads over up to 80. That choice is expressible in ttnn today, so
`GLM4_MOE_LITE_DS_CORE_CAP` caps the activation-sharding count and the model was measured both
ways:

| `DS_CORE_CAP` | ms/token |
|---|---:|
| 0 (ttnn default, up to 80 cores) | **33.2** |
| 8 (blaze's bank-matched layout) | **33.4** |

No improvement — very slightly worse. So the core-count component of blaze's 1.75x–9.17x
per-op advantage does **not** reach the step. This is the most direct evidence in this document
that those per-op deltas are off the critical path, and it agrees with the arithmetic: they sum
to 31.5% of a step whose entire weight-bandwidth budget is 11.7%.

**2. Traced decode now runs in blaze's tree, and the trees are close.** The `ttnn.copy`
divergence is fixed (see F13), so the model can be measured traced in the same process as blaze
ops. Reaching a running decode there costs three default-on optimizations, so the honest
comparison holds the config fixed across both trees:

| config | our tt-metal | blaze's tt-metal (v0.75.0-dev) |
|---|---:|---:|
| full default flags | **33.2** | cannot run — needs our ttnn extensions |
| `FUSE_DOWN_ROUTING_SCALE=0`, `FUSED_COLLECTIVE_EPILOGUE=0`, `FUSED_ROUTER=0` | **34.1** | **34.8** |

Two readings, both useful. Those three optimizations are worth **0.9 ms** together (33.2 -> 34.1),
which is a reasonable check on their documented values. And blaze's older tt-metal costs a further
**0.7 ms** at identical config — so the tree swap is nearly free, and **34.8 ms is the baseline any
blaze op substitution must beat.**

**3. No blaze op has been substituted into the model yet**, so there is still no
blaze-attributable step improvement. What remains is genuinely mechanical rather than blocked:
`DRAMStreamingMatmul` needs its weights DRAM-width-sharded *and* column-major tile-shuffled (a
load-time transform), its activation replicated height-sharded across the 8 bank workers as a
1x32 tile, and its output resharded back to what the model's next op expects. The activation and
output resharding are per-call ops that did not exist before, and at ~35 us of saving per call
they could plausibly consume the win — which is exactly why blaze fuses whole stages instead of
single matmuls, and why the fused ops (blocked on F4's down projection and F11's gather) are
where the real prize sits.

### A GLM-specific FusedOp, built from micro-ops: `GLMQKVAProjection` — 9.43x

blaze's own MLA fused ops cannot express this model (F3: 20 heads), and its MoE ones are blocked
(F4 down, F11 gather). But `DRAMStreamingMatmul` is correct at every GLM shape, so the
projections can be fused directly. `blaze/ops/glm_qkv_a_projection/` composes the two that share
an activation — modelled on `swiglu`'s gate/up pair, stopping before the Gather that deadlocks:

    act ──► DRAMStreamingMatmul(w_q_a)  ──► q_a_out    pop_act=False, act stays live
        └─► DRAMStreamingMatmul(w_kv_a) ──► kv_a_out   pop_act=True, last consumer

| impl | dispatches | cores | µs | q_a PCC | kv_a PCC |
|---|---:|---:|---:|---:|---:|
| ttnn: 2x `dram_sharded_linear` | 2 | 64 | 89.4 | 0.9999 | 0.9999 |
| ttnn: **1x fused `q_kv_a`** — what the model actually runs | 1 | 64 | **45.1** | — | — |
| **blaze: `GLMQKVAProjection`** | **1** | 8 | **9.5** | 0.9999 | 0.9999 |

**4.76x against the real baseline**, ~1.67 ms/token over 47 layers as an upper bound.

**A correction:** the first version of this measured only against two separate ttnn matmuls and
claimed 9.43x. That is not the shipping path — `GLM4_MOE_LITE_FUSE_QKV_A=1` is in the winning
defaults, so the model already concatenates these into ONE 2048x1344 matmul and slices it. The
two-matmul baseline overstated blaze by 2x. The lesson generalises to the rest of this document:
a per-op A/B is only as honest as its baseline is representative of the shipping configuration. It also proves blaze is usable for this model *without* waiting on F3/F4/F11: a
GLM-shaped FusedOp can be assembled from micro-ops that already work on a harvested grid.

**But read where the win comes from.** The two matmuls measured *separately* are 4.9 + 5.0 =
9.9 µs; fused they are 9.5 µs. **Fusion itself contributes ~0.4 µs, about 4%.** The 9.43x is
almost entirely `DRAMStreamingMatmul` beating ttnn's matmul, not the fusing. That matters because
the fusion is the part that was supposed to survive the critical-path problem: keeping an
intermediate in L1 is worth something only if there was a DRAM round-trip to avoid, and between
these two projections there is not one — they share an input rather than chaining.

The mechanism-2 case therefore still rests on a *chained* fusion (norm -> proj, or matmul ->
eltwise -> matmul), where an intermediate genuinely round-trips today. That is the next op to
build, and it is the honest test of whether fusion beats the ~12% ceiling.

### The chained fusion, at last: `GLMQANormQBProjection` — 1.22x

The previous op fused two projections that *share* an input, so no intermediate round-tripped and
fusion contributed only ~4%. This one is the real mechanism-2 test: `q_a -> RMSNorm -> q_b` is the
one place in GLM's decode attention where a norm feeds straight into a matmul, and the normalized
q_a exists solely to be consumed. It uses FlashNorm deferred normalization -- gamma folds into the
weight offline, so the matmul runs on un-normalized q_a and `1/RMS` rides the
`DRAMStreamingMatmul` scalar epilogue. No new kernel.

**Correctness (both gates required, both pass):**

| gate | against | value | bar |
|---|---|---:|---|
| `pcc_vs_device` | `(1/RMS) * (q_a @ W'_bf8)` | **0.99989** | ≥ 0.99 |
| `pcc_vs_model` | `RMSNorm(q_a, gamma) @ W_q_b` | **0.99988** | ≥ 0.99 |

**Cluster timing** (separate from any end-to-end claim):

| impl | dispatches | cores | µs |
|---|---:|---:|---:|
| ttnn `rms_norm` + `q_b` matmul | 2 | 24 | 25.9 |
| **blaze `GLMQANormQBProjection`** | **1** | 8 | **21.1** |

**1.22x, ~0.22 ms/token over 47 layers as an upper bound.** Modest — and informative precisely
because it is. Removing the DRAM round-trip works, but the intermediate it removes is small: q_b
is the *weakest* of the six matmul shapes against ttnn (1.75x), and the ttnn norm it deletes is
only ~5 µs at 768 elements. So mechanism 2 is real and now demonstrated end to end, but at this
cluster's size it is worth about a fifth of what the matmul substitution alone is worth.

Two traps the handoff called correctly, both confirmed: K must pad 768 -> 1024 for `SumOfSquares`
(768 is a multiple of neither 512 nor 1024, so `interpret_tile` would silently cover 512 of 768
and produce a wrong RMS) while the reduce still divides by the logical 768; and
`fp32_dest_acc_en` must be set on the **FusedProgram**, since the norm's `DST_ACCUM_MODE` comes
from the program's ComputeConfigDescriptor, not from `emit`.

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
| `shared_expert`, `glm_moe` | FusedOp | blocked on the **down** projection only (F4 revised); gate/up solves to 48/48 |

## 4. Findings

### F1 — `DRAMStreamingMatmul` is numerically wrong at m=32

At GLM's o_proj shape (5120x2048): **PCC 0.0074** at m=32; m=1 and m=8 pass at 0.9999. Reproduced
with blaze's own `_run_and_compare`, not a custom harness. The shipped tests only cover
m ∈ {1,4,8} — `test_dram_streaming_matmul.py:440`, and no test in the file goes above 8 — so m=32
is untested territory. m=32 is a full tile of rows and is the shape a batched decode step needs,
which is why the bs=1 wins in section 2 do not extend to batched throughput.

Coverage patch (an xfail case, not a fix — root cause not investigated) at
[`blaze_eval/upstream/05-dram-streaming-matmul-m32-coverage.patch`](blaze_eval/upstream/05-dram-streaming-matmul-m32-coverage.patch).

### F2 — `ab_harness` cannot import its PCC helper

`blaze-vs-ttnn-bench/scripts/ab_harness.py:132` does
`from tests.blaze.utils.torch_golden import comp_pcc`. That can never resolve: `tt-blaze/tests`
has no `__init__.py` (namespace portion only), while `tt-blaze/tt-metal/tests` **does** and is
also on `PYTHONPATH` — a regular package terminates the namespace search and shadows it.
The harness is unusable out of the box, and reordering `sys.path` does not help — a namespace
portion is never merged with a regular package, so the regular one wins wherever it sits. Pytest
is unaffected because it collects rootdir-relative rather than through this import, which is
presumably how it survived. Fix upstream is adding `tests/__init__.py` and
`tests/blaze/__init__.py`; we worked around it by registering the module by path. Confirmed
empirically off-device with a minimal directory pair. Patch at
[`blaze_eval/upstream/06-tt-blaze-tests-namespace-packages.patch`](blaze_eval/upstream/06-tt-blaze-tests-namespace-packages.patch).

### F3 (resolved, rewritten) — GLM's 20 heads do fit; the original finding blamed the wrong file

**Superseded.** The earlier version of this finding said MLA "cannot express" 20 heads and named
`tests/blaze/backed/layout_plan.py:186`'s `n_heads_per_device % 8 == 0` as the blocker. That
requirement is real but it is **not a production constraint**: `layout_plan.py` lives under
`tests/`, and production derives the Q-branch geometry from the *weight shard spec* —
`q_heads/op.py:155-157` takes `qnope_grid` from `kv_b1_proj_weights.core_ranges` (or its
`shard_spec.grid`), and `create_q_heads/op.py:67` reads `qnope_cols` off that grid's width. The
`_QB_GRID_ROWS = 8` was a hand-written constant in a test helper, not an invariant of the ops.

The two constraints that **are** binding, both verified in source:

1. **`create_q_heads/kernels/op.hpp` hardcodes two RoPE heads per core.** At HEAD, `:253`
   computes the destination as `2 * qrope_col * qrope_head_size_bytes` and `:255-260` transfers
   `double_qrope_head_size_bytes = qrope_head_size_bytes * 2` under a `NOC_MAX_BURST_SIZE`
   static_assert. This — not `qb_per_core` — is the true source of the 2:1 nope:rope assumption.
   `layout_plan.py:199` at HEAD derived `qrope_heads_per_core = qk_nope_head_dim //
   qk_rope_head_dim`, which is 2 for DSv3 and Kimi and therefore *coincidentally* matched the
   kernel's literal. The real relation is a layout one — a sender row lays down `qnope_cols` RoPE
   parts, so `qrope_cols * qrope_heads_per_core == qnope_cols` — and it has nothing to do with the
   dimension ratio.
2. **`flash_mla/op.py:535` asserts `B == grid.CORES_PER_BLOCK`**, and only two SDPA grids exist:
   `FlashMLAOptimalGridNOC0` with 8 (`:95`) and `FlashMLAOptimal4CoreGridNOC0` with 4 (`:260`),
   with `select_flash_mla_grid` (`:299-305`) choosing between them. So the receiver count has
   exactly **two legal values**.

Under those, GLM-4.7-Flash's 20 heads fit: **4 sender rows × (5 NoPE + 1 RoPE) columns, B=4**,
`qrope_heads_per_core = 5`, `heads_per_receiver = 5`. `blaze/models/config/mla_q_grid.py`
searches that constraint set instead of assuming a shape, and its preference order is chosen so
DSv3 (8 rows × (8+4), B=8) and Kimi (8 rows × (4+2), B=4) come out exactly as the hand-written
constants gave them. **CPU-only, 20/20 pass** (`tests/blaze/backed/test_mla_q_grid.py`); the
kernel side is **NOT silicon-validated** — `qrope_heads_per_core` is now a real CT arg
(`create_q_heads/op.py:277`) with `MAX_WORKERS = 8` and `MAX_QNOPE_COLS = 8` enforced
(`:225-262`), but nobody has run MLA at GLM's shape on hardware.

**GLM needs `scattered_q_heads`, and for a reason worth stating precisely.** A ttnn shard spec
carries one shard shape, so a single `q_b` tensor can only span the NoPE and RoPE core sets when
they want the same per-core width — `qk_nope_head_dim == qrope_heads_per_core * qk_rope_head_dim`.
Every 2:1 model satisfies that at `qrope_heads_per_core = 2` (128 either side). GLM lands on 192
NoPE columns against 5×64 = **320** RoPE columns, so `q_b` runs on its own uniform grid and
`ScatterOffset` routes the blocks out (`scattered_q_heads/op.py:6-30`). The cost is the lost DEST
fusion that `QHeads` gets. `MLAQGridLayout.q_b_shard_is_uniform` is the predicate.

Separately true and separately notable: GLM is the first model in this tree where **`v_head_dim`
(256) differs from `qk_nope_head_dim` (192)** — DSv3 and both Kimis are 128/128. GLM-5.1 and
GLM-5.2 share GLM-4.7-Flash's 192/64/256, so the 3:1 ratio is **not** unique to this model, which
makes the hardcoded 2 a live problem for the GLM family generally rather than a one-off. Note the
v_head_dim asymmetry is *not* what forces `scattered_q_heads` — the non-uniform `q_b` shard above
is — and an earlier draft of this note conflated the two.

**One pre-existing failure surfaced, unrelated to the solver.**
`tests/blaze/backed/test_layout_plan.py:187-189` asserts Kimi's `heads_per_receiver == 4`, but
HEAD's own arithmetic gives **8** (`n_sdpa_cores = max(1, 32//8) = 4` receivers, `32//4 = 8` heads
each — `layout_plan.py:183-192`, comment *"derive such that heads_per_receiver = 8"*). So that
test fails at HEAD, before any change here, and the solver reproduces HEAD's 8 rather than the
test's 4. Meanwhile `create_q_heads/op.py:25` and `flash_mla/op.py:498-499` both describe Kimi as
`heads_per_receiver=4`. Three places in tt-blaze disagree about what Kimi runs; we have not
resolved it and deliberately did not pick a side (`mla_q_grid.py:26-32`). **Someone with a Kimi
device should settle it** — the answer changes the SDPA grid.

### F4 (revised) — gate/up is already solved upstream; only `down` is blocked

Earlier revisions said the shared-expert gate/up split "only balances at 130 cores" and would
need upstream work. **Half of that is wrong.** Blaze already ships a general, grid-aware solver,
`moe_grid_layout.solve_shared_gate_up_grid(gc, k_dim, n_dim)`, which derives
`(k_parallel, n_parallel)` from the real dims and refuses to return an unbalanced split. On this
12x10 grid (118 usable cores) it produces balanced splits for both models:

| model | k_dim x n_dim | k_par | n_par | gate | up |
|---|---|---:|---:|---:|---:|
| GLM-4.7-Flash shared expert | 2048 x 1536 | 1 | 48 | **48** | **48** |
| GLM-5.1 / DSv3 (tp=8) | 6144 x 256 | 6 | 8 | 48 | 48 |

So the 64/54 imbalance and the `gate=(8,8) up=(8,7)` error come from the **legacy hardcoded
path** — `gate_up_coords_from_device`, with `NUM_SHARED_GATE_UP_MM_CORES = 64` and a fixed column
pattern — which the GLM-5 config and test use instead of the solver. Routing through the solver
is the fix, and it needs no new arithmetic.

**What IS still blocked is the down projection.** `NUM_SHARED_DOWN_MM_CORES = 112` is hardcoded
and `shared_down_coords_from_device` returns `get_matmul_cores()[:112]`, but on this harvested
grid that yields only **56** cores — short by half. Unlike gate/up there is no solver for it. So
`shared_expert`, and therefore `glm_moe`, remain unreachable here, but for a narrower and more
tractable reason than previously recorded.

Our `GLM4_FLASH_BLAZE_CONFIG` consequently leaves all three shared-expert coord tuples empty and
targets the routed path only, which `sanity_check_model_config` permits.

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

### F10 — the routed-expert test's gate bias is hardcoded to 256 experts

`tests/blaze/glm5_1/test_glm5_routed_expert.py:209` did
`gate_mm_bias_torch.reshape(-1).reshape(1, 16, 16)`. The bias is `[1,1,1,num_experts]`, so
this only works when `num_experts` exactly fills a 16x16 face — GLM-5's 256 does, GLM-4.7-Flash's
64 does not, and it raised `shape '[1, 16, 16]' is invalid for input of size 64`. The op
itself is parameterized by `num_experts`; only the test was fixed to 256.

Fixed by padding to a `face_h x face_w` face, which is byte-identical for 256 experts.

**Corrected since first written:** the original patch padded with **zeros**, and that is wrong.
The gate ranks `act + bias` over every lane, so a zero-act zero-bias padding lane scores 0 and
beats any real expert whose score went negative. The pad value must be a large negative bias —
`-100.0`, which is what blaze's own `create_gate_tensors` uses (`generic_moe_gate/op.py:93`). This
is the caller half of the contract F14 describes; neither half is sufficient alone. Patch now at
[`blaze_eval/upstream/03-glm5-gate-bias-face-dim-general.patch`](blaze_eval/upstream/03-glm5-gate-bias-face-dim-general.patch).

### F11 — `glm_routed_expert` still hangs at GLM-4.7-Flash dims, with a valid config

With `GLM4_FLASH_BLAZE_CONFIG` (below) — 2 gate cores, sender (11,9), all sanity checks
passing — and F10 fixed, the routed expert **still hangs** at 2048 / 1536 / 64 experts. Killed
at 700 s; the GLM-5 shape completes in ~190 s on the same build. **The Galaxy recovered
without a reset**, as in F6.

So F6's gate-core mismatch was not the whole story: something else in this path does not
tolerate GLM's dims. Candidates not yet eliminated — 2 gate cores versus GLM-5's 8 changes the
gather/handshake width, and top-k is 4 here versus 8 there.

**Localised with tt-triage** (callstacks archived at
[`blaze_eval/F11_triage_callstacks.log`](blaze_eval/F11_triage_callstacks.log)). Stuck roles,
by op:

| stuck in | frames | reading |
|---|---:|---|
| `swiglu__gather::sender_impl` | 16 | gather senders blocked — **head of the chain** |
| `swiglu__gather::receiver_impl` | 1 | gather receiver waiting for pages |
| `swiglu__post_gather_mcast::sender_impl` | 1 | waiting on gather output |
| `swiglu__post_gather_mcast::receiver_impl` | 111 | waiting on that sender |

So the **Gather** deadlocks and the mcast stall is derivative — the 111 waiting receivers are a
symptom, not the bug. The gather's resolved CT args are self-consistent at GLM's shape
(`noc1_num_senders = 8`, `dst_num_pages = 48` = 8 cores x 6 tiles for hidden 1536), and
`compute_cores` / `out_num_tiles` are *derived* from the gate matmul's output handle rather than
hardcoded, so the mismatch is not in that accounting. Resolving it needs kernel-level DPRINT
(`BLAZE_DEBUG_KERNELS=1`, which per `WRITING_A_FUSED_OP.md` marks each phase boundary) or
`cb-tap` / `cb-inject`. Running triage needs `TT_METAL_INSPECTOR=1`, and the logs land in
`<cwd>/generated/inspector`, **not** the `/tmp/tt-metal/inspector` the error message suggests.

### F12 — a hang DEGRADES the device, and open/close does not detect it

Earlier revisions of this document said the Galaxy "recovered without a reset" after the F6 and
F11 hangs. **That was wrong**, and the health check behind it was too weak: opening a device,
reading its grid size and closing it succeeds on a degraded device. After several hangs, real
kernel work began hanging unconditionally — including `o_proj`, which had passed in 23 s
minutes earlier.

`tt-smi -r` fixed it, and the 6/6 shape sweep then passed in 12 s. The control experiment is
what caught it: **re-run a known-good case after any hang**, and treat a previously-passing test
that now hangs as a degraded device rather than a new finding. A "1536 hangs" result was
attributed to the shape before the control disproved it.

`tt-smi` was not on PATH but is installed at
`tt-metal/python_env/bin/tt-smi`. On Galaxy it warns that CPLD FW v1.16+ is needed for `-r` and
suggests `-glx_reset` as the fallback; `-r` worked here.

### F13 — replicated mesh tensors report the concatenated shape on older tt-metal

The last thing standing between GLM and a traced run in blaze's tree. `ttnn.copy` rejected the
per-step RoPE write:

    copy_device_operation.cpp: out_tensor.logical_shape() == input_tensor_a.logical_shape()
    input Shape([1,1,1,64]) does not match output Shape([1,1,32,64])

Cause: on that tt-metal a *replicated mesh tensor* built by `from_torch(..., mesh_mapper=...)`
reports the **concatenated** shape -- 32 rows on a 32-device mesh -- while the device ops feeding
it (`transpose` -> `interleaved_to_sharded`) report the per-device shape. The check therefore
compares a global shape against a per-device one and can never pass.

Nothing works from the consumer side: `ttnn.reshape` fails on volume (2048 vs 64) and
`ttnn.assign` hits the same validation. What does work is `allocate_tensor_on_device` with an
explicit `TensorSpec`, which gives per-device semantics on both versions -- verified by writing
7.0 through the copy and reading it back from the original buffer.

`model_tt._match_rope_buffer_shape` applies this **only when the shapes actually disagree**, so on
our own tt-metal it is a no-op and the buffers are byte-identical (confirmed: the log line never
fires, and the step stays at 33.1-33.2 ms). It also re-zeros the freshly allocated buffer, since
`allocate_tensor_on_device` does not, and these feed RoPE.

Two smaller divergences fell out on the way: `ttnn.argmax` has no `use_multicore` on older
tt-metal (handled by `_argmax_mc`, probed once and remembered), and a run-summary line called
`ttnn.device.get_default_dispatch_core_type()` directly -- a reporting line should never be what
makes a model unrunnable, so it goes through `dispatch_core_label()` now.

### F9 (revised) — the trees are asymmetric: the MODEL runs in blaze's tree

Blaze needs its own tt-metal — not for C++20 as its README says, but for SFPI flags ours lacks
(`-ftt-nttp -ftt-constinit -ftt-consteval -ftt-no-dyninit`). The original conclusion drawn from
that — "benchmarks run in blaze's tree, the model runs in ours, and the two are not
interchangeable" — was **too pessimistic in one direction that matters**.

GLM-4.7-Flash is pure Python over ttnn, and it **imports and prefills successfully inside
blaze's tt-metal tree** (v0.75.0-dev): weights load, 47 layers build, prefill completes in
~1.2 s. That is the route to the step-level measurement this evaluation needs — blaze ops and
the model in one process — and it does not require porting blaze into our tree.

Getting there means clearing genuine divergences. Cleared so far:

| divergence | why | fix |
|---|---|---|
| `get_default_dispatch_core_type/axis` absent | newer helpers | derive WORKER/COL vs ROW from `is_blackhole()` |
| `permute/slice/concat(sub_core_grids=…)` rejected | **our** ttnn addition | `linear_helpers.scg_kwargs` — omit the kwarg when None |
| `sparse_matmul(post_scale=…)` rejected | **our** ttnn addition | omit when None; the fused down-routing-scale needs `=0` there |
| `fast_reduce_nc(epilogue_input_a/b=…)` rejected | **our** ttnn addition | omit when None; fused collective epilogue needs `=0` |
| `topk_router_gpt` asserts `num_experts == 128` | blaze's copy is OLDER than ours | run with `FUSED_ROUTER=0` |

Three of those were latent bugs in this model regardless of blaze: it passed **our own** ttnn
extensions unconditionally, even as `None`, so it could not run on stock ttnn at all. The
`scg_kwargs` / conditional-kwarg fixes are portability improvements in their own right, and are
verified non-regressing — 33.2 ms/token and correct output, unchanged.

Remaining, and the current stopping point: `ttnn.copy` in blaze's tree requires matching logical
shapes (`Shape([1,1,1,64])` vs `Shape([1,1,32,64])`) where our newer tt-metal broadcasts. That
is in the decode trace-capture path, so it needs a real model change rather than a flag.

**Caveat for whoever finishes this.** Reaching a running decode there costs three default-on
optimizations (`FUSE_DOWN_ROUTING_SCALE`, `FUSED_COLLECTIVE_EPILOGUE`, `FUSED_ROUTER`), so a
step time measured in blaze's tree is **not** comparable to our 33.2 ms. The valid experiment is
same-tree, same-flags, with and without the blaze op swapped in.

### F14 — the MoE gate ranked uninitialised L1 as an expert

This is the whole of the PCC 0.0235 that replaced F11's hang, and the mechanism is worth reading
in full because every step of it is a legal operation.

**The bookkeeping error.** The gate face is assembled by a `Gather` that writes one 32-wide page
per gate-MM core, so it fills `ceil(num_experts/32)*32` lanes. The SFPU top-k ranks
`num_total_experts` lanes, and it must be a multiple of 128 —
`ckernel_sfpu_generic_moe_gate_topk.h:176-177` static_asserts `>= 128 && % 128 == 0`
(`..._top8.h:178-179` likewise), so 64 experts are ranked as **128 lanes**. `glm_moe_router` then
passed `num_total_experts=256` as a literal (`glm_moe_router/op.py:137` at HEAD). Either way, at
GLM's 64 experts over 2 gate cores, lanes 64 upward were never written: the destination was
`f.cb_scratch`, a **shared arena slot** with documented temporal reuse, so those lanes held
whatever last occupied that L1.

**Why the existing sentinel could not save it.** blaze's own reference tensor builder pads *both*
halves — `generic_moe_gate/op.py:92-93` fills the act face with `0.0` and the bias face with
`-100.0`. That pairing is the contract, and it is a pair for a reason: the gate ranks `act + bias`
per lane, so bounding one side bounds nothing. The router path had **no act pad at all**, and its
gathered logits are post-sigmoid, hence in (0,1) and scoring at most ~1. Any stale bf16 word above
~100 outranks every real expert even with `-100.0` in the bias. The contract was documented all
along and only half-followed.

**Why it returned finite garbage instead of faulting.** A winning padding lane yields an expert id
in `[num_experts, num_total_experts)`, which has no weights behind it. The reader applies that id
with **no bounds check**: `dram_streaming_matmul/kernels/op.hpp:182-191` reads
`index_ptr[index_offset]` and computes `expert_offset_bytes = expert_idx * expert_size_bytes`. At
these dims the per-bank expert stride is such that the bad offset lands *inside a different real
weight tensor* — a legal DRAM read of real bytes. That is precisely why the symptom changed from a
hang to **PCC 0.0235**: a hang is a synchronisation failure, and this was not one. Inferred from
the arithmetic and the absence of a check, not instrumented.

**The fix** pins a zero-filled tensor-backed face whenever the gather cannot fill what the kernel
ranks. `gate_face_padding_plan()` separates lanes written from lanes ranked;
`make_gate_gather_dst()` returns the unchanged `f.cb_scratch` when they agree and an
`f.named_tensor`-backed zero face when they do not. Three properties make the zeros hold where a
scratch slot's would not: named tensors are pinned in the compiled program's `lifetime_tensors`
(`blaze/compiler.py:317-329`) so the buffer is never re-let; the gather writes only
`sender_idx * data_size_bytes` from the base (`gather/kernels/op.hpp:108`); and the one-page CB is
popped every iteration (`generic_moe_gate/kernels/op.hpp:92`) so the write pointer returns to base.
Configs whose experts fill the ranked block — GLM-5 and DeepSeek-V4 at 256 — take a byte-identical
path.

**VERIFIED on silicon:** the device then selected expert **44**, a real lane. Additionally
**CPU-only, 36/36 pass**, `tests/blaze/micro_ops/moe/test_gate_tail_padding_contract.py` and
`test_moe_router_gate_wiring.py` — they reproduce the half-padded configuration through
`GenericMoeGate.golden` and show the sentinel failing, then passing once the act tail is zeroed,
over tail values of 1e4, bf16 max, NaN, 1.0 and 0.51.

**The lesson that generalises.** Nothing in the existing suite could have caught this: every
device test in this family runs 128 or 256 experts, where the rounding is a no-op and the tail does
not exist, and the micro-op tests hand the gate a fully populated face. A defect that only appears
below a kernel block size is invisible to a suite that only tests at and above it.

### F15 — `full_sort=False` promises a set; the golden asserted an order

With F14 fixed, the remaining failure was a **tie**. Experts 44 and 35 both scored exactly
`1.0234375` — the bf16 value sitting at the k-th rank cutoff. Torch's `topk` broke it one way, the
SFPU the other. **Both are correct.** `GLMMoERouter` runs `GenericMoeGate` with
`full_sort=False`, and that mode promises the top-k *set* — neither an order across output lanes
nor a tie-break rule. `GLMRoutedExpert.golden` read `top_indices[0, lane]`, i.e. assumed lane `i`
holds the i-th largest.

This is a **golden defect, not a device defect**. Nothing was wrong on hardware.

Three things make it worth stating rather than patching around:

- The rest of the family already knows. `test_generic_moe_gate.py:252-258` compares **sorted sets**
  for exactly this reason, and only the `full_sort=True` branch (`:229-241`) asserts descending
  order. The routed-expert golden is the one place that forgot.
- **Exact ties are the expected case here, not a freak event.** bf16 carries ~8 mantissa bits and
  the gate ranks 64 post-sigmoid scores in (0,1). Any future numerics gate on this op will hit
  this, and the natural reaction — loosening the PCC threshold until it passes — hides real errors
  while not fixing the tie.
- The obvious cheap comparison is unsound in a second way. PCC is **scale-invariant**, so comparing
  the device against a single candidate cannot tell whether it paired the right routing weight with
  the right expert. `golden_lane_candidates()` therefore returns each legal candidate *with the
  score that expert would have earned*, which restores that check. Not hypothetical: it is how we
  found that the test passed `"routing_scaling_factor"` while `GLMRoutedExpert.compose` reads
  `"scaling_factor"` — the key was never consumed, so the device ran the 2.5 default against a
  golden using GLM-5's real factor, and scale invariance hid it for the life of the test. That is a
  fifth instance of F16.

**VERIFIED on silicon:** this is the change that took the GLM-4.7-Flash routed-expert probe from
failing on the 44-vs-35 tie to passing.

### F16 — a bug class: a parameter accepted, threaded, and then dropped at the call

**The single most valuable thing in this document to raise upstream**, and it should be filed as a
class rather than as four bugs. The shape is always identical: a parameter is threaded through the
public surface, carries a default that is correct for the model the op was written against, and is
then ignored or replaced by a literal at the inner call. Nothing fails. The model runs with the
default. Line numbers at HEAD, all under `blaze/ops/`:

| # | site | what happens | consequence |
|---|---|---|---|
| 1 | `glm_moe_router/op.py:137` | `num_total_experts=256` literal, while `num_experts` is accepted at `:94`, read at `:63` and passed at `:71` | F14 — 64 experts ranked over 256 lanes |
| 2 | `glm_moe_router/op.py:138` | `pop_act=True` literal at the gate call, while `pop_act` is a *live* parameter for the gate MM at `:103` | none — the arg is inert (below) |
| 3 | `glm_moe/op.py` | never accepted `num_experts` at all, in `compose()` or `emit()` | no caller could set it; the 256 default always won |
| 4 | `glm_moe_no_shared/op.py` | accepted in **both** `compose()` and `emit()`, then omitted at the `GLMRoutedExpert.emit` call | the most legible instance: plumbing complete except the last hop |
| 5 | `tests/blaze/glm5_1/test_glm5_routed_expert.py` | passed `"routing_scaling_factor"`; `GLMRoutedExpert.compose` reads `"scaling_factor"` | device ran the 2.5 default; PCC's scale invariance hid it (F15) |

On #2: the gate's `pop_act` is **inert**. `generic_moe_gate/kernels/op.hpp:47` is its only
occurrence in the kernel — act and bias are popped unconditionally at `:92-93`. A comment claiming
it controlled popping the gathered CB was wrong. A parameter that looks live and is not is exactly
how #1 survived, so the recommendation upstream is to delete it from both sides rather than repair
its plumbing.

**Three further call sites carry the same defect and fail loudly instead**, which is the part that
makes this a class: `gptoss_moe_router/op.py:162`, `ffn_interleaved/op.py:476` and
`deepseekv4_moe_large_router/op.py:156` all pass `num_total_experts=num_experts` — the raw,
unrounded model count. Below 128, or at anything not a multiple of 128, that is a JIT
static_assert failure at build time. So the identical mistake is safe when it trips a bound and
unsafe when it hits a default, and which one you get is incidental to the mistake. All three also
pass the inert `pop_act=True` (`:163`, `:477`, `:157`). We did not otherwise change them — no
numerics gate for those models, **NOT RUN**.

Patches: [`blaze_eval/upstream/`](blaze_eval/upstream/) 01 and 02, with a CPU-only `ast`-based
wiring test that asserts the derivation directly and would have failed on all four op instances.

### F17 — `DRAMStreamingMatmul` stalls under multi-device mesh execution

**The current critical-path blocker.** It works single-device and stalls on a mesh, which blocks
every fused op built on it — including all three GLM clusters — at the 32-device model boundary.
Everything below is read from the investigating agent's session logs; the eliminations are the
valuable part.

**The stall, on the stock unmodified op.** `blaze/ops/dram_streaming_matmul/` is unmodified in the
working tree; only the test file and the DPRINT-emitting `kernel_codegen.py` are. Driving the op's
own Graph API test on a **4×2 submesh** of the Galaxy fixture with no fabric, at GLM's q_a
geometry (k=2048, n=768, bf16 weights, `subblock_k=1`): **EXIT 124 — hung**, killed at 90 s. The
`BLAZE_DEBUG_KERNELS=1` phase markers on DRAM worker core (0,9) are identical on **all 8 devices**
of the submesh (0, 1, 4, 5, 8, 9, 12, 13):

| RISC | `DRAM_STREAMING_MATMUL start` | `done` |
|---|---:|---:|
| NCRISC (DM1, the weight streamer) | 8 | **0** |
| TRISC0 | 8 | **0** |
| TRISC1 | 8 | **0** |
| TRISC2 | 8 | 8 |
| BRISC | 8 | 8 |

TRISC0/1 get past the activation wait (`[CB] matmul act=0 after_wait`) and then stop; NCRISC never
completes the phase. Under `GLMQKVAProjection` the same signature appears with the phase named:
NCRISC/TRISC0/TRISC1 stall at `GLM_QKV_A__Q_A_PROJ start` on all 8 devices, `KV_A_PROJ` is never
reached, and BRISC/TRISC2 complete. So this is **not** a `GLMQKVAProjection` bug — the base op
does it, and the earlier reading that blamed the fused op was wrong.

**Eliminated, each by a run that still stalled (all EXIT 124):**

| hypothesis | test | result |
|---|---|---|
| weight distribution / per-device addresses | replicated full-K weights, lockstep addresses on all 32 devices | still stalls |
| our harness, not blaze's | production `BlazeCompiler`, 4×2 submesh, `FABRIC_2D`, Galaxy mesh graph descriptor | still stalls |
| fabric involvement | same submesh with fabric *off* | still stalls |
| per-device harvesting | 13 distinct raw Tensix masks across the 32 chips, but identical logical-to-physical maps, and the pinned bank-to-worker assignment matches TTNN's actual assignment on every device | eliminated |

*Caveat on the last row:* that is recorded from the investigating agent's session. **We did not
re-derive the masks**, and it is the one elimination in this table not backed by a log we read.

**A correction to an earlier reading.** An earlier note called this a "selective failure" because
core (11,9) completed every phase while (0,9) stalled. **That was wrong.** (11,9) is not a DRAM
worker — the eight are `(0,9), (0,0), (0,7), (0,3), (7,9), (7,1), (7,6), (7,4)`
(`blaze_eval/glm4_flash_blaze_config.py:50`); (11,9) is the **sender** (`:82`). Its
`DRAM_STREAMING_MATMUL` phases are no-ops, so completing them is not evidence of anything. Beware
this shape of mistake with DPRINT: a core that has nothing to do always looks healthy.

**Why nobody caught it.** The focused tests cover **one device only**.
`test_dram_streaming_matmul_graph_api` is parametrised `mesh_device=[1]`
(`test_dram_streaming_matmul.py:601`), and every other test in the file takes the single-`device`
fixture. The 89/89 pass recorded in section 3 is 89 single-device passes. The multi-device cases
(`..._graph_api_mesh`, `..._bh_submesh_no_fabric`, `..._bh_submesh_with_fabric`) are new, opt-in
behind env vars, and are the ones that fail.

**Open, and being worked right now.** Root cause is not established: NCRISC stalling immediately
after phase start on every device simultaneously is consistent with a global synchronisation or
address-resolution difference between the single-device and mesh program descriptors, but that is a
**hypothesis, not a measurement**. The investigation was live when this was written, including an
in-flight end-to-end substitution attempt, so treat this section as the state at the checkpoint
rather than the last word.

### F18 — `index_offset >= k` blocks production, and the fix is a modelling decision

`glm_moe/op.py:170` derives the routed-expert output lane from **mesh position**:

    index_offset = (row * cols + col) // routed_expert_tp + routed_index_offset

That ranges over `[0, num_devices / routed_expert_tp)` — **0..31** on a 4×8 mesh at tp=1. But the
gate only fills lanes `[0, num_selected_experts)`, and `zero_tail=True` blanks `[k, 16)`. A device
whose lane is beyond k therefore reads **expert id 0 with score 0**: it recomputes expert 0 and
multiplies the result by zero. At GLM-4.7-Flash's **k=4 that is 28 of 32 devices doing silent
no-op work**; at GLM-5's k=8 it is 24 of 32. Nothing raises.

**VERIFIED by reading the derivation and the `zero_tail` semantics; not measured** — it is not
observable as a failure, which is the problem.

The validity condition is `num_devices / routed_expert_tp <= k`. Satisfying it is an
**expert-parallel mapping decision, not a local patch**: it needs either k devices per expert group
with the group id folded into the offset, or a lane-to-device assignment the router emits rather
than one derived from mesh position. We deliberately did not patch it and instead marked both
`glm_moe.emit` and `GLMRoutedExpert.emit` UNRESOLVED in source so the next reader does not have to
rediscover it. `GLMRoutedExpert.golden` now raises on `device_id >= k` rather than returning a
reference the device cannot produce.

This is why the single-device routed-expert tests pass while the path is not production-ready: at
one device `index_offset` is 0 and always valid.

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
| `ab_all_shapes_bench.py` | the same A/B across all 6 decode matmul shapes |
| `glm47_all_shapes_check.py` | correctness of DRAMStreamingMatmul at all 6 shapes |
| `F11_triage_callstacks.log` | tt-triage output localising F11 to the gather |
| `ab_rmsnorm_bench.py` | the rmsnorm A/B |
| `ab_m32_check.py` | reproduces F1 at m ∈ {1,8,32} |
| `no_grid_guard_plugin.py` | stubs `requires_grid_size` (F7) |
| `remap_col12_plugin.py` | column-12 → 11 remap (F5) + gate-core truncation (F6) |
| `blaze/models/glm4_flash/glm4_flash.model_config.json` | GLM-4.7-Flash in blaze's schema |
| `blaze/models/glm4_flash/glm4_flash_blaze_config.py` | `GLM4_FLASH_BLAZE_CONFIG` (F11 target) |
| `tests/blaze/glm5_1/test_glm47_routed_expert_dims.py` | drives the routed expert at our dims |

Copies of all of the above are archived under [`blaze_eval/`](blaze_eval/) so they survive a
re-clone of tt-blaze; see [`blaze_eval/README.md`](blaze_eval/README.md). They are not runnable
from this tree — they import `blaze`.

## 6. Next steps

1. **Step-level measurement is the only thing that settles the matmul question.** All six
   shapes win on kernel time (1.75x–9.17x) but sum to 31.5% of the step against an 11.7%
   bandwidth budget, so the translation rate is unknown and probably low. The route is now
   **mapped and partly cleared** (F9 revised): the model imports and prefills inside blaze's
   tree, five divergences are fixed, and one remains — `ttnn.copy`'s shape strictness in the
   decode trace path. Finish that, then measure same-tree/same-flags with and without the blaze
   op swapped in. Adopt in per-call-win order — mlp_gate_up, q_a, kv_a first — and gate on F1
   (keep m=1 for decode).
2. **`GLM4_FLASH_BLAZE_CONFIG` is written** — [`blaze_eval/glm4_flash_blaze_config.py`](blaze_eval/glm4_flash_blaze_config.py),
   constructs cleanly and passes every `sanity_check_model_config` assertion. The open question
   from the previous revision is resolved: shared-expert coords **may be empty** (only the MoE
   gate / DRAM-worker relations are checked, and `dflash/config.py` sets them empty too), so a
   routed-only config is legal despite F4.
   Its live home is `blaze/models/glm4_flash/` inside a tt-blaze checkout. Notable choices:
   2 gate cores at (11,0)–(11,1) — column 11 is the phantom column on a 12-wide grid, the same
   relationship GLM-5 has to its column 12; DRAM worker order preserved from
   `get_pinned_optimal_dram_bank_to_logical_worker_assignment` because that ordering *is* the
   bank-id assignment; and `attn_sdpa_tp = attn_sdpa_cp = 1`, since the checks require
   `n_heads % tp == 0` and 20 is not divisible by 8.
   The MLA cores in it satisfy the config's divisibility checks so the object is constructible,
   but MLA remains unusable per F3 — that blocker is in `layout_plan.py`, not the config.
   **Finishing the A/B is now blocked on F11, not on the config.**
3. **File F1 and F2 upstream** — both are small, self-contained, and affect any blaze user.
4. **Raise F3 and F4 with the blaze team** — GLM-4.7-Flash is unlikely to be the only model
   with a non-8-divisible head count or a 3:1 nope:rope ratio, and F4 blocks every harvested
   Galaxy, not just this one.
