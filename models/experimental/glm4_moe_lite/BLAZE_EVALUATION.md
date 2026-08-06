# tt-blaze evaluation for GLM-4.7-Flash on Blackhole Galaxy

**Status:** `DRAMStreamingMatmul` is correct and faster on kernel time at **all six** of the
model's decode matmul shapes (1.75x–9.17x, every PCC 0.9999) — the adoption surface is the whole
dense matmul set, not one op. But the per-call wins sum to more than the model's entire measured
weight-bandwidth budget, so most of it cannot be on the critical path: the next milestone is a
**step-level** measurement, not more cluster A/Bs. `glm_moe_router` and `glm_routed_expert` run
on this grid at GLM-5 dims; at GLM-4.7-Flash dims the routed expert hangs, now localised to the
gather (F11). Fused ops beyond that are blocked on layout assumptions in blaze — narrowed to the shared
expert's *down* projection (F4 revised) — documented below with evidence.

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
| ttnn: 2x `dram_sharded_linear` | 2 | 64 | 89.3 | 0.9999 | 0.9999 |
| **blaze: `GLMQKVAProjection`** | **1** | 8 | **9.5** | 0.9999 | 0.9999 |

**9.43x, and 3.75 ms/token over 47 layers as an upper bound** — the largest single opportunity in
the table. It also proves blaze is usable for this model *without* waiting on F3/F4/F11: a
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

Fixed by zero-padding to a `face_h x face_w` face, which is byte-identical for 256 experts.
Patch preserved at
[`blaze_eval/glm5_routed_expert_gate_bias_dim_general.patch`](blaze_eval/glm5_routed_expert_gate_bias_dim_general.patch)
— small and self-contained, **worth upstreaming** alongside F1/F2.

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
