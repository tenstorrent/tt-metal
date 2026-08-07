# GLM-4.7-Flash on Blackhole Galaxy: the tt-blaze evaluation

**Status: no Blaze fused op is enabled in the model. The model runs at 33.2 ms/token
(34.6 ms/token in the Blaze tree, which is where any Blaze comparison must be made).**

This report covers everything attempted from the Wormhole → Blackhole Galaxy port onward: what was
built, what was measured, which measurements were wrong and why, and the evidence for the
conclusion that Blaze fused ops cannot currently improve this model's end-to-end decode.

Every number below is from hardware — a 32-chip Blackhole Galaxy (4×8 mesh, 1×-harvested, 12×10
Tensix grid, 8 DRAM banks, 1.5 MB L1/core) at ISL 128, batch 1, traced decode.

---

## 1. Baselines, and which one to compare against

| configuration | ms/token |
|---|---:|
| our tt-metal tree, all optimisations on | **33.2** |
| our tree, three flags off | 34.1 |
| **Blaze's tt-metal tree, three flags off** | **34.6 – 34.8** |

The model must run inside Blaze's own tt-metal checkout to use Blaze at all (it needs the
`-ftt-nttp` / `-ftt-constinit` / `-ftt-consteval` / `-ftt-no-dyninit` SFPI flags). Blaze's ttnn is
older and lacks parameters three of our optimisations use, so
`GLM4_MOE_LITE_FUSE_DOWN_ROUTING_SCALE`, `GLM4_MOE_LITE_FUSED_COLLECTIVE_EPILOGUE` and
`GLM4_MOE_LITE_FUSED_ROUTER` must be off there.

**Any Blaze claim must be measured against 34.6, not 33.2.** Mixing them overstates Blaze by
1.4 ms/token before a single kernel changes.

---

## 2. The Blackhole port (completed, shipping)

Work done to get GLM-4.7-Flash running on BH Galaxy at all:

- **`tt/runtime_config.py`** — `is_ubb_galaxy()`, `is_blackhole_galaxy()`, `hw_max_ccl_links()`,
  `galaxy_fabric_config()`, `default_ccl_topology()`, `dispatch_core_config()` (with an
  `AttributeError` fallback deriving WORKER/COL vs ROW from `is_blackhole()`), `dispatch_core_label()`,
  and `ccl_settings_from_env()` as the single clamp point for CCL settings.
- **`tt/linear_helpers.py`** — `worker_grid_x()`, `sdpa_grid_x()`, and `scg_kwargs(scg)` applied at
  28 call sites so ops can be confined to the prefetcher's worker region.
- **`ttnn/.../topk_router_gpt/device/kernels/compute.cpp`** — a real architecture bug: ZEROACC
  needed `#ifdef ARCH_BLACKHOLE` with `use_32_bit_mode = !glm_mode`. This is a ttnn fix, not a
  model workaround.
- **`ttnn` permute/transpose** — added `sub_core_grids`, needed because those ops size their grid
  from the full device and are rejected under the prefetcher's SubDevice.
- **`tt/model_tt.py`** — `_match_rope_buffer_shape()` (mesh-shape divergence, allocates with an
  explicit `TensorSpec` only when shapes disagree) and `_argmax_mc()`.

Result: **33.2 ms/token**, correct output, suite green.

---

## 3. What actually bounds the decode step (measured before any Blaze work)

This was established first, and it governs everything after. From commit `18479ed4ad0` (Tracy
capture plus on-device A/B of every candidate lever):

| lever | effect |
|---|---:|
| `DENSE_TT_DTYPE=bf16` (2× weight bytes) | **+9.8 ms** |
| `SHARDED_NORM=0` | +2.8 ms |
| `CCL_NUM_LINKS=1` | +1.6 ms |
| `CCL_TOPOLOGY=linear` | 0.0 ms |
| **`USE_DECODE_ROPE=0` (removes ~564 ops/step, 23% of all ops)** | **0.0 ms** |

Two conclusions, both load-bearing for the Blaze work:

1. **The step is weight-bandwidth-bound.** Doubling weight bytes costs 9.8 ms.
2. **Op count is not a lever under trace.** Removing 23% of all ops changed nothing. The profiler
   shows 26.3 ms of device kernel time in a 50.1 ms step and implies ~25 ms of per-op firmware
   overhead — but under trace replay the command stream is pre-recorded and that setup is
   *pipelined*. **The profiler serialises what trace overlaps.**

Blaze's central mechanism is fusion: fewer ops, fewer DRAM round trips. Point 2 says the first half
of that is a measured null on this model. Point 1 says the second half doesn't help either unless
it reduces *weight* bytes — and Blaze streams the same bf8 weights.

That was the warning sign, before any Blaze op existed.

---

## 4. Blaze ops built and evaluated

| op | origin | outcome |
|---|---|---|
| `GLMQKVAProjection` (q_a + kv_a from one activation) | built here | superseded — see §6 |
| `GLMQANormQBProjection` (q_a RMSNorm → q_b) | handoff | folded into the Q stage |
| `GLMOProjResidual` (o_proj + residual, 5 dispatches → 1) | second agent | **0.37×** |
| `glm_post_sdpa` (the structurally "right" op) | Blaze | **infeasible** |
| routed MoE / `glm_routed_expert` | Blaze | **blocked** |

**Why `glm_post_sdpa` is infeasible:** it is the correct shape and is grid-legal at GLM's dims, but
`Matmul`/`KNMatmul` read weights from **L1**. `w_o` is 11.1 MB/layer × 47 layers = **523 MB** of
o_proj alone, against 120 cores × 1.5 MB = 180 MB of L1. Every layer would re-upload its weights
from DRAM, spending exactly the bandwidth the fusion is meant to save. This is why everything here
is built on `DRAMStreamingMatmul` instead.

**Why routed MoE is blocked:** `num_total_experts`/`zero_tail` gives PCC **0.0235** at GLM's 64
experts (GLM-5's 256-expert shape passes), and some configurations deadlock. An earlier hang was
traced to leaving 8 gate cores wired for 64 experts, which hangs with no error.

---

## 5. Measurement errors found — and why they matter more than any single number

Four separate methodology faults were found and corrected. They are listed because **every
per-op figure in the earlier evaluation was affected**, and because the same traps will bite the
next person.

| # | fault | consequence |
|---|---|---|
| 1 | **Wrong baseline.** The first o_proj A/B used 8 shards (DRAM banks) where ttnn uses up to 80 cores | PCC −0.014; the comparison was meaningless |
| 2 | **Wrong reference op.** "9.43×" compared against two unfused ttnn matmuls when `FUSE_QKV_A=1` is default and the model runs *one* concatenated matmul | corrected to 4.76× |
| 3 | **Program build inside the timed callable.** `FusedProgram` + `emit()` re-run every iteration; ttnn pays no equivalent because it caches programs | measured host composition, not kernel time |
| 4 | **`ab_harness.set_profiler_env()` after the ttnn import.** `_measure()` returns `None` on zero profiler samples instead of raising | silent undercounting; the "9.5 µs" core was really **36.5 µs** |

Fault 4 is the worst of them: it silently deflated the headline. The op the whole evaluation was
built around was ~1.3× faster than ttnn, not 4.8×.

**Rules adopted after this:** hoist the build out of the timed callable and assert repeated
`run()` is bit-identical; call `set_profiler_env()` before importing ttnn; gate PCC against **the
actual ttnn op being replaced**, never a torch golden.

---

## 6. What was fixed, and what it bought

Three real defects were found and fixed in Blaze code. All are correctness-gated.

### 6.1 `TileRowReplicate` — a NOC barrier inside the loop

It read a whole 2 KB tile to extract 64 bytes (one row) and called `noc_async_read_barrier()`
**every iteration**. At K=2048 that is 64 serialised DRAM round-trips ≈ 40 µs — more than the
36.5 µs matmul it fed.

Fixed by chunking (8 pages per barrier): **117.3 → 94.0 µs**, PCC unchanged.

> A faster variant that read only the two 32-byte face lines was 36 µs better and **wrong** —
> PCC 0.71. Successive L1 destinations land at 32-mod-64 while Blackhole DRAM wants 64-byte
> alignment. Rejected.

### 6.2 `DRAMStreamingMatmul` — one worker per DRAM bank

The root cause of every loss. It pins **exactly one worker per DRAM bank** — 8 cores — while ttnn
spreads the same matmul over up to 80.

```
w_o = 5120×2048 bf8 ≈ 10.5 MB
blaze  187.8 µs →  56 GB/s   (~7 GB/s/core)   11% of peak
ttnn    69.1 µs → 152 GB/s                    30% of peak
device peak        512 GB/s
```

Added `BLAZE_DSM_WORKERS_PER_BANK` (default 1, a verified no-op) putting W workers per bank, each
taking a contiguous byte offset into its bank's shard — the column-major tile-shuffled layout makes
a column sub-range a byte range, so no page-index arithmetic changes.

| W | cores | core time |
|---|---:|---:|
| 1 | 8 | 48.4 µs |
| 2 | 16 | 29.9 µs |
| 4 | 32 | **16.1 µs** |

**3.01× on the matmul core.**

> Constraint discovered: `N % (32 · banks · W) == 0`, because a core's shard must be whole 32-wide
> tiles. GLM's `q_a`=768 / `kv_a`=576 pad to 1024 at W≥2 (33%/78% waste); `o_proj`'s N=2048 divides
> exactly.
>
> A first version ordered cores `[b0..b7, b0..b7]`, so sender *i* was not output column block *i*
> and the columns were assembled wrong — **PCC 0.13**, caught only by the gate. Fixed with
> bank-major ordering.

### 6.3 `GatherRowToDRAM` — same barrier defect, plus 32× write amplification

Barrier per page, and it zeroed a full 2 KB tile with a scalar loop to deliver **64 bytes**.
Chunked the barriers, then replaced the whole thing with two 32-byte writes per tile at page
offsets 0 and 512.

> The alignment trap did **not** recur here, and the distinction is worth keeping: both DRAM
> offsets are multiples of 64. The `TileRowReplicate` failure was on the **L1 destination**
> advancing by 32. Same optimisation, other side of the transfer, opposite outcome.

### 6.4 Cumulative

| step | full op | vs ttnn 47.4 µs |
|---|---:|---:|
| original | 117.3 µs | 0.40× |
| + replicate chunking | 94.0 | 0.50× |
| + matmul widening (W=4) | 86.4 | 0.55× |
| + gather barrier chunking | 78.6 | 0.60× |
| + **gather row-0-only writes** | **36.8** | **1.29×** |

**3.19× faster than where it started, and ahead of the ttnn op it replaces.** W=4 is load-bearing —
0.85× at W=1.

---

## 7. Validated on the real 32-chip Galaxy

Every number above was taken on `ttnn.open_device(device_id=0)` — **one chip**. The model runs on
32. `blaze_eval/mesh_qkv_a_ab.py` closed that gap:

| | per chip |
|---|---:|
| blaze q_kv_a (W=4) | **33.7 µs** |
| ttnn q_kv_a | 47.0 µs |
| | **1.39×** |

**PCC 0.999881 on all 32 shards**, checked per device — a device-0-only check would let a program
that landed on one chip pass while leaving 31 wrong.

Two mesh-only traps:
- `ab_harness._measure` **sums** kernel durations across all devices; the raw 32-chip figure is 32×
  a per-chip cost. Divide by `get_num_devices()`.
- **Dots in a Blaze prefix break mesh dispatch** — `prefix="qkv_a.mm"` yields the named arg
  `mm.bank_id` and `program.cpp:427` requires valid C++ identifiers. Underscores only.

---

## 8. The end-to-end result — the decisive measurement

Same script, same flags, back to back, 32 chips, traced, bs=1:

| | ms/token |
|---|---:|
| baseline (no Blaze) | **34.6** (min 34.4, max 34.9) |
| `GLM4_MOE_LITE_BLAZE_QKV_A=1`, `BLAZE_DSM_WORKERS_PER_BANK=4` | **36.8** (min 36.4, max 37.1) |

**+2.2 ms/token — a 6.4% regression — from an op measured at 1.39× faster on the same 32 chips.**

The integration is correct: `GENERATED_IDS` are **bit-identical** to baseline. This is a faithful
drop-in that wins on kernel time and loses on step time. (Trace capture went 7.0 s → 43.3 s
building 47 Blaze programs; one-time, not in the per-token figure.)

### This is the third time a cluster win failed to translate

| | |
|---|---:|
| removing 23% of all ops under trace | 0.0 ms |
| `DS_CORE_CAP=8` (ttnn on Blaze's 8-core layout) | +0.2 ms |
| a real 1.39× kernel win on the real mesh | **+2.2 ms** |

**The op-boundary A/B and the traced step time disagree in _sign_, not just magnitude.** No per-op
number in this evaluation — including the 1.39× — predicts step time.

---

## 9. Stage fusion: the right idea, blocked

If the loss is per-dispatch overhead that trace cannot overlap, a larger stage should amortise it.
Measured on the two-matmul Q stage (`bench_e_fused_stage.py`):

| W | cores | fused stage | **intra-stage L1 boundary** | ttnn same 2 matmuls |
|---|---:|---:|---:|---:|
| 1 | 8 | 63.4 µs | — | 18.8 |
| 2 | 16 | 39.6 µs | **0.5 µs** | 18.8 |
| 4 | 32 | **31.9 µs** | **1.0 µs** | 18.8 |

**The key number in this report: chaining ops inside one program costs ~1 µs; crossing back to the
model's tensor boundary costs ~40 µs.** A 40× argument for larger stages. Fusing the two matmuls
alone took the unfused pair 92.3 → 63.4 µs purely by deleting one crossing.

But the stage is still 1.7× slower than this bench's ttnn reference, and **it deadlocks in the
model**.

### The deadlock: triaged, eight hypotheses eliminated

Stall site (core (0,0), trisc0):
```
llk_wait_tiles ← cb_wait_front (cb_api.h:44)
  ← blaze::DRAMStreamingMatmul::Op<ct_args::glm_q_stage__norm_q_b__q_b_proj>
     at dram_streaming_matmul/kernels/op.hpp:329
```
256 cores (8 bank workers × 32 chips) parked at the mcast consumer, sender silent.

| # | hypothesis | test | result |
|---|---|---|---|
| 1 | shared `_PROGRAM_SEMAPHORES` across 47 layers | `..._OWN_SEM=1` | still hangs |
| 2 | Mcast broken by trace capture | no `--enable-trace` | still hangs |
| 3 | Mcast broken on a mesh | mesh port of the bench | **clean**, PCC 0.9998 |
| 4 | >1 program | N=2..5 | **clean** |
| 5 | receiver-core collision with compute | reversed core scan | still hangs |
| 6 | L1 vs DRAM activation | bench with L1 act | **clean** |
| 7 | scale | **47 programs** | **clean** |
| 8 | L1 contention from interleaved ttnn work | L1 churn between all 47 runs | **clean** |

**The bench cannot reproduce it.** The stage is correct at the model's exact configuration and
scale. The next step is instrumentation of a real run — dump the stage's CB ids, core ranges and L1
offsets from `_build_q_stage_program` and compare against SDPA's static CB region — not another
repro attempt.

---

## 10. Why Blaze cannot currently improve this model end-to-end

Four independent findings, each measured:

1. **Op count is a 0.0 ms lever under trace.** Blaze's primary mechanism — fewer dispatches — was
   measured as a null before any Blaze op existed.
2. **The step is weight-bandwidth-bound**, and `DRAMStreamingMatmul` streams the *same* bf8 weight
   bytes. Blaze does not reduce the quantity that dominates.
3. **A `ttnn.generic_op` dispatch does not overlap under trace the way native ttnn ops do.** This is
   the only explanation consistent with a 1.39× kernel win producing a 6.4% step-time loss, and it
   means the op boundary is the wrong unit of optimisation here.
4. **`DRAMStreamingMatmul` reaches 56 GB/s against ttnn's 152 GB/s of a 512 GB/s device** at the
   default 8 workers. Widening to 32 recovers 3.01× on the core, but the padding constraint
   (`N % (32·banks·W) == 0`) then taxes GLM's 768/576 projections by 33–78%.

**What would change the answer:** stages large enough to amortise the ~40 µs boundary over many ops
— the whole attention block or MoE block, not two matmuls — with the in-model deadlock fixed first.
Intra-stage chaining at ~1 µs says that is architecturally sound; nothing measured here says it is
reachable with the current building blocks.

---

## 11. The larger lever this work uncovered, which is not Blaze

At bs=1 the dense/attention weights are **replicated across all 32 chips**
(`ReplicateTensorToMesh` in `layer_weights.py`); only the MoE experts are sharded. Every chip reads
the full dense weight set to compute the same result.

| | per-chip weight bytes/token | bs=1 latency effect |
|---|---|---|
| replicate (today) | full model | 32× redundant |
| pipeline (4 stages × 8 chips) | ¼ of layers | **no latency gain** — the token still traverses all 47 layers serially |
| **tensor parallel** | 1/N of every weight | **~N× less serial DRAM time per layer** |

`GLM4_MOE_LITE_TP` is in `perf_defaults.PINNED_OFF` for a **correctness** regression, not a
performance one. Since the step is weight-bandwidth-bound, fixing that bug is a far larger lever
than op fusion — and it is a bug to fix, not a kernel to write.

(Intra-Galaxy pipelining *is* supported — `model_pipeline.py` accepts `4 (single-galaxy)`
processes, i.e. 4 stages × 8 chips via `create_submesh`. It helps throughput and per-chip memory,
not single-token latency.)

---

## 12. What is committed and reusable

**In the model tree** (all default-off / inert):
- `tt/blaze_ops.py` — import-guarded seam; `GLM4_MOE_LITE_BLAZE_QKV_A`, `GLM4_MOE_LITE_BLAZE_Q_STAGE`,
  per-layer PCC gates, weight prep at load time in `layer_weights.py`.
- `blaze_eval/` — A/B harnesses that gate PCC against the real ttnn op:
  `native_boundary_ab.py`, `boundary_decompose.py`, `oproj_residual_ab.py`, `mesh_qkv_a_ab.py`,
  `bench_e_fused_stage_mesh.py`, `bench_e_stage_47.py`.
- `blaze_eval/RESUME_HERE.md` — ordered next steps with commands.

**Blaze-side fixes, as patches** (`blaze_eval/upstream/`) — these directories are **untracked** in
the tt-blaze tree, so the patches are the only durable copy:
- `07-tile-row-replicate-chunked-reads.patch`
- `08-dram-streaming-matmul-workers-per-bank.patch`
- `09-gather-row-to-dram-chunked-writes.patch`

**Operational notes:**
- A device hang needs **two** `tt-smi -r` cycles; after one, the control plane may still report
  `Physical chip id 0 not found in control plane chip mapping`.
- To triage a hang, poll for a frozen log + `pcpu==0` and **do not kill the process** —
  `triage.py --run=dump_callstacks` needs the live inspector RPC.
- Check `df -h /` first; repeated JIT compiles filled the root filesystem (28 GB in
  `~/.cache/tt-metal-cache`) and the bench guard aborts below 2 GB free.

---

## 13. Bottom line

The Blaze work produced a genuinely faster op — `GLMQKVAProjection` went **117.3 → 36.8 µs**,
from 0.40× to **1.39×** against the ttnn op it replaces, correct on all 32 Galaxy chips — via three
real fixes to shared Blaze code.

**It is not enabled, because enabling it makes the model 6.4% slower.**

The gap between those two sentences is the main result: on this model, under trace, kernel-time
wins do not translate into step-time wins, and can invert. Any future Blaze work here should be
gated on a traced end-to-end measurement from the start, and should target stages large enough that
the ~40 µs model-boundary crossing is amortised — or should target the weight-bandwidth problem
directly, where the numbers are much larger.
