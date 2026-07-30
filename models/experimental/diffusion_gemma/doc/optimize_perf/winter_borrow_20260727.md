# Winter diff: what was borrowed, what was refused, and why the token-gather MoE died (2026-07-27)

Status: current for the concat MoE, the reference bars and the traps; provenance-only for every
§§1–12 absolute ms/step (host Gumbel, deleted 2026-07-28 — convert with the 1.94x ratio below).
Owns: why the token-gather denoise MoE was deleted, the concat-MoE mechanism and cost, and the
fused-MoE-kernel infeasibility verdict plus the two landed `sparse_matmul` gates (absorbed from the
deleted `fused_moe_kernel.md`).
See also: [refuted list](../REFUTED.md), [optimize_perf hub](README.md).
Well over the 100-line cap on purpose: ~12 measurement traps, 3 repro pins, 2 open contradictions and
the arithmetic that killed a whole MoE path. None of those is cut for length.

An independent TTNN implementation of this model (`winter/`, 2026-06-15, ~3.3k lines, 4xP150 1x4 mesh)
was diffed line-by-line against this branch. Every number below is ours.

## Why the token-gather denoise MoE was deleted

At the shipped `C = S = 256` (`E=128, H=2816, I_dev=192`), per layer per device:

| | token-gather path | concat-experts |
|---|---:|---:|
| expert gate+up+down MACs | 5.31e10 | 5.31e10 (**identical**) |
| routing / dispatch MACs | **4.72e10** (gather + combine) | 8.05e8 (one expand matmul) |
| **total** | **1.00e11 (+89%)** | 5.39e10 |
| activation DRAM | ~900 MiB (two ~184 MiB intermediates written **and** read) | ~40 MiB |

At `C = S` the gather cannot save any expert work — the gathered `[1,E,C,H]` is **~94% zero rows**, so
the dispatch is pure overhead. Deleted 2026-07-29 (`7417bd7d69d`).

**TRAP — the win was measured against the wrong baseline.** "~5x faster/step than dense-128" and
"10.54 ms/layer (13.0x vs dense 137.6)" used gemma4's `PREFILL_CHUNK_SIZE=32` serial per-expert
`sparse_matmul` path as the denominator, not a well-configured dense matmul, and the numerator was
measured at the capacity that dropped most of the routing.

**THE INERT-CONFIG BUG.** `746cfe53cb6` (2026-07-15) moved the MoE capacity default from 32 to the
canvas length because 32 was silently discarding 41–84% of active routes per layer; the same commit gave
the tuned program configs a `C == DEFAULT_CAPACITY` (=32) condition. Production passes 256, so for **12
days** every MoE matmul ran `program_config=None` while the docstring and README advertised the tuned
geometry as the default. The condition existed for a real reason: at C=256 the down matmul's per-core
output block is `per_core_M x per_core_N = 8 x 88` tiles = **2.9 MB** against ~1.4 MB usable L1, so the
tuned config was **ILLEGAL**, not merely suboptimal — the old `_pick_in0_block_w` modelled only the in1
block against a flat 176-tile budget, with no in0 CB, no output CB and no partials. (`DG_SPARSE_MOE`,
`DG_SPARSE_MOE_TUNED`, `DG_MOE_CONCAT` and `DG_ROPE_FUSED` were deleted with that path; dead flag names
are listed once, in [flag triage](flag_triage_20260728.md).)

## The concat MoE — now the only denoise MoE

Relayout gate/up to `[1,1,H,E*I]` and down to `[1,1,E*I,H]` once, then

    g    = geglu(x @ gate_cat, x @ up_cat)
    rexp = routing @ expand      # expand = repeat_interleave(I(E), I), a static [1,1,E,E*I]
    out  = (g * rexp) @ down_cat

The down fold is exact by linearity — `sum_e W_down_e @ (r_e * g_e) == (r ⊙ g) @ down_cat` — and never
materializes the `[1,E,S,H]` per-expert output. `apply_geglu` is **ours** (tanh GeLU), deliberately not
copied from winter, which uses `fast_and_approximate` GeLU in the shared MLP and erf-GeLU in the
self-conditioning gate; both disagree with this checkpoint.

**Memory.** `gate_cat` and `up_cat` are a second copy of those weights at 132 MiB each = 264 MiB per
layer per device at bf16 = **~7.7 GiB over 30 layers** (measured 7.773); the originals cannot be freed
because prefill still runs the ragged top-8 path over them. `down_cat` is **free** (same byte order at
bf16 TILE) — a view, which is what makes the total 7.7 and not 11.6 GiB, and
`verify_down_concat_is_free` checks that on device instead of assuming it.

**SCOPE WARNING: not denoise-only.** The batched commit runs the same layer body through the same
`_denoise_moe_forward` seam (`tt/commit_batched.py:703`) and is the shipped default; commit hidden
states build the committed-prefix KV, so it compounds across blocks.

**Measured** (30L, `reveal_pmax` 4096, 48 forced steps, 4 GiB trace region, 2 reps): token-gather
**31.737 s/block (661 ms/step)** vs concat **22.234 s/block (463 ms/step) = −29.9%**; against the
morning production path (34.642 s/block) that is **−35.8%, a 1.56x speedup, 722 → 463 ms/step**. Free
DRAM went 27.87 → 14.41 (model) → **4.93 GiB** (concat), confirming the 7.8 GiB relayout on device.
**It serves:** `run_upfront_gpqa.sh smoke` with device Gumbel in thinking mode gave 2/2 exact match,
block latency 5.38 s / 7.79 s, denoise 15 and 26 steps, commit 2.02 s — no statistical weight, but it
proves the path is wired, not just the microbenchmark.

## Measurement traps (the reusable part of this file)

1. **A halt that fires invalidates a step sweep.** At the shipped `entropy_stop_threshold = 0.005` the
   halt fired at `denoise_steps_per_block = [9, 2, 2]`, so a nominal 48-step sweep timed mostly the fixed
   commit cost — and a lever that changes numerics moves where the halt fires, so arms do unequal work.
   `serving_smoke` gained `--entropy-stop-threshold` (negative disables it); every arm must verify
   `[48, 48, 48]`, and the sweep script prints per-arm step counts next to the latency.
   > **OPEN CONTRADICTION (unexplained):** early halt is described as "a no-op under #48291, so steps are
   > not reduced" ([op profile](whole_gen_opprofile/README.md)) yet was measured here firing at
   > `[9, 2, 2]`; other readings are `[9,17,2]/48` and `K=10–43` ([refuted list](../REFUTED.md)). Not explained.
2. **Never quote per-step and per-block interchangeably.** At the shipped halt a block is roughly 55%
   denoise / 45% commit (2.38 s/block at ~660 ms/step plus ~1.0 s commit), so −8.8%/step becomes roughly
   −6% on a block that halts at 2 steps.
3. **Every absolute ms/step in §§1–12 is host-Gumbel, ~2x the served one.** The directly measured
   host→device ratio is **1.94x** (the same concat configuration is 22.234 s/block host, 11.436 device).
   Host Gumbel was deleted 2026-07-28 and `sweep_denoise_arms.sh` now defaults to device, so those arms
   are no longer re-runnable as written.
4. **ms/step is span-specific.** The ~428 ms in earlier docs was taken at `reveal_pmax=1024` (1280 key
   rows/layer); this sweep runs 4096 (4352 key rows).
5. **A 2L/6L extrapolation would have missed the concat win entirely** — the delta reads −5.5% at 2
   layers and −5.4% at 6, implying ~−3% at 30; the real answer is −30%. At small layer counts the fixed
   terminal and self-conditioning cost dominates and DRAM/L1 pressure differs. **Measure a lever at the
   depth it ships at.**
6. **A flag advertised in a docstring is not evidence it does anything** — three silent no-op flags were
   found in one day: the `C == DEFAULT_CAPACITY` MoE gate, the unconditional `tt-smi` requirement, and
   `DG_TERMINAL_SHARDED`.
7. **`DG_SKIP` output is garbage by construction** (it replaces a component with a shape-preserving
   `ttnn.mul(x, 0.0)` at its seam so the rest of the graph is untouched), so a `DG_SKIP` run must never
   feed a `committed_sha256` comparison. Live, measurement-only.
8. **A synchronizing profiler over-predicts.** Winter's tier-2 `_pmark` calls `ttnn.synchronize_device`
   at every stage boundary, draining dispatch — which is why winter's ~570 ms serial profile
   over-predicts its ~150 ms traced step by 3.8x. `prof_step_breakdown.py` is async-pipelined with one
   final sync and lands within ~8% of traced.
9. **Winter's ~0.15 s warm traced step has NO committed measurement artifact** in that tree — it is a
   README assertion, and the "200–350 tok/s" figure in circulation is not supported by it (0.15 s/step
   gives 107–213 tok/s at 8–16 steps; 350 tok/s at 8 steps needs 91 ms/step). The earlier impression of a
   4–10x gap came from comparing our **served** number (which includes a 2.02 s commit per block) against
   winter's **denoise-only** arithmetic.
10. **A commit citation in this file's own header is wrong and the tree disagrees with itself.**

> **OPEN CONTRADICTION (unexplained):** this file's 2026-07-28 header credits the language-drift fix to
> `d0936d4da4f`, which was reverted the same day on a void 44-prompt comparison; the shipped default-ON
> `DG_DENOISE_HIDE_PREFILL_PADS` came from `205e87956cc`. Both SHAs exist in history and the tree states
> both. Not explained.

## Where the step goes (`DG_SKIP`; 30L, device Gumbel, concat, 48 forced steps, 4 GiB, 2 reps)

| arm | steady s/block | ms/step | component | share |
|---|---|---|---|---|
| `full` | 11.429 | 238.1 | — | — |
| `DG_SKIP=moe` | 7.805 | 162.6 | **75.5 ms** | 31.7% |
| `DG_SKIP=attn` | 10.006 | 208.5 | 29.6 ms | 12.4% |
| `DG_SKIP=shared` | 10.993 | 229.0 | 9.1 ms | 3.8% |
| *(residual — in no seam at all)* | | | **124.0 ms** | **52.1%** |

**HEADLINE: 52.1% of the step is in no seam at all** — per-layer norms and residual adds, embed,
self-conditioning, lm_head, the terminal sampler and CCL. Layer-level matmul work is no longer where the
time is. The full-canvas norm measured **238.3 → 195.7 ms/step (−17.9%)** in this harness, a different
configuration from its other two readings — [l1 residency](l1_residency.md).

**PREFIX-SPAN SWEEP** (`NUM_BLOCKS=2`, full-canvas norm on, sweeping `DG_DENOISE_REVEAL_PMAX`): 576 →
8.724 s/block (181.8 ms/step); 1024 → 8.819 (183.7, +1.1%); 2048 → 9.043 (188.4, +3.7%); 4096 → 9.446
(196.8, +8.3%). All four arms produced **identical committed tokens** — the span only extends the
masked-out region.

## Landed defaults from this diff

- **Unpinning the denoise SDPA from an 8-core grid is −8.8% traced block with the SAME committed
  tokens** — reassigning the Q axis across cores does not touch the flash K-reduction (`k_chunk_size`
  unchanged); corroborated by the 2026-07-24 q-chunk sweep over 6 runs and again here at a 4x larger
  prefix span. Arms (3 reps, within-arm spread ≤1.5%, ordering held in every rep): `auto` 34.642 s/block
  (722 ms/step, sha `304e8023…`), `tuned` 34.369 (−0.8%, 716, `2ac3efcc…`), `+ device SDPA grid` 31.586
  (**−8.8%**, 658, `2ac3efcc…`).
- **Encoder `layer_scalar` guard.** Winter loads a separate encoder copy
  (`winter/tt_model.py:260-266`), but measured on `diffusiongemma-26B-A4B-it` **max |encoder − decoder|
  = 0.0 across all 30 layers**, and the `model.encoder.` prefix holds nothing else on the text path (its
  other 356 keys are vision tower / embed_vision). `checkpoint.validate_encoder_layer_scalar_tie` now
  runs on every load and raises if a future checkpoint diverges, so a wrong per-layer scalar compounding
  into the prompt KV cannot become a silent correctness bug.

## Trace region — the 12 GiB reservation was ~8 GiB of unusable DRAM

`bisect_trace_region.sh` (30L, `reveal_pmax` 4096): 12 GiB OK with 4.702 GiB free, 8 GiB OK with 8.702,
6 GiB OK with 10.702, 4 GiB OK with 12.702, and **3 GiB FAILS** with `TT_FATAL: Creating trace buffers of
size 3259146240B ... but only 3221225472B is allocated`. So the 48 up-front traces need **3.04 GiB**, not
the 1.41–1.44 GiB `doc/vllm_integration/traced_serving.md` claimed, and the free pool tracks the
reservation one-for-one. As of `d0551c78bda` (2026-07-29) every sweep/verify script and all five gate
arms default to **4 GiB**; only `run_upfront_gpqa.sh` still carries a 6 GiB default.

**The "re-run the bisect whenever `reveal_pmax` changes" advice is MEASURED FALSE** and was removed from
the scripts: all 48 up-front traces were captured inside a 4 GiB region at `reveal_pmax=16384`
(MeshTraceId 0..47, run `local100_trace4g`, 2026-07-29), where proportional growth from 3.04 GiB at 4096
would have demanded roughly 12 GiB.

## `ttnn.topk` width cliff

Measured on QB2 and reproducing exactly (`tests/test_device_topk_width_cliff.py`, 7 passed): index
agreement versus torch is **0.129 at shard width 16384, 0.129 at 32768, 1.000 at 49152, 1.000 at 65536**.
V=262144 over tp=8 is exactly 32768, so a Galaxy 4x8 bring-up lands on the cliff; at tp=4 (what we serve)
the width is 65536 and `argmax_last_dim` is exact. `ttnn.max` stayed finite and correct at every width,
which is why winter's workaround (pad to 49152 with `-inf` for the *index*, take the *value* from `max`)
is the right shape if tp=8 is ever needed. **TEST-DESIGN TRAP:** the router arm initially failed at
0.9961 set overlap on plain random input — that was **tie-breaking, not an op error** (the 8th and 9th
routing values land within a bf16 ulp on some rows); the test now separates the winners by a wide margin
and passes at 1.0000.

## Reference bars

**CUDA**, measured the same day on `ssh a100` (one NVIDIA A100 80GB PCIe, bf16, upstream vLLM, single
stream, same prompt, 768 tokens, default sampler, reference server idle): **0.54–0.62 s per 256-token
block = 410–473 tok/s**, against QB2's ~2.3 s / ~108 tok/s on the shipped early-halt path — TT is ~4x
slower per block on 4 chips, ~17x per chip. **Both caveats must be stated together:** those TT blocks
halt after 2–9 denoise steps and blocks 1–2 emit IDENTICAL committed tokens, so TT is partly fast for the
wrong reason (#48291); and the A100 number is single-stream on an 80 GB part that holds the whole model,
so it is a latency comparison, not a cost or throughput one.

**GPQA**, same reference server, 198 samples, flexible-extract, thinking at 262k: **70.71%** and
**70.20%** across two repetitions.

> **OPEN CONTRADICTION (unexplained):** this document states the reference run-to-run spread as **0.5 pp
> across those two repetitions** (§11) and as **~1.1 pp over three repetitions** (§12), giving a
> resolvable-difference floor of ~1–2 pp. Both readings kept; not explained.

**Normalized against winter's operating point** (30L, canvas 256, device Gumbel): production that morning
~371 ms/step (derived, 34.642 s/block host ÷ 1.94), + SDPA grid + concat MoE 238.3, + full-canvas norm
195.7, + prefix span at winter's geometry (`reveal_pmax` 384–576) ~181. On winter's own accounting (256
tokens per canvas, prefix excluded) we are at **177 tok/s vs winter's 213 at 8 steps, 141 vs 171 at 10,
110 vs 107 at 16** — 0.83x winter's documented step time at 8 steps and slightly ahead at 16, at bf16
where winter runs bf4 experts and bf8 attention.

**OPEN:** the **2.02 s/block commit** is 26–38% of a served block and no lever here touches it; winter
avoids it only by re-encoding the whole prefix every canvas, which is O(P²) and worse for us.

## Fused per-layer MoE kernel — infeasibility verdict, and what is still in the tree

The design targeted a deleted subject (the token-gather MoE), but three parts outlive it.

**REFUTED — an in-reader per-row gather from a TILE-layout hidden cannot be landed as a working, faster
gather.** `hidden` is forced TILE (`sparse_matmul_device_operation.cpp:87`); logical row `t` lives at
intra-tile row `t%32` of tile-row `t/32` and spans two horizontal 16x16 faces = two discontiguous 32-byte
runs 256 elements apart. One dest tile-column of one expert therefore costs `32 rows x 2 face-runs = 64`
sub-tile reads versus **1** tile read today, and over `H/32=88` tile-cols that is **~5632 reads/expert vs
88** — a ~64x NoC blow-up on a movement/op-count-bound step. The only variant that reduces movement is a
**ROW-MAJOR hidden plus on-the-fly tilize** (row `t` becomes one contiguous `H*BPE` read, 32
reads/expert), but the matmul reader has no tilize stage — `ttnn.embedding(..., layout=TILE)` already
does exactly this in the ragged prefill path, and folding it into the matmul reader is the multi-week
body. Precedents for why op-count is the enemy: a RoPE-unchunk removing ~128 tiny ops gave **+34%**, a
single-op `transpose_a` tweak gave **0**, and a compact embedding gather measured **~0 or slower**.

**GATHER ADDRESS MATH, preserved for any future attempt** (`BPE=2` for bf16):
`page_id = (t/32)*(H/32) + tile_col`; `face_row = (t%32)/16`, `r16 = (t%32)%16`;
`run0 = (face_row*2 + 0)*256*BPE + r16*16*BPE`, `run1 = (face_row*2 + 1)*256*BPE + r16*16*BPE`.

**THE CROSS-CORE SCATTER-ACCUMULATE HAZARD (a hardware fact that outlives the deleted MoE):** the NoC has
**no read-modify-add primitive**, so a token receiving up to `top_k=8` contributions placed on different
cores cannot be accumulated by concurrent `+=` — accumulation must be *structured*. Three structures,
cheapest first: (1) keep combine as a reduction over the compact route-weighted output using `embedding`
+ `fast_reduce_nc`; (2) home-core reduction with a per-(token, k-slot) scratch page nobody else owns;
(3) serialize by semaphore — **REJECTED**, it serializes the hot path.

**REVERTED 2026-07-30 — two ttnn env gates that no longer exist.** Both were scaffolds for the
fused-MoE experiment this doc records as refuted, both lived in the shared `ttnn/cpp` tree, and neither
had a caller: `TTNN_SPARSE_MATMUL_WRITER_SCALE` (a `WRITER_SCALE` define into the shared sparse writer
kernel `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`, scaling every output tile of an active
batch by the bf16 value already in the `cb_sparsity` L1 page) and `TTNN_SPARSE_MATMUL_IN0_GATHER` (a
`SPARSE_MATMUL_IN0_GATHER` define plus an in0-reader hook whose `#else` path was textually identical to
the pre-scaffold read, i.e. a no-op on or off). They went out with the rest of the out-of-folder
changes; recover from `af08af2c304` and `9f3f558319d` respectively. Their tests
(`test_sparse_matmul_writer_scale`, `::test_sparse_matmul_in0_gather_scaffold`,
`::test_sparse_matmul_in0_gather_reference`) were deleted in the same commit.

**Two facts from that work worth keeping.** (1) The writer-scale trick is legal **with no new host
tensor**, because the op uses only `== 0` of the sparsity value as an active/skip gate — confirmed by
`test_sparse_matmul_with_nnz`, which puts `torch.rand` values in `sparsity` and compares against a plain
`torch.matmul` with no scale. (2) **PROGRAM-CACHE TRAP:** an env gate read inside a program factory is
**not part of the program hash**, so a run must not reuse a cached program built with the flag in the
opposite state for the same shapes — any future gate of this shape needs a distinct shape per arm, or
the env var set before the first op. Also: these kernels are JIT-compiled on device, so such tests are
host-buildable but **REQUIRE a Tenstorrent device to run** — a `.so` build/link proves nothing.

**DO-NOT-REINVENT** if anyone revives this: the `cb_sparsity` per-batch skip page;
`SPARSE_OUTPUT`/`compact_output`, which packs only the nnz active batch pairs so the output is
`[nnz_rows,H]` not `[EC,H]`; `SparseMatmulMultiCoreReuseMcast1DProgramFactory`; and
`matmul_reduce_scatter_async` as the template for folding the TP all-reduce into the op. **Effort:**
increment 1 was ~2 days and is ~5% of the work; ~6 weeks of ttnn C++/Python remain, the on-kernel gather
and single-op fuse being the bulk. **Owner scope:** `tt/sparse_moe.py`,
`ttnn/cpp/ttnn/operations/matmul/device/sparse/**` plus its quasar mirror, the sparse-matmul unit test,
and this doc — never gemma4 or the denoise/loop/sampling/self-cond/model files.

## Reproduction and artifacts

`sweep_denoise_arms.sh` runs interleaved arms over `demo/serving_smoke.py --upfront` at 30 layers, canvas
256, `reveal_pmax` 4096, `max_seq_len` 4096, seed 0, 3 blocks x 48 denoise steps, 3 repetitions; steady
state = `mean(per_block_latency_s[1:])` because block 0 carries the ~85 s 48-trace capture. Trace-region
bisect: `bisect_trace_region.sh`. Env: see [plan](../../plan.md). The two GPQA arms ran as
`/home/zni/dg_runs/gpqa_{base,concat}_20260728` at `TRACE_REGION_SIZE=4 GiB`, `RESET_BEFORE=0`.
