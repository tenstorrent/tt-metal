# DSV4.1-Flash decode: how far can persistent / megakernel-style kernels go on BH Galaxy? (scoping + stage-A prototype)

Date 2026-10-03. Author: agent h47p. Device runs: host 10.82.97.45, overlay `/mnt/tt-data/ssinghal/wt/h47p` (diff: `/mnt/tt-data/ssinghal/wt/h47p/changes.diff`), logs `/mnt/tt-data/ssinghal/dsv4-logs/h47p_*.log`.
Tags: **[V]** verified (code read, or measured by me on device), **[I]** inferred / estimate.
Related: `2026-10-03-dsv41-gpu-fusion-research.md` (what the GPU stacks fuse), `2026-10-03-dsv41-dram-overlap-research.md` (sub-devices / prefetch).

## 0. Summary

* A cross-core, in-program synchronised kernel works on our stack today with plain `ttnn.generic_op`: per-role `KernelDescriptor`s on disjoint core sets, `ttnn.SemaphoreDescriptor`, direct NoC writes into another core's L1, `noc_semaphore_inc/wait`. No b1 infrastructure is needed for that [V].
* **Prototype (stage A):** mHC mixes (projection on 40..80 cores + post/Sinkhorn on 1 extra core) as ONE program. Bit-identical to the 2-program path (max |diff| = 0.0 on pre/post/comb, T=4/16/32, random + realistic inputs, plus 3 consecutive launches), PCC vs torch reference >= 0.999995. **Gain is small: 26.7 -> 24.3 us at T=4 (-2.4 us), 49.6 -> 47.6 (T=16), 91.7 -> 87.7 (T=32)** [V]. At 2 calls/layer that is ~5 us/layer = ~0.2 ms/token (0.3%).
* The measured program-boundary cost in a trace is therefore only **~2-4 us**, not tens of us. A megakernel that merely removes launches is a weak lever here. The levers that matter are (a) what happens inside the programs (DRAM-bound weight read, serial Sinkhorn) and (b) **concurrency**: mHC `mixes` outputs are not needed until a whole sub-block later (see 3), so they could be hidden entirely if something else ran at the same time.
* Stage D (expand fused with the next mixes, one program) is **slower** than separate programs at T=4/16 (54.3 vs 44.1 us; 117.7 vs 103.4), even though the megafied variant beats h45m's two-program EP (56.6 / 121.7) by 2.3 / 4 us [V]. Do not pursue as built.
* b1 (`deepseek_v3_b1`) is not reusable as is: M=1 single user, 7168-hidden, no fp32 CBs, CCL ops hard-wired to 2-device / 4-ring axes, persistent mode needs slow-dispatch + socket token injection, incompatible with our traced fast-dispatch loop and `moe_compute`.

## 1. What exists (study notes)

**b1 pattern [V, read by a research sub-agent; refs relative to `models/demos/deepseek_v3_b1`]**
* `UnifiedKernelDescriptor` (`unified_kernel_descriptor.py`): one `.cpp` compiled for NCRISC/BRISC/TRISC, per-core role flags as named compile-time args (`if constexpr` kills dead roles), one kernel group per role combination. `fused_ops/decoder_block/decoder_block_kernel.cpp` (~3800 lines) = MLA + MoE for ONE user in one program over ~129 cores, assembled as a `MeshProgramDescriptor` (one program per chip).
* Sub-ops chain through CBs, mcast/gather and semaphores. CB ids (limit 64) are reused across attention and MoE phases with `reconfig_cb_interfaces` (rewrites CB config from an L1 tensor).
* Persistence: device-side loop (`persistent_loop.hpp`), gated per iteration by a semaphore on the broadcast root, terminated by host writing a termination semaphore + one dummy token. Needs `TT_METAL_SLOW_DISPATCH_MODE=1` (`demo/cli.py`, `model_pipeline.py:66`); tokens enter via sockets/d2d. Decoder-block tests use plain non-persistent `generic_op`; trace appears only in CCL/LM-head benchmarks.
* Sync primitives: `noc_semaphore_inc/wait/set`; consumer resets after wait; separate semaphores per concurrent use (MoE uses 23; sharing across back-to-back rounds loses increments). Global semaphores (`ttnn.create_global_semaphore`) for cross-device.
* CCL in program: `ttnn.setup_fabric_connection(...)` per device program, dedicated fabric worker cores (row 9 / cols 11-12 of the 13x10 grid), `WorkerToFabricEdmSender` in-kernel. `ccl_all_reduce`: exactly 2 devices on the axis, 1-2 links. `ccl_all_gather`: ring of exactly 4, 1 link, torus. `ccl_broadcast`: general 1-hop tree (8x4 example).
* Gate: `deepseek_moe_gate.hpp` is 256 experts / 8 groups / top-8 on one 16x16 face. The repo's `generalized_moe_gate` (your uncommitted edits) supports 256-multiples (`num_blocks`), topk 4/6/8.

**Ours [V]:** `tt/mhc_*.py`, `tt/mhc_kernels/*.cpp` (h45m): proj2 (x@fn^T + sum of squares), post2 (sum partials, RMS scale, sigmoid, SFPU Sinkhorn), collapse+norm CN (already 32 cores synchronising through L1 slot writes + flag words, 30 -> 16 us), expand2, EP (expand + next projection). These are one-or-two programs each, DRAM in/out between them.

## 2. Blockers (our case)

| # | Blocker | Tag | Evidence / consequence |
|---|---|---|---|
| 1 | b1 ops assume M=1 (one tile row), 7168 hidden, single user; matmul asserts `Mt == 1`, no fp32 circular buffers, flash_mla is MLA (512 nope + 64 rope), RoutedExpert K=7168 and 112-core o_proj grids baked in | V (asserts) / I (porting cost) | Our T=4..32 rows, hidden 5120, 4 fp32 residual streams, head_dim 512, GQA-style 8 heads/chip need new dataflow, not a flag. b1 is a pattern library, not code to reuse |
| 2 | Persistent device loop needs slow dispatch + socket token injection; we run fast-dispatch traces with host-visible per-step state (positions, KV paging, engram) | V (b1) / I (conflict) | A never-returning program on the CQ cannot coexist with traced ops around it. Per-layer or per-sub-block programs launched inside our trace are the compatible unit |
| 3 | Cross-core sync inside ONE program | V (works) | Prototype: sender NoC-writes into receiver L1 then `noc_semaphore_inc`; receiver waits, resets. Needs: remote L1 address known on senders (solved by declaring the receiver's CB on ALL cores, same id => same address), physical NoC coords (`worker_core_from_logical_core`), an acked write barrier before the inc (cost ~1 us, could be relaxed), distinct CB ids per role (we use 0-13 and 16-25). 300+ launches incl. 50-replay traced chains, no hang or stale data |
| 4 | CCL/fabric from inside a fused program | V (b1 limits) / I (4x8) | b1 all_reduce (2 devices), all_gather (ring 4 + torus) do not fit a 4x8 mesh with FABRIC_1D_RING; only the broadcast tree generalises. Needs per-device ProgramDescriptors (MeshProgramDescriptor), fabric worker cores, `setup_fabric_connection`; our replicated `generic_op` helpers build ONE program for all chips. Large work, high hang risk; none of our fused kernels needs it today (collectives stay ttnn ops) |
| 5 | L1 budget | V | `moe_compute` keeps ~650 KB/bank of L1 outputs + a persistent global semaphore alive (see `tt/moe_block.py` warmup); static CB region of every other program is capped ~770 KB, and SDPA decode at head_dim 512 already sits near it. A layer-wide megakernel would have to live in that ~770 KB (b1 shrinks `worker_l1_size` to 1.37-1.43 MB and trims SRAM experts to fit). Per-sub-block kernels are fine: mixes mega uses ~145 KB on projection cores + 20 KB (T=4) / 80 KB (T=16/32) partial-page CB on every core in the program |
| 6 | Trace / hazard behaviour | V (ours) / I (rest) | Semaphore reset-by-consumer is trace-safe (verified chain of 3 launches per trace, 50 replays). Program hash must cover every tensor address/accessor arg (`custom_program_hash`), otherwise stale-address reuse. A killed run can leave non-zero semaphores; b1 alternates semaphores across traced iterations [V]. I did not verify that a stale semaphore wedges the next run (I) |
| 7 | `moe_compute` is a monolithic C++ op with its own ring / dispatch / persistent semaphore | V (L1 footprint) / I (internals) | It already is the "Mega-MoE" analogue; we cannot merge our kernels into it, only fuse at its edges (router output -> dispatch input, combine -> expand epilogue). Anything persistent that spans it must leave its L1 alone |
| 8 | Hang / debug cost | V | A lost semaphore = device wedged until reset (reset needs permission). The CN kernel uses bounded spins; my post reader's `noc_semaphore_wait` is unbounded (should be bounded before use in the model). Watcher unsupported for some b1 ops on BH (flash_mla skip); DPRINT only; device profiler zones worked well (section 4) |
| 9 | Programming-model gaps | V | `generic_op` has no role DSL: one KernelDescriptor per (kernel, core set), CB ids manual, per-role compute kernels must not share CB ids (had to shift ids by 16, copied the post compute kernel); no cross-phase CB reuse helper outside b1's id manager; every new kernel is a JIT compile and a new bit-exactness problem (fp32 HiFi4 matmul-as-reduce tricks) |
| 10 | No concurrency inside a command queue | V (dram-overlap doc, sub-device spike showed gaps) | Independent work (mixes, shared expert, indexer) cannot overlap the critical path unless it lives in the same kernel or on sub-devices. This, not launch count, is the megakernel's real prize here |
| 11 | mHC projection is DRAM-bound by its own padded weights | V | fn^T is stored fp32 and padded 24 -> 32 columns: 2.6 MB per mixes call (roofline counts 1.97 MB). At ~500 GB/s that is ~5 us of the 24 us; weight reads are still being issued at 7.2 us (profile below) |

## 3. Dependency structure that decides what a megakernel can hide [V from `tt/layer.py`]

`mixes(x)` yields `pre` (consumed by the NEXT sub-block's collapse), `post`/`comb` (consumed by THIS sub-block's expand, after attention or MoE, hundreds of us later). So the whole mixes chain (24 us x 2 per layer at T=4) is off the critical path in principle; collapse+norm (16 us) and expand are on it. Only a design with real concurrency (sub-device, or a single program with mixes roles on spare cores) can exploit that [I].

## 4. Prototype measurements (T per device, 32 identical chips, traced chain timing = (3 calls - 1 call)/2)

Files: `tt/mhc_mega.py`, `tt/mhc_kernels/mhc_mega_{proj_writer,post_reader,post_compute,ep_writer}.cpp`, flags `DSV41_MHC_MIXES_MEGA=1`, `DSV41_MHC_EP_MEGA=1` in `tt/mhc.py`; tests `tests/test_mhc_mega.py`, `test_mhc_mega_ep.py`, `test_mhc_mega_prof.py` + `parse_zones.py`. Base = snapshot of h45m's mhc files as of 03:42 (copied into the overlay; nothing edited in h45m).

| call | T=4 | T=16 | T=32 |
|---|---|---|---|
| mixes v1 (proj+post, older) | 68.4 | 117.4 | 184.2 |
| mixes v2, 2 programs (h45m) | 26.7 | 49.6 | 91.7 |
| **mixes mega, 1 program** | **24.3** | **47.6** | **87.7** |
| expand2 + mixes v2 (separate) | 44.1 | 103.4 | - |
| EP (h45m, expand+proj, then post2) | 56.6 | 121.7 | - |
| EP_MEGA (1 program) | 54.3 | 117.7 | - |

(us per call.) PCC vs reference: pre/post/comb 0.999995-1.0 (all cases), x_new 0.999999+.

**Where a mega-mixes call spends its 24 us (device profiler, T=4, device 0, us from first event):** weight/activation reads issued until 7.2 us; reads landed + matmuls done ~11-12.7 us; last partial sent 13.4; post core starts 14.0; raw/mixes/pre-post done 17.5; Sinkhorn + permute + comb write finish 23.9. So ~11 us DRAM-bound projection, ~3.5 us post prologue, ~6 us serial single-core Sinkhorn. Launch/boundary overhead is the small part.

## 5. Staged plan and expected saving (us per layer, T=4; savings counted on the critical path)

Cost model: ~2.5 us per removed program boundary [V from stage A], plus whatever the fusion removes inside.

| Stage | Content | Status | Saving / layer | Notes |
|---|---|---|---|---|
| A | mHC mixes in 1 program | DONE, correct | **~5 (2 calls)** | Productise: bounded spin in post reader; drop barrier before inc if ordering verified |
| A+ | Cut mixes weight traffic (fn in bf16/bfp8 or unpadded 24 cols), spread Sinkhorn over more lanes/cores | I, accuracy-gated | 2 x 3-5 = 6-10 | DRAM 2.6 MB -> 1.3 MB; Sinkhorn 6 us is serial; mHC is fp32 on purpose, needs a PCC/error-budget run |
| B | Attention front end: norm + q_a/kv matmuls + `attn_pre` | I | 10-20 | Eliminates ~4-6 boundaries (2.5 us each) + prologue overlap; bounded by DRAM streaming of weights (63 us floor vs 235 us); needs DRAM-streaming matmul at M=4 (b1 `dram_streaming_matmul` is M=1) |
| C | Router + shared expert prologue | I | 10-20 | router 40 -> ~20; 384 experts is not a multiple of 256: needs padded 512-block or a custom gate; shared expert overlap with moe_compute is the bigger win (dram-overlap doc, ~4 ms/token) and needs concurrency, not fusion |
| D | expand -> next mixes in one program | tried, **negative** (+10 us at T=4) | 0 / negative | Expand and projection serialise on the same cores; needs a rebalanced core split before it can win |
| E | Overlap mixes (and other off-critical-path work) with attention/MoE | I | up to ~48 (2 x 24) | The real megakernel prize; requires sub-devices in a trace or one program holding both; not attempted here |

Realistic total for A-C: ~25-45 us/layer = 1-1.8 ms/token of 58 (2-3%). E could double that. Nothing here approaches the 26.9 ms floor; the floor gap is dominated by per-op latency at M=4, which fusion shrinks only a few us at a time.

## 6. Recommendation

1. Merge stage A behind its flag after adding a bounded wait; keep stage D off.
2. Before more fusion, measure/attack inside-the-program costs: weight bytes (A+) and the serial Sinkhorn.
3. Pursue concurrency (E, shared expert overlap) via sub-devices; it addresses blockers 2, 5, 10 at once, which fusion cannot.
4. Do not port b1 kernels: shape (M=1, 7168), dtype (no fp32) and CCL (2-device / 4-ring) assumptions all fail; copy its patterns (role flags, semaphore hygiene, CB id reuse) instead.
