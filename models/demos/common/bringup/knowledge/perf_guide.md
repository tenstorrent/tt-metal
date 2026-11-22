# Performance guide (prefill bring-ups)

Rules of thumb for picking, building and judging a performance change. Model-agnostic: every number here is an
**example** from one bring-up (a large MoE prefill on a 32-chip mesh, 2026-10/11) to show the order of magnitude,
never a target. The entries named `[K-n]` / `[<MODEL>-n]` are in the shared or a model's `known_issues.md`; read them
for the details. Who reads this: the `perf` agent before it touches code, and whoever picks entries from
`opportunities.md`. The owner's rules (agent rules 14 and 16-18, the skill's perf-pick decisions) rank above it.

## 1. Read the profile before choosing anything

The plain profile and the A/B table give, per chunk (warm):

| metric | what it tells you |
|---|---|
| `device_ms_total`, `device_ms_<section>` | device time, slowest chip per section (real-time profiler). A program's time is start to end, so a CCL or a fused op that waits for a peer is **charged the wait**. |
| `chunk_wall_ms` (+ `_spread`) | the clean wall time (profiler off, proven warm). The spread is the min-max of the timed runs: read it before the mean. |
| `host_overhead_ms` | wall minus device: how long the device waits on the host. |
| `host_launch_calls`, `host_launch_ms`, `profiled_programs` | host dispatches a chunk and their cost; programs = dispatches x chips. |
| `host_cprofile_ms` | the same chunk under cProfile (slower than the wall; use it for ranking host callers, not as the wall). |
| chip spread (opportunities.md) | max - min of a section over chips: load imbalance, or one chip waiting for another. |

**First decide the regime:**

| sign | regime | levers that move the wall |
|---|---|---|
| wall >> device (example: wall 600 ms, device 220 ms) | host-bound | fewer launches, cheaper launches, host work off the main thread, trace (section 3) |
| wall ~ device + a small constant | device-bound | the largest sections: fusion, layout, formats, kernel work (sections 4-8) |
| a device win that does not show in the wall | host-bound in that span, or noise | count launches; check the spread |

A device-time cut on a host-bound chunk is still worth having (it shows once the host shrinks) but does not move the
wall; a launch cut on a device-bound chunk shows only as lower `host_overhead_ms`. A chunk often moves from one regime
to the other during a perf campaign (example: 624 ms wall / 220 ms device at the start, 187 / 176 ms after ~35 picks),
so re-read the regime after every few accepted picks.

**Profile at more than one position** (`ab_positions`, e.g. an early and a long-context chunk): position-dependent work
(attention over the prefix, top-k over all keys, cache writes) and fixed work (projections, experts) shift shares.
A pick can be flat at one position and a clear win at the other (example: host -1.7 ms at 50k, -11.7 ms at 500k).
Time a **non-last** chunk too when the last chunk does extra work (serving copies, snapshots): the steady-state chunk
is what most of a long prompt runs.

## 2. Pick the target

- Rank by device ms (opportunities.md does), then estimate what is **reachable**: the work a change removes, not the
  section's total. A 5 % section that a fusion deletes outright beats a 50 % section already at its bound.
- Before proposing a kernel change, decide what bounds the op (section 5). A compute change to a DRAM-bound pass and
  a program-config sweep of a weight-bandwidth-bound matmul both measure ~0 (example: two full matmul sweeps, each
  < 0.5 ms a chunk on ~170 ms; [DS41-191]).
- On a host-bound chunk rank host callers by launch count x cost a call from a cProfile census
  (`BRINGUP_PROFILE_HOST=1`), not by device time. A census that lists every Python-built op, every allocation and
  every per-chip program usually yields the first several picks ([DS41-171]).
- Write the estimate down (ms a chunk, device and host separately) in the brief / BREADCRUMBS before measuring; it
  makes a result that misses it visible (example: a host bundle estimated at -30 ms delivered -5 to -13 ms).
- One idea per pick where possible. A bundle is fine for small, independent host cleanups (one A/B instead of five),
  never for a change whose accuracy or speed you need to judge on its own.

## 3. Host-side overhead

Host cost scales with **launches x chips x per-core arguments**, not with tensor size. Examples on a 32-chip mesh:
a single-program op 30-80 us, an op built per chip 100-200 us, a Python-built program rebuilt each call ~0.7-1 ms.

| lever | when it pays | example |
|---|---|---|
| Prepared programs for every Python-built op (`ttnn.bringup.prepare_program` / `run_prepared`; owner rule) | always; a `generic_op` / descriptor op that rebuilds per call | -79 ms and -50 ms wall a chunk for two sites |
| One program for the mesh instead of one per chip | the per-chip difference is a constant the kernel can read (its chip id from the routing table, a common-arg table) | 101 -> 52 us, 336 -> 110 us a call ([DS41-147], [DS41-225]) |
| Common runtime args instead of per-core args | many cores, same values | 150-210 -> ~60 us a call ([DS41-145]) |
| Outputs written into buffers allocated at load | fixed-shape intermediates; each output alloc + free costs tens of us on a big mesh | ~1/3 of a fork's host time, -13.5 ms a chunk ([DS41-172], [DS41-174]) |
| Fewer ops (fusion, folding typecasts / copies into producers) | every removed launch saves its host cost even when the device time does not move | -89 to -296 launches, -9 to -35 ms host a chunk |
| Host work off the critical path (a worker thread, prefetch) | host prep that does not depend on the previous layer; keep the GIL released (no `.numpy().tobytes()` copies on the worker) | -40 ms wall; GIL trap [DS41-136] |
| Settings read once at load | per-call `os.environ` / config reads in the forward | small, but free |
| Trace | after the rest: only it removes the per-launch floor. Capture after warm runs add no program; buffers must not move ([DS41-100]) | example: 104 -> 87 ms a chunk |

Traps:
- **Address-keyed program caches.** A cache keyed on buffer addresses misses when any earlier allocation moves (a
  persistent buffer allocated mid-forward is enough): rebuilds then cost more than the change saved (example: warm
  chunk 426 -> 612 ms). Key on shape and patch addresses; allocate persistent buffers at load ([DS41-146]).
- **Many cached programs can be slower than one patched program** (example: one program per layer, 40 x 32-chip
  programs, was slower than patching 7500 addresses into one; [DS41-131]). Measure before splitting a shared program.
- A descriptor op that takes its output in `tensor_args` loses the binding fast path ([DS41-187]).
- A small, short tail chunk is host-bound even when its device time drops (example: device -20 %, wall -3 %;
  [DS41-205]).

## 4. Fusion: what usually pays

Fusion pays through three things; know which one you are buying:
1. **Launches** (host time and per-program fixed cost on the device, typically a few us each).
2. **DRAM round-trips** of intermediates (the big one for elementwise / norm / residual chains).
3. **Collectives removed or overlapped.**

Usually pays:
- Chains of small elementwise / rotary / norm ops on the same rows into one program (example: two rotary chains as
  single fork ops, -12 to -16 ms device and -296 launches a chunk).
- A norm (or scale, cast, bias) folded into the pass that already reads or writes those rows: the producer writes
  its consumer's input in the consumer's dtype and buffer (removes a typecast and a round-trip; [DS41-184]).
- The consumer's first step folded into the producer's epilogue when the producer already holds the values in
  DEST / L1 (example: the next step's inputs written by the residual pass, 79 launches a chunk removed).
- Packing / compressing on the producer chip before a gather (fewer bytes on the fabric; example: gather traffic
  183 -> 104 MB a chunk a chip).

Loses or is neutral when:
- **The fused pass is DRAM-bound.** Removing compute from a bandwidth-bound pass saves nothing (example: removing a
  fused kernel's matmul / SFPU work saved <= 0.01 ms; its reads were 0.09 ms; [DS41-80]). What helps there is fewer
  bytes (narrower formats, one read of a row instead of two) and batched reads behind one barrier.
- **The absorbing kernel gets a serial step on its critical path.** Data-movement RISC-V code is slow on scalar
  work (example: a 32x32 fp32 word-copy transpose 8 us; per-value branches made a writer 4x slower; [DS41-222],
  [DS41-85]). A fused norm first cost +27 us a call where the op it replaced cost 19 us; it paid only after the
  copies were moved to the SFPU or shrunk. Profile the fused kernel's zones before deciding it does not pay.
- **A collective fused into a compute op moves the wait.** The fused op now ends when the slowest peer's data lands,
  so the skew moves into its section (example: experts -3.6 ms, the producer's section +1.5 ms, net -2.2 ms).
  Judge on the **total**, not on the section you targeted.
- **The fusion changes the op's program mode.** An option can switch a fork to a slower path (example: an output
  dtype option moved a norm to a one-core-per-row mode, 2.6 -> 5.6 ms; [DS41-186]). Read the fork's README Limits.
- The removed op was already cheap and its launches were not on a host-bound span.

Correctness traps when fusing (all seen in practice):
- On a program-cache hit, per-core runtime args read back padded to the longest list: put shared values in common
  args or fixed slots before any variable-length list ([DS41-210]).
- Stock SFPU inits share programmable constants (sqrt-family vs reciprocal): phase the epilogue and re-init
  ([DS41-190]).
- A core that skips the main loop keeps the hardware-startup data formats: reconfigure before every phase a core
  can enter cold ([DS41-223]).
- A fused op is not required to be bit-identical to the old path (agent rule 14), only within the accuracy limits
  and deterministic; say in BREADCRUMBS which bits change and why.

## 5. Bound-ness: DRAM, compute, RISC, host

| bound | typical case | how to tell | what helps |
|---|---|---|---|
| DRAM / weight bandwidth | small M per chip (a few hundred rows or less) x large weights; streaming eltwise passes; LoFi / low-bit matmuls | time ~ bytes / bandwidth; fidelity and program config change nothing | fewer bytes (bfp8 / bfp4, bf16 instead of fp32 streams), DRAM-sharded weights, fewer passes over the same rows |
| compute (FPU) | large M per chip, HiFi2-4 | time scales with fidelity and M | lower fidelity where the operands have few significant bits, better program config, more cores (within the core cap) |
| data-movement RISC | writer / reader epilogues, scalar loops, per-row NoC commands | NoC command count, cycles per word | tile-sized reads, 32-bit loads, branchless code, move per-value work to the SFPU |
| host | many small ops, per-chip programs | wall >> device | section 3 |

The same op flips with the shape: weight formats that save time at small M save only memory at large M (example:
bfp8 dense weights at 1024 rows a chip: 0 ms; [DS41-79]); bfp4 experts save time at LoFi (weight-stream bound) but not
at HiFi4 (compute-bound) ([DS41-74]).

## 6. Trade-offs that flip

| trade | the cheap side wins when | the other side wins when |
|---|---|---|
| **More matmul work vs fewer collectives** (replicate weights / keep rows local, instead of resharding heads <-> rows) | the collectives and their launches cost more than the extra weight reads (example: projections ~3x slower each, yet attention -11 ms and fewer launches a chunk with no reshard all_to_all; [DS41-139]) | the weights are large and the chip is DRAM-bound already, or the collective is overlapped |
| **Sequence-parallel degree** (rows a chip) | higher: fewer collectives and launches, local norms / residuals (example: the lower degree measured +12 to +17 ms device, +340 launches a chunk) | lower: the per-chip rows get too few for matmul efficiency, or a carry / window needs more rows a chip than it has ([DS41-204]) |
| **Neighbour exchange vs all-gather** | each chip needs only an adjacent slice (windows, carries): -10 ms device, -280 launches a chunk (example) | every chip needs everything |
| **Persistent buffers vs shared L1** | the buffer has a fixed shape and nothing else needs the space | two ops time-share the L1 (arenas): a permanent buffer collides |
| **One program, patched vs many cached** | a few patched addresses (cheap) | patching is expensive and switching is cheap; measure ([DS41-131]) |
| **One pass at a narrower accumulator vs multi-pass at fp32** | L1 cannot hold fp32 partials for all rows (a bf16 accumulator: one pass) | the narrower sum misses the accuracy limits |
| **Exact vs approximate algorithm (skipping layers / work)** | the accuracy limits hold and the saved work is on the critical path | the saved device time hides behind the host (example: device -20 %, wall -3 %) |
| **Lower precision of a collective's payload** | the producer already writes the narrow dtype | it needs casts around the collective: they cost more than the bytes saved ([DS41-67]) |

## 7. Precision and formats vs bandwidth

- Formats follow the reference (agent rule 7, rule 17: no parity-only rounding). Within that, the narrowest format
  the device computes or stores in is the speed lever **only where the op is bandwidth-bound** (section 5).
- Weight formats: halving weight bytes halves a weight-bound op (example: experts bfp4 vs bfp8: +26 ms device a
  chunk for bfp8). A better rounding of the same format (least-squares shared exponent) costs no time and halves the
  error ([DS41-198]).
- Activation streams: bf16 instead of fp32 in DRAM-bound passes saves time and can even be more exact where the FPU
  reads fp32 at reduced precision ([DS41-104]).
- Caches: a wider KV / index format costs attention time in proportion to the bytes read (example: bf16 KV +5.8 ms
  attention a chunk vs the low-bit formats). One stored copy is cheaper than a compute copy plus a served copy.
- Fidelity: HiFi2 / LoFi is free accuracy-wise where an operand has few significant bits (fp8 / fp4 values fit; the
  result can be bit-identical, [DS41-68]); elsewhere it biases chained matmuls and is the owner's call.
- fp32 DEST can switch ops to slower kernels (streaming SDPA off, [K-47]; auto matmul configs spilling partials,
  [K-51]); keep it only where the reference accumulates in fp32.
- A precision change is never "free" on the metrics: report the margin used per metric (the A/B table's share column)
  and watch metrics already near their limit.

## 8. Collectives (CCL) placement

- Count collectives per layer in the plan; each costs launches (often per-chip programs) plus link time plus the
  peers' skew. Fewer, larger collectives beat many small ones.
- Prefer layouts that keep rows on their chip through the residual / norm path and reshard only at the matmuls that
  need it (example: residual streams on each chip's own rows removed an all_to_all and the norms' collectives:
  -9 to -15 ms device a chunk).
- Move a collective into the producer (write over the fabric from the epilogue) only for launch / host savings; the
  device time just moves (section 4).
- Pack, compress or cast in the producer before a gather, not after.
- Use the mesh's real topology: a torus closes rings (Ring instead of Linear can halve the busiest link's bytes;
  check which one the op uses); a fabric
  kernel must route by the forwarding direction the fabric reports, not by position ([DS41-224]).
- A CCL's section time includes waiting for the slowest peer: a large chip spread on a CCL section is imbalance
  upstream, not a slow collective.
- Persistent CCL outputs pay only once no address-keyed cache downstream misses (section 3).

## 9. Matmul and kernel configs

- Sweep program configs once per layout on the deployment box (shapes differ per mesh); after that a re-sweep rarely
  finds > 1 % ([DS41-191]). Stay inside the core cap (agent rule 18).
- Check the auto config on big fp32-DEST matmuls ([K-51], [DS41-112]): an explicit config or `minimal_matmul` can be
  several times faster. Small M (<= ~8 tiles): 1D multicast; narrow N: 2D.
- Batched matmuls with per-batch weights: only some program families keep fp32 partials ([DS41-113]).
- Attention: largest Q chunk that fits L1 cuts prefix traffic ([K-49]); one compiled program per chunk offset unless
  the offset is a tensor ([K-48]).
- Bring-up forks first (agent rule 16); a fork that is slow for this model is optimized in the fork behind an option.

## 10. Measuring correctly

The orchestrator measures every pick on and off (`orchestrator measure`, the A/B table); the agent's own runs are for
iterating only. What the framework already does: one process per side, the model loaded once; a clean wall run with
the profiler off, timed only when proven warm (no new program-cache entry); 20 timed runs after warm-up; device time
from the real-time profiler (no syncs).

- **Noise.** Typical run-to-run noise (example, 32-chip mesh, ~200 ms chunk): device total +-2-3 ms, wall +-3 ms on a
  quiet host and up to +-15 ms on a busy one; a section moving +-1 ms is noise unless it repeats at both positions.
  An outlier run inflates the mean and the spread (example: spread 279 ms on one side, 8 ms on a re-run of the same
  code). When the spread of a side is several times the other's, re-measure before concluding anything.
- **What counts as a win** (the skill's perf-pick rule): the gate passes, every accuracy metric within its limit,
  and it is faster on device or wall time (or saves memory) beyond the noise; or it only removes work by
  construction (fewer launches, cached programs, work off the main thread) with identical accuracy, even inside the
  wall noise. State the impact per chunk (ms, %), never per call.
- **Judge on totals.** A section win with an equal loss in another section is not a win (time moved, section 4).
- **Launch count is a hard number, wall time is not.** A drop in `host_launch_calls` / `profiled_programs` is exact;
  it shows in the wall only when the chunk is host-bound in that span.
- **Bit-identical changes** need no accuracy argument; all others report the margin used. Some metrics are chaotic
  (example: the last layer's error swings 0.70-0.91 of a limit on rounding-only changes; [DS41-188]): judge
  rounding-level changes on the stable ones.
- Warm only: the first run compiles programs (example: first long run ~5x slower; [K-48]). The hybrid component
  harness is not the model's speed ([K-50]).
- Profile per op only on representative layers and only when the section times cannot explain a result (the perf
  role text). Device-side zones in a kernel are for probing; remove them before the final measure.
- Long device jobs (> ~9 min) start in the background with a long timeout from the start, never under `timeout`
  ([DS41-183]).
- A/B hygiene: an env var both sides need is not part of `ab.change`; a switch-only pick that should not ship as the
  default uses `ab.keep_default`; a code drop without a switch uses `ab.before` (skill).

## 11. Common traps (checklist)

- Optimizing device time on a host-bound chunk and expecting the wall to move.
- Tuning compute (fidelity, program config, SFPU work) of a DRAM-bound op.
- A fusion that deletes one section's time and adds it to the next (a wait, a serial copy).
- Persistent buffers allocated mid-forward that move addresses under an address-keyed cache.
- Narrow-dtype collectives with casts on both sides.
- Per-chip programs or per-core args where one program with common args would do.
- A precision lever applied where the op is compute-bound (saves memory, not time) or the reverse.
- Reading one noisy A/B side as a regression; reading a 0.3 ms total change as a win or a loss.
- An option that silently switches a fork to a slower program mode.
- Python host conversions holding the GIL on a "background" thread.

## 12. Record what you learn

A measured lesson (a lever that paid or did not, and why) goes into the model's `bringup/known_issues.md` (short bullet,
detail in `known_issues_detail.md`) under its id; a lesson that holds for other models goes to the shared
`known_issues.md` Performance section, and a new rule of thumb into this file. Rejected and neutral picks are lessons
too: record the reason (no gain, moved time, noise, host-bound) with the A/B numbers.
