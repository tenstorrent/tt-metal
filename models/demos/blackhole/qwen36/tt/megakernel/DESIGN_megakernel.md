# Qwen3.8 decode megakernel (one persistent program per verify pass): design v0

Goal: verify ≤13.7 ms at 32K u8 and ≤9.4 ms at u1 (≥50% of 512 GB/s at today's bytes). Today: 17.82 ms (u8) and
14.21 ms (u1) with the shippable set (flag batch + KV BFP4 + multi-user SDPA cores 16).
Constraint: a plain tt-metal/ttnn generic_op, captured in the qwen36 vLLM decode trace (Blaze is reference only).

## 1. Per-chip bytes per layer (TP8, one P150)

| Op | Shape per chip | Dtype | MB/chip | Today µs (u8) |
|---|---|---|---|---|
| GDN in (qkvz + ba) | 5120 × ~2060 | BFP4 | 5.9 | 22.0 |
| GDN spec step (state math) | 48 V heads / 8 | fp32 state | 0.30 GB / pass total | ~44 (compute-bound) |
| GDN out | 768 × 5120 | BFP4 | 2.2 | 9.1 |
| Attn qkv (KV-split pair wqkvg_pair) | 5120 × 3584 | BFP8 | 19.5 | 49.5 (~394 GB/s) |
| SDPA (verify, 32K, u8) | KV per device | BFP4 KV | ~35 / layer | ~190 (KV-read bound) |
| Attn wo | → 5120 | BFP8 | 8.4 | 25.2 |
| MLP gate_up | 5120 × 4352 | BFP4 | 12.5 | 34-37 |
| MLP down | 2176 × 5120 | BFP4 | 6.3 | 18 |
| RMSNorm ×2, glue, gaps | — | — | — | ~45 / layer |
| All-reduce ×2 (one-shot, 2 links, ring) | ~240 KB / link | bf16 | — | 18-20 each (~9.6 on the wire) |

GDN layer weights: 5.9 + 2.2 + 18.8 = **26.9 MB**. Attention layer weights: 19.5 + 8.4 + 18.8 = **46.7 MB**.
Per pass (48 GDN + 16 attention + LM head/MTP): ~2.08 GB of weights + 0.56 GB KV (BFP4, u8) + 0.30 GB GDN state
= **~2.95 GB**.

## 2. Analytical model (what the sim must validate)

With a persistent program, two resources run concurrently:
- **DRAM stream**: ~2.95 GB at the v11-measured ~400 GB/s ≈ **7.4 ms** (u8). At u1 it's ~2.4 GB → ~6.0 ms.
- **Critical-path chain** (things that can't wait for "bytes alone"), per pass at u8:
  - GDN step compute: 48 × ~44 µs ≈ 2.1 ms (the GDN restructure could take it to ~1.2 ms)
  - SDPA: 16 × ~190 µs ≈ 3.0 ms, which is itself KV-read time, so it overlaps the DRAM-stream term
  - all-reduces: 128 × ~18 µs ≈ 2.3 ms on the chain (they can't overlap their own consumer)
  - matmul compute once weights are already in L1: ~5 µs per projection × 4 per layer ≈ 1.3 ms
  - per-op syncs inside the program: ~1-2 µs × ~6 per layer ≈ 0.6 ms (replaces ~1,200 launches/gaps ≈ 2.4 ms today)

Weights for the next op(s) keep streaming while the chain waits on GDN/SDPA/all-reduce, as long as L1 can hold
them. L1 budget: ~110 worker cores × ~1.2 MB free ≈ 130 MB per chip, and a whole attention layer is 46.7 MB. So a
prefetch depth of 2+ ops is easy; the L1 constraint is the working set of SDPA/GDN cores, not the weights.

Estimate: pass ≈ max(DRAM 7.4, chain non-DRAM ~6.3) + imperfect overlap (~1-2 ms) ≈ **9-11 ms at u8** (gate 13.7)
and **~7-9 ms at u1** (gate 9.4). The DRAM term dominates, so per-bank tail and multicast contention (which the sim
under-models) are the main risks.

## 3. Program structure

One generic_op per verify pass. Each core runs a loop over a per-pass descriptor table in L1. An entry is one
*phase*: {op type, layer, weight base page, K, N slice, input buffer, output buffer, sync}. Each core has a role per
phase type:
- **Stream/compute cores (80, the v11 rectangle)**: bank-local weight slices for every projection (gate_up, down,
  gdn_in, gdn_out, qkv, wo). Two DM RISCs issue the reads for phase p+1..p+D (depth D) as soon as phase p's reads are
  issued; slots are explicitly addressed (fm11 slot fix).
- **GDN cores (~16-32)**: own the recurrent state per (head, value block); run the spec step on the in-proj output,
  broadcast via roots. Today's kernel is compute-bound at ~44 µs; the all-users prefix restructure (in progress in
  another agent) drops in here.
- **SDPA cores (rest, ~16-24 per group)**: read KV (BFP4) and run verify SDPA at the ~16-core-per-group setting.
- **Roots (4)**: multi-root broadcasts of x/h between phases (v8/v11 scheme, ~5 µs per broadcast).
- **All-reduce**: kept as the existing one-shot all_reduce_async semantics, but issued from inside the program at
  phase boundaries. Its receive stage writes x straight into the stream cores' x buffers (v8 XPRE contract), with the
  RMSNorm folded into the next projection's input path.

Layer dependencies: in-proj → mixer (GDN/SDPA) → out-proj → AR → [norm] → gate_up → down → AR → next layer's
in-proj. The chain is strictly serial through activations; only weight streaming runs ahead.

## 4. Build plan (sim first, silicon checkpoints on d27u08)

MK1 (this step): a "layer skeleton" persistent kernel. The fm11 engine is generalised from 2 fixed matmuls to an op
list of streaming projections with per-op K/N and prefetch depth D, plus a calibrated *stall phase* standing in for
GDN step / SDPA / all-reduce (a fixed-cycle busy phase on the chain while the stream cores keep prefetching).
Correctness: PCC per projection against torch. Measure:
- µs per GDN layer and per attention layer
- stream utilisation during stalls
- per-phase sync cost
- whether D=2 keeps the stream continuous across the stalls

MK2: real GDN step on GDN cores inside the program (reuse the prod kernel as a phase).
MK3: real SDPA phase.
MK4: all-reduce issued from the program with the XPRE receive stage (multi-chip; silicon only, since the sim hangs
on >2-chip collectives).

## 5. Sim assumptions (stated)
- All-reduces are modeled as fixed stalls of 18 µs on the chain (silicon one-shot, 2 links). No overlap with their own
  consumer.
- GDN step and SDPA are modeled as stalls of their measured µs (44 / 190 at u8; 42→~30 / ~75 at u1) until MK2/MK3.
- KV and GDN-state DRAM traffic during stalls is NOT modeled in MK1, so MK1 is optimistic on DRAM contention.
  Correct for it with the analytical DRAM term above.
- The sim is optimistic on multicast contention and L1 spin-wait caching (fused MLP: sim 47.7 µs vs silicon
  63.2 µs single-launch). Every sim number gets a silicon checkpoint on d27u08 before it is used for a projection.

## 6. Interface spec v0 (shared by the MLP/engine, attention, GDN and AR/vLLM tracks)

Branches: base `anatarajan/qwen38-r16-megakernel` (engine, MLP, loop, orchestration, this doc). Tracks branch off it:
`-attn`, `-gdn`, `-ar`. Code lives in `models/demos/blackhole/qwen36/tt/megakernel/` (kernels in `kernels/`).
Interface changes go through this doc first: change the doc, tell the other tracks, then the code.

**Execution model.** One generic_op per verify pass. Every one of the 110 worker cores (P150 11×10 logical grid) runs
the same three kernels (reader/NCRISC/NOC1, writer/BRISC/NOC0, compute). Each is a *phase interpreter* over a
per-pass phase table in L1. Compute is time-multiplexed across phases: a core does whatever phase its role list
says, in pass order. Weight prefetch (DM RISCs) runs ahead of compute, up to depth D phases.

**Core roles** (runtime arg `role_mask`, bit per role; a core may have several):
- `STREAM` (bit 0): the 80-core rectangle x∈[0,9], y∈[0,7] (two multicast rects split at the DRAM column).
  Owns bank-local weight slices for every projection: GDN in/out, attn qkv/wo, MLP gate_up/down.
- `ROOT` (bit 1): 4 roots (logical cores 76-79) for multi-root h / activation broadcast (fm11 scheme).
- `ATTN` (bit 2): SDPA cores. v0: all 110 cores in groups of 16 (multi-user setting), time-multiplexed with STREAM.
- `GDN` (bit 3): GDN cores, two phases per GDN layer (v0.2, GDN track):
  - `GDN_PREFIX` (type 10): one core per (head, chunk). Row-wise and stateless for all users at once: conv window
    rows, 4-tap conv + silu, q/k L2 norm, silu(z). Output is qc|kc|vc|zs tiles per head (fp32, 32 rows). Runs right
    after gdn_in.
  - `GDN_STEP` (type 4): one core per (user, head, value block). Replay/WY state math + gated norm, consuming the
    prefix output. A ~16-32 core GDN set runs PREFIX then STEP back to back.
  - Why two phases: the all-users prefix does not fit in one gdn_spec_step program (the 70,656 B kernel config buffer
    overflows by ~0.5 KB at B=8).
  - v0.5 GDN_PREFIX data rules (GDN track, milestone A on silicon: conv+silu correct at B=8/B=1, standalone
    53.4 µs at B=8, mostly launch overhead):
    (a) **Megakernel-owned compact conv window:** one bf16 tile per (head, chunk) holding all users' K-1=3 rows
        (8 × 3 = 24 rows). GDN_PREFIX writes it back itself after the conv: the new window is rows t..t+2 of W per
        user, chosen by the accepted-draft index `mi`. That is 1 read + 1 write per chunk instead of 8 + 8.
        **The write-back must use the committed (accepted) position**, i.e. the same rollback rule as the GDN
        state under speculative verify.
    (b) Selector tiles (2K one-hots per (B, T)) stay resident in L1 for the whole pass (constant, inside the GDN
        working region).
    (c) Raw qkv rows come straight from the gdn_in phase's L1 output (STREAM cores → prefix cores by unicast at
        the final offset, under the activation distribution rule), with no DRAM round-trip.
    Estimated steady PREFIX phase ≈ 6-8 µs per layer at u8, replacing ~33 µs of per-user row-wise work in the step.
- `AR` (bit 4): all-reduce send/recv cores (AR track defines; receive writes x into STREAM cores' CB_X).

**Phase table** (L1, array of 16-u32 records, base address in runtime arg; `n_phases` in runtime arg):
| word | field |
|---|---|
| 0 | type: 1 PROJ, 2 MLP_GU, 3 MLP_DN, 4 GDN_STEP, 5 SDPA, 6 AR_SEND, 7 AR_RECV, 8 NORM, 9 STALL, 10 GDN_PREFIX, 11 GDN_STEP_U1 |
| 1 | layer index |
| 2 | weight tensor id (0 gate_up, 1 down, 2 gdn_in, 3 gdn_out, 4 qkv, 5 wo, 6 kv cache, 7 gdn state) |
| 3 | weight page base for this core (bank-local slice), or KV/state page base |
| 4 | K tiles |
| 5 | N tiles for this core |
| 6 | input CB id |
| 7 | output CB id |
| 8 | wait sem id (input ready, cumulative value = phase seq + 1) |
| 9 | signal sem id (output ready) |
| 10 | stall cycles (STALL / sim stand-ins) |
| 11 | flags (bit 0 last-in-layer, bit 1 residual add, bit 2 norm folded) |
| 12 | block-private |
| 13 | **engine: prefetch budget** = max weight bytes in flight on STREAM writers during this phase (0 = unlimited). SDPA phases set it so the STREAM cores' V reads aren't starved |
| 14-15 | block-private (attention/GDN tracks define) |

**CB ids (fixed across tracks):**
0 CB_X (32×5120 bf16 activation, replicated on STREAM cores) · 1 CB_W (even weight groups, NOC1) ·
5 CB_W2 (odd groups, NOC0) · 4 CB_WD (down/out-proj slice) · 6 CB_WM (mixer in-proj slice) · 2 CB_HL (local h tile) ·
3 CB_H (gathered h) · 7 CB_YM (mixer out) · 8 CB_GO (phase token) · 9-12 ATTN (Q, K, V, O) ·
13-15 GDN (state, conv window, out) · 16 CB_OUT (block output → AR send) · 17-20 AR (send, recv, residual, norm w) ·
24-31 block scratch (owner = phase type).

**Semaphores (v0.1, per the AR track):** tt-metal allows only 16 *program* semaphores per core (NUM_SEMAPHORES = 16).
- Program semaphores: 0-11 MLP roots (fm11: h counters 0-3, h flags 4-7, down-done 8-11), 12-15 phase/layer
  barriers (cumulative, value = phase seq + 1).
- Every other track (ATTN, GDN, AR) uses **global semaphores**: `ttnn.create_global_semaphore(mesh, core_ranges, 0)`,
  i.e. L1 words at a uniform address on every core and device. Pass the addresses in runtime args or phase-table
  words 12-15, and use `noc_semaphore_*` on the raw addresses. The uniform cross-device address is also what fabric
  fused write + atomic-inc needs.
- Exception: a track may keep a program semaphore on a core that is not a STREAM core (e.g. the AR sender's
  producer counter on (10,0)).
- setup_fabric_connection appends its own program semaphores on fabric sender cores, so keep those off STREAM cores.

**Activation distribution rule:** x (and every replicated activation) goes through the 4 roots. Each core unicasts its
slice into root k's buffer at its final offset, and each root multicasts its range. Never have many cores multicast
into overlapping rectangles at once (fused MLP v1/v2: 150 µs stalls on silicon).

**L1 budget per STREAM core (BH ~1.45 MB usable after FW/kernel/CB config):**
| Item | KB |
|---|---|
| CB_X (activation, bf16; candidate: BFP8 to halve) | 320 |
| weight prefetch ring (depth D=2 phases; attention qkv slice is the largest at ~244 KB) | 500 |
| CB_H + CB_HL | 140 |
| ATTN / GDN working set (time-multiplexed, shared region) | 300 |
| AR buffers | 100 |
| slack | ~90 |
Rule: a block may use the shared working region only during its own phases. Weight prefetch must never touch it.

**Attention-layer L1 during SDPA (v0.3):** STREAM weight residents during SDPA are this layer's wo slice (~105 KB) +
the next MLP's even gate|up (MKA_GUPRE: the whole even half, 110 KB, prefetched during SDPA to cut the ~93 µs
post-SDPA tail) + odd W2 + WD, all inside the 500 KB weight ring. qkv streams through the ring in chunks and is
consumed before SDPA, so it never co-resides with GUPRE. Keep GUPRE if the MK2 build keeps ≥16 KB headroom on STREAM
cores; otherwise single-buffer WD.

**Prefetch throttle (v0.3, attention track request):** during a phase with word 13 > 0, the engine issues next-phase
weight reads in chunks and keeps at most that many bytes outstanding (trid-ring waits), interleaving one weight chunk
per KV group. Why: unthrottled, the STREAM cores' NOC0 V reads finish last (~190 µs on top-row cores vs ~60 µs on
ATTN-only cores), which binds the SDPA phase. Role-weighting the KV shares helped only ~5%.

**MLP tail rules (v0.3, from the -mk-tail track, silicon d27u08; MKTAIL_RESULTS.md):**
1. CB_H is published per root chunk (HT/4) in fixed root order, so consumers start down on chunk i (MK_DNPIPE).
   Numerically identical.
2. Root flags are ordered with `noc_async_writes_flushed`, not the acked barrier, between the data and flag
   multicasts (MK_FLAG_FLUSH). Numerically identical.
3. The h landing buffer may be BFP8 (MK_H_BFP8; CB_HL/CB_H bfloat8_b, XTB 1088). It halves the root multicast
   (4.5-5.7 → 2.9 µs while the stream is active); PCC 0.99998 → 0.99993. **Default OFF until it passes the output-length
   gate** (lesson from KV BFP4).
4. Measured down tail: ~7 → 5.3-5.7 µs per layer, about −0.19 ms per pass.
5. Cold-start MIXFIRST stays off: it improves the median but not the max, and costs ~50 µs once per pass. If revived,
   it uses a global semaphore, not program sem 12, which is reserved for barriers.

**SDPA phase facts (attention track, silicon d30u20, 32K u8, branch -mk-attn, DESIGN_attn.md §5):**
- K is read on the reader/NOC1 and V on the writer/NOC0. Single-NoC KV took 186 µs.
- Interleaved KV beats bank-local: NoC distance dominates.
- The tree reduce runs on the reader, so the writer's weight prefetch never blocks it.
- KV dtype is set by MKA_KVDT (bfp8 default, bfp4 optional).
- Measured at BFP8: KV-only 166-175 µs (71 MB, ~420 GB/s). With 27 MB of STREAM weight prefetch running alongside:
  234-249 µs (98 MB, ~400 GB/s aggregate).
- L1 at BFP8: NBB=2 uses 288 KB, inside the 300 KB working region.

**BH NoC alignment rule:** DRAM→L1 NoC reads need equal source/destination offsets mod 64 B. Row moves go through
an aligned staging buffer plus an L1 copy; otherwise the data is silently corrupted.

**Block contract:** a block's input is CB_X on its cores once wait-sem ≥ seq+1. Its output goes to CB_OUT (per-core
shard), and it raises signal-sem. The AR track turns CB_OUT (all chips) into the next CB_X (+ residual, + norm).
Blocks never call launch-level ops. They must be trace-capturable (no host round trips inside the pass).

## 7. Release bundle (user decision 2026-11-10: ships with the megakernel ≥50% release, not standalone)
- Multi-image, up to 4 images: tt-metal anatarajan/qwen38-r13-mm @c8c90fe (QWEN36_MM_MAX_IMAGES), merged into
  this base branch (merge 80b077432f1). Every megakernel test runs with multi-image available.
- Vision-embedding cache: anatarajan/qwen38-r16-mm-vcache @8c2faf2ec95 (QWEN36_MM_VISION_CACHE; repeat 3-image
  turns 2.0 → 1.0 s). To merge into the base before release.
- Router image-cost routing; VLLM_ROUTER_MAX_IMAGES=4 at release.
- Shippable set: flag batch + QWEN36_R16_KV_BFP4 + multi-user SDPA cores 16 (−8.3 % step u8; acceptance unchanged).

## 8. Projection update (2026-11-10, KV BFP4 failed the GPQA output-length gate: answers 1.28× longer)
KV is **BFP8 by default**. The attention block keeps the KV dtype configurable (BFP8 default, BFP4 optional behind a
flag). The re-projection uses MK1 silicon non-stall times, r15 per-op GDN 64 / 42 µs (u8 / u1), SDPA at BFP8
(236 µs u8, ~94 µs u1 per attention layer, no ×0.8), 2 all-reduces × 18 µs per layer, and 0.5 ms LM head/MTP:

| | GDN layer | Attention layer | Pass | Gate | Margin |
|---|---|---|---|---|---|
| u8 | ~138 µs | ~322 µs | **~12.3 ms** | 13.7 | ~1.4 ms |
| u1 | ~113 µs | ~175 µs | **~8.7 ms** | 9.4 | ~0.7 ms |

DRAM at u8 with BFP8 KV: ~3.5 GB / 12.3 ms ≈ 285 GB/s, under the ~400 GB/s drain. Biggest chain levers, in order:
all-reduces (~2.3 ms), the GDN step (the prefix split targets ~−0.9 ms at u8), and the MLP down tail (13 vs 4 µs).

Attention-track silicon confirms the §8 attention-layer SDPA term: ~240 µs **including** the next layer's weight
prefetch running alongside (it was modeled as 236 µs SDPA + hidden stream). The §8 projection stands (u8 ~12.3 ms,
u1 ~8.7 ms).

## 8b. Projection update: measured in-engine attention layer + all-reduce sensitivity (2026-11-10)
The attention track measured the attention layer *inside the engine* on silicon at **368 µs steady** (32K, u8, BFP8
KV, with modeled 18 µs AR stalls), against the 322 µs estimated in §8.

Model: pass = 48 × (GDN non-AR + 2·AR) + 16 × (attention non-AR + 2·AR) + 0.5 ms (LM head/MTP) = base + 0.128 ms
per µs of AR. Non-AR per layer: u8 GDN 102 µs / attention 332 µs; u1 GDN 77 / attention 139. The u1 attention has
not been re-measured in the engine; the last row assumes it grows +46 µs like u8.

| Per-AR µs | 18 | 25 | 40 | 60 | 116 | 181 | **break-even** |
|---|---|---|---|---|---|---|---|
| u8 pass (gate 13.7) | **13.0** | 13.9 ✗ | 15.8 ✗ | 18.4 ✗ | 25.6 ✗ | 33.9 ✗ | **≤ 23.4 µs** |
| u1 pass (gate 9.4) | **8.7** | 9.6 ✗ | 11.5 ✗ | 14.1 ✗ | 21.3 ✗ | 29.6 ✗ | **≤ 23.3 µs** |
| u1 if attention +46 µs | 9.5 ✗ | 10.4 ✗ | 12.3 ✗ | 14.8 ✗ | 22.0 ✗ | 30.3 ✗ | **≤ 17.5 µs** |

**AR RISK (blocking):** the gate is met only if the in-megakernel all-reduce is about as fast as today's one-shot
all_reduce_async (18-20 µs), i.e. ≤ ~23 µs per call (≤ ~17.5 µs if u1 attention grows like u8). The real
megakernel AR is currently 116-181 µs per call (v1/v2), and v3 hung. At those numbers the pass is 25-34 ms, worse than
today. AR latency is now the make-or-break item, ahead of the GDN restructure and the MLP tail.
Mitigations:
(a) reuse the proven all_reduce_async kernels' one-shot fabric path as a phase inside the program (same packets,
    same 2 links), rather than a new AR design;
(b) overlap: issue the MLP AR send as soon as each core's down chunk is ready (per-root-chunk publication already
    exists), so part of the AR hides under the down tail;
(c) cut AR count: fuse the attention/GDN out-proj AR with the MLP-in residual where the math allows (not possible
    across a norm without a 2-stage AR);
(d) fallback: per-sub-block programs with today's AR op between them. That costs ~7 µs launch per boundary
    (×128 ≈ 0.9 ms) on top of 18-20 µs ARs, about 14 ms at u8, which misses the gate.

## 9. Spec v0.4: deferred RMSNorm (block contract change, AR track proposal, 2026-11-10)
Measured (AR track): the one-shot AR phase is **21.7 µs**, bit-identical to all_reduce_async. With the RMSNorm on the
owners plus 4-root x delivery it was 71.9 µs (norm chain ~25 µs, x multicast ~16-30 µs). The norm chain must leave the
critical path.

**New block contract:**
- The AR delivers the **un-normed residual x** (bf16 32×5120) straight into the STREAM cores' CB_X via the 4-root
  rule. Alongside it come **rstd partials**: each AR owner computes Σx² over its reduced slice and appends fp32 32 rows
  × n_owners to the all-gather payload (CB 20, renamed CB_RSTD).
- Consumers (MLP gate_up, attention qkv, GDN in-proj) use weights with **γ (× NORM_FIX) folded into the rows**
  (diag(γ)W, done at load). They compute rstd = 1/sqrt(ΣΣx²/5120 + eps) from the partials (a few ops) and **scale
  each output row by rstd at pack time**.
- Exact in real arithmetic: RMSNorm(x)·γ @ W = rstd ⊙ (x @ diag(γ)W).
- The residual add stays with the AR owners (they hold the residual).
- **Numerics gate:** folding γ re-quantizes the BFP4/BFP8 weights (different rounding), so it must pass the GPQA
  output-length gate before it becomes default. Fallback: unchanged weights + x ⊙ γ on the consumer (~7 µs per
  projection phase on the critical path).

**§8b re-run with the AR split into on-path vs overlappable** (base + 0.128 ms per on-path µs):

| On-path µs per AR | 21.7 (all on path) | 18 | 15 | 12 |
|---|---|---|---|---|
| u8 (gate 13.7) | 13.49 ✓ (0.2 margin) | 13.01 | 12.63 | 12.25 |
| u1 (gate 9.4) | 9.20 ✓ (0.2 margin) | 8.72 | 8.34 | 7.96 |
| u1 if attention +46 µs | 9.94 ✗ | 9.46 ✗ | 9.08 ✓ | 8.70 ✓ |

Overlappable candidates: the MLP AR send starts per down chunk (DNPIPE publication), hiding part of the
reduce-scatter under the down tail. The gather multicast for the next phase can overlap the consumer's γ-folded
weight prefetch, which is already resident. Target on-path **≤ 15 µs** for a real margin at both u8 and u1.

## 8c. Correction: steady attention layer = 377 µs (2026-11-10)
The attention track's true steady figure is the mean of the middle layers of a 5-layer pass: **377 µs** (GUPRE,
NBB=1, 16 KB budget). The earlier 368 µs was a last layer, which is 40-60 µs cheaper because it has no next-layer
prefetch. Non-AR per layer at u8: GDN 102, attention 341. The u1 attention is not yet measured in the engine; the
last row assumes it grows like u8 (+55 µs).

| On-path µs per AR | 21.7 | 18 | 15 | 12 | break-even |
|---|---|---|---|---|---|
| u8 (gate 13.7) | **13.63** (0.07 margin) | 13.16 | **12.77** | 12.39 | 22.2 µs |
| u1 (gate 9.4) | 9.20 | 8.72 | **8.34** | 7.96 | 23.3 µs |
| u1 if attention +55 µs | 10.08 ✗ | 9.60 ✗ | **9.22** | 8.84 | 16.4 µs |

**The ≤15 µs on-path AR target is now REQUIRED, not optional.** At 21.7 µs, u8 has essentially no margin, and u1 fails
if its attention layer grows like u8's. At 15 µs: u8 12.77 (0.9 ms margin), u1 8.34-9.22.

## 10. Spec v0.5 proposal: all ~120 cores stream weights when not in their own phase (2026-11-10)
**Why:** at u1 the attention layer measures ~270 µs in-engine (§8 assumed 175). Cause (attention track): the layer is
bound by the 80 STREAM cores' weight stream (~46.7 MB of next-layer qkv chunks, wo, gate|up, MLP; ~64.5 MB total at
~240-290 GB/s), while the 40 ATTN-only cores finish their 148 KB KV slice and idle. They also idle through every
GDN-type layer.

u1 now decides the gate: base 7.94 ms → 10.72 at 21.7 µs per AR, 9.86 at 15 µs (gate 9.4: **FAIL**); break-even
11.4 µs per AR.

**Change (roles become per phase, not per core):**
- New role bit `STREAM_EXT` (bit 5) on the ATTN-only cores. A STREAM_EXT core owns bank-local slices of the
  *projection* weights (qkv, wo, GDN in, GDN out) and streams and computes them like a STREAM core, whenever it is not
  in its own SDPA phase (before SDPA starts, after its KV slice is done, and through every GDN-type layer).
- Step 1 (minimum, attention track): qkv/wo PROJ across all 120 cores at u1.
- ~~Step 2: GDN in/out across all 120 cores during GDN-type layers.~~ DROPPED (§13): the mixer is 4.2 µs with resident weights.
- Step 3, only if needed: MLP gate|up/down on 120 cores. fm11 assumes the 80-core rectangle (68 gate|up columns,
  4 roots), so going to 120 needs extra multicast rectangles and new h-chunk ownership. Not in v0.5.
- Rules unchanged: bank-local placement (each STREAM_EXT slice in its nearest bank) and the 4-root rule for every
  replicated activation (x to STREAM_EXT cores goes through the same 4 roots, adding more multicast rectangles).
- L1: STREAM_EXT cores have no MLP buffers (no CB_W/CB_W2/CB_WD/CB_H), so they hold a larger projection ring
  (~600 KB) next to their 300 KB SDPA working set.

**Re-projection, u1** (non-AR GDN 77 µs; attention layer at the stated µs; 0.128 ms per µs of on-path AR):

| Attention layer, u1 | AR 21.7 | AR 15 | AR 12 | break-even AR |
|---|---|---|---|---|
| today 270 µs | 10.72 ✗ | 9.86 ✗ | 9.48 ✗ | 11.4 µs |
| v0.5 conservative 230 µs (stream ~340 GB/s) | 10.08 ✗ | **9.22** | 8.84 | 16.4 µs |
| v0.5 at ~400 GB/s, 210 µs | 9.76 ✗ | **8.90** | 8.52 | 18.9 µs |

**u8:** little change expected. The ATTN-only cores are already busy with KV (u8 SDPA is KV-read-bound), so the gain
is limited to the post-SDPA tail (perhaps 377 → ~355-365 µs). u8 at 15 µs AR stays ~12.6-12.8 ms (gate 13.7).
**Both gates need both changes: AR ≤15 µs on path AND v0.5 streaming at u1.**

## 11. Mixed-group silicon checkpoint (2026-11-10, d27u08 chip 0, branch -mk-mixed @183d9b41350)
32K, u8, BFP8 KV, NL=8 = 2 × [G,G,G,A], GUPRE, NBB 1, 16 KB budget, GDN stand-in 64 µs, AR stalls 18 µs. The CSV is
on the laptop at .claude/tmp/r16/mega/si_mixed8/.
- **Correctness:** mixer PCC 0.99999; SDPA L3/L7 0.99998; MLP last layer 0.99998.
- **Per layer (µs):** 220.4 (L0, cold), 145.4, 141.3, **382.4** (L3, attention, middle), 151.3, 141.2, 146.1,
  330.1 (L7, attention, last).
- **Steady:** GDN-type 145.1 µs (model 138); attention middle 382.4 µs (model 377).
- **Group:** 817.7 µs vs the 791 µs estimate (+3.4 %).
- **u8 full pass from measured layers:** 13.58 ms at 18 µs AR stalls; ~13.19 ms at 15 µs on-path AR; ~14.05 ms at
  21.7 µs. The gate (13.7) needs AR ≤15 µs **plus** the GDN restructure (replaces the 64 µs stand-in, target
  ~-19 µs/layer ≈ -0.9 ms) and/or v0.5 streaming.

## 12. AR / x-delivery layout pinned (v0.4) + the x-delivery problem (2026-11-10, AR track f5342c9cbd3)
**Pinned layout:**
- CB_X: un-normed residual, bf16 [32, 5120]. Chunk j = 10 tiles at offset j × 20 KB; owner j = (0,8)-(7,9),
  row-major.
- CB_RSTD: fp32 [16 owners][32 rows] at offset 0, P_j[r] = Σ_{c∈chunk j} x[r,c]² · 2^-12. The tag for owner j is at
  2048 + 16·j and is set to SEQ+1.
- Consumer: rstd' = rsqrt(Σ_j P_j + 1.25·eps) = rstd / NORM_FIX, so γ·NORM_FIX is folded into the weights.
- Flags: 4 root x-flags (4-root mode).

**Silicon (d26u02, 1x8 ring):**
- The residual is bit-identical on all devices, and CB_X == residual on all 80 cores. Partials max relative error
  8.2e-4.
- AR alone 21.7 µs; AR + 1 chunk + partials 24.7 µs.
- **x delivery (bf16, 320 KB to 80 cores) is multicast-throughput-bound:**
  - one 20 KB multicast to 40 cores takes ~1.3-1.5 µs;
  - 16 owners multicasting at once thrash: ~92-102 µs per call;
  - serial or token-chained delivery: ~101-107 µs;
  - 4 roots × 80 KB: ~96-102 µs in trace, 20 µs median per root multicast.
- The practical floor is ~16-20 µs (theory 3.7 µs). **Replicated full-K x cannot be on the critical path.**

**Options:**
1. Overlap: K-ordered root flags so consumers start on chunk 0 (weights are resident), and start delivering each
   chunk as soon as the all-gather lands it. Removes up to ~10 µs; delivery still ≥ AR tail.
2. Fewer bytes: BFP8 x (≈½; numerics, so it needs the output-length gate); gate_up-only targets (68 cores) for
   the MLP.
3. **Proposed v0.6, split-K tiling (removes replication):** arrange the 80 STREAM cores as 8 K-rows × 10 N-columns.
   Core (k, n) owns W[K-slice k, N-slice n]. x chunk k (40 KB) goes only to K-row k: 8 owners multicast to 8
   *non-overlapping* row rectangles at once, ~40 KB per core instead of 320 KB, no thrash (~2-4 µs). Each N-column
   then reduces its 8 partial outputs (32 × N/10, small) to the column's root before the nonlinearity (SwiGLU) or
   the next consumer. Extra reduce traffic is ~2 MB/layer vs ~25 MB of x multicast. This changes fm11's gate|up/down
   layout and the qkv/GDN in-proj layouts. Biggest remaining lever for the ≤15 µs on-path AR + delivery target.
   Needs a sim prototype first (fused MLP v12).

## 13. v0.5 step 2 dropped; engine to-dos (2026-11-10, -mk-gdnext GDNEXT_RESULTS.md)
- **Step 2 dropped:** with resident weights the GDN-type mixer compute is only 4.2 µs, so spreading GDN in/out over
  120 cores has at most ~2-3 µs headroom. v0.5 = step 1 only (qkv/wo on 120 cores, the u1 attention lever).
- **To integrate into the mixed engine:**
  - MKA_MIXTRID (separate mixer trids 6/7; GDN-layer MLP slots pushed in W-GDN before plan_layer(L+1)): cold
    layer 0 mixer 114 → 89 µs, layer 0 247 → 235 µs, PCC unchanged. From anatarajan/qwen38-r16-mk-gdnext.
  - Port MK_DNPIPE / MK_FLAG_FLUSH (from -mk-tail, currently only in fm11_common.hpp) into mka2_writer/compute's
    MLP tail: ~1-2 µs per layer.
  - **Owner:** a0992fda706a58a7c is porting DNPIPE, FLAG_FLUSH and MIXTRID into the MKA2 kernels on -mk-gdnext
    (default-off, measured with mixed NL=8 on d30u20). The base integrates from there.
- **GDN-type layer anatomy (silicon, steady ~145 µs):** ~100 µs of stand-ins (GDN step 64 + 2 ARs × 18) plus
  ~35 µs of real work (mixer ~4 with resident weights, MLP, h-broadcast/down tail, syncs). The levers are the
  **GDN step** (prefix split: ~6-8 µs prefix + reduced step vs 64) and the **AR + x delivery** (§12). Projection:
  u8 GDN layer ≈ 35 + GDN(prefix+step) + 2·AR_on-path.

**v0.6 split-K: AR-side check (AR track) and decisions:**
- **Fits the owner layout.** Keep **16 owners, 2 per K-row** (owners 2k and 2k+1, 10 tiles each, ~40 KB per core),
  which preserves the 10-tile reduction parallelism.
- Each K-row is 2 physical rectangles (column split at logical x=7: 7 + 3 cores), so 2 multicasts per chunk.
  Destinations never overlap, so no destination thrash.
- **Next bottleneck: the shared vertical link.** On NOC1 all chunks for a rectangle climb one column (logical 6 for
  rect0, 9 for rect1), ~320 KB through one link (≥3.7 µs). Rule: alternate NoCs per K-row (even rows NOC1, odd rows
  NOC0 entering at the x0 corner) and/or give each K-row a distinct entry column. Check in sim.
- **rstd goes only to the 10 N-column roots** (applied after the N-column reduce): 2.3 KB to 10 cores instead of 80.
- Pipelining with the all-gather is already the default (owners multicast each chunk as soon as it lands).
  Landing order is not K order, so K-ordered flags in the replicated layout can stall. Split-K avoids that because
  each K-row waits only for its own chunk.
- Next: fused MLP v12 (split-K gate|up + N-column reduce + SwiGLU at the column root, split-K down) in sim, then
  silicon.

**v0.6 down-stage layout (adopted, -mk-v12 agent's proposal):** gate|up gives h_n at column root n (68 (g,u) pairs
over 10 columns: 7×8 + 6×2). Root n multicasts h_n down its own N-column (10 non-overlapping column rectangles).
Core (k, n) computes out[:, N-slice k (20 tiles)] over K = h_n, then sends its 2 × 10-tile halves to K-row k's 2
owners, which sum the 10 column partials. **The down output lands directly in the next layer's x chunk layout**, so no h
redistribution and no separate x gather: the AR's reduce-scatter input is already per owner.
Partials bf16 first; fp32 if PCC needs it. Branch anatarajan/qwen38-r16-mk-v12 (stage A kernels from the coordinator;
silicon, down stage and persistence by the -mk-v12 agent; sim by the coordinator).
- MLP tail port merged-ready (-mk-gdnext top): DNPIPE in mka2 + MIXTRID, numerically identical, GDN-type layer
  ~-2 to -2.5 µs. Default ON at integration.
- **Replicated-x delivery measured with pipelining (AR track, fresh d26u02):** consumers can start (K-cumulative
  chunk 0) only at ~44 µs after AR start. Everything lands by 57 µs (per-chunk mode) or 57.2 µs (4-root mode, root 0
  last), against 21.7 µs for the AR alone. Replication adds ~20-35 µs on the path even when pipelined, and landing order
  ≠ K order makes it worse. **≤15-23 µs is not reachable with replication; split-K (v0.6) is required.** The rstd
  partial multicasts also contend with x. In split-K they go to only the 10 column roots, sent after x.

**v0.5 step 1 stand-in result (attention track, 1d559f26341; 5-layer middle mean, 32K BFP8 KV, GUPRE, NBB 1, 16 KB):**
- u1 attention layer **239.8 µs** with STREAM_EXT, vs 256.7 µs on 80 cores. u8 370.0 vs 374.5.
- **Bench fix:** the wo/extra stand-in reads all hit DRAM bank 0 (that was the "column skew"). MKA_WOBANK=1 spreads
  them (core i reads bank i%8) and took u1 from 270 to 257 µs. **The mixed engine (-mk-mixed @183d9b41350, off
  efe85cf56ca) has the same hotspot**, so its 382 µs attention and GDN layer timings are pessimistic. Rebase onto
  1d559f26341 and re-measure.
- With EXT, u1 is DRAM-bound in the SDPA window (~400 GB/s, 64.5 MB in ~160 µs) plus a ~74 µs chain (wo + S_WO 10 +
  2 × 18 AR + MLP).
- **u1 pass** (GDN non-AR 77 assumed): base 7.46 ms → AR 15 µs: **9.38** (gate 9.4, 0.02 margin); AR 13: 9.12;
  AR 12: 8.99. Break-even 15.2 µs per AR. With split-K (v0.6) removing the x gather, the on-path AR should approach
  the 21.7 µs AR-alone or below, so u1 also needs the AR itself ≤~13 µs, or a shorter post-SDPA chain (fold wo into
  the tree tail, overlap the attention AR with the MLP's gate|up K-stream).

## 8d. Projection of record (2026-11-10), all measured in-engine on silicon except the u1 GDN layer
Inputs (32K, BFP8 KV):
- GDN-type layer u8 145.1 µs (mixed checkpoint) − 2.25 µs (DNPIPE+MIXTRID port) → non-AR 106.85 µs.
- Attention layer u8 370.0 µs, u1 239.8 µs (v0.5 step 1 STREAM_EXT, MKA_WOBANK bank fix) → non-AR 334 / 203.8 µs.
- u1 GDN non-AR is assumed 77 − 2.25 = 74.75 µs (not yet measured in-engine at u1).
- +0.5 ms LM head/MTP; 2 ARs per layer.

| On-path µs per AR | 21.7 | 18 | **15** | 13 | 12 | break-even |
|---|---|---|---|---|---|---|
| u8 (gate 13.7) | 13.75 ✗ | 13.28 | **12.89 (0.8 margin)** | 12.64 | 12.51 | 21.3 µs |
| u1 (gate 9.4) | 10.13 ✗ | 9.65 ✗ | **9.27 (0.13 margin)** | 9.01 | 8.88 | 16.0 µs |

Still stand-ins: the GDN step (64 µs; prefix split targets ~−19 µs/layer ≈ −0.9 ms), real qkv/wo PROJ phases
(attention track), and the AR itself (split-K v0.6 removes x replication; the AR alone is 21.7 µs today, so the AR core
must also drop to ≤15 µs, or overlap). The mixed-engine numbers predate the bank-0 fix (MKA_WOBANK), so its attention
layers are pessimistic; rebase onto 1d559f26341 before the next mixed run.

**v0.6 owner placement (changed): owners IN-ROW at logical x = 10, 11, rows 0-7** (-mk-v12 silicon, d30u20,
1ebb16fd568, v12/XDEL_RESULTS.md):

| x delivery to 80 cores | median |
|---|---|
| Replicated (8 owners × 40 KB) | 49.0 µs |
| Split-K, pinned owners (0..7, 8..9), alternating NoC | 14.0 µs (all NOC0 35.6, all NOC1 17.0; shared vertical links wrap the torus) |
| **Split-K, owners (10,k) and (11,k) for K-row k, NOC1 multicast west within the row** | **4.42 µs** (max 7.75) |

- New pinning: owner 2k → (10,k), owner 2k+1 → (11,k). The CB_X chunk mapping is unchanged (chunk j = 10 tiles).
- The down stage's owner sums also land in-row: (k,n) partials go to (10,k) and (11,k).
- Role overlap is allowed (roles are per phase). Columns 10-11 are also STREAM_EXT (v0.5 qkv/wo) and part of the ATTN
  set. The AR-owner phase (reduce-scatter landing, rstd partials, row multicast) is time-disjoint from their SDPA/EXT
  phases. L1 on these cores: AR buffers ≤~100 KB (CB 17-20) + EXT ring + SDPA working set must fit, so check the
  budget at integration.
- **The AR track must move its owner coordinates** from (0..7, 8..9) to (10..11, 0..7).

## 8e. Projection of record, updated with in-engine u1 GDN (2026-11-10, -mk-mixed2 @746a7169b44, d27u08)
Mixed NL=8 (2 × [G,G,G,A]), 32K, BFP8 KV, with EXT + bank fix + tail port (DNPIPE, FLAG_FLUSH, MIXTRID). All PCCs as
before (mixer 0.99999, SDPA 0.99998, MLP 0.99998). CSVs are on the laptop at .claude/tmp/r16/mega/si_mx2_{u8,u1}/.

| Layer (µs, middle means, with 2 × 18 AR stalls) | u8 | u1 |
|---|---|---|
| GDN-type (5 middle layers; GDN step stand-in 64) | 138.0 (136.8 / 140.0 / 135.1 / 141.1 / 137.2) | 142.3 (150.3 / 136.5 / 145.8 / 134.0 / 145.0) |
| Attention-type (L3 middle) | 367.3 | 242.0 |

- Non-AR per layer: u8 GDN 102.0 (incl. the 64 µs stand-in), attention 331.3.
- u1 GDN 84.3 (the 64 µs stand-in replaced by r15's u1 GDN step of 42), attention 206.0.
- **The u1 GDN layer is 84.3 µs non-AR, not the assumed 74.75.**

| On-path µs per AR | 12 | **15** | 18 | 21.7 | 25 | break-even |
|---|---|---|---|---|---|---|
| u8 (gate 13.7) | 12.23 | **12.62 ✓** | 13.00 | 13.48 | 13.90 ✗ | 23.4 µs |
| u1 (gate 9.4) | 9.38 ✓ | **9.76 ✗** | 10.15 | 10.62 | 11.04 | **12.2 µs** |

**u1 is the binding gate: on-path AR ≤ ~12 µs, or ~0.4 ms of other u1 savings.** With split-K and in-row owners, x
delivery is ~4.4-7.8 µs, so the on-path AR ≈ AR core (21.7 today; part of its in-chip gather is now done by the down
stage's owner sums) + the unoverlapped delivery. u1 levers:
- AR core speed (the dominant one);
- the u1 attention post-SDPA chain (fold wo into the tree tail; overlap the attention AR with the gate|up K-stream);
- the u1 GDN step (42 µs; the prefix split does not help at u1, which has one user per head).

## 14. Spec v0.7: GDN_STEP_U1 (phase type 11) for the u1 gate (GDN track, -mk-gdn-u1v2 @3de08997fd3, DESIGN_gdn_step_u1.md)
- **Phase type 11, GDN_STEP_U1:** one core per (head, value block): 24 cores, or 48 with split-K. Replay + WY +
  gated norm, with the norm's Σo² combined by a scalar exchange. It consumes GDN_PREFIX output (phase 10 on 72 cores,
  plus l2norm and silu(z)/gates on aux cores).
- **Selection:** the host builds the per-pass phase table with type 11 when the verify batch has users == 1, and
  type 4 (existing step) otherwise. Both are compiled into the program, with no runtime branch inside a phase.
- **Resources:** CB 13-15 keep their GDN meaning (state, conv window, out). CB 24-31 scratch is owned by the running
  GDN phase. GDN global semaphores: up to 8 (prefix + step), addresses in runtime args / phase words 14-15. No program
  semaphores.
- **Output contract:** the gated-norm output o goes to the gdn_out PROJ input (on STREAM cores, v0.6 split-K layout),
  and gdn_out writes the AR send buffer (CB_OUT). If the step writes CB_OUT directly (fusing gdn_out), the fused
  phase must keep the owner/partial layout of §12.
- **Estimate:** ~42 → ~21 µs per u1 GDN op, about −1.0 ms per u1 pass. §8e u1 at 15 µs AR: 9.76 → ~**8.75 ms** (gate
  9.4, ~0.65 margin). At 18 µs: ~9.14.

## 15. Split-K AR on silicon (AR track, -mk-ar @92e213cb005, d26u02 1x8, trace per call; bit-identical on 8 devices)

| Config | µs/call |
|---|---|
| RS + AG core, in-row owners (10..11, 0..7) + residual, senders (10..11, 8..9) | 23.0 (22.6 without residual) |
| + split-K x delivery (owner j → K-row j//2 slot j%2, NOC1 west) | 26.2 (+3.2 on path) |
| + Σx² partials in-row | 28.4 |
| + Σx² partials multicast to row-0 column roots | 45.5 (cross-row multicast from rows 1-7 stalls) |

**Pinned (v0.7):** Σx² partials go in-row. CB_RSTD on (k,n): slot p = j%2 at p × 128 B, tag at 2048 + p × 16. Each
(k,n) forwards its 2 partials to the column root inside its column-reduce payload (column root = row 0, end of the
column tree), and the root sums the 16. No extra multicast.

**AR cost model:** chunk sweep 10/5/2/1 tiles → 23.1/17.6/14.2/12.8 µs ≈ 11.7 µs fixed + 0.036 µs/KB. About 7 µs
of each traced call is dispatch, so **in-megakernel device time ≈ 16 µs** (~5 fixed + ~10 bandwidth). Adding split-K
delivery brings the on-path total to **~19 µs**. 2 links only.
Levers:
- persistent fabric connections, −3 to −5 µs fixed;
- BFP8 RS/AG payload, ~−5 µs, numerics change, so it needs the gate;
- start the RS per down-chunk, hiding it under the MLP tail.

**L1:** the one-shot reduction buffer is 160 KB per owner (> ~100 KB budget), so a streaming reduce (accumulate on
arrival) is required.

**Gate arithmetic with GDN_STEP_U1 (§14, −21 µs per u1 GDN layer):** u1 break-even AR rises from 12.2 to **~20.0 µs**;
u8 break-even stays 23.4 µs. So **~19 µs on-path AR (today's in-megakernel estimate) plus GDN_STEP_U1 passes both
gates**, with thin u1 margin. Persistent fabric connections add margin; BFP8 payload stays gated.

**Persistent AR on silicon (AR track, -mk-ar @4229b02c43f, after reset; bit-identical at every N):**

| ARs per program | µs per program | µs per AR |
|---|---|---|
| 1 | 23.4 | 23.4 |
| 2 | 41.4 | 20.7 |
| 4 | 77.6 | 19.4 |
| 8 | 149.9 | 18.7 |

- Marginal **18.07 µs per AR**; launch + setup ~5.3 µs.
- The earlier hang was the CB-wrap bug, now fixed. The single-AR sanity pass is clean, so this was not hardware.
- Per-iteration profile at N=8:
  - RS send ~8 µs (~9 GB/s per link-direction worker);
  - owners wait 15.7 µs for the 7 RS chunks; reduce 1.5 µs;
  - AG send ~2 µs; x ready at 17.8 µs.
- **On-path AR ≈ 18 + 3.2 split-K delivery ≈ 21 µs.** u8 passes (break-even 23.4). u1 with GDN_STEP_U1 is
  ~1 µs over its ~20 µs break-even.
- **AR work order (decided):**
  1. **Start the RS per down-chunk, under the MLP tail.** In v12 the owners sum column partials chunk by chunk, so
     each owner can start its RS as soon as its chunk is complete. That hides up to ~8 µs of RS send with no numerics
     change. Prototype now with a synthetic progressive producer (chunks released at the measured down cadence); same
     idea for the attention AR after wo.
  2. **Streaming reduce** (accumulate on arrival): required to fit the ~100 KB L1 budget at integration.
  3. Pipelined per-packet flush (rotating headers).
  4. BFP8 payload: flag only, gated.

**RS started per down-chunk (AR track, -mk-ar @a17e60b0376; synthetic producer, bit-identical on 8 devices):**
On-path = marginal per-iteration time − the 15δ release window.

| Release order | δ 0 | 0.5 µs | 1 µs | 2 µs |
|---|---|---|---|---|
| Chunk index, same on every device | 18.13 | — | 16.7-17.6 | — |
| **rot**: per device, farthest-owned chunks first, own chunks last | — | 11.08 | **8.97** | 8.70 |

**Pinned contract (v0.7) for every AR producer (v12 down stage, attention wo):** on each device, owners complete
their chunks in **descending ring distance from the chunk's owner device, the device's own chunks last**. This is a
per-device permutation of the owner (column-sum) completion order. Equivalent if owners can't be reordered: rotate
per device which K-row the down stage finishes first.

**Result:** on-path AR ≈ 9 µs + 3.2 µs split-K delivery ≈ **12 µs**. Projection (§8e layers):

| On-path AR | u8 (gate 13.7) | u1 (gate 9.4) | u1 + GDN_STEP_U1 |
|---|---|---|---|
| 12 µs | **12.23 ms** | **9.38 ms** (0.02 margin) | **~8.37 ms** |
| 15 µs | 12.62 ms | 9.76 ✗ | ~8.75 ms |

**Both gates pass at ~12 µs on-path AR. With GDN_STEP_U1, u1 has ~1 ms margin.** Remaining AR items: streaming reduce
(L1), rotating packet headers.

**AR slot L1 decision (v0.7):** no streaming reduce. The AR owners (10..11, 0..7) are not STREAM cores. They hold no
CB_X (320 KB), weight rings, CB_H or MLP buffers, so the ~100 KB AR budget (written for STREAM cores) does not apply.
Owner-core L1 map (~1.45 MB usable):

| Region | KB |
|---|---|
| **AR slot region, 8 sources × 20 KB, shared by attention-AR and MLP-AR** (sound by the AR dependency chain: attention-AR(L) writes only after MLP-AR(L-1) completes everywhere, MLP-AR(L) writes only after attention-AR(L) completes everywhere) | 160 |
| SDPA working set (NBB 1, BFP8), never aliased with AR slots | ≤300 |
| EXT projection ring (v0.5), capped on these 16 cores | ≤500 |
| CB_RSTD, AR flags, rstd partials, misc | ≤40 |
| **Total** | ≤1,000 (≥~400 KB headroom) |

The AR slots never overlap SDPA/EXT: a faster remote device can write attention-AR(L) while we are still in SDPA(L).
Alternatives not taken: credit-based streaming (adds a fabric round trip on the late sources); separate
attention-AR owners (needs a second in-row placement); BFP8 slots (gated).

**§15 amendments (v0.7.1):**
- *rot-order contract, clarified:* in the v0.6 down layout all 16 owners finish together, 2-4 µs after h lands. No K-row is delayed on purpose; that would add latency, not hide AR. The contract is now:
  - (a) Each owner releases (flag + multicast) the moment its own 10-partial sum is done, with no stage-end barrier.
  - (b) Within a K-row, the sub-block order of 2k / 2k+1 is a per-device runtime arg: the chunk whose RS destination is farther around the ring goes first.
  - (c) The AR side polls owner flags and issues RS sends in rot order, so among chunks that finish together, the farthest destinations go first.
  - The attention wo producer follows (a)-(c) the same way.
- *Owner-core L1, revised with the EXT x shard:*
  - The x shard on the ext cores (320 KB bf16, at the same lockstep address as the STREAM x) is separate from the AR slots. A faster remote device can write attention-AR(L) while we are still in qkv(L), so they cannot alias.
  - Owner-core totals: EXT ring 348 + SDPA ~250 + misc 28 + x 320 + AR 160 ≈ **1.11 MB of ~1.45 MB**. The 2-tile ext ring stays.

**§15 risk note (v0.7.2):** the down stage finishes all 16 owner chunks within ~1 µs, 2-4 µs after h lands (v12 report). So the progressive-RS overlap with the down stage is small. The RS is fabric-link bound (~200 KB per link direction, ~25 GB/s; header rotation gave 0 gain), so on-path AR may land between 12 and 18 µs, not at the ~12 µs of record.
Gate sensitivity (128 ARs per pass, ≈0.13 ms per µs of on-path AR):

| On-path AR | u8 (≤13.7) | u1 (≤9.4) | u1 + STEP_U1 |
|---|---|---|---|
| 12 µs | 12.23 | 9.38 | 8.37 |
| 15 µs | 12.62 | 9.76 ✗ | 8.75 |
| 18 µs | 13.01 | 10.15 ✗ | 9.14 |

**Consequence:** GDN_STEP_U1 is **required** for the u1 gate, not optional. u8 passes even with a fully exposed 18 µs AR.

**STEP_U1 silicon budget (v0.7.3):** 48 GDN layers per pass. At 18 µs on-path AR, u1 has 0.26 ms margin, so the full u1 op (prefix + step) must be **≤26.4 µs on silicon**. At 12 µs AR the limit is ≤42 µs. The ~21 µs figure is a sim/zone estimate; with the 1.3-1.5× sim optimism seen on MK1, silicon could land at 27-31 µs, which is marginal at the top of the AR band.

**§15 measured burst AR (v0.7.4, AR track, d26u02):** all 16 chunks released at t0, rot order. t0 → x ready on owners is **17.6 µs**, **≈20.8 µs with split-K delivery**. Index order is 19.1-19.8 µs, so rot order saves 1.5-2 µs and stays the default. Nothing hides behind the down stage.
- Sensitivity stays **1:1 (0.128 ms per µs)**. The pass is chain-bound, not DRAM-bound: the non-AR chain is ~167 µs/layer, while DRAM at peak BW is ~107 µs/layer. The writer already prefetches all of layer L+1's weights during layer L's stall. Deeper prefetch (L+2) would not shorten the chain, and there is no L1 for it.
- At 20.8 µs: u8 **13.36 ms** ✓ (0.34 margin); u1 + STEP_U1 (21 µs/op) **9.50 ms ✗**.
- **Plan:** the AR track builds a chain RS: two 1-hop chains per chunk (4 + 3 devices) with partial-sum forwarding, so link bytes per direction drop from ~200 KB to ~75 KB. The estimate is RS 6-7 + reduce 1 + AG 4 + delivery 3.2 ≈ **15 µs**. Target **≤15 µs, ideally ≤14**, including delivery. Possible hybrid: chain for far devices, direct sends for near ones.
- Equally valuable: every µs cut from the non-AR chain counts 1:1, e.g. STEP_U1 and attention qkv/wo.

**Joint u1 constraint (v0.7.5):** the u1 margin is shared between the AR and STEP_U1 (48 GDN layers, 128 ARs).
- STEP_U1 silicon budget by on-path AR: AR 14 µs → ≤37 µs; AR 15 µs → ≤34 µs; AR 18 µs → ≤26 µs; AR 20.8 µs → nothing passes.
- So the chain RS is the must-have. STEP_U1's ~21 µs sim estimate clears ≤34 µs even at 1.5× sim optimism.
