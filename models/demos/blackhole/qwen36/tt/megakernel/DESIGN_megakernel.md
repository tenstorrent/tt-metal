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
- `AR` (bit 4): all-reduce send/recv cores (AR track defines; receive writes x into STREAM cores' CB_X).

**Phase table** (L1, array of 16-u32 records, base address in runtime arg; `n_phases` in runtime arg):
| word | field |
|---|---|
| 0 | type: 1 PROJ, 2 MLP_GU, 3 MLP_DN, 4 GDN_STEP, 5 SDPA, 6 AR_SEND, 7 AR_RECV, 8 NORM, 9 STALL, 10 GDN_PREFIX |
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
| 12-15 | block-private (attention/GDN tracks define) |

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
