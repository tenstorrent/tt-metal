# Regime-A DRAM-optimal matmul — State of the Art & Analysis (2026-07-08)

Standalone summary of where the regime-A (M≪N, low arithmetic intensity) matmul stands, to inform the next task.
Vehicle: `tests/tt_metal/tt_metal/perf_microbenchmark/regime_a_mm/` (constant-input correctness `out==K`, device-
profiler kernel-time; env-gated experimental modes). Full chronological record: `SMALL_MT_IMPL_PLAN.md`.

## 0. The problem & the machine
- Compute `out[M,N] = in0[M,K] @ in1[K,N]`, regime A = **M ≪ N** (in1 is the big operand). Targets: FLUX.2 / LTX
  attention/MLP projections. Mt = M/32 tiles; skinny = Mt small (1–16).
- **BH p150b:** 8 DRAM channels, 11×10 compute grid (110 cores), 1.35 GHz, **DRAM read ~500 GB/s**, compute
  ~304 TFLOP/s (**2.76 TF/core**), roofline ridge = 608 FLOP/byte. Grid has a physical column gap (logical
  cols 0-6→phys x1-7, 7-10→phys x10-13; rows 0-9→phys y2-11).
- **Guiding principle: reader==consumer** — a core reads its big-operand slice from DRAM in large bursts AND
  computes it; peak read BW only happens when the reader is also the consumer (no forwarding funnel).
- **Roofline note:** AI_dom against the dominant operand simplifies to **min(M,N)** FLOP/byte (bf16). Ridge 608
  ⇒ a shape is DRAM-bound iff min(M,N) < ~518. BUT the *achievable* compute rate depends on K-block depth (see §6);
  at shallow-K the effective ridge drops to ~234, making high-Mt shapes compute-bound in practice.

## 1. Methods of in1 reading (in1 = [K,N], the BIG operand)
in1 is DRAM-interleaved but each core reads a **contiguous slice of its bank's shard** (bank b holds columns
[b·N_band : (b+1)·N_band] as [K, N_band] k-major).

| kernel / mode | pattern | when | note |
|---|---|---|---|
| `reader_sharded` | whole-bank contiguous, 16KB bursts, 2-3 rotating TRIDs | preaders=1 | peak single-reader read |
| `reader_subband` | N-sub-band, 1 packet/row (Nsb≤8 tiles), strided by N_band | N-split (P readers/bank) | contiguous within a row |
| `reader_ring` (contiguous) | full-N_band, 16KB bursts, **rotated K-shard order** (matches in0 ring) | K-split ring | in1[k] pairs with in0[k] |
| `reader_ring` (strided nsb) | Nsb tiles/k-row, stride N_band, rotated shards | large-Mt N-sub-block | needed to fit L1 |
| `reader_ring` (deep, in0direct/mshard) | one contiguous [Kt_local, Nsub] block/nb, natural K | deep-K path | pairs with contiguous in0 |
| in1 mcast/forward (milestone 2) | m=0 reader reads once + forwards to Sm-1 M-slaves (unicast or 1 mcast) | M-split | in1 read ONCE, shared |
| ablation `--skipin1` | reserve+push empty CBs, no DRAM | isolate compute | (deadlocks with M-split fwd) |

**Split-NOC multi-reader:** P readers/bank alternate NOC0/NOC1 (each near its per-NoC-optimal DRAM core) →
sustains ~500 GB/s to 96 cores vs ~390 all-NOC0. This is why regime A reaches near-peak read.

**KEY RESULT — in1 delivery is SOLVED / essentially FREE.** Ablation (N-slice, reader==consumer): compute-only
174µs vs in1+compute 179µs → in1 adds ~5µs, fully overlapped. Strided vs contiguous read = identical (the read
is per-RISC-issue / access-pattern bound, not layout-bound). in1 is never the bottleneck when read as a
contiguous per-bank slice by its consumer.

## 2. Methods of in0 reading (in0 = [M,K], the SMALL, SHARED operand)
in0 is shared across N (every output column needs all of in0), so the challenge is DELIVERY (a copy to every
core), not the DRAM read itself.

| kernel / mode | mechanism | read cost | delivery cost | verdict |
|---|---|---|---|---|
| per-core direct (`in0_writer`,`in0_reduce_writer`) | each core reads its in0 from DRAM | redundant if shared | none | fine when in0/core small (K-split) |
| broadcast (`--bcast` full / `--bstream` streaming) | 1-few loaders read once, **mcast** to all | once | **~13 GB/s (mcast tree-bound)** | ❌ REJECTED — fanout-independent, ~200µs |
| ring all-gather (`in0_ring_writer`) | 8-bank ring: each reads 1/8, rotates cyclically → full | once | short 8-hop, ~20-27µs | ✅ BEST general delivery |
| store-and-forward (`--fwd`) | 1 injector reads k-slice, unicast chain | once | chain | ~ring but no parallel read |
| global read-once ring (`--gring`) | ONE 8P-core ring, each reads 1/(8P) once | once (no redundancy) | 8P hops | ❌ divisibility → few cores → compute-starved |
| M-shard ring (`--mshard`) | shard in0 by **M**; contiguous cb0 (deep-K), read-once | once | 8-hop but SERIAL (cb0 pushed once) | deep-K but no overlap → loses |
| direct contiguous (`--in0direct`) | each reads full [M-block,k-slice] contiguous (deep-K) | 8× redundant | none | deep-K but redundant-read-bound |
| ablations | `--skipin0` (read free), `--skipfwd` (forward free) | — | — | isolate read vs forward |

**KEY RESULTS:**
- in0 **READ** is cheap (~1-6µs, overlapped) — proven by `--skipin0`.
- in0 **DELIVERY** (getting a copy to all cores) is the historical wall, but only for **N-slice** (which needs the
  *full* in0[M,K] per core). Delivering full in0 to ~96 cores is ~140-200µs by any method (broadcast 13 GB/s;
  per-group ring P× redundant; global ring divisibility-starved).
- The **ring all-gather** wins because it distributes BOTH the read (8 cores read 1/8 in parallel) AND the
  forwarding (nearest-neighbor short hops). in0 forwarding ≈ 20-27µs (real but small).
- **Broadcast is a dead end** (single-source mcast caps at ~13 GB/s regardless of loaders/receivers).

## 3. How in1 can be FRACTURED (partition the [K,N] read across cores)
- **N-slice** (across banks + sub-bands): each core owns a distinct N-sub-band `[K, N_band/Ns]`, reads it itself
  (reader==consumer, peak). **No reduction.** in1 delivery is free. THIS is the in1-optimal structure.
- **K-slice (split-K):** each core owns `[Kt/Pk, N_band]`. Requires reduction across the Pk partials. Used to
  add cores / share in0 cheaply.
- **N-sub-block (`nsb`):** a single core loops `N_bpc = N_band/Nsb` sub-blocks — shrinks the fp32 output
  accumulator (Mt·Nsb) to fit L1 for large Mt. (Serial N loop, not extra cores.)
- **Shared across M-slaves (mcast):** with M-split, the Sm cores that share the same [K, N-band] receive in1 via
  one reader + small forward (fanout Sm-1). mcast beats unicast at fanout≥3 (1.54× at Sm=4).

## 4. How in0 can be FRACTURED (partition the [M,K] operand)
- **K-slice (split-K):** each core reads/holds `in0[M, Kt/Pk]` (a k-slice) — SMALL per-core in0 (this is why
  K-split makes in0 cheap). Costs a Pk-deep reduction (serial-latency-bound: Pk12=+68µs, Pk6=+3µs).
- **M-split (`msplit` Sm):** each core owns an M-BLOCK `in0[Mt/Sm, K]` — shrinks in0/core AND (at fixed core
  count) lets you trade K-split depth for M-parallelism → SHALLOWER reduction. Costs some compute efficiency
  (smaller M-block: noreduce M16=181, M8=201, M4=313µs — big M-blocks compute better).
- **M-shard (within the ring):** shard `in0[M-block, k-slice]` by M across the 8 banks → contiguous [M, Kt_local]
  cb0 (enables deep-K), read-once. But cb0 pushed once ⇒ serializes with compute.
- **Full (N-slice):** each core needs the ENTIRE in0[M,K] (shared across all N it doesn't split) — the expensive
  delivery case; only viable if in0 is small (Mt=1/2) or read-once via ring.

## 5. Results by Mt

### Mt=1 (M=32) — GEMV-like skinny — SOLVED, WINS 1.14–1.49×
Best: **ring K-split (Pk 2–3, 16–24 cores), in0 read on the in1 RISC (in0=same) or 2nd RISC (in0=other)**, contiguous
in1. Util 72–97% (90-98% of DRAM cap on deep-K shapes). Examples: 32×2048×512 = 71.7%/1.49×, 32×6144×1536 =
95%/1.27×, 32×6144×6144 = 97.4%/1.27×. Very-large-N (9216+) or K-too-shallow (Kt=8): N-split + in0-broadcast.

### Mt=2 (M=64) — WINS 1.19–1.26×
Best: **ring K-split (Pk 2–3)**, util 89–95%. Large-N (9216): N-split + in0-broadcast (ring OOMs). 64×6144×1536 =
91.5%/1.19×, 64×6144×4608 = 89.2%/1.20×.

### Mt=4 (M=128) — boundary of regime A (AI 108–124)
Historically the gap (0.3–0.9× branch): 128×2304×6144 = 63%/0.88×, 128×6144×768 = 47%/0.77×. Higher AI → less
DRAM-bound; the same large-Mt levers (2D M-split, deep-K) apply but weren't re-swept post-fixes. Likely lands
mid-pack; needs a targeted sweep with the current toolkit.

### Mt=8 (M=256) & Mt=16 (M=512) — the LARGE-Mt regime — MUCH IMPROVED, not yet roofline
The journey (512×3072×6144, AI 410, DRAM roofline 94µs):
1. Original nsb ring, kb=1: **30–41% util** (OOM'd or fell back; 205µs on this shape).
2. **2D M-split** (K-split × M-split): fixes OOM + trades reduction depth for M-parallelism. Reduction is
   depth-bound (Pk12 = +68µs, Pk6 = +3µs); M-split cures it. Sweet spot **Pk6/Sm2** ≈ 204µs, 1.26× over pure
   K-split. First scheme to beat the plain ring at Mt=16. in1 shared via small mcast (fanout Sm).
3. **Deep-K blocking** (kb = ring-max = Kt_local/8): the biggest single lever. On real FLUX K=6144:

   | shape | best kb | deep-K (util) | kb=1 (util) | speedup |
   |---|--:|--:|--:|--:|
   | 512×6144×4608 | 4 | 246µs (55%) | 303 (45%) | 1.23× |
   | 512×6144×2304 | 4 | 169µs (44%) | 196 (38%) | 1.16× |
   | 512×3072×6144 | 2 | 178µs (53%) | 205 (46%) | 1.15× |

   **Net large-Mt: ~44–55% util** (up from 30–41%), best config = **M-split Pk6/Sm2, kb = Kt_local/8, unicast
   bank-adjacent**. Still ~2× off the DRAM roofline. LTX 256×2048×1024 (Mt=8) similar trend.
- Not reached: DRAM-bound (roofline). The 8-bank ring caps kb ≤ Kt_local/8 (= semi-deep, e.g. 4); the ~90%
  compute floor needs kb ≥ 16 (SP2), which requires either thin-M (Sm4 ⇒ worse compute) or Pk1 (OOM). Deep-K
  delivery alternatives (mshard serial / direct redundant / overlapped-mshard) each serialize, read redundantly,
  or OOM (see §6).

## 6. How to keep compute fed (the current frontier)
The `--skipin0/--skipin1/--skipfwd/--noreduce` ablations decomposed the large-Mt cost (Pk6/Sm2, 96c):
- in0 read ≈ 2-9µs (overlapped), in0 forward ≈ 20-27µs, in1 delivery ≈ free/5µs, reduction ≈ 3µs (at shallow
  Pk6). **So delivery + reduction are all small; the dominant residual is the COMPUTE itself.**
- **Compute is UNPACK-bound, and the lever is K-block DEPTH** (SP2_compute_floor_findings.md): the compute engine
  reaches ~2.4–2.5 TF/core (**90%**) only with **deep K-blocks (kb ≥ 16, ideally 32–64)**; shallow kb collapses it
  (kb=1 → ~40%). The per-output-block pack + data-format reconfig must be amortized over a deep matmul. It is NOT
  fidelity-bound (LoFi == HiFi2 == 167µs at kb=1) nor subblock-bound (2×1 ≈ 2×2 ≈ 4×1). Confirmed directly:
  kb=1→2 = 1.30× pure compute (167→128); kb=16 = 108µs.
- **The central conflict blocking DRAM-bound at large Mt:** deep-K, ring-overlap, L1-fit, and read-once are
  mutually exclusive with the current architecture:
  - the ring overlaps ONLY by sharding the K (reduction) dim (commutative) ⇒ **shallow kb** (≤ Kt_local/8);
  - deep-K needs the full k-slice contiguous *before* the matmul ⇒ SERIAL (mshard) or 8× REDUNDANT (direct);
  - deep-K at Pk1 (no reduction) ⇒ full-K in0+in1 resident ⇒ OOM (Kt ≳ 64).

**To keep compute fed / reach the roofline (ranked next moves):**
1. **bf8 inputs** — halves unpack bandwidth ⇒ deep-K compute ~2× ⇒ below the DRAM roofline (SP2: ~85% util at
   bf8). Biggest single lever; it's an input-precision decision (analogous to disallowed bf16-accum).
2. **Deep-K-capable delivery redesign** — decouple compute-kb from the 8-shard ring without the full-k-slice OOM.
   The reduction-aware overlapped M-shard (`--moverlap` prototype: streaming M-shard ring + compute `IN1_RESIDENT`
   + M_blocks_per_core=G) is the intended vehicle, but currently (a) OOMs on Kt≥96 (full-K residents) and
   (b) hangs (credit deadlock). Needs Pk>1 (to shrink residents) + reduction + deadlock fix.
3. **Right-size cores per shape** (SP2): most low-AI shapes are READ-set (need only 16–32 cores); using fewer
   cores cuts NoC pressure and keeps reader==consumer clean. Large-Mt is the exception (compute-set → wants ~96+).

## 7. Env-gated modes in `regime_a_mm` (all correctness-checked; default path = plain ring)
`--ksplit Pk` `--msplit Sm` `--nslice Ns` `--ring` `--nsb Nsb` `--kb` `--in1mcast` `--in0direct` `--mshard`
`--moverlap`/`--modepth` `--gring`/`--nsring`/`--nsdepth` `--bstream`/`--bcast`/`--nloaders`/`--bdepth`/`--bcontig`
`--nosplit` `--rect` `--fwd` `--chain` `--in0risc` `--in0order`; ablations `--skipin0 --skipin1 --skipfwd
--noreduce --lofi --sbh --nsbcontig`. Kernels: `reader_{sharded,subband,ring}.cpp`, `in0_{writer,reduce_writer,
fwd_reduce_writer,ring_writer,nsring_writer,mshard_overlap_writer,bcast_loader,bstream_loader}.cpp`, reused
`minimal_matmul/device/kernels/compute.cpp` (flags `REDUCE_K`, `IN0_KSLICE_RESIDENT`, `IN1_RESIDENT`).

## 8. One-paragraph bottom line
Regime A splits cleanly into two regimes. **Mt=1/2 (and much of Mt=4): SOLVED** — ring K-split delivers in0
read-once + reader==consumer in1, hitting 72–97% util and 1.14–1.49× over the branch; in1 read is free, in0
read is cheap, delivery is a short ring hop. **Mt=8/16 (FLUX M=512, LTX M=256): substantially improved but not
finished** — the 2D M-split (shallow reduction) + deep-K blocking lifts util from 30–41% to 44–55% (~1.2–1.3× over
naive), and every DELIVERY cost (in0 read/forward, in1, reduction) is now small. The remaining ~2× to the DRAM
roofline is a pure **compute-feed** problem: bf16 unpack efficiency needs deep K-blocks (kb≥16), which the
8-shard ring architecturally caps at ~4. The two ways through are **bf8 inputs** (halve the unpack) or a
**deep-K-capable in0 delivery** that keeps read-once + overlap + L1-fit simultaneously (the unsolved redesign).
