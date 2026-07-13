# Multicast-forward feasibility for Mt>1 DRAM-BW-optimal matmul (Blackhole p150b)

**Question.** For Mt>1 we need >8 compute cores but only 8 bank-adjacent readers hit DRAM peak
(1 reader/channel = ~494 GB/s; multi-reader/bank drops to ~440). Hypothesis: keep the 8 peak readers,
and have each reader **multicast** its in1 stream (all Mt workers of a band need the SAME in1) to K worker
cores on the **idle NoC (NOC1)**. Per-reader forward requirement = 494/8 ≈ **62 GB/s**; one NoC link ≈ 64.
Does an optimized multicast forward sustain it?

**Verdict: NO.** The multicast forward does **not** keep the read at peak. Aggregate read-through funnels
to **~149 GB/s (K=2)** and **~130 GB/s (K=4)** — ~30% and ~26% of the 454-494 read ceiling. Two
independent mechanisms kill it: (1) per-core mcast egress *degrades with K* (it is NOT free router
replication), and (2) 8 concurrent multicasts on NOC1 contend severely, so aggregate egress collapses
to ~145 GB/s (≈18 GB/s/core, vs 68 isolated).

Hardware: BH p150b, 8 DRAM channels, grid 11×10, 1.3498 GHz, bf16 tiles (2048 B). Kernel-time BW via
device profiler (`tools/mm_sweep/parse_kernel_bw.py`, FREQ=1.35e9). Code: `sp_forward/`.

---

## Read ceiling anchor (this exact shard, no forward)
8 bank-adjacent readers, 54 MB (Kt=192 × N_band=18 × 8 banks), contiguous, `sp_bankread` contig depth=8:
**454 GB/s** (BRISC kernel-time). (Peak one_packet+triple-TRID pattern reaches ~494; 454 is the plain
double-buffer probe on this shard. Either way the read ceiling is ~450-494.)

## EXPERIMENT 1 — single-core L1→L1 multicast egress ceiling (NOC1)
One source core multicasts a 32 MB logical stream to K receivers (contiguous valid worker rectangle).
Egress BW = logical_bytes / NCRISC-kernel-time (counted once — routers replicate). Selected results:

| K | chunk | depth | egress GB/s |
|---|-------|-------|-------------|
| 1 | 8 (16KB) | 8 | **72.2** |
| 2 | 8 | 8 | **68.3** |
| 4 | 8 | 8 | **61.9** |
| 8 | 8 | 8 | **52.4** |
| 1 | 4 (8KB) | 8 | 63.5 |
| 1 | 8 | 4 | 67.3 |
| 1 | 1 (2KB) | 1 | 11.9 (worst) |

Findings:
- **Bigger chunk (16KB) and deeper pipeline (depth 8) are essential** — 2KB/depth-1 is ~12 GB/s; 16KB/
  depth-8 is ~6× higher. Chunk matters more than depth.
- **Egress is NOT independent of K** (contradicts the "routers replicate for free" assumption). It drops
  monotonically: 72 → 68 → 62 → 52 for K = 1 → 2 → 4 → 8. More destinations cost the source more egress.
- **Isolated, one core meets ~62 only for K≤4** (K=2: 68 ✓, K=4: 62 borderline, K=8: 52 ✗) — and only at
  the best chunk/depth, leaving essentially **zero headroom** for the concurrency losses seen in Exp 2.

## EXPERIMENT 2 — full forward read-through (decisive)
8 bank-adjacent readers each read their 6.75 MB shard at the peak pattern on **BRISC/NOC0** and multicast
every block to K workers on **NCRISC/NOC1**, pipelined through an L1 CB (cb0). Workers receive-only.
Aggregate BW = 54 MB / max-core kernel time. `block` = mcast chunk (tiles), `cbd` = cb0 depth (blocks),
`md` = mcasts in flight before barrier.

| K | block | cbd | md | READ (BRISC) GB/s | MCAST (NCRISC) GB/s |
|---|-------|-----|----|-------------------|---------------------|
| 2 | 8  | 8  | 4 | 133.4 | 132.4 |
| 2 | 8  | 16 | 8 | 140.5 | 139.0 |
| 2 | 16 | 8  | 8 | 144.7 | 143.5 |
| 2 | 16 | 16 | 8 | **148.9** | 145.5 |
| 4 | 8  | 8  | 4 | 117.6 | 116.8 |
| 4 | 16 | 8  | 8 | 126.4 | 125.2 |
| 4 | 16 | 16 | 4 | **129.7** | 120.8 |
| 4 | 16 | 16 | 8 | 129.4 | 125.9 |

Findings:
- **Best read-through: K=2 → 148.9 GB/s, K=4 → 129.7 GB/s.** vs 454-494 ceiling → the read funnels to
  **~30% (K=2) / ~26% (K=4)**. Nowhere near sustaining peak, and far below even the ~440 multi-reader
  ceiling the forward was meant to beat.
- **READ time ≈ MCAST time in every single config**, and READ BW *rises monotonically as mcast params
  improve* (block 8→16, cbd/md up). This proves the read is **entirely mcast-backpressure-bound** (CB
  backpressure), not read-bound — the funnel is caused by the forward, unambiguously. (Standalone the same
  shard reads at 454.)
- **Aggregate mcast collapses under concurrency.** Single-core isolated egress was 68 (K=2)/62 (K=4).
  Eight concurrent multicasts on NOC1 give only ~145 (K=2)/~125 (K=4) aggregate ≈ **~18 GB/s per core** —
  a ~3.8× per-core degradation. 8 simultaneous multicasts contend on shared NOC1 links; NOC1 does not
  supply 8×62 ≈ 496 GB/s of multicast transport for this pattern.
- Larger block (16 tiles = 32 KB) and deeper cbd/md help a little (~+12% K=2) but cannot close a 3.3×gap.

## Why it funnels (mechanism) — see Experiment 3 for the isolation
NOTE: my first hypothesis here (NOC1 transport contention) was WRONG — Exp 3 disproved it. The real
mechanism:
1. **Producer/consumer coupling.** The reader (producer) is CB-backpressured to the mcast (slow consumer);
   deeper credits don't help. Decoupled, the read runs at full 509 GB/s (Exp 3b).
2. **On-core L1 arbitration starves the mcast.** A core doing a heavy NOC0 DRAM read (~64 GB/s/core of L1
   write-landings) starves the NOC1 mcast's L1 source-reads down to ~18 GB/s/core (vs 68 isolated) — the
   read wins the L1/NIU arbitration (asymmetric). This is NOT NoC-fabric contention: 8 concurrent mcasts
   with no read scale perfectly to 544 GB/s (Exp 3a).
3. **Per-core mcast egress degrades with K** (Exp 1) — a secondary effect (62-72 for K=1..4).

## Conclusion
The multicast-forward *as tested* (one core both reads DRAM and multicasts) is **not viable** — it caps
read-through at ~149 (K=2)/~130 (K=4). But the cause is **on-core read↔send L1 arbitration + CB coupling**,
NOT the NoC fabric and NOT the multicast: the fabric sustains 544 GB/s of concurrent mcast (Exp 3a), and a
reader core reads at 509 while an mcast coexists (Exp 3b). So the *idea* is not dead — the *co-residency*
is. Path forward = separate reader cores from sender cores, or worker-pull (see Experiment 3 consequence).
Compared to alternatives, single-core-does-both (146 delivered) still loses to the branch distributed read
(~369) and DRAM-sharded (~450), so it must be restructured to compete.

## EXPERIMENT 3 — mechanism isolation

### 3a. NOC1 egress under concurrency, NO DRAM read (`test_egress_concurrency`)
nsrc source cores each drive a logical stream (mcast to K, or unicast to 1) on NCRISC/NOC1, chunk=16KB,
depth=8, no reads:

| mode | nsrc | K | aggregate GB/s | per-core | scaling |
|------|------|---|----------------|----------|---------|
| unicast | 1 | 1 | 76.8 | 76.8 | — |
| unicast | 8 | 1 | **614.3** | 76.8 | 8.0× perfect |
| mcast | 8 | 1 | **577.3** | 72.2 | 8.0× |
| mcast | 8 | 2 | **544.5** | 68.1 | 8.0× |
| mcast | 8 | 4 | **494.6** | 61.8 | 8.0× |

**NOC1 fabric does NOT contend** — 8 concurrent mcasts scale perfectly linearly. Unicast == mcast here, so
multicast replication is cheap. The Exp-2 collapse is therefore NOT fabric contention.

### 3b. Decoupled read+mcast on the SAME cores (`test_forward_readthrough --decouple 1`)
8 reader cores each read their shard (BRISC/NOC0) AND mcast (NCRISC/NOC1), decoupled: reader uses a private
ring (no cb backpressure), mcast sends a static L1 buffer (no cb_wait). block=16, cbd=16, md=8:

| config | READ (BRISC) GB/s | MCAST (NCRISC) GB/s |
|--------|-------------------|---------------------|
| K=2 coupled (cb)  | 148.3 | 144.1 |
| K=2 **decoupled** | **509.1** | 145.9 |
| K=4 coupled (cb)  | 130.8 | 127.1 |
| K=4 **decoupled** | **509.6** | 127.2 |

1. **The read is not the victim** — decoupled it runs at full 509 GB/s *while* an mcast runs on the same
   core. The CB coupling is what dragged it to 148 (reader stalls at cb_reserve behind the slow consumer,
   even at cbd=16). Funnel = producer/consumer coupling.
2. **The read starves the mcast** — the mcast that does 544 isolated (3a) does only 146 co-resident with a
   DRAM read (3b), asymmetrically (read unhurt). = on-core L1/NIU arbitration: NOC0 read-landings win over
   NOC1 mcast source-reads. NOT fabric, NOT the read itself.

### Consequence for synchronization
Credits vs synchronous handshake does not help: the CB is already a credit scheme, and cbd 8→16 barely
moved the number because the limiter is a *throughput* cap (co-resident mcast ~146), not credit depth.
Structural fixes only: (i) separate reader cores from sender cores, or (ii) **worker-pull** — readers only
read into L1 (509), workers NoC-read the shared in1 straight from reader L1 (remote-CB style), charging the
L1 read-out to the workers, not the reader. Worth prototyping next.

### Caveats
- Multicast on NOC1 needs the rectangle corners passed SWAPPED (start=max, end=min); DYNAMIC_NOC flips
  them into valid ascending order. A min-corner start hangs multi-core mcast on NOC1 (single-core masks it).
- Data is constant/garbage (BW test, data-independent). Mcast confirmed real: Exp-1/3a timing scales with
  K/chunk/depth/nsrc; Exp-2 read BW tracks mcast params 1:1; decouple flips read 148→509.

## EXPERIMENT 4 — worker-pull (`test_worker_pull`)
8 bank-adjacent readers stream their shard into a D-deep L1 ring (BRISC/NOC0) and ONLY signal readiness
(multicast valid semaphore). Their K workers NoC-read (pull) the shared in1 straight out of the reader's
L1 (NCRISC/NOC1) and credit-return freed slots. Goal: move the L1 read-OUT off the reader onto the workers.
Aggregate BW = 54 MB (unique in1) / max-core kernel time; read (BRISC) and delivery (NCRISC) reported.

| K (workers/reader) | READ = DELIVERY (GB/s) |
|---|---|
| 1 | **168** (flat vs block 8/16/32, ring 8-32, WD 8-16) |
| 2 | 114 |
| 3 | 69 |
| 4 | 45 |
| 5 | 42 |
| 6 | 34 |
| 7 | 29 |
| 8 | **27** |

**Worker-pull is the worst option and degrades with K — the opposite of what we need.** Two mechanisms:
1. **K× L1-read amplification at the reader.** All K workers of a band read the *same* in1 out of one
   reader's L1, so the reader's L1 must egress K× the shard (mcast avoids this via router replication;
   pull does not). The reader's L1 read-port saturates → read/delivery ∝ ~1/K.
2. **Low per-worker remote-L1 read even at K=1** (~168 aggregate = ~21 GB/s/core), insensitive to pipeline
   depth (WD) or block size — L1→L1 NoC read is issue/latency-bound per core, well under the reader's
   509. And the reader is credit-coupled to it, so the read funnels to the pull rate.

Net vs alternatives (unique-in1 delivery): worker-pull K=1 **168** / K=2 **114** < mcast-forward **146** <
branch distributed read **369** < DRAM-sharded **450**. Worker-pull loses to everything and is the only
one that gets worse with more workers.

## Overall verdict across all fan-out schemes
Fanning ONE reader's in1 shard out to K workers on-chip funnels the read regardless of mechanism:
- **mcast-send** (reader reads DRAM + sources the L1→L1 send): on-core L1 arbitration starves the send to
  ~18 GB/s/core → ~146 aggregate.
- **worker-pull** (workers read the reader's L1): K× L1-read amplification + low per-core remote read →
  168 (K=1) falling to 27 (K=8).
The NoC fabric itself is NOT the limit (8 concurrent mcasts hit 544; a lone reader hits 509 with an mcast
coexisting). The limit is **on-chip L1 fan-out bandwidth from a single producer core**. Recommendation for
Mt>1: do NOT centralize the in1 read then fan it out. The winning approach is Experiment 5 below.

## EXPERIMENT 5 — multi-reader per bank (THE SCALABLE Mt>1 SOLUTION) (`test_multireader`)
Give each core its OWN unique 1/(8P) slice of in1 to read (reader==consumer, the M=1 winning pattern) and
scale the number of readers. P readers/bank × 8 banks. Lever = which NoC each reader uses. **Key HW fact:
each BH DRAM channel has multiple NoC subchannel endpoints, and NOC0 vs NOC1 reach DIFFERENT endpoints with
opposite routing.** Readers placed near the per-NoC optimal core. Aggregate = 54 MB / max-core time, block=16.

| P | reader cores | noc0 (all NOC0) | noc1 (all NOC1) | **split (NOC0+NOC1)** | dual (2 RISC/core) |
|---|---|---|---|---|---|
| 1 | 8  | 509 | 342 | — | — |
| 2 | 16 | 390 | 327 | **509** | 509 (8 cores) |
| 3 | 24 | 428 | — | **509** | — |
| 4 | 32 | 403 | — | **509** | 510 (16 cores) |
| 6 | 48 | — | — | **509** | 509 (24 cores) |
| 8 | 64 | — | — | **503** | 509 (32 cores) |
| 9 | 72 | — | — | **498** | — |
| 12 | 96 | — | — | **503** | 507 (48 cores) |

**Split-NOC is the answer.** Readers on BOTH NoCs (equal volume → different subchannel endpoints) sustain
the DRAM cap (~503-509 = 98-99% of 512) while scaling reader cores from 8 to **96**. Each core reads a
unique in1 slice and computes its own M-slice → arbitrary compute-core count for Mt>1 with NO on-chip fan-out.

Mechanism:
- **Same-NOC multi-reader contends**: P=2 all-NOC0 = 390 (< P=1's 509). Two streams to the SAME subchannel
  endpoint arbitrate poorly (this is the prior "2 readers/channel doesn't scale" result — a *same-endpoint*
  problem, not fundamental).
- **NOC1 alone is weaker** (342 vs 509) — matters only when unbalanced; in split both NoCs carry half the
  volume and the aggregate is DRAM-channel-limited, so both hit ~509 together.
- Ultimate cap = **DRAM channel bandwidth** (~64 GB/s/channel), reached in every split config.

**split vs dual:** *split* (recommended for Mt>1) uses ONE DM RISC/core for the read, leaving the other DM
RISC + TRISC to compute → every one of the 8P cores is a full compute worker. *dual* uses both DM RISCs to
read (509 with half the cores) but leaves only TRISC for compute.

### Recommendation for Mt>1
**Split-NOC multi-reader.** Pick P so 8P ≈ desired compute-core count (validated 16→96 cores at ~500 GB/s);
assign readers alternately to NOC0/NOC1, each near its per-NoC optimal core, each reading a unique 1/(8P)
contiguous slice of in1. Keeps the peak read AND supplies the extra compute cores Mt>1 needs — solving the
problem all fan-out schemes could not. (Ranking recap: worker-pull 168/K, mcast-forward 146, branch 369,
DRAM-sharded 450, **multi-reader split ~503-509 with up to 96 cores**.)

## in0 HANDLING for split-NOC multi-reader (regime A) — design ideation
Partition is **N-parallel**: each of the 8P cores owns a unique N-band of in1 (full K) and computes
`out[:, N-band] = in0[Mt,Kt] @ in1[Kt, N-band]`. Therefore **every core needs the FULL in0[Mt,Kt]** → in0
is a BROADCAST, not a partition.

Traffic ratios (the governing fact): in0_total = Mt·Kt tiles, in1_total = Kt·N tiles.
- Broadcast in0 (read once, replicate): DRAM & NoC load = **Mt/N** of in1 → negligible in regime A (N huge).
- Per-core redundant in0 DRAM read (each core reads its own copy): **8P·Mt/N** of in1 — scales with P and
  bites for smaller-N regime-A (e.g. N=1152t, P=8, Mt=1 → +5.6% DRAM stolen from in1). AVOID.
So the whole game is: broadcast in0 once; never let 8P cores each read it from DRAM.

Recommended designs (ranked):
1. **Dedicated in0-loader core(s) + multicast (top pick).** 1-2 cores that do nothing but read in0 once
   from interleaved DRAM (keep in0 interleaved, it's tiny) and multicast it to the compute cores. The 8P
   in1-readers stay purely on the peak contiguous pattern and never touch in0 DRAM.
   - 1a. **Up-front (prologue)** if Mt·Kt fits L1 (small K/Mt): one mcast before in1 streaming; in0 resident
     in L1, reused across the whole matmul; ZERO contention with the in1 stream. Hide the ~few-µs prologue
     behind the in1 readers ramping their CBs. Simplest.
   - 1b. **K-block streamed / double-buffered** for large K (in0 doesn't fit L1): loader reads+mcasts one
     in0 K-block ahead of compute; each K-block is reused across the core's whole N-band. Total in0 NoC
     load still Mt·Kt (once); overlaps in1.
2. **Deliver in0 on each core's IDLE NoC.** In split mode ~half the cores read in1 on NOC0, half on NOC1.
   Broadcast in0 to the NOC0-in1 group over NOC1, and to the NOC1-in1 group over NOC0 (two mcasts). This
   puts in0 on the NoC each receiver is NOT using for in1 → zero in0/in1 NoC contention. The two groups are
   already distinct clusters (near opt0 vs opt1). Since in0 is Mt/N tiny, this is belt-and-suspenders, but
   free to do.
3. **Mind the BH mcast rectangle gotcha** (non-contiguous physical grid, x=8-9 gap): the in0 broadcast must
   target contiguous valid worker rectangles — mcast in two column-bands (logical cols 0-6 and 7-10), or
   lay compute cores within one band. (Same corner-swap rule for NOC1: start=max, end=min.)

Why in0 is not a limiter: it is reused across each core's entire N-band (huge), so once resident/prefetched
one K-block ahead, compute stays in1-bound. With broadcast delivery, in0 adds only Mt/N of DRAM+NoC load —
independent of P — so it stays non-limiting at any regime-A shape and any reader count.

Alternatives considered / rejected: per-core in0 DRAM read (8P× redundant, not P-scalable); split-K to make
in0 unique-per-core (adds an output cross-core reduction — unnecessary, since broadcast in0 is already
negligible).

## EXPERIMENT 6 — in0 concurrency, MEASURED (`test_in0_concurrent`)
Split-NOC in1 readers (8P, ~505) + L dedicated loader cores reading in0 interleaved, K-block streamed
(Kb=8), with pipelined read(BRISC)∥mcast(NCRISC). Test shape: K=6144, N=4608 (144 N-tiles), so in0/in1 =
Mt/N-tiles = Mt/144 (a small-N regime-A case → pessimistic dent; larger N shrinks it). in1_reader_BW =
in1 BW with loader cores EXCLUDED (isolates real in1 dent from loader time).

| config | in1_reader_BW | **AGG DRAM (in0+in1)** | agg vs 503 |
|---|---|---|---|
| baseline (no in0) | 503 | **503** | — |
| Mt=1 read-only (L=1) | 485 | **488** | −3% |
| Mt=1 mcast NOC0 (L=1) | 483 | **487** | −3% |
| Mt=1 mcast NOC1 (L=1) | 483 | **486** | −3% |
| Mt=4 read-only (L=1) | 436 | **448** | −11% |
| Mt=4 mcast NOC0 (L=1) | 434 | **447** | −11% |
| Mt=4 mcast NOC1 (L=1) | 436 | **448** | −11% |
| Mt=4 mcast NOC0 (**L=2**) | 467 | **480** | **−5%** |
| Mt=4 mcast NOC1 (L=2) | 460 | **472** | −6% |
| Mt=4 mcast NOC1 (L=3) | 438 | **450** | −11% (over-contends) |
| P=12 Mt=1 mcast (L=1) | 476 | **479** | −5% |
| P=12 Mt=4 mcast (L=2) | 450 | **462** | −8% |

**Aggregate DRAM (in0+in1) is what matters, and the contention is LOSSY — total drops below the 503
in1-only peak, it is not free redistribution.** Reason: the in0 read is INTERLEAVED (less DRAM-efficient
than the contiguous DRAM-sharded in1 read), so DRAM cycles diverted to in0 yield less than peak → total
falls. Magnitude: Mt=1 → ~487 (−3%); Mt=4 → ~448 (L=1) recovered to **~480 (L=2, the sweet spot)**; L=3
over-contends (450). NoC choice still ~irrelevant for the aggregate. **Lever to shrink the loss further:
make the in0 read more DRAM-efficient** (DRAM-shard / contiguous in0, or fewer larger interleaved bursts)
so the loaders read closer to peak-per-bank — currently the interleaved access is the source of the loss.

Findings (decisive):
1. **The broadcast is FREE and the NoC choice is irrelevant.** mcast NOC0 == NOC1 (481 vs 482; 435 vs 436),
   and mcast ≈ read-only (the mcast adds ~0 on top of the read). → **Idea 1 (dedicated loader + simple
   single-NoC K-block-streamed broadcast) is correct; Idea 2 (idle-NoC split delivery) is unnecessary
   complexity.** (Consistent with Exp 3a: the mcast fabric has headroom.)
2. **1-2 loader cores are fast enough.** The loader delivers all of in0 within 72-76% of the in1 read time
   even at P=12 (96 in1 cores) — it is never the bottleneck. (The earlier "34% mcast dent" was a strawman
   sequential-barrier loader; a pipelined read∥mcast loader removes it.)
3. **The only cost is DRAM-bank contention from the in0 READ.** in0 is interleaved → it touches the same 8
   channels in1 saturates, so a concurrent in0 read dents in1 ~Mt/N-ish (amplified ~5× at small N because
   any extra reader disrupts the peak in1 pipeline). Mt=1: −5%; Mt=4: −14% (L=1). **Two mitigations that
   work**: (a) use L=2 loaders → shorter contention window, dent halves (−14%→−8% at Mt=4); (b) larger N
   (real regime-A) shrinks it since in0/in1 = Mt/N. For small in0 that fits L1, an up-front prologue read
   (before in1 ramps) avoids the concurrent dent entirely — optional refinement.

### Final in0 recommendation
**Dedicated loader core(s) + K-block-streamed broadcast on either NoC (Idea 1).** Use L=1 for Mt=1, L=2 for
Mt≥4. Pipeline the loader's read (BRISC) and mcast (NCRISC) so it finishes fast and its DRAM-contention
window is short. Do NOT bother with idle-NoC delivery (Idea 2) — measured identical. Residual in1 dent is
~5% (Mt=1) to ~8% (Mt=4, L=2) at N=4608, shrinking as N grows; acceptable and far better than any per-core
in0 DRAM read.

## EXPERIMENT 7 — Mt scaling: when does the in0-broadcast design break? (`test_in0_concurrent`)
Swept Mt with split-NOC in1 (P=8) + K-block-streamed in0 broadcast. AGG = (in0+in1)/wall. Test N=144 tiles.

| Mt | read-only AGG (L=2) | mcast AGG (L=2) | mcast − read |
|----|--------------------|-----------------|--------------|
| 4  | 479 | 473 | ~0 |
| 8  | 434 | 340 | **−22%** |
| 16 | 367 | 218 | **−41%** |

Reducing P at fixed Mt (mcast): recovers BW as the broadcast burden falls —
| Mt=16 | P=8 (64c) | P=4 (32c) | P=2 (16c) | P=1 (8c) |
|---|---|---|---|---|
| AGG | 218 | 224 | 243 | 324 |
| burden 8P·Mt/N | 7.1 | 3.5 | 1.7 | 0.8 |

**Root cause = the in0 BROADCAST, not the in0 read.** read-only stays high at Mt=16 (367); the mcast is what
collapses (218). Contiguous (DRAM-sharded) in0 read does NOT help (it's not row-buffer thrashing). The
governing quantity is the **per-core in0 burden = 8P·Mt/N**: each of the 8P cores receives the FULL in0
(Mt·Kt) but reads only 1/(8P) of in1, so aggregate on-chip in0 L1-write traffic = 8P·Mt·Kt vs in1 = N·Kt.
The broadcast stays subordinate only while **8P·Mt ≲ N**. For N=144, P=8 that is Mt ≲ ~2; Mt=4 is fine,
Mt=8/16 blow up. It is NOT fixable by L (loaders) or by making in0 contiguous — it's fan-out volume.

**Answer to "issues at Mt>4?":** Yes, and it's not just a parameter change. The 1D N-parallel + broadcast-
in0 scheme is a **regime-A design** (valid while 8P·Mt ≲ N). Past that:
- Lowering P recovers BW (confirms the formula) but sacrifices the compute cores you added Mt for — a real
  tension: broadcast volume = 8P·Mt·Kt grows with BOTH core count and Mt.
- The proper fix is a **different partition: 2D M×N** (M-partition into Mg groups so each core gets only
  Mt/Mg of in0 → burden ÷ Mg). The cost is in1 must then be shared/re-read across the Mg M-groups (either
  redundant in1 DRAM reads or in1 fan-out — the very problem Exp 2/4 studied). That is acceptable at large
  Mt because you're **compute-bound there** (compute ∝ Mt; in1 read is fixed), so the read no longer needs
  to be at peak. In short: small Mt → 1D N-parallel multi-reader (this work); large Mt → standard 2D matmul.
- The crossover Mt scales as **N/(8P)**, so genuine low-AI regime-A shapes (small Mt, large N by definition)
  sit comfortably in the sweet spot; the break only appears when you push Mt out of regime A.

## PROJECTED SPEEDUP vs branch sliced-matmul on the regime-A FLUX/LTX shapes (bh_skinny_results.md)
These shapes are DRAM-bound (AI ≪ ridge 608), so runtime ≈ traffic / achieved_BW. Both variants move the
same minimal traffic (the multi-reader reads in1 EXACTLY once — no K-par/slicing re-reads), so
**projected speedup = new_BW% / branch_BW%** (branch_BW% = "br BW%" column). new_BW% is grounded in Exp5/6
(measured at K=6144, N=4608): Mt=1 ≈ 96%, Mt=2 ≈ 92%, Mt=4 ≈ 87% (large N); it degrades when the in0
broadcast burden 8P·Mt/N grows (Exp7) and for tiny shapes (overhead-bound). Regime A here = N>M.

Grouped projection (regime-A = N>M shapes only; excludes the M>N "transposed" shapes 1216×4096×32,
4864×4096×32, 512×6144×128, 1024/2048/4096/8192/16384×6144×128):

| shape | Mt | br BW% | new_BW% | proj speedup | notes |
|---|---|---|---|---|---|
| 32×6144×1536 | 1 | 74.6 | 96 | **1.29×** | large K, solid |
| 32×6144×2304 | 1 | 73.7 | 96 | **1.30×** | solid |
| 32×6144×3072 | 1 | 72.7 | 96 | **1.32×** | solid |
| 32×6144×6144 | 1 | 76.4 | 96 | **1.26×** | solid |
| 32×6144×9216 | 1 | 77.9 | 96 | **1.23×** | solid |
| 32×2048×2048 | 1 | 74.7 | 95 | **1.27×** | solid |
| 32×2048×1536 | 1 | 69.4 | 94 | **1.35×** | solid |
| 32×2048×512  | 1 | 48.1 | ~80 | ~1.4-1.9× | SMALL (2.3MB) overhead-bound, discount |
| 32×256×6144  | 1 | 63.2 | ~85 | ~1.3-1.5× | SMALL (3.6MB) + shallow K, discount |
| 64×6144×1536 | 2 | 76.8 | 92 | **1.20×** | solid |
| 64×15360×1536| 2 | 71.8 | 92 | **1.28×** | solid |
| 64×4608×6144 | 2 | 75.4 | 92 | **1.22×** | solid |
| 64×6144×4608 | 2 | 74.3 | 92 | **1.24×** | solid |
| 64×6144×9216 | 2 | 77.8 | 92 | **1.18×** | solid |
| 128×6144×2304| 4 | 73.9 | 87 | **1.18×** | large N, burden ok |
| 128×2304×6144| 4 | 71.8 | 87 | **1.21×** | large N |
| 128×6144×4608| 4 | 76.4 | 87 | **1.14×** | large N |
| 128×6144×768 | 4 | 60.5 | ~78 | ~1.2-1.3× | small N=24t → burden 8P·4/24>1, dent |
| 128×15360×768| 4 | 64.0 | ~78 | ~1.2× | small N, burden dent (deep K) |
| 512×6144×1536| 16| 61.5 | ~55-62 | **~1.0× (no win / risk)** | Mt=16 broadcast collapses + AI=361 compute-bound |

**Bottom line:**
- **Solid wins on the bulk (Mt=1,2,4, large N, deep K): ~1.15-1.35×**, bounded above by the branch already
  hitting 72-78% BW-util (the DRAM cap headroom is only to ~95-97%). The mechanism: the multi-reader reads
  in1 at ~98% of cap (vs branch 72-78%) AND moves minimal bytes (no K-par/slice re-reads).
- **Larger nominal wins (1.4-1.9×) on the small/awkward shapes** where the branch has poor BW-util (48-64%):
  32×2048×512, 32×256×6144, 128×…×768. BUT these are either tiny (overhead-bound — the new variant also
  pays dispatch/ramp/in0-prologue on a ~2-4MB op, so realized gain < nominal) or have in0-broadcast burden
  >1 (small N). Real gains likely ~1.2-1.4×, still positive.
- **One likely non-win: 512×6144×1536 (Mt=16)** — the in0 broadcast burden collapses it (Exp7) and AI=361
  is the least DRAM-bound (branch already 28→36% MAC). Not a regime-A sweet-spot shape; expect ~parity or
  use a 2D partition there.
- Geomean over the solid regime-A shapes ≈ **~1.25×** over the branch (which is itself ~1.7× over main).
  Every regime-A shape is expected to win or tie; none should regress except possibly the Mt=16 outlier.

## Assets
- `sp_forward/test_in0_concurrent.cpp` (+ `kernels/loader_read.cpp`, `loader_mcast.cpp`) — Exp 6/7, `--preaders P --loaders L --mt Mt --mode {none|read|mc0|mc1|mc1c}` (mc1c = DRAM-sharded contiguous in0).
- `sp_forward/test_multireader.cpp` (+ `kernels/reader_mr.cpp`) — Exp 5, `--preaders P --noc-mode {noc0|noc1|split|dual}`.
- `sp_forward/test_mcast_egress.cpp` (+ `kernels/mcast_egress.cpp`) — Exp 1 single-core egress.
- `sp_forward/test_forward_readthrough.cpp` (+ `kernels/reader_fwd.cpp`, `mcast_fwd.cpp`, `noop.cpp`) — Exp 2
  read-through; `--decouple 1` = Exp 3b.
- `sp_forward/test_egress_concurrency.cpp` (+ `kernels/unicast_egress.cpp`) — Exp 3a concurrency, mcast|unicast.
- `sp_forward/test_worker_pull.cpp` (+ `kernels/reader_pull.cpp`, `worker_pull.cpp`) — Exp 4 worker-pull, `--k 1..8`.
- Registered in `tests/tt_metal/tt_metal/perf_microbenchmark/sources.cmake`.
- Run: `TT_METAL_DEVICE_PROFILER=1 <bin> ...`; parse `tools/mm_sweep/parse_kernel_bw.py <csv> <bytes> [BRISC|NCRISC]`.
