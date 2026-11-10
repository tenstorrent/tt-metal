# Megakernel AR track: status 2026-11-10 (branch anatarajan/qwen38-r16-mk-ar)

Standalone generic_op prototypes of the AR_SEND/AR_RECV phases (ar_track.py, kernels/mk_ar*), spec v0.1 (global
semaphores; the program-semaphore limit is 16 per core). No host C++: built from Python via MeshProgramDescriptor +
ttnn.setup_fabric_connection (same pattern as gpt-oss fused_decode/boundary.py). So it needs no in-pod rebuild.

| Variant | What | Silicon (d26u02, 1x8 ring, trace, per call) | Correctness |
|---|---|---|---|
| v0 | reduce on the 80 STREAM cores, 4-root x | hangs (dispatch timeout); dropped | - |
| v1 | reduce on 20 non-STREAM AR cores (rows 8-9), x multicast + 20 flags | 130 us; 116 us with 8 KB packets | residual bit-identical on all devices; CB_X exact on all 80 cores |
| v2 | v1 + RMSNorm on the AR cores (stats exchange) | 193 us; 181 us with 8 KB packets | + normed x within 0.3% (gamma x NORM_FIX folded into the consumer) |
| baseline | all_reduce_async one_shot + fused residual, RES40 out (prod) | 22.7 us | - |

The baseline does not replicate x into 80 cores' CB_X (320 KB each), and it does not norm.

Zone breakdown (v1, 4 KB packets): fabric sender 61 us median (80 packets/link over one core's 2 RISCs, fused
atomic per packet); AR-core wait 76 us; slot sum <1 us; stage 2.5 us; x multicast 6-37 us (20 concurrent 16 KB
multicasts into the 80-core rect); end-to-end STREAM x-wait 125 us. v2 adds a 50 us stats exchange (20 concurrent
4 KB multicasts among the AR cores + flags).

Next, in order of expected gain:
1. Sender parallelism: prod gets 22.7 us for the same 320 KB with 2 workers per link (num_workers_per_link=2). Use
   4-8 sender RISCs on 2-4 non-STREAM cores (one per link and direction or more), each with its own fabric
   connection. Expect the sender to drop from ~55 to ~10-15 us.
2. Stats exchange: leader-based instead of all-to-all (19 unicasts of one tile to a leader, leader sums and
   multicasts one rstd tile). Expect 50 -> ~5 us.
3. x delivery: 20 concurrent multicasts contend. Stagger them, or use 4 aggregator roots (fm11 scheme), toward the
   ~10 us floor for 320 KB into every STREAM core.
4. Overlap in the megakernel: the AR cores and the sender are non-STREAM, so STREAM DM RISCs keep prefetching the
   next phase's weights during the whole AR. Only CB_X consumption waits for x.
Weight repack for the vLLM integration: weights.py (fm11 page layout, gamma x NORM_FIX fold, checked 3.6e-7).

## 2026-11-10 ~11:30Z update: v3 hang, root cause, pivot

- v3 (v2 + 4 sender workers as 2 per link from 2 cores + leader stats + 4-root x) HUNG on d26u02 and wedged the column
  (a fresh ttnn.add also times out). It needs `tt-smi -r`. DO NOT RUN ArTrackV3 as written.
- Root cause: 1D fabric routers have `num_sender_channels_1d = 2` (tt_metal/fabric/builder/fabric_builder_config.hpp):
  channel 0 = the local worker, channel 1 = forwarding. That allows ONE worker connection per (link, direction);
  two workers on the same link collide.
- The fast path that respects this is what all_reduce_async one_shot does (program_factory.cpp
  create_one_shot_descriptor, 18-20 us / 22.7 us traced):
  - one sender per (link, direction), each holding only that direction's connection (4 senders on a ring with
    2 links);
  - reduce-scatter by direct unicast of remote-owned 1/8 slices into the owners' slots (fused write + sem inc);
  - owner-only reduction (same compute kernel / slot order as the default path, bit-identical);
  - a line-multicast all-gather of the reduced slices.
  Bytes per link are far lower than v1/v2's "multicast every partial to every device" (every link carried the full
  320 KB per direction there).
- Pivot (coordinator): run the one-shot reader / writer / receiver kernels as a phase inside the megakernel program
  (fabric connections opened once per pass), then add x delivery into CB_X (4-root multicast) and overlap the MLP
  AR's send with the per-chunk down output. Plan:
  1. port create_one_shot_descriptor's core and arg layout to Python (owners = 40 RES40-equivalent cores or the AR
     cores; 4 sender cores on non-STREAM cores, one per link and direction; its CB ids remapped into the megakernel's
     AR range), as a generic_op first; time it against all_reduce_async (target = the same 18-23 us);
  2. add the RMSNorm on the owners (each owner holds 1/8 of every row: leader-based stats over 8 owners, or
     over the 40 owner cores) and the 4-root x delivery into CB_X;
  3. fold into the megakernel phase table (fabric connections opened once per pass; per-layer phase = RS send,
     owner reduce, AG multicast).

## 2026-11-10 ~11:40Z: pivot steps 1-2 on silicon (d26u02 after reset, 1x8 ring, trace, per call)

| Step | What | us/call | Correctness |
|---|---|---|---|
| 1 | Python port of all_reduce_async one_shot (oneshot_ar.OneShotAR), RES40 layout + residual | 21.7 (op itself 22.7) | bit-identical to all_reduce_async on all 8 devices |
| 2a | same, megakernel layout (80 STREAM CB_OUT in -> 16 non-STREAM owners (0,8)-(7,9), 10 tiles each) | 21.7 | bit-identical across devices; max err 0.09 at 14 vs torch |
| 2 | 2a + RMSNorm on the owners (leader stats over 16) + 4-root x into CB_X (OneShotARNorm) | 71.9 | residual bit-identical; normed x within 0.3%; CB_X equal on all 80 cores |

Step-2 tail zones (medians; compute zones include their input waits): stats 15, leader 7.7 (rstd 5.5), rstd wait
4.9, scale 5.3, x unicast to roots 1.4, 4-root 80 KB multicast 16.3 (p90 30). The tail adds ~50 us to the 21.7 us AR.

To get to <=23 us per AR the tail must leave the critical path:
1. Norm into the consumer: deliver the un-normed new residual r as x. Every STREAM consumer holds the full row, so
   it takes sum(x^2) while its K loop streams and scales its matmul output rows by rstd at the end (with gamma and
   NORM_FIX folded into the weights). This removes the stats/leader/rstd/scale chain (~25 us) from the AR.
2. x delivery merged into the all-gather: CB_X needs the full r on 80 cores (x is only read by the 68 gate_up
   cores in fm11; down cores read h). Make the AG land directly in the STREAM cores' CB_X (fabric packet -> NOC
   multicast at the destination, plus a local NOC multicast), so the 16-30 us 4-root multicast overlaps the AG.
   Option: BFP8 CB_X halves the 320 KB per core.
3. In the megakernel the remaining x delivery overlaps the consumer: deliver x in K order with per-block flags, so
   gate_up starts on the first K blocks.
Step 3 (phase table, fabric connections opened once per pass) not started.

## Step 3: persistent all-reduces (OneShotARP, 2026-11-10, d26u02 1x8 ring)
N all-reduces in one program, fabric connections opened once, in-row owner layout (v0.6), + residual,
bit-identical on 8 devices at every N (traced per program):

| N | us/program | us/AR |
|---|---|---|
| 1 | 23.4 | 23.4 |
| 2 | 41.4 | 20.7 |
| 4 | 77.6 | 19.4 |
| 8 | 149.9 | 18.7 |

Marginal per-AR 18.07 us (4 -> 8); launch + setup ~5.3 us. The first 8-iteration run hung: the single-pass
RS/AG CB sizes made the CB pointers drift past the CB end from iteration 2; persistent mode sizes them per sender.

Device profile, N = 8, per iteration (medians): writer iteration 18.2 us = RS send ~8 us (~70 KB per (link, dir)
worker, ~9 GB/s) + wait for the owners' reduced chunks 8.1 us + AG send ~2 us; owner wait for the 7 RS chunks
15.7 us; reduce 1.5 us; gather cores have x 17.8 us after the iteration starts. The RS (one worker per link
direction, chunks multi-hop) dominates.

## Step 3b: RS started per owner chunk (progressive producer, 2026-11-10)
Synthetic down stage (mk_ps_producer.cpp on the owners' BRISC): per persistent iteration, owner chunk released at
rank x delta; the sender's reader stages each RS chunk only once released and stages this device's reduced chunk for
the all-gather as soon as it is ready (interleaved with the RS sends). On path = marginal per-iteration time (N 4 -> 8)
minus the 15 x delta release window. Bit-identical on 8 devices in every run.

| release order | delta 0 | 0.5 us | 1 us | 2 us |
|---|---|---|---|---|
| chunk index on every device (no interleave) | 18.13 | | 16.69 | |
| chunk index (AG interleaved) | 18.13 | | 17.58 | |
| **rot: per device farthest owner first, own chunks last** | | 11.08 | **8.97** | 8.70 |

With the same chunk order on every device, the last-released chunks all belong to one device, so 14 RS chunks
converge on it after the last release. Rotated per device, the last releases are each device's own chunks (no RS hop),
so after the last release only the local reduce + all-gather remain: ~8.7-9 us on path (+3.2 us split-K delivery
~ 12 us), from 18.1 us. Down-stage contract: owners should finish chunks in this per-device order
(MK_PS_ORDER=rot: hops descending, own chunks last).

## Step 3c: rotating RS packet headers (MK_PS_HDRROT, default off)
2 headers per direction, flush every second packet: marginal per-AR 18.14 us with and without (N 4 -> 8). No gain:
the RS is fabric-link bound, not per-packet-flush bound (one-shot RS puts ~200 KB on each forward link direction
per AR, multi-hop, ~25 GB/s). Remaining levers on link bytes: BFP8 payload (flag, gate) or a ring-style RS.
Decision (§15): no streaming reduce; the 8 x 20 KB slots are a dedicated 160 KB region on the owners shared by the
attention-AR and the MLP-AR.

## Burst release + chain reduce-scatter (2026-11-10)
**Burst** (all 16 chunks released within <=1 us, as the v0.6 down stage produces them), persistent one-shot:
t0 -> x on all owners 17.6 us (rot order) / 19.1-19.7 (index order); + 3.2 us split-K delivery ~ 20.8 us.

**Chain RS** (chain_ar.py, kernels/mk_ch_*.cpp; 1-hop sends, each member adds its partial; owner adds both chain
sums + own + residual, then one-shot AG; persistent; identical on all devices, PCC 0.99999 vs fp32):

| chain depth K (rest send direct, multi-hop) | marginal us/AR |
|---|---|
| 4 (pure chain) | 19.93-20.03 |
| 2 | 18.48 |
| 1 | 19.40 |
| one-shot (reference) | 18.07 (burst rot 17.6) |

Per stage (K=4 profile): hop send 0.58 us per 20 KB (~34 GB/s), but one chain stage (receive -> core push ->
compute add -> sender NOC copy -> send -> fabric hop) ~3 us, so 4 stages ~12 us; owner reduce 1.4 us; AG tail 4.8 us.
The chain is latency-bound, not byte-bound: no gain over one-shot.

## BFP8 knobs (MK_AR_RS_BFP8 / MK_AR_AG_BFP8, default off; numerics -> output-length gate)
OneShotARSK, in-row partials (MK_SK_PDST=row), traced per call (incl. ~5.3 us launch + setup), x in CB_X identical on
all 8 devices and all cores of a K-row, PCC / max abs err of x vs the fp32 sum of the bf16 partials + residual
(|ref| max 12.86):

| RS BFP8 | AG BFP8 | us/call | delta | x PCC | x max abs err | owner residual max err |
|---|---|---|---|---|---|---|
| 0 | 0 | 28.4 | - | 0.999998 | 0.086 | 0.086 |
| 0 | 1 | 26.1 | -2.3 | 0.999971 | 0.109 | 0.086 (bf16 kept on the owner) |
| 1 | 0 | 25.2 | -3.2 | 0.999973 | 0.118 | 0.118 |
| 1 | 1 | 22.7 | -5.7 | 0.999944 | 0.167 | 0.118 |

In-megakernel t0 -> x in CB_X (burst release, 20.8 us bf16) by the same deltas: AG 18.5, RS 17.6, both ~15.1 us.

## Chain sub-chunk pipelining (MK_CH_S)
Marginal per AR (persistent, exact; PCC 0.99999): K=2 S=1 18.35, **K=2 S=2 17.84**, K=2 S=5 19.02, **K=4 S=2 17.82**
(K=4 S=1 was 20.0). Pipelining recovers ~2 us on the pure chain but only ties the one-shot burst (17.6 rot).

## Row-compact AR (OneShotARP rows=R, MK_AR_ROWS; exact)
Only rows 0..R-1 move: the sender reader / owner receiver pack each input tile's two runs (face 0 / face 1 rows
0..R-1, R x 32 B each) into a packed chunk of ceil(640 R / 2048) pseudo-tiles; RS, reduce (+ packed residual, owner
only) and AG run on the pseudo-tiles (elementwise add does not care about the layout). Persistent, burst release
(delta 1 ns), rot order; identical on 8 devices, PCC 0.999996 vs fp32 at every R:

| R | AR only (us/AR) | + x delivery into the K-row CB_X (t0 -> x in CB_X) |
|---|---|---|
| 4 | 10.14 | 28.29 unpacked in place (40 small multicasts/chunk: bad) / **11.05 packed** (consumer unpacks locally) |
| 8 | 11.10 | |
| 16 | 13.02 | |
| 32 (off) | 17.59 | 20.22 |

Packed delivery (MK_PS_DELIVER=2): the owner multicasts the 2.5 KB packed chunk; the consumer's reader must unpack the
2 x R x 32 B runs per tile into tile layout before its matmul (local L1 copies, est. ~0.3-0.5 us, not measured).

## RS hop sweep + reduce-on-arrival + direct AG handoff (after the 2nd reset)
Persistent receiver now uses monotonic semaphore counts (no per-iteration reset): fixes the HOPSEL=99 hang mechanism
(with no RS, nothing back-pressures an owner's next all-gather, so a gather core's equality wait on 1 could see 2).
Normal path unchanged (18.12 us/AR, bit-identical).

**RS by hop distance** (MK_PS_HOPSEL, timing only; each run still has the full reduce + AG; max KB per link direction
from the routing model):

| RS restricted to | KB / link dir | 8 KB packets | 4 KB | 6 KB | 15 KB | 2 KB |
|---|---|---|---|---|---|---|
| all (normal) | 160 | 18.14 | 17.91 | 17.91 | 19.36 | 25.67 |
| 1 hop | 20 | 10.92 | 10.52 | | 11.81 | |
| 2 hops | 40 | 11.70 | | | | |
| 3 hops | 60 | 12.84 | | | | |
| 4 hops | 40 | 14.02 | 12.93 | 13.23 | 16.28 | 15.87 |

Same bytes, 2 -> 4 hops: +2.3 us (~1.1 us per extra hop at 8 KB packets; ~0.6 at 4 KB). 20 -> 60 KB at fixed-ish
latency: ~19-25 GB/s marginal per link direction. The one-shot RS is bound by bytes x hops; 4-6 KB packets save ~0.2 us
only (fabric-global setting, prod uses 8 KB).

**Reduce on arrival** (MK_PS_SRED, per-source RS flags, fp32 ping-pong accumulators): 20.65 us/AR (vs 18.12), exact.
Worse: arrivals bunch at the end and each accumulate step (reconfig + fp32 pack) costs more than it hides.
**Owner -> AG staging direct** (MK_PS_AGD): 18.30 us/AR, exact. No gain. Both stay default off.
