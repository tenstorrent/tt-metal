# L4 temporal-banded K-skip ring SDPA — kernel plan (M2 blueprint)

## Recommendation: MODIFY the ring SDPA kernel; do NOT fork the MSA op.
The MSA sparse-flash op IS already in-tree (ttnn/.../transformer/sdpa/device/kernels/
{compute/sparse_sdpa_msa_compute.cpp, dataflow/sparse_sdpa_msa_reader.cpp}) but it's
decode-shaped (Sq=1, no ring) and its value-add is a CONTENT indexer — which LTX does
NOT need: LTX's temporal band is STATIC (deterministic from the F=19×H=34×W=60 grid).
The ring kernel already has the pieces: STANDARD path has proven two-sided sliding-window
K-skip (sliding_window_geometry.hpp + reader_interleaved.cpp:370-384 + compute_streaming.hpp:2084);
RING path has causal_k_limit skip + active_ring_iter_mask (ring_joint_sdpa.cpp:203-208).
Gap = port the static band into the ring path's per-Q-chunk K-loop + tighten active_ring_iter_mask.

## Band mapping (no indexer, no runtime tensor)
Per Q-chunk tile range → frame range [f_lo,f_hi] → widen by window w → absolute K-tile band
[K_lo,K_hi); skip a k_chunk iff it doesn't intersect. Integer arithmetic from (F,HW,w,
Sq_chunk_t=6, Sk_chunk_t=16, kv_local_padded_Nt=152). ±3f ≈ 35% kept ≈ 2.7×. Compile-time
band → hashed op attr → zero per-chunk overhead, trace-captured. Dense (band=None) = today.

## Edit sites (all in ttnn/.../transformer/sdpa/device/)
op params RingJointSDPAParams (+TemporalBand); program_factory build_ring_work_plan_impl
(:204-242 tighten active_ring_iter_mask) + emit band CT args; compute_streaming.hpp sdpa_ring_v2
(:2400 per-Q-chunk band [K_lo,K_hi), :2428-2440 pre-scan, :2459-2466 K-loop skip + CB pop_front);
ring_joint_reader.cpp:592-623 same band test on the gather; attention_ltx.py:490 thread temporal_band=.
Depth-adaptive: per-block window list from opt/l4/depth_map_ltxperftip.txt (dense 4/8/16/20; windowed 12/24/32/40/47).

## ***P0 GO/NO-GO (do BEFORE funding the ~1.5-week build)***
Band-skip saves SDPA COMPUTE + local K/V fill but NOT inter-device all-gather fabric bytes
(ring_joint_sdpa.cpp:203-205 rotates the all-gather BEFORE the skip). So the win only exists
if the QK/PV flash matmul is EXPOSED, not hidden behind the fabric-bound all-gather.
**STRONG NEGATIVE PRIOR:** SDPA-LoFi (dropped SDPA matmul fidelity HiFi2→LoFi = halved that
compute) measured a 0.06s WASH → the SDPA compute is already fabric-hidden. If so, L4 saves
~0 wall-clock on this comms-bound DiT. Confirm with a single-Stage-2-block Tracy (device-active
vs op-gap on RingJointSDPA) before building. If hidden → L4 is a quality-viable but wall-clock-
NEUTRAL lever here, same as SDPA-LoFi; the real bottleneck is the inter-device fabric bytes
(num_links=2 = BH galaxy wall) and the ONLY latency levers left are fewer DiT passes (Tier-C
4-step distill, training) or a fabric-bytes reduction.

## Phasing (~1-1.5 eng-weeks AFTER a positive P0): P0 gate(0.5d) → host band+coarse skip(1-2d)
→ fine per-Q-chunk ring band(2-4d) → depth-adaptive dispatch(1d) → quality gate+tune(1-2d).
Expected IF P0 positive: ~5.6-5.7s. Risk #1 (existential): fabric-hidden compute (see P0).

## CORRECTION (the P0 conclusion was wrong — windowing DOES cut comms)
Verified in-tree: the fused ring SDPA runs a FULL RingAttentionAllGatherAsync (all K/V
shards -> all devices) on DEDICATED AG cores, CONCURRENT with SDPA compute (non-overlapping
coregrids, ring_joint_sdpa_device_operation.cpp:258-268). active_ring_iter_mask gates only the
SDPA reader/compute/writer consumption — NOT the all-gather fabric. So the "modify the SDPA
K-loop" approach saves compute (which SDPA-LoFi proved is hidden) but not comms — hence it
looked dead. THE REAL LEVER: make the ALL-GATHER banded. In the ring, a K/V shard from device s
only needs to reach devices [s-w, s+w]; cap each shard's ring-hop count to the temporal window
(~2w+1 hops vs full 8) -> ~2x less inter-device K/V fabric. That attacks the actual comms-bound
bottleneck (SDPA-LoFi wash confirms fabric-bound). Bigger change (modify RingAttentionAllGatherAsync,
not just the SDPA kernel) but it's THE lever for a comms-bound DiT. Est: banding the K/V gather ~2x
on the fabric portion of RingJointSDPA (29%) -> ~0.3-0.5s; combined with fewer-frame-K compute.
Depth-adaptive: dense early blocks (4/8/16/20), banded late blocks (12/24/32/40/47) per the map.

## BANDED ALL-GATHER — concrete design (verified, conditional GO)
Hop knob = (num_targets_forward,num_targets_backward) at ccl_common.cpp:1726-1727 (4/3 for ring=8).
Device d ALREADY receives band [d-backward, d+forward]; full gather = degenerate band. So banding =
clamp counts to window W. Edits (opt-in, dense default byte-identical):
1. Add optional kv_window to the AG helper (ring_attention_all_gather_async_..._program_factory.hpp:85-91,
   clamp num_targets_* after :165-166) + same clamp in ring_joint_sdpa build_ring_write_plan (:315).
   Leave get_forward_backward_configuration untouched (5+ CCL callers = blast radius).
2. MANDATORY loop-bound fix: replace hard `ring_size` loop bound with compile-time ring_iters_to_run=
   1+f+b in the 3 SDPA consumer kernels (compute ring_joint_sdpa.cpp:202, reader ring_joint_reader.cpp:475,
   writer ring_joint_writer.cpp:533) — else RingIdSequencer over-runs -> HANG.
3. Ring wrap: band [d-W,d+W] wraps at edges. Option A (rec): Ring+uniform W + edge-mask (clear out-of-line
   bits in build_ring_work_plan_impl:204-242) — interior 4 dev get full 1.75x, edge 4 save compute only.
   Option B: Linear topology on SP axis only (exact, more bytes saved, higher max-hop latency; sp_axis
   SDPA is separable from tp_axis to_out AGMM). Chain constraint: counts MUST be uniform=W (per-device
   truncation starves downstream). depth_map used a TRUE window -> Ring-wrap band must be RE-GATED.
4. Depth-adaptive dispatch (attention_ltx.py:490-518, key like _ring_pc_by_n): dense 4/8/16/20;
   W=2 band-5 for 12/24/32/40; W=1 band-3 for 47.
Fabric: W=2 -> 32/56 slice-hops = 1.75x; W=1 -> 3.5x. ~half blocks -> ~0.3-0.5s e2e -> ~5.6-5.7s.
No cheaper primitive (WAN exp_ring_sdpa has no windowing hook; gather_valid_Ht bounds shard-LENGTH not count).
**P0 (0.5d, before funding ~1.5-2wk build):** hand-patch f=b=W=2 + ring_iters_to_run=5 on one Stage-2
block; run the 8-step e2e (broker_watch guards the hang risk) — confirm (a) no hang, (b) Total drops. If
RingJointSDPA is startup-latency-bound not byte-bound -> NO-GO. Direct e2e Total delta is sufficient (no
Tracy needed for the wall-clock go/no-go).
