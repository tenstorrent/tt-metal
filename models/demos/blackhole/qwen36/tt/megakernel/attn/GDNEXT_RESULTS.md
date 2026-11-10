# v0.5 step 2 (GDN in/out on all ~120 cores): measured headroom, 2026-11-10, d30u20 chip 0

Setup: mixed NL=8 = 2 × [G,G,G,A], 32K, u8 SDPA, BFP8 KV, GUPRE, NBB 1, budget 16 KB, GDN stand-in 64 us, AR stalls 18 us
(run_si_gdnext.sh, CSVs in .claude/tmp/r16/mega/si_gx_*).

GDN-type layers on d30u20: 135.4 / 138.0 / 135.5 / 145.6 / 134.4 us (base), 135.2-135.8 / 133.6-135.6 (repeat).
d27u08 had 145.1.

Per-RISC breakdown of a steady GDN-type layer (TRISC_0 = unpack, real compute; base run, layer 2):
- previous layer's down end → MLP AR stand-in (18 us) → this layer's mixer starts
- mixer (GDN in-proj stand-in, 7.37 MB over 80 STREAM cores): **4.2 us compute; its weights are already resident**
  (prefetched during the previous layer's 82 us stall)
- GDN stand-in + AR stand-in: 83.4 us
- gate|up 8.0 us (max end +6.6 us straggler) → h broadcast / down → down end ~23 us after median gate|up end

So a GDN-type layer is ~100 us of stand-ins plus ~35 us of real work, and the projection phase is not weight-bound.
Spreading GDN in/out over 120 cores (STREAM_EXT step 2) can recover at most ~2-3 us of the 4.2 us mixer compute
(and adds x distribution to 40 more cores). **Recommendation: skip step 2.** STREAM_EXT is worth it where the layer is
stream-bound: u1 attention (§10, step 1).

MKA_MIXTRID=1 (this branch): the mixer slice gets its own transaction ids (6/7), so it is released when its own slice
lands instead of after the layer's ~26 MB mixer + gate|up + down read. Steady layers: no change (already resident).
Cold start: layer 0 C-MIX 114.5 → 89.0 us, layer 0 235.1 vs 244.9-249.9 us. PCCs unchanged (mixer 0.99999, SDPA
0.99998, MLP 0.99998). MK_DNPIPE / MK_FLAG_FLUSH are not wired into the MKA2 kernels (no effect here).
