# r02-b02-a02 result: 1.5191 (ok), neutral vs parent r02-b02-a01 (1.5279, -0.6%, inside the ±1% noise)

## What happened vs expected
All 9 shapes are valid and bit-identical to the parent and the baseline (PCC, max_abs and mean_abs equal to every
printed digit). That was expected: only the request VC of the weight reads changed. No hang. The kernel-only change
JIT-compiled first try. The eval did a host rebuild, but only because the shared eval checkout had changed.

Per shape vs the parent (µs, noise ±1%):

| shape | parent | this | Δ |
|---|---|---|---|
| dsv3 | 565.0 | 561.2 | -0.7% |
| kimi-k2.7 | 765.9 | 768.8 | +0.4% |
| kimi-k3 | 1391.7 | 1398.3 | +0.5% |
| glm | 492.9 | 489.0 | -0.8% |
| mistral | 190.6 | 193.6 | +1.5% |
| dsv4-flash | 332.8 | 337.3 | +1.4% |
| dsv4-pro | 1125.6 | 1127.5 | +0.2% |
| minimax-m3 | 392.7 | 387.7 | -1.3% |
| gpt-oss | 269.0 | 280.1 | **+4.1%** |

I expected the per-call slowest group to move toward the mean on mm3, mistral, dsv4-flash and gpt-oss. That didn't
happen.

## Why (profiler evidence)
Per-call TRISC1 kernel end over the active cores, averaged over the 10 measured calls
(/tmp/routed_r02b02a02/spread.py on reports/<node>/.logs/profile_log_device.csv), as mean-core / max and max/mean:

| shape | parent | this node |
|---|---|---|
| mm3 | 360 / 391 (1.089) | 355 / 386 (1.087) |
| mistral | 174 / 189 (1.090) | 174 / 192 (1.109) |
| dsv4-flash | 266 / 331 (1.248) | 275 / 336 (1.223) |
| gpt-oss | 248 / 268 (1.078) | 252 / 279 (1.107) |
| kimi-k3 | 1280 / 1391 (1.086) | 1292 / 1397 (1.081) |

- **The group-tail lottery is untouched.** mm3's per-call group ends still scatter 302-433 µs for equal work (e.g.
  320/366/433/326 in one call), and the slow group still changes from call to call (/tmp/routed_r02b02a02/grp.py).
  So FCFS arbitration of weight read REQUESTS within one VC is not what makes one group slow for a whole call.
  Request packets are tiny. The contended resource is the read RESPONSE data and the DRAM banks, and the request
  VC doesn't steer either.
- gpt-oss lost 4%. Its last 2-row group (rows 8-9, VCs (gx+3)&3) now ends ~10-15 µs later than the other groups in
  most calls (e.g. 310/282/283/286 vs ~250-265). That suggests some VC/route combinations are served worse than the
  default VC 1, and their share of DRAM gets smaller. This is a small, plausibly real effect, but it sits on top of
  the lottery's noise. The rest of the shapes are within noise.

## Classification
flawed idea (neutral within noise; request-VC fairness is not the cause of the group-tail scatter). The implementation
is correct and harmless, but a child should revert it (the 3 kernel files), because it may cost gpt-oss a few %.

## What a child of this node should try next
1. Start from r02-b02-a01's code (revert this node's VC argument), or keep it with `weight_read_vc` returning
   NOC_UNICAST_WRITE_VC.
2. **The group-tail scatter is now the largest measured waste on the few-expert shapes**: max/mean group end ~1.09
   on mm3/mistral/gpt-oss and 1.22-1.28 on dsv3/glm/dsv4-flash (those also include real load imbalance). Ruled out
   so far: pacing (r01-b01-a03), ring depth (r01-b04-a03), and request VCs (this node). Untested causes: response
   path / DRAM bank hotspots from all groups walking K in lock-step over the same bank sequence. Every expert tensor
   has the same ND-shard -> bank map, bank = (11*k + col) % banks, and all groups start at K-block 0 together.
   A cheap probe that keeps numerics bit-identical: delay each group's start by a fraction of one K-block. Or give
   each group a different DRAM bank phase by reading its K-rows inside a block in a rotated order (k = (k0 + g*s) % w).
   The read order within a block doesn't change which tiles land in which L1 slot, so the numerics are unaffected.
3. Measure before more DM tweaks. One DeviceZoneScopedN around each sender's per-block DRAM read + barrier (reader
   gate, writer up) on mm3 would show whether the slow group's senders wait on DRAM (bank/response contention) or on
   receiver acks (compute or x).
4. On the device facts: the compute grid is 11x10 (2-column harvested BH, the 12th column is dispatch;
   "Max Compute Cores: 120" in the profiler header), so no extra columns are available. Recalibrating the cost
   model's per-chunk fixed term (F = 2..5) on this lineage only moves dsv4-pro back to 2x5
   (/tmp/routed_r02b02a02/pick.py), so it isn't worth an attempt.
