# r02-b01-a01: route each output tile on the NoC with the short (non-wrapping) x-path to its DRAM column

## Motivation
On the round root (r01-b04-a04, 1.2132) the output drain is the end of the kernel and its speed is set by
throughput, not by compute. At h7168 the AG ends at ~11.4 µs, TRISC ends at ~17.3 and the last W_DRAIN ends
at ~19.7 (r01-b04-a04 / r01-b01-a04 reflections). 2.24 MB/chip leaves in ~8.3 µs (~270 GB/s), and the drain
tail after compute is 1-4 µs on every 20-worker node so far.

The evidence says the limit is NoC link congestion, not DRAM:
- r01-b02-a04 (NoC0 only, 20 workers): drain end grows with core x inside each half
  (x=1 18.8 µs -> x=7 21.9). That is the signature of NoC0's eastward row links. Every write from a left-half
  core leaves the row eastward through x=7->8, including writes to the WEST DRAM column (x=0), which go east
  all the way around the torus. Right-half cores' writes to the EAST column (x=9) also wrap through the left half.
- The same node's position-blind 50/50 NoC0/NoC1 split was a big regression: left-half cores got much worse.
  NoC1 goes west, so a left-half core's tiles for the EAST column (x=9) went west around the whole ring.
  r01-b03-a04 (80 cores) saw the same: a fixed split moved the worst case to the cores that are far on the
  other NoC.
Both results fit one model: each tile should go on the NoC whose x-direction reaches its DRAM column without
wrapping. No attempt has tried a split that depends on the destination.

## Mechanism
Kernel-only change in `dit_rmsnorm_fused_worker_writer.cpp`, W_DRAIN:
- For each output tile, take the NoC0 address of its page (accessor) and decode the destination x. The DRAM is
  in the west column (translated x 17 / physical 0) or the east column (translated x 18 / physical 9).
  The core's own side is `my_x[0] < 9` (left of the east DRAM column) or right of it.
- Left-half core: west-column tile -> NoC1 (west, short), east-column tile -> NoC0 (east, short).
  Right-half core: east-column tile -> NoC1 (west from x to 9), west-column tile -> NoC0 (east through the
  x=16->0 wrap link, one hop).
- BRISC issues the NoC1 writes itself. A second `Noc` object is set up on the other NoC, with
  `noc_local_state_init(other)` to sync BRISC's software counters for it (dedicated-NoC mode only syncs its
  own NoC at kernel start). This is safe: NCRISC (the NoC1 owner) only issues READS and has finished them
  before any output tile exists, so the NoC1 write/ack counters are BRISC's alone. Flush both NoCs per block
  before the pop, and barrier both at the end.
No host change, no CB change, no protocol change, and compute is untouched.

## Why this is not a repeat
- r01-b02-a04 / r01-b03-a04 split by CB position (even/odd), blind to the destination. Half of each core's NoC1
  tiles still wrapped around the row. Their reflections suggest this variant ("pick the NoC per tile by the
  destination DRAM bank's column ... taking the path that does not wrap") but nobody tried it.
- No NCRISC handshake (drain_sem) is needed: BRISC owns the whole drain, so it isn't the r01-b02-a04 protocol.
- Bank de-phasing (r01-b03-a03) is about DRAM bank queues. It barely changed the drain, which points at the NoC.

## Expected effect and risk
- Under the link model, the hottest eastward row link carries ~3x less, and the east DRAM column's southward
  links ~1.4x less. Expected: the drain tail after TRISC end shrinks on all shapes, mostly on wide ones.
  h7168 -1 to -2 µs, h3584 -0.5 to -1 µs. Score ~1.25-1.30 if the model holds.
- Accuracy: bit-identical (same data, same pages).
- Hang risk: BRISC's NoC1 counters. If `noc_local_state_init` isn't effective, the flush on NoC1 would spin
  (hang -> eval reports `hang`). Wrong-endpoint risk: the per-NoC endpoint comes from the accessor with the
  matching noc id, so the address is correct for each NoC.
- If the drain doesn't speed up, then DRAM write ingress (not the NoC) is the limit. In that case W_DRAIN end -
  TRISC end stays at 1-4 µs, and the per-core drain-end gradient over x should flatten anyway.
