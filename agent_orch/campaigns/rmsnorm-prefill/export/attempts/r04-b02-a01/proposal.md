# r04-b02-a01: forwarder-push all-gather release: the stats scratch lives in the forwarder's own L1 (height-sharded on the forwarder core); after out_ready the forwarder multicasts the gathered pages + the go flag to the worker rectangle, and compute unpacks each worker's stick in place, so the 20 serial go incs and the 8 post-go remote stick reads leave the critical path

## Motivation
The post-AG fixed chain on the best node r03-b02-a02 (its `ag.py`, medians over measured calls, all 4 chips):

| | h3584 | h4096 | h6144 | h7168 |
|---|---|---|---|---|
| go -> W_DRAIN start (stick read + push) | 0.60 | 0.57 | 0.60 | 0.60 |
| F_FABRIC end -> C_POST start, max over workers | 1.64 | 1.61 | 1.65 | 1.64 |

From r02-b02-a04's `go.py`: the forwarder's 20 serial unicast go incs reach the first worker 0.17 us and the last
0.52 us after F_FABRIC end. So between "all four chips' stats are on this chip" and "compute can start the combine"
each worker spends 0.17-0.52 us waiting for its go and then ~0.6 us reading 8 x 64 B face-rows from 4 L1 bank
cores, barriering and pushing. r03-b02-a02 showed that only ~0.09 us of that 0.6 us was memory latency; the rest is
issue/sync of 8 TensorAccessor-addressed remote reads + barrier + CB push. That is ~0.8-1.1 us per call of pure
handshake on the critical path of every shape (6-9% of h3584, 4-6% of h7168), and it is the same on every shape.
The rest of the post-AG chain (0.38 us combine, POST, drain) is what the other branches attack.

r03-b02-a02's reflection #1 names exactly this ("remove the worker-side gathered-stick read entirely: the forwarder
pushes the data, the go rides behind it on the same NoC"). It was never tried.

## Mechanism
New path `gather_mcast`, enabled only for RMS, TP>1, one forwarder, one row per worker (max_rounds == 1),
<= 32 workers, ring_size <= 8 (all four campaign shapes: 20 workers, 1 forwarder, ring 4). Everything else keeps
the existing path unchanged.

1. **Stats scratch on the forwarder core** (`types.hpp`, factory `compute_sizing` / `make_stats_tensor_spec`, device-op
   validation): the persistent buffer becomes L1 HEIGHT_SHARDED with one shard (all `ring*1` pages) on the forwarder's
   logical core (core index `num_workers` in row-major order, the same formula the factory uses). It is still a
   caller-owned, ping-ponged, mesh-coherent MeshBuffer (same address on every chip), so peer chips' fabric fused
   write+inc land in the forwarder's own L1. Same shape/dtype/page size, only the memory config changes.
2. **Stick layout that is a valid fp32 tile row 0 at a per-slot offset** (worker writer): worker `s` writes its two
   64 B face-rows into the forwarder packet at `L(s)` and `L(s)+1024`, `L(s) = (s/16)*2048 + (s%16)*64`. So for every
   device page, `page + L(s)` *is* the start of an fp32 tile whose row 0 holds slot s's 32 sums (face_00 row 0 at +0,
   face_01 row 0 at +1024). The packet span becomes 3328 B for 20 slots (fits the 4352 B page).
3. **Forked RMS forwarder** (`kernels/dataflow/dit_rmsnorm_forwarder.cpp`, factory points at it; GroupNorm keeps the
   common one): after issuing the fabric sends it multicasts its OWN page to every worker's gathered CB at
   `G + my_device*4352` (overlaps the fabric latency). After `out_ready` it multicasts the 3 remote pages (from its own
   L1) to `G + d*4352`, then `set_multicast`s go = r+1 on the same NoC/VC (ordered behind the data, the matmul mcast
   pattern). One rectangle: logical rows `0 .. num_workers/grid_x` full width, which contains all workers + the
   forwarder (+ idle cores whose copies of the grid-uniform CB/sem are unused), loopback-src multicast.
4. **Gathered CB with a 64 B page** (factory): `stats_transposed_gathered_cb` becomes `gathered_pages` x 64 B on the whole
   grid (19392 B for ring 4 / 20 workers). The worker reserves it before the go wait and pushes all pages right after go
   — no reads. Compute (`dit_rmsnorm_fused_compute.cpp`) waits for all pages and runs the same ELWADDs with tile indices
   `d*68 + L(s)/64` (LLK unpack addresses tile i at `rd_ptr + i*fifo_page_size`; the eltwise unpack doesn't use the
   tile-size GPR), then the unchanged add_rsqrt / transpose_dest / pack. New compute RT arg 2 = `L(slot)/64`.

## Why this is not a repeat
- r02-b02-a04 multicast only the go flag (a 4 B sem) and kept the 8 DRAM stick reads after go: the go mcast alone was
  ~0.1 us slower to the first worker than a unicast inc and saved nothing. Here the multicast *carries the data*, so the
  whole 0.6 us post-go read window disappears, and only one rectangle is used (r02-b02-a04 did 2 serial ones).
- r03-b01-a02 / r03-b02-a02 moved the scratch DRAM -> L1 (-0.09 us): latency was not the cost, the read handshake was.
  This node removes the read handshake itself.
- No other node changed who moves the gathered data on-chip.

## Expected effect and risk
- Expected: F_FABRIC end -> C_POST start drops from ~1.64 us to ~0.8-1.0 us on every worker, and the go-order spread
  (0.35 us) disappears. -0.4..-0.7 us per shape: h3584 ~12.0, h4096 ~13.5, h6144 ~17.0, h7168 ~18.8 us, score ~1.37.
- Costs: 3 x 3328 B multicast after out_ready (~0.15 us), +768 B per fabric packet (~tens of ns), the own-page mcast
  is hidden under the fabric wait.
- Risks: (a) multicast rectangle / num_dests wrong -> hang (eval -> hang); (b) wrong tile indexing -> PCC collapse
  (accuracy_fail); (c) CB/L1-buffer clash on the forwarder core between the 19 KB grid-wide CB and the sharded
  scratch -> program validation error (runtime_error); (d) the mcast lands on idle cores in the rectangle — harmless
  because the CB and semaphore are created on the whole grid.
- How to tell: new zone F_MCAST on the forwarder, the existing W_AGWAIT / W_DRAIN start and TRISC_0 C_COMB/C_POST
  zones: go -> C_COMB start should be ~0.05 us instead of ~0.62 us.
