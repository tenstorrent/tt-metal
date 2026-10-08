# r03-b01-a03 result: 1.3255 (ok)

## What happened vs expected
All shapes valid. PCC 0.9999985 and max_abs 0.0204-0.0240 are bit-identical to the parent, as expected since only BRISC
scheduling changed. Per shape, parent r03-b01-a02 -> this node (µs, chip mean): h3584 12.66 -> 12.54 (-1.0%),
h4096 14.09 -> 14.17 (+0.6%), h6144 17.92 -> 17.77 (-0.9%), h7168 19.41 -> 19.46 (+0.2%). Score 1.3255 vs 1.3220
(+0.3%), inside the ±1% noise band. Below the campaign best r03-b02-a02 (1.3333). I expected -0.3..-0.5 µs on
h6144/h7168 and got about -0.1..-0.2 µs at the AG start, which the chip-mean metric does not resolve.

## Why (profiler evidence)
Scripts in this dir, run as `python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`. Medians over measured calls and
all 4 chips.

`pre.py`, parent -> this (µs from first worker kernel start):

| shape | slowest pusher: push start - its R_INPUT end | W_PUSH end max (= F_COLLECT gate) | AG wait end max | kernel end |
|---|---|---|---|---|
| h3584 | 0.81 -> 0.77 | 4.82 -> 4.80 | 7.64 -> 7.64 | 11.88 -> 11.87 |
| h4096 | 0.99 -> 0.96 | 5.62 -> 5.59 | 8.43 -> 8.42 | 13.64 -> 13.65 |
| h6144 | 1.26 -> 1.16 | 7.72 -> **7.57** | 10.59 -> **10.38** | 17.47 -> **17.28** |
| h7168 | 1.16 -> 1.02 | 8.19 -> **8.08** | 11.53 -> 11.54 | 18.83 -> 18.76 |

`gbar.py` (new W_GBAR zone = the gamma-landing wait, now polled):

| shape | W_GBAR dur median / p90 | pushes fired inside the poll | pushes after the gamma wait (their W_PUSH dur) |
|---|---|---|---|
| h3584 | 0.31 / 0.82 | 105 / 800 | 2 (0.71) |
| h4096 | 0.31 / 0.33 | 38 / 800 | 21 (0.71) |
| h6144 | 0.31 / 0.87 | 181 / 800 | 74 (0.71) |
| h7168 | 0.32 / 0.97 | 309 / 800 | 239 (0.71) |

1. **The mechanism engages.** Up to 309/800 core-calls now push from inside the gamma wait (p90 durations include the
   0.63 µs push itself). The slowest pusher moved 0.1-0.15 µs earlier on h6144/h7168, and on h6144 that carried through
   to AG end (-0.21) and kernel end (-0.19).
2. **My premise overstated the gating.** The bare gamma-landing wait is only ~0.31 µs, not ~0.5-1 µs. The "push
   starts right after W_GAMMA end, never later" pattern I used as evidence is partly an artifact: W_PUSH's zone opens
   *before* its `cb_stats_local.wait_front`, so a push entered after the barrier also covers any wait for the stat.
   Here the pushes that still start after the gamma wait last 0.71 µs vs 0.63 µs, so the stat arrived ~0.08 µs after
   gamma landed. On wide shapes, gamma landing and stat-ready nearly coincide. The likely reason is that both wait on the
   same DRAM queues draining at the end of the input read. So removing the barrier could only save the overlap, ≤0.3 µs.
3. Nothing else moved: push duration 0.63 µs, stick read after go 0.57-0.60, POST and drain timings identical.

## Classification
neutral (within noise). The mechanism is correct and harmless (keep it; it removes a real ≤0.3 µs hole on wide
shapes). The premise was only partly right, because the zone placement made the gate look bigger than it was.

## What a child of this node should try next
1. **Shorten the stick-push handshake itself (W_PUSH = 0.63 µs on every core, on the AG-start critical path).** Today:
   2 x 64 B writes -> `async_write_barrier` (write-ack round trip) -> atomic inc -> `async_atomic_barrier`. Use
   `noc.async_writes_flushed()` instead of the write barrier before the inc (writes and `Semaphore::up(noc,x,y)` both
   use NOC_UNICAST_WRITE_VC on the same NoC). The fabric router relies on exactly this for its flush=true fused write+inc
   (`fabric_edm_packet_transmission.hpp` NOC_FUSED_UNICAST_ATOMIC_INC: flush, then `noc_semaphore_inc` on the same
   noc/vc). Move the atomic barrier off the path (before kernel end). Expected: the forwarder sees the arrival one write
   round trip earlier, ~0.2-0.3 µs, on every shape. Also put a zone *after* the CB wait if you want W_PUSH to mean
   handshake time only.
2. **PRE x*x at HiFi2 (still untried, suggested 4 times).** Hard numbers now exist: tt-metal#58723 measures BH ELWMUL
   math at 82.6 (HiFi4) vs 34.6 (HiFi2) cycles/tile, with the pipeline at 86.1 vs 38.1. `fid.py` here emulates the BH
   fidelity masks (tt-llk golden: HiFi2 drops srcB's last bf16 mantissa bit) on the test's data. sum(x^2) is biased
   -0.28%, max_abs goes 0.0156 -> 0.0226 in emulation (HW today 0.024 -> expect ~0.03 of the 0.05 gate), and PCC is
   unchanged at 0.9999986. The PRE tail at the slowest pusher is still 1.0-1.16 µs after its read ends vs 0.55-0.64 at
   the fastest cores, so compute lag is plausible. Call the math LLK init/op with an explicit HiFi2 template for the
   PRE mul only. Keep the ones*S^T matmul and POST at HiFi4.
3. Don't revisit the gamma placement: gamma lands about when the input row does on every shape, and the remaining wait
   (0.31 µs) is now overlapped with the push.
