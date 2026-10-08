# r04-b01-a01 result: 1.3436 (ok)

## What happened vs expected
Valid on all shapes. PCC 0.9999985 and max_abs 0.0204-0.0240 are the parent's values (same data, same math; only when
gamma reaches compute changed). Per shape, parent r03-b02-a02 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.59 | 12.43 | -0.16 (-1.3%) |
| h4096 | 14.05 | 14.12 | +0.07 (+0.5%, noise) |
| h6144 | 17.55 | 17.73 | +0.18 (+1.0%, noise) |
| h7168 | 19.32 | **18.68** | **-0.64 (-3.3%)** |

Score 1.3436 vs 1.3333 (+0.8% geomean). That is inside the ±1% band as a geomean, but the h7168 move is 3x the noise
and has a mechanism behind it (below), so the h7168 part is real. The other three shapes are noise. That was the
prediction: h7168 ~18.5-18.7 µs, the narrow shapes unchanged, score ~1.34-1.35. h7168 at 18.68 µs is the
campaign's fastest h7168 (r03-b04-a03 had 18.72).

## Why (profiler evidence)
`analysis/late2.py` (r03-b04-a03's script, copied here; output in `analysis/late2_out.txt`; parent numbers from
r03-b04-a03's `late2_r03-b02-a02.txt`). Measured calls, per device:

| | parent dev0 h7168 | this dev0 h7168 | this dev1-3 h7168 |
|---|---|---|---|
| straggler calls (one core's POST >0.5 µs later after go than the median's) | 9/10 | **0/10** | 1/10, 0/10, 1/10 |
| last - median core drain end | 1.58 µs | **0.39 µs** | 0.35-0.44 |
| max POST lag - median | 1.71 µs | 0.06 µs | 0.05-0.06 |

- **The dev-0 cross-call straggler loop is gone on this lineage too.** The streamed gamma lets the late core start
  x*gamma at PRE end, so it no longer finishes last and doesn't launch late next call. The AG couples the chips, so all
  four chips' h7168 kernels now sit at 18.60-18.64 µs (parent: 19.39-19.52).
- min(C_COMB start - W_GAMMA end) dropped 0.6-1.4 µs on every shape (e.g. h7168 dev0 3.32 -> 1.93). This is expected:
  W_GAMMA now ends later because the in-loop trid waits and pushes sit inside it. Compute consumes the chunks as they
  land, so this costs nothing on the critical path.
- h6144: dev 0 had a 19.05 µs per-device span in the parent (1/10 straggler there, plus the dev0 launch-skew tail),
  and now it has 17.73 like the others. But devs 2-3 went from 16.2-16.8 to 17.7-17.8, so the chip mean is flat. The
  devices are now uniform rather than skewed. Treat it as noise.
- Nothing else moved: last-median drain on the other shapes is 0.26-0.68 µs, the same as the parent.

## Classification
win on h7168 (-3.3%, beyond noise, mechanism confirmed by straggler counts); neutral elsewhere. Geomean +0.8% is
nominally inside noise, but this is the new campaign best (1.3436), and it combines the best node's L1 scratch +
fused add_rsqrt combine with r03-b04-a03's straggler fix. Use it as the base.

## What a child of this node should try next
With the straggler gone, every core follows the same timeline, so per-core fixed costs now map straight to kernel time:
1. **Stick-push handshake (W_PUSH ~0.63 µs on every core, on the AG-start path; untried, r03-b01-a03 #1).** Replace the
   `async_write_barrier` before the arrival atomic inc with `async_writes_flushed()` (same NoC/VC ordering the fabric's
   fused write+inc relies on). Move the atomic barrier off the path. Expected -0.2-0.3 µs on all shapes.
2. **PRE x*x at lower fidelity (suggested 5 times, never tried).** Explicit HiFi3 (near exact) or HiFi2 template for
   the PRE mul only (BH ELWMUL ~83 -> ~35 cycles/tile at HiFi2 per tt-metal#58723; r03-b01-a03's fid.py emulation says
   max_abs ~0.03 of the 0.05 gate). Keep POST and the stat matmul at HiFi4. The PRE tail is ~0.7 µs after the last
   input lands on every shape.
3. **Posted output writes** for the drain (r03-b02-a03 #2): the drain is per-core-bound at ~100 ns/tile and trails
   POST by 1-1.9 µs. Cmd-buf and VC tricks (r03-b02-a03, r03-b03-a03) didn't help, so posted is the remaining cheap test.
4. Don't revisit gamma placement: gamma is no longer on any core's critical path.
