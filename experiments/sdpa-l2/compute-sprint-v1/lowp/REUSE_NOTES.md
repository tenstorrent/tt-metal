# Correction-vector replay reuse

This is a scheduling change, not an identity-correction approximation. It also executes when the correction is not1.

Blackhole ISA reference pinned to `5287a62727350bcef35f7b411d1b8a706172ec4c`: `BlackholeA0/TensixTile/TensixCoprocessor/SFPLOAD.md`. Its functional model reads `Row=(Addr & ~3)+Lane/8`, `Column=(Lane & 7)*2 + ((Addr & 2) != 0)`. The old compensator advances DST by2 after every replay, so a pair at offsets `d,d+2` reads even/odd columns of the same four rows. The correction tile has already been produced by `unary_bcast<BroadcastType::COL>`, making these two correction vectors equal lane-by-lane. Numerator and denominator use their original correction tiles; no maximum comparison is skipped.

Keep the existing replay programs unchanged:

- Numerator entries0–14: entry0 loads L6 correction; entries1–14 execute the two original compensated states.
- Paired denominator entries15–30: entries15/16 load each row's separate L6/L7 correction; entries17–30 execute the two original states.

Execute the entire program for the even-column vector, then only the final14 entries for its odd-column vector. Retain the original macro round/store order, DST address modifier, all32 data vectors, and final three SFPNOP drains. No skip persists across DST halves, rows or state updates. One-state fallback invokes the original function.

For the fixed Q256/D128 loop, there are16 paired numerator updates and4 paired denominator updates per recurrent K chunk. This removes `16*16 + 4*32 = 384` correction-load instructions from9728 replay-body instructions (3.947%). That is an instruction count, **not a runtime-speedup prediction**. The first paired E resident measurement improves0.731% in total time; it does not establish universal hardware utilization or end-to-end speedup.

Qualification splits numerator-only and denominator-only wrappers before testing both together. `e-reuse-num-distinct-v1.json` passes Q2048/K8192 distinct/max-changing input; `g-reuse-denom-distinct-v1.json` passes the same shape with outliers and changing maxima. Both compare raw output bytes to the unmodified kernel and canonical adapter, plus two actual trace replays. Complete E/G combined stress/performance qualification is separate.

The broader `setup_cache` experiment failed exact equivalence and is not part of this change. Its replay/macro state was retained across operations; this optimization retains only a correction vector across adjacent equal-column-broadcast vectors inside one original SFPU call.
