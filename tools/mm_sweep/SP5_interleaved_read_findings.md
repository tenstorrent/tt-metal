# SP5 — reader==consumer on INTERLEAVED big operand (Regime B go/no-go)

**Question:** for Regime B (N≪M, in0 is the big read and MUST stay DRAM-interleaved), can each
compute core read its own contiguous tile-id range of interleaved in0 (reader==consumer, no
forwarding) fast enough? Or is it stuck at the ~2 KB-burst floor?

**Tool:** custom `tests/.../sp5_interleaved_read/test_interleaved_read` (+ kernel
`reader_interleaved.cpp`) — many cores, each reads a contiguous tile-id range from a bf16
DRAM-interleaved buffer via the interleaved addr-gen (`noc_async_read_page`), double-buffered with
outstanding depth `depth`. **Kernel-time** BW (device profiler), 256 MB, 8 DRAM banks.

## Result — interleaved read saturates at ~419 GB/s (82 % of the ~510 ceiling)
Core-count sweep (depth=8, best):
| cores | kernel-time BW |
|---:|---:|
| 8 | 205 |
| 16 | 294 |
| 32 | 330 |
| 64 | 403 |
| 96 | 413 |
| 110 | **419** |

Depth sweep (110 cores): depth 8→419, 32→418, 64→414, 128→408, 256→400 (more outstanding slightly
*worse* — congestion; shallow depth is fine).

## Interpretation
- **The interleaved 2 KB-burst penalty is real but not fatal.** Contiguous 8-core reads hit ~510
  (SP1); the *same 8 cores* reading interleaved get only 205 (2 KB bursts, per-core under-parallelized).
  But interleaved consecutive tile-ids round-robin all 8 banks, so **adding cores adds parallel
  in-flight requests** and recovers to ~419 at 96–110 cores.
- **Regime B is a CONDITIONAL GO.** Achievable pure-read ceiling ≈ **419 GB/s = 82 % of peak**, using
  most of the grid as readers. That is above where today's matmul lands on most big-M shapes
  (device-profiler ~290–400), so there is real headroom — but **Regime B cannot reach ~500**. The
  interleaved-in0 constraint structurally caps it at ~82 % because logical-order reads are 2 KB bursts.
- **Contrast with Regime A:** in1 is DRAM-shardable → K-deep contiguous 16 KB bursts → can target the
  full ~510 ceiling. Regime A is the higher-value, higher-ceiling half; Regime B is headroom-limited.

## Consequences for the plan
1. **Set honest per-regime targets:** Regime A → ~90 % of 510 (~460+). Regime B → ~90 % of 419
   (~375–400), NOT 500. Update success criteria #12/#4 accordingly.
2. **Regime B needs MANY reader cores** (≥64, ideally ~96–110), not the 8 that suffice for a
   contiguous/sharded read. So Regime B's core budget is set by the *read* (needs ~all cores to hit
   419), which conveniently also gives plenty of compute cores for large M.
3. **Shallow outstanding depth (~8 tiles) is optimal**; deep prefetch hurts. Keep reader CBs small.
4. **Possible future Regime-B lever (not tested):** a larger DRAM interleave *page* (e.g. page = a
   K-row of tiles instead of 1 tile) would give larger contiguous bursts while still being
   "interleaved" — could lift Regime B toward peak, but changes in0's memory config (composability
   question). Flag for later; out of scope now.

## EXTENDED (follow-up): can FEWER cores hit peak? placement / NoC / issue-rate?

Motivation: in the integrated matmul the NoC also carries in-operand broadcast + output write, so
minimizing reader cores is valuable. Investigated depth, dual-RISC/dual-NoC, and NoC congestion (via
tt-metal noc-event trace + direct link analysis; tt-npe's op-grouping returned empty, so analyzed the
trace JSON directly — `analyze_noc_trace.py`).

**1. The original SP5 core-sweep used depth=8, which is too shallow for few cores.** At low core count
the limit is OUTSTANDING DEPTH (bandwidth-delay product), not issue rate:
| cores | best BW | at depth |
|---:|---:|:--|
| 8 | 348 | 128 |
| 16 | 372 | 128 |
| 24 | **400** | 32 |
| 32 | 405 | 64 |
| 64 | 409 | 64 |
| 110 | 419 | 8 |

⇒ **~24–32 cores reach ~400–405 GB/s (95–97 % of the 419 peak)** with the right depth. We do NOT need
all 110 cores for the read. Tradeoff: few cores → deep outstanding (big reader CB); many cores →
shallow depth (deep prefetch congests). Total outstanding ≈ cores×2×depth has a sweet spot (~BDP).

**2. Dual-RISC (BRISC/NOC0 + NCRISC/NOC1) is a LOSS, not a win.** 8c: 190 (dual) vs 343 (single);
only ties single at 64c. Two RISCs on one core share the same NoC injection point and interfere;
NOC1's path to the banks is worse. ⇒ single-RISC + deep depth is the right design; don't split reads
across both RISCs on a core.

**3. The ~419 cap is DRAM 2 KB-burst efficiency, NOT NoC link congestion.** Noc-trace evidence
(32 cores, NOC_0): all 8 DRAM endpoints receive EXACTLY equal traffic (512/512 reads) — perfectly
balanced. Approx XY-routing link model shows mild concentration (max/mean ≈ 2.4) on the vertical links
of the two DRAM columns (x=0, x=9) — a secondary effect of banks being clustered in 2 columns, not a
saturated hotspot. Plateau across 24–110 cores + dual-RISC-no-help both confirm DRAM-granularity is
the primary limiter. Contiguous 16 KB bursts hit 510 (SP1); the 419 vs 510 gap is purely 2 KB vs
16 KB DRAM burst efficiency and cannot be closed by placement/VC while staying interleaved.

**Marginal untested levers (won't break past ~419, may help reach it with fewer cores):** bank-adjacent
reader placement + explicit VC assignment (like `8_dram_adjacent`) could trim the vertical-column
concentration and recover the ~405→419 gap at low core count. Low priority given DRAM is the cap.

**Refined guidance for integration:**
- Regime B read is satisfied by **~24–32 cores** (deep outstanding, single-RISC). SP2 (updated:
  compute ceiling ~2.4 TF/core, ~90 % util) shows compute needs even fewer, so **the read sets the core
  count for Regime B (~32 cores)** and the shape stays read-bound. Reader==consumer means readers are
  the compute cores anyway; ~80 cores stay free.
- Regime A (in1 sharded, 16 KB bursts): **~8–16 cores already saturate at ~510** (SP1) — very few
  readers needed; frees the rest. This is the higher-value regime.
- (bf16 / HiFi2 fixed scope; reads are bf16 2 KB tiles throughout.)

## Artifacts
- `tests/tt_metal/tt_metal/perf_microbenchmark/sp5_interleaved_read/` (test + kernel, `--dual-risc`,
  `--depth`, `--num-cores`), in `sources.cmake`. Build: `cmake --build build_Release --target
  test_interleaved_read`.
- Run: `TT_METAL_DEVICE_PROFILER=1 test_interleaved_read --input-size <B> --depth <d> --num-cores <c>
  --num-tests 6`, then `tools/mm_sweep/parse_kernel_bw.py <csv> <input-size>`.
- NoC trace: add `TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1
  TT_METAL_DEVICE_PROFILER_NOC_EVENTS_RPT_PATH=<dir>`; analyze with
  `tools/mm_sweep/analyze_noc_trace.py <dir>/noc_trace_dev0_ID0.json`.
