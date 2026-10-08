# r05-b03-a01 result: 1.5174 (ok)

## What happened vs expected
Valid on every shape. PCC 0.9999985 and max_abs 0.0205-0.0240 are the parent's values, so the half-row partial sticks,
the per-wave pages and the 8-tile combine are all exact. No hang in 13 calls x 4 shapes. Nothing was iterated: the
first build compiled and the first device run is this result.

Score 1.5174 against the parent/root r04-b04-a02's 1.3997 (+8.4%). This is the campaign's best valid node: the higher
HiFi2 nodes are now forbidden_edit. µs, chip mean:

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 11.96 | **11.47** | -0.49 (-4.1%) |
| h4096 | 13.62 | **12.76** | -0.86 (-6.3%) |
| h6144 | 16.91 | **15.25** | -1.66 (-9.8%) |
| h7168 | 17.93 | **16.02** | -1.91 (-10.7%) |

Every shape is outside the ±1% noise band. The gain grows with width, as it should when the hidden quantity is a
DRAM phase. I predicted about -3 µs at h7168 and -1.5..-2 at h3584 (score 1.55-1.7). The result is roughly 60% of the
model on the wide shapes and about a third on h3584, for the reasons below.

## Why (profiler evidence)
`analysis/waves.py <report>` gives per-wave medians over measured calls x 4 chips, in µs from the chip's first kernel
start. Wave B = cores with an R_WAVEWAIT zone. Outputs: `waves_out.txt` (this node) and `waves_parent_out.txt`.

h7168 (h3584 in brackets):

| | parent (20 full rows) | wave A | wave B |
|---|---|---|---|
| R_INPUT end max | 6.56 (3.43) | **3.59** (2.08) | 6.41 (3.42) |
| W_PUSH end max | 7.81 | 4.51 (2.94) | 7.32 (4.42) |
| forwarder send end | (F_COLLECT 7.92) | 5.42 (3.49) | 7.83 (4.84) |
| go max | 10.93 (7.43) | 8.67 (6.37) | 10.70 (7.78) |
| C_POST start max (T0) | 12.18 (8.54) | 10.16 (7.88) | 12.58 (9.47) |
| drain end max | 17.72 (11.63) | 13.53 (9.60) | **15.73** (11.13) |
| drain dur med | 5.73 | 3.50 | 3.66 |
| chip kernel end | 17.86 (11.78) | | 15.87 (11.26) |

1. **The pipeline works as designed.**
   - Wave A's read ends at 3.6 µs instead of 6.6, so its stick, all-gather and POST all start ~2.2-3 µs earlier.
   - Wave B is released at 2.7 µs (lead = 2 blocks). Its read ends at 6.4, the same time the parent's whole read
     ended, so the read stays aggregate-bound and nothing is lost there.
   - Wave B's all-gather (send 7.8 -> go 10.7) runs while wave A drains (10.2 -> 13.5).
   - Each wave's drain is half as long (3.5 vs 5.7 µs). The kernel end is now wave B's drain end, 1.9 µs earlier
     than the parent's single drain.
2. **Why less than the model:**
   - **The gather after go got slower.** go -> C_POST start is 1.25 µs in the parent, 1.49 in wave A and 1.88 in
     wave B. Cause: 16 x 64 B stick reads instead of 8, plus an 8-tile combine (4 ELWADDs on fp32 tiles) instead of
     4. Wave B pays more because its post-go reads and combine overlap wave A's drain.
   - **Each wave pays the full all-gather latency.** Send -> go is ~3.1 µs (A) and ~2.9 µs (B). These are medians
     over chips, so they include cross-chip launch skew. Wave B's AG is not shorter than wave A's.
   - **Narrow shapes have the least read to hide.** h3584's full read is only 3.4 µs, so the wave split hides about
     1.5 µs of read and drain. The extra ~0.4-0.6 µs of post-go gather eats a third of that.
3. Wave-B cores restart up to 1.4 µs late on h6144/h7168 ("B BRISC start max" 1.41-1.44). They drain last, so the
   previous call's per-core end spread carries over. Wave B is gated until ~2.6 µs anyway, so this is hidden.
   Wave A starts within 0.10 µs.
4. Per-device durations (`eval/ops.csv`) alternate between ~15 and ~17 µs call to call on h7168. That is the usual
   cross-chip launch skew and not a new effect.

## Classification
win (+8.4% over the root, every shape outside noise; best valid node in the campaign). It is a structurally new
mechanism: two-wave overlap of the DRAM phases with the all-gather, made possible by splitting each row across two
cores. It repairs r03-b04-a02 (waves of 10 cores) as that node's reflection suggested.

## What a child of this node should try next
1. **Shrink the post-go gather (go -> POST start is 1.5 / 1.9 µs; it was 1.25 in the parent).** Both halves of a row
   are on adjacent slots, so their sticks are contiguous: [h0: f0r0 f1r0][h1: f0r0 f1r0].
   - Simplest fix: change the stick layout so a row's two halves land in tile rows 0 and 1 of ONE gathered tile.
     The half-h worker writes its face rows at slot offsets {h*64, 128 + h*64} into a 256 B row slot. One 128 B
     read then fills face0 rows 0-1, and one fills face1 rows 0-1. That is 8 reads and 4 tiles again.
   - But then compute must add row 1 into row 0 before add_rsqrt. That is an SFPU lane shift, or a
     `transpose_dest` + a 2-column add.
   - Alternative: keep 8 tiles but read each device's 256 B (both halves) with fewer, larger reads into a staging
     page, and add on the FPU.
   - Measure with `waves.py` (go -> T0 C_POST start).
2. **More waves.** col_split 4 / 4 waves of 20 cores (80 workers + forwarder; 8-bit wave fields, 4 scratch pages)
   would hide more of the read and the drain behind three all-gathers. Do (1) first: the post-go gather grows with
   col_split (32 reads, 16-tile combine) and would eat the gain. Also check whether the AG per wave (~3 µs median,
   mostly cross-chip skew) can overlap with itself. Wave k+1's send currently happens before wave k's go, so it
   already does.
3. **Wave B's drain overlaps nothing**: 12.6 -> 15.7 at h7168, the kernel tail. An uneven split, where wave B gets
   fewer rows (e.g. 12 / 8 rows with col_split 2 = 24 / 16 workers), would shorten the exposed tail. Each wave must
   stay ≤ 34 sticks.
4. Tune the B release lead (kWaveLeadBlocks = 2). Wave A's read (3.45 µs at h7168) is slightly longer than half the
   parent's read, so B's early start slows A's tail a little. Try lead 1 vs 3.
5. Caveat carried over: the posted drain still needs a per-bank non-posted fence for production (r04-b04-a01 #2).
   The wave pipeline only engages for BH, RMS, one link, even rows ≤ 34, even width, broadcast gamma, no
   rope/bias/heads. Other configs keep the old path; the stats scratch has one unused page if sizing chose waves but
   the factory fell back.
