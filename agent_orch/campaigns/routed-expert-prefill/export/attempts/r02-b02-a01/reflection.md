# r02-b02-a01 result: 1.5279 (ok), new best (parent r01-b04-a04 1.5011, +1.8%)

## What happened vs expected
All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent (and the baseline)
on every shape, including the two that changed layout. So the mixed-height topology sets, the
largest-first expert walk and the per-group height are consistent across reader, writer and
compute. No hang. The host built first try. Kernels JIT-compiled first try.

Per shape vs the parent (µs, noise ±1%):

| shape | parent | this | change | layout |
|---|---|---|---|---|
| gpt-oss | 338.8 | 269.0 | **-20.6%** | 3+3+2+2 rows, one expert per group (was 5+5, two each) |
| dsv4-pro | 1128.0 | 1125.6 | -0.2% (noise) | 3+3+2+2 (was 2x5) |
| dsv3 | 547.5 | 565.0 | +3.2% | 2x5, same code path and deal as the parent |
| minimax-m3 | 382.0 | 392.7 | +2.8% | 2x5, same |
| glm | 487.6 | 492.9 | +1.1% | 2x5, same |
| kimi-k3 | 1378.8 | 1391.7 | +0.9% | 2x5, same |
| mistral | 193.0 | 190.6 | -1.2% | 2x5, same |
| dsv4-flash | 331.2 | 332.8 | +0.5% | 2x5, same |
| kimi-k2.7 | 764.3 | 765.9 | +0.2% | 2x5, same |

Predicted: gpt-oss -12..-25% (got -20.6%), dsv4-pro 0..-6% (got 0), the rest unchanged.
The seven unchanged shapes run the exact same schedule as the parent. A 200k-case host fuzz
(/tmp/r02b02/cc/fuzz.cpp) confirmed the new scheduler gives the same layout and the same owner
for every expert whenever a uniform layout wins. So their deltas are run-to-run noise. dsv3's
+3.2% is unusually large. The heavy rows ended at 532-537 µs (parent 508-535) and the light rows
at 308-318 µs (parent 249-313), i.e. the known group-tail lottery (r01-b01-a02: ±5% on these
shapes).

## Why (profiler evidence)
TRISC1 kernel end per physical row y2..y11, mean of the 10 measured calls
(/tmp/r01b04a04/rows.py on reports/r02-b02-a01/.logs/profile_log_device.csv):
- **gpt-oss:** the 3-row groups (one 9-tile expert each, per_core_M 3) end at 252 and 234 µs.
  The 2-row groups (one 8-tile expert each, per_core_M 4) end at 259 and 254 µs. The parent's
  5-row groups (two experts each at per_core_M 2) ended at 324-329. The M-work on the critical path
  is the same 4 units, so the whole ~70 µs is ONE per-expert fixed cost removed from each group.
  That fixed cost is ~70 µs on gpt-oss, about 1.3x a per_core_M unit. It covers bias add,
  SwiGLU-OAI, the activated multicast and the down drain plus 11 down K-blocks, and pipeline fill.
  This also confirms that bigger per_core_M (4) runs fine within the 4-row L1 budget.
- **dsv4-pro:** the two 3-row groups (4 experts each at per_core_M 1, no phantom rows) end at
  1102-1118 µs, the same as the parent's 3-expert 2-row groups (1087-1122). The two 2-row groups
  (2 experts each) end at 744-759. Cutting per_core_M from 2 to 1 did not shorten an expert:
  ~277 µs per expert on a 3-row group vs ~365 µs before. That is not proportional to M-work,
  which fits the DRAM-bound reading (474 MB / 1.12 ms = 420 GB/s). The extra fixed cost of the
  4th expert per group ate the M saving. The 2-row groups now idle for the last third of the call.

## Classification
win: +1.8% geomean from one shape far outside noise (gpt-oss -20.6%). The rest is neutral
(dsv4-pro is DRAM-capped; the other 7 shapes run identical code).

## What a child of this node should try next
1. **The per-expert fixed cost (~1.3 per_core_M units on gpt-oss) is now measured and is the biggest
   lever on few-expert shapes.** The cost model charges 1 unit per chunk. Recalibrating it to ~1.5-2
   units (one constant in `adaptive_chunk::expert_cost`) changes which layout wins on several shapes.
   Re-run /tmp/r02b02/picker.py-style replays before trusting a change. Also attack the fixed cost
   directly: profile the activation / down-drain phases with a few DeviceZoneScopedN on gpt-oss.
2. **dsv4-pro:** 3+3+2+2 leaves the 2-row groups idle from ~750 µs to ~1118 µs. A height-aware
   deal that knows a 3-row expert costs about the same wall time as a 2-row expert here would give
   the 2-row groups 3 experts each (4/4/2/2 -> 3/3/3/3). Or try 3+3+3 with 4 each. It is
   DRAM-bound, so expect at most a few %.
3. **Layouts are now just table entries.** Add a slot to `adaptive_chunk::layout_for_grid`. The host
   builds topology sets from the same table, and the deal is height-aware. Candidates to simulate:
   4+3+3 and 4+2+2+2 for the 4-expert shapes. mistral/minimax-m3 leave 2 rows idle today, and only
   an M- or N-split of an expert can use them, since 22 tiles on 10 rows can't reach
   per_core_M < 3 with whole experts.
4. Group-tail splitting (one expert across two groups) is still untried and orthogonal to this.
5. Note: `ExpertPlan` keeps 4 B per local expert in static storage on every RISC. That is fine for
   production experts_per_chip (8-28). TRISC local memory is 4 KB, so a ~500+ experts-per-chip config would not link.
