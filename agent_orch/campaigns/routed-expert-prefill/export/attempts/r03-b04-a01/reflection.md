# r03-b04-a01 result: 1.5813 (ok)

## What happened vs expected
All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the root and the baseline on every printed digit. That confirms the restructured planner produces exactly the root's plan on the test routing, as the 7.2M-case host fuzz said. There was no hang, and the kernels JIT-compiled first try (kernel-only change, no host rebuild).

The score is 1.5813 against the root's (r02-b02-a04) 1.5697: +0.74% geomean, inside the ±1% noise band. I expected +0.8..+1.2%.

Per shape vs the root (µs; noise ±1%, about ±5% on mistral / minimax-m3 / gpt-oss). Median / max core is the TRISC1 kernel end, mean of the 10 measured calls (/tmp/r02b03a02/med.py):

| shape | root | this | Δ | median/max core root -> this | NCRISC RE-PLAN (this node) |
|---|---|---|---|---|---|
| kimi-k3 | 1405.9 | 1378.8 | **-1.9%** | **1321**/1405 -> **1299**/1378 | 17.7 µs |
| dsv4-flash | 327.9 | 320.7 | **-2.2%** | 303/327 -> 298/320 | 6.9 µs |
| dsv3 | 529.1 | 520.2 | **-1.7%** | 500/528 -> 495/519 | 3.9 µs |
| dsv4-pro | 1080.7 | 1072.5 | -0.8% | 1076/1079 -> 1067/1072 | 5.1 µs |
| minimax-m3 | 381.6 | 369.1 | -3.3% (noisy shape) | 347/381 -> 342/368 | 2.7 µs |
| glm | 453.5 | 454.3 | +0.2% | 433/453 -> 437/453 | 4.0 µs |
| mistral | 195.6 | 196.0 | +0.2% | 167/195 -> 170/195 | 2.7 µs |
| kimi-k2.7 | 736.5 | 741.1 | +0.6% | 733/735 -> 738/739 | 5.3 µs |
| gpt-oss | 267.0 | 273.4 | +2.4% (lottery) | 245/267 -> 242/273 | 3.6 µs |

## Why (best explanation, with profiler evidence)
- **The planner was the cost, and it shrank as modelled.** The new RE-PLAN zone on the reader (NCRISC, script /tmp/rep-r03b04a01-rO2J/plan_zone.py) measures the trimmed planner directly: 17.7 µs on kimi-k3, 6.9 on dsv4-flash, 5.1-5.3 on kimi-k2.7 / dsv4-pro, ~4 on dsv3 / glm, 2.7-3.6 on the 4-expert shapes.
  - In the BH -Os RV32 instruction simulation the new kimi-k3 plan is 22.2k instructions. 17.7 µs is 23.9k cycles, about 1.08 cycles per instruction, so instruction count is a good proxy on these cores.
  - With that rate, the root's planner (50.1k instructions on kimi-k3; 11.6k-17.4k on the 8-12-expert shapes) would cost ~40 µs on kimi-k3 and 9-14 µs on dsv3 / glm / kimi-k2.7 / dsv4.
  - The predicted saving is therefore ~22 µs on kimi-k3 and ~5-7 µs on the 8-12-expert shapes. kimi-k3's median core moved exactly -22 µs. dsv3, dsv4-flash and dsv4-pro moved -5, -5 and -9 µs.
- **Where the root spent it.** In the simulation, 1312 L1 loads (`tiles_of` re-read on every planner pass), 893 `divu` (expert_cost / num_chunks in the two-phase model), the uniform layouts scheduled twice, and LPT walking every expert once per distinct count. The BH RISC-Vs DO have `divu` (-mcpu=tt-bh is rv32im). r02-b04-a01's "no hardware divider" note is wrong, but divides are still multi-cycle.
- **glm and kimi-k2.7 did not move**, although they planned ~5-7 µs faster. Both are DRAM-bound with all rows finishing together. Their medians moved +4 / +5 µs, which is within the ~±5 µs run-to-run spread of the DRAM-bound medians seen across earlier nodes with identical code. The gpt-oss +2.4% is the max-core lottery: its median core moved -3 µs.
- The `start`-read overlap shows in the zones: kernel start -> RE-PLAN end is only ~1.1 µs more than RE-PLAN itself (counts multicast + start read issue). The read's latency no longer sits in front of the plan.
- The UNPACK-thread RE-PLAN zone did not appear in the profile: only NCRISC rows exist. Custom zones on TRISC_0 seem not to be recorded here. The RE-* MATH zones in earlier nodes were on TRISC_1. The zone is harmless.

## Classification
neutral (within noise) by the campaign metric: +0.74% geomean, under the ±1% band. The mechanism itself is a real and measured win: bit-identical, and kimi-k3's median core -22 µs as predicted. dsv3, dsv4-flash and dsv4-pro also moved by the predicted ~5-9 µs. It is cheap, orthogonal to every other lever and safe to stack. A child or a combination node should keep it.

## What a child of this node should try next
1. **Stack it with the round-3 PACK-thread activation pipeline combination** (r03-b01-a01 / r03-b02-a01 were porting r02-b03-a04's fused_swiglu.cpp onto this root). The two touch different things: this node changes `adaptive_chunk.hpp`, two lines each in reader/writer, and a zone in fused_swiglu's UNPACK block. Both gains should add: about +0.7% from this node on top of theirs.
2. **kimi-k3 still plans for 17.7 µs** (1.3% of its time). Remaining simulated cost (22.2k instructions) by function:
   - schedule_lpt over 2 mixed layouts: 4.6k;
   - uniform_makespans: 3.8k;
   - the final schedule_in_order + lpt_may_fit in plan_groups: 3.5k;
   - model_single_phase: 3.5k;
   - model_expert: 2.5k;
   - the PlanTiles insertion sort: 2k.

   Cheap exact cuts:
   - (a) Record each uniform layout's deal (group per expert) in uniform_makespans and reuse it in model_single_phase (skip its first pass) and for the final uniform owner map. That saves ~5k. Gate it on NumExperts <= 64 to bound the TRISC stack.
   - (b) Run model_single_phase only after make_plan's cheap rejections.

   Expect ~6-8 µs on kimi-k3 (+0.5%), with almost nothing on other shapes. Low priority. Use /tmp/rep-r03b04a01-rO2J/sim (rvsim.py / prof.py) to count instructions before an eval, and fuzz.cpp to prove the plan stays identical.
3. **The large remaining waste is the per-expert fixed cost on the compute-bound shapes.** Fitting T = F + U·pcM gives F ≈ 100 µs per expert and U ≈ 25 µs per tile-row: mistral is one expert at pcM 3 (mean core ~170), and dsv4-flash is two experts at pcM 2 (~300) with the same weights. U is at the LoFi ideal (~15-17 cycles per tile-mult); gpt-oss pcM 3 -> 4 also costs exactly the ideal. So mistral, minimax-m3, dsv4-flash and gpt-oss are not limited by the matmul rate. They lose ~100-150 µs per expert to something that does not scale with per_core_M.

   Before changing anything, measure it: put DeviceZoneScopedN zones on NCRISC around (i) the gate DRAM read + barrier, (ii) the wait for receivers / mcast_done, and (iii) the x-owner `cb_x_tl` wait + x multicast, on one shape (mistral or minimax-m3). Then compare per-block wait time against the 8+11 blocks per expert. NCRISC zones are recorded reliably (this node's RE-PLAN); TRISC_0 zones were not.
4. Don't retry: request VCs (r02-b02-a02), down look-ahead / deeper rings (r02-b02-a03, r01-b04-a03), read-ahead (r01-b03-a01, r01-b04-a01), writer-owned down stream (r02-b04-a02).
