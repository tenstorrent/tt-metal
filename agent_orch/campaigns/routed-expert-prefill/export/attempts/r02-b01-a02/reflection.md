# r02-b01-a02 result: 1.4889 (ok): neutral vs r01-b04-a04 (1.5011), +0.2% vs parent r02-b01-a01 (1.4854)

## What happened vs expected
All 9 shapes are valid and bit-identical to the baseline (PCC, max_abs and mean_abs equal to every printed
digit). So the `start` page delivered by the counts multicast is correct for both the reader (c_15) and the
writer (c_14). No hang. The host built first try.

Per shape vs r01-b04-a04, whose picker this node restores (noise ±1%):
- dsv3 555 vs 547 (+1.4%), kimi-k2.7 767 vs 764, kimi-k3 1367 vs 1379 (-0.9%), glm 487 vs 488,
  dsv4-flash 326 vs 331 (-1.6%), dsv4-pro 1128 vs 1128, gpt-oss 342 vs 339 (+0.9%).
- mistral 202 vs 193 (+4.8%) and minimax-m3 392 vs 382 (+2.7%). These are the one-expert-per-group shapes
  with the known ±5% run-to-run spread.
- Overall -0.8%, inside noise.
- The picker revert worked: the unit counts per group in the zones match r01-b04-a04's layouts (gpt-oss
  5+5 rows with 2 experts per group, 2-row groups elsewhere).

I expected +0.5% from the fold. It bought nothing measurable, and the zones show why.

## Why (profiler evidence: first per-phase measurement in the campaign)
Zones on TRISC_1 (MATH): RE-GU, RE-ACT and RE-DN per expert chunk, plus RE-PRO on NCRISC.
- Scripts: /tmp/r02b01a02/zones.py, perexp.py, percore.py, run on reports/r02-b01-a02/.logs/profile_log_device.csv.
- All numbers are means over the 10 measured calls. Real N-columns only: column x=15 (gx 10) is wholly
  phantom on gpt-oss, so its GU is ~1 µs and its waits fall into ACT. Exclude it.

**1. The prologue is ~1 µs.**
- RE-PRO (counts + start broadcast) takes ~1.0 µs on every shape.
- The first RE-GU starts 2.5-6 µs after kernel start (6.4 on kimi-k3, which has the 28-expert plan).
- There was nothing to win by folding `start`, and nothing left in the prologue for later nodes.

**2. Per-expert phase split** (µs per expert chunk, real columns):

| shape | GU (first / later) | ACT | DN | ACT share | DN share |
|---|---|---|---|---|---|
| dsv3 | 186 (209 / 149) | 0.6 | 79 | — | 30% |
| kimi-k2.7 | 178 (194 / 167) | 0.4 | 82 | — | 32% |
| kimi-k3 | 104 (136 / 97) | **39.5** | 78 | **18%** | 35% |
| glm | 162 (181 / 131) | 0.6 | 69 | — | 30% |
| mistral | 117 | 0.6 | 55 | — | 32% |
| dsv4-flash | 92 (109 / 63) | **18.3** | 49 | 11% | 31% |
| dsv4-pro | 246 (269 / 230) | **22.9** | 129 | 6% | 32% |
| minimax-m3 | 211 | **38.7** | 105 | 11% | 30% |
| gpt-oss | 97 (99 / 95) | **25.9** | 40 | 16% | 25% |

- **SiLU shapes** (dsv3, kimi-k2.7, glm, mistral): the SiLU gate copy runs inside matmul_phase_fused_gu, so it
  is counted in GU. Their ACT is only the bf8 x bf16 multiply pass (~0.5 µs). Their activation cost is unmeasured.
- **Binary-SFPU activations** (SiTU-GLU on kimi-k3, clamped SiLU-GLU on dsv4, SwiGLU-OAI on mm3 and gpt-oss)
  cost 1.4-2.2 µs per output tile (~2-3k cycles):
  - kimi-k3: 18 tiles per core per expert, 39.5 µs. That is ~237 µs of the critical 6-expert group's ~1365.
  - gpt-oss: ~1.4 µs per tile including the bias adds.
  - During ACT the group's weight stream idles (only the 2-slot down ring prefills). This is the biggest
    identifiable non-matmul cost in the op.
- **DN is ~30% of each expert on every shape**, which equals its share of MACs and weight bytes (1/3). So the
  down phase is not anomalously slow. Its per-block handshake chain (rotating activated multicast, reader
  waits for receivers before its DRAM read) is NOT where the fixed cost is.
- **The first expert's GU is 15-60 µs slower than later experts' GU** (dsv3 +60, glm +50, dsv4-flash +46,
  kimi-k3 +39). Two possible causes:
  - All groups start in lockstep and contend for DRAM, and later experts run after groups have desynchronised.
  - The first expert also pays pipeline fill.
  - It is a per-call cost on every shape, but probably mostly contention, not latency.
- gpt-oss real cores per expert: GU 95-100, ACT 26, DN 35-45. The r02-b04-a01 fit "F ~150 µs per expert"
  is therefore not a fixed latency. GU alone is ~100 µs at pcM 2, and the whole expert is GU + ACT + DN with
  no idle gaps between zones. F is the activation (~26) plus the part of GU/DN that does not scale with pcM,
  i.e. weight delivery at ~1 group's stream rate.

## Classification
neutral (within noise) for the mechanism: the `start` hotspot was not a real cost (prologue ~1 µs total). The
code is correct and harmless (fewer DRAM reads, bit-identical), so it can stay.

The node's value is the measurement. It also restores r01-b04-a04's picker on branch b01.

## What a child of this node should try next
1. **Hide or shrink the binary activation phase.** It costs kimi-k3 ~18%, gpt-oss ~16%, dsv4-flash and mm3
   ~11% per expert, while that group's DRAM stream idles.
   - (a) Cheapest probe: split RE-ACT into the copy_tile/bias part and the SFPU part (one more zone inside
     binary_activation_phase, MATH thread). Also check whether the pack of bf8 activated tiles or the unpack
     of two bf16 partials dominates. At ~2-3k cycles per tile, the SFPU op itself is the suspect for SiTU-GLU.
   - (b) Overlap: run the SFPU op from the PACK thread (bmm_fused_activation.hpp's
     apply_activation_from_pack pattern) so MATH can proceed. MATH then only pays copy_tile, and the SFPU
     runs on the other DST half. Real overlap needs MATH work that does not depend on this activation. The
     next expert's first gate/up K-blocks are the natural candidate (cross-expert pipelining; needs a second
     partials pair or the reload-free ordering). High effort, but now a measured 6-18% target.
   - (c) Precision rule: keep the approximation modes and formats exactly as they are. Only the thread and
     schedule may change.
2. **SiLU shapes:** move the silu gate copy out of matmul_phase_fused_gu into its own zone and measure it
   before attacking it. It is unmeasured here.
3. **First-expert GU penalty (15-60 µs per call):** stagger the groups' start (e.g. give the heaviest groups
   their smallest expert first), or check whether it is pure DRAM contention. Compare per-group first-GU
   time against group count (5 groups on dsv3 vs 2 on gpt-oss: gpt-oss shows no penalty, 99 vs 95).
4. Don't spend attempts on the kernel prologue (start/counts/plan: ~1-6 µs total) or on down-phase delivery
   timing (DN is exactly its MAC share).
5. Zone tooling: the zones cost nothing measurable. The scripts in /tmp/r02b01a02 parse them, but remember
   to skip phantom N-columns. A child may keep the zones or delete the three `#ifdef TRISC_MATH` blocks.
