# r02-b01-a02: fold the per-core `start` DRAM page read into the one-core counts multicast (on r01-b04-a04's picker), plus per-expert MATH phase zones

## Motivation
- Parent r02-b01-a01 (1.4854) regressed against r01-b04-a04 (1.5011). Its own reflection says start from
  r01-b04-a04's picker, so this node reverts `kernels/adaptive_chunk.hpp` to r01-b04-a04 first.
- Floor arithmetic (bf4 bytes / ~440 GB/s DRAM, and tile-mults at ~16.7 cycles, which is the U ~ 30 µs per
  per_core_M unit that r02-b04-a01 measured on gpt-oss):
  - kimi-k2.7 (761 vs 764 µs), dsv3 (532 vs 548) and dsv4-pro (1077 vs 1128) already sit at the DRAM floor.
  - The few-experts-per-group shapes carry a large fixed overhead above it: mistral ~129 floor vs 192
    measured, minimax-m3 ~289 vs 382, dsv4-flash ~258 vs 331, glm ~386 vs 487, gpt-oss ~130 vs 338.
  - That overhead is the "per-expert fixed cost F" (2-5 per_core_M units) that r02-b02-a01 and r02-b04-a01
    measured but could not attribute. Five reflections asked for a per-phase measurement and no node has
    taken one.
- One fixed cost is visible in the code. Every reader (NCRISC) and every writer (BRISC), ~220 RISCs, reads
  the SAME single `start` (expert_region_offsets) DRAM page at kernel start, behind a barrier, before any
  weight read:
  - The page is 0.5-3.5 KB (num_experts x 4 B; kimi-k3 has 896 experts), so ~0.1-0.8 MB is pulled through
    one bank in the prologue.
  - counts/idx were already moved to a one-reader + L1 multicast scheme for exactly this hotspot reason
    (COUNTS_BCAST), but `start` was left behind. r01-b01-a03 flagged it.

## Mechanism
1. Revert `adaptive_chunk.hpp` to r01-b04-a04 (units+chunks picker; 10/5/2 rows).
2. Reader, COUNTS_BCAST block: the counts-reader core also reads the `start` page and multicasts it, linked,
   into every core's reader start CB (c_15) AND writer start CB (c_14), on the same linked path, ahead of the
   counts_valid sem multicast.
   - Other readers get `start` with the counts. Their own DRAM read of `start` is deleted.
   - The writer no longer reads `start` from DRAM. It already waits on cb_counts_scratch (pushed by the
     reader after counts_valid), so `start` in c_14 is valid by then.
   - The host passes the writer's start CB index to the reader as a define (UNIFIED_RE_CB_START_WRITER).
   - Files: reader.cpp, writer.cpp, program_factory.cpp (one define).
3. Instrumentation, not a mechanism: three DeviceZoneScopedN zones on TRISC_MATH per expert chunk
   ("RE-GU": fused gate/up phase including the SiLU gate copy, "RE-ACT": activation / multiply, "RE-DN": down
   K-loop + copy pass), plus one "RE-PRO" zone on the reader prologue (counts/start broadcast). These
   produce the first measured split of the per-expert fixed cost. The profiler buffer is bounded (zones
   past the room are dropped, op-time markers are guaranteed), and there are at most ~20 zones per RISC
   per op.

## Why this is not a repeat
- Nearest: COUNTS_BCAST (baseline code) did the same for counts/idx. No node touched `start`.
- r01-b01-a04 / r01-b03-a03 attacked the unit-boundary x / weight feed (neutral). This attacks the prologue,
  which precedes every weight read on every core.
- No node has run with per-phase zones yet.

## Expected effect and risk
- Bit-identical numerics: same bytes, delivered by multicast instead of DRAM.
- Expected gain: a few µs off the prologue on every shape, which matters most on short shapes and on
  kimi-k3 (largest page):
  - mistral / dsv4-flash: +1-2%.
  - kimi-k3: ~+0.5%.
  - Overall maybe +0.5% over r01-b04-a04 (~1.50), i.e. ~+1.5-2% over the parent.
  - Could be within noise, in which case the zones are the value of this node.
- Risks:
  - A wrong c_14 address in the writer would give wrong output rows. The accuracy gate catches it.
  - A hang if the sem ordering breaks. The tool times out.
  - The zones could add a few hundred cycles per expert (negligible).
