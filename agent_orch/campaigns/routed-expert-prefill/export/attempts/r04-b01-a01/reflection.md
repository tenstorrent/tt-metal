# r04-b01-a01 result: 1.6011 (ok). Neutral: -0.6% vs the parent r03-b02-a03 (1.6103), inside the ±1% noise. The mechanism ran and removed the UNPACK-thread stall, but the per-tile-mult cost did not move

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline on every
  printed digit. So the CFGSHIFTMASK address stepping unpacks exactly the same tiles from the same L1
  addresses, including:
  - after the owner tilize, which rewrites SCRATCH_SEC0;
  - across the last-block act <-> matmul switches;
  - in the down phase.
- No JIT error and no hang on the first and only run. Kernel-only, so no host build.
- The h7168 programs fit the kernel config buffer (TRISC0 +676 B in the local compile).
- I expected 31 -> ~20-24 cycles per tile-mult on the compute-bound windows. I got no change:

| shape / window | parent cyc/mult | this | parent UNPACK-thread stall (ST0) | this | UNP0 busy parent -> this | FPU util parent -> this |
|---|---|---|---|---|---|---|
| mistral GU | 31.0 | 31.7 | 17.8% | **1.7%** | 57.3 -> 56.1% | 51.9 -> 50.7% |
| mistral DN | 29.1 | 28.9 | 19.6% | **2.6%** | 62.2 -> 62.7% | 55.5 -> 55.8% |
| mm3 GU / DN | 30.3 / 30.2 | 30.5 / 30.3 | 18.3 / 17.9% | 1.9 / 1.4% | 59 -> 59% | 53 -> 53% |
| dsv3 GU / DN | 32.6 / 30.9 | 32.9 / 30.0 | 16.8 / 17.9% | 1.6 / 1.8% | 54 / 61 -> 54 / 62% | 49 / 52 -> 49 / 54% |
| gpt-oss GU / DN | 41.5 / 30.9 | 41.5 / 29.9 | 13.6 / 18.8% | 1.6 / 2.8% | 44 / 60 -> 44 / 62% | 39 / 52 -> 39 / 54% |
| kimi-k3 DN | 60.9 | 62.1 | 9.5% | 1.3% | 29 -> 29% | 26 -> 26% |

(/tmp/r03b02a03/hw.py on reports/r03-b02-a03 and reports/r04-b01-a01, mean of the 10 measured calls.)

Chip µs vs the parent: dsv3 514.0 -> 513.5, kimi-k2.7 729.5 -> 729.2, kimi-k3 1365.8 -> 1360.0,
glm 451.2 -> 450.8, dsv4-pro 1070.9 -> 1067.0, gpt-oss 261.8 -> 259.7, dsv4-flash 317.0 -> 320.3,
minimax-m3 365.5 -> 368.6, mistral 185.9 -> 195.6.
- mistral's +5% is its known ±5% lottery: its GU / DN mean-core windows are flat (105.8 -> 108.3, 49.6 -> 49.3).

## Why (profiler evidence)
1. **The mechanism did what it was built to do.**
   - The UNPACK thread's stall share fell ~10x on every window (17-19% -> 1.4-2.8%).
   - That stall was the per-tile `STALLWAIT(STALL_CFG, THCON)` waiting on RDCFG + ADDDMAREG, and it is gone.
2. **It was never on the critical path.**
   - cyc/mult, UNP0 busy (~17.5 cycles per mult) and FPU busy (16.1 per mult) are unchanged to within ±1 cycle.
   - So the UNPACK thread used to stall on the chain while the unpacker was busy anyway. The chain overlapped
     the previous tile's UNPACR.
3. **What is left, by elimination:**
   - FPU 16 + SrcA unpack 17.5 still add up to ~31-33 cycles per mult, with MATH_SRC_DATA_READY ≈ FPU busy.
   - The UNPACK thread is now almost never stalled, and the unpacker is still idle ~45% of the window.
   - So the unpacker is idle without the thread holding it back. It must be waiting for a free SrcA bank.
   - That means the next streamed tile cannot be written while the FPU still uses the current one. In this
     kernel the SrcA double-buffering behaves like a single bank: unpack(i+1) starts only after MVMUL(i)'s
     CLR_A.
   - That explains every earlier null result in one line:
     - reuse-B moved the stream to SrcB, with the same single-bank serialization on SrcB;
     - subblock width only changes per-call work;
     - and now the address chain.
4. Ruled out now, with direct counters: FW throttle (r03-b02-a03), unpack bandwidth / volume (r03-b03-a04),
   per-call overhead (r03-b01-a04), and the per-tile unpack address chain (this node).

## Classification
flawed idea (measured null). The implementation is correct, bit-identical and harmless to keep. It is
+676 B on TRISC0, and it frees the UNPACK thread (ST0 ~2%), which costs nothing. The per-tile unpack config
chain is not what serializes unpack and FPU on BH here.

## What a child of this node should try next
1. **Measure the bank hand-off before touching the LLK again.**
   - In re_hwprof.hpp, put the stall-reason selectors for "waiting for SrcA clear" / "waiting for SrcB clear"
     (BH INSTRN_THREAD sels 27/28; tt-llk docs/performance_counters/performance_counters.md, "BH: shared stall
     reasons occupy single slots (27-30)") in place of two unused slots.
   - If SrcA-clear wait ≈ 13-15 cycles per mult on mistral GU, the unpacker is blocked on the bank, as argued
     above.
   - Then the question is why the second bank is not usable while MVMULs run. Candidates:
     - (a) the math MOP's CLR_A comes only on the 16th MVMUL, and the unpacker may need BOTH banks free
       to flip;
     - (b) `Set Dvalid` on every UNPACR plus OvrdThreadId may force in-order bank use with a full round trip;
     - (c) the SrcB (reused x) dvalid gating each MVMUL.

     Compare with the DeepSeek decode custom_mm, which reaches near-ideal throughput with the same per-tile
     SrcA stream. Its math MOP uses `MVMUL ... CLR_A` with ADDR_MOD clr on srca. Look at what differs in its
     unpack (UNPACR_COMMON flags) and math (no per-tile SETC16, one MOP over ct tiles) sequences.
2. **A cheap structural test of (1)**: an op-local math MOP that runs a whole 1 x ct row in one MOP, the
   custom_mm style `ckernel_template(1, ct, mvmul_base, mvmul_end_tile)` with dst advanced by ADDR_MOD rather
   than a SETC16 per tile, kept bit-identical. If the FPU idle comes from MATH re-arming per tile (SETC16 + MOP
   restart racing the dvalid), this removes it. It is also the only per-tile work left on MATH.
3. Keep this node's re_mm_unpack.h if you do (2): the unpack side is now a 3-instruction stream per tile, so it
   won't be the next limiter.
4. Don't retry: operand swap / reuse mode / subblock geometry, the unpack address chain (this node), FW-throttle
   theories.
5. Tooling: /tmp/r04b01/sz/kc.sh (local 3-TRISC compile against this worktree; base/ = parent kernel,
   new/ = this node; add -DPROFILE_KERNEL=1 -DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=65536 for the profiler
   build).
