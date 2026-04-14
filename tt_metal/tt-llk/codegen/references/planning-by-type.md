# Kernel-Type-Specific Planning Steps

Read ONLY the section for your kernel type. The other sections are irrelevant.

---

## SFPU Kernels

SFPU kernels are typically the most straightforward. Key steps:
1. Check what SFPI library functions are available on the target
2. Check what LUT modes are available (`lut` vs `lut2` vs `lut2_sign`)
3. Check if any hardware-accelerated instructions exist for the operation
4. Follow the template structure from existing target SFPU kernels

---

## Math Kernels

Math kernels involve MOP (Macro Operation) configuration. Key steps:
1. Identify the MOP structure from the closest existing target math kernel
2. Check `MathFidelity` modes and `ReduceDim`/`PoolType` handling
3. Verify TTI instruction availability (GMPOOL, GAPOOL, ELWADD, ELWMUL, etc.)
4. Follow the three-thread synchronization pattern (unpack → math → pack)

---

## Pack Kernels (8 Planning Steps)

Pack kernels require careful hardware configuration:
1. Read the closest existing target pack kernel LINE BY LINE
2. Identify the MOP template type and configuration pattern
3. Identify PACR instruction usage and packer resource selection
4. Check tile increment patterns (TT_OP_PACR0_TILE_INC, etc.)
5. Check data format handling and dest tile offset calculation
6. Identify init/uninit patterns and what hardware state they modify
7. Verify all constants are explicit values (NEVER boolean expressions for hardware params)
8. Copy the replay buffer / MOP loop pattern exactly from existing code

---

## Unpack Kernels (5 Planning Steps)

Unpack kernels have the most architectural variation:
1. Read the closest existing target unpack kernel LINE BY LINE
2. Identify the MOP template type (`ckernel_template` vs `ckernel_unpack_template`)
   ```bash
   grep -r "ckernel_template\|ckernel_unpack_template" tt_llk_{target_arch}/llk_lib/llk_unpack_*.h
   ```
3. Check replay buffer API and config write patterns:
   ```bash
   grep -r "load_replay_buf\|replay_insn" tt_llk_{target_arch}/llk_lib/ --include="*.h"
   grep -r "TTI_WRCFG\|TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/ --include="*.h"
   ```
4. Check context-based addressing pattern:
   ```bash
   grep -r "unp_cfg_context" tt_llk_{target_arch}/llk_lib/ --include="*.h"
   ```
5. Check tile dimension configuration (for tilize/untilize modes):
   ```bash
   grep -r "Tile_x_dim\|config_unpacker_x_end" tt_llk_{target_arch}/llk_lib/ --include="*.h"
   ```
