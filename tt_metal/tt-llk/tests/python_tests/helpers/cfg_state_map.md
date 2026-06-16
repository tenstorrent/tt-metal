# Tensix CFG state map — what LLK / Metal actually write

Companion to the CFG-pollution harness (`cfg_pollution.py`, `cfg_bisect.py`). The pollution
test finds, empirically, which CFG registers a kernel *depends on* (reads/uses but doesn't
re-write). This map is the other half: which CFG registers the LLK + Metal compute stack
*writes*, and from where. Together they classify a pollution finding:

- **actionable** — kernel K depends on register R, and some *other reachable* op writes R to a
  non-default value → real reconfig-escape (run that op, then K, K breaks).
- **latent** — kernel K depends on R, but no reachable op ever writes R → R stays at its reset
  default in practice, so the dependency can't actually be violated. Real per the
  "relies on unverified state" criterion, but not a live bug.
- **over-reach** — R is firmware/hardware-owned (not compute config); pollution there is a
  harness artifact, not a kernel finding. These are masked in `_PRESERVE_BITS` / `_BOOT_OWNED_ADDR32`.

> **Reachability matters.** A write *site* in source does not make a register live — the writer
> must be *called*. Example: `PCK0_ADDR_BASE_REG_0` has exactly one writer,
> `program_packer_dest_offset_registers()` (cpack_common.h:740), which has **zero callers** in
> all of `tt_metal` → `PCK0_ADDR_BASE` is latent, not live.

Generated 2026-06-16 from a static sweep (4 agents) over `tt_llk_blackhole/` + the Blackhole
Metal layers, plus `grep`. Addr32 values are Blackhole. Reproduce by grepping the CFG-write
primitives (`TTI/TT_WRCFG`, `TTI/TT_RMWCIB*`, `TTI/TT_SETC16`, `cfg_reg_rmw_tensix`, `cfg_rmw*`,
`get_cfg_pointer()[X_ADDR32]=`, `addr_mod_t::set`).

## Live CFG-write surface (written by reachable LLK ops)

### Unpack (`cunpack_common.h`, `llk_unpack_common.h`, `llk_unpack_*.h`)
- **ALU_FORMAT_SPEC_REG0/1/2, ALU_ROUNDING_MODE, ALU_ACC_CTRL** (addr32 0–2) ← `configure_unpack_AB`, `_llk_unpack_reconfig_data_format_*`, `_llk_unpack_configure_stoch_rnd_`, `_llk_unpack_reduce_init_`
- **THCON_SEC0/1_REG0_TileDescriptor** (64–65, 112–113) ← `configure_unpack_AB`, reconfig, tilize/untilize init/uninit
- **THCON_SEC0/1_REG2_Out_data_format / Haloize_mode / Unpack_if_sel** (72–73, 120) ← `configure_unpack_AB`, every `_llk_unpack_*_init_` (Haloize), reconfig, `unpack_to_dest`/`set_dst_write_addr`
- **THCON_SEC0/1_REG3_Base_address[_cntx1]** (76–77, 124–125) ← unpack execute paths (per-tile L1 src addr), via `_llk_unpack_configure_addresses_/_single_address_`
- **THCON_SEC0_REG5_Dest_cntx0/1_address, Tile_x_dim_cntx0** (84, 86) ← `configure_unpack_AB`, `unpack_to_dest`, tilize/untilize
- **THCON_SEC0/1_REG7_Offset[_cntx1]_address** (92–93, 140–141) ← `unpacker_wrapup`, untilize
- **THCON_SEC0/1_REG1_Unp_LF8_4b_exp** (71, 119) ← `configure_unpack_AB`, reconfig
- **UNP0/1_ADDR_CTRL_*_stride** (56, 57, 59), **UNP0_ADD_DEST_ADDR_CNTR** (50), **SRCA_SET_Base** (5), **UNPACK_MISC_CFG_CfgContextOffset_0** (41), **SCRATCH_SEC0** (209)

### Math / SFPU (`cmath_common.h`, `llk_math_*.h`, `common/inc/sfpu/*`)
- **ALU_ACC_CTRL_Fp32/SFPU_Fp32/INT8_math/Zero_Flag** (1–2) ← `_llk_math_hw_configure_`, `_llk_math_set_fp32_dest_acc_`, datacopy bcast paths (Zero_Flag), `transpose_dest_32b`, reduce
- **ALU_FORMAT_SPEC_REG0_SrcA** (1) ← `transpose_dest_32b`
- **DEST_TARGET_REG_CFG_MATH_Offset** (1) ← `set_dst_write_addr<…,SrcRegs>`, `dest_section_flip`, `set_dest_section_base`
- **DEST_ACCESS_CFG_remap/swizzle/zeroacc_abs** (220) ← `_llk_math_hw_configure_`, `_llk_math_reconfig_remap_`
- **CLR_DVALID_SrcA_Disable** (7) ← every `_llk_math_*_init_`
- **FP16A_FORCE_Enable** (55) ← reduce transpose (int-fpu)
- **ADDR_MOD_AB/DST/BIAS SEC0–7** (12–19, 28–35, 47–54) ← `addr_mod_t::set(idx)` from configure_addrmod in transpose/reduce/fast_tilize/fast_untilize/matmul_custom/sfpu. Index usage: 0–3 common (transpose/reduce/datacopy math uses **ADDR_MOD_3**); **4** only `fast_untilize` + `datacopy_custom`; 5–7 fast_untilize/binary_custom/matmul_custom/sfpu.
- **addr32 2 bulk-zeroed** by `_llk_math_eltwise_sfpu_done_with_addrmod_reset_()` via `TTI_SETC16(2,0)` — SFPU-only path; clobbers the whole word (incl. STACC_RELU + BP bits).

### Pack (`cpack_common.h`, `llk_pack*.h`)
- **THCON_SEC0_REG1_*** (format / L1_Dest_addr / Exp_section / Exp_threshold / Pack_L1_Acc / Disable_pack_zero_flags / LF8) ← `set_packer_config`, `reconfig_packer_data_format`, `program_packer_destination`, `reconfigure_packer_l1_acc`, `reconfigure_exp_threshold`
- **PCK_DEST_RD_CTRL** ← `set_packer_config`, reconfig, `_llk_pack_set_fp32_dest_acc_`
- **PCK0_ADDR_CTRL_XY/ZW strides** ← `set_packer_strides`, fast_tilize/untilize, untilize
- **PCK0_ADDR_BASE_REG_1** ← fast_untilize output-row-stride
- **PACK_COUNTERS_SEC0** ← `configure_pack`, reduce-mask, `_llk_pack_rows_init_`
- **PCK_EDGE_OFFSET_SEC0/1, TILE_ROW_SET_MAPPING_0/1** ← `configure_pack`, reduce-mask
- **DEST_TARGET_REG_CFG_PACK_SEC0_Offset** (128-bit) ← `select_packer_dest_registers` (dest-sync), `_llk_init_packer_dest_offset_registers_`
- **STACC_RELU_ApplyRelu/ReluThreshold** ← `configure_pack`, `_llk_pack_relu_config_`
- **ALU_FORMAT_SPEC_REG2_Dstacc** ← `configure_pack`, reconfig

### Common (`ckernel.h`)
- **CFG_STATE_ID_StateID** (0) ← `flip_cfg_state_id`  ·  **TENSIX_TRISC_SYNC** ← `set_ttsync_enables`  ·  **PRNG_SEED** ← `init_prng_seed`

### Metal layers
All `metal/llk_api/llk_{math,pack,unpack}_*_api.h` and `hw/inc/api/compute/*` are pass-through
to `_llk_*` — **no direct CFG writes**. The only Metal-direct CFG writes are in three Blackhole
SFPU kernels (`metal/llk_api/llk_sfpu/ckernel_sfpu_{exp,quant,reduce}.h`) writing **ADDR_MOD_5/6/7**.

## Latent registers (relied-upon by kernels, but no reachable writer)
- **DEST_REGW_BASE** (addr32 6) — no writer anywhere in `tt_metal`.
- **PCK0_ADDR_BASE_REG_0** (addr32 16, bits 0–17) — only writer is the uncalled `program_packer_dest_offset_registers()`.

## Over-reach (firmware/hardware-owned — masked, not findings)
- **DISABLE_RISC_BP** (addr32 2, bits 22–31) — branch prediction; `brisc disable_branch_prediction()` + ttexalens debug-halt. No LLK writer.
- **CFG_STATE_ID.StateID** (addr32 0, bit 0) — active-shadow select.
- **WH TRISC_RESET_PC_SEC0/1/2 + RESET_PC_OVERRIDE** (addr32 158–161) — boot vectors (`device_setup`).

## Datacopy triage (Blackhole)
Bisected implicit dependencies, classified via this map:

| addr32 | register | class |
|---|---|---|
| 2 (bits 22–31) | DISABLE_RISC_BP | over-reach (masked) |
| 6 | DEST_REGW_BASE | latent (no writer) |
| 16 | PCK0_ADDR_BASE_REG_0 / ADDR_MOD_AB_SEC4 | latent (PCK0 dead-writer; ADDR_MOD_4 not used by plain datacopy) |

→ No live reconfig-escape in datacopy. Its implicit dependencies are all on registers nothing
actually dirties. Actionable escapes are expected in kernels that share the heavily-reconfigured
surface above (data-format, addr-mod indices, packer strides/dest-offset, STACC_RELU) where one
op writes a register another op reads-but-doesn't-reset.
