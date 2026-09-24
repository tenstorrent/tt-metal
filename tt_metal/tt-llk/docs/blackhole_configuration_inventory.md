# Blackhole architectural CFG inventory

This document inventories architectural configuration-register access under
`tt_llk_blackhole/`. It is intended to guide the typed HAL work in
`tt_llk_blackhole/llk_lib/hal/cfg.h`.

The baseline inventory was taken on 2026-07-23, before the typed HAL landed.
The coverage and migration-status sections below were re-measured on
2026-08-25 against the tree with the `hal::cfg` migration applied. Macro
definitions and commented-out calls are excluded. Calls inside
`tt_llk_blackhole/llk_lib/hal/` are the abstraction boundary rather than
migration candidates.

## Scope

The inventory includes only instructions and helpers that access the Thread or
Tensix architectural configuration-register files:

- `WRCFG`, `RDCFG`, `SETC16`, `CFGSHIFTMASK`, and `RMWCIB*`.
- `cfg_reg_rmw_tensix<>`.
- State-aware MMIO access through `cfg[]`, `cfg_rmw`, and `cfg_read`.

It intentionally excludes address counters, math read/write counters, SFPU
local state, MOP state, DMA/GPR loads, and per-operation instruction modifiers.

## Architectural CFG coverage

Measured on 2026-08-25, **239 raw architectural CFG accesses** remain outside
`llk_lib/hal/`, down from 347 in the 2026-07-23 baseline:

| Access path | Writes | Reads | Notes |
|---|---:|---:|---|
| Compile-time Tensix instructions (`TTI_*`) | 85 | 15 | 48 `WRCFG`, 32 `SETC16`, 5 `CFGSHIFTMASK`, 15 `RDCFG` |
| Tensix byte RMW helper (`cfg_reg_rmw_tensix<>`) | 117 | 0 | Commented calls and the helper definition are excluded |
| State-aware MMIO pointer (`cfg[...]`) | 15 | 0 | Remaining whole-word address writes; every MMIO `cfg[]` read is migrated |
| MMIO RMW/read helpers | 0 | 1 | All nine baseline `cfg_rmw` field writes are migrated; one `cfg_read` remains |
| Runtime instruction macros | 4 | 0 | Four raw `TT_SETC16`; the raw `TT_WRCFG` is migrated |
| Pre-encoded instruction words | 2 | 0 | Two `TT_OP_WRCFG` words in the untilize MOP |
| **Total** | **223** | **16** | |

The same tree has 60 `hal::cfg::write` call sites outside `llk_lib/hal/`
covering the accesses migrated so far.

The 15 `RDCFG` calls are not setters, but they participate in save/modify/restore
sequences and therefore belong in the same typed interface.

## Logical register groups

Site counts in the group tables below are the 2026-07-23 baseline that shaped
the HAL design; several families have since moved behind `hal::cfg`. See
"Migration status" for what remains today.

### 1. Thread-local execution policy

Access mechanism: `SETC16` / `TT_SETC16`.

| Register or field | Raw compile-time sites | Purpose |
|---|---:|---|
| `CLR_DVALID.SrcA_Disable` | 17 | Math/SFPU source-valid cleanup policy |
| `DISABLE_IMPLIED_SRCA_FMT.Base` | 12 | Explicit versus inferred SrcA format |
| `UNPACK_MISC_CFG` packed word | 6 | Unpacker context offset/reset/increment |
| Address-modifier tables | 4 | SrcA/SrcB, destination, bias, and pack increments |
| `SRCA_SET` packed word | 3 | Source-address swizzle/base-set selection |
| `DEST_TARGET_REG_CFG_MATH.Offset` | 2 compile-time + runtime sites | Destination bank/offset selection |
| `FP16A_FORCE.Enable` | 2 | Integer-mode destination read behavior |
| `TENSIX_TRISC_SYNC.TrackGlobalCfg` | 1 | CFG hazard tracking |
| Literal register address `2` | 1 | Same word as `DISABLE_IMPLIED_SRCA_FMT`; a magic-address cleanup candidate |
| `CFG_STATE_ID.StateID` | runtime | Selects the active configuration bank |

Recommended API:

```cpp
hal::cfg::write<hal::cfg::Access::TensixCfgUnit,
                hal::cfg::ClrDvalid::SrcA_Disable,
                hal::cfg::Sec::S0,
                0>();
```

For thread CFG, `SETC16` replaces all 16 bits. Put every intended field from
one physical word in the same assignment run; unspecified bits are written as
zero:

```cpp
hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
    hal::cfg::set<hal::cfg::UnpackMiscCfg::CfgContextOffset_0, hal::cfg::Sec::S0, 4>(),
    hal::cfg::set<hal::cfg::UnpackMiscCfg::CfgContextCntReset_0, hal::cfg::Sec::S0, 1>());
```

Address-modifier helpers that select their section through a constant-propagated
runtime argument keep the packed compatibility overload because `set()` requires
a compile-time `Sec`.

### 2. Unpack format and tile geometry

Access mechanisms: `WRCFG`, `RDCFG`, `cfg_reg_rmw_tensix<>`, and MMIO.

Key targets:

| Register family | Direct `TTI_WRCFG` | Direct `TTI_RDCFG` | Role |
|---|---:|---:|---|
| `THCON_SEC[01]_REG0_TileDescriptor` | 6 | 4 | Source format and X/Y/Z/W tile geometry |
| `THCON_SEC[01]_REG2_Out_data_format` | 4 | 1 | Unpack destination format and mode |
| `THCON_SEC0_REG5_Tile_x_dim_cntx0` | 10 | 4 | Context tile dimensions |
| `THCON_SEC[01]_REG3_Base[_cntx1]_address` | 6 | 4 | Unpacker L1 base addresses |
| `THCON_SEC[01]_REG7_Offset[_cntx1]_address` | 6 | 0 | Context address offsets |
| `UNP[01]_ADDR_CTRL_*` | 3 | 2 | X/Y/Z/W strides |

The same families dominate `cfg_reg_rmw_tensix<>`:

- `THCON_SEC0_REG2_Haloize_mode`: 13 sites.
- unpack Y-stride: 6 sites.
- unpack Z-stride: 3 sites.
- tile-descriptor words/fields: 11 sites.
- unpack format, LF8 mode, destination address, and interface selection:
  16 sites.

Recommended split:

- Field writes: use `hal::cfg::write<Access, Field, Sec>()`.
- Whole descriptor/config writes: use the resolved-pointer whole-word overload
  `write<Access::MMIO, Anchor, Sec>(cfg, value)` or the consecutive array
  overload; do not force prepacked words through a scalar field write.
- Save/restore: add `read_word<Access, Reg, Sec>()` and a small RAII or explicit
  snapshot object. This directly covers all `RDCFG` sequences.
- Context 0/1 and unpacker 0/1 should be type or enum axes, not names baked into
  call sites.

### 3. Pack format, destination, and address generation

Key targets:

| Register family | Direct `TTI_WRCFG` sites | Role |
|---|---:|---|
| `PCK0_ADDR_CTRL_XY/ZW_REG_*` | 9 | Packer strides |
| `PCK0_ADDR_BASE_REG_*` | 3 | Packer base address |
| `PCK_EDGE_OFFSET_SEC*` | 4 | Edge masks |
| `TILE_ROW_SET_MAPPING_*` | 3 | Row mapping |
| `THCON_SEC0_REG1_L1_Dest_addr` | 2 | Output address |
| `THCON_SEC0_REG1_Row_start_section_size` | 2 | Packed row layout |
| `DEST_TARGET_REG_CFG_PACK` | 1 | Destination offsets |

Related RMW families include pack counters, destination read width, L1
accumulation, LF8 mode, exponent thresholds/section size, and ReLU controls.

Recommended split:

- `PackAddressing` for base/stride/output address.
- `PackLayout` for edge masks, row mappings, row/section size.
- `PackFormat` for destination format, LF8/exponent controls, and source format.
- `PackCounters` for pack-per-plane limits.

These are logical facades over generated fields; the generated register
descriptors remain the silicon source of truth.

### 4. Math/ALU and destination behavior

The largest RMW cluster is math configuration:

| Family | `cfg_reg_rmw_tensix<>` sites |
|---|---:|
| `ALU_ACC_CTRL` | 39 |
| `ALU_FORMAT_SPEC*` | 31 |
| `DEST_ACCESS_CFG` | 3 |
| `STACC_RELU` | 2 |

The most repeated individual fields are
`ALU_FORMAT_SPEC_REG0_SrcA` (22),
`ALU_ACC_CTRL_Fp32_enabled` (19),
`ALU_ACC_CTRL_Zero_Flag_disabled_src` (12), and
`ALU_ACC_CTRL_INT8_math_enabled` (4).

Recommended facade:

```cpp
MathFormatConfig{
    .src_a = DataFormat::Tf32,
    .fp32_acc = true,
    .int8_math = false,
}.apply<Access::TensixCfgUnit>();
```

This is more useful than exposing repeated independent writes to callers and
can enforce Blackhole format invariants in one place.

### 5. Scratch and atomic CFG mutation

| Mechanism | Sites | Targets |
|---|---:|---|
| `WRCFG` scratch words | 3 | `SCRATCH_SEC0`, `SCRATCH_SEC2` |
| `CFGSHIFTMASK` | 5 | Pack output address and unpack base/context address |

Scratch registers should remain typed whole-word registers. `CFGSHIFTMASK` is
an atomic transformation rather than a normal field write, so add a distinct
operation such as `transform_word<Reg>(operation, scratch)`; hiding it behind
`write()` would lose important semantics.

## Multi-field write interface

`hal::cfg::write` now accepts field assignments from different generated
structs when those fields resolve to the same physical CFG word:

```cpp
hal::cfg::write<hal::cfg::Access::TensixCfgUnit>(
    hal::cfg::set<hal::cfg::AluFormatSpecReg0::SrcA, hal::cfg::Sec::S0>(src_a),
    hal::cfg::set<hal::cfg::AluFormatSpecReg1::SrcB, hal::cfg::Sec::S0>(src_b),
    hal::cfg::set<hal::cfg::AluFormatSpecReg2::Dstacc, hal::cfg::Sec::S0>(dst),
    hal::cfg::set<hal::cfg::AluAccCtrl::Fp32_enabled, hal::cfg::Sec::S0>(fp32));
```

Although these fields belong to four generated structs, all resolve to CFG word
1. `write()` groups them by resolved register file and address, combines their
masks and shifted values, and rejects overlapping fields at compile time.

Fields targeting different physical words can remain in the same call. The HAL
groups them automatically and emits distinct words in first-address-seen order:

```cpp
hal::cfg::write<hal::cfg::Access::MMIO>(
    hal::cfg::set<hal::cfg::Pck0AddrCtrlXyReg0::Xstride, hal::cfg::Sec::S0>(x),
    hal::cfg::set<hal::cfg::Pck0AddrCtrlXyReg0::Ystride, hal::cfg::Sec::S0>(y),
    hal::cfg::set<hal::cfg::Pck0AddrCtrlZwReg0::Zstride, hal::cfg::Sec::S0>(z),
    hal::cfg::set<hal::cfg::Pck0AddrCtrlZwReg0::Wstride, hal::cfg::Sec::S0>(w));
```

Hardware constraints still apply:

- MMIO emits one store for a full word or one read-modify-write for a partial
  word.
- The Tensix backend emits only the `RMWCIB` byte lanes touched by the combined
  mask; one logical word update may therefore be several instructions.
- Thread fields sharing a word produce one `SETC16`. Because `SETC16` replaces
  the whole 16-bit word, unspecified bits are written as zero.
- Different word addresses require one hardware update per word even though
  they share one C++ call.

## Proposed API layers

Use three layers:

1. **Generated silicon descriptors** (`cfg.h`): address, field width, mask,
   register file, section count/stride.
2. **Mechanism-complete primitive access** (`hal/cfg.h`):
   field RMW, whole-word read/write, multiword read/write, snapshot/restore,
   and atomic transform. Backends remain explicit (`MMIO` versus
   `Access::TensixCfgUnit`; runtime versus compile-time emission).
3. **Logical facades**: `UnpackTileConfig`, `UnpackAddressing`,
   `PackAddressing`, `PackLayout`, `MathFormatConfig`, and thread-policy word
   composers.

Do not encode logical grouping into the generated YAML hierarchy. The YAML
describes physical registers; logical facades often span several physical
registers and need validation that a generated register description cannot
express.

## Migration status

Steps follow the original migration order; status re-measured on 2026-08-25.

1. **Repeated single-field RMWs (ALU/format, haloize, pack counters) — in
   progress.** `cfg_reg_rmw_tensix<>` is down from 141 to 117 sites;
   `ALU_ACC_CTRL` (38), `ALU_FORMAT_SPEC*` (26), and the THCON
   tile-descriptor/format families remain the largest clusters.
2. **Safe single-field thread writes — partially done.** Sync tracking
   (`TENSIX_TRISC_SYNC`) is migrated; `CLR_DVALID` (17),
   `DISABLE_IMPLIED_SRCA_FMT` (12), and `FP16A_FORCE` (2) `TTI_SETC16`
   sites remain.
3. **Whole-word and multiword primitives; MMIO `cfg[]` writes — mostly
   done.** All nine `cfg_rmw` field writes and all MMIO `cfg[]` reads are
   migrated; 15 of the 30 baseline `cfg[]` whole-word writes remain
   (unpacker base/context addresses on hot paths).
4. **Typed reads and snapshot/restore — not started.** All 15 `RDCFG`
   save/modify/restore paths and one `cfg_read` remain raw.
5. **Packed thread-word composers — done.** `UNPACK_MISC_CFG`, `SRCA_SET`,
   and the address-modifier tables are written through `hal::cfg::write`;
   no raw `SETC16` sites for these words remain outside `llk_lib/hal/`.
6. **Logical facades — not started;** blocked on completing the primitive
   coverage above.

## Verification searches

Use these as migration ratchets:

```bash
rg -n 'TTI_(WRCFG|RDCFG|SETC16|CFGSHIFTMASK|RMWCIB[0-3])' tt_llk_blackhole
rg -n '\bTT_(WRCFG|RDCFG|SETC16|CFGSHIFTMASK|RMWCIB[0-3])\s*\(' tt_llk_blackhole
rg -n 'TT_OP_(WRCFG|RDCFG|SETC16|CFGSHIFTMASK|RMWCIB[0-3])' tt_llk_blackhole
rg -n 'cfg_reg_rmw_tensix<' tt_llk_blackhole
rg -n '\bcfg\s*\[[^]]+\]\s*=' tt_llk_blackhole
rg -n '=\s*cfg\s*\[[^]]+\]' tt_llk_blackhole
rg -n '\bcfg_(read|write|rmw|rmw_gpr)\s*\(' tt_llk_blackhole
```

Expected raw-site counts should only decrease as sites move behind `hal::cfg`.
Keep the low-level implementations in `ckernel_ops.h`, `ckernel.h`, and
`hal/cfg.h` allowlisted rather than trying to eliminate the instruction
names completely.
