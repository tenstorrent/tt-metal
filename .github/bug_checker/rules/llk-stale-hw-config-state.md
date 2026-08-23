# LLK Stale Hardware Config State Across Kernel Calls

## Description

LLK compute kernels program the Tensix backend by writing *persistent* hardware
configuration: unpacker tile descriptors and strides, packer output formats and
L1 offsets, ALU format-spec / accumulate-control fields, ADDR_MOD slots, and the
software-side trackers that mirror them. None of this state is reset between
kernel invocations. Whatever the *previous* op left in a config register is what
the *next* op starts with.

A bug in this class appears when an op's `_init_` / `_reconfig_` / `_uninit_`
path fails to fully re-establish the state it depends on, so behaviour becomes
dependent on which op ran before it. The symptom is order-dependent: the op is
correct in isolation and in its own unit test, and wrong only when preceded by
some other op in a fused sequence or a program-cache-warmed second call. There
is no crash — the packer emits subtly wrong datums, or the pipeline hangs.

Four concrete shapes recur in tt-metal's history:

- A **reconfig writes only some of the fields** it owns, leaving a sibling field
  at the previous op's value.
- A **full-word `WRCFG_32b` / `cfg[ADDR]=` write clobbers a neighbouring field**
  that shares the same 32-bit config word but belongs to another thread or
  another concern (the classic `STACC_RELU_*` vs
  `ALU_ACC_CTRL_Zero_Flag_disabled_*` collision).
- A **software state tracker goes stale**: code writes a config register
  directly, bypassing the tracker that records what was last programmed, so the
  tracker's skip-if-already-set fast path then suppresses a genuinely needed
  re-apply.
- An **`_init_` / `_uninit_` pair is asymmetric or mis-ordered**, so the op
  leaves global state it never restores, breaking the *next* op rather than
  itself.

This is an actively-audited area: `tt_metal/tt-llk/.claude/skills/` carries
dedicated `reconfig-stall-audit`, `cfg-word-overlap-audit`, and
`srcreg-bank-sync-audit` skills, and the `llk::san::` sanitizer namespace
enforces init/uninit contracts at runtime. Roughly thirty separate merged `fix`
PRs share this root cause, many tagged `[LLK]` / `[SAN]`.

## What to Look For

1. **Partial field update in a reconfig**: a `_reconfig_` / `_init_` function
   that writes some fields of a config register group but not others that the
   same op depends on. Cross-check against the matching `_llk_*_hw_configure_`
   for that unit — every field the full configure sets is a field a reconfig
   must either set or provably not care about. Also diff the
   `tt_llk_wormhole_b0` / `tt_llk_blackhole` / `tt_llk_quasar` siblings: a field
   handled on one arch and missing on another is a strong signal.

2. **Unmasked full-word write to a shared config word**: `TTI_WRCFG(...,
   p_cfg::WRCFG_32b, X_ADDR32)` or a raw `cfg[X_ADDR32] = ...` targeting a word
   that holds more than one named field. This zeroes every sibling field in the
   word. The correct form is a masked read-modify-write —
   `cfg_reg_rmw_tensix<X_RMW>(value)` — which is byte-atomic and touches only
   its own bits.

3. **Direct config write that bypasses a software state tracker**: writing
   `ALU_ACC_CTRL_Zero_Flag_disabled_src` (or any tracked field) directly without
   calling `_invalidate_src_zero_flag_state_()` afterwards. The tracker's
   configurators (`_configure_default_zero_flag_state_`,
   `_configure_unary_preserve_zero_flag_state_`,
   `_configure_mov_ops_zero_flag_state_`) all early-return when
   `src_zero_flag_state` already matches, so a stale tracker silently suppresses
   the next re-apply.

4. **Missing or mis-ordered `_uninit_`**: an `_init_` that installs an ADDR_MOD,
   MOP program, or format override with no `_uninit_` restoring it, or an
   `_uninit_` whose LLK calls run in an order that leaves the unit in a
   half-configured state. Check that `llk::san::operation_init<...>()` is paired
   with the matching `llk::san::operation_uninit<...>()`, and that the API-order
   contract the sanitizer asserts is actually satisfied.

5. **Config rewrite with no execution-unit drain**: config registers that the
   hardware samples *during* an in-flight op must not be reprogrammed while that
   unit is running. Packer config needs a preceding
   `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK)`, unpacker config needs
   `p_stall::UNPACK0` / `UNPACK1`, and math config needs **both** engines:
   `p_stall::MATH | p_stall::WAIT_SFPU`. A `p_stall::THCON`-only stall orders the
   GPR-to-config write but drains no execution unit, and is the classic
   insufficient guard.

## Bad Code Examples

```cpp
// BUG: unmasked full-word write — STACC_RELU_ApplyRelu shares its 32-bit config
// word with ALU_ACC_CTRL_Zero_Flag_disabled_src/dst, which the math thread owns.
// This zeroes the math thread's zero-flag bits every time relu is configured.
inline void _llk_pack_relu_config_(const std::uint32_t config)
{
    TTI_WRCFG(p_gpr::TMP0, p_cfg::WRCFG_32b, STACC_RELU_ApplyRelu_ADDR32);
}
```

```cpp
// BUG: writes the tracked zero-flag register directly, bypassing the tracker.
// src_zero_flag_state still reads DEFAULT, so the next
// _configure_default_zero_flag_state_() call early-returns and never restores
// the flag — the following matmul runs with the wrong zero-substitution setting.
inline void _llk_unpack_to_dest_32b_block_()
{
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    // ... block body ...
    // missing: _invalidate_src_zero_flag_state_();
}
```

```cpp
// BUG: math config reconfig drains only the FPU. The SFPU shares the math path
// and also reads ALU_FORMAT_SPEC, so an in-flight SFPU op can sample the
// half-updated format.
inline void _llk_math_reconfig_data_format_(const std::uint32_t srca_fmt, const std::uint32_t srcb_fmt)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);   // missing | p_stall::WAIT_SFPU
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(srca_fmt);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(srcb_fmt);
}
```

```cpp
// BUG: reconfig updates the SrcA format but leaves the format-derived ch1 Z/Y
// strides at the previous operand's values, so the unpacker walks the new tile
// with the old stride and reads garbage after the first face.
inline void _llk_unpack_reconfig_data_format_srca_impl_(
    const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t tile_size)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_RMW>(unpack_src_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    // missing: reprogram the ch1 Z/Y strides that the new format implies
}
```

```cpp
// BUG: init installs a MOP program and an ADDR_MOD but the uninit only tears
// down the sanitizer contract, leaving the ADDR_MOD slot owned by this op.
// The next op that assumes the default ADDR_MOD silently mis-addresses.
inline void _llk_pack_untilize_uninit_()
{
    llk::san::operation_uninit<llk::san::Operation::PackUntilize>();
    // missing: restore the default packer addrmod / MOP state installed by _init_
}
```

## Good Code Examples

```cpp
// GOOD: masked read-modify-write touches only the relu bits, so the math
// thread's zero-flag bits in the same 32-bit word survive.
inline void _llk_pack_relu_config_(const std::uint32_t config)
{
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, hw_relu_mask>(config);
}
```

```cpp
// GOOD: the direct write is followed by an explicit tracker invalidation, so the
// next configurator re-applies instead of taking its skip-if-already-set path.
inline void _llk_unpack_to_dest_32b_block_()
{
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(1);
    // ... block body ...
    _invalidate_src_zero_flag_state_();
}
```

```cpp
// GOOD: both math engines are drained before the shared ALU config is rewritten.
inline void _configure_src_zero_flag_(const bool disable)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(disable ? 1 : 0);
}
```

```cpp
// GOOD: the reconfig re-establishes every format-derived field it owns, so no
// field carries over from the previously unpacked operand.
inline void _llk_unpack_reconfig_data_format_srca_impl_(
    const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t tile_size)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_RMW>(unpack_src_format);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(unpack_dst_format);
    _llk_unpack_reconfig_tile_shape_srca_</*issue_stall=*/false>(tile_size);
    _configure_default_zero_flag_state_(unpack_dst_format, src_zero_flag_srcb_fmt);
}
```

```cpp
// GOOD: the state the init installed is explicitly restored, and the sanitizer
// contract is closed in the required order.
inline void _llk_pack_untilize_uninit_(const std::uint32_t pack_dst_format)
{
    _llk_pack_configure_addrmod_();                       // restore default addrmod
    _llk_pack_mop_config_<false>(pack_dst_format);        // restore default MOP
    llk::san::operation_uninit<llk::san::Operation::PackUntilize>();
}
```
