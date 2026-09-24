


<table>
<thead>
<tr>
<th width="50%">Current repository code</th>
<th width="50%">Proposed refactored call site</th>
</tr>
</thead>
<tbody>
<tr>
<td valign="top">
<pre><code class="language-cpp">// Mixed path: compile-time mode, runtime data formats
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(
    const std::uint32_t srca_data_format,
    const std::uint32_t srcb_data_format)
{
    // LLK sanitizer hooks
    llk::san::math_operand_configure(
        srca_data_format,
        srcb_data_format);

    // Configure ZEROACC to auto-detect
    // destination bank.
    cfg_reg_rmw_tensix<
        DEST_ACCESS_CFG_
        zeroacc_absolute_tile_mode_RMW >(0);

    TTI_STALLWAIT(
        p_stall::STALL_CFG,
        p_stall::MATH);

    std::uint32_t int8_math_enabled =
        is_int8_or_int32_format(
            srca_data_format) ||
        is_int8_or_int32_format(
            srcb_data_format);

    cfg_reg_rmw_tensix<
        ALU_ACC_CTRL_
        INT8_math_enabled_RMW >(
            int8_math_enabled);

    cfg_reg_rmw_tensix<
        ALU_ACC_CTRL_Fp32_enabled_RMW >(
            is_fp32_dest_acc_en);

    cfg_reg_rmw_tensix<
        ALU_ACC_CTRL_SFPU_Fp32_enabled_RMW >(
            is_fp32_dest_acc_en);

    _configure_default_zero_flag_state_(
        srca_data_format,
        srcb_data_format);
}</code></pre>
</td>
<td valign="top">
<pre><code class="language-cpp">// Mixed path: compile-time mode, runtime data formats
template <bool is_fp32_dest_acc_en = false>
inline void _llk_math_hw_configure_(
    const std::uint32_t srca_data_format,
    const std::uint32_t srcb_data_format)
{
    // LLK sanitizer hooks
    llk::san::math_operand_configure(
        srca_data_format,
        srcb_data_format);

    const bool int8_math_enabled =
        is_int8_or_int32_format(
            srca_data_format) ||
        is_int8_or_int32_format(
            srcb_data_format);

    hal::math::apply({
        .fp32_accumulation =
            is_fp32_dest_acc_en,
        .sfpu_fp32 =
            is_fp32_dest_acc_en,
        .int8_math =
            int8_math_enabled,
        .zeroacc_addressing =
            hal::math::ZeroAccAddressing::
                RelativeToSelectedBank,
    });

    _configure_default_zero_flag_state_(
        srca_data_format,
        srcb_data_format);
}</code></pre>
</td>
</tr>
</tbody>
</table>

## Proposed reusable HAL helper

The hardware-specific fields and ordering would move into a helper such as
`hal/math_config.h`:

```cpp
namespace hal::math {

enum class ZeroAccAddressing {
    RelativeToSelectedBank,
    AbsoluteTile,
};

struct ModeConfig {
    bool fp32_accumulation;
    bool sfpu_fp32;
    bool int8_math;
    ZeroAccAddressing zeroacc_addressing;
};

inline void apply(const ModeConfig& config)
{
    cfg::write<
        cfg::Access::TensixCfgUnit,
        cfg::DestAccessCfg::zeroacc_absolute_tile_mode,
        cfg::Sec::S0>(
        config.zeroacc_addressing ==
        ZeroAccAddressing::AbsoluteTile);

    TTI_STALLWAIT(
        p_stall::STALL_CFG,
        p_stall::MATH);

    // These fields share CFG word 1 and can be updated together.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluAccCtrl::INT8_math_enabled, cfg::Sec::S0>(
            config.int8_math),
        cfg::set<cfg::AluAccCtrl::Fp32_enabled, cfg::Sec::S0>(
            config.fp32_accumulation),
        cfg::set<cfg::AluAccCtrl::SFPU_Fp32_enabled, cfg::Sec::S0>(
            config.sfpu_fp32));
}

} // namespace hal::math
```
