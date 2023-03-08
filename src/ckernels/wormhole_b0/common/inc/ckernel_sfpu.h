#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

sfpi_inline vInt sfpu_is_fp16_zero(const vFloat& v, uint exponent_size_8)
{
    if (exponent_size_8) {
        // fp16b
        return v == 0.0F;
    } else {
        // fp16a
        // if math data format is fp16, SFPU will convert 5 bit exp to 8 bit exp
        // in grayskull, this unconditionally adds bias value to exp (even for zero)
        vInt tmp = 0x3800; // loads {0, 8'd112, 10'b0}
        tmp += reinterpret<vInt>(v);

        return tmp == 0;
    }
}

sfpi_inline vFloat sfpu_exp(vFloat val)
{
    // If exponent is > -1 extract it and replace with -1
    vInt exp = exexp(val);
    v_if (exp >= 0) {
        val = setexp(val, 126);
    }
    v_endif;

    // Run series in Horner form
    vFloat tmp = val * vConst0p8373 + s2vFloat16b(0.863281);
    val = val * tmp + vConst1;

    v_if (exp >= 0) {
        val = val * val;
        for (int s_iter = 0; s_iter < 7; s_iter++) {
            exp = exp - 1;
            // Narrow predication on each loop
            v_and(exp >= 0);
            val = val * val;
        }
    }
    v_endif;

    return val;
}

template <int max_iter = 3>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in)
{
    // Force sign to 1 (make number negative)
    vFloat val = setsgn(in, 1);

    val = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    // Use 1.44 as first guess at x, ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
    vFloat vConstLn2Recip = vConstFloatPrgm0;
    vFloat two = vConstFloatPrgm1;
    vFloat result = vConstLn2Recip * (val * vConstLn2Recip + two);

    for (int s_iter = 0; s_iter < (max_iter-1); s_iter++) {
        result = result * (val * result + two);
    }

    vInt orig_exp = exexp(in);
    vInt new_exp = exexp(result);

    // "Subtract" exponents, and re-bias.
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp < 0) {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result = 0.0F;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    return setexp(result, new_exp);
}

inline void init_dropout_seed(uint16_t p2){
    FWLOG1("calculate_dropout() -- input seed:%x", p2);

    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);

    uint16_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint16_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    uint16_t per_tensix_input_seed = p2 ^ (my_x << my_y);

    FWLOG1("calculate_dropout() -- calculated seed:%x", per_tensix_input_seed);

    LRegAssigner lra;
    vInt result = lra.assign(LRegs::LReg3);

    vInt tmp = vConstTileId << 10;
    vInt ptis = per_tensix_input_seed;
    result = ~(tmp & ptis) & (tmp | ptis);
}

template <bool APPROXIMATION_MODE>
inline void configure_programmable_constants(SfpuType operation)
{
    switch (operation) {
    case SfpuType::exponential:
        if (APPROXIMATION_MODE) {
            vConstFloatPrgm0 = 1.442695f; // ln2_recip
            vConstFloatPrgm1 = s2vFloat16b(p_exp::C23_73);
            vConstFloatPrgm2 = s2vFloat16b(p_exp::ADJ_EXP);
            break;
        }

        // Fall through
    case SfpuType::gelu_derivative:
        vConstFloatPrgm2 = 0.863281f;

        // Fall through
    case SfpuType::reciprocal:
        vConstFloatPrgm0 = 1.442695f; // ln2_recip
        vConstFloatPrgm1 = 2.0f;
        break;

    case SfpuType::log:
        // ln2
        vConstFloatPrgm0 = 0.692871f; // ln2

        // XXXXX could do these to higher precision
        vConstFloatPrgm1 = 0.1058f;
        vConstFloatPrgm2 = -0.7166f;
        break;

    case SfpuType::sqrt:
        if (APPROXIMATION_MODE) {
            vConstFloatPrgm0 = s2vFloat16b(127 << 7);
        } else {
            vConstFloatPrgm0 = s2vFloat16b(0x5f37);
        }
        break;

    case SfpuType::dropout:
        vConstIntPrgm0 = 0xb400;
        vConstIntPrgm1 = 0x1; // binary 0b1 - used to extract LSB
        break;

    default:
        // Should result in compile time error??
        break;
    }
}

template <bool APPROXIMATION_MODE>
inline void sfpu_init(SfpuType operation, uint param0 = 0)
{
    configure_programmable_constants<APPROXIMATION_MODE>(operation);
    uint imm0;
    uint imm1;
    uint imm2;
    switch (operation) {
    case SfpuType::tanh:
    case SfpuType::tanh_derivative:
        imm0 = 0x1DFF; //0.90625*x
        imm1 = 0x481A; //0.09375*x + 0.8125
        imm2 = 0xFF00; //1
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::sigmoid:
        imm0 = 0x3DFF;
        imm1 = 0x21D8;
        imm2 = 0xFF10;
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::gelu_derivative:
        imm0 = 0x28FF;
        imm1 = (APPROXIMATION_MODE)? 0x313C : 0x3020;
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        break;
    case SfpuType::gelu:
        imm0 = 0x18FF;
        imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
        imm2 = 0xFF00;
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::dropout:
        init_dropout_seed(param0);
        break;
    default:
        // Should result in compile time error??
        break;
    }
}

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_exponential_body(vFloat in)
{
    vFloat out;

    if constexpr (APPROXIMATION_MODE)
    {
        constexpr int FRAC_BITS = 3;
        constexpr uint SP_BIAS = 127 << FRAC_BITS;

        // * by 1/ln2 and add convert to 7.3 FxP format
        vFloat vConstLn2Recip = vConstFloatPrgm0;
        vFloat conv = in * vConstLn2Recip;

        // Clear exp bits
        vInt c23_73 = p_exp::C23_73;
        vInt tmp = reinterpret<vInt>(conv) - c23_73;

        // Add bias
        tmp += SP_BIAS;

        // SHL to move integer bits to exponent
        out = reinterpret<vFloat>(tmp << (10 - FRAC_BITS));
    }
    else
    {
        // Force sign to 0 (make number positive)
        out = sfpu_exp(setsgn(in, 0));

        v_if (in < 0) {
            out = sfpu_reciprocal(out);
        }
        v_endif;
    }

    return out;
}

/*
template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN>
void calculate_cube(uint16_t exp_base_scale_factor = 0)
{
    for (int d = 0; d < 8; d++)
    {

        TTI_SFPLOAD(p_sfpu::LREG3, 0, 0); // load from dest
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPNOP; TTI_SFPNOP;
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);
        TTI_SFPNOP; TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG2, 0, 0); // Store from lreg[1] into dest registers
        TTI_INCRWC(0, 2, 0, 0);
    }
}
*/

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN>
void calculate_exponential(uint16_t exp_base_scale_factor = 0)
{
    // Unroll 8 best for approx, unroll 0 for precise, compiler figures this out
    for (int d = 0; d < 8; d++)
    {
        vFloat val = dst_reg[0];
        if constexpr(SCALE_EN){
            val = val * s2vFloat16a(exp_base_scale_factor);
        }
        if constexpr (APPROXIMATION_MODE)
        {
            // * by 1/ln2 and add convert to 7.3 FxP format
            vFloat vConstLn2Recip = vConstFloatPrgm0;
            vFloat c23_73 = vConstFloatPrgm1;
            vInt adj_exp = vConstIntPrgm2;
            val = val * vConstLn2Recip + c23_73;

            // Remove Exponent of 7 and bias the Mantissa to 127.
            vInt val_short = adj_exp + reinterpret<vInt>(val);

            // SHL to move integer bits to exponent
            val_short <<= 10 - p_exp::FRAC_BITS;
            dst_reg[0] = reinterpret<vFloat>(val_short);

            // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
            // without using Relu in Packer to clamp -ve Infinity to 0.
            if constexpr (ZERO_NEGATIVE)
            {
                v_if (val_short < 0) {
                    dst_reg[0] = vConst0;
                }
                v_endif;
            }
        }
        else
        {
            // Force sign to 0 (make number positive)
            vFloat result = sfpu_exp(setsgn(val, 0));

            v_if (val < 0) {
                result = sfpu_reciprocal(result);
            }
            v_endif;

            dst_reg[0] = result;
        }

        dst_reg++;
    }
}
template <bool APPROXIMATION_MODE>
inline vFloat calculate_gelu_core(vFloat in)
{
    // SFPU microcode:
    // result = (APPROX_MODE == 1)
    //   ? (1 + erf(x/sqrt(2)))
    //   : (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )
    vFloat result;
    if constexpr (APPROXIMATION_MODE) {
        result = in;
    } else {
        // f = (0.044715*x^3 + x)
        result = (in * in) * (in * s2vFloat16b(0.044715f)) + in;
        result *= s2vFloat16b(0.79788f);
    }

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_gelu()
{
    LRegAssigner lra;
    vUInt l0 = lra.assign(LRegs::LReg0);
    vUInt l1 = lra.assign(LRegs::LReg1);
    vUInt l2 = lra.assign(LRegs::LReg2);
    vFloat half = 0.5f;

    #pragma GCC unroll 8
    for (int d = 0; d < 8; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = calculate_gelu_core<APPROXIMATION_MODE>(in);

        vFloat half_in = in * half;
        result = lut(result, l0, l1, l2);
        result = half_in * result + half_in;

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_sigmoid()
{
    LRegAssigner lra;
    vUInt l0 = lra.assign(LRegs::LReg0);
    vUInt l1 = lra.assign(LRegs::LReg1);
    vUInt l2 = lra.assign(LRegs::LReg2);

    #pragma GCC unroll 8
    for (int d = 0; d < 8; d++)
    {
        vFloat val = dst_reg[0];

        dst_reg[0] = lut(val, l0, l1, l2) + 0.5f;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_tanh()
{
    // SFPU microcode
    LRegAssigner lra;
    vUInt l0 = lra.assign(LRegs::LReg0);
    vUInt l1 = lra.assign(LRegs::LReg1);
    vUInt l2 = lra.assign(LRegs::LReg2);

    #pragma GCC unroll 8
    for (int d = 0; d < 8; d++)
    {
        vFloat val = dst_reg[0];
        val = lut(val, l0, l1, l2);
        dst_reg[0] = val;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_hardtanh(uint param0, uint param1, uint param2)
{
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    vFloat p0 = s2vFloat16(param0);
    vFloat p1 = s2vFloat16(param1);
    vFloat p2 = s2vFloat16(param2);
    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++)
    {
        vFloat val = dst_reg[0];

        val += p0;// 12 bits
        v_if (val < 0.0f) {
            val = 0.0f;
        }
        v_endif;

        val += p1;// 12 bits
        v_if (val >= 0.0f) {
            val = 0.0f;
        }
        v_endif;

        val += p2;// 12 bits

        dst_reg[0] = val;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH>
inline void calculate_tanh_derivative()
{
    LRegAssigner lra;
    vUInt l0 = lra.assign(LRegs::LReg0);
    vUInt l1 = lra.assign(LRegs::LReg1);
    vUInt l2 = lra.assign(LRegs::LReg2);

    // tanh'(x) = 1 - (tanh(x))^2
    for (int d = 0; d < 8; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr (!WITH_PRECOMPUTED_TANH) {
            val = lut(val, l0, l1, l2);
        }

        val = val * (-val) + vConst1;
        dst_reg[0] = val;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_gelu_derivative()
{
    constexpr uint imm2 = 0xFF10;

    LRegAssigner lra;
    vUInt l0 = lra.assign(LRegs::LReg0);
    vUInt l1 = lra.assign(LRegs::LReg1);

    // SFPU microcode:
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++)
    {
        vFloat in = dst_reg[0];
        vFloat neg_half_sq_in = in * in * -0.5f;

        // exp = e^(val)
        vFloat exp = calculate_exponential_body<false>(neg_half_sq_in);

        // exp = exp * 1/sqrt(2*pi)
        vFloat partial = exp * in * s2vFloat16b(0.3989423F);

        vFloat result = calculate_gelu_core<true>(in);

        result = lut(result, l0, l1, imm2);

        dst_reg[0] = partial + result + 0.5f;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_reciprocal()
{
    #pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat in = dst_reg[0];
        vFloat out = sfpu_reciprocal<APPROXIMATION_MODE ? 2 : 3>(in);

        v_if (in < 0.0F) {
            // Invert sign on calculated value if CC=1 (number is negative)
            out = -out;
        }
        v_endif;

        dst_reg[0] = out;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS=2>
inline void calculate_sqrt()
{
    #pragma GCC unroll 8
    for (int d = 0; d < 8; d++)
    {
        vFloat val = dst_reg[0];

        if constexpr (APPROXIMATION_MODE)
        {
            vUInt magic = vConstIntPrgm0;

            //sqrt initial approximation
            // adjust bias
            vUInt val_s = magic + reinterpret<vUInt>(val);

            // approximation of square root
            val_s >>= 1;
            dst_reg[0] = reinterpret<vFloat>(val_s);
        }
        else
        {
            // Recip root method
            //// Init approx
            //u.i = SQRT_MAGIC_F - (u.i >> 1);
            v_if (val != 0.0f)
            {
                vUInt magic = vConstIntPrgm0;
                vFloat approx = reinterpret<vFloat>(magic - (reinterpret<vUInt>(val) >> 1));

                //Reciproot iterations
                for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
                {
                    //x*r*(1.5f - xhalf*r*r);
                    approx = ((approx * approx) * (val * -0.5f) + 1.5f) * approx;
                }

                dst_reg[0] = approx * val;
            }
            v_endif;
        }

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_dropout(uint prob, uint scale)
{
    // SFPU microcode

    FWLOG1("calculate_dropout() -- prob:%x", prob);
    FWLOG1("calculate_dropout() -- scale:%x", scale);

    LRegAssigner lra;
    vUInt rand = lra.assign(LRegs::LReg3);

    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        ////////////////////////
        // Scale samples
        ///////////////////////
        dst_reg[0] = dst_reg[0] * s2vFloat16b(scale);

        ////////////////////////
        // Drop samples
        ///////////////////////
        v_if (rand < prob) {
            dst_reg[0] = vConst0;
        }
        v_endif;

        ////////////////////////
        // 16-bit PRNG update
        ///////////////////////
        vUInt lfsr = vConstIntPrgm1;
        vUInt tmp = lfsr & rand;
        rand = rand >> 1;
        v_if (tmp != 0) {
            vUInt mask = vConstIntPrgm0;
            rand ^= mask;
        }
        v_endif;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_lrelu(uint slope)
{
    // SFPU microcode
    vFloat s = s2vFloat16b(slope);

    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];

        v_if (v < 0.0f) {
            v *= s;
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_power(uint exponent)
{
    for (int d = 0; d < 8; d++)
    {
        vFloat in = dst_reg[0];
        vFloat result = in * in;
        for (uint i = 2; i < exponent; i++) {
            result *= in;
        }

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_multiply()
{
    for (int d = 0; d < 8; d++)
    {
        TTI_WRCFG(p_gpr_math::DEST_OP0_BASE, 0, DEST_REGW_BASE_Base_ADDR32);
        TTI_SFPNOP;
        TTI_SFPLOAD(2, 0, ADDR_MOD_3, 0);	// load from dest

        TTI_WRCFG(p_gpr_math::DEST_OP1_BASE, 0, DEST_REGW_BASE_Base_ADDR32);
        TTI_SFPNOP;
        TTI_SFPLOAD(3, 0, ADDR_MOD_3, 0);	// load from dest

        TTI_SFPMUL(2, 3, p_sfpu::LCONST_0, 1, 0); //lreg1 = x^2
        TTI_SFPNOP;

        TTI_SFPSTORE(1, 0, ADDR_MOD_3, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
template <bool HAS_BASE_SCALING>
sfpi_inline void calculate_log_body(const uint log_base_scale_factor)
{
    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    vFloat in = dst_reg[0];
    vFloat x = setexp(in, 127);    // set exp to exp bias (put in range of 1-2)

    // XXXXXX ask Namal? if we can derive the coefficients below to higher precision
    ////////////////////////////
    // Calculate Cheby Approximation using Horner Form Multiplication: 3rd Order
    // x* ( x* (A*x + B) + C) + D
    // A :0.1058, B: -0.3942, C: 0.9813, D: 0.006
    // Run above on (x-1) so x is in ln(x+1), plug (x-1 into equation above to
    // save the subtract and get A',B',C',D'):
    // A' = A
    // B' = -3A + B
    // C' = 3a -2B + C
    // D' = -A + B - C + D
    // A':0.1058, B':-0.7116, C':2.0871, D':-1.4753
    ////////////////////////////
    vFloat a = vConstFloatPrgm1;
    vFloat b = vConstFloatPrgm2;
    // XXXXX try variants of the below: B'=.7122, C'=2.0869
    vFloat series_result = x * (x * (x * a + b) + 2.0871) + -1.4753f;

    ////////////////////////////
    // Convert exponent to float
    ////////////////////////////
    vInt exp = exexp(in);
    v_if (exp < 0) {
        exp = setsgn(~exp + 1, 1);
    }
    v_endif;

    vFloat expf = int2float(exp, 0);
    vFloat vConstLn2 = vConstFloatPrgm0;
    vFloat result = expf * vConstLn2 + series_result; // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        result *= s2vFloat16a(log_base_scale_factor);
    }

    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    v_if (in == 0.0F) { // Reload for register pressure
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;

    dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING>
inline void calculate_log(uint log_base_scale_factor)
{
    #pragma GCC unroll 4
    for(int d = 0; d < 8; d++){
        calculate_log_body<HAS_BASE_SCALING>(log_base_scale_factor);
        dst_reg++;
    }
}

sfpi_inline void calculate_comp_init_flag(bool check, vFloat& flag1, vFloat& flag2, float init)
{
    flag1 = init;
    if (check) {
        flag2 = init;
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE>
inline void calculate_comp(uint exponent_size_8)
{
    //invert output and use same comparison check
    constexpr bool invert_output = ((COMP_MODE == SfpuType::greater_than_equal_zero) ||
                                    (COMP_MODE == SfpuType::not_equal_zero) ||
                                    (COMP_MODE == SfpuType::greater_than_zero));

    // output_0 and output_1 hold the outputs use use when a zero or negative check is true/false.
    // False = 0.0 = kCONST_0 (5/8-bit exponent format)
    // True  = 1.0 = kCONST_1_FP16B (8-bit exponent format)
    // SFPU uses 8-bit exponent in operations so loading these constants in 8-bit exponent format.
    // Although a command flag can tell SFPU to re-bias a 5-bit exponent to 8-bit, we are loading 8-bit
    // exponent and telling SFPU to not add any bias to these constants.
    constexpr float output_0 = invert_output ? 0.0f : 1.0f;
    constexpr float output_1 = invert_output ? 1.0f : 0.0f;

    constexpr bool check_zero = (COMP_MODE == SfpuType::equal_zero) || (COMP_MODE == SfpuType::not_equal_zero);
    constexpr bool second_check = (COMP_MODE == SfpuType::less_than_equal_zero) || (COMP_MODE == SfpuType::greater_than_zero);

    for (int d = 0; d < 8; d++)
    {
        vFloat v = dst_reg[0];
        vFloat flag1, flag2;
        if constexpr(check_zero)
        {
            v_if (sfpu_is_fp16_zero(v, exponent_size_8)) {
                calculate_comp_init_flag(second_check, flag1, flag2, output_0);
            } v_else {
                calculate_comp_init_flag(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }
        else
        {
            v_if (v < 0.0F) {
                calculate_comp_init_flag(second_check, flag1, flag2, output_0);
            } v_else {
                calculate_comp_init_flag(second_check, flag1, flag2, output_1);
            }
            v_endif;
        }

        vFloat result;
        if constexpr (second_check)
        {
            // SfpuType::less_than_equal_zero
            // flag1 = 0x3F80(1.0) if DST < 0 else 0
            // flag2 = 0x3F80(1.0) if DST == 0 else 0
            // Do a bitwise Or (flag1 | flag2) to get <= condition.
            // flag1 < 0 OR flag2 == 0 => DST is Less than or Equal to zero.
            // Result will be either 0x0000(0.0) or 0x3F80(1.0)
            if constexpr (COMP_MODE == SfpuType::less_than_equal_zero) {
                result = reinterpret<vFloat>(reinterpret<vUInt>(flag1) | reinterpret<vUInt>(flag2));
            }
            else
            {
                // SfpuType::greater_than_zero
                // flag1 = 0x3F80(1.0) if DST >= 0 else 0
                // flag2 = 0x3F80(1.0) if DST != 0 else 0
                // Do a bitwise And (flag1 & flag2) to get > condition.
                // flag2 >= 0 AND flag1 != 0 => DST is Greater than zero
                // Result will be either 0x0000(0.0) or 0x3F80(1.0)
                result = reinterpret<vFloat>(reinterpret<vUInt>(flag1) & reinterpret<vUInt>(flag2));
            }
        } else {
            result = flag1;
        }

        dst_reg[0] = result;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_clamp(uint param0, uint param1, uint param2)
{
    // All params are in FP16 format
    // param0 = min
    // param1 = max

    //uint format = (param0 >> 16)&0x1;
    s2vFloat16::Format format = s2vFloat16::fp16a;

    // SFPU microcode
    vFloat min = s2vFloat16(param0, format);
    vFloat max = s2vFloat16(param1, format);
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++)
    {
        vFloat val = dst_reg[0];

        v_if (val < min) {
            val = s2vFloat16(param0, format);
        } v_elseif (val >= max) {
            val = s2vFloat16(param1, format);
        }
        v_endif;

        dst_reg[0] = val + s2vFloat16b(param2); // 12 bits

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_abs()
{
    // SFPU microcode
    for (int d = 0; d < 8; d++)
    {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void calculate_sign(uint exponent_size_8)
{
    // All params are in FP16 format
    // uint format = 1;
    for (int d = 0; d < 8; d++)
    {
        vFloat v = dst_reg[0];
        dst_reg[0] = vConst1;
        v_if (v < 0.0F) {
            dst_reg[0] = vConstNeg1;
        }
        v_endif;

        //param0 == 0 is Bfp8 format. It does not require bias removal.
        //param0 != 0 is Float16 format and exp bias needs to be removed for zero check.
        v_if (sfpu_is_fp16_zero(v, exponent_size_8)) {
            dst_reg[0] = vConst0;
        }
        v_endif;

        dst_reg++;
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int SfpuType_PARAM=0, int ITERATIONS=8>
inline void calculate_sfpu(uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0)
{
    if constexpr (operation == SfpuType::exponential) {
        calculate_exponential<APPROXIMATION_MODE, APPROXIMATION_MODE, false>(param0);
    }
    else if constexpr (operation == SfpuType::exp_with_base) {
        calculate_exponential<APPROXIMATION_MODE, false, true>(param0);
    }
    else if constexpr (operation == SfpuType::tanh) {
        calculate_tanh<APPROXIMATION_MODE>();
    }
    else if constexpr (operation == SfpuType::hardtanh) {
        calculate_hardtanh<APPROXIMATION_MODE>(param0, param1, param2);
    }
    else if constexpr (operation == SfpuType::gelu) {
        calculate_gelu<APPROXIMATION_MODE>();
    }
    else if constexpr (operation == SfpuType::reciprocal) {
        calculate_reciprocal<APPROXIMATION_MODE, ITERATIONS>();
    }
    else if constexpr (operation == SfpuType::sigmoid) {
        calculate_sigmoid<APPROXIMATION_MODE>();
    }
    else if constexpr (operation == SfpuType::sqrt) {
        calculate_sqrt<APPROXIMATION_MODE, ITERATIONS, 2>();
    }
    else if constexpr (operation == SfpuType::tanh_derivative) {
        calculate_tanh_derivative<APPROXIMATION_MODE, SfpuType_PARAM>();
    }
    else if constexpr (operation == SfpuType::lrelu) {
        calculate_lrelu<APPROXIMATION_MODE>(param0);
    }
    else if constexpr (operation == SfpuType::dropout) {
        calculate_dropout<APPROXIMATION_MODE>(param0, param1);
    }
    else if constexpr (operation == SfpuType::power) {
        calculate_power<APPROXIMATION_MODE>(param0);
    }
    else if constexpr (operation == SfpuType::multiply) {
        calculate_multiply<APPROXIMATION_MODE>();
    }
    else if constexpr (operation == SfpuType::log) {
        calculate_log<APPROXIMATION_MODE, false>(param0);
    }
    else if constexpr (operation == SfpuType::log_with_base) {
        calculate_log<APPROXIMATION_MODE, true>(param0);
    }
    else if constexpr (operation == SfpuType::gelu_derivative) {
        calculate_gelu_derivative<APPROXIMATION_MODE>();
    }
    else if constexpr ((operation == SfpuType::equal_zero) ||
                       (operation == SfpuType::not_equal_zero) ||
                       (operation == SfpuType::less_than_zero) ||
                       (operation == SfpuType::greater_than_equal_zero) ||
                       (operation == SfpuType::less_than_equal_zero) ||
                       (operation == SfpuType::greater_than_zero)) {
        calculate_comp<APPROXIMATION_MODE, operation>(param5);
    }
    else if constexpr (operation == SfpuType::clamp) {
        calculate_clamp<APPROXIMATION_MODE>(param0, param1, param2);
    }
    else if constexpr (operation == SfpuType::abs) {
        calculate_abs<APPROXIMATION_MODE>();
    }
    else if constexpr (operation == SfpuType::sign) {
        calculate_sign<APPROXIMATION_MODE>(param5);
    }
}

} // namespace sfpu
} // namespace ckernel
