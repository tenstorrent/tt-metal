#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

namespace ckernel
{
namespace sfpu
{

struct csfpu_params_t {
    uint leaky_slope;
    uint format;
    uint sfpu_scale_val;
    uint sfpu_rnd_fmt;
    bool sfpu_rnd_unsigned_int;
    bool fp32_acc;
    bool sfpu_stoch_rnd;

    csfpu_params_t()
        : leaky_slope(0x3dd1), format(0) {} 

};
    
inline void sfpu_push_cc()
{
    TTI_SFPPUSHC(0, 0, 0, 0);
}
inline void sfpu_pop_cc()
{
    TTI_SFPPOPC(0, 0, 0, 0);
}

inline void sfpu_comp_cc()
{
    TTI_SFPCOMPC(0, 0, 0, 0);
}

inline void sfpu_toggle_enable_cc()
{
    TTI_SFPENCC(0, 0, 0, 1);
}

inline void sfpu_enable_cc()
{
    TTI_SFPENCC(1, 0, 0, 2);
}

inline void sfpu_disable_cc()
{
    TTI_SFPENCC(0, 0, 0, 2);
}

inline void sfpu_flip_cc_flag()
{
    TTI_SFPSETCC(0, 0, 0, 8);
}

inline void sfpu_set_cc_flag()
{
    TTI_SFPSETCC(1, 0, 0, 1);
}

inline void sfpu_set_cc_from_reg0_sign()
{
    // Set predicate based on sign of lreg[0]
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);
}

inline void sfpu_set_cc_from_reg1_sign()
{
    // Set predicate based on sign of lreg[1]
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 0);
}

inline void sfpu_set_cc_from_reg2_sign()
{
    // Set predicate based on sign of lreg[2]
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 0);
}

inline void sfpu_set_cc_from_reg3_sign()
{
    // Set predicate based on sign of lreg[3]
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, 0);
}

inline void sfpu_set_cc_from_reg3_if_zero()
{
    //if math data format is fp16, SFPU has to convert 5 bit exp to 8 bit exp
    //in grayskull, this unconditionally adds bias value to exp (even for zero)
    //in wormhole, this bug is fixed and zero will be 0b'0
    // Set predicate based on value of lreg[3] being 0
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, 6);
}

inline void sfpu_set_cc_from_reg2_if_zero()
{
    // Set predicate based on value of lreg[2] being 0
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6);
}

inline void sfpu_set_cc_if_sign_neg(uint reg)
{
    TTI_SFPSETCC(0, reg, 0, 0);
}

inline void sfpu_set_cc_if_sign_pos(uint reg)
{
    TTI_SFPSETCC(0, reg, 0, 4);
}

inline void sfpu_set_cc_if_zero(uint reg)
{
    TTI_SFPSETCC(0, reg, 0, 6);
}

inline void sfpu_set_cc_if_not_zero(uint reg)
{
    TTI_SFPSETCC(0, reg, 0, 2);
}

inline void sfpu_load_imm32(const uint dest, const uint upper16, const uint lower16)
{
        TTI_SFPLOADI(dest, 0xA, lower16);  // insmod == A will write the lower bits, and not affect the upper bits; 
        TTI_SFPLOADI(dest, 0x8, upper16);  // insmod == 8 will write the upper bits, and not affect the lower bits; 
}

inline void sfpu_load_imm32(const uint dest, const uint val)
{
        TT_SFPLOADI(dest, 0xA, (val & 0xFFFF));  // insmod == A will write the lower bits, and not affect the upper bits; 
        TT_SFPLOADI(dest, 0x8, (val>>16) & 0xFFFF);  // insmod == 8 will write the upper bits, and not affect the lower bits; 
}

inline void sfpu_access_even_cols() 
{
    // Using insmod == 5 means to use the immediate field as an AND mask; This restores the read/write columns to default value of 0 to point to even columns
    TTI_SFPCONFIG(0xFF3F, 15, 0x5);              // #define TT_SFPCONFIG(imm16_math, config_dest, instr_mod1) 
    TTI_SFPNOP;
}

inline void sfpu_access_odd_cols() 
{
    // Using insmod == 3 means to use the immediate field as an OR mask; This will flip the read/write columns (bits 6 and 7)
    TTI_SFPCONFIG(0x00C0, 15, 0x3);              // #define TT_SFPCONFIG(imm16_math, config_dest, instr_mod1) 
    TTI_SFPNOP;
}

inline void sfpu_exp()
{
    // If exponent is > -1 extract it and replace with -1
    // Extract exponent to lreg[0] (debiased)
    sfpu_toggle_enable_cc();
    // sfpu_set_cc_flag();

    TTI_SFPEXEXP(0, 3, 0, 0xA);

    // If exponent is greater than -1 - set it to -1
    // Set exponent to 126 to make the number in 0-1 range in lreg[3]
    TTI_SFPSETEXP(126, 3, 3, 1);

    sfpu_toggle_enable_cc();

    // Run series in Horner form
    // Load coefficients needed for first mad into lreg[1] and lreg[2]
//  TTI_SFPLOADI(1, 0, 0x3F56); // 0.8373
//  TTI_SFPNOP; TTI_SFPNOP;
    TTI_SFPLOADI(2, 0, 0x3F5D);  // 0.8634
    TTI_SFPMAD(3, p_sfpu::LCONST_0_8373, 2, 2, 0);   // lreg[8] has hard-coded value of 0.8373
    TTI_SFPNOP;

//  TTI_SFPLOADI(1, 0, 0x3F80);  // 1.0077
//  TTI_SFPMAD(3, 2, 6,  3, 0);   // lreg[6] has hard-coded value of 1.0077
    TTI_SFPMAD(3, 2, p_sfpu::LCONST_1, 3, 0);   // lreg[10] has hard-coded value of 1.0 (0x3F80) 
    TTI_SFPNOP;

    // Run predicated loop of squarings
    // Enable predication
    sfpu_toggle_enable_cc();
    // sfpu_set_cc_from_reg0_sign();
    // sfpu_flip_cc_flag();
    TTI_SFPSETCC(0, 0, 0, 4);

    for (uint s_iter = 0; s_iter < 8; s_iter++)
    {
        TTI_SFPMAD(3, 3, p_sfpu::LCONST_0, 3, 0);
        TTI_SFPIADD(0xFFF, 0, 0, 9);
    }

    // Disable predication
    sfpu_toggle_enable_cc();
}

template <int max_iter = 3, uint ADDR_MOD>
inline void sfpu_reciprocal()
{
    if constexpr (max_iter == 1)
    {
        // If we are only doing one iteration of the MAD loop, then we only need to use one LREG for the MAD instructions because we have our "first guess" in a hard-coded register
        // This allows us to avoid having to load back in the original value later on
        TTI_SFPEXEXP(0, 3, 2, 0); // Extract exponent from original number to lreg[2]
    }
    TTI_SFPMOV(0, 3, 3, 1); // invert sign on loaded value

    TTI_SFPSETEXP(126, 3, 3, 1); // Set exponent to 126 to make the number in 0.5-1 range in lreg[3]

    TTI_SFPLOADI(1, 1, 0x4000); // Load 2.0 into lreg[1]
//  TTI_SFPLOADI(0, 0, 0x3F40); // Load first guess at x in lreg[0] (0.75)


    if constexpr (max_iter == 1)
    {
        // If we are only doing one iteration of the MAD loop, then we only need to use one LREG for the MAD instructions because we have our "first guess" in a hard-coded register
        // This allows us to avoid having to load back in the original value later on
        TTI_SFPMAD(3, p_sfpu::LCONST_ln2_recip, 1, 0, 0);  // Use 1.44   as first guess at x (hard-coded in lreg[7])    -- the ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
        TTI_SFPNOP;
        TTI_SFPMAD(p_sfpu::LCONST_ln2_recip, 0, p_sfpu::LCONST_0, 0, 0);  // Use 1.44   as first guess at x (hard-coded in lreg[7])    -- the ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
        TTI_SFPNOP;
    }
    else
    {
        TTI_SFPMAD(3, p_sfpu::LCONST_ln2_recip, 1, 2, 0);  // Use 1.44   as first guess at x (hard-coded in lreg[7])    -- the ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
        TTI_SFPNOP;
        TTI_SFPMAD(p_sfpu::LCONST_ln2_recip, 2, p_sfpu::LCONST_0, 0, 0);  // Use 1.44   as first guess at x (hard-coded in lreg[7])    -- the ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
        TTI_SFPNOP;
    }
    for (uint s_iter = 0; s_iter < (max_iter-1); s_iter++)
    {
        TTI_SFPMAD(3, 0, 1, 2, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(0, 2, p_sfpu::LCONST_0, 0, 0);
        TTI_SFPNOP;
    }

    if constexpr (max_iter == 1)
    {
        TTI_SFPEXEXP(0, 0, 1, 0); // Extract exponent from result to lreg[1]
    }
    else
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // reload original number (pre exponent wrangling) into lreg[3]

        TTI_SFPEXEXP(0, 0, 1, 0); // Extract exponent from result to lreg[1]

        TTI_SFPEXEXP(0, 3, 2, 0); // Extract exponent from original number to lreg[2]

    }
    // Execute: -1 - exp; put result back in lreg[2]
    // Invert EXP from original number and add 1 to it (2's complement)
    // removes the need to explicitly subtract 1
    TTI_SFPNOT(0xFFF, 2, 2, 0);

    // Subtract exponents
    TTI_SFPIADD(0, 1, 2, 4);

    // Re-bias exponent
    TTI_SFPIADD(127, 2, 2, 5);

    sfpu_push_cc();
    sfpu_set_cc_from_reg2_sign();
    // if (lreg[2] < 0) {
        // if rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);         // Move 0 to lreg[0]

        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);         // Move 0 to lreg[2]
    // }
    sfpu_comp_cc();
    // else {}
    sfpu_pop_cc();

    // Set newly denormalized exponent to result exponent field and put number in lreg[0]
    TTI_SFPSETEXP(0, 0, 2, 0);
}

template<uint ADDR_MOD>
inline void sfpu_reciprocal_lreg_reduced(const uint max_iter = 3)
{
    // This function performs reciprocal function without using LREG[1]
    // Perf penalty is that constant 2.0 is being reloaded every time we need it
    // Use this function as part of bigger procedure when one of LREGs keeps some other result
    
    if (max_iter == 1)
    {
        // If we are only doing one iteration of the MAD loop, then we only need to use one LREG for the MAD instructions because we have our "first guess" in a hard-coded register
        // This allows us to avoid having to load back in the original value later on
        
        // Extract exponent from original number to lreg[2]
        TTI_SFPEXEXP(0, 3, 2, 0);
    }
    
    // invert sign on loaded value
    TTI_SFPMOV(0, 3, 3, 1);
    // Set exponent to 126 to make the number in 0.5-1 range in lreg[3]
    TTI_SFPSETEXP(126, 3, 3, 1);
    
    // MT: keep loading 1.0 before it's used to keep LREG[1] unused
    //  TTI_SFPLOADI(1, 1, 0x4000); // Load 2.0 into lreg[1]

    if (max_iter == 1)
    {
        // If we are only doing one iteration of the MAD loop, then we only need to use one LREG for the MAD instructions because we have our "first guess" in a hard-coded register
        // This allows us to avoid having to load back in the original value later on
        
        // lreg[0] = 2.0
        TTI_SFPLOADI(p_sfpu::LREG0, 1, 0x4000);
        // lreg[0] = lreg[3]*1.44 + lreg[0]
        // Use 1.44   as first guess at x (loaded in lreg[13])    -- the ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_ln2_recip, p_sfpu::LREG0, p_sfpu::LREG0, 0);TTI_SFPNOP;
        // lreg[0] = lreg[0]*1.44 + 0
        // Use 1.44   as first guess at x (hard-coded in lreg[13])    -- the ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LCONST_ln2_recip, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);TTI_SFPNOP;
        // Extract exponent from result to lreg[3]
        TTI_SFPEXEXP(0, 0, 3, 0);
    }
    else
    {
        // lreg[7] = 2.0
        TTI_SFPLOADI(p_sfpu::LREG7, 1, 0x4000);
        // lreg[2] = lreg[3]*1.44 + lreg[7]
        // Use 1.44   as first guess at x (hard-coded in lreg[13])    -- the ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
        TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_ln2_recip, p_sfpu::LREG7, p_sfpu::LREG2, 0); TTI_SFPNOP;
        // lreg[0] = 1.44*lreg[2] + 0
        // Use 1.44   as first guess at x (hard-coded in lreg[13])    -- the ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
        TTI_SFPMAD(p_sfpu::LCONST_ln2_recip, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); TTI_SFPNOP;

        for (uint s_iter = 1; s_iter < max_iter; s_iter++){
            // lreg[7] = 2.0
            // lreg[2] = lreg[3]*lreg[0] + lreg[7]
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LREG2, 0);TTI_SFPNOP;
            // lreg[0] = lreg[0]*lreg[2] + 0
            TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);TTI_SFPNOP;
        }

        // reload original number (pre exponent wrangling) into lreg[3]
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0);

        // Extract exponent from original number to lreg[2]
        TTI_SFPEXEXP(0, 3, 2, 0);
        // Extract exponent from result to lreg[3]
        TTI_SFPEXEXP(0, 0, 3, 0);
    }

    // Execute: -1 - exp; put result back in lreg[2]
    // Invert EXP from original number and add 1 to it (2's complement)
    // removes the need to explicitly subtract 1
    TTI_SFPNOT(0xFFF, 2, 2, 0);

    // Subtract exponents
    TTI_SFPIADD(0, 3, 2, 4);

    // Re-bias exponent
    TTI_SFPIADD(127, 2, 2, 5);

    sfpu_push_cc();
    sfpu_set_cc_from_reg2_sign();
    // if (lreg[2] < 0) {
        // if rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        
        // Move 0 to lreg[0]
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        // Move 0 to lreg[2]
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
    // }
    sfpu_comp_cc();
    // else {}
    sfpu_pop_cc();

    // Set newly denormalized exponent to result exponent field and put number in lreg[3]
    TTI_SFPSETEXP(0, 0, 2, 0);
    // lreg[3] = lreg[2]
    TTI_SFPMOV(0x0, 2, 3, 0);
}

inline void init_dropout_seed(uint16_t p2){
    FWLOG1("calculate_dropout() -- input seed:%x", p2);
    
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);

    uint16_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint16_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    uint16_t per_tensix_input_seed = p2 ^ (my_x << my_y);

    FWLOG1("calculate_dropout() -- calculated seed:%x", per_tensix_input_seed);
    
    TTI_SFPMOV(0x0, 15, p_sfpu::LREG2, 0);                                                      // lreg[2] <= lreg15 (tile-id)
    TTI_SFPSHFT(10, 0, p_sfpu::LREG2, 1);                                                       //lreg2 <<=10
    TTI_SFPMOV(0x0, p_sfpu::LREG2, p_sfpu::LREG3, 0);                                           // lreg[3] <= lreg2 (tile-id)
    TT_SFPLOADI(p_sfpu::LREG0, 0xA, per_tensix_input_seed);                                     // insmod == A will write the lower bits, and not affect the upper bits;
    TTI_SFPLOADI(p_sfpu::LREG0, 0x8, 0);                                                        // zero out upper 16-bits
    //TT_SFPLOADI(p_sfpu::LREG0, 0 , per_tensix_input_seed);//0xa94b);                          // lreg[0] <= p2

    // XOR per-tensix random seed with tile id
    // XOR breakdown lreg3 = lreg2 ^ lreg0
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);                                             // lreg2 = lreg2 AND lreg0
                                                                                                //       =  tile-id AND p2
    TTI_SFPNOT(0, p_sfpu::LREG2, p_sfpu::LREG2, 0);                                             // lreg2 = ~lreg2

    TTI_SFPOR(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);                                              // lreg3 = lreg3 OR lreg0
                                                                                                //       = tile-id or p2
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);                                             // lreg3 = lreg2 AND lreg3
                                                                                                //       = ~(tile-id AND p2) AND (tile-id OR p2
}

inline void configure_programmable_constants(){

    TTI_SFPLOADI(0, 0xA, 0x0000);          
    TTI_SFPLOADI(0, 0x8, 0xBF80);
    TTI_SFPCONFIG(0, p_sfpu::LCONST_neg1, 0);

    TTI_SFPLOADI(0, 0xA, 0x6000);          
    TTI_SFPLOADI(0, 0x8, 0x3f31);
    TTI_SFPCONFIG(0, p_sfpu::LCONST_ln2, 0);

    TTI_SFPLOADI(0, 0xA, 0xAA3B);          
    TTI_SFPLOADI(0, 0x8, 0x3FB8);
    TTI_SFPCONFIG(0, p_sfpu::LCONST_ln2_recip, 0);

    TTI_SFPLOADI(0, 0xA, 0x0000);          
    TTI_SFPLOADI(0, 0x8, 0xBF00);
    TTI_SFPCONFIG(0, p_sfpu::LCONST_neg_point_5, 0);

    TTI_SFPLOADI(0, 0xA, 0x0000);          
    TTI_SFPLOADI(0, 0x8, 0x0000);

}

template <bool APPROXIMATION_MODE>
inline void sfpu_init(SfpuType operation, uint param0 = 0) 
{
    configure_programmable_constants();
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
    case SfpuType::gelu:
        imm0 = 0x18FF;
        imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
        imm2 = 0xFF00;
        TTI_SFPLOADI(0, 2, imm0);
        TTI_SFPLOADI(1, 2, imm1);
        TTI_SFPLOADI(2, 2, imm2);
        break;
    case SfpuType::sqrt:
        imm2 = (APPROXIMATION_MODE)? 127 << 7 : 0x5f37;
        TTI_SFPLOADI(2, 0, imm2);
        break;
    case SfpuType::exponential:
        if constexpr(APPROXIMATION_MODE) {
            TTI_SFPLOADI(p_sfpu::LREG0, 0, p_exp::C23_73);
            TTI_SFPLOADI(p_sfpu::LREG2, 0, p_exp::ADJ_EXP);
        }
        break;
    case SfpuType::dropout:
        init_dropout_seed(param0);
        // store binary value of 0b1 - used to extract LSB
        sfpu_load_imm32(p_sfpu::LREG7, 0x0, 0x1);
        break;
    default:
        // Should result in compile time error??
        break;
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_exponential_body()
{
    
    // TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest
    if constexpr (APPROXIMATION_MODE)
    {
        constexpr int FRAC_BITS = 3;
        constexpr int C23_73 = 0x4340; // Based on FRAC_BITS and INT_BITS
                                    //constexpr int INT_BITS		  = 6;
        constexpr uint SP_BIAS = 127 << FRAC_BITS;

        TTI_SFPLOADI(0, 0, C23_73);
        // * by 1/ln2 and add convert to 7.3 FxP format
        TTI_SFPMAD(3, p_sfpu::LCONST_ln2_recip, 0, 3, 0); // lreg3 = lreg3 * 1.442 + lreg[0] (c23_73)
        TTI_SFPNOP;

        // Clear exp bits
        TTI_SFPIADD(0, 3, 0, 0x2); // lreg0 = lreg3 - lreg0

        // Add bias
        TTI_SFPIADD(SP_BIAS, 0, 0, 1); // lreg0 += SP_BIAS

        // SHL to move integer bits to exponent
        TTI_SFPSHFT(10 - FRAC_BITS, 0, 3, 1); // lreg[3] = lreg[0] << 7

        // TTI_SFPSTORE(3, 0, ADDR_MOD, 0); // Store from lreg[2] into dest registers
    }
    else
    {
        // Force sign to 0 (make number positive)
        TTI_SFPSETSGN(0, 3, 3, 1);

        // lreg[3] = exp^(lreg[3])
        sfpu_exp();

        // Load input value, to determine whether reciprocal needs to be run
        // lreg[0] = val
        TTI_SFPLOAD(0, 0, ADDR_MOD, 0);

        // store tentatively e^x
        // reciprocal function relies on reloading input
        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);

        // Enable predication
        sfpu_toggle_enable_cc();
        sfpu_set_cc_from_reg0_sign();

        // CC = sign(X) => reciprocal is conditional
        // if (lreg[0] < 0) {
            // lreg[3] = 1/lreg[3]
            sfpu_reciprocal_lreg_reduced<ADDR_MOD>();
            // TTI_SFPSTORE(3, 0, ADDR_MOD, 0); // Store from lreg[3] into dest registers
        // }
        sfpu_toggle_enable_cc();
    }
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

template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN, uint ADDR_MOD>
void calculate_exponential(uint16_t exp_base_scale_factor = 0)
{
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD, 0); // load from dest
        if constexpr(SCALE_EN){
            TT_SFPLOADI(p_sfpu::LREG0, 1, exp_base_scale_factor);   
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);  // input scaling: x * ln(a) = x * exp_base_scale_factor
            TTI_SFPNOP;
        }
        if constexpr (APPROXIMATION_MODE)
        {
            // * by 1/ln2 and add convert to 7.3 FxP format
            TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_ln2_recip, p_sfpu::LREG0, p_sfpu::LREG3, 0); // lreg3 = lreg3 * 1.442 + lreg[0] (c23_73)
            TTI_SFPNOP;

            // Remove Exponent of 7 and bias the Mantissa to 127.
            // LREG2 already holds 2's complement value so we simply do REG2 + REG3 
            TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // lreg3 = lreg2 + lreg3.

            // SHL to move integer bits to exponent
            TTI_SFPSHFT(10 - p_exp::FRAC_BITS, p_sfpu::LREG3, p_sfpu::LREG3, 1); // lreg3 = lreg3 << 7

            TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD, 0); // Store from lreg[3] into dest registers

            // Needed for fused kernels such as math_row_softmax_tables which call calculate_exponential()
            // without using Relu in Packer to clamp -ve Infinity to 0.
            if constexpr (ZERO_NEGATIVE)
            {
                sfpu_enable_cc();
                sfpu_set_cc_from_reg3_sign();
                TTI_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD, 0); // Store 0 into dest registers
                sfpu_disable_cc();
            }
        }
        else
        {
            // Enable predication
            // sfpu_toggle_enable_cc();
            // sfpu_set_cc_from_reg3_sign();

            // Force sign to 0 (make number positive)
            TTI_SFPSETSGN(0, 3, 3, 1);

            // sfpu_push_cc();
            // sfpu_toggle_enable_cc();

            sfpu_exp();

            TTI_SFPLOAD(1, 0, ADDR_MOD, 0); // load from dest

            if constexpr(SCALE_EN){
                TT_SFPLOADI(p_sfpu::LREG0, 1, exp_base_scale_factor);   
                TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);  // input scaling: x * ln(a) = x * exp_base_scale_factor
                TTI_SFPNOP;
            }

            TTI_SFPSTORE(3, 0, ADDR_MOD, 0); // Store from lreg[3] into dest registers

            // Enable predication
            sfpu_toggle_enable_cc();
            sfpu_set_cc_from_reg1_sign();

            // sfpu_pop_cc();

            // CC = sign(X) => reciprocal is conditional
            sfpu_reciprocal<3, ADDR_MOD>();
            TTI_SFPSTORE(2, 0, ADDR_MOD, 0); // Store from lreg[2] into dest registers
            sfpu_toggle_enable_cc();
        }
        TTI_INCRWC(0, 2, 0, 0);
    }
}
template <bool APPROXIMATION_MODE>
inline void calculate_gelu_core()
{

    constexpr uint imm0 = 0x18FF;
    constexpr uint imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    constexpr uint imm2 = 0xFF00;

    TTI_SFPLOADI(0, 2, imm0); //reload lreg0
    
    // SFPU microcode: 
    // result = (APPROX_MODE == 1) 
    //   ? (1 + erf(x/sqrt(2)))
    //   : (1 + tanh( sqrt(2/pi) * (x + 0.44715*x^3) )
    if constexpr (APPROXIMATION_MODE)
    {
        TTI_SFPLOADI(1, 2, imm1); //reload lreg1
        TTI_SFPLOADI(2, 2, imm2); //reload lreg2
    } else {
        // copy to lreg2
        TTI_SFPMOV(0x0, 3, 2, 0); //lreg2 = lreg3

        //f = (0.044715*x^3 + x)
        TTI_SFPMUL(3, 3, p_sfpu::LCONST_0, 1, 0); //lreg1 = x^2
        TTI_SFPNOP;
        TTI_SFPMUL(1, 3, p_sfpu::LCONST_0, 3, 0); //lreg3 = x*x^2
        TTI_SFPNOP;

        TTI_SFPMULI(0x3d37, 3, 0); //lreg3 = .044715*x3
        TTI_SFPNOP;

        TTI_SFPADD(2, p_sfpu::LCONST_1, 3, 3, 0); //lreg3 = lreg3 + lreg2 (x+lreg1)
        TTI_SFPNOP;
        TTI_SFPLOADI(1, 2, imm1); //reload lreg1
        TTI_SFPLOADI(2, 2, imm2);  //reload lreg2
        TTI_SFPMULI(0x3f4c, 3, 0); //lreg3 = lreg3 * sqrt(2/pi)
        TTI_SFPNOP;
    }

    // sfpu_instr(`SFPU_LUT,0,0,0,0,3,2);
    TTI_SFPLUT(p_sfpu::LREG3, 4, 0);
    TTI_SFPNOP;

    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LCONST_1, 3, 3, 0); // lreg3 = 1 + lreg3
    TTI_SFPNOP;
    TTI_SFPMULI(0x3f00, 3, 0); // lreg3 = lreg3*0.5
    TTI_SFPNOP;
    
    // TTI_SFPSTORE(3, 0, 0);
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_gelu()
{

    constexpr uint imm1 = (APPROXIMATION_MODE)? 0x212C : 0x2010;
    constexpr uint imm2 = 0xFF00;
    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest

        if constexpr (APPROXIMATION_MODE)
        {
            TTI_SFPLOADI(1, 2, imm1); //reload lreg1
            TTI_SFPLOADI(2, 2, imm2); //reload lreg2
        } else {
            // copy to lreg2
            TTI_SFPMOV(0x0, 3, 2, 0); //lreg2 = lreg3

            //f = (0.044715*x^3 + x)
            TTI_SFPMUL(3, 3, p_sfpu::LCONST_0, 1, 0); //lreg1 = x^2
            TTI_SFPNOP;
            TTI_SFPMUL(1, 3, p_sfpu::LCONST_0, 3, 0); //lreg3 = x*x^2
            TTI_SFPNOP;

            TTI_SFPMULI(0x3d37, 3, 0); //lreg3 = .044715*x3
            TTI_SFPNOP;

            TTI_SFPADD(2, p_sfpu::LCONST_1, 3, 3, 0); //lreg3 = lreg3 + lreg2 (x+lreg1)
            TTI_SFPNOP;
            TTI_SFPLOADI(1, 2, imm1); //reload lreg1

            TTI_SFPLOADI(2, 2, imm2);  //reload lreg2
            TTI_SFPMULI(0x3f4c, 3, 0); //lreg3 = lreg3 * sqrt(2/pi)
            TTI_SFPNOP;
        }

        // sfpu_instr(`SFPU_LUT,0,0,0,0,3,2);
        TTI_SFPLUT(p_sfpu::LREG3, 4, 0);
        TTI_SFPNOP;

        TTI_SFPLOAD(1, 0, ADDR_MOD, 0);        // re-load from dest -> lreg1
        TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LCONST_1, 3, 2, 0); // lreg2 = 1+tanh()
        TTI_SFPNOP;
        TTI_SFPMULI(0x3f00, 1, 0); // lreg1 = x*0.5
        TTI_SFPNOP;
        TTI_SFPMUL(1, 2, p_sfpu::LCONST_0, 3, 0); // lreg3 = 		lreg2*lreg1
        TTI_SFPNOP;
        
        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_sigmoid()
{
    constexpr uint add_imm0 = (0 << 15) | (126 << 7) | (0 << 0);
    // SFPU microcode
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest

        // sfpu_instr(`SFPU_LUT,0,0,0,0,3,2);
        TTI_SFPLUT(p_sfpu::LREG3, 4, 0);
        TTI_SFPNOP;

        TTI_SFPADDI(add_imm0, 3, 0);
        TTI_SFPNOP;

        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_tanh()
{
    // SFPU microcode
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest

        // sfpu_instr(`SFPU_LUT,0,0,0,0,3,2);
        TTI_SFPLUT(p_sfpu::LREG3, 4, 0);
        TTI_SFPNOP;

        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_hardtanh(uint param0, uint param1, uint param2)
{
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest

        TT_SFPADDI(param0, 3, 0); // lreg3 = param0 + lreg3
        TTI_SFPNOP;

        // Enable predication
        sfpu_toggle_enable_cc();

        // if lreg[3] < 0
        TTI_SFPSETCC(0, p_sfpu::LREG3, 0, 0);
        TTI_SFPNOP;
        TTI_SFPLOADI(p_sfpu::LREG3, 1, 0x0000);  // lreg[3] = 0

        sfpu_toggle_enable_cc();

        TT_SFPADDI(param1, 3, 0); // lreg3 = param1 + lreg3
        TTI_SFPNOP;

        sfpu_toggle_enable_cc();

        // if lreg[3] >= 0
        TTI_SFPSETCC(0, p_sfpu::LREG3, 0, 4);
        TTI_SFPNOP;
        TTI_SFPLOADI(p_sfpu::LREG3, 1, 0x0000);  // lreg[3] = 0

        // Disable predication
        sfpu_toggle_enable_cc();

        TT_SFPADDI(param2, 3, 0); // lreg3 = param2 + lreg3
        TTI_SFPNOP;

        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH, uint ADDR_MOD>
inline void calculate_tanh_derivative()
{
    // SFPU microcode: tanh'(x) = 1 - (tanh(x))^2
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest

        if constexpr (!WITH_PRECOMPUTED_TANH) {
            TTI_SFPLUT(p_sfpu::LREG3, 4, 0); // lreg3 = tanh(x)
            TTI_SFPNOP;
        }
        TTI_SFPMUL(3, 3, p_sfpu::LCONST_0, 3, 0); // lreg3 = lreg3^2
        TTI_SFPNOP;
        TTI_SFPADD(3, p_sfpu::LCONST_neg1, p_sfpu::LCONST_1, 3, 0); // lreg3 = 1 - lreg3
        TTI_SFPNOP;

        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}



template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_gelu_derivative()
{
    // SFPU microcode:
    #pragma GCC unroll 0 
    for (int d = 0; d < 8; d++)
    {
        // lreg[3] = x;
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0);

        // lreg[3] = lreg[3]^2
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        
        // Reload x to lreg[1] and keep it saved
        // lreg[1] = x;
        TTI_SFPLOAD(1, 0, ADDR_MOD, 0);

        // lreg[3] = x*(-0.5)
        TTI_SFPMULI(0xbf00, 3, 0);TTI_SFPNOP;

        // Dest = lreg[3];
        // Store intermediate result since exp(x) reloads value from dest
        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);

        // lreg[3] = e^(lreg[3])
        calculate_exponential_body<false, ADDR_MOD>(); // Gelu_derivative needs accurate exponent

        // lreg[3] = lreg[3] * 1/sqrt(2*pi)
        TTI_SFPMULI(0x3ecc, 3, 0);TTI_SFPNOP;
        
        // lreg[0] = lreg[3] * lreg[1]
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); TTI_SFPNOP;
        
        // Store intermediate result
        TTI_SFPSTORE(0, 0, ADDR_MOD, 0);

        // lreg[3] = lreg[1]
        // FIXME MT: try to conver this to loadi if possible since its faster. Can do it right away since input value is stored in lreg[1]
        // TTI_SFPLOAD(3, 0, ADDR_MOD, 0);

        // lreg[3] = lreg[1]
        TTI_SFPMOV(0x0, 1, 3, 0);
        
        // lreg[3] = (APPROX_MODE == 1) 
        //   ? 0.5 * (1 + erf(x/sqrt(2)))
        //   : 0.5 * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) )
        calculate_gelu_core<true>();

        // Load previous result
        TTI_SFPLOAD(0, 0, ADDR_MOD, 0);

        // lreg[3] = lreg[3] + lreg[0]
        TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG3, 0); TTI_SFPNOP;

        // dest = lreg[3]
        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);

        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, uint ADDR_MOD>
inline void calculate_reciprocal()
{
    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest

        // // Enable code conditinal
        // sfpu_toggle_enable_cc();
        // sfpu_set_cc_from_reg3_sign();

        // Force sign to 0 (make number positive)
        TTI_SFPSETSGN(0, 3, 3, 1);

        // sfpu_push_cc();

        // Enable predication
        sfpu_toggle_enable_cc();

        if constexpr (APPROXIMATION_MODE)
        {
            sfpu_reciprocal<2, ADDR_MOD>();
        }
        else
        {
            sfpu_reciprocal<3, ADDR_MOD>();
        }

        // sfpu_pop_cc();
        TTI_SFPLOAD(1, 0, ADDR_MOD, 0); // load from dest

        sfpu_set_cc_from_reg1_sign();

        // Invert sign on calculated value if CC=1 (number is negative)
        TTI_SFPMOV(0, 2, 2, 1);

        // Disable code conditional
        sfpu_toggle_enable_cc();

        TTI_SFPSTORE(2, 0, ADDR_MOD, 0); // Store from lreg[2] into dest registers
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, int RECIPROCAL_ITERATIONS=2, uint ADDR_MOD>
inline void calculate_sqrt()
{

    // SFPU microcode
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest
        //TTI_SFPNOP; TTI_SFPNOP;
        if constexpr (APPROXIMATION_MODE)
        {
            //sqrt initial approximation
            // adjust bias
            TTI_SFPIADD(0, 2, 3, 0); // lreg3 += lreg2
            // approximation of square root
            TTI_SFPSHFT(0xfff, 0, 3, 1); // lreg3 >>= 1
            TTI_SFPSTORE(3, 0, ADDR_MOD, 0); // Store from lreg[3] into dest registers
        }
        else
        {
            // check if zero to avoid NAN output
            sfpu_enable_cc();

            // if cond {
            sfpu_set_cc_if_not_zero(p_sfpu::LREG3);

            // Recip root method
            // calculate y_0 = 1/sqrt(x) using fast inverse square root method, then
            // use iteration to come up with better approximation y_n and finally get sqrt(x) by doing x * y_n
            //// Init approx
            //u.i = SQRT_MAGIC_F - (u.i >> 1);
            TTI_SFPMOV(0x0, 3, 0, 0); //lreg0 = lreg3

            TTI_SFPSHFT(0xfff, 0, 0, 1); // lreg0 >>= 1

            TTI_SFPIADD(0, 2, 0, 0x6); // lreg0 = lreg2 - lreg0 (make exponent negative)

            //Reciproot iterations
            for (int r = 0; r < RECIPROCAL_ITERATIONS; r++)
            {
                //x*r*(1.5f - xhalf*r*r);
                TTI_SFPMUL(0, 0, p_sfpu::LCONST_0, 1, 0); //lreg1 = r^2 (lreg0)
                TTI_SFPNOP;
                TTI_SFPMUL(1, 3, p_sfpu::LCONST_0, 1, 0); //lreg1 *= x  (lreg3)
                TTI_SFPNOP;
                TTI_SFPMUL(1, p_sfpu::LCONST_neg_point_5, p_sfpu::LCONST_0, 1, 0); // lreg1 *- -0.5
                TTI_SFPNOP;
                TTI_SFPADDI(0x3fc0, 1, 0); // lreg1 = 1.5f + lreg1
                TTI_SFPNOP;
                TTI_SFPMUL(1, 0, p_sfpu::LCONST_0, 0, 0); // lreg0 = lreg1 * r lreg0
                TTI_SFPNOP;
            }

            TTI_SFPMUL(0, 3, p_sfpu::LCONST_0, 3, 0); // lreg3 = lreg0 x lreg3
            TTI_SFPNOP;

            TTI_SFPSTORE(3, 0, ADDR_MOD, 0); // Store from lreg[3] into dest registers

            // }
            sfpu_disable_cc();
        }

        TTI_INCRWC(0, 2, 0, 0);
    }
}


template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_dropout(uint prob, uint scale)
{
    // SFPU microcode

    //TTI_SFPLOADI(p_sfpu::LREG3, 0 , 0xa94b); //lreg[3] = aeed //PRNG is pre-seeded in lreg3
    FWLOG1("calculate_dropout() -- prob:%x", prob);
    FWLOG1("calculate_dropout() -- scale:%x", scale);

    sfpu_load_imm32(p_sfpu::LREG0, 0x0, 0xB400);                                                    //lreg[0] = mask
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        ////////////////////////
        // Scale samples
        ///////////////////////
        TT_SFPLOADI(p_sfpu::LREG1, 0 , scale);                                                      //lreg[1] = scale
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD, 0);                                                 //lreg[2] = dest
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);               //lreg1 = lreg[2] * lreg[1]
        TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD, 0);                                                // Store lreg1 to dest

        ////////////////////////
        // Drop samples
        ///////////////////////
        sfpu_load_imm32(p_sfpu::LREG2, prob);                                                       // lreg[2] <= prob
        TTI_SFPMOV(0x0, p_sfpu::LREG3, p_sfpu::LREG1, 0);                                           // lreg[1] <= current lfsr (lreg[3])

        
        // Compare random num in lreg[3] and set cc
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG2, 0x2);                                         // lreg2 <= lreg1 - lreg2

        
        sfpu_toggle_enable_cc();
        sfpu_set_cc_from_reg2_sign();
        TTI_SFPSTORE(p_sfpu::LCONST_0, 0, ADDR_MOD, 0);                                             // Store (zero) to dest if random number (LFSR state) is less than prob (an integer)
        sfpu_toggle_enable_cc();
        
        ////////////////////////
        // 16-bit PRNG update
        ///////////////////////
        TTI_SFPMOV(0x0, p_sfpu::LREG3, p_sfpu::LREG2, 0);                                           // lreg[2] <= current lfsr (lreg[3])
        TTI_SFPAND(0x0, p_sfpu::LREG7, p_sfpu::LREG2, 0x0);                                         // lreg[2] <= LSB of current lfsr (lreg[3])
        // PRNG SHR by one

        TTI_SFPSHFT(0xFFF, 0, p_sfpu::LREG3, 0x1);                                                  // lreg3 = lreg3 >> 1

        // if sign is negative 
        sfpu_toggle_enable_cc();
        sfpu_set_cc_if_not_zero(p_sfpu::LREG2);                                                     // Check LSB of LFSR before right shift
        TTI_SFPMOV(0x0, p_sfpu::LREG3, p_sfpu::LREG2, 0);                                           // lreg[2] <= current lfsr (lreg[3]) (shifted version)

        // XOR breakdown
        TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);                                             // lreg2 = lreg2 AND lreg0
        TTI_SFPNOT(0, p_sfpu::LREG2, p_sfpu::LREG2, 0);                                             // lreg2 = ~lreg2
        

        TTI_SFPOR(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);                                              // lreg3 = lreg3 OR lreg0

        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);                                             // lreg3 = lreg2 AND lreg3
        
        sfpu_toggle_enable_cc();

        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_lrelu(uint slope)
{
    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        TTI_SFPLOAD(2, 0, ADDR_MOD, 0);	// load from dest
        
        TT_SFPLOADI(1, 0, slope);
        
        sfpu_toggle_enable_cc();
        sfpu_set_cc_from_reg2_sign();

        TTI_SFPMAD(2, 1, p_sfpu::LCONST_0, 2, 0);
        TTI_SFPNOP;

        sfpu_toggle_enable_cc();
        
        TTI_SFPSTORE(2, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_power(uint exponent)
{
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0); // load from dest
        TTI_SFPMUL(3, 3, p_sfpu::LCONST_0, 1, 0); //lreg1 = x^2
        TTI_SFPNOP;

        for (uint i = 2; i < exponent; i++) {
            TTI_SFPMUL(1, 3, p_sfpu::LCONST_0, 1, 0); //lreg1 = lreg1 * x
            TTI_SFPNOP;
        }

        TTI_SFPSTORE(1, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_multiply()
{
    for (int d = 0; d < 8; d++)
    {
        TTI_WRCFG(p_gpr_math::DEST_OP0_BASE, 0, DEST_REGW_BASE_Base_ADDR32);
        TTI_SFPNOP;
        TTI_SFPLOAD(2, 0, ADDR_MOD, 0);	// load from dest

        TTI_WRCFG(p_gpr_math::DEST_OP1_BASE, 0, DEST_REGW_BASE_Base_ADDR32);
        TTI_SFPNOP;
        TTI_SFPLOAD(3, 0, ADDR_MOD, 0);	// load from dest

        TTI_SFPMUL(2, 3, p_sfpu::LCONST_0, 1, 0); //lreg1 = x^2
        TTI_SFPNOP;

        TTI_SFPSTORE(1, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}
template <bool HAS_BASE_SCALING, uint ADDR_MOD>
inline void calculate_log_body(const uint log_base_scale_factor)
{
    ////////////////////////////
    // Load From dest + "normalize to calculation range"
    ////////////////////////////
    // lreg[3] = dest
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD, 0);
    // set exponent to 127 (force the value to range 1-2)
    TTI_SFPSETEXP(127, p_sfpu::LREG3, p_sfpu::LREG3, 1);                             // set exp to exp bias (put in range of 1-2)
    // Subtract a 1 to account for ln(x+1) action
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LCONST_neg1, p_sfpu::LREG3, 0);  // Subtract 1 to get x in ln(x+1) 

    // Load coefficients for Horner form multiplication:
    // x* ( x* (A*x + B) + C) + D
    sfpu_load_imm32(p_sfpu::LREG1, 0x3dd8, 0xadac);         // A :0.1058 * x^3
    sfpu_load_imm32(p_sfpu::LREG2, 0xbec9, 0xd495);         // B: -0.3942 * x^2
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG2, 0);

    sfpu_load_imm32(p_sfpu::LREG1, 0x3f7b, 0x367a);         // C: 0.9813 * x
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG2, 0);

    sfpu_load_imm32(p_sfpu::LREG1, 0x3a1d, 0x4952);         // D: 0.006
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LREG2, 0);

    // Lreg[2] holds the series result
    
    ///////
    // Convert exponent to float
    ///////
    // Reload argument into lreg[3]; Set lreg[1] to 0
    TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD, 0);

    // Extract exponent, debiased, 2's complement integer and convert to sign + magnitude
    sfpu_toggle_enable_cc();
    TTI_SFPEXEXP(0, p_sfpu::LREG3, p_sfpu::LREG1, 2);   // set cc flag based on the sign of the extracted exp
    // TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    // sfpu_set_cc_from_reg0_sign();
    // if (lreg[1] < 0 ) {
        // ~exp + 1
        TTI_SFPNOT(0, p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPIADD(1, p_sfpu::LREG1, p_sfpu::LREG1, 5);     // IADD with immediate op = +1
        // set negative sign
        TTI_SFPSETSGN(1, p_sfpu::LREG1, p_sfpu::LREG1, 1);
    // }
    sfpu_toggle_enable_cc();

    // Cast to fp32
    TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);

    ////////////////////////////
    // Exp Correction + Base Correction
    ////////////////////////////

    // !! lreg[2] contains series result,
    // !! lreg[1] contains exponent converted to float
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_ln2, p_sfpu::LREG2, p_sfpu::LREG2, 0);  // exp correction: ln(1+x) + exp*ln(2)

    if constexpr (HAS_BASE_SCALING) {
        TT_SFPLOADI(p_sfpu::LREG0, 1, log_base_scale_factor);  // load constant for non-natural base
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);  // base correction: ln(x)*log_mult_const, log_mult_const is 1/ln(base)
        TTI_SFPNOP;
    }
    
    ////////////////////////////
    // Base case when input is 0. ln(0) = -inf
    ////////////////////////////
    sfpu_toggle_enable_cc();
    sfpu_set_cc_if_zero(p_sfpu::LREG3);
    // if (lreg[3] == 0) {
        // Load -inf
        sfpu_load_imm32(p_sfpu::LREG2, 0xff80, 0x0000);
    // }
    sfpu_toggle_enable_cc();

    TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD, 0);
}

template <bool APPROXIMATION_MODE, bool HAS_BASE_SCALING, uint ADDR_MOD>
inline void calculate_log(uint log_base_scale_factor)
{
    for(int d = 0; d < 8; d++){
        calculate_log_body<HAS_BASE_SCALING, ADDR_MOD>(log_base_scale_factor);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, SfpuType COMP_MODE, uint ADDR_MOD>
inline void calculate_comp(uint exponent_size_8)
{
    //invert output and use same comparison check
    constexpr bool invert_output = ((COMP_MODE == SfpuType::greater_than_equal_zero) ||
                                    (COMP_MODE == SfpuType::not_equal_zero) ||
                                    (COMP_MODE == SfpuType::greater_than_zero));

    // output_0 and output_1 hold the oututs use use when a zero or negative check is true/false.
    // False = 0.0 = kCONST_0 (5/8-bit exponent format)
    // True  = 1.0 = kCONST_1_FP16B (8-bit exponent format)
    // SFPU uses 8-bit exponent in operations so loading these constants in 8-bit exponent format.
    // Although a command flag can tell SFPU to re-bias a 5-bit exponent to 8-bit, we are loading 8-bit 
    // exponent and telling SFPU to not add any bias to these constants.
    constexpr auto output_0 = invert_output ? p_sfpu::kCONST_0 : p_sfpu::kCONST_1_FP16B;
    constexpr auto output_1 = invert_output ? p_sfpu::kCONST_1_FP16B : p_sfpu::kCONST_0;
    constexpr auto instr_mode = p_sfpu::kCONST_Exp_8Bit;

    constexpr bool check_zero = (COMP_MODE == SfpuType::equal_zero) || (COMP_MODE == SfpuType::not_equal_zero);
    constexpr bool second_check = (COMP_MODE == SfpuType::less_than_equal_zero) || (COMP_MODE == SfpuType::greater_than_zero);

    // SFPU microcode
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD, 0); // load from dest

        sfpu_enable_cc();
        // if cond {
        if constexpr(check_zero)
        {
            if(!exponent_size_8)
            {
                //this loads bias const into LREG[0] 
                //and modifies LREG[3]
                sfpu_set_cc_from_reg3_if_zero();
            }
            else
            {
                sfpu_set_cc_if_zero (p_sfpu::LREG3); // if lreg[3] == 0
            }
        }
        else{
            sfpu_set_cc_from_reg3_sign();
        }
        TTI_SFPLOADI(p_sfpu::LREG3, instr_mode, output_0);
        // } 
        // else {
        sfpu_comp_cc();
        TTI_SFPLOADI(p_sfpu::LREG3, instr_mode, output_1);
        // }
        // Disable predication
        sfpu_disable_cc();

        if constexpr (second_check)
        {
            TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD, 0); // load from dest
            //Enable predication
            sfpu_enable_cc();
            //if cond {
            if (!exponent_size_8)
            {
                sfpu_set_cc_from_reg2_if_zero();
            }
            else
            {
                sfpu_set_cc_if_zero (p_sfpu::LREG2); // if lreg[2] == 0
            }

            TTI_SFPLOADI(p_sfpu::LREG2, instr_mode, output_0);
            // else {
            sfpu_comp_cc();

            TTI_SFPLOADI(p_sfpu::LREG2, instr_mode, output_1);
            // }
            // Disable Predication
            sfpu_disable_cc();

            // sfpu_operation::less_than_equal_zero
            // LREG3 = 0x3F80(1.0) if DST < 0 else 0
            // LREG2 = 0x3F80(1.0) if DST == 0 else 0
            // Do a bitwise Or (LREG2 | LREG3) to get <= condition.
            // LREG3 < 0 OR LREG2 == 0 => DST is Less than or Equal to zero.
            // Result will be either 0x0000(0.0) or 0x3F80(1.0) in LREG3
            if constexpr (COMP_MODE == SfpuType::less_than_equal_zero){
                TTI_SFPOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
            }
            else
            {
                // sfpu_operation::greater_than_zero
                // LREG3 = 0x3F80(1.0) if DST >= 0 else 0
                // LREG2 = 0x3F80(1.0) if DST != 0 else 0
                // Do a bitwise And (LREG2 & LREG3) to get > condition.
                // LREG3 >= 0 AND LREG2 != 0 => DST is Greater than zero 
                // Result will be either 0x0000(0.0) or 0x3F80(1.0) in LREG3
                TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);
            }

        }
        TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_clamp(uint param0, uint param1, uint param2)
{
    // All params are in FP16 format
    // param0 = min
    // param1 = max

    //uint format = (param0 >> 16)&0x1;
    uint format = 1;

    //TT_SETDMAREG(0, format, 0, LO_16(60));
    //TT_SETDMAREG(0, param0, 0, LO_16(60));
    //TT_SETDMAREG(0, param1, 0, LO_16(61));

    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD, 0); // load from dest into lreg3
        TT_SFPLOADI(p_sfpu::LREG2,format,param0); // Load param0 into lreg2 (format fp16_a or fp16_b) 

        TTI_SFPMAD(p_sfpu::LREG2,p_sfpu::LCONST_neg1, p_sfpu::LREG3, p_sfpu::LREG2, 0); // lreg2 = lreg2*(-1) + lreg3
        TTI_SFPNOP;

        // Enable predication
        sfpu_toggle_enable_cc();

        // if lreg[2] < 0
        sfpu_set_cc_if_sign_neg(p_sfpu::LREG2);
        TT_SFPLOADI(p_sfpu::LREG3,format,param0); // Load param0 into lreg3 if (lreg[2]<0)

        sfpu_toggle_enable_cc();

        TT_SFPLOADI(p_sfpu::LREG2,format,param1); // Load param1 into lreg2 (format fp16_a or fp16_b) 

        TTI_SFPMAD(p_sfpu::LREG2,p_sfpu::LCONST_neg1, p_sfpu::LREG3, p_sfpu::LREG2, 0); // lreg2 = lreg2*(-1) + lreg3
        TTI_SFPNOP;

        sfpu_toggle_enable_cc();

        // if lreg[4] > 0
        sfpu_set_cc_if_sign_pos(p_sfpu::LREG2);
        TT_SFPLOADI(p_sfpu::LREG3,format,param1); // Load param1 into lreg3 if (lreg[2]>0)

        // Disable predication
        sfpu_toggle_enable_cc();

        TT_SFPADDI(param2, 3, 0); // lreg3 = param2 + lreg3 (lreg3 = datum-max+max = datum)
        TTI_SFPNOP;

        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_abs()
{
        // SFPU microcode
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD, 0); // load from dest into lreg3
        TTI_SFPABS(0, p_sfpu::LREG3, p_sfpu::LREG2, 1);
        TTI_SFPSTORE(2, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <bool APPROXIMATION_MODE, uint ADDR_MOD>
inline void calculate_sign(uint exponent_size_8)
{
    // All params are in FP16 format
    //uint format = 1;
    // SFPU microcode
    #pragma GCC unroll 0
    for (int d = 0; d < 8; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD, 0); // load from dest into lreg2
        // Enable predication
        sfpu_enable_cc();
        //if sign is negative {
        sfpu_set_cc_if_sign_neg(p_sfpu::LREG2);
        TTI_SFPMOV(0, p_sfpu::LCONST_neg1, p_sfpu::LREG3, 0); // Load -1 into lreg3 if (lreg[2]<0)
        //} else {
        sfpu_comp_cc();
        TTI_SFPMOV(0, p_sfpu::LCONST_1, p_sfpu::LREG3, 0);  // Load +1 into lreg3 if (lreg[2]<0)
        //}
        sfpu_enable_cc();
        //param0 == 0 is Bfp8 format. It does not require bias removal.
        //param0 != 0 is Float16 format and exp bias needs to be removed for zero check.
        //if cond {
        if (!exponent_size_8)
        {
            //if zero
            sfpu_set_cc_from_reg2_if_zero();
        }
        else
        {
            //if zero
            sfpu_set_cc_if_zero (p_sfpu::LREG2); // if lreg[2] == 0
        }
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
        //}
        sfpu_disable_cc();

        TTI_SFPSTORE(3, 0, ADDR_MOD, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int SfpuType_PARAM=0, int ITERATIONS=8, uint ADDR_MOD=ADDR_MOD_3>
inline void calculate_sfpu(uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0)
{
    if constexpr (operation == SfpuType::exponential) {
        calculate_exponential<APPROXIMATION_MODE, APPROXIMATION_MODE, false, ADDR_MOD>(param0);
    }
    else if constexpr (operation == SfpuType::exp_with_base) {
        calculate_exponential<APPROXIMATION_MODE, false, true, ADDR_MOD>(param0);
    }
    else if constexpr (operation == SfpuType::tanh) {
        calculate_tanh<APPROXIMATION_MODE, ADDR_MOD>();
    }
    else if constexpr (operation == SfpuType::hardtanh) {
        calculate_hardtanh<APPROXIMATION_MODE, ADDR_MOD>(param0, param1, param2);
    }
    else if constexpr (operation == SfpuType::gelu) {
        calculate_gelu<APPROXIMATION_MODE, ADDR_MOD>();
    }
    else if constexpr (operation == SfpuType::reciprocal) {
        calculate_reciprocal<APPROXIMATION_MODE, ITERATIONS, ADDR_MOD>();
    }
    else if constexpr (operation == SfpuType::sigmoid) {
        calculate_sigmoid<APPROXIMATION_MODE, ADDR_MOD>();
    }
    else if constexpr (operation == SfpuType::sqrt) {
        calculate_sqrt<APPROXIMATION_MODE, ITERATIONS, 2, ADDR_MOD>();
    }
    else if constexpr (operation == SfpuType::tanh_derivative) {
        calculate_tanh_derivative<APPROXIMATION_MODE, SfpuType_PARAM, ADDR_MOD>();
    }
    else if constexpr (operation == SfpuType::lrelu) {
        calculate_lrelu<APPROXIMATION_MODE, ADDR_MOD>(param0);
    }
    else if constexpr (operation == SfpuType::dropout) {
        calculate_dropout<APPROXIMATION_MODE, ADDR_MOD>(param0, param1);
    }
    else if constexpr (operation == SfpuType::power) {
        calculate_power<APPROXIMATION_MODE, ADDR_MOD>(param0);
    }
    else if constexpr (operation == SfpuType::multiply) {
        calculate_multiply<APPROXIMATION_MODE, ADDR_MOD>();
    }
    else if constexpr (operation == SfpuType::log) {
        calculate_log<APPROXIMATION_MODE, false, ADDR_MOD>(param0);
    }
    else if constexpr (operation == SfpuType::log_with_base) {
        calculate_log<APPROXIMATION_MODE, true, ADDR_MOD>(param0);
    }
    else if constexpr (operation == SfpuType::gelu_derivative) {
        calculate_gelu_derivative<APPROXIMATION_MODE, ADDR_MOD>();
    }
    else if constexpr ((operation == SfpuType::equal_zero) || 
                       (operation == SfpuType::not_equal_zero) ||
                       (operation == SfpuType::less_than_zero) || 
                       (operation == SfpuType::greater_than_equal_zero) ||
                       (operation == SfpuType::less_than_equal_zero) ||
                       (operation == SfpuType::greater_than_zero)) {
        calculate_comp<APPROXIMATION_MODE, operation, ADDR_MOD>(param5);
    }
    else if constexpr (operation == SfpuType::clamp) {
        calculate_clamp<APPROXIMATION_MODE, ADDR_MOD>(param0, param1, param2);
    }
    else if constexpr (operation == SfpuType::abs) {
        calculate_abs<APPROXIMATION_MODE, ADDR_MOD>();
    }
    else if constexpr (operation == SfpuType::sign) {
        calculate_sign<APPROXIMATION_MODE, ADDR_MOD>(param5);
    }
}

} // namespace sfpu
} // namespace ckernel
