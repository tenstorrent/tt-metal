// Compile-only discriminator for converting TopK compare/swap to typed SFPI.
namespace ckernel
{
extern volatile unsigned int* instrn_buffer;
}
#include "sfpi.h"

void topk_typed_compare_swap_probe()
{
    sfpi::vFloat value0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat value2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    auto sorted         = sfpi::min_max(value0, value2, 0xf);
    sfpi::l_reg[sfpi::LRegs::LReg0] = sorted.first;
    sfpi::l_reg[sfpi::LRegs::LReg2] = sorted.second;

    // TopK's SFPU_CONTROL_REG.IndexEn makes the physical SFPSWAP above also
    // update companion index registers L4/L6.  Keep them live to make the
    // missing compiler dataflow visible: the builtin has no index operands or
    // results, so RTL still treats these values as unchanged.
    sfpi::l_reg[sfpi::LRegs::LReg4].in_use();
    sfpi::l_reg[sfpi::LRegs::LReg6].in_use();
}

void topk_typed_transpose_probe()
{
    sfpi::vFloat value0 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vFloat value1 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat value2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vFloat value3 = sfpi::l_reg[sfpi::LRegs::LReg3];
    sfpi::subvec_transp(value0, value1, value2, value3);
    sfpi::l_reg[sfpi::LRegs::LReg0] = value0;
    sfpi::l_reg[sfpi::LRegs::LReg1] = value1;
    sfpi::l_reg[sfpi::LRegs::LReg2] = value2;
    sfpi::l_reg[sfpi::LRegs::LReg3] = value3;

    // Physical SFPTRANSP also transposes the companion L4--L7 group used for
    // TopK indices.  The typed builtin has only four inputs/results and its
    // backend pattern hard-codes just L0--L3.
    sfpi::l_reg[sfpi::LRegs::LReg4].in_use();
    sfpi::l_reg[sfpi::LRegs::LReg5].in_use();
    sfpi::l_reg[sfpi::LRegs::LReg6].in_use();
    sfpi::l_reg[sfpi::LRegs::LReg7].in_use();
}
