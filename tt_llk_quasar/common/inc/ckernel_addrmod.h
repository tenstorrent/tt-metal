// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

/*

OVERVIEW
--------

Each TRISC has six register counters:
  - srcA counter
  - srcB counter
  - Dest counter
  - Bias counter
  - Pack src counter
  - Pack dest counter

These counters manage a single memory address into the corresponding
register file (except the pack src counter, which points into dest, and
the pack dest counter, which points into L1). Some of these counters
also manage a "carriage return" address of the same bit-width. (The
carriage return is a bit tricky to understand and is explained later).

When a math instruction is issued, it will use the srcA and srcB
counters to fetch its input operands, and will use the Dest counter for
saving the result. Likewise, pack instructions use the Pack src counter
to point at data in Dest and the Pack dest counter to point at the
desired packed L1 location. I'm not 100% how the bias counter works, but
I'll come and edit this once I do.

-> Yes, even though we never use TRISC 1 to issue pack instructions, it
   still has its own Pack src and Pack dest counters.

-> Also, to make things more complicated, the Pack src and Pack dest
   counters are actually split into a y-component and a z-component. I
   think the two counter values are added together to compute the final
   address for each transaction.

Anyway, the srcA, srcB, and Dest counters can be updated using SETRWC
and INCRWC instructions. The Bias counter can be set with the SETIBRWC
instruction. Finally, the pack counters can be updated using SETADCXY,
INCADCXY, ADDRCRXY, SETADCZW, INCADCZW, and ADDRCRZW instructions.

Here's the rub: editing the address counters with separate tensix
instructions after every single math/pack instruction is excruciatingly
slow. Instead, each TRISC also has eight so-called "ADDR_MOD registers"
and eight so-called "ADDR_MOD_PACK registers" that can be selected using
a few bits in a math or pack instruction (respectively). Once the
instruction finishes, the selected ADDR_MOD (or ADDR_MOD_PACK) register
will be used to automatically increment the counters.

This header file defines two top-level structs: an addr_mod_t and an
addr_mod_pack_t. Notice that the addr_mod_t struct contains sub-structs
for srcA counter increments, srcB counter increments, Dest counter
increments, and bias increments. (Also something to do with fidelity,
but I have no sweet clue what that means and we're getting rid of it in
Blackhole anyway). Likewise, the addr_mod_pack_t struct contains
increments for the pack counters. This structure mirrors the arrangement
of registers in the Tensix core: math instructions select one of the
ADDR_MOD registers with srcA, srcB, Dest, and Bias increments, whereas
pack instructions select one of the ADDR_MOD_PACK registers with Pack
src and Pack dest increments.






ADDR_MOD BANKS
--------------

To make things a little more complicated, it turns out we only have
enough room in the Tensix instructions for two ADDR_MOD selection bits.
This means we can only choose between four ADDR_MOD registers at a time.
So, the eight ADDR_MOD (and ADDR_MOD_PACK) registers are divided into
two banks of four registers each. We now have an extra bit in the Tensix
thread-private config registers (ADDR_MOD_BANK_SEL_BankSel) that selects
which bank is currently active.

This bit can be set in one of three ways. This first is to simply issue
a SETC16 instruction and directly write the value.

The next way is to make use of the NextBank field of the ADDR_MOD_AB
register; for every math instruction that specifies an ADDR_MOD, this
value is _always_ written to ADDR_MOD_BANK_SEL_BankSel. This means that
you need to be careful about how you set it!
-> The ADDR_MOD_AB_NextBank field will be assigned with the
   addr_mod_t::next_bank::bank field when you issue a call to
   addr_mot_t::set.

The last way is to make use of the NextBank field of the ADDR_MOD_PACK2
register; for every PACR instruction that specifies an ADDR_MOD, this
value is _always_ written to ADDR_MOD_BANK_SEL_BankSel. This means that
you need to be careful about how you set it!
-> The ADDR_MOD_PACK2_NextBank field will be assigned with the
   addr_mod_pack_t::next_bank::bank field when you issue a call to
   addr_mot_pack_t::set.

HOW ADDR_MOD INCREMENTS UPDATE ADDRESS COUNTERS
-----------------------------------------------

The format of ADDR_MODs are slightly different depending on the
particular counter (I guess it's because each counter is slightly
different).

The Bias counter is the simplest. It only has two fields in an ADDR_MOD:
an increment and a clear. If the clear bit is set, the Bias counter will
be reset to zero when that ADDR_MOD is applied. Otherwise, the increment
value will be added to the counter.

 - clear: if 1, reset counter to zero
 - incr:  counter = counter + incr

Unlike Bias counters, the srcA, srcB, and Dest counters also contain a
"carriage return". This is a second counter of the same bit-width as the
main counter (i.e. the one that actually addresses memory). In addition
to using clear and increment fields in the ADDR_MOD, they also use a
third single-bit field called "CR". When this bit is not set, the clear
and increment fields have the same meaning as before. However, when the
bit is is set, the increment will instead be applied to the carriage
return, AND the carriage return is copied into the main counter.

  Case 1, CR bit is set to 0:
   - clear: if 1, reset counter and carriage return to zero. incr ignored
   - incr:  counter         = counter + incr
            carriage_return = <unchanged>

  Case 2, CR bit is set to 1:
   - clear: if 1, reset counter and carriage return to zero. incr ignored
   - incr:  counter         = carriage_return + incr
            carriage_return = carriage_return + incr

For whatever reason, the Dest counter also supports an extra bit in the
ADDR_MOD register called "CToCR", which has the effect of copying the
main counter to the carriage return (after increments are applied):

  Case 3, CToCR bit is set to 1:
   - clear: if 1, reset counter and carriage return to zero. incr ignored
   - incr:  counter         = counter + incr
            carriage_return = counter + incr

As previously mentioned, the Pack src and Pack dest counters are actually
split into a y-component and a z-component. The y-components counters
have carriage returns, but the z-components do not. In other words, the
y-components behave like the srcA/srcB counters, and the z-components
behave like the Bias counter.

*/

#include "tensix.h"

namespace ckernel
{

constexpr uint8_t ADDR_MOD_0 = 0;
constexpr uint8_t ADDR_MOD_1 = 1;
constexpr uint8_t ADDR_MOD_2 = 2;
constexpr uint8_t ADDR_MOD_3 = 3;
constexpr uint8_t ADDR_MOD_4 = 4;
constexpr uint8_t ADDR_MOD_5 = 5;
constexpr uint8_t ADDR_MOD_6 = 6;
constexpr uint8_t ADDR_MOD_7 = 7;

struct addr_mod_t
{
    // CLR, CR, INCR(4 bits)
    struct addr_mod_src_t
    {
        uint8_t incr = 0;
        uint8_t clr  = 0;
        uint8_t cr   = 0;

        constexpr uint8_t val() const
        {
            return (incr & 0x3F) | ((cr & 0x1) << 6) | ((clr & 0x1) << 7);
        }
    };

    // MM Apr 6 2021: Dest incr width changed from 8 to 12
    // MM Feb 16 2022: Dest increment changed a few months ago to be 10,
    // but I guess I forgot to update this comment
    // CLR, CR, INCR(10 bits)
    struct addr_mod_dest_t
    {
        uint16_t incr   = 0;
        uint8_t clr     = 0;
        uint8_t cr      = 0;
        uint8_t c_to_cr = 0;

        constexpr uint16_t val() const
        {
            return (incr & 0x3FF) | ((cr & 0x1) << 10) | ((clr & 0x1) << 11) | ((c_to_cr & 0x1) << 12); // Updated manually when dest incr changed to 10 bits
        }
    };

    // CLR, INCT (2 bits)
    struct addr_mod_fidelity_t
    {
        uint8_t incr = 0;
        uint8_t clr  = 0;

        constexpr uint16_t val() const
        {
            return (incr & 0x3) | ((clr & 0x1) << 2);
        }
    };

    // CLR, INCT (4 bits)
    /*struct addr_mod_bias_t{
            uint8_t incr = 0;
            uint8_t clr = 0;
            constexpr uint16_t val() const {
                    return (incr & 0xF) | ((clr & 0x1) << 4);
            }
    };*/

    // Set defaults so that we can skip unchanged in initialization list
    addr_mod_src_t srca          = {};
    addr_mod_src_t srcb          = {};
    addr_mod_dest_t dest         = {};
    addr_mod_fidelity_t fidelity = {};

    // SrcA/B register is combination of A and B values
    constexpr uint16_t src_val() const
    {
        return srca.val() | (srcb.val() << 8);
    }

    constexpr uint16_t dest_val() const
    {
        return dest.val() | fidelity.val() << 13;
    }

    // List of addresses of src/dest registers
    constexpr static uint32_t addr_mod_src_reg_addr[] = {
        ADDR_MOD_AB_SEC0_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC1_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC2_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC3_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC4_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC5_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC6_SrcAIncr_ADDR32,
        ADDR_MOD_AB_SEC7_SrcAIncr_ADDR32};

    constexpr static uint32_t addr_mod_dest_reg_addr[] = {
        ADDR_MOD_DST_SEC0_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC1_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC2_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC3_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC4_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC5_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC6_DestIncr_ADDR32,
        ADDR_MOD_DST_SEC7_DestIncr_ADDR32};

#define NUM_MATH_ADDR_MODS 8

    // Program source and dest registers
    __attribute__((always_inline)) inline addr_mod_t const& set(const uint8_t mod_index, uint32_t thread_id = 1) const
    {
        auto cfg                                                                = (volatile uint32_t*)TENSIX_CFG_BASE;
        cfg[addr_mod_src_reg_addr[mod_index] + NUM_MATH_ADDR_MODS * thread_id]  = src_val();
        cfg[addr_mod_dest_reg_addr[mod_index] + NUM_MATH_ADDR_MODS * thread_id] = dest_val();
        return *this;
    }
};

} // namespace ckernel
