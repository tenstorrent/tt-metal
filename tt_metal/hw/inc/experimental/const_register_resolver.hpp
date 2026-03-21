#pragma once

#include <type_traits>

// This file should probably go to tt_llk repository.
//
// Idea here is to have compile time resolution for constant registers, e.g.
// For each init automatically determine which constant registers to use
// E.g., if two inits are called from init and both require constant registers
// Resolve to share vConst and LREG properly
// Example use (very abstracted)
//
// void init_function() {
//    using ExpRegisters = ConstRegisterResolver<3, void>;
//    using RecipRegisters = ConstRegisterResolver<1, ExpRegisters>;
//    _init_exp_<ExpRegisters>();
//    _init_reciprocal_<RecipRegisters>();
//    }
//
// void calculate_function() {
//   using ExpRegisters = ConstRegisterResolver<3, void>;
//   using RecipRegisters = ConstRegisterResolver<1, ExpRegisters>;
//   for (int d = 0; d < ITERATIONS; d++) {
//      sfpi::vFloat crs_0 = ExpRegisters::template const_register<0, true>();
//      sfpi::vFloat crs_1 = ExpRegisters::template const_register<1, true>();
//      sfpi::vFloat crs_2 = ExpRegisters::template const_register<2, true>();
//      sfpi::vFloat crs_3 = RecipRegisters::template const_register<0, true>();
//      sfpi::vFloat val = sfpi::dst_reg[0];
//      sfpi::vFloat result = _sfpu_sigmoid_<ExpRegisters>(val);
//      result = _sfpu_reciprocal_<RecipRegisters>(result);
//      sfpi::dst_reg[0] = result;
//      sfpi::dst_reg++;
//      // Store back constant registers to make sure L_REGS are preserved
//      ExpRegisters::template store_const_register<0>(crs_0);
//      ExpRegisters::template store_const_register<1>(crs_1);
//      ExpRegisters::template store_const_register<2>(crs_2);
//      RecipRegisters::template store_const_register<0>(crs_3);
//   }
//

template <int REG_NO, bool float_or_int, typename CRC>
inline constexpr auto const_register() {
    static_assert(REG_NO >= 0, "Register number must be non-negative.");
    constexpr auto real_reg = REG_NO + CRC::first_reg;
    static_assert(real_reg < CRC::last_reg, "Trying to access unavailable register.");
    if constexpr (real_reg == 0) {
        if constexpr (float_or_int) {
            return sfpi::vConstFloatPrgm0;
        } else {
            return sfpi::vConstIntPrgm0;
        }
    } else if constexpr (real_reg == 1) {
        if constexpr (float_or_int) {
            return sfpi::vConstFloatPrgm1;
        } else {
            return sfpi::vConstIntPrgm1;
        }
    } else if constexpr (real_reg == 2) {
        if constexpr (float_or_int) {
            return sfpi::vConstFloatPrgm2;
        } else {
            return sfpi::vConstIntPrgm2;
        }
    } else if constexpr (real_reg < 12) {
        return sfpi::l_reg[static_cast<const enum sfpi::LRegs>(real_reg - 3)];
    } else {
        static_assert(false, "No more registers available.");
    }
}

template <int REG_NO, typename CRC, typename V>
inline constexpr void store_const_register(V& val) {
    static_assert(REG_NO >= 0, "Register number must be non-negative.");
    constexpr auto real_reg = REG_NO + CRC::first_reg;
    static_assert(real_reg < CRC::last_reg, "Trying to access unavailable register.");
    if constexpr (real_reg <= 2) {
        // No need to store
    } else if constexpr (real_reg < 12) {
        sfpi::l_reg[static_cast<const enum sfpi::LRegs>(real_reg - 3)] = val;
    } else {
        static_assert(false, "No more registers available.");
    }
}

template <int REG_COUNT, typename CRR>
struct ConstRegisterResolver : public CRR {
    static const int first_reg = CRR::last_reg;
    static const int last_reg = CRR::last_reg + REG_COUNT;

    template <int REG_NO, bool float_or_int>
    inline static auto const_register() {
        return ::const_register<REG_NO, float_or_int, ConstRegisterResolver<REG_COUNT, CRR>>();
    }

    template <int REG_NO, typename V>
    inline static void store_const_register(V& val) {
        ::store_const_register<REG_NO, ConstRegisterResolver<REG_COUNT, CRR>, V>(val);
    }
};

template <int REG_COUNT>
struct ConstRegisterResolver<REG_COUNT, void> {
    static const int first_reg = 0;
    static const int last_reg = REG_COUNT;

    template <int REG_NO, bool float_or_int>
    inline static auto const_register() {
        return ::const_register<REG_NO, float_or_int, ConstRegisterResolver<REG_COUNT, void>>();
    }

    template <int REG_NO, typename V>
    inline static void store_const_register(V& val) {
        ::store_const_register<REG_NO, ConstRegisterResolver<REG_COUNT, void>, V>(val);
    }
};
