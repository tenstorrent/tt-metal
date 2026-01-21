#pragma once
#ifndef __TT_METAL_FABRIC_UARCH_LSU_HPP__
#define __TT_METAL_FABRIC_UARCH_LSU_HPP__

/*-------------------------------------------------------
   ERISC Register File
-------------------------------------------------------
struct EriscRegisterFile : public RegisterFile {
    using NumGeneralPurposeRegisters = std::integral_constant<uint32_t, 32U>;
    using NumFloatingPointRegisters = std::integral_constant<uint32_t, 15U>;

    template<uint32_t index>
    using X {
        static_assert(index < NumGeneralPurposeRegisters::value, "GPR index out of bounds");
        using is_general_purpose = std::true_type;
        using is_floating_point = std::false_type;
        using index = std::integral_constant<uint32_t, index>;
    };

    template<uint32_t index>
    using FP {
        static_assert(index < NumFloatingPointRegisters::value, "FPR index out of bounds");
        using is_general_purpose = std::false_type;
        using is_floating_point = std::true_type;

        using index = std::integral_constant<uint32_t, index>;
    };

    using X0 = X<0U>;
    using X1 = X<1U>;
    using X2 = X<2U>;
    using X3 = X<3U>;
    using X4 = X<4U>;
    using X5 = X<5U>;
    using X6 = X<6U>;
    using X7 = X<7U>;
    using X8 = X<8U>;
    using X9 = X<9U>;
    using X10 = X<10U>;
    using X11 = X<11U>;
    using X12 = X<12U>;
    using X13 = X<13U>;
    using X14 = X<14U>;
    using X15 = X<15U>;
    using X16 = X<16U>;
    using X17 = X<17U>;
    using X18 = X<18U>;
    using X19 = X<19U>;
    using X20 = X<20U>;
    using X21 = X<21U>;
    using X22 = X<22U>;
    using X23 = X<23U>;
    using X24 = X<24U>;
    using X25 = X<25U>;
    using X26 = X<26U>;
    using X27 = X<27U>;
    using X28 = X<28U>;
    using X29 = X<29U>;
    using X30 = X<30U>;
    using X31 = X<31U>;

    using FP0 = FP<0U>;
    using FP1 = FP<1U>;
    using FP2 = FP<2U>;
    using FP3 = FP<3U>;
    using FP4 = FP<4U>;
    using FP5 = FP<5U>;
    using FP6 = FP<6U>;
    using FP7 = FP<7U>;
    using FP8 = FP<8U>;
    using FP9 = FP<9U>;
    using FP10 = FP<10U>;
    using FP11 = FP<11U>;
    using FP12 = FP<12U>;
    using FP13 = FP<13U>;
    using FP14 = FP<14U>;

    template<typename T>
    using is_general_purpose_register = std::conditional_t<std::is_base_of<EriscRegisterFile, T>::value, std::true_type, std::false_type>;
    template<typename T>
    using is_floating_point_register = std::conditional_t<std::is_base_of<EriscRegisterFile, T>::value, std::true_type, std::false_type>;

    static constexpr uint32_t x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30, x31;
    static constexpr float fp0, fp1, fp2, fp3, fp4, fp5, fp6, fp7, fp8, fp9, fp10, fp11, fp12, fp13, fp14;

    template<T index>
    static constexpr FORCED_INLINE uint32_t& get() {
        static_assert(index::is_general_purpose::value || index::is_floating_point::value, "Index must be GPR or FPR");
        if constexpr(index::is_general_purpose::value) {
            return get_gpr<index::index::value>();
        }
        else if constexpr(index::is_floating_point::value) {
            return get_fpr<index::index::value>();
        }
        
        return InvalidAddress::value; // Should never reach here
    }
        
    template<uint32_t index>
    static constexpr FORCED_INLINE uint32_t& get_gpr() {
        static_assert(index < NumGeneralPurposeRegisters::value, "GPR index out of bounds");

        if constexpr(index == 0U) { return x0; }
        else if constexpr(index == 1U) { return x1; }
        else if constexpr(index == 2U) { return x2; }
        else if constexpr(index == 3U) { return x3; }
        else if constexpr(index == 4U) { return x4; }
        else if constexpr(index == 5U) { return x5; }
        else if constexpr(index == 6U) { return x6; }
        else if constexpr(index == 7U) { return x7; }
        else if constexpr(index == 8U) { return x8; }
        else if constexpr(index == 9U) { return x9; }
        else if constexpr(index == 10U) { return x10; }
        else if constexpr(index == 11U) { return x11; }
        else if constexpr(index == 12U) { return x12; }
        else if constexpr(index == 13U) { return x13; }
        else if constexpr(index == 14U) { return x14; }
        else if constexpr(index == 15U) { return x15; }
        else if constexpr(index == 16U) { return x16; }
        else if constexpr(index == 17U) { return x17; }
        else if constexpr(index == 18U) { return x18; }
        else if constexpr(index == 19U) { return x19; }
        else if constexpr(index == 20U) { return x20; }
        else if constexpr(index == 21U) { return x21; }
        else if constexpr(index == 22U) { return x22; }
        else if constexpr(index == 23U) { return x23; }
        else if constexpr(index == 24U) { return x24; }
        else if constexpr(index == 25U) { return x25; }
        else if constexpr(index == 26U) { return x26; }
        else if constexpr(index == 27U) { return x27; }
        else if constexpr(index == 28U) { return x28; }
        else if constexpr(index == 29U) { return x29; }
        else if constexpr(index == 30U) { return x30; }
        else { return x31; } //index == 31U
    }

    template<uint32_t index>
    static constexpr FORCED_INLINE float& get_fpr() {
        static_assert(index < NumFloatingPointRegisters::value, "FPR index out of bounds");
        if constexpr(index == 0U) { return fp0; }
        else if constexpr(index == 1U) { return fp1; }
        else if constexpr(index == 2U) { return fp2; }
        else if constexpr(index == 3U) { return fp3; }
        else if constexpr(index == 4U) { return fp4; }
        else if constexpr(index == 5U) { return fp5; }
        else if constexpr(index == 6U) { return fp6; }
        else if constexpr(index == 7U) { return fp7; }
        else if constexpr(index == 8U) { return fp8; }
        else if constexpr(index == 9U) { return fp9; }
        else if constexpr(index == 10U) { return fp10; }
        else if constexpr(index == 11U) { return fp11; }
        else if constexpr(index == 12U) { return fp12; }
        else if constexpr(index == 13U) { return fp13; }
        else { return fp14; } //index == 14U
    }
};

struct LoadStoreUnit : public EriscRegisterFile {
    using InvalidAddress = std::integral_constant<uint32_t, -1U>;

    template<typename T>
    static FORCED_INLINE uint32_t load() {
        if constexpr(OverlayRegisterFile::IsOverlayRegister<T>::value) {
            return OverlayRegisterFile::load<T>();
        }
        else if constexpr(std::is_base_of<EriscRegisterFile, T>::value) {
            return EriscRegisterFile::get<T>();
        }

        return InvalidAddress::value; // Invalid address
    }

    static FORCED_INLINE bool is_valid_address(uint32_t const address) {
        return address != InvalidAddress::value;
    }

    template<typename T>
    constexpr FORCED_INLINE bool is_valid_address() {
        return std::conditional_t<T::ADDRESS != InvalidAddress::value, std::true_type, std::false_type>::value;
    }
};
*/

#endif