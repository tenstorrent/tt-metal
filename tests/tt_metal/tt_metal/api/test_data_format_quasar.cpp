// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only: the Quasar unpack-to-dest route and UInt32 canonicalisation derived in jit_build/data_format.

#include <gtest/gtest.h>

#include <array>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-umd/types/arch.hpp>
#include "jit_build/data_format.hpp"

namespace {

using tt::DataFormat;
using tt::tt_metal::UnpackToDestMode;

constexpr auto kDefault = UnpackToDestMode::Default;
constexpr auto kToDestFp32 = UnpackToDestMode::UnpackToDestFp32;

TEST(DataFormatQuasar, Int32BufferEnablesToDestWithoutModeEntry) {
    const std::array<DataFormat, 4> int32_formats{
        DataFormat::Int32, DataFormat::Float16_b, DataFormat::Invalid, DataFormat::Invalid};
    const std::vector<UnpackToDestMode> all_default(int32_formats.size(), kDefault);
    EXPECT_TRUE(tt::quasar_unpack_to_dest_en(int32_formats, all_default));
    // An empty mode vector (legacy path with no modes) must not hide the format-derived route.
    EXPECT_TRUE(tt::quasar_unpack_to_dest_en(int32_formats, {}));

    const std::array<DataFormat, 4> uint32_formats{
        DataFormat::Float16_b, DataFormat::UInt32, DataFormat::Invalid, DataFormat::Invalid};
    EXPECT_TRUE(tt::quasar_unpack_to_dest_en(uint32_formats, all_default));

    const std::array<DataFormat, 4> bf16_formats{
        DataFormat::Float16_b, DataFormat::Float16_b, DataFormat::Invalid, DataFormat::Invalid};
    EXPECT_FALSE(tt::quasar_unpack_to_dest_en(bf16_formats, all_default));
    EXPECT_FALSE(tt::quasar_unpack_to_dest_en(bf16_formats, {}));

    const std::array<DataFormat, 1> fp32_formats{DataFormat::Float32};
    EXPECT_TRUE(tt::quasar_unpack_to_dest_en(fp32_formats, {kToDestFp32}));
    EXPECT_FALSE(tt::quasar_unpack_to_dest_en(fp32_formats, {kDefault}));
    EXPECT_FALSE(tt::quasar_unpack_to_dest_en(fp32_formats, {}));
}

TEST(DataFormatQuasar, CanonicalFormatFoldsUInt32IntoInt32) {
    EXPECT_EQ(tt::quasar_canonical_format(DataFormat::UInt32), DataFormat::Int32);
    EXPECT_EQ(tt::quasar_canonical_format(DataFormat::Int32), DataFormat::Int32);
    EXPECT_EQ(tt::quasar_canonical_format(DataFormat::Float32), DataFormat::Float32);
    EXPECT_EQ(tt::quasar_canonical_format(DataFormat::Float16_b), DataFormat::Float16_b);
    EXPECT_EQ(tt::quasar_canonical_format(DataFormat::RawUInt32), DataFormat::RawUInt32);
    EXPECT_EQ(tt::quasar_canonical_format(DataFormat::Invalid), DataFormat::Invalid);
}

TEST(DataFormatQuasar, UInt32MapsToInt32OnQuasarOnly) {
    const std::array<DataFormat, 1> uint32_formats{DataFormat::UInt32};

    EXPECT_EQ(tt::get_unpack_src_formats(uint32_formats, tt::ARCH::QUASAR)[0], DataFormat::Int32);
    EXPECT_EQ(
        tt::get_unpack_dst_formats(
            uint32_formats,
            DataFormat::Float16_b,
            /*fp32_dest_acc_en=*/true,
            {},
            /*int_fpu_en=*/false,
            tt::ARCH::QUASAR)[0],
        DataFormat::Int32);
    EXPECT_EQ(tt::get_pack_dst_formats(uint32_formats, tt::ARCH::QUASAR)[0], DataFormat::Int32);

    EXPECT_EQ(tt::get_unpack_src_formats(uint32_formats, tt::ARCH::BLACKHOLE)[0], DataFormat::UInt32);
    EXPECT_EQ(
        tt::get_unpack_dst_formats(
            uint32_formats,
            DataFormat::Float16_b,
            /*fp32_dest_acc_en=*/true,
            {},
            /*int_fpu_en=*/false,
            tt::ARCH::BLACKHOLE)[0],
        DataFormat::UInt32);
    EXPECT_EQ(tt::get_pack_dst_formats(uint32_formats, tt::ARCH::BLACKHOLE)[0], DataFormat::UInt32);

    // Int32 and the Raw formats are untouched by the Quasar canonical mapping.
    const std::array<DataFormat, 2> other_formats{DataFormat::Int32, DataFormat::RawUInt32};
    const auto quasar_src = tt::get_unpack_src_formats(other_formats, tt::ARCH::QUASAR);
    EXPECT_EQ(quasar_src[0], DataFormat::Int32);
    EXPECT_EQ(quasar_src[1], DataFormat::Float32);
}

TEST(DataFormatQuasar, Float32ToDestKeepsFloat32) {
    const std::array<DataFormat, 1> fp32_formats{DataFormat::Float32};
    EXPECT_EQ(
        tt::get_unpack_dst_formats(
            fp32_formats,
            DataFormat::Tf32,
            /*fp32_dest_acc_en=*/true,
            {kToDestFp32},
            /*int_fpu_en=*/false,
            tt::ARCH::QUASAR)[0],
        DataFormat::Float32);
    // Without the mode entry Float32 still lowers to the conditional SrcA format.
    EXPECT_EQ(
        tt::get_unpack_dst_formats(
            fp32_formats,
            DataFormat::Tf32,
            /*fp32_dest_acc_en=*/true,
            {kDefault},
            /*int_fpu_en=*/false,
            tt::ARCH::QUASAR)[0],
        DataFormat::Tf32);
}

}  // namespace
