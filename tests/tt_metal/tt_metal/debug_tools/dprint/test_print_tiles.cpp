// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "common/bfloat8.hpp"
#include "common/bfloat4.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A simple test for checking DPRINTs from all harts.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

static constexpr uint32_t elements_in_tile = 32 * 32;

static std::vector<uint32_t> GenerateInputTile(tt::DataFormat data_format) {
    uint32_t tile_size_bytes = tile_size(data_format);
    std::vector<uint32_t> u32_vec;
    if (data_format == tt::DataFormat::Float32) {
        u32_vec.resize(elements_in_tile);
        for (int i = 0; i < u32_vec.size(); i++) {
            float val = -12.3345 + static_cast<float>(i);  // Rebias to force some negative #s to be printed
            u32_vec.at(i) = *reinterpret_cast<uint32_t*>(&val);
        }
    } else if (data_format == tt::DataFormat::Float16_b) {
        std::vector<bfloat16> fp16b_vec(elements_in_tile);
        for (int i = 0; i < fp16b_vec.size(); i++) {
            uint16_t val = 0x3dfb + i;  // Start at some known value (~0.1226) and increment for new numbers
            fp16b_vec[i] = bfloat16(val);
        }
        u32_vec = pack_bfloat16_vec_into_uint32_vec(fp16b_vec);
    } else if (data_format == tt::DataFormat::Bfp8_b) {
        std::vector<float> float_vec(elements_in_tile);
        for (int i = 0; i < float_vec.size(); i++) {
            float_vec[i] = 0.012345 * i * (i % 32 == 0 ? -1 : 1);  // Small increments and force negatives for testing
        }
        u32_vec = pack_fp32_vec_as_bfp8_tiles(float_vec, true, false);
    } else if (data_format == tt::DataFormat::Bfp4_b) {
        std::vector<float> float_vec(elements_in_tile);
        for (int i = 0; i < float_vec.size(); i++) {
            float_vec[i] = 0.012345 * i * (i % 16 == 0 ? -1 : 1);  // Small increments and force negatives for testing
        }
        u32_vec = pack_fp32_vec_as_bfp4_tiles(float_vec, true, false);
    } else if (data_format == tt::DataFormat::Int8) {
        std::vector<int8_t> int8_vec(elements_in_tile);
        for (int i = 0; i < int8_vec.size(); i++) {
            int8_vec[i] = ((i / 2) % 256) - 128;  // Force prints to be different (/2), within the int8 range (%256),
                                                  // and include negatives (-128) for testing purposes.
        }
        uint32_t datums_per_32 = sizeof(uint32_t) / tt::datum_size(data_format);
        u32_vec.resize(elements_in_tile / datums_per_32);
        std::memcpy(u32_vec.data(), int8_vec.data(), elements_in_tile * sizeof(int8_t));
    } else if (data_format == tt::DataFormat::UInt8) {
        std::vector<uint8_t> uint8_vec(elements_in_tile);
        for (int i = 0; i < uint8_vec.size(); i++) {
            uint8_vec[i] = ((i / 2) % 256);  // Same as int8, just no negatives
        }
        uint32_t datums_per_32 = sizeof(uint32_t) / tt::datum_size(data_format);
        u32_vec.resize(elements_in_tile / datums_per_32);
        std::memcpy(u32_vec.data(), uint8_vec.data(), elements_in_tile * sizeof(uint8_t));
    } else if (data_format == tt::DataFormat::UInt16) {
        std::vector<uint16_t> uint16_vec(elements_in_tile);
        for (int i = 0; i < uint16_vec.size(); i++) {
            uint16_vec[i] = (i % 0x10000);  // Force to within uint16 range
        }
        uint32_t datums_per_32 = sizeof(uint32_t) / tt::datum_size(data_format);
        u32_vec.resize(elements_in_tile / datums_per_32);
        std::memcpy(u32_vec.data(), uint16_vec.data(), elements_in_tile * sizeof(uint16_t));
    } else if (data_format == tt::DataFormat::Int32) {
        u32_vec.resize(elements_in_tile);
        for (int i = 0; i < u32_vec.size(); i++) {
            u32_vec[i] = (i % 2) ? i : i * -1;  // Make every other number negative for printing purposes
        }
    } else if (data_format == tt::DataFormat::UInt32) {
        u32_vec.resize(elements_in_tile);
        for (int i = 0; i < u32_vec.size(); i++) {
            u32_vec[i] = i;
        }
    }
    return u32_vec;
}

static string GenerateExpectedData(tt::DataFormat data_format, std::vector<uint32_t> &input_tile) {
    string data = "";
    if (data_format == tt::DataFormat::Float32) {
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{:.6} {:.6} {:.6} {:.6}",
                *reinterpret_cast<float *>(&input_tile[col * 32 + 0]),
                *reinterpret_cast<float *>(&input_tile[col * 32 + 8]),
                *reinterpret_cast<float *>(&input_tile[col * 32 + 16]),
                *reinterpret_cast<float *>(&input_tile[col * 32 + 24]));
        }
    } else if (data_format == tt::DataFormat::Float16_b) {
        std::vector<bfloat16> fp16b_vec = unpack_uint32_vec_into_bfloat16_vec(input_tile);
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{:.6} {:.6} {:.6} {:.6}",
                fp16b_vec[col * 32 + 0].to_float(),
                fp16b_vec[col * 32 + 8].to_float(),
                fp16b_vec[col * 32 + 16].to_float(),
                fp16b_vec[col * 32 + 24].to_float());
        }
    } else if (data_format == tt::DataFormat::Bfp8_b) {
        std::vector<float> float_vec = unpack_bfp8_tiles_into_float_vec(input_tile, true, false);
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{:.6} {:.6} {:.6} {:.6}",
                *reinterpret_cast<float *>(&float_vec[col * 32 + 0]),
                *reinterpret_cast<float *>(&float_vec[col * 32 + 8]),
                *reinterpret_cast<float *>(&float_vec[col * 32 + 16]),
                *reinterpret_cast<float *>(&float_vec[col * 32 + 24]));
        }
    } else if (data_format == tt::DataFormat::Bfp4_b) {
        std::vector<float> float_vec = unpack_bfp4_tiles_into_float_vec(input_tile, true, false);
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{:.6} {:.6} {:.6} {:.6}",
                *reinterpret_cast<float *>(&float_vec[col * 32 + 0]),
                *reinterpret_cast<float *>(&float_vec[col * 32 + 8]),
                *reinterpret_cast<float *>(&float_vec[col * 32 + 16]),
                *reinterpret_cast<float *>(&float_vec[col * 32 + 24]));
        }
    } else if (data_format == tt::DataFormat::Int8) {
        int8_t* int8_ptr = reinterpret_cast<int8_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                int8_ptr[col * 32 + 0],
                int8_ptr[col * 32 + 8],
                int8_ptr[col * 32 + 16],
                int8_ptr[col * 32 + 24]);
        }
    } else if (data_format == tt::DataFormat::UInt8) {
        uint8_t* uint8_ptr = reinterpret_cast<uint8_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                uint8_ptr[col * 32 + 0],
                uint8_ptr[col * 32 + 8],
                uint8_ptr[col * 32 + 16],
                uint8_ptr[col * 32 + 24]);
        }
    } else if (data_format == tt::DataFormat::UInt16) {
        uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                uint16_ptr[col * 32 + 0],
                uint16_ptr[col * 32 + 8],
                uint16_ptr[col * 32 + 16],
                uint16_ptr[col * 32 + 24]);
        }
    } else if (data_format == tt::DataFormat::Int32) {
        int32_t* int32_ptr = reinterpret_cast<int32_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                int32_ptr[col * 32 + 0],
                int32_ptr[col * 32 + 8],
                int32_ptr[col * 32 + 16],
                int32_ptr[col * 32 + 24]);
        }
    } else if (data_format == tt::DataFormat::UInt32) {
        uint32_t* uint32_ptr = reinterpret_cast<uint32_t*>(input_tile.data());
        for (uint32_t col = 0; col < 32; col += 8) {
            data += fmt::format(
                "\n{} {} {} {}",
                uint32_ptr[col * 32 + 0],
                uint32_ptr[col * 32 + 8],
                uint32_ptr[col * 32 + 16],
                uint32_ptr[col * 32 + 24]);
        }
    }
    return data;
}

static string GenerateGoldenOutput(tt::DataFormat data_format, std::vector<uint32_t> &input_tile) {
    string data = GenerateExpectedData(data_format, input_tile);
    string expected = fmt::format("Print tile from Data0:{}", data);
    expected += fmt::format("\nPrint tile from Unpack:{}", data);
    expected += fmt::format("\nPrint tile from Math:\nWarning: MATH core does not support TileSlice printing, omitting print...");
    expected += fmt::format("\nPrint tile from Pack:{}", data);
    expected += fmt::format("\nPrint tile from Data1:{}", data);
    return expected;
}

static void RunTest(DPrintFixture* fixture, Device* device, tt::DataFormat data_format) {
    // Set up program + CQ, run on just one core
    constexpr CoreCoord core = {0, 0};
    Program program = Program();

    // Create an input CB with the right data format
    uint32_t tile_size = detail::TileSize(data_format);
    CircularBufferConfig cb_src0_config = CircularBufferConfig(tile_size, {{CBIndex::c_0, data_format}})
                                              .set_page_size(CBIndex::c_0, tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // Dram buffer to send data to, device will read it out of here to print
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device, .size = tile_size, .page_size = tile_size, .buffer_type = tt_metal::BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();

    // Create kernels on device
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program,
        llrt::RunTimeOptions::get_instance().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/print_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle ncrisc_print_kernel_id = CreateKernel(
        program,
        llrt::RunTimeOptions::get_instance().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/print_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    KernelHandle trisc_print_kernel_id = CreateKernel(
        program,
        llrt::RunTimeOptions::get_instance().get_root_dir() +
            "tests/tt_metal/tt_metal/test_kernels/misc/print_tile.cpp",
        core,
        ComputeConfig{});

    // BRISC kernel needs dram info via rtargs
    tt_metal::SetRuntimeArgs(program, brisc_print_kernel_id, core, {dram_buffer_src_addr, (std::uint32_t)0});

    // Create input tile
    std::vector<uint32_t> u32_vec = GenerateInputTile(data_format);
    /*for (int idx = 0; idx < u32_vec.size(); idx+= 16) {
        string tmp = fmt::format("data[{:#03}:{:#03}]:", idx - 1, idx - 16);
        for (int i = 0; i < 16; i++)
            tmp += fmt::format(" {:#08x}", u32_vec[idx + 15 - i]);
        log_info("{}", tmp);
    }*/

    // Send input tile to dram
    if (fixture->IsSlowDispatch()) {
        tt_metal::detail::WriteToBuffer(src_dram_buffer, u32_vec);
    } else {
        CommandQueue& cq = device->command_queue();
        EnqueueWriteBuffer(cq, src_dram_buffer, u32_vec, true);
    }

    // Run the program
    fixture->RunProgram(device, program);

    // Check against expected prints
    string expected = GenerateGoldenOutput(data_format, u32_vec);
    // log_info("Expected output:\n{}", expected);
    EXPECT_TRUE(
        FilesMatchesString(
            DPrintFixture::dprint_file_name,
            expected
        )
    );
}

TEST_F(DPrintFixture, TestPrintTilesFloat32) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::Float32); }, device);
    }
}
TEST_F(DPrintFixture, TestPrintTilesFloat16_b) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::Float16_b); }, device);
    }
}
TEST_F(DPrintFixture, TestPrintTilesBfp4_b) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::Bfp4_b); }, device);
    }
}
TEST_F(DPrintFixture, TestPrintTilesBfp8_b) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::Bfp8_b); }, device);
    }
}
TEST_F(DPrintFixture, TestPrintTilesInt8) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::Int8); }, device);
    }
}
TEST_F(DPrintFixture, TestPrintTilesInt32) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::Int32); }, device);
    }
}
TEST_F(DPrintFixture, TestPrintTilesUInt8) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::UInt8); }, device);
    }
}
TEST_F(DPrintFixture, TestPrintTilesUInt16) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::UInt16); }, device);
    }
}
TEST_F(DPrintFixture, TestPrintTilesUInt32) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [&](DPrintFixture* fixture, Device* device) { RunTest(fixture, device, tt::DataFormat::UInt32); }, device);
    }
}
