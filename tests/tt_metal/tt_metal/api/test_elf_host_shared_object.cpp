// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The binary cache reads an emulated kernel's binary (a shared object built for the host) by its sections only, and
// everything else, a RISC-V device image above all, as a device image. ElfFile::IsHostSharedObject makes that choice
// from the file alone, and ReadSections rejects anything that is not a shared object.

#include <gtest/gtest.h>

#include <elf.h>
#include <link.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "tt_metal/llrt/tt_elffile.hpp"

namespace ll_api {
namespace {

// A shared object loaded into this test: a host shared object by construction.
std::string loaded_shared_object() {
    std::string found;
    dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* data) {
            auto& out = *static_cast<std::string*>(data);
            if (info->dlpi_name != nullptr && std::strstr(info->dlpi_name, ".so") != nullptr &&
                std::filesystem::exists(info->dlpi_name)) {
                out = info->dlpi_name;
                return 1;
            }
            return 0;
        },
        &found);
    return found;
}

class ElfHostSharedObject : public ::testing::Test {
protected:
    void SetUp() override {
        auto pattern = (std::filesystem::temp_directory_path() / "elf_host_so_XXXXXX").string();
        ASSERT_NE(mkdtemp(pattern.data()), nullptr);
        directory_ = pattern;
    }
    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(directory_, ec);
    }

    // An ELF file that is only a header, with the given class, type and machine.
    std::string header_only(const std::string& name, bool is_64, uint16_t type, uint16_t machine) const {
        const auto path = (directory_ / name).string();
        std::ofstream out(path, std::ios::binary);
        if (is_64) {
            Elf64_Ehdr hdr{};
            fill_ident(hdr.e_ident, ELFCLASS64);
            hdr.e_type = type;
            hdr.e_machine = machine;
            hdr.e_version = EV_CURRENT;
            hdr.e_ehsize = sizeof(hdr);
            out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        } else {
            Elf32_Ehdr hdr{};
            fill_ident(hdr.e_ident, ELFCLASS32);
            hdr.e_type = type;
            hdr.e_machine = machine;
            hdr.e_version = EV_CURRENT;
            hdr.e_ehsize = sizeof(hdr);
            out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
        }
        return path;
    }

private:
    static void fill_ident(unsigned char* ident, unsigned char elf_class) {
        ident[EI_MAG0] = ELFMAG0;
        ident[EI_MAG1] = ELFMAG1;
        ident[EI_MAG2] = ELFMAG2;
        ident[EI_MAG3] = ELFMAG3;
        ident[EI_CLASS] = elf_class;
        ident[EI_DATA] = ELFDATA2LSB;
        ident[EI_VERSION] = EV_CURRENT;
    }

    std::filesystem::path directory_;
};

TEST_F(ElfHostSharedObject, CPU_AHostSharedObjectIsReadBySections) {
    const std::string path = loaded_shared_object();
    ASSERT_FALSE(path.empty()) << "no shared object is loaded into the test";
    EXPECT_TRUE(ElfFile::IsHostSharedObject(path)) << path;
    ElfFile elf;
    EXPECT_NO_THROW(elf.ReadSections(path)) << path;
}

TEST_F(ElfHostSharedObject, CPU_ARiscvDeviceImageIsNotAHostSharedObject) {
    EXPECT_FALSE(ElfFile::IsHostSharedObject(header_only("riscv_exec", /*is_64=*/false, ET_EXEC, EM_RISCV)));
    EXPECT_FALSE(ElfFile::IsHostSharedObject(header_only("riscv_dyn", /*is_64=*/false, ET_DYN, EM_RISCV)));
}

TEST_F(ElfHostSharedObject, CPU_AHostExecutableIsNotAHostSharedObject) {
#if defined(__x86_64__)
    constexpr uint16_t host_machine = EM_X86_64;
#elif defined(__aarch64__)
    constexpr uint16_t host_machine = EM_AARCH64;
#endif
    EXPECT_FALSE(ElfFile::IsHostSharedObject(header_only("host_exec", /*is_64=*/true, ET_EXEC, host_machine)));
    EXPECT_FALSE(ElfFile::IsHostSharedObject(header_only("host_rel", /*is_64=*/true, ET_REL, host_machine)));
}

TEST_F(ElfHostSharedObject, CPU_AnUnreadableOrNonElfFileIsNotAHostSharedObject) {
    EXPECT_FALSE(ElfFile::IsHostSharedObject("/nonexistent/kernel.so"));
    EXPECT_FALSE(ElfFile::IsHostSharedObject("/proc/self/cmdline"));
}

TEST_F(ElfHostSharedObject, CPU_ReadSectionsRejectsAnElfThatIsNotASharedObject) {
    const std::string path = header_only("host_exec", /*is_64=*/true, ET_EXEC, EM_X86_64);
    ElfFile elf;
    try {
        elf.ReadSections(path);
        FAIL() << "ReadSections accepted an ET_EXEC ELF";
    } catch (const std::exception& e) {
        EXPECT_NE(std::string(e.what()).find("must be a shared object"), std::string::npos) << e.what();
    }
}

}  // namespace
}  // namespace ll_api
