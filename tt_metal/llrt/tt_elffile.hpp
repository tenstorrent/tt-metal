// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// C++
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

// An ELF executable loader
// This is a replacement for tt_hexfile stuff.

namespace ll_api {

class ElfFile {
   public:
    // ELF32
    using address_t = std::uint32_t;  // Address in memory
    using offset_t = std::uint32_t;   // Offset within region
    using word_t = std::uint32_t;     // Contents

    struct Segment {
        std::span<word_t const> contents;  // Non-owning span
        address_t address = 0;             // word address or 0 for XIP
        offset_t bss = 0;                  // words of BSS

       public:
        constexpr Segment(std::span<word_t const> contents, address_t addr, offset_t bss) :
            contents(contents), address(addr), bss(bss) {}
    };

   public:
    ElfFile(std::string const &path);
    ~ElfFile();

    // Uncopyable -- because of the owning buffer
    ElfFile(ElfFile const &) = delete;
    ElfFile operator=(ElfFile const &) = delete;

    // Move constructable -- take ownership
    ElfFile(ElfFile &&s) : contents_(std::move(s.contents_)), segments_(std::move(s.segments_)) {
        s.contents_ = std::span<std::byte>();
    }
    ElfFile &operator=(ElfFile &&s) {
        std::swap(contents_, s.contents_);
        segments_ = std::move(s.segments_);
        return *this;
    }

   public:
    std::vector<Segment> const &GetSegments() const { return segments_; }

   private:
    class Impl;
    std::span<std::byte> contents_;  // Owning buffer
    std::vector<Segment> segments_;
};

}  // namespace ll_api
