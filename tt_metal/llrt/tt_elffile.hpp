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
        address_t address = 0;             // byte address or 0 for XIP
        offset_t bss = 0;                  // words of BSS

       public:
        constexpr Segment(std::span<word_t const> contents, address_t addr, offset_t bss) :
            contents(contents), address(addr), bss(bss) {}
    };

   public:
    ElfFile() = default;
    ~ElfFile();

    // Uncopyable -- because of the owning buffer & pimpl object.
    ElfFile(ElfFile const &) = delete;
    ElfFile operator=(ElfFile const &) = delete;

    // Move constructable & assignable -- take ownership
    ElfFile(ElfFile &&s) : pimpl_(s.pimpl_),
	contents_(std::move(s.contents_)), segments_(std::move(s.segments_)) {
        s.contents_ = std::span<std::byte>();
	s.pimpl_ = nullptr;
    }
    ElfFile &operator=(ElfFile &&s) {
        std::swap(contents_, s.contents_);
        segments_ = std::move(s.segments_);
	std::swap(pimpl_, s.pimpl_);
        return *this;
    }

   public:
    std::vector<Segment> const &GetSegments() const { return segments_; }

   public:
    // Release the implementation data, leaving the segments and
    // contents. Use this, after processing, if the elf object is long-lived.
    void ReleaseImpl();

    // Read an elf file, populate segments vector.
    // Path must remain live throughout processing.
    void ReadImage(std::string const &path);

    // Write the (now-processed) elf file.
    void WriteImage(std::string const &path);

    // Weaken data symbols, remove all others. Keep STRONG_NAMES
    // strong (can be non-data symbols).  Names can be exact or simple
    // globs ending in '*'.
    void WeakenDataSymbols(std::span<std::string_view const> strong_names);

   private:
    class Impl;
    // We can't use unique_ptr here, because the above move semantics
    // would require Impl be complete at this point, which is what
    // we're trying to avoid.
    Impl *pimpl_ = nullptr;

    std::span<std::byte> contents_;  // Owning buffer

    // The first segment is the text segment, regardless of VMA ordering.
    std::vector<Segment> segments_;
};

}  // namespace ll_api
