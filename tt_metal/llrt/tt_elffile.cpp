// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_elffile.hpp"

#include "common/assert.hpp"
// C
#include <errno.h>
// OS
#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// Verify some knowledge of, and compatibilty with, RiscV
#ifndef EM_RISCV
#error "Don't know RISCV elf details"
#endif

// Having the same endianness as RISCV makes things easier.
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Host must be little endian"
#endif

#ifdef PT_RISCV_ATTRIBUTES
#warning "PT_RISCV_ATTRIBUTES available, remove workaround"
#else
// Missing from my elf.h
#define PT_RISCV_ATTRIBUTES (PT_LOPROC + 3)
enum {
    Tag_RISCV_arch = 5,
};
#endif

// Sadly the toolchain's usurped some machine numbers.  With any luck
// this will go away at some point.
#define EM_RISCV_GRAYSKULL 242
#define EM_RISCV_WORMHOLE 0x5151
#define EM_RISCV_BLACKHOLE 0x6151

using namespace ll_api;

class ElfFile::Impl {
   private:
    std::span<Elf32_Phdr const> phdrs_;
    std::span<Elf32_Shdr const> shdrs_;
    std::string const &path_;

   private:
    ElfFile &owner_;

   public:
    Impl(ElfFile &owner, std::string const &path) : owner_(owner), path_(path) {}
    ~Impl() = default;

   public:
    void LoadImage();

   private:
    Elf32_Ehdr const &GetHeader() const { return *reinterpret_cast<Elf32_Ehdr const *>(GetContents().data()); }
    std::span<Elf32_Phdr const> GetPhdrs() const { return phdrs_; }
    std::span<Elf32_Shdr const> GetShdrs() const { return shdrs_; }
    Elf32_Shdr const &GetShdr(unsigned ix) const { return shdrs_[ix]; }
    std::vector<Segment> &GetSegments() const { return owner_.segments_; }
    std::span<std::byte> &GetContents() const { return owner_.contents_; }
    std::span<std::byte> GetContents(Elf32_Phdr const &phdr) {
        return GetContents().subspan(phdr.p_offset, phdr.p_filesz);
    }
    std::span<std::byte> GetContents(Elf32_Shdr const &shdr) {
        return GetContents().subspan(shdr.sh_offset, shdr.sh_size);
    }
    char const *GetString(size_t offset, unsigned ix) {
        if (ix >= GetShdrs().size())
        bad:
            return "*bad*";
        auto &shdr = GetShdr(ix);
        if (shdr.sh_type != SHT_STRTAB)
            goto bad;
        auto strings = GetContents(GetShdr(ix));
        if (offset >= strings.size())
            goto bad;
        return ByteOffset<char const>(strings.data(), offset);
    }
    char const *GetName(Elf32_Shdr const &shdr) { return GetString(shdr.sh_name, GetHeader().e_shstrndx); }
    std::span<Elf32_Sym> GetSymbols(Elf32_Shdr const &shdr) {
        auto section = GetContents(shdr);
        return std::span(ByteOffset<Elf32_Sym>(section.data()), section.size() / shdr.sh_entsize);
    }
    char const *GetName(Elf32_Sym const &sym, unsigned lk) { return GetString(sym.st_name, lk); }
    std::span<Elf32_Rela> GetRelocations(Elf32_Shdr const &shdr) {
        auto section = GetContents(shdr);
        return std::span(ByteOffset<Elf32_Rela>(section.data()), section.size() / shdr.sh_entsize);
    }

    static bool IsInSegment(Segment const &segment, Elf32_Shdr const &shdr) {
        // Remember, Segments use word_t sizes
        return shdr.sh_flags & SHF_ALLOC && shdr.sh_addr >= segment.address &&
	    shdr.sh_addr + shdr.sh_size <=
	    segment.address + (segment.contents.size() + segment.bss) * sizeof (word_t);
    }
    bool IsInSegment(unsigned ix, Elf32_Shdr const &shdr) const { return IsInSegment(GetSegments()[ix], shdr); }
    bool IsInText(Elf32_Shdr const &shdr) const { return IsInSegment(GetSegments().front(), shdr); };
    int GetSegmentIx(Elf32_Shdr const &shdr) const {
        for (unsigned ix = GetSegments().size(); ix--;)
            if (IsInSegment(ix, shdr))
                return ix;
        return -1;
    };
    bool IsTextSymbol(Elf32_Sym const &symbol) const {
        return symbol.st_shndx < GetShdrs().size() && IsInText(GetShdr(symbol.st_shndx));
    }
    bool IsDataSymbol(Elf32_Sym const &symbol) const {
        return symbol.st_shndx < GetShdrs().size() && GetSegmentIx(GetShdr(symbol.st_shndx)) > 0;
    }

   private:
    template <typename T = std::byte>
    static T *ByteOffset(std::byte *base, size_t offset = 0) {
        return reinterpret_cast<T *>(base + offset);
    }
    template <typename T = std::byte>
    static T const *ByteOffset(std::byte const *base, size_t offset = 0) {
        return reinterpret_cast<T const *>(base + offset);
    }
};

ElfFile::~ElfFile() {
    ReleaseImpl();
    if (!contents_.empty())
        munmap(contents_.data(), contents_.size());
}

void ElfFile::ReleaseImpl() {
    delete pimpl_;
    pimpl_ = nullptr;
}

void ElfFile::ReadImage(std::string const &path) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat st;
    void *buffer = MAP_FAILED;
    if (fd >= 0 && fstat(fd, &st) >= 0)
        buffer = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    if (fd >= 0)
        // It is acceptable to close a mapped file -- the mapping stays.
        close(fd);
    if (buffer == MAP_FAILED)
        TT_THROW("{}: cannot map elf file into memory: {}", path, strerror(errno));

    contents_ = std::span(reinterpret_cast<std::byte *>(buffer), st.st_size);

    pimpl_ = new Impl(*this, path);
    pimpl_->LoadImage();
}

void ElfFile::Impl::LoadImage() {
    auto &hdr = GetHeader();

    // Make sure it's ELF
    if (hdr.e_ident[EI_MAG0] != 0x7f || hdr.e_ident[EI_MAG1] != 'E' || hdr.e_ident[EI_MAG2] != 'L' ||
        hdr.e_ident[EI_MAG3] != 'F')
        TT_THROW("{}: no ELF magic found", path_);

    // Of the expected address size, endianness and version
    if (hdr.e_ident[EI_CLASS] != ELFCLASS32 || hdr.e_ident[EI_DATA] != ELFDATA2LSB ||
        hdr.e_ident[EI_VERSION] != EV_CURRENT)
        TT_THROW("{}: incompatible address size or endianness", path_);

    if (hdr.e_type != ET_EXEC)
        TT_THROW("{}: not an executable", path_);

    if (hdr.e_machine != EM_RISCV
        // Hopefully these can go way at some point.
        && hdr.e_machine != EM_RISCV_GRAYSKULL && hdr.e_machine != EM_RISCV_WORMHOLE &&
        hdr.e_machine != EM_RISCV_BLACKHOLE)
        TT_THROW("{}: incompatible architecture {}", path_, hdr.e_machine);

    if (!hdr.e_phoff || hdr.e_phoff & (sizeof(address_t) - 1) || hdr.e_phentsize != sizeof(Elf32_Phdr) ||
        (hdr.e_phoff + hdr.e_phnum * sizeof(Elf32_Phdr) > GetContents().size()))
        TT_THROW("{}: PHDRS are missing or malformed", path_);
    phdrs_ = std::span(ByteOffset<Elf32_Phdr const>(GetContents().data(), hdr.e_phoff), hdr.e_phnum);
    if (!hdr.e_shoff || hdr.e_shoff & (sizeof(address_t) - 1) || hdr.e_shentsize != sizeof(Elf32_Shdr) ||
        (hdr.e_shoff + hdr.e_shnum * sizeof(Elf32_Shdr) > GetContents().size()))
        TT_THROW("{}: sections are missing or malformed", path_);
    shdrs_ = std::span(ByteOffset<Elf32_Shdr const>(GetContents().data(), hdr.e_shoff), hdr.e_shnum);
    if (!hdr.e_shstrndx || hdr.e_shstrndx >= GetShdrs().size())
        TT_THROW("{}: string table is missing or malformed", path_);

    // We care about the location of some sections.
    for (auto const &section : GetShdrs())
        if ((section.sh_flags & SHF_ALLOC || section.sh_type == SHT_RELA || section.sh_type == SHT_SYMTAB) &&
                (section.sh_offset | section.sh_addr) & (sizeof(word_t) - 1) ||
            section.sh_offset + section.sh_size > GetContents().size())
            TT_THROW("{}: section {} is misaligned", path_, GetName(section));

    GetSegments().reserve(hdr.e_phnum);
    int textIx = -1;
    for (auto const &phdr : GetPhdrs()) {
        if (phdr.p_type == PT_RISCV_ATTRIBUTES)
            // TODO: verify Arch is ok?
            continue;
        if (phdr.p_type != PT_LOAD)
            continue;
        if (!phdr.p_memsz)
            // Have observed zero-sized segments, ignore them
            continue;

        // Require loadable segments to be nicely aligned
        if ((phdr.p_offset | phdr.p_vaddr) & (sizeof(word_t) - 1))
            TT_THROW("{}: loadable segment {} is misaligned", path_, unsigned(GetSegments().size()));

        auto contents = GetContents(phdr);
        // We require the entry point to be the start of the text segment,
        // so use a simple comparison -- if the entry point is elsewhere
        // we'll complain about lack of text segment.
        if (hdr.e_entry == phdr.p_vaddr)
            textIx = GetSegments().size();

        // This word-size rounding up means the span can occupy some bytes
        // outside the range of the original span, but those bytes will
        // still be inside the span covering the whole file, so that's ok.
        offset_t file_size = (phdr.p_filesz + sizeof(word_t) - 1) / sizeof(word_t);
        offset_t mem_size = (phdr.p_memsz + sizeof(word_t) - 1) / sizeof(word_t);
        GetSegments().emplace_back(
            std::span(reinterpret_cast<word_t const *>(contents.data()), file_size),
            phdr.p_vaddr, mem_size - file_size);
    }
    if (textIx < 0)
        TT_THROW("{}: cannot find text segment", path_);
    if (textIx > 0) {
        auto &segments = GetSegments();
	auto text = std::next(segments.begin(), textIx);
        std::rotate(segments.begin(), text, std::next(text, 1));
    }
}
