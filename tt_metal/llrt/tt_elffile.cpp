// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_elffile.hpp"

#include <algorithm>
#include <array>

#include "common/assert.hpp"
// C++
#include <map>
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

// We have to translate these two instructions
static constexpr uint32_t insn_opc_auipc = 0x00000017;
static constexpr uint32_t insn_opc_lui = 0x00000037;
static constexpr uint32_t insn_mask_u = 0x0000007f;
static constexpr uint32_t mask_hi20 = 0x00000fff;
static constexpr unsigned mask_hi20_shift = 12;
static constexpr uint32_t mask_lo12_i = 0x000fffff;
static constexpr unsigned mask_lo12_i_shift = 20;
static constexpr uint32_t mask_lo12_s = 0x01fff07f;
static constexpr unsigned mask_lo12_s_split = 5;
static constexpr unsigned mask_lo12_s_shift_1 = 7;
static constexpr unsigned mask_lo12_s_shift_2 = 25;

using namespace ll_api;

class ElfFile::Impl {
   private:
    std::span<Elf32_Phdr> phdrs_;
    std::span<Elf32_Shdr> shdrs_;
    std::string const &path_;
    ElfFile &owner_;

   private:
    class Weakener;

   public:
    Impl(ElfFile &owner, std::string const &path) : owner_(owner), path_(path) {}
    ~Impl() = default;

   public:
    void LoadImage();
    void WeakenDataSymbols(std::span<std::string_view const> strong_names);
    void XIPify();

   private:
    [[nodiscard]] auto GetHeader() const -> Elf32_Ehdr const & { return *ByteOffset<Elf32_Ehdr>(GetContents().data()); }
    [[nodiscard]] auto GetPhdrs() const -> std::span<Elf32_Phdr const> { return phdrs_; }
    [[nodiscard]] auto GetShdrs() const -> std::span<Elf32_Shdr const> { return shdrs_; }
    [[nodiscard]] auto GetShdr(unsigned ix) const -> Elf32_Shdr const & { return shdrs_[ix]; }
    [[nodiscard]] auto GetSegments() const -> std::vector<Segment> & { return owner_.segments_; }
    [[nodiscard]] auto GetContents() const -> std::span<std::byte> & { return owner_.contents_; }
    [[nodiscard]] auto GetContents(Elf32_Phdr const &phdr) const -> std::span<std::byte> {
        return GetContents().subspan(phdr.p_offset, phdr.p_filesz);
    }
    [[nodiscard]] auto GetContents(Elf32_Shdr const &shdr) const -> std::span<std::byte> {
        return GetContents().subspan(shdr.sh_offset, shdr.sh_size);
    }
    [[nodiscard]] auto GetString(size_t offset, Elf32_Shdr const &shdr) const -> char const * {
        return ByteOffset<char const>(GetContents(shdr).data(), offset);
    }
    [[nodiscard]] auto GetName(Elf32_Shdr const &shdr) const -> char const * {
        return GetString(shdr.sh_name, GetShdr(GetHeader().e_shstrndx));
    }
    [[nodiscard]] auto GetSymbols(Elf32_Shdr const &shdr) const -> std::span<Elf32_Sym> {
        auto section = GetContents(shdr);
        return std::span(ByteOffset<Elf32_Sym>(section.data()), section.size() / shdr.sh_entsize);
    }
    [[nodiscard]] auto GetName(Elf32_Sym const &sym, unsigned link) const -> char const * {
        return GetString(sym.st_name, GetShdr(link));
    }
    [[nodiscard]] auto GetRelocations(Elf32_Shdr const &shdr) const -> std::span<Elf32_Rela> {
        auto section = GetContents(shdr);
        return std::span(ByteOffset<Elf32_Rela>(section.data()), section.size() / shdr.sh_entsize);
    }

    [[nodiscard]] static bool IsInSegment(Segment const &segment, Elf32_Shdr const &shdr) {
        // Remember, Segments use word_t sizes
        return shdr.sh_flags & SHF_ALLOC && shdr.sh_addr >= segment.address &&
	    shdr.sh_addr + shdr.sh_size <=
	    segment.address + (segment.contents.size() + segment.bss) * sizeof (word_t);
    }
    [[nodiscard]] bool IsInSegment(unsigned _ix, Elf32_Shdr const &shdr) const {
        return IsInSegment(GetSegments()[_ix], shdr);
    }
    [[nodiscard]] bool IsInText(Elf32_Shdr const &shdr) const { return IsInSegment(GetSegments().front(), shdr); };
    [[nodiscard]] int GetSegmentIx(Elf32_Shdr const &shdr) const {
        for (unsigned ix = GetSegments().size(); ix--;)
            if (IsInSegment(ix, shdr))
                return ix;
        return -1;
    };
    [[nodiscard]] bool IsTextSymbol(Elf32_Sym const &symbol) const {
        return symbol.st_shndx < GetShdrs().size() && IsInText(GetShdr(symbol.st_shndx));
    }
    [[nodiscard]] bool IsDataSymbol(Elf32_Sym const &symbol) const {
        return symbol.st_shndx < GetShdrs().size() && GetSegmentIx(GetShdr(symbol.st_shndx)) > 0;
    }

   private:
    template <typename T = std::byte>
    [[nodiscard]] static T *ByteOffset(std::byte *base, size_t offset = 0) {
        return reinterpret_cast<T *>(base + offset);
    }
    template <typename T = std::byte>
    [[nodiscard]] static T const *ByteOffset(std::byte const *base, size_t offset = 0) {
        return reinterpret_cast<T const *>(base + offset);
    }

    uint32_t Read32(Elf32_Shdr const &shdr, address_t addr) {
        return *ByteOffset<uint32_t>(GetContents(shdr).data(), addr - shdr.sh_addr);
    }
    void Write32(Elf32_Shdr const &shdr, address_t addr, uint32_t value) {
        *ByteOffset<uint32_t>(GetContents(shdr).data(), addr - shdr.sh_addr) = value;
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

void ElfFile::WriteImage(std::string const &path) {
    // open is an os-defined varadic function, it the API to use.
    int file_descriptor = open(
        path.c_str(),
        O_WRONLY | O_CLOEXEC | O_CREAT | O_TRUNC,
        S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    bool failed = file_descriptor < 0;
    if (!failed) {
        failed = write(file_descriptor, contents_.data(), contents_.size()) != ssize_t(contents_.size());
        close(file_descriptor);
    }
    if (failed)
        TT_THROW("{}: cannot map elf file into memory: {}", path, strerror(errno));
}

void ElfFile::WeakenDataSymbols(std::span<std::string_view const> strong) { pimpl_->WeakenDataSymbols(strong); }

void ElfFile::MakeExecuteInPlace() { pimpl_->XIPify(); }

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
    phdrs_ = std::span(ByteOffset<Elf32_Phdr>(GetContents().data(), hdr.e_phoff), hdr.e_phnum);
    if (!hdr.e_shoff || hdr.e_shoff & (sizeof(address_t) - 1) || hdr.e_shentsize != sizeof(Elf32_Shdr) ||
        (hdr.e_shoff + hdr.e_shnum * sizeof(Elf32_Shdr) > GetContents().size()))
        TT_THROW("{}: sections are missing or malformed", path_);
    shdrs_ = std::span(ByteOffset<Elf32_Shdr>(GetContents().data(), hdr.e_shoff), hdr.e_shnum);
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

class ElfFile::Impl::Weakener {
    enum { LOCAL, GLOBAL, HWM };

    Elf32_Shdr const &shdr_;
    std::span<Elf32_Sym> syms_in_;
    std::vector<unsigned> remap_;
    std::vector<Elf32_Sym> syms_out_[HWM];

   public:
    Weakener(Elf32_Shdr const &shdr, std::span<Elf32_Sym> symbols) :
        shdr_(shdr), syms_in_(symbols.subspan(shdr.sh_info)) {
        unsigned reserve = syms_in_.size();
        remap_.reserve(reserve);
        std::ranges::for_each(syms_out_, [=](std::vector<Elf32_Sym> &syms) { syms.reserve(reserve); });
    }

    void WeakenOrLocalizeSymbols(Impl &impl, std::span<std::string_view const> strong) {
        auto name_matches = [](std::string_view name, std::span<std::string_view const> list) {
            return std::ranges::any_of(list, [&](std::string_view pattern) {
                return pattern.back() == '*' ? name.starts_with(pattern.substr(0, pattern.size() - 1))
                                             : name == pattern;
            });
        };

        // Weaken or hide globals
        for (auto &sym : syms_in_) {
            auto kind = GLOBAL;
            if ((ELF32_ST_BIND(sym.st_info) == STB_GLOBAL || ELF32_ST_BIND(sym.st_info) == STB_WEAK) &&
                !name_matches(impl.GetName(sym, shdr_.sh_link), strong)) {
                unsigned bind = impl.IsDataSymbol(sym) ? STB_WEAK : STB_LOCAL;
                sym.st_info = ELF32_ST_INFO(bind, ELF32_ST_TYPE(sym.st_info));
                if (bind == STB_LOCAL)
                    kind = LOCAL;
            }
            remap_.push_back(syms_out_[kind].size() ^ (kind == GLOBAL ? ~0U : 0U));
            syms_out_[kind].push_back(sym);
        }
    }

    void UpdateRelocations(std::span<Elf32_Rela> relocs) {
        // Adjust relocs using remap array.
        const unsigned num_locals = shdr_.sh_info;
        for (auto &reloc : relocs) {
            unsigned sym_ix = ELF32_R_SYM(reloc.r_info);
            if (sym_ix < num_locals)
                continue;

            sym_ix = remap_[sym_ix - num_locals];
            if (bool(sym_ix & (~0U ^ (~0U >> 1))))
                sym_ix = ~sym_ix + syms_out_[LOCAL].size();
            reloc.r_info = ELF32_R_INFO(ELF32_R_TYPE(reloc.r_info), sym_ix + num_locals);
        }
    }

    void RewriteSymbols() {
        // Rewrite the symbols
        std::copy(syms_out_[LOCAL].begin(), syms_out_[LOCAL].end(), syms_in_.begin());
        const_cast<Elf32_Shdr &>(shdr_).sh_info += syms_out_[LOCAL].size();

        std::copy(
            syms_out_[GLOBAL].begin(),
            syms_out_[GLOBAL].end(),
            std::next(syms_in_.begin(), ssize_t(syms_out_[LOCAL].size())));
    }
};

// Any global symbol matching STRONG is preserved.
// Any global symbol in a data-segment section is weakened
// Any other global symbol is made local
void ElfFile::Impl::WeakenDataSymbols(std::span<std::string_view const> strong) {
    for (unsigned ix = GetShdrs().size(); bool(ix--);) {
        auto &shdr = GetShdr(ix);
        if (shdr.sh_type != SHT_SYMTAB || bool(shdr.sh_flags & SHF_ALLOC))
            continue;

        Weakener weakener(shdr, GetSymbols(shdr));
        weakener.WeakenOrLocalizeSymbols(*this, strong);

        for (auto const &relhdr : GetShdrs())
            if (relhdr.sh_type == SHT_RELA && relhdr.sh_link == ix)
                weakener.UpdateRelocations(GetRelocations(relhdr));

        weakener.RewriteSymbols();
    }
}

void ElfFile::Impl::XIPify() {
    // In general there can be several lo12 relocs for a hi20
    // reloc. This is particularly true for lui/{addi,lw,sw,etc}
    // pairs -- a load and a store might share a single lui, as
    // the compiler now emits those insns separately. Thus we have
    // to build a work list and then process it. Furthermore,
    // although auipc/lo12 pairings are clear because the lo12
    // part directly points at the auipc, that is not true of
    // lui/lo12 pairings. We have to use heuristics to locate the
    // matching relocs and that could get arbitrarily hard. We
    // presume (a) the compiler doesn't duplicate lui insns, and
    // (b) the lui preceeds the lo12 in program counter
    // order. Thus we look for a hi20 reloc matching the symbol at
    // a lower offset than the lo12 in question. Fortunately we
    // only need to do this for relocs that need translating, and
    // those happen to be rare when all data-like sections are in
    // the data segment (so putting .rodata in text is
    // problematic). If that proves insufficient here are some
    // ideas:

    // * Insert fn boundaries from symbols of FNtype -- you'll
    //   need to tweak the fn address to not cause collisions in
    //   the reloc map. this might fail with hot/cold block
    //   splitting.

    // * Construct the CFG by examining R_RISCV_BRANCH
    //   relocs. Then walk it (backwards) from each lo12 to find
    //   the reachable hi20. This would be able to deal with
    //   hot/cold splitting, if one constructed the complete
    //   section CFG, not as a per-fn entity. One might get away
    //   with not disasembling to discover ret instructions that
    //   terminate the CFG.

    struct ComposedReloc {
        std::vector<Elf32_Rela *> lo_relocs;
        Elf32_Rela *hi_reloc = nullptr;  // the high part

        ComposedReloc(Elf32_Rela *hi) : hi_reloc(hi) {}
    };

    enum { ABS, PCREL, HWM };
    static char const *const r_names[][2] = {
        {"R_RISCV_HI20", "R_RISCV_LO12"}, {"R_RISCV_PCREL_HI20", "R_RISCV_PCREL_LO12"}};

    auto check_relaxed = [&](Elf32_Rela const &reloc) {
        // If RELOC is the final reloc, this will
        // be out of bounds (and probably fail),
        // but we kind of want that anyway
        if (ELF32_R_TYPE((&reloc)[1].r_info) != R_RISCV_RELAX)
            log_debug(tt::LogLLRuntime, "{}: Relocation at {x} is not relaxed", path_, reloc.r_offset);
    };

    // TODO:We'll eventually place at zero, but this allows the null transformation
    const address_t text_placement = GetSegments().front().address;
    unsigned num_reloc_sections = 0;
    for (auto const &relocHdr : GetShdrs()) {
        if (relocHdr.sh_type != SHT_RELA)
            continue;

        // Is this relocating a section of interest?
        unsigned section_ix = relocHdr.sh_info;
        auto &section = GetShdr(section_ix);
        if (!(section.sh_flags & SHF_ALLOC && section.sh_type != SHT_NOBITS))
            continue;

        int segment_ix = GetSegmentIx(section);
        if (segment_ix < 0)
            continue;

        num_reloc_sections++;
        std::map<offset_t, ComposedReloc> composed[HWM];
        std::vector<Elf32_Rela *> lo[HWM];

        auto symbols = GetSymbols(GetShdr(relocHdr.sh_link));
        auto relocs = GetRelocations(relocHdr);
        bool is_from_text = !segment_ix;

        // ADD32/SUB32 pairs are used for switch tables. Make sure
        // they're consistent.
        Elf32_Rela const *sub_reloc = nullptr;  // Active sub reloc.
        for (auto ix = relocs.size(); ix--;) {
            auto &reloc = relocs[ix];
            if (reloc.r_offset & 3 || reloc.r_offset - section.sh_addr >= section.sh_size)
                TT_THROW(
                    "{}: relocation @ {x} is {} section {}",
                    path_,
                    reloc.r_offset,
                    reloc.r_offset & 3 ? "misaligned in" : "outside of",
                    GetName(section));

            auto type = ELF32_R_TYPE(reloc.r_info);
            auto sym_ix = ELF32_R_SYM(reloc.r_info);
            auto const *symbol = &symbols[sym_ix];
            bool is_to_text = IsTextSymbol(*symbol);

            // Check add/sub relocs are paired and do not cross text/non-text boundary.
            if (bool(sub_reloc) != (type == R_RISCV_ADD32) || (sub_reloc && sub_reloc->r_offset != reloc.r_offset))
            unpaired_sub:
                TT_THROW(
                    "{}: unpaired {} reloc at {x}",
                    path_,
                    sub_reloc ? "sub32" : "add32",
                    (sub_reloc ? sub_reloc : &reloc)->r_offset);
            if (type == R_RISCV_ADD32) {
                auto const *sub_symbol = &symbols[ELF32_R_SYM(sub_reloc->r_info)];
                bool sub_is_to_text = IsTextSymbol(*sub_symbol);
                if (is_to_text != sub_is_to_text)
                    TT_THROW(
                        "{}: mismatched add32/sub32 relocs at {x} & {x}", path_, reloc.r_offset, sub_reloc->r_offset);
            }
            sub_reloc = nullptr;
            if (type == R_RISCV_SUB32) {
                sub_reloc = &reloc;
                if (!ix)
                    goto unpaired_sub;
            }

            unsigned kind = PCREL;
            switch (type) {
                // Abs relocs to text will need fixing up
                case R_RISCV_LO12_I:
                case R_RISCV_LO12_S:
                    if (!is_to_text)
                        break;
                    kind = ABS;
                    [[fallthrough]];

                // PCrel relocs not to text will need fixing up. At
                // this point we don't know the symbol from the LO12
                // relocs, as that points at the hi20 reloc.
                case R_RISCV_PCREL_LO12_I:
                case R_RISCV_PCREL_LO12_S: lo[kind].push_back(&reloc); break;

                case R_RISCV_HI20: kind = ABS; [[fallthrough]];

                case R_RISCV_PCREL_HI20:
                    if (is_to_text && !is_from_text)
                        TT_THROW(
                            "{}: segment-crossing {} relocation found at {x}", path_, r_names[kind][0], reloc.r_offset);

                    if (!is_to_text && kind == ABS)
                        break;
                    composed[kind].emplace(reloc.r_offset, ComposedReloc(&reloc));
                    break;

                case R_RISCV_32: {
                    if (!is_to_text)
                        break;
                    // Emit dynamic reloc
                    log_debug(
                        tt::LogLLRuntime, "{}: emitting dynamic R_RISCV_32 relocation at {x}", path_, reloc.r_offset);
                    address_t value =
                        (symbol->st_value + reloc.r_addend - GetSegments().front().address + text_placement);
                    Write32(section, reloc.r_offset, value);
                    auto &seg = GetSegments()[segment_ix];
                    seg.relocs.push_back(reloc.r_offset - seg.address);
                } break;

                case R_RISCV_JAL:
                    if (is_from_text != is_to_text)
                        TT_THROW("{}: segment-crossing R_RISCV_JAL relocation found at {x}", path_, reloc.r_offset);
                    break;

                case R_RISCV_CALL:
                case R_RISCV_CALL_PLT:
                    TT_THROW("{}: R_RISCV_CALL{,_PLT} relocation found at {x}", path_, reloc.r_offset);
                    break;

                case R_RISCV_32_PCREL:
                    TT_THROW("{}: R_RISCV_32_PCREL relocation found at {x}", path_, reloc.r_offset);
                    break;
            }
        }

        // Combine hi/lo relocs

        // We can't do abs ones in general with complete accuracy,
        // because there could be multiple possible matching hi
        // relocs. If we construct the CFG then it becomes more
        // accurate, but it's always going to be somewhat
        // heuristic. Let's hope CFG construction is unnecessary. A
        // first step in that direction might be to insert function
        // boundaries, to stop the search.
        for (unsigned kind = HWM; kind--;) {
            for (auto *lo_reloc : lo[kind]) {
                // Find the matching hi-reloc by searching backwards. This
                // presumes block reordering hasn't done something to
                // break that.
                unsigned sym_ix = ELF32_R_SYM(lo_reloc->r_info);
                auto hi_reloc = composed[kind].begin();

                if (kind == ABS) {
                    hi_reloc = composed[kind].lower_bound(lo_reloc->r_offset);
                    while (hi_reloc != composed[kind].begin()) {
                        --hi_reloc;
                        if (ELF32_R_SYM(hi_reloc->second.hi_reloc->r_info) == sym_ix)
                            goto found;
                    }
                } else {
                    uint32_t hi_offset = symbols[sym_ix].st_value + lo_reloc->r_addend;
                    hi_reloc = composed[kind].find(hi_offset);
                    if (hi_reloc != composed[kind].end())
                        goto found;
                }
                TT_THROW(
                    "{}: {} relocation at {x} has no matching {}",
                    path_,
                    r_names[kind][true],
                    lo_reloc->r_offset,
                    r_names[kind][false]);
            found:
                hi_reloc->second.lo_relocs.push_back(lo_reloc);
            }
        }

        // Process composed relocations
        for (unsigned kind = HWM; kind--;) {
            for (auto &slot : composed[kind]) {
                if (slot.second.lo_relocs.empty())
                    TT_THROW(
                        "{}: R_RISCV_{}HI20 relocation at {x} has no matching R_RISCV_{}LO12",
                        path_,
                        r_names[kind][false],
                        r_names[kind][true],
                        slot.first);

                auto hi_reloc = slot.second.hi_reloc;
                unsigned sym_ix = ELF32_R_SYM(hi_reloc->r_info);
                auto const &symbol = symbols[sym_ix];
                bool is_to_text = IsTextSymbol(symbol);
                if (is_to_text == is_from_text)
                    continue;

                address_t value = symbol.st_value + hi_reloc->r_addend;
                if (kind == ABS) {
                    value -= slot.first;
                    sym_ix = 0;
                }

                // translate hi
                check_relaxed(*hi_reloc);
                uint32_t insn = Read32(section, hi_reloc->r_offset);
                log_debug(
                    tt::LogLLRuntime,
                    "{}: translating {} at {x} to {}",
                    path_,
                    r_names[kind][false],
                    hi_reloc->r_offset,
                    r_names[HWM - 1 - kind][false]);
                if ((insn & insn_mask_u) != (kind == ABS ? insn_opc_lui : insn_opc_auipc))
                    TT_THROW(
                        "{}: translating instruction at {x} is not `{}'",
                        path_,
                        hi_reloc->r_offset,
                        kind == ABS ? "lui" : "auipc");
                insn &= mask_hi20;                      // Remove old immediate
                insn ^= insn_opc_auipc ^ insn_opc_lui;  // Convert opcode
                // Insert new immediate
                insn |= ((value + (1 << 11)) >> 12) << mask_hi20_shift;
                Write32(section, hi_reloc->r_offset, insn);
                hi_reloc->r_info ^= ELF32_R_INFO(0, R_RISCV_HI20 ^ R_RISCV_PCREL_HI20);

                // translate lo
                for (auto *lo_reloc : slot.second.lo_relocs) {
                    unsigned type = ELF32_R_TYPE(lo_reloc->r_info);
                    bool is_form_i = type == (kind == PCREL ? R_RISCV_PCREL_LO12_I : R_RISCV_LO12_I);
                    check_relaxed(*lo_reloc);
                    uint32_t insn = Read32(section, lo_reloc->r_offset);
                    log_debug(
                        tt::LogLLRuntime,
                        "{}: translating R_RISCV{}_LO12 at {x} to R_RISCV{}_LO12",
                        path_,
                        r_names[kind][true],
                        lo_reloc->r_offset,
                        r_names[HWM - 1 - kind][true]);
                    if (is_form_i) {
                        insn &= mask_lo12_i;
                        insn |= (value & 0x0fff) << mask_lo12_i_shift;
                    } else {
                        // S form splits the immediate
                        insn &= mask_lo12_s;
                        insn |= (value & ((1 << mask_lo12_s_split) - 1)) << mask_lo12_s_shift_1;
                        insn |= ((value & 0x0fff) >> mask_lo12_s_split) << mask_lo12_s_shift_2;
                    }
                    Write32(section, lo_reloc->r_offset, insn);

                    // We can't convert to PCREL with fidelity, as
                    // that involves adding a symbol. Instead, let's
                    // use a null symbol and an addend.
                    lo_reloc->r_info = ELF32_R_INFO(
                        sym_ix,
                        type ^ (is_form_i ? (R_RISCV_LO12_I ^ R_RISCV_PCREL_LO12_I)
                                          : (R_RISCV_LO12_S ^ R_RISCV_PCREL_LO12_S)));
                    lo_reloc->r_addend = kind == PCREL ? slot.second.hi_reloc->r_addend
                                                       : slot.second.hi_reloc->r_offset - lo_reloc->r_offset;
                }
            }
        }
    }

    if (!num_reloc_sections)
        // Hm, that's suspicious
        TT_THROW("{}: there are no relocation sections", path_);

    // The text segment is now XIP
    GetSegments().front().address = text_placement;
}
