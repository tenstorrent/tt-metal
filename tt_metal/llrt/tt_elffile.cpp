// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_elffile.hpp"

#include <tt_stl/assert.hpp>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <iterator>
#include <map>
#include <type_traits>

#include <tt-logger/tt-logger.hpp>

// Verify some knowledge of, and compatibilty with, RiscV
#ifndef EM_RISCV
#error "Don't know RISCV elf details"
#endif

// Having the same endianness as RISCV makes things easier.
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Host must be little endian"
#endif

#ifndef PT_RISCV_ATTRIBUTES
// Missing from my elf.h
#define PT_RISCV_ATTRIBUTES (PT_LOPROC + 3)
enum {
    Tag_RISCV_arch = 5,
};
#endif

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
protected:
    ElfFile& owner_;
    // This is a view of the caller's object, which must remain live
    // for the lifetime of this object. (As that's the case anyway,
    // there's no burden on the caller). See the document on
    // ReadImage's declaration.
    const std::string_view path_;

public:  // NOLINT
    Impl(ElfFile& owner, std::string_view path) : owner_(owner), path_(path) {}
    virtual ~Impl() = default;

public:  // NOLINT
    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&) = delete;
    Impl& operator=(Impl&&) = delete;

public:  // NOLINT
    static Impl* Make(ElfFile& owner, const std::string& path);

public:  // NOLINT
    virtual void LoadImage() = 0;
    virtual void WeakenDataSymbols(std::span<const std::string_view> strong_names) = 0;
    virtual void XIPify() = 0;

private:  // NOLINT
    template <bool Is64>
    class Elf;
};

template <bool Is64>
class ElfFile::Impl::Elf final : public Impl {
public:  // NOLINT
    using Ehdr = std::conditional_t<Is64, Elf64_Ehdr, Elf32_Ehdr>;
    using Phdr = std::conditional_t<Is64, Elf64_Phdr, Elf32_Phdr>;
    using Shdr = std::conditional_t<Is64, Elf64_Shdr, Elf32_Shdr>;
    using Sym = std::conditional_t<Is64, Elf64_Sym, Elf32_Sym>;
    using Rela = std::conditional_t<Is64, Elf64_Rela, Elf32_Rela>;

private:  // NOLINT
    std::span<Phdr> phdrs_;
    std::span<Shdr> shdrs_;

    class Weakener;

public:  // NOLINT
    Elf(ElfFile& owner, std::string_view path) : Impl(owner, path) {}
    virtual ~Elf() override = default;  // NOLINT

    virtual void LoadImage() override;                                                        // NOLINT
    virtual void WeakenDataSymbols(std::span<const std::string_view> strong_names) override;  // NOLINT
    virtual void XIPify() override;                                                           // NOLINT

private:  // NOLINT
    [[nodiscard]] auto GetHeader() const -> const Ehdr& { return *ByteOffset<Ehdr>(GetContents().data()); }
    [[nodiscard]] auto GetPhdrs() const -> std::span<const Phdr> { return phdrs_; }
    [[nodiscard]] auto GetShdrs() const -> std::span<const Shdr> { return shdrs_; }
    [[nodiscard]] auto GetShdr(unsigned ix) const -> const Shdr& { return shdrs_[ix]; }
    [[nodiscard]] auto GetSegments() const -> std::vector<Segment>& { return owner_.segments_; }
    [[nodiscard]] auto GetContents() const -> std::span<std::byte>& { return owner_.contents_; }
    [[nodiscard]] auto GetContents(const Phdr& phdr) const -> std::span<std::byte> {
        return GetContents().subspan(phdr.p_offset, phdr.p_filesz);
    }
    [[nodiscard]] auto GetContents(const Shdr& shdr) const -> std::span<std::byte> {
        return GetContents().subspan(shdr.sh_offset, shdr.sh_size);
    }
    [[nodiscard]] auto GetString(size_t offset, const Shdr& shdr) const -> const char* {
        return ByteOffset<char const>(GetContents(shdr).data(), offset);
    }
    [[nodiscard]] auto GetName(const Shdr& shdr) const -> const char* {
        return GetString(shdr.sh_name, GetShdr(GetHeader().e_shstrndx));
    }
    [[nodiscard]] auto GetSymbols(const Shdr& shdr) const -> std::span<Sym> {
        auto section = GetContents(shdr);
        return std::span(ByteOffset<Sym>(section.data()), section.size() / shdr.sh_entsize);
    }
    [[nodiscard]] auto GetName(const Sym& sym, unsigned link) const -> const char* {
        return GetString(sym.st_name, GetShdr(link));
    }
    [[nodiscard]] auto GetRelocations(const Shdr& shdr) const -> std::span<Rela> {
        auto section = GetContents(shdr);
        return std::span(ByteOffset<Rela>(section.data()), section.size() / shdr.sh_entsize);
    }

    [[nodiscard]] static bool IsInSegment(const Segment& segment, const Shdr& shdr) {
        // Remember, Segments use word_t sizes. If a zero-sized
        // section is at the end of a segment, it is considered in
        // that segment. Fortunately, we do not have abutting
        // segments, so do not have to consider the case of a zero
        // length section sitting at that boundary. We also take
        // advantage of the (a) fact that sections cannot straddle
        // segment boundaries -- they're either wholey inside or
        // wholey outside, and (b) unsigned arithmetic.
        return shdr.sh_flags & SHF_ALLOC && shdr.sh_addr + shdr.sh_size - segment.address <= segment.membytes;
    }
    [[nodiscard]] bool IsInSegment(unsigned _ix, const Shdr& shdr) const {
        return IsInSegment(GetSegments()[_ix], shdr);
    }
    [[nodiscard]] bool IsInText(const Shdr& shdr) const { return IsInSegment(GetSegments().front(), shdr); };
    [[nodiscard]] int GetSegmentIx(const Shdr& shdr) const {
        for (unsigned ix = GetSegments().size(); ix--;) {
            if (IsInSegment(ix, shdr)) {
                return ix;
            }
        }
        return -1;
    };
    [[nodiscard]] bool IsTextSymbol(const Sym& symbol) const {
        return symbol.st_shndx < GetShdrs().size() && IsInText(GetShdr(symbol.st_shndx));
    }
    [[nodiscard]] bool IsDataSymbol(const Sym& symbol) const {
        return symbol.st_shndx < GetShdrs().size() && GetSegmentIx(GetShdr(symbol.st_shndx)) > 0;
    }

    template <typename T = std::byte>
    [[nodiscard]] static T* ByteOffset(std::byte* base, size_t offset = 0) {
        return reinterpret_cast<T*>(base + offset);
    }
    template <typename T = std::byte>
    [[nodiscard]] static T const* ByteOffset(std::byte const* base, size_t offset = 0) {
        return reinterpret_cast<T const*>(base + offset);
    }

    uint32_t Read32(const Shdr& shdr, address_t addr) {
        return *ByteOffset<uint32_t>(GetContents(shdr).data(), addr - shdr.sh_addr);
    }
    void Write32(const Shdr& shdr, address_t addr, uint32_t value) {
        *ByteOffset<uint32_t>(GetContents(shdr).data(), addr - shdr.sh_addr) = value;
    }

private:  // NOLINT
    static unsigned char GetSymBind(const Sym& sym) {
        if constexpr (Is64) {
            return ELF64_ST_BIND(sym.st_info);
        } else {
            return ELF32_ST_BIND(sym.st_info);
        }
    }
    static unsigned char GetSymType(const Sym& sym) {
        if constexpr (Is64) {
            return ELF64_ST_TYPE(sym.st_info);
        } else {
            return ELF32_ST_TYPE(sym.st_info);
        }
    }
    static void SetSymInfo(Sym& sym, unsigned bind, unsigned type) {
        if constexpr (Is64) {
            sym.st_info = ELF64_ST_INFO(bind, type);
        } else {
            sym.st_info = ELF32_ST_INFO(bind, type);
        }
    }
    static unsigned GetRelocSymIx(const Rela& rel) {
        if constexpr (Is64) {
            return ELF64_R_SYM(rel.r_info);
        } else {
            return ELF32_R_SYM(rel.r_info);
        }
    }
    static unsigned GetRelocType(const Rela& rel) {
        if constexpr (Is64) {
            return ELF64_R_TYPE(rel.r_info);
        } else {
            return ELF32_R_TYPE(rel.r_info);
        }
    }
    static void SetRelocInfo(Rela& rel, unsigned sym, unsigned type) {
        if constexpr (Is64) {
            rel.r_info = ELF64_R_INFO(sym, type);
        } else {
            rel.r_info = ELF32_R_INFO(sym, type);
        }
    }
    static void XorRelocInfo(Rela& rel, unsigned sym, unsigned type) {
        if constexpr (Is64) {
            rel.r_info ^= ELF64_R_INFO(sym, type);
        } else {
            rel.r_info ^= ELF32_R_INFO(sym, type);
        }
    }
};

ElfFile::~ElfFile() {
    ReleaseImpl();
    if (!contents_.empty()) {
        munmap(contents_.data(), contents_.size());
    }
}

void ElfFile::ReleaseImpl() {
    delete pimpl_;
    pimpl_ = nullptr;
}

void ElfFile::ReadImage(const std::string& path) {
    pimpl_ = Impl::Make(*this, path);
    pimpl_->LoadImage();
}

ElfFile::Impl* ElfFile::Impl::Make(ElfFile& owner, const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat st{};
    void* buffer = MAP_FAILED;
    if (fd >= 0 && fstat(fd, &st) >= 0) {
        buffer = mmap(nullptr, st.st_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    }
    if (fd >= 0) {
        // It is acceptable to close a mapped file -- the mapping stays.
        close(fd);
    }
    if (buffer == MAP_FAILED) {
        TT_THROW("{}: cannot map elf file into memory: {}", path, strerror(errno));
    }

    owner.contents_ = std::span(reinterpret_cast<std::byte*>(buffer), st.st_size);

    // Sniff the header
    const unsigned char* ident = reinterpret_cast<const unsigned char*>(buffer);

    // Make sure it's ELF ...
    if (!(ident[EI_MAG0] == 0x7f && ident[EI_MAG1] == 'E' && ident[EI_MAG2] == 'L' && ident[EI_MAG3] == 'F')) {
        TT_THROW("{}: no ELF magic found", path);
    }

    // ... of the expected address size, endianness and version
    bool is_32 = ident[EI_CLASS] == ELFCLASS32;
    bool is_64 = ident[EI_CLASS] == ELFCLASS64;
    if (!(is_32 || is_64) || !(ident[EI_DATA] == ELFDATA2LSB && ident[EI_VERSION] == EV_CURRENT)) {
        TT_THROW("{}: incompatible address size or endianness", path);
    }

    if (is_64) {
        return new Elf<true>(owner, path);
    } else {
        return new Elf<false>(owner, path);
    }
}

void ElfFile::WriteImage(std::string const& path) {
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
    if (failed) {
        TT_THROW("{}: cannot map elf file into memory: {}", path, strerror(errno));
    }
}

void ElfFile::WeakenDataSymbols(std::span<std::string_view const> strong) { pimpl_->WeakenDataSymbols(strong); }

void ElfFile::MakeExecuteInPlace() { pimpl_->XIPify(); }

template <bool Is64>
void ElfFile::Impl::Elf<Is64>::LoadImage() {
    auto& hdr = GetHeader();

    if (hdr.e_type != ET_EXEC) {
        TT_THROW("{}: not an executable", path_);
    }

    if (hdr.e_machine != EM_RISCV) {
        TT_THROW("{}: incompatible architecture {}", path_, hdr.e_machine);
    }

    if (!hdr.e_phoff || hdr.e_phoff & (sizeof(address_t) - 1) || hdr.e_phentsize != sizeof(Phdr) ||
        (hdr.e_phoff + hdr.e_phnum * sizeof(Phdr) > GetContents().size())) {
        TT_THROW("{}: PHDRS are missing or malformed", path_);
    }
    phdrs_ = std::span(ByteOffset<Phdr>(GetContents().data(), hdr.e_phoff), hdr.e_phnum);
    if (!hdr.e_shoff || hdr.e_shoff & (sizeof(address_t) - 1) || hdr.e_shentsize != sizeof(Shdr) ||
        (hdr.e_shoff + hdr.e_shnum * sizeof(Shdr) > GetContents().size())) {
        TT_THROW("{}: sections are missing or malformed", path_);
    }
    shdrs_ = std::span(ByteOffset<Shdr>(GetContents().data(), hdr.e_shoff), hdr.e_shnum);
    if (!hdr.e_shstrndx || hdr.e_shstrndx >= GetShdrs().size()) {
        TT_THROW("{}: string table is missing or malformed", path_);
    }

    GetSegments().reserve(hdr.e_phnum);
    bool haveStack = false;
    for (auto const& phdr : GetPhdrs()) {
        if (phdr.p_type == PT_RISCV_ATTRIBUTES) {
            // TODO: verify Arch is ok?
            continue;
        }

        if (phdr.p_type == PT_GNU_STACK) {
            haveStack = true;
        } else if (phdr.p_type != PT_LOAD) {
            continue;
        } else if (haveStack) {
            TT_THROW("{}: loadable segments after stack segment", path_);
        }

        log_debug(
            tt::LogLLRuntime,
            "{}: loadable segment {}: [{},+{}/{})@{}",
            path_,
            unsigned(GetSegments().size()),
            phdr.p_vaddr,
            phdr.p_filesz,
            phdr.p_memsz,
            phdr.p_offset);

        // Require loadable segments to be nicely aligned
        if (((phdr.p_offset | phdr.p_vaddr | phdr.p_paddr) & (sizeof(word_t) - 1)) ||
            // Only support loading into the first 4GB
            (Is64 && ((phdr.p_vaddr | phdr.p_paddr) + phdr.p_memsz) > UINT32_MAX)) {
            TT_THROW(
                "{}: loadable segment {} is misaligned or misplaced, [{}({}),+{}/{})@{}",
                path_,
                unsigned(GetSegments().size()),
                phdr.p_vaddr,
                phdr.p_paddr,
                phdr.p_filesz,
                phdr.p_memsz,
                phdr.p_offset);
        }

        auto contents = GetContents(phdr);
        // We require the first segment to be text, and that the entry
        // point is the start of that segment.
        if (GetSegments().empty() && hdr.e_entry != phdr.p_vaddr) {
            TT_THROW("{}: first loadable segment is not text", path_);
        }

        // This word-size rounding up means the span can occupy some bytes
        // outside the range of the original span, but those bytes will
        // still be inside the span covering the whole file, so that's ok.
        offset_t file_words = (phdr.p_filesz + sizeof(word_t) - 1) / sizeof(word_t);
        offset_t mem_bytes = (phdr.p_memsz + sizeof(word_t) - 1) & ~(sizeof(word_t) - 1);
        GetSegments().emplace_back(
            std::span(reinterpret_cast<word_t const*>(contents.data()), file_words),
            phdr.p_vaddr,
            phdr.p_paddr,
            mem_bytes);
    }

    // Check sections
    for (auto const& section : GetShdrs()) {
        // We care about alignment of allocatable sections,
        // relocations and symbols.
        if ((section.sh_flags & SHF_ALLOC || section.sh_type == SHT_RELA || section.sh_type == SHT_SYMTAB) &&
            (section.sh_offset | section.sh_addr) & (sizeof(word_t) - 1)) {
            TT_THROW(
                "{}: section {} is misaligned [{},+{})@{}",
                path_,
                GetName(section),
                section.sh_addr,
                section.sh_size,
                section.sh_offset);
        }
        // If it's allocatable, make sure it's in a segment.
        if (section.sh_flags & SHF_ALLOC && GetSegmentIx(section) < 0) {
            TT_THROW(
                "{}: allocatable section {} [{},+{})@{} is not in known segment",
                path_,
                GetName(section),
                section.sh_addr,
                section.sh_size,
                section.sh_offset);
        }
        // If the name begins with .empty. it should be empty.  We can
        // generate a better error here than the linker can -- and one
        // has the binary to examine.
        if (section.sh_flags & SHF_ALLOC && section.sh_size != 0) {
            auto* name = GetName(section);
            constexpr auto* prefix = ".empty.";
            if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
                TT_THROW("{}: {} section has contents (namespace-scope constructor present?)", path_, name);
            }
        }
        if (!(section.sh_flags & SHF_ALLOC) && section.sh_type == SHT_PROGBITS &&
             std::strcmp(GetName(section), ".phdrs") == 0) {
            // Specifies phdr size limits
            auto bytes = GetContents(section);
            auto words = std::span(reinterpret_cast<uint32_t const *>(bytes.data()), bytes.size() / sizeof(uint32_t));
            for (unsigned ix = 0; ix != words.size(); ix++) {
                if (ix >= GetSegments().size())
                    continue;
                uint32_t limit = words[ix];
                auto const &seg = GetSegments()[ix];
                if (seg.membytes > limit) {
                    TT_THROW("{}: phdr[{}] [{},+{}) overflows limit of {} bytes, {}",
                             path_, ix, seg.address, seg.membytes, limit,
                             ix == 0 ? "reduce the code size" :
                             ix == 1 ? "reduce the number of statically allocated variables (e.g, globals)" :
                             "examine executable for segment details"
                        );
                }
            }
        }
        if (std::strcmp(GetName(section), ".data") == 0) {
            // Verify this is at the start of segment 1 -- we had a
            // linker script bug at one point.
            bool in_range = GetSegments().size() >= 2;
            if (!in_range || section.sh_addr != GetSegments()[1].address) {
                TT_THROW("{}: .data section at [{},+{}) not at start of data segment at [{},+{})",
                         path_,
                         section.sh_addr, section.sh_size,
                         in_range ? GetSegments()[1].address : 0,
                         in_range ? GetSegments()[1].membytes : 0);
            }
        }
    }
    if (haveStack) {
        // Remove the stack segment, now we used it for checking the sections.
        GetSegments().pop_back();
    }
}

template <bool Is64>
class ElfFile::Impl::Elf<Is64>::Weakener {
    enum { LOCAL, GLOBAL, HWM };

    const Shdr& shdr_;
    std::span<Sym> syms_in_;
    std::vector<unsigned> remap_;
    std::vector<Sym> syms_out_[HWM];

public:
    Weakener(const Shdr& shdr, std::span<Sym> symbols) : shdr_(shdr), syms_in_(symbols.subspan(shdr.sh_info)) {
        unsigned reserve = syms_in_.size();
        remap_.reserve(reserve);
        std::ranges::for_each(syms_out_, [=](std::vector<Sym>& syms) { syms.reserve(reserve); });
    }

    void WeakenOrLocalizeSymbols(Elf<Is64>& impl, std::span<const std::string_view> strong) {
        auto name_matches = [](std::string_view name, std::span<std::string_view const> list) {
            return std::ranges::any_of(list, [&](std::string_view pattern) {
                return pattern.back() == '*' ? name.starts_with(pattern.substr(0, pattern.size() - 1))
                                             : name == pattern;
            });
        };

        // Weaken or hide globals
        for (auto& sym : syms_in_) {
            auto kind = GLOBAL;
            auto bind = impl.GetSymBind(sym);
            if ((bind == STB_GLOBAL || bind == STB_WEAK) && !name_matches(impl.GetName(sym, shdr_.sh_link), strong)) {
                bind = impl.IsDataSymbol(sym) ? STB_WEAK : STB_LOCAL;
                impl.SetSymInfo(sym, bind, impl.GetSymType(sym));
                if (bind == STB_LOCAL) {
                    kind = LOCAL;
                }
            }
            remap_.push_back(syms_out_[kind].size() ^ (kind == GLOBAL ? ~0U : 0U));
            syms_out_[kind].push_back(sym);
        }
    }

    void UpdateRelocations(std::span<Rela> relocs) {
        // Adjust relocs using remap array.
        const unsigned num_locals = shdr_.sh_info;
        for (auto& reloc : relocs) {
            unsigned sym_ix = GetRelocSymIx(reloc);
            if (sym_ix < num_locals) {
                continue;
            }

            sym_ix = remap_[sym_ix - num_locals];
            if (bool(sym_ix & (~0U ^ (~0U >> 1)))) {
                sym_ix = ~sym_ix + syms_out_[LOCAL].size();
            }
            SetRelocInfo(reloc, sym_ix + num_locals, GetRelocType(reloc));
        }
    }

    void RewriteSymbols() {
        // Rewrite the symbols
        std::copy(syms_out_[LOCAL].begin(), syms_out_[LOCAL].end(), syms_in_.begin());
        const_cast<Shdr&>(shdr_).sh_info += syms_out_[LOCAL].size();

        std::copy(
            syms_out_[GLOBAL].begin(),
            syms_out_[GLOBAL].end(),
            std::next(syms_in_.begin(), ssize_t(syms_out_[LOCAL].size())));
    }
};

// Any global symbol matching STRONG is preserved.
// Any global symbol in a data-segment section is weakened
// Any other global symbol is made local
template <bool Is64>
void ElfFile::Impl::Elf<Is64>::WeakenDataSymbols(std::span<const std::string_view> strong) {
    for (unsigned ix = GetShdrs().size(); bool(ix--);) {
        auto& shdr = GetShdr(ix);
        if (shdr.sh_type != SHT_SYMTAB || bool(shdr.sh_flags & SHF_ALLOC)) {
            continue;
        }

        Weakener weakener(shdr, GetSymbols(shdr));
        weakener.WeakenOrLocalizeSymbols(*this, strong);

        for (auto const& relhdr : GetShdrs()) {
            if (relhdr.sh_type == SHT_RELA && relhdr.sh_link == ix) {
                weakener.UpdateRelocations(GetRelocations(relhdr));
            }
        }

        weakener.RewriteSymbols();
    }
}

template <bool Is64>
void ElfFile::Impl::Elf<Is64>::XIPify() {
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
        std::vector<Rela*> lo_relocs;
        Rela* hi_reloc = nullptr;  // the high part

        ComposedReloc(Rela* hi) : hi_reloc(hi) {}
    };

    enum { ABS, PCREL, HWM };
    static char const* const r_names[][2] = {
        {"R_RISCV_HI20", "R_RISCV_LO12"}, {"R_RISCV_PCREL_HI20", "R_RISCV_PCREL_LO12"}};

    auto check_relaxed = [&](const Rela& reloc) {
        // If RELOC is the final reloc, this will
        // be out of bounds (and probably fail),
        // but we kind of want that anyway
        if (GetRelocType((&reloc)[1]) != R_RISCV_RELAX) {
            log_debug(tt::LogLLRuntime, "{}: Relocation at {} is not relaxed", path_, reloc.r_offset);
        }
    };

    unsigned num_reloc_sections = 0;
    for (auto const& relocHdr : GetShdrs()) {
        if (relocHdr.sh_type != SHT_RELA) {
            continue;
        }

        // Is this relocating a section of interest?
        unsigned section_ix = relocHdr.sh_info;
        auto& section = GetShdr(section_ix);
        if (!(section.sh_flags & SHF_ALLOC && section.sh_type != SHT_NOBITS)) {
            continue;
        }

        int segment_ix = GetSegmentIx(section);
        if (segment_ix < 0) {
            continue;
        }

        num_reloc_sections++;
        std::map<offset_t, ComposedReloc> composed[HWM];
        std::vector<Rela*> lo[HWM];

        auto symbols = GetSymbols(GetShdr(relocHdr.sh_link));
        auto relocs = GetRelocations(relocHdr);
        bool is_from_text = !segment_ix;

        // ADD32/SUB32 pairs are used for switch tables. Make sure
        // they're consistent.
        const Rela* sub_reloc = nullptr;  // Active sub reloc.
        for (auto ix = relocs.size(); ix--;) {
            auto& reloc = relocs[ix];
            // We can get a RISCV_NONE right at the end (!)
            if (reloc.r_offset & 3 ||
                reloc.r_offset - section.sh_addr >= section.sh_size + int(GetRelocType(reloc) == R_RISCV_NONE)) {
                TT_THROW(
                    "{}: relocation @ {} is {} section {}",
                    path_,
                    reloc.r_offset,
                    reloc.r_offset & 3 ? "misaligned in" : "outside of",
                    GetName(section));
            }

            auto type = GetRelocType(reloc);
            auto sym_ix = GetRelocSymIx(reloc);
            auto const* symbol = &symbols[sym_ix];
            bool is_to_text = IsTextSymbol(*symbol);

            auto throw_unpaired = [&]() {
                TT_THROW(
                    "{}: unpaired {} reloc at {}",
                    path_,
                    sub_reloc ? "sub32" : "add32",
                    (sub_reloc ? sub_reloc : &reloc)->r_offset);
            };

            // Check add/sub relocs are paired and do not cross text/non-text boundary.
            if (bool(sub_reloc) != (type == R_RISCV_ADD32) || (sub_reloc && sub_reloc->r_offset != reloc.r_offset)) {
                throw_unpaired();
            }
            if (type == R_RISCV_ADD32) {
                const auto* sub_symbol = &symbols[GetRelocSymIx(*sub_reloc)];
                bool sub_is_to_text = IsTextSymbol(*sub_symbol);
                if (is_to_text != sub_is_to_text) {
                    TT_THROW(
                        "{}: mismatched add32/sub32 relocs at {} & {}", path_, reloc.r_offset, sub_reloc->r_offset);
                }
            }
            sub_reloc = nullptr;
            if (type == R_RISCV_SUB32) {
                sub_reloc = &reloc;
                if (!ix) {
                    throw_unpaired();
                }
            }

            unsigned kind = PCREL;
            // NOLINTBEGIN(bugprone-switch-missing-default-case)
            switch (type) {
                case R_RISCV_LO12_I:
                case R_RISCV_LO12_S: kind = ABS; [[fallthrough]];

                case R_RISCV_PCREL_LO12_I:
                case R_RISCV_PCREL_LO12_S:
                    if (kind == ABS && !is_to_text) {
                        // Abs relocs not to text do not need to be translated.
                        break;
                    }

                    // PCrel relocs to text will not need translation,
                    // but at this point we don't know the symbol as
                    // these relocs point to the hi20 reloc.  Record
                    // them all and filter later.
                    lo[kind].push_back(&reloc);
                    break;

                case R_RISCV_HI20: kind = ABS; [[fallthrough]];

                case R_RISCV_PCREL_HI20:
                    if (is_to_text && !is_from_text) {
                        TT_THROW(
                            "{}: segment-crossing {} relocation found at {}", path_, r_names[kind][0], reloc.r_offset);
                    }

                    if (kind == ABS && !is_to_text) {
                        // Abs relocs not to text do not need to be translated.
                        break;
                    }

                    composed[kind].emplace(reloc.r_offset, ComposedReloc(&reloc));
                    break;

                case R_RISCV_32: {
                    if (!is_to_text) {
                        break;
                    }
                    // Emit dynamic reloc
                    log_debug(
                        tt::LogLLRuntime, "{}: emitting dynamic R_RISCV_32 relocation at {}", path_, reloc.r_offset);
                    address_t value = (symbol->st_value + reloc.r_addend - GetSegments().front().address);
                    Write32(section, reloc.r_offset, value);
                    auto& seg = GetSegments()[segment_ix];
                    seg.relocs.push_back(reloc.r_offset - seg.address);
                } break;

                case R_RISCV_JAL:
                case R_RISCV_CALL:
                case R_RISCV_CALL_PLT:
                    if (is_from_text != is_to_text) {
                        TT_THROW(
                            "{}: segment-crossing R_RISCV_(JAL|CALL|CALL_PLT) relocation found at {}",
                            path_,
                            reloc.r_offset);
                    }
                    break;

                case R_RISCV_32_PCREL:
                    TT_THROW("{}: R_RISCV_32_PCREL relocation found at {}", path_, reloc.r_offset);
                    break;
            }
            // NOLINTEND(bugprone-switch-missing-default-case)
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
            for (auto* lo_reloc : lo[kind]) {
                // Find the matching hi-reloc by searching backwards. This
                // presumes block reordering hasn't done something to
                // break that.
                unsigned sym_ix = GetRelocSymIx(*lo_reloc);
                auto hi_reloc = composed[kind].begin();

                if (kind == ABS) {
                    hi_reloc = composed[kind].lower_bound(lo_reloc->r_offset);
                    while (hi_reloc != composed[kind].begin()) {
                        --hi_reloc;
                        if (GetRelocSymIx(*hi_reloc->second.hi_reloc) == sym_ix) {
                            goto found;
                        }
                    }
                } else {
                    uint32_t hi_offset = symbols[sym_ix].st_value + lo_reloc->r_addend;
                    hi_reloc = composed[kind].find(hi_offset);
                    if (hi_reloc != composed[kind].end()) {
                        goto found;
                    }
                }
                TT_THROW(
                    "{}: {} relocation at {} has no matching {}",
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
            for (auto& slot : composed[kind]) {
                if (slot.second.lo_relocs.empty()) {
                    TT_THROW(
                        "{}: R_RISCV_{}HI20 relocation at {} has no matching R_RISCV_{}LO12",
                        path_,
                        r_names[kind][false],
                        r_names[kind][true],
                        slot.first);
                }

                auto hi_reloc = slot.second.hi_reloc;
                unsigned sym_ix = GetRelocSymIx(*hi_reloc);
                auto const& symbol = symbols[sym_ix];
                bool is_to_text = IsTextSymbol(symbol);
                if (kind == PCREL && is_to_text == is_from_text) {
                    // intra-text PCREL is ok.
                    continue;
                }

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
                    "{}: translating {} at {} to {}",
                    path_,
                    r_names[kind][false],
                    hi_reloc->r_offset,
                    r_names[HWM - 1 - kind][false]);
                if ((insn & insn_mask_u) != (kind == ABS ? insn_opc_lui : insn_opc_auipc)) {
                    TT_THROW(
                        "{}: translating instruction at {} is not `{}'",
                        path_,
                        hi_reloc->r_offset,
                        kind == ABS ? "lui" : "auipc");
                }
                insn &= mask_hi20;                      // Remove old immediate
                insn ^= insn_opc_auipc ^ insn_opc_lui;  // Convert opcode
                // Insert new immediate
                insn |= ((value + (1 << 11)) >> 12) << mask_hi20_shift;
                Write32(section, hi_reloc->r_offset, insn);
                XorRelocInfo(*hi_reloc, 0, R_RISCV_HI20 ^ R_RISCV_PCREL_HI20);

                // translate lo
                for (auto* lo_reloc : slot.second.lo_relocs) {
                    unsigned type = GetRelocType(*lo_reloc);
                    bool is_form_i = type == (kind == PCREL ? R_RISCV_PCREL_LO12_I : R_RISCV_LO12_I);
                    check_relaxed(*lo_reloc);
                    uint32_t insn = Read32(section, lo_reloc->r_offset);
                    log_debug(
                        tt::LogLLRuntime,
                        "{}: translating R_RISCV{}_LO12 at {} to R_RISCV{}_LO12",
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
                    SetRelocInfo(
                        *lo_reloc,
                        sym_ix,
                        type ^ (is_form_i ? (R_RISCV_LO12_I ^ R_RISCV_PCREL_LO12_I)
                                          : (R_RISCV_LO12_S ^ R_RISCV_PCREL_LO12_S)));
                    lo_reloc->r_addend = kind == PCREL ? slot.second.hi_reloc->r_addend
                                                       : slot.second.hi_reloc->r_offset - lo_reloc->r_offset;
                }
            }
        }
    }

    if (!num_reloc_sections) {
        // Hm, that's suspicious
        TT_THROW("{}: there are no relocation sections", path_);
    }

    // The text segment is now XIP
    GetSegments().front().address = 0;
}
