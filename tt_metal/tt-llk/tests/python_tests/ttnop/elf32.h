#ifndef TTNOP_ELF32_H
#define TTNOP_ELF32_H

// read-only ELF32 little-endian RISC-V reader for `ttnop` scan
// read the .text words and the symbols that bound the scan and the cave.

#include <elf.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

[[noreturn]] inline void die(const std::string& message)
{
    std::fprintf(stderr, "ttnop: error: %s\n", message.c_str());
    std::exit(1);
}

struct Section
{
    Elf32_Shdr header {};
    std::string name;
};

struct Symbol
{
    std::uint32_t value = 0; // st_value -> the L1 virtual address where the symbol points
    std::uint32_t size  = 0; // st_size -> symbol size (0 for markers)
    std::uint8_t type   = 0; // ELF32_ST_TYPE (Eg. run_kernel is a STT_FUNC)
};

struct ElfImage
{
    std::vector<std::uint8_t> bytes;
    Elf32_Ehdr header {};
    std::vector<Section> sections;
    std::map<std::string, Symbol> symbols;
    int text_index = -1;

    const Section& text() const
    {
        return sections.at(static_cast<std::size_t>(text_index));
    }

    // .text words are addressed by link address, which is also the device L1 address.
    std::uint32_t word_at(std::uint32_t vaddr) const
    {
        const Elf32_Shdr& text_hdr         = text().header;
        const std::uint32_t text_start     = text_hdr.sh_addr;
        const std::uint32_t offset_in_text = vaddr - text_start;
        const std::uint32_t offset_in_elf  = text_hdr.sh_offset + offset_in_text;
        std::uint32_t out                  = 0;
        std::memcpy(&out, bytes.data() + offset_in_elf, sizeof(out));
        return out;
    }
};

// Read the whole ELF and parse sections (.text and the
// symbol tables) and symbols.
inline void load_elf(const std::string& path, ElfImage& elf)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        die("cannot open " + path);
    }
    elf.bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (elf.bytes.size() < sizeof(Elf32_Ehdr) || std::memcmp(elf.bytes.data(), ELFMAG, SELFMAG) != 0)
    {
        die(path + " is not an ELF file");
    }
    // Replace any previous parse so a reused ElfImage doesn't append stale sections/symbols.
    elf.header = {};
    elf.sections.clear();
    elf.symbols.clear();
    elf.text_index = -1;

    // copy the elf header bytes
    std::memcpy(&elf.header, elf.bytes.data(), sizeof(Elf32_Ehdr));

    // Section-header table: e_shoff is the file offset of entry 0
    // each entry is e_shentsize bytes.
    const std::uint32_t num_section_header   = elf.header.e_shnum;
    const std::uint32_t section_table_offset = elf.header.e_shoff;
    const std::uint32_t section_size         = elf.header.e_shentsize;
    elf.sections.reserve(num_section_header);

    // extract section headers from the elf
    for (std::uint32_t i = 0; i < num_section_header; ++i)
    {
        Section section;
        const std::uint8_t* section_address = elf.bytes.data() + section_table_offset + (i * section_size);
        std::memcpy(&section.header, section_address, sizeof(Elf32_Shdr));
        elf.sections.push_back(section);
    }

    // Resolve section names from the table named by e_shstrndx.
    // needed to extract .text from the ELF
    const Elf32_Shdr& shstrtab = elf.sections[elf.header.e_shstrndx].header;
    for (Section& section : elf.sections)
    {
        section.name = reinterpret_cast<const char*>(elf.bytes.data() + shstrtab.sh_offset + section.header.sh_name);
    }

    for (std::size_t i = 0; i < elf.sections.size(); ++i)
    {
        if (elf.sections[i].name == ".text")
        {
            elf.text_index = static_cast<int>(i); // record the index of .text
        }
    }
    if (elf.text_index < 0) // default val of test_index is -1
    {
        die("no .text section in " + path);
    }

    for (const Section& section : elf.sections)
    {
        if (section.header.sh_type != SHT_SYMTAB)
        {
            continue;
        }
        // sh_link is the section index of the string table (.strtab)
        const Elf32_Shdr& strtab  = elf.sections.at(section.header.sh_link).header;
        const std::uint32_t count = section.header.sh_size / sizeof(Elf32_Sym);

        // extract the symbols from the elf
        for (std::uint32_t i = 0; i < count; ++i)
        {
            Elf32_Sym sym {};
            std::memcpy(&sym, elf.bytes.data() + section.header.sh_offset + i * sizeof(Elf32_Sym), sizeof(sym));
            if (!sym.st_name) // skip is symbol is unnamed
            {
                continue;
            }
            // extract the string name -> used to look up run_kernel and _etext
            const std::string name = reinterpret_cast<const char*>(elf.bytes.data() + strtab.sh_offset + sym.st_name);
            if (!elf.symbols.count(name))
            {
                elf.symbols[name] = Symbol {sym.st_value, sym.st_size, ELF32_ST_TYPE(sym.st_info)};
            }
        }
    }
}

#endif
