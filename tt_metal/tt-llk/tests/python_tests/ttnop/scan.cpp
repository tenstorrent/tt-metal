// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// One pass over a kernel ELF's .text. Emits, as JSON on stdout:
//   - the kernel body range (where detour sites may live)
//   - the cave range (scratch space the injector may overwrite)
//   - which unpacker(s) the kernel actually loads with
//   - every candidate detour site
//
//   scan [--mode sync|all] <elf>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "elf32.h"
#include "tensix_isa.h"

namespace
{

struct Site
{
    std::uint32_t addr = 0;
    std::uint32_t word = 0;
    std::string op;
    bool sfpu = false;
};

struct Range
{
    std::uint32_t start = 0;
    std::uint32_t end   = 0;
    const char* source  = "none";

    bool valid() const
    {
        return end > start;
    }
};

// Optimisation folds the LLK helpers into one function, but the entry point still carries
// the name, so prefer it. The symbol is C++-mangled, hence the substring match.
Range find_body(const ElfImage& elf, const Range& cave)
{
    const Symbol* best = nullptr;
    for (const auto& [name, symbol] : elf.symbols)
    {
        if (symbol.type == STT_FUNC && symbol.size > 0 && name.find("run_kernel") != std::string::npos)
        {
            if (best == nullptr || symbol.size > best->size)
            {
                best = &symbol;
            }
        }
    }
    if (best != nullptr)
    {
        return {best->value, best->value + best->size, "run_kernel"};
    }

    const std::uint32_t text_start = elf.text().header.sh_addr;
    // Metal strips the entry symbol. An in-text cave is linked after all real code, so
    // everything ahead of it is kernel body.
    if (cave.valid() && cave.start > text_start)
    {
        return {text_start, cave.start, "text_to_cave"};
    }
    return {text_start, text_start + elf.text().header.sh_size, "text"};
}

// Two ways to get scratch space, one per runtime. Metal reserves it inside .text at link
// time so both detour jumps stay PC-relative through XIP. LLK links its kernel into a
// fixed 16K L1 region whose tail is unused, and .loader_init is the next section after
// that region, so its start doubles as the region bound.
Range find_cave(const ElfImage& elf)
{
    const auto reserved_start = elf.symbols.find("__kernel_cave_start");
    const auto reserved_end   = elf.symbols.find("__kernel_cave_end");
    if (reserved_start != elf.symbols.end() && reserved_end != elf.symbols.end())
    {
        return {reserved_start->second.value, reserved_end->second.value, "linker"};
    }

    const auto etext       = elf.symbols.find("_etext");
    const auto loader_init = elf.symbols.find("__loader_init_start");
    if (etext != elf.symbols.end() && loader_init != elf.symbols.end())
    {
        const std::uint32_t start = (etext->second.value + 15u) & ~15u;
        if (loader_init->second.value > start)
        {
            return {start, loader_init->second.value, "l1_tail"};
        }
    }
    return {};
}

std::string tensix_op_name(std::uint32_t opcode)
{
    if (const char* name = sync_op_name(opcode))
    {
        return name;
    }
    if (opcode == TT_OP_SFPNOP)
    {
        return "SFPNOP";
    }
    if (opcode == TT_OP_SETADCXX)
    {
        return "SETADCXX";
    }
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), is_sfpu_opcode(opcode) ? "SFPU_0x%02x" : "TTI_0x%02x", opcode);
    return buffer;
}

struct ScanResult
{
    std::vector<Site> sites;
    std::uint32_t unpacker_mask = 0;
};

ScanResult scan_text(const ElfImage& elf, const Range& body, bool all_instructions)
{
    ScanResult result;
    const Elf32_Shdr& text         = elf.text().header;
    const std::uint32_t text_start = text.sh_addr;
    const std::uint32_t text_end   = text_start + text.sh_size;
    // Sites only come from the body; word count is a hard upper bound on push_backs.
    if (body.valid())
    {
        result.sites.reserve((body.end - body.start) / 4u);
    }

    // Start from the top of .text rather than the body so the replay-payload counter is
    // already correct by the time the body begins.
    std::uint32_t replay_payload = 0;
    for (std::uint32_t vaddr = text_start; vaddr + 4u <= text_end; vaddr += 4u)
    {
        const std::uint32_t word = elf.word_at(vaddr);
        const bool tensix        = is_tensix_word(word);

        // A replay buffer being loaded holds Tensix words that are recorded, not executed.
        // The compiler can interleave RISC-V in there, so only Tensix words count down.
        if (replay_payload > 0)
        {
            replay_payload -= tensix ? 1u : 0u;
            continue;
        }

        const bool in_body = vaddr >= body.start && vaddr < body.end;

        if (!tensix)
        {
            if (all_instructions && in_body && is_relocatable_riscv(word))
            {
                char buffer[16];
                std::snprintf(buffer, sizeof(buffer), "RV32_0x%02x", word & RISCV_OPCODE_MASK);
                result.sites.push_back({vaddr, word, buffer, false});
            }
            continue;
        }

        const std::uint32_t op     = rotate_right_2(word);
        const std::uint32_t opcode = (op >> TT_OP_OPCODE_SHIFT) & TT_OP_OPCODE_MASK;
        const std::uint32_t params = op & TT_OP_PARAMS_MASK;

        if (opcode == TT_OP_REPLAY && (params & TT_REPLAY_LOAD_MODE))
        {
            replay_payload = (params >> TT_REPLAY_LEN_SHIFT) & TT_REPLAY_LEN_MASK;
            continue;
        }
        // The census covers all of .text: the SETADCXX that programs a datum count may be
        // hoisted out of the body by the optimiser.
        if (opcode == TT_OP_SETADCXX)
        {
            result.unpacker_mask |= (params >> TT_SETADCXX_CNTSETMASK_SHIFT) & TT_SETADCXX_UNP_MASK;
        }

        if (!in_body || !is_detourable_tensix(opcode))
        {
            continue;
        }

        const bool sfpu = is_sfpu_opcode(opcode) || (opcode == TT_OP_STALLWAIT && stallwait_touches_sfpu(params));
        if (all_instructions || sync_op_name(opcode) != nullptr)
        {
            result.sites.push_back({vaddr, word, tensix_op_name(opcode), sfpu});
        }
    }
    return result;
}

void emit_json(const std::string& path, const ElfImage& elf, const Range& body, const Range& cave, const ScanResult& scanned, const char* mode)
{
    const Elf32_Shdr& text = elf.text().header;
    std::printf("{\n");
    std::printf("  \"elf\": \"%s\",\n", path.c_str());
    std::printf("  \"mode\": \"%s\",\n", mode);
    std::printf("  \"text\": {\"start\": %u, \"end\": %u},\n", text.sh_addr, text.sh_addr + text.sh_size);
    std::printf("  \"body\": {\"start\": %u, \"end\": %u, \"source\": \"%s\"},\n", body.start, body.end, body.source);
    if (cave.valid())
    {
        std::printf("  \"cave\": {\"start\": %u, \"limit\": %u, \"source\": \"%s\"},\n", cave.start, cave.end, cave.source);
    }
    else
    {
        std::printf("  \"cave\": null,\n");
    }
    std::printf("  \"unpacker_mask\": %u,\n", scanned.unpacker_mask);
    // Published so the host never has to keep its own copy of these encodings.
    std::printf(
        "  \"fillers\": {\"tti_nop\": %u, \"sfpnop\": %u, \"unpacr0\": %u, \"unpacr1\": %u},\n", FILLER_TTI_NOP, FILLER_SFPNOP, FILLER_UNPACR0, FILLER_UNPACR1);
    std::printf("  \"sites\": [");
    for (std::size_t i = 0; i < scanned.sites.size(); ++i)
    {
        const Site& site = scanned.sites[i];
        std::printf(
            "%s\n    {\"index\": %zu, \"addr\": %u, \"word\": %u, \"op\": \"%s\", \"sfpu\": %s}",
            i ? "," : "",
            i,
            site.addr,
            site.word,
            site.op.c_str(),
            site.sfpu ? "true" : "false");
    }
    std::printf("%s]\n}\n", scanned.sites.empty() ? "" : "\n  ");
}

} // namespace

int main(int argc, char** argv)
{
    const char* mode = "sync";
    const char* path = nullptr;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
        {
            mode = argv[++i];
        }
        else
        {
            path = argv[i];
        }
    }
    if (path == nullptr || (std::strcmp(mode, "sync") != 0 && std::strcmp(mode, "all") != 0))
    {
        std::fprintf(stderr, "usage: scan [--mode sync|all] <elf>\n");
        return 2;
    }

    ElfImage elf;
    load_elf(path, elf);

    const Range cave      = find_cave(elf);
    const Range body      = find_body(elf, cave);
    const ScanResult scan = scan_text(elf, body, std::strcmp(mode, "all") == 0);

    if (cave.valid() && cave.start >= body.start && cave.start < body.end)
    {
        die("cave overlaps the kernel body; detouring would corrupt live code");
    }
    emit_json(path, elf, body, cave, scan, mode);
    return 0;
}
