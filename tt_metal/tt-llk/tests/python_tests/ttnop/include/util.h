#ifndef UTIL_H
#define UTIL_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

// basic utility functions

[[noreturn]] static void die(const std::string& m)
{
    std::fprintf(stderr, "ttnop: error: %s\n", m.c_str());
    std::exit(1);
}

static void warn(const std::string& m)
{
    std::fprintf(stderr, "ttnop: warning: %s\n", m.c_str());
}

static bool host_is_le()
{
    std::uint32_t x = 1;
    return *reinterpret_cast<std::uint8_t*>(&x) == 1;
}

// Parse a C-style integer ("0x.." hex or decimal).
static bool parse_u32(const std::string& s, std::uint32_t& out)
{
    if (s.empty())
    {
        return false;
    }
    errno           = 0;
    char* end       = nullptr;
    unsigned long v = std::strtoul(s.c_str(), &end, 0);
    if (errno || !end || *end != '\0')
    {
        return false;
    }
    out = static_cast<std::uint32_t>(v);
    return true;
}

// RV32 machine code

constexpr std::uint32_t OP_LOAD    = 0x03;
constexpr std::uint32_t OP_OPIMM   = 0x13;
constexpr std::uint32_t OP_AUIPC   = 0x17;
constexpr std::uint32_t OP_STORE   = 0x23;
constexpr std::uint32_t OP_AMO     = 0x2f;
constexpr std::uint32_t OP_OP      = 0x33;
constexpr std::uint32_t OP_LUI     = 0x37;
constexpr std::uint32_t OP_BRANCH  = 0x63;
constexpr std::uint32_t OP_JALR    = 0x67;
constexpr std::uint32_t OP_JAL     = 0x6f;
constexpr std::uint32_t OP_SYSTEM  = 0x73;
constexpr std::uint32_t OP_MISCMEM = 0x0f; // fence

constexpr std::uint32_t NOP    = 0x00000013; // addi x0,x0,0
constexpr std::uint32_t REG_SP = 2, REG_T0 = 5;

std::uint32_t opcode(std::uint32_t insn)
{
    return insn & 0x7f;
}

std::uint32_t rd(std::uint32_t insn)
{
    return (insn >> 7) & 0x1f;
}

std::uint32_t funct3(std::uint32_t insn)
{
    return (insn >> 12) & 0x7;
}

std::uint32_t rs1(std::uint32_t insn)
{
    return (insn >> 15) & 0x1f;
}

std::uint32_t rs2(std::uint32_t insn)
{
    return (insn >> 20) & 0x1f;
}

std::int32_t sext(std::uint32_t v, int bits)
{
    std::uint32_t m = 1u << (bits - 1);
    return static_cast<std::int32_t>((v ^ m) - m);
}

// J-type immediate (JAL): imm[20|10:1|11|19:12], byte offset, multiple of 2.
std::int32_t jal_imm(std::uint32_t insn)
{
    std::uint32_t i = (((insn >> 31) & 1) << 20) | (((insn >> 21) & 0x3ff) << 1) | (((insn >> 20) & 1) << 11) | (((insn >> 12) & 0xff) << 12);
    return sext(i, 21);
}

std::uint32_t enc_jal_imm(std::int32_t off)
{
    std::uint32_t o = static_cast<std::uint32_t>(off);
    return (((o >> 20) & 1) << 31) | (((o >> 1) & 0x3ff) << 21) | (((o >> 11) & 1) << 20) | (((o >> 12) & 0xff) << 12);
}

// B-type immediate (BRANCH): imm[12|10:5|4:1|11], byte offset, multiple of 2.
std::int32_t branch_imm(std::uint32_t insn)
{
    std::uint32_t i = (((insn >> 31) & 1) << 12) | (((insn >> 25) & 0x3f) << 5) | (((insn >> 8) & 0xf) << 1) | (((insn >> 7) & 1) << 11);
    return sext(i, 13);
}

std::uint32_t enc_branch_imm(std::int32_t off)
{
    std::uint32_t o = static_cast<std::uint32_t>(off);
    return (((o >> 12) & 1) << 31) | (((o >> 5) & 0x3f) << 25) | (((o >> 1) & 0xf) << 8) | (((o >> 11) & 1) << 7);
}

// jal rd, (to - from). Range checked by caller.
std::uint32_t make_jal(std::uint32_t reg, std::int64_t from, std::int64_t to)
{
    return OP_JAL | (reg << 7) | enc_jal_imm(static_cast<std::int32_t>(to - from));
}

static bool jal_in_range(std::int64_t from, std::int64_t to)
{
    std::int64_t d = to - from;
    return d >= -(1 << 20) && d < (1 << 20);
}

#endif
