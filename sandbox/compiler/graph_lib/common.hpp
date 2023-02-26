#pragma once

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace tt {

struct BudaBlocks {
    int z;
    int ublock_rt, ublock_ct;
    int mblock_m, mblock_n;
};

struct BudaName
{
    std::string name;
    BudaName(std::string const &name) : name(name) {}
};

using BudaOpAttr = ::std::variant<bool, int, float, std::string, std::vector<int>>;
using BudaOpAttrs = ::std::map<std::string, BudaOpAttr>;

std::ostream &operator<<(std::ostream &os, const BudaName &name);
std::ostream &operator<<(std::ostream &os, const BudaBlocks &bb);
std::ostream &operator<<(std::ostream &os, const BudaOpAttr &attr);

enum class DataFormat : std::uint8_t
{
    Float32   = 0,
    Float16   = 1,
    Bfp8      = 2,
    Bfp4      = 3,
    Bfp2      = 11,
    Float16_b = 5,
    Bfp8_b    = 6,
    Bfp4_b    = 7,
    Bfp2_b    = 15,
    Lf8       = 10,
    UInt16    = 12,
    Int8      = 14,
    RawUInt8  = 0xf0,
    RawUInt16 = 0xf1,
    RawUInt32 = 0xf2,
    Invalid   = 0xff
};

enum class MathFidelity : uint8_t
{
    LoFi          = 0,
    HiFi2         = 2,
    HiFi3         = 3,
    HiFi4         = 4,
    Invalid       = 0xff,
};

std::uint32_t data_format_byte_size(DataFormat df, int elements = 1);

std::ostream &operator<<(std::ostream &os, DataFormat const &df);
std::ostream &operator<<(std::ostream &os, MathFidelity const &mf);

struct PytorchTensorDesc
{
    const void* ptr;
    std::uint32_t itemsize;
    DataFormat format;
    std::array<std::uint32_t, 4> shape;   // outer-most dimension first
    std::array<std::uint32_t, 4> strides; // outer-most dimension first, in bytes
};

}  // namespace tt
