#include "common/assert.hpp"
#include "graph_lib/common.hpp"

namespace tt {

static bool contains(std::string const &str, std::string const &substr) {
    return str.find(substr) != std::string::npos;
}

std::ostream &operator<<(std::ostream &os, const BudaName &name)
{
    bool needs_quotes = contains(name.name, " ") or contains(name.name, "/");
    if (needs_quotes)
        os << "\"";
    os << name.name;
    if (needs_quotes)
        os << "\"";
    return os;
}

std::ostream &operator<<(std::ostream &os, const BudaBlocks &bb) {
    TT_ASSERT(bb.z > 0);
    TT_ASSERT(bb.mblock_m > 0);
    TT_ASSERT(bb.mblock_n > 0);
    TT_ASSERT(bb.ublock_rt > 0);
    TT_ASSERT(bb.ublock_ct > 0);
    os << "t: " << bb.z << ", ";
    os << "mblock: [" << bb.mblock_m << ", " << bb.mblock_n << "], ";
    os << "ublock: [" << bb.ublock_rt << ", " << bb.ublock_ct << "]";
    return os;
}

std::ostream &operator<<(std::ostream &os, const BudaOpAttr &attr) {
    if (const bool *v = std::get_if<bool>(&attr)) {
        os << (*v ? "true" : "false");
    } else if (const int *v = std::get_if<int>(&attr)) {
        os << *v;
    } else if (const float *v = std::get_if<float>(&attr)) {
        os << std::scientific << *v;
    } else if (const std::string *v = std::get_if<std::string>(&attr)) {
        os << *v;
    }
    else if (const std::vector<int> *v = std::get_if<std::vector<int>>(&attr))
    {
        bool first = true;
        os << "[";
        for (int i : *v)
        {
            if (not first)
                os << ", ";
            os << i;
            first = false;
        }
        os << "]";
    }
    else
    {
        TT_ASSERT(false, "Unhandled variant type for BudaOpAttr");
    }

    return os;
}

std::ostream& operator<<(std::ostream &os, const DataFormat &format) {
    switch (format) {
        case DataFormat::Bfp2: os << "Bfp2"; break;
        case DataFormat::Bfp2_b: os << "Bfp2_b"; break;
        case DataFormat::Bfp4: os << "Bfp4"; break;
        case DataFormat::Bfp4_b: os << "Bfp4_b"; break;
        case DataFormat::Bfp8: os << "Bfp8"; break;
        case DataFormat::Bfp8_b: os << "Bfp8_b"; break;
        case DataFormat::Float16: os << "Float16"; break;
        case DataFormat::Float16_b: os << "Float16_b"; break;
        case DataFormat::Float32: os << "Float32"; break;
        case DataFormat::Int8: os << "Int8"; break;
        case DataFormat::Lf8: os << "Lf8"; break;
        case DataFormat::UInt16: os << "UInt16"; break;
        case DataFormat::RawUInt8: os << "RawUInt8"; break;
        case DataFormat::RawUInt16: os << "RawUInt16"; break;
        case DataFormat::RawUInt32: os << "RawUInt32"; break;
        case DataFormat::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown format");
    }
    return os;
}

std::ostream& operator<<(std::ostream &os, const MathFidelity &fidelity) {
    switch (fidelity) {
        case MathFidelity::LoFi: os << "LoFi"; break;
        case MathFidelity::HiFi2: os << "HiFi2"; break;
        case MathFidelity::HiFi3: os << "HiFi3"; break;
        case MathFidelity::HiFi4: os << "HiFi4"; break;
        case MathFidelity::Invalid: os << "Invalid"; break;
        default: throw std::invalid_argument("Unknown fidelity");
    }
    return os;
}

std::uint32_t data_format_byte_size(DataFormat df, int elements)
{
    switch (df) {
        case DataFormat::Float32: return 4 * elements;
        case DataFormat::UInt16:
        case DataFormat::Float16_b:
        case DataFormat::Float16: return 2 * elements;
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp8: return (elements + elements/16); 
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp4: return (elements / 2 + elements / 16);
        case DataFormat::Bfp2_b:
        case DataFormat::Bfp2: return (elements / 4 + elements / 16);
        case DataFormat::Lf8:
        case DataFormat::Int8: return elements;
        case DataFormat::RawUInt8: return elements;
        case DataFormat::RawUInt16: return 2 * elements;
        case DataFormat::RawUInt32: return 4 * elements;
        case DataFormat::Invalid: return 0;
    }
    throw std::runtime_error("Invalid format");

}

}

