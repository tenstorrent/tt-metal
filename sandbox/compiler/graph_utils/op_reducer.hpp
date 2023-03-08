#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cassert>

enum class ReducedOpType {
    Input,
    Output,
    Constant,
    Convolution,
    Matmul,
    EltwiseBinary,
    Sfpu,
    Other
};

constexpr const char* reduced_op_type_to_string(ReducedOpType e) throw()
{
    switch (e)
    {
        case ReducedOpType::Input: return "Input";
        case ReducedOpType::Output: return "Output";
        case ReducedOpType::Constant: return "Constant";
        case ReducedOpType::Convolution: return "Convolution";
        case ReducedOpType::Matmul: return "Matmul";
        case ReducedOpType::EltwiseBinary: return "EltwiseBinary";
        case ReducedOpType::Sfpu: return "Sfpu";
        case ReducedOpType::Other: return "Other";
        default: assert(false);
    }
};

ReducedOpType reduce_pybuda_op(std::string pybuda_op);
