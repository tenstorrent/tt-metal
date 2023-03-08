#include "op_reducer.hpp"
#include <map>

std::map<std::string, ReducedOpType> pybuda_op_map {
    {"matmul", ReducedOpType::Matmul},
    {"convolution", ReducedOpType::Convolution},
    {"input::parameter", ReducedOpType::Input},
    {"constant", ReducedOpType::Constant},
    {"output", ReducedOpType::Output},
    {"relu", ReducedOpType::Sfpu},
    {"add", ReducedOpType::EltwiseBinary},
    {"mul", ReducedOpType::EltwiseBinary},
    {"sub", ReducedOpType::EltwiseBinary},
    {"div", ReducedOpType::EltwiseBinary},
};

ReducedOpType reduce_pybuda_op(std::string pybuda_op) {
    if(pybuda_op_map.find(pybuda_op) != pybuda_op_map.end()) {
        return pybuda_op_map[pybuda_op];
    }
    else {
        return ReducedOpType::Other;
    }
}
