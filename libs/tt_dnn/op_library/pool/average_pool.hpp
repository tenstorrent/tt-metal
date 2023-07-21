#pragma once

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"


namespace tt {
namespace tt_metal {

struct PoolType {
    enum Enum { AVG = 0 };
    static const vector<Enum> all() { return { AVG }; }
};

Tensor average_pool_2d(const Tensor& input);

}  // namespace tt_metal
}  // namespace tt
