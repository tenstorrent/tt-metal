#pragma once

#include "ll_buda/tensor/tensor.hpp"

namespace tt {

namespace ll_buda {

// TODO: Accept parallelization

Tensor tilize (const Tensor &a);
}  // namespace ll_buda

}  // namespace tt
