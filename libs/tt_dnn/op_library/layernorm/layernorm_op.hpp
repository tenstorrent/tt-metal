#pragma once

#include "libs/tensor/tensor.hpp"

namespace tt { namespace tt_metal {

Tensor layernorm(const Tensor &a, float eps, bool out_dram);
Tensor layernorm_gamma(const Tensor &a, float eps, const Tensor& gamma, bool out_dram);
Tensor layernorm_gamma_beta(const Tensor &a, float eps, const Tensor& gamma, const Tensor& beta, bool out_dram);

// computes layernorm(a+b)*gamma+beta
Tensor add_layernorm_gamma_beta(const Tensor& a, const Tensor &b, float eps, const Tensor& gamma, const Tensor& beta, bool out_dram);

} }  // namespace tt::tt_metal
