// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// How a bound TensorParameter's TensorSpec feeds the kernel's JIT cache key (compute_hash).

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::ProgramSpecTestGen1;

// ============================================================================
// Kernel hash sensitivity to TensorParameter spec
// ============================================================================
//
// The kernel's JIT cache key is its compute_hash(); two kernels that hash equal share a cached
// binary. The TensorParameter's TensorSpec flows into the kernel's positional CTAs (via
// ResolveTensorParameterStaticCTAs), and compile_time_args_ is part of the hash, so:
//   - same kernel source + different TensorSpec  =>  different hash  =>  forced recompile
//   - same kernel source + identical TensorSpec  =>  same hash       =>  cache reuse
// These are regression canaries for the cache-key chain. If a future refactor drops a field from
// the CTA payload, or changes compute_hash to skip compile_time_args_, the silent failure mode is
// a stale cached binary running with mismatched layout metadata — nasty to debug in prod.

TEST_F(ProgramSpecTestGen1, CPU_DifferentTensorSpecProducesDifferentKernelHash) {
    // Same kernel source, same accessor name. The two TensorParameters differ only in BufferType
    // (DRAM vs L1), which flips the is_dram bit in the binding's args_config CTA word.
    auto make_spec = [](tt::tt_metal::BufferType buffer_type) {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor", buffer_type)};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };
    Program prog_dram = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::BufferType::DRAM));
    Program prog_l1 = MakeProgramFromSpec(*mesh_device_, make_spec(tt::tt_metal::BufferType::L1));

    auto hash_dram = prog_dram.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_l1 = prog_l1.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_NE(hash_dram, hash_l1);
}

TEST_F(ProgramSpecTestGen1, CPU_IdenticalTensorSpecProducesIdenticalKernelHash) {
    // Determinism canary: two specs constructed identically must hash identically. If this ever
    // fails, something nondeterministic crept into the hash (iteration order over a hash map,
    // pointer values, timestamps, etc.).
    auto make_spec = [] {
        ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
        spec.tensor_parameters = {MakeMinimalTensorParameter("input_tensor")};
        BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");
        return spec;
    };
    Program prog_a = MakeProgramFromSpec(*mesh_device_, make_spec());
    Program prog_b = MakeProgramFromSpec(*mesh_device_, make_spec());

    auto hash_a = prog_a.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_b = prog_b.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_EQ(hash_a, hash_b);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
