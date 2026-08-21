// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "quasar_test_common.h"

static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "dest_sync_chain only supports the L1_TO_L1 run type");

constexpr std::uint32_t chain_num_l1_iterations()
{
    std::uint32_t count = 0;
    for (std::uint32_t i = 0; i < CHAIN_LENGTH; i++)
    {
        if (CHAIN_OPS[i] == ChainOp::PACK)
        {
            count++;
        }
    }
    return count;
}

constexpr std::uint32_t NUM_L1_ITERATIONS = chain_num_l1_iterations();

constexpr std::uint32_t iteration_begin(const std::uint32_t iteration)
{
    std::uint32_t begin = 0;
    for (std::uint32_t seen = 0; seen < iteration; begin++)
    {
        if (CHAIN_OPS[begin] == ChainOp::PACK)
        {
            seen++;
        }
    }
    return begin;
}

constexpr ChainOp iteration_producer(const std::uint32_t iteration)
{
    return CHAIN_OPS[iteration_begin(iteration)];
}

constexpr std::uint32_t iteration_sfpu_count(const std::uint32_t iteration)
{
    std::uint32_t count = 0;
    for (std::uint32_t i = iteration_begin(iteration); CHAIN_OPS[i] != ChainOp::PACK; i++)
    {
        if (CHAIN_OPS[i] == ChainOp::SFPU)
        {
            count++;
        }
    }
    return count;
}

constexpr bool iteration_has_sfpu(const std::uint32_t iteration)
{
    return iteration_sfpu_count(iteration) > 0;
}

constexpr bool chain_has(const ChainOp op)
{
    for (std::uint32_t i = 0; i < CHAIN_LENGTH; i++)
    {
        if (CHAIN_OPS[i] == op)
        {
            return true;
        }
    }
    return false;
}

constexpr std::uint32_t DEST_SECTION_REPEATS = 256;

constexpr bool CHAIN_HAS_UNPACK_TO_DEST = chain_has(ChainOp::UNPACK);
constexpr bool CHAIN_HAS_FPU            = chain_has(ChainOp::FPU);
constexpr bool CHAIN_HAS_SFPU           = chain_has(ChainOp::SFPU);
constexpr bool MIXED_PRODUCERS          = CHAIN_HAS_UNPACK_TO_DEST && CHAIN_HAS_FPU;

constexpr bool uniform_sfpu_participation()
{
    for (std::uint32_t i = 0; i < NUM_L1_ITERATIONS; i++)
    {
        if (iteration_has_sfpu(i) != iteration_has_sfpu(0))
        {
            return false;
        }
    }
    return true;
}

constexpr bool UNIFORM_DVALID_CHAIN = !MIXED_PRODUCERS && uniform_sfpu_participation();

template <ckernel::dest_dvalid_client CLIENT>
inline void configure_iteration_dvalid_chain(const ChainOp producer, const bool has_sfpu)
{
    using ckernel::dest_dvalid_client;
    if (producer == ChainOp::UNPACK)
    {
        if (has_sfpu)
        {
            ckernel::set_up_dest_dvalid_per_thread<CLIENT>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        }
        else
        {
            ckernel::set_up_dest_dvalid_per_thread<CLIENT>({dest_dvalid_client::UNPACK, dest_dvalid_client::PACK});
        }
    }
    else
    {
        if (has_sfpu)
        {
            ckernel::set_up_dest_dvalid_per_thread<CLIENT>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        }
        else
        {
            ckernel::set_up_dest_dvalid_per_thread<CLIENT>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});
        }
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_bfd_alloc.h"
#include "llk_math_common.h"
#include "llk_sync.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"

inline void program_unpack_mop(const ChainOp producer, const std::uint32_t buf_desc_id, const ckernel::TensorShape& tensor_shape)
{
    if (producer == ChainOp::UNPACK)
    {
        if constexpr (USE_DVALID_SCHEME)
        {
            _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false, is_fp32_dest_acc_en>(buf_desc_id, tensor_shape, 1);
        }
        else
        {
            _llk_unpack_unary_operand_init_<p_unpacr::UNP_DEST, false, is_fp32_dest_acc_en, EltwiseBinaryReuseDestType::NONE, true>(
                buf_desc_id, tensor_shape, 1);
        }
    }
    else
    {
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, is_fp32_dest_acc_en>(buf_desc_id, tensor_shape, 1);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces);

    {
        ZONE_SCOPED("INIT")
        if constexpr (CHAIN_HAS_UNPACK_TO_DEST)
        {
            _llk_math_upk_to_dest_hw_configure_<false, is_fp32_dest_acc_en, false>();
        }

        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Unp0>(tensor_shape, L1_ADDRESS(buffer_A[0]), formats.unpack_A_src);
        _llk_unpack_configure_unary_<p_unpacr::UNP_A>(static_cast<DataFormat>(formats.unpack_A_dst));

        if constexpr (!MIXED_PRODUCERS)
        {
            program_unpack_mop(iteration_producer(0), ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape);
        }

        if constexpr (USE_DVALID_SCHEME)
        {
            if constexpr (UNIFORM_DVALID_CHAIN)
            {
                configure_iteration_dvalid_chain<ckernel::dest_dvalid_client::UNPACK>(iteration_producer(0), iteration_has_sfpu(0));
            }
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_unpack();
            if constexpr (CHAIN_HAS_UNPACK_TO_DEST)
            {
                _llk_sync_init_(semaphore::UNPACK_MATH, dest_sync == ckernel::DstSync::SyncHalf ? 2 : 1, 0);
            }
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        {
            for (std::uint32_t iteration = 0; iteration < NUM_L1_ITERATIONS; iteration++)
            {
                const ChainOp producer = iteration_producer(iteration);

                if constexpr (MIXED_PRODUCERS)
                {
                    program_unpack_mop(producer, ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Unp0>(), tensor_shape);
                }

                if constexpr (USE_DVALID_SCHEME && !UNIFORM_DVALID_CHAIN)
                {
                    configure_iteration_dvalid_chain<ckernel::dest_dvalid_client::UNPACK>(producer, iteration_has_sfpu(iteration));
                }

                for (std::uint32_t rep = 0; rep < DEST_SECTION_REPEATS; rep++)
                {
                    if (producer == ChainOp::UNPACK)
                    {
                        if constexpr (USE_DVALID_SCHEME)
                        {
                            _llk_unpack_unary_operand_<p_unpacr::UNP_DEST>(0, tensor_shape);
                            _llk_unpack_dest_dvalid_section_done_<dest_sync>();
                        }
                        else
                        {
                            _llk_unpack_unary_operand_<p_unpacr::UNP_DEST, EltwiseBinaryReuseDestType::NONE, true, dest_sync>(0, tensor_shape);
                        }
                    }
                    else
                    {
                        _llk_unpack_unary_operand_<p_unpacr::UNP_A>(0, tensor_shape);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sync.h"
#include "sfpu_operations_quasar.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    [[maybe_unused]] const DataFormat math_format = static_cast<DataFormat>(formats.math);

    {
        ZONE_SCOPED("INIT")
        if constexpr (CHAIN_HAS_FPU || CHAIN_HAS_SFPU)
        {
            _llk_math_srcAB_hw_configure_<true, is_fp32_dest_acc_en, false>(math_format, math_format);
        }

        if constexpr (CHAIN_HAS_FPU)
        {
            _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en>(params.num_faces * params.TEST_FACE_R_DIM, 1);
        }
        if constexpr (CHAIN_HAS_SFPU)
        {
            _llk_math_eltwise_sfpu_init_();
            test_utils::init_unary_sfpu_operation_quasar<SfpuType::abs, is_fp32_dest_acc_en, false>();
        }

        if constexpr (USE_DVALID_SCHEME)
        {
            if constexpr (UNIFORM_DVALID_CHAIN)
            {
                if constexpr (CHAIN_HAS_FPU)
                {
                    configure_iteration_dvalid_chain<dest_dvalid_client::FPU>(iteration_producer(0), iteration_has_sfpu(0));
                }
                if constexpr (CHAIN_HAS_SFPU)
                {
                    configure_iteration_dvalid_chain<dest_dvalid_client::SFPU>(iteration_producer(0), iteration_has_sfpu(0));
                }
            }
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_math();
            set_up_zero_dest_dvalid_handshake_for_sfpu();
            _llk_math_pack_sync_init_<dest_sync>();
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        {
            for (std::uint32_t iteration = 0; iteration < NUM_L1_ITERATIONS; iteration++)
            {
                const ChainOp producer       = iteration_producer(iteration);
                const bool has_sfpu          = iteration_has_sfpu(iteration);
                const std::uint32_t sfpu_ops = iteration_sfpu_count(iteration);

                if constexpr (USE_DVALID_SCHEME && !UNIFORM_DVALID_CHAIN)
                {
                    if constexpr (CHAIN_HAS_FPU)
                    {
                        configure_iteration_dvalid_chain<dest_dvalid_client::FPU>(producer, has_sfpu);
                    }
                    if constexpr (CHAIN_HAS_SFPU)
                    {
                        configure_iteration_dvalid_chain<dest_dvalid_client::SFPU>(producer, has_sfpu);
                    }
                }

                for (std::uint32_t rep = 0; rep < DEST_SECTION_REPEATS; rep++)
                {
                    if constexpr (!USE_DVALID_SCHEME)
                    {
                        _llk_math_wait_for_dest_available_();
                    }

                    if (producer == ChainOp::FPU)
                    {
                        _llk_math_eltwise_unary_datacopy_(0);
                        if constexpr (USE_DVALID_SCHEME)
                        {
                            _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
                        }
                    }
                    else if constexpr (!USE_DVALID_SCHEME)
                    {
                        _llk_sync_wait_<p_stall::STALL_SYNC, p_stall::STALL_ON_ZERO>(semaphore::UNPACK_MATH);
                        _llk_sync_get_<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::UNPACK_MATH);
                    }

                    for (std::uint32_t sfpu_op = 0; sfpu_op < sfpu_ops; sfpu_op++)
                    {
                        test_utils::call_unary_sfpu_operation_quasar<SfpuType::abs, dest_sync, is_fp32_dest_acc_en, false, 32>(
                            0, static_cast<DataFormat>(formats.math));
                    }

                    if constexpr (USE_DVALID_SCHEME)
                    {
                        if (has_sfpu)
                        {
                            _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();
                        }
                    }
                    else
                    {
                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_ISOLATE_SFPU

void run_kernel(RUNTIME_PARAMETERS params)
{
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_bfd_alloc.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_Res = params.buffer_Res;
#endif
    const ckernel::TensorShape tensor_shape = ckernel::tensor_shape_from_num_faces(params.TEST_FACE_R_DIM, params.num_faces);

    {
        ZONE_SCOPED("INIT")
        ckernel::trisc::bfd_alloc_and_program<ckernel::trisc::BfdResource::Pack0>(tensor_shape, L1_ADDRESS(buffer_Res[0]), formats.pack_dst);
        _llk_pack_hw_configure_<p_pacr::PACK0, is_fp32_dest_acc_en>(static_cast<DataFormat>(formats.pack_src), ckernel::ReluConfig::none());
        _llk_pack_init_(ckernel::trisc::bfd_current<ckernel::trisc::BfdResource::Pack0>(), tensor_shape, 1);

        if constexpr (USE_DVALID_SCHEME)
        {
            if constexpr (UNIFORM_DVALID_CHAIN)
            {
                configure_iteration_dvalid_chain<ckernel::dest_dvalid_client::PACK>(iteration_producer(0), iteration_has_sfpu(0));
            }
        }
        else
        {
            set_up_zero_dest_dvalid_handshake_for_pack();
            _llk_pack_dest_init_<p_pacr::PACK0, dest_sync>();
        }
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        {
            for (std::uint32_t iteration = 0; iteration < NUM_L1_ITERATIONS; iteration++)
            {
                if constexpr (USE_DVALID_SCHEME && !UNIFORM_DVALID_CHAIN)
                {
                    configure_iteration_dvalid_chain<ckernel::dest_dvalid_client::PACK>(iteration_producer(iteration), iteration_has_sfpu(iteration));
                }

                for (std::uint32_t rep = 0; rep < DEST_SECTION_REPEATS; rep++)
                {
                    if constexpr (!USE_DVALID_SCHEME)
                    {
                        _llk_packer_wait_for_math_done_();
                    }

                    _llk_pack_(0, 0, tensor_shape);

                    if constexpr (USE_DVALID_SCHEME)
                    {
                        _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                    else
                    {
                        _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, dest_sync, is_fp32_dest_acc_en>();
                    }
                    ckernel::wait_pack_idle();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
