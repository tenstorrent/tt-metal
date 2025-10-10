// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compile_time_args.h"

enum class FabricChannelPoolType {
    STATIC = 0,
    ELASTIC = 1,
};


template <typename DERIVED>
struct CtArgConsumer {
    static constexpr size_t NUM_ARGS_USED = DERIVED::GET_NUM_ARGS_CONSUMED();
};

template<size_t CT_ARG_IDX_BASE>
struct StaticChannelPool : public CtArgConsumer<StaticChannelPool<CT_ARG_IDX_BASE>> {
    static constexpr size_t base_address = get_compile_time_arg_val(CT_ARG_IDX_BASE);

    static constexpr size_t GET_NUM_ARGS_CONSUMED() {
        return 1;
    }
};

template <size_t CT_ARG_IDX_BASE>
struct ElasticChannelPool : public CtArgConsumer<ElasticChannelPool<CT_ARG_IDX_BASE>> {
    static constexpr size_t num_chunks = get_compile_time_arg_val(CT_ARG_IDX_BASE);
    static constexpr size_t num_slots_per_chunk = get_compile_time_arg_val(CT_ARG_IDX_BASE + 1);
    
    static constexpr size_t ARGS_PER_CHUNK = 1;
    static constexpr size_t CHUNKS_CT_ARGS_IDX_BASE = CT_ARG_IDX_BASE + 2;

    static constexpr std::array<size_t, num_chunks> chunk_base_addresses = fill_array_with_next_n_args<size_t, CHUNKS_CT_ARGS_IDX_BASE, num_chunks>();

    static constexpr size_t GET_NUM_ARGS_CONSUMED() {
        return 2 + num_chunks * ARGS_PER_CHUNK;
    }
};

// Helper to build pools tuple recursively - uses array of pool types
template<size_t CT_ARG_IDX, size_t NumPools, typename PoolTypesArray, size_t... Indices>
struct PoolsBuilderImpl;

template<size_t CT_ARG_IDX, size_t NumPools, typename PoolTypesArray, size_t CurrentIdx, size_t... RestIndices>
struct PoolsBuilderImpl<CT_ARG_IDX, NumPools, PoolTypesArray, CurrentIdx, RestIndices...> {
    static constexpr size_t current_pool_type_val = PoolTypesArray::types[CurrentIdx];
    static constexpr FabricChannelPoolType current_pool_type = 
        static_cast<FabricChannelPoolType>(current_pool_type_val);
    
    // Determine current pool type
    using CurrentPoolType = typename std::conditional<
        current_pool_type_val == static_cast<size_t>(FabricChannelPoolType::STATIC),
        StaticChannelPool<CT_ARG_IDX>,
        ElasticChannelPool<CT_ARG_IDX>
    >::type;
    
    static constexpr size_t next_ct_arg_idx = CT_ARG_IDX + CurrentPoolType::GET_NUM_ARGS_CONSUMED();
    
    using RestOfPools = typename PoolsBuilderImpl<next_ct_arg_idx, NumPools, PoolTypesArray, RestIndices...>::type;
    
    using type = decltype(std::tuple_cat(
        std::declval<std::tuple<CurrentPoolType>>(),
        std::declval<RestOfPools>()
    ));
    
    static constexpr size_t final_ct_arg_idx = RestOfPools::final_ct_arg_idx;
};

// Base case: no more pools to build
template<size_t CT_ARG_IDX, size_t NumPools, typename PoolTypesArray>
struct PoolsBuilderImpl<CT_ARG_IDX, NumPools, PoolTypesArray> {
    using type = std::tuple<>;
    static constexpr size_t final_ct_arg_idx = CT_ARG_IDX;
};

// Index sequence generator
template<size_t... Is>
struct index_sequence {};

template<size_t N, size_t... Is>
struct make_index_sequence_impl : make_index_sequence_impl<N-1, N-1, Is...> {};

template<size_t... Is>
struct make_index_sequence_impl<0, Is...> {
    using type = index_sequence<Is...>;
};

template<size_t N>
using make_index_sequence = typename make_index_sequence_impl<N>::type;

// Wrapper that holds the pool types array
template<size_t N, size_t TypesBaseIdx>
struct PoolTypesHolder {
    static constexpr std::array<size_t, N> types = fill_array_with_next_n_args<size_t, TypesBaseIdx, N>();
};

// Main PoolsBuilder that uses index sequence
template<size_t CT_ARG_IDX, size_t NumPools, size_t TypesBaseIdx, typename Indices>
struct PoolsBuilder;

template<size_t CT_ARG_IDX, size_t NumPools, size_t TypesBaseIdx, size_t... Indices>
struct PoolsBuilder<CT_ARG_IDX, NumPools, TypesBaseIdx, index_sequence<Indices...>> {
    using PoolTypes = PoolTypesHolder<NumPools, TypesBaseIdx>;
    using Impl = PoolsBuilderImpl<CT_ARG_IDX, NumPools, PoolTypes, Indices...>;
    using type = typename Impl::type;
    static constexpr size_t final_ct_arg_idx = Impl::final_ct_arg_idx;
};

template<size_t CT_ARG_IDX_BASE>
struct ChannelPoolCollection : public CtArgConsumer<ChannelPoolCollection<CT_ARG_IDX_BASE>> {
    static constexpr size_t num_channel_pools = get_compile_time_arg_val(CT_ARG_IDX_BASE);
    static constexpr size_t channel_pool_types_base_idx = CT_ARG_IDX_BASE + 1;
    static constexpr size_t pools_data_base_idx = channel_pool_types_base_idx + num_channel_pools;
    
    static constexpr std::array<size_t, num_channel_pools> channel_pool_types = 
        fill_array_with_next_n_args<size_t, channel_pool_types_base_idx, num_channel_pools>();
    
    using PoolsTuple = typename PoolsBuilder<
        pools_data_base_idx, 
        num_channel_pools, 
        channel_pool_types_base_idx, 
        make_index_sequence<num_channel_pools>
    >::type;
    
    static constexpr size_t GET_NUM_ARGS_CONSUMED() {
        return PoolsBuilder<
            pools_data_base_idx, 
            num_channel_pools, 
            channel_pool_types_base_idx, 
            make_index_sequence<num_channel_pools>
        >::final_ct_arg_idx - CT_ARG_IDX_BASE;
    }
};

