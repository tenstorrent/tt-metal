// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compile_time_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"

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
    static constexpr size_t num_slots = get_compile_time_arg_val(CT_ARG_IDX_BASE + 1);

    // i.e. if this is sender 0, then we also store remote sender 0 address here
    // They are not conceptually related, but to simplify migration, we keep them together
    // for the time being.
    static constexpr size_t remote_address = get_compile_time_arg_val(CT_ARG_IDX_BASE + 2);
    static constexpr size_t remote_num_slots = get_compile_time_arg_val(CT_ARG_IDX_BASE + 3);

    static constexpr size_t GET_NUM_ARGS_CONSUMED() { return 4; }
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
template <size_t CT_ARG_IDX, size_t NumPools, typename PoolTypesArray, size_t... Indices>
struct PoolsBuilderImpl {
    using type = std::tuple<>;
    static constexpr size_t final_ct_arg_idx = CT_ARG_IDX;
};

// Base case: no more pools to build
template<size_t CT_ARG_IDX, size_t NumPools, typename PoolTypesArray>
struct PoolsBuilderImpl<CT_ARG_IDX, NumPools, PoolTypesArray> {
    using type = std::tuple<>;
    static constexpr size_t final_ct_arg_idx = CT_ARG_IDX;
};

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

    // static constexpr size_t final_ct_arg_idx = RestOfPools::final_ct_arg_idx;
    static constexpr size_t final_ct_arg_idx = PoolsBuilderImpl<next_ct_arg_idx, NumPools, PoolTypesArray, RestIndices...>::final_ct_arg_idx;
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
struct PoolsBuilder {
    static constexpr size_t final_ct_arg_idx = CT_ARG_IDX;
};

template<size_t CT_ARG_IDX, size_t NumPools, size_t TypesBaseIdx, size_t... Indices>
struct PoolsBuilder<CT_ARG_IDX, NumPools, TypesBaseIdx, index_sequence<Indices...>> {
    using PoolTypes = PoolTypesHolder<NumPools, TypesBaseIdx>;
    using Impl = PoolsBuilderImpl<CT_ARG_IDX, NumPools, PoolTypes, Indices...>;
    using type = typename Impl::type;
    static constexpr size_t final_ct_arg_idx = Impl::final_ct_arg_idx;
};

template <size_t CT_ARG_IDX_BASE, size_t NumSenderChannels, size_t NumReceiverChannels>
struct ChannelPoolCollection
    : public CtArgConsumer<ChannelPoolCollection<CT_ARG_IDX_BASE, NumSenderChannels, NumReceiverChannels>> {
    // 0 - special start tag
    static constexpr size_t special_tag_idx = CT_ARG_IDX_BASE;
    static constexpr size_t special_tag = get_compile_time_arg_val(special_tag_idx);
    static_assert(
        special_tag == 0xabcd1234,
        "Special tag not found. This implies some arguments were misaligned between host and device. Double check the "
        "CT args.");

    // 1) number of pools: n_pools
    // 2) array of pool types: pool_types[n_pools]
    // 3) for_each(pool): get args
    // 4) sender channel to pool mapping
    // 5) receiver channel to pool mapping
    static constexpr size_t num_channel_pools = get_compile_time_arg_val(CT_ARG_IDX_BASE + 1);
    static constexpr size_t channel_pool_types_base_idx = CT_ARG_IDX_BASE + 2;
    static constexpr size_t pools_data_base_idx =
        channel_pool_types_base_idx + // start of pool types
        num_channel_pools;            // end of channel pool types


    // pool types list
    static constexpr std::array<size_t, num_channel_pools> channel_pool_types =
        fill_array_with_next_n_args<size_t, channel_pool_types_base_idx, num_channel_pools>();

    // pools tuple (args)
    // unpacks args according to the pool types list
    using PoolsTuple = typename PoolsBuilder<
        pools_data_base_idx,
        num_channel_pools,
        channel_pool_types_base_idx,
        make_index_sequence<num_channel_pools>>::type;

    // sender channel to pool mapping
    static constexpr size_t sender_channel_to_pool_index_base_idx = PoolsBuilder<
                   pools_data_base_idx,
                   num_channel_pools,
                   channel_pool_types_base_idx,
                   make_index_sequence<num_channel_pools>>::final_ct_arg_idx;
    static constexpr std::array<size_t, NumSenderChannels> sender_channel_to_pool_index =
        fill_array_with_next_n_args<size_t, sender_channel_to_pool_index_base_idx, NumSenderChannels>();
    static constexpr std::array<size_t, NumReceiverChannels> receiver_channel_to_pool_index =
        fill_array_with_next_n_args<
            size_t,
            sender_channel_to_pool_index_base_idx + NumSenderChannels, //channel_pool_types_base_idx + num_channel_pools + NumSenderChannels,
            NumReceiverChannels>();


    static constexpr size_t GET_NUM_ARGS_CONSUMED() {
        return (sender_channel_to_pool_index_base_idx + NumSenderChannels + NumReceiverChannels) -
               CT_ARG_IDX_BASE;
    }
};

template <size_t NumPools, typename CHANNEL_POOL_COLLECTION>
struct ChannelPoolLookup {
    // Index into CHANNEL_POOL_COLLECTION::PoolsTuple using PoolTypes
    using pools_tuple_full = typename CHANNEL_POOL_COLLECTION::PoolsTuple;

    // Helper to get pool type at index
    template <size_t Index>
    using GetPoolType = typename std::tuple_element<Index, pools_tuple_full>::type;

    // Build tuple from pool types
    template <typename IndexSequence>
    struct BuildTuple;

    template <size_t... Indices>
    struct BuildTuple<index_sequence<Indices...>> {
        using type = std::tuple<GetPoolType<Indices>...>;
    };

    // Final type alias
    using pools_tuple = typename BuildTuple<
        typename make_index_sequence<NumPools>::type
    >::type;
};
