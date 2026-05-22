// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Hardware address generator (addrgen) integration for TensorAccessor (Quasar-only).
//
// Phase 1 scope: interleaved tensors only (DRAM and L1).
//   BANK_INNER banking + sentinel inner loop; walk is page-id order for interleaved layouts.
//   L1 interleaved uses the same placeholder DRAM ATT IDs (20..23, shift 36) until L1 ATT
//   entries exist — peek/pop works; NOC routing is not correct until then.
//

#ifndef ARCH_QUASAR
#error "tensor_accessor_addrgen.h is Quasar-only"
#endif

#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"

static constexpr uint32_t QUASAR_DRAM_FIRST_ATT_ID = 20;
static constexpr uint32_t QUASAR_DRAM_ATT_BANK_ENDPOINT_ID_SHIFT = 36;
static constexpr uint64_t ADDRGEN_LOOP_END_SENTINEL = (uint64_t)0xFFFFF;

template <typename DSpecT>
struct noc_traits_t<TensorAccessor<DSpecT, AddrgenMode::READ>> {
    struct src_args_type {};
    struct dst_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };

    template <Noc::AddressType address_type>
    static uint64_t src_addr(
        const TensorAccessor<DSpecT, AddrgenMode::READ>&, const Noc&, const src_args_type&) {
        return overlay::pop_src_addrgen_0();
    }

    template <Noc::AddressType address_type>
    static auto dst_addr(
        const TensorAccessor<DSpecT, AddrgenMode::READ>& dst,
        const Noc& noc,
        const dst_args_type& args) {
        uint64_t noc_addr = dst.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }
};

template <typename DSpecT>
struct noc_traits_t<TensorAccessor<DSpecT, AddrgenMode::WRITE>> {
    struct src_args_type {
        uint32_t page_id{};
        uint32_t offset_bytes = 0;
    };
    struct dst_args_type {};

    template <Noc::AddressType address_type>
    static auto src_addr(
        const TensorAccessor<DSpecT, AddrgenMode::WRITE>& src,
        const Noc& noc,
        const src_args_type& args) {
        uint64_t noc_addr = src.get_noc_addr(args.page_id, args.offset_bytes, noc.get_noc_id());
        if constexpr (address_type == Noc::AddressType::LOCAL_L1) {
            ASSERT(noc.is_local_addr(noc_addr));
            return static_cast<uint32_t>(noc_addr);
        } else {
            return noc_addr;
        }
    }

    template <Noc::AddressType address_type>
    static uint64_t dst_addr(
        const TensorAccessor<DSpecT, AddrgenMode::WRITE>&, const Noc&, const dst_args_type&) {
        return overlay::pop_dest_addrgen_0();
    }
};

namespace tensor_accessor::detail {

template <typename DSpecT>
constexpr uint32_t interleaved_num_banks() {
    if constexpr (DSpecT::is_dram) {
        return NUM_DRAM_BANKS;
    } else {
        return NUM_L1_BANKS;
    }
}

template <typename DSpecT, AddrgenMode Mode>
void configure_interleaved_addrgen_src(TensorAccessor<DSpecT, Mode>& acc) {
    const uint64_t page_size = static_cast<uint64_t>(acc.get_aligned_page_size());
    const uint32_t num_banks = interleaved_num_banks<DSpecT>();

    overlay::setup_src_base_start_addrgen_0(static_cast<uint64_t>(acc.get_bank_base_address()));
    overlay::setup_src_banking_addrgen_0(overlay::BankingConfig{
        .endpoint_id_shift = QUASAR_DRAM_ATT_BANK_ENDPOINT_ID_SHIFT,
        .size = num_banks,
        .skip = 1,
        .base = QUASAR_DRAM_FIRST_ATT_ID,
        .offset = 0,
        .bank_order = overlay::BANK_INNER,
    });
    overlay::setup_src_inner_loop_addrgen_0(page_size, ADDRGEN_LOOP_END_SENTINEL * page_size);
}

template <typename DSpecT, AddrgenMode Mode>
void configure_interleaved_addrgen_dst(TensorAccessor<DSpecT, Mode>& acc) {
    const uint64_t page_size = static_cast<uint64_t>(acc.get_aligned_page_size());
    const uint32_t num_banks = interleaved_num_banks<DSpecT>();

    overlay::setup_dest_base_start_addrgen_0(static_cast<uint64_t>(acc.get_bank_base_address()));
    overlay::setup_dest_banking_addrgen_0(overlay::BankingConfig{
        .endpoint_id_shift = QUASAR_DRAM_ATT_BANK_ENDPOINT_ID_SHIFT,
        .size = num_banks,
        .skip = 1,
        .base = QUASAR_DRAM_FIRST_ATT_ID,
        .offset = 0,
        .bank_order = overlay::BANK_INNER,
    });
    overlay::setup_dest_inner_loop_addrgen_0(page_size, ADDRGEN_LOOP_END_SENTINEL * page_size);
}

template <typename DSpecT, AddrgenMode Mode>
void configure_addrgen_src(TensorAccessor<DSpecT, Mode>& acc) {
    overlay::reset_addrgen_0();
    if constexpr (DSpecT::is_interleaved) {
        configure_interleaved_addrgen_src(acc);
        return;
    }
    ASSERT(false);
}

template <typename DSpecT, AddrgenMode Mode>
void configure_addrgen_dst(TensorAccessor<DSpecT, Mode>& acc) {
    overlay::reset_addrgen_0();
    if constexpr (DSpecT::is_interleaved) {
        configure_interleaved_addrgen_dst(acc);
        return;
    }
    ASSERT(false);
}

}  // namespace tensor_accessor::detail
