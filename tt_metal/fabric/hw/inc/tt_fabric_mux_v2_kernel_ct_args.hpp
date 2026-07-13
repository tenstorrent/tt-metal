// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compile_time_args.h"

namespace tt::tt_fabric::mux_v2::ct_args {

#define FABRIC_MUX_V2_NAMED_CT_ARG(name) get_named_compile_time_arg_val(name)

constexpr uint32_t num_buffers_per_channel = FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_num_buffers_per_channel");
constexpr uint32_t num_channels = FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_num_channels");
constexpr uint32_t mux_status_address = FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_mux_status_address");
constexpr uint32_t channel_region_base_address =
    FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_channel_region_base_address");
constexpr uint32_t connection_info_region_base_address =
    FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_connection_info_region_base_address");
constexpr uint32_t connection_handshake_region_base_address =
    FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_connection_handshake_region_base_address");
constexpr uint32_t shared_ring_region_base_address =
    FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_shared_ring_region_base_address");
constexpr uint32_t shared_control_region_base_address =
    FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_shared_control_region_base_address");
constexpr uint32_t credit_notify_scratch_region_base_address =
    FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_credit_notify_scratch_region_base_address");
constexpr uint32_t channel_buffer_size_bytes = FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_channel_buffer_size_bytes");
constexpr uint32_t per_channel_scalar_region_stride_bytes =
    FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_per_channel_scalar_region_stride_bytes");
constexpr uint32_t forwarder_service_burst_size =
    FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_forwarder_service_burst_size");
constexpr uint32_t shared_trid_ring_capacity = FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_shared_trid_ring_capacity");
constexpr uint32_t forwarder_ready_sem_id = FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_forwarder_ready_sem_id");
constexpr uint32_t manager_init_done_sem_id = FABRIC_MUX_V2_NAMED_CT_ARG("fabric_mux_v2_manager_init_done_sem_id");

static_assert(num_buffers_per_channel > 0, "FabricMuxV2 num_buffers_per_channel must be greater than zero");
static_assert(num_channels > 0, "FabricMuxV2 num_channels must be greater than zero");
static_assert(channel_buffer_size_bytes > 0, "FabricMuxV2 channel_buffer_size_bytes must be greater than zero");
static_assert(
    per_channel_scalar_region_stride_bytes > 0,
    "FabricMuxV2 per_channel_scalar_region_stride_bytes must be greater than zero");
static_assert(forwarder_service_burst_size > 0, "FabricMuxV2 forwarder service burst size must be greater than zero");
static_assert(shared_trid_ring_capacity > 0, "FabricMuxV2 shared TRID ring capacity must be greater than zero");
static_assert(
    shared_trid_ring_capacity != 0 && (shared_trid_ring_capacity & (shared_trid_ring_capacity - 1)) == 0,
    "Shared TRID ring capacity must be a power of two");
static_assert(
    shared_trid_ring_capacity <= (NOC_MAX_TRANSACTION_ID + 1),
    "FabricMuxV2 shared TRID ring capacity must not exceed available transaction IDs");

constexpr bool num_buffers_per_channel_is_pow2 = (num_buffers_per_channel & (num_buffers_per_channel - 1)) == 0;
constexpr uint32_t num_buffers_per_channel_mask = num_buffers_per_channel_is_pow2 ? (num_buffers_per_channel - 1) : 0;
constexpr uint32_t shared_trid_ring_mask = shared_trid_ring_capacity - 1;

#undef FABRIC_MUX_V2_NAMED_CT_ARG

}  // namespace tt::tt_fabric::mux_v2::ct_args
