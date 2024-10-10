// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstddef>


namespace tt_metal {

namespace routing {
enum class TransportMedium : uint8_t {
    NOC,
    ETHERNET,

    // For testing in shared memory environment (e.g. basic host unit tests)
    SHARED_MEM,
};

enum class PayloadMode : uint8_t {
    // Header is included with the payload
    PACKETIZED,

    // No header, just raw data
    PAGED
};

enum class RemoteCoreLocality : uint8_t {
    LOCAL_CHIP,
    REMOTE_CHIP
};


// Can we piggy back of another enum here instead of creating a new one?
enum class RemoteCoreType : uint8_t {
    WORKER,
    ETHERNET,
    DRAM
};



using l1_address_t = uint32_t;
using noc_address_t = uint32_t;


// TODO: split these off into separate headers for the separate use cases
template <TransportMedium transport_medium> struct remote_payload_addr_t { using type = nullptr_t; };

using host_address_t = uint64_t;
// Host mem
template <> struct remote_payload_addr_t<TransportMedium::SHARED_MEM> { using type = host_address_t; };
template <> struct remote_payload_addr_t<TransportMedium::NOC> { using type = noc_address_t; };
template <> struct remote_payload_addr_t<TransportMedium::ETHERNET> { using type = l1_address_t; };

} // namespace routing
} // namespace tt_metal
