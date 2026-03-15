// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

namespace llk::debug
{

class tensix_dump
{
private:
    static constexpr std::uintptr_t TENSIX_DUMP_MAILBOX_ADDRESS = 0x16AFE4;
    enum class mailbox_state : std::uint32_t
    {
        DONE      = 0,
        REQUESTED = 1,
    };

public:
    /**
     * @brief Requests a tensix state dump and blocks until the host completes it.
     * @note All TRISC cores must call this for the host to fulfill the request.
     */
    static void request()
    {
        ckernel::tensix_sync(); // make sure all changes to tensix state are written

        volatile mailbox_state* const DUMP_MAILBOX = reinterpret_cast<mailbox_state*>(TENSIX_DUMP_MAILBOX_ADDRESS);

        ckernel::store_blocking(&DUMP_MAILBOX[COMPILE_FOR_TRISC], mailbox_state::REQUESTED); // signal host to start dumping tensix state

        do
        {
            ckernel::invalidate_data_cache(); // prevent polling from cache.
            asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;");
        } while (ckernel::load_blocking(&DUMP_MAILBOX[COMPILE_FOR_TRISC]) == mailbox_state::REQUESTED); // wait while the host is dumping tensix state.
    }
};

} // namespace llk::debug
