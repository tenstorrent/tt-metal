// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "ckernel.h"
#include "llk_assert.h"

namespace llk
{

template <std::uint32_t BUFFER_SIZE>
class Stream
{
    // non volatile; manually managed;
    std::uint32_t write_idx;
    std::uint32_t read_idx;

    volatile char buffer[BUFFER_SIZE];

    std::uint32_t capacity() const
    {
        return BUFFER_SIZE - 1;
    }

    std::uint32_t producer_poll_free()
    {
        return (capacity() - write_idx + ckernel::load_force(read_idx)) % BUFFER_SIZE;
    }

    std::uint32_t producer_wait_free()
    {
        std::uint32_t free = producer_poll_free();
        while (free == 0)
        {
            asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;");
            ckernel::invalidate_data_cache();
            free = producer_poll_free();
        }

        return free;
    }

    std::uint32_t consumer_poll_avail()
    {
        return (ckernel::load_force(write_idx) - read_idx + BUFFER_SIZE) % BUFFER_SIZE;
    }

    std::uint32_t consumer_wait_avail()
    {
        std::uint32_t avail = consumer_poll_avail();
        while (avail == 0)
        {
            asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;");
            ckernel::invalidate_data_cache();
            avail = consumer_poll_avail();
        }

        return avail;
    }

public:
    Stream()                         = delete;
    Stream(const Stream&)            = delete;
    Stream(Stream&&)                 = delete;
    Stream& operator=(const Stream&) = delete;
    Stream& operator=(Stream&&)      = delete;

    /**
     * @brief Initialize the stream. Should be called only once, before all operations.
     */
    void init()
    {
        ckernel::store_force(write_idx, static_cast<std::uint32_t>(0));
        ckernel::store_force(read_idx, static_cast<std::uint32_t>(0));
    }

    /**
     * @brief Push `size` chars into the stream.
     * @param inputv the input buffer
     * @param size the number of chars to push
     */
    void push(const void* inputv, std::size_t size)
    {
        const char* input     = reinterpret_cast<const char*>(inputv);
        std::size_t remaining = size;
        std::size_t offset    = 0;

        while (remaining > 0)
        {
            std::uint32_t free = producer_wait_free();

            std::size_t chunk = std::min<std::size_t>(remaining, free);

            const std::size_t head = std::min<std::size_t>(chunk, BUFFER_SIZE - write_idx);

            // memcpy_blocking: returns only after the copy is present in memory
            ckernel::memcpy_blocking(&buffer[write_idx], &input[offset], head);
            ckernel::memcpy_blocking(&buffer[0], &input[offset + head], chunk - head);

            // fence_compiler: prevents the compiler from reordering store_force before the memcpy_blocking
            ckernel::fence_compiler();

            // store_force: prevents the compiler from optimizing away the update to write_idx
            ckernel::store_force(write_idx, (write_idx + chunk) % BUFFER_SIZE);

            offset += chunk;
            remaining -= chunk;
        }
    }

    /**
     * @brief Peek at the next char in the stream without consuming it.
     * @return the next char if available, std::nullopt otherwise
     */
    [[nodiscard]] std::optional<char> peek()
    {
        if (consumer_poll_avail() > 0)
        {
            return buffer[read_idx];
        }

        return std::nullopt;
    }

    /**
     * @brief Pop the next `size` chars in the stream and consume them.
     * @param outputv the output buffer
     * @param size the number of chars to pop
     */
    void pop(void* outputv, std::size_t size)
    {
        char* output          = reinterpret_cast<char*>(outputv);
        std::size_t remaining = size;
        std::size_t offset    = 0;

        while (remaining > 0)
        {
            std::uint32_t avail = consumer_wait_avail();

            std::size_t chunk = std::min<std::size_t>(remaining, avail);

            const std::size_t head = std::min<std::size_t>(chunk, BUFFER_SIZE - read_idx);

            // memcpy_blocking: returns only after the copy is present in memory
            ckernel::memcpy_blocking(&output[offset], &buffer[read_idx], head);
            ckernel::memcpy_blocking(&output[offset + head], &buffer[0], chunk - head);

            // fence_compiler: prevents the compiler from reordering store_force before the memcpy_blocking
            ckernel::fence_compiler();

            // store_force: prevents the compiler from optimizing away the update to read_idx
            ckernel::store_force(read_idx, (read_idx + chunk) % BUFFER_SIZE);

            offset += chunk;
            remaining -= chunk;
        }
    }
};

} // namespace llk
