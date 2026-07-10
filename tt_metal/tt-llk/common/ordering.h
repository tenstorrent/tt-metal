// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel
{
/**
 * @brief Copies data from src -> dest, blocking until the copy is completed.
 * @note Addresses are marked volatile because it's assumed that this function is used for sync between threads.
 * @param dst volatile destination address
 * @param src volatile source address
 * @param len number of bytes to copy
 * @return pointer to the destination
 */
inline volatile void *memcpy_blocking(volatile void *dst, const volatile void *src, std::size_t len)
{
    // I'm prioritizing correctness and simplicity over complexity and performance at this point.
    // Therefore this is definitely slow. I don't expect this to become a bottleneck, so we can optimize it later.

    // https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md

    // this code provides a blocking memcpy by doing the following:
    // - issue a LOAD from src[i]
    // - issue a STORE to dst[i]
    //     - the STORE flushes the L0 (DCACHE) line, so the subsequent LOAD will read from L1
    // - issue a LOAD from dst[i]
    //     - this LOAD is ordered after the STORE to the same address
    // - issue 7 NOPs after the final LOAD
    //     - the retire-order queue has 8 slots; the final LOAD + 7 NOPs fill it completely
    //     - the final LOAD only retires (frees its slot) once the read-response arrives from the memory subsystem arrives
    //     - the final LOAD completes after the STORE, so once it retires, the memcpy is fully committed to memory
    //     - until the LOAD retires, no new instruction can enter the retire-order queue
    //     - subsequent LOADs/STOREs can't enter the LSQ and emit transactions because the retire-order queue is full
    //     - this ensures that no memory transactions can be issued until the memcpy is fully committed to underlying memory
    // - memory clobber
    //     - prevents the COMPILER from reordering memory accesses across this boundary

    volatile char *dstc       = reinterpret_cast<volatile char *>(dst);
    const volatile char *srcc = reinterpret_cast<const volatile char *>(src);

    for (std::size_t i = 0; i < len; i++)
    {
        dstc[i] = srcc[i];
    }

    for (std::size_t i = 0; i < len; i++)
    {
        (void)(dstc[i]);
    }

#ifdef WORMHOLE
    asm volatile(
        "nop\n\t"
        "nop\n\t"
        "nop\n\t"
        "nop\n\t"
        "nop\n\t"
        "nop\n\t"
        "nop\n\t" ::
            : "memory");
#else
    asm volatile("fence" ::: "memory");
#endif

    return dst;
}


/**
 * @brief Issues a load transaction that will block the core until the transaction is completed.
 * @tparam T 32-bit type to load
 * @param ptr address to read from
 * @return value read from the address
 */
template <typename T, typename = std::enable_if_t<std::is_trivially_copyable_v<T>>>
inline T load_blocking(volatile T *ptr)
{
    static_assert(sizeof(T) == sizeof(std::uint32_t), "load_blocking: operand must be 32-bit");

    // https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md

    // important note: FENCE on Wormhole is a NOP
    //
    // this code provides a blocking load by doing the following:
    // - issue a LOAD transaction to the address
    //     - actual load that was requested
    // - issue an instruction that requires the data from the LOAD transaction
    //     - block the pipeline until the LOAD transaction completes
    // - memory clobber
    //     - prevent reordering of transactions that occur after the load before the load by the COMPILER

    std::uint32_t raw;

    asm volatile(
        "lw %[raw], (%[ptr])\n\t"
        "and %[raw], %[raw], %[raw]"
        : [raw] "=r"(raw)
        : [ptr] "r"(ptr)
        : "memory");

    T val;
    std::memcpy(&val, &raw, sizeof(T)); // trickery to return T loaded into register

    return val;
}


/**
 * @brief Issues a store transaction that will block the core until the transaction is completed.
 * @tparam T 32-bit type to store
 * @tparam U type of the value to store, must be trivially assignable to T
 * @param ptr address to write to
 * @param val value to write
 */
template <typename T, typename U, typename = std::enable_if_t<std::is_trivially_copyable_v<T> && std::is_trivially_assignable_v<T &, U>>>
inline void store_blocking(volatile T *ptr, U &&val)
{
    static_assert(sizeof(T) == sizeof(std::uint32_t), "store_blocking: operand must be 32-bit");

    T typed = static_cast<T>(std::forward<U>(val));

    std::uint32_t raw;
    std::memcpy(&raw, &typed, sizeof(raw));

    // https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/BabyRISCV/MemoryOrdering.md

    // important note: FENCE on Wormhole is a NOP
    //
    // this code provides a blocking store by doing the following:
    // - issue a STORE transaction to the address
    //     - actual store that was requested
    // - issue a LOAD transaction to the address
    //     - must complete after the STORE transaction
    // - issue an instruction that requires the data from the LOAD transaction
    //     - block the pipeline until the LOAD transaction completes, ensuring that the STORE is complete
    // - memory clobber
    //     - prevent reordering of transactions that occur after the store before the store by the COMPILER

    asm volatile(
        "sw %[raw], (%[ptr])\n\t"
        "lw %[raw], (%[ptr])\n\t"
        "and %[raw], %[raw], %[raw]\n\t"
        : [raw] "+r"(raw)
        : [ptr] "r"(ptr)
        : "memory");
}


/**
 * @brief Forces the compiler to load @p ref from memory.
 *
 * @note Does NOT enforce ordering in code or memory.
 * @note Guarantees that a load will be performed.
 *
 * @tparam T type of the referenced object
 * @param ref to load from memory
 * @return loaded value
 *
 * @par Example
 * Consumer waits for producer to create entries in a ringbuffer.
 * @code
 * // Producer updates write_idx, so we need to invalidate when polling
 * while ((ckernel::load_force(write_idx) - read_idx + BUFFER_SIZE) % BUFFER_SIZE == 0);
 * @endcode
 */
template <typename T>
[[nodiscard]] inline T load_force(T &ref)
{
    // "=m" output constraint: tells the compiler that ref may have been modified by external code
    // Effect: prevents the compiler from reusing a stale register-cached value.
    asm volatile("" : "=m"(ref));
    return ref;
}


/**
 * @brief Assigns @p val to @p ref and prevents the compiler from eliminating or deferring the store.
 *
 * @note Does NOT enforce ordering in code or memory.
 * @note Guarantees that a store will be performed.
 *
 * @tparam T type of the referenced object
 * @tparam U type of the value to store
 * @param ref reference to the object to store into
 * @param val value to assign to @p ref
 *
 * @par Example
 * Producer signals entries have been written to a ringbuffer.
 * @code
 * // Consumer polls write_idx, so we need to ensure the store is committed
 * ckernel::store_force(write_idx, (write_idx + chunk) % BUFFER_SIZE);
 * @endcode
 */
template <typename T, typename U>
inline void store_force(T &ref, U &&val)
{
    ref = std::forward<U>(val);

    // "m" input constraint: tells compiler this asm reads from ref
    // Effect: compiler must flush any pending write to ref before this point
    asm volatile("" : : "m"(ref));
}


/**
 * @brief Compiler-only barrier: prevents reordering of memory accesses across this point.
 * @note Does not enforce CPU or system memory ordering by itself.
 */
inline void fence_compiler()
{
    asm volatile("" ::: "memory");
}

// LLK INFRA SPECIFIC HELPERS

} // namespace ckernel
