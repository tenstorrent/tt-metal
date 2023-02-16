#pragma once

#include "src/firmware/riscv/common/fw_debug.h"
#include "src/firmware/riscv/common/memory_management/mm_common_types.hpp"

namespace allocators {

/**
 * @brief Allocate many times, then free all at once. All operations O(1).
 *
 * Maintais only a single pointer to the next free address, and limits on the range of the owned memory.
 * Must only be used with trivial types, or destructor must be called externally,
 *   as it does not maintain a destructor list or any thing like that.
 *
 * @p num_alignment_bits_ is the number of low bits that must be 0 in returned addresses.
 *   Ie., alignment in bytes is @c pow(2,num_alignment_bits_) .
 */
template <std::uint8_t num_alignment_bits_>
class Arena : public SingleSpan, public PowerOf2Aligned<num_alignment_bits_> {
 public:
  using PowerOf2Aligned<num_alignment_bits_>::alignments;
  static const std::size_t min_block_size = alignments.size;

  /**
   * @brief Construct an empty arena
   */
  Arena() : Arena({nullptr, nullptr}) {}

  /**
   * @brief Construct an arena allocator that owns the memory in @p mem_span
   */
  Arena(MemorySpan mem_span) : SingleSpan(mem_span), next_free(mem_span.begin()) {}

  // move only -- copies would lead to two things owning the same memory
  Arena(const Arena&) = delete;
  Arena(Arena&&) = default;
  Arena& operator=(const Arena&) = delete;
  Arena& operator=(Arena&&) = default;

  /**
   * @brief request a block of size @p size .
   */
  std::byte* allocate(std::size_t size) {
    size = this->round_up_to_alignment(size);
    auto result = next_free;
    if (next_free + size > mem_span.end())
      return nullptr;
    next_free += size;
    return result;
  }

  void reset() { next_free = mem_span.begin(); }

 private:
  std::byte* next_free;
};

}  // end namespace allocators
