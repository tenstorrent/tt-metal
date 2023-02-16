#pragma once

#include <cstddef>

/**
 * @brief Represents a single contiguous range of addresses.
 */
class MemorySpan {
 public:
  using Ptr = std::byte*;

  /** @brief Construct from a @p begin and past-the @p end addresses. */
  MemorySpan(Ptr begin, Ptr end) : begin_(begin), end_(end) {}

  /** @brief Alternate construction method useful then the @p begin expression is complicated. */
  static MemorySpan from_pointer_and_size(Ptr begin, std::size_t size) { return {begin, begin + size}; }

  /** @brief Same as the usual constructor, but saves the caller from two casts */
  static MemorySpan fromU32(std::uint32_t begin, std::uint32_t end) {
    return {reinterpret_cast<Ptr>(begin), reinterpret_cast<Ptr>(end)};
  }

  /** @brief Lowest address in this span */
  Ptr begin() const { return begin_; }

  /** @brief One address past the last address in this span */
  Ptr end() const { return end_; }

  /** @brief Number of bytes owned by the span */
  auto size() const { return end_ - begin_; }

  /** @brief Is @p ptr in this range? */
  bool contains_address(Ptr ptr) const { return begin_ <= ptr && ptr < end_; }

  /** @brief Does either @c this or @p other contain common addresses? */
  bool overlaps_with(const MemorySpan& other) {
    return this->contains_address(other.begin()) || other.contains_address(this->begin());
  }

 private:
  Ptr begin_, end_;
};

/**
 * @brief Number of bits needed for aligning to X number of bytes
 */
namespace AlignmentBits {
static constexpr std::uint8_t B8 = 3;
static constexpr std::uint8_t B16 = 4;
static constexpr std::uint8_t B32 = 5;
static constexpr std::uint8_t B64 = 6;
}  // namespace AlignmentBits

namespace allocators {

/**
 * @brief Mixin base for allocators that own a single range of memory addresses.
 *
 * Allocators like that should (non-virtually!) inherit from this.
 */
struct SingleSpan {
  SingleSpan(MemorySpan mem_span) : mem_span(mem_span) {}
  MemorySpan get_memory_span() const { return mem_span; }
  bool owns_address(std::byte* ptr) const { return mem_span.contains_address(ptr); }

 protected:
  MemorySpan mem_span;
};

/**
 * @brief Mixin base for allocators that have a compile-time alignment that is a power of 2
 *
 * @p num_alignment_bits_ is the number of low bits that must be 0.
 *   Ie., alignment in bytes is @c pow(2,num_alignment_bits_) .
 */
template <std::uint8_t num_alignment_bits_>
struct PowerOf2Aligned {
  static constexpr std::size_t num_alignment_bits = num_alignment_bits_;  //< 2^this is alignment
  static constexpr std::size_t alignment_size = 1 << num_alignment_bits;  //< actual alignment, in bytes
  static constexpr std::size_t alignment_mask = alignment_size - 1;       //< bits that need to be low

  /**
   * @brief Bitmask-based rounding up by a power of 2. Likely a dependent name, so will need @c this-> when calling.
   */
  template <typename Int>
  static Int round_up_to_alignment(Int size) {
    return round_down_to_alignment(size + alignment_size - 1);
  }

  /**
   * @brief Bitmask-based rounding down by a power of 2. Likely a dependent name, so will need @c this-> when calling.
   */
  template <typename Int>
  static Int round_down_to_alignment(Int size) {
    return size & ~alignment_mask;
  }

  /**
   * @brief Class and variable for consise re-exporting in derived classes.
   *
   * Deriving classes are likely templates too, which means that the names in this class are 'dependent',
   *   and will not be found witout explicitly @c using them.
   * More precise explanation at https://isocpp.org/wiki/faq/templates#nondependent-name-lookup-members
   *
   * Ex.:
   * @code{.cpp}
   *   using PowerOf2Aligned<num_alignment_bits_>::alignments;
   *   static const std::size_t min_block_size = alignments.size;
   * @endcode
   */
  static constexpr struct Constants {
    static constexpr auto num_bits = num_alignment_bits;
    static constexpr auto size = alignment_size;
    static constexpr auto mask = alignment_mask;
  } alignments = {};
};

}  // namespace allocators
