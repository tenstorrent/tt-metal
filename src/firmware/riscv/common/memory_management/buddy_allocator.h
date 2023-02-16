#pragma once

#include <bitset>

#include "fw_debug.h"
#include "memory_management/mm_common_types.hpp"

/**
 * @brief A constant-sized binary tree of bits.
 *
 * @p num_levels_ is the depth of the tree. All functions that take a level expect (positive) values less than this
 * @p log2_num_roots_ has the effect of multiplying the size of each level by @c pow(2,log2_num_roots_)
 *
 * The 'top' level (the one with the fewest nodes) is level 0, and the the 'bottom' (most nodes) is @c num_levels_-1 .
 */
template <std::size_t num_levels_, std::size_t log2_num_roots_ = 0>
struct BitTree {
  using LevelIndex = std::int8_t;
  static constexpr std::size_t log2_num_roots = log2_num_roots_;
  static constexpr std::size_t num_roots = 1 << log2_num_roots;
  static constexpr LevelIndex num_levels = num_levels_;
  static_assert(num_levels != 0, "bit tree must have at least one level");

  static constexpr std::size_t offset_for_level(LevelIndex ilevel) {
    return (1 << (ilevel + log2_num_roots)) - num_roots;
  }

  static constexpr std::size_t size_of_level(LevelIndex ilevel) { return 1 << (ilevel + log2_num_roots); }

  static constexpr std::size_t num_bits = offset_for_level(num_levels);
  static constexpr std::size_t biggest_level_size = size_of_level(num_levels - 1);

  std::bitset<num_bits> bits;

  /**
   * @brief Direct, constant time access to the bit at level @p ilevel and offset @p iblk
   */
  decltype(auto) get(LevelIndex ilevel, std::size_t offset_in_level) {
    const auto offset = offset_for_level(ilevel) + offset_in_level;
    // FWASSERT("accessing out-of-range bit", offset < bits.size());
    return bits[offset];
  }

  decltype(auto) get(LevelIndex ilevel, std::size_t offset_in_level) const {
    const auto offset = offset_for_level(ilevel) + offset_in_level;
    // FWASSERT("accessing out-of-range bit", offset < bits.size());
    return bits[offset];
  }

  void reset() { bits.reset(); }
};

namespace allocators {

/**
 * @brief Buddy allocator, with a fixed number of divisions that uses trees of bits instead of nodes.
 *
 * @p num_divisions_ is the number of times the range passed to the constructor will be subdivided.
 *   The 'maximum' number of allocations is equal to @c pow(2,num_divisions_) min-sized blocks.
 * @p num_alignment_bits_ is the number of low bits that must be 0 in returned addresses.
 *   Ie., alignment in bytes is @c pow(2,num_alignment_bits_) .
 *
 * Instead of nodes, this buddy allocator maintains two binary trees of bits called @c is_full and @c is_used.
 * Each node of of the trees corresponds to a possible sub-division of the memory range.
 * Nodes in @c is_full represent the lack of any free descendant nodes, including this node.
 * Nodes in @c is_used represent that at least one descendant, including this node are used.
 * If a block was allocated in a given subdivision, then the corresponding node will be set in *both* trees.
 * Space usage is approximately @c 2*pow(2,num_levels)/8 bytes.
 *
 * Allocation is done by scanning across the @c is_used bits for the correct level,
 *   skipping forward if any ancestors are found to have @c is_full set.
 * This results in worst-case @c O(M/B*N_d) runtime,
 *   where @c M is the range size, @c B is the block size and @c N_d is @p num_divisions_.
 * The first allocation is always @c O(1) , and as the range fills up, more skipping forward will
 *   generally happen, resulting in better runtime than the worst-case.
 * @c O(N_d) runtime is possible with per-size trees, but this will have higher constants due to more bookkeeping.
 *
 * Deallocation is done by clearing all @c is_full bits and any now-unused nodes' @c is_used bits.
 * This results in @c O(N_d) runtime.
 */
template <std::size_t num_divisions_, std::size_t min_block_size_, std::uint8_t num_alignment_bits_, std::size_t num_unusable_divisions_ = num_divisions_ / 3>
class NodelessBuddy : public SingleSpan, public PowerOf2Aligned<num_alignment_bits_> {
  using PowerOf2Aligned<num_alignment_bits_>::alignments;

 public:
  using LevelIndex = int;
  using Offset = int;

  static constexpr std::size_t num_divisions = num_divisions_;
  static constexpr std::size_t min_block_size = min_block_size_;

  /**
   * @brief This many of the top levels (ie. block sizes) won't exist.
   *
   * An optimization. Trading-off the ability to allocate the largest blocks in exchange for not having to
   *   check or update the nodes for those blocks.
   * Has almost no effect on memory requirements -- the top levels are small
   * Could be a template paramater, if needed.
   */
  static constexpr std::size_t num_unusable_divisions = num_unusable_divisions_;

  static constexpr LevelIndex num_levels = 1 + num_divisions - num_unusable_divisions;  ///< Number of blocks sizes
  using AllLevelsBitTree = BitTree<num_levels, num_unusable_divisions>;

  static constexpr std::size_t max_block_size = min_block_size * (1 << (num_levels - 1));
  static constexpr std::size_t max_num_allocations = 1 << num_divisions;  ///< Max. number of min-size blocks
  static constexpr std::size_t min_required_memory_range_size = min_block_size * max_num_allocations;

  /**
   * @brief Construct an empty allocator
   */
  NodelessBuddy() : NodelessBuddy({nullptr, nullptr}) {}

  /**
   * @brief Construct an allocator that owns the memory in @p mem_span
   */
  NodelessBuddy(MemorySpan mem_span) : SingleSpan(mem_span) {
    if (mem_span.size() != 0) {
      FWASSERT("Memory range not large enough for buddy", (int)min_required_memory_range_size <= mem_span.size());
    }
  }

  // move only -- copies would lead to two things owning the same memory
  NodelessBuddy(const NodelessBuddy&) = delete;
  NodelessBuddy(NodelessBuddy&&) = default;
  NodelessBuddy& operator=(const NodelessBuddy&) = delete;
  NodelessBuddy& operator=(NodelessBuddy&&) = default;

  /**
   * @brief Allocate and return a block that is at least @p size bytes.
   *
   * @p size will be rounded up to a power of 2 times the min_block_size.
   * Return @c nullptr if no free block can be found.
   */
  std::byte* allocate(std::size_t size);

  /**
   * @brief Release the block at @p ptr into the pool of blocks.
   */
  void deallocate(std::byte* ptr);

  /**
   * @brief Mark all blocks as available; reset to the initial state after construction.
   */
  void reset() {
    is_used_bit_tree.reset();
    is_full_bit_tree.reset();
  }

  /**
   * @brief Used for debugging tests. Definition in test_buddy_allocator.cpp
   */
  template <typename NodelessBuddy_>
  friend void dump_buddy_allocator(const NodelessBuddy_& nb);

 private:
  /**
   * @brief Search up the @c is_full tree for the highest set parent.
   *
   * Starts seach at the parent of the specified node
   * In the return value, the bool indicates if a set parent was found, and therefore the validity
   *   of the other member of the pair, the level.
   */
  std::pair<LevelIndex, bool> find_full_parent_level(LevelIndex child_level, Offset iblk) const {
    if (child_level == 0) {
      return {0, false};  // "base case" -- has no parents
    }
    // go up the whole tree, find biggest is_full block
    std::pair<LevelIndex, bool> result = {0, false};
    for (int plevel = child_level - 1; plevel >= 0; --plevel) {
      iblk /= 2;
      if (is_full_bit_tree.get(plevel, iblk)) {
        result = {plevel, true};  // we're going from small to big, so last seen full node will be result
      } else if (result.second) {
        // if already found an is_full block, and this (parent of it) isn't, there's no point in going further.
        // this is an optimization -- should probably try profiling this to see if the br instruction is worth it
        break;
      }
    }
    return result;
  }

  std::size_t block_size_at_level(LevelIndex ilevel) const {
    return (1 << (num_levels - (ilevel + 1))) * min_block_size;
  }

  LevelIndex get_level_for_size(std::size_t size) const {
    LevelIndex ilevel;
    // start at the smallest
    for (ilevel = num_levels - 1; ilevel != 0; --ilevel) {
      if (size <= block_size_at_level(ilevel)) {
        break;  // if it fits, return.
      }
    }
    return ilevel;
  }

 private:
  /**
   * @brief is block[level,iblk] used by anything at or below it
   *
   * Invariant: "if either child is set, then their parent must be"
   *   is_used[level+1,iblk*2] || is_used[level+1,iblk*2+1] <= is_used[level,iblk]
   */
  AllLevelsBitTree is_used_bit_tree = {};

  /**
   * @brief Is block[level,iblk] used directly, or are both child blocks marked true
   *
   * Invariant: "if both children are set, then their parent must be"
   *   is_full[level+1,iblk*2] && is_full[level+1,iblk*2+1] <= is_full[level,iblk]
   */
  AllLevelsBitTree is_full_bit_tree = {};

  static_assert(min_block_size % alignments.size == 0);
  static_assert(max_num_allocations == AllLevelsBitTree::biggest_level_size);
  static_assert(min_required_memory_range_size == max_block_size * AllLevelsBitTree::size_of_level(0));
  static_assert(min_required_memory_range_size == min_block_size * AllLevelsBitTree::biggest_level_size);
};

/*****************
 * Implementations
 *****************/

template <std::size_t a, std::size_t b, std::uint8_t c, std::size_t d>
inline std::byte* NodelessBuddy<a, b, c, d>::allocate(std::size_t size) {
  if (size > max_block_size || mem_span.size() == 0) {
    return nullptr;
  }
  size = this->round_up_to_alignment(size);
  const auto ilevel = get_level_for_size(size);

  // walk across the level for this size, checking the parents of any free blocks
  Offset iblk = 0;
  for (;;) {
    if (iblk == (Offset)is_used_bit_tree.size_of_level(ilevel)) {
      return nullptr;  // hit the end -- no free blocks!
    } else if (const auto& [plevel, found] = find_full_parent_level(ilevel, iblk); found) {
      iblk += 1 << (ilevel - plevel);                     // skip forward past the full parent
    } else if (not is_used_bit_tree.get(ilevel, iblk)) {  // putting this here pessimises many blocks of the same size
      break;                                              // found an empty slot!
    } else {
      ++iblk;  // just look at the next block in this level
    }
  }

  // mark this block as alloc
  is_used_bit_tree.get(ilevel, iblk) = true;
  is_full_bit_tree.get(ilevel, iblk) = true;

  // set all parents' is_used bit. And, set all parents' is_full bit, if both children are now full
  Offset prev_iblk = iblk;
  bool prev_node_full = true;
  for (LevelIndex plevel = ilevel - 1; plevel >= 0; --plevel) {
    const auto prev_node_sibling_iblk = prev_iblk + (prev_iblk % 2 ? -1 : 1);
    const auto curr_iblk = prev_iblk / 2;
    const auto child_level = plevel + 1;
    const bool prev_node_sibling_is_full = is_full_bit_tree.get(child_level, prev_node_sibling_iblk);

    // all bit setting is here
    is_used_bit_tree.get(plevel, curr_iblk) = true;
    if (prev_node_full && prev_node_sibling_is_full) {
      is_full_bit_tree.get(plevel, curr_iblk) = true;
    } else {
      prev_node_full = false;  // remember that this node is_full, so we only have to check the other child next iter
    }

    prev_iblk = curr_iblk;
  }

  return mem_span.begin() + iblk * block_size_at_level(ilevel);
}

template <std::size_t a, std::size_t b, std::uint8_t c, std::size_t d>
inline void NodelessBuddy<a, b, c, d>::deallocate(std::byte* ptr) {
  int prev_iblk = (ptr - mem_span.begin()) / min_block_size;

  // unset everything for the min-size block
  is_full_bit_tree.get(num_levels - 1, prev_iblk) = false;
  is_used_bit_tree.get(num_levels - 1, prev_iblk) = false;
  bool prev_node_is_used = false;

  // walk up tree starting at second-lowest level. (lowest level handled above)
  // clear is_full for all parents. And, clear is_used if both children are now not is_used
  for (LevelIndex plevel = num_levels - 2; plevel >= 0; --plevel) {
    const auto prev_node_sibling_iblk = prev_iblk + (prev_iblk % 2 ? -1 : 1);
    const auto curr_iblk = prev_iblk / 2;
    const auto child_level = plevel + 1;
    const bool prev_node_sibling_is_used = is_used_bit_tree.get(child_level, prev_node_sibling_iblk);

    // all bit setting is here
    is_full_bit_tree.get(plevel, curr_iblk) = false;
    if (not prev_node_is_used && not prev_node_sibling_is_used) {
      is_used_bit_tree.get(plevel, curr_iblk) = false;
    } else {
      prev_node_is_used = true;  // remember that this node is_used, so we only have to check the other child next iter
    }

    prev_iblk = curr_iblk;
  }
}

}  // end namespace allocators
