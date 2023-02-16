#pragma once

#include "src/firmware/riscv/common/containers/static_hash_table.hpp"
#include "src/firmware/riscv/common/fw_debug.h"
#include "src/firmware/riscv/common/memory_management/arena_allocator.hpp"
#include "src/firmware/riscv/common/memory_management/buddy_allocator.h"

enum class MemoryType {
  Dram,
  L1,
};

struct MemoryLocation {
  std::byte* ptr;
  MemoryType type;
};

class MemoryManager {
 public:
  using BufferId = std::uint32_t;
  using RefCountUint = std::uint16_t;
  struct BufferInfo {
    std::byte* ptr;
    RefCountUint reference_count;
    bool is_dram;
    // lots of padding, still.
  };

  // Just an assert to let us know if the size changes. If needed, just change it. Nothing should rely on it.
  static_assert(
      sizeof(BufferInfo) == sizeof(std::byte*) * 2, "BufferInfo size changed. Just thought you might want to know");

  // should probably be a constant that numodel and/or LLIR has access to...
  static constexpr std::size_t max_concurrent_buffers = 512;
  static constexpr std::size_t buffer_info_data_size = max_concurrent_buffers * sizeof(BufferInfo);

  MemoryManager(MemorySpan l1_buffer_range, MemorySpan dram_range, MemorySpan l1_private_range) :
      l1_private_alloc(l1_private_range),
      l1_buffer_alloc(l1_buffer_range),
      dram_buffer_alloc(dram_range),
      buf_id_to_address(
          (BufferId*)l1_private_alloc.allocate(sizeof(BufferId[max_concurrent_buffers])),
          (BufferInfo*)l1_private_alloc.allocate(sizeof(BufferInfo[max_concurrent_buffers]))) {
    FWASSERT("l1_buffer_range overlaps l1_private_range", not l1_buffer_range.overlaps_with(l1_private_range));
  }

  /**
   * @brief General allocation function. Puts buffer wherever it will fit, prefering L1.
   */
  MemoryLocation allocate_buffer(BufferId bid, std::size_t size, RefCountUint initial_ref_count) {
    return allocate_buffer_impl(bid, size, initial_ref_count, false);
  }

  /**
   * @brief Records existence of a buffer at a given location, without trying to allocate it
   */
  std::byte *record_buffer(BufferId bid, std::byte *ptr, MemoryType type, RefCountUint initial_ref_count) {
    const bool recorded = buf_id_to_address.insert(bid, {ptr, initial_ref_count, type == MemoryType::Dram}).second;
    FWASSERT("recording the buffer's allocation information failed", recorded);
    (void)recorded;
    return ptr;
  }

  /**
   * @brief Allocate a buffer specifically in DRAM
   */
  MemoryLocation allocate_buffer_in_dram(BufferId bid, std::size_t size, RefCountUint initial_ref_count) {
    return allocate_buffer_impl(bid, size, initial_ref_count, true);
  }

  /**
   * @brief Decrement the reference count for @p bid, and possibly free it if the RC reaches 0
   *
   * Current implementation doesn't reclaim any memory from L1 (or DRAM) unless
   *   there are zero active buffers in L1 (or DRAM), as the L1 (or DRAM) allocator is an arena allocator
   */
  void release_buffer(BufferId bid);

  /**
   * @brief Remove buffer from the buffer map. Do not deallocate or decrement refcount, its assumed to be 1.
   */
  void remove_buffer(BufferId bid) {
    auto& binfo = buf_id_to_address.at(bid);
    buf_id_to_address.erase_iter(&binfo);
  }

  void reset() {
    buf_id_to_address.clear();
    l1_buffer_alloc.reset();
    dram_buffer_alloc.reset();
  }

  std::byte* get_buffer_pointer(BufferId bid) const { return buf_id_to_address.at(bid).ptr; }

  MemoryType get_buffer_type(BufferId bid) const { return buf_id_to_address.at(bid).is_dram ? MemoryType::Dram : MemoryType::L1; }

 private:
  /**
   * @brief Implementation "god" function that handles all allocation requests.
   */
  MemoryLocation allocate_buffer_impl(BufferId bid, std::size_t size, RefCountUint initial_ref_count, bool skip_l1);

 public:
  using L1PrivateAllocator = allocators::Arena<AlignmentBits::B8>;
  //using L1BufferAllocator = allocators::NodelessBuddy<10, 512 + 32, AlignmentBits::B16, 2>;  // 544B is a 1-face buffer
  using L1BufferAllocator = allocators::NodelessBuddy<10, 512 + 16, AlignmentBits::B16, 2>;  // 512+16 is more efficient for larger buffers
  using DRAMBufferAllocator = allocators::NodelessBuddy<10, 4096, AlignmentBits::B32>;
  using BufferInfoTable =
      StaticHashTable<BufferId, BufferInfo, max_concurrent_buffers, ExternalArray<BufferInfo>, ExternalArray<BufferId>>;

 private:
  L1PrivateAllocator l1_private_alloc;
  L1BufferAllocator l1_buffer_alloc;
  DRAMBufferAllocator dram_buffer_alloc;
  BufferInfoTable buf_id_to_address;
};

inline void MemoryManager::release_buffer(BufferId bid) {
  auto& binfo = buf_id_to_address.at(bid);
  FWASSERT("reference count already zero?", binfo.reference_count != 0);
  --binfo.reference_count;

  if (binfo.reference_count == 0) {
    if (not binfo.is_dram) {
      FWASSERT_NSE("L1 allocator does not own that address", l1_buffer_alloc.owns_address(binfo.ptr));
      l1_buffer_alloc.deallocate(binfo.ptr);
    } else {
      FWASSERT_NSE("DRAM allocator does not own that address", dram_buffer_alloc.owns_address(binfo.ptr));
      dram_buffer_alloc.deallocate(binfo.ptr);
    }
    buf_id_to_address.erase_iter(&binfo);
  }
}

inline MemoryLocation MemoryManager::allocate_buffer_impl(
    BufferId bid, std::size_t size, RefCountUint initial_ref_count, bool skip_l1) {
  FWASSERT_NSE("already allocated", buf_id_to_address.find(bid) == nullptr);
  MemoryLocation result = {nullptr, MemoryType::L1};
  if (not skip_l1) {
    result = {l1_buffer_alloc.allocate(size), MemoryType::L1};
  }
  if (not result.ptr) {
    result = {dram_buffer_alloc.allocate(size), MemoryType::Dram};
  }

  const bool recorded =
      buf_id_to_address.insert(bid, {result.ptr, initial_ref_count, result.type == MemoryType::Dram}).second;
  FWASSERT("recording the buffer's allocation information failed", recorded);
  (void)recorded;

  #ifdef DUMP_FAILED_ALLOCATION
  // Turn on to dump json file to see the current memory allocation in reportify at the fime of failure
  if (not result.ptr) {
    printf("Failed to allocate %lu bytes for buffer id %d. Paste the following into emule_memory_dynamic_analysis.json for the test to view in reportify.\n", size, bid);
    printf("{%s}", core->get_memory_analysis_json().c_str());
  }
  #endif
  FWASSERT("block could not be allocated in L1 or DRAM", result.ptr);  // or make the caller handle it?
  return result;
}
