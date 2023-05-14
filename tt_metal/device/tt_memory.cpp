
#include <cassert>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "tensix.h"
#include "tt_memory.h"
#include "tt_hexfile.h"
#include <l1_address_map.h>

using std::numeric_limits;
using std::runtime_error;
using std::string;
using std::vector;

namespace ll_api {

// We use stoul to parse an address. We cast address_t to unsigned long to format with %08lX.
static_assert(
    numeric_limits<unsigned long>::max() >= numeric_limits<memory::address_t>::max(),
    "unsigned long can't cover whole range of addresses");

// We cast word_t to unsigned long to format with %08lX.
static_assert(
    numeric_limits<unsigned long>::max() >= numeric_limits<memory::word_t>::max(),
    "unsigned long can't cover whole range of words");

memory::memory(address_t base, address_t size) : m_base(base), m_content(size / sizeof(word_t)) {
  assert(size % sizeof(word_t) == 0);
  assert_invariants();
}

memory::memory(address_t base, const std::vector<word_t> &content) : m_base(base), m_content(content) {
  assert_invariants();
}

memory::memory(address_t base, std::vector<word_t> &&content) : m_base(base), m_content(std::move(content)) {
  assert_invariants();
}

memory memory::from_contiguous_hex(std::istream &in) {
  string line;
  if (!getline(in, line))
    throw runtime_error("Empty memory file.");

  if (line.size() == 0 || line[0] != '@')
    throw runtime_error("Memory file does not start with address line.");

  unsigned long base = stoul(string(line.begin() + 1, line.end()), nullptr, 16);
  if (base > numeric_limits<address_t>::max() / sizeof(word_t))
    throw runtime_error("Memory file base address is too high.");

  vector<word_t> content = read_contiguous_hex_file(in);

  if (content.size() > numeric_limits<address_t>::max() / sizeof(word_t) + 1 - base)
    throw runtime_error("Memory file wraps the address space.");

  return memory(base * sizeof(word_t), std::move(content));
}

memory memory::from_discontiguous_hex(std::istream &in) {
  const auto word_size = sizeof(memory::word_t);
  memory result(0, 0);
  read_discontiguous_hex_file(in, [&](memory::address_t word_addr, memory::word_t value) {
    if (result.size() == 0) {  // first call
      result = memory(word_addr * word_size, word_size);
    }

    if (result.limit_word() <= word_addr) {
      result.resize(word_addr - result.base_word() + 1);
    }
    result[word_addr - result.base_word()] = value;
  });
  return result;
}

memory memory::from_discontiguous_risc_hex(std::istream& in, memory::risc_type_t risc_type) {
  const auto word_size = sizeof(memory::word_t);

  bool is_trisc = false;
  bool is_brisc = false;
  bool is_ncrisc = false;
  unsigned int risc_l1_local_memory_base_addr = 0x0;

  switch (risc_type) {
     case TRISC0: is_trisc=true;
           risc_l1_local_memory_base_addr = l1_mem::address_map::TRISC0_LOCAL_MEM_BASE >> 2;
           break;
     case TRISC1: is_trisc=true;
           risc_l1_local_memory_base_addr = l1_mem::address_map::TRISC1_LOCAL_MEM_BASE >> 2;
           break;
     case TRISC2: is_trisc=true;
           risc_l1_local_memory_base_addr = l1_mem::address_map::TRISC2_LOCAL_MEM_BASE >> 2;
           break;
     case NCRISC: is_ncrisc=true;
           risc_l1_local_memory_base_addr = l1_mem::address_map::NCRISC_LOCAL_MEM_BASE >> 2;
           break;
     case BRISC: is_brisc=true;
           risc_l1_local_memory_base_addr = l1_mem::address_map::BRISC_LOCAL_MEM_BASE >> 2;
           break;
     default:
           throw runtime_error("Invalid risc core type.");
  }

  memory result(0, 0);
  read_discontiguous_hex_file(in, [&](memory::address_t word_addr, memory::word_t value) {
    if (result.size() == 0) {  // first call
      // Remap local memory address to temp location in l1
      if ((is_trisc || is_ncrisc) && (((word_addr << 2) & 0xfff00000) == MEM_LOCAL_BASE)) {
        word_addr = (word_addr & ((l1_mem::address_map::MAX_L1_LOADING_SIZE >> 2) - 1)) + risc_l1_local_memory_base_addr;
      } else if ((is_ncrisc) && (((word_addr << 2) & 0xfff00000) == MEM_NCRISC_IRAM_BASE)) {
        word_addr = (word_addr & ((l1_mem::address_map::MAX_L1_LOADING_SIZE >> 2) - 1)) + (MEM_NCRISC_FIRMWARE_BASE >> 2);
      }
      if (is_ncrisc) {
        // For ncrisc, first block of hex data is L1 resident and goes @ address 0x9000. But the base address for ncrisc
        // hex image in L1 has to be 0x5000 which maps to l1_mem::address_map::NCRISC_FIRMWARE_BASE and is 16 KB in size
        // from 0x5000 to 0x8FFF in L1. Followed by L1 residend ncrisc code @ 0x9000 in L1.
        result = memory(MEM_NCRISC_FIRMWARE_BASE, word_size);
      } else {
        result = memory(word_addr * word_size, word_size);
      }
    }
    // Remap local memory address from hex into location in l1 where copy of the local  memory content resides.
    // During *risc init firmware will copy data from l1 into local memory
    // During brisc init firmware will use DMA to copy ncrisc firmware to l1 to iram
    if (((word_addr << 2) & 0xfff00000) == MEM_LOCAL_BASE) {
       word_addr = (word_addr & ((l1_mem::address_map::MAX_L1_LOADING_SIZE >> 2) - 1)) + risc_l1_local_memory_base_addr;
    } else if ((is_ncrisc) && (((word_addr << 2) & 0xfff00000) == MEM_NCRISC_IRAM_BASE)) {
       word_addr = (word_addr & ((l1_mem::address_map::MAX_L1_LOADING_SIZE >> 2) - 1)) + (MEM_NCRISC_FIRMWARE_BASE >> 2);
    }

    if (result.limit_word() <= word_addr) {
      result.resize(word_addr - result.base_word() + 1);
    }
    result[word_addr - result.base_word()] = value;
  });
  return result;
}

void memory::load_relative_hex(const std::string &filename) {
  std::ifstream input(filename);

  read_discontiguous_hex_file(input, [this](memory::address_t addr, memory::word_t value) {
    if (addr > m_content.capacity())
      throw runtime_error("Attempt to load relative hex file that is bigger than the memory.");

    m_content[addr] = value;
  });
}

void memory::write_hex(std::ostream &os) const {
  discontiguous_hex_file_writer writer(os);

  writer.add(m_base / sizeof(word_t), m_content.begin(), m_content.end());
}

void memory::assert_invariants() const {
  assert(m_base % sizeof(word_t) == 0);
  assert(m_content.size() <= (numeric_limits<address_t>::max() - m_base) / sizeof(word_t) + 1);
}

memory slice_memory(const memory &mem, memory::address_t base, memory::address_t start, memory::address_t end) {
  assert(mem.base() <= start);
  assert(mem.limit() >= end);
  assert(((end - start) % sizeof(memory::word_t)) == 0);

  auto num_words_to_copy = (end - start) / sizeof(memory::word_t);
  auto starting_word = (start - base) / sizeof(memory::word_t);
  assert(starting_word >= 0);

  auto contents = std::vector<memory::word_t>();
  for (int i = 0; i < num_words_to_copy; i++) {
    contents.push_back(mem.data()[starting_word + i]);
  }
  return memory(base, contents);
}

}  // namespace ll_api
