#pragma once

#include <cassert>
#include <cstdint>
#include <istream>
#include <sstream>
#include <ostream>
#include <string>
#include <vector>

// #include "command_assembler/memory.h"

namespace ll_api {

class memory {
 public:
  typedef enum {
    BRISC,
    TRISC0,
    TRISC1,
    TRISC2,
    NCRISC,
  } risc_type_t;
  typedef std::uint32_t address_t;
  typedef std::uint32_t word_t;
  memory() : memory(0, 0){};
  memory(address_t base, address_t size);
  memory(address_t base, const std::vector<word_t>& content);
  memory(address_t base, std::vector<word_t>&& content);

  void resize(std::size_t num_words) { m_content.resize(num_words, 0); }

  /**
   * @brief Read from @p is to create a memory that is exactly the size of the hex.
   *
   * Multiple @ lines will cause an exception to be thrown.
   */
  static memory from_contiguous_hex(std::istream& is);
  static memory load_hex(std::istream& is) { return from_contiguous_hex(is); }

  /**
   * @brief Read from @p is to create a memory that encompasses all assigned addresses. Other locations will be 0.
   *
   * Can have as many @ lines as necessary.
   */
  static memory from_discontiguous_hex(std::istream& is);
  static memory from_discontiguous_risc_hex(std::istream& is, risc_type_t risc_type);

  void load_relative_hex(const std::string& filename);

  // in bytes
  address_t base() const { return m_base; }
  address_t size() const { return count() * sizeof(word_t); }
  address_t limit() const { return base() + size(); }

  // in words
  address_t base_word() const { return base() / sizeof(word_t); }
  address_t count() const { return m_content.size(); }
  address_t limit_word() const { return base_word() + count(); }

  const word_t& operator[](address_t i) const { return m_content.at(i); }
  word_t& operator[](address_t i) { return m_content.at(i); }

  word_t* data() { return m_content.data(); }
  const word_t* data() const { return m_content.data(); }

  void write_hex(std::ostream& os) const;

  auto consume() && { return std::move(m_content); }
  auto contents() { return m_content; }
  std::vector<word_t> get_content() { return m_content; }

  bool operator==(const memory& rhs) const { return this->m_base == rhs.m_base && this->m_content == rhs.m_content; }
  friend std::ostream& operator<<(std::ostream& os, const memory& mem) {
    mem.write_hex(os);
    return os;
  }

  std::string to_string() {
    std::ostringstream s;
    for (int index = 0; index < count(); index++) {
      // s << boost::format("%1$#x: %2$#x\n") % (base() + (index * 4)) % m_content.at(index);
      s << std::hex << "0x" << (base() + (index * 4)) << ": 0x" << m_content.at(index) << std::dec << std::endl;
    }
    return s.str();
  }

  // // TODO: Remove CA Translation
  // operator CommandAssembler::memory() const { return CommandAssembler::memory(m_base, m_content); }

 private:
  address_t m_base;
  std::vector<word_t> m_content;

  void assert_invariants() const;
};

inline memory::address_t WriteBlob(
    memory& mem, memory::address_t start, const memory::word_t* data, long long unsigned count) {
  assert(mem.base() <= start);
  const auto one_past_end = start + count * sizeof(memory::word_t);
  assert(one_past_end <= mem.limit());
  const auto offset_from_base = start - mem.base();
  for (int idx = offset_from_base / sizeof(memory::word_t), span_pos = 0; span_pos < count; ++idx, ++span_pos) {
    mem[idx] = data[span_pos];
  }
  return one_past_end;
}

template <typename SpanLike>
auto WriteBlob(memory& mem, memory::address_t start, const SpanLike& data_span) {
  return WriteBlob(mem, start, data_span.data(), data_span.size());
}

memory slice_memory(const memory& mem, memory::address_t base, memory::address_t start, memory::address_t end);

}  // namespace ll_api
