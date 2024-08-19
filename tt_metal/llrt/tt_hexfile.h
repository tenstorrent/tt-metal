// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <istream>
#include <ostream>
#include <vector>

#include "tt_memory.h"

namespace ll_api {

// A "contiguous" hex file has no @address lines, just data lines.
std::vector<memory::word_t> read_contiguous_hex_file(std::istream& input);

// Reads a contiguous hex file (see above), passing each element to the callback.
// The address passed to the callback is the data index + base. Data is passed in increasing address order.
// The return value is the highest address passed to the callback + 1.
memory::address_t read_contiguous_hex_file(
    std::istream&, const std::function<void(memory::address_t, memory::word_t)>& callback, memory::address_t base = 0);

memory::address_t read_discontiguous_hex_file(
    std::istream& input, const std::function<void(memory::address_t, memory::word_t)>& callback);

// A discontiguous hex file has @address lines that set the address for the next line.
// It must start with an address, and an address must be followed by at least one data line.
// Addresses must be strictly increasing.
class discontiguous_hex_file_writer {
 public:
  // Until destroyed, this object owns the state of output.
  explicit discontiguous_hex_file_writer(std::ostream& output);

  void add(memory::address_t address, memory::word_t value);

  template <class Iterator>
  inline void add(memory::address_t start_address, Iterator first, Iterator last);

  void end() {}

 private:
  std::ostream& output;

  bool first = true;
  memory::address_t last_address;
};

template <class Iterator>
inline void discontiguous_hex_file_writer::add(memory::address_t start_address, Iterator first, Iterator last) {
  while (first != last) add(start_address++, *first++);
}

}  // namespace ll_api
