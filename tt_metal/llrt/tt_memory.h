// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <cassert>
#include <cstdint>
#include <istream>
#include <sstream>
#include <string>
#include <vector>

// #include "command_assembler/memory.h"

namespace ll_api {

class memory {
 public:
  typedef std::uint64_t address_t;
  typedef std::uint32_t word_t;

 private:
  static constexpr uint32_t initial_data_space_ = 0x400;
  static constexpr uint32_t initial_span_space_ = 4;

  struct span {
      // Note: the offset of the data for a span in data_ is generated on the
      // fly by processing spans in order
      address_t addr;    // byte address in device memory
      size_t len;
      bool operator==(const span& other) const { return addr == other.addr && len == other.len; }
  };

  std::vector<word_t> data_;
  std::vector<struct span> link_spans_;
  uint32_t text_size_;
  uint32_t packed_size_;

 public:
  memory();
  memory(std::string const &path);

 public:
  const std::vector<word_t>& data() const { return this->data_; }

  // memory& operator=(memory &&src);
  bool operator==(const memory& other) const;

  void set_text_size(uint32_t size) { this->text_size_ = size; }
  void set_packed_size(uint32_t size) { this->packed_size_ = size; }
  uint32_t get_text_size() const { return this->text_size_; }
  uint32_t get_packed_size() const { return this->packed_size_; }

  size_t size() const { return data_.size(); }

  size_t num_spans() const { return link_spans_.size(); }

private:
  // Read from file
  void fill_from_discontiguous_hex(std::string const &path);

public:
  // Process spans in arg mem to fill data in *this (eg, from device)
  void fill_from_mem_template(const memory& mem_template, const std::function<void (std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback);

  // Iterate over spans_ to act on data_ (eg., to device)
  void process_spans(const std::function<void (std::vector<uint32_t>::const_iterator, uint64_t addr, uint32_t len)>& callback) const;
  void process_spans(const std::function<void (std::vector<uint32_t>::iterator, uint64_t addr, uint32_t len)>& callback);

  void pack_data_into_text(std::uint64_t text_start, std::uint64_t data_start);
};

}  // namespace ll_api
