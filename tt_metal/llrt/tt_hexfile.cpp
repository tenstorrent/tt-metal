// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


// #include <boost/spirit/include/qi.hpp>
#include <cassert>
#include <iomanip>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <iostream>

#include "tt_hexfile.h"
#include "common/tt_rounding.h"

using namespace std;
using namespace ll_api;

#define USE_REGEX_PARSER 1

namespace {

// Select between std::regex parsing and boost::spirit parsing.
// libc++ (llvm c++ standard library) regex is very slow.

unsigned int address_hex_digits = round_up_div(std::numeric_limits<memory::address_t>::digits, 4);
unsigned int data_hex_digits = round_up_div(std::numeric_limits<memory::word_t>::digits, 4);

// Hex input consists of elements of optional @ followed by optional 0x followed by 1-8 hex digits, optionally followed
// by a comma. The @ denotes an address line, which is only permitted in certain cases.

// The optional @ is captured (so its presence can be detected) and the hex digits are captured.
bool parse_hex_line(const std::string& line, bool* seen_at, memory::address_t* hex_address) {

#ifdef USE_REGEX_PARSER

    // captures:                  (1 )          (2                )
    static const regex re(R"__(\s*(@?)(?:0[Xx])?([[:xdigit:]]{1,8})\s*,?\s*)__");

    smatch m;
    if (!regex_match(line, m, re))
      return false;

    *seen_at = (m.length(1) != 0);
    *hex_address = stoul(m[2], nullptr, 16);

    return true;

#else
    using boost::spirit::ascii::char_;
    using boost::spirit::ascii::space;
    using boost::spirit::ascii::xdigit;
    using boost::spirit::qi::copy;
    using boost::spirit::qi::lexeme;
    using boost::spirit::qi::matches;
    using boost::spirit::qi::omit;
    using boost::spirit::qi::parse;
    using boost::spirit::qi::repeat;
    using boost::spirit::qi::uint_parser;

    auto parser = copy(
        omit[*space] >> matches[char_('@')] >> omit[-(char_('0') >> char_("Xx"))] >>
        uint_parser<memory::address_t, 16, 1, 8>() >> omit[*space] >> omit[-char_(',')] >> omit[*space]);

    auto first = line.begin();
    auto last = line.end();

    bool good_parse = parse(first, last, parser, *seen_at, *hex_address);

    return (good_parse && first == last);
#endif

}

template <class Function>
void read_contiguous_hex_file_impl(std::istream& input, Function&& callback) {
  string line;

  while (getline(input, line)) {
    bool seen_at = false;
    memory::address_t hex_address;

    if (!parse_hex_line(line, &seen_at, &hex_address))
      throw std::runtime_error("Memory image has unreadable data line.");

    if (seen_at)
      throw std::runtime_error("Memory file has multiple base addresses.");

    callback(hex_address);
  }
}

}  // namespace

namespace ll_api {

std::vector<memory::word_t> read_contiguous_hex_file(std::istream& input) {
  vector<memory::word_t> content;

  read_contiguous_hex_file_impl(input, [&content](memory::word_t value) { content.push_back(value); });

  return content;
}

memory::address_t read_contiguous_hex_file(
    std::istream& input,
    const std::function<void(memory::address_t, memory::word_t)>& callback,
    memory::address_t base) {
  memory::address_t current_address = base;

  read_contiguous_hex_file_impl(
      input, [&callback, &current_address](memory::word_t value) { callback(current_address++, value); });

  return current_address;
}

memory::address_t read_discontiguous_hex_file(
    std::istream& input, const std::function<void(memory::address_t, memory::word_t)>& callback) {
  string line;
  bool seen_at;
  memory::address_t addr;

  getline(input, line);
  if (input.bad() || input.fail()) {
    throw std::runtime_error(
        "Problem getting initial line from hexfile stream. It may be reading a file that doesn't exist or ulimit -n too low.");
  }
  if (line.empty() && input.eof()) {
    // Allow empty files.
    return 0;
  }

  if (!parse_hex_line(line, &seen_at, &addr) || !seen_at)
    throw std::runtime_error("Memory image does not start with address line.");

  while (getline(input, line)) {
    memory::address_t value;
    if (!parse_hex_line(line, &seen_at, &value))
      throw std::runtime_error("Memory image has unreadable line.");

    if (seen_at) {
      memory::address_t new_addr = value;

      if (new_addr < addr) {
        cout << "new_addr = " << std::hex << new_addr << ", addr = " << addr << std::dec << endl;
        throw std::runtime_error("Memory image address goes backwards.");
      }


      addr = new_addr;
    } else {
      callback(addr++, value);
    }
  }

  return addr;
}

discontiguous_hex_file_writer::discontiguous_hex_file_writer(std::ostream& output) : output(output) {
  output << std::hex << std::uppercase << std::noshowbase << std::noshowpos << std::right << std::setfill('0')
         << std::setw(1);
}

void discontiguous_hex_file_writer::add(memory::address_t address, memory::word_t value) {
  assert(first || address > last_address);

  if (first || last_address + 1 != address) {
    first = false;

    output << '@' << std::setw(address_hex_digits) << address << '\n';
  }

  last_address = address;

  output << std::setw(data_hex_digits) << value << '\n';
}

}  // namespace ll_api
