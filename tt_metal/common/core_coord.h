
#pragma once

#include <string>
#include <regex>
// #include <boost/functional/hash.hpp>
// #include <command_assembler/xy_pair.h>


struct CoreCoord {

  constexpr CoreCoord() : x{}, y{} {}
  constexpr CoreCoord(std::size_t x, std::size_t y) : x(x), y(y) {}
  // explicit CoreCoord(const CommandAssembler::xy_pair &p) : CoreCoord(p.x, p.y) {}

  std::size_t x;
  std::size_t y;

  std::string str() const { return "(x=" + std::to_string(x) + ",y=" + std::to_string(y) + ")"; }

  // TODO: Remove CA Translation
  // operator CommandAssembler::xy_pair() const { return CommandAssembler::xy_pair(x, y); }

};

constexpr inline bool operator==(const CoreCoord &a, const CoreCoord &b) { return a.x == b.x && a.y == b.y; }

constexpr inline bool operator!=(const CoreCoord &a, const CoreCoord &b) { return !(a == b); }

constexpr inline bool operator<(const CoreCoord &left, const CoreCoord &right) {
  return (left.x < right.x || (left.x == right.x && left.y < right.y));
}


struct tt_cxy_pair : public CoreCoord {

  tt_cxy_pair() : CoreCoord{}, chip{} {}
  tt_cxy_pair(std::size_t ichip, CoreCoord xy_pair) : CoreCoord(xy_pair.x, xy_pair.y), chip(ichip) {}
  tt_cxy_pair(std::size_t ichip, std::size_t x, std::size_t y) : CoreCoord(x,y), chip(ichip) {}

  std::size_t chip;

  std::string str() const { return "(chip=" + std::to_string(chip) + ",x=" + std::to_string(x) + ",y=" + std::to_string(y) + ")"; }
};

constexpr inline bool operator==(const tt_cxy_pair &a, const tt_cxy_pair &b) { return a.x == b.x && a.y == b.y && a.chip == b.chip; }

constexpr inline bool operator!=(const tt_cxy_pair &a, const tt_cxy_pair &b) { return !(a == b); }

constexpr inline bool operator<(const tt_cxy_pair &left, const tt_cxy_pair &right) {
  return (left.chip < right.chip || (left.chip == right.chip && left.x < right.x) || (left.chip == right.chip && left.x == right.x && left.y < right.y));
}

namespace std {
template <>
struct hash<CoreCoord> {
  std::size_t operator()(CoreCoord const &o) const {
    std::size_t seed = 0;
    seed = std::hash<std::size_t>()(o.x) ^ std::hash<std::size_t>()(o.y) << 1;
    return seed;
  }
};
}

namespace std {
template <>
struct hash<tt_cxy_pair> {
  std::size_t operator()(tt_cxy_pair const &o) const {
    std::size_t seed = 0;
    seed = std::hash<std::size_t>()(o.chip) ^ (std::hash<std::size_t>()(o.x) << 1) ^ (std::hash<std::size_t>()(o.y) << 2);
    return seed;
  }
};
}
