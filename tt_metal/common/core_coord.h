
#pragma once

#include <string>
#include <regex>
#include <set>
// #include <boost/functional/hash.hpp>
// #include <command_assembler/xy_pair.h>


struct CoreCoord {

  constexpr CoreCoord() : x{}, y{} {}
  constexpr CoreCoord(std::size_t x, std::size_t y) : x(x), y(y) {}
  // explicit CoreCoord(const CommandAssembler::xy_pair &p) : CoreCoord(p.x, p.y) {}

  std::size_t x = 0;
  std::size_t y = 0;

  std::string str() const { return "(x=" + std::to_string(x) + ",y=" + std::to_string(y) + ")"; }

};

constexpr inline bool operator==(const CoreCoord &a, const CoreCoord &b) { return a.x == b.x && a.y == b.y; }

constexpr inline bool operator!=(const CoreCoord &a, const CoreCoord &b) { return !(a == b); }

constexpr inline bool operator<(const CoreCoord &left, const CoreCoord &right) {
  return (left.x < right.x || (left.x == right.x && left.y < right.y));
}

struct RelativeCoreCoord {
  long x = 0;
  long y = 0;

  std::string str() const { return "(x=" + std::to_string(x) + ",y=" + std::to_string(y) + ")"; }

};

constexpr inline bool operator==(const RelativeCoreCoord &a, const RelativeCoreCoord &b) { return a.x == b.x && a.y == b.y; }

constexpr inline bool operator!=(const RelativeCoreCoord &a, const RelativeCoreCoord &b) { return !(a == b); }

inline CoreCoord get_core_coord_from_relative(const RelativeCoreCoord& in, const CoreCoord& grid_size) {
  CoreCoord coord;
  coord.x = in.x + ((in.x < 0)? grid_size.x : 0);
  coord.y = in.y + ((in.y < 0)? grid_size.y : 0);
  return coord;
}

struct tt_cxy_pair : public CoreCoord {

  tt_cxy_pair() : CoreCoord{}, chip{} {}
  tt_cxy_pair(std::size_t ichip, CoreCoord xy_pair) : CoreCoord(xy_pair.x, xy_pair.y), chip(ichip) {}
  tt_cxy_pair(std::size_t ichip, std::size_t x, std::size_t y) : CoreCoord(x,y), chip(ichip) {}

  std::size_t chip = 0;

  std::string str() const { return "(chip=" + std::to_string(chip) + ",x=" + std::to_string(x) + ",y=" + std::to_string(y) + ")"; }
};

constexpr inline bool operator==(const tt_cxy_pair &a, const tt_cxy_pair &b) { return a.x == b.x && a.y == b.y && a.chip == b.chip; }

constexpr inline bool operator!=(const tt_cxy_pair &a, const tt_cxy_pair &b) { return !(a == b); }

constexpr inline bool operator<(const tt_cxy_pair &left, const tt_cxy_pair &right) {
  return (left.chip < right.chip || (left.chip == right.chip && left.x < right.x) || (left.chip == right.chip && left.x == right.x && left.y < right.y));
}

struct CoreRange {
  CoreCoord start;
  CoreCoord end;

  std::string str() const { return "[" + start.str() + " - " + end.str() + "]"; }
};

constexpr inline bool operator==(const CoreRange &a, const CoreRange &b) { return a.start == b.start && a.end == b.end; }

constexpr inline bool operator!=(const CoreRange &a, const CoreRange &b) { return !(a == b); }

constexpr inline bool operator<(const CoreRange &left, const CoreRange &right) {
  return (left.start < right.start || (left.start == right.start && left.end < right.end));
}

using CoreRangeSet = std::set<CoreRange>;

inline bool core_coord_in_core_range_set(const CoreRangeSet &core_ranges, const CoreCoord &core_coord) {
  for (auto core_range : core_ranges) {
    bool in_x_range = (core_coord.x >= core_range.start.x) and (core_coord.x <= core_range.end.x);
    bool in_y_range = (core_coord.y >= core_range.start.y) and (core_coord.y <= core_range.end.y);
    if (in_x_range and in_y_range) {
      return true;
    }
  }
  return false;
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
template <>
struct hash<RelativeCoreCoord> {
  std::size_t operator()(RelativeCoreCoord const &o) const {
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
