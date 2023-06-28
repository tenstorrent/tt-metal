
#pragma once

#include <string>
#include <regex>
#include <set>
#include <utility>
#include <optional>

#include "common/assert.hpp"
// #include <boost/functional/hash.hpp>
// #include <command_assembler/xy_pair.h>

using std::pair;

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

constexpr inline bool operator<=(const CoreCoord &a, const CoreCoord &b) {
    return (a < b) or (a == b);
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

    CoreRange(CoreCoord start, CoreCoord end) {
        tt::log_assert(
            end.x >= start.x and end.y >= start.y,
            "Invalid core range for start: {}, end: {}", start.str(), end.str());

        this->start = start;
        this->end = end;
    }

    std::optional<CoreRange> intersects ( const CoreRange & other ) const
    {
        std::size_t x1 = std::max(this->start.x, other.start.x);
        std::size_t y1 = std::max(this->start.y, other.start.y);
        std::size_t x2 = std::min(this->end.x, other.end.x);
        std::size_t y2 = std::min(this->end.y, other.end.y);
        if (x1< x2 and y1< y2)
            return CoreRange( {x1, y1}, {x2, y2} );

        return {};
    }

    bool contains ( const CoreRange & other ) const
    {
        return this->start.x <= other.start.x <= this->end.x &&
                this->start.x <= other.end.x <= this->end.x &&
                this->start.y <= other.start.y <= this->end.y &&
                this->start.y <= other.end.y <= this->end.y;
    }

    // Merge adjacent lined-up rectangles
    std::optional<CoreRange> merge ( const CoreRange & cr) const
    {
        if ( this->start.x == cr.start.x && this->end.x == cr.end.x){
            if (this->end.y == cr.start.y ) return std::optional<CoreRange> {CoreRange ( {this->start.x, this->start.y}, {this->end.x, cr.end.y})};
            if ( this->start.y == cr.end.y) return std::optional<CoreRange> { CoreRange ( {this->start.x, cr.start.y}, {this->end.x, this->end.y})};
        }
        else if ( this->start.y == cr.start.y && this->end.y == cr.end.y){
            if (this->end.x == cr.start.x ) return std::optional<CoreRange> { CoreRange ( {this->start.x, this->start.y}, {cr.end.x, cr.end.y})};
            if ( this->start.x == cr.end.x) return std::optional<CoreRange> { CoreRange ( {cr.start.x, cr.start.y}, {this->end.x, this->end.y})};
        }

        return std::nullopt;
    }

    std::set<CoreRange> diff ( const CoreRange & cr) const
    {
        auto irect = this->intersects(cr);
        if (!irect.has_value())
            return {*this};

        std::set<size_t> xs = {this->start.x, this->end.x};
        std::set<size_t> ys = {this->start.y, this->end.y};

        if ( this->start.x < cr.start.x < this->end.x ) xs.insert(cr.start.x);
        if ( this->start.x < cr.end.x < this->end.x) xs.insert(cr.end.x);
        if ( this->start.y < cr.start.y < this->end.y ) ys.insert(cr.start.y);
        if ( this->start.y < cr.end.y < this->end.y ) ys.insert(cr.end.y);

        std::vector<size_t> vxs(xs.begin(), xs.end());
        std::vector<size_t> vys(ys.begin(), ys.end());
        std::set<CoreRange> ret;
        for (unsigned i = 0; i < vxs.size()-1; i++){
            for (unsigned j = 0; j < vys.size()-1; j++){
                CoreRange r( {vxs[i],vys[i]}, {vxs[i+1], vys[i+1]});
                if (r.start != irect.value().start || r.end != irect.value().end )
                  ret.insert(r);
            }
        }
        return ret;
    }

    std::string str() const { return "[" + start.str() + " - " + end.str() + "]"; }

    size_t size() const { return (this->end.x - this->start.x + 1) * (this->end.y - this->start.y + 1); }
};

constexpr inline bool operator==(const CoreRange &a, const CoreRange &b) { return a.start == b.start && a.end == b.end; }

constexpr inline bool operator!=(const CoreRange &a, const CoreRange &b) { return !(a == b); }

constexpr inline bool operator<(const CoreRange &left, const CoreRange &right) {
  return (left.start < right.start || (left.start == right.start && left.end < right.end));
}




struct CoresInCoreRangeGenerator {
    CoreCoord current;
    CoreCoord end;
    int num_worker_cores_x;
    int num_worker_cores_y;

    CoresInCoreRangeGenerator(const CoreRange& core_range, const CoreCoord& worker_grid_size) {
        this->current = core_range.start;
        this->end = core_range.end;

        this->num_worker_cores_x = worker_grid_size.x;
        this->num_worker_cores_y = worker_grid_size.y;
    }

    pair<CoreCoord, bool> operator() () {
        CoreCoord coord = this->current;
        CoreCoord new_coord;

        new_coord.x = (coord.x + 1) % this->num_worker_cores_x;
        new_coord.y = coord.y + (new_coord.x == 0); // It means we moved to next row

        this->current = new_coord;

        bool terminate = this->end == coord;

        return {coord, terminate};
    }
};

class CoreRangeSet {
  public:
    CoreRangeSet(const std::set<CoreRange> &core_ranges) {
      this->merge( core_ranges);
    }



    void merge ( const std::set<CoreRange> & core_ranges ){
      std::vector<CoreRange> vcr (core_ranges.begin(), core_ranges.end() );
      vcr.insert(vcr.end(), ranges_.begin(), ranges_.end());
      std::vector<CoreRange> diffs;

      //Diff pair-wise CoreRanges
      for ( unsigned i = 0; i < vcr.size(); i++){
        for (unsigned j = i+1; j < vcr.size(); j++){

          auto d1 = vcr[i].diff(vcr[j]);
          auto d2 = vcr[j].diff(vcr[i]);

          if (d1.size() < d2.size() )
          {
            diffs.insert(diffs.end(), d1.begin(), d1.end());
            diffs.push_back( vcr[j]);
          }
          else{
            diffs.insert(diffs.end(), d2.begin(), d2.end());
            diffs.push_back( vcr[i]);
          }
        }
      }

      // Remove subsumed CoreRanges, and also try to merge adjacent CoreRanges
      std::sort( diffs.begin(), diffs.end() );
      for (auto it1 = diffs.begin(); it1 != diffs.end(); it1++){
        for (auto it2 = diffs.begin(); it2 != diffs.end(); ){
          if ( it1 == it2 ) continue;
          else if ( it1->contains(*it2)){
            it2 = diffs.erase(it2);
          }
          else if (it2->contains(*it1)){
            *it1 = *it2;
            it2 = diffs.erase(it2);
          }
          else if ( auto merged = it1->merge(*it2) ){
            *it1 = merged.value();
            it2 = diffs.erase(it2);
          }
          else{
            ++it2;
          }
        }
      }
      ranges_ = {diffs.begin(), diffs.end()};
    }

    void merge ( const CoreRangeSet & s )
    {
      this->merge (s.ranges());
    }

    bool core_coord_in_core_ranges(const CoreCoord &core_coord) const {
      for (auto core_range : this->ranges_) {
        bool in_x_range = (core_coord.x >= core_range.start.x) and (core_coord.x <= core_range.end.x);
        bool in_y_range = (core_coord.y >= core_range.start.y) and (core_coord.y <= core_range.end.y);
        if (in_x_range and in_y_range) {
          return true;
        }
      }
      return false;
    }

    std::set<CoreRange> ranges() const { return this->ranges_; }

  private:
    std::set<CoreRange> ranges_;
};

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
