/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <string>
#include <regex>
#include <set>
#include <utility>
#include <optional>

#include "common/assert.hpp"
#include "third_party/umd/device/tt_xy_pair.h"

// #include <boost/functional/hash.hpp>
// #include <command_assembler/xy_pair.h>

using std::pair;

using CoreCoord = tt_xy_pair;

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
        if (x1 <= x2 and y1 <= y2)
            return CoreRange( {x1, y1}, {x2, y2} );

        return {};
    }

    bool contains ( const CoreRange & other ) const
    {
        return (other.start.x >= this->start.x ) &&
               (other.end.x <= this->end.x) &&
               (other.start.y >= this->start.y)  &&
               (other.end.y <= this->end.y);
    }

    // Merge lined-up (in x or y dimension) intersecting/adjacent rectangles
    std::optional<CoreRange> merge ( const CoreRange & cr) const
    {
        if ( this->intersects(cr) ){
          if ( this->start.x == cr.start.x && this->end.x == cr.end.x )
            return CoreRange ( {this->start.x, std::min(this->start.y, cr.start.y)} , { this->end.x, std::max( this->end.y, cr.end.y) } );

          else if ( this->start.y == cr.start.y && this->end.y == cr.end.y )
            return CoreRange ( { std::min( this->start.x, cr.start.x ), this->start.y}, { std::max( this->end.x, cr.end.x) , this->end.y });
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
    CoreRangeSet(const std::set<CoreRange> &core_ranges) : ranges_(core_ranges) {
      for (auto outer_it = this->ranges_.begin(); outer_it != this->ranges_.end(); outer_it++) {
        for (auto inner_it = this->ranges_.begin(); inner_it != this->ranges_.end(); inner_it++) {
          if (outer_it == inner_it) {
            continue;
          }
          CoreRange first_core_range = *outer_it;
          CoreRange second_core_range = *inner_it;
          bool first_core_left_of_second = first_core_range.end.x < second_core_range.start.x;
          bool first_core_right_of_second = first_core_range.start.x > second_core_range.end.x;
          bool first_core_above_second = first_core_range.end.y < second_core_range.start.y;
          bool first_core_below_second = first_core_range.start.y > second_core_range.end.y;
          auto no_overlap = first_core_left_of_second or first_core_right_of_second or first_core_above_second or first_core_below_second;
          if (not no_overlap) {
            TT_THROW("Cannot create CoreRangeSet with specified core ranges because core ranges " + first_core_range.str() + " and " + second_core_range.str() + " overlap!");
          }
        }
      }
    }

    CoreRangeSet(const CoreRangeSet &other) = default;
    CoreRangeSet& operator=(const CoreRangeSet &other) = default;

    CoreRangeSet(CoreRangeSet &&other) = default;
    CoreRangeSet& operator=(CoreRangeSet &&other) = default;

    CoreRangeSet merge ( const std::set<CoreRange> & core_ranges ){
      std::set<CoreRange> scr = core_ranges;
      scr.insert(ranges_.begin(), ranges_.end());
      std::vector<CoreRange> diffs;
      std::set<CoreRange> filter_set, tmp;
      // Find all CoreRanges that are completely contained by another CoreRange
      for (auto it1 = scr.begin(); it1 != scr.end(); it1++){
        for (auto it2 = scr.begin(); it2 != scr.end(); it2++ ){
          if ( it1 == it2 ){
            continue;
          }
          // std::cout << "Comparing " << it1->str() << " and " << it2->str() << std::endl;

          if ( it1->contains(*it2)){
            // std::cout << it1->str() << " contains " << it2->str() << std::endl;
            filter_set.insert(*it2);
          }
        }
      }

      std::set_difference( std::make_move_iterator( scr.begin() ),
                           std::make_move_iterator( scr.end() ),
                           filter_set.begin(), filter_set.end(),
        std::inserter(tmp, tmp.end()));

      scr.swap(tmp);
      filter_set.clear();
      tmp.clear();
      ranges_.clear();
      // Merge CoreRanges, where possible
      for (auto it1 = scr.begin(); it1 != scr.end(); it1++){
        for (auto it2 = scr.begin(); it2 != scr.end(); it2++ ){
          if ( it1 == it2 ){
            continue;
          }
          if ( auto merged = it1->merge(*it2) ){
            // std::cout << "merging " << it1->str() << " and " << it2->str() << std::endl;
            ranges_.insert ( merged.value());
            filter_set.insert(*it1);
            filter_set.insert(*it2);
          }
        }
      }
      std::set_difference( std::make_move_iterator( scr.begin() ),
                           std::make_move_iterator( scr.end() ),
                           filter_set.begin(), filter_set.end(),
                           std::inserter(tmp, tmp.end()));

      scr.swap(tmp);

      ranges_.insert(scr.begin(), scr.end());

      //TODO: Diff CoreRanges
      // for ( unsigned i = 0; i < vcr.size(); i++){
      //   for (unsigned j = i+1; j < vcr.size(); j++){

      //     auto d1 = vcr[i].diff(vcr[j]);
      //     auto d2 = vcr[j].diff(vcr[i]);

      //     if (d1.size() < d2.size() )
      //     {
      //       diffs.insert(diffs.end(), d1.begin(), d1.end());
      //       diffs.push_back( vcr[j]);
      //     }
      //     else{
      //       diffs.insert(diffs.end(), d2.begin(), d2.end());
      //       diffs.push_back( vcr[i]);
      //     }
      //   }
      // }
      return *this;
    }

    CoreRangeSet merge ( const CoreRangeSet & s )
    {
      return this->merge (s.ranges());
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
struct hash<RelativeCoreCoord> {
  std::size_t operator()(RelativeCoreCoord const &o) const {
    std::size_t seed = 0;
    seed = std::hash<std::size_t>()(o.x) ^ std::hash<std::size_t>()(o.y) << 1;
    return seed;
  }
};
}
