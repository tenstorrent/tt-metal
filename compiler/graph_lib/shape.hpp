#pragma once

#include <vector>
#include <ostream>
#include "common/assert.hpp"

namespace tt {

namespace graphlib {

using DimBroadcast = std::tuple<int, int, int>;  // operand, dim, size

class Shape {

    enum Type {
        FREE,   // any number of dimensions
        BUDA    // 4D, snapped to tile sizes
    };

    bool valid_ = false;
    Type type_ = FREE;
    std::vector<std::uint32_t> dims_;


public:
    constexpr static int BUDA_TILE_DIM = 32;
    constexpr static int BUDA_DIM_COUNT = 4;

    Shape() = default;

    static Shape create(std::vector<std::uint32_t> dims);
    static Shape create_buda(std::vector<std::uint32_t> dims);
    static Shape create_buda(std::uint32_t w, std::uint32_t z, std::uint32_t r, std::uint32_t c);
    static Shape to_buda(const Shape &other);

    std::uint32_t& operator[](int i);
    std::uint32_t operator[](int i) const;

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape &other) const;

    bool is_valid() const { return valid_; }

    std::vector<std::uint32_t> as_vector() const;
    std::string as_string() const;

    std::uint32_t size() const { return (std::uint32_t)dims_.size(); }
    std::uint32_t volume() const;
    bool is_single_tile() const;

    // Return a canonical copy, i.e. padded to 4d
    Shape canonical() const;
    Shape as_rank(std::uint32_t rank) const;

    // Return the list of dims (and amount) that need to be broadcast from current to other
    std::vector<DimBroadcast> broadcast_dims(const Shape &other) const;

    // Common factory func
    static Shape single_tile() { return create_buda(1, 1, BUDA_TILE_DIM, BUDA_TILE_DIM); }

    // Buda dimensions accessors
    std::uint32_t rt() const;
    std::uint32_t ct() const;
    std::uint32_t z() const;
    std::uint32_t w() const;

};

std::ostream &operator<<(std::ostream &out, const Shape &s);

} // namespace graphlib
} // namespace tt
