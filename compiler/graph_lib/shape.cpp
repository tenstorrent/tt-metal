#include <tuple>
#include <sstream>
#include "common/assert.hpp"
#include "graph_lib/shape.hpp"

namespace tt {
namespace graphlib {

Shape Shape::create(std::vector<std::uint32_t> values)
{
    Shape s;
    s.dims_ = values;
    s.valid_ = true;
    s.type_ = FREE;
    return s;
}

Shape Shape::create_buda(std::vector<std::uint32_t> values)
{
    TT_ASSERT(values.size() == BUDA_DIM_COUNT, "Shape must be 4-dimensional");
    TT_ASSERT(values[3] % BUDA_TILE_DIM == 0, "Column dimension must be divisible by tile size");
    TT_ASSERT(values[2] % BUDA_TILE_DIM == 0, "Row dimension must be divisible by tile size");
    Shape s;
    s.dims_ = values;
    s.valid_ = true;
    s.type_ = BUDA;
    return s;
}

Shape Shape::create_buda(std::uint32_t w, std::uint32_t z, std::uint32_t r, std::uint32_t c) {

    TT_ASSERT(r % BUDA_TILE_DIM == 0, "Row dimension must be divisible by tile size");
    TT_ASSERT(c % BUDA_TILE_DIM == 0, "Column dimension must be divisible by tile size");
    Shape s;
    s.dims_ = {w, z, r, c};
    s.valid_ = true;
    s.type_ = BUDA;
    return s;
}

Shape Shape::to_buda(const Shape &other)
{
    if (other.type_ == BUDA)
        return other;

    std::vector<std::uint32_t> dims = other.dims_;

    // Make it 4D
    while (dims.size() > BUDA_DIM_COUNT) {
        TT_ASSERT(dims[0] == 1, "Too many dimensions to convert to buda shape");
        dims.erase(dims.begin());
    }
    while (dims.size() < BUDA_DIM_COUNT) {
        dims.insert(dims.begin(), 1);
    }

    // Snap to tile sizes
    for (std::size_t dim = 2; dim <= 3; dim++) 
        if (dims[dim] % BUDA_TILE_DIM != 0) 
        {
            dims[dim] += BUDA_TILE_DIM - (dims[dim] % BUDA_TILE_DIM);
        }

    Shape s;
    s.dims_ = dims;
    s.valid_ = true;
    s.type_ = BUDA;
    return s;
}

std::uint32_t& Shape::operator[](int i) {
    if (i < 0) i += dims_.size();
    TT_ASSERT( (i >= 0) && (i < (int)dims_.size()), "Trying to access element outside of dimensions: " + std::to_string(i));
    TT_ASSERT(valid_, "Shape is not valid_.");
    return dims_[i];
}
std::uint32_t Shape::operator[](int i) const {
    if (i < 0) i += dims_.size();
    TT_ASSERT( (i >= 0) && (i < (int)dims_.size()), "Trying to access element outside of dimensions: " + std::to_string(i));
    TT_ASSERT(valid_, "Shape is not valid_.");
    return dims_[i];
}

bool Shape::operator==(const Shape& other) const {
    auto const &a = this->dims_;
    auto const &b = other.dims_;

    int size = (int)std::max(a.size(), b.size());
    for (int i = -1; i >= -size; --i)
    {
        int ai = (int)a.size() + i;
        int bi = (int)b.size() + i;
        if (ai < 0 and b[bi] != 1)
            return false;
        if (bi < 0 and a[ai] != 1)
            return false;
        if (ai < 0 or bi < 0)
            continue;
        if (a[ai] != b[bi])
            return false;
    }

    return true;
}

bool Shape::operator!=(const Shape &other) const { return not(*this == other); }

std::string Shape::as_string() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

std::ostream &operator<<(std::ostream &out, const Shape &s) {
    if (!s.is_valid()) 
        return out << "{INVALID_SHAPE}";

    const std::vector<std::uint32_t> dims_ = s.as_vector();
    TT_ASSERT(dims_.size() > 0);
    out << "{";
    out << s[0];
    for (std::size_t i = 1; i < dims_.size(); ++i) {
        out << ", " << dims_[i];
    }
    out << "}";
    return out;
}

std::uint32_t Shape::volume() const 
{
    std::uint32_t v = 1;
    for (auto i : dims_)
        v *= i;
    return v;
}

std::vector<std::uint32_t> Shape::as_vector() const {
    return dims_;
}

Shape Shape::canonical() const
{
    auto v = as_vector();
    while (v.size() < 4) v.insert(v.begin(), 1);
    return Shape::create(v);
}

Shape Shape::as_rank(std::uint32_t rank) const
{
    TT_ASSERT(rank > 0);
    auto v = as_vector();
    while ((std::uint32_t)v.size() < rank)
    {
        v.insert(v.begin(), 1);
    }
    while ((std::uint32_t)v.size() > rank)
    {
        TT_ASSERT(v.front() == 1, "Cannot squeeze a non-zero dim");
        v.erase(v.begin());
    }
    return graphlib::Shape::create(v);
}

// Return the list of dims_ (and amount) that need to be broadcast from current to other
std::vector<DimBroadcast> Shape::broadcast_dims(const Shape &other) const
{
    std::vector<DimBroadcast> ret;
    TT_ASSERT(valid_ && other.valid_);

    if (volume() == other.volume()) {
        // If volume is equal, we can't be bcasting, needed for reshape which isn't compatible with code below
        return {};
    }

    for (std::size_t i=0; i < dims_.size(); i++)
    {
        if (i >= other.dims_.size())
            break;

        std::size_t my_dim = dims_[dims_.size() - 1 - i];
        std::size_t other_dim = other.dims_[other.dims_.size() - 1 - i];

        if (my_dim < other_dim) {
            if (type_ == FREE)
                TT_ASSERT(my_dim == 1, "Invalid broadcast shapes: " + as_string() + " vs " + other.as_string());
            else
                TT_ASSERT( 
                    ((i >= 2) && my_dim == BUDA_TILE_DIM) ||
                    ((i < 2) && my_dim == 1), "Invalid broadcast shapes: " + as_string() + " vs " + other.as_string());
            ret.push_back(std::make_tuple(0, dims_.size() - 1 - i, other_dim));
        }
    }
    return ret;
}

std::uint32_t Shape::rt() const 
{ 
    TT_ASSERT(valid_, "Shape is not set."); 
    TT_ASSERT(type_ == BUDA, "Accessing buda dimensions in non-buda shape.");
    return dims_[2] / BUDA_TILE_DIM; 
}

std::uint32_t Shape::ct() const 
{ 
    TT_ASSERT(valid_, "Shape is not set."); 
    TT_ASSERT(type_ == BUDA, "Accessing buda dimensions in non-buda shape.");
    return dims_[3] / BUDA_TILE_DIM; 
}

std::uint32_t Shape::z() const 
{ 
    TT_ASSERT(valid_, "Shape is not set."); 
    TT_ASSERT(type_ == BUDA, "Accessing buda dimensions in non-buda shape.");
    return dims_[1];
}

std::uint32_t Shape::w() const 
{ 
    TT_ASSERT(valid_, "Shape is not set."); 
    TT_ASSERT(type_ == BUDA, "Accessing buda dimensions in non-buda shape.");
    return dims_[0];
}

bool Shape::is_single_tile() const
{
    TT_ASSERT(type_ == BUDA, "'Single-tile' is only meaningful in buda context.");
    return w() == 1 && z() == 1 && rt() == 1 && ct() == 1;
}

}  // namespace graphlib
}  // namespace tt
