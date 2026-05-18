// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "serialization/serializable.hpp"

using ttml::serialization::get_value_type;
using ttml::serialization::SerializableType;
using ttml::serialization::StateDict;
using ttml::serialization::ValueType;

namespace {

// Build a single-key StateDict whose value is the given variant alternative.
template <typename T>
StateDict make_dict(T&& v) {
    StateDict d;
    d.emplace("k", SerializableType{ValueType{std::forward<T>(v)}});
    return d;
}

}  // namespace

// ---------------------------------------------------------------------------
// Integer branch: same-type round trip
// ---------------------------------------------------------------------------

TEST(GetValueType, IntegerSameType_SizeT) {
    auto d = make_dict(std::size_t{42});
    EXPECT_EQ(get_value_type<std::size_t>(d, "k"), 42u);
}

TEST(GetValueType, IntegerSameType_Int) {
    auto d = make_dict(int{-7});
    EXPECT_EQ(get_value_type<int>(d, "k"), -7);
}

TEST(GetValueType, IntegerSameType_SizeTAboveIntMax_RoundTrips) {
    // A size_t value that does not fit in int. The same-type read must
    // preserve it bit-for-bit, i.e. the implementation must not narrow
    // through int.
    constexpr std::size_t big = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    auto d = make_dict(big);
    EXPECT_EQ(get_value_type<std::size_t>(d, "k"), big);
}

// ---------------------------------------------------------------------------
// Integer branch: regular-integer group widening (the round-trip fix)
// ---------------------------------------------------------------------------

TEST(GetValueType, IntegerWidening_IntToSizeT) {
    // Mirrors the AdamW "steps" round-trip: written as size_t, comes back as
    // int after a Python boundary, must read back as size_t.
    auto d = make_dict(int{7});
    EXPECT_EQ(get_value_type<std::size_t>(d, "k"), 7u);
}

TEST(GetValueType, IntegerWidening_SizeTToInt) {
    auto d = make_dict(std::size_t{7});
    EXPECT_EQ(get_value_type<int>(d, "k"), 7);
}

TEST(GetValueType, IntegerWidening_Uint32ToSizeT) {
    auto d = make_dict(std::uint32_t{12345});
    EXPECT_EQ(get_value_type<std::size_t>(d, "k"), 12345u);
}

TEST(GetValueType, IntegerWidening_IntToUint32) {
    auto d = make_dict(int{12345});
    EXPECT_EQ(get_value_type<std::uint32_t>(d, "k"), 12345u);
}

// ---------------------------------------------------------------------------
// Integer branch: bool group is isolated
// ---------------------------------------------------------------------------

TEST(GetValueType, BoolSameType) {
    auto d = make_dict(true);
    EXPECT_EQ(get_value_type<bool>(d, "k"), true);
}

TEST(GetValueType, BoolGuard_BoolReaderRejectsInt) {
    auto d = make_dict(int{1});
    EXPECT_THROW(get_value_type<bool>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, BoolGuard_SizeTReaderRejectsBool) {
    auto d = make_dict(true);
    EXPECT_THROW(get_value_type<std::size_t>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, BoolGuard_IntReaderRejectsBool) {
    auto d = make_dict(false);
    EXPECT_THROW(get_value_type<int>(d, "k"), std::bad_variant_access);
}

// ---------------------------------------------------------------------------
// Integer branch: char group is isolated
// ---------------------------------------------------------------------------

TEST(GetValueType, CharSameType) {
    auto d = make_dict(char{'a'});
    EXPECT_EQ(get_value_type<char>(d, "k"), 'a');
}

TEST(GetValueType, CharGuard_CharReaderRejectsInt) {
    auto d = make_dict(int{65});  // 'A' as int
    EXPECT_THROW(get_value_type<char>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, CharGuard_IntReaderRejectsChar) {
    auto d = make_dict(char{'A'});
    EXPECT_THROW(get_value_type<int>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, CharGuard_BoolReaderRejectsChar) {
    auto d = make_dict(char{'a'});
    EXPECT_THROW(get_value_type<bool>(d, "k"), std::bad_variant_access);
}

// ---------------------------------------------------------------------------
// Integer branch: negative-to-unsigned guard
// ---------------------------------------------------------------------------

TEST(GetValueType, NegativeToUnsigned_IntToSizeT_Throws) {
    // Without the guard this would static_cast<size_t>(-1) -> SIZE_MAX.
    auto d = make_dict(int{-1});
    EXPECT_THROW(get_value_type<std::size_t>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, NegativeToUnsigned_IntToUint32_Throws) {
    auto d = make_dict(int{-42});
    EXPECT_THROW(get_value_type<std::uint32_t>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, NegativeToSigned_IntToInt_Allowed) {
    // Sanity: the negative guard must not fire when the destination is signed.
    auto d = make_dict(int{-1});
    EXPECT_EQ(get_value_type<int>(d, "k"), -1);
}

TEST(GetValueType, ZeroToUnsigned_NotRejected) {
    // The negative guard rejects strictly-negative values; zero should pass.
    auto d = make_dict(int{0});
    EXPECT_EQ(get_value_type<std::size_t>(d, "k"), 0u);
}

// ---------------------------------------------------------------------------
// Integer branch: held type is non-integer
// ---------------------------------------------------------------------------

TEST(GetValueType, IntegerReader_RejectsString) {
    auto d = make_dict(std::string{"hi"});
    EXPECT_THROW(get_value_type<std::size_t>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, IntegerReader_RejectsFloat) {
    auto d = make_dict(float{3.14F});
    EXPECT_THROW(get_value_type<int>(d, "k"), std::bad_variant_access);
}

// ---------------------------------------------------------------------------
// Floating-point branch
// ---------------------------------------------------------------------------

TEST(GetValueType, FloatSameType) {
    auto d = make_dict(float{0.5F});
    EXPECT_EQ(get_value_type<float>(d, "k"), 0.5F);
}

TEST(GetValueType, DoubleSameType) {
    auto d = make_dict(double{0.1});
    EXPECT_EQ(get_value_type<double>(d, "k"), 0.1);
}

TEST(GetValueType, DoubleSameType_NotRepresentableInFloat_RoundTrips) {
    // 0.1 has no exact binary representation; the nearest float and the
    // nearest double differ. The same-type read must preserve full double
    // precision, i.e. the implementation must not narrow through float.
    constexpr double v = 0.1;
    static_assert(static_cast<double>(static_cast<float>(v)) != v, "0.1 must differ between float and double");
    auto d = make_dict(v);
    EXPECT_EQ(get_value_type<double>(d, "k"), v);
}

TEST(GetValueType, FloatToDouble_Widens) {
    // A double that came back from Python in the float slot because it was
    // exactly representable in float32 -- must widen to double on read.
    auto d = make_dict(float{0.5F});
    EXPECT_EQ(get_value_type<double>(d, "k"), 0.5);
}

TEST(GetValueType, DoubleToFloat_Narrows) {
    // Lossless when the source double is exactly representable in float32.
    auto d = make_dict(double{0.5});
    EXPECT_EQ(get_value_type<float>(d, "k"), 0.5F);
}

TEST(GetValueType, FloatReader_RejectsInt) {
    auto d = make_dict(int{42});
    EXPECT_THROW(get_value_type<float>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, DoubleReader_RejectsString) {
    auto d = make_dict(std::string{"hi"});
    EXPECT_THROW(get_value_type<double>(d, "k"), std::bad_variant_access);
}

// ---------------------------------------------------------------------------
// Other branch: std::get<T> passthrough for non-integer non-FP types
// ---------------------------------------------------------------------------

TEST(GetValueType, String_SameType) {
    auto d = make_dict(std::string{"hello"});
    EXPECT_EQ(get_value_type<std::string>(d, "k"), "hello");
}

TEST(GetValueType, String_RejectsMismatch) {
    auto d = make_dict(int{42});
    EXPECT_THROW(get_value_type<std::string>(d, "k"), std::bad_variant_access);
}

TEST(GetValueType, VectorInt_SameType) {
    std::vector<int> v{1, 2, 3};
    auto d = make_dict(v);
    EXPECT_EQ(get_value_type<std::vector<int>>(d, "k"), v);
}

TEST(GetValueType, VectorString_SameType) {
    std::vector<std::string> v{"a", "b"};
    auto d = make_dict(v);
    EXPECT_EQ(get_value_type<std::vector<std::string>>(d, "k"), v);
}

// ---------------------------------------------------------------------------
// Missing key
// ---------------------------------------------------------------------------

TEST(GetValueType, MissingKey_Throws) {
    auto d = make_dict(int{42});
    EXPECT_THROW(get_value_type<int>(d, "no_such_key"), std::out_of_range);
}
