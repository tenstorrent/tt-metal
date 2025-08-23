#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace tt::tt_metal::detail {

class StructInfo {
private:
    // For a struct with n fields, x arrays of distinct sizes and y struct fields:
    // [0] -> size of entire struct
    // [1..n) -> offset of field 1..n-1
    // [n..n+x) -> distinct size of arrays
    // [n+x, n+x+y) -> pointers to StructInfo of the struct fields
    const uintptr_t* offsets_;

public:
    StructInfo(const uintptr_t* offsets) : offsets_(offsets) {}

    // These getters are only used by generated code.  Code generator
    // guarantees type correctness.
    size_t get(size_t index) const { return reinterpret_cast<size_t>(offsets_[index]); }
    StructInfo get_info(size_t index) const { return *reinterpret_cast<const StructInfo*>(offsets_[index]); }
    size_t get_size() const { return get(0); }
    size_t offset_of(size_t i) const { return i ? offsets_[i] : 0; }
};

template <typename T, typename Derived>
class BaseStructView;

template <typename T>
    requires std::derived_from<T, BaseStructView<typename T::byte_type, T>>
class StructSpan {
public:
    using byte_type = typename T::byte_type;
    StructSpan(const StructInfo info, byte_type* base, size_t size) : info_(info), base_(base), size_(size) {}
    byte_type* data() const { return base_; }
    size_t size() const { return size_; }
    operator std::span<byte_type>() const { return {data(), size()}; }

    T operator[](size_t i) const { return {info_, base_ + i * info_.get_size()}; }
    class iterator {
    private:
        const StructInfo info_;
        byte_type* ptr_;
        iterator(const StructInfo info, byte_type* ptr) : info_(info), ptr_(ptr) {}

    public:
        iterator& operator++() {
            ptr_ += info_.get_size();
            return *this;
        }
        bool operator!=(const iterator& other) const { return ptr_ != other.ptr_; }
        T operator*() const { return {info_, ptr_}; }
    };
    iterator begin() const { return {info_, base_}; }
    iterator end() const { return {info_, base_ + size_ * info_.get_size()}; }

private:
    const StructInfo info_;
    byte_type* base_;
    size_t size_;
};

template <typename T, typename Derived>
class BaseStructView {
public:
    using byte_type = T;
    BaseStructView(const StructInfo info, byte_type* base) : info_(info), base_(base) {}
    byte_type* data() const { return base_; }
    size_t size() const { return info_.get_size(); }
    operator std::span<byte_type>() const { return {data(), size()}; }
    template <typename U = Derived>
    size_t offset_of(U::fields i) const {
        return info_.offset_of(static_cast<size_t>(i));
    }

protected:
    byte_type* address_of(size_t i) const { return base_ + info_.offset_of(i); }
    template <typename U, typename element_type = std::conditional_t<std::is_const_v<T>, const U, U>>
    element_type& scalar_field(size_t i) const {
        return *reinterpret_cast<element_type*>(address_of(i));
    }
    template <typename U, typename element_type = std::conditional_t<std::is_const_v<T>, const U, U>>
    std::span<element_type> scalar_array(size_t i, size_t array_idx) const {
        return {reinterpret_cast<element_type*>(address_of(i)), info_.get(array_idx)};
    }
    template <typename U>
        requires std::derived_from<U, BaseStructView<T, U>>
    U struct_field(size_t i, size_t j) const {
        return {info_.get_info(j), address_of(i)};
    }
    template <typename U>
        requires std::derived_from<U, BaseStructView<T, U>>
    StructSpan<U> struct_array(size_t i, size_t struct_idx, size_t array_idx) const {
        return {info_.get_info(struct_idx), address_of(i), info_.get(array_idx)};
    }

private:
    const StructInfo info_;
    byte_type* base_;
};

template <template <typename> class Impl, typename Derived>
class StructStorage {
public:
    StructStorage(const ::tt::tt_metal::detail::StructInfo info) : info_(info), storage_(info.get_size()) {}
    using view = Impl<std::byte>;
    using const_view = Impl<const std::byte>;
    using fields = view::fields;
    operator view() { return {info_, storage_.data()}; }
    operator const_view() const { return {info_, storage_.data()}; }
    std::byte* data() { return storage_.data(); }
    const std::byte* data() const { return storage_.data(); }
    size_t size() const { return info_.get_size(); }
    template <typename T = Derived>
    size_t offset_of(T::fields i) const {
        return info_.offset_of(static_cast<size_t>(i));
    }

protected:
    const ::tt::tt_metal::detail::StructInfo info_;
    std::vector<std::byte> storage_;
};

}  // namespace tt::tt_metal::detail
