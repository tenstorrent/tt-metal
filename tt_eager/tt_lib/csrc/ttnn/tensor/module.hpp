// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "tensor/tensor.hpp"

namespace py = pybind11;

namespace ttnn {
namespace tensor {

namespace detail {
template <std::size_t Rank>
tt::tt_metal::Shape compute_ttl_shape(
    const std::array<uint32_t, Rank>& shape, const std::array<std::array<uint32_t, 2>, Rank>& padding) {
    auto ttl_shape = std::array<uint32_t, Rank>{};
    for (auto index = 0; index < Rank; index++) {
        ttl_shape[index] = shape[index] + padding[index][0] + padding[index][1];
    }
    return tt::tt_metal::Shape{
        tt::tt_metal::Shape{ttl_shape}, tt::tt_metal::Padding{padding, tt::tt_metal::Padding::PadValue::Any}};
}
}  // namespace detail

struct Shape {
    const std::size_t rank;
    const tt::tt_metal::Shape value;

    explicit Shape(tt::tt_metal::Shape&& shape) : rank{shape.rank()}, value(shape) {}
    explicit Shape(const tt::tt_metal::Shape& shape) : rank{shape.rank()}, value(shape) {}

    template <std::size_t Rank>
    explicit Shape(
        const std::array<uint32_t, Rank>& shape, const std::optional<std::array<uint32_t, Rank>>& padded_shape) :
        rank{Rank}, value{shape, padded_shape} {}

    template <std::size_t Rank>
    explicit Shape(const std::array<uint32_t, Rank>& shape, const std::array<std::array<uint32_t, 2>, Rank>& padding) :
        rank{Rank}, value{detail::compute_ttl_shape(shape, padding)} {}

    Shape padded() const { return Shape{tt::tt_metal::Shape{this->value, Padding{this->value.rank()}}}; }

    template <std::size_t Rank>
    Shape operator+(const std::array<std::array<uint32_t, 2>, Rank>& padding) const {
        auto shape = this->value;
        const auto& current_padding = this->value.padding();
        auto accumulated_padding = padding;
        for (auto index = 0; index < Rank; index++) {
            shape[index] += padding[index][0] + padding[index][1];
            accumulated_padding[index][0] += current_padding[index].front;
            accumulated_padding[index][1] += current_padding[index].back;
        }
        return Shape{tt::tt_metal::Shape{
            shape, tt::tt_metal::Padding{accumulated_padding, tt::tt_metal::Padding::PadValue::Any}}};
    }

    bool operator==(const Shape& other) const { return this->value == other.value; }
};

std::ostream& operator<<(std::ostream& os, const Shape& self) {
    os << "ttnn.Shape([";
    const auto shape = self.value.without_padding();
    const auto& padding = self.value.padding();
    const auto& padded_shape = self.value;
    for (auto i = 0; i < shape.rank(); ++i) {
        if (i > 0) {
            os << ", ";
        }
        if (padding[i].front > 0) {
            os << padding[i].front << " + ";
        }
        os << shape[i];
        if (padding[i].back > 0) {
            os << " + " << padding[i].back;
        }
    }
    os << "])";
    return os;
}

struct Tensor {
    const tt::tt_metal::Tensor value;

    explicit Tensor(tt::tt_metal::Tensor&& tensor) : value{tensor} {}
    explicit Tensor(const tt::tt_metal::Tensor& tensor) : value{tensor} {}
};

void py_module(py::module& m_tensor) {
    py::class_<Shape>(m_tensor, "Shape")
        .def(py::init<tt::tt_metal::Shape>())
        .def(
            py::init<const std::array<uint32_t, 1>&, const std::array<std::array<uint32_t, 2>, 1>&>(),
            py::arg("shape"),
            py::arg("padding"))
        .def(
            py::init<const std::array<uint32_t, 2>&, const std::array<std::array<uint32_t, 2>, 2>&>(),
            py::arg("shape"),
            py::arg("padding"))
        .def(
            py::init<const std::array<uint32_t, 3>&, const std::array<std::array<uint32_t, 2>, 3>&>(),
            py::arg("shape"),
            py::arg("padding"))
        .def(
            py::init<const std::array<uint32_t, 4>&, const std::array<std::array<uint32_t, 2>, 4>&>(),
            py::arg("shape"),
            py::arg("padding"))
        .def(
            py::init<const std::array<uint32_t, 1>&, const std::optional<std::array<uint32_t, 1>>&>(),
            py::arg("shape"),
            py::arg("padded_shape") = std::nullopt)
        .def(
            py::init<const std::array<uint32_t, 2>&, const std::optional<std::array<uint32_t, 2>>&>(),
            py::arg("shape"),
            py::arg("padded_shape") = std::nullopt)
        .def(
            py::init<const std::array<uint32_t, 3>&, const std::optional<std::array<uint32_t, 3>>&>(),
            py::arg("shape"),
            py::arg("padded_shape") = std::nullopt)
        .def(
            py::init<const std::array<uint32_t, 4>&, const std::optional<std::array<uint32_t, 4>>&>(),
            py::arg("shape"),
            py::arg("padded_shape") = std::nullopt)
        .def(
            "__add__",
            [](const Shape& self, const std::array<std::array<uint32_t, 2>, 1>& padding) { return self + padding; })
        .def(
            "__add__",
            [](const Shape& self, const std::array<std::array<uint32_t, 2>, 2>& padding) { return self + padding; })
        .def(
            "__add__",
            [](const Shape& self, const std::array<std::array<uint32_t, 2>, 3>& padding) { return self + padding; })
        .def(
            "__add__",
            [](const Shape& self, const std::array<std::array<uint32_t, 2>, 4>& padding) { return self + padding; })
        .def_property_readonly("value", [](const Shape& self) { return self.value; })
        .def("__len__", [](const Shape& self) { return self.value.rank(); })
        .def(
            "__getitem__",
            [](const Shape& self, std::int64_t index) {
                auto shape = self.value.without_padding();
                return shape[index];
            })
        .def("__iter__", [](const Shape& self) { return py::iter(py::cast(self.value.without_padding())); })
        .def("__eq__", [](const Shape& self, const Shape& other) { return self == other; })
        .def(
            "__eq__",
            [](const Shape& self, const std::array<uint32_t, 1>& other) {
                return Shape{self.value.without_padding()} == Shape{tt::tt_metal::Shape{other}};
            })
        .def(
            "__eq__",
            [](const Shape& self, const std::array<uint32_t, 2>& other) {
                return Shape{self.value.without_padding()} == Shape{tt::tt_metal::Shape{other}};
            })
        .def(
            "__eq__",
            [](const Shape& self, const std::array<uint32_t, 3>& other) {
                return Shape{self.value.without_padding()} == Shape{tt::tt_metal::Shape{other}};
            })
        .def(
            "__eq__",
            [](const Shape& self, const std::array<uint32_t, 4>& other) {
                return Shape{self.value.without_padding()} == Shape{tt::tt_metal::Shape{other}};
            })
        .def("__eq__", [](const Shape& self, const py::none) { return false; })
        .def(
            "__repr__",
            [](const Shape& self) {
                std::stringstream ss;
                ss << self;
                return ss.str();
            })
        .def_property_readonly("rank", [](const Shape& self) { return self.rank; })
        .def("padded", [](const Shape& self) { return self.padded(); });

    py::class_<Tensor>(m_tensor, "Tensor")
        .def(py::init<tt::tt_metal::Tensor>())
        .def_property_readonly("value", [](const Tensor& self) -> auto& { return self.value; })
        .def("__repr__", [](const Tensor& self) { return self.value.write_to_string(Layout::ROW_MAJOR, true); })
        .def_property_readonly("shape", [](const Tensor& self) { return py::cast(Shape{self.value.shape()}); })
        .def_property_readonly("dtype", [](const Tensor& self) { return self.value.dtype(); })
        .def_property_readonly("layout", [](const Tensor& self) { return self.value.layout(); });
}

}  // namespace tensor
}  // namespace ttnn
