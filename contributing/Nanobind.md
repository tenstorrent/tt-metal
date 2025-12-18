## Resources:

- [Nanobind Documentation](https://nanobind.readthedocs.io/en/latest/)
- [Nanobind Porting Guide](https://nanobind.readthedocs.io/en/latest/porting.html)
- [Nanobind Exchanging Information (Type Casters)](https://nanobind.readthedocs.io/en/latest/exchanging.html)
- [Nanobind GitHub Repository](https://github.com/wjakob/nanobind)
- [Nanobind GitHub Q&A](https://github.com/wjakob/nanobind/discussions)

---

## Table of Contents

0. [Common Issues In Practice](#common-issues-in-practice)
1. [Quick Reference Table](#quick-reference-table)
2. [Namespace and Module Changes](#namespace-and-module-changes)
3. [Include Statements](#include-statements)
4. [Module Macro](#module-macro)
5. [Class and Property Bindings](#class-and-property-bindings)
6. [Function Arguments and Defaults](#function-arguments-and-defaults)
7. [Constructor Bindings](#constructor-bindings)
8. [Return Value Policies](#return-value-policies)
9. [Iterator Bindings](#iterator-bindings)
10. [Operator Bindings](#operator-bindings)
11. [Custom Type Casters](#custom-type-casters)
12. [Optional and None Handling](#optional-and-none-handling)
13. [Overload Patterns](#overload-patterns)
14. [STL Container Bindings](#stl-container-bindings)
15. [Buffer Protocol and ndarray](#buffer-protocol-and-ndarray)
16. [Implicit Conversions](#implicit-conversions)
17. [Keep Alive and Lifetime Management](#keep-alive-and-lifetime-management)
18. [Other Common Gotchas and Fixes](#other-common-gotchas-and-fixes)
19. [Migration Checklist](#migration-checklist)
20. [Debugging Checklist](#debugging-checklist)

---

## Common Issues In Practice

### `shared_ptr` / `unique_ptr` in `nb::class_` template arguments not needed
https://nanobind.readthedocs.io/en/latest/porting.html#shared-pointers-and-holders

### Custom constructors require placement new
https://nanobind.readthedocs.io/en/latest/porting.html#custom-constructors

### `TypeError: __call__(): incompatible function arguments`

#### `kw_only` misplaced or kwarg used without keyword

`nb::kw_only()` forces all arguments after `nb::kw_only()` to actually be keyword only. In the argument list given for
the bound function call, you will see a kwargs = {...} field. Double check that the candidate function args after
`, * ,` are all kwargs. Pybind is less strict and will let you put `kw_only` anywhere in the binding signature.

#### Missing STL typecasters

You didn't include the required typecaster files for STL containers. This can be especially confusing if one of the types
arguments you are using is a typedef/alias. Most commonly you'll see a `using NAME = std::variant<...>;` for some type in your
argument lists. For this reason there's a `#include <nanobind/stl/variant.h>` in `ttnn-nanobind/decorators.hpp`, since
that is a prolific header.

See also: https://nanobind.readthedocs.io/en/latest/porting.html#type-casters

#### `arg.noconvert()` is more strict

**DO NOT USE `.noconvert()` WITH OPTIONALS!**
`.noconvert()` is more strongly enforced in nanobind. Most commonly you'll run into issues if you use `.noconvert()` on
an argument of type `std::optional`. Other common issues can be from using `.noconvert` on a numerical type like a
`float` or `int`.

#### Passing a python `None` value

If an `optional` argument is bound, nanobind **requires** the `nb::arg("name") = nb::none()`. If you don't, passing in
a `None` will give you an incompatible function argument `NoneType` error.

#### `std::reference_wrapper`

Nanobind does not work smoothly with `std::reference_wrapper`, most commonly seen in the pybind code as an
`std::optional<std::reference_wrapper<...>>`. This can be worked around by using a `std::optional<YOUR_TYPE*>` instead.
Then in the implementation, you can call a convenient helper from `#include "ttnn-nanobind/nanobind_helpers.hpp`:
`nbh::rewrap_optional` to make the transition down from the binding layer into the bound c++ functions.

#### Returning `std::unique_ptr` to python

To get a `std::unique_ptr` properly handled by python when returning from C++, the `unique_ptr` needs a custom deleter
provided by nanobind. There is a convenience function to rewrap a `std::unique_ptr` with the nanobind deleter in
`#include "ttnn-nanobind/nanobind_helpers.hpp"`. Called as `nbh::steal_rewrap_unique`.

See also: https://nanobind.readthedocs.io/en/latest/ownership.html#unique-pointers

#### Setting default optional args to `std::nullopt` instead of `nb::none()`

`nb::arg("name") = std::nullopt` does not work. You need `nb::arg("name") = nb::none()`.

See also: https://nanobind.readthedocs.io/en/latest/porting.html#none-null-arguments

#### Typing: placeholders for types covered by typecasters

```cpp
// MatmulProgramConfig is a std::variant that is covered by the nanobind variant typecaster,
// but the "MatmulProgramConfig" name is used explicitly in `__init__.py` for type annotations.
// The easiest way to work around this is making a placeholder class to define the symbol.

// NB_MAKE_OPAQUE is probably not what you want here

struct MatmulProgramConfigPlaceholder {};

auto matmul_program_config = nb::class_<MatmulProgramConfigPlaceholder>(mod, "MatmulProgramConfig", R"doc(
    Variant defining matmul program config
)doc");
```

#### `module` is a reserved name

C++20 added modules to the standard. Regardless of availability, please avoid naming your `nb::module_ module` to avoid
keyword clashes. Prefer names such as `mod`, `m`, `module_<NAME>`, etc.

#### Nanobind enum map entries in python uses `_member_map_` instead of `__entries`

Error message:
```
ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
/usr/lib/python3.10/enum.py:437: in __getattr__
    raise AttributeError(name) from None
```

Patch:
```py
def get_types_from_binding_framework():
    if hasattr(ttnn.DataType, "__entries"):
        # pybind
        ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]
    elif hasattr(ttnn.DataType, "_member_map_"):
        # nanobind
        ALL_TYPES = [dtype for _, dtype in ttnn.DataType._member_map_.items() if dtype != ttnn.DataType.INVALID]
    else:
        raise Exception(
            "test_rand.py: ttnn.DataType has unexpected way of holding values. Not matching pybind/nanobind."
        )

    return ALL_TYPES

ALL_TYPES = get_types_from_binding_framework()
```

#### TypeError: Unable to convert function return value to a Python type!

Error message:
```
Traceback (most recent call last):
  File "tt-metal/./test_topk.py", line 44, in <module>
    tensor = ttnn.from_torch(tensor)
  File ".../decorators.py", line 729, in __call__
    output = self.decorated_function(*function_args, **function_kwargs)
  File ".../decorators.py", line 541, in call_wrapper
    if ttnn.CONFIG.report_path is not None:

TypeError: Unable to convert function return value to a Python type!
The signature was:
    (self) -> std::optional<std::filesystem::__cxx11::path>
```

**What this means**: A typecaster header was missing in the place where the binding was defined.

In this case, the `CONFIG` member `report_path` didn't have the typecasters included where it was defined.
If we search for `report_path` in `ttnn/cpp/ttnn-nanobind`, we find the binding definition in `core.cpp`.
The error message identifies the types `std::optional` and `std::filesystem::__cxx11::path`, so we know
that the typecaster headers required are `#include <nanobind/stl/optional.h>` and `#include <nanobind/stl/filesystem.h>`.


---

# Pybind11 to Nanobind Migration Guide

This guide documents the common patterns, bugfixes, and differences observed during the migration from pybind11 to nanobind in this codebase. It is designed to help developers understand what changes are needed when converting binding code.


---

## Quick Reference Table

| pybind11 | nanobind | Notes |
|----------|----------|-------|
| `namespace py = pybind11;` | `namespace nb = nanobind;` | Namespace alias |
| `PYBIND11_MODULE(name, m)` | `NB_MODULE(name, m)` | Module macro |
| `py::module_` | `nb::module_` | Note the underscore |
| `py::module::import("json")` | `nb::module_::import_("json")` | Note trailing underscore in `import_` |
| `py::function` / `py::object` / `py::handle` | `nb::callable` / `nb::object` / `nb::handle` | Object types |
| `.def_readwrite(...)` | `.def_rw(...)` | Read-write member |
| `.def_readonly(...)` | `.def_ro(...)` | Read-only member |
| `.def_property(...)` | `.def_prop_rw(...)` | Property with getter/setter |
| `.def_property_readonly(...)` | `.def_prop_ro(...)` | Read-only property |
| `py::return_value_policy::*` | `nb::rv_policy::*` | Return policy |
| `py::keep_alive<0, 1>()` | `nb::keep_alive<0, 1>()` | Same syntax |
| `py::arg("name") = std::nullopt` | `nb::arg("name") = nb::none()` | None defaults |
| `py::self == py::self` | `nb::self == nb::self` | Comparison operator shorthand |
| `py::overload_cast<Args...>` | `nb::overload_cast<Args...>` | Overload casting |
| `py::overload_cast<...>(..., py::const_)` | `nb::overload_cast<...>(..., nb::const_)` | Const overloads |
| `py::init<>()` | `nb::init<>()` | Constructor |
| `py::init_implicit<T>()` | `nb::init_implicit<T>()` | Implicit constructor |
| `py::implicitly_convertible<T, U>()` | `nb::implicitly_convertible<T, U>()` | Implicit conversion |
| `py::enum_<E>(...)` | `nb::enum_<E>(...)` | Enum binding |
| `py::reinterpret_borrow<T>(x)` | `nb::borrow<T>(x)` | Borrow reference |
| `py::reinterpret_steal<T>(x)` | `nb::steal<T>(x)` | Steal reference |
| `py::type::of<T>()` | `nb::type<T>()` | Type introspection |
| `py::kw_only()` | `nb::kw_only()` | More strict about placement |

---

## Namespace and Module Changes

### Pybind11

```cpp
namespace py = pybind11;

void py_module(py::module& module) {
    // module is passed by reference
}
```

### Nanobind

```cpp
namespace nb = nanobind;

void py_module(nb::module_& mod) {
    // Note: module_ has an underscore, passed by reference
}
```

**Important**: The `module_` type name includes an underscore in nanobind.

---

## Include Statements

### Pybind11

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Covers most STL types
#include <pybind11/operators.h>
```

### Nanobind

Nanobind requires explicit includes for each STL type:

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/operators.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
```

**Tip**: Missing STL headers is a common source of `TypeError: __call__(): incompatible function arguments` errors at runtime.

**Key Difference**: Nanobind does **not** auto-enable STL conversions. You must explicitly include the desired header in every translation unit that uses those types.

---

## Module Macro

### Pybind11

```cpp
PYBIND11_MODULE(_ttnn, module) {
    module.doc() = "Python bindings for TTNN";
    // ...
}
```

### Nanobind

```cpp
NB_MODULE(_ttnn, mod) {
    mod.doc() = "Python bindings for TTNN";
    // ...
}
```

---

## Class and Property Bindings

### Pybind11

```cpp
py::class_<MyClass>(module, "MyClass")
    .def_readwrite("member", &MyClass::member)
    .def_readonly("const_member", &MyClass::const_member)
    .def_property("prop", &MyClass::get_prop, &MyClass::set_prop)
    .def_property_readonly("readonly_prop", &MyClass::get_readonly_prop);
```

### Nanobind

```cpp
nb::class_<MyClass>(mod, "MyClass")
    .def_rw("member", &MyClass::member)
    .def_ro("const_member", &MyClass::const_member)
    .def_prop_rw("prop", &MyClass::get_prop, &MyClass::set_prop)
    .def_prop_ro("readonly_prop", &MyClass::get_readonly_prop);
```

### Example from Codebase (tensor.cpp)

**Pybind11:**

```cpp
.def_property_readonly("tile_shape", &Tile::get_tile_shape)
```

**Nanobind:**

```cpp
.def_prop_ro("tile_shape", &Tile::get_tile_shape)
```

### Read-Write Properties with `def_prop_rw`

The nanobind `def_prop_rw` method allows defining properties with both getter and setter:

```cpp
.def_prop_rw(
    "tensor_id",
    [](const Tensor& self) { return self.tensor_id; },
    [](Tensor& self, std::size_t tensor_id) { self.tensor_id = tensor_id; });
```

---

## Function Arguments and Defaults

### Pybind11

```cpp
module.def(
    "my_function",
    &my_function,
    py::arg("input_tensor"),
    py::arg("device") = std::nullopt,
    py::arg("memory_config") = std::nullopt,
    py::kw_only(),
    py::arg("queue_id") = std::nullopt);
```

### Nanobind

```cpp
mod.def(
    "my_function",
    &my_function,
    nb::arg("input_tensor"),
    nb::arg("device") = nb::none(),
    nb::arg("memory_config") = nb::none(),
    nb::kw_only(),
    nb::arg("queue_id") = nb::none());
```

**Key Difference**: Use `nb::none()` instead of `std::nullopt` for default None values.

### Keyword-Only Arguments (`kw_only`)

Nanobind is **stricter** about `kw_only()` placement. It must appear **before** the first keyword-only argument.

**Wrong (accepted by pybind11, rejected by nanobind):**

```cpp
m.def("foo", &foo_impl,
      nb::arg("x"),
      nb::kw_only(),              // ❌ too late
      nb::arg("y") = 42);
```

**Correct:**

```cpp
m.def("foo", &foo_impl,
      nb::kw_only(),              // ✅ immediately before keyword-only args
      nb::arg("x"),
      nb::arg("y") = 42);
```

**Rule of thumb**: Place `nb::kw_only()` **between positional args and keyword-only args**, before the first keyword-only `nb::arg(...)`.

---

## Constructor Bindings

### Pybind11 (Lambda-based)

```cpp
py::class_<CoreCoord>(module, "CoreCoord")
    .def(py::init<std::size_t, std::size_t>())
    .def(py::init<>([](std::tuple<std::size_t, std::size_t> core_coord) {
        return CoreCoord(std::get<0>(core_coord), std::get<1>(core_coord));
    }));
```

### Nanobind (Placement New)

Nanobind uses placement new for custom constructors:

```cpp
nb::class_<CoreCoord>(mod, "CoreCoord")
    .def(nb::init<std::size_t, std::size_t>())
    .def(
        "__init__",
        [](CoreCoord* t, std::tuple<std::size_t, std::size_t> core_coord) {
            new (t) CoreCoord(std::get<0>(core_coord), std::get<1>(core_coord));
        });
```

**Key Difference**: Nanobind requires using `__init__` with placement new (`new (t)`) for custom constructors instead of returning a value.

### Example: Complex Constructor with Multiple Arguments

**Pybind11:**

```cpp
.def(
    py::init<>([](const ttnn::Shape& shape,
                  DataType dtype,
                  Layout layout,
                  BufferType buffer_type,
                  const std::optional<Tile>& tile) {
        return TensorSpec(shape, TensorLayout(dtype, PageConfig(layout, tile),
                         MemoryConfig(TensorMemoryLayout::INTERLEAVED, buffer_type)));
    }),
    py::arg("shape"),
    py::arg("dtype"),
    py::arg("layout"),
    py::arg("buffer_type") = BufferType::DRAM,
    py::arg("tile") = std::nullopt)
```

**Nanobind:**

```cpp
.def(
    "__init__",
    [](TensorSpec* t,
       const ttnn::Shape& shape,
       DataType dtype,
       Layout layout,
       BufferType buffer_type,
       const std::optional<Tile>& tile) {
        new (t) TensorSpec(shape, TensorLayout(dtype, PageConfig(layout, tile),
                          MemoryConfig(TensorMemoryLayout::INTERLEAVED, buffer_type)));
    },
    nb::arg("shape"),
    nb::arg("dtype"),
    nb::arg("layout"),
    nb::arg("buffer_type") = BufferType::DRAM,
    nb::arg("tile") = nb::none())
```

---

## Return Value Policies

### Pybind11

```cpp
.def("device", &Tensor::device, py::return_value_policy::reference)
```

### Nanobind

```cpp
.def("device", &Tensor::device, nb::rv_policy::reference)
```

**Policy Names:**

| pybind11 | nanobind |
|----------|----------|
| `py::return_value_policy::reference` | `nb::rv_policy::reference` |
| `py::return_value_policy::copy` | `nb::rv_policy::copy` |
| `py::return_value_policy::move` | `nb::rv_policy::move` |
| `py::return_value_policy::reference_internal` | `nb::rv_policy::reference_internal` |
| `py::return_value_policy::automatic` | `nb::rv_policy::automatic` |

---

## Iterator Bindings

### Pybind11

```cpp
.def(
    "__iter__",
    [](const HostBuffer& self) {
        return py::make_iterator(self.view_bytes().begin(), self.view_bytes().end());
    },
    py::keep_alive<0, 1>())
```

### Nanobind

```cpp
#include <nanobind/make_iterator.h>

.def(
    "__iter__",
    [](const HostBuffer& self) {
        return nb::make_iterator<nb::rv_policy::reference_internal>(
            nb::type<tt::tt_metal::HostBuffer>(),
            "iterator",
            self.view_bytes().begin(),
            self.view_bytes().end());
    },
    nb::keep_alive<0, 1>())
```

**Key Differences:**
- Nanobind's `make_iterator` requires specifying the return value policy
- Requires the parent type and iterator name as additional arguments
- Requires `#include <nanobind/make_iterator.h>`
- Often needs `nb::rv_policy::reference_internal` to avoid dangling references

---

## Operator Bindings

### Pybind11

```cpp
.def(py::self == py::self)
.def(py::self != py::self)
```

### Nanobind

```cpp
.def(nb::self == nb::self)
.def(nb::self != nb::self)
```

**Note**: For type stubs/typing, you may need to add signature annotations:

```cpp
.def(
    nb::self == nb::self,
    nb::sig("def __eq__(self, arg: object, /) -> bool"))
```

---

## Custom Type Casters

### SmallVector Caster - Pybind11

```cpp
// small_vector_caster.hpp
namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename T, size_t PREALLOCATED_SIZE>
struct type_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>>
    : list_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>, T> {};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
```

### SmallVector Caster - Nanobind

```cpp
// small_vector_caster.hpp
#include <nanobind/stl/detail/nb_list.h>
#include <tt_stl/small_vector.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T, size_t PREALLOCATED_SIZE>
struct type_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>>
    : list_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>, T> {};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
```
**Key Difference**: Nanobind uses `NB_TYPE_CASTER(...)` with `from_python(...)` and `from_cpp(...)` compared to pybind11's `PYBIND11_TYPE_CASTER` + `load(...)` + `cast(...)`.

### Custom Dtype Traits

Nanobind requires explicit dtype traits for custom numeric types:

```cpp
namespace nanobind::detail {
template <>
struct dtype_traits<::bfloat16> {
    static constexpr dlpack::dtype value{
        static_cast<uint8_t>(nanobind::dlpack::dtype_code::Bfloat),
        16,  // size in bits
        1    // lanes (simd), usually set to 1
    };
    static constexpr auto name = const_name("bfloat16");
};
}
```

---

## Optional and None Handling

One of the most significant differences between pybind11 and nanobind is how `None` values are handled.

### Default Values for Optional Arguments

**Pybind11:**

```cpp
py::arg("device") = std::nullopt
```

**Nanobind:**

```cpp
nb::arg("device") = nb::none()
```

### Allowing None for Non-Optional Arguments

In nanobind, if you need to allow Python `None` for a function argument that doesn't have a default, you can use the `.none()` modifier on the argument:

```cpp
// Allows the argument to accept None
m.def("func", &func, "arg"_a.none());
```

This is documented in the [nanobind porting guide](https://nanobind.readthedocs.io/en/latest/porting.html):

> Unlike pybind11, nanobind does not allow None-valued function arguments by default.
> To permit them, add a `.none()` annotation to the argument or set a `None` default value.

### Setting Default to None

```cpp
// Equivalent: setting nb::none() as default also allows None
m.def("func", &func, nb::arg("arg") = nb::none());
```

### Allowing Python to Set Properties to None

Nanobind's porting guide notes that `None` arguments must be explicitly allowed unless the default is `nb::none()`. The same principle applies to property setters: if you want `obj.prop = None` to work, your setter must accept and interpret `None`.

Here is a pattern that uses `nb::none()` as a default on the property setter argument, allowing Python code to set the value to `None`:

```cpp
nb::class_<MyType>(mod, "MyType")
    .def_prop_rw(
        "maybe_config",
        [](const MyType &self) -> nb::object {
            if (self.maybe_config.has_value())
                return nb::cast(*self.maybe_config);
            return nb::none();
        },
        [](MyType &self, nb::object value) {
            if (value.is_none()) {
                self.maybe_config.reset();
            } else {
                self.maybe_config = nb::cast<Config>(value);
            }
        },
        // Key point: explicitly accept None by setting default to nb::none()
        nb::arg("value") = nb::none());
```

**Python behavior:**

```python
x = ttnn.MyType()
x.maybe_config = None          # works: calls setter with None
x.maybe_config = some_config   # works: casts and stores
```

**Important Notes:**
- `nb::none()` is a value, not a type. It must be passed by value, not by reference.
- Nanobind enforces that `None` defaults match the declared C++ type (often `nb::object` or a pointer type). A mismatch triggers a compile-time error instead of a runtime `TypeError`.
- If a parameter can be multiple unrelated types (e.g. bool/str/None), nanobind ports often accept an `nb::object` and do explicit runtime checks with `.is_none()`.

---

## Overload Patterns

### Decorator Helper Types

**Pybind11:**

```cpp
ttnn::pybind_overload_t{
    [](const OperationType& self,
       const ttnn::Tensor& input_tensor,
       const float value) { return self(input_tensor, value); },
    py::arg("input_tensor"),
    py::arg("value"),
    py::arg("memory_config") = std::nullopt}
```

**Nanobind:**

```cpp
ttnn::nanobind_overload_t{
    [](const OperationType& self,
       const ttnn::Tensor& input_tensor,
       const float value) { return self(input_tensor, value); },
    nb::arg("input_tensor"),
    nb::arg("value"),
    nb::arg("memory_config") = nb::none()}
```

### Composite Operation Call Binding

In pybind, composite ops were bound by passing the overload lambda directly.

In nanobind (in this repo), composite ops use a *signature-preserving wrapper*:
- `resolve_composite_operation_call_method(...)`

If you port a composite op and `__call__` suddenly has the wrong behavior/signature or overload resolution breaks, ensure you're using the nanobind composite wrapper (see `ttnn/cpp/ttnn-nanobind/decorators.hpp`).

---

## STL Container Bindings

### Required Includes

Unlike pybind11's `<pybind11/stl.h>` catch-all, nanobind requires explicit includes:

```cpp
// For std::vector
#include <nanobind/stl/vector.h>

// For std::array
#include <nanobind/stl/array.h>

// For std::optional
#include <nanobind/stl/optional.h>

// For std::variant
#include <nanobind/stl/variant.h>

// For std::map
#include <nanobind/stl/map.h>

// For std::unordered_map
#include <nanobind/stl/unordered_map.h>

// For std::tuple
#include <nanobind/stl/tuple.h>

// For std::set
#include <nanobind/stl/set.h>

// For std::string
#include <nanobind/stl/string.h>

// For std::string_view
#include <nanobind/stl/string_view.h>

// For std::shared_ptr
#include <nanobind/stl/shared_ptr.h>

// For std::unique_ptr
#include <nanobind/stl/unique_ptr.h>

// For std::function
#include <nanobind/stl/function.h>
```

**Rule of thumb:**
1. Use **type casters** (`vector.h`, `map.h`, …) for containers when copies are fine.
2. Switch to **bindings** (`bind_vector.h`, `bind_map.h`) only when you need mutation without copies or observe performance issues.

**Developer mental model:**
- **Type casters** (`nanobind/stl/vector.h`, …) convert between Python containers and STL containers by value.
- **Container bindings** (`nanobind/stl/bind_vector.h`, …) expose an actual container type to Python (mutations can go back to C++), but come with a different API surface.
  - Will likely need NB_MAKE_OPAQUE to make sure the vector typecaster is not applied.

In TTNN bindings, the dominant pattern is **type casters**.

### Shared/Unique Pointer Handling

**Pybind11:**

```cpp
py::class_<MyClass, std::shared_ptr<MyClass>>(module, "MyClass")
    .def(py::init<>());
```

**Nanobind:**

Nanobind removed holder types. You don't need to specify holders anymore:

```cpp
nb::class_<MyClass>(mod, "MyClass")
    .def(nb::init<>());

// But if you exchange shared_ptr across boundary, include the caster
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
```

**Migration bugfix pattern:**
- Remove holder template params from `class_`.
- Add `nanobind/stl/shared_ptr.h` and/or `nanobind/stl/unique_ptr.h` in the binding TU if those types cross the boundary.

### Optional Reference Wrapper

Nanobind doesn't bind `std::optional<std::reference_wrapper<T>>` properly.

**Solution:** Use pointer-based approach instead:

```cpp
// Instead of std::optional<std::reference_wrapper<T>>
// Use std::optional<T*>
// You can call this function in `ttnn-nanobind/nanobind_helpers.hpp` under the `nbh` namespace
// in the binding function body for convenience.

template <typename T>
constexpr std::optional<std::reference_wrapper<T>> rewrap_optional(std::optional<T*> arg) noexcept {
    if (arg.has_value() && arg.value() != nullptr) {
        return std::make_optional(std::ref(*arg.value()));
    }
    return std::nullopt;
}
```

---

## Buffer Protocol and ndarray

### Pybind11 (Buffer Protocol)

```cpp
py::class_<tt::tt_metal::HostBuffer>(m_tensor, "HostBuffer", py::buffer_protocol())
    .def_buffer([](HostBuffer& self) -> py::buffer_info {
        return py::buffer_info(
            self.view_bytes().data(),
            sizeof(std::byte),
            py::format_descriptor<unsigned char>::format(),
            1,
            {self.view_bytes().size()},
            {sizeof(std::byte)});
    });
```

### Nanobind (ndarray)

Nanobind dropped `buffer_protocol` in favor of `ndarray`. The implementation is more involved:

```cpp
nb::class_<tt::tt_metal::HostBuffer>(m_tensor, "HostBuffer")
    .def(nb::init<>())
    .def("__getitem__", [](const HostBuffer& self, std::size_t index) {
        return self.view_bytes()[index];
    })
    .def("__len__", [](const HostBuffer& self) {
        return self.view_bytes().size();
    })
    .def(
        "__array__",
        [](HostBuffer& self) {
            return nb::ndarray<uint8_t, nb::array_api, nb::device::cpu, nb::shape<-1>, nb::c_contig>(
                self.view_bytes().data(), {self.view_bytes().size()});
        },
        nb::rv_policy::reference_internal)
    .def("__dlpack_device__", [](nb::handle) {
        return std::make_pair(nb::device::cpu::value, 0);
    })
    .def("__dlpack__", [](nb::pointer_and_handle<tt::tt_metal::HostBuffer> self, nb::kwargs kwargs) {
        using array_api_t = nb::ndarray<uint8_t, nb::array_api, nb::device::cpu, nb::shape<-1>, nb::c_contig>;
        nb::object aa = nb::cast(
            array_api_t(self.p->view_bytes().data(), {self.p->view_bytes().size()}),
            nb::rv_policy::reference_internal,
            self.h);
        return aa.attr("__dlpack__")(**kwargs);
    });
```

**Migration bugfix pattern:**
- If your pybind binding relied on the buffer protocol, a nanobind port needs to define an explicit array interop strategy:
  - `nb::ndarray<nb::array_api>` for "array API" / DLPack-capable views
  - `__array__`, `__dlpack__`, `__dlpack_device__` helpers where appropriate

### Python Tensor Inputs

Pybind code in this repo frequently accepts `torch.Tensor` / `numpy.ndarray` as a generic `py::object` and then does runtime dtype detection + conversion.

Nanobind code is migrating toward `nb::ndarray<nb::array_api>` (i.e., an *array-api / DLPack compatible* input) for the same constructor path.

---

## Implicit Conversions

### Pybind11

```cpp
py::implicitly_convertible<std::tuple<std::size_t, std::size_t>, CoreCoord>();
py::implicitly_convertible<ttnn::SmallVector<uint32_t>, ttnn::Shape>();
```

### Nanobind

```cpp
nb::implicitly_convertible<std::tuple<std::size_t, std::size_t>, CoreCoord>();
nb::implicitly_convertible<ttsl::SmallVector<uint32_t>, ttnn::Shape>();
```

For implicit constructor conversion (replaces `py::implicitly_convertible` in some cases):

```cpp
.def(nb::init_implicit<ttsl::SmallVector<uint32_t>>());
```

### Python Tuple Conversions

Nanobind has its own tuple wrappers (e.g. `nb::detail::tuple<...>`). In this repo, some constructors accept both:
- `std::tuple<size_t, size_t>` and
- `nb::detail::tuple<nb::int_, nb::int_>`

If a pybind constructor "just worked" with a Python tuple and the nanobind port rejects it, add:
- an overload taking `nb::detail::tuple<...>`, and/or
- `nb::implicitly_convertible<...>()` for the specific tuple wrapper.

**Note**: Nanobind is stricter about implicit conversions. You may need to explicitly allow them or use `.noconvert()` on arguments to forbid them if strictness is desired (which is the default in nanobind for some types, unlike pybind11).

---

## Keep Alive and Lifetime Management

### Pybind11

```cpp
.def("to_device", &Tensor::to_device,
     py::arg("device"),
     py::keep_alive<0, 2>())  // Keep argument 2 (device) alive while return value (0) exists
```

### Nanobind

```cpp
.def("to_device", &Tensor::to_device,
     nb::arg("device"),
     nb::keep_alive<0, 2>())  // Same syntax
```

**Index Reference:**
- `0` = return value
- `1` = `this` (self)
- `2` = first argument
- `3` = second argument
- etc.

---

## Other Common Gotchas and Fixes

### 1. Missing STL Type Casters

**Error**: `TypeError: __call__(): incompatible function arguments`

**Fix**: Add the appropriate nanobind STL header:

```cpp
#include <nanobind/stl/optional.h>  // For std::optional
#include <nanobind/stl/variant.h>   // For std::variant
```

**Common failure mode**: Missing casters frequently manifests as `TypeError: __call__(): incompatible function arguments` at runtime.

### 2. None Default Values

**Error**: Function doesn't accept `None` when it should

**Fix**: Use `nb::none()` instead of `std::nullopt`:

```cpp
// Wrong
nb::arg("param") = std::nullopt

// Correct
nb::arg("param") = nb::none()
```

### 3. Custom Constructor Syntax

**Error**: Constructor doesn't work properly

**Fix**: Use placement new with `__init__`:

```cpp
// Wrong (pybind11 style)
.def(py::init<>([](Args...) { return MyClass(...); }))

// Correct (nanobind style)
.def("__init__", [](MyClass* t, Args...) { new (t) MyClass(...); })
```

### 4. Module Import

**Pybind11:**

```cpp
py::module::import("torch").attr("Tensor")
```

**Nanobind:**

```cpp
nb::module_::import_("torch").attr("Tensor")
```

Note the underscore after `module_` and underscore after `import_`.

### 5. Cast Syntax

**Pybind11:**

```cpp
py::cast(tt::tt_metal::DispatchCoreConfig{})
```

**Nanobind:**

```cpp
nb::cast(tt::tt_metal::DispatchCoreConfig{})
```

### 6. Default Argument Won't Bind / Wrong Default Appears in Python

**Likely cause**: Default isn't representable as a Python object the way nanobind expects.

**Fix**: Wrap default in `nb::cast(...)` (common across TTNN nanobind bindings):

```cpp
nb::arg("buffer_type") = nb::cast(tt::tt_metal::BufferType::L1)
```

### 7. Tracy Integration

When using Tracy profiler with nanobind:

```cpp
#if defined(TRACY_ENABLE)
    nb::callable tracy_decorator = nb::module_::import_("tracy.ttnn_profiler_wrapper")
        .attr("callable_decorator");
    tracy_decorator(m_device);
#endif
```

### 8. Leak Warnings

Nanobind has leak detection that can be disabled if needed:

```cpp
nb::set_leak_warnings(false);  // TODO: Re-enable after fixing leaks
```

### 9. Iterator Crashes / Dangling References

**Likely cause**: Missing keep-alive relationship or wrong `rv_policy`.

**Fix**: Use `nb::make_iterator<nb::rv_policy::reference_internal>(...)` and `nb::keep_alive<0, 1>()` where the iterator views internal storage.

### 10. Generic "Buffer-like" Arguments May Need Manual Dtype Dispatch

Nanobind will happily cast a `nb::object` to "some vector type", but if you accept generic python sequences, you can accidentally cast to the wrong C++ type.

**Prefer:**
- explicit `nb::ndarray` inputs, or
- an explicit dtype + switch-based cast.

### 11. Error Handling

Pybind11's `throw py::type_error` translated to regular `std::invalid_argument` or `nb::raise<TypeError>()` in nanobind.

---

## Module Initialization Best Practices

Both pybind11 and nanobind follow these rules (nanobind tends to enforce them more strictly):

1. **Bind types before functions**: Classes/enums must be registered before any `def(...)` that references them.
2. **Define shared submodules exactly once**: All `def_submodule(...)` calls should live in the module entrypoint, then individual components attach to those submodules.

**Example from codebase**: The comment block in `ttnn/cpp/ttnn-{pybind,nanobind}/__init__.cpp` describes this explicitly.

---

## Migration Checklist

When creating `*_nanobind.cpp` from `*_pybind.cpp`:

- [ ] **Headers**
  - Replace `#include <pybind11/pybind11.h>` with `#include <nanobind/nanobind.h>`
  - Replace `#include <pybind11/stl.h>` with specific `#include <nanobind/stl/*.h>` includes
  - Add any additional nanobind headers needed (`make_iterator.h`, `ndarray.h`, etc.)

- [ ] **Namespace + Module Type**
  - `py::module& module` → `nb::module_& mod`
  - `namespace py = pybind11;` → `namespace nb = nanobind;`

- [ ] **Macros**
  - `PYBIND11_MODULE` → `NB_MODULE` (only in module entrypoint)

- [ ] **Arguments and Keyword-Only**
  - `py::arg(...)` → `nb::arg(...)`
  - `py::kw_only()` → `nb::kw_only()` (ensure correct placement)
  - `std::nullopt` / `py::none()` defaults → `nb::none()`

- [ ] **Class Bindings**
  - `py::class_<T>` → `nb::class_<T>`
  - `.def_readwrite` → `.def_rw`
  - `.def_readonly` → `.def_ro`
  - `.def_property_readonly` → `.def_prop_ro`
  - `.def_property` → `.def_prop_rw`
  - `py::init<>()` → `nb::init<>()`
  - Remove holder types (e.g., `std::shared_ptr<T>`) from `class_` template

- [ ] **Overloads**
  - `py::overload_cast<...>(..., py::const_)` → `nb::overload_cast<...>(..., nb::const_)`
  - If the pybind version used `pybind_overload_t`, use `nanobind_overload_t`
  - For composite operations, follow the repo's nanobind decorator patterns

- [ ] **Lifetimes and Return Policies**
  - `py::keep_alive` → `nb::keep_alive`
  - `py::return_value_policy::*` → `nb::rv_policy::*`

- [ ] **Module Imports**
  - `py::module::import("name")` → `nb::module_::import_("name")`

- [ ] **Enums**
  - `py::enum_<E>` → `nb::enum_<E>`

- [ ] **Iterators**
  - Add `#include <nanobind/make_iterator.h>`
  - Update `make_iterator` calls with scope/name parameters and return value policy
  - Add appropriate `keep_alive` policies

- [ ] **Buffer Protocol**
  - Replace `py::buffer_protocol()` and `.def_buffer()` with `nb::ndarray` and `__array__` method
  - Add DLPack hooks (`__dlpack__`, `__dlpack_device__`) where appropriate

- [ ] **Optional Parameters**
  - Replace `std::nullopt` defaults with `nb::none()`
  - For properties that should accept `None`, use `nb::arg("value") = nb::none()` in `def_prop_rw`

- [ ] **Custom Type Casters**
  - Update custom type casters to use nanobind API (`NB_TYPE_CASTER`, `from_python`, `from_cpp`)
  - Include custom caster headers in binding translation units

- [ ] **Containers / Spans / Custom Collections**
  - Ensure you have casters for STL containers
  - For TTNN custom types (e.g. `SmallVector`), include the repo's caster headers (`ttnn-nanobind/small_vector_caster.hpp`) in the binding TU

---

## Debugging Checklist

When a nanobind port fails, check:

- **`TypeError: ... incompatible function arguments`**
  - Add missing `nanobind/stl/*.h` casters for the argument/return types.
  - Ensure your `SmallVector`/custom casters are included in that `.cpp`.

- **Segfaults / use-after-free**
  - Audit `keep_alive` relationships.
  - Audit `rv_policy` (especially iterator + returned references).

- **Python tuple rejected**
  - Add an overload taking `nb::detail::tuple<...>`.
  - Consider adding `nb::implicitly_convertible<...>()`.

- **Array/buffer interop regressions**
  - Replace buffer protocol assumptions with `nb::ndarray` + DLPack.

- **None values rejected**
  - Ensure optional parameters default to `nb::none()`.
  - For properties, use `nb::arg("value") = nb::none()` in `def_prop_rw`.

- **Compile-time errors**
  - Check `kw_only()` placement (must be before keyword-only args).
  - Verify `nb::none()` is used as a value, not a reference.
  - Ensure STL type casters are included in the translation unit.

- **Default argument won't bind**
  - Wrap default in `nb::cast(...)` for enums, arrays, complex objects.
