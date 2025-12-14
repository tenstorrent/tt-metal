# Resources:

Official Docs: https://nanobind.readthedocs.io/en/latest/

Porting Guide: https://nanobind.readthedocs.io/en/latest/porting.html
* Has a top level of converting from pybind11 -> nanobind.

Github Discussion Pages: https://github.com/wjakob/nanobind/discussions
* Searchable Q&A with the devs.

# Common pybind -> nanobind issues in practice

### `shared_ptr` / `unique_ptr` in `nb::class_` template arguments not needed
https://nanobind.readthedocs.io/en/latest/porting.html#shared-pointers-and-holders

### Custom constructors require placement new
https://nanobind.readthedocs.io/en/latest/porting.html#custom-constructors

### `TypeError: __call__(): incompatible function arguments`

#### kw_only misplaced or kwarg used without keyword

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
