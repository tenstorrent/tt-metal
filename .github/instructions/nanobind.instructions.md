---
description: 'PR review for nanobind Python/C++ bindings — signature correctness, lifetime safety, and common pitfalls'
applyTo: 'ttnn/**/*nanobind*.cpp,ttnn/**/*nanobind*.hpp,ttnn/**/nanobind/**,tt-train/**/nanobind/**'
excludeAgent: "cloud-agent"
---

# Nanobind Bindings Review

For full reference on conventions and common pitfalls, see [contributing/Nanobind.md](../../contributing/Nanobind.md).

## 🔴 CRITICAL

- **Signature mismatch**: the bound function's Python-facing argument types, order, and defaults must exactly match the C++ declaration. A mismatch silently produces wrong results or segfaults at runtime.
- **Lifetime violation**: returning a raw pointer or reference from C++ to Python without an appropriate return value policy (`nb::rv_policy::reference_internal`, `keep_alive`, etc.) creates a dangling reference. Flag any missing lifetime annotation on functions returning pointers or references.
- **Missing `nb::kw_only()` placement**: arguments after `nb::kw_only()` become keyword-only in Python. Misplacement silently changes the public API.
- **`std::unique_ptr` returned without nanobind deleter**: returning a `std::unique_ptr` to Python requires the nanobind custom deleter. Use `nbh::steal_rewrap_unique` from `ttnn-nanobind/nanobind_helpers.hpp`. Without it, Python will call the wrong deleter on garbage collection.

## 🟡 IMPORTANT

- **`shared_ptr`/`unique_ptr` in `nb::class_<>` template args**: nanobind does not need holder types in the class template. Flag `nb::class_<T, std::shared_ptr<T>>` — it causes subtle issues.
- **Use `= nb::none()` for optional default values**: if a parameter defaults to `None` in Python, the binding must use `nb::arg("name") = nb::none()`. Using `= std::nullopt` does NOT work — it silently fails to accept `None` from Python.
- **`.noconvert()` with optionals**: using `.noconvert()` on `std::optional` arguments can reject valid `None` values in some contexts. Flag it when combined with `std::optional<std::reference_wrapper<T>>` or when the argument has no default — in those cases nanobind will refuse the implicit `None` conversion.
- **Missing STL typecaster includes**: if a bound function uses STL containers (`std::vector`, `std::variant`, `std::optional`, etc.) in its signature, the corresponding `<nanobind/stl/...>` header must be included. Missing typecasters produce confusing `TypeError` at runtime. Check `typedef`/`using` aliases — they often hide containers.
- **`std::reference_wrapper` not supported**: nanobind doesn't handle `std::reference_wrapper`. Use `std::optional<T*>` instead and `nbh::rewrap_optional` to convert at the binding boundary.
- **Overload resolution**: when binding overloaded functions, all overloads must be registered. A missing overload causes confusing `TypeError` at runtime instead of a compile-time failure.
- **Custom constructors require placement new**: nanobind custom constructors must use `.def("__init__", [](T* self, ...) { new (self) T(...); })` with explicit placement new. Simple constructors can use `nb::init<Args...>()`. Flag factory-style lambdas that return a constructed object instead of using placement new — they won't work in nanobind.

## 🟢 SUGGESTION

- Group related bindings together (e.g., all methods of a class in one block) for readability.
- Add a Python-visible docstring (`nb::doc(...)`) for every public-facing function.
- Prefer `nb::rv_policy::move` for functions returning by value to avoid unnecessary copies.
- Use `mod` or `m` for the module variable name — `module` clashes with the C++20 keyword.

## Review Checklist

- [ ] Bound signature matches C++ declaration (types, order, defaults)
- [ ] Return value policy specified for pointer/reference returns
- [ ] `nb::kw_only()` placement matches intended Python API
- [ ] No holder types in `nb::class_<>` template args
- [ ] Optional parameters use `nb::arg("name") = nb::none()`, not `std::nullopt`
- [ ] `.noconvert()` not used on optionals without a default value
- [ ] All required `<nanobind/stl/...>` headers included for container types
- [ ] All overloads registered
- [ ] `std::unique_ptr` returns use nanobind-compatible deleter
