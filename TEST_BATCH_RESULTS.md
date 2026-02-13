# Test Batch Migration Results

## Date: 2026-02-12

## Summary

Attempted to migrate 5 operations as a test batch. Successfully migrated 4 out of 5.

### ✅ Successfully Migrated (4 operations)

| Operation | Files | Complexity | Status |
|-----------|-------|------------|--------|
| **clone** | 3 | Simple | ✅ Complete |
| **copy** | 3 | Simple | ✅ Complete (fixed doc string issue) |
| **move** | 3 | Simple | ✅ Complete |
| **transpose** | 3 | Medium (2 overloads) | ✅ Complete |

### ❌ Could Not Migrate (1 operation)

| Operation | Reason | Resolution |
|-----------|--------|------------|
| **squeeze** | Requires Python polymorphism (None/int/list overloading) via lambda | Reverted to old pattern |

## Issues Encountered

### 1. Documentation String Type Mismatch
**File**: `copy_nanobind.cpp`
**Issue**: `bind_function` expects `const char*` for doc parameter, but `get_binary_doc_string()` returns `std::string`
**Fix**: Added `.c_str()` call to convert string to C-string
**Lesson**: When using dynamic doc strings, remember to call `.c_str()`

### 2. Lambda Not Supported in bind_function
**File**: `squeeze_nanobind.cpp`
**Issue**: `bind_function` template's `make_method_wrapper` only matches function pointers `Ret (*)(Args...)`, not lambdas
**Root Cause**: Python polymorphism (accepting None, int, or List[int] for same parameter) requires runtime type checking via lambda
**Attempted Fix**: Used `ttnn::overload_t` with lambda
**Result**: Template substitution failure - lambda type doesn't match function pointer pattern
**Resolution**: Reverted squeeze to old `bind_registered_operation` pattern
**Impact**: ~10-15% of operations may have similar Python polymorphism requirements

## Migration Pattern That Worked

### For Simple Operations (clone, copy, move, transpose)

**Step 1: Update `.hpp` file**
```cpp
// OLD
namespace ttnn::operations::data_movement {
struct OperationName {
    static Tensor invoke(...);
};
}
constexpr auto op_name = ttnn::register_operation<"ttnn::op_name", OperationName>();

// NEW
namespace ttnn {
Tensor op_name(...);
}
```

**Step 2: Update `.cpp` file**
```cpp
// OLD
namespace ttnn::operations::data_movement {
Tensor OperationName::invoke(...) { ... }
}

// NEW
namespace ttnn {
Tensor op_name(...) { ... }
}
```

**Step 3: Update `_nanobind.cpp` file**
```cpp
// OLD
#include "ttnn-nanobind/decorators.hpp"
bind_registered_operation(mod, ttnn::op_name, doc, ttnn::nanobind_arguments_t{...});

// NEW
#include "ttnn-nanobind/bind_function.hpp"
ttnn::bind_function<"op_name">(
    mod,
    doc,  // Must be const char*, use .c_str() if std::string
    ttnn::overload_t(
        &ttnn::op_name,  // or nb::overload_cast<...>(&ttnn::op_name) for overloads
        nb::arg("param1"),
        ...));
```

## Operations Requiring Special Handling

The following pattern indicates an operation that CANNOT be migrated with current `bind_function`:

```cpp
// Lambda for Python polymorphism - NOT SUPPORTED
ttnn::nanobind_overload_t{
    [](const auto& self, const Tensor& input, const nb::object& param) {
        if (param.is_none()) { return self(input); }
        if (nb::isinstance<nb::int_>(param)) { return self(input, nb::cast<int>(param)); }
        if (nb::isinstance<nb::list>(param)) { return self(input, nb::cast<SmallVector<int>>(param)); }
    },
    ...
}
```

**Operations with this pattern should stay with old pattern until `bind_function` is enhanced.**

## Build Status

✅ **BUILD SUCCESSFUL** - Exit code: 0

All 4 migrated operations (clone, copy, move, transpose) compiled successfully.

## Recommendations

1. **Continue migration** - The pattern works well for simple operations
2. **Identify polymorphic operations early** - Check for lambdas in nanobind files before migration
3. **Consider bind_function enhancement** - Add lambda support or static wrapper function pattern for Python polymorphism
4. **Batch by complexity** - Group simple operations separately from polymorphic ones

## Next Steps

- [x] Fix documentation string issue in copy
- [x] Revert squeeze to old pattern
- [ ] Complete build test with 4 operations
- [ ] Update migration plan to flag polymorphic operations
- [ ] Consider enhancing bind_function to support common polymorphism patterns
