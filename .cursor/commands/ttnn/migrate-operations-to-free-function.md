# Migrate operations to free function

## Pattern migration from struct-based operations to free functions:

Before:

```c++
struct Operation {
    static Tensor invoke(...);
};
constexpr auto operation = ttnn::register_operation<"ttnn::operation", Operation>();

// Python binding
bind_registered_operation(mod, ttnn::operation, doc, nanobind_overload_t{...});
```

After:

```c++
Tensor operation(...);

// Python binding - single function without overloads
ttnn::bind_function<"operation">(mod, doc, &operation, args...);

// Python binding - multiple overloads
ttnn::bind_function<"operation">(mod, doc, ttnn::overload_t(&operation, args...), ...);
```

- The original code using `bind_registered_operation` should be replaced with the new code that uses `bind_function` function.
- "Operation" structures should be rewritten to use free functions instead.
- ALWAYS remove original structure definition
- NEVER create standalone function wrapper around original function definition, ALWAYS rewrite the operation struct and the invoke function to the free function definition.

## Common pitfals

1. Using `bind_function` with lambda argument without decaying it to pointer  -- `bind_function/overload_t` argument
2. Using functions accepting `Span` as an argument, or using lambdas with `Span` argument -- the conversion should accept `SmallVector` instead, and then pass it to the `Span`.
3. NEVER change the names of the python or C++ arguments.
4. NEVER change the order of placement of the KV arguments in the python bindings.
5. Use `overload_t` only if there are more than one argument in the `bind_function`.
6. Transition the python docstrings as is
7. NEVER add or remove documentation comments or comments in the implementation. Transition all comments exactly as they are used in the code.
8. ALWAYS preserve the comments and annotatiosn on the nanobind arguments.
9. NEVER introduce new macro-based solutions to the code, use templates.
10. NEVER change structure and text of the documentation comments and documentation strings for python and C++ code.
11. NEVER add or remove default values for the nanobind wrappers.
12. NEVER introduce empty namespace definitions with no content. `namespace ttnn::operations::data_movement {}  // namespace ttnn::operations::data_movement` -- code like this is not allowed

## Migration operations


- Order of migration:
  - Migrate the hpp operation structure definitions to free functions
  - Migrate operation structure and invoke definition to the free function in the cpp file
  - Migrate the python nanobind wrappers for the new header API
- Do not suggest to run the build command during migration, the developer will perform build manually and report any issues back.
- ALWAYS Migrate operations in batch -- if the nanobind file contains more than one operation slated for migration, rewrite them all in one go, and then adjust other relevant files where the operation logic is defined.

##

- Do not use local lambdas for wrapping the functions. If the function signature cannot be used directly in the wrapper (due to span usage, multiple overloads, changed order of arguments in the actual lambda and function), consider two options:
  - Use `nb::overload_cast` to select the correct overload -- in case the lambda does not change the order of arguments between the python bindings and the C++ API
  - Define an overload wrapper in the anonymous namespace if the function needs to perform additional operations on the arguments, or it needs to change the order of arguments between the python and C++ code.

Find all local lambdas used in nanobind wrapping that pass the arguments to the function, without modifying any of the values. Apply the conversion rules. If the call to `bind_function` already uses the overload cast, regular function pointer etc., do not change this -- only change the code if it uses the lambda.
