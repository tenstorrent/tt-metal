# Name and value type tests

`AccessorName` (alias `ArgName`) is the type of every name kernel code sees; `ArgValue` is the type of every kernel
argument value. Both accept the strings and numbers users write in place, so idioms like
`runtime_arg_values["n"][node] = n` and `{{"Ht", 8}}` compile as they did when names were `std::string` and values
`uint32_t`.

- `accessor_name.cpp` and `arg_value.cpp` test the types themselves: conversions, comparison, hashing and
  formatting.
- `api_idioms.cpp` pins the user-facing idioms that must keep compiling, mostly at compile time.

How argument values reach the kernel is tested with the rest of each behavior: run-time values in
[program_run_args](../program_run_args/) (`named_runtime_args.cpp`), compile-time values in
[program](../program/) (`kernel_args_layout.cpp`).
