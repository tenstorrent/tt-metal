# ArgValue tests

`ArgValue` is the type of every kernel argument value. It accepts the numbers users write in place, so idioms like
`runtime_arg_values["n"][node] = n` and `{{"Ht", 8}}` compile as they did when values were `uint32_t`.

- `arg_value.cpp` tests the type itself: conversions, comparison, hashing and formatting.
- `api_idioms.cpp` pins the user-facing idioms that must keep compiling, mostly at compile time.

How argument values reach the kernel is tested with the rest of each behavior: run-time values in
[program_run_args](../program_run_args/) (`named_runtime_args.cpp`), compile-time values in
[program](../program/) (`kernel_args_layout.cpp`).
