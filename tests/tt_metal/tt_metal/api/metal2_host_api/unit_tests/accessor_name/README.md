# AccessorName tests

`AccessorName` (alias `ArgName`) is the type of every name kernel code sees: binding accessor names, argument names
and tensor binding sequence names. It accepts the strings users write in place, so idioms like
`.runtime_arg_names = {"num_tiles"}` and `runtime_arg_values["n"][node] = n` compile as they did when names were
`std::string`.

- `accessor_name.cpp` tests the type itself: conversions, comparison, ordering, hashing and formatting.
- `api_idioms.cpp` pins the user-facing idioms that must keep compiling, mostly at compile time.
