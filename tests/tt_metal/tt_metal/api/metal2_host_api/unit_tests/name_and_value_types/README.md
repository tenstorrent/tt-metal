# Name and value type tests

`AccessorName` (alias `ArgName`) is the type of every name kernel code sees; `ArgValue` is the type of every kernel
argument value. Both must accept the strings and numbers users already write, so the API idioms built on them keep
compiling unchanged.

- `accessor_name.cpp` and `arg_value.cpp` test the types themselves: conversions, comparison, hashing and
  formatting.
- `api_idioms.cpp` pins the user-facing idioms that must keep compiling, mostly at compile time.

How argument values reach the dispatch buffer is tested with the rest of the run-args behavior, in
[program_run_args](../program_run_args/).
