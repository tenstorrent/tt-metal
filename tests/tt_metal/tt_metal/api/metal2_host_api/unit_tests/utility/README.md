# Utility tests

Host-only tests, with no device, of the containers in `metal2_host_api/utility/`.

Suites: `TableTest`, `TableMiscTest`, `TableHashTest`.

| File | Tests | What is covered |
|---|---|---|
| `table.cpp` | 45 | `Table<K, V>` (`table.hpp`): construction, lookup, insert, emplace, erase, iteration, equality, hashing |

A `Table` never holds duplicate keys: building one from an initializer list, span or range with a repeated key
throws, and `operator[]`, `insert` and `emplace` never add a second entry for a key. That is why Table-typed spec
fields such as `compile_time_args`, `compiler_options.defines`, `unpack_modes` and `num_runtime_varargs_per_node`
need no uniqueness check of their own.

`group.hpp` has no tests yet.
