# Binding compilation tests

Compile-only tests of kernels that construct resources from the binding tokens that Metal 2.0 emits into
`kernel_bindings_generated.h` (`dfb::<name>`, `scratch::<name>`, `tensor::<name>`). Mock Wormhole
(`ProgramSpecTestGen1`).

| File | Tests | What is compiled |
|---|---|---|
| `tensor_bindings.cpp` | 7 | A DM kernel building a `TensorAccessor` from `tensor::<name>`; tensor binding sequences with several, empty and singleton member lists, one binding in two sequences, on a compute kernel, and a sequence named like a DFB accessor |
| `scratchpad_bindings.cpp` | 3 | `Scratchpad` from `scratch::<name>` in DM and compute kernels; range-based `for` over a scratchpad |
| `get_token_if_present.cpp` | 8 | `get_token_if_present<"name">()` returns the token or `nullptr`; constructing a DFB, scratchpad or tensor accessor from it; kernels with no bindings; telling several bindings apart |

The spec-level rules for these bindings are in
[`../../unit_tests/invariant_tests/kernel_spec/`](../../unit_tests/invariant_tests/kernel_spec/) and
[`../../unit_tests/invariant_tests/advanced_options/`](../../unit_tests/invariant_tests/advanced_options/).
