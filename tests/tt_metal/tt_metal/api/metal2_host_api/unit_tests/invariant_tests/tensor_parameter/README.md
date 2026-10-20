# tensor_parameter invariant tests

Local invariants of `TensorParameter` (`tensor_parameter.hpp`). It has none, so this folder has no test files. The rule
on the TensorParameter a DFB borrows from is in `../dataflow_buffer_spec/`; that every TensorParameter name resolves
and every TensorParameter is used are structural and live in `../program_spec/`.

## Listed invariants

`TensorParameter` as declared in `tensor_parameter.hpp`, with every field and only its invariants.
To be filled in as the header grows.

```cpp
struct TensorParameter {
    TensorParamName unique_id;

    tt::tt_metal::TensorSpec spec;

    TensorSpecRelaxations relaxations;
};
```
