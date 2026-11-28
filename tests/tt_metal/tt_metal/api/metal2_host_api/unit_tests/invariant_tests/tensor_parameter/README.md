# tensor_parameter invariant tests

Local invariants of `TensorParameter` (`tensor_parameter.hpp`). The MX tile rule applies to both common and per-node
tensor parameters and is covered by [resource tile metadata tests](../../resource_tile_metadata.cpp). The rule on the
TensorParameter a DFB borrows from is in `../dataflow_buffer_spec/`; that every TensorParameter name resolves and
every TensorParameter is used are structural and live in `../program_spec/`.

## Listed invariants

`TensorParameter` as declared in `tensor_parameter.hpp`, with every field and only its invariants.

```cpp
struct TensorParameter {
    TensorParamName unique_id;

    // Invariant: For an MX data type, the tensor's tile contains a multiple of 32 elements.
    tt::tt_metal::TensorSpec spec;

    TensorSpecRelaxations relaxations;
};
```
