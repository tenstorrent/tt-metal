# Layer 4: Device Operation Interface

DeviceOperation struct and type definitions.

## Type Definitions

```yaml
types:
  operation_attributes_t:
    content: [bool, float, DataType, MemoryConfig, scalars]
    rules:
      - "Must be hashable"
      - "No Tensor references"
      - "No raw pointers"

  tensor_args_t:
    content: ["const Tensor&", "optional<Tensor>"]
    rules:
      - "All input tensors"
      - "Optional preallocated output"

  spec_return_value_t:
    types: [TensorSpec, "vector<TensorSpec>"]

  tensor_return_value_t:
    types: [Tensor, "vector<Tensor>"]
```

## Static Methods

```yaml
methods:
  select_program_factory:
    signature: "static program_factory_t select_program_factory(attrs, tensor_args)"
    purpose: "Dispatch to correct factory"

  validate_on_program_cache_miss:
    signature: "static void validate_on_program_cache_miss(attrs, tensor_args)"
    checks: [Layouts, DTypes, Shapes, Alignment, MemoryConfig]

  validate_on_program_cache_hit:
    signature: "static void validate_on_program_cache_hit(attrs, tensor_args)"
    typical: "Delegates to cache_miss"

  compute_output_specs:
    signature: "static spec_return_value_t compute_output_specs(attrs, tensor_args)"
    rule: "No side effects"

  create_output_tensors:
    signature: "static tensor_return_value_t create_output_tensors(attrs, tensor_args)"
    pattern: "Return preallocated or create_device_tensor()"

  compute_program_hash:
    signature: "static hash_t compute_program_hash(attrs, tensor_args)"
    include: [structural_attrs, input_dtype, memory_config, shape_volume]
    exclude: [seed, addresses]

  invoke:
    signature: "static tuple<attrs, tensor_args> invoke(...)"
    rule: "Pack args only, no logic"
```

## Prim Registration

```yaml
registration:
  namespace: "ttnn::prim"
  pattern: |
    constexpr auto <op> = ttnn::register_operation<
        "ttnn::prim::<op>",
        <Op>DeviceOperation>();
  invariant: "All calls via ttnn::prim::<op>, not struct directly"
```

## Invariants

```yaml
invariants:
  - { id: I1, rule: "All methods are static" }
  - { id: I2, rule: "No instance state" }
  - { id: I3, rule: "Same attrs/args = same hash" }
  - { id: I4, rule: "compute_output_specs consistent with create_output_tensors" }
```
