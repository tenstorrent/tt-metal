# Layer 4: Program Factory Interface

Program creation and runtime argument override.

## Shared Variables

```yaml
shared_variables_t:
  purpose: "Handles for updating program without recompile"
  content:
    - { type: KernelHandle, examples: [reader_id, writer_id, compute_id] }
    - { type: CBHandle, examples: [cb_in, cb_out] }
    - { type: CoreRangeSet, examples: [core_group_1, core_group_2] }
    - { type: "vector<CoreCoord>", examples: [cores] }
    - { type: uint32_t, examples: [tiles_per_core, num_cores] }
```

## Create Method

```yaml
create:
  signature: "static cached_program_t create(attrs, tensor_args, output)"

  steps:
    1: { action: "CreateProgram()", result: program }
    2: { action: "CreateCircularBuffer()", result: cb_handles }
    3: { action: "CreateKernel()", result: kernel_handles }
    4: { action: "SetRuntimeArgs()", args: "addresses, seeds, counts" }
    5: { action: "return", value: "cached_program_t{program, shared_vars}" }

  order_invariant: "CBs → Kernels → RuntimeArgs"
```

## Override Method

```yaml
override:
  signature: "static void override_runtime_arguments(cached_program, attrs, tensor_args, output)"

  allowed:
    - "UpdateDynamicCircularBufferAddress()"
    - "SetRuntimeArgs() for addresses, seeds"

  forbidden:
    - "Change CB size"
    - "Change kernel defines"
    - "Change core grid"
```

## CB Creation

```yaml
cb_creation:
  standard:
    config: "CircularBufferConfig(size, {{index, format}}).set_page_size(index, page_size)"
    call: "CreateCircularBuffer(program, core_grid, config)"

  zero_copy:
    config: "...set_globally_allocated_address(*buffer)"
    update: "UpdateDynamicCircularBufferAddress(program, cb_handle, *buffer)"
```

## Kernel Creation

```yaml
kernel_creation:
  reader:
    config: "ReaderDataMovementConfig(compile_args)"

  writer:
    config: "WriterDataMovementConfig(compile_args)"

  compute:
    config: |
      ComputeConfig{
        .math_fidelity = HiFi4,
        .fp32_dest_acc_en = false,
        .math_approx_mode = false,
        .compile_args = compile_args
      }
```

## Argument Binding

```yaml
argument_rules:
  compile_time:
    - "CB indices"
    - "Block sizes"
    - "LLK parameters (converted from float)"

  runtime:
    - "Buffer addresses"
    - "Tile counts"
    - "Offsets"
    - "Seeds"
```

## Mesh Workload

```yaml
mesh_workload:
  type: "AdaptedCachedMeshWorkload<shared_variables_t>"
  use_case: "Per-device coordinate logic (e.g., seed + device_id)"
  pattern: |
    for mesh_coord in tensor_coords:
        attrs.seed += device->id()
        create(attrs, tensor_args, output)
```

## Invariants

```yaml
invariants:
  - { id: I1, rule: "Resource order: CBs → Kernels → RuntimeArgs" }
  - { id: I2, rule: "CT args immutable after create()" }
  - { id: I3, rule: "RT args mutable in override()" }
  - { id: I4, rule: "shared_vars contains all handles needed for override" }
  - { id: I5, rule: "Same program reusable with different RT args" }
```
