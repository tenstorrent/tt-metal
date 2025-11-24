# Device Operation Specification Schema

YAML blocks are the source of truth. Prose is minimal.

## Layer Index

| Layer | Scope | Document Pattern |
|:------|:------|:-----------------|
| 1 | Algorithm | `per-OP/<op>.md` → Section 1 (`algorithm:` block) |
| 2 | Hardware Strategy | `per-OP/<op>.md` → Section 2 (`hw_strategy:` block) |
| 3 | LLK Selection | `per-OP/<op>.md` → Section 3 (`llk_selection:` block) |
| 4 | Data Flow | `per-OP/<op>.md` → Section 4 (`data_flow_graph:` block) |
| 5 | Optimizations | `per-OP/<op>.md` → Section 5 (`optimizations_applied:` block) |
| 6 | Optimized Algorithm | `per-OP/<op>.md` → Section 6 (`optimized_algorithm:` block) |
| 7 | Full Implementation | `per-OP/<op>.md` → Section 7 (kernel code) |
| 8 | C++ Binding | `per-OP/<op>.md` → Section 8 (`cpp_binding:` block) |
| LLK | Primitive Catalog | `LLK/primitives_catalog.md` |

## YAML Schema Reference

### Preconditions

```yaml
preconditions:
  - id: P1
    entity: Input.shape        # Target entity
    attr: rank                  # Attribute being checked
    rel: "=="                   # Relation: ==, !=, <, <=, >, >=, %, in
    value: 4                    # Expected value
```

### Postconditions

```yaml
postconditions:
  - id: O1
    entity: Output.shape
    attr: value
    rel: "=="
    expr: "[Input.shape[0], 1, Input.shape[3], Input.shape[2]]"
```

### Circular Buffers

```yaml
circular_buffers:
  - name: cb_in
    index: c_0
    size: "tiles_per_core * tile_size(input_dtype)"
    page_size: "tile_size(input_dtype)"
    format: input_dtype
    zero_copy: true              # Optional, default false
    buffer: "input.buffer()"     # Required if zero_copy
```

### Kernels

```yaml
kernels:
  reader:
    path: "device/kernels/reader_<op>.cpp"
    type: ReaderDataMovement
    compile_args:
      - { idx: 0, name: cb_in, expr: "tt::CBIndex::c_0" }
    runtime_args:
      - { idx: 0, name: total_tiles, expr: "total_tiles_per_core" }

  compute:
    path: "device/kernels/compute/<op>.cpp"
    type: Compute
    config:
      math_fidelity: HiFi4
      fp32_dest_acc_en: false
      math_approx_mode: false
    compile_args:
      - { idx: 0, name: cb_in, expr: "tt::CBIndex::c_0" }
      - { idx: 1, name: cb_out, expr: "tt::CBIndex::c_2" }
    runtime_args:
      - { idx: 0, name: seed, expr: "args.seed" }

  writer:
    path: "device/kernels/writer_<op>.cpp"
    type: WriterDataMovement
    compile_args:
      - { idx: 0, name: cb_out, expr: "tt::CBIndex::c_2" }
    runtime_args:
      - { idx: 0, name: dst_addr, expr: "output_buffer->address()" }
```

### LLK Contract

```yaml
llk:
  ref: "LLK/<primitive>.md"
  init:
    - "init_sfpu(cb_in, cb_out)"
    - "dropout_kernel_init(seed)"
  primitives:
    - name: dropout_tile
      params: [idst, probability_int, scale_bits]
      pre: [DST_ACQUIRED, DST_HAS_DATA]
      post: [DST_MODIFIED]
  cleanup: []
```

### Shared Variables

```yaml
shared_variables:
  - { name: cb_in, type: CBHandle }
  - { name: cb_out, type: CBHandle }
  - { name: reader_kernel_id, type: KernelHandle }
  - { name: writer_kernel_id, type: KernelHandle }
  - { name: compute_kernel_id, type: KernelHandle }
  - { name: cores, type: "vector<CoreCoord>" }
  - { name: tiles_per_core, type: uint32_t }
```

### Override Targets

```yaml
override_targets:
  - target: cb_in
    method: UpdateDynamicCircularBufferAddress
    value: "input_buffer"
  - target: reader_kernel_id
    arg_idx: 0
    value: "src_buffer->address()"
  - target: compute_kernel_id
    arg_idx: 0
    value: "args.seed"
```

### Types

```yaml
types:
  operation_attributes:
    - { name: memory_config, type: MemoryConfig }
    - { name: dtype, type: DataType }
    - { name: seed, type: uint32_t, default: 0 }
  tensor_args:
    - { name: input, type: "const Tensor&" }
    - { name: preallocated_output, type: "optional<Tensor>" }
```

## File Structure

```
_dev_TODO.now/
├── Idea.md                     # Vision and principles
├── Global_Structural.md        # This file (YAML schema reference)
├── Global_Architecture.md      # HW overview, memory, work distribution
├── Global_Algorithm.md         # Section 1 concepts
├── Common_Optimizations.md     # Reusable optimization patterns
├── Global_Cpp_*.md             # C++ interface concepts
│
├── LLK/
│   ├── primitives_catalog.md   # CENTRAL: All primitives, state machines
│   ├── dropout_tile.md         # Per-primitive details
│   ├── copy_tile.md
│   ├── pack_tile.md
│   ├── transpose_wh_tile.md
│   ├── pack_untilize_dest.md
│   ├── eltwise_unary_dropout.md  # Pattern usage
│   └── transpose_untilize.md     # Pattern usage
│
└── per-OP/
    ├── dropout.md              # Full spec (Sections 1-9)
    └── convert_to_chw.md       # Full spec (Sections 1-8)
```

## Invariants

```yaml
global_invariants:
  - id: G1
    rule: "CB indices passed as compile-time args"
  - id: G2
    rule: "Buffer addresses passed as runtime args"
  - id: G3
    rule: "LLK params converted in create() before passing to kernels"
  - id: G4
    rule: "CBs created before Kernels, Kernels before RuntimeArgs"
  - id: G5
    rule: "Zero-copy CB requires set_globally_allocated_address()"
```
