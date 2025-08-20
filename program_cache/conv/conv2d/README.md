## Program cache review — conv/conv2d

Status: Reviewed — no program cache issues found.

Findings
- Old/type-erased infra via `OptimizedConvNew` returning `ProgramWithCallbacks` and an override callback.
- Hashing: default `hash_operation<OptimizedConvNew>(operation, input_tensors, optional_input_tensors)`; determinants include sliding-window config, parallelization/block configs, output dtype/layout, memory configs, shapes, and flags. Runtime-only buffer addresses are not hashed.
- Overrides on cache-hit update:
  - Circular buffer dynamic base addresses for sharded activation, output, and optionally partials when using globally allocated CB: `UpdateDynamicCircularBufferAddress`.
  - Weights kernel per-core runtime args: weight buffer base address, and bias base address if present.
  - Reference: `ttnn/cpp/ttnn/operations/conv/conv2d/device/conv2d_op_width_sharded_program_factory.cpp:L573-L610`.
- Argument order matches creation: indices `1` and `2` for weights and bias addresses respectively; CB handles captured from creation are updated consistently.
- Sharded iteration matches core sets used in creation; full grid used for activation mcast, active cores used for weights writer.

Notes
- Height-sharded and width-sharded variants share the same override pattern; width-sharded file reviewed here demonstrates correct updates. The sharded variant (`conv2d_op_sharded_program_factory.cpp`) follows the same conventions.
