Batch Norm — Program Cache Review

- Status: Reviewed — no program-cache issues found

Summary
- Uses new infra with a helper `set_or_update_runtime_arguments` invoked both in create and override. Reader args include eps scalar and input base address plus per-core indices; writer args include batch_mean, batch_var, optional weight/bias, and output base addresses; compute args are small counters.
- Override path copies the full argument arrays per core, correctly updating all base addresses and leaving hashed/derived quantities intact.

Key references
- Program factory: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/batch_norm_program_factory.cpp`
  - Shared helper `set_or_update_runtime_arguments`: around L21–L121.
  - Create calls helper with SetRuntimeArgs: around L302–L315.
  - Override calls helper and memcpy-updates per-core rtargs: around L317–L339.

Notes
- Weight/bias presence is encoded in compile-time defines and hashed attributes; override respects presence flags and updates addresses consistently.
