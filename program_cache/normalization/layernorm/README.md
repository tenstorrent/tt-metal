LayerNorm program cache review

Status: Reviewed — no program cache issues found.

Reviewed files
- `device/multi_core/layernorm_op_multi_core.cpp`

Findings
- Interleaved and sharded variants use override to update:
  - Input/output base addresses in reader/writer kernel args.
  - Optional `b`, `gamma`, `beta`, and `stats` buffer addresses when present.
  - Dynamic CB addresses for sharded inputs/outputs and optional tensors.
- All compile-time selections (block sizes, kernel variants, mcast/mode flags) are keyed by shapes/layouts and attributes; override refrains from changing hashed properties.

Recommendation
- No changes required.
