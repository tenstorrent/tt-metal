### prefix_scan program cache review

- OP: `ttnn/cpp/ttnn/operations/experimental/ssm/prefix_scan`

Findings:
- Program uses CBs globally attached to a/bx/h/output buffers and updates them on cache hits via `UpdateDynamicCircularBufferAddress(...)`.
- Per-core reader/writer/compute runtime args are recomputed from current tensor shapes in override.

No issues identified.
