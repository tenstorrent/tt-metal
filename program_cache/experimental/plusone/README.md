### plusone program cache review

- OP: `ttnn/cpp/ttnn/operations/experimental/plusone`
- Infra: old/type-erased `ProgramWithCallbacks`

Findings:
- Single-core program. Reader kernel runtime args contain only the input buffer base address.
- Override updates the reader runtime arg[0] with `input.buffer()->address()` for cache-hit runs.
- No other runtime-only scalars; default hash suffices.

No issues identified.
