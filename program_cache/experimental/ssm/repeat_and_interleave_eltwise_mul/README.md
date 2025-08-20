### repeat_and_interleave_eltwise_mul program cache review

- OP: `ttnn/cpp/ttnn/operations/experimental/ssm/repeat_and_interleave_eltwise_mul`

Findings:
- Program computes reader/writer/compute runtime args per core using input/output buffers and shapes; override recomputes them and updates buffer base addresses.
- Compile-time defines depend on shapes but are part of the hash via tensor args; default hashing is sufficient.

No issues identified.
