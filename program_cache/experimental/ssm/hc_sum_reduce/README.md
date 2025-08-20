### hc_sum_reduce program cache review

- OP: `ttnn/cpp/ttnn/operations/experimental/ssm/hc_sum_reduce`

Findings:
- Program computes per-core reader/writer/compute args from current tensor shapes and updates them in override. Buffer base addresses are updated.
- No runtime-only scalars outside hash observed; default hash should be sufficient.

No issues identified.
