# Beta/decay V3 row-only unary candidate

Source-only controlled follow-up to exact-passing beta180. Frozen
`beta_decay_v3_on185.py` changes only integrated185's `_gate178_COMPUTE`
string. No decoder method, reader, writer, CB, runtime address, tensor layout,
or precision policy changes. All prior artifacts remain untouched.

The input CBs contain meaningful data only in row0. V2 runs sigmoid and
softplus over all4faces ×8vector iterations/face. V3 uses native
VectorMode::R (Face0+Face1) with ITERATIONS=1 for both operations, reducing
nominal unary vectors32→2. Native FPU `add_init`/`add_tiles` and the final
FP32 `mul_binary_tile` remain exactly unchanged. Native sigmoid's BF16
numerical specialization, explicit nearest rounding, native FP32 softplus's
polynomial and init constants also remain unchanged.

Legal source support:

- `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_sfpu_common.h`
  implements VectorMode::R as the first two faces and explicitly skips the
  bottom two faces.
- `ckernel_sfpu_sigmoid.h::calculate_sigmoid` accepts independent ITERATIONS.
- `ckernel_sfpu_softplus.h::calculate_softplus` accepts independent ITERATIONS;
  its INP_FLOAT32 branch processes one vector per iteration, so the BF16
  two-vector even-iteration restriction does not apply.

The first vector may include extra padded lanes; no lane reductions occur.
Unvisited lanes retain their zero-initialized input/add values. Physical
padding can differ from V2/native sigmoid-transformed padding but must remain
finite and repeat-deterministic. Only logical [B,1,48] values are consumed by
the existing recurrent reader. Stage1 and2 diagnostic output controls remain
available to test native add and softplus boundaries independently.

Artifacts:

- `candidates/gdn_beta_decay_fused_v3.py`
- `candidates/make_beta_decay_candidate_v3.py`
- `candidates/beta_decay_v3_on185.py`
- `candidates/beta_decay_v3_on185_source.json`
- `tests/run_beta_decay_probe_v3.py`

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_beta_decay_probe_v3 \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_beta_decay_fused_v3.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/beta188_probe.json
```

The V3 probe checks all12 B1/3/31/32 cases with real weights, native exact beta,
decay, add and softplus, changed input/program-cache reuse, supplied output
address reuse, repeat equality, and finite/repeat-equal **padded** buffers.
It intentionally does not compare native unspecified physical padding.

Python syntax and AST scope checks pass. Hardware exactness and warmed traced
whole-decoder latency are pending parent execution. Source work predicts less
SFPU work, not a measured latency gain; reader, FPU add, multiply, pack, and
launch costs remain unchanged. No quality gate is waived for this candidate.
