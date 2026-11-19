# Beta / decay V2: reproduce native mixed-input add boundary

`beta177_probe.json` is preserved evidence for V1: all12 beta cases are bit
exact; decay equal fractions are approximately0.77–0.94 with max absolute
error0.00131845 and PCC approximately0.99999998. No whole-decoder performance
or correctness claims were made from that failed exactness gate.

Source audit finds a concrete arithmetic difference before softplus:

- `binary/binary.hpp::resolve_fast_and_approximate_mode` documents ADD/SUB's
  unset flag as true, choosing the faster FPU implementation.
- `binary_ng_device_operation.cpp::is_binary_sfpu_op` selects the SFPU ADD under
  that flag only for matching FP32/integer input types. Mixed BF16 A / FP32
  dt_bias therefore uses FPU even with dtype=FP32 output.
- `binary_ng_program_factory.cpp` enables FP32 DST for these inputs, but leaves
  FPU inputs' unpack mode at Default. The native FPU add's source conversion
  and accumulation boundary differ from V1's precise SFPU addition.
- Parent notes checkpoint dt_bias is BF16 before upload, so truncating that
  input alone does **not** establish the numerical cause. The add boundary
  must be measured directly. V2 makes no claim that bias truncation explains
  all observed differences.

V2 uses BF16 CB0 for A and FP32 CB2 for bias, `compute_kernel_hw_startup(0,2,5)`,
`add_init(0,2)`, `add_tiles(0,2,0,0,0)` with default unpack for those FPU inputs.
After the add it restores FP32 direct-unpack SrcA for the retained native
softplus and FP32 multiplication path. Beta math is unchanged from V1.
This reproduces the native FPU operation rather than hand-coding an assumed
mantissa truncation. No native/shared source was changed.

New immutable artifacts:

- `candidates/gdn_beta_decay_fused_v2.py`
- `candidates/make_beta_decay_candidate_v2.py`
- `candidates/beta_decay_v2_on167.py`
- `candidates/beta_decay_v2_on167_source.json`
- `tests/run_beta_decay_probe_v2.py`

The helper has optional diagnostic_stage1 (emit add) or2 (emit softplus) in
its FP32 output; normal stage0 emits decay. Compile-time stage selection
removes diagnostic branches from the normal path. These controls are probe
only; the frozen decoder always uses stage0. New helper paths allocate no
extra persistent buffers. V1 and its provenance remain untouched.

The V2 probe checks all12 prior cases, native/fused beta and decay equality,
plus exact native/fused add and softplus intermediates. For add it separately
records CPU full-FP32 errors and first mismatch samples. New source provenance
includes binary dispatch code as well as unary arithmetic and helper/harness
hashes. Exactness remains mandatory; high PCC alone is insufficient.

```sh
python -m models.autoports.qwen_qwen3_6_27b.tests.run_beta_decay_probe_v2 \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/gdn_beta_decay_fused_v2.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/beta178_probe.json
```

Python syntax passes. Hardware is not executed by this source-only helper
agent; parent owns the serialized compile/probe and any necessary repair.
