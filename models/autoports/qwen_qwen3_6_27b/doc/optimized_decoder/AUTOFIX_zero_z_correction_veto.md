# Numerical veto: remove only packed GDN Z correction

Source-only candidate `candidates/zero_z_correction_on134.py`, based on frozen
integrated134. **No performance claim**: primary and correction matrices keep
their full shapes, sharding, programs, matmuls and adds. No device or TTNN
execution was performed while preparing this candidate.

Parent evidence for phase139 (removing all input correction during prefill)
reported minimum 512-step PCC 0.994980452, below the unchanged 0.995 bar. That
candidate was rejected. This experiment removes a narrower contribution in
both prefill and decode, testing whether the recurrent Q/K/V/B/A correction can
be retained while omitting the Z gate correction.

## Exact setup change and quantization order

The existing packed input matrix is `[5120,16512]`:

| Columns | Meaning |
| --- | --- |
| 0:10240 | Q, K, V convolution inputs |
| 10240:16384 | Z output gate |
| 16384:16448 | 48 B values plus 16 padding columns |
| 16448:16512 | 48 A values plus 16 padding columns |

The primary is uploaded to BFP8 using the original setup path. Its stored
values are read back, and the FP32 residual is formed as `original - quantized`
exactly as before. **After that subtraction**, the candidate sets only
`residual[:,10240:16384] = 0`, before the existing correction BFP8 upload.
The primary tensor is never modified or repacked by the candidate. Existing
padding to physical N=16640 and C40/R5 geometry remain unchanged.

Both Z boundaries are multiples of 32, so the removed region consists of whole
N tiles. BFP8 shared-exponent groups remain within those tiles; zeroing Z cannot
change quantization groups for QKV or B/A. The host packing implementation is
`tt_metal/impl/data_format/bfloat8.cpp:22-24`, delegating to tile/face-local
`pack_as_bfp_tiles` in `blockfloat_common.cpp:347` onward. The real-weight probe
below verifies stored elements outside Z rather than relying only on this
source argument.

Given identical external hidden inputs, Z is used after recurrence/readout by
the current output gate. It does not enter convolution carry, Q/K/V, B/A or the
recurrent-state update. Therefore the expected local cache state is unchanged;
layer outputs may change. This does not establish full-model equivalence or
permit relaxing the decoder output PCC gate. Subsequent layers or autoregressive
feedback would observe any changed output.

## Artifacts and validation

- Frozen candidate: `candidates/zero_z_correction_on134.py`.
- Generator: `candidates/make_zero_z_correction_candidate.py`; refuses overwrite.
- Matched control: candidate policy `zero_z_correction=False`.
- Real-weight probe: `tests/run_zero_z_weights_probe.py`.
- Source check metadata: `candidates/zero_z_correction_source.json`.

Python syntax checks passed. The generator changes only the weight-loading
method; runtime methods, primary upload source, projection shapes and programs
are unchanged.

The real-weight probe opens **no device**. It loads actual checkpoint QKV/Z/B/A
weights, reproduces TTNN host BFP8 packing, forms full and zero-Z corrections,
and requires identical stored correction elements in every QKV and B/A column,
zero stored Z elements, and unchanged stored primary values. It was prepared
but not run by this source-only helper:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_zero_z_weights_probe \
  --layer 0 \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/zero_z_weights_probe.json
```

Parent should run the normal real-weight short test and B32 512-step sequence on
the frozen decoder, including prefill PCC and cache comparison, before any
narrow-matrix implementation. If output PCC fails, record the numerical veto
and keep the full correction. A passing short test alone is insufficient.

## Conditional performance follow-up

Only after this numerical candidate passes, a separate layout candidate could
pack correction QKV followed by B/A into logical N=10368, physical N=11520 at
C40/R5. Primary stays full. This would reduce correction weight traffic by
`(16640-11520)/32 * (5120/32) * 1088 = 27,852,800` bytes, about 54.4 us at an
idealized 512 GB/s. That is a bandwidth bound, not a predicted latency gain.
The new split/add/output assembly must preserve the existing FP32-primary plus
FP32-correction to BF16 boundary for corrected regions, preserve primary-only Z
rounding, and prove whole-path latency. None of that narrower runtime is
implemented in this first numerical experiment.
