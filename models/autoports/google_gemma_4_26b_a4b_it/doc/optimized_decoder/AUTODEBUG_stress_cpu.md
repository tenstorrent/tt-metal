# CPU precision controls for the S=1025, 512-step stress pair

The BF16 cache and RoPE boundaries are sufficient to reproduce nine shared HF
failures across the two layer kinds. They do **not** explain every stress
failure or the optimized-only differences. PCC 0.995 remains unchanged.

Evidence is in `stress_cpu_precision_layer0.json`,
`stress_cpu_precision_layer5.json`, and the compact
`stress_cpu_precision_summary.json`. The new helper is
`tests/hf_optimized_stress_precision_controls.py`; inherited helpers and runtime
code were not modified for these controls. Both runs completed successfully
with four PyTorch threads and confirmed `ttnn_imported=false`.

## Input and oracle validation

The saved `stress_pair_{fused,optimized}_layer{0,5}.pt` fixtures contain output
tensors only. They contain no input tensors to compare bitwise. The helper
reconstructs the device harness's seed 42, meta-layer construction, real pinned
checkpoint loading, and BF16-roundtripped random prefill/decode input stream.
It asserts equality if explicit input tensors are present in a future fixture.

Every reconstructed FP32-oracle versus saved-TT decode PCC matches the original
harness report within 2.69e-8 for sliding layer 0 and 1.72e-8 for full layer 5.
This is strong indirect validation of the input stream and oracle; it is not a
claim that absent input tensors were compared. Reports include input hashes,
fixture hashes, source hashes and the exact software version.

## Controlled changes and observed boundaries

The FP32 HF oracle and both CPU controls share real weights and exact generated
inputs. The cache-only control rounds newly stored K/V through BF16 and returns
FP32 tensors to otherwise FP32 HF computation. The second control additionally
rounds RoPE cos/sin tables through BF16. These match actual inherited precision
boundaries: `run_decoder.py:116` uploads BF16 by default, including the caches
and RoPE tables; `fused_decoder.py:650` casts prefill Q/K/V to BF16, and
`fused_decoder.py:621` casts decode cache updates to BF16. FP32 table casts in
the rotary implementation do not restore values already lost during upload.

| Layer | CPU control | Shared TT-vs-FP32 failures reproduced | Lowest TT-vs-control PCC at those positions |
| --- | --- | --- | ---: |
| 0, sliding | BF16 cache | 1310 | 0.998806 |
| 0, sliding | BF16 cache and RoPE | 1067, 1181, 1191, 1310, 1354, 1420 | 0.996178 |
| 5, full | BF16 cache | 1189, 1289, 1382 | 0.997832 |
| 5, full | BF16 cache and RoPE | 1189, 1289, 1382 | 0.997832 |

For each listed position, the CPU control itself fails against FP32, both TT
implementations fail against FP32, and both TT implementations pass against
that CPU control. In the CPU controls, attention PCC remains 0.9999959 or
better at the six sliding matches and 0.9999985 or better at the three full
matches. Every matching CPU control replaces one of the selected eight experts.

For example, sliding position 1310 has an FP32 rank-eight/rank-nine logit gap
of 0.0002872. BF16 cache plus RoPE changes the selected tail expert from 7 to
22. Attention PCC is 0.9999963, router-score PCC is 0.9999969, expert-output PCC
drops to 0.868039, and final-output PCC becomes 0.982815. Saved fused and
optimized outputs agree with this control at 0.999913 and 0.998821.

Full position 1189 shows the same mechanism using cache rounding alone: the
FP32 tail gap is 0.0004349, expert 87 is replaced by 110, attention PCC is
0.9999986, expert-output PCC is 0.920088, and final-output PCC is 0.990113.
Saved fused and optimized outputs agree with this control at 0.999856 and
0.998174. These are measured CPU routing changes; actual TT route identities
were not captured by these output-only fixtures.

## Remaining failures and limitations

After the combined cache/RoPE control, shared failures still unexplained are:

- Sliding: 1098, 1143, 1165, 1173, 1184, 1240, 1285, 1293, 1451.
- Full: 1360.

The optimized-only HF failures remain sliding 1108, 1175, 1519 and full 1398;
the fused-only failures remain sliding 1428 and full 1396. These isolated CPU
controls do not explain those implementation differences. They also create
some CPU-only failures where TT passes, so the quantized CPU control is not a
replacement acceptance oracle. In particular, it does not emulate device
reduction order, prefill query rounding, attention/projection kernels, or the
optimized common-normalized router path.

The evidence identifies inherited precision boundaries that can cross routing
decisions while attention PCC is very high. It does not justify accepting all
stress failures, relaxing the threshold, or changing runtime defaults.

Reproduction, one layer at a time:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python_env/bin/python \
  -m models.autoports.google_gemma_4_26b_a4b_it.tests.hf_optimized_stress_precision_controls \
  --layer 0
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python_env/bin/python \
  -m models.autoports.google_gemma_4_26b_a4b_it.tests.hf_optimized_stress_precision_controls \
  --layer 5
```

Both commands pass. Python compilation, Black, and `git diff --check` also pass.
