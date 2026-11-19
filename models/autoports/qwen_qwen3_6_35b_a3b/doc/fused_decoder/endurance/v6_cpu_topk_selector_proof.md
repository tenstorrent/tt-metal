# V6 CPU selector reproduction

The scalar reference reproduces installed Torch CPU topk8 indices exactly on all 5,032 tested finite rows, both when supplied BF16-logit keys and when supplied actual FP32-softmax probability keys. This is algorithm evidence, not decoder acceptance or device performance evidence.

The [executable proof](../../../tests/v6_cpu_topk_selector_proof.py) contains `select_top8` as the Python specification and `PORTABLE_CPP` as a self-contained scalar C++ implementation. `v6_select_top8(uint32_t* q, int depth=16)` consumes caller-owned scratch containing 256 entries `(BF16_bits << 16) | original_id`; the first eight entries become selected payloads in ATen order. There are no allocations, standard-library calls, recursion, large local arrays, or index tie-breaks. Signed zeros compare equal; NaNs compare ahead of numbers like ATen `largest=True`.

The source reproduces GCC13 median-to-first, unguarded partition, introselect with depth 16, exact heap-select fallback, and insertion sort of the first seven selected entries. The eighth is already the nth element. Source hashes for installed ATen, GCC13 algorithm and heap headers, the portable string, proof script, captures, and permutation fixtures are recorded in the [result](v6_cpu_topk_selector_proof.json).

Validation includes independent calls into GCC13 `std::__introselect` at forced depths 0, 1, 2 and 16, so the rarely reached heap fallback is exercised without claiming random rows reached it. The exact portable C++ string is separately compiled and checked against the Python reference on every finite fixture and against Torch on nonfinite direct-topk fixtures. Nonfinite softmax inputs are explicitly outside the softmax equivalence contract.

| Input group | Rows | Selector mismatch versus Torch on identical keys | Direct-logit versus FP32-softmax membership mismatch |
| --- | ---: | ---: | ---: |
| Saved actual HF/TT captures | 24 | 0 | 0 |
| Saved lower-value permutation fixtures | 3 | 0 | 0 |
| Gaussian standard deviation 1 | 1,000 | 0 | 0 |
| Gaussian standard deviation 5 | 1,000 | 0 | 0 |
| Quantized random ties | 1,000 | 0 | 0 |
| Gaussian standard deviation 100 | 1,000 | 0 | 390 |
| Gaussian standard deviation 1e-8 | 1,000 | 0 | 1,000 |
| Equal, signed-zero, ordered and extreme rows | 5 | 0 | 0 |

The logit-key substitution is **not universally equivalent** to CPU FP32 softmax followed by topk. Rounding can merge distinct probabilities, and underflow can collapse low scores to zero. Even ties below the selection boundary can change the partition path and tied membership. The wide-input rows additionally show 498 order mismatches. Therefore no blanket exact-router claim is justified by this proof. All saved model rows agree, supporting an integrated candidate experiment with unchanged quality gates.

Command, CPU only:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_cpu_topk_selector_proof
```

No TTNN import, hardware access, shared/native source change, or production decoder change occurs in this proof.
