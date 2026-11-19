# Remove known-zero setup and unused inverse reads

Norm179 passed all 54 exact cases and whole short/512/B1 gates. Its B32 short
latency was 1.605525 ms versus 1.613159 ms for integrated173; long latency was
1.621856 versus 1.629856 ms. This next candidate builds on that accepted norm.

The output reader clears a 1024-word zero tile in a scalar loop before issuing
any NoC reads. Each output tile then unpacks that known-zero tile and the native
column-normalization kernel broadcasts its zeros in every row band. This setup
is unnecessary because hidden RMSNorm's centering mean is identically +0.

The candidate removes only the zero buffer and its producer/consumer work.
`_hidden_normalize_zero_` initializes LREG7 to negated +0 and loads the four data
slots. It calls the existing native `_finish_col_normalize_row_band_` unchanged,
which performs the same inverse broadcast, data+(-0), FP32 inverse multiply and
stores. The zero-subtract operation is retained, including signed-zero behavior.
Gamma multiplication and explicit BF16 RNE remain unchanged.

The inverse is consumed only through column zero of its two left faces. The
reader now fetches those two 1024-byte faces at their original offsets rather
than all 4096 bytes. This halves traffic from the common L1 inverse page. Right
faces are intentionally not initialized: no normalization load addresses them,
and direct FP32 unpack does not perform arithmetic on their contents. Inverse
producer padding for inactive users remains unchanged.

Fold and inverse kernels, the exact reduction tree, L1 scratch, output core
partition, output writer and final BF16 boundary are unchanged. The performance
hypothesis is reduced serial reader setup, fewer unpack handshakes, and less
hot-source NoC traffic. No latency improvement is claimed before measurement.

Frozen artifacts:

- `candidates/hidden_rmsnorm_exact_tree_zero_mean.py`
- `candidates/hidden_norm_zero_mean_on179.py`
- `candidates/make_hidden_norm_zero_mean_candidate.py`
- `candidates/check_hidden_norm_zero_mean_cpu.py`
- `candidates/hidden_norm_zero_mean_cpu.json`

CPU/source checks passed: all 65,280 finite BF16 encodings retain identical
zero-subtract bits; native zero broadcast equals the specialized negative-zero
constant; all 256 inverse values physically loaded by SFPU are inside the fetched
faces; the output kernel reverts exactly to its parent after undoing the zero
specialization. Python compilation passed. No hardware or main/shared source
was touched. Device instruction scheduling, exact output and latency remain
unverified.

Parent-serialized probe:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_norm_exact_l1_comparison \
  --candidate models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/hidden_rmsnorm_exact_tree_zero_mean.py \
  --reference models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/candidates/hidden_rmsnorm_exact_tree_l1_80.py \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/hidden_norm_zero_probe.json
```

Require all 54 default exact cases, then whole B32
short/512 and B1 using `hidden_norm_zero_mean_on179.py`. Combine with independent
recurrence changes only after its own gate passes.
