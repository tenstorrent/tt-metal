# Device K mean probe

`center_mean.build(device, src, mode="bf16_fpu")` returns `(bias, invoke)`.
The source is device BF16 `[1,H,N,128]`, TILE/DRAM. Every invocation recomputes
the column mean over N on device and fills a stable BF16 `[1,H,32,128]` bias.
No host-computed bias is uploaded. The probe then passes that tensor directly
to the separately qualified `center_preprocess.build`.

Two explicit precision/cost choices:

- `bf16_fpu`: original BF16 source, HiFi4 FPU/GMPOOL reduction, FP32 DST kept
  across the entire H tile loop, final BF16 output. **The scaler `1/N` is
  truncated to BF16, not rounded to nearest.** Power-of-two N has an exact
  reciprocal; other lengths may have systematic scaling error. Setting
  `fast_and_approximate_mode=False` does not select SFPU for BF16 inputs.
- `fp32_sfpu`: device BF16-to-FP32 widening, full-FP32 SFPU reduction and FP32
  reciprocal post-multiply, then explicit FP32-to-BF16 output rounding. Full
  source conversion, its extra DRAM traffic, narrowing and bias materialization
  are all included in every invocation and timing. The source remains the
  original BF16 tensor; widening cannot recover any pre-BF16 input information.

The compact mean has logical H=1 and padded H=32. Its padded rows must not be
reinterpreted as valid repeated means. A broadcast add to a device-zero tensor
materializes all 32 rows into the stable bias. This avoids the current repeat
composite's H=1 TILE-to-row-major-to-TILE path. Adding zero to an already BF16
mean incurs no further ordinary-value precision loss.

The tiled H reduction distributes work over H*4 output tile columns, not the
long N dimension: H=10 can use only 40 column tasks. This existing implementation
is a practical starting point, not necessarily the fastest device mean. A
future chunked partial reduction can expose more cores, at the cost of another
reduction and FP32 intermediate storage.

The CLI reports device bias error against a verification-only FP64 mean of the
BF16 source, ideal final BF16 rounding error, BF16 reciprocal truncation error,
and exact centered-quantization agreement using the **actual device bias**.
It measures mean+bias production alone and mean+bias+center+quantization.
Mean buffers allocated by public TTNN operations are retained by `invoke` for
trace safety. Keep the builder alive until traces are released; discard it
afterward. Full-size widening and narrow-mean buffers are preallocated/reused.
For a long-lived service, replace retained public-operation outputs with a
fixed graph/primitive preallocation plan; this is a bounded research probe.

```sh
python experiments/sdpa-l2/bfp4-lofi-v2/center_mean.py --label bf16-normal-v1 --mean-mode bf16_fpu
python experiments/sdpa-l2/bfp4-lofi-v2/center_mean.py --label fp32-normal-v1 --mean-mode fp32_sfpu
python experiments/sdpa-l2/bfp4-lofi-v2/center_mean.py --label bf16-common-v1 --distribution common_k
python experiments/sdpa-l2/bfp4-lofi-v2/center_mean.py --label bf16-25920-v1 --length 25920 --heads 5
python experiments/sdpa-l2/bfp4-lofi-v2/center_mean.py --label fp32-perf-v1 --mean-mode fp32_sfpu --heads 10 --length 32768 --cores 110 --iters 10
```

Use `--quant-mode b4_rne` for the other center quantizer. `--trace-repeats 0`
selects synchronized eager timing, explicitly including host dispatch overhead.
Preparation did not run any JIT or device jobs; the card owner performs them.

## Repository evidence

- Python mean signature and fast/accurate flag:
  `ttnn/cpp/ttnn/operations/reduction/generic/generic_reductions_nanobind.hpp:192`.
- Mean lowers to AVG with scalar/reduced_volume:
  `.../generic/generic_reductions.cpp:303`.
- Full-FP32 SFPU eligibility and mean post-scaling:
  `.../generic/device/reduce_op.cpp:178` and `:205`.
- BF16-vs-FP32 scaler CB and H*4 work split:
  `.../generic/device/reduce_op_multi_core_h_program_factory.cpp:54` and `:110`.
- FP32 DST lifetime across the H loop:
  `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:619` through final pack.
- BF16 reciprocal truncation (`bits >> 16`):
  `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl:38`.
- Preallocated typecast binding:
  `ttnn/cpp/ttnn-nanobind/operations/copy.cpp:57`.
- H=1 repeat layout fallback:
  `ttnn/cpp/ttnn/operations/data_movement/repeat/repeat.cpp:455`.
