# NC.1 layer check: native C++ ttnn.grouped_matmul vs the Python prepared route (wo_a)

Written by `models/demos/deepseek_v41_flash_d_p/tests/woa/test_gmm_native_layer.py` (op level on the held mesh, no model load; fixed-seed bfloat8_b inputs of the real shapes, the model's knobs, dtypes, config and grid). Model-wide verification is deferred to the end of the prepared-op ports (owner).

| case | knobs | mismatch | repeat | cache growth (C++) | Python LRU growth | host us native | host us native_direct | host us generic | device us native | device us generic |
|---|---|---|---|---|---|---|---|---|---|---|
| interleaved256 | block_w 2, rt 1, depth 2 | 0.0 | 0.0 | 0 | 0 | 29.6 | 29.1 | 118.4 | 150.0 | 150.19 |
| bank256 | block_w 2, rt 1, depth 4, w_inflight 1, x_ahead 1, one_group True | 0.0 | 0.0 | 0 | 0 | 27.5 | 26.3 | 102.5 | 129.28 | 129.29 |
| interleaved160 | block_w 4, rt 1, depth 4 | 0.0 | 0.0 | 0 | 0 | 22.6 | 22.3 | 89.2 | 130.85 | 130.3 |
| bank160 | block_w 4, rt 1, depth 4, w_inflight 2, x_ahead 1, one_group True | 0.0 | 0.0 | 0 | 0 | 27.3 | 27.4 | 69.1 | 112.21 | 113.2 |

Output dtype DataType.BFLOAT8_B, grid 11-10. Host us: a warm call with the output buffer passed (as attention._out_wo), perf_counter over 50 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.grouped_matmul (the model's entry, the C++ op under the ttnn.bringup registration), native_direct = ttnn.grouped_matmul. Device us: real-time profiler, slowest chip, 20 calls a section.
