# NC.1 layer check: native C++ ttnn.grouped_matmul vs the Python prepared route (wo_a)

Written by `models/demos/deepseek_v41_flash_d_p/tests/woa/test_gmm_native_layer.py` (op level on the held mesh, no model load; fixed-seed bfloat8_b inputs of the real shapes, the model's knobs, dtypes, config and grid). Model-wide verification is deferred to the end of the prepared-op ports (owner).

| case | knobs | mismatch | repeat | cache growth (C++) | Python LRU growth | host us native | host us native_direct | host us generic | device us native | device us generic |
|---|---|---|---|---|---|---|---|---|---|---|
| interleaved256 | block_w 2, rt 1, depth 2 | 0.0 | 0.0 | 0 | 0 | 29.4 | 28.9 | 114.3 | 150.27 | 149.91 |
| bank256 | block_w 2, rt 1, depth 4, w_inflight 1, x_ahead 1, one_group True | 0.0 | 0.0 | 0 | 0 | 27.2 | 25.5 | 99.3 | 129.48 | 129.34 |
| interleaved160 | block_w 4, rt 1, depth 4 | 0.0 | 0.0 | 0 | 0 | 21.8 | 21.4 | 91.9 | 130.63 | 130.54 |
| bank160 | block_w 4, rt 1, depth 4, w_inflight 2, x_ahead 1, one_group True | 0.0 | 0.0 | 0 | 0 | 26.5 | 25.3 | 65.9 | 112.3 | 112.32 |

Output dtype DataType.BFLOAT8_B, grid 11-10. Host us: a warm call with the output buffer passed (as attention._out_wo), perf_counter over 50 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.grouped_matmul (the model's entry, the C++ op under the ttnn.bringup registration), native_direct = ttnn.grouped_matmul. Device us: real-time profiler, slowest chip, 20 calls a section.
