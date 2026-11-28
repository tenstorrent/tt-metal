# NC.2 layer check: native C++ ttnn.fabric_reduce_scatter vs the Python prepared route

Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_reduce_scatter.py` (op level on the held mesh, no model load; fixed-seed partials of the model's shapes and the callers' options). Model-wide verification is deferred to the end of the port series (owner, NC.8).

| case | partial a chip | options | mismatch | repeat | cache growth (C++) | host us native | host us native_direct | host us generic | device us native | device us generic |
|---|---|---|---|---|---|---|---|---|---|---|
| attn_out1024 | [1024, 5120] bf16 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 46.5 | 45.5 | 74.4 | 181.91 | 182.03 |
| attn_out640 | [640, 5120] bf16 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 46.0 | 45.1 | 72.7 | 117.41 | 117.74 |
| q_a1024 | [1024, 1792] bf16 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 46.7 | 45.7 | 72.8 | 70.01 | 73.98 |
| comp1024 | [1024, 1024] bf16 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 46.4 | 45.9 | 72.6 | 48.35 | 66.33 |
| idx1024 | [1024, 32] bf16 dim 2 | cluster_axis 1, num_links 2, dim 2, cache_descriptors True, program_hash True, prepared True | 0.0 | 0.0 | 0 | 47.5 | 46.5 | 72.5 | 40.52 | 40.53 |
| exact1024 | [1024, 5120] fp32 dim 3 | cluster_axis 1, num_links 2, dim 3, cache_descriptors True, program_hash True, prepared True, exact_fp32_add True | 0.0 | 0.0 | 0 | 73.5 | 72.0 | 116.9 | 477.91 | 479.26 |
| moe1024 | [1024, 5120] bf16 dim 2 | cluster_axis 1, topology Topology.Ring, num_links 2 | 0.0 | 0.0 | 0 | 45.8 | 45.2 | 7413.2 | 145.92 | 164.44 |

Host us: a warm call with the output buffer passed, perf_counter over 50 calls, the median of 5 rounds alternating the routes. native = ttnn.bringup.fabric_reduce_scatter (the model's entry), native_direct = ttnn.fabric_reduce_scatter (the C++ op), generic = the pre-port Python route (cached descriptors, prepared programs). Device us: real-time profiler, slowest chip, 20 calls a section.
