# Round 2 proposals (G2-r2, automatic: no floor, precision-policy or C++ change)

Base: c97b6f32cae, paper / accuracy 272.7 ms. Savings at paper (mid layer x5 plus the last layer).

| # | item | rows now | expected | numerics |
|---|---|---|---|---|
| 1 | fused SwiGLU, `ttnn.experimental.minimal_matmul(fuse_swiglu=True)`, interleaved [w1, w2] weights and biases (swiglu_interleave pattern); sweep its blocks | w1 5.48 + w2 4.71 + mul 3.58 | -18 to -23 ms | changing (the product is not rounded to bf16): full gate |
| 2 | z + y folded into layer_norm (`residual_input_tensor`) | add 1.80 | -9 ms | near-neutral: full gate |
| 3 | g*y + D*u in one op (`ttnn.addcmul`, D row-broadcast), if supported | 1.80 | -9 ms | near-neutral: full gate |
| 4 | B / C proj tiling: second sweep (out_block_w < N, other subblocks with fp32 dest) | 8.22 + 6.46 | -10 to -15 ms | the accumulation order only |
| 5 | time-segmented route-K scan (G2-r1 item 3, carried): long_context profile first, then the B0 bounds of test_s5.py | long_context 734 ms | about -70 ms at long_context, <= -3 at paper | changing (reassociation): test_s5 + full gate |

Each candidate is committed before its jobs; gate per candidate: bench paper down above noise, suite passes,
check_gates 0 failures against `iter_00/fixed_record.json`, score_broken 0 passes on its device dumps.
**Deferred:** bf16 x into C, HiFi2 / bf16 B weights, LoFi gate (precision policy: exceptions, X gate).
