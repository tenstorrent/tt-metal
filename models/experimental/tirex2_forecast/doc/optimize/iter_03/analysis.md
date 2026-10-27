# Round 3 analysis (HEAD c6764be1895; no fresh profile: disk 27 GB < 30 GB watermark)

Estimates from the round-1 profile scaled by round-2 measured deltas (analyst, read-only). Paper/acc 914.3 ms.

Paper:
1. sLSTM op loop (558 steps x ~0.46 ms, ~22 ops/step): ~255 ms (layers.py slstm_recurrence). Kernel with an init state at ~40 us/step: ~-230 ms.
2. mLSTM chunkwise fp32 binaries (~29/layer): ~107 ms; fused input/post activations: -25..-40 ms.
3. mLSTM TM ops (cumsum, transposes, slices, C/n/m concats): ~70 ms; triangular-matmul cumsum, transpose_b, preallocated state: -25..-35 ms.
4. sLSTM step slices: ~50 ms (inside 1; gone with the kernel).
5. slstm pre/post transposes: ~57 ms; time-major stream on paper: -30..-45 ms.
6. sLSTM recurrent matmul (3/4 FLOPs wasted, block-diagonal R): ~55 ms (moot with the kernel).
7. slstm_post remainder (ys concat, layer_norm, binaries): ~135 ms (rough); preallocated ys: -20..-30 ms.
8. mLSTM gate linears (4 fp32 per-head linears + concat + tanh + x15): ~15-20 ms; one stacked linear: -12..-15 ms.
9. mLSTM q/k slices: ~8 ms. 10. FFN w_up 50 / w_in 29 / w_down 22 ms (efficient already).

Multivariate (2407 ms): variate block ~450-550 ms (subtract/keep/add -> one op: -40..-60 ms; mask add into softmax -15 ms);
mLSTM chunkwise x6 ~600 ms (items 2, 3, 8: -120..-150 ms); matmuls ~500 ms; remaining permutes ~100 ms.
