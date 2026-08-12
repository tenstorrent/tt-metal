# Full-model decode lower-bound accounting

The retained optimized TP4 decoder measurement is 0.414822 ms/layer over 100
warmed replays. Forty layers therefore cost 16.592880 ms. The final one-layer
full-terminal control is 1.747149 ms, so final norm, BF16 sharded LM head,
Sampling1D, device token feedback, and trace orchestration add 1.332327 ms over
one layer. The composed wall target is therefore 17.925207 ms/token.

The final 128-token host-free token-out window measures 18.364851 ms/token. The
unexplained remainder is 0.439644 ms, or 2.394% of measured token-out time. This
is below the 10–15% gap trigger, so no additional speculative graph rewrite is
warranted.

The corresponding retained profiler device accounting is 348.538 us/layer and
153.954 us/layer byte-only DRAM floor. The refreshed one-layer full terminal is
1,856.835 us/replay of merged device time. Subtracting one layer and composing
with 40 gives 15.449817 ms of device kernels. The reduced report's 42.6%
modeled DRAM roofline implies 791.012 us for that graph; composing its terminal
increment with the decoder floor gives a 6.795218 ms full-path byte floor.
Device-kernel, warmed wall, and roofline numbers are not substituted for one
another.

In the refreshed reduced trace, matmul is 68.526%, all-gather 10.378%, TopK
6.732%, Sampling 1.467%, ManualSeed 0.928%, and async all-reduce 2.037%. TopK,
Sampling, and ManualSeed together are 169.475 us/replay (9.127%), so the sampler
family does not dominate the complete token-out path.
