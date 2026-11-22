# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 14: block type moe_shared (layer 2) with moe_combine swapped in last.

Rendered from models/demos/common/bringup/testing/templates.py. Frozen: an implementation may not edit this file.
Runs the whole block of layer 2 (moe_shared) with these steps on the device and the rest on the CPU reference:
    attn_hc
    attn_hc_pre
    attn_norm
    q_a
    topk_shared
    attention
    attn_residual
    ffn_hc
    ffn_hc_pre
    ffn_norm
    router
    experts
    shared_expert
    moe_combine

Reviewed (S.moe_shared.14.test.1): every swap-13 check is kept at its limits (the reviews of swaps 13, 12, 11, 10,
09 and 08 follow); the moe_combine checks are the moe_combine block of test_swap_moe_full_14_moe_combine.py (layer
1) at its limits, which are also the limits of the layer-2 component test test_c_moe_shared_moe_combine.py. The
swapped step is mlp_out [S, H] = experts_out + shared_out (HF HYV4MoE), both addends now from the device (the harness
hands the module the block's own experts_out / shared_out); block out = h_mid + post_j x mlp_out (CPU ffn_residual).
At layer 2 the addends are balanced (||experts_out|| 1769, ||shared_out|| 1669, ||mlp_out|| 2646). Measured on the
CPU (golden s4096 chunk 1, 2048 rows; the golden's layer-2 experts_out / shared_out as the addends, moe_combine
replaced by mutations; study /tmp/hy4_ms14/study.py, log study.log, outside the repo). "out" = block out PCC / rel vs
golden (limit 0.01); "c" = vs the exact fp32 sum of the same addends: rel [per-row norm ratio] worst row; "add" = per
addend (e = experts, s = shared; delta = mlp_out - other addend vs this addend, float64) coef / rel / worst row;
"g" = vs golden rel / coef:

    variant                 out PCC  out rel | c                          | add e coef/rel/row  s coef/rel/row  | g rel/coef
    fp32 a + b (reference)  0.999997 0.00228 | 0 [1, 1] 0                 | 1 / 0 / 0           1 / 0 / 0       | 0.0023 1.0
    bf16 output             0.999997 0.00240 | 0.0018 [0.9999, 1.0001] 0.0019 | 1 / 0.0026 / 0.019 1 / 0.0028 / 0.013 | 0.0028
    bf16 inputs             0.999997 0.00228 | 0 [1, 1] 0                 | 1 / 0 / 0           1 / 0 / 0       | 0.0023 1.0
    1.003 / 0.997 x shared  0.999997 0.00240 | 0.0019 [1.0001, 1.0029] 0.0029 | 1.0005/0.0028/0.033  1.0030      | 0.0029 (p)
    1.005 / 0.995 x shared  0.999997 0.00262 | 0.0032 [1.0001, 1.0048] 0.0049 | 1.0009/0.0047/0.054  1.0050      | 0.0039 (p)
    1.003 / 0.997 x experts 0.999997 0.00244 | 0.0020 [1.0001, 1.0029] 0.0030 | 1.0030  1.0006/0.0032/0.024      | 0.0030 (p)
    1.005 x experts         0.999997 0.00271 | 0.0033 [1.0002, 1.0049] 0.0049 | 1.0050  1.0010/0.0053/0.040      | 0.0040 (p)
    1.005 x both            0.999996 0.00311 | 0.0050 [1.0050, 1.0050] 0.0050 | 1.0059/0.0075  1.0060/0.0079    | 0.0055 (p)
    1.01 x experts          0.999994 0.00371 | 0.0067 [1.0003, 1.0098] 0.0099 | 1.0100  1.0020/0.0106           | 0.0071 (p)
    shared rows 1023 / 1024 0.999987 0.00511 | 0.0322 [0.673, 1.224] 1.05 | 1.0/0.048/9.3  0.9987/0.051/1.5   | 0.0323 (p)
    rows 1023 / 1024 swapped 0.999983 0.00581 | 0.0376 [0.912, 1.096] 1.22 | -              -                 | 0.0377 (p)
    golden shared (not own) the identity here; on the device addends the (experts, -shared) probe sees it     (p)
    cached golden mlp_out   0.999997 0.00229 | 0.0023 [0.9998, 1.0002] 0.0024 | 1/0.0034/0.026  1/0.0036/0.019   | 0      (p)
    last row zeroed, last 32 columns / rows zeroed: out rel >= 0.0115 (fail); a + 0.5 b: out rel 0.130 (fail)
    shared column / SP halves swapped, shared dropped, experts dropped, 2 (a + b), zero stub: out PCC <= 0.978 (fail)

"(p)" = passes the 0.98 out gate and the block out rel <= 0.01. So the test also asserts, on the swapped step (the
layer-1 checks and limits, unchanged):
  - vs the exact fp32 sum of the block's own device experts_out + shared_out: rel L2 <= 0.003, per-row norm ratio in
    [0.997, 1.003], worst row <= 0.005;
  - per addend on those inputs: |coef - 1| <= 0.002, add rel <= 0.008 (shared) / 0.005 (experts), add worst row
    <= 0.03 (bf16 output 0.019 / 0.013, the tightest margin; 0.997 x shared moves the experts row to 0.033);
  - the module again on (experts_out, -shared_out) and on (experts_out, 0) vs the exact sums: rel <= 0.004, worst
    row <= 0.005 (a module that ignores an input or returns a cached sum fails);
  - vs golden (backstop): rel <= 0.02, global coef within 0.004 of 1, and on the rows routed as in the golden per-row
    ratio in [0.98, 1.02], worst row <= 0.03.
Every "(p)" row fails at least one of these (0.997 x shared by the shared coefficient and the experts worst row,
1.003 x experts by the experts coefficient, the cached / golden-addend cases by the probes); bf16 rounding of the
inputs or the output passes them. A golden experts_out in place of the module's input is the identity in this study
(the addends are the golden's); on the device addends (experts vs golden rel 0.021) both probes see it. Known gap:
a scale error on one addend below 0.2 % (the balanced addends make the coefficient check tighter than at layer 1).
Reference / stub: BRINGUP_IMPL=reference passes (out 0.999995; mlp_out vs CPU 0, coefs 1, probes 0; vs golden
0.00727 coef 0.99995, 2003 matched rows [0.99932, 1.00068] row 0.0029; out rel 0.00304); the zero stub fails (every
check).
Device run (this gate, tt/mlp.py:TtMoeCombine, fp32 ttnn.add): pcc_swap_out 0.999974; mlp_out vs the CPU sum of the
device addends rel 0 / row 0, addend coefs 1.000000, probes 0; vs golden 0.01418 coef 0.99941, on 1939 matched rows
[0.99408, 1.00491] row 0.0126; out rel 0.00717 (swap 13: 0.00717). Every swap-13 check as in swap 13.

Swap 13's review (S.moe_shared.13.test.1): every swap-12 check is kept at its limits (the reviews of swaps 12, 11, 10, 09 and
08 follow); the shared_expert checks are the shared_expert block of test_swap_moe_full_13_shared_expert.py (layer 1),
with the limits of the layer-2 component test test_c_moe_shared_shared_expert.py (the same as layer 1, except the
clamp probe scale: 3, not 2). The swapped step is shared_out [S, H] = down(silu(gate(x)) x up(x)) (HF HYV4MLP as
mlp.shared_experts, intermediate 2048, unclamped) on x = ffn_norm, now from the device; mlp_out = experts_out +
shared_out (CPU moe_combine), block out = h_mid + post_j x mlp_out. Measured on the CPU (golden s4096 chunk 1, 2048
rows; the golden's layer-2 ffn_norm / h_mid / ffn_hc / experts_out, shared_expert replaced by mutations of the fp32
step; study /tmp/hy4_ss13/study.py, log study.log, outside the repo). "out" = block out PCC / rel vs golden; "c" = vs
the CPU shared_expert on the same ffn_norm: rel [per-row norm ratio] worst row; "g" = vs golden; "x3" = on ffn_norm
x 3 (bf16) vs the CPU step, rel / worst row:

    variant                  out PCC   out rel | c                              | g                              | x3
    fp32 reference           0.999997  0.00225 | 0 [1, 1] 0                     | 0.0020 [0.9996, 1.0006] 0.0025 | 0 / 0
    bf16 output              0.999997  0.00229 | 0.0017 [0.9999, 1.0001] 0.0017 | 0.0020 [0.9996, 1.0006] 0.0028 | 0.0017 / 0.0017
    bf16 gate / up / h       0.999997  0.00238 | 0.0030 [0.9959, 1.0029] 0.0058 | 0.0036 [0.9957, 1.0031] 0.0062 | 0.0029 / 0.0054
    bfp8 weights             0.999995  0.00303 | 0.0078 [0.9989, 1.0011] 0.0099 | 0.0080 [0.9988, 1.0012] 0.0102 | 0.0081 / 0.0110
    x 1.005                  0.999997  0.00259 | 0.0050 [1.0050, 1.0050] 0.0050 | 0.0054 [1.0046, 1.0056] 0.0060 | 0.0050 (p)
    x 1.01                   0.999995  0.00342 | 0.0100 [1.0100, 1.0100] 0.0100 | 0.0102 [1.0096, 1.0106] 0.0108 | 0.0100 (p)
    x 1.02 / x 1.03          >= 0.99998 <= 0.0081 | 0.020 / 0.030                | 0.020 / 0.031                  | 0.020 / 0.030 (p)
    clamped SwiGLU at 10     0.999997  0.00225 | 0 [1, 1] 0 (golden-blind)      | 0.0020 (as reference)          | 0.068 / 0.40 (p)
    HiFi2-like               0.999981  0.00742 | 0.027 [0.973, 0.977] 0.028     | 0.027 [0.973, 0.977] 0.028     | 0.027 / 0.029 (p)
    rows 1023 / 1024 swapped 0.999987  0.00508 | 0.051 [0.666, 1.502] 1.48      | 0.051                          | 0.051 / 1.41 (p)
    last row zeroed          0.999941  0.01083 | 0.036 [0, 1] 1.0                | 0.036 [0, 1.0006] 1.0          | 0.037 / 1.0 (p)
    last 32 rows zeroed      0.999322  0.03686 | 0.124 [0, 1] 1.0                | 0.124                          | 0.126 / 1.0 (p)
    gelu_tanh                0.999453  0.03565 | 0.138 [0.901, 1.108] 0.29      | 0.138                          | 0.066 / 0.19 (p)
    quarter of the intermediate dropped / gate-up swapped / one K half of down (no reduce) / no silu / doubled /
    input x 0.5: out PCC 0.9935 / 0.9890 / 0.9867 / 0.9866 / 0.9851 / 0.9842 (rel >= 0.118) (p)
    sigmoid, gate/up shards mixed, down K halves swapped, output column halves swapped, rows shifted, SP row halves
    swapped, shared dropped (zero stub): out PCC <= 0.973 (fail)

"(p)" = passes the 0.98 out gate (20 of 27 mutations; the shared expert carries more of mlp_out at layer 2 than at
layer 1, so the zero stub now fails the gate at 0.972). The golden cannot see the clamp (gate <= 5.3, up <= 5.7 on
the layer-2 golden; at x 2 the clamp scores only rel 0.0069, hence x 3). So the test also asserts, on the swapped
step:
  - vs the CPU shared_expert on the same device ffn_norm, at the component limits: rel L2 <= 0.008, per-row norm
    ratio in [0.99, 1.01], worst row <= 0.015, float64 global coefficient <got, want> / <want, want> within 0.003 of 1;
  - the module again on the device ffn_norm x 3 (bf16; gate / up pass +-10), vs the CPU step on the same input:
    rel <= 0.006, ratio in [0.99, 1.01], worst row <= 0.012, coefficient within 0.003 of 1;
  - vs golden (backstop): rel <= 0.01, ratio in [0.985, 1.015], worst row <= 0.02, coefficient within 0.004 of 1.
Every "(p)" row fails at least one of these (x 1.005 by the coefficient, 1.0050); bf16 rounding and bfp8 weights pass
them (bfp8 x3 worst row 0.0110 of 0.012, the tightest). Known gap (as the component test): a scale error below ~0.3 %.
Reference / stub: BRINGUP_IMPL=reference passes (out 0.999995; shared vs CPU 0 / x3 0; vs golden 0.00217
[0.99926, 1.00069] row 0.0036 coef 0.99999; out rel 0.00304); the zero stub fails (every check).
Device run (this gate, tt/mlp.py:TtDenseMLP on mlp.shared_experts of layer 2, HiFi4, bf16 weights): pcc_swap_out
0.999974; shared vs CPU 0.00073 [0.99916, 0.99963] row 0.0010 coef 0.99945 (the layer-1 module's steady -0.06 %
scale, 5x inside the limit); x 3 vs CPU 0.00074 [0.99918, 0.99963] row 0.0010 coef 0.99945; vs golden 0.00377
[0.99520, 1.00482] row 0.0115 coef 0.99937; tail 0.0035; out rel 0.00717 (swap 12: 0.0035 / 0.00718). Every swap-12
check as in swap 12.

Swap 12's review (S.moe_shared.12.test.1): every swap-11 check is kept at its limits (swap 11's review and the earlier ones
follow); the experts checks are the experts block of test_swap_moe_full_12_experts.py (layer 1), with the limits of
the layer-2 component test test_c_moe_shared_experts.py. The swapped step is experts_out [S, H] (HF HYV4Experts: sum
over the 8 routed pairs of router[t, e] x down_e(silu(min(g, 10)) x clamp(u, -10, 10))), from the device ffn_norm and
the device dense routing matrix; mlp_out = experts_out + shared_out (CPU), block out = h_mid + post_j x mlp_out.
The step and its bugs are those of layer 1, and the layer-2 mutation table (drop an expert / pair, scale, clamp, gelu,
limit, rows swapped or zeroed, capacity) is in test_c_moe_shared_experts.py (CPU study /tmp/hy4_exp2); it is not
re-measured here. The 0.98 out gate misses most of those bugs (layer 1: 19 of 26 mutations pass it), so the test also
asserts, on the swapped step:
  - vs the CPU experts on the same device ffn_norm and device routing: rel L2 <= 0.015, per-token norm ratio in
    [0.98, 1.02], worst token <= 0.018 (layer 1: 0.03; at layer 2 dropping the 1-token expert 180 scores 0.0197),
    float64 global coefficient within 0.004 of 1;
  - the module again on the device ffn_norm x 2 (exact in bf16; the gate reaches ~19.5, so the clamp fires, which it
    never does on the layer-2 golden) with the device routing, vs the CPU experts on the same input, same limits (the
    only check that sees the clamp bugs at layer 2: no clamp 0.205, clamp up only 0.097, limit 9 0.054 in the
    component study);
  - vs golden: rel L2 <= 0.03, global coefficient within 0.004 of 1, and on the rows whose top-8 set equals the
    golden's (a near-tie flip swaps a whole expert in the other rows) per-token ratio in [0.97, 1.03], worst row
    <= 0.04 (layer-1 limits; backstops, the vs-CPU checks carry the detection). No row routed as in the golden, an
    all-zero reference (nan coefficient) or a wrong shape fails cleanly instead of raising.
Known gap (as the component test): x 1.003, and one token's smallest pair when it moves its row by < 0.018.
Reference / stub: BRINGUP_IMPL=reference passes (out 0.999995; experts vs CPU 0 / x2 0; vs golden 0.0106, on 2003
matched rows [0.99887, 1.00103] row 0.0031, coef 0.99990; out rel 0.00304); the zero stub fails (every check).
Device run (this gate, tt/experts.py:TtHy4Experts, bfp8 weights, HiFi4): pcc_swap_out 0.999974; experts vs CPU
0.00822 [0.99399, 1.00533] row 0.01091 coef 1.00018; x 2 vs CPU 0.00814 [0.99512, 1.00520] row 0.01146 coef 1.00032;
vs golden 0.0209, on 1939 matched rows [0.98971, 1.00695] row 0.0164, coef 0.99931; tail 0.0035; out rel 0.00718
(swap 11: 0.0025 / 0.00674). Every swap-11 check as in swap 11.

Swap 11's review (S.moe_shared.11.test.1): every swap-10 check is kept at its limits (the reviews of swaps 10, 09 and 08
follow); the router checks are at the end of this docstring. Swap 10's review (S.moe_shared.10.test.1): every swap-09 check is kept at its limits; the ffn_norm checks are at the end of
this docstring. Swap 09's review (S.moe_shared.09.test.1): every swap-08 check is kept at its limits; the ffn_hc_pre
checks follow swap 08's. Swap 08's review (S.moe_shared.08.test.1) follows.
Built from test_swap_moe_shared_07_attn_residual.py (layer-2 setup: ctx.extra
["shared_topk"] = the golden's L{src}.topk, src = cfg.topk_source(2) = 1, in both contexts; every check and limit of
swap 07 unchanged) plus the ffn_hc checks of test_swap_moe_full_08_ffn_hc.py, re-tuned for layer 2 with the column
set of test_c_moe_shared_ffn_hc.py. The gated metric is pcc_swap_out (PCC, float [2048, 24576] golden = the 4 iHC
streams, spec block threshold 0.98). The swapped step is ffn_hc, the iHC gates [S, 8] fp32 (pre 0-3 | post 4-7) from
h_mid with the hc_mlp_layer weights of layer 2. Column means are pre 1.1e-3 / 1.4e-6 / 1.0 / 0.10, post 5.4e-4 /
9.6e-4 / 3.5e-4 / 0.049: pre gate 1 sits at hc_eps (the only near-eps column at layer 2; layer 1 had 0 / 1 / 6), pre 2
is saturated, and the two gate scales are equal (swapping them is a no-op). Measured on the CPU (golden s4096 chunk 1,
2048 rows; ffn_hc replaced by mutations of the fp32 step, every other step the fp32 CPU reference; study
/tmp/hy4_ssh8/study.py and study2.py, outside the repo, ~5 s per variant). "gates c" = vs the CPU ffn_hc on the same
h_mid: rel / worst of columns 0, 2-7 / column 1 / post worst row; "ffn_x c" = rel / row; "post" = out vs ffn_residual
with the CPU gates and the same mlp_out, worst stream / row; "tail" = out vs the whole CPU tail, rel:

    variant                   gates c                      | ffn_x c         | post          | tail   | out PCC
    fp32 reference            0 / 0 / 0 / 0                | 0 / 0           | 0 / 0         | 0      | 0.999995
    bf16 output               0.0003 / 0.0018 / 0.0018 / 0.0039 | 0.0006 / 0.0047 | 0.0007 / 0.0025 | 0.0012 | 0.999995 (ok)
    bf16 input and fn         0.0000 / 0.0006 / 0.0005 / 0.0010 | 0.0001 / 0.0011 | 0.0001 / 0.0005 | 0.0014 | (ok)
    scales swapped (equal)    0.0000 / 0.0006 / 0.0007 / 0.0004 | 0.0000 / 0.0001 | 0.0001 / 0.0002 | 0.0001 | (ok)
    post x 1.005              0.0003 / 0.0050 / 0 / 0.0050 | 0 / 0           | 0.0022 / 0.0041 | 0.0021 | 0.999994 (p)
    post x 1.01               0.0006 / 0.0100 / 0 / 0.0100 | 0 / 0           | 0.0043 / 0.0083 | 0.0043 | 0.999991 (p)
    post x 1.02 / 1.1         cols 0.020 / 0.10            | 0 / 0           | 0.0087 / 0.043  | <= 0.043 | >= 0.99962 (p)
    pre x 1.01 / 1.02         0.010 / 0.010 / 0.010 / 0; x 2 | 0.010; 0.020  | 0 / 0         | <= 0.0011 | 0.999995 (p)
    sigmoid abs err 1e-4      0.0004 / 0.22 / 42 / 0.048   | 0.0006 / 0.0016 | 0.013 / 0.14  | 0.0037 | 0.999990 (p)
    post = 1 x sigmoid        0.032 / 0.50 / 0 / 0.50      | 0 / 0           | 0.22 / 0.41   | 0.21   | 0.986256 (p)
    gate 0 zeroed             0.0030 / 1.0 / 0 / 0         | 0.0031 / 0.049  | 0 / 0         | 0.0041 | 0.999988 (p)
    gate 1 zeroed             0 / 0 / 1.0 / 0              | 0 / 0           | 0 / 0         | 0      | 0.999995 (p)
    gate 1 = gate 0           0.0030 / 0 / 1261 / 0        | 0.0025 / 0.041  | 0 / 0         | 0.0025 | 0.999993 (p)
    gate 4 / 5 / 6 zeroed     <= 0.0016 / 1.0 / 0 / >= 0.048 | 0 / 0         | >= 0.043 / 0.15 | <= 0.009 | >= 0.99996 (p)
    base 0 / 1 swapped        0.0004 / 0.14 / 0.13 / 0     | 0.0004 / 0.0065 | 0 / 0         | 0.0005 | 0.999995 (p)
    base 1 / 2 swapped        0 / 0 / 3.6 / 0              | 0 / 0.0003      | 0 / 0         | 1e-5   | 0.999995 (p)
    base 4/5, 5/6, 6/7        cols >= 0.41, post row >= 0.11 | 0 / 0         | >= 0.061 / 0.21 | >= 0.006 | >= 0.99487 (p)
    fn rows 4 / 5 swapped     0.0019 / 1.5 / 0 / 0.32      | 0 / 0           | 0.083 / 0.35  | 0.010  | 0.999941 (p)
    fn streams 0/1, 0/2       cols >= 0.19, post row >= 0.10 | >= 0.030      | >= 0.018      | >= 0.025 | >= 0.99851 (p)
    rows 1023/1024, 100/1500  cols <= 0.025, post row >= 0.34 | row >= 0.16  | row >= 0.085  | <= 0.007 | >= 0.99997 (p)
    last row = prev / zeroed  cols >= 0.024, post row >= 0.52 | row >= 0.42  | row >= 0.18   | ~0.01  | >= 0.99993 (p)
    rms_norm_eps 5e-6         0.0021 / 0.014 / 0.002 / 0.17 | 0.0018 / 0.040 | 0.0021 / 0.040 | 0.0026 | 0.999993 (p)
    rms_norm_eps 1e-6 / 2e-5  cols 0.027 / 0.029, post row 0.30 / 0.36 | row 0.077 / 0.066 | row >= 0.071 | 0.004 | (p)
    rms_norm_eps 1e-4         0.022 / 0.46 / 0.37 / 3.6    | 0.017 / 0.31    | 0.039 / 0.80  | 0.038  | 0.999341 (p)
    hc_eps dropped            0 / 0.0011 / 0.42 / 0.0002   | 0 / 0           | 0.0001 / 0.0007 | 1e-5 | 0.999995 (p)
    TP: one chip's sumsq      0.043 / 0.90 / 0.80 / 0.81   | 0.10 / 0.33     | 0.22 / 0.45   | 0.21   | 0.985121 (p)
    gate 3 = gate 2, gate 7 zeroed, base 2 / 3, fn rows 1 / 2 or 6 / 7, fn streams 2 / 3, RMS over one stream,
    chip-major columns, attn_hc weights, pre | post halves swapped, SP row halves swapped, rows shifted by 1, zero stub:
    out PCC <= 0.969 (fail)

"(p)" = passes the 0.98 out gate: 36 of 49 mutations do, 33 of them real bugs (post = 1 x sigmoid at 0.986 and one
chip's partial sumsq at 0.985 among them). A dropped hc_eps, gate 1 zeroed and base 1 / 2 swapped move ffn_x and out
by < 1e-4 and show only in column 1. So the test also asserts (informational metrics):
  - everything swap 07 (test_swap_moe_shared_07_attn_residual.py) asserts, at its limits: attn_hc, attn_x (+ vs CPU,
    rotated pre gates), attn_norm and q_resid (vs golden, vs the CPU step, eps runs), topk (exact per-row sets vs
    golden and vs the shared input, pad layout, chunk 0), attn_out five ways, h_mid (vs golden, per stream, stream
    ratio, vs the CPU attn_residual, addend, rotated post gates), router top-8 overlap >= 0.98, block out finite and
    rel L2 <= 0.01;
  - ffn_hc vs golden: rel L2 <= 0.01, per column <= 0.015 (column 1: <= 0.03), post worst row <= 0.02 (the
    component's 0.01 / 0.02 / 0.015 widened for the device h_mid); ffn_x vs golden rel <= 0.01, worst row <= 0.05;
  - ffn_hc vs the CPU ffn_hc on the same device h_mid: rel L2 <= 0.005, per column <= 0.01 (column 1: <= 0.02), post
    worst row <= 0.015; the pre gates through ffn_x (CPU ffn_hc_pre from the device gates vs from the CPU gates):
    rel <= 0.005, worst row <= 0.02;
  - the post gates through out: block out vs ffn_residual(h_mid, CPU gates, the block's own mlp_out), per stream
    rel <= 0.003 (the layer-2 component's limit; post x 1.01 scores 0.0043), worst (row, stream) <= 0.02;
  - block out vs the whole CPU tail (ffn_hc .. ffn_residual) from the device h_mid: rel <= 0.01 (worst row not gated:
    near-tie expert flips give 0.04 even for bf16 input rounding).
Every "(p)" bug above fails one of these except post x 1.005 (vs-CPU column 0.0050, post stream 0.0022), a 0.5 % post
scale that moves out by rel 0.002. No scaled-input eps probe: rms_norm_eps 5e-6 already fails the per-column check
(0.014) and the post worst row (0.17) at the block's own scale. The trail line pcc_swap_topk is positional match
(template default), diagnostic only; the topk checks above are order-free.
Device run (this gate): pcc_swap_out 0.999977; ffn_hc vs golden 0.00043, columns [0.0037, 0.0105, 0.0, 0.0029,
0.0068, 0.0055, 0.0074, 0.0028], post row 0.0088; vs CPU 0.00018, columns [0.0032, 0.0100, 0.0, 0.0007, 0.0066,
0.0059, 0.0074, 0.0024], post row 0.0067; ffn_x vs golden 0.0047 / 0.0100, vs CPU 0.00038 / 0.0013; post through out
stream <= 0.0010 / row 0.0026; tail 0.0016; h_mid 0.00146 / streams <= 0.0067; router 0.99341; out rel 0.00686.
Tightest margins: vs-CPU column 6 (0.0074 of 0.01) and column 1 (0.0100 of 0.02).

ffn_hc_pre (swap 09). The swapped step is ffn_x [S, H] = sum_j pre_j x h_mid stream j (pre = ffn_hc columns 0-3),
feeding ffn_norm -> router / experts / shared expert; block out = h_mid + post_j x mlp_out. At layer 2 the pre column
means are 1.1e-3 / 1.4e-6 / 1.0 / 0.10 (gate 2 saturated at 1.0, gate 1 at hc_eps), contribution norms 1.5e-3 /
1.4e-6 / 0.94 / 0.40, so stream 1 carries ~1e-6 of ffn_x and stream 0 shows only on the few rows where gate 0 reaches
0.06; ffn_norm then removes any row scale of ffn_x before the MoE. Layer-2 mutations of the step, measured on the CPU
on this golden (test_c_moe_shared_ffn_hc_pre.py, study /tmp/hy4_ssh_ffnhcpre2/study.py, outside the repo). "c" = vs
the CPU ffn_hc_pre on the same h_mid and gates, rel / worst row; "rot" = the step again with each row's pre gates
rotated by row mod 4, vs the CPU step:

    variant                      c rel / row       | rot rel / row
    bf16 output                  0.0017 / 0.0018   | 0.0016 / 0.0018   (ok)
    bf16 accumulation            0.0021 / 0.0028   | 0.0016 / 0.0028   (ok)
    pre x 1.005                  0.0050 / 0.0050   | 0.0050 / 0.0050   (fails c)
    stream 1 dropped             0 / 0             | 0.125 / 0.98      (fails rot)
    stream 0 dropped             0.0031 / 0.049    | 0.149 / 0.99      (fails both)
    streams 0 / 1 swapped        0.0014 / 0.024    | 0.092 / 0.73      (fails both)
    streams 0 / 2 swapped        -                 | 0.101 / 0.82      (component: golden rel 0.41)
    SP rows 1023 / 1024 swapped  0.014 / 1.22      |                   (fails c)
    pre + 3e-4                   0.0018 / 0.0048   |                   (not caught: below gate 2's bf16 rounding)

At the block, a pre-gate error on streams 0 / 1 or a row-scale error barely reaches out (the layer-1 swap-09 study:
dropped stream 0 / 1 and a 0 / 1 swap leave out at PCC 0.999997, pre x 1.02 moves the tail by 0.001), so the out gate
cannot see them. On top of swap 08's checks (ffn_x vs golden rel <= 0.01 / row <= 0.05 now sees the device step; out
vs the CPU tail, which starts from the CPU gates and the CPU ffn_x, now covers ffn_hc and ffn_hc_pre together), the
test asserts the component test's limits (test_c_moe_shared_ffn_hc_pre.py, the same as layer 1):
  - ffn_x (device) vs the CPU ffn_hc_pre on the same device h_mid and device gates: rel L2 <= 0.003, worst row
    <= 0.006 (bf16 accumulation 0.0021 / 0.0028 passes; pre x 1.005 0.0050 fails);
  - the ffn_hc_pre module once more on that h_mid with each row's pre gates rotated by row mod 4, vs the CPU step:
    rel L2 <= 0.004, worst row <= 0.01 (stream 1 / 0 dropped 0.125 / 0.149, streams 0 / 1 swapped 0.092).
Not caught: pre + 3e-4 on every gate; the gates are the step's input, so the module cannot make that error by itself.
Device run (this gate, tt/ihc.py:TtHcPre, fp32): pcc_swap_out 0.999977; ffn_hc_pre vs CPU rel 0 / row 0, rotated
rel 0 / row 0; ffn_x vs golden 0.0047 / 0.0100; tail 0.0016; router 0.99341; out rel 0.00686 (all as swap 08).

ffn_norm (swap 10). The swapped step is ffn_norm [S, H] = w x ffn_x x rsqrt(mean(ffn_x^2) + 1e-5)
(post_attention_layernorm, w in [0.070, 0.175]). It feeds the router, the routed experts and the shared expert, and
block out = h_mid + post_j x moe(ffn_norm). Layer-2 ffn_x has row rms in [0.0025, 0.062], and 15% of rows have
mean(x^2) < eps, so eps and the RMS reduction both show on the golden. The checks and limits are those of
test_swap_moe_full_10_ffn_norm.py, which are the component test's (test_c_moe_shared_ffn_norm.py, the same as
layer 1). They were re-measured at layer 2 on the CPU (golden s4096 chunk 1, 2048 rows; ffn_norm replaced by
mutations, every other step the fp32 CPU reference; shared_topk from the golden; study /tmp/hy4_ssh10/study.py,
outside the repo). "g" = vs golden rel [row norm ratio] worst row; "c" = vs the CPU ffn_norm on the same ffn_x
(rel / worst row); "x0.1" / "x30" = the step on ffn_x x 0.1 / x 30 (bf16) vs the CPU step (rel / worst row); "rt" =
router top-8 overlap vs golden; "out g" = block out vs golden rel (the CPU tail check is within 8e-4 of it):

    variant                  ffn_norm g                     | c               | x0.1            | x30              | rt     | out PCC   out g
    fp32 reference           0.0028 [0.9998, 1.0002] 0.0034 | 0 / 0           | 0 / 0           | 0 / 0            | 0.9973 | 0.999995  0.0030
    bf16 output              0.0031 [0.9998, 1.0002] 0.0037 | 0.0017 / 0.0017 | 0.0017 / 0.0017 | 0.0017 / 0.0017  | 0.9967 | 0.999993  0.0037 (ok)
    bf16 everywhere          0.0043 [0.9959, 1.0036] 0.0056 | 0.0033 / 0.0049 | 0.0028 / 0.0046 | 0.0029 / 0.0046  | 0.9955 | 0.999983  0.0058 (ok)
    x 1.01                   0.0104 [1.0098, 1.0102] 0.0107 | 0.0100 / 0.0100 | 0.0100 / 0.0100 | 0.0100 / 0.0100  | 0.9973 | 0.999970  0.0109 (p)
    x 1.05 / x 1.2           0.050 / 0.20                   | 0.050 / 0.20    | ...             | ...              | 0.9908 | 0.99936 / 0.99009  0.054 / 0.24 (p)
    eps 1.2e-5               0.0230 [0.9441, 0.9997] 0.0560 | 0.0228 / 0.0558 | 0.058 / 0.087   | 0.0001 / 0.0002  | 0.9968 | 0.999983  0.0062 (p)
    eps 2e-5 / 1e-6 / 0      0.093 / 0.16 / 0.19            | same            | 0.21 / 0.75 / 1.76 | <= 0.0003     | 0.991  | >= 0.99906  0.022 .. 0.046 (p)
    RMS over half the cols   0.0082 [0.9796, 1.0264] 0.0266 | 0.0077 / 0.0264 | 0.0038 / 0.0130 | 0.0092 [0.9722, 1.0272] 0.0278 | 0.9973 | 0.999953  0.0099 (p)
    RMS over a quarter       0.0131 [0.9540, 1.0576] 0.058  | 0.0128 / 0.058  | 0.0067 / 0.027  | 0.0156 / 0.060   | 0.9971 | 0.999882  0.0156 (p)
    LayerNorm instead of RMS 0.0126 [0.9997, 1.0001] 0.0385 | 0.0123 / 0.0384 | 0.0139 / 0.0384 | 0.0117 / 0.0384  | 0.9902 | 0.999974  0.0072 (p)
    input_layernorm's w      0.277 [1.248, 1.277] 0.29      | 0.277 / 0.29    | 0.277 / 0.29    | 0.278 / 0.29     | 0.9138 | 0.984507  0.31 (p)
    w halves / quarters (TP) 0.100 / 0.099 [0.982, 1.006] 0.11 | 0.10 / 0.11  | 0.10 / 0.11     | 0.10 / 0.11      | 0.90   | 0.9993    0.041 (p)
    rows 1023 / 1024 swapped 0.034 [0.965, 1.036] 1.08      | 0.034 / 1.08    | 0.018 / 1.20    | 0.033 / 1.06     | 0.9963 | 0.999981  0.0062 (p)
    last row zeroed          0.0250 [0.0, 1.0002] 1.0       | 0.025 / 1.0     | 0.032 / 1.0     | 0.022 / 1.0      | 0.9968 | 0.999932  0.0117 (p)
    sum instead of mean, 1 + w, no weight, output column halves swapped, SP row halves swapped, rows shifted by 1, no
    norm, w x x / sqrt(eps), zero stub: out PCC <= 0.935 (fail)

"(p)" = passes the 0.98 out gate: 15 of 24 mutations do, all real bugs. Swap 09's out checks (out vs golden rel
<= 0.01, out vs the CPU tail rel <= 0.01) catch 11 of them; x 1.01 is caught only just (out g 0.0109). Four are not
caught: eps 1.2e-5 (out 0.0062), RMS over half the columns (0.0099, the missing all-gather of the column-split
ffn_x), LayerNorm (0.0072) and SP rows 1023 / 1024 swapped (0.0062). So the test also asserts (informational metrics):
  - ffn_norm vs golden: rel L2 <= 0.01, per-row norm ratio in [0.99, 1.01], worst row <= 0.03 (upstream device error
    included: ffn_x is 0.0047 / row 0.0100 off the golden);
  - ffn_norm vs the CPU ffn_norm on the same device ffn_x: rel L2 <= 0.008, ratio in [0.993, 1.007], worst row
    <= 0.015 (RMS over half the columns: ratio 0.9796 and row 0.026; LayerNorm row 0.038; eps 1.2e-5 row 0.056);
  - the module once more on that ffn_x x 0.1 (bf16, eps dominates most rows) vs the CPU step: rel L2 <= 0.01, worst
    row <= 0.02 (eps 1.2e-5 0.058 / 0.087; ratio printed, not gated, as in the component test);
  - and on that ffn_x x 30 (bf16, mean(x^2) >> eps, the RMS reduction dominates) vs the CPU step: rel L2 <= 0.006,
    ratio in [0.993, 1.007], worst row <= 0.015 (RMS over half the columns 0.9722 / 0.028, LayerNorm 0.038).
Every "(p)" row above fails at least one of these. bf16 everywhere (the pessimistic device estimate) passes them with
about 2x margin (worst: vs CPU row 0.0049 of 0.015, ratio 0.9960 of 0.993). Not caught: a row scale of about 0.5%
(inside the bf16 ratio band). A 1% scale fails.
Device run (this gate, tt/norm.py:TtGatheredRmsNorm): pcc_swap_out 0.999978. ffn_norm vs golden 0.00518
[0.99703, 1.00073] row 0.0097; vs CPU 0.00174 [0.99914, 1.00065] row 0.0019; x0.1 0.00169 / 0.0019; x30 0.00169
[0.99948, 1.00085] row 0.0019. Tail 0.0026, router 0.99335, out rel 0.00670. Every swap-09 check is as in swap 09.

router (swap 11). The swapped step is the dense routing matrix [S, 256] fp32 (HYV4TopkRouter, as on layer 1: fp32
logits ffn_norm @ W^T, sigmoid, top-8 on sigmoid + e_score_correction_bias, weights = the unbiased sigmoids of the 8,
renormalized, x routed_scaling_factor 2.827), feeding the routed experts. The checks and limits are those of
test_swap_moe_full_11_router.py, except the golden-overlap floor, which stays at swap 10's 0.98 (layer 2 is more
precision-bound: median 8th-9th choice gap 0.0017, 691 of 2048 rows < 1e-3; the device ffn_norm alone costs the CPU
router 0.004 of overlap, 0.99335 at swap 10). Re-measured at layer 2 on the CPU (golden s4096 chunk 1, 2048 rows; the
router mutated on the golden ffn_norm; study /tmp/hy4_ssh11/study.py, outside the repo). "g" = vs golden: mean top-8
overlap / matched-row weight rel L2 / row sum / 2.827; "c" = vs the CPU router on the same input: overlap / matched
rel / worst row overlap:

    variant                  g ov    mrel    sum/2.827        | c ov    mrel    row
    fp32 reference           0.99823 0.00172 [1.0000, 1.0000] | 1.0     0       1.0    (ok)
    bf16 output              0.99823 0.00110 [0.9983, 1.0018] | 1.0     0.00170 1.0    (ok)
    bias bf16                0.99841 0.00172                  | 0.99969 0       0.875  (ok)
    logits bf16              0.98969 0.00205                  | 0.98907 0.00110 0.875  (fails c ov)
    sigmoid bf16             0.98792 0.00229                  | 0.98804 0.00147 0.750  (fails c ov)
    choice keys bf16         0.98395 0.00173                  | 0.98383 0       0.750  (fails c ov)
    all bf16                 0.97894 0.00254                  | 0.97882 0.00186 0.750  (fails g ov, c ov)
    weights from choice      0.99823 0.04067                  | 1.0     0.04059 1.0    (fails both mrel)
    weights x 1.005          0.99823 0.00531 [1.0050, 1.0050] | 1.0     0.00500 1.0    (fails sum, c mrel)
    logits x 1.01            0.99805 0.00533                  | 0.99969 0.00502 0.875  (fails c mrel)
    logits + 0.1             0.99268 0.01896                  | 0.99268 0.01888 0.875  (fails g mrel, c ov, c mrel)
    rows 1023 / 1024 swapped 0.99725 0.00172                  | 0.99902 0       0.0    (fails c row)
    last row zeroed          0.99774 0.00172 [0.0, 1.0] nnz 0 | 0.99951 0       0.0    (fails nnz, sum, c row)
    no bias                  0.88086                          | 0.88092         0.5    (fails g ov, c ov, c row)
    scale 2.5                0.99823 0.11566 [0.8843, 0.8843] | 1.0     0.11567 1.0    (fails sum, both mrel)

The swap asserts, on top of every swap-10 check: exactly 8 nonzeros per row, finite non-negative weights, every row
sum / 2.827 within 0.004 of 1; vs golden mean overlap >= 0.98 and matched-row weight rel L2 <= 0.01; vs the CPU router
on the same device ffn_norm mean overlap >= 0.996, matched-row rel L2 <= 0.004, worst row overlap >= 0.75 (adjacent
rows share at most 5 of 8 experts at layer 2, so a swapped or zeroed row scores <= 0.625). Every bug above fails at
least one of these; the bf16-output and bf16-bias rows pass. No scaled-input probe (as swap 11 on layer 1).
Reference / stub: BRINGUP_IMPL=reference passes (out 0.999995, router vs golden 0.99725 / 0.00173, vs CPU 1.0); the
zero stub fails every check.
Device run (this gate, tt/router.py:TtHy4Router): pcc_swap_out 0.999977; router nnz 8, row sums / 2.827 [1.00000,
1.00000]; vs golden overlap 0.99335 (1939 matched rows, rel 0.00206); vs the CPU router on the device ffn_norm overlap
0.99963, worst row 0.875, 2042 matched rows, rel 0.000059; tail 0.0025; out rel 0.00674. Every swap-10 check as in
swap 10.
"""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.interface import Ctx, run_block
from models.demos.common.bringup.testing.component import _step, module_under_test
from models.demos.common.bringup.testing.harness import (
    compare,
    component_golden,
    default_mode,
    device_ctx,
    mesh_parametrize,
    reference_ctx,
    spec,
    threshold,
)

S = spec()
BLOCK_TYPE = "moe_shared"
SWAPPED = [
    "attn_hc",
    "attn_hc_pre",
    "attn_norm",
    "q_a",
    "topk_shared",
    "attention",
    "attn_residual",
    "ffn_hc",
    "ffn_hc_pre",
    "ffn_norm",
    "router",
    "experts",
    "shared_expert",
    "moe_combine",
]
THRESHOLD = None  # None = spec thresholds.block (default 0.98)
HC_MULT = 4
GATES_MAX_REL_L2 = 0.01  # attn_hc [S, 8] vs golden: ||got - want|| / ||want||
GATES_MAX_COL_REL = 0.01  # per gate column (pre 0-3, post 4-7), rel L2 vs golden
GATES_MAX_POST_ROW_REL = 0.015  # worst row, rel L2 over the post columns
X_MAX_REL_L2 = 0.005  # attn_x vs golden, whole tensor
X_RATIO = (0.996, 1.004)  # attn_x per-row ||got|| / ||want|| vs golden
X_MAX_ROW_REL = 0.01  # attn_x vs golden, worst row
CPU_MAX_REL_L2 = 0.003  # attn_x vs the CPU hc_pre on the same inputs
CPU_MAX_ROW_REL = 0.006
ROT_MAX_REL_L2 = 0.004  # rotated pre gates, vs the CPU hc_pre on the same inputs
ROT_MAX_ROW_REL = 0.01
H_MID_MAX_REL = 0.005  # h_mid vs golden, whole tensor
H_MID_MAX_STREAM_REL = 0.01  # h_mid, rel L2 of each stream (stream 3 dominates the whole tensor at layer 2;
# swap 05: 0.005. The device attention's own error now reaches streams 0-2: device 0.0067, bf16 estimate 0.0049)
H_MID_MAX_ROW_REL = 0.02  # h_mid, worst (row, stream)
MID_STREAM_RATIO = (0.99, 1.01)  # h_mid per token and stream ||got|| / ||want|| vs golden
RES_MAX_REL_L2 = 5e-4  # h_mid vs the CPU attn_residual on the same inputs (device fp32: bit-identical), and rotated
RES_MAX_ROW_REL = 1e-3  # the same, worst (row, stream)
ADD_COEF_TOL = (
    0.005  # per stream |coef_j - 1| <= this + ROUND_MULT * r_j / ||t_j|| (test_c_moe_shared_attn_residual.py)
)
MAX_ADD_REL = 0.005  # per stream ||delta_j - t_j|| <= this * ||t_j|| + ROUND_MULT * r_j
MAX_ADD_ROW_REL = 0.05  # per token ||delta - t|| <= this * ||t|| + ROUND_MULT * r + ROW_FLOOR
ROUND_MULT = 2.0  # allowance for the bf16 rounding of the output
ROW_FLOOR = 1e-6
N_MAX_REL_L2 = 0.008  # attn_norm vs golden, and vs the CPU attn_norm on the same input
N_RATIO = (0.993, 1.007)  # attn_norm per-row norm ratio vs golden
N_MAX_ROW_REL = 0.015  # attn_norm worst row, vs golden and vs the CPU step on the same input
SYN_SCALE = 0.1  # eps check: the device attn_x x SYN_SCALE (bf16) through the module vs the CPU step
SYN_MAX_REL_L2 = 0.01
SYN_MAX_ROW_REL = 0.02
Q_MAX_REL_L2 = 0.008  # q_resid vs golden, and vs the CPU q_a on the same input
Q_RATIO = (0.994, 1.006)  # q_resid per-row norm ratio vs golden
Q_MAX_ROW_REL = 0.015  # q_resid worst row, vs golden and vs the CPU step on the same input
Q_SYN_SCALE = 0.01  # eps check: the device attn_norm x Q_SYN_SCALE (bf16) through the q_a module vs the CPU step
Q_SYN_MAX_REL_L2 = 0.01
Q_SYN_MAX_ROW_REL = 0.02
ATT_LIMITS = (0.012, (0.99, 1.01), 0.025, 0.004)  # attn_out (the swapped step): max rel L2, per-row norm ratio,
# worst row rel L2, max |<got, want> / <want, want> - 1| (test_c_moe_shared_attention.py checks 1-3): vs golden, vs CPU,
# chunk 0, probe
ATT_SCALED_LIMITS = (0.03, (0.99, 1.01), 0.05, 0.004)  # the attn_norm x 1e-3 run (component check 4)
PROBE_KEYS = 64  # probe topk: this many random causal positions per row, unsorted, the rest -1
PROBE_SEED = 0
ATT_SYN_SCALE = 1e-3  # eps check: the device attn_norm x ATT_SYN_SCALE (bf16), where kv_a_layernorm's eps matters
ROUTER_MIN_OVERLAP = 0.98  # router top-8 selection overlap vs golden
FHC_SMALL_COLS = (1,)  # ffn_hc gate at hc_eps at layer 2 (pre 1, column mean 1.4e-6; test_c_moe_shared_ffn_hc.py)
FHC_MAX_REL_L2 = 0.01  # ffn_hc [S, 8] vs golden, whole tensor
FHC_MAX_COL_REL = 0.015  # ffn_hc vs golden, per column rel L2 (columns 0, 2-7)
FHC_MAX_SMALL_COL_REL = 0.03  # the same on FHC_SMALL_COLS
FHC_MAX_POST_ROW_REL = 0.02  # ffn_hc vs golden, worst row over the post columns
FHC_CPU_MAX_REL_L2 = 0.005  # ffn_hc vs the CPU ffn_hc on the same device h_mid
FHC_CPU_MAX_COL_REL = 0.01  # the same, per column (columns 0, 2-7)
FHC_CPU_MAX_SMALL_COL_REL = 0.02  # the same on FHC_SMALL_COLS
FHC_CPU_MAX_POST_ROW_REL = 0.015  # the same, worst row over the post columns
FX_MAX_REL_L2 = 0.01  # ffn_x (now the device ffn_hc_pre) vs golden
FX_MAX_ROW_REL = 0.05  # the same, worst row
FX_CPU_MAX_REL_L2 = 0.005  # ffn_x from the device gates vs from the CPU gates on the same h_mid
FX_CPU_MAX_ROW_REL = 0.02  # the same, worst row
FXD_MAX_REL_L2 = 0.003  # ffn_x (device ffn_hc_pre) vs the CPU ffn_hc_pre on the same device h_mid and gates
FXD_MAX_ROW_REL = 0.006  # the same, worst row
FROT_MAX_REL_L2 = 0.004  # ffn_hc_pre on per-row rotated pre gates vs the CPU step on the same inputs
FROT_MAX_ROW_REL = 0.01  # the same, worst row
POST_MAX_STREAM_REL = 0.003  # out vs ffn_residual(h_mid, CPU gates, the block's mlp_out): per stream rel L2
POST_MAX_ROW_REL = 0.02  # the same, worst (row, stream)
TAIL_MAX_REL_L2 = 0.01  # block out vs the CPU tail (ffn_hc .. ffn_residual) from the same device h_mid
OUT_MAX_REL_L2 = 0.01  # block output, whole tensor
FN_MAX_REL_L2 = 0.01  # ffn_norm vs golden (upstream device error included)
FN_RATIO = (0.99, 1.01)  # ffn_norm per-row norm ratio vs golden
FN_MAX_ROW_REL_L2 = 0.03  # ffn_norm worst row vs golden
FN_CPU_MAX_REL_L2 = 0.008  # ffn_norm vs the CPU ffn_norm on the same device ffn_x (the component's golden limits)
FN_CPU_RATIO = (0.993, 1.007)
FN_CPU_MAX_ROW_REL_L2 = 0.015
FN_EPS_SCALE = 0.1  # eps check: the device ffn_x x FN_EPS_SCALE (bf16), eps dominates most rows, vs the CPU step
FN_EPS_MAX_REL_L2 = 0.01
FN_EPS_MAX_ROW_REL_L2 = 0.02
FN_SYN_SCALE = 30.0  # RMS check: the device ffn_x x FN_SYN_SCALE (bf16), mean(x^2) >> eps, vs the CPU step
FN_SYN_MAX_REL_L2 = 0.006
FN_SYN_RATIO = (0.993, 1.007)
FN_SYN_MAX_ROW_REL_L2 = 0.015
SENTINEL = 0xFFFFFFFF  # topk_large_indices' / sparse_sdpa's pad
TOP_K = 8
ROUTE_SCALE = 2.827  # routed_scaling_factor: every routing row sums to it (norm_topk_prob)
RT_MAX_ROW_SUM_ERR = 0.004  # |sum(row) / 2.827 - 1| (test_c_moe_shared_router.py)
RT_MAX_MATCHED_REL_L2 = 0.01  # router vs golden, weights on rows whose selected set equals the golden's
RT_CPU_MIN_OVERLAP = 0.996  # router vs the CPU router on the same device ffn_norm: mean per-row overlap
RT_CPU_MAX_MATCHED_REL_L2 = 0.004  # the same, matched-row weight rel L2
RT_CPU_MIN_ROW_OVERLAP = 0.75  # the same, worst row (a near-tie flip costs 1 of 8; other rows share <= 5 of 8)
EX_MAX_REL_L2 = (
    0.015  # experts_out vs the CPU experts on the same device ffn_norm + router (test_c_moe_shared_experts.py)
)
EX_RATIO = (0.98, 1.02)  # the same, per-token norm ratio
EX_MAX_ROW_REL = 0.018  # the same, worst token (layer 2: dropping 1-token expert 180 scores 0.0197; layer 1 0.03)
EX_MAX_COEF_ERR = 0.004  # the same, |<got, want> / <want, want> - 1| (float64)
EX_SYN_SCALE = 2.0  # clamp probe: the device ffn_norm x 2 (exact in bf16) with the device routing, vs the CPU experts
EX_G_MAX_REL_L2 = 0.03  # experts_out vs golden, whole tensor (routing flips and upstream device error included)
EX_G_RATIO = (0.97, 1.03)  # experts_out vs golden, per-token norm ratio on rows routed as in the golden
EX_G_MAX_ROW_REL = 0.04  # the same rows, worst token rel L2
EX_G_MAX_COEF_ERR = 0.004  # experts_out vs golden, global coefficient (whole tensor)
SE_MAX_REL_L2 = 0.008  # shared_out vs the CPU shared_expert on the device ffn_norm (component limits)
SE_RATIO = (0.99, 1.01)  # the same, per-row norm ratio
SE_MAX_ROW_REL = 0.015  # the same, worst row
SE_MAX_COEF_ERR = 0.003  # the same, |<got, want> / <want, want> - 1| (float64)
SE_SYN_SCALE = 3.0  # clamp probe: the device ffn_norm x 3 (bf16), gate / up pass +-10 (layer 2), vs CPU
SE_SYN_MAX_REL_L2 = 0.006
SE_SYN_RATIO = (0.99, 1.01)
SE_SYN_MAX_ROW_REL = 0.012
SE_G_MAX_REL_L2 = 0.01  # shared_out vs golden (upstream device error included)
SE_G_RATIO = (0.985, 1.015)
SE_G_MAX_ROW_REL = 0.02
SE_G_MAX_COEF_ERR = 0.004
MC_MAX_REL_L2 = 0.003  # mlp_out vs the exact fp32 sum of the block's own device addends (bf16 output 0.0017)
MC_RATIO = (0.997, 1.003)  # the same, per-row norm ratio (bf16 output [0.9999, 1.0001])
MC_MAX_ROW_REL = 0.005  # the same, worst row (bf16 output 0.0017; shared rows 1023 / 1024 swapped 0.50)
MC_MAX_COEF_DEV = 0.002  # per addend |coef - 1| on those addends (bf16 output 1e-5; 1.003 x experts 0.003)
MC_MAX_ADD_REL = {"shared_out": 0.008, "experts_out": 0.005}  # per addend ||delta - t|| / ||t|| (bf16 0.0038 / 0.0021)
MC_MAX_ADD_ROW_REL = 0.03  # per addend, worst row (bf16 0.013 / 0.014; 1.005 x shared moves the experts row 0.041)
MC_PROBE_MAX_REL = 0.004  # the module on (experts, -shared) and (experts, 0) vs the exact sums
MC_PROBE_MAX_ROW_REL = 0.005
MC_G_MAX_REL_L2 = 0.02  # mlp_out vs golden (the exact sum of the device addends: 0.0111)
MC_G_MAX_COEF_ERR = 0.004  # the same, global coefficient (exact sum 1.00056)
MC_G_RATIO = (0.98, 1.02)  # the same, per-row ratio on rows routed as in the golden (exact sum [0.9903, 1.0074])
MC_G_MAX_ROW_REL = 0.03  # the same rows, worst row (exact sum 0.0176)


def _rel(got, want):
    got, want = got.float().reshape(want.shape), want.float()
    return ((got - want).norm() / want.norm().clamp_min(1e-12)).item()


def _worst_row_rel(got, want, streams=1):
    got = got.float().reshape(want.shape[0], streams, -1)
    want = want.float().reshape(want.shape[0], streams, -1)
    return ((got - want).norm(dim=-1) / want.norm(dim=-1).clamp_min(1e-12)).max().item()


def _ratio(got, want):
    r = got.float().reshape(want.shape).norm(dim=-1) / want.float().norm(dim=-1).clamp_min(1e-12)
    return r.min().item(), r.max().item()


def _errors(got, want):
    """rel L2, per-row norm ratio (min, max), worst row rel L2."""
    if want.shape[0] == 0:  # no rows selected (e.g. none routed as the golden): fail every limit, do not crash
        return float("inf"), 0.0, float("inf"), float("inf")
    got, want = got.float().reshape(want.shape), want.float()
    rel = ((got - want).norm() / want.norm().clamp_min(1e-12)).item()
    wn = want.norm(dim=-1).clamp_min(1e-12)
    ratio = got.norm(dim=-1) / wn
    return rel, ratio.min().item(), ratio.max().item(), ((got - want).norm(dim=-1) / wn).max().item()


def _rotated(gates):
    """Each row's pre gates rotated by (row mod 4), so every stream meets the large gate on a quarter of the rows."""
    n = gates.shape[0]
    idx = (torch.arange(HC_MULT)[None, :] + torch.arange(n)[:, None]) % HC_MULT
    gs = gates.clone()
    gs[:, :HC_MULT] = torch.gather(gates[:, :HC_MULT], 1, idx)
    return gs


def _rotated_post(gates):
    """Each row's post gates (columns 4-7) rotated by (row mod 4), so every stream meets the large post gates."""
    n = gates.shape[0]
    idx = (torch.arange(HC_MULT)[None, :] + torch.arange(n)[:, None]) % HC_MULT
    gs = gates.clone()
    gs[:, HC_MULT:] = torch.gather(gates[:, HC_MULT:], 1, idx)
    return gs


def _stream_rel(got, want, streams):
    got = got.float().reshape(want.shape[0], streams, -1)
    want = want.float().reshape(want.shape[0], streams, -1)
    return ((got - want).norm(dim=(0, 2)) / want.norm(dim=(0, 2)).clamp_min(1e-12)).tolist()


def _norm_topk(out, want):
    """int64 [S, k] with every pad (negative, or the uint32 sentinel) as -1; None if not integer / wrong size."""
    if out.is_floating_point() or out.numel() != want.numel():
        return None
    t = out.reshape(want.shape).to(torch.int64)
    return torch.where((t < 0) | (t == SENTINEL), torch.full_like(t, -1), t)


def _topk_check(got, want, start, tag):
    """Exact per-row set equality with want plus sparse_sdpa's pad layout; returns failure messages."""
    fails = []
    valid = got >= 0
    pos = torch.arange(start, start + got.shape[0])[:, None]
    noncausal = (valid & (got > pos)).sum().item()
    if noncausal:
        fails.append(f"{noncausal} selected positions are after their query row (non-causal)")
    srt, ws = got.sort(dim=-1).values, want.sort(dim=-1).values
    dup = ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).sum().item()
    if dup:
        fails.append(f"{dup} repeated positions within rows")
    bad = (srt != ws).any(-1).nonzero().flatten()
    if bad.numel():
        r = bad[0].item()
        fails.append(
            f"{bad.numel()} rows differ from the wanted set (first: row {r} at position {start + r}: "
            f"{valid[r].sum().item()} valid, want {(want[r] >= 0).sum().item()})"
        )
    tail_bad = (valid[:, 1:] & ~valid[:, :-1]).any(-1).sum().item()
    if tail_bad:
        fails.append(f"{tail_bad} rows have a valid position after a pad (sparse_sdpa needs a contiguous pad tail)")
    empty = (valid.sum(-1) == 0).sum().item()
    if empty:
        fails.append(f"{empty} rows have no valid key (sparse_sdpa needs >= 1)")
    rows = torch.tensor(
        [torch.isin(b[b >= 0], a[a >= 0]).float().mean().item() if (b >= 0).any() else 1.0 for a, b in zip(got, want)]
    )
    metrics.record(f"{tag}topk_overlap_swap", rows.mean().item())
    metrics.record(f"{tag}topk_worst_row_overlap_swap", rows.min().item())
    print(
        f"{tag}topk: mean set overlap={rows.mean().item():.6f} worst row={rows.min().item():.5f} (== 1) "
        f"differing rows={bad.numel()} pads={(~valid).sum().item()} (want {(want < 0).sum().item()}) "
        f"positional match={(got == want).float().mean().item():.4f} (not gated)"
    )
    return [f"{tag}topk: {m}" for m in fails]


def _probe_topk(start, rows, k, n, seed):
    """[rows, k] int64: per row min(n, pos + 1) distinct random positions in [0, pos], unsorted, then -1 pads."""
    gen = torch.Generator().manual_seed(seed)
    out = torch.full((rows, k), -1, dtype=torch.int64)
    for i in range(rows):
        p = start + i
        m = min(n, p + 1)
        out[i, :m] = torch.randperm(p + 1, generator=gen)[:m]
    return out


@mesh_parametrize
def test_swap(mesh_device):
    g, c = component_golden(S)
    layer = S.representative_layer(BLOCK_TYPE)
    ref = S.hooks().reference(S, layers=[layer], dtype=torch.float32)
    steps = ref.block_graph(layer)
    rctx, dctx = reference_ctx(ref, layer, g, c), device_ctx(layer, g, c)
    # A shared layer reuses the latest full layer's top-k for this chunk; that layer does not run here, so take it
    # from the golden (the source layer's recorded topk, int64 [S, index_topk]).
    src = ref.cfg.topk_source(layer)
    shared_topk = g.layer(c, src)["topk"]
    assert src != layer and shared_topk.shape[0] == g.chunk, f"layer {src} topk {tuple(shared_topk.shape)}"
    rctx.extra["shared_topk"] = shared_topk
    dctx.extra["shared_topk"] = shared_topk
    overrides, muts = {}, {}
    for name in SWAPPED:
        _step(ref, layer, name)
        mut = module_under_test(S, ref, mesh_device, layer, name)
        assert not getattr(mut, "cpu_bridge", False), f"device_component returned a CPU bridge for {name}"
        muts[name] = mut
        overrides[name] = lambda ctx, *x, mut=mut: mut(ctx, dctx, *x)
    gl = g.layer(c, layer)
    seen = {}
    run_block(
        steps,
        lambda n: ref.component(layer, n),
        rctx,
        gl["in"].float(),
        rec=lambda n, t: seen.__setitem__(n, t),
        overrides=overrides,
    )
    for n, t in seen.items():
        if n not in ("in", "out") and n in gl:
            compare(f"pcc_swap_{n}", t, gl[n], default_mode(gl[n]), 0.0)  # the trail, for diagnosis; not gated
    thr = threshold(S, "block") if THRESHOLD is None else THRESHOLD
    _, ok = compare("pcc_swap_out", seen["out"], gl["out"], "pcc", thr)

    # Extra checks (see the module docstring). Recorded as informational metrics, asserted here.
    failures = [] if ok else [f"pcc_swap_out below {thr}"]

    def finite_shape(n):
        got, want = seen[n], gl[n]
        if got.numel() != want.numel() or got.shape[-1] != want.shape[-1]:
            failures.append(f"{n}: shape {tuple(got.shape)} vs golden {tuple(want.shape)}")
            return False
        if not torch.isfinite(got.float()).all():
            failures.append(f"{n}: non-finite")
            return False
        return True

    # attn_hc: the iHC gates [S, 8] (pre 0-3 | post 4-7), as swap 01.
    gates_ok = finite_shape("attn_hc")
    if gates_ok:
        want = gl["attn_hc"].float()
        got = seen["attn_hc"].float().reshape(want.shape)
        rel = _rel(got, want)
        col_rel = ((got - want).norm(dim=0) / want.norm(dim=0).clamp_min(1e-12)).tolist()
        post_row = _worst_row_rel(got[:, HC_MULT:], want[:, HC_MULT:])
        metrics.record("rel_l2_swap_attn_hc", rel)
        metrics.record("max_col_rel_l2_swap_attn_hc", max(col_rel))
        metrics.record("post_worst_row_rel_l2_swap_attn_hc", post_row)
        print(
            f"attn_hc: rel_l2={rel:.6f} (<= {GATES_MAX_REL_L2}) col rel (pre 0-3 | post 4-7)="
            f"{[round(v, 5) for v in col_rel]} (<= {GATES_MAX_COL_REL}) post worst row={post_row:.5f} "
            f"(<= {GATES_MAX_POST_ROW_REL})"
        )
        if rel > GATES_MAX_REL_L2:
            failures.append(f"attn_hc: rel L2 {rel:.5f} > {GATES_MAX_REL_L2}")
        bad = [j for j, v in enumerate(col_rel) if v > GATES_MAX_COL_REL]
        if bad:
            failures.append(f"attn_hc: rel L2 > {GATES_MAX_COL_REL} in columns {bad}")
        if post_row > GATES_MAX_POST_ROW_REL:
            failures.append(f"attn_hc: post worst row rel L2 {post_row:.5f} > {GATES_MAX_POST_ROW_REL}")

    # attn_x vs golden (swap 02).
    x_ok = finite_shape("attn_x")
    if x_ok:
        rel = _rel(seen["attn_x"], gl["attn_x"])
        rmin, rmax = _ratio(seen["attn_x"], gl["attn_x"])
        row = _worst_row_rel(seen["attn_x"], gl["attn_x"])
        metrics.record("rel_l2_swap_attn_x", rel)
        metrics.record("worst_row_rel_l2_swap_attn_x", row)
        print(
            f"attn_x vs golden: rel_l2={rel:.6f} (<= {X_MAX_REL_L2}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(X_RATIO)}) worst_row_rel_l2={row:.5f} (<= {X_MAX_ROW_REL})"
        )
        if rel > X_MAX_REL_L2:
            failures.append(f"attn_x: rel L2 {rel:.5f} > {X_MAX_REL_L2}")
        if not (X_RATIO[0] <= rmin and rmax <= X_RATIO[1]):
            failures.append(f"attn_x: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(X_RATIO)}")
        if row > X_MAX_ROW_REL:
            failures.append(f"attn_x: worst row rel L2 {row:.5f} > {X_MAX_ROW_REL}")

        if gates_ok:
            # vs the CPU hc_pre on the same inputs (block input + the gates the device produced).
            cpu = ref.component(layer, "attn_hc_pre")
            gates = seen["attn_hc"].float().reshape(gl["attn_hc"].shape)
            same = cpu(rctx, gl["in"].float(), gates).float()
            crel, crow = _rel(seen["attn_x"], same), _worst_row_rel(seen["attn_x"], same)
            metrics.record("rel_l2_swap_attn_x_vs_cpu", crel)
            print(
                f"attn_x vs CPU hc_pre on the same inputs: rel_l2={crel:.6f} (<= {CPU_MAX_REL_L2}) "
                f"worst_row_rel_l2={crow:.5f} (<= {CPU_MAX_ROW_REL})"
            )
            if crel > CPU_MAX_REL_L2 or crow > CPU_MAX_ROW_REL:
                failures.append(f"attn_x vs CPU on the same inputs: rel {crel:.5f} / worst row {crow:.5f}")

            # Stream order: the module again with each row's pre gates rotated by (row mod 4).
            rot = (gl["in"].float(), _rotated(gates))
            rot_want = cpu(rctx, *rot).float()
            rot_out = muts["attn_hc_pre"](rctx, dctx, *rot)
            if rot_out.numel() != rot_want.numel() or not torch.isfinite(rot_out.float()).all():
                failures.append(f"attn_hc_pre rotated gates: shape {tuple(rot_out.shape)} or non-finite")
            else:
                yrel, yrow = _rel(rot_out, rot_want), _worst_row_rel(rot_out, rot_want)
                metrics.record("rot_rel_l2_swap_attn_x", yrel)
                print(
                    f"attn_hc_pre with rotated pre gates vs CPU: rel_l2={yrel:.6f} (<= {ROT_MAX_REL_L2}) "
                    f"worst_row_rel_l2={yrow:.5f} (<= {ROT_MAX_ROW_REL})"
                )
                if yrel > ROT_MAX_REL_L2 or yrow > ROT_MAX_ROW_REL:
                    failures.append(f"attn_hc_pre rotated gates: rel {yrel:.5f} / worst row {yrow:.5f} (stream order?)")

    def rel_row(tag, got, want, max_rel, max_row, ratio=None, what="vs golden"):
        rel, row = _rel(got, want), _worst_row_rel(got, want)
        metrics.record(f"rel_l2_swap_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_{tag}", row)
        msg = f"{tag} {what}: rel_l2={rel:.6f} (<= {max_rel})"
        if ratio is not None:
            rmin, rmax = _ratio(got, want)
            msg += f" row norm ratio=[{rmin:.5f}, {rmax:.5f}] (in {list(ratio)})"
            if not (ratio[0] <= rmin and rmax <= ratio[1]):
                failures.append(f"{tag} {what}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio)}")
        if max_row is not None:
            msg += f" worst_row_rel_l2={row:.5f} (<= {max_row})"
            if row > max_row:
                failures.append(f"{tag} {what}: worst row rel L2 {row:.5f} > {max_row}")
        print(msg)
        if rel > max_rel:
            failures.append(f"{tag} {what}: rel L2 {rel:.5f} > {max_rel}")

    # attn_norm (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    n_ok = finite_shape("attn_norm")
    if n_ok:
        rel_row("attn_norm", seen["attn_norm"], gl["attn_norm"], N_MAX_REL_L2, N_MAX_ROW_REL, N_RATIO)
        if x_ok:
            cpu_n = ref.component(layer, "attn_norm")
            x = seen["attn_x"].float().reshape(gl["attn_x"].shape)
            rel_row(
                "attn_norm_vs_cpu",
                seen["attn_norm"],
                cpu_n(rctx, x).float(),
                N_MAX_REL_L2,
                N_MAX_ROW_REL,
                what="vs CPU attn_norm on the device attn_x",
            )
            xs = (x * SYN_SCALE).bfloat16().float()
            syn_want = cpu_n(rctx, xs).float()
            syn_out = muts["attn_norm"](rctx, dctx, xs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"attn_norm scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, yrow = _rel(syn_out, syn_want), _worst_row_rel(syn_out, syn_want)
                ymin, ymax = _ratio(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_attn_norm", yrel)
                metrics.record("syn_worst_row_rel_l2_swap_attn_norm", yrow)
                print(
                    f"attn_norm on attn_x x{SYN_SCALE} vs CPU: rel_l2={yrel:.6f} (<= {SYN_MAX_REL_L2}) row norm "
                    f"ratio=[{ymin:.5f}, {ymax:.5f}] worst_row_rel_l2={yrow:.5f} (<= {SYN_MAX_ROW_REL})"
                )
                if yrel > SYN_MAX_REL_L2 or yrow > SYN_MAX_ROW_REL:
                    failures.append(f"attn_norm scaled input: rel {yrel:.5f} / worst row {yrow:.5f} (wrong eps?)")

    # q_resid (the swapped step): vs golden, vs the CPU step on the same input, and on a scaled input (eps).
    if finite_shape("q_resid"):
        rel_row("q_resid", seen["q_resid"], gl["q_resid"], Q_MAX_REL_L2, Q_MAX_ROW_REL, Q_RATIO)
        if n_ok:
            cpu_q = ref.component(layer, "q_a")
            xn = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
            rel_row(
                "q_resid_vs_cpu",
                seen["q_resid"],
                cpu_q(rctx, xn).float(),
                Q_MAX_REL_L2,
                Q_MAX_ROW_REL,
                what="vs CPU q_a on the device attn_norm",
            )
            xs = (xn * Q_SYN_SCALE).bfloat16().float()
            syn_want = cpu_q(rctx, xs).float()
            syn_out = muts["q_a"](rctx, dctx, xs)
            if syn_out.numel() != syn_want.numel() or not torch.isfinite(syn_out.float()).all():
                failures.append(f"q_a scaled input: shape {tuple(syn_out.shape)} or non-finite")
            else:
                yrel, yrow = _rel(syn_out, syn_want), _worst_row_rel(syn_out, syn_want)
                ymin, ymax = _ratio(syn_out, syn_want)
                metrics.record("syn_rel_l2_swap_q_resid", yrel)
                metrics.record("syn_worst_row_rel_l2_swap_q_resid", yrow)
                print(
                    f"q_a on attn_norm x{Q_SYN_SCALE} vs CPU: rel_l2={yrel:.6f} (<= {Q_SYN_MAX_REL_L2}) row norm "
                    f"ratio=[{ymin:.5f}, {ymax:.5f}] worst_row_rel_l2={yrow:.5f} (<= {Q_SYN_MAX_ROW_REL})"
                )
                if yrel > Q_SYN_MAX_REL_L2 or yrow > Q_SYN_MAX_ROW_REL:
                    failures.append(f"q_a scaled input: rel {yrel:.5f} / worst row {yrow:.5f} (wrong eps?)")

    # topk (the swapped step): exact per-row sets vs golden and vs the shared input, pad layout, then chunk 0.
    want_tk = gl["topk"].long()
    got_tk = _norm_topk(seen["topk"], want_tk)
    if got_tk is None:
        failures.append(f"topk: {seen['topk'].dtype} {tuple(seen['topk'].shape)}, want integer {tuple(want_tk.shape)}")
    else:
        start = c * g.chunk
        failures += _topk_check(got_tk, want_tk, start, "")
        failures += _topk_check(got_tk, shared_topk.long(), start, "vs_shared_input_")
        g0 = g.layer(0, layer)
        shared0 = g.layer(0, src)["topk"]
        rctx0, dctx0 = reference_ctx(ref, layer, g, 0), device_ctx(layer, g, 0)
        rctx0.extra["shared_topk"] = shared0.clone()
        dctx0.extra["shared_topk"] = shared0.clone()
        want0 = g0["topk"].long()
        got0 = _norm_topk(muts["topk_shared"](rctx0, dctx0, g0["attn_norm"].float()), want0)
        if got0 is None:
            failures.append("chunk0_topk: not integer or wrong size")
        else:
            failures += _topk_check(got0, want0, 0, "chunk0_")

    # attn_out (the swapped step): vs golden, vs the CPU step on the same device inputs, then the module on chunk 0,
    # on a probe topk and on a scaled attn_norm, each vs the CPU step on the same inputs.
    def att_check(tag, got, want, limits=ATT_LIMITS):
        max_rel, ratio_lim, max_row, max_coef = limits
        if got.numel() != want.numel() or not torch.isfinite(got.float()).all():
            failures.append(f"attn_out {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)} or non-finite")
            return
        rel, row = _rel(got, want), _worst_row_rel(got, want)
        rmin, rmax = _ratio(got, want)
        gd, wd = got.double().reshape(want.shape), want.double()
        coef = ((gd * wd).sum() / (wd * wd).sum()).item()
        metrics.record(f"rel_l2_swap_attn_out_{tag}", rel)
        metrics.record(f"worst_row_rel_l2_swap_attn_out_{tag}", row)
        metrics.record(f"scale_coef_swap_attn_out_{tag}", coef)
        print(
            f"attn_out {tag}: rel_l2={rel:.6f} (<= {max_rel}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(ratio_lim)}) worst_row_rel_l2={row:.5f} (<= {max_row}) coef={coef:.6f} (within {max_coef})"
        )
        if rel > max_rel:
            failures.append(f"attn_out {tag}: rel L2 {rel:.5f} > {max_rel}")
        if not (ratio_lim[0] <= rmin and rmax <= ratio_lim[1]):
            failures.append(f"attn_out {tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio_lim)}")
        if row > max_row:
            failures.append(f"attn_out {tag}: worst row rel L2 {row:.5f} > {max_row}")
        if abs(coef - 1) > max_coef:
            failures.append(f"attn_out {tag}: global scale coefficient {coef:.5f} not within {max_coef} of 1")

    if finite_shape("attn_out"):
        att_check("vs_golden", seen["attn_out"], gl["attn_out"])
        if n_ok and "q_resid" in seen and got_tk is not None:
            cpu = ref.component(layer, "attention")
            xn = seen["attn_norm"].float().reshape(gl["attn_norm"].shape)
            qr = seen["q_resid"].float().reshape(gl["q_resid"].shape)
            att_check("vs_cpu", seen["attn_out"], cpu(reference_ctx(ref, layer, g, c), xn, qr, got_tk).float())

            # (c) golden chunk 0: start 0, empty prefix, -1 pads in every row.
            g0 = g.layer(0, layer)
            in0 = (g0["attn_norm"].float(), g0["q_resid"].float(), g0["topk"].long())
            dctx0 = Ctx(layer, 0, g.chunk, None, {"state_prefix": g.state(layer), "prefix_len": 0, "max_seq": g.seq})
            want0 = cpu(reference_ctx(ref, layer, g, 0), *in0).float()
            att_check("chunk0", muts["attention"](reference_ctx(ref, layer, g, 0), dctx0, *in0), want0)

            # (d) probe topk: 64 random causal positions per row, unsorted, the rest -1.
            start = c * g.chunk
            ptk = _probe_topk(start, xn.shape[0], gl["topk"].shape[-1], PROBE_KEYS, PROBE_SEED)
            pwant = cpu(reference_ctx(ref, layer, g, c), xn, qr, ptk).float()
            att_check("probe", muts["attention"](reference_ctx(ref, layer, g, c), dctx, xn, qr, ptk), pwant)

            # (e) eps: attn_norm scaled so kv_a_layernorm's eps 1e-6 is visible.
            xs = (xn * ATT_SYN_SCALE).bfloat16().float()
            swant = cpu(reference_ctx(ref, layer, g, c), xs, qr, got_tk).float()
            sout = muts["attention"](reference_ctx(ref, layer, g, c), dctx, xs, qr, got_tk)
            att_check("scaled", sout, swant, ATT_SCALED_LIMITS)
        else:
            failures.append("attn_out: vs-CPU, chunk 0, probe and scaled checks skipped (unusable upstream output)")

    # h_mid (the swapped step): vs golden (+ per-stream norm ratio), vs the CPU step on the same device inputs (+ the
    # addend per stream), and the module on rotated post gates vs the CPU step.
    if finite_shape("h_mid"):
        rel = _rel(seen["h_mid"], gl["h_mid"])
        wr = _worst_row_rel(seen["h_mid"], gl["h_mid"], HC_MULT)
        metrics.record("rel_l2_swap_h_mid", rel)
        metrics.record("worst_row_rel_l2_swap_h_mid", wr)
        print(f"h_mid: rel_l2={rel:.6f} (<= {H_MID_MAX_REL}) worst (row, stream) rel={wr:.6f} (<= {H_MID_MAX_ROW_REL})")
        if rel > H_MID_MAX_REL:
            failures.append(f"h_mid rel L2 {rel:.5f} > {H_MID_MAX_REL}")
        if wr > H_MID_MAX_ROW_REL:
            failures.append(f"h_mid worst row rel L2 {wr:.5f} > {H_MID_MAX_ROW_REL}")
        sr = _stream_rel(seen["h_mid"], gl["h_mid"], HC_MULT)
        metrics.record("max_stream_rel_l2_swap_h_mid", max(sr))
        print(f"h_mid: per-stream rel_l2={[round(v, 6) for v in sr]} (<= {H_MID_MAX_STREAM_REL})")
        bad = [j for j, v in enumerate(sr) if v > H_MID_MAX_STREAM_REL]
        if bad:
            failures.append(f"h_mid streams {bad} rel L2 above {H_MID_MAX_STREAM_REL}: {sr}")

        want = gl["h_mid"].float()
        n = want.shape[0]
        hm = seen["h_mid"].float().reshape(n, HC_MULT, -1)
        sr = hm.norm(dim=-1) / want.view(n, HC_MULT, -1).norm(dim=-1).clamp_min(1e-12)
        smin, smax = sr.min().item(), sr.max().item()
        metrics.record("stream_norm_ratio_min_swap_h_mid", smin)
        metrics.record("stream_norm_ratio_max_swap_h_mid", smax)
        print(f"h_mid per-token stream norm ratio=[{smin:.5f}, {smax:.5f}] (in {list(MID_STREAM_RATIO)})")
        if not (MID_STREAM_RATIO[0] <= smin and smax <= MID_STREAM_RATIO[1]):
            failures.append(f"h_mid: stream norm ratio [{smin:.5f}, {smax:.5f}] outside {list(MID_STREAM_RATIO)}")

        a_ok = seen["attn_out"].numel() == gl["attn_out"].numel() and torch.isfinite(seen["attn_out"].float()).all()
        g_ok = seen["attn_hc"].numel() == gl["attn_hc"].numel() and torch.isfinite(seen["attn_hc"].float()).all()
        if a_ok and g_ok:
            cpu_r = ref.component(layer, "attn_residual")
            xs_in = gl["in"].float()
            gt = seen["attn_hc"].float().reshape(gl["attn_hc"].shape)
            y = seen["attn_out"].float().reshape(gl["attn_out"].shape)
            same = cpu_r(rctx, xs_in, gt, y).float()
            crel, crow = _rel(seen["h_mid"], same), _worst_row_rel(seen["h_mid"], same, HC_MULT)
            metrics.record("rel_l2_swap_h_mid_vs_cpu", crel)
            metrics.record("worst_row_rel_l2_swap_h_mid_vs_cpu", crow)
            print(
                f"h_mid vs CPU attn_residual on the same inputs: rel_l2={crel:.7f} (<= {RES_MAX_REL_L2}) "
                f"worst (row, stream) rel={crow:.7f} (<= {RES_MAX_ROW_REL})"
            )
            if crel > RES_MAX_REL_L2 or crow > RES_MAX_ROW_REL:
                failures.append(f"h_mid vs CPU on the same inputs: rel {crel:.6f} / worst row {crow:.6f}")

            # The addend on each stream: delta_j = h_mid_j - in_j vs t_j = post_j * attn_out, rounding-aware limits
            # (the component's; r = bf16 rounding error of the exact fp32 result), float64 statistics.
            xs = xs_in.view(n, HC_MULT, -1)
            tgt32 = gt[:, HC_MULT:].unsqueeze(-1) * y.view(n, 1, -1)
            exact = xs + tgt32
            rnd = (exact.bfloat16().float() - exact).double()
            tgt = tgt32.double()
            delta = (hm - xs).double()
            err = delta - tgt
            tn = tgt.norm(dim=(0, 2)).clamp_min(1e-30)
            rs = rnd.norm(dim=(0, 2))
            coef = (delta * tgt).sum(dim=(0, 2)) / (tn * tn)
            coef_x = ((coef - 1).abs() / (ADD_COEF_TOL + ROUND_MULT * rs / tn)).tolist()  # > 1 fails
            excess = (err.norm(dim=(0, 2)) / (MAX_ADD_REL * tn + ROUND_MULT * rs)).tolist()  # > 1 fails
            row_x = err.norm(dim=-1) / (MAX_ADD_ROW_REL * tgt.norm(dim=-1) + ROUND_MULT * rnd.norm(dim=-1) + ROW_FLOOR)
            worst = row_x.max().item()
            worst_at = divmod(int(row_x.argmax().item()), HC_MULT)
            metrics.record("add_coef_min_swap_h_mid", coef.min().item())
            metrics.record("add_coef_max_swap_h_mid", coef.max().item())
            metrics.record("add_excess_swap_h_mid", max(excess))
            metrics.record("add_worst_row_excess_swap_h_mid", worst)
            print(
                f"h_mid addend per stream: coef={[round(v, 5) for v in coef.tolist()]} (|coef-1|/tol="
                f"{[round(v, 3) for v in coef_x]} <= 1) excess={[round(v, 3) for v in excess]} (<= 1) "
                f"worst row excess={worst:.3f} (<= 1) at (row, stream)={worst_at}"
            )
            bad = [j for j, v in enumerate(coef_x) if v > 1]
            if bad:
                failures.append(f"h_mid addend: coefficient off on streams {bad}: {coef.tolist()}")
            bad = [j for j, v in enumerate(excess) if v > 1]
            if bad:
                failures.append(f"h_mid addend: error above the limit on streams {bad}: excess {excess}")
            if worst > 1:
                failures.append(f"h_mid addend: worst row excess {worst:.3f} at (row, stream) {worst_at}")

            # Rotated post gates: stream 3 (post ~0.003) meets the large gates on three quarters of the rows.
            gr = _rotated_post(gt)
            rwant = cpu_r(rctx, xs_in, gr, y).float()
            rout = muts["attn_residual"](rctx, dctx, xs_in, gr, y)
            if rout.numel() != rwant.numel() or not torch.isfinite(rout.float()).all():
                failures.append(f"attn_residual rotated post gates: shape {tuple(rout.shape)} or non-finite")
            else:
                prel, prow = _rel(rout, rwant), _worst_row_rel(rout, rwant, HC_MULT)
                metrics.record("rot_rel_l2_swap_h_mid", prel)
                metrics.record("rot_worst_row_rel_l2_swap_h_mid", prow)
                print(
                    f"attn_residual with rotated post gates vs CPU: rel_l2={prel:.7f} (<= {RES_MAX_REL_L2}) "
                    f"worst (row, stream) rel={prow:.7f} (<= {RES_MAX_ROW_REL})"
                )
                if prel > RES_MAX_REL_L2 or prow > RES_MAX_ROW_REL:
                    failures.append(f"attn_residual rotated post gates: rel {prel:.6f} / worst row {prow:.6f}")
        else:
            failures.append("h_mid: attn_hc / attn_out unusable, cannot check attn_residual against the CPU step")

    # ffn_hc (the swapped step): the iHC gates [S, 8] from h_mid, vs golden and vs the CPU ffn_hc on the same device
    # h_mid; per column (the hc_eps column 1 carries bugs nothing downstream sees), the pre gates through ffn_x,
    # the post gates through out with the block's own mlp_out (no routing in between).
    def gate_check(tag, got, want, max_rel, max_col, max_small, max_row):
        err = got - want
        rel = (err.norm() / want.norm().clamp_min(1e-12)).item()
        col = (err.norm(dim=0) / want.norm(dim=0).clamp_min(1e-12)).tolist()
        prow = (err[:, HC_MULT:].norm(dim=-1) / want[:, HC_MULT:].norm(dim=-1).clamp_min(1e-12)).max().item()
        big = max(v for j, v in enumerate(col) if j not in FHC_SMALL_COLS)
        small = max(col[j] for j in FHC_SMALL_COLS)
        metrics.record(f"rel_l2_swap_ffn_hc_{tag}", rel)
        metrics.record(f"max_col_rel_l2_swap_ffn_hc_{tag}", big)
        metrics.record(f"max_small_col_rel_l2_swap_ffn_hc_{tag}", small)
        metrics.record(f"post_worst_row_rel_l2_swap_ffn_hc_{tag}", prow)
        print(
            f"ffn_hc {tag}: rel_l2={rel:.6f} (<= {max_rel}) col rel={[round(v, 5) for v in col]} (<= {max_col}, "
            f"columns {list(FHC_SMALL_COLS)} <= {max_small}) post worst row={prow:.5f} (<= {max_row})"
        )
        if rel > max_rel:
            failures.append(f"ffn_hc {tag}: rel L2 {rel:.5f} > {max_rel}")
        bad = [j for j, v in enumerate(col) if v > (max_small if j in FHC_SMALL_COLS else max_col)]
        if bad:
            failures.append(f"ffn_hc {tag}: column rel L2 above the limit in columns {bad}")
        if prow > max_row:
            failures.append(f"ffn_hc {tag}: post worst row rel L2 {prow:.5f} > {max_row}")

    mid_ok = seen["h_mid"].numel() == gl["h_mid"].numel() and torch.isfinite(seen["h_mid"].float()).all()
    if finite_shape("ffn_hc"):
        want_hc = gl["ffn_hc"].float()
        gt = seen["ffn_hc"].float().reshape(want_hc.shape)
        gate_check(
            "vs_golden", gt, want_hc, FHC_MAX_REL_L2, FHC_MAX_COL_REL, FHC_MAX_SMALL_COL_REL, FHC_MAX_POST_ROW_REL
        )
        if finite_shape("ffn_x"):
            rel_row("ffn_x", seen["ffn_x"], gl["ffn_x"], FX_MAX_REL_L2, FX_MAX_ROW_REL)
        if mid_ok:
            hm = seen["h_mid"].float().reshape(gl["h_mid"].shape)
            cpu_hc, cpu_pre = ref.component(layer, "ffn_hc"), ref.component(layer, "ffn_hc_pre")
            cpu_g = cpu_hc(rctx, hm).float()
            gate_check(
                "vs_cpu",
                gt,
                cpu_g,
                FHC_CPU_MAX_REL_L2,
                FHC_CPU_MAX_COL_REL,
                FHC_CPU_MAX_SMALL_COL_REL,
                FHC_CPU_MAX_POST_ROW_REL,
            )
            cpu_fx = cpu_pre(rctx, hm, cpu_g).float()
            rel_row(
                "ffn_x_vs_cpu",
                cpu_pre(rctx, hm, gt).float(),
                cpu_fx,
                FX_CPU_MAX_REL_L2,
                FX_CPU_MAX_ROW_REL,
                what="from the device gates vs from the CPU gates on the device h_mid",
            )

            # ffn_hc_pre (the swapped step): the device ffn_x vs the CPU step on the same device h_mid and gates, then
            # the module on per-row rotated pre gates (pre gate 1 is at hc_eps and gate 0 ~1e-3 here, so a dropped
            # stream 0 or 1, or a 0 / 1 swap, is otherwise invisible).
            def pre_check(tag, got, want, max_rel, max_row):
                if got.numel() != want.numel() or not torch.isfinite(got.float()).all():
                    failures.append(f"ffn_hc_pre {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)} or non-finite")
                    return
                rel, row = _rel(got, want), _worst_row_rel(got, want)
                rmin, rmax = _ratio(got, want)
                metrics.record(f"rel_l2_swap_ffn_hc_pre_{tag}", rel)
                metrics.record(f"worst_row_rel_l2_swap_ffn_hc_pre_{tag}", row)
                print(
                    f"ffn_hc_pre {tag}: rel_l2={rel:.7f} (<= {max_rel}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
                    f"worst_row_rel_l2={row:.6f} (<= {max_row})"
                )
                if rel > max_rel or row > max_row:
                    failures.append(f"ffn_hc_pre {tag}: rel {rel:.5f} / worst row {row:.5f} (stream order or scale?)")

            if finite_shape("ffn_x"):
                pre_check("vs_cpu", seen["ffn_x"], cpu_pre(rctx, hm, gt).float(), FXD_MAX_REL_L2, FXD_MAX_ROW_REL)
            rg = _rotated(gt)
            pre_check(
                "rotated",
                muts["ffn_hc_pre"](rctx, dctx, hm, rg),
                cpu_pre(rctx, hm, rg).float(),
                FROT_MAX_REL_L2,
                FROT_MAX_ROW_REL,
            )

            # Post gates through out: the block's out vs ffn_residual with the CPU gates and the block's own mlp_out.
            res = ref.component(layer, "ffn_residual")
            if "mlp_out" in seen and seen["mlp_out"].numel() * HC_MULT == hm.numel():
                mo = seen["mlp_out"].float().reshape(hm.shape[0], -1)
                pw = res(rctx, hm, cpu_g, mo).float()
                n = pw.shape[0]
                d = seen["out"].float().reshape(n, HC_MULT, -1) - pw.view(n, HC_MULT, -1)
                srel = (d.norm(dim=(0, 2)) / pw.view(n, HC_MULT, -1).norm(dim=(0, 2)).clamp_min(1e-12)).tolist()
                prow = _worst_row_rel(seen["out"], pw, HC_MULT)
                metrics.record("post_stream_rel_l2_swap_out_vs_cpu_gates", max(srel))
                metrics.record("post_worst_row_rel_l2_swap_out_vs_cpu_gates", prow)
                print(
                    f"out vs ffn_residual(h_mid, CPU gates, block mlp_out): stream rel={[round(v, 5) for v in srel]} "
                    f"(<= {POST_MAX_STREAM_REL}) worst (row, stream) rel={prow:.5f} (<= {POST_MAX_ROW_REL})"
                )
                if max(srel) > POST_MAX_STREAM_REL or prow > POST_MAX_ROW_REL:
                    failures.append(f"out vs CPU post gates: stream rel {max(srel):.5f} / worst row {prow:.5f}")
            else:
                failures.append("mlp_out missing or misshapen, cannot check the post gates through out")

            # Block out vs the whole CPU tail from the same device h_mid (routing flips make rows noisy: rel only).
            fnm = ref.component(layer, "ffn_norm")(rctx, cpu_fx)
            rt = ref.component(layer, "router")(rctx, fnm)
            mo_c = ref.component(layer, "moe_combine")(
                rctx, ref.component(layer, "experts")(rctx, fnm, rt), ref.component(layer, "shared_expert")(rctx, fnm)
            )
            tail = res(rctx, hm, cpu_g, mo_c).float()
            trel, trow = _rel(seen["out"], tail), _worst_row_rel(seen["out"], tail, HC_MULT)
            metrics.record("rel_l2_swap_out_vs_cpu_tail", trel)
            metrics.record("worst_row_rel_l2_swap_out_vs_cpu_tail", trow)  # informational (near-tie expert flips)
            print(
                f"out vs CPU tail from the device h_mid: rel_l2={trel:.6f} (<= {TAIL_MAX_REL_L2}) "
                f"worst (row, stream) rel={trow:.5f} (not gated)"
            )
            if trel > TAIL_MAX_REL_L2:
                failures.append(f"out vs CPU tail: rel {trel:.5f} > {TAIL_MAX_REL_L2}")
        else:
            failures.append("ffn_hc: h_mid unusable, cannot check ffn_hc against the CPU step")

    # ffn_norm (the swapped step): vs golden, vs the CPU step on the same device ffn_x, and the module on that ffn_x
    # scaled down (x 0.1, eps dominates most rows) and up (x 30, the RMS reduction dominates), each vs the CPU step.
    def fn_check(tag, got, want, max_rel, ratio, max_row):
        if got.numel() != want.numel() or not torch.isfinite(got.float()).all():
            failures.append(f"ffn_norm {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)} or non-finite")
            return
        rel, rmin, rmax, row = _errors(got, want)
        metrics.record(f"rel_l2_swap_ffn_norm_{tag}", rel)
        metrics.record(f"row_norm_ratio_min_swap_ffn_norm_{tag}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_ffn_norm_{tag}", rmax)
        metrics.record(f"worst_row_rel_l2_swap_ffn_norm_{tag}", row)
        rtxt = "not gated" if ratio is None else f"in {list(ratio)}"
        print(
            f"ffn_norm {tag}: rel_l2={rel:.6f} (<= {max_rel}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"({rtxt}) worst_row_rel_l2={row:.5f} (<= {max_row})"
        )
        if rel > max_rel:
            failures.append(f"ffn_norm {tag}: rel L2 {rel:.5f} > {max_rel}")
        if ratio is not None and not (ratio[0] <= rmin and rmax <= ratio[1]):
            failures.append(f"ffn_norm {tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio)}")
        if row > max_row:
            failures.append(f"ffn_norm {tag}: worst row rel L2 {row:.5f} > {max_row}")

    if finite_shape("ffn_norm"):
        fn_check("vs_golden", seen["ffn_norm"], gl["ffn_norm"].float(), FN_MAX_REL_L2, FN_RATIO, FN_MAX_ROW_REL_L2)
        fx_ok = "ffn_x" in seen and seen["ffn_x"].numel() == gl["ffn_x"].numel()
        if fx_ok and torch.isfinite(seen["ffn_x"].float()).all():
            cpu_fn = ref.component(layer, "ffn_norm")
            fxd = seen["ffn_x"].float().reshape(gl["ffn_x"].shape)
            fn_check(
                "vs_cpu",
                seen["ffn_norm"],
                cpu_fn(rctx, fxd).float(),
                FN_CPU_MAX_REL_L2,
                FN_CPU_RATIO,
                FN_CPU_MAX_ROW_REL_L2,
            )
            xe = (fxd * FN_EPS_SCALE).bfloat16().float()
            fn_check(
                "eps",
                muts["ffn_norm"](rctx, dctx, xe),
                cpu_fn(rctx, xe).float(),
                FN_EPS_MAX_REL_L2,
                None,
                FN_EPS_MAX_ROW_REL_L2,
            )
            xs = (fxd * FN_SYN_SCALE).bfloat16().float()
            fn_check(
                "scaled",
                muts["ffn_norm"](rctx, dctx, xs),
                cpu_fn(rctx, xs).float(),
                FN_SYN_MAX_REL_L2,
                FN_SYN_RATIO,
                FN_SYN_MAX_ROW_REL_L2,
            )
        else:
            failures.append("ffn_norm: ffn_x unusable, cannot check ffn_norm against the CPU step")

    # router (the swapped step): structure and row sums, selection and matched-row weights vs golden, and vs the CPU
    # router on the same device ffn_norm (mean and worst-row overlap, matched-row weights).
    def selection(got, want):
        gs, ws = got != 0, want != 0
        row = (gs & ws).sum(-1).float() / ws.sum(-1).clamp_min(1)
        m = (gs == ws).all(-1)
        mrel = ((got[m] - want[m]).norm() / want[m].norm().clamp_min(1e-12)).item() if m.any() else float("inf")
        return row.mean().item(), row.min().item(), mrel, int(m.sum().item())

    if finite_shape("router"):
        want_rt = gl["router"].float()
        got_rt = seen["router"].float().reshape(want_rt.shape)
        nnz = (got_rt != 0).sum(-1)
        rsum = got_rt.sum(-1) / ROUTE_SCALE
        rmin, rmax = rsum.min().item(), rsum.max().item()
        overlap, _, mrel, nm = selection(got_rt, want_rt)
        metrics.record("router_overlap_swap", overlap)
        metrics.record("router_matched_rel_l2_swap", mrel)
        metrics.record("router_row_sum_min_swap", rmin)
        metrics.record("router_row_sum_max_swap", rmax)
        print(
            f"router: nnz/row {nnz.min().item()}..{nnz.max().item()} (== {TOP_K}) row_sum / {ROUTE_SCALE} = "
            f"[{rmin:.5f}, {rmax:.5f}] (1 +- {RT_MAX_ROW_SUM_ERR})\n"
            f"router vs golden: top-8 overlap={overlap:.5f} (>= {ROUTER_MIN_OVERLAP}) matched_rows={nm}/{want_rt.shape[0]} "
            f"matched_rel_l2={mrel:.5f} (<= {RT_MAX_MATCHED_REL_L2})"
        )
        if not (nnz == TOP_K).all():
            failures.append(f"router: nonzeros per row in [{nnz.min().item()}, {nnz.max().item()}], want {TOP_K}")
        if not (got_rt >= 0).all():
            failures.append("router: negative routing weight")
        if not (1 - RT_MAX_ROW_SUM_ERR <= rmin and rmax <= 1 + RT_MAX_ROW_SUM_ERR):
            failures.append(f"router: row sum / {ROUTE_SCALE} in [{rmin:.5f}, {rmax:.5f}] (route scale or renorm bug)")
        if overlap < ROUTER_MIN_OVERLAP:
            failures.append(f"router selection overlap {overlap:.4f} < {ROUTER_MIN_OVERLAP}")
        if mrel > RT_MAX_MATCHED_REL_L2:
            failures.append(f"router vs golden: matched-row weight rel L2 {mrel:.5f} > {RT_MAX_MATCHED_REL_L2}")
        fn_ok = "ffn_norm" in seen and seen["ffn_norm"].numel() == gl["ffn_norm"].numel()
        if fn_ok and torch.isfinite(seen["ffn_norm"].float()).all():
            fnd = seen["ffn_norm"].float().reshape(gl["ffn_norm"].shape)
            cpu_rt = ref.component(layer, "router")(rctx, fnd).float().reshape(want_rt.shape)
            c_ov, c_row, c_mrel, c_nm = selection(got_rt, cpu_rt)
            metrics.record("router_overlap_swap_vs_cpu", c_ov)
            metrics.record("router_worst_row_overlap_swap_vs_cpu", c_row)
            metrics.record("router_matched_rel_l2_swap_vs_cpu", c_mrel)
            print(
                f"router vs CPU router on the device ffn_norm: overlap={c_ov:.5f} (>= {RT_CPU_MIN_OVERLAP}) worst row "
                f"overlap={c_row:.4f} (>= {RT_CPU_MIN_ROW_OVERLAP}) matched_rows={c_nm}/{want_rt.shape[0]} "
                f"matched_rel_l2={c_mrel:.6f} (<= {RT_CPU_MAX_MATCHED_REL_L2})"
            )
            if c_ov < RT_CPU_MIN_OVERLAP:
                failures.append(
                    f"router vs CPU: overlap {c_ov:.5f} < {RT_CPU_MIN_OVERLAP} (bf16 logits / scores / keys?)"
                )
            if c_row < RT_CPU_MIN_ROW_OVERLAP:
                failures.append(f"router vs CPU: worst row overlap {c_row:.4f} < {RT_CPU_MIN_ROW_OVERLAP} (row order?)")
            if c_mrel > RT_CPU_MAX_MATCHED_REL_L2:
                failures.append(f"router vs CPU: matched-row weight rel L2 {c_mrel:.5f} > {RT_CPU_MAX_MATCHED_REL_L2}")
        else:
            failures.append("router: ffn_norm unusable, cannot check the router against the CPU step")

    # experts (the swapped step): vs the CPU experts on the same device ffn_norm and routing (component limits), the
    # module again on ffn_norm x 2 (the clamp fires), and vs golden (whole tensor; per token on rows routed as in the
    # golden, since a near-tie flip swaps a whole expert in a row).
    def ex_check(tag, got, want, max_rel, ratio, max_row, max_coef, rows=None):
        if got.numel() != want.numel() or not torch.isfinite(got.float()).all():
            failures.append(f"experts {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)} or non-finite")
            return
        if rows is not None and not rows.any():
            failures.append(f"experts {tag}: no row is routed as in the golden (router broken?)")
            return
        got, want = got.float().reshape(want.shape), want.float()
        rel = _rel(got, want)
        g2, w2 = (got, want) if rows is None else (got[rows], want[rows])
        _, rmin, rmax, row = _errors(g2, w2)
        g64, w64 = got.double(), want.double()
        coef = ((g64 * w64).sum() / (w64 * w64).sum()).item()
        metrics.record(f"rel_l2_swap_experts_{tag}", rel)
        metrics.record(f"row_norm_ratio_min_swap_experts_{tag}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_experts_{tag}", rmax)
        metrics.record(f"worst_row_rel_l2_swap_experts_{tag}", row)
        metrics.record(f"coef_swap_experts_{tag}", coef)
        on = "" if rows is None else f" on {int(rows.sum().item())} rows routed as golden"
        ctxt = "not gated" if max_coef is None else f"1 +- {max_coef}"
        print(
            f"experts {tag}: rel_l2={rel:.6f} (<= {max_rel}){on} row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(ratio)}) worst_row_rel_l2={row:.5f} (<= {max_row}) coef={coef:.5f} ({ctxt})"
        )
        if rel > max_rel:
            failures.append(f"experts {tag}: rel L2 {rel:.5f} > {max_rel}")
        if not (ratio[0] <= rmin and rmax <= ratio[1]):
            failures.append(f"experts {tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio)}")
        if row > max_row:
            failures.append(f"experts {tag}: worst row rel L2 {row:.5f} > {max_row} (dropped expert / pair, rows?)")
        if max_coef is not None and not abs(coef - 1) <= max_coef:  # nan (all-zero want) fails too
            failures.append(f"experts {tag}: global scale {coef:.5f} not within {max_coef} of 1")

    if finite_shape("experts_out"):
        fn_ok = "ffn_norm" in seen and seen["ffn_norm"].numel() == gl["ffn_norm"].numel()
        rt_ok = "router" in seen and seen["router"].numel() == gl["router"].numel()
        if fn_ok and rt_ok and torch.isfinite(seen["ffn_norm"].float()).all() and torch.isfinite(seen["router"]).all():
            cpu_ex = ref.component(layer, "experts")
            fnd = seen["ffn_norm"].float().reshape(gl["ffn_norm"].shape)
            rtd = seen["router"].float().reshape(gl["router"].shape)
            ex_check(
                "vs_cpu",
                seen["experts_out"],
                cpu_ex(rctx, fnd, rtd).float(),
                EX_MAX_REL_L2,
                EX_RATIO,
                EX_MAX_ROW_REL,
                EX_MAX_COEF_ERR,
            )
            xs = (fnd * EX_SYN_SCALE).bfloat16().float()
            ex_check(
                f"x{EX_SYN_SCALE:g}",
                muts["experts"](rctx, dctx, xs, rtd),
                cpu_ex(rctx, xs, rtd).float(),
                EX_MAX_REL_L2,
                EX_RATIO,
                EX_MAX_ROW_REL,
                EX_MAX_COEF_ERR,
            )
            same = ((rtd != 0) == (gl["router"].float() != 0)).all(-1)
            ex_check(
                "vs_golden",
                seen["experts_out"],
                gl["experts_out"].float(),
                EX_G_MAX_REL_L2,
                EX_G_RATIO,
                EX_G_MAX_ROW_REL,
                EX_G_MAX_COEF_ERR,
                rows=same,
            )
        else:
            failures.append("experts: ffn_norm / router unusable, cannot check the experts against the CPU step")

    # shared_expert (the swapped step): vs the CPU step on the same device ffn_norm (component limits + global scale),
    # the module again on ffn_norm x 2 (the clamp would fire there), and vs golden (backstop).
    def se_check(tag, got, want, max_rel, ratio, max_row, max_coef):
        if got.numel() != want.numel() or not torch.isfinite(got.float()).all():
            failures.append(f"shared_expert {tag}: shape {tuple(got.shape)} vs {tuple(want.shape)} or non-finite")
            return
        got, want = got.float().reshape(want.shape), want.float()
        rel, rmin, rmax, row = _errors(got, want)
        g64, w64 = got.double(), want.double()
        coef = ((g64 * w64).sum() / (w64 * w64).sum().clamp_min(1e-300)).item()
        metrics.record(f"rel_l2_swap_shared_out_{tag}", rel)
        metrics.record(f"row_norm_ratio_min_swap_shared_out_{tag}", rmin)
        metrics.record(f"row_norm_ratio_max_swap_shared_out_{tag}", rmax)
        metrics.record(f"worst_row_rel_l2_swap_shared_out_{tag}", row)
        metrics.record(f"coef_swap_shared_out_{tag}", coef)
        print(
            f"shared_expert {tag}: rel_l2={rel:.6f} (<= {max_rel}) row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(ratio)}) worst_row_rel_l2={row:.5f} (<= {max_row}) coef={coef:.6f} (1 +- {max_coef})"
        )
        if rel > max_rel:
            failures.append(f"shared_expert {tag}: rel L2 {rel:.5f} > {max_rel}")
        if not (ratio[0] <= rmin and rmax <= ratio[1]):
            failures.append(f"shared_expert {tag}: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(ratio)}")
        if row > max_row:
            failures.append(f"shared_expert {tag}: worst row rel L2 {row:.5f} > {max_row} (dropped / misplaced rows?)")
        if not abs(coef - 1) <= max_coef:
            failures.append(f"shared_expert {tag}: global scale {coef:.6f} not within {max_coef} of 1")

    if finite_shape("shared_out"):
        fn_ok = "ffn_norm" in seen and seen["ffn_norm"].numel() == gl["ffn_norm"].numel()
        if fn_ok and torch.isfinite(seen["ffn_norm"].float()).all():
            cpu_se = ref.component(layer, "shared_expert")
            fnd = seen["ffn_norm"].float().reshape(gl["ffn_norm"].shape)
            se_check(
                "vs_cpu",
                seen["shared_out"],
                cpu_se(rctx, fnd).float(),
                SE_MAX_REL_L2,
                SE_RATIO,
                SE_MAX_ROW_REL,
                SE_MAX_COEF_ERR,
            )
            xs = (fnd * SE_SYN_SCALE).bfloat16().float()
            se_check(
                f"x{SE_SYN_SCALE:g}",
                muts["shared_expert"](rctx, dctx, xs),
                cpu_se(rctx, xs).float(),
                SE_SYN_MAX_REL_L2,
                SE_SYN_RATIO,
                SE_SYN_MAX_ROW_REL,
                SE_MAX_COEF_ERR,
            )
        else:
            failures.append("shared_expert: ffn_norm unusable, cannot check the shared expert against the CPU step")
        se_check(
            "vs_golden",
            seen["shared_out"],
            gl["shared_out"].float(),
            SE_G_MAX_REL_L2,
            SE_G_RATIO,
            SE_G_MAX_ROW_REL,
            SE_G_MAX_COEF_ERR,
        )

    # moe_combine (the swapped step): vs the exact fp32 sum of the block's own device addends (the module's inputs),
    # per addend coefficient / error on those, the module again on (experts, -shared) and (experts, 0), and vs golden.
    def mc_ok(n):
        return n in seen and seen[n].numel() == gl[n].numel() and torch.isfinite(seen[n].float()).all()

    if finite_shape("mlp_out"):
        n = gl["mlp_out"].shape[0]
        mo = seen["mlp_out"].float().reshape(n, -1)
        if mc_ok("experts_out") and mc_ok("shared_out"):
            ex = seen["experts_out"].float().reshape(n, -1)
            sh = seen["shared_out"].float().reshape(n, -1)
            exact = ref.component(layer, "moe_combine")(rctx, ex, sh).float().reshape(n, -1)
            rel, rmin, rmax, row = _errors(mo, exact)
            metrics.record("rel_l2_swap_mlp_out_vs_cpu", rel)
            metrics.record("row_norm_ratio_min_swap_mlp_out_vs_cpu", rmin)
            metrics.record("row_norm_ratio_max_swap_mlp_out_vs_cpu", rmax)
            metrics.record("worst_row_rel_l2_swap_mlp_out_vs_cpu", row)
            print(
                f"mlp_out vs CPU moe_combine on the device addends: rel_l2={rel:.7f} (<= {MC_MAX_REL_L2}) row norm "
                f"ratio=[{rmin:.6f}, {rmax:.6f}] (in {list(MC_RATIO)}) worst_row_rel_l2={row:.6f} (<= {MC_MAX_ROW_REL})"
            )
            if rel > MC_MAX_REL_L2:
                failures.append(f"mlp_out vs CPU: rel L2 {rel:.5f} > {MC_MAX_REL_L2}")
            if not (MC_RATIO[0] <= rmin and rmax <= MC_RATIO[1]):
                failures.append(f"mlp_out vs CPU: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(MC_RATIO)}")
            if row > MC_MAX_ROW_REL:
                failures.append(f"mlp_out vs CPU: worst row rel L2 {row:.5f} > {MC_MAX_ROW_REL} (misplaced rows?)")

            # Each addend on its own (float64): delta = mlp_out - other addend vs this addend.
            named = {"experts_out": ex.double(), "shared_out": sh.double()}
            g64 = mo.double()
            for name, t in named.items():
                other = sum(v for k, v in named.items() if k != name)
                d = g64 - other
                err = d - t
                tn = t.norm().clamp_min(1e-300)
                coef = ((d * t).sum() / (tn * tn)).item()
                arel = (err.norm() / tn).item()
                arow = (err.norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-30)).max().item()
                tag = name.replace("_out", "")
                metrics.record(f"add_coef_{tag}_swap_mlp_out", coef)
                metrics.record(f"add_rel_l2_{tag}_swap_mlp_out", arel)
                metrics.record(f"add_worst_row_{tag}_swap_mlp_out", arow)
                print(
                    f"mlp_out addend {name}: coef={coef:.6f} (1 +- {MC_MAX_COEF_DEV}) rel={arel:.6f} "
                    f"(<= {MC_MAX_ADD_REL[name]}) worst row={arow:.5f} (<= {MC_MAX_ADD_ROW_REL})"
                )
                if not abs(coef - 1) <= MC_MAX_COEF_DEV:
                    failures.append(f"mlp_out: {name} enters with coefficient {coef:.5f} (scaled addend?)")
                if arel > MC_MAX_ADD_REL[name]:
                    failures.append(f"mlp_out: {name} addend error {arel:.5f} > {MC_MAX_ADD_REL[name]}")
                if arow > MC_MAX_ADD_ROW_REL:
                    failures.append(f"mlp_out: {name} worst row addend error {arow:.4f} > {MC_MAX_ADD_ROW_REL}")

            # Probes: the module must add its own inputs.
            for label, a, b in (("experts - shared", ex, -sh), ("experts + 0", ex, torch.zeros_like(sh))):
                pe = a + b
                p = muts["moe_combine"](rctx, dctx, a, b)
                if p.numel() != pe.numel() or not torch.isfinite(p.float()).all():
                    failures.append(f"moe_combine probe {label}: shape {tuple(p.shape)} or non-finite")
                    continue
                pg = p.float().reshape(n, -1)
                prel = ((pg - pe).norm() / pe.norm().clamp_min(1e-12)).item()
                prow = _worst_row_rel(pg, pe)
                ptag = label.replace(" ", "").replace("-", "minus").replace("+", "plus")
                metrics.record(f"probe_rel_l2_{ptag}_swap_mlp_out", prel)
                metrics.record(f"probe_worst_row_rel_l2_{ptag}_swap_mlp_out", prow)
                print(
                    f"moe_combine probe {label}: rel={prel:.7f} (<= {MC_PROBE_MAX_REL}) worst row={prow:.6f} "
                    f"(<= {MC_PROBE_MAX_ROW_REL})"
                )
                if prel > MC_PROBE_MAX_REL or prow > MC_PROBE_MAX_ROW_REL:
                    failures.append(
                        f"moe_combine probe {label}: rel {prel:.5f} / worst row {prow:.5f} (module ignores its inputs?)"
                    )
        else:
            failures.append("moe_combine: experts_out / shared_out unusable, cannot check mlp_out against the CPU step")

        # vs golden (backstop): whole tensor, global scale, and per row on the rows routed as in the golden.
        want = gl["mlp_out"].float().reshape(n, -1)
        grel = _rel(mo, want)
        gcoef = ((mo.double() * want.double()).sum() / (want.double() ** 2).sum().clamp_min(1e-300)).item()
        rt_ok = "router" in seen and seen["router"].numel() == gl["router"].numel()
        if rt_ok:
            same = ((seen["router"].float().reshape(gl["router"].shape) != 0) == (gl["router"].float() != 0)).all(-1)
        else:
            same = torch.ones(n, dtype=torch.bool)
        _, rmin, rmax, row = _errors(mo[same], want[same])
        metrics.record("rel_l2_swap_mlp_out_vs_golden", grel)
        metrics.record("coef_swap_mlp_out_vs_golden", gcoef)
        metrics.record("row_norm_ratio_min_swap_mlp_out_vs_golden", rmin)
        metrics.record("row_norm_ratio_max_swap_mlp_out_vs_golden", rmax)
        metrics.record("worst_row_rel_l2_swap_mlp_out_vs_golden", row)
        print(
            f"mlp_out vs golden: rel_l2={grel:.6f} (<= {MC_G_MAX_REL_L2}) coef={gcoef:.6f} (1 +- {MC_G_MAX_COEF_ERR}) "
            f"on {int(same.sum().item())} rows routed as golden: row norm ratio=[{rmin:.5f}, {rmax:.5f}] "
            f"(in {list(MC_G_RATIO)}) worst_row_rel_l2={row:.5f} (<= {MC_G_MAX_ROW_REL})"
        )
        if grel > MC_G_MAX_REL_L2:
            failures.append(f"mlp_out vs golden: rel L2 {grel:.5f} > {MC_G_MAX_REL_L2}")
        if not abs(gcoef - 1) <= MC_G_MAX_COEF_ERR:
            failures.append(f"mlp_out vs golden: global scale {gcoef:.5f} not within {MC_G_MAX_COEF_ERR} of 1")
        if not (MC_G_RATIO[0] <= rmin and rmax <= MC_G_RATIO[1]):
            failures.append(f"mlp_out vs golden: row norm ratio [{rmin:.5f}, {rmax:.5f}] outside {list(MC_G_RATIO)}")
        if row > MC_G_MAX_ROW_REL:
            failures.append(f"mlp_out vs golden: worst row rel L2 {row:.5f} > {MC_G_MAX_ROW_REL}")

    # Block out.
    out_rel = _rel(seen["out"], gl["out"])
    out_row = _worst_row_rel(seen["out"], gl["out"], HC_MULT)
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("worst_row_rel_l2_swap_out", out_row)  # informational only (near-tie expert flips)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2}) worst (row, stream) rel={out_row:.5f} (not gated)")
    if not (torch.isfinite(seen["out"]).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)
