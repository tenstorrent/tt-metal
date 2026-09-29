# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Swap test 10: block type moe_shared (layer 2) with ffn_norm swapped in last.

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

Reviewed (S.moe_shared.10.test.1): every swap-09 check is kept at its limits; the ffn_norm checks are at the end of
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

    # Router selection (top-8 of 256; the dense routing weights are nonzero on the selected experts).
    want_sel = gl["router"] != 0
    got_sel = seen["router"].reshape(want_sel.shape) != 0
    overlap = ((got_sel & want_sel).sum(-1).float() / want_sel.sum(-1).clamp_min(1)).mean().item()
    metrics.record("router_overlap_swap", overlap)
    print(f"router top-8 overlap={overlap:.5f} (>= {ROUTER_MIN_OVERLAP})")
    if overlap < ROUTER_MIN_OVERLAP:
        failures.append(f"router selection overlap {overlap:.4f} < {ROUTER_MIN_OVERLAP}")

    # Block out.
    out_rel = _rel(seen["out"], gl["out"])
    out_row = _worst_row_rel(seen["out"], gl["out"], HC_MULT)
    metrics.record("rel_l2_swap_out", out_rel)
    metrics.record("worst_row_rel_l2_swap_out", out_row)  # informational only (near-tie expert flips)
    print(f"rel_l2_swap_out={out_rel:.6f} (<= {OUT_MAX_REL_L2}) worst (row, stream) rel={out_row:.5f} (not gated)")
    if not (torch.isfinite(seen["out"]).all() and out_rel <= OUT_MAX_REL_L2):
        failures.append(f"block out rel L2 {out_rel:.4f} > {OUT_MAX_REL_L2} (or non-finite)")
    assert not failures, "; ".join(failures)
