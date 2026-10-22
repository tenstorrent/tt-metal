# Matmul sweep (accuracy)

| role | shape | x | default ms | best candidate | best ms | speedup | full-model saving ms | PCC |
|---|---|---|---|---|---|---|---|---|
| qkv | 768x2304 | 24 | 5.994 | `2d[11x10,obh=11,ibw=12,sb=1x7]` | 2.727 | 2.20x | 78.4 | 0.99989 |
| gate_sigmoid | 768x768 +sigmoid | 30 | 4.576 | `2d[8x10,obh=11,ibw=12,sb=1x3]` | 1.477 | 3.10x | 93.0 | 0.99989 |
| proj768 | 768x768 | 36 | 3.526 | `minimal[8,8,8,sb=1x8]` | 1.197 | 2.95x | 83.9 | 0.99996 |
| up_silu | 768x2048 +silu | 24 | 8.056 | `2d[11x10,obh=11,ibw=12,sb=1x6]` | 2.924 | 2.75x | 123.2 | 0.99984 |
| mlp_gate | 768x2048 | 24 | 5.312 | `2d[11x10,obh=11,ibw=12,sb=1x6]` | 2.412 | 2.20x | 69.6 | 0.99989 |
| down | 2048x768 | 24 | 8.251 | `minimal[8,8,8,sb=1x8]` | 1.805 | 4.57x | 154.7 | 0.99996 |
| embed_hidden | 64x3072 +bias +silu | 1 | 6.546 | `2d[11x10,obh=11,ibw=2,sb=1x3]` | 3.134 | 2.09x | 3.4 | 0.99998 |
| embed_out | 3072x768 +bias | 1 | 11.722 | `minimal[tt_dit heuristic]` | 2.557 | 4.58x | 9.2 | 0.99996 |
| embed_res | 64x768 +bias | 1 | 1.091 | `minimal[tt_dit heuristic]` | 0.740 | 1.47x | 0.4 | 0.99999 |

Default ms for activation roles includes the separate unary op.

## qkv

| candidate | ms | PCC / error |
|---|---|---|
| `2d[11x10,obh=11,ibw=12,sb=1x7]` | 2.727 | 0.99989 |
| `2d[11x10,obh=7,ibw=12,sb=1x7]` | 2.867 | 0.99989 |
| `2d[11x10,obh=14,ibw=8,sb=1x7]` | 2.886 | 0.99993 |
| `2d[11x10,obh=11,ibw=8,sb=1x7]` | 2.941 | 0.99993 |
| `2d[11x10,obh=14,ibw=6,sb=1x7]` | 3.070 | 0.99994 |
| `2d[11x10,obh=7,ibw=8,sb=1x7]` | 3.088 | 0.99993 |
| `2d[11x10,obh=4,ibw=12,sb=1x7]` | 3.112 | 0.99989 |
| `2d[11x10,obh=11,ibw=6,sb=1x7]` | 3.118 | 0.99994 |
| `minimal[8,8,4,sb=2x4]` | 3.174 | 0.99996 |
| `2d[11x10,obh=14,ibw=4,sb=1x7]` | 3.205 | 0.99996 |
| `2d[11x10,obh=7,ibw=6,sb=1x7]` | 3.222 | 0.99994 |
| `2d[11x10,obh=11,ibw=4,sb=1x7]` | 3.230 | 0.99996 |
| `minimal[16,4,4,sb=2x4]` | 3.236 | 0.99998 |
| `2d[11x10,obh=14,ibw=3,sb=1x7]` | 3.264 | 0.99996 |
| `2d[11x10,obh=11,ibw=3,sb=1x7]` | 3.298 | 0.99996 |
| `2d[11x10,obh=7,ibw=4,sb=1x7]` | 3.303 | 0.99996 |
| `2d[11x10,obh=14,ibw=2,sb=1x7]` | 3.338 | 0.99997 |
| `2d[11x10,obh=4,ibw=8,sb=1x7]` | 3.349 | 0.99993 |
| `2d[11x10,obh=11,ibw=2,sb=1x7]` | 3.371 | 0.99997 |
| `2d[11x10,obh=7,ibw=3,sb=1x7]` | 3.384 | 0.99996 |
| `2d[11x10,obh=4,ibw=6,sb=1x7]` | 3.461 | 0.99994 |
| `2d[11x10,obh=7,ibw=2,sb=1x7]` | 3.466 | 0.99997 |
| `minimal[8,4,8,sb=1x8]` | 3.473 | 0.99998 |
| `minimal[8,8,8,sb=1x8]` | 3.546 | 0.99996 |
| `2d[11x10,obh=4,ibw=3,sb=1x7]` | 3.572 | 0.99996 |
| `2d[11x10,obh=4,ibw=4,sb=1x7]` | 3.585 | 0.99996 |
| `2d[11x10,obh=14,ibw=1,sb=1x7]` | 3.597 | 0.99997 |
| `minimal[default]` | 3.608 | 0.99996 |
| `2d[11x10,obh=11,ibw=1,sb=1x7]` | 3.671 | 0.99997 |
| `2d[11x10,obh=4,ibw=2,sb=1x7]` | 3.692 | 0.99997 |
| `2d[11x10,obh=7,ibw=1,sb=1x7]` | 3.887 | 0.99997 |
| `2d[11x10,obh=4,ibw=1,sb=1x7]` | 4.299 | 0.99997 |
| `minimal[4,8,8,sb=1x8]` | 4.554 | 0.99996 |
| `1d_full_n[obh=2,ibw=2,sb=1x8]` | 4.627 | 0.99997 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 4.841 | 0.99997 |
| `default` | 5.994 | 0.99997 |
| `1d_full_n[obh=1,ibw=3,sb=1x8]` | 6.366 | 0.99996 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 6.421 | 0.99997 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 6.568 | 0.99997 |
| `core_grid` | 6.775 | 0.99997 |
| `minimal[tt_dit heuristic]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| `minimal[16,8,8,sb=1x8]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |

## gate_sigmoid

| candidate | ms | PCC / error |
|---|---|---|
| `2d[8x10,obh=11,ibw=12,sb=1x3]` | 1.477 | 0.99989 |
| `1d_full_n[obh=4,ibw=8,sb=1x8]` | 1.509 | 0.99993 |
| `2d[8x10,obh=14,ibw=12,sb=7x1]` | 1.538 | 0.99989 |
| `1d_full_n[obh=4,ibw=6,sb=1x8]` | 1.558 | 0.99994 |
| `2d[8x10,obh=11,ibw=8,sb=1x3]` | 1.590 | 0.99993 |
| `2d[8x10,obh=14,ibw=8,sb=7x1]` | 1.630 | 0.99993 |
| `1d_full_n[obh=4,ibw=3,sb=1x8]` | 1.699 | 0.99996 |
| `minimal[8,8,4,sb=2x4]` | 1.710 | 0.99996 |
| `1d_full_n[obh=7,ibw=3,sb=1x8]` | 1.713 | 0.99996 |
| `2d[8x10,obh=7,ibw=12,sb=7x1]` | 1.718 | 0.99989 |
| `2d[8x10,obh=11,ibw=6,sb=1x3]` | 1.738 | 0.99994 |
| `2d[8x10,obh=14,ibw=6,sb=7x1]` | 1.763 | 0.99994 |
| `1d_full_n[obh=7,ibw=4,sb=1x8]` | 1.763 | 0.99996 |
| `1d_full_n[obh=2,ibw=8,sb=1x8]` | 1.770 | 0.99993 |
| `1d_full_n[obh=2,ibw=6,sb=1x8]` | 1.776 | 0.99994 |
| `1d_full_n[obh=7,ibw=2,sb=1x8]` | 1.809 | 0.99997 |
| `1d_full_n[obh=4,ibw=4,sb=1x8]` | 1.815 | 0.99996 |
| `1d_full_n[obh=4,ibw=2,sb=1x8]` | 1.830 | 0.99997 |
| `1d_full_n[obh=2,ibw=4,sb=1x8]` | 1.842 | 0.99996 |
| `2d[8x10,obh=7,ibw=8,sb=7x1]` | 1.843 | 0.99993 |
| `1d_full_n[obh=2,ibw=3,sb=1x8]` | 1.859 | 0.99996 |
| `minimal[tt_dit heuristic]` | 1.889 | 0.99996 |
| `minimal[16,4,4,sb=2x4]` | 1.908 | 0.99997 |
| `2d[8x10,obh=4,ibw=12,sb=2x3]` | 1.951 | 0.99989 |
| `1d_full_n[obh=2,ibw=2,sb=1x8]` | 1.960 | 0.99997 |
| `2d[8x10,obh=11,ibw=4,sb=1x3]` | 1.991 | 0.99996 |
| `2d[8x10,obh=7,ibw=6,sb=7x1]` | 1.998 | 0.99994 |
| `2d[8x10,obh=14,ibw=3,sb=7x1]` | 2.002 | 0.99996 |
| `2d[8x10,obh=14,ibw=4,sb=7x1]` | 2.004 | 0.99996 |
| `2d[8x10,obh=11,ibw=3,sb=1x3]` | 2.028 | 0.99996 |
| `1d_full_n[obh=7,ibw=1,sb=1x8]` | 2.047 | 0.99997 |
| `1d_full_n[obh=4,ibw=1,sb=1x8]` | 2.068 | 0.99997 |
| `2d[8x10,obh=4,ibw=8,sb=2x3]` | 2.108 | 0.99993 |
| `2d[8x10,obh=14,ibw=2,sb=7x1]` | 2.138 | 0.99997 |
| `2d[8x10,obh=11,ibw=2,sb=1x3]` | 2.176 | 0.99997 |
| `2d[8x10,obh=7,ibw=4,sb=7x1]` | 2.214 | 0.99996 |
| `2d[8x10,obh=4,ibw=6,sb=2x3]` | 2.243 | 0.99994 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 2.261 | 0.99997 |
| `2d[8x10,obh=7,ibw=3,sb=7x1]` | 2.288 | 0.99996 |
| `2d[8x10,obh=11,ibw=1,sb=1x3]` | 2.369 | 0.99997 |
| `2d[8x10,obh=14,ibw=1,sb=7x1]` | 2.378 | 0.99997 |
| `2d[8x10,obh=4,ibw=4,sb=2x3]` | 2.407 | 0.99996 |
| `2d[8x10,obh=7,ibw=2,sb=7x1]` | 2.420 | 0.99997 |
| `2d[8x10,obh=4,ibw=3,sb=2x3]` | 2.465 | 0.99996 |
| `1d_full_n[obh=1,ibw=4,sb=1x8]` | 2.524 | 0.99996 |
| `1d_full_n[obh=1,ibw=8,sb=1x8]` | 2.526 | 0.99993 |
| `1d_full_n[obh=1,ibw=6,sb=1x8]` | 2.529 | 0.99994 |
| `1d_full_n[obh=1,ibw=3,sb=1x8]` | 2.581 | 0.99996 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 2.663 | 0.99997 |
| `2d[8x10,obh=7,ibw=1,sb=7x1]` | 2.716 | 0.99997 |
| `2d[8x10,obh=4,ibw=2,sb=2x3]` | 2.721 | 0.99997 |
| `minimal[8,8,8,sb=1x8]` | 2.764 | 0.99996 |
| `minimal[4,8,8,sb=1x8]` | 2.767 | 0.99996 |
| `minimal[8,4,8,sb=1x8]` | 2.879 | 0.99997 |
| `minimal[default]` | 2.925 | 0.99996 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 3.031 | 0.99997 |
| `2d[8x10,obh=4,ibw=1,sb=2x3]` | 3.183 | 0.99997 |
| `core_grid` | 4.171 | 0.99997 |
| `default` | 4.576 | 0.99997 |
| `minimal[16,8,8,sb=1x8]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |

## proj768

| candidate | ms | PCC / error |
|---|---|---|
| `minimal[8,8,8,sb=1x8]` | 1.197 | 0.99996 |
| `minimal[tt_dit heuristic]` | 1.202 | 0.99996 |
| `minimal[8,8,4,sb=2x4]` | 1.210 | 0.99996 |
| `2d[8x10,obh=11,ibw=12,sb=1x3]` | 1.213 | 0.99989 |
| `minimal[default]` | 1.236 | 0.99996 |
| `minimal[16,4,4,sb=2x4]` | 1.257 | 0.99998 |
| `2d[8x10,obh=14,ibw=12,sb=7x1]` | 1.258 | 0.99989 |
| `minimal[4,8,8,sb=1x8]` | 1.270 | 0.99996 |
| `2d[8x10,obh=11,ibw=8,sb=1x3]` | 1.354 | 0.99993 |
| `2d[8x10,obh=14,ibw=8,sb=7x1]` | 1.360 | 0.99993 |
| `2d[8x10,obh=7,ibw=12,sb=7x1]` | 1.375 | 0.99989 |
| `minimal[8,4,8,sb=1x8]` | 1.403 | 0.99998 |
| `2d[8x10,obh=4,ibw=12,sb=2x3]` | 1.441 | 0.99989 |
| `1d_full_n[obh=4,ibw=8,sb=1x8]` | 1.452 | 0.99993 |
| `2d[8x10,obh=7,ibw=8,sb=7x1]` | 1.471 | 0.99993 |
| `1d_full_n[obh=4,ibw=6,sb=1x8]` | 1.477 | 0.99994 |
| `2d[8x10,obh=11,ibw=6,sb=1x3]` | 1.486 | 0.99994 |
| `2d[8x10,obh=14,ibw=6,sb=7x1]` | 1.515 | 0.99994 |
| `2d[8x10,obh=11,ibw=4,sb=1x3]` | 1.580 | 0.99996 |
| `2d[8x10,obh=4,ibw=8,sb=2x3]` | 1.597 | 0.99993 |
| `2d[8x10,obh=14,ibw=4,sb=7x1]` | 1.603 | 0.99996 |
| `2d[8x10,obh=7,ibw=6,sb=7x1]` | 1.611 | 0.99994 |
| `1d_full_n[obh=2,ibw=8,sb=1x8]` | 1.651 | 0.99993 |
| `1d_full_n[obh=4,ibw=3,sb=1x8]` | 1.653 | 0.99996 |
| `2d[8x10,obh=14,ibw=3,sb=7x1]` | 1.660 | 0.99996 |
| `1d_full_n[obh=2,ibw=6,sb=1x8]` | 1.663 | 0.99994 |
| `2d[8x10,obh=11,ibw=3,sb=1x3]` | 1.674 | 0.99996 |
| `1d_full_n[obh=7,ibw=3,sb=1x8]` | 1.684 | 0.99996 |
| `1d_full_n[obh=2,ibw=4,sb=1x8]` | 1.718 | 0.99996 |
| `2d[8x10,obh=7,ibw=4,sb=7x1]` | 1.722 | 0.99996 |
| `1d_full_n[obh=7,ibw=4,sb=1x8]` | 1.723 | 0.99996 |
| `2d[8x10,obh=4,ibw=6,sb=2x3]` | 1.733 | 0.99994 |
| `1d_full_n[obh=2,ibw=3,sb=1x8]` | 1.746 | 0.99996 |
| `1d_full_n[obh=7,ibw=2,sb=1x8]` | 1.761 | 0.99997 |
| `1d_full_n[obh=4,ibw=4,sb=1x8]` | 1.765 | 0.99996 |
| `1d_full_n[obh=4,ibw=2,sb=1x8]` | 1.781 | 0.99997 |
| `2d[8x10,obh=14,ibw=2,sb=7x1]` | 1.830 | 0.99997 |
| `2d[8x10,obh=11,ibw=2,sb=1x3]` | 1.846 | 0.99997 |
| `2d[8x10,obh=7,ibw=3,sb=7x1]` | 1.849 | 0.99996 |
| `1d_full_n[obh=2,ibw=2,sb=1x8]` | 1.854 | 0.99997 |
| `2d[8x10,obh=4,ibw=4,sb=2x3]` | 1.910 | 0.99996 |
| `2d[8x10,obh=4,ibw=3,sb=2x3]` | 1.948 | 0.99996 |
| `2d[8x10,obh=7,ibw=2,sb=7x1]` | 1.991 | 0.99997 |
| `1d_full_n[obh=7,ibw=1,sb=1x8]` | 2.027 | 0.99997 |
| `1d_full_n[obh=4,ibw=1,sb=1x8]` | 2.030 | 0.99997 |
| `2d[8x10,obh=14,ibw=1,sb=7x1]` | 2.070 | 0.99997 |
| `2d[8x10,obh=11,ibw=1,sb=1x3]` | 2.091 | 0.99997 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 2.153 | 0.99997 |
| `2d[8x10,obh=4,ibw=2,sb=2x3]` | 2.206 | 0.99997 |
| `1d_full_n[obh=1,ibw=8,sb=1x8]` | 2.290 | 0.99993 |
| `1d_full_n[obh=1,ibw=6,sb=1x8]` | 2.304 | 0.99994 |
| `2d[8x10,obh=7,ibw=1,sb=7x1]` | 2.312 | 0.99997 |
| `1d_full_n[obh=1,ibw=4,sb=1x8]` | 2.321 | 0.99996 |
| `1d_full_n[obh=1,ibw=3,sb=1x8]` | 2.416 | 0.99996 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 2.425 | 0.99997 |
| `2d[8x10,obh=4,ibw=1,sb=2x3]` | 2.673 | 0.99997 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 2.816 | 0.99997 |
| `default` | 3.526 | 0.99997 |
| `core_grid` | 4.000 | 0.99997 |
| `minimal[16,8,8,sb=1x8]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |

## up_silu

| candidate | ms | PCC / error |
|---|---|---|
| `2d[11x10,obh=11,ibw=12,sb=1x6]` | 2.924 | 0.99984 |
| `2d[11x10,obh=11,ibw=8,sb=1x6]` | 3.013 | 0.99989 |
| `2d[11x10,obh=11,ibw=6,sb=1x6]` | 3.095 | 0.99992 |
| `2d[11x10,obh=11,ibw=4,sb=1x6]` | 3.207 | 0.99994 |
| `2d[11x10,obh=14,ibw=8,sb=7x1]` | 3.220 | 0.99989 |
| `2d[11x10,obh=14,ibw=4,sb=7x1]` | 3.220 | 0.99994 |
| `2d[11x10,obh=11,ibw=3,sb=1x6]` | 3.313 | 0.99995 |
| `2d[11x10,obh=14,ibw=3,sb=7x1]` | 3.338 | 0.99995 |
| `2d[11x10,obh=14,ibw=6,sb=7x1]` | 3.355 | 0.99992 |
| `2d[11x10,obh=11,ibw=2,sb=1x6]` | 3.392 | 0.99995 |
| `2d[11x10,obh=7,ibw=12,sb=7x1]` | 3.403 | 0.99984 |
| `2d[11x10,obh=7,ibw=8,sb=7x1]` | 3.555 | 0.99989 |
| `2d[11x10,obh=14,ibw=2,sb=7x1]` | 3.582 | 0.99995 |
| `2d[11x10,obh=7,ibw=4,sb=7x1]` | 3.599 | 0.99994 |
| `minimal[8,8,8,sb=1x8]` | 3.600 | 0.99995 |
| `2d[11x10,obh=4,ibw=12,sb=4x2]` | 3.615 | 0.99984 |
| `minimal[8,8,4,sb=2x4]` | 3.650 | 0.99995 |
| `2d[11x10,obh=4,ibw=8,sb=4x2]` | 3.679 | 0.99989 |
| `2d[11x10,obh=7,ibw=6,sb=7x1]` | 3.682 | 0.99992 |
| `2d[11x10,obh=11,ibw=1,sb=1x6]` | 3.747 | 0.99995 |
| `minimal[default]` | 3.770 | 0.99995 |
| `2d[11x10,obh=4,ibw=6,sb=4x2]` | 3.778 | 0.99992 |
| `2d[11x10,obh=7,ibw=3,sb=7x1]` | 3.851 | 0.99995 |
| `2d[11x10,obh=4,ibw=4,sb=4x2]` | 3.929 | 0.99994 |
| `2d[11x10,obh=14,ibw=1,sb=7x1]` | 3.933 | 0.99995 |
| `minimal[8,4,8,sb=1x8]` | 3.936 | 0.99997 |
| `minimal[4,8,8,sb=1x8]` | 3.968 | 0.99995 |
| `2d[11x10,obh=7,ibw=2,sb=7x1]` | 4.004 | 0.99995 |
| `minimal[16,4,4,sb=2x4]` | 4.078 | 0.99997 |
| `2d[11x10,obh=4,ibw=3,sb=4x2]` | 4.103 | 0.99995 |
| `2d[11x10,obh=4,ibw=2,sb=4x2]` | 4.151 | 0.99995 |
| `1d_full_n[obh=2,ibw=2,sb=1x8]` | 4.294 | 0.99995 |
| `2d[11x10,obh=7,ibw=1,sb=7x1]` | 4.416 | 0.99995 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 4.517 | 0.99995 |
| `core_grid` | 4.530 | 0.99995 |
| `2d[11x10,obh=4,ibw=1,sb=4x2]` | 4.734 | 0.99995 |
| `1d_full_n[obh=1,ibw=3,sb=1x8]` | 6.003 | 0.99995 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 6.054 | 0.99995 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 6.193 | 0.99995 |
| `default` | 8.056 | 0.99995 |
| `minimal[tt_dit heuristic]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| `minimal[16,8,8,sb=1x8]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |

## mlp_gate

| candidate | ms | PCC / error |
|---|---|---|
| `2d[11x10,obh=11,ibw=12,sb=1x6]` | 2.412 | 0.99989 |
| `2d[11x10,obh=11,ibw=8,sb=1x6]` | 2.646 | 0.99993 |
| `2d[11x10,obh=14,ibw=8,sb=7x1]` | 2.694 | 0.99993 |
| `2d[11x10,obh=7,ibw=12,sb=7x1]` | 2.756 | 0.99989 |
| `2d[11x10,obh=11,ibw=6,sb=1x6]` | 2.781 | 0.99994 |
| `2d[11x10,obh=11,ibw=4,sb=1x6]` | 2.834 | 0.99996 |
| `2d[11x10,obh=14,ibw=6,sb=7x1]` | 2.860 | 0.99994 |
| `2d[11x10,obh=7,ibw=8,sb=7x1]` | 2.876 | 0.99993 |
| `2d[11x10,obh=14,ibw=4,sb=7x1]` | 2.913 | 0.99996 |
| `minimal[8,8,4,sb=2x4]` | 2.917 | 0.99996 |
| `2d[11x10,obh=4,ibw=12,sb=4x2]` | 2.924 | 0.99989 |
| `2d[11x10,obh=11,ibw=3,sb=1x6]` | 2.957 | 0.99996 |
| `2d[11x10,obh=7,ibw=6,sb=7x1]` | 3.025 | 0.99994 |
| `2d[11x10,obh=14,ibw=3,sb=7x1]` | 3.035 | 0.99996 |
| `2d[11x10,obh=11,ibw=2,sb=1x6]` | 3.042 | 0.99997 |
| `minimal[8,8,8,sb=1x8]` | 3.084 | 0.99996 |
| `minimal[16,4,4,sb=2x4]` | 3.089 | 0.99998 |
| `2d[11x10,obh=7,ibw=4,sb=7x1]` | 3.097 | 0.99996 |
| `2d[11x10,obh=4,ibw=8,sb=4x2]` | 3.111 | 0.99993 |
| `minimal[default]` | 3.119 | 0.99996 |
| `minimal[8,4,8,sb=1x8]` | 3.127 | 0.99998 |
| `2d[11x10,obh=14,ibw=2,sb=7x1]` | 3.131 | 0.99997 |
| `2d[11x10,obh=4,ibw=6,sb=4x2]` | 3.142 | 0.99994 |
| `2d[11x10,obh=7,ibw=3,sb=7x1]` | 3.202 | 0.99996 |
| `2d[11x10,obh=4,ibw=4,sb=4x2]` | 3.280 | 0.99996 |
| `2d[11x10,obh=4,ibw=3,sb=4x2]` | 3.311 | 0.99996 |
| `2d[11x10,obh=7,ibw=2,sb=7x1]` | 3.313 | 0.99997 |
| `2d[11x10,obh=11,ibw=1,sb=1x6]` | 3.375 | 0.99997 |
| `2d[11x10,obh=14,ibw=1,sb=7x1]` | 3.449 | 0.99997 |
| `2d[11x10,obh=4,ibw=2,sb=4x2]` | 3.486 | 0.99997 |
| `2d[11x10,obh=7,ibw=1,sb=7x1]` | 3.736 | 0.99997 |
| `minimal[4,8,8,sb=1x8]` | 3.866 | 0.99996 |
| `core_grid` | 3.919 | 0.99997 |
| `2d[11x10,obh=4,ibw=1,sb=4x2]` | 4.061 | 0.99997 |
| `1d_full_n[obh=2,ibw=2,sb=1x8]` | 4.157 | 0.99997 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 4.379 | 0.99997 |
| `default` | 5.312 | 0.99997 |
| `1d_full_n[obh=1,ibw=3,sb=1x8]` | 5.714 | 0.99996 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 5.783 | 0.99997 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 5.937 | 0.99997 |
| `minimal[tt_dit heuristic]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |
| `minimal[16,8,8,sb=1x8]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |

## down

| candidate | ms | PCC / error |
|---|---|---|
| `minimal[8,8,8,sb=1x8]` | 1.805 | 0.99996 |
| `minimal[8,8,4,sb=2x4]` | 1.828 | 0.99996 |
| `minimal[16,4,4,sb=2x4]` | 1.915 | 0.99997 |
| `minimal[tt_dit heuristic]` | 1.931 | 0.99996 |
| `minimal[8,4,8,sb=1x8]` | 1.997 | 0.99997 |
| `minimal[default]` | 2.072 | 0.99996 |
| `minimal[4,8,8,sb=1x8]` | 2.190 | 0.99996 |
| `2d[8x10,obh=11,ibw=8,sb=1x3]` | 2.423 | 0.99992 |
| `2d[8x10,obh=14,ibw=8,sb=7x1]` | 2.437 | 0.99992 |
| `1d_full_n[obh=4,ibw=8,sb=1x8]` | 2.525 | 0.99992 |
| `2d[8x10,obh=7,ibw=8,sb=7x1]` | 2.645 | 0.99992 |
| `1d_full_n[obh=2,ibw=8,sb=1x8]` | 2.701 | 0.99992 |
| `1d_full_n[obh=2,ibw=4,sb=1x8]` | 2.814 | 0.99995 |
| `2d[8x10,obh=4,ibw=8,sb=2x3]` | 2.992 | 0.99992 |
| `1d_full_n[obh=4,ibw=4,sb=1x8]` | 3.015 | 0.99995 |
| `2d[8x10,obh=11,ibw=4,sb=1x3]` | 3.133 | 0.99995 |
| `1d_full_n[obh=4,ibw=2,sb=1x8]` | 3.135 | 0.99996 |
| `2d[8x10,obh=14,ibw=4,sb=7x1]` | 3.141 | 0.99995 |
| `1d_full_n[obh=7,ibw=4,sb=1x8]` | 3.190 | 0.99995 |
| `1d_full_n[obh=2,ibw=2,sb=1x8]` | 3.264 | 0.99996 |
| `1d_full_n[obh=7,ibw=2,sb=1x8]` | 3.275 | 0.99996 |
| `2d[8x10,obh=7,ibw=4,sb=7x1]` | 3.297 | 0.99995 |
| `2d[8x10,obh=14,ibw=2,sb=7x1]` | 3.336 | 0.99996 |
| `2d[8x10,obh=11,ibw=2,sb=1x3]` | 3.504 | 0.99996 |
| `2d[8x10,obh=4,ibw=4,sb=2x3]` | 3.534 | 0.99995 |
| `2d[8x10,obh=7,ibw=2,sb=7x1]` | 3.706 | 0.99996 |
| `1d_full_n[obh=7,ibw=1,sb=1x8]` | 3.843 | 0.99994 |
| `1d_full_n[obh=4,ibw=1,sb=1x8]` | 3.849 | 0.99994 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 4.130 | 0.99994 |
| `2d[8x10,obh=14,ibw=1,sb=7x1]` | 4.218 | 0.99994 |
| `2d[8x10,obh=4,ibw=2,sb=2x3]` | 4.221 | 0.99996 |
| `2d[8x10,obh=11,ibw=1,sb=1x3]` | 4.289 | 0.99994 |
| `1d_full_n[obh=1,ibw=8,sb=1x8]` | 4.411 | 0.99992 |
| `1d_full_n[obh=1,ibw=4,sb=1x8]` | 4.500 | 0.99995 |
| `2d[8x10,obh=7,ibw=1,sb=7x1]` | 4.746 | 0.99994 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 4.764 | 0.99996 |
| `2d[8x10,obh=4,ibw=1,sb=2x3]` | 5.646 | 0.99994 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 5.854 | 0.99994 |
| `default` | 8.251 | 0.99996 |
| `core_grid` | 9.246 | 0.99994 |
| `minimal[16,8,8,sb=1x8]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |

## embed_hidden

| candidate | ms | PCC / error |
|---|---|---|
| `2d[11x10,obh=11,ibw=2,sb=1x3]` | 3.134 | 0.99998 |
| `2d[11x10,obh=11,ibw=1,sb=1x3]` | 3.237 | 0.99998 |
| `2d[11x10,obh=14,ibw=1,sb=7x1]` | 3.440 | 0.99998 |
| `2d[11x10,obh=14,ibw=2,sb=7x1]` | 3.472 | 0.99998 |
| `2d[11x10,obh=4,ibw=1,sb=2x3]` | 3.536 | 0.99998 |
| `2d[11x10,obh=4,ibw=2,sb=2x3]` | 3.543 | 0.99998 |
| `2d[11x10,obh=7,ibw=2,sb=7x1]` | 3.620 | 0.99998 |
| `2d[11x10,obh=7,ibw=1,sb=7x1]` | 3.634 | 0.99998 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 4.073 | 0.99998 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 4.404 | 0.99998 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 4.442 | 0.99998 |
| `minimal[tt_dit heuristic]` | 4.720 | 0.99998 |
| `core_grid` | 4.824 | 0.99998 |
| `minimal[default]` | 5.583 | 0.99998 |
| `default` | 6.546 | 0.99998 |

## embed_out

| candidate | ms | PCC / error |
|---|---|---|
| `minimal[tt_dit heuristic]` | 2.557 | 0.99996 |
| `minimal[16,4,4,sb=2x4]` | 2.587 | 0.99996 |
| `minimal[8,8,4,sb=2x4]` | 2.649 | 0.99996 |
| `minimal[8,8,8,sb=1x8]` | 2.682 | 0.99996 |
| `minimal[8,4,8,sb=1x8]` | 2.726 | 0.99996 |
| `minimal[4,8,8,sb=1x8]` | 2.937 | 0.99996 |
| `minimal[default]` | 2.992 | 0.99996 |
| `2d[8x10,obh=14,ibw=8,sb=7x1]` | 3.541 | 0.99996 |
| `2d[8x10,obh=11,ibw=8,sb=1x3]` | 3.558 | 0.99996 |
| `1d_full_n[obh=4,ibw=8,sb=1x8]` | 3.606 | 0.99996 |
| `1d_full_n[obh=2,ibw=8,sb=1x8]` | 3.607 | 0.99996 |
| `2d[8x10,obh=11,ibw=12,sb=1x3]` | 3.629 | 0.99994 |
| `2d[8x10,obh=14,ibw=12,sb=7x1]` | 3.639 | 0.99994 |
| `1d_full_n[obh=2,ibw=4,sb=1x8]` | 3.753 | 0.99996 |
| `1d_full_n[obh=2,ibw=6,sb=1x8]` | 3.842 | 0.99996 |
| `2d[8x10,obh=7,ibw=8,sb=7x1]` | 3.861 | 0.99996 |
| `2d[8x10,obh=7,ibw=12,sb=7x1]` | 3.901 | 0.99994 |
| `1d_full_n[obh=4,ibw=3,sb=1x8]` | 4.000 | 0.99996 |
| `2d[8x10,obh=14,ibw=6,sb=7x1]` | 4.003 | 0.99996 |
| `2d[8x10,obh=11,ibw=6,sb=1x3]` | 4.039 | 0.99996 |
| `1d_full_n[obh=2,ibw=3,sb=1x8]` | 4.069 | 0.99996 |
| `2d[8x10,obh=4,ibw=12,sb=2x3]` | 4.142 | 0.99994 |
| `1d_full_n[obh=7,ibw=3,sb=1x8]` | 4.210 | 0.99996 |
| `2d[8x10,obh=4,ibw=8,sb=2x3]` | 4.218 | 0.99996 |
| `1d_full_n[obh=4,ibw=4,sb=1x8]` | 4.231 | 0.99996 |
| `1d_full_n[obh=4,ibw=6,sb=1x8]` | 4.237 | 0.99996 |
| `1d_full_n[obh=4,ibw=2,sb=1x8]` | 4.250 | 0.99995 |
| `2d[8x10,obh=14,ibw=3,sb=7x1]` | 4.273 | 0.99996 |
| `2d[8x10,obh=7,ibw=6,sb=7x1]` | 4.284 | 0.99996 |
| `2d[8x10,obh=11,ibw=3,sb=1x3]` | 4.365 | 0.99996 |
| `1d_full_n[obh=2,ibw=2,sb=1x8]` | 4.416 | 0.99995 |
| `1d_full_n[obh=7,ibw=2,sb=1x8]` | 4.428 | 0.99995 |
| `1d_full_n[obh=7,ibw=4,sb=1x8]` | 4.463 | 0.99996 |
| `2d[8x10,obh=14,ibw=4,sb=7x1]` | 4.493 | 0.99996 |
| `2d[8x10,obh=4,ibw=6,sb=2x3]` | 4.502 | 0.99996 |
| `2d[8x10,obh=11,ibw=4,sb=1x3]` | 4.527 | 0.99996 |
| `2d[8x10,obh=14,ibw=2,sb=7x1]` | 4.598 | 0.99995 |
| `2d[8x10,obh=11,ibw=2,sb=1x3]` | 4.697 | 0.99995 |
| `2d[8x10,obh=7,ibw=4,sb=7x1]` | 4.762 | 0.99996 |
| `2d[8x10,obh=7,ibw=3,sb=7x1]` | 4.827 | 0.99996 |
| `2d[8x10,obh=4,ibw=4,sb=2x3]` | 5.108 | 0.99996 |
| `2d[8x10,obh=4,ibw=3,sb=2x3]` | 5.223 | 0.99996 |
| `2d[8x10,obh=7,ibw=2,sb=7x1]` | 5.230 | 0.99995 |
| `1d_full_n[obh=7,ibw=1,sb=1x8]` | 5.302 | 0.99992 |
| `1d_full_n[obh=4,ibw=1,sb=1x8]` | 5.312 | 0.99992 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 5.571 | 0.99992 |
| `2d[8x10,obh=14,ibw=1,sb=7x1]` | 5.920 | 0.99992 |
| `1d_full_n[obh=1,ibw=8,sb=1x8]` | 5.971 | 0.99996 |
| `2d[8x10,obh=4,ibw=2,sb=2x3]` | 5.997 | 0.99995 |
| `1d_full_n[obh=1,ibw=6,sb=1x8]` | 6.003 | 0.99996 |
| `1d_full_n[obh=1,ibw=4,sb=1x8]` | 6.048 | 0.99996 |
| `2d[8x10,obh=11,ibw=1,sb=1x3]` | 6.065 | 0.99992 |
| `1d_full_n[obh=1,ibw=3,sb=1x8]` | 6.283 | 0.99996 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 6.639 | 0.99995 |
| `2d[8x10,obh=7,ibw=1,sb=7x1]` | 6.771 | 0.99992 |
| `2d[8x10,obh=4,ibw=1,sb=2x3]` | 8.112 | 0.99992 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 8.173 | 0.99992 |
| `default` | 11.722 | 0.99995 |
| `core_grid` | 13.797 | 0.99992 |
| `minimal[16,8,8,sb=1x8]` | nan | FAILED: TT_THROW @ /home/andy/tt-metal/tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3087: tt::exception |

## embed_res

| candidate | ms | PCC / error |
|---|---|---|
| `minimal[tt_dit heuristic]` | 0.740 | 0.99999 |
| `2d[8x10,obh=11,ibw=2,sb=1x3]` | 0.813 | 0.99999 |
| `2d[8x10,obh=11,ibw=1,sb=1x3]` | 0.837 | 0.99999 |
| `2d[8x10,obh=4,ibw=2,sb=2x3]` | 0.850 | 0.99999 |
| `2d[8x10,obh=14,ibw=1,sb=7x1]` | 0.862 | 0.99999 |
| `2d[8x10,obh=14,ibw=2,sb=7x1]` | 0.868 | 0.99999 |
| `minimal[default]` | 0.883 | 0.99999 |
| `2d[8x10,obh=4,ibw=1,sb=2x3]` | 0.892 | 0.99999 |
| `2d[8x10,obh=7,ibw=1,sb=7x1]` | 0.911 | 0.99999 |
| `2d[8x10,obh=7,ibw=2,sb=7x1]` | 0.920 | 0.99999 |
| `1d_full_n[obh=7,ibw=2,sb=1x8]` | 1.039 | 0.99999 |
| `1d_full_n[obh=7,ibw=1,sb=1x8]` | 1.049 | 0.99999 |
| `1d_full_n[obh=4,ibw=2,sb=1x8]` | 1.052 | 0.99999 |
| `1d_full_n[obh=4,ibw=1,sb=1x8]` | 1.055 | 0.99999 |
| `core_grid` | 1.061 | 0.99999 |
| `1d_full_n[obh=2,ibw=2,sb=1x8]` | 1.088 | 0.99999 |
| `1d_full_n[obh=2,ibw=1,sb=1x8]` | 1.091 | 0.99999 |
| `default` | 1.091 | 0.99999 |
| `1d_full_n[obh=1,ibw=2,sb=1x8]` | 1.163 | 0.99999 |
| `1d_full_n[obh=1,ibw=1,sb=1x8]` | 1.176 | 0.99999 |
