# unary: the stacked activations change and the gelu lever (B) against the previous head's files (P); C is B with a wrong
# exp_21f c2 in Prgm0 (positive control); P2 repeats P (A/A).
BITS_VARIANTS="P F P2 C"
OPS="gelu,gelu_approx,gelu_tanh,gelu_bw_poly,celu,elu,selu,exp,exp_approx,exp2,expm1,sigmoid,sigmoid_approx,sigmoid_fastexp,silu,mish,softplus,tanhshrink,logsigmoid,snake_beta,i1,softcap,xielu,tanh"
bits() {
  SFPU_FIX_DUMP_DIR=$1 R3_DUMP_OPS=$OPS timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider $S/tests/test_sfpu_unary_r3_dump.py
  SFPU_FIX_DUMP_DIR=$1 R3_MM_MODE=dump R3_MM_ACTS=gelu,silu,sigmoid,selu,softplus timeout -s INT -k 60 1200 python3 -m pytest -q -p no:cacheprovider $S/tests/test_sfpu_unary_r3_mm.py
}
COMPARE=("P F" "P P2" "P C")
PROF_PAIRS=("P F")
PROF_CASES="gelu_dram_1024x4096 gelu_hs silu_dram_32x14336 silu_hs sigmoid_dram_1024x1024 silu_mul_32x14336 silu_mul_128x14336 gelu_linear_1024x1024x4096"
PROF_SCRIPT=prof_su.py
