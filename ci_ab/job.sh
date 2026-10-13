# unary: the head with the interleaved-input gate (G) against the previous head's files (P), both merged with main;
# C is G with 1/ln2 one part in 10^4 off on the gated path (positive control); P2 repeats P (A/A).
BITS_VARIANTS="P G P2 C"
OPS="gelu,gelu_approx,gelu_tanh,gelu_bw_poly,celu,elu,selu,exp,exp_approx,exp2,expm1,sigmoid,sigmoid_approx,sigmoid_fastexp,silu,mish,softplus,tanhshrink,logsigmoid,snake_beta,i1,softcap,xielu,tanh"
bits() {
  SFPU_FIX_DUMP_DIR=$1 R3_DUMP_OPS=$OPS timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider $S/tests/test_sfpu_unary_r3_dump.py
  SFPU_FIX_DUMP_DIR=$1 R3_MM_MODE=dump R3_MM_ACTS=gelu,silu,sigmoid,selu,softplus timeout -s INT -k 60 1200 python3 -m pytest -q -p no:cacheprovider $S/tests/test_sfpu_unary_r3_mm.py
}
COMPARE=("P G" "P P2" "P C")
PROF_PAIRS=()
