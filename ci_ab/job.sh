# unary: the init-once and chain forms ported to eltwise_sfpu_metal2.cpp (Q, the head) against the head without them
# (P: main's metal2 kernel); C is Q leaving the first tile of a single op unprocessed (positive control); P2 repeats P.
BITS_VARIANTS="P Q P2 C"
OPS="gelu,gelu_approx,gelu_tanh,gelu_bw_poly,celu,elu,selu,exp,exp_approx,exp2,expm1,sigmoid,sigmoid_approx,sigmoid_fastexp,silu,mish,softplus,tanhshrink,logsigmoid,snake_beta,i1,softcap,xielu,tanh"
put() { for f in $2/*; do [ -f "$f" ] && mv "$f" "$1/$3__$(basename $f)"; done; rm -rf "$2"; }
bits() {
  for L in dram dram16 sh4; do
    R3_DUMP_LAYOUT=$L SFPU_FIX_DUMP_DIR=$1/_t R3_DUMP_OPS=$OPS timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider $S/tests/test_sfpu_unary_r3_dump.py
    put $1 $1/_t op_$L
  done
  for L in dram sh4 sh2; do
    R3_DUMP_LAYOUT=$L R3_CHAIN_MODE=dump SFPU_FIX_DUMP_DIR=$1/_t timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider $S/tests/test_sfpu_unary_r3_chains.py
    put $1 $1/_t chain_$L
  done
  R3_DUMP_LAYOUT=dram16 SFPU_FIX_DUMP_DIR=$1/_t timeout -s INT -k 60 3000 python3 -m pytest -q -p no:cacheprovider $S/tests/test_sfpu_unary_r3_chain_pairs.py
  put $1 $1/_t pair_dram16
}
COMPARE=("P Q" "P P2" "P C")
PROF_PAIRS=()
