for i in $(seq 0 50);
do
 TT_METAL_SLOW_DISPATCH_MODE=1 TT_NOP_INSERT=$i ./test_sdpa_nd_bug
done
