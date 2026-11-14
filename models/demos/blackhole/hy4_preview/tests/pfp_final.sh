#!/bin/bash
# final evidence batch on .45 slot 2 (run from any host with ssh to .45): tag prefix f_
cd /mnt/tt-data/ssinghal/wt/hy4_pfpool/models/demos/blackhole/hy4_preview/tests
for L in 0 1 2; do ./pfp_go.sh f_l$L pfp_layer.py LAYER=$L CHUNKS=6 TIMING=0 REPEAT=1 N=3 W=10 >/dev/null; done
./pfp_go.sh f_l1_pfopt pfp_layer.py LAYER=1 CHUNKS=6 TIMING=0 HY4_PF_OPT=all DEPTHS=5,63,255,1023,2047 N=3 W=10 >/dev/null
./pfp_go.sh f_l2_pfopt pfp_layer.py LAYER=2 CHUNKS=6 TIMING=0 HY4_PF_OPT=all DEPTHS=5,63,255,1023,2047 N=3 W=10 >/dev/null
./pfp_go.sh f_l1_1024 pfp_layer.py LAYER=1 CHUNKS=6 TIMING=0 PF_ROWS=1024 N=3 W=10 >/dev/null
./pfp_go.sh f_l2_1024 pfp_layer.py LAYER=2 CHUNKS=6 TIMING=0 PF_ROWS=1024 N=3 W=10 >/dev/null
./pfp_go.sh f_carry pfp_carry.py N=3 W=10 >/dev/null
./pfp_go.sh f_merge pfp_merge.py N=3 W=10 >/dev/null
./pfp_go.sh f_needle pfp_needle.py CTXS=65536,262144,1048576,999424 N=3 W=10 >/dev/null
./pfp_go.sh f_kernel pfp_kernel.py PATS=mix,256 N=3 W=10 >/dev/null
./pfp_go.sh f_idx pfp_idx.py N=3 W=10 >/dev/null
