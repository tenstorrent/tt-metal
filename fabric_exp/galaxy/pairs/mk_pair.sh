#!/bin/bash
# mk_pair.sh NAME HOST_A HOST_B: the pair's FSD slice (fsd_subset.py), per-host mock cluster descriptors from it
# (generate_cluster_descriptor), each stamped with its hostname (cluster_id, as the supercluster descriptors carry), and
# the rank mapping. Output: $S/pair_NAME/.
S=${PAIRS_WORK:-/home/aagarwal/tt-metal/fabric_exp/galaxy/pairs/work}  # pair dirs and mock outputs
F=/data/scaleout_configs/tt-cluster-config-sources/exabox-latest-main/factory_system_descriptor.textproto
n=$1; a=$2; b=$3; d=$S/pair_$n
rm -rf $d; mkdir -p $d
python3 $(dirname "$0")/fsd_subset.py $F $d/fsd.textproto $a $b > $d/subset.txt || exit 1
/home/aagarwal/tt-metal-mainline/build_Release/tools/scaleout/generate_cluster_descriptor --fsd $d/fsd.textproto -o $d -b cd > $d/gen.txt 2>&1 || { cat $d/gen.txt; exit 1; }
k=0; for h in $a $b; do sed -i "1i cluster_id: $h" $d/cd_rank_$k.yaml; k=$((k + 1)); done
cat $d/subset.txt
