#!/bin/sh
# A/B timing across fresh processes. B1 varies ~0.1 ms between processes, so one run
# per config is not enough. Alternates A and B, N rounds each, prints every mean.
# Usage (tt-metal root): ab_bench.sh <batch> <rounds> <file> <A copy> <B copy>
# <file> is overwritten with each copy in turn and restored from <A copy> at the end.
B=$1; N=$2; F=$3; A_SRC=$4; B_SRC=$5
H=$(dirname "$0")
for i in $(seq 1 "$N"); do
    for v in A B; do
        if [ $v = A ]; then cp "$A_SRC" "$F"; else cp "$B_SRC" "$F"; fi
        m=$(python_env/bin/python -u "$H/bench_nomask.py" "$B" 200 2>&1 | grep -a '^BENCH' | sed 's/.* mean=\([0-9.]*\).*/\1/')
        echo "$v round$i ${m:-FAIL}"
    done
done
cp "$A_SRC" "$F"
