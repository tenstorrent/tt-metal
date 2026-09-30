#!/bin/bash
# t19: CPU A/B of one latent crop. Usage: crop.sh <seed> <name> t0 h0 w0 T H W   (latent units; 2.5 1080p/145f = 19x34x60)
H=$(cd "$(dirname "$0")" && pwd); D=$H/data; T12=/home/smarton/fasth3/tt-metal/tt-project/research/t12
s=$1 n=$2; shift 2; o=$D/crops/seed${s}_$n; mkdir -p $o
export PYTHONPATH=/home/smarton/tmp/vaecmp/LTX-2/packages/ltx-core/src LF=19 LH=34 LW=60 WHICH=conv,diff0 NT=${NT:-32}
PY=/home/smarton/fasth3/tt-metal/python_env/bin/python
$PY $T12/cpu_ab.py $D/lat/seed$s.pt $o "$@" > $o/decode.log 2>&1 && (cd $o && $PY $T12/score.py $o diff0) > $o/score.log 2>&1
cat $o/score.log
