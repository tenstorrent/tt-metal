#!/bin/bash
# prep_round.sh VARIANT...: make the derived variant dirs a round needs, next to each variant:
#   <v>_strict  the variant plus the test kernel sending flushed packets as FLUSH_STRICT
#   <v>_hook    the variant plus perturb.py's hook probe (needs the STABILITY_PROBE markers)
# Then build each of them offline at -O3 (all logged configs) so a broken one is caught before the device job.
set -e
cd /home/aagarwal/tt-metal
CON=fabric_exp/consolidation
V=fabric_exp/variants
T=tests/tt_metal/tt_fabric/test_infra/kernels/tt_fabric_test_kernels_utils.hpp
TMP=/tmp/claude-4004/-home-aagarwal-tt-metal/76f59f51-9416-4ea6-898e-740976fca7f2/scratchpad/prep
for v in "$@"; do
    rm -rf "${V:?}/${v:?}_strict" "${V:?}/${v:?}_hook" "${TMP:?}/${v:?}"
    cp -r $V/$v $V/${v}_strict
    mkdir -p $V/${v}_strict/$(dirname $T)
    sed 's/kActiveFlushMode = tt::tt_fabric::NocFlushType::FLUSH_RELAXED;/kActiveFlushMode = tt::tt_fabric::NocFlushType::FLUSH_STRICT;/' \
        $T > $V/${v}_strict/$T
    grep -q 'kActiveFlushMode = tt::tt_fabric::NocFlushType::FLUSH_STRICT;' $V/${v}_strict/$T
    python3 $CON/perturb.py $V/$v $TMP/$v hook > /dev/null
    cp -r $TMP/$v/hook $V/${v}_hook
    for b in $v ${v}_hook; do
        python3 $CON/offline_build.py build --variant $V/$b --out $TMP/build_$b --jobs 48 | tail -1
    done
done
