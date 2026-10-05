#!/bin/bash
# t133: submodules + lean Release build, then LTX CPU tests with /dev/tenstorrent hidden. Writes tmp/t133/DONE.
set -u
cd "$(dirname "$0")/../.."
W=$PWD; L=$W/tmp/t133
rm -f $L/DONE
{ git submodule update --init --recursive && echo SUBMODULES_OK; } > $L/submodules.log 2>&1
{ ./build_metal.sh --release; echo "BUILD_EXIT=$?"; } > $L/build.log 2>&1
grep -q 'BUILD_EXIT=0' $L/build.log || { echo build_failed > $L/DONE; exit 1; }
unshare -r -m sh -c 'mount -t tmpfs none /dev/tenstorrent && cd "$0" && exec timeout 2400 '"$W"'/../../../python_env/bin/python -m pytest -q -p no:cacheprovider -rfE models/tt_dit/tests/models/ltx/' "$W" > $L/ltx_cpu.log 2>&1
echo "PYTEST_EXIT=$?" >> $L/ltx_cpu.log
tail -1 $L/ltx_cpu.log > $L/DONE
