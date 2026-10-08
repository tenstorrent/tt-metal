#!/bin/bash
# vrun.sh <name> <tree> <test ids, separated by |> [ENV=VAL ...]: one Versim run with VCD in its own copy of the tree; writes /tmp/v_<name>.log, keeps the VCD at /tmp/vcd/<name>.vcd
name=$1; tree=$2; id=$3; shift 3; IFS="|" read -ra IDS <<< "$id"
w=/tmp/vw_$name; c=$(git -C $tree rev-parse HEAD)
[ -d $w ] || git -C /home/nstojic/tt-metal worktree add -q --detach $w $c
git -C $w checkout -q --detach $c
cd $w/tt_metal/tt-llk/tests && for l in .venv sfpi; do ln -sfn /home/nstojic/tt-metal/tt_metal/tt-llk/tests/$l $l; done; ln -sfn ../../sfpi-info.sh sfpi-info.sh; ln -sfn ../../sfpi-version sfpi-version
cd python_tests && source ../.venv/bin/activate
mkdir -p /tmp/vcd
env CHIP_ARCH=wormhole LLK_HOME=$w/tt_metal/tt-llk RUNNER_TEMP=/tmp/build-v_$name NNG_SOCKET_NAME=v_$name \
  TT_METAL_SIMULATOR=/proj_sw/user_dev/ndivnic/tt-umd-simulators/build/versim-wormhole-b0 TT_SIMULATOR_LOCALHOST=1 LLK_SIM_TIMEOUT=14400 "$@" \
  python -u -m pytest --run-simulator -p no:randomly -v "${IDS[@]}" > /tmp/v_$name.log 2>&1 < /dev/null
mv -f 1-1-core_dump.vcd /tmp/vcd/$name.vcd 2>/dev/null
grep -h "TILE_LOOP" $(dirname $(grep -o "Wrote run Parquet batch: [^ ]*" /tmp/v_$name.log | tail -1 | awk '{print $NF}'))/*/*.csv 2>/dev/null | grep -v post > /tmp/v_$name.tl
echo done > /tmp/v_$name.done
