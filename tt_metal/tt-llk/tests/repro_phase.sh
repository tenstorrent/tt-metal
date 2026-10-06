#!/bin/bash
# Run the Wormhole packer-phase repro once and print the pack TILE_LOOP cycles.
#
#   ./repro_phase.sh sim|hw  K=<n> [NOPS=<n>] [TAIL=<n>] [LOOP=<n>] [VCD=1]
#
#   sim   Versim (needs /proj_sw); hw = the Wormhole card in this machine
#   K     spin iteration at which the math RISC-V does one L1 load during the pack loop (0 = no load).
#         A runtime value: the host writes it to L1 0x16AFE4, so the ELF is the same for every K.
#   NOPS  4-byte nops in the Wormhole _llk_pack_init_ (a code change that does no work). Default 0.
#   TAIL  spin iterations unpack and math wait before ending their zone (6000 = they stay idle past the
#         pack loop; 0 = they end during it). Default 6000.
#   LOOP  tiles packed (loop factor). Default 256.
#   VCD=1 keep the Versim waveform (several GB; written to the run directory).
#   INJ   15 (default): the K loop is compiled in for every K, so K=0 and K=100 run the same ELF.
set -e
mode=$1; shift
K=0; NOPS=0; TAIL=6000; LOOP=256; VCD=0; INJ=15
for a in "$@"; do eval "$a"; done
here=$(cd "$(dirname "$0")" && pwd)
name=${mode}_k${K}_n${NOPS}_t${TAIL}_l${LOOP}_i${INJ}
out=${OUT_DIR:-/tmp/repro_phase}/$name
mkdir -p "$out"
cd "$here/python_tests"
source ../.venv/bin/activate
export CHIP_ARCH=wormhole LLK_HOME="$here/.." LLK_PERF_RUN_TYPES=PACK_ISOLATE
export REPRO_RT=$K REPRO_PACK_NOPS=$NOPS REPRO_TAIL=$TAIL REPRO_LOOP_FACTOR=$LOOP REPRO_INJ=$INJ
export RUNNER_TEMP="$out/build" NNG_SOCKET_NAME=$name
sim=""
if [ "$mode" = sim ]; then
  export TT_METAL_SIMULATOR=${TT_METAL_SIMULATOR:-/proj_sw/user_dev/ndivnic/tt-umd-simulators/build/versim-wormhole-b0}
  export TT_SIMULATOR_LOCALHOST=1
  [ "$VCD" = 1 ] || export USER=gitlab-ci   # Versim writes a VCD unless USER=gitlab-ci
  sim=--run-simulator
fi
python -u -m pytest $sim -p no:randomly -v \
  "perf_math_matmul.py::test_perf_math_matmul[repro_knob0-MathFidelity.LoFi-matmul_config1-0-1]" 2>&1 | tee "$out/run.log"
csv=$(dirname "$(grep -o 'Wrote run Parquet batch: [^ ]*' "$out/run.log" | tail -1 | awk '{print $NF}')")/perf_math_matmul/perf_math_matmul.csv
cp "$csv" "$out/"
[ "$mode" = sim ] && mv versim_*_"$name".log versim_*"$name"*.vcd "$out/" 2>/dev/null || true
python3 - "$out/perf_math_matmul.csv" <<'PY'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    print(f"{r['marker']:10s} {float(r['mean(PACK_ISOLATE)']):10.0f} cycles")
PY
echo "results: $out"
