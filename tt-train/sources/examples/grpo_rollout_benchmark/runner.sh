#!/bin/bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# GRPO rollout benchmark launcher. One backend per invocation:
#
#   ttml backend (local generation, single process):
#     runner.sh --backend ttml --ttml-devices {1,2,4,8} [--steps K] [--repeats R]
#
#   ttt backend (remote generation, two tt-run ranks):
#     runner.sh --backend ttt --ttml-devices N --ttt-devices M [--steps K] [--repeats R]
#     supported splits (N+M): 2+2 (local4) and 4+4 (local8)
#
# --steps K   GRPO steps per run (default 20)
# --repeats R how many times to run the whole bench (default 1); each repeat is a
#             FRESH process and appends to the same CSV, tagged by run index.

set -euo pipefail

if [[ -z "${TT_METAL_HOME:-}" ]]; then
    echo "TT_METAL_HOME is not set" >&2
    exit 1
fi

EX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND=""
TTML_DEVICES=""
TTT_DEVICES=""
STEPS=20
REPEATS=1

usage() {
    sed -n '4,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit "${1:-1}"
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --backend)      shift; BACKEND="${1:-}" ;;
        --ttml-devices) shift; TTML_DEVICES="${1:-}" ;;
        --ttt-devices)  shift; TTT_DEVICES="${1:-}" ;;
        --steps)        shift; STEPS="${1:-}" ;;
        --repeats)      shift; REPEATS="${1:-}" ;;
        -h|--help)      usage 0 ;;
        *) echo "Unknown argument: $1" >&2; usage 1 ;;
    esac
    shift
done

[[ -n "${BACKEND}" ]] || { echo "--backend is required (ttml|ttt)" >&2; usage 1; }
[[ -n "${TTML_DEVICES}" ]] || { echo "--ttml-devices is required" >&2; usage 1; }
[[ "${STEPS}" =~ ^[0-9]+$ && "${STEPS}" -ge 1 ]] || { echo "--steps must be a positive integer" >&2; exit 1; }
[[ "${REPEATS}" =~ ^[0-9]+$ && "${REPEATS}" -ge 1 ]] || { echo "--repeats must be a positive integer" >&2; exit 1; }

# cd here so relative config / mesh_graph_desc_path resolve against the example dir.
cd "${EX_DIR}"

case "${BACKEND}" in
    ttml)
        case "${TTML_DEVICES}" in
            1|2|4|8) ;;
            *) echo "ttml backend: --ttml-devices must be one of {1,2,4,8} (got ${TTML_DEVICES})" >&2; exit 1 ;;
        esac
        CONFIG="configs/ttml_${TTML_DEVICES}dev.yaml"
        [[ -f "${CONFIG}" ]] || { echo "missing ${CONFIG}" >&2; exit 1; }
        for i in $(seq 1 "${REPEATS}"); do
            echo "=== [ttml ${TTML_DEVICES}dev] repeat ${i}/${REPEATS} (steps=${STEPS}) ==="
            python3 bench_ttml.py --config "${CONFIG}" --steps "${STEPS}" --run-index "${i}"
        done
        ;;

    ttt)
        [[ -n "${TTT_DEVICES}" ]] || { echo "ttt backend: --ttt-devices is required" >&2; usage 1; }
        case "${TTML_DEVICES}x${TTT_DEVICES}" in
            2x2) CONFIG_DIR="local4" ;;
            4x4) CONFIG_DIR="local8" ;;
            *) echo "ttt backend: unsupported split ${TTML_DEVICES}+${TTT_DEVICES} (expected 2+2 or 4+4)" >&2; exit 1 ;;
        esac
        RANK_BINDINGS="configurations/${CONFIG_DIR}/rank_bindings.yaml"
        HOST_FILE="configurations/${CONFIG_DIR}/hosts.txt"
        [[ -f "${RANK_BINDINGS}" && -f "${HOST_FILE}" ]] || { echo "missing ${CONFIG_DIR} rank bindings" >&2; exit 1; }

        export GRPO_BENCH_TTML_DEVICES="${TTML_DEVICES}"
        export GRPO_BENCH_TTT_DEVICES="${TTT_DEVICES}"
        export GRPO_BENCH_STEPS="${STEPS}"
        for i in $(seq 1 "${REPEATS}"); do
            echo "=== [ttt ${TTML_DEVICES}+${TTT_DEVICES}] repeat ${i}/${REPEATS} (steps=${STEPS}) ==="
            export GRPO_BENCH_RUN="${i}"
            "${TT_METAL_HOME}/ttnn/ttnn/distributed/ttrun.py" \
                --rank-binding "${RANK_BINDINGS}" \
                --mpi-args "--hostfile ${HOST_FILE} --tag-output \
                    -x GRPO_BENCH_TTML_DEVICES -x GRPO_BENCH_TTT_DEVICES -x GRPO_BENCH_STEPS -x GRPO_BENCH_RUN" \
                python3 bench_ttt.py
        done
        ;;

    *)
        echo "--backend must be 'ttml' or 'ttt' (got ${BACKEND})" >&2; exit 1 ;;
esac
