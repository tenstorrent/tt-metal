#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# The disaggregated prefill server (host 40, mesh 0) and the decode server (host 41, mesh 1) as the two ranks of ONE
# tt-run job on the two-mesh fabric (PREFILL_D_P_FABRIC_PUSH.md, phase 0.3). The control plane of a two-mesh fabric
# all-gathers over every rank when it starts, so the servers come up together, and one exiting ends the job.
#
# Both ranks open the same router settings: 2D torus fabric, STRICT_INIT, and the prefill's 6208 B packet payload
# (MOE_BLOCK_SPLIT_ROWS=on; decode takes it through DEEPSEEK_V4_FABRIC_PAYLOAD).
#
# Run on host 40 from the repo root, venv active; host 41 has the same checkout, built, at the same path:
#
#   models/experimental/deepseek_v41_flash/demo/run_d_p_servers.sh --hf /path/to/DeepSeek-V4.1-Flash \
#     [--prefill-port 9041] [-- decode server args...]
#
# Prefill logs are tagged [1,0]<stdout>, decode's [1,1]<stdout>. The decode server listens on host 41 (default
# port 8000); the prefill server on host 40 at --prefill-port. Decode args for the disaggregated prefill with the KV
# pushed over the fabric: -- --prefill-url 10.82.97.40:9041 --prefill-push (--no-prefill-push: the KV over TCP).
set -euo pipefail

SCRIPT=models/experimental/deepseek_v41_flash/demo/run_d_p_servers.sh
PREFILL_HOST=DC16-2-BG-0202-u20-40
DECODE_HOST=DC16-2-BG-0202-u13-41
MPI_IFACE=ens5f0np0.1

hf="" prefill_port=9041
while [ $# -gt 0 ]; do
  case "$1" in
    --hf) hf="$2"; shift 2 ;;
    --prefill-port) prefill_port="$2"; shift 2 ;;
    # (--decode-args: the ranks' own separator, as tt-run drops a "--" from the command it runs)
    --|--decode-args) shift; break ;;
    *) echo "unknown argument $1 (decode server args go after --)" >&2; exit 2 ;;
  esac
done
decode_args=("$@")

if [ -z "${OMPI_COMM_WORLD_RANK:-}" ]; then
  [ -n "$hf" ] || { echo "--hf (the DeepSeek-V4.1-Flash checkpoint, for the prefill server) is required" >&2; exit 2; }
  [ -f "$SCRIPT" ] || { echo "run from the repo root" >&2; exit 2; }
  export TT_METAL_HOME=$PWD
  export DEEPSEEK_V4_FABRIC_PAYLOAD=6208
  exec tt-run --rank-binding tests/tt_metal/distributed/config/dual_bh_galaxy_8x4_hosts_40_41_two_mesh_rank_bindings.yaml \
    --mpi-args "--host $PREFILL_HOST,$DECODE_HOST \
      --map-by rankfile:file=$PWD/tests/tt_metal/distributed/config/dual_8x4_hosts_40_41.rankfile \
      --mca btl self,tcp --mca btl_tcp_if_include $MPI_IFACE --bind-to none --tag-output" \
    bash "$PWD/$SCRIPT" --hf "$hf" --prefill-port "$prefill_port" --decode-args "${decode_args[@]}"
fi

cd "$TT_METAL_HOME"
export PYTHONPATH=$TT_METAL_HOME LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib VIRTUAL_ENV=$TT_METAL_HOME/python_env
python=$TT_METAL_HOME/python_env/bin/python

case "$OMPI_COMM_WORLD_RANK" in
  0)
    export BRINGUP_SPEC=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml BRINGUP_HF=$hf
    exec "$python" models/experimental/deepseek_v41_flash/demo/prefill_server.py --port "$prefill_port"
    ;;
  1)
    # (the prefill server listens only after its warm-up: a --prefill-url decode server keeps retrying until then)
    export DEEPSEEK_V41_PREFILL_WAIT=${DEEPSEEK_V41_PREFILL_WAIT:-7200}
    echo "decode server: server.py ${decode_args[*]}"
    exec "$python" models/experimental/deepseek_v41_flash/demo/server.py "${decode_args[@]}"
    ;;
  *)
    echo "rank $OMPI_COMM_WORLD_RANK: this job has two ranks" >&2
    exit 1
    ;;
esac
