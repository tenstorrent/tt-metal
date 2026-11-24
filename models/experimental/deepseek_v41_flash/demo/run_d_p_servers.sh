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
# Run on the prefill host from the repo root, venv active; the decode host has the same checkout, built, at the same
# path, and takes passwordless ssh at DECODE_IP (environment). The decode host's name comes from ssh, and the
# network interface MPI runs over is the one each host routes to the other through; MPI_IFACE (environment)
# overrides it.
# The prefill server reads the default Hugging Face cache
# (~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017).
#
#   DECODE_IP=... [MPI_IFACE=...] \
#   models/experimental/deepseek_v41_flash/demo/run_d_p_servers.sh [--prefill-port 9041] [-- decode server args...]
#
# Prefill logs are tagged [1,0]<stdout>, decode's [1,1]<stdout>. The decode server listens on host 41 (default
# port 8000); the prefill server on host 40 at --prefill-port. Decode args for the KV cache over the fabric:
# -- --prefill-url 10.82.97.40:9041 --kv-transfer-mode fabric (--kv-transfer-mode tcp: the KV over TCP).
# The prefill server logs a per-stage profile of every request; DEEPSEEK_V41_PREFILL_PROFILE_CHUNKS=1 makes its
# prefill split exact (it waits for the device after every chunk).
set -euo pipefail

SCRIPT=models/experimental/deepseek_v41_flash/demo/run_d_p_servers.sh
prefill_port=9041
while [ $# -gt 0 ]; do
  case "$1" in
    --prefill-port) prefill_port="$2"; shift 2 ;;
    # (--decode-args: the ranks' own separator, as tt-run drops a "--" from the command it runs)
    --|--decode-args) shift; break ;;
    *) echo "unknown argument $1 (decode server args go after --)" >&2; exit 2 ;;
  esac
done
decode_args=("$@")

if [ -z "${OMPI_COMM_WORLD_RANK:-}" ]; then
  [ -f "$SCRIPT" ] || { echo "run from the repo root" >&2; exit 2; }
  export TT_METAL_HOME=$PWD
  export DEEPSEEK_V4_FABRIC_PAYLOAD=6208
  [ -n "${DECODE_IP:-}" ] || { echo "set DECODE_IP to the decode host's IP" >&2; exit 2; }
  # The route to the decode host: the interface it leaves through, and this host's address on that interface.
  route=$(ip -o route get "$DECODE_IP") || { echo "no route to DECODE_IP=$DECODE_IP" >&2; exit 2; }
  prefill_iface=$(sed -n 's/.* dev \([^ ]*\).*/\1/p' <<<"$route")
  prefill_src=$(sed -n 's/.* src \([^ ]*\).*/\1/p' <<<"$route")
  # The decode host's name, and the interface it routes back to this host through.
  remote=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$DECODE_IP" bash -s -- "$prefill_src" <<'EOF'
hostname
ip -o route get "$1" | sed -n 's/.* dev \([^ ]*\).*/\1/p'
EOF
  ) || { echo "ssh to DECODE_IP=$DECODE_IP failed (the launch needs passwordless ssh there)" >&2; exit 2; }
  PREFILL_HOST=$(hostname)
  DECODE_HOST=$(sed -n 1p <<<"$remote")
  decode_iface=$(sed -n 2p <<<"$remote")
  if [ -z "${MPI_IFACE:-}" ]; then
    if [ -z "$prefill_iface" ] || [ -z "$decode_iface" ]; then
      echo "no interface between $PREFILL_HOST (${prefill_iface:-?}) and $DECODE_HOST (${decode_iface:-?});" \
        "set MPI_IFACE" >&2
      exit 2
    fi
    # (btl_tcp_if_include takes a list, and each host uses the names it has, so the two may differ)
    MPI_IFACE=$prefill_iface
    [ "$decode_iface" = "$prefill_iface" ] || MPI_IFACE+=",$decode_iface"
  fi
  echo "prefill: $PREFILL_HOST, decode: $DECODE_HOST ($DECODE_IP), MPI interface: $MPI_IFACE"
  # Rank 0 (prefill, mesh 0) on this host, rank 1 (decode, mesh 1) on the decode host.
  rankfile=$(mktemp /tmp/d_p_servers.XXXXXX.rankfile)
  printf 'rank 0=%s slot=0\nrank 1=%s slot=0\n' "$PREFILL_HOST" "$DECODE_HOST" > "$rankfile"
  exec tt-run --rank-binding tests/tt_metal/distributed/config/dual_bh_galaxy_8x4_hosts_40_41_two_mesh_rank_bindings.yaml \
    --mpi-args "--host $PREFILL_HOST,$DECODE_HOST \
      --map-by rankfile:file=$rankfile \
      --mca btl self,tcp --mca btl_tcp_if_include $MPI_IFACE --bind-to none --tag-output" \
    bash "$PWD/$SCRIPT" --prefill-port "$prefill_port" --decode-args "${decode_args[@]}"
fi

cd "$TT_METAL_HOME"
export PYTHONPATH=$TT_METAL_HOME LD_LIBRARY_PATH=$TT_METAL_HOME/build/lib VIRTUAL_ENV=$TT_METAL_HOME/python_env
python=$TT_METAL_HOME/python_env/bin/python

case "$OMPI_COMM_WORLD_RANK" in
  0)
    exec "$python" models/experimental/deepseek_v41_flash/demo/prefill_server.py --port "$prefill_port"
    ;;
  1)
    # (the prefill server listens only after its warm-up: a --prefill-url decode server keeps retrying until then)
    export DEEPSEEK_V41_PREFILL_WAIT=${DEEPSEEK_V41_PREFILL_WAIT:-7200}
    # (--instances 1 sizes its mesh from this rather than asking the control plane before the fabric is set)
    export DEEPSEEK_V4_MESH_SHAPE=${DEEPSEEK_V4_MESH_SHAPE:-8x4}
    echo "decode server: server.py ${decode_args[*]}"
    exec "$python" models/experimental/deepseek_v41_flash/demo/server.py "${decode_args[@]}"
    ;;
  *)
    echo "rank $OMPI_COMM_WORLD_RANK: this job has two ranks" >&2
    exit 1
    ;;
esac
