#!/usr/bin/env bash
set -uo pipefail
cd /home/tt-admin/sdawle/pi05_openpi_upstream_bh_glx_trace/tt-metal
source _bench_runs/pi05_production.env 2>/dev/null || true
export TT_METAL_HOME="$PWD"
export TT_VISIBLE_DEVICES="$(seq -s, 0 31)"
export PYTHONPATH="$PWD:/home/tt-admin/pi05_cache/libero_repo"
export PROBE_FABRIC="${PROBE_FABRIC:-1d}"
export PROBE_ROW="${PROBE_ROW:-1}"
python_env/bin/python -u _bench_runs/probe_socket_d2d.py
echo "SOCKPROBE_EXIT=$?"
