#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
source /home/ttuser/ornith-9b/lane-env.sh
D="$MODEL_DIR/doc/optimized_multichip_decoder"
label=${1:?recovery label}
if [[ "${STAGE5_LOCK_HELD:-0}" != 1 ]]; then
  exec 9>/home/ttuser/ornith-9b/tt-device-stage5.lock
  flock 9
fi
# Preserve host watcher dumps and kernel map before any reset or smoke can
# replace them. Watcher has already dumped a stopped core before aborting.
mkdir -p "$D/${label}_watcher"
cp -a "$TT_METAL_HOME/generated/watcher/." "$D/${label}_watcher/" 2>/dev/null || true
recovery_started=$(date -u +%s)
record_recovery() {
  recovery_status=$?
  python - "$D" "$label" "$recovery_started" "$recovery_status" <<'PYLOG'
import datetime, json, pathlib, sys, time
art, label, started, status = sys.argv[1:]
start = float(started)
row = dict(label=label, start=datetime.datetime.fromtimestamp(start, datetime.timezone.utc).isoformat(),
           end=datetime.datetime.now(datetime.timezone.utc).isoformat(), seconds=time.time()-start,
           exit_code=int(status), devices=[0,1,2,3], regime="serial", purpose="failed_run_recovery",
           commands=["fuser", "timeout 60 tt-smi -ls --local", "timeout 180 tt-smi -r", "timeout 60 tt-smi -ls --local", "timeout 60 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_mesh_smoke"])
smoke = pathlib.Path(art)/(label+"_clean_close_smoke.log")
if smoke.exists() and not (pathlib.Path(art)/(label+"_reset.log")).exists():
    row["commands"] = ["fuser", "timeout 60 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_mesh_smoke"]
    row["purpose"] = "clean_close_validation_health_check"
with (pathlib.Path(art)/"recovery_jobs.jsonl").open("a") as output:
    output.write(json.dumps(row)+"\n")
PYLOG
}
trap record_recovery EXIT
if fuser /dev/tenstorrent/0 /dev/tenstorrent/1 /dev/tenstorrent/2 /dev/tenstorrent/3 > "$D/${label}_owners.log" 2>&1; then
  printf 'Devices owned; recovery refused\n'
  exit 2
fi
# A cleanly closed validation failure need not reset healthy devices. Prove
# dispatch with a bounded mesh smoke; preserve the original failure either way.
failed_log="$D/${label%_recovery}.log"
if [[ -f "$failed_log" ]] && { rg -q 'Closing devices in cluster completed' "$failed_log" || ! rg -q 'Opening user mode device driver' "$failed_log"; }; then
  if timeout 60 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_mesh_smoke > "$D/${label}_clean_close_smoke.log" 2>&1; then
    printf 'RECOVERY_PASS %s (clean close; mesh smoke; no reset)\n' "$label"
    exit 0
  fi
fi
timeout 60 tt-smi -ls --local > "$D/${label}_before.log" 2>&1
timeout 180 tt-smi -r > "$D/${label}_reset.log" 2>&1
timeout 60 tt-smi -ls --local > "$D/${label}_after.log" 2>&1
timeout 60 python -m models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_mesh_smoke > "$D/${label}_smoke.log" 2>&1
printf 'RECOVERY_PASS %s\n' "$label"
