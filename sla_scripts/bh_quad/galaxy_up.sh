#!/bin/bash
# Bring up one Galaxy's server for the quad (runs ON that Galaxy, from tt-metal/).
#   RESET=1   chip-level reset first (tt-smi -r). It resets only the Tenstorrent chips; the NVMe drives are untouched.
#             Do NOT use tt-smi -glx_reset_auto on shared hosts: its IPMI tray reset re-probes every NVMe drive on
#             PCIe root complex 00, and mounts there (home directories, other users' drives) return I/O errors
#             until remounted.
#   START=1   start serve.sh in the background (port 8000). Log: $J/runs/serve_<host>/server.log
# Refuses to touch chips another process holds. Env: J (work dir, default /mnt/nvme0/jerrywang).
J=${J:-/mnt/nvme0/jerrywang}
TTSMI=$(command -v tt-smi || echo /home/ttuser/.local/bin/tt-smi)
n=$(sudo -n fuser /dev/tenstorrent/* 2>/dev/null | wc -w)
[ "$n" = 0 ] || { echo "chips busy ($n holders), not touching them"; exit 1; }
if [ "${RESET:-0}" = 1 ]; then
  sync
  # TT_CLAIM_OVERRIDE: this is the claim holder's own reset (claim.sh guards tt-smi resets against everyone else)
  (cd /tmp && TT_CLAIM_OVERRIDE=1 "$TTSMI" -r > "$J/runs/tt_smi_r.log" 2>&1) || { echo "tt-smi -r failed, see $J/runs/tt_smi_r.log"; exit 1; }
  echo "chip reset done"
fi
[ -d "$J/tt-metal/sla_scripts/bh_quad" ] || { echo "$J/tt-metal missing"; exit 1; }
if [ "${START:-0}" = 1 ]; then
  cd "$J/tt-metal"
  setsid nohup bash sla_scripts/bh_quad/serve.sh > "$J/runs/serve.out" 2>&1 < /dev/null &
  echo "server starting (pid $!), log $J/runs/serve_$(hostname -s)/server.log"
fi
