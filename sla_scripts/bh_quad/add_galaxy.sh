#!/bin/bash
# Add one Galaxy to the running quad. Run on the balancer host (which holds the full setup in $J):
#   bash sla_scripts/bh_quad/add_galaxy.sh 10.82.97.44
# Steps: check the target is free -> copy weights, tensor cache, tt-metal and plugin (rsync, resumable, ~10 min)
# -> chip reset + start its server (galaxy_up.sh) -> wait until it serves -> sanity check (cold + cached long prompt)
# -> register it with the balancer (it takes traffic from then on).
# Before the first run, the target needs /mnt/nvme0 -> a directory with ~400 GB free (README "Adding a Galaxy").
# Env: J (default /mnt/nvme0/jerrywang), LB (default http://localhost:9000), STAGE=0 skips copying, RESET=0 skips the
#      chip reset, WEIGHTS (HF snapshot to copy; default: where $J/weights/Llama-3.3-70B-Instruct points).
set -e
IP=${1:?usage: add_galaxy.sh <ip>}
J=${J:-/mnt/nvme0/jerrywang}; LB=${LB:-http://localhost:9000}
WEIGHTS=${WEIGHTS:-$(readlink -f "$J/weights/Llama-3.3-70B-Instruct")}
S() { ssh -o BatchMode=yes -o ConnectTimeout=20 "$IP" "$@"; }
R() { rsync -a --partial -e "ssh -o BatchMode=yes -c aes128-gcm@openssh.com" "$@"; }
log() { echo "$(date -u +%T) $IP $*"; }

S "test -d $J" || { log "no $J on the target: set up storage and the /mnt/nvme0 link first"; exit 1; }
n=$(S "sudo -n fuser /dev/tenstorrent/* 2>/dev/null | wc -w")
[ "$n" = 0 ] || { log "chips busy ($n holders): someone is using this Galaxy"; exit 1; }
curl -sf "$LB/health" > /dev/null || { log "balancer $LB is not up"; exit 1; }

if [ "${STAGE:-1}" = 1 ]; then
  S "mkdir -p $J/weights $J/cache $J/tmp $J/runs $J/jit-cache $J/xdg-cache"
  log "copying weights";      R -L --exclude=original/ "$WEIGHTS/" "$IP:$J/weights/Llama-3.3-70B-Instruct/"
  log "copying tensor cache"; R "$J/cache/llama70b/" "$IP:$J/cache/llama70b/"
  # trailing slashes matter: they anchor /generated/ inside tt-metal (841 GB of profiler output stays behind)
  log "copying tt-metal";     R --exclude=/generated/ --exclude=__pycache__/ --exclude=/.pytest_cache/ \
                                --exclude=/torchinductor_ttuser/ "$J/tt-metal/" "$IP:$J/tt-metal/"
  log "copying plugin";       R "$J/vllm-tt-plugin-gio/" "$IP:$J/vllm-tt-plugin-gio/"
fi

log "pre-flight: python env"
S "$J/tt-metal/python_env/bin/python -c 'import ttnn'" > /dev/null 2>&1 || {
  log "python_env does not run there (its uv interpreter link may be missing: compare"
  log "~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu with this host)"; exit 1; }
log "reset + start"
S "J=$J RESET=${RESET:-1} START=1 bash $J/tt-metal/sla_scripts/bh_quad/galaxy_up.sh"
log "waiting for the server (first start compiles kernels, ~25 min)"
until curl -sf -m 5 "http://$IP:8000/v1/models" > /dev/null; do
  # anchored: an unanchored pattern also matches this ssh command's own shell and never sees the server die
  S "pgrep -f '^bash sla_scripts/bh_quad/serve.sh' > /dev/null" || { log "server died: see $J/runs/serve_*/server.log there"; exit 1; }
  sleep 30
done
log "sanity check"
python "$J/tt-metal/sla_scripts/bh_tp32/sanity_check.py" "http://$IP:8000" || { log "failed sanity: NOT added"; exit 1; }
curl -s -X POST "$LB/lb/backend" -H "Content-Type: application/json" -d "{\"url\": \"http://$IP:8000\", \"add\": true}"
echo
log "added to the balancer"
