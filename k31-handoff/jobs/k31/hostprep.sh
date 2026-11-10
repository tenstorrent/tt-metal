#!/bin/bash
# hostprep.sh <stage>: per-host step of sc4_job.sh, run once per host through srun.
#   prep : wait out other users' device work, tt-smi health (no reset), uv python for the venv, purge my
#          stale prefill shm and UMD locks, publish this host's ssh key.
#   trust: trust every published key and check ssh to every host in HOSTS_SP.
set -u
S=/data/$USER/jobs/k31 REPO=${REPO:-/data/$USER/tt-metal} me=$(id -u) h=$(hostname -s)
case "$1" in
prep)
  others() {
    ps -eo uid=,user=,etime=,args= | awk -v me="$me" '$1!=me && $1>=1000 && $1<60000 && /[p]ython|[p]ytest|[p]rterun|[p]rted|[t]t-smi|[d]ocker run/' \
      | grep -v -e bmc-inband -e fabric-manager -e serve_wasm.py -e tt_device_mcp.server | cut -c1-200
    local f pid u
    for f in /dev/shm/TT_UMD_LOCK.CHIP_IN_USE_*_PCIe; do
      [ -e "$f" ] || continue
      pid=$(od -An -t d4 -j 52 -N 4 "$f" 2>/dev/null | tr -d " ")
      [ -n "$pid" ] && [ "$pid" -gt 0 ] && u=$(stat -c %u /proc/$pid 2>/dev/null) && [ "$u" != "$me" ] && echo "chip lock $f held by pid $pid uid $u"
    done
  }
  for i in $(seq 1 20); do busy=$(others); [ -z "$busy" ] && break; echo "[prep] $h waiting on other users ($(wc -l <<<"$busy"))"; sleep 30; done
  [ -z "$busy" ] || { echo "[prep] $h ABORT: other users' device work"; echo "$busy"; exit 3; }
  snap=$(timeout 300 tt-smi -s --snapshot_no_tty 2>&1); rc=$?
  if [ $rc -ne 0 ] || grep -q 'Error in detecting devices' <<<"$snap"; then echo "[prep] $h UNHEALTHY: tt-smi rc $rc"; exit 4; fi
  n=$(python3 -c 'import json,sys
s=sys.stdin.read(); d=json.loads(s[s.index("{"):]); print(len(d["device_info"]))' <<<"$snap" 2>/dev/null)
  [ "${n:-32}" = 32 ] || { echo "[prep] $h UNHEALTHY: tt-smi sees $n chips"; exit 4; }
  echo "[prep] $h tt-smi ok: ${n:-?} chips"
  if [ ! -x $REPO/python_env/bin/python ]; then
    export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH
    command -v uv >/dev/null || (cd $REPO && scripts/install-uv.sh >/dev/null) || exit 1
    uv python install "$(sed -n 's/^version_info = //p' $REPO/python_env/pyvenv.cfg)" >/dev/null || exit 1
    [ -x $REPO/python_env/bin/python ] || { echo "[prep] $h venv python missing"; exit 1; }
  fi
  find /dev/shm -maxdepth 1 -user "$me" \( -name '*prefill*' -o -name 'TT_UMD_LOCK*' \) -delete 2>/dev/null
  mkdir -p ~/.ssh $S/keys; chmod 700 ~/.ssh
  [ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519
  cp ~/.ssh/id_ed25519.pub $S/keys/$h.pub
  echo "[prep] $h ok, root disk free $(df -BG --output=avail / | tail -1 | tr -dc 0-9)GB"
  ;;
trust)
  for k in $S/keys/*.pub; do grep -qF "$(cat $k)" ~/.ssh/authorized_keys 2>/dev/null || cat $k >> ~/.ssh/authorized_keys; done
  chmod 600 ~/.ssh/authorized_keys
  for x in $HOSTS_SP; do ssh-keyscan -H $x </dev/null >> ~/.ssh/known_hosts 2>/dev/null; done
  bad=0; for x in $HOSTS_SP; do ssh -n -o BatchMode=yes -o ConnectTimeout=10 $x true || { echo "[trust] $h -> $x FAIL"; bad=1; }; done
  [ $bad = 0 ] && echo "[trust] $h ok"; exit $bad
  ;;
esac
