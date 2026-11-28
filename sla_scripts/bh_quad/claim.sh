#!/bin/bash
# Reserve this Galaxy visibly, so other users and their agents see it before touching the chips. Run ON the Galaxy.
#   claim.sh claim "<who>" "<until>" "<what>"   install everything below
#   claim.sh release                             undo everything (restores tt-smi exactly)
#   claim.sh status
# What a claim installs (all reversible):
#   /tmp/GALAXY_CLAIM            the reservation text (+ ~/CLAIMED_BY_<who>.txt)
#   /etc/profile.d/zz-galaxy-claim.sh   prints the reservation on every interactive login
#   tt-smi reset guard           tt-smi -r / -glx_reset* refuse while claimed unless TT_CLAIM_OVERRIDE=1 (other
#                                tt-smi uses run as normal); original launcher kept as <path>.preclaim
#   /tmp/dsv4_dev.lock           held (flock) so job chains that queue on it wait
#   wall                         one broadcast to logged-in terminals
set -e
C=/tmp/GALAXY_CLAIM
BANNER=/etc/profile.d/zz-galaxy-claim.sh
TTSMI=$(for f in /usr/local/bin/tt-smi /home/ttuser/.local/bin/tt-smi; do [ -e "$f" ] && echo "$f"; done | head -1)
GUARD='if [ -f /tmp/GALAXY_CLAIM ] && [ "$TT_CLAIM_OVERRIDE" != 1 ]; then case " $* " in *" -r"*|*"-glx_reset"*|*" --reset"*) { echo; echo "tt-smi: RESET REFUSED - this Galaxy is reserved:"; cat /tmp/GALAXY_CLAIM; echo "A reset kills the servers running on it. Ask the owner first; to override anyway set TT_CLAIM_OVERRIDE=1."; echo; } >&2; exit 1;; esac; fi'

case "$1" in
claim)
  who=${2:?who}; until=${3:?until}; what=${4:-a multi-Galaxy test}
  printf '%s\n' "*** GALAXY $(hostname -s) RESERVED by $who until $until for $what ***" \
    "*** Running servers hold the chips; a reset (tt-smi -r / -glx_reset) or a new device job will kill them. ***" \
    "*** Please ask $who before using this machine. Claimed $(date '+%Y-%m-%d %H:%M %Z'). ***" > $C
  cp $C /home/ttuser/CLAIMED_BY_$who.txt
  printf '%s\n' '[ -f /tmp/GALAXY_CLAIM ] && case $- in *i*) printf "\033[1;31m"; cat /tmp/GALAXY_CLAIM; printf "\033[0m";; esac' \
    | sudo -n tee $BANNER > /dev/null
  if [ -n "$TTSMI" ] && [ ! -e "$TTSMI.preclaim" ]; then
    cp -p "$TTSMI" "$TTSMI.preclaim" 2>/dev/null || sudo -n cp -p "$TTSMI" "$TTSMI.preclaim"
    if head -1 "$TTSMI" | grep -q bash; then
      { head -1 "$TTSMI.preclaim"; echo "$GUARD"; tail -n +2 "$TTSMI.preclaim"; } > /tmp/tt-smi.claim.$$
    else  # a python entry point: keep it as .preclaim and exec it from a guarded bash wrapper
      printf '#!/bin/bash\n%s\nexec "%s" "$@"\n' "$GUARD" "$TTSMI.preclaim" > /tmp/tt-smi.claim.$$
    fi
    chmod 755 /tmp/tt-smi.claim.$$
    cp /tmp/tt-smi.claim.$$ "$TTSMI" 2>/dev/null || sudo -n cp /tmp/tt-smi.claim.$$ "$TTSMI"
    rm -f /tmp/tt-smi.claim.$$
  fi
  touch /tmp/dsv4_dev.lock
  if ! pgrep -f "flock /tmp/dsv4_dev.lock sleep" > /dev/null; then
    setsid nohup flock /tmp/dsv4_dev.lock sleep 86400 > /dev/null 2>&1 < /dev/null &
  fi
  wall < $C 2>/dev/null || true
  echo "claimed: $(head -1 $C)";;
release)
  if [ -n "$TTSMI" ] && [ -e "$TTSMI.preclaim" ]; then
    cp -p "$TTSMI.preclaim" "$TTSMI" 2>/dev/null || sudo -n cp -p "$TTSMI.preclaim" "$TTSMI"
    rm -f "$TTSMI.preclaim" 2>/dev/null || sudo -n rm -f "$TTSMI.preclaim"
  fi
  sudo -n rm -f $BANNER
  pkill -f "flock /tmp/dsv4_dev.lock sleep" || true
  rm -f $C /home/ttuser/CLAIMED_BY_*.txt /tmp/CLAIMED_BY_*.txt
  echo "released $(hostname -s)";;
status)
  [ -f $C ] && cat $C || echo "not claimed"
  [ -n "$TTSMI" ] && { [ -e "$TTSMI.preclaim" ] && echo "tt-smi guard: on ($TTSMI)" || echo "tt-smi guard: off"; }
  [ -f $BANNER ] && echo "login banner: on" || echo "login banner: off"
  pgrep -f "flock /tmp/dsv4_dev.lock sleep" > /dev/null && echo "dsv4_dev.lock: held" || echo "dsv4_dev.lock: not held";;
*) sed -n 2,13p "$0"; exit 1;;
esac
