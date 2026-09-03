# galaxy-kit shared remote plumbing (sourced by stage.sh/run_bench.sh/collect.sh)
# Route: quietbox -> Mac relay (ssh alias, key-auth) -> exabox slurm login
# (the exabox key lives ONLY in the Mac's qz agent sock) -> srun --overlap.
# All overridable by env:
LK_RELAY=${LK_RELAY:-mac-relay}
LK_LOGIN=${LK_LOGIN:-nkapre@slurm-login.exabox.tenstorrent.com}
LK_AGENT_SOCK=${LK_AGENT_SOCK:-\$HOME/.ssh/qz-exabox-agent.sock}
LK_DEST=${LK_DEST:-/data/nkapre/craq-laneLK}

_mac() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$LK_RELAY" "$@"; }

# run one command string on the exabox login node (NB: macOS has no
# `timeout` — keep remote commands self-terminating)
exa() {
  _mac "export SSH_AUTH_SOCK=$LK_AGENT_SOCK; ssh -o BatchMode=yes -o ConnectTimeout=20 $LK_LOGIN $(printf '%q' "$1")"
}

# stream stdin to a file on the login node (never touches Mac disk —
# the relay has ~100MB free)
exa_put() {  # exa_put <remote-path>
  _mac "export SSH_AUTH_SOCK=$LK_AGENT_SOCK; ssh -o BatchMode=yes -o ConnectTimeout=20 $LK_LOGIN 'mkdir -p $(dirname "$1") && cat > $1'"
}

# stream a remote command's stdout back (for pulling result tarballs)
exa_get() {  # exa_get <remote-command>
  _mac "export SSH_AUTH_SOCK=$LK_AGENT_SOCK; ssh -o BatchMode=yes -o ConnectTimeout=20 $LK_LOGIN $(printf '%q' "$1")"
}

route_check() {
  _mac true 2>/dev/null || {
    echo "REFUSE: relay '$LK_RELAY' unreachable (the Mac must be awake" \
         "on the LAN; see README prerequisites)" >&2
    return 2
  }
  exa "true" >/dev/null 2>&1 || {
    echo "REFUSE: exabox login unreachable through the relay (is the qz" \
         "agent sock alive on the Mac?)" >&2
    return 2
  }
}
