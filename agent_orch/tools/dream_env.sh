# Machine-level settings for agent_orch tools. Sourced by the shell tools; every
# value can be overridden from the environment.
: "${DREAM_HOME:=/localdev/$USER/dream}"                         # worktrees, reports, ledger, eval checkout
: "${DREAM_PYENV:=/localdev/$USER/tt-metal/python_env}"          # venv with ttnn deps (editable ttnn is overridden via PYTHONPATH)
: "${DREAM_CPM_CACHE:=/localdev/$USER/tt-metal/.cpmcache}"       # reuse the main checkout's CPM downloads
: "${CCACHE_DIR:=/localdev/$USER/.ccache}"                       # home is NFS and nearly full
: "${DREAM_TRACY_PORT:=8086}"
: "${DREAM_LOCK:=$DREAM_HOME/device.lock}"                      # one build+test at a time on this machine
export DREAM_HOME DREAM_PYENV DREAM_CPM_CACHE CCACHE_DIR DREAM_TRACY_PORT DREAM_LOCK

DREAM_TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DREAM_PY="$DREAM_PYENV/bin/python"
export DREAM_TOOLS DREAM_PY

dream_py() { PYTHONPATH="$DREAM_TOOLS${PYTHONPATH:+:$PYTHONPATH}" "$DREAM_PY" "$@"; }
dream_die() { echo "error: $*" >&2; exit 1; }
