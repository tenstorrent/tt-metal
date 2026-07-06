# shellcheck shell=bash
# ---------------------------------------------------------------------------
# perf_automation — one bootstrap for bring-up + optimize.
#
# SOURCE this (do not execute) so the exports land in your shell:
#     source models/experimental/perf_automation/setup_env.sh
#
# It self-detects the checkout + build venv (NO hardcoded paths), installs the
# agent pip deps into that venv the first time, exports TT_METAL_HOME /
# PYTHONPATH / PATH, and runs a preflight that fails loudly if anything the
# tool needs is missing. Safe to source every run — the install is skipped once
# the deps import.
#
# Knobs (all optional, override before sourcing):
#     TT_METAL_VENV=python_env      # venv dir name under the checkout root
#     PERF_AUTO_INSTALL=1           # force `pip install -r` even if deps import
# ---------------------------------------------------------------------------

# --- must be sourced, not executed (else exports vanish with the subshell) ---
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this script, don't execute it:" >&2
    echo "         source ${BASH_SOURCE[0]}" >&2
    exit 1
fi

# Don't leave the caller's interactive shell in -e/-u mode; scope our own
# strictness to a function and report via return code instead.
_perf_auto_setup() {
    set -uo pipefail

    local red green yellow reset
    red=$'\033[31m'; green=$'\033[32m'; yellow=$'\033[33m'; reset=$'\033[0m'
    local ok="${green}ok${reset}" fail="${red}FAIL${reset}"
    local self="${BASH_SOURCE[1]}"

    # 1. Locate the checkout root from THIS file's location (git-worktree aware),
    #    never from the caller's cwd or a hardcoded path.
    local script_dir ttm
    script_dir="$(cd "$(dirname "$self")" && pwd)"
    if ! ttm="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)"; then
        # not a git checkout — fall back to walking up from perf_automation/
        ttm="$(cd "$script_dir/../../.." && pwd)"
    fi
    local perf_dir="$ttm/models/experimental/perf_automation"

    # 2. Find the build venv (the ONE interpreter with torch+ttnn+pytest).
    local venv="${TT_METAL_VENV:-python_env}"
    local py="$ttm/$venv/bin/python"
    if [[ ! -x "$py" ]]; then
        echo "  [$fail] build venv not found at: $ttm/$venv/bin/python" >&2
        echo "         set TT_METAL_VENV=<dir> to point at the venv that has torch+ttnn," >&2
        echo "         or build tt-metal first (this is NOT pip-installable)." >&2
        return 1
    fi

    # 3. Export the env the tool relies on. venv FIRST on PATH (it shells out to
    #    bare `python`), then ~/.local/bin so the `claude` CLI resolves.
    export TT_METAL_HOME="$ttm"
    export PYTHONPATH="$ttm${PYTHONPATH:+:$PYTHONPATH}"
    export PATH="$ttm/$venv/bin:$HOME/.local/bin:$PATH"

    # 4. Install agent pip deps into the build venv — only if they don't already
    #    import (idempotent), unless PERF_AUTO_INSTALL forces it.
    local req="$perf_dir/requirements-agent.txt"
    local need_install="${PERF_AUTO_INSTALL:-0}"
    if [[ "$need_install" != "1" ]]; then
        "$py" - <<'PYEOF' >/dev/null 2>&1 || need_install=1
import claude_agent_sdk, tt_perf_report, dotenv  # noqa: F401
PYEOF
    fi
    if [[ "$need_install" == "1" ]]; then
        if [[ ! -f "$req" ]]; then
            echo "  [$fail] requirements file missing: $req" >&2
            return 1
        fi
        echo "  installing agent deps into $venv ..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$py" -r "$req" || { echo "  [$fail] pip install failed" >&2; return 1; }
        else
            "$py" -m pip install -r "$req" || { echo "  [$fail] pip install failed" >&2; return 1; }
        fi
    fi

    # 5. Preflight — every check the tool silently depends on.
    local status=0

    # 5a. all five imports resolve in the ONE interpreter
    if "$py" - <<'PYEOF' >/dev/null 2>&1; then
import torch, ttnn, claude_agent_sdk, tt_perf_report, dotenv  # noqa: F401
PYEOF
        echo "  [$ok] deps import (torch, ttnn, claude_agent_sdk, tt_perf_report, dotenv)"
    else
        echo "  [$fail] one of torch/ttnn/claude_agent_sdk/tt_perf_report/dotenv won't import in $py" >&2
        "$py" -c "import torch, ttnn, claude_agent_sdk, tt_perf_report, dotenv" 2>&1 | sed 's/^/         /' >&2
        status=1
    fi

    # 5b. claude CLI on PATH (the Agent SDK / cc spawns it)
    if command -v claude >/dev/null 2>&1; then
        echo "  [$ok] claude CLI: $(command -v claude)"
    else
        echo "  [$fail] 'claude' CLI not on PATH — cc / the Agent SDK spawns it" >&2
        status=1
    fi

    # 5c. cc auth: native Anthropic — ANTHROPIC_API_KEY, else `claude` login. No .env.agent.
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "  [$ok] auth: ANTHROPIC_API_KEY exported"
    elif [[ -f "$HOME/.claude/.credentials.json" || -f "$HOME/.claude.json" ]]; then
        echo "  [$ok] auth: claude CLI login found"
    else
        echo "  [${yellow}warn${reset}] no ANTHROPIC_API_KEY and no claude login detected — run 'claude' to log in, or export ANTHROPIC_API_KEY" >&2
    fi

    # 5d. tt-smi on PATH or in ~/.tenstorrent-venv (the tool's env-check + device reset need it)
    if command -v tt-smi >/dev/null 2>&1; then
        echo "  [$ok] tt-smi: $(command -v tt-smi)"
    elif [[ -x "$HOME/.tenstorrent-venv/bin/tt-smi" ]]; then
        echo "  [$ok] tt-smi: $HOME/.tenstorrent-venv/bin/tt-smi (tool auto-discovers this)"
    else
        echo "  [$fail] 'tt-smi' not found — install it in its OWN venv: python3 -m venv ~/.tenstorrent-venv && ~/.tenstorrent-venv/bin/pip install tt-smi (NOT the tt-metal venv)" >&2
        status=1
    fi

    # 5e. transformers pinned at 5.10.2 (a stray downgrade to 5.8.1 breaks bring-up)
    local tfv; tfv="$("$py" -c "import transformers; print(transformers.__version__)" 2>/dev/null)"
    if [[ "$tfv" == "5.10.2" ]]; then
        echo "  [$ok] transformers 5.10.2"
    elif [[ -n "$tfv" ]]; then
        echo "  [${yellow}warn${reset}] transformers $tfv (repo pins 5.10.2; if the tool offers 5.8.1, decline / use --no-env-fix)" >&2
    else
        echo "  [${yellow}warn${reset}] transformers not importable" >&2
    fi

    # 5f. tt-lang (ttl) kernel rung — cp312 wheels only; unavailable on py3.10 (fine for bring-up)
    if "$py" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('ttl') else 1)" 2>/dev/null; then
        echo "  [$ok] tt-lang (ttl) importable — optimize kernel rung available"
    else
        local pyver; pyver="$("$py" -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>/dev/null)"
        echo "  [${yellow}note${reset}] tt-lang (ttl) not installed (py$pyver; wheels are cp312-only) — optimize rung-3 off; fine for bring-up" >&2
    fi

    # 5g. optimize's REVERT needs a git baseline; warn if the checkout is a linked
    #     worktree (the known-bad setup for tracy kernel JIT).
    local common_dir
    common_dir="$(git -C "$ttm" rev-parse --git-common-dir 2>/dev/null || echo '')"
    if [[ -n "$common_dir" && "$common_dir" != ".git" && "$common_dir" != "$ttm/.git" ]]; then
        echo "  [${yellow}warn${reset}] this looks like a LINKED git worktree; tracy/optimize needs a" >&2
        echo "         standalone clone (kernel JIT mixes worktree .cpp with main .hpp → no trace)." >&2
    fi

    if [[ "$status" == "0" ]]; then
        echo "  ${green}environment ready${reset}  TT_METAL_HOME=$ttm  venv=$venv"
    else
        echo "  ${red}environment NOT ready — fix the FAIL lines above before running${reset}" >&2
    fi
    return "$status"
}

_perf_auto_setup
