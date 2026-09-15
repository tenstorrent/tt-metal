# Path contract for the harnesses and the runbook (portability)

Goal: a colleague copies the workspace into their own /proj_sw/user_dev/<user>/SDPA and runs
everything without editing a script. Nothing in a script may hardcode another user's home.

## Layout the runbook asks for (same shape as the reference workspace)

    $WORK/                      = /proj_sw/user_dev/<user>/SDPA
      tt-metal/                 clone, branch mvlahovic/analyze_single_chip_sdpa, plus the copied harnesses
      tt-metal-fresh/           clone or worktree at main tip, branch of your own
      polaris/                  clone, branch mvlahovic/roofline_model_sdpa or mvlahovic/roofline_model_topk
      handoff/revamp/           copied from the reference workspace: runners, campaigns, reducers, figs, data

## Variable names, resolution order

Shell scripts resolve in this order and never hardcode a user directory:

    DD        the directory the script itself sits in: ${DD:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
    WORK      ${SDPA_WORK:-<four levels above DD>}          (DD = $WORK/handoff/revamp/data/bh_zones)
    HANDOFF   ${HANDOFF:-$WORK/handoff/revamp}
    TTM       ${TTM:-$WORK/tt-metal}
    TTM_FRESH ${TTM_FRESH:-$WORK/tt-metal-fresh}
    POLARIS   ${POLARIS:-$WORK/polaris}
    PYENV     ${PYENV:-$TTM/python_env/bin/activate}

Python scripts use the same names through os.environ, with the same self-locating default:

    DD      = Path(os.environ.get("DD", Path(__file__).resolve().parent))
    HANDOFF = Path(os.environ.get("HANDOFF", <two or three levels above __file__, per the script's place>))
    WORK    = Path(os.environ.get("SDPA_WORK", <HANDOFF.parent.parent>))
    TTM     = Path(os.environ.get("TTM", WORK / "tt-metal"))
    POLARIS = Path(os.environ.get("POLARIS", WORK / "polaris"))

Rules: the default must reproduce today's behaviour in the reference workspace exactly, so no
existing run changes; every override is an environment variable, never an edited line; a script that
writes output writes it under DD or under a directory derived from HANDOFF, never an absolute path.

## Host and reservation

The card host, the container name and the emulator address are per reservation. Scripts and the
runbook use $(hostname) or a named variable, never the reference host string.
