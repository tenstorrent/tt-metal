# Tenstorrent Model Bring-up Rules

## Environment
- ARCH_NAME=wormhole_b0
- PYTHONPATH must include $(pwd) and $(pwd)/models
- Env settings for every command run: export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

## Standards
- PCC > 0.99 is MANDATORY for all ttnn blocks.
- Run `tt-smi -glx_reset` yourself when the device needs to be reset (after hangs/segfaults). Do not ask the user to do it. tt-smi is in `python_env/bin/tt-smi` — always `source python_env/bin/activate` before calling it.
- Follow the "Relay Race" flow: Architecture -> Reference -> TTNN -> Debug -> Opt.

## Session Management
- Read BRINGUP_LOG.md at the start of every session.
- Update BRINGUP_LOG.md before ending a session with: [Status, PCC, Block Hash].
