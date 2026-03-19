# Tenstorrent Model Bring-up Rules

## Environment
- ARCH_NAME=wormhole_b0
- PYTHONPATH must include $(pwd) and $(pwd)/models
- Env settings for every command run: export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

## Standards
- PCC > 0.99 is MANDATORY for all ttnn blocks.
- You can reset the device yourself using `tt-smi -glx_reset` (after hangs/segfaults); do not ask the user to run the reset.
- Follow the "Relay Race" flow: Architecture -> Reference -> TTNN -> Debug -> Opt.

## Session Management
- Read BRINGUP_LOG.md at the start of every session.
- Update BRINGUP_LOG.md before ending a session with: [Status, PCC, Block Hash].
