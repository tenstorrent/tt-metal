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

## Skills (Relay Race Workflow)

Use these skills to follow the model bring-up workflow:

| Skill | Command | Purpose |
|-------|---------|---------|
| Architecture | `/architecture` | Map model blocks to existing TTNN implementations, create ARCHITECTURE.md |
| Reference | `/reference` | Create standalone PyTorch reference modules for golden outputs |
| TTNN | `/tt` | Implement model blocks in TTNN, verify PCC > 0.99 |
| Debug | `/debug` | Diagnose PCC failures, device hangs, and other issues |
| Optimization | `/optimization` | Add tracing, memory optimization, and op fusion |

### Workflow Example
```
/architecture   # 1. Analyze model, find similar implementations
/reference      # 2. Create PyTorch reference, verify against HuggingFace
/tt             # 3. Implement in TTNN, achieve PCC > 0.99
/debug          # 4. Fix any issues
/optimization   # 5. Add tracing for performance
```
