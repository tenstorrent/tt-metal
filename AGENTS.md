# TTNN Model Bring-up Workflow

This repo uses a structured agentic experiment loop for model bring-up, refactoring, and performance optimization on Tenstorrent hardware.

## Current Task
GLM-4.7-Flash (`glm4_moe_lite`): refactor monolithic implementation into modular architecture, then optimize per-layer performance.

## Context Files (READ ON EVERY SESSION START)
| File | Purpose |
|------|---------|
| `experiments/goal.md` | Problem definition and success criteria |
| `experiments/status.md` | Current analysis, phase, and progress |
| `experiments/baseline.yaml` | Baseline numbers, commands, constraints |
| `experiments/ledger.md` (last 100 lines) | Structured experiment log |
| `experiments/resume.md` | Current state for session resumption |

## Bring-up Phases

Model bring-up follows a strict phase order. Each phase has clear entry/exit criteria.

```
Phase 1: UNDERSTAND
    Read torch reference, map ops to TTNN equivalents, document architecture.
    Exit: op-mapping table complete, architecture documented.

Phase 2: PROFILE BASELINE
    Profile the existing implementation (or torch reference) to establish baselines.
    Exit: baseline numbers recorded in baseline.yaml.

Phase 3: MODULARIZE / IMPLEMENT
    Implement or refactor the TTNN model layer-by-layer.
    Each layer is one experiment: implement -> test PCC -> log -> next.
    Exit: all layers pass PCC > 0.99 vs torch reference.

Phase 4: OPTIMIZE
    Profile each layer, identify bottlenecks, apply optimizations.
    Each optimization is one experiment: hypothesis -> change -> profile -> log.
    Exit: meets performance target.

Phase 5: INTEGRATE
    Wire into vLLM / serving stack, validate end-to-end.
    Exit: end-to-end demo runs correctly.
```

## The Experiment Loop

```
USER sends message ("continue", "next experiment", etc.)
    |
    v
AGENT reads context files (goal, status, baseline, ledger tail, resume)
    |
    v
AGENT summarizes current state (3-5 bullets)
    |
    v
AGENT proposes 1-3 hypotheses / next steps
    |
    v
AGENT chooses exactly ONE
    |
    v
AGENT explains why this is the best next step
    |
    v
AGENT makes ONE minimal change
    |
    v
AGENT rebuilds tt-metal (if C++ changed): ./build_metal.sh
    |
    v
AGENT runs validation:
    - Phase 3: PCC test vs torch reference
    - Phase 4: Tracy profiler run
    |
    v
AGENT parses results:
    - PCC pass/fail (threshold from goal.md)
    - kernel durations from ops CSV
    - any errors (grep 'FATAL|error|TT_THROW')
    |
    v
AGENT classifies result:
    - build failure
    - runtime failure (hang, crash)
    - PCC failure (regression)
    - PCC pass but slower
    - PCC pass and faster / on target
    |
    v
AGENT updates experiments/ledger.md with structured entry
AGENT updates experiments/resume.md with current state
    |
    v
If FAILED: AGENT reverts changes, proposes next hypothesis
If SUCCESS: AGENT commits, reports improvement
    |
    v
USER sends next message to continue loop
```

## CRITICAL RULES

1. **ONE experiment at a time** - never mix multiple changes in one step
2. **Always read context first** - goal, status, baseline, ledger tail, resume
3. **Always rebuild after C++ changes** - `./build_metal.sh`
4. **Always validate** - PCC test or profiler run after every change
5. **Always log to ledger** - every experiment, success or failure
6. **Always update resume.md** - especially before risky operations
7. **Revert failed experiments** - don't leave broken code in tree
8. **Test existing tests first** - run the test suite before AND after changes
9. **One module at a time** - during refactoring, extract one module, validate, commit, then next

## Validation Commands

```bash
# Rebuild (required after C++ changes)
cd /home/ubuntu/agent/agentic/tt-metal && ./build_metal.sh

# Activate venv
source python_env/bin/activate

# Layer 0 decode (fastest smoke test)
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
  pytest models/demos/glm4_moe_lite/tests/test_tt_decoder_layer0_decode_update_cache_optional.py -v

# MoE layer (validates routed experts)
TT_ENABLE_HW_TESTS=1 TT_ENABLE_LARGE_MODEL_TESTS=1 \
  pytest models/demos/glm4_moe_lite/tests/test_tt_moe_layer1_optional.py -v

# Full model greedy decode
python models/demos/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py --max-new-tokens 4

# Profile pytest test (Tracy ops report -- use -m for pytest)
TT_METAL_DEVICE_PROFILER=1 python -m tracy -v -r -p -n <run_name> -m "pytest <test_path> -v"

# Profile standalone script (no -m flag, pass script path directly)
TT_METAL_DEVICE_PROFILER=1 python -m tracy -v -r -p -n <run_name> <script_path> [args...]
```

## Key Files
| File | Purpose |
|------|---------|
| `models/demos/glm4_moe_lite/tt/model_tt.py` | Top-level model runner |
| `models/demos/glm4_moe_lite/tt/decoder_layer_tt.py` | Decoder layer (attention + MLP) |
| `models/demos/glm4_moe_lite/tt/moe_tt.py` | MoE implementation |
| `models/demos/glm4_moe_lite/tt/layer_weights.py` | Weight conversion |
| `models/demos/glm4_moe_lite/tt/config.py` | Hyperparameters |
| `models/demos/glm4_moe_lite/tt/generator_vllm.py` | vLLM backend |

## Autonomous Mode

Tell the agent: **"Run autonomously until done or 5 failures"**

The agent will:
1. NOT wait for user input between iterations
2. Keep looping: read state -> pick next step -> execute -> validate -> log -> next
3. Stop when:
   - PHASE COMPLETE: all layers pass PCC / perf target met
   - MAX_FAILURES: 5 consecutive failures
   - UNRECOVERABLE: build failure or device brick
   - USER_STOP: user sends "stop"

## Experiment Entry Format

```markdown
### Experiment YYYY-MM-DD-XX

### Phase
understand / profile / modularize / optimize / integrate

### What Changed
- file: description

### Validation
- test: pass / fail
- PCC: X.XXX (threshold: Y.YYY)
- duration: X us (baseline: Y us)

### Verdict
build failure / runtime failure / PCC fail / PCC pass slower / PCC pass on-target

### Revert Status
reverted / kept / committed

### Next Step
...
```

## Adapting for a New Model

To reuse this framework for a different model:

1. Copy `experiments/` directory, clear ledger.md and resume.md
2. Edit `experiments/goal.md` - new model name, torch reference path, success criteria
3. Edit `experiments/baseline.yaml` - target numbers, key files, commands
4. Edit `experiments/status.md` - new op-mapping table
5. Edit `.cursor/agents/ttnn-model-bringup.md` - update globs and model-specific rules
6. Update this AGENTS.md - current task description and key files table
