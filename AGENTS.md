# Performance Experiment Workflow

This repo uses a controlled experiment loop for performance optimization.
Adapted from [tt-vllm-simple](https://github.com/mickg10/tt-vllm-simple/tree/glm47_flash) agentic workflow.

## Current Task
Fused AllGather+Matmul async op: enable `num_links=4` for 7×8 grid on Llama prefill path.

## Context Files (READ ON EVERY SESSION START)
| File | Purpose |
|------|---------|
| `AG_MM_7x8_4links_goal.md` | Problem definition and success criteria |
| `AG_MM_7x8_4links_status.md` | Current analysis and experiment history |
| `experiments/current_baseline.yaml` | Baseline numbers, commands, constraints |
| `experiments/ledger.md` (last 100 lines) | Structured experiment log |
| `experiments/resume.md` | Current state for session resumption |

## Optimization Loop Workflow

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
AGENT proposes 1-3 hypotheses
    |
    v
AGENT chooses exactly ONE hypothesis
    |
    v
AGENT explains why this is the best next experiment
    |
    v
AGENT makes ONE minimal code change (gated to 7x8+4links)
    |
    v
AGENT rebuilds tt-metal (if kernel/cpp changed)
    |
    v
AGENT resets devices (if previous run hung): tt-smi -glx_reset && sleep 60
    |
    v
AGENT runs profiler
    |
    v
AGENT parses results:
    - fused op duration from prefill.csv
    - activation proof (grep 'AG+MM.*7x8')
    - any errors (grep 'FATAL|error|TT_THROW')
    |
    v
AGENT classifies result:
    - build failure
    - runtime failure (hang, crash)
    - 4-link not active
    - active but slower
    - active and faster
    |
    v
AGENT updates experiments/ledger.md with structured entry
    |
    v
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

1. **ONE experiment at a time** - never mix multiple hypotheses in one change
2. **Always read context first** - goal, status, baseline, ledger tail, resume
3. **Always rebuild after C++ changes** - `./build_metal.sh`
4. **Always reset after hangs** - `tt-smi -glx_reset && sleep 60`
5. **Always prove activation** - grep for log message, don't assume
6. **Always log to ledger** - every experiment, success or failure
7. **Always update resume.md** - especially before risky operations
8. **Revert failed experiments** - don't leave broken code in tree
9. **Gate all changes** - `grid_size.x == 7 && grid_size.y == 8 && num_links == 4`
10. **Feature-flag new paths** - safe defaults for non-7x8 configs

## Known Failure Modes (from previous experiments)
| Failure | Cause | Error Pattern |
|---------|-------|---------------|
| CoreRangeSet overlap | Mux cores at same position | `Cannot create CoreRangeSet.*overlap` |
| Illegal NOC usage | Mux cores on worker column | `Illegal NOC usage` |
| Not enough mux cores | num_links × 2 > available | `not enough cores for.*mux` |
| Hang during prefetcher | Worker-mux comm broken | Stuck at `TtLlamaPrefetcherSetup` |
| Path not taken | Python params wrong | No `AG+MM.*7x8` in logs |

## Key Files
| File | Purpose |
|------|---------|
| `ttnn/cpp/.../all_gather_minimal_matmul_async_program_factory.cpp` | C++ program factory |
| `models/demos/llama3_70b_galaxy/tt/llama_ccl.py` | Python CCL ops |
| `models/demos/llama3_70b_galaxy/tt/llama_mlp.py` | MLP layer (calls CCL) |

## Quick Reference Commands

```bash
# Rebuild (required after C++ changes)
cd /home/cust-team/teja/tt-metal && ./build_metal.sh

# Reset devices (required after hangs)
tt-smi -glx_reset && sleep 60

# Activate venv
source python_env/bin/activate

# Run profiler
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name <name>

# Check activation
grep 'AG+MM.*7x8' <output>

# Check errors
grep -E 'error|FATAL|TT_THROW' <output>
```

## Autonomous Mode

### Option A: Shell Script (fire and forget)
```bash
# Run up to 10 experiments autonomously
nohup ./experiments/run_loop.sh 10 > experiments/loop.log 2>&1 &

# Check progress
tail -f experiments/loop.log
cat experiments/last_result.txt
```

### Option B: Agent-Driven Autonomous Loop

Tell the agent: **"Run autonomously until success or 5 failures"**

The agent will:
1. NOT wait for user input between iterations
2. Keep looping: hypothesis → change → build → test → log → next
3. Stop when:
   - SUCCESS: fused op beats baseline (< 1851 µs) with 4-links active
   - MAX_FAILURES: 5 consecutive failures (configurable)
   - UNRECOVERABLE: build failure or device brick
   - USER_STOP: user sends "stop" message

### Autonomous Loop Protocol

When user says "run autonomously":

```
WHILE not success AND failures < max_failures:
    1. Read resume.md for current state
    2. Pick next untried hypothesis from resume.md
    3. If no hypotheses left:
       - Generate 3 new hypotheses based on failure patterns
       - Add to resume.md
    4. Make ONE code change
    5. Rebuild (if C++ changed)
    6. Reset devices
    7. Run profiler with timeout (10 min)
    8. Parse results
    9. Log to ledger.md
    10. Update resume.md
    11. If success: STOP, report
    12. If failure: increment counter, revert, continue
    13. If unrecoverable: STOP, report

REPORT final state to user
```

### Stop Conditions

| Condition | Action |
|-----------|--------|
| `fused_duration < 1851 µs` AND `4-link active` | SUCCESS - stop and commit |
| 5 consecutive failures | PAUSE - ask user for guidance |
| Build failure | STOP - code is broken |
| Device unresponsive after reset | STOP - hardware issue |
| All hypotheses exhausted | PAUSE - need new ideas |
| User sends "stop" | STOP immediately |

## End Every Experiment With This Format

```markdown
### Experiment YYYY-MM-DD-XX

### Hypothesis
...

### Files Changed
- ...

### Build
pass / fail

### Runtime
pass / fail / hang

### Profiler
- fused op duration: X µs
- num_links=4 active: yes / no (proof: ...)
- baseline comparison: faster / slower / same

### Verdict
build failure / runtime failure / 4-link not active / active but slower / active and faster

### Revert Status
reverted / kept

### Next Step
...
```
