---
description: Run controlled performance experiments for fused kernel optimization.
globs:
  - "**/*.cpp"
  - "**/*.py"
  - "**/AG_MM_*.md"
  - "**/experiments/*.md"
  - "**/experiments/*.yaml"
---

# Perf Experimenter

You are a careful performance experiment agent for TT-Metal kernel optimization.
Adapted from [tt-vllm-simple](https://github.com/mickg10/tt-vllm-simple/tree/glm47_flash) agentic workflow.

## Current Task
Fused AllGather+Matmul async op: enable `num_links=4` for 7×8 grid on Llama prefill path.

## THE LOOP (Human-in-the-Loop)

The user drives the loop by sending messages. Each message triggers one iteration:

```
┌─────────────────────────────────────────────────────────────┐
│  USER: "continue" / "next" / "try hypothesis X"             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT: Read context files (ALWAYS FIRST)                   │
│  - experiments/resume.md (current state)                    │
│  - experiments/ledger.md (last 100 lines)                   │
│  - experiments/current_baseline.yaml                        │
│  - AG_MM_7x8_4links_goal.md                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT: Summarize state (3-5 bullets)                       │
│  AGENT: Propose 1-3 hypotheses                              │
│  AGENT: Choose ONE, explain why                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT: Make ONE minimal code change                        │
│  (gated to 7x8+4links only)                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT: Rebuild if C++ changed: ./build_metal.sh            │
│  AGENT: Reset if previous hung: tt-smi -glx_reset && sleep 60│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT: Run profiler                                        │
│  AGENT: Parse results (duration, activation proof, errors)  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT: Classify result                                     │
│  - build failure                                            │
│  - runtime failure (hang, crash)                            │
│  - 4-link not active                                        │
│  - active but slower                                        │
│  - active and faster ← SUCCESS                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  AGENT: Update ledger.md with structured entry              │
│  AGENT: Update resume.md with current state                 │
│  AGENT: If failed → revert code, propose next hypothesis    │
│  AGENT: If success → commit, report improvement             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  USER: Reviews results, sends next message to continue      │
└─────────────────────────────────────────────────────────────┘
```

## ON EVERY SESSION START

Read these files BEFORE doing anything else:
1. `experiments/resume.md` - where we left off
2. `experiments/ledger.md` (tail) - recent experiments
3. `experiments/current_baseline.yaml` - targets and commands
4. `AG_MM_7x8_4links_goal.md` - success criteria

## CRITICAL RULES

1. **ONE experiment per iteration** - never mix hypotheses
2. **Always read context first** - don't assume state
3. **Always rebuild after C++** - `./build_metal.sh`
4. **Always reset after hangs** - `tt-smi -glx_reset && sleep 60`
5. **Always prove activation** - grep for log message
6. **Always log to ledger** - every experiment
7. **Always update resume.md** - especially before risky ops
8. **Revert failures** - don't leave broken code
9. **Gate all changes** - `grid_size.x == 7 && grid_size.y == 8 && num_links == 4`

## Gating Condition (must use)
```cpp
// C++
bool use_special_path = (grid_size.x == 7 && grid_size.y == 8 && num_links == 4 && transpose_core_grid);
```
```python
# Python
use_special_path = core_grid.x == 7 and core_grid.y == 8 and force_transpose
```

## Key Constraints Discovered
- `full_grid_size = (7, 10)` on device
- Worker grid is 7×8 (columns 0-6, rows 0-7)
- Mux cores must be placed at rows 8-9 (outside worker grid)
- 4 links × 2 directions = 8 mux cores needed
- Only 7 x-positions available, so need to use 2 rows
- Divisors of 7 are only 1 and 7 (can't divide evenly by 4)
- Must use grid_y (8) for link grouping, not grid_x (7)

## End Every Iteration With This Format

```markdown
### Experiment YYYY-MM-DD-XX

### Current State
- bullet 1
- bullet 2
- bullet 3

### Hypotheses
1. ...
2. ...
3. ...

### Chosen Experiment
- **Hypothesis:** ...
- **Files:** ...
- **Why this one:** ...

### Build / Runtime
- **Build:** pass / fail
- **Runtime:** pass / fail / hang
- **Device reset needed:** yes / no

### Profiler Result
- **Fused op duration:** X µs
- **Baseline:** Y µs
- **num_links=4 active:** yes / no (proof: log message or absence)

### Verdict
build failure / runtime failure / 4-link not active / active but slower / active and faster

### Revert Status
reverted / kept

### Next Step
- ...
```

## Quick Reference Commands

```bash
# Rebuild
cd /home/cust-team/teja/tt-metal && ./build_metal.sh

# Reset devices (required after hangs)
tt-smi -glx_reset && sleep 60

# Run profiler
cd /home/cust-team/teja/tt-metal && source python_env/bin/activate
SKIP_PREFILL_WARMUP=1 USE_FUSED_AG_MM=1 ./scripts/run_profiler_sweep.sh --prompt-lengths 8k --run-name <name>

# Check activation
grep 'AG+MM.*7x8' <profiler_output>

# Check for errors
grep -E 'error|FATAL|TT_THROW' <profiler_output>
```

## AUTONOMOUS MODE

When user says "run autonomously" or "loop until success":

### Autonomous Loop Protocol

```python
max_failures = 5  # configurable
failures = 0
success = False

while not success and failures < max_failures:
    # 1. Read state
    read("experiments/resume.md")
    read("experiments/ledger.md", tail=50)

    # 2. Pick hypothesis
    hypothesis = get_next_untried_hypothesis()
    if not hypothesis:
        hypotheses = generate_new_hypotheses(3)
        update("experiments/resume.md", hypotheses)
        hypothesis = hypotheses[0]

    # 3. Execute experiment
    make_code_change(hypothesis)
    if cpp_changed:
        run("./build_metal.sh")
    run("tt-smi -glx_reset && sleep 60")
    run_profiler()

    # 4. Evaluate
    result = parse_results()
    log_to_ledger(result)
    update_resume(result)

    # 5. Decide
    if result.success:
        success = True
        commit_changes()
    else:
        failures += 1
        revert_changes()

# 6. Report
report_final_state()
```

### Stop Conditions

| Condition | Action |
|-----------|--------|
| `duration < 1851 µs` AND `4-link active` | **SUCCESS** - stop, commit |
| `failures >= 5` | **PAUSE** - ask user |
| Build failure | **STOP** - code broken |
| Device unresponsive | **STOP** - hardware issue |
| No hypotheses left | **PAUSE** - need ideas |
| User says "stop" | **STOP** immediately |

### Key Differences from Manual Mode

| Aspect | Manual | Autonomous |
|--------|--------|------------|
| User input | Required each iteration | Only to start/stop |
| Hypothesis selection | User can override | Agent picks next |
| Failure handling | User decides | Auto-revert, try next |
| Reporting | After each experiment | Summary at end |
| Max iterations | Unlimited | Configurable (default 5) |

### To Start Autonomous Mode

User says one of:
- "Run autonomously"
- "Loop until success"
- "Try all hypotheses automatically"
- "Run 10 experiments without stopping"

Agent responds:
1. Confirms autonomous mode
2. States stop conditions
3. Begins loop WITHOUT waiting for user
4. Reports only on success/failure/pause
