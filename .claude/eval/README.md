# run_eval.sh

Repeatable evaluation harness for Claude Code agentic workflows on tt-metal.

For each run, the script clones the repo fresh, builds it, sets up the environment, and runs Claude headlessly against your prompt(s). Multiple runs let you measure consistency.

## Quick start

```bash
# 1. Write your prompt
mkdir -p eval/prompts
cat > eval/prompts/my_task.txt << 'EOF'
Implement a pointwise sigmoid operation using the /create-op skill.
EOF

# 2. Run from the tt-metal repo root (uses current branch automatically)
./eval/run_eval.sh eval/prompts/my_task.txt
```

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `<prompt_file_or_dir>` | yes | | A single `.txt` prompt file, or a directory of them |
| `--runs N` | no | 1 | Number of independent runs per prompt |
| `--base-dir DIR` | no | `/localdev/$USER` | Where clones and results are stored |

The branch and remote URL are inferred from the current git repo — no `--branch` flag needed.

## What happens per run

1. **Clone** the current branch from the origin remote
2. **Create a unique branch** (`eval/<prompt_name>_run<N>_<timestamp>`)
3. **Init submodules** recursively
4. **Set up environment** in an isolated subshell (mirrors `.envrc` — `TT_METAL_HOME`, `PYTHONPATH`, etc.)
5. **Build** via `./build_metal.sh --enable-ccache`
6. **Create python venv** via `./create_venv.sh --force`
7. **Run Claude** headlessly with `--dangerously-skip-permissions --max-turns 100`
8. **Record result** — PASS, FAIL, or infra failure (BUILD_FAIL / VENV_FAIL)

Each run is fully isolated: own clone, own branch, own env vars, own subshell.

## Running multiple prompts

Point to a directory instead of a single file. Every `.txt` file in that directory becomes a separate eval:

```bash
./eval/run_eval.sh eval/prompts/ --runs 3
```

This runs each prompt 3 times (e.g. 4 prompts x 3 runs = 12 total clone+build+claude invocations).

## Output structure

```
/localdev/$USER/
├── eval_results/
│   └── mstaletovic_NoPlanner_20260302_143000/
│       ├── summary.txt
│       ├── create_reduce_avg/
│       │   ├── run_1/
│       │   │   ├── clone.log
│       │   │   ├── submodules.log
│       │   │   ├── build.log
│       │   │   ├── venv.log
│       │   │   ├── claude_output.json
│       │   │   ├── claude_stderr.log
│       │   │   ├── result.txt
│       │   │   └── duration_seconds.txt
│       │   └── run_2/
│       │       └── ...
│       └── create_sigmoid/
│           └── run_1/
│               └── ...
├── eval_..._create_reduce_avg_run1_.../tt-metal/   # clone for that run
└── eval_..._create_sigmoid_run1_.../tt-metal/
```

## Writing prompts

The prompt file is read as-is and passed to `claude -p`. Write it as you would type into a Claude Code session:

```text
Create a custom pointwise sigmoid operation at
ttnn/ttnn/operations/pointwise_sigmoid/ using the /create-op skill.
Use TILE layout, float32 dtype, and target Wormhole.
```

The prompt runs inside a freshly built tt-metal clone, so Claude has access to the full repo, build artifacts, and python venv.

## Result meanings

| Result | Meaning |
|---|---|
| `PASS` | Claude exited 0 |
| `FAIL (exit N)` | Claude ran but returned non-zero |
| `BUILD_FAIL` | `build_metal.sh` failed before Claude started |
| `VENV_FAIL` | `create_venv.sh` failed before Claude started |

## Notes

- Clones are **not cleaned up** automatically. Delete them manually when done.
- Runs are **sequential**. Each full run (clone + build + claude) can take a while.
- The script exits non-zero if any run failed.
