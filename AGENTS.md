# Codex repository guidance

## Repository overview

This is the tt-metal repository. It contains TT-Metalium, the low-level programming model, and
TTNN, the high-level neural-network operations library for Tenstorrent accelerators. Tensix compute
is tile-oriented: the normal compute tile is 32×32 elements.

Follow the nearest nested `AGENTS.md`, if one exists, for directory-specific rules.

## Architecture essentials

- A Tensix core contains two data-movement RISC-V processors, unpack/math/pack compute processors,
  local L1 SRAM, and NoC interfaces.
- Reader, compute, and writer kernels communicate through circular buffers.
- Reader kernels normally use NCRISC/`RISCV_1` and NoC0; writer kernels normally use
  BRISC/`RISCV_0` and NoC1.
- A stuck NCRISC usually points to the reader. A stuck BRISC usually points to the writer.
- Compute APIs operate on tiles. Row-major data used by FPU/SFPU compute must be tilized first and
  untilized again when row-major output is required.

Read `METALIUM_GUIDE.md` when working below the TTNN Python layer or when hardware behavior is
unclear.

## Build and environment

Activate the repository environment before Python work:

```bash
source python_env/bin/activate
```

Use the standard host build:

```bash
./build_metal.sh
```

Device kernels compile at runtime. Do not rebuild tt-metal merely because a kernel source changed.
Rebuild after host C++ or binding changes when the existing build does not cover them.

## Tests and device safety

Never invoke raw pytest for device tests. Use:

```bash
scripts/run_safe_pytest.sh [--dev] [--run-all] <pytest path or nodeid>
```

The wrapper owns the device lock, dispatch timeout, hang detection, triage, and reset. It stops on
the first failure by default; use `--run-all` only when complete counts are useful. Use `--dev` for
new kernels, suspected hangs, assertions, or detailed watcher/triage output.

Run device commands in the foreground. Never background `run_safe_pytest.sh` or `tt-probe.sh`.
Watch their output; a silent device test may be hung.

For an exploratory device script, use:

```bash
scripts/tt-probe.sh [--dev] <op_name> <<'PYEOF'
import torch
import ttnn
# focused probe
PYEOF
```

This stores the probe under `tests/ttnn/unit_tests/operations/<op_name>/probes/` and gives it the
same device protection as the safe pytest wrapper. Never run raw Python device code.

Never run `tt-smi -r` directly. The wrappers reset the device while holding the correct ownership;
a manual reset can corrupt another process's test.

## Hang triage

On a dispatch timeout, the wrappers write the exact triage path and normally produce:

```text
generated/tt-triage/triage.txt
```

Read it as CSV-formatted text. Start with `dump_callstacks.py` and
`dump_running_operations.py`. Useful signatures include:

- `cb_wait_front`: the consumer waited for tiles that a producer did not push;
- `cb_reserve_back`: the producer waited for space that a consumer did not release;
- `noc_async_read_barrier` or `noc_async_write_barrier`: a transfer did not complete;
- `ASSERT` or `LLK_ASSERT`: a development-mode kernel assertion fired;
- `Kernel Name`, `Go Message`, and `GO`: identify launched, unfinished user kernels.

Ignore command-queue dispatch infrastructure and idle firmware when locating the user kernel.

Standalone triage, when genuinely needed:

```bash
python3 tools/tt-triage.py --llm-output
python3 tools/tt-triage.py --llm-output-path=out.txt
```

## Simulator

Simulator runs use `TT_METAL_SIMULATOR` and `TT_METAL_SLOW_DISPATCH_MODE=1`. The safe wrappers adapt
their locking, reset, and timeout behavior automatically. Never issue a hardware reset in simulator
mode.

## TTNN operation work

For TTNN operation implementation, generalization, performance work, or kernel debugging, use the
repository skill at `.agents/skills/develop-ttnn-operation/`. It contains the technical workflow
and routes to the relevant layout, sharding, precision, blocking, and kernel references.

Keep operation-local focused development tests under `ttnn/ttnn/operations/<op_name>/` when working
inside a goal-driven evaluation. Run them through `scripts/run_safe_pytest.sh`.

For goal-driven operation work, establish correctness and performance on the declared target shapes
first, select a measured performant pattern, and only then generalize that pattern while preserving
the anchor benchmark.

For a goal-driven TTNN implementation/evaluation, read and follow
`.claude/agents_codex/goal-coordinator.md` before changing implementation code. It owns the adaptive
approach, rolling task list, official golden validation, commit protocol, publication, and recovery.
`python3 -m eval.goal_runner` owns deterministic state and database synchronization.

The canonical instruction map is `references/goal_driven/README.md`.

## Repository references

- `METALIUM_GUIDE.md`: Tensix architecture, NoC, kernels, circular buffers, and multi-chip concepts.
- `tech_reports/tensor_accessor/tensor_accessor.md`: logical pages and distributed memory access.
- `tech_reports/tensor_layouts/tensor_layouts.md`: tiled versus row-major and interleaved versus
  sharded layouts.
- `tech_reports/tensor_sharding/`: shard shapes, core grids, and sharded data movement.
- `ttnn/ttnn/operations/examples/master.md`: generic-operation examples and implementation status.

## General safeguards

- Preserve unrelated user changes.
- Never embed developer-specific absolute paths in reusable code, tests, prompts, skills, or docs.
  Discover paths from arguments, environment variables, git, `sys.prefix`, or portable defaults.
- Do not invoke subagents merely because custom TT agents exist. Delegate only when the user
  explicitly authorizes it, and give every write-capable candidate an isolated git worktree.
