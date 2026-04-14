# LLK CodeGen - Blackhole

AI-assisted code generation system for Tenstorrent Blackhole Low-Level Kernels (LLK).

## Overview

This system uses Claude AI agents to generate Blackhole LLK kernels by:
1. Analyzing reference implementations (from Wormhole)
2. Planning the Blackhole-specific implementation
3. Generating code following Blackhole conventions
4. Debugging compilation errors
5. Running functional tests

## Quick Start

```bash
cd codegen-bh
claude
> Generate sigmoid for Blackhole
```

## Directory Structure

```
codegen-bh/
├── CLAUDE.md              # Orchestrator instructions
├── README.md              # This file
├── agents/                # Agent playbooks
│   ├── llk-analyzer.md    # Analyze reference implementations
│   ├── llk-arch-lookup.md # Fetch architecture documentation
│   ├── llk-debugger.md    # Fix compilation errors
│   ├── llk-kernel-writer.md # Generate kernel code
│   ├── llk-planner.md     # Plan implementation strategy
│   └── llk-tester.md      # Run functional tests
├── references/            # Knowledge base
│   ├── blackhole-architecture.md  # Blackhole SFPU specifics
│   ├── common-errors.md   # Error patterns and fixes
│   ├── llk-architecture.md # LLK kernel types
│   ├── logging.md         # Logging reference
│   └── porting-guide.md   # WH→BH translation
├── artifacts/             # Generated artifacts
├── config/                # Configuration
└── scripts/               # Tools
```

## Installation

### Install Agents to Claude

Run the install script to copy agents to the local repo's Claude agents directory:

```bash
./scripts/install-agents.sh
```

This copies agent files to `.claude/agents/` in the repository root (with `bh-` prefix).

### Setup Test Environment

```bash
cd ../tests
./setup_testing_env.sh
```

## Usage

### Generate a Kernel

```bash
cd codegen-bh
claude
> Generate gelu for Blackhole
```

### Manual Compilation Check

```bash
cd codegen-bh
source ../tests/.venv/bin/activate
PYTHONPATH=.. python scripts/check_compile.py ../tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_gelu.h -v
```

### Run Tests

```bash
cd codegen-bh
source ../tests/.venv/bin/activate
python scripts/run_functional_test.py gelu -v --arch blackhole
```

## Kernel Types

| Type | Description | Target Path |
|------|-------------|-------------|
| SFPU | Vector operations (sigmoid, exp, etc.) | `tt_llk_blackhole/common/inc/sfpu/` |
| Math | Matrix operations (matmul, reduce) | `tt_llk_blackhole/llk_lib/` |
| Pack | Pack from dest to L1 | `tt_llk_blackhole/llk_lib/` |
| Unpack | Unpack from L1 to src | `tt_llk_blackhole/llk_lib/` |

## Blackhole vs Wormhole

Key differences from Wormhole:
- **Same SFPI library** - Core API is identical
- **More SFPU instructions** - SFPSHFT2, SFPLUTFP32, SFPLE, SFPGT, SFPMUL24, SFPARECIP
- **Enhanced LUT** - 6-piece piecewise linear with `lut2`, `lut2_sign`
- **Macro instructions** - SFPLOADMACRO for complex sequences
- **Replay buffer** - Record and replay instruction sequences

## Reference Architecture

When generating kernels, the system:
1. Analyzes Wormhole reference implementations
2. Maps constructs to Blackhole equivalents
3. Leverages Blackhole-specific optimizations (LUT, macros)

## Agents

| Agent | Purpose |
|-------|---------|
| llk-analyzer | Analyze reference implementations |
| llk-planner | Design implementation strategy |
| llk-kernel-writer | Generate kernel code |
| llk-debugger | Fix compilation errors |
| llk-tester | Run functional tests |
| llk-arch-lookup | Fetch architecture docs from Confluence |

## Troubleshooting

### Compilation Errors

See [references/common-errors.md](references/common-errors.md) for known error patterns.

### Missing Tests

Some kernels may not have functional tests. The tester agent will report when tests are unavailable.

### Architecture Questions

Use the llk-arch-lookup agent to fetch documentation from Confluence:
```
> /agent llk-arch-lookup "SFPU instruction set"
```
