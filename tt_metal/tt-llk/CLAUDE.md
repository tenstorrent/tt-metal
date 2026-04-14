# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Policy: Read-Only

Read-only git commands are allowed (`git rev-parse`, `git log`, `git status`, `git diff`, `git show`). **NEVER push, commit, checkout, restore, reset, or otherwise modify** the repo via git.

## LLK CodeGen System

To generate kernels for a target architecture, start Claude from the `codegen/` folder:
```bash
cd codegen
claude
> Generate gelu for Quasar
```

The codegen system discovers architectural patterns dynamically from Confluence, DeepWiki, assembly.yaml, and existing implementations. See [codegen/CLAUDE.md](codegen/CLAUDE.md) for orchestrator details.

## Repository Structure

```
tt-llk/
├── tt_llk_quasar/           # Quasar LLK implementations
├── tt_llk_blackhole/        # Blackhole LLK (reference)
├── tt_llk_wormhole_b0/      # Wormhole LLK (reference)
├── codegen/                 # AI code generation system
│   ├── CLAUDE.md            # Orchestrator instructions
│   ├── agents/              # Agent playbooks
│   ├── references/          # Knowledge base
│   ├── artifacts/           # Generated artifacts
│   └── scripts/             # Tools
└── tests/                   # Test infrastructure
```

## Commands

### Environment Setup
```bash
cd tests
./setup_testing_env.sh
```

### MCP Servers
Pre-configured in `.mcp.json`. Atlassian requires authentication on first use.
- **Atlassian** — Primary source for architecture knowledge. See `codegen/agents/llk-arch-lookup.md` for full page index. Key pages:
  - Page ID `48300268` — uarch tree root (80+ sub-pages with detailed microarchitecture)
  - Page ID `1256423592` — Quasar/Trinity SFPU Micro-Architecture Spec (primary SFPU reference)
  - Page ID `1170505767` — Tensix SFPU Instruction Set Architecture (per-instruction details)
  - Page ID `1613201604` — Tensix ISA (164 child pages, one per instruction)
  - Page ID `84508873` — Tensix NEO High Level Specification (general architecture overview)
  - Page ID `1612808713` — REPLAY instruction (replay buffer for ITERATIONS loops)
- **DeepWiki** — Query `tenstorrent/tt-isa-documentation` for reference architecture ISA docs

### Compilation Check
```bash
cd codegen
source ../tests/.venv/bin/activate
PYTHONPATH=.. python scripts/check_compile.py ../tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_{op}.h -v
```

### Run Tests
```bash
source tests/.venv/bin/activate
cd tests/python_tests/quasar
TT_UMD_SIMULATOR_PATH=/proj_sw/user_dev/vvukomanovic/tt-umd-simulators/build/emu-quasar-1x3 CHIP_ARCH=quasar pytest -x --run-simulator --port=5556 test_{op}_quasar.py
```

## Key Files

| File | Purpose |
|------|---------|
| `tt_llk_quasar/instructions/assembly.yaml` | Quasar ISA (277KB, use grep) |
| `tt_llk_blackhole/common/inc/sfpu/*.h` | Reference SFPU implementations |
| `tt_llk_quasar/common/inc/sfpu/*.h` | Quasar SFPU implementations |
