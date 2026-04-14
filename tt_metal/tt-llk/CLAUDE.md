# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## LLK CodeGen System

To generate Quasar kernels, start Claude from the `codegen/` folder:
```bash
cd codegen
claude
> Generate gelu for Quasar
```

See [codegen/CLAUDE.md](codegen/CLAUDE.md) for orchestrator details.

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

### Atlassian MCP (Optional)
Pre-configured in `codegen/.mcp.json`. Authenticate on first use.

### Compilation Check
```bash
cd codegen
source ../tests/.venv/bin/activate
PYTHONPATH=.. python scripts/check_compile.py ../tt_llk_quasar/common/inc/sfpu/ckernel_sfpu_{op}.h -v
```

### Run Tests
```bash
cd codegen
source ../tests/.venv/bin/activate
python scripts/run_test.py --list
python scripts/run_test.py {test_name} -v
```

## Key Files

| File | Purpose |
|------|---------|
| `tt_llk_quasar/instructions/assembly.yaml` | Quasar ISA (277KB, use grep) |
| `tt_llk_blackhole/common/inc/sfpu/*.h` | Reference SFPU implementations |
| `tt_llk_quasar/common/inc/sfpu/*.h` | Quasar SFPU implementations |
