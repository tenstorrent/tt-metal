# TT Transformers v2 (TTTv2)

A modular, composable library for implementing transformer models on Tenstorrent hardware.

## Key Features

- **Modular Design**: Core components with minimal dependencies
- **Clear Boundaries**: Model implementations are separate from core library
- **Semantic Versioning**: Stable API with predictable upgrades
- **Hardware Abstraction**: Clean interfaces for TT hardware configuration

## Directory Structure

```
tt_transformers_v2/
├── core/           # Core TTT modules (depends only on TTNN)
├── interfaces/     # Standard interfaces (demos, generators, HW configs)
├── config/         # ML model configuration APIs
├── tests/          # Unit tests for all modules
├── debug/          # Debug support utilities
└── examples/       # Example model implementations
```

## Design Principles

1. **Tightened Scope**: Only core modules are part of TTTv2
2. **Model Independence**: Model implementations use TTTv2 but are not part of it
3. **Minimal Dependencies**: Core modules depend only on TTNN
4. **Extensibility**: Easy to override defaults and customize behavior

## Usage

Model implementations should:
- Import TTTv2 modules as needed
- Override defaults where necessary
- Implement model-specific logic separately
- Pin to specific TTTv2 version for stability

## Version Compatibility

TTTv2 follows semantic versioning:
- Major version changes (2.0 → 3.0): Breaking API changes
- Minor version changes (2.0 → 2.1): Backwards compatible features
- Patch version changes (2.0.0 → 2.0.1): Bug fixes only
