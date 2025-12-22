# Sweep Framework: Transitive Imports Quick Reference

## Overview
This document provides a quick reference for all transitive imports in the sweep framework files documented in `sweep_framework.md`.

## Import Summary by Count

### Standard Library (27 modules)
`abc`, `argparse`, `builtins`, `click`, `collections`, `contextlib`, `copy`, `csv`, `dataclasses`, `datetime`, `enum`, `hashlib`, `importlib`, `json`, `math`, `multiprocessing`, `os`, `pathlib`, `queue`, `random`, `shutil`, `subprocess`, `sys`, `tempfile`, `time`, `typing`, `yaml`

### Third-Party Libraries (11 packages)
`click`, `enlighten`, `faster_fifo`, `loguru`, `numpy*`, `pandas`, `psycopg2*`, `pydantic`, `torch`, `ttnn`, `yaml`

*Optional dependencies with graceful degradation

**Note:** `elasticsearch` was removed in Phase 0 (December 2025) - no longer supported

### Internal Modules (17 components)

#### Framework Core (`framework/`)
- `device_fixtures` - Device lifecycle management
- `statuses` - Status enums (VectorValidity, TestStatus, VectorStatus)
- `tt_smi_util` - Device reset utilities
- `sweeps_logger` - Centralized logging (wraps loguru)
- `vector_source` - Vector loading (File/Export)
- `result_destination` - Result storage (File/Superset)
- `serialize` - TTNN object serialization
- `permutations` - Parameter space generation
- `database` - PostgreSQL operations
- `upload_sftp` - SFTP file transfer

**Removed:** `elastic_config` (December 2025 - Elasticsearch no longer supported)

#### Utilities (`sweep_utils/`)
- `perf_utils` - Performance measurement
- `roofline_utils` - Roofline model calculations

#### Tracy Profiler (`tools/tracy/`)
- `common` - Tracy constants and paths
- `process_ops_logs` - Device profiler data processing
- `process_device_log` - Device log parsing
- `device_post_proc_config` - Post-processing config

#### Infrastructure (`infra/`)
- `data_collection.pydantic_models` - Pydantic schemas (OpTest, OpRun, PerfMetric, etc.)

## Critical Dependencies

### Must Have (Hard Requirements)
```
ttnn                 # Tenstorrent core library
torch                # PyTorch for tensor operations
loguru               # Logging infrastructure
pydantic             # Data validation
faster_fifo          # IPC queues (process isolation)
enlighten            # Progress bars
```

### Should Have (Functional Degradation)
```
pandas               # Profiler data processing
click                # CLI features
yaml                 # Configuration
```

### Nice to Have (Optional Features)
```
numpy                # Numeric optimizations
psycopg2            # PostgreSQL support
```

## Import Chains (Depth Analysis)

### Deepest Import Chain (Depth: 4)
```
sweeps_runner.py
└─ sweep_utils.perf_utils
   └─ tracy.process_ops_logs
      └─ tracy.process_device_log
```

### Most Common Import Depth (Depth: 2-3)
```
sweeps_runner.py
└─ framework.result_destination
   └─ infra.data_collection.pydantic_models

sweeps_runner.py
└─ framework.vector_source
   └─ framework.statuses
```

## Hub Modules (Most Imported)

### Tier 1: Universal Dependencies
```
framework.sweeps_logger    (10+ importers)
framework.statuses          (5+ importers)
```

### Tier 2: Core Infrastructure
```
ttnn                        (4+ importers)
framework.serialize         (3+ importers)
```

**Removed:** `framework.elastic_config` (December 2025)

## Module Purpose Quick Lookup

| Module | Purpose | Key Exports |
|--------|---------|-------------|
| `sweeps_runner.py` | Test orchestration | `run_sweeps()`, `execute_suite()` |
| `sweeps_parameter_generator.py` | Vector generation | `generate_vectors()`, `generate_tests()` |
| `framework.device_fixtures` | Device management | `default_device()` |
| `framework.statuses` | Status enums | `TestStatus`, `VectorValidity`, `VectorStatus` |
| `framework.vector_source` | Vector loading | `VectorSourceFactory` |
| `framework.result_destination` | Result storage | `ResultDestinationFactory` |
| `framework.serialize` | Object serialization | `serialize()`, `deserialize()` |
| `framework.permutations` | Parameter expansion | `permutations()` |
| `sweep_utils.perf_utils` | Performance measurement | `run_single()`, `run_with_cache_comparison()` |
| `infra.data_collection.pydantic_models` | Data schemas | `OpTest`, `OpRun`, `PerfMetric` |

## Import by Feature

### Vector Generation
```python
sweeps_parameter_generator.py
├── framework.permutations      # Parameter expansion
├── framework.serialize         # Vector serialization
├── framework.statuses          # Status enums
└── framework.sweeps_logger     # Logging
```

### Vector Execution
```python
sweeps_runner.py
├── framework.vector_source          # Load vectors
├── framework.result_destination     # Save results
├── framework.device_fixtures        # Device access
├── sweep_utils.perf_utils          # Performance
└── multiprocessing.Process          # Isolation
```

### Performance Profiling
```python
sweep_utils.perf_utils
├── ttnn                            # Device ops
├── tracy.process_ops_logs          # Profiler data
├── tracy.common                    # Constants
└── sweep_utils.roofline_utils      # Metrics
```

### Data Upload (CI)
```python
framework.result_destination.SupersetResultDestination
└── framework.upload_sftp
    ├── subprocess                   # SFTP command
    └── tempfile                     # Key management
```

## Installation Command Reference

### Minimal Installation (Core Features)
```bash
pip install ttnn torch loguru pydantic faster-fifo enlighten
```

### Full Installation (All Features)
```bash
pip install ttnn torch loguru pydantic faster-fifo enlighten \
            pandas click pyyaml numpy psycopg2-binary
```

### CI Installation (Production)
```bash
pip install -r tests/sweep_framework/requirements-sweeps.txt
```

## Troubleshooting Import Issues

### Issue: "No module named 'ttnn'"
```bash
# ttnn is a proprietary Tenstorrent library
# Follow Tenstorrent installation guide
# Typically: pip install ttnn (from internal source)
```

### Issue: "No module named 'faster_fifo'"
```bash
pip install faster-fifo
# Note: May require C compiler for building
```

### Issue: "psycopg2 not available"
```bash
# Optional - only needed for PostgreSQL features
pip install psycopg2-binary  # Binary distribution (easier)
# OR
pip install psycopg2         # Source distribution (requires pg_config)
```

## Performance Considerations

### Import Time Breakdown (Approximate)
```
ttnn:           1-3 seconds    (device initialization)
torch:          0.5-1 second   (large library)
pandas:         0.3-0.5 second (data structures)
framework.*:    <0.01 second each (pure Python)
```

### Optimization Tips
1. **Preload in parent process**: Import heavy libraries before forking
2. **Lazy imports**: Import `ttnn` only when device is needed
3. **Optional features**: Skip unused dependencies (numpy, psycopg2)

## Related Documentation

- Full analysis: `TRANSITIVE_IMPORTS_ANALYSIS.md`
- Dependency graph: `IMPORT_DEPENDENCY_GRAPH.md`
- Framework overview: `tt-dots/knowledge/tt-metal/sweeps/sweep_framework.md`
- Installation: `requirements-sweeps.txt`

---

**Last Updated**: 2025-12-22
**Coverage**: sweeps_runner.py, sweeps_parameter_generator.py, and all transitive dependencies
