# Transitive Imports Analysis: Sweep Framework

This document identifies all transitive imports in the files mentioned in `tt-dots/knowledge/tt-metal/sweeps/sweep_framework.md`.

## Analyzed Files

### Primary Files
1. `tests/sweep_framework/sweeps_runner.py` - Main sweep runner orchestrator
2. `tests/sweep_framework/sweeps_parameter_generator.py` - Vector generation tool
3. `.github/workflows/ttnn-run-sweeps.yaml` - CI workflow (YAML, no Python imports)

## Complete Transitive Import Tree

### 1. sweeps_runner.py

#### Standard Library Imports
- `argparse` - CLI argument parsing
- `builtins` - Python built-in functions
- `contextlib.contextmanager` - Context manager decorator
- `dataclasses.dataclass` - Data class decorator
- `datetime` (as `dt`) - Date/time operations
- `importlib` - Dynamic module importing
- `multiprocessing.Process` - Process management
- `os` - Operating system interface
- `pathlib.Path` - File path operations
- `subprocess` - Process execution
- `sys` - System-specific parameters
- `queue.Empty` - Queue exception handling
- `typing.Optional` - Type hints

#### Third-Party Imports
- `enlighten` - Progress bar management
- `faster_fifo.Queue` - High-performance IPC queues

#### Internal Framework Imports
- `framework.device_fixtures.default_device`
  - Imports: `ttnn`

- `framework.statuses.{VectorValidity, TestStatus}`
  - `enum.Enum`

**Removed:** `framework.elastic_config.*` (December 2025 - Elasticsearch no longer supported)

- `framework.tt_smi_util` (as `tt_smi_util`)
  - `os`
  - `shutil`
  - `subprocess`
  - `time.sleep`
  - `tests.sweep_framework.framework.sweeps_logger.sweeps_logger`

- `framework.sweeps_logger.sweeps_logger` (as `logger`)
  - `os`
  - `sys`
  - `loguru.logger`

- `framework.vector_source.VectorSourceFactory`
  - `abc.{ABC, abstractmethod}`
  - `typing.{List, Dict, Optional, Tuple}`
  - `pathlib`
  - `json`
  - `framework.statuses.VectorStatus`
  - `framework.sweeps_logger.sweeps_logger`

- `framework.result_destination.ResultDestinationFactory`
  - `abc.{ABC, abstractmethod}`
  - `typing.{Optional, Any}`
  - `pathlib`
  - `json`
  - `datetime` (as `dt`)
  - `hashlib`
  - `os`
  - `math`
  - `framework.database.generate_error_hash`
  - `framework.serialize.{serialize, serialize_structured, deserialize, deserialize_structured, convert_enum_values_to_strings}`
  - `framework.sweeps_logger.sweeps_logger`
  - `infra.data_collection.pydantic_models.{OpTest, PerfMetric, TestStatus, OpParam, OpRun, RunStatus}`
  - `framework.upload_sftp.upload_run_sftp`
  - `numpy` (as `np`, optional)

- `framework.serialize.{deserialize, deserialize_vector_structured}`
  - `ttnn`
  - `json`
  - `tests.sweep_framework.framework.sweeps_logger.sweeps_logger`
  - `framework.statuses.{VectorValidity, VectorStatus}`
  - `torch`

- `sweep_utils.perf_utils.{run_with_cache_comparison, run_single}`
  - `subprocess`
  - `shutil`
  - `pathlib.Path`
  - `typing.{Any, Optional, Tuple, Dict}`
  - `framework.sweeps_logger.sweeps_logger`
  - `tracy.common.PROFILER_LOGS_DIR`
  - `tracy.process_ops_logs.get_device_data_generate_report`
  - `sweep_utils.roofline_utils.get_updated_message`
  - `ttnn` (imported at runtime)

### 2. sweeps_parameter_generator.py

#### Standard Library Imports
- `argparse` - CLI argument parsing
- `sys` - System operations
- `importlib` - Dynamic module importing
- `pathlib` - Path operations
- `datetime` - Date/time operations
- `os` - Operating system interface
- `hashlib` - Hash generation
- `json` - JSON serialization
- `random` - Random number generation

#### Internal Framework Imports
- `framework.permutations.*` (wildcard import)
  - No external imports (pure Python logic)

- `framework.serialize.serialize_structured`
  - (See serialize imports above)

- `framework.statuses.{VectorValidity, VectorStatus}`
  - (See statuses imports above)

- `framework.sweeps_logger.sweeps_logger` (as `logger`)
  - (See sweeps_logger imports above)

### 3. framework.database.py

#### Standard Library Imports
- `os`
- `datetime` (as `dt`)
- `json`
- `contextlib.contextmanager`
- `hashlib`

#### Third-Party Imports
- `psycopg2` (optional, with fallback)

#### Internal Imports
- `tests.sweep_framework.framework.sweeps_logger.sweeps_logger`

### 4. framework.upload_sftp.py

#### Standard Library Imports
- `os`
- `pathlib`
- `subprocess`
- `tempfile`
- `shutil`
- `typing.Optional`

#### Internal Imports
- `framework.sweeps_logger.sweeps_logger`

### 5. sweep_utils.roofline_utils.py

#### Third-Party Imports
- `ttnn`
- `loguru.logger`

#### Internal Imports
- `tests.ttnn.utils_for_testing.check_with_pcc`

### 6. tracy Integration

#### tracy.common.py
- `os`
- `shutil`
- `sys`
- `pathlib.Path`
- `loguru.logger`

#### tracy.process_ops_logs.py (first 100 lines analyzed)
- `os`
- `csv`
- `pathlib.Path`
- `json`
- `yaml`
- `datetime`
- `copy`
- `collections.{defaultdict, deque}`
- `typing.{Any, Dict, List, Optional, Set, Tuple}`
- `pandas` (as `pd`)
- `math.{nan, isnan}`
- `click`
- `loguru.logger`
- `tracy.process_device_log.import_log_run_stats`
- `tracy.common.{...}` (multiple constants)
- `tracy.device_post_proc_config`

### 7. infra.data_collection.pydantic_models.py

#### Standard Library Imports
- `datetime`
- `typing.{List, Optional, Union, Tuple}`
- `enum.Enum`

#### Third-Party Imports
- `pydantic.{BaseModel, Field, model_validator}`

## Summary: Complete Dependency Graph

### Python Standard Library (33 unique)
1. `argparse`
2. `builtins`
3. `click`
4. `collections` (defaultdict, deque)
5. `contextlib`
6. `copy`
7. `csv`
8. `dataclasses`
9. `datetime`
10. `enum`
11. `hashlib`
12. `importlib`
13. `json`
14. `math`
15. `multiprocessing`
16. `os`
17. `pathlib`
18. `queue`
19. `random`
20. `shutil`
21. `subprocess`
22. `sys`
23. `tempfile`
24. `time`
25. `typing`
26. `abc`
27. `yaml`

### Third-Party Libraries (11 unique)
1. `enlighten` - Progress bars
2. `faster_fifo` (Queue) - High-performance IPC
3. `loguru` (logger) - Logging
4. `numpy` (optional)
5. `pandas` - Data analysis
6. `psycopg2` (optional) - PostgreSQL
7. `pydantic` (BaseModel, Field, model_validator) - Data validation
8. `torch` - PyTorch
9. `ttnn` - Tenstorrent Neural Network library
10. `click` - CLI framework
11. `yaml` - YAML parser

**Removed:** `elasticsearch` (December 2025 - No longer supported)

### Internal Modules (Sweep Framework)

#### Core Framework (`framework/`)
1. `framework.device_fixtures`
2. `framework.statuses`
3. `framework.tt_smi_util`
4. `framework.sweeps_logger`
5. `framework.vector_source`
6. `framework.result_destination`
7. `framework.serialize`
8. `framework.permutations`
9. `framework.database`
10. `framework.upload_sftp`

**Removed:** `framework.elastic_config` (December 2025)

#### Utilities (`sweep_utils/`)
1. `sweep_utils.perf_utils`
2. `sweep_utils.roofline_utils`

#### Tracy Profiler (`tools/tracy/`)
1. `tracy.common`
2. `tracy.process_ops_logs`
3. `tracy.process_device_log`
4. `tracy.device_post_proc_config`

#### Infrastructure (`infra/`)
1. `infra.data_collection.pydantic_models`

#### Test Utilities
1. `tests.ttnn.utils_for_testing`

#### Dynamic Sweep Modules
- All modules under `tests/sweep_framework/sweeps/**/*.py` (dynamically imported at runtime)

## Import Categories by Purpose

### Device Management
- `ttnn` - Core device operations
- `framework.device_fixtures` - Device lifecycle
- `framework.tt_smi_util` - Device reset utilities

### Data Persistence
- `psycopg2` - PostgreSQL backend (optional)
- `json` - File-based storage (primary)
- `framework.database` - Database abstraction
- `framework.result_destination` - Result storage factory
- `framework.vector_source` - Vector loading factory

### Serialization
- `framework.serialize` - TTNN object serialization
- `json` - JSON operations
- `pickle` (implicit via multiprocessing)

### Performance Measurement
- `sweep_utils.perf_utils` - Performance utilities
- `sweep_utils.roofline_utils` - Roofline model calculations
- `tracy.*` - Tracy profiler integration

### Data Validation & Models
- `pydantic` - Schema validation
- `infra.data_collection.pydantic_models` - Data schemas

### Process Management
- `multiprocessing.Process` - Subprocess isolation
- `faster_fifo.Queue` - IPC
- `subprocess` - External command execution

### Configuration & Logging
- `framework.sweeps_logger` (loguru) - Structured logging
- `argparse` - CLI parsing
- `os` / `pathlib` - File system operations
- `framework.elastic_config` - Elastic configuration

### Data Upload
- `framework.upload_sftp` - SFTP file transfer

## Key Observations

1. **Heavy Third-Party Dependencies**: The framework relies on several specialized libraries:
   - `ttnn` (Tenstorrent's core library)
   - `faster_fifo` (performance-critical IPC)
   - `pydantic` (data validation)

2. **Optional Dependencies**: Some imports are optional with fallbacks:
   - `numpy` (graceful degradation)
   - `psycopg2` (database features optional)

3. **Dynamic Imports**: Sweep modules under `sweeps/` are imported dynamically at runtime

4. **Tracy Profiler Integration**: Deep integration with Tracy profiler for device-level performance measurement

5. **Multi-Backend Support**: The framework supports multiple storage backends:
   - PostgreSQL (optional)
   - File-based (JSON) - primary
   - SFTP upload

6. **Process Isolation**: Uses multiprocessing extensively to isolate test execution and prevent cascading failures

## Installation Requirements

Based on the imports, a complete installation would require:

```bash
# Core dependencies
pip install ttnn torch
pip install faster-fifo
pip install loguru
pip install enlighten
pip install pydantic
pip install click
pip install pyyaml
pip install pandas

# Optional dependencies
pip install numpy
pip install psycopg2-binary  # or psycopg2
```

**Removed:** `elasticsearch` (December 2025 - No longer required)

Note: Some dependencies like `ttnn` are proprietary Tenstorrent libraries and may have special installation procedures.
