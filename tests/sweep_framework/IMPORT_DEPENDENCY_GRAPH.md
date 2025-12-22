# Import Dependency Graph: Sweep Framework

## Visual Dependency Tree

```
sweeps_runner.py
├── Standard Library
│   ├── argparse
│   ├── builtins
│   ├── contextlib.contextmanager
│   ├── dataclasses.dataclass
│   ├── datetime (as dt)
│   ├── importlib
│   ├── multiprocessing.Process
│   ├── os
│   ├── pathlib.Path
│   ├── subprocess
│   ├── sys
│   ├── queue.Empty
│   └── typing.Optional
│
├── Third-Party
│   ├── enlighten
│   └── faster_fifo.Queue
│
└── Internal Framework
    ├── framework.device_fixtures
    │   └── ttnn
    │
    ├── framework.statuses
    │   └── enum.Enum
    │
    ├── framework.tt_smi_util
    │   ├── os
    │   ├── shutil
    │   ├── subprocess
    │   ├── time.sleep
    │   └── framework.sweeps_logger
    │       ├── os
    │       ├── sys
    │       └── loguru.logger
    │
    ├── framework.vector_source
    │   ├── abc.{ABC, abstractmethod}
    │   ├── typing.{List, Dict, Optional, Tuple}
    │   ├── pathlib
    │   ├── json
    │   ├── framework.statuses
    │   └── framework.sweeps_logger
    │
    ├── framework.result_destination
    │   ├── abc.{ABC, abstractmethod}
    │   ├── typing.{Optional, Any}
    │   ├── pathlib
    │   ├── json
    │   ├── datetime
    │   ├── hashlib
    │   ├── os
    │   ├── math
    │   ├── numpy (optional)
    │   ├── framework.database
    │   │   ├── os
    │   │   ├── datetime
    │   │   ├── json
    │   │   ├── contextlib.contextmanager
    │   │   ├── hashlib
    │   │   ├── psycopg2 (optional)
    │   │   └── framework.sweeps_logger
    │   │
    │   ├── framework.serialize
    │   │   ├── ttnn
    │   │   ├── json
    │   │   ├── torch
    │   │   ├── framework.sweeps_logger
    │   │   └── framework.statuses
    │   │
    │   ├── infra.data_collection.pydantic_models
    │   │   ├── datetime
    │   │   ├── typing.{List, Optional, Union, Tuple}
    │   │   ├── enum.Enum
    │   │   └── pydantic.{BaseModel, Field, model_validator}
    │   │
    │   ├── framework.upload_sftp
    │   │   ├── os
    │   │   ├── pathlib
    │   │   ├── subprocess
    │   │   ├── tempfile
    │   │   ├── shutil
    │   │   ├── typing.Optional
    │   │   └── framework.sweeps_logger
    │   │
    │   └── framework.sweeps_logger
    │
    └── sweep_utils.perf_utils
        ├── subprocess
        ├── shutil
        ├── pathlib.Path
        ├── typing.{Any, Optional, Tuple, Dict}
        ├── ttnn (runtime import)
        ├── framework.sweeps_logger
        ├── tracy.common
        │   ├── os
        │   ├── shutil
        │   ├── sys
        │   ├── pathlib.Path
        │   └── loguru.logger
        │
        ├── tracy.process_ops_logs
        │   ├── os
        │   ├── csv
        │   ├── pathlib.Path
        │   ├── json
        │   ├── yaml
        │   ├── datetime
        │   ├── copy
        │   ├── collections.{defaultdict, deque}
        │   ├── typing.{Any, Dict, List, Optional, Set, Tuple}
        │   ├── pandas
        │   ├── math.{nan, isnan}
        │   ├── click
        │   ├── loguru.logger
        │   ├── tracy.process_device_log
        │   ├── tracy.common
        │   └── tracy.device_post_proc_config
        │
        └── sweep_utils.roofline_utils
            ├── ttnn
            ├── loguru.logger
            └── tests.ttnn.utils_for_testing.check_with_pcc


sweeps_parameter_generator.py
├── Standard Library
│   ├── argparse
│   ├── sys
│   ├── importlib
│   ├── pathlib
│   ├── datetime
│   ├── os
│   ├── hashlib
│   ├── json
│   └── random
│
└── Internal Framework
    ├── framework.permutations
    │   └── (pure Python - no external imports)
    │
    ├── framework.serialize
    │   └── (see above)
    │
    ├── framework.statuses
    │   └── (see above)
    │
    └── framework.sweeps_logger
        └── (see above)
```

## Circular Dependency Analysis

No circular dependencies detected. The dependency graph is acyclic (DAG).

## Critical Path Analysis

### Longest Import Chain
```
sweeps_runner.py
  → sweep_utils.perf_utils
    → tracy.process_ops_logs
      → tracy.process_device_log
        → (device-specific processing)

Depth: 4 levels
```

### Most Connected Modules (Hub Analysis)

1. **framework.sweeps_logger** (imported by 10+ modules)
   - Central logging facility
   - Used by nearly all framework components

2. **framework.statuses** (imported by 5+ modules)
   - Enum definitions
   - Shared across vector generation and execution

3. **ttnn** (imported by 4+ modules)
   - Core device operations
   - Serialization
   - Performance utilities

4. **framework.serialize** (imported by 3+ modules)
   - Vector serialization/deserialization
   - Result preparation

## Module Coupling Analysis

### Tightly Coupled Modules

1. **result_destination ↔ serialize**
   - Result destinations need serialization for vector data
   - Bidirectional data flow

2. **perf_utils ↔ tracy**
   - Performance measurement requires Tracy profiler
   - Tight integration for device profiling

### Loosely Coupled Modules

1. **permutations** (standalone)
   - Pure functional module
   - No external dependencies

2. **statuses** (enum definitions)
   - Minimal dependencies
   - Pure data structures

3. **upload_sftp** (isolated utility)
   - Self-contained
   - Optional feature

## Import Categories by Layer

### Layer 1: Foundation (No Internal Dependencies)
```
framework.statuses
framework.permutations
framework.sweeps_logger (only depends on loguru)
```

### Layer 2: Core Infrastructure (Depends on Layer 1)
```
framework.device_fixtures
framework.tt_smi_util
framework.serialize
framework.database
framework.upload_sftp
```

**Removed:** `framework.elastic_config` (December 2025)

### Layer 3: Data Management (Depends on Layers 1-2)
```
framework.vector_source
framework.result_destination
infra.data_collection.pydantic_models
```

### Layer 4: Execution Utilities (Depends on Layers 1-3)
```
sweep_utils.roofline_utils
sweep_utils.perf_utils
tracy.common
tracy.process_ops_logs
```

### Layer 5: Orchestration (Depends on All Layers)
```
sweeps_runner.py
sweeps_parameter_generator.py
```

## Optional Import Analysis

### Gracefully Degraded Features
```
numpy (in result_destination.py)
└── Used for: Numeric type handling in hot paths
    Fallback: Pure Python numeric operations

psycopg2 (in database.py)
└── Used for: PostgreSQL database features
    Fallback: Mock objects that raise informative errors
```

## External Service Dependencies

### Network Services
```
PostgreSQL
├── Used by: database.py (optional)
├── Connection: psycopg2
├── Required for: Advanced database features
└── Fallback: File-based storage

SFTP Server
├── Used by: upload_sftp (optional)
├── Connection: SSH/SFTP protocol
├── Required for: CI result uploads
└── Fallback: Local file storage only
```

**Removed:** Elasticsearch (December 2025) - File-based storage (vectors_export, results_export) is now the default

## Import Weight Analysis

### Heavy Imports (>100KB typical size)
```
ttnn           - ~50MB+ (device library)
torch          - ~500MB+ (PyTorch)
pandas         - ~20MB (data analysis)
numpy          - ~20MB (numerical computing)
```

### Medium Imports (10-100KB)
```
pydantic       - ~2MB (validation)
loguru         - ~500KB (logging)
click          - ~800KB (CLI)
enlighten      - ~300KB (progress bars)
yaml           - ~200KB (YAML parser)
```

### Light Imports (<10KB)
```
faster_fifo    - ~100KB (IPC)
psycopg2       - ~500KB (PostgreSQL)
All stdlib     - Negligible (already loaded)
```

## Performance Impact

### Slow Imports (Cold Start)
1. `ttnn` - Device initialization (~1-3s)
2. `torch` - Large library loading (~0.5-1s)
3. `pandas` - Data structures initialization (~0.3-0.5s)

### Fast Imports
1. `framework.*` modules (pure Python) - <10ms each
2. `stdlib` modules - <5ms each
3. `loguru` - ~20ms

## Recommendation: Import Optimization

### Lazy Import Candidates
```python
# In perf_utils.py - import ttnn only when needed
def gather_single_test_perf(device, test_passed):
    import ttnn  # Lazy import here
    ttnn.ReadDeviceProfiler(device)

# In result_destination.py - import numpy only when available
try:
    import numpy as np
except ImportError:
    np = None
```

### Preload Recommendations for CI
```python
# For faster sweep execution, preload heavy libraries
import ttnn  # Preload in parent process
import torch  # Preload in parent process
# Then fork child processes (saves ~2s per test)
```

## Dependency Conflicts

### No Known Conflicts
All dependencies are compatible with each other.

### Version Constraints (Inferred)
```
Python >= 3.8 (f-strings, dataclasses, typing features)
ttnn: Proprietary (version controlled by Tenstorrent)
torch >= 1.0 (modern tensor operations)
pydantic >= 2.0 (model_validator syntax)
loguru >= 0.5 (modern logging features)
```
