# Logging with Loguru

The test infrastructure uses [loguru](https://github.com/Delgan/loguru) for structured logging.

## Quick start

```python
from helpers.logger import logger

logger.info("Test started for op={}", op_name)
logger.debug("Tile dims: {}x{}", rows, cols)
logger.warning("Skipping unsupported format: {}", fmt)
logger.error("Mismatch at index {}: expected={}, got={}", i, exp, got)
```

Loguru uses `{}` placeholders (not `%s`), and the arguments are only
formatted if the message is actually emitted, so there is zero overhead at
levels that are filtered out.

## Log levels

From most to least verbose:

| Level | Typical use |
|----------|----------------------------------------------|
| TRACE | Very fine-grained internals |
| DEBUG | Diagnostic details (configs, paths, sizes) |
| INFO | General progress ("compiling", "running") |
| SUCCESS | Positive outcomes |
| WARNING | Unexpected but non-fatal conditions |
| ERROR | Failures worth investigating |
| CRITICAL | Unrecoverable errors |

## Setting the log level

**Priority order** (highest first):

1. `--loguru-level` pytest CLI option
2. `LOGURU_LEVEL` environment variable
3. Default: `INFO`

```bash
# CLI option (highest priority)
pytest --loguru-level=DEBUG tests/

# Environment variable
LOGURU_LEVEL=TRACE pytest tests/

# CI sets it via workflow input (defaults to WARNING)
```

## Output destinations

| Destination | File | Behaviour |
|-------------|------|-----------|
| Session log | `test_run.log` | Overwritten each run (`mode="w"`) |
| Error log | `test_errors.log` | Appended across runs (`mode="a"`, ERROR+ only) |
| Terminal | via pytest live logging | Shown when `--loguru-level` is set |

## pytest-xdist (parallel workers)

When running with `-n N`, each worker writes to its own log files using the
`PYTEST_XDIST_WORKER` env var (set automatically by xdist):

```
test_run_gw0.log   test_errors_gw0.log
test_run_gw1.log   test_errors_gw1.log
...
test_run_gw9.log   test_errors_gw9.log
```

The controller process (if any) uses the base filenames without a suffix.
CI uploads all matching files via `test_run*.log` / `test_errors*.log` globs.

## Examples

```python
from helpers.logger import logger

# f-string style placeholders with lazy formatting
logger.info("Compiling {} variants for {}", len(variants), arch)

# Structured data
logger.debug("Config: {}", config.__dict__)

# Exception logging (loguru captures the traceback automatically)
try:
    run_test(params)
except Exception:
    logger.exception("Test failed for params={}", params)
```
