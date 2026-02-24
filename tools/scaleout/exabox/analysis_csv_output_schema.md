# Log Analysis CSV Schema Documentation

This document describes the CSV output schemas produced by the three log analysis scripts. All scripts support a `--csv PATH` flag that writes structured CSV files alongside the normal console/JSON output. These CSVs are designed for ingestion into Superset (or any BI tool) for visualization and dashboarding.

## Common Columns

The following columns appear across all CSV outputs:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | ISO 8601 string | When the analysis was run |
| `log_file` | string | Source log filename (basename only) |
| `domain` | string | Which analyzer produced this row: `dispatch`, `fabric`, or `validation` |

## Dispatch Analysis (`analyze_dispatch_results.py`)

Analyzes a single GTest dispatch test log file. Produces two CSV files.

### `dispatch_test_{run_id}_{run}_summary.csv`

One row per analyzed log file.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Analysis timestamp |
| `log_file` | string | Source log filename |
| `domain` | string | Always `dispatch` |
| `total_processes` | int | Number of MPI processes (GTest instances) |
| `tests_run` | int | Total tests executed |
| `tests_passed` | int | Number of passed tests |
| `tests_failed` | int | Number of failed tests |
| `tests_skipped` | int | Number of skipped tests |
| `warnings_count` | int | Runtime warnings detected |
| `critical_errors_count` | int | Critical errors (segfaults, core dumps) |
| `status` | string | `PASSED` or `FAILED` |

### `dispatch_test_{run_id}_{run}_details.csv`

One row per test result, plus one row per categorized runtime error.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Analysis timestamp |
| `log_file` | string | Source log filename |
| `domain` | string | Always `dispatch` |
| `record_type` | string | `test_result` or `error` |
| `test_name` | string | GTest name (e.g. `TestSuite.TestName`). Empty for `error` rows |
| `test_status` | string | `PASSED`, `FAILED`, or `SKIPPED`. Empty for `error` rows |
| `error_category` | string | Error classification (see Error Categories below) |
| `error_severity` | string | `critical`, `warning`, `error`, or `info` |
| `error_message` | string | Raw error text (truncated to 500 chars) |

## Fabric Analysis (`analyze_fabric_results.py`)

Analyzes a single fabric test log file. Produces two CSV files.

### `fabrich_test_{run_id}_{run}_summary.csv`

One row per analyzed log file.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Analysis timestamp |
| `log_file` | string | Source log filename |
| `domain` | string | Always `fabric` |
| `test_passed` | bool | Whether all hosts passed |
| `num_hosts` | int | Number of MPI hosts detected |
| `warnings_count` | int | Runtime warnings detected |
| `critical_errors_count` | int | Critical errors detected |
| `status` | string | `PASSED` or `FAILED` |

### `fabrich_test_{run_id}_{run}_details.csv`

One row per error category found, with occurrence count.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Analysis timestamp |
| `log_file` | string | Source log filename |
| `domain` | string | Always `fabric` |
| `record_type` | string | Always `error` |
| `error_category` | string | Error classification (see Error Categories below) |
| `error_severity` | string | `critical` or `warning` |
| `error_message` | string | Representative error message (truncated to 500 chars) |
| `error_count` | int | Number of occurrences of this error type |

## Validation Analysis (`analyze_validation_results.py`)

Analyzes a directory of iteration log files. Produces up to four CSV files.

### `validation_test_{run_id}_{run}_summary.csv`

One row per iteration (log file).

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Analysis timestamp |
| `iteration` | int | Iteration number |
| `log_file` | string | Source log filename |
| `domain` | string | Always `validation` |
| `hosts` | string | Comma-separated list of detected hostnames |
| `chips_per_host` | int | Number of chips detected per host |
| `category` | string | Primary result category (human-readable label) |
| `all_categories` | string | Pipe-separated list of all detected categories |
| `is_healthy` | bool | Whether iteration was classified as healthy |

### `validation_test_{run_id}_{run}_aggregate.csv`

Single row summarizing the entire validation run.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Analysis timestamp |
| `domain` | string | Always `validation` |
| `total_iterations` | int | Number of log files analyzed |
| `healthy_count` | int | Iterations classified as healthy |
| `success_rate` | float | Percentage of healthy iterations |
| `timeout_count` | int | Iterations with workload timeouts |
| `timeout_rate` | float | Percentage of timed-out iterations |
| `hosts` | string | Comma-separated list of all detected hosts |
| `chips_per_host` | int | Chips per host |
| `total_chips` | int | Total chips across all hosts |
| `status` | string | `PASSED` or `FAILED` (based on success rate threshold) |

### `validation_test_{run_id}_{run}_faulty_links.csv`

One row per faulty link per iteration. **Only produced when faulty links exist.**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Analysis timestamp |
| `iteration` | int | Iteration number |
| `log_file` | string | Source log filename |
| `domain` | string | Always `validation` |
| `error_category` | string | Always `Unhealthy link` |
| `host` | string | Hostname where link failure occurred |
| `tray` | int | Tray ID |
| `asic` | int | ASIC ID |
| `channel` | int | Channel ID |
| `port_id` | int | Port ID |
| `port_type` | string | Port type (e.g. `TRACE`, `QSFP`) |
| `retrains` | int | Number of link retrains |
| `crc_errors` | int | CRC error count |
| `uncorrected_cw` | int | Uncorrected codeword count |
| `mismatch_words` | int | Data mismatch word count |
| `failure_type` | string | Failure classification from the FAULTY LINKS REPORT |

### `validation_test_{run_id}_{run}_errors.csv`

One row per matched error line per iteration. **Only produced when non-healthy errors exist.**

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | string | Analysis timestamp |
| `iteration` | int | Iteration number |
| `log_file` | string | Source log filename |
| `domain` | string | Always `validation` |
| `error_category` | string | Human-readable category label |
| `error_severity` | string | `error`, `warning`, or `info` |
| `error_message` | string | Matched log line (truncated to 500 chars) |
| `line_number` | int | Line number in the source log file |

## Error Categories

Error categories are extensible -- adding a new type requires only adding an entry to the corresponding `CATEGORIES` or `ERROR_CATEGORIES` dict in the analyzer script. No CSV schema changes are needed.

### Dispatch Error Categories

| Category | Severity | What it detects |
|----------|----------|----------------|
| `test_failed` | error | GTest `[  FAILED  ]` marker |
| `test_skipped` | info | GTest `[  SKIPPED ]` marker |
| `segfault` | critical | Segmentation fault or core dump |
| `signal_abort` | critical | Signal abort (SIGABRT, SIGSEGV) |
| `arc_failure` | warning | ARC core failure |
| `timeout` | warning | Operation timed out |

### Fabric Error Categories

| Category | Severity | What it detects |
|----------|----------|----------------|
| `tt_fatal` | critical | TT_FATAL, segfault, signal abort, critical-level messages |
| `timeout` | warning | Timeout or failed operation warnings |

### Validation Error Categories

| Category Key | Label | Severity | What it detects |
|-------------|-------|----------|----------------|
| `healthy` | Healthy links | -- | All links healthy (success) |
| `unhealthy` | Unhealthy links | error | Faulty links detected |
| `missing_ports` | Missing port connections | info | Missing port/cable connections vs FSD |
| `missing_channels` | Missing channel connections | info | Missing channel connections vs FSD |
| `extra_connections` | Extra connections | warning | Unexpected links not in FSD |
| `workload_timeout` | Workload timeout | warning | Traffic test timed out |
| `dram_failure` | DRAM training failures | error | DRAM training or GDDR issues |
| `arc_timeout` | ARC timeout | warning | ARC communication timeout |
| `aiclk_timeout` | AICLK timeout | error | AICLK failed to settle |
| `mpi_error` | MPI error | error | Lost MPI communication |
| `ssh_error` | SSH error | warning | SSH authentication failure |
| `inconclusive` | Inconclusive | -- | Issues detected but outside known categories |

## Usage Examples

```bash
# Dispatch: analyze single log, produce CSV
python3 analyze_dispatch_results.py test.log --csv results/dispatch.csv

# Fabric: analyze single log, produce CSV
python3 analyze_fabric_results.py test.log --csv results/fabric.csv

# Validation: analyze directory of logs, produce CSV
python3 analyze_validation_results.py validation_output/ --csv results/validation.csv
```

The `--csv` flag can be combined with `--json` (both outputs are produced) or used standalone alongside the default text output.
