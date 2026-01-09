# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import hashlib
import json
import math
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any

from framework.database import generate_error_hash
from framework.serialize import (
    convert_enum_values_to_strings,
    deserialize,
    deserialize_structured,
    serialize,
    serialize_structured,
)
from framework.sweeps_logger import sweeps_logger as logger
from framework.upload_sftp import upload_run_sftp

from infra.data_collection.pydantic_models import (
    OpParam,
    OpRun,
    OpTest,
    PerfMetric,
    RunStatus,
    TestStatus,
)

# Optional numpy import for numeric handling in hot paths
try:
    import numpy as np
except ImportError:
    np = None


# --- Metric extraction helpers (module-private) ---
def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_device_metric_name(key: str, suffix: str | None = None) -> str:
    base = key if key.startswith("device_") else f"device_{key}"
    return f"{base}{suffix or ''}"


def _add_metric(metrics: set, name: str, value: Any) -> None:
    v = _to_float(value)
    if v is not None:
        metrics.add(PerfMetric(metric_name=name, metric_value=v))


def _add_device_perf_from_dict(metrics: set, perf: dict, suffix: str | None = None) -> None:
    for k, v in perf.items():
        _add_metric(metrics, _normalize_device_metric_name(k, suffix), v)


def _map_status(value: Any) -> TestStatus | None:
    if value is None:
        return None
    try:
        from framework.statuses import TestStatus as RunnerStatus

        # TODO: consider removing this mapping and just using the TestStatus enum directly
        if isinstance(value, RunnerStatus):
            mapping = {
                RunnerStatus.PASS: "pass",
                RunnerStatus.FAIL_ASSERT_EXCEPTION: "fail_assert_exception",
                RunnerStatus.FAIL_CRASH_HANG: "fail_crash_hang",
                RunnerStatus.NOT_RUN: "skipped",
                RunnerStatus.FAIL_L1_OUT_OF_MEM: "fail_l1_out_of_mem",
                RunnerStatus.FAIL_WATCHER: "fail_watcher",
                RunnerStatus.FAIL_UNSUPPORTED_DEVICE_PERF: "fail_unsupported_device_perf",
                RunnerStatus.XFAIL: "xfail",  # Expected failure
                RunnerStatus.XPASS: "xpass",  # Unexpected pass
            }
            return TestStatus(mapping.get(value, "error"))
    except Exception:
        pass
    # If already a string, trust but verify
    try:
        s = str(value)
        # Normalize common forms like "TestStatus.PASS"
        if s.startswith("TestStatus."):
            suffix = s.split(".", 1)[1].lower()
            if suffix == "pass":
                return TestStatus("pass")
            return TestStatus(suffix)
        return TestStatus(s)
    except Exception:
        return TestStatus("error")


def _collect_all_metrics(raw: dict[str, Any]) -> set[PerfMetric] | None:
    """Collect both e2e performance and device performance metrics into PerfMetric set"""
    metrics: set[PerfMetric] = set()

    # Collect e2e and device metrics via helpers
    _add_e2e_metrics(metrics, raw)
    _add_device_metrics(metrics, raw)
    _add_memory_metrics(metrics, raw)

    return metrics if metrics else None


def _coerce_to_optional_string(value: Any) -> str | None:
    """Convert any value to an optional string, handling common numeric types gracefully."""
    if value is None:
        return None
    if isinstance(value, str):
        return value

    # Handle numpy numeric types first (before checking for regular float/int)
    if np is not None and isinstance(value, np.number):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return str(value)

    # Handle regular Python numeric types
    if isinstance(value, (int, float)):
        # Handle special float cases
        if isinstance(value, float):
            if math.isnan(value):
                return None
            if math.isinf(value):
                return "inf" if value > 0 else "-inf"
        return str(value)

    # For any other type, convert to string
    return str(value)


def _add_e2e_metrics(metrics: set, raw: dict[str, Any]) -> None:
    e2e_perf = raw.get("e2e_perf")
    if e2e_perf is not None:
        if isinstance(e2e_perf, dict):
            _add_metric(metrics, "e2e_perf_uncached_ms", e2e_perf.get("uncached"))
            _add_metric(metrics, "e2e_perf_cached_ms", e2e_perf.get("cached"))
        else:
            _add_metric(metrics, "e2e_perf_ms", e2e_perf)

    # Also capture explicit fields when present (back/forward compat)
    _add_metric(metrics, "e2e_perf_uncached_ms", raw.get("e2e_perf_uncached"))
    _add_metric(metrics, "e2e_perf_cached_ms", raw.get("e2e_perf_cached"))


def _add_device_metrics(metrics: set, raw: dict[str, Any]) -> None:
    device_perf_raw = raw.get("device_perf")
    if device_perf_raw is not None:
        if isinstance(device_perf_raw, dict) and ("cached" in device_perf_raw or "uncached" in device_perf_raw):
            uncached_perf = device_perf_raw.get("uncached")
            if isinstance(uncached_perf, dict):
                _add_device_perf_from_dict(metrics, uncached_perf, suffix="_uncached")

            cached_perf = device_perf_raw.get("cached")
            if isinstance(cached_perf, dict):
                _add_device_perf_from_dict(metrics, cached_perf, suffix="_cached")
        else:
            # Original structure - single dict or list of dicts
            if isinstance(device_perf_raw, list):
                device_perf_raw = next((d for d in device_perf_raw if isinstance(d, dict)), None)
            if isinstance(device_perf_raw, dict):
                _add_device_perf_from_dict(metrics, device_perf_raw)

    # Also accept separate fields when provided
    device_perf_uncached = raw.get("device_perf_uncached")
    if isinstance(device_perf_uncached, dict):
        _add_device_perf_from_dict(metrics, device_perf_uncached, suffix="_uncached")

    device_perf_cached = raw.get("device_perf_cached")
    if isinstance(device_perf_cached, dict):
        _add_device_perf_from_dict(metrics, device_perf_cached, suffix="_cached")


def _add_memory_metrics(metrics: set, raw: dict[str, Any]) -> None:
    """Extract memory metrics from result dict and add to metrics set"""

    # Per-core memory metrics from extract_resource_usage_per_core
    _add_metric(metrics, "peak_l1_memory_per_core_bytes", raw.get("peak_l1_memory_per_core"))
    _add_metric(metrics, "peak_cb_per_core_bytes", raw.get("peak_cb_per_core"))
    _add_metric(metrics, "peak_l1_buffers_per_core_bytes", raw.get("peak_l1_buffers_per_core"))
    _add_metric(metrics, "num_cores", raw.get("num_cores"))

    # Aggregate and device-level metrics
    _add_metric(metrics, "peak_l1_memory_aggregate_bytes", raw.get("peak_l1_memory_aggregate"))
    _add_metric(metrics, "peak_l1_memory_device_bytes", raw.get("peak_l1_memory_device"))


class ResultDestination(ABC):
    """Abstract base class for test result destinations"""

    @abstractmethod
    def initialize_run(self, run_metadata: dict[str, Any]) -> str | None:
        """Initialize a new test run and return run_id if applicable"""
        pass

    @abstractmethod
    def export_results(self, header_info: list[dict], results: list[dict], run_context: dict[str, Any]) -> str:
        """Export test results and return status"""
        pass

    @abstractmethod
    def finalize_run(self, run_id: str | None, final_status: str) -> None:
        """Finalize the test run"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the destination is accessible"""
        pass


class FileResultDestination(ResultDestination):
    """File-based result destination (JSON export)"""

    def __init__(self, export_dir: pathlib.Path | None = None):
        if export_dir is None:
            self.export_dir = pathlib.Path(__file__).parent.parent / "results_export"
        else:
            self.export_dir = export_dir
        # In-memory aggregation for building OpRun at finalize_run
        self._run_metadata: dict[str, Any] | None = None
        self._collected_tests: list[dict[str, Any]] = []
        self._run_id: str | None = None

    def initialize_run(self, run_metadata: dict[str, Any]) -> str | None:
        """Prepare export directory and initialize run aggregation context."""
        if not self.export_dir.exists():
            self.export_dir.mkdir(parents=True)

        self._run_metadata = run_metadata
        # Generate a simple deterministic run id based on host and start timestamp
        try:
            host = str(run_metadata.get("host", "unknown"))
            # Use a short digest of run_contents to prevent overly long filenames
            run_contents = str(run_metadata.get("run_contents", "unknown"))
            digest = hashlib.sha256(run_contents.encode("utf-8")).hexdigest()[:12]
            start_ts = (
                run_metadata.get("start_time_ts")
                or run_metadata.get("run_start_ts")
                or dt.datetime.now(dt.timezone.utc)
            )
            ts_str = start_ts.strftime("%Y%m%d_%H%M%S")
            # Sanitize host to avoid path issues and keep the filename short
            safe_host = host.replace("/", "_")[:32]
            self._run_id = f"{safe_host}_{digest}_{ts_str}"
        except Exception:
            self._run_id = None

        # Reset any previously collected tests
        self._collected_tests = []

        return self._run_id

    def export_results(self, header_info: list[dict], results: list[dict], run_context: dict[str, Any]) -> str:
        """Export results to JSON file using Pydantic validation (OpTest)."""
        if not results:
            return "success"

        sweep_name = header_info[0]["sweep_name"]
        run_start_time = run_context.get("test_start_time")
        if run_start_time:
            timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
        else:
            # Fallback to current time if run_start_time is not available
            timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Keep filenames short and safe: use sweep short name + digest
        short_sweep = str(sweep_name).split(".")[0] if sweep_name else "sweep"
        name_digest = hashlib.sha256(str(sweep_name).encode("utf-8")).hexdigest()[:12] if sweep_name else "na"
        export_path = self.export_dir / f"{short_sweep}_{name_digest}_{timestamp}.json"

        git_hash = run_context.get("git_hash", "unknown")

        # this will be the list of OpTest objects that will be exported to the file
        validated_records = []

        # Map internal TestStatus enum (or strings) to file schema enum values via module helper

        for i in range(len(results)):
            header = header_info[i]
            raw = results[i]

            mapped_status = _map_status(raw.get("status"))
            is_success = mapped_status in (TestStatus.success, TestStatus.passed)
            is_skipped = mapped_status == TestStatus.skipped

            # Convert original vector data into a JSON-friendly dict suitable for JSONB/file storage
            normalized_vector = _normalize_original_vector_data(raw.get("original_vector_data")) or {}
            # Flatten nested structure into a single-level dict with dotted keys
            flattened_vector = _flatten_any_to_dotted(normalized_vector)
            op_param_list: list[OpParam] = []
            for k, v in flattened_vector.items():
                # Coerce to JSON-friendly primitives when needed, but preserve dict/list for JSON column
                coerced_value = v
                if isinstance(v, dict):
                    # keep dicts as-is for param_value_json
                    pass
                elif isinstance(v, list):
                    # keep lists as-is for param_value_json
                    pass
                elif isinstance(v, (int, float)):
                    pass
                elif isinstance(v, str):
                    pass
                else:
                    # fallback to string representation for unsupported types
                    coerced_value = str(v)

                # Map value into appropriate OpParam field
                if isinstance(coerced_value, (int, float)):
                    op_param_list.append(OpParam(param_name=k, param_value_numeric=float(coerced_value)))
                elif isinstance(coerced_value, str):
                    op_param_list.append(OpParam(param_name=k, param_value_text=coerced_value))
                elif isinstance(coerced_value, (list, dict)):
                    op_param_list.append(OpParam(param_name=k, param_value_json=coerced_value))
                else:
                    op_param_list.append(OpParam(param_name=k, param_value_text=str(coerced_value)))

            # Derive op_kind/op_name from full_test_name (sweep_name): first and last string segments
            full_name = header.get("sweep_name")
            try:
                _parts = str(full_name).split(".") if full_name is not None else []
            except Exception:
                _parts = []
            _op_kind = _parts[0] if len(_parts) > 0 and _parts[0] else (header.get("op_kind") or "unknown")
            _op_name = _parts[-1] if len(_parts) > 0 and _parts[-1] else (header.get("op_name") or "unknown")

            exception = str(raw.get("exception", None))
            error_hash = generate_error_hash(exception)

            # Extract machine info if available
            machine_info = header.get("traced_machine_info")
            card_type_str = "n/a"
            if machine_info and isinstance(machine_info, list) and len(machine_info) > 0:
                # machine_info is a list of dicts, take the first one
                first_machine = machine_info[0]
                if isinstance(first_machine, dict):
                    board_type = first_machine.get("board_type", "")
                    device_series = first_machine.get("device_series", "")
                    card_count = first_machine.get("card_count", "")
                    # Format as "Wormhole n300 (1 card)" or similar
                    if board_type and device_series:
                        card_type_str = f"{board_type} {device_series}"
                        if card_count:
                            cards_label = "card" if card_count == 1 else "cards"
                            card_type_str += f" ({card_count} {cards_label})"

            record = OpTest(
                github_job_id=run_context.get("github_job_id", None),
                full_test_name=header.get("sweep_name"),
                test_start_ts=raw.get("start_time_ts"),
                test_end_ts=raw.get("end_time_ts"),
                test_case_name=header.get("suite_name"),
                filepath=header.get("sweep_name"),
                success=is_success,
                skipped=is_skipped,
                error_message=exception,
                error_hash=error_hash,
                config=None,
                frontend="ttnn.op",
                model_name=header.get("traced_source", "n/a")
                if isinstance(header.get("traced_source"), str)
                else ", ".join(header.get("traced_source", ["n/a"])),
                op_kind=_op_kind,
                op_name=_op_name,
                framework_op_name="sweep",
                inputs=None,
                outputs=None,
                op_params=None,
                git_sha=git_hash,
                status=mapped_status,
                card_type=card_type_str,
                backend="n/a",
                data_source="ttnn op test",
                input_hash=raw.get("input_hash"),
                message=_coerce_to_optional_string(raw.get("message", None)),
                exception=_coerce_to_optional_string(raw.get("exception", None)),
                metrics=_collect_all_metrics(raw),
                op_params_set=op_param_list,
            )

            # Convert to JSON-ready dict and deeply flatten any nested types
            record_dict = record.model_dump(mode="json")
            record_dict = _flatten_serialized(record_dict)
            # Ensure deterministic ordering of metrics for stable outputs
            metrics = record_dict.get("metrics")
            if isinstance(metrics, list):
                try:
                    record_dict["metrics"] = sorted(metrics, key=lambda m: m.get("metric_name", ""))
                except Exception:
                    # Best-effort: if structure is unexpected, leave as-is
                    pass
            validated_records.append(record_dict)

        # Atomic write to avoid truncated/invalid JSON on interruptions
        tmp_path = export_path.with_suffix(export_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(validated_records, file, indent=2)
            try:
                file.flush()
                os.fsync(file.fileno())
            except Exception:
                pass
        os.replace(tmp_path, export_path)

        # Aggregate for OpRun export at finalize_run
        try:
            self._collected_tests.extend(validated_records)
        except Exception:
            # Do not fail the run on aggregation errors; file export already succeeded
            pass

        logger.info(f"Successfully exported {len(results)} results to {export_path}")
        return "success"

    def finalize_run(self, run_id: str | None, final_status: str) -> None:
        """Validate and write a run-level JSON conforming to OpRun schema."""
        if self._run_metadata is None:
            return

        # Map runner final status to OpRun.RunStatus
        try:
            status_map = {
                "success": RunStatus.passed,
                "failure": RunStatus.fail,
            }
            run_status = status_map.get(str(final_status).lower(), RunStatus.exception)
        except Exception:
            run_status = RunStatus.exception

        # Build OpRun record
        try:
            run_start_ts = self._run_metadata.get("run_start_ts")
            run_end_ts = dt.datetime.now(dt.timezone.utc)
            card_type = self._run_metadata.get("device") or self._run_metadata.get("card_type") or "unknown"

            oprun = OpRun(
                initiated_by=self._run_metadata.get("initiated_by", "unknown"),
                host=self._run_metadata.get("host", "unknown"),
                card_type=str(card_type),
                run_type=self._run_metadata.get("run_type", "sweeps"),
                run_contents=self._run_metadata.get("run_contents", "all_sweeps"),
                git_author=self._run_metadata.get("git_author", "unknown"),
                git_branch_name=self._run_metadata.get("git_branch_name", "unknown"),
                git_sha=(
                    self._run_metadata.get("git_commit_sha") or self._run_metadata.get("git_commit_hash") or "unknown"
                ),
                github_pipeline_id=self._run_metadata.get(
                    "github_pipeline_id",
                ),
                run_start_ts=run_start_ts,
                run_end_ts=run_end_ts,
                status=run_status,
                tests=self._collected_tests,
            )

            # Serialize for JSON output
            run_dict = oprun.model_dump(mode="json")

            # Choose filename based on generated run_id when available
            ts_fallback = run_start_ts or dt.datetime.now(dt.timezone.utc)
            run_id_str = run_id or self._run_id or ts_fallback.strftime("%Y%m%d_%H%M%S")
            run_path = self.export_dir / f"oprun_{run_id_str}.json"

            # Atomic write to avoid truncated/invalid JSON on interruptions
            tmp_path = run_path.with_suffix(run_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as file:
                json.dump(run_dict, file, indent=2)
                try:
                    file.flush()
                    os.fsync(file.fileno())
                except Exception:
                    pass
            os.replace(tmp_path, run_path)

            logger.info(f"Successfully exported run metadata to {run_path}")
        except Exception as e:
            logger.error(f"Failed to export run metadata: {e}")
            # Do not raise to avoid masking test execution failures
            return

    def validate_connection(self) -> bool:
        """Validate that the export directory exists or can be created"""
        try:
            if not self.export_dir.exists():
                self.export_dir.mkdir(parents=True)
            return self.export_dir.is_dir()
        except Exception as e:
            logger.error(f"File destination validation failed: {e}")
            return False


def _normalize_original_vector_data(original):
    """
    Convert the captured original_vector_data (which may contain pre-serialized
    values like {'type': '...', 'data': ...} or stringified ttnn enums) into a
    JSON-friendly dict suitable for JSONB/file storage.
    """
    if original is None:
        return None
    if not isinstance(original, dict):
        return original

    normalized = {}
    for k, v in original.items():
        obj = v
        # Try postgres-aware deserialization first, then generic
        try:
            obj = deserialize_structured(v)
        except Exception:
            try:
                obj = deserialize(v)
            except Exception:
                obj = v
        # Keep native JSON-friendly structures so downstream flattening can work
        if isinstance(obj, dict):
            # Convert enum integer fields to readable strings inside dicts
            try:
                normalized[k] = convert_enum_values_to_strings(obj)
            except Exception:
                normalized[k] = obj
        elif isinstance(obj, (list, str, int, float, bool)) or obj is None:
            normalized[k] = obj
        else:
            # For complex nanobind/ttnn objects, fall back to structured serialization
            try:
                normalized[k] = serialize_structured(obj)
            except Exception:
                normalized[k] = str(obj)
    return normalized


def _flatten_serialized(value):
    # For file output: strip {"type": "...", "data": ...} wrappers and keep readable data
    if isinstance(value, dict):
        if set(value.keys()) == {"type", "data"}:
            return _flatten_serialized(value["data"])
        return {k: _flatten_serialized(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_flatten_serialized(v) for v in value]
    if isinstance(value, dt.datetime):
        return value.isoformat()
    try:
        from enum import Enum

        if isinstance(value, Enum):
            return str(value)
    except Exception:
        pass
    return value


def _flatten_any_to_dotted(value: Any) -> dict[str, Any]:
    """
    Flatten an arbitrarily nested structure (dicts/lists/primitives) into a
    single-level dict with dotted keys for dict nesting. Lists are preserved
    under their current key (no index flattening).
    Example: {"a": {"b": [1, {"c": 2}]}} -> {"a.b": [1, {"c": 2}]}
    """
    flat: dict[str, Any] = {}

    def _recurse(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}.{k}" if prefix else str(k)
                _recurse(new_key, v)
        elif isinstance(obj, list):
            # Preserve lists as-is under the current key
            flat[prefix] = obj
        else:
            flat[prefix] = obj

    _recurse("", value)
    return flat


class SupersetResultDestination(FileResultDestination):
    """Superset destination: file export plus SFTP upload of oprun_*.json."""

    def __init__(self, export_dir: pathlib.Path | None = None):
        super().__init__(export_dir)

    def finalize_run(self, run_id: str | None, final_status: str) -> None:
        # First perform the standard file-based finalize to write oprun_*.json
        super().finalize_run(run_id, final_status)

        # Compute the path of the just-written oprun file
        try:
            run_id_str = run_id or self._run_id
            run_path = self.export_dir / f"oprun_{run_id_str}.json"
        except Exception as e:
            logger.error(f"Superset: failed to determine oprun file path for upload: {e}")
            return
        logger.info(f"Superset: run_path={run_path}")
        logger.info(f"Superset: run_id={run_id}")

        # Upload via SFTP if environment/configuration is available
        try:
            success = upload_run_sftp(run_path)
            if success:
                logger.info(f"Superset: successfully uploaded '{run_path.name}' via SFTP")
            else:
                logger.warning(
                    f"Superset: skipping SFTP upload for '{run_path.name}' (missing credentials or upload failed)"
                )
        except Exception as e:
            logger.error(f"Superset: unexpected error during SFTP upload of '{run_path}': {e}")
            # Do not raise; file export already succeeded


class ResultDestinationFactory:
    """Factory to create appropriate result destination based on configuration"""

    SUPPORTED_DESTINATIONS = {"results_export", "superset"}

    @staticmethod
    def create_destination(result_destination: str, **kwargs) -> ResultDestination:
        if result_destination == "results_export":
            export_dir = kwargs.get("export_dir")
            return FileResultDestination(export_dir)
        elif result_destination == "superset":
            export_dir = kwargs.get("export_dir")
            return SupersetResultDestination(export_dir)
        else:
            raise ValueError(
                f"Unknown result destination: {result_destination}. "
                f"Supported destinations: {', '.join(sorted(ResultDestinationFactory.SUPPORTED_DESTINATIONS))}"
            )
