# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pathlib
import json
import datetime as dt
import hashlib
import os
import math
from elasticsearch import Elasticsearch
from framework.database import (
    postgres_connection,
    initialize_postgres_database,
    push_run,
    update_run,
    generate_error_signature,
    map_test_status_to_run_status,
    generate_error_hash,
)
from framework.serialize import serialize, serialize_structured
from framework.serialize import deserialize, deserialize_structured
from framework.sweeps_logger import sweeps_logger as logger
from infra.data_collection.pydantic_models import OpTest, PerfMetric, TestStatus, OpParam, OpRun, RunStatus
from framework.upload_sftp import upload_run_sftp

# Optional numpy import for numeric handling in hot paths
try:
    import numpy as np
except ImportError:
    np = None


class ResultDestination(ABC):
    """Abstract base class for test result destinations"""

    @abstractmethod
    def initialize_run(self, run_metadata: Dict[str, Any]) -> Optional[str]:
        """Initialize a new test run and return run_id if applicable"""
        pass

    @abstractmethod
    def export_results(self, header_info: List[Dict], results: List[Dict], run_context: Dict[str, Any]) -> str:
        """Export test results and return status"""
        pass

    @abstractmethod
    def finalize_run(self, run_id: Optional[str], final_status: str) -> None:
        """Finalize the test run"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the destination is accessible"""
        pass


class PostgresResultDestination(ResultDestination):
    """PostgreSQL-based result destination"""

    def __init__(self):
        # No postgres_env parameter needed since database.py uses environment variables
        pass

    def initialize_run(self, run_metadata: Dict[str, Any]) -> Optional[str]:
        """Initialize PostgreSQL database and create run record"""
        initialize_postgres_database()  # No env parameter needed

        return push_run(
            initiated_by=run_metadata["initiated_by"],
            host=run_metadata["host"],
            git_author=run_metadata["git_author"],
            git_branch_name=run_metadata["git_branch_name"],
            git_commit_hash=run_metadata["git_commit_hash"],
            start_time_ts=run_metadata["start_time_ts"],
            status=run_metadata["status"],
            run_contents=run_metadata.get("run_contents"),
            device=run_metadata.get("device"),
            run_type="sweep",
        )

    def export_results(self, header_info: List[Dict], results: List[Dict], run_context: Dict[str, Any]) -> str:
        """Export results to PostgreSQL database"""
        if not results:
            logger.info("No test results to push to PostgreSQL database.")
            return "success"

        run_id = run_context["run_id"]
        test_start_time = run_context["test_start_time"]
        test_end_time = run_context["test_end_time"]

        try:
            with postgres_connection() as (conn, cursor):  # No env parameter needed
                sweep_name = header_info[0]["sweep_name"]

                # Create test record
                test_insert_query = """
                INSERT INTO tests (run_id, name, start_time_ts, end_time_ts, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """
                cursor.execute(test_insert_query, (run_id, sweep_name, test_start_time, test_end_time, "success"))
                test_id = cursor.fetchone()[0]

                # Insert test cases in batch
                test_statuses = []
                testcase_insert_query = """
                INSERT INTO sweep_testcases (
                    test_id, name, start_time_ts, end_time_ts,
                    status, suite_name, test_vector, message, exception,
                    e2e_perf, device_perf, error_signature
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """

                batch_values = []
                for i, result in enumerate(results):
                    db_status = self._map_test_status_to_db_status(result.get("status", None))
                    test_statuses.append(db_status)
                    testcase_name = f"{sweep_name}_{header_info[i].get('vector_id', 'unknown')}"
                    exception_text = result.get("exception", None)
                    error_sig = generate_error_signature(exception_text)
                    error_hash = generate_error_hash(exception_text)

                    testcase_values = (
                        test_id,
                        testcase_name,
                        result.get("start_time_ts", None),
                        result.get("end_time_ts", None),
                        db_status,
                        header_info[i].get("suite_name", None),
                        json.dumps(_normalize_original_vector_data(result.get("original_vector_data", None))),
                        result.get("message", None),
                        exception_text,
                        result.get("e2e_perf", None),
                        json.dumps(result.get("device_perf")) if result.get("device_perf") else None,
                        error_sig,
                    )
                    batch_values.append(testcase_values)

                if batch_values:
                    cursor.executemany(testcase_insert_query, batch_values)
                    logger.info(
                        f"Successfully pushed {len(batch_values)} testcase results to PostgreSQL database for test {test_id}"
                    )

                # Update test status
                test_status = map_test_status_to_run_status(test_statuses)
                test_update_query = "UPDATE tests SET status = %s WHERE id = %s"
                cursor.execute(test_update_query, (test_status, test_id))

                return test_status
        except Exception as e:
            logger.error(f"Failed to push test result to PostgreSQL database: {e}")
            raise

    def finalize_run(self, run_id: Optional[str], final_status: str) -> None:
        """Finalize the run in PostgreSQL"""
        if run_id:
            update_run(run_id, dt.datetime.now(), final_status)  # No env parameter needed

    def validate_connection(self) -> bool:
        """Validate PostgreSQL connection"""
        try:
            with postgres_connection() as (conn, cursor):  # No env parameter needed
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"PostgreSQL connection validation failed: {e}")
            return False

    def _map_test_status_to_db_status(self, test_status):
        """Map TestStatus enum to database status string"""
        from framework.statuses import TestStatus

        status_mapping = {
            TestStatus.PASS: "pass",
            TestStatus.FAIL_ASSERT_EXCEPTION: "fail_assert_exception",
            TestStatus.FAIL_L1_OUT_OF_MEM: "fail_l1_out_of_mem",
            TestStatus.FAIL_WATCHER: "fail_watcher",
            TestStatus.FAIL_CRASH_HANG: "fail_crash_hang",
            TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF: "fail_unsupported_device_perf",
            TestStatus.NOT_RUN: "skipped",
        }
        return status_mapping.get(test_status, "error")


class ElasticResultDestination(ResultDestination):
    """Elasticsearch-based result destination"""

    def __init__(self, connection_string: str, username: str, password: str):
        self.client = Elasticsearch(connection_string, basic_auth=(username, password))
        self.connection_string = connection_string

    def initialize_run(self, run_metadata: Dict[str, Any]) -> Optional[str]:
        """No specific run initialization needed for Elasticsearch"""
        return None

    def export_results(self, header_info: List[Dict], results: List[Dict], run_context: Dict[str, Any]) -> str:
        """Export results to Elasticsearch"""
        if not results:
            return "success"

        from framework.elastic_config import RESULT_INDEX_PREFIX

        sweep_name = header_info[0]["sweep_name"]
        results_index = RESULT_INDEX_PREFIX + sweep_name

        # Add git hash to results
        curr_git_hash = run_context.get("git_hash", "unknown")
        for result in results:
            result["git_hash"] = curr_git_hash

        for i in range(len(results)):
            result = header_info[i].copy()
            for elem in results[i].keys():
                if elem == "device_perf":
                    result[elem] = results[i][elem]
                    continue
                # Skip problematic fields that were added for PostgreSQL functionality
                if elem in ["start_time_ts", "end_time_ts", "original_vector_data"]:
                    continue
                result[elem] = serialize(results[i][elem])
            self.client.index(index=results_index, body=result)

        logger.info(f"Successfully exported {len(results)} results to Elasticsearch")
        return "success"

    def finalize_run(self, run_id: Optional[str], final_status: str) -> None:
        """No specific run finalization needed for Elasticsearch"""
        pass

    def validate_connection(self) -> bool:
        """Validate Elasticsearch connection"""
        try:
            # Basic connection test - just try to get cluster info
            self.client.info()
            return True
        except Exception as e:
            logger.error(f"Elasticsearch connection validation failed: {e}")
            return False


class FileResultDestination(ResultDestination):
    """File-based result destination (JSON export)"""

    def __init__(self, export_dir: Optional[pathlib.Path] = None):
        if export_dir is None:
            self.export_dir = pathlib.Path(__file__).parent.parent / "results_export"
        else:
            self.export_dir = export_dir
        # In-memory aggregation for building OpRun at finalize_run
        self._run_metadata: Optional[Dict[str, Any]] = None
        self._collected_tests: List[Dict[str, Any]] = []
        self._run_id: Optional[str] = None

    def initialize_run(self, run_metadata: Dict[str, Any]) -> Optional[str]:
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
            start_ts = run_metadata.get("start_time_ts") or run_metadata.get("run_start_ts") or dt.datetime.now()
            ts_str = start_ts.strftime("%Y%m%d_%H%M%S")
            # Sanitize host to avoid path issues and keep the filename short
            safe_host = host.replace("/", "_")[:32]
            self._run_id = f"{safe_host}_{digest}_{ts_str}"
        except Exception:
            self._run_id = None

        # Reset any previously collected tests
        self._collected_tests = []

        return self._run_id

    def export_results(self, header_info: List[Dict], results: List[Dict], run_context: Dict[str, Any]) -> str:
        """Export results to JSON file using Pydantic validation (OpTest)."""
        if not results:
            return "success"

        sweep_name = header_info[0]["sweep_name"]
        run_start_time = run_context.get("test_start_time")
        if run_start_time:
            timestamp = run_start_time.strftime("%Y%m%d_%H%M%S")
        else:
            # Fallback to current time if run_start_time is not available
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Keep filenames short and safe: use sweep short name + digest
        short_sweep = str(sweep_name).split(".")[0] if sweep_name else "sweep"
        name_digest = hashlib.sha256(str(sweep_name).encode("utf-8")).hexdigest()[:12] if sweep_name else "na"
        export_path = self.export_dir / f"{short_sweep}_{name_digest}_{timestamp}.json"

        git_hash = run_context.get("git_hash", "unknown")

        # this will be the list of OpTest objects that will be exported to the file
        validated_records = []

        # Map internal TestStatus enum (or strings) to file schema enum values
        def _map_status(value: Any) -> Optional[TestStatus]:
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

        def _coerce_device_perf(device_perf_raw: Any) -> Optional[set[PerfMetric]]:
            if device_perf_raw is None:
                return None
            # If list of dicts, merge or take first
            if isinstance(device_perf_raw, list):
                device_perf_raw = next((d for d in device_perf_raw if isinstance(d, dict)), None)
                if device_perf_raw is None:
                    return None
            if not isinstance(device_perf_raw, dict):
                return None
            metrics: set[PerfMetric] = set()

            def _to_float(v):
                try:
                    return float(v)
                except Exception:
                    return None

            for k, v in device_perf_raw.items():
                metrics.add(PerfMetric(metric_name=str(k), metric_value=_to_float(v)))
            return metrics if metrics else None

        def _coerce_to_optional_string(value: Any) -> Optional[str]:
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
            op_param_list: List[OpParam] = []
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

            record = OpTest(
                github_job_id=run_context.get("github_job_id", None),
                full_test_name=header.get("sweep_name"),
                test_start_ts=raw.get("start_time_ts"),
                test_end_ts=raw.get("end_time_ts"),
                test_case_name=header.get("suite_name"),
                filepath=header.get("sweep_name"),
                success=is_success,
                skipped=is_skipped,
                error_message=raw.get("exception", None),
                error_hash=generate_error_hash(raw.get("exception", None)),
                config=None,
                frontend="ttnn.op",
                model_name="n/a",
                op_kind=_op_kind,
                op_name=_op_name,
                framework_op_name="sweep",
                inputs=None,
                outputs=None,
                op_params=None,
                git_sha=git_hash,
                status=mapped_status,
                card_type="n/a",
                backend="n/a",
                data_source="ttnn op test",
                input_hash=header.get("input_hash"),
                message=_coerce_to_optional_string(raw.get("message", None)),
                exception=_coerce_to_optional_string(raw.get("exception", None)),
                metrics=raw.get("device_perf", None),
                op_params_set=op_param_list,
            )

            # Convert to JSON-ready dict and deeply flatten any nested types
            record_dict = record.model_dump(mode="json")
            record_dict = _flatten_serialized(record_dict)
            validated_records.append(record_dict)

        # Atomic write to avoid truncated/invalid JSON on interruptions
        tmp_path = export_path.with_suffix(export_path.suffix + ".tmp")
        with open(tmp_path, "w") as file:
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

    def finalize_run(self, run_id: Optional[str], final_status: str) -> None:
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
            run_end_ts = dt.datetime.now()
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
            run_id_str = run_id or self._run_id or run_start_ts.strftime("%Y%m%d_%H%M%S")
            run_path = self.export_dir / f"oprun_{run_id_str}.json"

            # Atomic write to avoid truncated/invalid JSON on interruptions
            tmp_path = run_path.with_suffix(run_path.suffix + ".tmp")
            with open(tmp_path, "w") as file:
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
                normalized[k] = convert_enum_values_to_strings(obj)  # type: ignore[name-defined]
            except Exception:
                normalized[k] = obj
        elif isinstance(obj, (list, str, int, float, bool)) or obj is None:
            normalized[k] = obj
        else:
            # For complex pybind/ttnn objects, fall back to structured serialization
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


def _flatten_any_to_dotted(value: Any) -> Dict[str, Any]:
    """
    Flatten an arbitrarily nested structure (dicts/lists/primitives) into a
    single-level dict with dotted keys. List indices are included in the key path.
    Example: {"a": {"b": [1, {"c": 2}]}} -> {"a.b.0": 1, "a.b.1.c": 2}
    """
    flat: Dict[str, Any] = {}

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

    def __init__(self, export_dir: Optional[pathlib.Path] = None):
        super().__init__(export_dir)

    def finalize_run(self, run_id: Optional[str], final_status: str) -> None:
        # First perform the standard file-based finalize to write oprun_*.json
        super().finalize_run(run_id, final_status)

        # Compute the path of the just-written oprun file
        try:
            run_id_str = run_id or self._run_id
            run_path = self.export_dir / f"oprun_{run_id_str}.json"
        except Exception as e:
            logger.error(f"Superset: failed to determine oprun file path for upload: {e}")
            return
        print(f"Superset: run_path: {run_path}")
        print(f"Superset: run_id: {run_id}")

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

    @staticmethod
    def create_destination(result_destination: str, **kwargs) -> ResultDestination:
        if result_destination == "postgres":
            return PostgresResultDestination()
        elif result_destination == "elastic":
            required_args = ["connection_string", "username", "password"]
            for arg in required_args:
                if arg not in kwargs:
                    raise ValueError(f"Missing required argument '{arg}' for elastic result destination")
            return ElasticResultDestination(**kwargs)
        elif result_destination == "results_export":
            export_dir = kwargs.get("export_dir")
            return FileResultDestination(export_dir)
        elif result_destination == "superset":
            export_dir = kwargs.get("export_dir")
            return SupersetResultDestination(export_dir)
        else:
            raise ValueError(f"Unknown result destination: {result_destination}")
