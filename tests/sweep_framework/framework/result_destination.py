# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import pathlib
import json
import datetime as dt
from elasticsearch import Elasticsearch
from framework.database import (
    postgres_connection,
    initialize_postgres_database,
    push_run,
    update_run,
    generate_error_signature,
    map_test_status_to_run_status,
)
from framework.serialize import serialize, serialize_for_postgres
from framework.serialize import deserialize, deserialize_for_postgres
from framework.sweeps_logger import sweeps_logger as logger
from infra.data_collection.pydantic_models import OpTestResultRecord, DevicePerfMetric


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

    def initialize_run(self, run_metadata: Dict[str, Any]) -> Optional[str]:
        """Ensure export directory exists"""
        if not self.export_dir.exists():
            self.export_dir.mkdir(parents=True)
        return None

    def export_results(self, header_info: List[Dict], results: List[Dict], run_context: Dict[str, Any]) -> str:
        """Export results to JSON file using Pydantic validation (OpTestResultRecord)."""
        if not results:
            return "success"

        module_name = header_info[0]["sweep_name"]
        export_path = self.export_dir / f"{module_name}.json"

        curr_git_hash = run_context.get("git_hash", "unknown")

        validated_records = []

        # Map internal TestStatus enum (or strings) to file schema enum values
        def _map_status(value: Any) -> Optional[str]:
            if value is None:
                return None
            try:
                from framework.statuses import TestStatus

                # TODO: consider removing this mapping and just using the TestStatus enum directly
                if isinstance(value, TestStatus):
                    mapping = {
                        TestStatus.PASS: "pass",
                        TestStatus.FAIL_ASSERT_EXCEPTION: "fail_assert_exception",
                        TestStatus.FAIL_CRASH_HANG: "fail_crash_hang",
                        TestStatus.NOT_RUN: "skipped",
                        TestStatus.FAIL_L1_OUT_OF_MEM: "fail_l1_out_of_mem",
                        TestStatus.FAIL_WATCHER: "fail_watcher",
                        TestStatus.FAIL_UNSUPPORTED_DEVICE_PERF: "fail_unsupported_device_perf",
                    }
                    return mapping.get(value, "error")
            except Exception:
                pass
            # If already a string, trust but verify
            try:
                s = str(value)
                # Normalize common forms like "TestStatus.PASS"
                if s.startswith("TestStatus."):
                    suffix = s.split(".", 1)[1].lower()
                    if suffix == "pass":
                        return "pass"
                    return suffix
                return s
            except Exception:
                return "error"

        def _coerce_device_perf(device_perf_raw: Any) -> Optional[set[DevicePerfMetric]]:
            if device_perf_raw is None:
                return None
            # If list of dicts, merge or take first
            if isinstance(device_perf_raw, list):
                device_perf_raw = next((d for d in device_perf_raw if isinstance(d, dict)), None)
                if device_perf_raw is None:
                    return None
            if not isinstance(device_perf_raw, dict):
                return None
            metrics: set[DevicePerfMetric] = set()

            def _to_float(v):
                try:
                    return float(v)
                except Exception:
                    return None

            for k, v in device_perf_raw.items():
                metrics.add(DevicePerfMetric(metric_name=str(k), metric_value=_to_float(v)))
            return metrics if metrics else None

        for i in range(len(results)):
            header = header_info[i]
            raw = results[i]

            exception_text = raw.get("exception", None)

            # Build a validated record via Pydantic
            record = OpTestResultRecord(
                test_name=header.get("sweep_name"),
                suite_name=header.get("suite_name"),
                vector_id=header.get("vector_id"),
                input_hash=header.get("input_hash"),
                test_start_ts=raw.get("start_time_ts"),
                test_end_ts=raw.get("end_time_ts"),
                status=_map_status(raw.get("status")),
                message=raw.get("message"),
                exception=exception_text,
                error_signature=generate_error_signature(exception_text),
                e2e_perf=raw.get("e2e_perf"),
                device_perf=_coerce_device_perf(raw.get("device_perf")),
                git_hash=curr_git_hash,
                host=raw.get("host"),
                user=raw.get("user"),
                original_vector_data=_flatten_serialized(
                    _normalize_original_vector_data(raw.get("original_vector_data"))
                ),
            )

            # Convert to JSON-ready dict and deeply flatten any nested types
            record_dict = record.model_dump(mode="json")
            record_dict = _flatten_serialized(record_dict)
            validated_records.append(record_dict)

        # Append to existing file or create new one
        if export_path.exists():
            try:
                with open(export_path, "r") as file:
                    old_data = json.load(file)
                if isinstance(old_data, list):
                    validated_records = old_data + validated_records
                else:
                    logger.warning(
                        f"Existing export file {export_path} is not a JSON list. Overwriting with validated records."
                    )
            except json.JSONDecodeError:
                # Corrupt or non-JSON file: back it up and proceed with fresh records
                try:
                    backup_path = export_path.with_suffix(export_path.suffix + ".bak")
                    export_path.rename(backup_path)
                    logger.warning(
                        f"Existing export file {export_path} contained invalid JSON. Backed up to {backup_path}."
                    )
                except Exception:
                    logger.warning(
                        f"Existing export file {export_path} contained invalid JSON and could not be backed up. Overwriting."
                    )

        with open(export_path, "w") as file:
            json.dump(validated_records, file, indent=2)

        logger.info(f"Successfully exported {len(results)} results to {export_path}")
        return "success"

    def finalize_run(self, run_id: Optional[str], final_status: str) -> None:
        """No specific run finalization needed for file export"""
        pass

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
            obj = deserialize_for_postgres(v)
        except Exception:
            try:
                obj = deserialize(v)
            except Exception:
                obj = v
        # Re-serialize into JSON-friendly shape with enum strings parsed
        try:
            normalized[k] = serialize_for_postgres(obj)
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
        else:
            raise ValueError(f"Unknown result destination: {result_destination}")
