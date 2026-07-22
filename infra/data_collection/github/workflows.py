# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import re
from datetime import datetime, timedelta
from functools import partial
from typing import List

from loguru import logger

from infra.data_collection import junit_xml_utils, pydantic_models

smi_pattern = re.compile(r'.*"tt_smi":\s*"([a-zA-Z0-9\-\.]+)"')
tt_smi_reset_pattern = re.compile(r'"tt_smi_reset":\s*(\[.*\])')
# Define a regex pattern to match timestamps in ISO 8601 format (e.g., 2025-03-26T19:18:31.7521333Z)
timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z")


def search_for_tt_smi_version_in_log_file_(log_file):
    # Defense-in-depth: resolve and confirm this is a real file before opening.
    log_file = pathlib.Path(log_file).resolve()
    assert log_file.is_file(), f"Not a readable log file: {log_file}"
    with open(log_file, "r") as log_f:
        for line in log_f:
            regex_match = smi_pattern.match(line)
            if regex_match:
                return regex_match.group(1)
    return None


def search_for_tt_smi_reset_in_log_file_(log_file):
    # Defense-in-depth: resolve and confirm this is a real file before opening.
    log_file = pathlib.Path(log_file).resolve()
    assert log_file.is_file(), f"Not a readable log file: {log_file}"
    ts_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z\s*")
    # Strip GitHub Actions annotation prefixes like ##[error], ##[warning]
    gh_annotation_pattern = re.compile(r"^##\[[a-z]+\]", re.IGNORECASE)

    def strip_ansi(text):
        return re.sub(r"\x1B[@-_][0-?]*[ -/]*[@-~]", "", text)

    def parse_ts(line):
        try:
            line = line.strip()
            if not ts_pattern.match(line):
                return None
            ts_str = line.split(" ")[0].rstrip("Z")
            if "." in ts_str:
                base, frac = ts_str.split(".")
                ts_str = f"{base}.{frac[:6]}"
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None

    def clean_line(line):
        """Strip ISO timestamp prefix and GitHub annotation prefix, return clean content."""
        s = line.strip()
        m = ts_pattern.match(s)
        if m:
            s = s[m.end() :]
        # Strip ##[error] / ##[warning] / ##[group] etc.
        s = gh_annotation_pattern.sub("", s).strip()
        return s

    with open(log_file, "r") as f:
        log = strip_ansi(f.read())

    lines = log.splitlines()

    # Check if there is any tt-smi reset activity in this log at all
    has_reset = any(
        "tt-smi reset" in line.lower() or ("tt_metal_infra" in line.lower() and ".sh" in line.lower()) for line in lines
    )

    if not has_reset:
        return [
            {
                "tt_smi_reset_attempt": 1,
                "final_status": "UNKNOWN",
                "total_reset_time_sec": None,
                "error_summary": None,
            }
        ]

    # Find where the reset section starts
    reset_start_idx = None
    for i, line in enumerate(lines):
        lower = line.lower()
        if (
            ("tt_metal_infra" in lower and ".sh" in lower)
            or "starting tt-smi reset" in lower
            or "tt-smi reset" in lower
        ):
            reset_start_idx = i
            break

    # If no explicit start found, scan from beginning (e.g. WH simple case)
    if reset_start_idx is None:
        reset_start_idx = 0

    block_lines = lines[reset_start_idx:]
    block_start_ts = parse_ts(lines[reset_start_idx])
    block_end_ts = None
    # Count internal tt-smi retry attempts via "===== START of output =====" occurrences
    num_smi_attempts = 0
    final_status = "UNKNOWN"
    seen_errors = set()
    error_lines = []

    reset_done = False

    for line in block_lines:
        lower = line.strip().lower()
        ts = parse_ts(line)
        # Only update end timestamp while reset is still in progress
        if ts and not reset_done:
            block_end_ts = ts
        if "===== start of output =====" in lower:
            num_smi_attempts += 1
        if "tt-smi reset was successful" in lower:
            final_status = "SUCCESS"
            if ts:
                block_end_ts = ts
            reset_done = True
        if "error:" in lower:
            # Always collect all conclusive failure lines (there can be multiple)
            content = clean_line(line)
            if content and content not in seen_errors:
                seen_errors.add(content)
                error_lines.append(content)
            if not reset_done:
                final_status = "FAILURE"
                if ts:
                    block_end_ts = ts
                reset_done = True

    # "===== START of output =====" counts internal tt-smi retry invocations.
    # If it never appeared but we got a clear SUCCESS, the reset ran once without retries.
    # If it never appeared and status is still UNKNOWN or FAILURE, no real tool ran —
    # it was a false positive (e.g. "tt-smi reset" / "Error:" in an unrelated log line).
    if num_smi_attempts == 0:
        if final_status == "SUCCESS":
            num_smi_attempts = 1
        else:
            return [
                {
                    "tt_smi_reset_attempt": 1,
                    "final_status": "UNKNOWN",
                    "total_reset_time_sec": None,
                    "error_summary": None,
                }
            ]

    duration = None
    if block_start_ts and block_end_ts:
        duration = (block_end_ts - block_start_ts).total_seconds()

    if final_status == "SUCCESS":
        error_summary = "tt-smi reset was successful"
    elif error_lines:
        error_summary = " | ".join(error_lines)
    else:
        error_summary = None

    return [
        {
            "tt_smi_reset_attempt": num_smi_attempts,
            "final_status": final_status,
            "total_reset_time_sec": duration,
            "error_summary": error_summary,
        }
    ]


def get_github_job_ids_to_tt_smi_versions(workflow_outputs_dir, workflow_run_id: int, workflow_attempt: int):
    logs_dir = _safe_logs_dir(workflow_outputs_dir, workflow_run_id)

    assert logs_dir is not None, f"Invalid or unsafe workflow_run_id: {workflow_run_id}"
    assert logs_dir.exists(), f"Logs dir does not exist: {logs_dir}"
    assert logs_dir.is_dir(), f"Logs path is not a dir: {logs_dir}"

    log_files = list(logs_dir.glob("*.log"))
    assert log_files, f"No log files found in {logs_dir}"

    github_job_ids_to_tt_smi_versions = {}
    github_job_ids_to_tt_smi_resets = {}

    for log_file in log_files:
        filename = log_file.stem

        assert filename.isdigit(), f"Unexpected filename format: {filename}"

        github_job_id = int(filename)
        assert github_job_id > 0

        # Re-derive the path from the validated integer ids (containment-checked) so the
        # path handed to the file readers below is sanitized, not the raw globbed path.
        safe_log_file = _safe_job_log_file(workflow_outputs_dir, workflow_run_id, github_job_id)
        assert safe_log_file is not None and safe_log_file.is_file(), f"Unsafe or missing log file: {log_file}"

        tt_smi_version = search_for_tt_smi_version_in_log_file_(safe_log_file)
        if tt_smi_version:
            github_job_ids_to_tt_smi_versions[github_job_id] = tt_smi_version

        tt_smi_reset = search_for_tt_smi_reset_in_log_file_(safe_log_file)
        for reset in tt_smi_reset:
            reset["workflow_attempt"] = workflow_attempt
        assert tt_smi_reset is not None, f"Parser returned None for {safe_log_file}"

        assert github_job_id not in github_job_ids_to_tt_smi_resets, f"Duplicate reset key detected: {github_job_id}"

        github_job_ids_to_tt_smi_resets[github_job_id] = tt_smi_reset

    return github_job_ids_to_tt_smi_versions, github_job_ids_to_tt_smi_resets


def parse_github_log_timestamp(line):
    timestamp_str = line.split("T")[0] + "T" + line.split("T")[1].split("Z")[0]
    # Wacky github workaround: truncate to 26 chars because github's timestamp
    # is 7 digits for fractional seconds instead of 6, which is the ISO format
    # E.g. 2024-09-25T14:33:11.1060679Z -> 2024-09-25T14:33:11.106067
    return datetime.fromisoformat(timestamp_str[:26])


def _resolve_within(base_dir, *parts):
    """Resolve base_dir / *parts, returning the path only if it stays within base_dir.

    Guards against path traversal from dynamic path components: returns None if the
    resolved path escapes base_dir (e.g. a component containing "..").
    """
    base = base_dir.resolve()
    candidate = base.joinpath(*parts).resolve()
    return candidate if candidate.is_relative_to(base) else None


def _safe_logs_dir(workflow_outputs_dir, workflow_run_id):
    """Resolved <workflow_outputs_dir>/<run_id>/logs, or None if run_id is not a plain
    integer or the path escapes workflow_outputs_dir."""
    if not str(workflow_run_id).isdigit():
        return None
    return _resolve_within(workflow_outputs_dir, str(workflow_run_id), "logs")


def _safe_job_log_file(workflow_outputs_dir, workflow_run_id, github_job_id):
    """Resolved <workflow_outputs_dir>/<run_id>/logs/<job_id>.log, or None if either id is
    not a plain integer or the path escapes workflow_outputs_dir."""
    if not (str(workflow_run_id).isdigit() and str(github_job_id).isdigit()):
        return None
    return _resolve_within(workflow_outputs_dir, str(workflow_run_id), "logs", f"{github_job_id}.log")


def is_job_hanging_from_job_log(error_snippet, workflow_outputs_dir, workflow_run_id: int, workflow_job_id: int):
    """
    Read the job output log to determine if a job is hanging or genuinely timed out.
    For each line, we store the associated github timestamp (if it exists)
    When we encounter the timeout error message, compare the timestamp of the generated message
    against the last output line, as well as the last line against the 2nd last.

    We calculate two time deltas because some hangs can generate a line of output the moment the process is terminated/timed out.
    Therefore we need to also check the second-last line's timestamp to confirm a hang has occurred.

    If the time delta between the lines is greater than 5 minutes** (max_time_delta_seconds)
    then consider the job as a hang. Otherwise it's most likely a regular timeout.

    ** Threshold may be reduced in the future
    """
    log_file = _safe_job_log_file(workflow_outputs_dir, workflow_run_id, workflow_job_id)
    max_time_delta_seconds = 300

    if log_file is None or not log_file.is_file():
        logger.warning(f"Unable to find github job log file for job: {workflow_job_id}")
        return False

    log_lines = []
    last_ts, second_last_ts = None, None
    with open(log_file, "r", encoding="utf-8-sig") as log_f:
        log_lines = log_f.readlines()

    for line in log_lines:
        # Skip lines that are empty or do not start with a valid timestamp
        if not line.strip() or not timestamp_pattern.match(line):
            continue

        if error_snippet in line:
            timeout_timestamp = parse_github_log_timestamp(line)

            # Check if we have the previous two timestamps
            if last_ts is None or second_last_ts is None:
                logger.warning("Not enough previous lines to compare time deltas.")
                return False

            # Compare with the last two timestamps
            # Hang message vs last output line
            delta_1 = timeout_timestamp - last_ts
            # Last output line vs 2nd last output line
            delta_2 = last_ts - second_last_ts

            # Check if any of the deltas is greater than 5 minutes
            if delta_1.total_seconds() > max_time_delta_seconds or delta_2.total_seconds() > max_time_delta_seconds:
                logger.info(f"Time difference between the timeout line and previous lines is greater than 5 minutes.")
                logger.info(f"Timeout timestamp: {timeout_timestamp}")
                logger.info(f"Previous timestamps: {second_last_ts}, {last_ts}")
                logger.info(f"Hang detected for job: {str(workflow_job_id)}")
                return True
            else:
                logger.info(
                    f"No hang detected for job: {str(workflow_job_id)}, Time differences are within the expected range."
                )
                return False

        # Update the second last and last timestamps
        second_last_ts = last_ts
        last_ts = parse_github_log_timestamp(line)
    return False


def get_workflow_run_uuids_to_test_reports_paths_(workflow_outputs_dir, workflow_run_id: int):
    artifacts_dir = workflow_outputs_dir / str(workflow_run_id) / "artifacts"

    test_report_dirs = artifacts_dir.glob("test_reports_*")

    workflow_run_test_reports_path = {}
    for test_report_dir in test_report_dirs:
        assert test_report_dir.exists()
        assert test_report_dir.is_dir(), f"{test_report_dir} is not dir"

        test_report_uuid = test_report_dir.name.replace("test_reports_", "")

        try:
            # read all *.xml in test_report_dir (gtest can have one xml files per test executable)
            xml_file_paths = [file.resolve(strict=True) for file in list(test_report_dir.glob("*.xml"))]
        except FileNotFoundError:
            logger.warning(f"No pytest or gtest xml file found in {test_report_dir}, skipping directory.")
        else:
            workflow_run_test_reports_path[test_report_uuid] = xml_file_paths

    return workflow_run_test_reports_path


def search_for_uuid_in_log_file_(log_file):
    with open(log_file) as log_f:
        for line in log_f:
            # [UPLOAD-ARTIFACT-UUID] test_reports_f04ff0aa-d54d-446b-b87b-a7317970ace3
            line_tokens = line.split(" ")

            is_uuid_line = len(line_tokens) == 3 and line_tokens[1] == "[UPLOAD-ARTIFACT-UUID]"

            if is_uuid_line:
                return line_tokens[2].strip()
    return None


def get_github_job_ids_to_workflow_run_uuids_(workflow_outputs_dir, workflow_run_id: int):
    """
    We know we can get a proper mapping of github_job_id -> uuid because based on anecdotal testing,
    it seems that the same GitHub job ID does not repeat for attempted runs, but because artifacts
    carry over, uuid will repeat. We want to ensure that the set of github_job_ids is fully captured, and
    it doesn't matter if the full set of uuids is.
    """
    logs_dir = workflow_outputs_dir / str(workflow_run_id) / "logs"

    log_files = logs_dir.glob("*.log")

    github_job_ids_to_workflow_run_uuids = {}
    for log_file in log_files:
        artifact_uuid_name = search_for_uuid_in_log_file_(log_file)
        if artifact_uuid_name:
            uuid = artifact_uuid_name.replace("test_reports_", "")
            filename = log_file.stem
            assert filename.isdigit(), f"Unexpected filename format: {filename}"
            github_job_id = int(filename)
            github_job_ids_to_workflow_run_uuids[github_job_id] = uuid
    return github_job_ids_to_workflow_run_uuids


def get_github_job_id_to_test_reports(workflow_outputs_dir, workflow_run_id: int, github_job_ids: List[int]):
    workflow_run_uuids_to_test_report_paths = get_workflow_run_uuids_to_test_reports_paths_(
        workflow_outputs_dir, workflow_run_id
    )

    github_job_ids_to_workflow_run_uuids = get_github_job_ids_to_workflow_run_uuids_(
        workflow_outputs_dir, workflow_run_id
    )

    github_job_id_to_test_reports = {}
    for github_job_id in github_job_ids:
        # It's possible the upload didn't go through, but if the test report
        # for a uuid exists, then that means the test must have printed a uuid
        # Unless the log file itself is gone because it's a stale pipeline

        # It could be a good idea to throw an exception for either continue
        # step in this loop and have the caller handle it. The key reason to do
        # that in our case would be to potentially snuff out infrastructure
        # issues and surface that at the job level.

        if github_job_id not in github_job_ids_to_workflow_run_uuids:
            """
            The potential infra failure is: internal error or logs were never pushed up because the runner died
            """
            logger.warning(
                f"No uuid was found for job {github_job_id}, meaning either the log file is not present or it doesn't have a uuid printed"
            )
            continue

        uuid = github_job_ids_to_workflow_run_uuids[github_job_id]

        if uuid not in workflow_run_uuids_to_test_report_paths:
            logger.warning(
                f"uuid {uuid} for job {github_job_id} does not have a matching test report path, usually meaning the report doesn't exist"
            )
            continue

        github_job_id_to_test_reports[github_job_id] = workflow_run_uuids_to_test_report_paths[uuid]

    return github_job_id_to_test_reports


# Markers printed by the CIv2 runner job-start hook.
# The same strings appear both as GitHub notice annotations and as plain stdout in the
# "Set up runner" step of the job log.
_CIV2_NODE_NAME_LOG_MARKER = "is running on Kubernetes node:"
_CIV2_SERIAL_LOG_MARKER = "serial number(s):"


def get_civ2_node_name_and_serial_from_annotations(annotation_info):
    """Extract the (node_name, serial) a CIv2 runner emitted at job start.

    CIv2 runners emit GitHub Actions notice annotations at job start
    (see tenstorrent/github-ci-infra#1408):
        ::notice title=k8s-node-name::CIV2 runner <name> is running on Kubernetes node: <node>
        ::notice title=tt-card-serial::CIV2 runner <name> has serial number(s): <serial>

    Returns (node_name, serial), each None if its annotation is absent (e.g. CPU-only
    runners have no card serial).
    """
    node_name, serial = None, None
    for _annot in annotation_info or []:
        title = _annot.get("title") or ""
        message = _annot.get("message") or ""
        if title == "k8s-node-name" and "Kubernetes node:" in message:
            node_name = message.rsplit("Kubernetes node:", 1)[-1].strip() or None
        elif title == "tt-card-serial" and _CIV2_SERIAL_LOG_MARKER in message:
            serial = message.rsplit(_CIV2_SERIAL_LOG_MARKER, 1)[-1].strip() or None
    logger.info(f"Extracted node name and serial from annotations: {node_name}, {serial}")
    return node_name, serial


def get_civ2_node_name_and_serial_from_job_log(workflow_outputs_dir, workflow_run_id: int, github_job_id: int):
    """Fallback parser for the CIv2 (node_name, serial) read from the job log.

    Annotations may be absent. The job-start hook also prints these lines to stdout e.g.:
        CIV2 runner <name> is running on Kubernetes node: <node>
        CIV2 runner <name> has serial number(s): <serial>

    Notes:
    I thought there was an instance where this happens,
    but it was actually because annotations don't get "reissued" on passing jobs when another attempt is launched
    (because the job doesn't get rerun)

    So on subsequent workflow run attempts, it looks like passing jobs don't have annotations;
    the annotations are actually still present, but only visible on the prior run attempt on the UI.
    From the API, we can always see the annotations.
    E.g. https://api.github.com/repos/tenstorrent/tt-metal/check-runs/<job id>/annotations

    So we should never have to exercise this fallback parser because we always generate pipeline data on each run attempt's jobs only.
    (Unless github has an outage and doesn't upload annotations for some reason)

    Returns (node_name, serial), each None if not found (CPU-only runners have no serial).
    """
    log_file = _safe_job_log_file(workflow_outputs_dir, workflow_run_id, github_job_id)
    if log_file is None or not log_file.is_file():
        logger.warning(f"Unable to find github job log file for job: {github_job_id}")
        return None, None

    ansi_pattern = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    node_name, serial = None, None
    with open(log_file, "r", encoding="utf-8-sig") as log_f:
        for line in log_f:
            line = ansi_pattern.sub("", line)
            if node_name is None and _CIV2_NODE_NAME_LOG_MARKER in line:
                node_name = line.rsplit(_CIV2_NODE_NAME_LOG_MARKER, 1)[-1].strip() or None
            elif serial is None and _CIV2_SERIAL_LOG_MARKER in line:
                serial = line.rsplit(_CIV2_SERIAL_LOG_MARKER, 1)[-1].strip() or None
            if node_name is not None and serial is not None:
                break
    logger.info(f"Extracted node name and serial from job log: {node_name}, {serial}")
    return node_name, serial


def get_github_job_id_to_annotations(workflow_outputs_dir, workflow_run_id: int):
    # Read <job_id>_annotations.json inside the (sanitized) logs dir.
    logs_dir = _safe_logs_dir(workflow_outputs_dir, workflow_run_id)
    if logs_dir is None or not logs_dir.is_dir():
        logger.warning(f"Annotations logs dir not found for run: {workflow_run_id}")
        return {}

    annot_json_files = logs_dir.glob("*_annotations.json")

    github_job_ids_to_annotation_jsons = {}
    for annot_json_file in annot_json_files:
        annot_json_info = None
        with open(annot_json_file, "r") as f:
            annot_json_info = json.load(f)
        if annot_json_info:
            # Map job id to annotation info (list of dict)
            filename = annot_json_file.name.replace("_annotations.json", "")
            assert filename.isdigit(), f"Unexpected filename format: {filename}"
            github_job_id = int(filename)
            github_job_ids_to_annotation_jsons[github_job_id] = annot_json_info
    return github_job_ids_to_annotation_jsons


def get_pydantic_test_from_testcase_(testcase, default_timestamp=datetime.now(), is_pytest=True, testsuite_name=None):
    skipped = junit_xml_utils.get_testcase_is_skipped(testcase)
    failed = junit_xml_utils.get_testcase_is_failed(testcase)
    error = junit_xml_utils.get_testcase_is_error(testcase)
    success = not (failed or error)

    error_message = None

    # Error is a scarier thing than failure because it means there's an infra error, expose that first
    if failed:
        error_message = junit_xml_utils.get_test_failure_message(testcase)

    if error:
        error_message = junit_xml_utils.get_test_error_message(testcase)

    # Error at the beginning of a test can prevent pytest from recording timestamps at all
    properties = None
    if not (skipped or error):
        if is_pytest:
            properties = junit_xml_utils.get_pytest_testcase_properties(testcase)
            # Check if properties is none to see if pytest recorded the timestamps
            if properties is not None:
                test_start_ts = datetime.fromisoformat(properties["start_timestamp"])
                if "end_timestamp" in properties:
                    test_end_ts = datetime.fromisoformat(properties["end_timestamp"])
                else:
                    # When a setup error occurs in a pytest, the end timestamp may sometimes not be recorded
                    # Set the end timestamp equal to the start timestamp since no test was executed
                    test_end_ts = test_start_ts
            else:
                # Check if there's a time attribute in the testcase
                if "time" in testcase.attrib:
                    pytest_elapsed_time = float(testcase.attrib["time"])
                    test_start_ts = default_timestamp
                    test_end_ts = default_timestamp + timedelta(seconds=pytest_elapsed_time)
                else:
                    test_start_ts = default_timestamp
                    test_end_ts = default_timestamp
        else:
            test_start_ts = default_timestamp
            # gtest stores elapsed time for the test in the time attribute
            gtest_elapsed_time = float(testcase.attrib["time"])
            test_end_ts = default_timestamp + timedelta(seconds=gtest_elapsed_time)
    else:
        test_start_ts = default_timestamp
        test_end_ts = default_timestamp

    test_case_name = testcase.attrib["name"].split("[")[0]

    if is_pytest:
        filepath_no_ext = testcase.attrib["classname"].replace(".", "/")
        filepath = f"{filepath_no_ext}.py"
    else:
        filepath = testcase.attrib.get("file", "")
        if filepath.startswith("/work/"):
            filepath = filepath.lstrip("/work/")

    def get_category_from_testcase_(testcase_, is_pytest=True):
        categories = ["models", "ttnn", "tt_eager", "tt_metal"]
        for category in categories:
            identifier_attrib = "classname" if is_pytest else "file"
            if category in testcase_.attrib.get(identifier_attrib, ""):
                return category
        return "other"

    category = get_category_from_testcase_(testcase, is_pytest=is_pytest)

    # leaving empty for now
    group = None

    # leaving empty for now
    owner = None

    if testsuite_name:
        full_test_name = f"{filepath}::{testsuite_name}::{testcase.attrib['name']}"
    else:
        full_test_name = f"{filepath}::{testcase.attrib['name']}"

    # to be populated with [] if available
    config = None

    tags = None
    if properties and "failure_type" in properties:
        tags = {"failure_type": properties["failure_type"]}

    return pydantic_models.Test(
        test_start_ts=test_start_ts,
        test_end_ts=test_end_ts,
        test_case_name=test_case_name,
        filepath=filepath,
        category=category,
        group=group,
        owner=owner,
        error_message=error_message,
        success=success,
        skipped=skipped,
        full_test_name=full_test_name,
        config=config,
        tags=tags,
    )


def is_valid_testcase_(testcase):
    """
    Some cases of invalid tests include:

    - GitHub times out pytest so it records something like this:
        </testcase>
        <testcase time="0.032"/>
    """
    if "name" not in testcase.attrib or "classname" not in testcase.attrib:
        # This should be able to capture all cases where there's no info
        logger.warning("Found invalid test case with: no name nor classname")
        return False
    else:
        return True


def deduplicate_tests_by_full_name(tests):
    """
    Deduplicate tests based on full_test_name.
    If there are multiple tests with the same name:
    - Take the first one with elapsed time > 0 (test_end_ts != test_start_ts)
    - If they all have 0 elapsed time, take the first instance
    """
    test_name_to_tests = {}

    for test in tests:
        test_name = test.full_test_name
        if test_name not in test_name_to_tests:
            test_name_to_tests[test_name] = []
        test_name_to_tests[test_name].append(test)

    deduplicated_tests = []
    for test_name, test_list in test_name_to_tests.items():
        if len(test_list) == 1:
            # Only one test with this name, keep it
            deduplicated_tests.append(test_list[0])
        else:
            # Multiple tests with same name, apply deduplication logic
            # First, try to find one with elapsed time > 0
            logger.warning(f"Found {len(test_list)} tests with the same full_test_name: {test_name}. Will deduplicate.")
            test_with_elapsed_time = None
            for test in test_list:
                if test.test_end_ts != test.test_start_ts:
                    test_with_elapsed_time = test
                    break

            # If found one with elapsed time > 0, use it; otherwise use the first one
            selected_test = test_with_elapsed_time if test_with_elapsed_time else test_list[0]
            deduplicated_tests.append(selected_test)

    return deduplicated_tests


def get_tests_from_test_report_path(test_report_path):
    report_root_tree = junit_xml_utils.get_xml_file_root_element_tree(test_report_path)

    report_root = report_root_tree.getroot()

    # Special case: Handle ctest: the report root is <testsuite>, not <testsuites>
    if report_root.tag == "testsuite":
        logger.info("Root tag is testsuite, found ctest xml")
        tests = []
        default_timestamp = datetime.fromisoformat(report_root.attrib["timestamp"])
        for testcase in report_root.findall("testcase"):
            if is_valid_testcase_(testcase):
                # Process ctest testcase
                pyd_test_info = get_pydantic_test_from_testcase_(
                    default_timestamp=default_timestamp, is_pytest=False, testsuite_name=None, testcase=testcase
                )
                tests.append(pyd_test_info)
        return deduplicate_tests_by_full_name(tests)

    is_pytest = junit_xml_utils.is_pytest_junit_xml(report_root)
    is_gtest = junit_xml_utils.is_gtest_xml(report_root)

    if is_pytest or is_gtest:
        logger.info(f"Found {len(report_root)} testsuites")
        tests = []
        for i in range(len(report_root)):
            testsuite = report_root[i]
            testsuite_name = testsuite.attrib.get("name") if is_gtest else None
            default_timestamp = datetime.fromisoformat(testsuite.attrib["timestamp"])
            get_pydantic_test = partial(
                get_pydantic_test_from_testcase_,
                default_timestamp=default_timestamp,
                is_pytest=is_pytest,
                testsuite_name=testsuite_name,
            )
            for testcase in testsuite:
                if is_valid_testcase_(testcase):
                    tests.append(get_pydantic_test(testcase))

        return deduplicate_tests_by_full_name(tests)
    else:
        logger.warning("XML is not pytest junit or gtest format, or no tests were found in the XML, skipping for now")
        return []
