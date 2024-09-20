# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pathlib
from datetime import datetime
from functools import partial
from typing import List

from loguru import logger

from infra.data_collection import junit_xml_utils, pydantic_models


def get_workflow_run_uuids_to_test_reports_paths_(workflow_outputs_dir, workflow_run_id: int):
    artifacts_dir = workflow_outputs_dir / str(workflow_run_id) / "artifacts"

    test_report_dirs = artifacts_dir.glob("test_reports_*")

    workflow_run_test_reports_path = {}
    for test_report_dir in test_report_dirs:
        assert test_report_dir.exists()
        assert test_report_dir.is_dir(), f"{test_report_dir} is not dir"

        test_report_uuid = test_report_dir.name.replace("test_reports_", "")
        workflow_run_test_reports_path[test_report_uuid] = (test_report_dir / "most_recent_tests.xml").resolve(
            strict=True
        )

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
            github_job_id = log_file.name.replace(".log", "")
            assert github_job_id.isnumeric(), f"{github_job_id}"
            github_job_id = int(github_job_id)
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


def get_pydantic_test_from_pytest_testcase_(testcase, default_timestamp=datetime.now()):
    skipped = junit_xml_utils.get_pytest_testcase_is_skipped(testcase)
    failed = junit_xml_utils.get_pytest_testcase_is_failed(testcase)
    error = junit_xml_utils.get_pytest_testcase_is_error(testcase)
    success = not (failed or error)

    error_message = None

    # Error is a scarier thing than failure because it means there's an infra error, expose that first
    if failed:
        error_message = junit_xml_utils.get_pytest_failure_message(testcase)

    if error:
        error_message = junit_xml_utils.get_pytest_error_message(testcase)

    # Error at the beginning of a test can prevent pytest from recording timestamps at all
    if not (skipped or error):
        properties = junit_xml_utils.get_pytest_testcase_properties(testcase)
        test_start_ts = datetime.strptime(properties["start_timestamp"], "%Y-%m-%dT%H:%M:%S")
        test_end_ts = datetime.strptime(properties["end_timestamp"], "%Y-%m-%dT%H:%M:%S")
    else:
        test_start_ts = default_timestamp
        test_end_ts = default_timestamp

    test_case_name = testcase.attrib["name"].split("[")[0]

    filepath_no_ext = testcase.attrib["classname"].replace(".", "/")
    filepath = f"{filepath_no_ext}.py"

    def get_category_from_pytest_testcase_(testcase_):
        categories = ["models", "ttnn", "tt_eager", "tt_metal"]
        for category in categories:
            if category in testcase_.attrib["classname"]:
                return category
        return "other"

    category = get_category_from_pytest_testcase_(testcase)

    # leaving empty for now
    group = None

    # leaving empty for now
    owner = None

    full_test_name = f"{filepath}::{testcase.attrib['name']}"

    # to be populated with [] if available
    config = None

    tags = None

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


def get_tests_from_test_report_path(test_report_path):
    report_root_tree = junit_xml_utils.get_xml_file_root_element_tree(test_report_path)

    report_root = report_root_tree.getroot()

    is_pytest = junit_xml_utils.is_pytest_junit_xml(report_root)

    if is_pytest:
        testsuite = report_root[0]
        default_timestamp = datetime.strptime(testsuite.attrib["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")

        get_pydantic_test = partial(get_pydantic_test_from_pytest_testcase_, default_timestamp=default_timestamp)

        return list(map(get_pydantic_test, testsuite))
    else:
        raise Exception("We only support pytest junit xml outputs for now")
