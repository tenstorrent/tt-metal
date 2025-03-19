# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pathlib
import re
import json
from datetime import datetime, timedelta
from functools import partial
from typing import List

from loguru import logger

from infra.data_collection import junit_xml_utils, pydantic_models


smi_pattern = re.compile(r'.*"tt_smi":\s*"([a-zA-Z0-9\-\.]+)"')


def search_for_tt_smi_version_in_log_file_(log_file):
    with open(log_file, "r") as log_f:
        for line in log_f:
            regex_match = smi_pattern.match(line)
            if regex_match:
                return regex_match.group(1)
    return None


def get_github_job_ids_to_tt_smi_versions(workflow_outputs_dir, workflow_run_id: int):
    """
    Read the job output log for the tt-smi version. The tt-smi version is printed in the
    Set up runner step, where we call tt-smi-metal -s to dump the smi output.
    The tt-smi version stored in the "host_sw_vers" dict is only available in
    higher versions of tt-smi (3.0.4+). For older versions we will not be able to extract
    the smi version directly from the smi output log dump.
    See: https://github.com/tenstorrent/tt-metal/issues/19095
    """
    logs_dir = workflow_outputs_dir / str(workflow_run_id) / "logs"

    log_files = logs_dir.glob("*.log")

    github_job_ids_to_tt_smi_versions = {}
    for log_file in log_files:
        tt_smi_version = search_for_tt_smi_version_in_log_file_(log_file)
        if tt_smi_version:
            github_job_id = log_file.name.replace(".log", "")
            assert github_job_id.isnumeric(), f"{github_job_id}"
            github_job_id = int(github_job_id)
            github_job_ids_to_tt_smi_versions[github_job_id] = tt_smi_version
    return github_job_ids_to_tt_smi_versions


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
        except FileNotFoundError as e:
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


def get_github_job_id_to_annotations(workflow_outputs_dir, workflow_run_id: int):
    # Read <job_id>_annotations.json inside the logs dir
    logs_dir = workflow_outputs_dir / str(workflow_run_id) / "logs"
    annot_json_files = logs_dir.glob("*_annotations.json")

    github_job_ids_to_annotation_jsons = {}
    for annot_json_file in annot_json_files:
        annot_json_info = None
        with open(annot_json_file, "r") as f:
            annot_json_info = json.load(f)
        if annot_json_info:
            # Map job id to annotation info (list of dict)
            github_job_id = annot_json_file.name.replace("_annotations.json", "")
            assert github_job_id.isnumeric(), f"{github_job_id}"
            github_job_id = int(github_job_id)
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
    if not (skipped or error):
        if is_pytest:
            properties = junit_xml_utils.get_pytest_testcase_properties(testcase)
            # Check if properties is none to see if pytest recorded the timestamps
            if properties is not None:
                test_start_ts = datetime.strptime(properties["start_timestamp"], "%Y-%m-%dT%H:%M:%S")
                test_end_ts = datetime.strptime(properties["end_timestamp"], "%Y-%m-%dT%H:%M:%S")
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


def get_tests_from_test_report_path(test_report_path):
    report_root_tree = junit_xml_utils.get_xml_file_root_element_tree(test_report_path)

    report_root = report_root_tree.getroot()

    # Special case: Handle ctest: the report root is <testsuite>, not <testsuites>
    if report_root.tag == "testsuite":
        logger.info("Root tag is testsuite, found ctest xml")
        tests = []
        # ctest timestamp format is not the same as pytest/gtest
        default_timestamp = datetime.strptime(report_root.attrib["timestamp"], "%Y-%m-%dT%H:%M:%S")
        for testcase in report_root.findall("testcase"):
            if is_valid_testcase_(testcase):
                # Process ctest testcase
                pyd_test_info = get_pydantic_test_from_testcase_(
                    default_timestamp=default_timestamp, is_pytest=False, testsuite_name=None, testcase=testcase
                )
                tests.append(pyd_test_info)
        return tests

    is_pytest = junit_xml_utils.is_pytest_junit_xml(report_root)
    is_gtest = junit_xml_utils.is_gtest_xml(report_root)

    if is_pytest or is_gtest:
        logger.info(f"Found {len(report_root)} testsuites")
        tests = []
        for i in range(len(report_root)):
            testsuite = report_root[i]
            testsuite_name = testsuite.attrib.get("name") if is_gtest else None
            default_timestamp = datetime.strptime(testsuite.attrib["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
            get_pydantic_test = partial(
                get_pydantic_test_from_testcase_,
                default_timestamp=default_timestamp,
                is_pytest=is_pytest,
                testsuite_name=testsuite_name,
            )
            for testcase in testsuite:
                if is_valid_testcase_(testcase):
                    tests.append(get_pydantic_test(testcase))

        return tests
    else:
        logger.warning("XML is not pytest junit or gtest format, or no tests were found in the XML, skipping for now")
        return []
