# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from pprint import pprint
from loguru import logger

from infra.data_collection.github.utils import (
    get_pipeline_row_from_github_info,
    get_job_rows_from_github_info,
    get_data_pipeline_datetime_from_datetime,
)
from infra.data_collection.github.workflows import (
    get_github_job_id_to_test_reports,
    get_tests_from_test_report_path,
)
from infra.data_collection import pydantic_models


def get_cicd_json_filename(pipeline):
    github_pipeline_start_ts = get_data_pipeline_datetime_from_datetime(pipeline.pipeline_start_ts)
    github_pipeline_id = pipeline.github_pipeline_id
    cicd_json_filename = f"pipeline_{github_pipeline_id}_{github_pipeline_start_ts}.json"


def create_cicd_json_for_data_analysis(
    workflow_outputs_dir,
    github_runner_environment,
    github_pipeline_json_filename,
    github_jobs_json_filename,
):
    with open(github_pipeline_json_filename) as github_pipeline_json_file:
        github_pipeline_json = json.load(github_pipeline_json_file)

    with open(github_jobs_json_filename) as github_jobs_json_file:
        github_jobs_json = json.load(github_jobs_json_file)

    raw_pipeline = get_pipeline_row_from_github_info(github_runner_environment, github_pipeline_json, github_jobs_json)

    raw_jobs = get_job_rows_from_github_info(github_pipeline_json, github_jobs_json)

    github_pipeline_id = raw_pipeline["github_pipeline_id"]
    github_pipeline_start_ts = raw_pipeline["pipeline_start_ts"]

    github_job_ids = []
    for raw_job in raw_jobs:
        github_job_id = int(raw_job["github_job_id"])
        github_job_ids.append(github_job_id)

    github_job_id_to_test_reports = get_github_job_id_to_test_reports(
        workflow_outputs_dir, github_pipeline_id, github_job_ids
    )

    jobs = []

    for raw_job in raw_jobs:
        github_job_id = raw_job["github_job_id"]

        logger.info(f"Processing raw GitHub job {github_job_id}")

        test_report_exists = github_job_id in github_job_id_to_test_reports
        if test_report_exists:
            test_report_path = github_job_id_to_test_reports[github_job_id]
            tests = get_tests_from_test_report_path(test_report_path)
        else:
            tests = []

        logger.info(f"Found {len(tests)} tests for job {github_job_id}")

        job = pydantic_models.Job(
            **raw_job,
            tests=tests,
        )

        jobs.append(job)

    pipeline = pydantic_models.Pipeline(
        **raw_pipeline,
        jobs=jobs,
    )

    return pipeline
