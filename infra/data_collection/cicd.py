# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
from loguru import logger

from infra.data_collection.github.utils import (
    get_pipeline_row_from_github_info,
    get_job_rows_from_github_info,
    get_data_pipeline_datetime_from_datetime,
)
from infra.data_collection.github.workflows import (
    deduplicate_tests_by_full_name,
    get_github_job_id_to_test_reports,
    get_github_job_id_to_annotations,
    get_tests_from_test_report_path,
    get_github_job_ids_to_tt_smi_versions,
)
from infra.data_collection import pydantic_models
from infra.data_collection.pydantic_models import Step
from infra.data_collection.pydantic_models import TtSmiReset


def get_cicd_json_filename(pipeline):
    github_pipeline_start_ts = get_data_pipeline_datetime_from_datetime(pipeline.pipeline_start_ts)
    github_pipeline_id = pipeline.github_pipeline_id
    cicd_json_filename = f"pipeline_{github_pipeline_id}_{github_pipeline_start_ts}.json"
    return cicd_json_filename


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

    github_pipeline_id = raw_pipeline["github_pipeline_id"]
    github_pipeline_start_ts = raw_pipeline["pipeline_start_ts"]
    workflow_attempt = github_pipeline_json["run_attempt"]
    raw_pipeline["workflow_attempt"] = workflow_attempt

    github_job_id_to_annotations = get_github_job_id_to_annotations(workflow_outputs_dir, github_pipeline_id)

    raw_jobs = get_job_rows_from_github_info(workflow_outputs_dir, github_jobs_json, github_job_id_to_annotations)

    github_job_ids = []
    for raw_job in raw_jobs:
        github_job_id = int(raw_job["github_job_id"])
        github_job_ids.append(github_job_id)

    github_job_id_to_test_reports = get_github_job_id_to_test_reports(
        workflow_outputs_dir, github_pipeline_id, github_job_ids
    )

    github_job_id_to_smi_versions, github_job_id_to_smi_resets = get_github_job_ids_to_tt_smi_versions(workflow_outputs_dir, github_pipeline_id, workflow_attempt)
    jobs = []
    tt_smi_resets = []

    for raw_job in raw_jobs:
        github_job_id = int(raw_job["github_job_id"])

        logger.info(f"Processing raw GitHub job {github_job_id}")

        # Ignore skipped jobs
        # Reason: if an entire matrix is skipped then we can get duplicate skipped jobs with the same pydantic keys
        # Which will fail pydantic model validation.
        if raw_job.get("job_status") == "skipped":
            logger.info(f"Job id:{github_job_id} is skipped. Skipping job upload.")
            continue

        test_report_exists = github_job_id in github_job_id_to_test_reports
        if test_report_exists:
            tests = []
            test_reports = github_job_id_to_test_reports[github_job_id]
            for test_report_path in test_reports:
                logger.info(f"Job id:{github_job_id} Analyzing test report {test_report_path}")
                tests += get_tests_from_test_report_path(test_report_path)
            tests = deduplicate_tests_by_full_name(tests)
        else:
            tests = []

        raw_steps = raw_job.get("steps")
        steps = [Step(**step) for step in raw_steps] if raw_steps else []

        # Remove 'steps' from raw_job to avoid double-passing of 'steps'
        raw_job = dict(raw_job)
        raw_job["workflow_attempt"] = workflow_attempt
        raw_job.pop("steps", None)
        raw_job.pop("tt_smi_reset", None)

        reset_data = github_job_id_to_smi_resets.get((workflow_attempt, github_job_id))

        if reset_data:
            for smi_reset in reset_data:
                smi_reset = dict(smi_reset)
                smi_reset["github_job_id"] = github_job_id
                smi_reset["workflow_attempt"] = workflow_attempt
                tt_smi_resets.append(TtSmiReset(**smi_reset))

        job = pydantic_models.Job(
            **raw_job,
            tt_smi_version=github_job_id_to_smi_versions.get(github_job_id),
            tests=tests,
            steps=steps,
        )
        jobs.append(job)

    # Collect SMI reset data for all earlier workflow attempts.
    # raw_jobs only contains jobs from the latest attempt (workflow_jobs.json is overwritten
    # per attempt during log download). Log files for all attempts are downloaded, so
    # github_job_id_to_smi_resets contains keys for every attempt. We iterate over those
    # earlier-attempt keys here so their reset data is captured in the pipeline record.
    for (attempt_num, job_id), reset_entries in github_job_id_to_smi_resets.items():
        if attempt_num == workflow_attempt:
            continue  # Already handled in the job loop above
        for smi_reset in reset_entries:
            smi_reset = dict(smi_reset)
            smi_reset["github_job_id"] = job_id
            smi_reset["workflow_attempt"] = attempt_num
            tt_smi_resets.append(TtSmiReset(**smi_reset))

    pipeline = pydantic_models.Pipeline(
        **raw_pipeline,
        jobs=jobs,
        tt_smi_resets=tt_smi_resets,
    )

    return pipeline
