# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import csv
import pathlib
import os
from datetime import datetime

from loguru import logger

PIPELINE_CSV_FIELDS = (
    "github_pipeline_id",
    "pipeline_submission_ts",
    "pipeline_start_ts",
    "pipeline_end_ts",
    "name",
    "project",
    "trigger",
    "vcs_platform",
    "git_commit_hash",
    "git_author",
    "orchestrator",
)

JOB_CSV_FIELDS = (
    "github_job_id",
    "host_name",
    "host_card_type",
    "host_os",
    "host_location",
    "name",
    "job_submission_ts",
    "job_start_ts",
    "job_end_ts",
    "job_success",
    "is_build_job",
    "job_matrix_config",
    "docker_image",
)


BENCHMARK_ENVIRONMENT_CSV_FIELDS = (
    "git_repo_name",
    "git_commit_hash",
    "git_commit_ts",
    "github_pipeline_id",
    "github_job_id",
    "user_name",
    "docker_image",
    "device_hostname",
    "device_ip",
    "device_type",
    "device_memory_size",
)


def assert_all_fieldnames_exist(fieldnames, row):
    assert set(row.keys()) == set(fieldnames)


def create_csv(filename, fieldnames, rows):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            assert_all_fieldnames_exist(fieldnames, row)
            writer.writerow(row)

    logger.info(f"Finished writing to file {filename}")


def get_datetime_from_github_datetime(github_datetime):
    return datetime.strptime(github_datetime, "%Y-%m-%dT%H:%M:%SZ")


def get_data_pipeline_datetime_from_datetime(requested_datetime):
    return requested_datetime.strftime("%Y-%m-%dT%H:%M:%S%z")


def get_pipeline_row_from_github_info(github_context_json, github_pipeline_json, github_jobs_json):
    github_pipeline_id = github_pipeline_json["id"]
    pipeline_submission_ts = github_pipeline_json["created_at"]

    jobs = github_jobs_json["jobs"]
    jobs_start_times = list(map(lambda job_: get_datetime_from_github_datetime(job_["started_at"]), jobs))
    sorted_jobs_start_times = sorted(jobs_start_times)
    pipeline_start_ts = get_data_pipeline_datetime_from_datetime(sorted_jobs_start_times[0])

    pipeline_end_ts = github_pipeline_json["created_at"]
    name = github_pipeline_json["name"]

    logger.warning("Using hardcoded value tt-metal for project value")
    project = "tt-metal"

    trigger = github_context_json["event_name"]

    logger.warning("Using hardcoded value github for vcs_platform value")
    vcs_platform = "github"

    git_commit_hash = github_pipeline_json["head_sha"]

    git_author = github_pipeline_json["head_commit"]["author"]["name"]

    logger.warning("Using hardcoded value github_actions for orchestrator value")
    orchestrator = "github_actions"

    return {
        "github_pipeline_id": github_pipeline_id,
        "pipeline_submission_ts": pipeline_submission_ts,
        "pipeline_start_ts": pipeline_start_ts,
        "pipeline_end_ts": pipeline_end_ts,
        "name": name,
        "project": project,
        "trigger": trigger,
        "vcs_platform": vcs_platform,
        "git_commit_hash": git_commit_hash,
        "git_author": git_author,
        "orchestrator": orchestrator,
    }


def return_first_string_starts_with(starting_string, strings):
    for string in strings:
        if string.startswith(starting_string):
            return string
    raise Exception(f"{strings} do not have any that match {starting_string}")


def get_job_row_from_github_job(github_job):
    github_job_id = github_job["id"]

    logger.info(f"Processing github job with ID {github_job_id}")

    host_name = github_job["runner_name"]

    labels = github_job["labels"]

    if not host_name:
        logger.debug("Detected null host_name, so will return null host_location")
        host_location = None
    elif "GitHub Actions " in host_name:
        host_location = "github"
    else:
        host_location = "tt_cloud"

    if host_location == "github":
        try:
            ubuntu_version = return_first_string_starts_with("ubuntu-", labels)
        except Exception as e:
            logger.error(e)
            logger.error(f"{labels} for a GitHub runner seem to not specify an ubuntu version")
            raise e
        if ubuntu_version == "ubuntu-latest":
            logger.warning("Found ubuntu-latest, replacing with ubuntu-22.04 but may not be case for long")
            ubuntu_version = "ubuntu-22.04"
    elif host_location == "tt_cloud":
        logger.warning("Assuming ubuntu-20.04 for tt cloud, but may not be the case soon")
        ubuntu_version = "ubuntu-20.04"
    else:
        ubuntu_version = None

    host_os = ubuntu_version

    name = github_job["name"]

    assert github_job["status"] == "completed", f"{github_job_id} is not completed"

    logger.warning(
        "Using labels to heuristically look for card type, but we should be using future arch- label instead"
    )
    if "grayskull" in labels:
        host_card_type = "grayskull"
    elif "wormhole_b0" in labels:
        host_card_type = "wormhole_b0"
    else:
        host_card_type = "unknown"

    job_submission_ts = github_job["created_at"]

    job_start_ts = github_job["started_at"]

    job_end_ts = github_job["completed_at"]

    job_success = github_job["conclusion"] == "success"

    is_build_job = "build" in name or "build" in labels

    logger.warning("Returning null for job_matrix_config because difficult to get right now")
    job_matrix_config = "null"

    logger.warning("docker_image erroneously used in pipeline data model, but should be moved. Returning null")
    docker_image = None

    return {
        "github_job_id": github_job_id,
        "host_name": host_name,
        "host_card_type": host_card_type,
        "host_os": host_os,
        "host_location": host_location,
        "name": name,
        "job_submission_ts": job_submission_ts,
        "job_start_ts": job_start_ts,
        "job_end_ts": job_end_ts,
        "job_success": job_success,
        "is_build_job": is_build_job,
        "job_matrix_config": job_matrix_config,
        "docker_image": docker_image,
    }


def get_job_rows_from_github_info(github_pipeline_json, github_jobs_json):
    return list(map(get_job_row_from_github_job, github_jobs_json["jobs"]))


def create_csvs_for_data_analysis(
    github_context_json_filename,
    github_pipeline_json_filename,
    github_jobs_json_filename,
    github_pipeline_csv_filename=None,
    github_jobs_csv_filename=None,
):
    with open(github_context_json_filename) as github_context_json_file:
        github_context_json = json.load(github_context_json_file)

    with open(github_pipeline_json_filename) as github_pipeline_json_file:
        github_pipeline_json = json.load(github_pipeline_json_file)

    with open(github_jobs_json_filename) as github_jobs_json_file:
        github_jobs_json = json.load(github_jobs_json_file)

    pipeline_row = get_pipeline_row_from_github_info(github_context_json, github_pipeline_json, github_jobs_json)

    job_rows = get_job_rows_from_github_info(github_pipeline_json, github_jobs_json)

    github_pipeline_id = pipeline_row["github_pipeline_id"]
    github_pipeline_start_ts = pipeline_row["pipeline_start_ts"]

    if not github_pipeline_csv_filename:
        github_pipeline_csv_filename = f"pipeline_{github_pipeline_id}_{github_pipeline_start_ts}.csv"

    if not github_jobs_csv_filename:
        github_jobs_csv_filename = f"job_{github_pipeline_id}_{github_pipeline_start_ts}.csv"

    create_csv(github_pipeline_csv_filename, PIPELINE_CSV_FIELDS, [pipeline_row])
    create_csv(github_jobs_csv_filename, JOB_CSV_FIELDS, job_rows)


def get_github_benchmark_environment_csv_filename():
    logger.info("We are assuming generated/benchmark_data exists from previous passing test")

    current_utils_path = pathlib.Path(__file__)
    benchmark_data_dir = current_utils_path.parent.parent.parent.parent / "generated/benchmark_data"
    assert benchmark_data_dir.exists()
    assert benchmark_data_dir.is_dir()

    measurement_csv_paths = list(benchmark_data_dir.glob("measurement_*.csv"))
    assert (
        len(measurement_csv_paths) == 1
    ), f"There needs to be exactly one measurement csv as we're assuming a single test is run for now: {measurement_csv_paths}"
    timestamp_str = str(measurement_csv_paths[0].name).replace("measurement_", "").replace(".csv", "")

    csv_filename = str(benchmark_data_dir / f"environment_{timestamp_str}.csv")
    logger.info(f"Saving to {csv_filename}")
    return csv_filename


def create_csv_for_github_benchmark_environment(github_benchmark_environment_csv_filename):
    assert "GITHUB_REPOSITORY" in os.environ
    git_repo_name = os.environ["GITHUB_REPOSITORY"]

    assert "GITHUB_SHA" in os.environ
    git_commit_hash = os.environ["GITHUB_SHA"]

    logger.warning("Hardcoded null for git_commit_ts")
    git_commit_ts = ""

    assert "GITHUB_RUN_ID" in os.environ
    github_pipeline_id = os.environ["GITHUB_RUN_ID"]

    logger.warning("Hardcoded null for github_job_id")
    github_job_id = ""

    assert "GITHUB_TRIGGERING_ACTOR" in os.environ
    user_name = os.environ["GITHUB_TRIGGERING_ACTOR"]

    logger.warning("Hardcoded null for ")
    docker_image = ""

    assert "RUNNER_NAME" in os.environ
    device_hostname = os.environ["RUNNER_NAME"]

    logger.warning("Hardcoded null for device_ip")
    device_ip = ""

    assert "ARCH_NAME" in os.environ
    device_type = os.environ["ARCH_NAME"]
    assert device_type in ("grayskull", "wormhole_b0")

    logger.warning("Hardcoded null for device_memory_size")
    device_memory_size = ""

    benchmark_environment_row = {
        "git_repo_name": git_repo_name,
        "git_commit_hash": git_commit_hash,
        "git_commit_ts": git_commit_ts,
        "github_pipeline_id": github_pipeline_id,
        "github_job_id": github_job_id,
        "user_name": user_name,
        "docker_image": docker_image,
        "device_hostname": device_hostname,
        "device_ip": device_ip,
        "device_type": device_type,
        "device_memory_size": device_memory_size,
    }

    create_csv(github_benchmark_environment_csv_filename, BENCHMARK_ENVIRONMENT_CSV_FIELDS, [benchmark_environment_row])
