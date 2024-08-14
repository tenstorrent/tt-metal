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
    "card_type",
    "os",
    "location",
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
    "device_info",
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


def get_pipeline_row_from_github_info(github_runner_environment, github_pipeline_json, github_jobs_json):
    github_pipeline_id = github_pipeline_json["id"]
    pipeline_submission_ts = github_pipeline_json["created_at"]

    logger.warning("Using hardcoded value for repository_url")
    repository_url = "https://github.com/tenstorrent/tt-metal"

    jobs = github_jobs_json["jobs"]
    jobs_start_times = list(map(lambda job_: get_datetime_from_github_datetime(job_["started_at"]), jobs))
    sorted_jobs_start_times = sorted(jobs_start_times)
    pipeline_start_ts = get_data_pipeline_datetime_from_datetime(sorted_jobs_start_times[0])

    pipeline_end_ts = github_pipeline_json["updated_at"]
    name = github_pipeline_json["name"]

    logger.warning("Using hardcoded value tt-metal for project value")
    project = "tt-metal"

    trigger = github_runner_environment["github_event_name"]

    logger.warning("Using hardcoded value github for vcs_platform value")
    vcs_platform = "github"

    git_commit_hash = github_pipeline_json["head_sha"]

    git_author = github_pipeline_json["head_commit"]["author"]["name"]

    logger.warning("Using hardcoded value github_actions for orchestrator value")
    orchestrator = "github_actions"

    return {
        "github_pipeline_id": github_pipeline_id,
        "repository_url": repository_url,
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
        location = None
        host_name = None
    elif "GitHub Actions " in host_name:
        location = "github"
    else:
        location = "tt_cloud"

    if location == "github":
        try:
            ubuntu_version = return_first_string_starts_with("ubuntu-", labels)
        except Exception as e:
            logger.error(e)
            logger.error(f"{labels} for a GitHub runner seem to not specify an ubuntu version")
            raise e
        if ubuntu_version == "ubuntu-latest":
            logger.warning("Found ubuntu-latest, replacing with ubuntu-24.04 but may not be case for long")
            ubuntu_version = "ubuntu-24.04"
    elif location == "tt_cloud":
        logger.warning("Assuming ubuntu-20.04 for tt cloud, but may not be the case soon")
        ubuntu_version = "ubuntu-20.04"
    else:
        ubuntu_version = None

    os = ubuntu_version

    name = github_job["name"]

    assert github_job["status"] == "completed", f"{github_job_id} is not completed"

    # Best effort card type getting

    get_overlap = lambda labels_a, labels_b: set(labels_a) & set(labels_b)
    labels_have_overlap = lambda labels_a, labels_b: bool(get_overlap(labels_a, labels_b))

    try:
        detected_config = return_first_string_starts_with("config-", labels).replace("config-", "")
    except Exception as e:
        logger.error(e)
        logger.info("Seems to have no config- label, so assuming no special config requested")
        detected_config = None

    if labels_have_overlap(["grayskull", "arch-grayskull"], labels):
        detected_arch = "grayskull"
    elif labels_have_overlap(["wormhole_b0", "arch-wormhole_b0"], labels):
        detected_arch = "wormhole_b0"
    elif labels_have_overlap(["arch-blackhole"], labels):
        detected_arch = "blackhole"
    else:
        detected_arch = None

    single_cards_list = ("E150", "N150", "N300", "BH")
    single_cards_overlap = get_overlap(single_cards_list, labels)

    if detected_config:
        if not detected_arch:
            raise Exception(f"There must be an arch detected for config {detected_config}")
        card_type = f"{detected_config}-{detected_arch}"
    elif single_cards_overlap:
        logger.info(f"Detected overlap in single cards: {single_cards_overlap}")
        card_type = list(single_cards_overlap)[0]
    else:
        card_type = None

    job_submission_ts = github_job["created_at"]

    job_start_ts = github_job["started_at"]

    job_submission_ts_dt = get_datetime_from_github_datetime(job_submission_ts)
    job_start_ts_dt = get_datetime_from_github_datetime(job_start_ts)

    if job_submission_ts_dt > job_start_ts_dt:
        logger.warning(
            f"Job {github_job_id} seems to have a start time that's earlier than submission. Setting equal for data"
        )
        job_submission_ts = job_start_ts

    job_end_ts = github_job["completed_at"]

    job_success = github_job["conclusion"] == "success"

    is_build_job = "build" in name or "build" in labels

    logger.warning("Returning None for job_matrix_config because difficult to get right now")
    job_matrix_config = None

    logger.warning("docker_image erroneously used in pipeline data model, but should be moved. Returning null")
    docker_image = None

    return {
        "github_job_id": github_job_id,
        "host_name": host_name,
        "card_type": card_type,
        "os": os,
        "location": location,
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
    github_runner_environment,
    github_pipeline_json_filename,
    github_jobs_json_filename,
    github_pipeline_csv_filename=None,
    github_jobs_csv_filename=None,
):
    with open(github_pipeline_json_filename) as github_pipeline_json_file:
        github_pipeline_json = json.load(github_pipeline_json_file)

    with open(github_jobs_json_filename) as github_jobs_json_file:
        github_jobs_json = json.load(github_jobs_json_file)

    pipeline_row = get_pipeline_row_from_github_info(github_runner_environment, github_pipeline_json, github_jobs_json)

    job_rows = get_job_rows_from_github_info(github_pipeline_json, github_jobs_json)

    github_pipeline_id = pipeline_row["github_pipeline_id"]
    github_pipeline_start_ts = pipeline_row["pipeline_start_ts"]

    if not github_pipeline_csv_filename:
        github_pipeline_csv_filename = f"pipeline_{github_pipeline_id}_{github_pipeline_start_ts}.csv"

    if not github_jobs_csv_filename:
        github_jobs_csv_filename = f"job_{github_pipeline_id}_{github_pipeline_start_ts}.csv"

    create_csv(github_pipeline_csv_filename, PIPELINE_CSV_FIELDS, [pipeline_row])
    create_csv(github_jobs_csv_filename, JOB_CSV_FIELDS, job_rows)


def get_github_benchmark_environment_csv_filenames():
    logger.info("We are assuming generated/benchmark_data exists from previous passing test")

    current_utils_path = pathlib.Path(__file__)
    benchmark_data_dir = current_utils_path.parent.parent.parent.parent / "generated/benchmark_data"
    assert benchmark_data_dir.exists()
    assert benchmark_data_dir.is_dir()

    measurement_csv_paths = list(benchmark_data_dir.glob("measurement_*.csv"))
    assert len(
        measurement_csv_paths
    ), f"There needs to be at least one measurement csv since we're making an environment CSV for each one"
    timestamp_strs = list(
        map(
            lambda csv_path_: str(csv_path_.name).replace("measurement_", "").replace(".csv", ""), measurement_csv_paths
        )
    )

    csv_filenames = list(
        map(lambda timestamp_str_: str(benchmark_data_dir / f"environment_{timestamp_str_}.csv"), timestamp_strs)
    )
    logger.info(f"The following environment CSVs should be created: {csv_filenames}")
    return csv_filenames


def get_github_runner_environment():
    assert "GITHUB_EVENT_NAME" in os.environ
    github_event_name = os.environ["GITHUB_EVENT_NAME"]

    return {
        "github_event_name": github_event_name,
    }


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
    assert device_type in ("grayskull", "wormhole_b0", "blackhole")

    logger.warning("Hardcoded null for device_memory_size")
    device_memory_size = ""

    device_info = json.dumps(
        {
            "device_type": device_type,
            "device_memory_size": device_memory_size,
        }
    )

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
        "device_info": device_info,
    }

    create_csv(github_benchmark_environment_csv_filename, BENCHMARK_ENVIRONMENT_CSV_FIELDS, [benchmark_environment_row])
