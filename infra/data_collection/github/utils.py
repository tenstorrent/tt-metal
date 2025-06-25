# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import csv
import pathlib
import os
from datetime import datetime
from typing import Optional, Union

from loguru import logger

from infra.data_collection.github.workflows import is_job_hanging_from_job_log
from infra.data_collection.models import InfraErrorV1, TestErrorV1
from infra.data_collection.pydantic_models import CompleteBenchmarkRun


def get_datetime_from_github_datetime(github_datetime):
    return datetime.strptime(github_datetime, "%Y-%m-%dT%H:%M:%SZ")


def get_data_pipeline_datetime_from_datetime(requested_datetime):
    return requested_datetime.strftime("%Y-%m-%dT%H:%M:%S%z")


def get_pipeline_row_from_github_info(github_runner_environment, github_pipeline_json, github_jobs_json):
    github_pipeline_id = github_pipeline_json["id"]
    pipeline_submission_ts = github_pipeline_json["created_at"]

    repository_url = github_pipeline_json["repository"]["html_url"]

    jobs = github_jobs_json["jobs"]
    jobs_start_times = list(map(lambda job_: get_datetime_from_github_datetime(job_["started_at"]), jobs))
    # We filter out jobs that started before because that means they're from a previous attempt for that pipeline
    eligible_jobs_start_times = list(
        filter(
            lambda job_start_time_: job_start_time_ >= get_datetime_from_github_datetime(pipeline_submission_ts),
            jobs_start_times,
        )
    )
    sorted_jobs_start_times = sorted(eligible_jobs_start_times)
    assert (
        sorted_jobs_start_times
    ), f"It seems that this pipeline does not have any jobs that started on or after the pipeline was submitted, which should be impossible. Please directly inspect the JSON objects"
    pipeline_start_ts = get_data_pipeline_datetime_from_datetime(sorted_jobs_start_times[0])

    pipeline_end_ts = github_pipeline_json["updated_at"]
    name = github_pipeline_json["name"]

    project = github_pipeline_json["repository"]["name"]

    trigger = github_runner_environment["github_event_name"]

    logger.warning("Using hardcoded value github for vcs_platform value")
    vcs_platform = "github"

    git_branch_name = github_pipeline_json["head_branch"]

    git_commit_hash = github_pipeline_json["head_sha"]

    git_author = github_pipeline_json["head_commit"]["author"]["name"]

    logger.warning("Using hardcoded value github_actions for orchestrator value")
    orchestrator = "github_actions"

    github_pipeline_link = github_pipeline_json["html_url"]

    pipeline_status = github_pipeline_json["conclusion"]

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
        "git_branch_name": git_branch_name,
        "git_commit_hash": git_commit_hash,
        "git_author": git_author,
        "orchestrator": orchestrator,
        "github_pipeline_link": github_pipeline_link,
        "pipeline_status": pipeline_status,
    }


def return_first_string_starts_with(starting_string, strings):
    for string in strings:
        if string.startswith(starting_string):
            return string
    raise Exception(f"{strings} do not have any that match {starting_string}")


def get_job_failure_signature_(github_job, failure_description, workflow_outputs_dir) -> Optional[Union[InfraErrorV1]]:
    error_snippet_to_signature_mapping = {
        "has timed out": str(InfraErrorV1.JOB_UNIT_TIMEOUT_FAILURE),
        "exceeded the maximum execution time": str(InfraErrorV1.JOB_CUMULATIVE_TIMEOUT_FAILURE),
        "lost communication with the server": str(InfraErrorV1.RUNNER_COMM_FAILURE),
        "runner has received a shutdown signal": str(InfraErrorV1.RUNNER_SHUTDOWN_FAILURE),
        "No space left on device": str(InfraErrorV1.DISK_SPACE_FAILURE),
        "API rate limit exceeded": str(InfraErrorV1.API_RATE_LIMIT_FAILURE),
        "Tenstorrent cards seem to be in use": str(InfraErrorV1.RUNNER_CARD_IN_USE_FAILURE),
    }

    # Check the mapping dictionary for specific failure signature types
    for error_snippet in error_snippet_to_signature_mapping:
        if error_snippet in failure_description:
            error_signature = error_snippet_to_signature_mapping[error_snippet]
            # Determine if timeout is a hang
            if error_signature in [
                str(InfraErrorV1.JOB_CUMULATIVE_TIMEOUT_FAILURE),
                str(InfraErrorV1.JOB_UNIT_TIMEOUT_FAILURE),
            ] and is_job_hanging_from_job_log(
                error_snippet,
                workflow_outputs_dir=workflow_outputs_dir,
                workflow_run_id=github_job["run_id"],
                workflow_job_id=github_job["id"],
            ):
                error_signature = str(InfraErrorV1.JOB_HANG)
            return error_signature

    # If failure occurred in runner setup, classify as set up failure
    for step in github_job["steps"]:
        is_generic_setup_failure = (
            step["name"] == "Set up runner"
            and step["status"] in ("completed", "cancelled")
            and step["conclusion"] != "success"
            and step["started_at"] is not None
            and step["completed_at"] is None
        )
        if is_generic_setup_failure:
            return str(InfraErrorV1.GENERIC_SET_UP_FAILURE)

    # generic catch-all
    return str(InfraErrorV1.GENERIC_FAILURE)


def get_failure_signature_and_description_from_annotations(
    github_job, github_job_id_to_annotations, workflow_outputs_dir
):
    failure_signature, failure_description = None, None

    # Don't return any failure info if job passed
    if github_job["conclusion"] == "success":
        return failure_signature, failure_description

    # Otherwise, check the job's annotation info for failure reason
    job_id = github_job["id"]
    if job_id in github_job_id_to_annotations:
        annotation_info = github_job_id_to_annotations[job_id]

        for _annot in annotation_info:
            if _annot["annotation_level"] == "failure":
                # Unit test failure: a failure exists where the annotation path is not .github
                if _annot["path"] != ".github":
                    failure_description = _annot["path"]
                    if ".py" in failure_description:
                        failure_signature = str(TestErrorV1.PY_TEST_FAILURE)
                    elif ".cpp" in failure_description:
                        failure_signature = str(TestErrorV1.CPP_TEST_FAILURE)
                    else:
                        failure_signature = str(TestErrorV1.UNKNOWN_TEST_FAILURE)
                    return failure_signature, failure_description
                else:
                    # Infrastructure error
                    failure_description = _annot.get("message")
                    if failure_description:
                        failure_signature = get_job_failure_signature_(
                            github_job, failure_description, workflow_outputs_dir
                        )
                        return failure_signature, failure_description
    return failure_signature, failure_description


def get_job_row_from_github_job(github_job, github_job_id_to_annotations, workflow_outputs_dir):
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

    # Clean up ephemeral runner names
    if host_name and host_name.startswith("tt-beta"):
        parts = host_name.split("-")
        # Issue: https://github.com/tenstorrent/tt-metal/issues/21694
        # Remove non-constant ephemeral runner suffix from tt-beta runner names only if the second last part is "runner"
        # We don't want to remove the suffix for non-ephemeral tt-beta runners (e.g. tt-beta-ubuntu-2204-xlarge)
        # E.g. tt-beta-ubuntu-2204-n150-large-stable-nk6pd-runner-5g5f9 -> tt-beta-ubuntu-2204-n150-large-stable-nk6pd
        if len(parts) >= 2 and parts[-2] == "runner":
            host_name = "-".join(parts[:-1])

    os = ubuntu_version

    name = github_job["name"]

    if github_job["status"] != "completed":
        logger.warning(f"{github_job_id} is not completed, skipping this job")
        return None

    # Best effort card type getting

    get_overlap = lambda labels_a, labels_b: set(labels_a) & set(labels_b)
    labels_have_overlap = lambda labels_a, labels_b: bool(get_overlap(labels_a, labels_b))

    try:
        detected_config = return_first_string_starts_with("config-", labels).replace("config-", "")
    except Exception as e:
        logger.error(e)
        logger.info("Seems to have no config- label, so assuming no special config requested")
        detected_config = None

    if labels_have_overlap(["E150", "grayskull", "arch-grayskull"], labels):
        detected_arch = "grayskull"
    elif labels_have_overlap(["N150", "N300", "wormhole_b0", "arch-wormhole_b0", "config-t3000"], labels):
        detected_arch = "wormhole_b0"
    elif labels_have_overlap(["BH", "arch-blackhole"], labels):
        detected_arch = "blackhole"
    else:
        detected_arch = None

    single_cards_list = ("E150", "N150", "N300", "BH")
    single_cards_overlap = get_overlap(single_cards_list, labels)

    # In order of preference
    if detected_config:
        if not detected_arch:
            # This will occur for jobs where runs-on: has a config-* label but doesn't have an arch-* or card-specific label
            logger.warning(f"No arch label found for config {detected_config} in job label, unable to infer card type")
            card_type = None
        else:
            card_type = f"{detected_config}-{detected_arch}"
    elif single_cards_overlap:
        logger.info(f"Detected overlap in single cards: {single_cards_overlap}")
        card_type = list(single_cards_overlap)[0]
    elif detected_arch:
        card_type = detected_arch
    else:
        card_type = None

    job_submission_ts = github_job["created_at"]

    job_start_ts = github_job["started_at"]

    job_submission_ts_dt = get_datetime_from_github_datetime(job_submission_ts)
    job_start_ts_dt = get_datetime_from_github_datetime(job_start_ts)

    if job_submission_ts_dt > job_start_ts_dt or github_job["conclusion"] == "skipped":
        if github_job["conclusion"] == "skipped":
            # When the job is skipped, github may set the start timestamp to an invalid value
            # In this case, just set the started_at timestamp to the created_at timestamp
            # See https://github.com/tenstorrent/tt-metal/issues/24151 for an example
            logger.warning(
                f"Job {github_job_id} is skipped. Setting start timestamp equal to submission timestamp for data"
            )
            job_start_ts = job_submission_ts
        else:
            logger.warning(
                f"Job {github_job_id} seems to have a start time that's earlier than submission. Setting start timestamp equal to submission timestamp for data"
            )
            job_submission_ts = job_start_ts

    job_end_ts = github_job["completed_at"]

    # skipped jobs are considered passing jobs (nothing was run)
    job_success = github_job["conclusion"] in ["success", "skipped"]
    job_status = github_job["conclusion"]

    is_build_job = "build" in name or "build" in labels

    logger.warning("Returning None for job_matrix_config because difficult to get right now")
    job_matrix_config = None

    logger.warning("docker_image erroneously used in pipeline data model, but should be moved. Returning null")
    docker_image = None

    github_job_link = github_job["html_url"]

    failure_signature, failure_description = get_failure_signature_and_description_from_annotations(
        github_job, github_job_id_to_annotations, workflow_outputs_dir
    )

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
        "job_status": job_status,
        "is_build_job": is_build_job,
        "job_matrix_config": job_matrix_config,
        "docker_image": docker_image,
        "github_job_link": github_job_link,
        "failure_signature": failure_signature,
        "failure_description": failure_description,
        "job_label": ",".join(labels),
    }


def get_job_rows_from_github_info(workflow_outputs_dir, github_jobs_json, github_job_id_to_annotations):
    job_rows = list(
        map(
            lambda job: get_job_row_from_github_job(job, github_job_id_to_annotations, workflow_outputs_dir),
            github_jobs_json["jobs"],
        )
    )
    return [x for x in job_rows if x is not None]


def get_github_partial_benchmark_json_filenames():
    logger.info("We are assuming generated/benchmark_data exists from previous passing test")

    current_utils_path = pathlib.Path(__file__)
    benchmark_data_dir = current_utils_path.parent.parent.parent.parent / "generated/benchmark_data"
    assert benchmark_data_dir.exists()
    assert benchmark_data_dir.is_dir()

    benchmark_json_paths = list(benchmark_data_dir.glob("partial_run_*.json"))
    assert len(
        benchmark_json_paths
    ), f"There needs to be at least one benchmark data json since we're completing the environment data for each one"

    logger.info(
        f"The following partial benchmark data JSONs should be completed with environment data: {benchmark_json_paths}"
    )
    return benchmark_json_paths


def get_github_runner_environment():
    assert "GITHUB_EVENT_NAME" in os.environ
    github_event_name = os.environ["GITHUB_EVENT_NAME"]

    return {
        "github_event_name": github_event_name,
    }


def create_json_with_github_benchmark_environment(github_partial_benchmark_json_filename):
    assert "GITHUB_REPOSITORY" in os.environ
    git_repo_name = os.environ["GITHUB_REPOSITORY"]

    assert "GITHUB_SHA" in os.environ
    git_commit_hash = os.environ["GITHUB_SHA"]

    logger.warning("Hardcoded null for git_commit_ts")
    git_commit_ts = None

    assert "GITHUB_REF_NAME" in os.environ
    git_branch_name = os.environ["GITHUB_REF_NAME"]

    assert "GITHUB_RUN_ID" in os.environ
    github_pipeline_id = os.environ["GITHUB_RUN_ID"]

    github_pipeline_link = f"https://github.com/{git_repo_name}/actions/runs/{github_pipeline_id}"

    logger.warning("Hardcoded null for github_job_id")
    github_job_id = None

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

    device_info = {"card_type": device_type, "dram_size": device_memory_size}

    with open(github_partial_benchmark_json_filename, "r") as f:
        partial_benchmark_data = json.load(f)

    partial_benchmark_data["git_repo_name"] = git_repo_name
    partial_benchmark_data["git_commit_hash"] = git_commit_hash
    partial_benchmark_data["git_commit_ts"] = git_commit_ts
    partial_benchmark_data["git_branch_name"] = git_branch_name
    partial_benchmark_data["github_pipeline_id"] = github_pipeline_id
    partial_benchmark_data["github_pipeline_link"] = github_pipeline_link
    partial_benchmark_data["github_job_id"] = github_job_id
    partial_benchmark_data["user_name"] = user_name
    partial_benchmark_data["docker_image"] = docker_image
    partial_benchmark_data["device_hostname"] = device_hostname
    partial_benchmark_data["device_ip"] = device_ip
    partial_benchmark_data["device_info"] = device_info

    complete_benchmark_run = CompleteBenchmarkRun(**partial_benchmark_data)

    json_data = complete_benchmark_run.model_dump_json()

    # Save complete run json
    output_path = pathlib.Path(str(github_partial_benchmark_json_filename).replace("partial_run_", "complete_run_"))
    with open(output_path, "w") as f:
        f.write(json_data)

    # Delete partial run json
    os.remove(github_partial_benchmark_json_filename)
