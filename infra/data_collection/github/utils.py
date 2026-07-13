# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import functools
import pathlib
import pickle
import os
from datetime import datetime
from typing import Optional, Union

import yaml
from loguru import logger

from infra.data_collection.github.workflows import is_job_hanging_from_job_log
from infra.data_collection.models import InfraErrorV1, TestErrorV1, CodeQualityErrorV1
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
    workflow_attempt = github_pipeline_json["run_attempt"]

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
        "workflow_attempt": workflow_attempt,
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
        "Error response from daemon": str(InfraErrorV1.DOCKER_REGISTRY_FAILURE),
        "Failed to CreateArtifact": str(InfraErrorV1.ARTIFACT_UPLOAD_FAILURE),
        "device timeout, potential hang detected, the device is unrecoverable": str(InfraErrorV1.TT_TRIAGE_JOB_HANG),
        # Git checkout / submodule clone failures (transient GitHub infra issues)
        "fatal: clone of": str(InfraErrorV1.CHECKOUT_FAILURE),
        "Failed to clone": str(InfraErrorV1.CHECKOUT_FAILURE),
        "could not read Username": str(InfraErrorV1.CHECKOUT_FAILURE),
        "terminal prompts disabled": str(InfraErrorV1.CHECKOUT_FAILURE),
        "Fetched in submodule path": str(InfraErrorV1.CHECKOUT_FAILURE),
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

    # If failure occurred in a checkout step, classify as checkout failure
    for step in github_job.get("steps", []):
        step_name = step.get("name", "")
        step_conclusion = step.get("conclusion", "")

        is_checkout_failure = "checkout" in step_name.lower() and step_conclusion == "failure"

        if is_checkout_failure:
            return str(InfraErrorV1.CHECKOUT_FAILURE)

    # If failure occurred in clang-tidy step, classify as code quality failure
    for step in github_job.get("steps", []):
        step_name = step.get("name", "")
        step_conclusion = step.get("conclusion", "")

        is_clang_tidy_failure = "analyze code with clang-tidy" in step_name.lower() and step_conclusion == "failure"

        if is_clang_tidy_failure:
            return str(CodeQualityErrorV1.CLANG_TIDY_VIOLATION)

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

        # First, look for test failures (prioritize these over infrastructure failures)
        for _annot in annotation_info:
            # Unit test failure: a failure exists where the annotation path is not .github
            if _annot["annotation_level"] == "failure" and _annot["path"] != ".github":
                failure_description = _annot["path"]
                if ".py" in failure_description:
                    failure_signature = str(TestErrorV1.PY_TEST_FAILURE)
                elif ".cpp" in failure_description:
                    failure_signature = str(TestErrorV1.CPP_TEST_FAILURE)
                else:
                    failure_signature = str(TestErrorV1.UNKNOWN_TEST_FAILURE)
                return failure_signature, failure_description

        # If no test failures found, fall back to infrastructure failures
        for _annot in annotation_info:
            # Infrastructure error
            if _annot["annotation_level"] == "failure" and _annot["path"] == ".github":
                failure_description = _annot.get("message")
                if failure_description:
                    failure_signature = get_job_failure_signature_(
                        github_job, failure_description, workflow_outputs_dir
                    )
                    return failure_signature, failure_description
    return failure_signature, failure_description


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
        elif title == "tt-card-serial" and "serial number(s):" in message:
            serial = message.rsplit("serial number(s):", 1)[-1].strip() or None
    return node_name, serial


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
    if host_name and host_name.startswith("tt-ubuntu"):
        parts = host_name.split("-")
        # Issue: https://github.com/tenstorrent/tt-metal/issues/21694
        # Issue: https://github.com/tenstorrent/tt-metal/issues/26445
        # Remove non-constant ephemeral runner suffix from tt-beta/tt-ubuntu runner names only if the second last part is "runner"
        # We don't want to remove the suffix for non-ephemeral runners (e.g. tt-beta-ubuntu-2204-xlarge)
        # E.g. tt-beta-ubuntu-2204-n150-large-stable-nk6pd-runner-5g5f9 -> tt-beta-ubuntu-2204-n150-large-stable-nk6pd
        if len(parts) >= 2 and parts[-2] == "runner":
            host_name = "-".join(parts[:-1])

        # For CIv2 runners (tt-ubuntu), the ephemeral runner identity is not useful for data
        # analysis since it changes every pod. Instead, identify the physical node/card that ran
        # the job using the <node name>_<serial> emitted by the job-start hook annotations
        # Fall back to the truncated name above when the
        # annotations are unavailable (e.g. not downloaded, or CPU-only runners without a serial).
        if host_name.startswith("tt-ubuntu"):
            node_name, serial = get_civ2_node_name_and_serial_from_annotations(
                github_job_id_to_annotations.get(github_job_id)
            )
            if node_name and serial:
                host_name = f"{node_name}_{serial}"

    # Cleanup GitHub-hosted runner names because we're sending the whole thing, which is unnecessary
    # and clogs up the data with 1000s of hosts
    if host_name and location == "github":
        host_name = "GitHub Actions"

    os = ubuntu_version

    name = github_job["name"]

    if github_job["status"] != "completed":
        logger.warning(f"{github_job_id} is not completed, skipping this job")
        return None

    card_type = _card_type_from_job_labels(labels)

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
        "workflow_attempt": github_job.get("run_attempt"),
        "steps": github_job.get("steps", []),
    }


def get_job_rows_from_github_info(workflow_outputs_dir, github_jobs_json, github_job_id_to_annotations):
    job_rows = list(
        map(
            lambda job: get_job_row_from_github_job(job, github_job_id_to_annotations, workflow_outputs_dir),
            github_jobs_json["jobs"],
        )
    )
    return [x for x in job_rows if x is not None]


def _get_repo_root() -> pathlib.Path:
    """Return the repository root directory (parent of infra/)."""
    return pathlib.Path(__file__).resolve().parents[3]


@functools.lru_cache(maxsize=1)
def _load_sku_config_skus() -> dict:
    sku_config_path = _get_repo_root() / ".github" / "sku_config.yaml"
    with open(sku_config_path) as f:
        config = yaml.safe_load(f)
    return config.get("skus") or {}


@functools.lru_cache(maxsize=1)
def _sku_config_sku_names() -> tuple[str, ...]:
    return tuple(_load_sku_config_skus().keys())


def _is_sku_name_prefix(prefix: str, sku_name: str) -> bool:
    if sku_name == prefix:
        return True
    if len(prefix) >= len(sku_name) or not sku_name.startswith(prefix):
        return False
    return sku_name[len(prefix)] in "_-"


@functools.lru_cache(maxsize=1)
def _generic_runner_labels() -> frozenset[str]:
    """
    Runner labels from sim_* runs_on entries in sku_config.yaml.

    These are shared CPU pools where strict sku_config matching cannot distinguish
    sim tests from other jobs on the same label; card_type stores the label itself.
    """
    labels: set[str] = set()
    for sku_name, sku_entry in _load_sku_config_skus().items():
        if sku_name.startswith("sim_"):
            labels.update(sku_entry.get("runs_on") or [])
    return frozenset(labels)


# Longest-first suffixes stripped when promoting a variant SKU to its root name.
_CARD_TYPE_ROOT_SUFFIXES: tuple[str, ...] = (
    "_civ2_viommu_prio",
    "_civ2_viommu",
    "_civ2_prio",
    "_merge_gate",
    "_civ2",
    "_viommu",
    "_perf",
    "_prio",
    "_iommu",
    "-blitz",
    "-mgd",
)


def _uses_generic_runner_labels(label_set: set[str]) -> bool:
    return bool(label_set) and label_set <= _generic_runner_labels()


def _card_type_from_generic_runner_labels(label_set: set[str]) -> Optional[str]:
    """
    Map sim_* shared CPU pools back to their runner label for card_type.

    When job labels are only pools listed on sim_* SKUs in sku_config.yaml, return
    the CPU runner label itself rather than a sim_* SKU name.
    """
    if not _uses_generic_runner_labels(label_set):
        return None
    return sorted(label_set)[0]


@functools.lru_cache(maxsize=128)
def _root_sku_for(sku_name: str) -> str:
    known_skus = set(_sku_config_sku_names())
    candidate = sku_name

    while True:
        promoted = False
        for suffix in _CARD_TYPE_ROOT_SUFFIXES:
            if not candidate.endswith(suffix):
                continue
            stripped = candidate[: -len(suffix)]
            if stripped in known_skus:
                candidate = stripped
                promoted = True
                break
        if not promoted:
            break

    if candidate in known_skus:
        return candidate

    candidates = [name for name in known_skus if _is_sku_name_prefix(name, sku_name)]
    if candidates:
        return min(candidates, key=lambda name: (len(name), name))
    return sku_name


# Runner labels checked in order when strict sku_config matching fails.
_CARD_TYPE_LABEL_FALLBACK: tuple[tuple[str, str], ...] = (
    ("P300-viommu", "bh_p300"),
    ("P300", "bh_p300"),
    ("P150", "bh_p150"),
    ("P100", "bh_p100"),
    ("N300", "wh_n300"),
    ("N150", "wh_n150"),
)


def _card_type_from_sku_config(labels: list[str]) -> Optional[str]:
    """
    Match job labels to a pipeline SKU from sku_config.yaml.

    Every SKU whose runs_on labels are present on the job is a match (sim_* SKUs are
    skipped). Matches are promoted to their root SKU via suffix stripping / sku_config
    prefix lookup.
    """
    label_set = set(labels)
    matching_skus: list[str] = []

    for sku_name, sku_entry in _load_sku_config_skus().items():
        if sku_name.startswith("sim_"):
            continue

        runs_on = sku_entry.get("runs_on") or []
        if not runs_on:
            continue

        if frozenset(runs_on).issubset(label_set):
            matching_skus.append(sku_name)

    if not matching_skus:
        return None

    roots = {_root_sku_for(sku_name) for sku_name in matching_skus}
    return sorted(roots)[0]


def _card_type_fallback_from_job_labels(labels: list[str]) -> Optional[str]:
    """
    Best-effort card type when sku_config has no full runs_on match.

    Maps a single hardware runner label to the corresponding root SKU.
    """
    label_set = set(labels)
    for runner_label, card_type in _CARD_TYPE_LABEL_FALLBACK:
        if runner_label in label_set:
            return card_type
    return None


def _card_type_from_job_labels(labels: list[str]) -> Optional[str]:
    label_set = set(labels)

    card_type = _card_type_from_generic_runner_labels(label_set)
    if card_type is not None:
        logger.info(f"Matched job labels to generic runner label {card_type!r}")
        return card_type

    card_type = _card_type_from_sku_config(labels)
    if card_type is None:
        card_type = _card_type_fallback_from_job_labels(labels)
        if card_type is not None:
            logger.info(f"Matched job labels to SKU {card_type!r} via label fallback")
            return card_type
        return None

    logger.info(f"Matched job labels to SKU {card_type!r}")
    return card_type


def get_github_partial_benchmark_data_filenames():
    logger.info("We are assuming generated/benchmark_data exists from previous passing test")

    benchmark_data_dir = _get_repo_root() / "generated/benchmark_data"
    assert benchmark_data_dir.exists()
    assert benchmark_data_dir.is_dir()

    benchmark_data_paths = list(benchmark_data_dir.glob("partial_run_*.pkl"))
    assert len(
        benchmark_data_paths
    ), f"There needs to be at least one benchmark data pkl since we're completing the environment data for each one"

    logger.info(
        f"The following partial benchmark data PKLs should be completed with environment data: {benchmark_data_paths}"
    )
    return benchmark_data_paths


def get_github_runner_environment():
    assert "GITHUB_EVENT_NAME" in os.environ
    github_event_name = os.environ["GITHUB_EVENT_NAME"]

    return {
        "github_event_name": github_event_name,
    }


def _get_device_type_from_runner_environment(sku_from_test: Optional[str] = None) -> str:
    """
    Infer device/card type (wormhole_b0, blackhole) from runner environment.
    RUNNER_NAME is a GitHub Actions env var that must be set.

    When sku_from_test is provided (e.g. from workflow), look up sku_config for that SKU's
    runs_on labels; if any label contains "blackhole" or "wormhole", return the arch.
    """
    assert "RUNNER_NAME" in os.environ, "RUNNER_NAME must be set (GitHub Actions env var)"
    runner_name = os.environ["RUNNER_NAME"]
    runner_lower = runner_name.lower()

    # This assumes all CIv2 runner names start with tt-ubuntu
    if runner_lower.startswith("tt-ubuntu"):
        if "blackhole" in runner_lower or "bh-" in runner_lower or "p100" in runner_lower or "p150" in runner_lower:
            return "blackhole"
        if "n150" in runner_lower or "n300" in runner_lower or "wormhole" in runner_lower:
            return "wormhole_b0"
        return "unknown"

    # Not tt-ubuntu: check .github/sku_config.yaml for arch from runs_on labels matching runner
    if sku_from_test:
        sku_config_path = _get_repo_root() / ".github" / "sku_config.yaml"
        if sku_config_path.exists():
            with open(sku_config_path) as f:
                config = yaml.safe_load(f)
            skus = config.get("skus") or {}

            if sku_from_test in skus:
                # Use sku_from_test from workflow: get runs_on labels for this SKU
                runs_on = skus.get(sku_from_test, {}).get("runs_on") or []
                for label in runs_on:
                    label_lower = label.lower()
                    # Only checks CIv1 style labels in sku_config for now
                    if "blackhole" in label_lower:
                        return "blackhole"
                    if "wormhole" in label_lower:
                        return "wormhole_b0"

    # Failed to parse from CIv2 runner name and failed to parse from sku_config:
    # Fallback to using ARCH_NAME env var
    if "ARCH_NAME" in os.environ:
        arch = os.environ["ARCH_NAME"]
        if arch in ("wormhole_b0", "blackhole"):
            return arch

    logger.warning(
        f"Could not infer device type from RUNNER_NAME={runner_name!r}. "
        "Set ARCH_NAME env var (wormhole_b0, blackhole) for accurate benchmark data."
    )
    return "unknown"


def create_json_with_github_benchmark_environment(
    github_partial_benchmark_data_filename, sku_from_test: Optional[str] = None
):
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

    device_type = _get_device_type_from_runner_environment(sku_from_test=sku_from_test)

    logger.warning("Hardcoded null for device_memory_size")
    device_memory_size = ""

    with open(github_partial_benchmark_data_filename, "rb") as f:
        partial_benchmark_data = pickle.load(f)

    existing_device_info = partial_benchmark_data.device_info or {}
    device_info = existing_device_info | {"card_type": device_type, "dram_size": device_memory_size}
    if sku_from_test:
        device_info["sku"] = sku_from_test

    partial_benchmark_data = partial_benchmark_data.model_copy(
        update={
            "git_repo_name": git_repo_name,
            "git_commit_hash": git_commit_hash,
            "git_commit_ts": git_commit_ts,
            "git_branch_name": git_branch_name,
            "github_pipeline_id": github_pipeline_id,
            "github_pipeline_link": github_pipeline_link,
            "github_job_id": github_job_id,
            "user_name": user_name,
            "docker_image": docker_image,
            "device_hostname": device_hostname,
            "device_ip": device_ip,
            "device_info": device_info,
        }
    )

    complete_benchmark_run = CompleteBenchmarkRun(**partial_benchmark_data.model_dump())

    json_data = complete_benchmark_run.model_dump_json()

    # Save complete run json
    output_path = pathlib.Path(
        str(github_partial_benchmark_data_filename).replace("partial_run_", "complete_run_")
    ).with_suffix(".json")
    with open(output_path, "w") as f:
        f.write(json_data)

    # Delete partial run pkl
    os.remove(github_partial_benchmark_data_filename)
