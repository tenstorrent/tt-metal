# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Definition of the pydantic models used for data production.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Test(BaseModel):
    """
    Table containing information about the execution of CI/CD tests, each one associated
    with a specific CI/CD job execution.

    Only some CI/CD jobs execute tests, which are executed sequentially.
    """

    test_start_ts: datetime = Field(description="Timestamp with timezone when the test execution started.")
    test_end_ts: datetime = Field(description="Timestamp with timezone when the test execution ended.")
    test_case_name: str = Field(description="Name of the pytest function.")
    filepath: str = Field(description="Test file path and name.")
    category: str = Field(description="Name of the test category.")
    group: Optional[str] = Field(None, description="Name of the test group.")
    owner: Optional[str] = Field(None, description="Developer of the test.")
    error_message: Optional[str] = Field(None, description="Succinct error string, such as exception type.")
    success: bool = Field(description="Test execution success.")
    skipped: bool = Field(description="Some tests in a job can be skipped.")
    full_test_name: str = Field(description="Test name plus config.")
    config: Optional[dict] = Field(None, description="Test configuration key/value " "pairs.")
    tags: Optional[dict] = Field(None, description="Tags associated with the test, as key/value pairs.")


class Job(BaseModel):
    """
    Contains information about the execution of CI/CD jobs, each one associated with a
    specific CI/CD pipeline.

    Each job may execute multiple tests, which are executed sequentially on a unique
    host.
    """

    github_job_id: Optional[int] = Field(
        None,
        description="Identifier for the Github Actions CI job, for pipelines " "orchestrated and executed by Github.",
    )
    name: str = Field(description="Name of the job.")
    job_submission_ts: datetime = Field(description="Timestamp with timezone when the job was submitted.")
    job_start_ts: datetime = Field(description="Timestamp with timezone when the job execution started.")
    job_end_ts: datetime = Field(description="Timestamp with timezone when the job execution ended.")
    job_success: bool = Field(
        description="Job execution success, independently from the test success "
        "criteria. Failure mechanisms that are only descriptive of the "
        "job itself."
    )
    docker_image: Optional[str] = Field(None, description="Name of the Docker image used for the CI job.")
    is_build_job: bool = Field(description="Flag identifying if the job is a software build.")
    job_matrix_config: Optional[dict] = Field(
        None, description="This attribute is included for future feature enhancement."
    )
    host_name: Optional[str] = Field(description="Unique host name.")
    card_type: Optional[str] = Field(description="Card type and version.")
    os: Optional[str] = Field(description="Operating system of the host.")
    location: Optional[str] = Field(description="Where the host is located.")
    tests: List[Test] = []


class Pipeline(BaseModel):
    """
    Contains information about the execution of CI/CD pipelines, which consist of the
    sequential execution of one or more jobs.

    Each pipeline is associated with a specific code repository and a specific commit.;
    """

    github_pipeline_id: Optional[int] = Field(
        None,
        description="Identifier for the Github Actions CI pipeline, for pipelines "
        "orchestrated and executed by Github.",
    )
    pipeline_submission_ts: datetime = Field(
        description="Timestamp with timezone when the pipeline was submitted for " "execution.",
    )
    pipeline_start_ts: datetime = Field(description="Timestamp with timezone when the pipeline execution started.")
    pipeline_end_ts: datetime = Field(description="Timestamp with timezone when the pipeline execution ended.")
    name: str = Field(description="Name of the pipeline.")
    project: Optional[str] = Field(None, description="Name of the software project.")
    trigger: Optional[str] = Field(None, description="Type of trigger that initiated the pipeline.")
    vcs_platform: Optional[str] = Field(
        None,
        description="Version control software used for the code tested in the pipeline.",
    )
    repository_url: str = Field(description="URL of the code repository.")
    git_branch_name: Optional[str] = Field(description="Name of the Git branch tested by the pipeline.")
    git_commit_hash: str = Field(description="Git commit that triggered the execution of the pipeline.")
    git_author: str = Field(description="Author of the Git commit.")
    orchestrator: Optional[str] = Field(None, description="CI/CD pipeline orchestration platform.")
    jobs: List[Job] = []
