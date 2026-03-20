# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
TT-Train metrics schema and serialization utilities.

"""
from pydantic import BaseModel, Field
import json


class TtTrainMetricsData(BaseModel):
    """
    A Pydantic model for validating TT-Train metrics from JSON files.

    Based on the same schema from: data_airflow/dags/pipelines/tt_train_metrics/pydantic_models.py
    """

    test_ts: float = Field(
        ...,
        description="Unix timestamp in microseconds when the training run was executed.",
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Display name of the model, e.g. Linear Regression TP+DP.",
    )
    model_filename: str = Field(
        ...,
        min_length=1,
        description="Base filename of the model binary, e.g. linear_regression_tp_dp.",
    )
    binary_name: str = Field(
        ...,
        min_length=1,
        description="Full path to the binary and environment variables.",
    )
    args: str = Field(
        ...,
        description="Command-line arguments passed to the training binary.",
    )
    git_commit_hash: str = Field(
        ...,
        min_length=1,
        description="Git commit hash for reproducibility of the run.",
    )
    model_dram_mb: float = Field(
        ...,
        description="Net DRAM allocated by the model weights (MB).",
    )
    optimizer_dram_mb: float = Field(
        ...,
        description="Net DRAM allocated by the optimizer state (MB).",
    )
    activations_dram_mb: float = Field(
        ...,
        description="Net DRAM allocated by activations during the forward pass (MB).",
    )
    gradients_dram_mb: float = Field(
        ...,
        description="Peak DRAM allocated by gradients during the backward pass (MB).",
    )
    unaccounted_dram_mb: float = Field(
        ...,
        description="Unaccounted DRAM (MB).",
    )
    total_dram_mb: float = Field(
        ...,
        description="Total DRAM allocated for the trace (MB).",
    )
    device_memory_mb: float = Field(
        ...,
        description="Device memory capacity (MB).",
    )
    last_loss: float = Field(
        ...,
        description="The loss value in the last iteration of the run.",
    )
    average_iteration_time_ms: float = Field(
        ...,
        description="The average iteration time after skipping the first 2 iterations (ms).",
    )

    model_config = {"from_attributes": True, "protected_namespaces": ()}


def write_json(pydantic_model, output_filename):
    """
    Serialize a Pydantic model to a JSON file.

    Args:
        pydantic_model: A Pydantic model instance (e.g. TtTrainMetricsData) to serialize.
        output_filename: Path to the output JSON file. Overwrites if it exists.

    Raises:
        OSError: If the file cannot be opened or written.
    """
    pydantic_json = pydantic_model.model_dump(mode="json")
    with open(output_filename, "w") as jsonfile:
        json.dump(pydantic_json, jsonfile)
