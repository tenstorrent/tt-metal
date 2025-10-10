# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for transformer training."""
import os
import yaml


def get_config(path: str):
    """Load configuration from YAML file.

    Args:
        path: Path to config file relative to configs directory

    Returns:
        Dictionary containing configuration
    """
    path = f'{os.environ["TT_METAL_HOME"]}/tt-train/configs/{path}'
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


class DeviceConfig:
    """Configuration for device mesh and distributed training."""

    def __init__(self, yaml_config: dict):
        """Initialize device configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        device_config = yaml_config.get("device_config", {})
        self.mesh_shape = device_config.get("mesh_shape", [1, 1])
        self.device_ids = device_config.get("device_ids", [])
        self.enable_tp = device_config.get("enable_tp", False)
        self.enable_ddp = device_config.get("enable_ddp", False)

        # we currently support only [1, N] mesh shapes
        assert (
            self.mesh_shape[0] == 1
        ), f"Only [1, N] mesh shapes are supported, got {self.mesh_shape}"

    def total_devices(self) -> int:
        """Get total number of devices in mesh.

        Returns:
            Total device count
        """
        return self.mesh_shape[0] * self.mesh_shape[1]


class TrainingConfig:
    """Configuration for training hyperparameters."""

    def __init__(self, yaml_config: dict):
        """Initialize training configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        tc = yaml_config.get("training_config", {})
        self.batch_size = int(tc.get("batch_size", 64))
        self.steps = int(tc.get("max_steps", 1000))
        self.eval_every = int(tc.get("eval_every", 200))
        self.gradient_accumulation_steps = int(tc.get("gradient_accumulation_steps", 1))

        self.transformer_config = TransformerConfig(tc.get("transformer_config", {}))
        self.seq_len = int(self.transformer_config.max_sequence_length)


class TransformerConfig:
    """Configuration for transformer model hyperparameters."""

    def __init__(self, yaml_config: dict):
        """Initialize transformer configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """

        # Base parameters
        self.runner_type = yaml_config.get("runner_type", "default")
        self.num_heads = int(yaml_config.get("num_heads", 6))
        self.embedding_dim = int(yaml_config.get("embedding_dim", 384))
        self.dropout_prob = float(yaml_config.get("dropout_prob", 0.2))
        self.num_blocks = int(yaml_config.get("num_blocks", 6))
        self.vocab_size = int(yaml_config.get("vocab_size", 96))
        self.weight_tying = yaml_config.get("weight_tying", None)
        self.max_sequence_length = int(yaml_config.get("max_sequence_length", 128))

        # Llama-specific
        self.intermediate_dim = yaml_config.get("intermediate_dim", None)
        self.theta = yaml_config.get("theta", None)
        self.num_groups = yaml_config.get("num_groups", 3)

        # RoPE
        self.rope = yaml_config.get("rope_scaling", None)
        if self.rope:
            self.scaling_factor = self.rope.get("scaling_factor", None)
            self.high_freq_factor = self.rope.get("high_freq_factor", None)
            self.low_freq_factor = self.rope.get("low_freq_factor", None)
            self.original_context_length = self.rope.get(
                "original_context_length", None
            )


class PipelineParallelHostConfig:
    """Host-side representation of pipeline-parallel configuration.

    Parsed from YAML under multihost_config.pipeline_parallel_config.
    """

    def __init__(self, cfg: dict):
        self.num_blocks = int(cfg.get("num_blocks", 0))
        self.blocks_per_rank = {
            int(k): int(v) for k, v in dict(cfg.get("blocks_per_rank", {})).items()
        }


class MultiHostConfig:
    """Configuration for multihost (multi-process) execution.

    Captures transport and optional pipeline-parallel settings.
    """

    def __init__(self, yaml_config: dict):
        mh = yaml_config.get("multihost_config", {})
        self.enabled = bool(mh.get("enabled", False))
        self.num_workers = int(mh.get("num_workers", 1))
        # Keep as lowercase string to avoid importing native enums here
        self.socket_type = str(mh.get("socket_type", "mpi")).strip().lower()

        pp_cfg = mh.get("pipeline_parallel_config")
        self.pipeline_parallel_config = (
            PipelineParallelHostConfig(pp_cfg) if isinstance(pp_cfg, dict) else None
        )
