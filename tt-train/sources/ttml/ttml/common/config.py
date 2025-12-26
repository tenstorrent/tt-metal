# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for transformer training."""
import os
import yaml
from typing import Union
from ttml.common.utils import get_tt_metal_home


class DeviceConfig:
    """Configuration for device mesh and distributed training."""

    def __init__(self, yaml_config: Union[dict, str]):
        """Initialize device configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        device_config = {}
        if isinstance(yaml_config, str):
            device_config = load_config(yaml_config)
        elif isinstance(yaml_config, dict):
            device_config = yaml_config.get("device_config", {})

        self.mesh_shape = device_config.get("mesh_shape", [1, 1])
        self.device_ids = device_config.get("device_ids", None)
        self.enable_tp = device_config.get("enable_tp", False)
        self.enable_ddp = device_config.get("enable_ddp", False)

        # Based on current configs, DDP and TP cannot be both enabled
        assert not (
            self.enable_ddp and self.enable_tp
        ), "DDP and TP cannot be both enabled."

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

    def __init__(self, yaml_config: Union[dict, str]):
        """Initialize training configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """

        tc = {}
        if isinstance(yaml_config, str):
            tc = load_config(yaml_config)
        elif isinstance(yaml_config, dict):
            tc = yaml_config.get("training_config", {})

        self.seed = int(tc.get("seed", 42))
        self.batch_size = int(tc.get("batch_size", 64))
        self.validation_batch_size = int(
            tc.get("validation_batch_size", max(self.batch_size // 2, 1))
        )
        self.steps = int(tc.get("max_steps", 1000))
        self.epochs = int(tc.get("num_epochs", 1))
        self.eval_every = int(tc.get("eval_every", 200))
        self.save_every = int(tc.get("model_save_interval", 500))
        self.gradient_accumulation_steps = int(tc.get("gradient_accumulation_steps", 1))
        self.model_config = tc.get("model_config", None)
        tokenizer_type = tc.get("tokenizer_type", "bpe")

        if tokenizer_type == "bpe":
            self.use_bpe = True
        elif tokenizer_type == "char":
            self.use_bpe = False
        else:
            raise ValueError(f"Unknown tokenizer_type: {tokenizer_type}")

        self.lr = float(tc.get("learning_rate", 3e-4))
        self.beta1 = float(tc.get("beta1", 0.9))
        self.beta2 = float(tc.get("beta2", 0.999))
        self.eps = float(tc.get("eps", 1e-8))
        self.weight_decay = float(tc.get("weight_decay", 0.01))
        self.checkpoint_dir = tc.get("checkpoint_dir", "checkpoints")


class TransformerConfig:
    """Configuration for transformer model hyperparameters."""

    def __init__(self, yaml_config: Union[dict, str]):
        """Initialize transformer configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """

        tc = {}
        if isinstance(yaml_config, str):
            tc = load_config(yaml_config)
        elif isinstance(yaml_config, dict):
            tc = yaml_config.get("transformer_config", {})

        # Base parameters
        self.model_type = tc.get("model_type", "gpt2")
        self.runner_type = tc.get("runner_type", "default")
        self.num_heads = int(tc.get("num_heads", 6))
        self.embedding_dim = int(tc.get("embedding_dim", 384))
        self.dropout_prob = float(tc.get("dropout_prob", 0.2))
        self.num_blocks = int(tc.get("num_blocks", 6))
        self.vocab_size = int(tc.get("vocab_size", 96))
        self.weight_tying = tc.get("weight_tying", None)
        self.max_sequence_length = int(tc.get("max_sequence_length", 128))

        # Llama-specific
        self.intermediate_dim = tc.get("intermediate_dim", None)
        self.theta = tc.get("theta", None)
        self.num_groups = tc.get("num_groups", 3)

        # RoPE
        self.rope = tc.get("rope_scaling", None)
        if self.rope:
            self.scaling_factor = self.rope.get("scaling_factor", None)
            self.high_freq_factor = self.rope.get("high_freq_factor", None)
            self.low_freq_factor = self.rope.get("low_freq_factor", None)
            self.original_context_length = self.rope.get(
                "original_context_length", None
            )


class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    def __init__(self, yaml_config: dict):
        """Initialize scheduler configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        sc = yaml_config.get("scheduler_config", {})
        tc = yaml_config.get("training_config", {})
        self.max_lr = float(sc.get("max_lr", 0.001))
        self.min_lr = float(sc.get("min_lr", 0.0))
        self.warmup_steps = int(sc.get("warmup_steps", 100))
        self.hold_steps = int(sc.get("hold_steps", 0))
        self.total_steps = int(tc.get("total_steps", 1000))

        # optional momentum warmup (beta1 ramp)
        self.beta1_start = float(sc.get("beta1_start", 0.85))
        self.beta1_end = float(sc.get("beta1_end", 0.95))
        self.beta1_warmup_steps = int(sc.get("beta1_warmup_steps", 0))


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

    def __init__(self, yaml_config: Union[dict, str]):
        mh = {}
        if isinstance(yaml_config, str):
            mh = load_config(yaml_config).get("multihost_config", {})
        elif isinstance(yaml_config, dict):
            mh = yaml_config.get("multihost_config", {})

        self.enabled = bool(mh.get("enabled", False))
        self.num_workers = int(mh.get("num_workers", 1))
        # Keep as lowercase string to avoid importing native enums here
        self.socket_type = str(mh.get("socket_type", "mpi")).strip().lower()

        pp_cfg = mh.get("pipeline_parallel_config")
        self.pipeline_parallel_config = (
            PipelineParallelHostConfig(pp_cfg) if isinstance(pp_cfg, dict) else None
        )


def yaml_deep_update(original: dict, updates: dict) -> dict:
    """Recursively update a dictionary with another dictionary.

    Args:
        original: Original dictionary to be updated
        updates: Dictionary containing updates

    Returns:
        Updated dictionary
    """
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            original[key] = yaml_deep_update(original[key], value)
        else:
            original[key] = value
    return original


def load_config(path: str, configs_root: str = None) -> dict:
    """Load configuration from YAML file.

    Args:
        path: Path to config file relative to configs directory

    Returns:
        Dictionary containing configuration
    """

    if configs_root is None:
        configs_root = f"{get_tt_metal_home()}/tt-train/configs/"

    # if the path is relative, make it absolute
    if not (os.path.isabs(path)):
        path = os.path.join(configs_root, path)

    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_training_config(
    training_config_src: str,
    configs_root: str = f"{get_tt_metal_home()}/tt-train/configs/training_configs",
) -> TrainingConfig:
    """Load training configuration given its filename."""

    training_config = load_config(training_config_src, configs_root)
    training_config = TrainingConfig(training_config)

    return training_config


def get_device_config(
    device_config_src: str,
    configs_root: str = f"{get_tt_metal_home()}/tt-train/configs/training_configs/",
) -> DeviceConfig:
    """Load device configuration given its filename."""

    device_config = load_config(device_config_src, configs_root)
    device_config = DeviceConfig(device_config)

    return device_config


def get_model_config(
    model_config_src: str,
    configs_root: str = f"{get_tt_metal_home()}/tt-train/",
) -> TransformerConfig:
    """Load model configuration given its filename."""

    model_config = load_config(model_config_src, configs_root)
    model_config = TransformerConfig(model_config)

    return model_config


def get_multihost_config(
    multihost_config_src: str,
    configs_root: str = f"{get_tt_metal_home()}/tt-train/configs/multihost_configs/",
) -> MultiHostConfig:
    """Load multihost configuration given its filename."""

    multihost_config = load_config(multihost_config_src, configs_root)
    multihost_config = MultiHostConfig(multihost_config)

    return multihost_config
