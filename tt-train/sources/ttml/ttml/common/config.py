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
        assert self.mesh_shape[0] == 1, f"Only [1, N] mesh shapes are supported, got {self.mesh_shape}"

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
        self.epochs = int(tc.get("num_epochs", 1))
        self.eval_every = int(tc.get("eval_every", 200))
        self.save_every = int(tc.get("model_save_interval", 500))
        self.gradient_accumulation_steps = int(tc.get("gradient_accumulation_steps", 1))
        self.checkpoint_dir = tc.get("checkpoint_dir", "checkpoints")

        self.transformer_config = TransformerConfig(tc.get("transformer_config", {}))
        self.seq_len = int(self.transformer_config.max_sequence_length)

    def update_config(self, yaml_config: dict):
        """Update training configuration from another YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        tc = yaml_config.get("training_config", {})
        self.batch_size = int(tc.get("batch_size", self.batch_size))
        self.steps = int(tc.get("max_steps", self.steps))
        self.epochs = int(tc.get("num_epochs", self.epochs))
        self.eval_every = int(tc.get("eval_every", self.eval_every))
        self.save_every = int(tc.get("model_save_interval", self.save_every))
        self.gradient_accumulation_steps = int(tc.get("gradient_accumulation_steps", self.gradient_accumulation_steps))
        self.checkpoint_dir = tc.get("checkpoint_dir", self.checkpoint_dir)

        if "transformer_config" in tc:
            self.transformer_config = self.transformer_config.update_config(tc.get("transformer_config", {}))
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
            self.original_context_length = self.rope.get("original_context_length", None)

    def update_config(self, yaml_config: dict):
        """Update transformer configuration from another YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        if "transformer_config" not in yaml_config:
            return

        tc = yaml_config.get("transformer_config", {})
        self.runner_type = tc.get("runner_type", self.runner_type)
        self.num_heads = int(tc.get("num_heads", self.num_heads))
        self.embedding_dim = int(tc.get("embedding_dim", self.embedding_dim))
        self.dropout_prob = float(tc.get("dropout_prob", self.dropout_prob))
        self.num_blocks = int(tc.get("num_blocks", self.num_blocks))
        self.vocab_size = int(tc.get("vocab_size", self.vocab_size))
        self.weight_tying = tc.get("weight_tying", self.weight_tying)
        self.max_sequence_length = int(tc.get("max_sequence_length", self.max_sequence_length))

        self.intermediate_dim = tc.get("intermediate_dim", self.intermediate_dim)
        self.theta = tc.get("theta", self.theta)
        self.num_groups = tc.get("num_groups", self.num_groups)

        if "rope_scaling" in tc:
            self.rope = tc.get("rope_scaling", self.rope)
            if self.rope:
                self.scaling_factor = self.rope.get("scaling_factor", self.scaling_factor)
                self.high_freq_factor = self.rope.get("high_freq_factor", self.high_freq_factor)
                self.low_freq_factor = self.rope.get("low_freq_factor", self.low_freq_factor)
                self.original_context_length = self.rope.get("original_context_length", self.original_context_length)


class SchedulerConfig:
    """Configuration for learning rate scheduler."""

    def __init__(self, yaml_config: dict):
        """Initialize scheduler configuration from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        sc = yaml_config.get("scheduler_config", {})
        self.max_lr = float(sc.get("max_lr", 0.001))
        self.min_lr = float(sc.get("min_lr", 0.0))
        self.warmup_steps = int(sc.get("warmup_steps", 100))
        self.hold_steps = int(sc.get("hold_steps", 0))
