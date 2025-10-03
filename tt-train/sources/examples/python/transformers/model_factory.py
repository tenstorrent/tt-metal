# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Factory for creating transformer models from configuration."""
import ttml
from config import DeviceConfig
from utils import round_up_to_tile


def adjust_vocab_size(vocab_size: int, enable_tp: bool, total_devices: int) -> int:
    """Adjust vocabulary size for tiling and tensor parallelism.

    Args:
        vocab_size: Original vocabulary size
        enable_tp: Whether tensor parallelism is enabled
        total_devices: Total number of devices

    Returns:
        Adjusted vocabulary size
    """
    if enable_tp:
        return round_up_to_tile(vocab_size, total_devices * 32)
    return round_up_to_tile(vocab_size, 32)


def map_runner_type(rt: str):
    """Map runner type string to enum.

    Args:
        rt: Runner type as string

    Returns:
        RunnerType enum value
    """
    name = str(rt).strip().lower()
    if name == "memory_efficient":
        return ttml.models.RunnerType.MemoryEfficient
    return ttml.models.RunnerType.Default


class TransformerModelFactory:
    """Factory class for creating transformer models from configuration."""

    def __init__(self, yaml_config: dict):
        """Initialize model factory from YAML config.

        Args:
            yaml_config: Dictionary containing configuration
        """
        self.device_config = DeviceConfig(yaml_config)
        training_config = yaml_config.get("training_config", {})
        self.model_type = training_config.get("model_type", "gpt2")
        self.transformer_config = training_config.get("transformer_config", {})

    def _create_gpt2(self):
        """Create GPT-2 model from configuration.

        Returns:
            GPT-2 model instance
        """
        gcfg = ttml.models.gpt2.GPT2TransformerConfig()
        gcfg.num_heads = self.transformer_config.get("num_heads", 6)
        gcfg.embedding_dim = self.transformer_config.get("embedding_dim", 384)
        gcfg.num_blocks = self.transformer_config.get("num_blocks", 6)
        vs = self.transformer_config.get("vocab_size", 256)
        gcfg.vocab_size = adjust_vocab_size(vs, self.device_config.enable_tp, self.device_config.total_devices())
        gcfg.max_sequence_length = self.transformer_config.get("max_sequence_length", 256)
        gcfg.dropout_prob = self.transformer_config.get("dropout_prob", 0.2)
        # Optional runner type: accept enum or string names; fallback to defaults if absent
        if "runner_type" in self.transformer_config:
            gcfg.runner_type = map_runner_type(self.transformer_config["runner_type"])  # type: ignore[arg-type]
        if self.device_config.enable_tp:
            return ttml.models.distributed.gpt2.create_gpt2_model(gcfg)
        return ttml.models.gpt2.create_gpt2_model(gcfg)

    def _create_llama(self):
        """Create Llama model from configuration.

        Returns:
            Llama model instance
        """
        lcfg = ttml.models.llama.LlamaConfig()
        tc = self.transformer_config

        # Core fields with sensible defaults
        lcfg.num_heads = tc.get("num_heads", 6)
        lcfg.num_groups = tc.get("num_groups", 3)
        lcfg.embedding_dim = tc.get("embedding_dim", 384)
        lcfg.num_blocks = tc.get("num_blocks", 6)
        vs = tc.get("vocab_size", 256)
        lcfg.vocab_size = adjust_vocab_size(vs, self.device_config.enable_tp, self.device_config.total_devices())
        lcfg.max_sequence_length = tc.get("max_sequence_length", 256)
        lcfg.dropout_prob = tc.get("dropout_prob", 0.0)

        # Optional fields
        if "intermediate_dim" in tc:
            lcfg.intermediate_dim = tc["intermediate_dim"]
        if "theta" in tc:
            lcfg.theta = tc["theta"]

        # Runner type (simple mapping like GPT2)
        rt = tc.get("runner_type", "default")
        lcfg.runner_type = map_runner_type(rt)

        if "weight_tying" in tc:
            lcfg.weight_tying = tc["weight_tying"]

        # Optional RoPE scaling from nested block
        rope = tc.get("rope_scaling")
        if rope:
            if "scaling_factor" in rope:
                lcfg.scaling_factor = rope["scaling_factor"]
            if "high_freq_factor" in rope:
                lcfg.high_freq_factor = rope["high_freq_factor"]
            if "low_freq_factor" in rope:
                lcfg.low_freq_factor = rope["low_freq_factor"]
            if "original_context_length" in rope:
                lcfg.original_context_length = rope["original_context_length"]

        if self.device_config.enable_tp:
            return ttml.models.distributed.llama.create_llama_model(lcfg)
        return ttml.models.llama.create_llama_model(lcfg)

    def create_model(self):
        """Create model based on model_type configuration.

        Returns:
            Model instance

        Raises:
            ValueError: If model type is not supported
        """
        if self.model_type == "gpt2":
            return self._create_gpt2()
        elif self.model_type == "llama":
            return self._create_llama()
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
