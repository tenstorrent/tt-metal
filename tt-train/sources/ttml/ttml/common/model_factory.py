# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Factory for creating transformer models from configuration."""
import ttml
from ttml.common.config import DeviceConfig, TransformerConfig, MultiHostConfig
from ttml.common.utils import round_up_to_tile


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
        self.multihost_config = MultiHostConfig(yaml_config)
        self.transformer_config = TransformerConfig(yaml_config)
        self.model_type = self.transformer_config.model_type

    def _create_gpt2(self):
        """Create GPT-2 model from configuration.

        Returns:
            GPT-2 model instance
        """
        gcfg = ttml.models.gpt2.GPT2TransformerConfig()
        gcfg.num_heads = self.transformer_config.num_heads
        gcfg.embedding_dim = self.transformer_config.embedding_dim
        gcfg.num_blocks = self.transformer_config.num_blocks
        vs = self.transformer_config.vocab_size
        gcfg.vocab_size = adjust_vocab_size(
            vs, self.device_config.enable_tp, self.device_config.total_devices()
        )
        gcfg.max_sequence_length = self.transformer_config.max_sequence_length
        gcfg.dropout_prob = self.transformer_config.dropout_prob
        gcfg.runner_type = map_runner_type(self.transformer_config.runner_type)  # type: ignore[arg-type]

        if self.transformer_config.weight_tying:
            gcfg.weight_tying = (
                ttml.models.WeightTyingType.Enabled
                if "enabled" in self.transformer_config.weight_tying
                else ttml.models.WeightTyingType.Disabled
            )

        if self.device_config.enable_tp:
            return ttml.models.distributed.gpt2.create_gpt2_model(gcfg)
        return ttml.models.gpt2.create_gpt2_model(gcfg)

    def _create_llama(self):
        """Create Llama model from configuration.

        Returns:
            Llama model instance
        """
        lcfg = ttml.models.llama.LlamaConfig()

        # Core fields with sensible defaults
        lcfg.num_heads = self.transformer_config.num_heads
        lcfg.num_groups = self.transformer_config.num_groups
        lcfg.embedding_dim = self.transformer_config.embedding_dim
        lcfg.num_blocks = self.transformer_config.num_blocks
        vs = self.transformer_config.vocab_size
        lcfg.vocab_size = adjust_vocab_size(
            vs, self.device_config.enable_tp, self.device_config.total_devices()
        )
        lcfg.max_sequence_length = self.transformer_config.max_sequence_length
        lcfg.dropout_prob = self.transformer_config.dropout_prob

        # Optional fields
        if self.transformer_config.intermediate_dim:
            lcfg.intermediate_dim = self.transformer_config.intermediate_dim
        if self.transformer_config.theta:
            lcfg.theta = self.transformer_config.theta

        # Runner type (simple mapping like GPT2)
        lcfg.runner_type = map_runner_type(self.transformer_config.runner_type)

        if self.transformer_config.weight_tying:
            lcfg.weight_tying = (
                ttml.models.WeightTyingType.Enabled
                if "enabled" in self.transformer_config.weight_tying
                else ttml.models.WeightTyingType.Disabled
            )

        # Optional RoPE scaling from nested block
        rope = self.transformer_config.rope
        if rope:
            if self.transformer_config.scaling_factor:
                lcfg.scaling_factor = self.transformer_config.scaling_factor
            if self.transformer_config.high_freq_factor:
                lcfg.high_freq_factor = self.transformer_config.high_freq_factor
            if self.transformer_config.low_freq_factor:
                lcfg.low_freq_factor = self.transformer_config.low_freq_factor
            if self.transformer_config.original_context_length:
                lcfg.original_context_length = (
                    self.transformer_config.original_context_length
                )

        # Pipeline-parallel config (optional, under multihost_config)
        mh = self.multihost_config
        if mh.pipeline_parallel_config:
            # Build PP config object
            pp = (
                ttml.models.distributed.pipeline_parallel.llama.PipelineParallelConfig()
            )
            pp.num_blocks = mh.pipeline_parallel_config.num_blocks
            pp.blocks_per_rank = mh.pipeline_parallel_config.blocks_per_rank
            return ttml.models.distributed.pipeline_parallel.llama.create_llama_model(
                lcfg, pp, self.device_config.enable_tp
            )

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
