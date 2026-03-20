from dataclasses import dataclass
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.utils import initialize_device, set_seed, get_tt_metal_runtime_root
from ttml.common.config import TransformerConfig, load_config
from ttml.optimizers import create_optimizer
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from .llama_overrides import LlamaCompositeKV
import logging
import os


@dataclass
class InferenceCtx:
    tt_model: object
    tokenizer: object
    transformer_config: object
    pad_token: int
    max_tokens_to_complete: int
    temperature: float
    tile_size: int = 32
    group_size: int = 1
    sample_seed: int = 42
    _kv_cache: object = None
    _B: int = None
    _N: int = None


@dataclass
class GrpoConfig:
    clip_eps: float
    base_lr: float
    warmup_steps: int
    micro_batch_size: int
    num_mini_epochs: int  # from training_config.num_epochs
    prompts_to_train: int
    completions_batch_size: int  # from training_config.batch_size


@dataclass
class TrainingCtx:
    inference: InferenceCtx
    grpo_cfg: GrpoConfig
    optimizer: object
    output_dir: str
    checkpoint_dir: str
    logger: logging.Logger

    def save_checkpoint(self, step: int) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        tensors = {}
        for name, param in self.inference.tt_model.parameters().items():
            tensors[name] = param.to_numpy(ttnn.DataType.FLOAT32)

        filepath = os.path.join(self.checkpoint_dir, f"grpo_step_{step}.safetensors")
        save_file(tensors, filepath)
        self.logger.info(f"Saved checkpoint ({len(tensors)} tensors) to {filepath}")
        return filepath


def setup_inference(yaml_config_path, hf_model_id, load_pretrained, setup_optimizer=False) -> InferenceCtx:
    yaml_config = load_config(yaml_config_path, get_tt_metal_runtime_root())

    # training_config -> model_config path
    model_config = load_config(yaml_config["training_config"]["model_config"])

    temperature = float(yaml_config["eval_config"]["temperature"])

    # GRPO runtime knobs from training yaml
    grpo_cfg = yaml_config.get("grpo_config", {})

    max_tokens_to_complete = int(grpo_cfg["max_tokens_to_complete"])
    group_size = int(grpo_cfg["group_size"])

    initialize_device(yaml_config)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id

    transformer_config = TransformerConfig(model_config)
    transformer_config.vocab_size = len(tokenizer)

    rope_scaling = LlamaRopeScalingConfig(
        scaling_factor=getattr(transformer_config, "scaling_factor", 0.0) or 0.0,
        high_freq_factor=getattr(transformer_config, "high_freq_factor", 4.0) or 4.0,
        low_freq_factor=getattr(transformer_config, "low_freq_factor", 1.0) or 1.0,
        original_context_length=getattr(transformer_config, "original_context_length", 0) or 0,
    )

    runner_type = RunnerType.from_string(str(transformer_config.runner_type))
    weight_tying = WeightTyingType.Disabled
    if transformer_config.weight_tying:
        weight_tying = WeightTyingType.from_string(str(transformer_config.weight_tying))

    llama_cfg = LlamaConfig(
        hidden_size=transformer_config.embedding_dim,
        intermediate_size=transformer_config.intermediate_dim,
        num_hidden_layers=transformer_config.num_blocks,
        num_attention_heads=transformer_config.num_heads,
        num_key_value_heads=transformer_config.num_groups,
        vocab_size=len(tokenizer),
        max_position_embeddings=transformer_config.max_sequence_length,
        rope_theta=transformer_config.theta or 10000.0,
        attention_dropout=transformer_config.dropout_prob,
        mlp_dropout=transformer_config.dropout_prob,
        runner_type=runner_type,
        weight_tying=weight_tying,
        rope_scaling=rope_scaling,
    )

    tt_model = LlamaCompositeKV(llama_cfg)
    if load_pretrained:
        model_repo_path = snapshot_download(
            repo_id=hf_model_id,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
        load_from_safetensors(tt_model, model_repo_path, llama_cfg)

    ctx = InferenceCtx(
        tt_model=tt_model,
        tokenizer=tokenizer,
        pad_token=pad_token,
        temperature=temperature,
        max_tokens_to_complete=max_tokens_to_complete,
        transformer_config=transformer_config,
        group_size=group_size,
        tile_size=32,
        sample_seed=42,
    )

    return ctx


def setup_grpo_config(yaml_config_path) -> GrpoConfig:
    yaml_config = load_config(yaml_config_path, get_tt_metal_runtime_root())

    grpo_cfg = yaml_config.get("grpo_config", {})
    training_cfg = yaml_config["training_config"]

    grpo_config = GrpoConfig(
        clip_eps=float(grpo_cfg.get("clip_eps", 0.2)),
        base_lr=float(training_cfg["optimizer"]["lr"]),
        warmup_steps=int(grpo_cfg.get("warmup_steps", 20)),
        micro_batch_size=int(grpo_cfg.get("micro_batch_size", 16)),
        prompts_to_train=int(grpo_cfg.get("prompts_to_train", 1536)),
        num_mini_epochs=int(training_cfg.get("num_epochs", 1)),
        completions_batch_size=int(training_cfg.get("batch_size", 4)),
    )

    return grpo_config


def setup_training_optimizer(yaml_config_path, tt_model):
    yaml_config = load_config(yaml_config_path, get_tt_metal_runtime_root())
    return create_optimizer(
        yaml_config["training_config"]["optimizer"],
        tt_model.parameters(),
    )
