from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import (
    initialize_device,
    set_seed,
    get_tt_metal_home,
)
from ttml.optimizers import create_optimizer
from ttml.common.config import load_config
from datasets import load_dataset
import time
from typing import List, Tuple
from batched_inference import (
    InferenceCtx,
    generate_answers_multiple_prompts,
    generate_answers_one_prompt,
)

from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from llama_overrides import LlamaCompositeKV
from typing import Iterator, Sequence
from string import Template


def setup(yaml_config_path, hf_model_id, load_pretrained, setup_optimizer=False) -> InferenceCtx:
    set_seed(42)

    yaml_config = load_config(yaml_config_path, f"{get_tt_metal_home()}/tt-train/configs/training_configs")

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

    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = len(tokenizer)
    transformer_config = tt_model_factory.transformer_config

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

    optimizer = None
    if setup_optimizer:
        optimizer = create_optimizer(
            yaml_config["training_config"]["optimizer"],
            tt_model.parameters(),
        )

    ctx = InferenceCtx(
        optimizer=optimizer,
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


def extract_hash_answer(text: str) -> float | None:
    if "####" not in text:
        return None
    s = text.split("####")[1].strip()

    if s is None:
        return float("nan")

    import re

    number_re = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?$")

    if not number_re.fullmatch(s):
        return float("nan")

    return float(s.replace(",", ""))


def get_gsm8k(
    ctx: InferenceCtx, system_prompt, user_prompt_template_str, split="train", shuffle_seed=None
) -> Tuple[List[str], List[float]]:
    data = load_dataset("openai/gsm8k", "main")[split]
    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    dataset = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )

    questions = [ex["question"] for ex in dataset]
    answers = [ex["answer"] for ex in dataset]

    t = Template(user_prompt_template_str)

    prompts = [
        ctx.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": t.substitute(question=q)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    return prompts, answers
