#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GSM8K Fine-tuning Script
Fine-tunes a Llama model on the GSM8K math word problems dataset using TT-Metal.
"""

import os
from functools import partial

import datasets
import numpy as np
import ttnn
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

import ttml
from ttml.common.config import (
    TrainingConfig,
    DeviceConfig,
    SchedulerConfig,
    load_config,
    yaml_deep_update,
)
from ttml.common.schedulers import SpeedrunScheduler
from ttml.common.utils import (
    round_up_to_tile,
    initialize_device,
    build_logits_mask,
    no_grad,
    get_tt_metal_runtime_root,
)
from ttml.common.data import build_causal_mask
from ttml.datasets import Batch, InMemoryDataloader
from ttml.models import RunnerType, WeightTyingType
from ttml.models.nanogpt import NanoGPT, NanoGPTConfig, NanoGPTExperimentalConfig, load_gpt2_from_safetensors
from ttml.models.llama import Llama, LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from ttml.modules import LoraConfig
from ttml.trainers import SFTConfig, SFTTrainer, TrainerCallback


class DDPCallback(TrainerCallback):
    """Synchronise gradients across all DDP devices before the optimiser step."""

    def on_before_optimizer_step(self, trainer):
        ttml.core.distributed.synchronize_gradients(trainer.model.parameters())


class MetricsLogger(TrainerCallback):
    """Write metric lines to output.txt in the format expected by slurm_training_service.

    Format: LR: <lr>, training_loss: <loss>, [val_loss: <val>,] step: <step>, epoch: 1
    """

    def __init__(self, path: str = "output.txt"):
        self._path = path
        self._last_loss: float = 0.0
        self._last_lr: float = 0.0

    def on_step_end(self, trainer, step, loss, lr):
        self._last_loss = loss
        self._last_lr = lr
        with open(self._path, "a") as f:
            f.write(f"LR: {lr}, training_loss: {loss:.4f}, step: {step}, epoch: 1\n")

    def on_eval_end(self, trainer, step, eval_loss):
        with open(self._path, "a") as f:
            f.write(
                f"LR: {self._last_lr}, training_loss: {self._last_loss:.4f}, "
                f"val_loss: {eval_loss:.4f}, step: {step}, epoch: 1\n"
            )


# Configuration
CONFIG = "training_gsm8k_tinyllama.yaml"


def gsm8k_collate_fn(
    batch: list,
    eos_token_id: int,
    max_sequence_length: int,
    mapper=None,
) -> Batch:
    """Collate (question_tokens, answer_tokens) pairs into a :class:`Batch`.

    Loss is computed only on answer (completion) tokens; question (prompt) tokens
    and padding are masked out via ``Batch.loss_mask``.
    """
    X, Y = map(list, zip(*batch))
    batch_size = len(X)

    data_np = np.full((batch_size, max_sequence_length), eos_token_id, dtype=np.uint32)
    prompt_lens = []

    for i, (x_tokens, y_tokens) in enumerate(zip(X, Y)):
        x_len = len(x_tokens)
        y_len = len(y_tokens)
        total_len = x_len + y_len

        if total_len > max_sequence_length:
            available_space = max_sequence_length - y_len
            if available_space > 0:
                # Truncate question, keep full answer
                x_tokens = x_tokens[:available_space]
                x_len = available_space
                data_np[i, :x_len] = x_tokens
                data_np[i, x_len : x_len + y_len] = y_tokens
            else:
                # Answer alone is too long: keep only (part of) the answer
                y_tokens = y_tokens[:max_sequence_length]
                y_len = max_sequence_length
                data_np[i, :y_len] = y_tokens
                x_len = 0
        else:
            data_np[i, :x_len] = x_tokens
            data_np[i, x_len : x_len + y_len] = y_tokens

        prompt_lens.append(x_len)

    # input_ids: [B, 1, 1, T]
    input_ids_np = data_np.reshape(batch_size, 1, 1, max_sequence_length)

    # labels: shift left by 1 for next-token prediction
    labels_np = np.full((batch_size, max_sequence_length), eos_token_id, dtype=np.uint32)
    labels_np[:, :-1] = input_ids_np[:, 0, 0, 1:]

    # loss_mask: 0 for prompt tokens and padding, 1 elsewhere, then normalised
    loss_mask_np = np.ones((batch_size, 1, max_sequence_length, 1), dtype=np.float32)
    for i, prompt_len in enumerate(prompt_lens):
        loss_mask_np[i, :, :prompt_len, :] = 0.0
        pad_positions = input_ids_np[i, 0, 0, :] == eos_token_id
        loss_mask_np[i, :, pad_positions, :] = 0.0

    total_weight = loss_mask_np.sum()
    if total_weight > 0:
        loss_mask_np *= (batch_size * max_sequence_length) / total_weight

    return Batch(
        input_ids=ttml.autograd.Tensor.from_numpy(input_ids_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper),
        labels=ttml.autograd.Tensor.from_numpy(labels_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper),
        loss_mask=ttml.autograd.Tensor.from_numpy(loss_mask_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper),
    )


def generate_text_tt(
    model,
    tokenizer: AutoTokenizer,
    question: str,
    max_sequence_length: int,
    causal_mask: ttml.autograd.Tensor,
    temperature: float,
    logits_mask_tensor: ttml.autograd.Tensor,
    max_gen_tokens: int = 576,
    pad_token_id: int = None,
    return_with_prompt: bool = False,
):
    """
    Greedy/temperature=0 generation that prints the *full* text once at the end.
    Uses a sliding window if prompt exceeds max_sequence_length.

    model: TT model
    tokenizer: HuggingFace tokenizer
    question: input question string
    max_sequence_length: maximum sequence length
    causal_mask: causal mask tensor
    temperature: sampling temperature (0.0 for greedy)
    logits_mask_tensor: logits mask tensor (mask that keeps answer tokens)
    max_gen_tokens: maximum number of tokens to generate
    pad_token_id: padding token id
    return_with_prompt: if True, return full text including prompt
    """
    model.eval()

    # --- Tokenize once ---
    prompt_tokens = tokenizer.encode(question)
    if pad_token_id is None:
        # Try tokenizer.pad_token_id, else fall back to 0
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    generated_tokens = []

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    # Preallocate once
    padded_prompt_tokens = np.full((1, 1, 1, max_sequence_length), pad_token_id, dtype=np.uint32)

    with no_grad():
        for _ in range(max_gen_tokens):
            # Sliding window for long prompts
            if len(prompt_tokens) > max_sequence_length:
                start_idx = len(prompt_tokens) - max_sequence_length
                window = prompt_tokens[start_idx:]
            else:
                start_idx = 0
                window = prompt_tokens

            # Refill buffer (fully) to avoid stale ids
            padded_prompt_tokens[...] = pad_token_id
            padded_prompt_tokens[0, 0, 0, : len(window)] = np.asarray(window, dtype=np.uint32)

            # [1,1,1,T] -> TT tensor
            padded_prompt_tensor = ttml.autograd.Tensor.from_numpy(
                padded_prompt_tokens,
                ttnn.Layout.ROW_MAJOR,
                ttnn.DataType.UINT32,
            )

            # Forward: logits [1,1,T,V]
            logits = model(padded_prompt_tensor, causal_mask)

            # Sample: next tokens for all positions [1,1,T,1]
            # With temperature=0.0 this behaves like argmax/greedy.
            next_token_tensor = ttml.ops.sample.sample_op(logits, 0.0, np.random.randint(low=1e7), logits_mask_tensor)

            # Take the token at the last active position in the current window
            next_token_idx = max_sequence_length - 1 if len(prompt_tokens) > max_sequence_length else len(window) - 1
            next_token = int(next_token_tensor.to_numpy(composer=composer).reshape(-1, 1)[next_token_idx][0])

            if next_token == tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token)
            prompt_tokens.append(next_token)

        # Decode once at the end
        out = tokenizer.decode(prompt_tokens if return_with_prompt else generated_tokens)

    model.train()

    return out


def validate(
    tt_model,
    tokenizer: AutoTokenizer,
    val_batch_generator,
    testing_data,
    loss_fn,
    causal_mask: ttml.autograd.Tensor,
    logits_mask_tensor: ttml.autograd.Tensor,
    max_sequence_length: int,
    current_step: int,
):
    """
    Validation function that computes loss and generates answers for a few samples.

    tt_model: TT model
    tokenizer: HuggingFace tokenizer
    val_batch_generator: generator yielding validation batches (from get_batch_generator)
    testing_data: tokenized testing dataset
    loss_fn: loss function
    causal_mask: causal mask tensor
    logits_mask_tensor: logits mask tensor (mask that keeps answer tokens)
    max_sequence_length: maximum sequence length
    current_step: current training step
    """
    reduce = ttml.ops.ReduceType.NONE

    tt_model.eval()

    with no_grad():
        eval_batch_count = 4
        cur_val_losses = []
        for _ in range(eval_batch_count):
            val_X, val_y, val_loss_scaler = next(val_batch_generator)
            val_logits = tt_model(val_X, causal_mask)

            # Compute validation loss
            val_loss = loss_fn(val_logits, val_y, reduce)
            val_loss = val_loss * val_loss_scaler
            val_loss = ttml.ops.unary.mean(val_loss)
            cur_val_losses.append(get_loss_over_devices(val_loss))

        checks_count = 4

    with open("validation.txt", "a+") as val_file:
        val_file.write(f"Validation at step {current_step}\n")
        for check in range(checks_count):
            val_file.write(f"Validation check: {check}\n")
            val_file.write("====================================\n")

            tokenized_question, tokenized_answer = testing_data[check]
            question = tokenizer.decode(tokenized_question, skip_special_tokens=True)

            val_file.write(f"Question: {question}\n")
            val_file.write("====================================\n")

            gen_text = generate_text_tt(
                tt_model,
                tokenizer,
                question,
                max_sequence_length,
                causal_mask,
                0.0,
                logits_mask_tensor,
            )

            val_file.write(f"Generated Answer: {gen_text}\n")
            val_file.write("\n====================================\n")

        val_file.write(f"Last validation loss: {float(np.mean(cur_val_losses)):.4f}\n\n\n")

    tt_model.train()
    return np.mean(cur_val_losses)


class TokenizedDataset:
    """
    A simple Dataset class for tokenized data.

    X: list of tokenized questions
    y: list of tokenized answers
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def tokenize_dataset(data, tokenizer: AutoTokenizer) -> TokenizedDataset:
    """
    Tokenizes the questions and answers in the dataset using the provided tokenizer.

    data: dataset with "question" and "answer" fields
    tokenizer: HuggingFace tokenizer
    """
    X = [sample["question"] for sample in data]
    y = [sample["answer"] for sample in data]

    tok = lambda texts: tokenizer(texts, return_tensors="np", add_special_tokens=False)["input_ids"]
    return TokenizedDataset(tok(X), tok(y))


def train():
    """
    Main training loop for fine-tuning on GSM8K dataset.
    """
    yaml_config = load_config(CONFIG, f"{get_tt_metal_runtime_root()}/tt-train/configs/training_configs")
    model_config = load_config(yaml_config["training_config"]["model_config"])

    override_config_path = os.environ.get(
        "TT_TRAIN_OVERRIDES_PATH",
        f"{get_tt_metal_runtime_root()}/tt-train/configs/training_overrides.yaml",
    )

    if os.path.isfile(override_config_path):
        print("Applying training overrides...")

        override_config = load_config(override_config_path)

        yaml_config = yaml_deep_update(yaml_config, override_config)
        model_config = yaml_deep_update(model_config, override_config)

        # pretty output of yaml config
        import yaml

        print("Loaded YAML config:")
        print(yaml.dump(yaml_config, sort_keys=False, default_flow_style=False))
        print("*********************************\n\n")

    training_config = TrainingConfig(yaml_config)
    scheduler_config = SchedulerConfig(yaml_config)

    batch_size = training_config.batch_size

    # initialize device
    device_config = DeviceConfig(yaml_config)

    # no need to initialize device if #devices=1
    if device_config.total_devices() > 1:
        initialize_device(yaml_config)

    ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(
        ttml.autograd.DistributedConfig(enable_ddp=device_config.enable_ddp, enable_tp=device_config.enable_tp)
    )

    use_ddp = device_config.enable_ddp and device_config.total_devices() > 1
    mapper = None
    if use_ddp:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)

    tc = model_config["transformer_config"]
    model_type = tc["model_type"]

    if model_type == "gpt2":
        repo_id = "gpt2"
    else:
        repo_id = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    print("Loading tokenizer...")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    print("Downloading safetensors...")
    safetensors_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
    )
    safetensors_path = safetensors_path.replace("model.safetensors", "")
    print(f"Safetensors path: {safetensors_path}")

    orig_vocab_size = tokenizer.vocab_size
    padded_vocab_size = round_up_to_tile(orig_vocab_size, 32)

    print("Creating model...")
    if model_type == "gpt2":
        runner_type = RunnerType.from_string(tc.get("runner_type", "default"))
        weight_tying = WeightTyingType.from_string(tc.get("weight_tying", "disabled"))
        gpt2_config = NanoGPTConfig(
            vocab_size=padded_vocab_size,
            block_size=tc.get("max_sequence_length", 1024),
            n_embd=tc.get("embedding_dim", 768),
            n_layer=tc.get("num_blocks", 12),
            n_head=tc.get("num_heads", 12),
            dropout=tc.get("dropout_prob", 0.2),
            runner_type=runner_type,
            weight_tying=weight_tying,
            experimental=NanoGPTExperimentalConfig(
                use_composite_layernorm=tc.get("experimental", {}).get("use_composite_layernorm", False),
            ),
        )
        tt_model = NanoGPT(gpt2_config)
        max_sequence_length = gpt2_config.block_size
        print("Loading GPT-2 weights from safetensors...")
        load_gpt2_from_safetensors(tt_model, safetensors_path, gpt2_config)
    elif model_type == "llama":
        runner_type = RunnerType.from_string(tc.get("runner_type", "default"))
        weight_tying = WeightTyingType.from_string(tc.get("weight_tying", "disabled"))
        rope_scaling_cfg = LlamaRopeScalingConfig()
        if "rope_scaling" in tc:
            rs = tc["rope_scaling"]
            rope_scaling_cfg = LlamaRopeScalingConfig(
                scaling_factor=rs.get("scaling_factor", 0.0),
                high_freq_factor=rs.get("high_freq_factor", 4.0),
                low_freq_factor=rs.get("low_freq_factor", 1.0),
                original_context_length=rs.get("original_context_length", 0),
            )
        llama_config = LlamaConfig(
            hidden_size=tc.get("embedding_dim", 384),
            num_hidden_layers=tc.get("num_blocks", 6),
            num_attention_heads=tc.get("num_heads", 6),
            num_key_value_heads=tc.get("num_groups", 3),
            vocab_size=padded_vocab_size,
            max_position_embeddings=tc.get("max_sequence_length", 256),
            rope_theta=tc.get("theta", 10000.0),
            attention_dropout=tc.get("dropout_prob", 0.0),
            mlp_dropout=tc.get("dropout_prob", 0.0),
            runner_type=runner_type,
            weight_tying=weight_tying,
            rope_scaling=rope_scaling_cfg,
        )
        tt_model = Llama(llama_config)
        max_sequence_length = llama_config.max_position_embeddings
        print("Loading Llama weights from safetensors...")
        load_from_safetensors(tt_model, safetensors_path, llama_config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported: gpt2, llama")

    # Load dataset
    print("Loading GSM8K dataset...")
    training_data = datasets.load_dataset("gsm8k", "main", split="train", ignore_verifications=True)
    testing_data = datasets.load_dataset("gsm8k", "main", split="test", ignore_verifications=True)

    training_data = tokenize_dataset(training_data, tokenizer)
    testing_data = tokenize_dataset(testing_data, tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collate = partial(
        gsm8k_collate_fn,
        eos_token_id=tokenizer.eos_token_id,
        max_sequence_length=max_sequence_length,
        mapper=mapper,
    )

    num_devices = device_config.total_devices()
    train_loader = InMemoryDataloader(training_data, collate, batch_size, shuffle=True)
    eval_loader = InMemoryDataloader(
        testing_data,
        collate,
        training_config.validation_batch_size * num_devices,
    )

    tokens_per_batch = batch_size * max_sequence_length
    print("Tokens per micro-batch:", tokens_per_batch)
    print(
        "Tokens per accumulated batch:",
        tokens_per_batch * training_config.gradient_accumulation_steps,
    )

    sft_config = SFTConfig(
        max_steps=training_config.steps,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        eval_interval=training_config.eval_every,
        save_interval=training_config.save_every,
        checkpoint_dir=training_config.checkpoint_dir,
        max_seq_len=max_sequence_length,
        learning_rate=scheduler_config.max_lr,
        warmup_steps=scheduler_config.warmup_steps,
    )

    optimizer_cfg = {
        "type": "AdamW",
        "lr": scheduler_config.max_lr,
    }

    sched = SpeedrunScheduler(scheduler_config)

    mask_np = build_causal_mask(max_sequence_length)
    causal_mask = ttml.autograd.Tensor.from_numpy(mask_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    callbacks = [MetricsLogger()]
    if use_ddp:
        callbacks.append(DDPCallback())

    peft_config = None
    lora_cfg = yaml_config.get("lora_config")
    if lora_cfg:
        peft_config = LoraConfig(
            rank=lora_cfg.get("rank", 8),
            alpha=lora_cfg.get("alpha", 16),
            target_modules=lora_cfg.get("target_modules", ["q_linear", "kv_linear", "out_linear"]),
            lora_dropout=lora_cfg.get("lora_dropout", 0.05),
            use_rslora=lora_cfg.get("use_rslora", False),
            is_bias_trainable=lora_cfg.get("is_bias_trainable", False),
            verbose=True,
        )

    trainer = SFTTrainer(
        model=tt_model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=sft_config,
        optimizer=optimizer_cfg,
        lr_schedule=sched.lr_at,  # SpeedrunScheduler uses 0-based step index
        attention_mask=causal_mask,
        callbacks=callbacks,
        peft_config=peft_config,
    )

    print(f"Starting training for max {training_config.steps} steps...")
    trainer.train()
    print("Training completed!")

    # Cleanup
    ttml.autograd.AutoContext.get_instance().close_device()


if __name__ == "__main__":
    train()
