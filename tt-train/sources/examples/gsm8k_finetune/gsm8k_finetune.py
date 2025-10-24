#!/usr/bin/env python3
"""
GSM8K Fine-tuning Script
Fine-tunes a Llama model on the GSM8K math word problems dataset using TT-Metal.
"""

import os
import sys
from time import time
import datasets
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

# Add TT-Metal path
sys.path.append(f"{os.environ['TT_METAL_HOME']}/tt-train/sources/ttml")
import ttml
from ttml.common.config import get_config, TransformerConfig, TrainingConfig, SchedulerConfig, DeviceConfig
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import set_seed, round_up_to_tile, create_optimizer, initialize_device
from ttml.common.data import build_causal_mask

import tt_serialization  # noqa: F401

# Configuration
CONFIG = "training_shakespeare_tinyllama.yaml"
VALIDATION_BATCH_SIZE_PER_DEVICE = 6


class SpeedrunScheduler:
    """Linear warmup -> optional hold -> linear decay; optional beta1 warmup."""

    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg

    def lr_at(self, step: int) -> float:
        s = step
        w = max(0, self.cfg.warmup_steps)
        h = max(0, self.cfg.hold_steps)
        T = max(1, self.cfg.total_steps)
        peak = self.cfg.max_lr
        min_lr = self.cfg.min_lr

        if s <= w:
            # linear warmup 0 -> lr_max
            return peak * (s / max(1, w))
        elif s <= w + h:
            # hold at lr_max
            return peak
        else:
            # linear decay from lr_max at (w+h) to min_lr at T
            s2 = min(s, T)
            frac = (s2 - (w + h)) / max(1, (T - (w + h)))
            return peak + (min_lr - peak) * frac

    def beta1_at(self, step: int) -> Optional[float]:
        if self.cfg.beta1_start is None or self.cfg.beta1_end is None or self.cfg.beta1_warmup_steps <= 0:
            return None
        s = min(step, self.cfg.beta1_warmup_steps)
        t = s / float(self.cfg.beta1_warmup_steps)
        return (1.0 - t) * self.cfg.beta1_start + t * self.cfg.beta1_end


class OptimParamSetter:
    def __init__(self, optim):
        self.optim = optim
        self._warned_lr = False
        self._warned_beta1 = False

    def set_lr(self, lr: float):
        ok = False
        self.optim.set_lr(float(lr))

    def set_beta1(self, beta1: float):
        raise NotImplementedError("set_beta1 is not implemented in TTML AdamW optimizer.")


def build_logits_mask(vocab_size: int, padded_vocab_size: int) -> ttml.autograd.Tensor:
    logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
    logits_mask[:, :, :, vocab_size:] = 1e4
    return ttml.autograd.Tensor.from_numpy(
        logits_mask, ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
    )  # [1,1,1,T], bfloat16"


class CollateFn:
    def __init__(self, eos_token_id, max_sequence_length, padded_vocab_size):
        self.eos_token_id = eos_token_id
        self.max_sequence_length = max_sequence_length
        self.padded_vocab_size = padded_vocab_size

    def collate_fn(self, batch):
        X = [sample[0] for sample in batch]
        Y = [sample[1] for sample in batch]

        batch_size = len(X)

        data_np = np.full((batch_size, self.max_sequence_length), self.eos_token_id, dtype=np.uint32)
        mask_lens = []

        for i in range(batch_size):
            x_tokens = X[i]
            y_tokens = Y[i]

            # Concatenate question + answer
            combined_length = len(x_tokens) + len(y_tokens)
            if combined_length > self.max_sequence_length:

                # Truncate if too long, prioritizing keeping the answer
                available_space = self.max_sequence_length - len(y_tokens)
                if available_space > 0:
                    x_tokens = x_tokens[:available_space]
                    data_np[i, : len(x_tokens)] = x_tokens
                    data_np[i, len(x_tokens) : len(x_tokens) + len(y_tokens)] = y_tokens

                else:
                    # If answer is too long, just use the answer
                    data_np[i, :self.max_sequence_length] = y_tokens[:self.max_sequence_length]
                    x_tokens = []

            else:
                # Normal case: concatenate question + answer

                data_np[i, : len(x_tokens)] = x_tokens
                data_np[i, len(x_tokens) : len(x_tokens) + len(y_tokens)] = y_tokens

            mask_lens.append(len(x_tokens))

        # Shape: [batch_size, 1, 1, max_sequence_length]
        X_np = np.expand_dims(data_np, axis=(1, 2))

        y_np = np.full(
            (batch_size, self.max_sequence_length), self.eos_token_id, dtype=np.uint32
        )  # Shape: [batch, seq_len]
        y_np[:, 0:-1] = X_np[:, 0, 0, 1:]  # Shift left by 1


        loss_scaler_np = np.full((batch_size, 1, self.max_sequence_length, 1), 1.0, dtype=np.float32)
        for i, mask_len in enumerate(mask_lens):
            loss_scaler_np[i, :, :mask_len, :] = 0.0
            pad_positions = X_np[i, 0, 0, :] == self.eos_token_id
            loss_scaler_np[i, :, pad_positions, :] = 0.0
        loss_scaler_ratio = self.max_sequence_length * batch_size / np.sum(loss_scaler_np)
        loss_scaler_np = loss_scaler_np * loss_scaler_ratio

        return X_np, y_np, loss_scaler_np

    def __call__(self, batch):
        return self.collate_fn(batch)

def get_batch_generator(dataloader, batch_size, max_sequence_length, padded_vocab_size, tokenizer, device_config=None):
    """Custom data generator for GSM8K dataset."""
    mapper = None
    if device_config is not None:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
    
    while True:
        for batch in dataloader:
            X_np, y_np, loss_scaler_np = batch

            X = ttml.autograd.Tensor.from_numpy(X_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32, mapper)
            y = ttml.autograd.Tensor.from_numpy(y_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32, mapper)
            loss_scaler = ttml.autograd.Tensor.from_numpy(loss_scaler_np, ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16, mapper)

            yield (X, y, loss_scaler)


def generate_text_tt(
    model,
    tokenizer,
    question,
    max_sequence_length,
    causal_mask,
    temperature,
    logits_mask_tensor,
    max_gen_tokens=576,
    pad_token_id=None,
    return_with_prompt=False,
):
    """
    Greedy/temperature=0 generation that prints the *full* text once at the end.
    Uses a sliding window if prompt exceeds max_sequence_length.
    """
    model.eval()
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)

    # --- Tokenize once ---
    prompt_tokens = tokenizer.encode(question)
    if pad_token_id is None:
        # Try tokenizer.pad_token_id, else fall back to 0
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0

    generated_tokens = []

    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)

    # Preallocate once
    padded_prompt_tokens = np.full((1, 1, 1, max_sequence_length), pad_token_id, dtype=np.uint32)
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
            padded_prompt_tokens, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32
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
    out = tokenizer.decode(generated_tokens)
    if return_with_prompt:
        out = tokenizer.decode(prompt_tokens)
    
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    return out


def validate(
    tt_model,
    tokenizer,
    val_batch_generator,
    testing_data,
    loss_fn,
    causal_mask,
    logits_mask_tensor,
    max_sequence_length,
    current_step,
):
    reduce = ttml.ops.ReduceType.NONE
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    tt_model.eval()
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

        val_file.write("Last validation loss: {:.4f}\n\n\n".format(np.mean(cur_val_losses)))

    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    tt_model.train()
    return np.mean(cur_val_losses)

def adjust_logits(logits, binary_mask, add_mask):
    masked_logits = binary_mask * logits
    masked_logits = masked_logits + add_mask

    return masked_logits

def get_loss_over_devices(loss):
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
    loss_numpy = loss.to_numpy(composer=composer)
    return loss_numpy.mean()

def tokenize_dataset(data, tokenizer):
    X = [sample['question'] for sample in data]
    y = [sample['answer'] for sample in data]

    X = tokenizer(X, return_tensors="np", add_special_tokens=False)["input_ids"]
    y = tokenizer(y, return_tensors="np", add_special_tokens=False)["input_ids"]
    return X, y


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = len(X)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train():
    print("Loading tokenizer and config...")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # Disable tokenizer parallelism to avoid conflicts with DataLoader multiprocessing
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    yaml_config = get_config(CONFIG)

    override_config_path = f"{os.environ['TT_METAL_HOME']}/tt-train/configs/training_overrides.yaml"
    if os.path.isfile(override_config_path):
        print("Applying training overrides...")
        override_config = get_config(override_config_path)
        def deep_update(a: dict, b: dict) -> dict:
            for k, v in b.items():
                if (
                    k in a
                    and isinstance(a[k], dict)
                    and isinstance(v, dict)
                ):
                    deep_update(a[k], v)  # recursive update
                else:
                    a[k] = v              # overwrite or add
            return a
        deep_update(yaml_config, override_config)

        # pretty output of yaml config
        import yaml
        print("Loaded YAML config:")
        print(yaml.dump(yaml_config, sort_keys=False, default_flow_style=False))
        print('*********************************\n\n')


    training_config = TrainingConfig(yaml_config)
    scheduler_config = SchedulerConfig(yaml_config)

    batch_size = training_config.batch_size

    # initialize device
    device_config = DeviceConfig(yaml_config)
    # no need to initialize device if #devices=1
    if device_config.total_devices() > 1:
        initialize_device(yaml_config)

    # Download safetensors
    print("Downloading safetensors...")
    safetensors_path = hf_hub_download(
        repo_id="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", filename="model.safetensors"
    )
    safetensors_path = safetensors_path.replace("model.safetensors", "")
    print(f"Safetensors path: {safetensors_path}")

    # Setup model
    print("Setting up model...")
    orig_vocab_size = tokenizer.vocab_size
    tt_model_factory = TransformerModelFactory(yaml_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
    print("Created Model Factory")

    max_sequence_length = tt_model_factory.transformer_config.max_sequence_length

    print("Creating model...")
    tt_model = tt_model_factory.create_model()
    print("Loading from safetensors...")
    tt_model.load_from_safetensors(safetensors_path)

    padded_vocab_size = round_up_to_tile(orig_vocab_size, 32)

    # Load dataset
    print("Loading GSM8K dataset...")
    training_data = datasets.load_dataset("gsm8k", "main", split="train")
    testing_data = datasets.load_dataset("gsm8k", "main", split="test")

    training_data_x, training_data_y = tokenize_dataset(training_data, tokenizer)
    testing_data_x, testing_data_y = tokenize_dataset(testing_data, tokenizer)
    training_data = TokenizedDataset(training_data_x, training_data_y)
    testing_data = TokenizedDataset(testing_data_x, testing_data_y)

    training_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the dataset for each epoch
        drop_last=True,
        num_workers=0,
        collate_fn=CollateFn(tokenizer.eos_token_id, max_sequence_length, padded_vocab_size),
    )

    num_devices = device_config.total_devices()
    testing_dataloader = DataLoader(
        testing_data,
        batch_size=VALIDATION_BATCH_SIZE_PER_DEVICE * num_devices,
        shuffle=False,  # Disable shuffling for validation
        drop_last=True,
        num_workers=0,
        collate_fn=CollateFn(tokenizer.eos_token_id, max_sequence_length, padded_vocab_size),
    )

    # Setup training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optim = create_optimizer(tt_model, yaml_config)
    causal_mask = build_causal_mask(max_sequence_length)

    causal_mask = ttml.autograd.Tensor.from_numpy(causal_mask, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.BFLOAT16)

    logits_mask_tensor = build_logits_mask(orig_vocab_size, padded_vocab_size)

    loss_fn = ttml.ops.loss.cross_entropy_loss
    reduce = ttml.ops.ReduceType.NONE

    # Training setup
    tt_model.train()
    train_losses = []
    val_losses = []

    # remove output.txt if it exists
    if os.path.exists("output.txt"):
        os.remove("output.txt")

    train_batch_generator = get_batch_generator(
        training_dataloader, batch_size, max_sequence_length, padded_vocab_size, tokenizer, device_config
    )

    val_batch_generator = get_batch_generator(testing_dataloader, VALIDATION_BATCH_SIZE_PER_DEVICE * num_devices, max_sequence_length, padded_vocab_size, tokenizer, device_config)

    tokens_per_batch = batch_size * max_sequence_length
    print("Tokens per micro-batch:", tokens_per_batch)
    print("Tokens per accumulated batch:", tokens_per_batch * training_config.gradient_accumulation_steps)

    sched = SpeedrunScheduler(scheduler_config)
    setter = OptimParamSetter(optim)

    f = open("validation.txt", "w")
    f.write("Validation log\n")
    f.write("===============\n")
    f.close()

    print(f"Starting training for {training_config.epochs} epochs, max {training_config.steps} steps...")
    bar = tqdm(range(1, training_config.steps + 1))

    total_steps = 0
    last_val_loss = 0
    accum_steps = training_config.gradient_accumulation_steps


    # ========== Training Loop ===========
    for opt_step in bar:
        # LR (and optional beta1) updated once per optimizer step
        optim.zero_grad()
        lr_now = sched.lr_at(opt_step - 1)  # zero-based inside scheduler
        setter.set_lr(lr_now)

        # ---- internal micro-steps ----
        # Aggregate the true (unscaled) mean losses across micro-steps to report per optimizer step.
        micro_losses = []

        for micro in range(accum_steps):
            X, y, loss_scaler = next(train_batch_generator)

            # Forward
            logits = tt_model(X, causal_mask)  # [B,1,T,V]

            # CE on masked logits
            loss = loss_fn(logits, y, reduce)  # [B,1,T,1] shape reduced later
            loss = loss * loss_scaler
            loss = ttml.ops.unary.mean(loss)  # scalar

            # Track true loss for reporting
            # micro_losses.append(float(loss.to_numpy()))
            micro_losses.append(get_loss_over_devices(loss))

            # Scale for accumulation and backward
            scaled_loss = ttml.ops.binary.mul(loss, 1.0 / float(accum_steps))   # check if accum_steps > 1 
            scaled_loss.backward(False)
            ttml.autograd.AutoContext.get_instance().reset_graph()

        #Synchronize gradients if DDP is enabled
        if device_config.enable_ddp:
            ttml.core.distributed.synchronize_parameters(tt_model.parameters())

        # Optimizer step after micro-steps
        optim.step()

        # Average loss across micro-steps (this corresponds to the optimizer step)
        step_loss = float(np.mean(micro_losses)) if len(micro_losses) > 0 else 0.0
        train_losses.append(step_loss)

        # tqdm postfix
        postfix = {"train_loss": f"{step_loss:.4f}", "lr": f"{lr_now:.6f}"}
        if last_val_loss is not None:
            postfix["val_loss"] = f"{last_val_loss:.4f}"
        bar.set_postfix(postfix, refresh=False)

        # Validation every eval_every steps
        if total_steps % training_config.eval_every == 0 or \
            total_steps + 1 == training_config.steps:
            last_val_loss = validate(
                tt_model,
                tokenizer,
                val_batch_generator,
                testing_data,
                loss_fn,
                causal_mask,
                logits_mask_tensor,
                max_sequence_length,
                total_steps,
            )
            val_losses.append(last_val_loss)
        """
        if training_config.save_every > 0 and (total_steps % training_config.save_every == 0):
            ckpt_path = tt_serialization._save_model_npz(
                tt_model, training_config.checkpoint_dir, step=total_steps
            )
            print(f"[Checkpoint] Saved model to {ckpt_path}")
        """

        with open("output.txt", "a") as f:
            f.write(
                f"LR: {lr_now:.6f}, training_loss: {step_loss:.4f}, val_loss: {last_val_loss:.4f}, step: {total_steps}, epoch: 1\n"
            )
        total_steps += 1

    print("Training completed!")

    # Plot training curves
    print("Plotting training curves...")
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(train_losses, color="blue", label="Train Loss")
    axs.plot(np.arange(0, len(val_losses)) * training_config.eval_every, val_losses, color="orange", label="Val Loss")
    axs.set_title("Training Loss")
    axs.set_xlabel("Steps")
    axs.set_ylabel("Loss")
    axs.legend()
    plt.savefig("training_curves.png")
    plt.show()


if __name__ == "__main__":
    train()
