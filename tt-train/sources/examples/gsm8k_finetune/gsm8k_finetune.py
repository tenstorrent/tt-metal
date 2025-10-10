#!/usr/bin/env python3
"""
GSM8K Fine-tuning Script
Fine-tunes a GPT-2 model on the GSM8K math word problems dataset using TT-Metal.
"""

import os
import sys
import datasets
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from tqdm import tqdm
import debugpy
from typing import List, Tuple, Dict, Optional

# Add TT-Metal path
sys.path.append(f"{os.environ['TT_METAL_HOME']}/tt-train/sources/ttml")
import ttml
from ttml.common.config import get_config, TransformerConfig, TrainingConfig
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import set_seed, round_up_to_tile, create_optimizer
from ttml.common.data import build_causal_mask

import tt_serialization  # noqa: F401

# Configuration
CONFIG = "training_shakespeare_tinyllama.yaml"


class GradientAccumulator:
    """Helper class for gradient accumulation."""

    def __init__(self, accumulation_steps: int):
        self.m_accumulation_steps = int(max(1, accumulation_steps))
        self.m_total_loss: float = 0.0
        self.m_total_samples: int = 0
        self.m_steps: int = 0  # micro-steps seen

    def should_zero_grad(self) -> bool:
        return (self.m_steps % self.m_accumulation_steps) == 0

    def should_step(self) -> bool:
        return (self.m_steps % self.m_accumulation_steps) == (self.m_accumulation_steps - 1)

    def scale(self, tensor):
        if self.m_accumulation_steps > 1:
            return ttml.ops.binary.mul(tensor, 1.0 / float(self.m_accumulation_steps))
        return tensor

    def update(self, loss_value: float, samples: int):
        self.m_total_loss += float(loss_value) * float(samples) * float(self.m_accumulation_steps)
        self.m_total_samples += int(samples)
        self.m_steps += 1

    def reset(self):
        self.m_total_loss = 0.0
        self.m_total_samples = 0
        self.m_steps = 0

    def average_loss(self) -> float:
        return (self.m_total_loss / float(self.m_total_samples)) if self.m_total_samples > 0 else 0.0


@dataclass
class SchedulerConfig:
    lr_max: float = 3e-4
    min_lr: float = 0.0
    warmup_steps: int = 2000
    hold_steps: int = 0
    total_steps: int = 150_000  # full micro-steps or optimizer steps; we apply on optimizer steps
    # optional momentum warmup (beta1 ramp)
    beta1_start: Optional[float] = None  # e.g., 0.85
    beta1_end: Optional[float] = None  # e.g., 0.9
    beta1_warmup_steps: int = 0


class SpeedrunScheduler:
    """Linear warmup -> optional hold -> linear decay; optional beta1 warmup."""

    def __init__(self, cfg: SchedulerConfig):
        self.cfg = cfg

    def lr_at(self, step: int) -> float:
        s = step
        w = max(0, self.cfg.warmup_steps)
        h = max(0, self.cfg.hold_steps)
        T = max(1, self.cfg.total_steps)
        peak = self.cfg.lr_max
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


def get_batch_generator(dataloader, batch_size, max_sequence_length, padded_vocab_size, tokenizer):
    """Custom data generator for GSM8K dataset."""

    while True:
        for batch in dataloader:
            batch = next(iter(dataloader))
            X = batch["question"]
            Y = batch["answer"]

            # Batch tokenize questions and answers
            x_tokens_batch = tokenizer(X, return_tensors="np", add_special_tokens=False)["input_ids"]
            y_tokens_batch = tokenizer(Y, return_tensors="np", add_special_tokens=False)["input_ids"]

            data_np = np.full((batch_size, max_sequence_length), tokenizer.eos_token_id, dtype=np.uint32)
            mask_lens = []

            for i in range(batch_size):
                x_tokens = x_tokens_batch[i]
                y_tokens = y_tokens_batch[i]

                # Concatenate question + answer
                combined_length = len(x_tokens) + len(y_tokens)
                if combined_length > max_sequence_length:
                    # Truncate if too long, prioritizing keeping the answer
                    available_space = max_sequence_length - len(y_tokens)
                    if available_space > 0:
                        x_tokens = x_tokens[:available_space]
                        data_np[i, : len(x_tokens)] = x_tokens
                        data_np[i, len(x_tokens) : len(x_tokens) + len(y_tokens)] = y_tokens
                    else:
                        # If answer is too long, just use the answer
                        data_np[i, :max_sequence_length] = y_tokens[:max_sequence_length]
                        x_tokens = []
                else:
                    # Normal case: concatenate question + answer
                    data_np[i, : len(x_tokens)] = x_tokens
                    data_np[i, len(x_tokens) : len(x_tokens) + len(y_tokens)] = y_tokens

                mask_lens.append(len(x_tokens))

            # Shape: [batch_size, 1, 1, max_sequence_length]
            X_np = np.expand_dims(data_np, axis=(1, 2))

            y_np = np.full(
                (batch_size, max_sequence_length), tokenizer.eos_token_id, dtype=np.uint32
            )  # Shape: [batch, seq_len]
            y_np[:, 0:-1] = X_np[:, 0, 0, 1:]  # Shift left by 1

            logits_mask_np = np.ones((batch_size, 1, max_sequence_length, padded_vocab_size), dtype=np.float32)

            for i, mask_len in enumerate(mask_lens):
                # Mask out the question tokens (first mask_len tokens)
                logits_mask_np[i, :, : mask_len - 1, :] = 0.0
                # Also mask padding tokens
                pad_positions = X_np[i, 0, 0, :] == tokenizer.eos_token_id
                logits_mask_np[i, :, pad_positions, :] = 0.0

            scaler_np = logits_mask_np[..., 0].sum(axis=-1, keepdims=True)  # Shape: [batch_size, 1, 1]
            scaler_np = max_sequence_length / scaler_np
            scaler_np = np.expand_dims(scaler_np, axis=-1)
            scaler_np = np.repeat(scaler_np, max_sequence_length, axis=2)

            logits_add_mask_np = np.zeros_like(logits_mask_np, dtype=np.float32)

            # Find positions where mask is 0 (question tokens)
            mask_positions = logits_mask_np[:, 0, :, 0] == 0.0  # Shape: [batch_size, max_sequence_length]

            # Use advanced indexing to set the corresponding target token positions to 1e3
            batch_indices, seq_indices = np.where(mask_positions)
            target_tokens = y_np[batch_indices, seq_indices]
            logits_add_mask_np[batch_indices, 0, seq_indices, target_tokens] = 1e3

            logits_add_mask = ttml.autograd.Tensor.from_numpy(
                logits_add_mask_np, ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
            )

            logits_mask = ttml.autograd.Tensor.from_numpy(
                logits_mask_np, ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16
            )

            scaler = ttml.autograd.Tensor.from_numpy(scaler_np, ttml.Layout.TILE, ttml.autograd.DataType.BFLOAT16)

            X = ttml.autograd.Tensor.from_numpy(X_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32)
            y = ttml.autograd.Tensor.from_numpy(y_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32)

            yield (X, y, logits_mask, logits_add_mask, scaler)


def generate_text_tt(
    model, tokenizer, question, max_sequence_length, causal_mask, temperature, logits_mask_tensor, max_gen_tokens=100
):
    """Generate text given a question."""
    model.eval()
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)

    # Tokenize and prepare input
    prompt_tokens = tokenizer.encode(question)

    # Generate tokens autoregressively
    generated_tokens = []
    padded_prompt_tokens = np.zeros((1, 1, 1, max_sequence_length), dtype=np.uint32)
    start_idx = 0

    for _ in range(max_gen_tokens):
        if len(prompt_tokens) > max_sequence_length:
            start_idx = len(prompt_tokens) - max_sequence_length

        padded_prompt_tokens[0, 0, 0, : len(prompt_tokens)] = prompt_tokens[start_idx:]
        padded_prompt_tensor = ttml.autograd.Tensor.from_numpy(
            padded_prompt_tokens, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32
        )  # [1,1,1, max_seq_len], uint32

        logits = model(padded_prompt_tensor, causal_mask)  # out=[1,1,seq_len, vocab_size], bf16

        next_token_tensor = ttml.ops.sample.sample_op(
            logits, 0.0, np.random.randint(low=1e7), logits_mask_tensor
        )  # out=[1,1,seq_len,1], uint32

        next_token_idx = max_sequence_length - 1 if len(prompt_tokens) > max_sequence_length else len(prompt_tokens) - 1
        next_token = next_token_tensor.to_numpy().flatten()[next_token_idx]

        generated_tokens.append(next_token)
        print(tokenizer.decode(next_token), end="", flush=True)
        prompt_tokens.append(next_token)

    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.ENABLED)


def generate_text_tinyllama(tokenizer, question, max_gen_tokens=100, temperature=0.7):
    """Generate text using HuggingFace TinyLlama for comparison."""
    # Load TinyLlama model with BF16 precision
    tinyllama_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", torch_dtype=torch.bfloat16
    )
    tinyllama_model.eval()

    # Tokenize input
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generate text
    print(f"\nTinyLlama Generation for: {question}")
    print("=" * 50)
    print(question, end="", flush=True)

    with torch.no_grad():
        generated_ids = tinyllama_model.generate(
            input_ids,
            max_new_tokens=max_gen_tokens,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (excluding input)
    generated_text = tokenizer.decode(generated_ids[0][input_ids.shape[1] :], skip_special_tokens=True)
    print(generated_text)
    print("=" * 50)

    return generated_text


def adjust_logits(logits, binary_mask, add_mask):
    masked_logits = binary_mask * logits
    masked_logits = masked_logits + add_mask

    return masked_logits


def main():
    print("Loading tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    yaml_config = get_config(CONFIG)

    training_config = TrainingConfig(yaml_config)
    batch_size = training_config.batch_size

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

    training_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the dataset for each epoch
    )

    testing_dataloader = DataLoader(
        testing_data,
        batch_size=6,
        shuffle=True,  # Shuffle the dataset for each epoch
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
    data_steps = (len(training_data) // batch_size) + 1
    train_losses = []
    val_losses = []

    train_batch_generator = get_batch_generator(
        training_dataloader, batch_size, max_sequence_length, padded_vocab_size, tokenizer
    )
    val_batch_generator = get_batch_generator(testing_dataloader, 6, max_sequence_length, padded_vocab_size, tokenizer)

    accum = GradientAccumulator(training_config.gradient_accumulation_steps)
    print("Gradient Accumulation Steps:", training_config.gradient_accumulation_steps)
    tokens_per_batch = batch_size * max_sequence_length
    print("Tokens per micro-batch:", tokens_per_batch)
    print("Tokens per accumulated batch:", tokens_per_batch * training_config.gradient_accumulation_steps)
    optim_steps_done = 0

    total_optim_steps = max(1, training_config.steps // max(1, training_config.gradient_accumulation_steps))
    sched = SpeedrunScheduler(
        SchedulerConfig(
            lr_max=1e-4,
            min_lr=3e-5,
            warmup_steps=250,
            hold_steps=1000,
            total_steps=total_optim_steps,
            beta1_start=0.85,
            beta1_end=0.9,
            beta1_warmup_steps=3000,
        )
    )
    setter = OptimParamSetter(optim)

    print(f"Starting training for {training_config.epochs} epochs, max {training_config.steps} steps...")
    bar = tqdm(range(1, data_steps + 1))

    total_steps = 0
    last_val_loss = 0
    accum_loss = 0

    # ========== Training Loop ===========
    for step in bar:
        if accum.should_zero_grad():
            optim.zero_grad()

            lr_now = sched.lr_at(optim_steps_done)
            setter.set_lr(lr_now)

        X, y, logits_mask, logits_add_mask, loss_scaler = next(train_batch_generator)

        # Forward pass: input is the concatenated sequence
        logits = tt_model(X, causal_mask)  # Shape: [batch, 1, seq_len, vocab_size]

        # Apply mask to logits (zero out question token logits)
        masked_logits = adjust_logits(logits, logits_mask, logits_add_mask)

        # Compute cross-entropy loss on masked logits
        unscaled_loss = loss_fn(masked_logits, y, reduce)

        loss = unscaled_loss * loss_scaler  # Scale loss based on non-masked tokens
        loss = ttml.ops.unary.mean(loss)

        scaled_loss = accum.scale(loss)
        scaled_loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        train_loss = float(loss.to_numpy())
        accum_loss += train_loss
        accum.update(train_loss, tokens_per_batch)

        if step == 1:
            print(f"First loss: {train_loss:.4f}")

        if accum.should_step():
            optim.step()
            optim_steps_done += 1

            train_losses.append(train_loss)

            accum_loss /= training_config.gradient_accumulation_steps

            postfix = {"train_loss": f"{accum_loss:.4f}"}
            if last_val_loss is not None:
                postfix["val_loss"] = f"{last_val_loss:.4f}"
            bar.set_postfix(postfix, refresh=False)

            accum_loss = 0.0
            total_steps += 1

        # Validation every eval_every steps
        if step % training_config.eval_every == 0:
            ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)
            tt_model.eval()

            val_X, val_y, val_logits_mask, val_logits_add_mask, val_loss_scaler = next(val_batch_generator)
            val_logits = tt_model(val_X, causal_mask)

            # Apply mask to validation logits
            val_masked_logits = adjust_logits(val_logits, val_logits_mask, val_logits_add_mask)

            # Compute validation loss
            val_loss = loss_fn(val_masked_logits, val_y, reduce)
            val_loss = val_loss * val_loss_scaler
            val_loss = ttml.ops.unary.mean(val_loss)
            val_losses.append(val_loss.to_numpy().item())
            last_val_loss = val_loss.to_numpy().item()

            print("Validation check")
            print("====================================")
            print(f"Question: {testing_data[-1]['question']}")
            print("====================================")

            generate_text_tt(
                tt_model,
                tokenizer,
                testing_data[-1]["question"],
                max_sequence_length,
                causal_mask,
                0.7,
                logits_mask_tensor,
            )

            print("\n====================================")

            ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            tt_model.train()

        if training_config.save_every > 0 and (step % training_config.save_every == 0):
            ckpt_path = tt_serialization._save_model_npz(
                tt_model, training_config.checkpoint_dir, step=step, extra={"optim_steps_done": optim_steps_done}
            )
            print(f"[Checkpoint] Saved model to {ckpt_path}")

        if total_steps >= training_config.steps:
            break

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

    # Test generation
    print("\nTesting text generation...")
    test_question = testing_data[-1]["question"]
    print("Test question:", test_question)

    # Generate with TinyLlama for comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Tinyllama (HuggingFace) Generation")
    print("=" * 60)
    gpt2_answer = generate_text_tinyllama(tokenizer, test_question, max_gen_tokens=100, temperature=0.0)

    print(gpt2_answer)

    # Generate with fine-tuned TT-Metal model
    print("\n" + "=" * 60)
    print("FINE-TUNED: TT-Metal Model Generation")
    print("=" * 60)
    print(test_question, end="", flush=True)
    generate_text_tt(tt_model, tokenizer, test_question, max_sequence_length, causal_mask, 0.0, logits_mask_tensor, 100)

    print("\n" + "=" * 60)
    print("EXPECTED ANSWER:")
    print("=" * 60)
    print(testing_data[-1]["answer"])


if __name__ == "__main__":
    main()
