#!/usr/bin/env python3
"""
GSM8K Fine-tuning Script
Fine-tunes a GPT-2 model on the GSM8K math word problems dataset using TT-Metal.
"""

import os
import sys
import datasets
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from matplotlib import pyplot as plt
from tqdm import tqdm

# Add TT-Metal path
sys.path.append(f"{os.environ['TT_METAL_HOME']}/tt-train/sources/ttml")
import ttml
from ttml.common.config import get_config, TransformerConfig, TrainingConfig
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import set_seed, round_up_to_tile, create_optimizer
from ttml.common.data import get_batch, build_causal_mask
from ttml.common.trainer import train

# Configuration
CONFIG = "training_shakespeare_gpt2s.yaml"
BATCH_SIZE = 4


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


def get_batch_generator(data, batch_size, max_sequence_length, tokenizer):
    """Custom data generator for GSM8K dataset."""

    def get_batch(data, batch_size=32):
        curr_idx = 0

        while curr_idx < len(data):
            X = data[curr_idx : min(curr_idx + batch_size, len(data))]["question"]
            Y = data[curr_idx : min(curr_idx + batch_size, len(data))]["answer"]

            data_np = np.empty((batch_size, max_sequence_length), dtype=np.uint32)
            mask_lens = []

            for i, (x_str, y_str) in enumerate(zip(X, Y)):
                # Tokenize question and answer separately
                x_tokens = tokenizer(x_str, return_tensors="np")["input_ids"].flatten()
                y_tokens = tokenizer(y_str, return_tensors="np")["input_ids"].flatten()

                # Concatenate question + answer
                data_point = np.concatenate([x_tokens, y_tokens])
                mask_lens.append(len(x_tokens))  # Length of question (to mask in loss)

                # Pad or truncate to max_sequence_length
                if len(data_point) > max_sequence_length:
                    data_point = data_point[:max_sequence_length]
                    # Adjust mask_len if question was truncated
                    if mask_lens[-1] > max_sequence_length:
                        mask_lens[-1] = max_sequence_length
                elif len(data_point) < max_sequence_length:
                    data_point = np.pad(
                        data_point, (0, max_sequence_length - len(data_point)), constant_values=tokenizer.eos_token_id
                    )

                data_np[i] = data_point.astype(np.uint32)

            # Shape: [batch_size, 1, 1, max_sequence_length]
            data_np = np.expand_dims(data_np, axis=(1, 2))
            X = ttml.autograd.Tensor.from_numpy(data_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32)

            curr_idx += batch_size
            yield (X, np.array(mask_lens))

    return get_batch(data, batch_size)


def generate_text(model, tokenizer, question, max_sequence_length, causal_mask, max_gen_tokens=100):
    """Generate text given a question."""
    model.eval()
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)

    # Tokenize and prepare input
    test_tokens = tokenizer(
        question, return_tensors="np", truncation=True, padding="max_length", max_length=max_sequence_length
    )["input_ids"].flatten()
    test_input_np = np.expand_dims(np.expand_dims(test_tokens.astype(np.uint32), axis=(0, 1)), axis=0)

    # Generate tokens autoregressively
    generated_tokens = []
    input_seq = test_input_np.copy()

    for _ in range(max_gen_tokens):
        input_tensor = ttml.autograd.Tensor.from_numpy(input_seq, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32)
        logits = model(input_tensor, causal_mask)
        next_token_logits = logits.to_numpy()[0, 0, len(generated_tokens) + test_tokens.shape[0] - 1]
        next_token = np.argmax(next_token_logits)  # Greedy decoding

        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)

        # Append the new token to the input sequence
        if len(generated_tokens) + test_tokens.shape[0] < max_sequence_length:
            input_seq[0, 0, 0, len(generated_tokens) + test_tokens.shape[0] - 1] = next_token
        else:
            break

    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.ENABLED)

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text


def main():
    print("Loading tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    yaml_config = get_config(CONFIG)

    # Download safetensors
    print("Downloading safetensors...")
    safetensors_path = hf_hub_download(repo_id="gpt2", filename="model.safetensors")
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

    # Split test data
    val_data = testing_data.select(range(400))
    testing_data = testing_data.select(range(400, len(testing_data)))

    # Setup training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_config = TrainingConfig(yaml_config)
    optim = create_optimizer(tt_model, yaml_config)
    causal_mask = build_causal_mask(max_sequence_length)

    causal_mask = ttml.autograd.Tensor.from_numpy(causal_mask, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.BFLOAT16)

    loss_fn = ttml.ops.loss.cross_entropy_loss
    reduce = ttml.ops.ReduceType.MEAN

    # Training setup
    tt_model.train()
    data_steps = (len(training_data) // BATCH_SIZE) + 1
    train_losses = []
    val_losses = []

    train_batch_generator = get_batch_generator(training_data, BATCH_SIZE, max_sequence_length, tokenizer)
    val_batch_generator = get_batch_generator(val_data, 4, max_sequence_length, tokenizer)

    accum = GradientAccumulator(training_config.gradient_accumulation_steps)
    print("Gradient Accumulation Steps:", training_config.gradient_accumulation_steps)
    tokens_per_batch = BATCH_SIZE * max_sequence_length
    optim_steps_done = 0

    print(f"Starting training for {data_steps} steps...")
    bar = tqdm(range(1, data_steps + 1))

    for step in bar:
        if accum.should_zero_grad():
            optim.zero_grad()

        X, mask_lens = next(train_batch_generator)

        # Forward pass: input is the concatenated sequence
        logits = tt_model(X, causal_mask)  # Shape: [batch, 1, seq_len, vocab_size]

        # Create targets: shift input by 1 position (standard causal LM)
        X_np = X.to_numpy()  # Shape: [batch, 1, 1, seq_len]
        targets_np = np.roll(X_np, -1, axis=-1)  # Shift left by 1
        targets_np[:, :, :, -1] = tokenizer.eos_token_id  # Last token target
        targets_np = targets_np.squeeze(axis=(1, 2))  # Shape: [batch, seq_len]

        targets = ttml.autograd.Tensor.from_numpy(targets_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32)

        # Create mask to zero out logits corresponding to question tokens
        logits_np = logits.to_numpy()  # Shape: [batch, 1, seq_len, vocab_size]
        logits_mask_np = np.ones_like(logits_np, dtype=np.float32)

        for i, mask_len in enumerate(mask_lens):
            # Mask out the question tokens (first mask_len tokens)
            logits_mask_np[i, :, :mask_len, :] = 0.0
            # Also mask padding tokens
            pad_positions = X_np[i, 0, 0, :] == tokenizer.eos_token_id
            logits_mask_np[i, :, pad_positions, :] = 0.0

        logits_mask = ttml.autograd.Tensor.from_numpy(
            logits_mask_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.BFLOAT16
        )

        # Apply mask to logits (zero out question token logits)
        masked_logits = logits * logits_mask

        # Compute cross-entropy loss on masked logits
        loss = loss_fn(masked_logits, targets, reduce)

        scaled_loss = accum.scale(loss)
        scaled_loss.backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()

        train_loss = float(loss.to_numpy())
        accum.update(train_loss, tokens_per_batch)

        if accum.should_step():
            optim.step()
            optim_steps_done += 1

        train_losses.append(train_loss)
        avg_loss = np.array(train_losses[max(0, step - 20) :]).mean()

        postfix = {"train_loss": f"{train_loss:.4f}", "avg_loss": f"{avg_loss:.4f}"}
        bar.set_postfix(postfix, refresh=False)

        # Validation every 100 steps
        if step % 100 == 0:
            ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)
            tt_model.eval()

            val_X, val_mask_lens = next(val_batch_generator)
            val_logits = tt_model(val_X, causal_mask)

            # Same target and masking logic for validation
            val_X_np = val_X.to_numpy()
            val_targets_np = np.roll(val_X_np, -1, axis=-1)
            val_targets_np[:, :, :, -1] = tokenizer.eos_token_id
            val_targets_np = val_targets_np.squeeze(axis=(1, 2))

            val_targets = ttml.autograd.Tensor.from_numpy(
                val_targets_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.UINT32
            )

            # Create logits mask for validation
            val_logits_np = val_logits.to_numpy()
            val_logits_mask_np = np.ones_like(val_logits_np, dtype=np.float32)

            for i, mask_len in enumerate(val_mask_lens):
                val_logits_mask_np[i, :, :mask_len, :] = 0.0
                pad_positions = val_X_np[i, 0, 0, :] == tokenizer.eos_token_id
                val_logits_mask_np[i, :, pad_positions, :] = 0.0

            val_logits_mask = ttml.autograd.Tensor.from_numpy(
                val_logits_mask_np, ttml.Layout.ROW_MAJOR, ttml.autograd.DataType.BFLOAT16
            )

            # Apply mask to validation logits
            val_masked_logits = val_logits * val_logits_mask

            # Compute validation loss
            val_loss = loss_fn(val_masked_logits, val_targets, reduce)
            val_losses.append(val_loss.to_numpy().item())

            ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.ENABLED)
            tt_model.train()

    print("Training completed!")

    # Plot training curves
    print("Plotting training curves...")
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(train_losses, color="blue", label="Train Loss")
    axs.plot(np.arange(0, len(val_losses)) * 100, val_losses, color="orange", label="Val Loss")
    axs.set_title("Training Loss")
    axs.set_xlabel("Steps")
    axs.set_ylabel("Loss")
    axs.legend()
    plt.savefig("training_curves.png")
    plt.show()

    # Test generation
    print("\nTesting text generation...")
    test_question = testing_data[0]["question"]
    print("Test question:", test_question)

    generated_answer = generate_text(tt_model, tokenizer, test_question, max_sequence_length, causal_mask)
    print("Generated answer:", generated_answer)

    print("Expected answer:", testing_data[0]["answer"])


if __name__ == "__main__":
    main()
