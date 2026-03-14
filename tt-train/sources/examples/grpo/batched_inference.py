#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GRPO training
This script takes the GSM8K dataset and trains the llama_3_2_1b model to respond with a short first sentence. It uses the GRPO reinforcement learning mechanism.

Here are the main elements of each training step:
* Take 1 prompt from the dataset, generate a group of completions for the same prompt. (complete_tokens)

* Compute rewards for those completions, take a mean across the group and compute advantages.
   The advantage of a completion roughly means: "How well did the completion do to achieve a better reward?".
   Larger advantage means better completion. If the temperature is large enough, the completions and rewards in the group will differ.

So far the first few steps are done in the evaluation mode, without computing the graphs or doing backpropagation.
In the next steps, the model is called in the training mode

* Split the completed tokens into different 'passes'
* Take all the completed tokens of the pass and compute their probabilities (compute_probs).
  Each row of the sequences_np matrix corresponds to 1 completion.
* Ignore probabilities of tokens from the original prompt and tokens that were padded. (ignore_probs)
* Use the probabilities to compute the GRPO 'loss' formula. (compute_loss)
* Take loss.backward() in each pass
* optimizer.step() after all passes are done

The completions are done sequentially (inefficient) and the probabilities are computed in a batch (more efficient).
"""

from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import (
    initialize_device,
    set_seed,
    create_optimizer,
    get_tt_metal_home,
    no_grad,
)
from ttml.common.config import load_config
import ttnn
import ttml
import numpy as np
from datasets import Dataset, load_dataset
import time
from typing import List, TypeAlias, Tuple

CONFIG = "training_grpo_unsloth_llama_3_2_1b_instruct.yaml"
HF_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
LOAD_PRETRAINED = True

# Set when the training & model .yaml files are parsed
temperature: float = None
max_tokens_to_complete: int = None
num_layers: int = None
num_groups: int = None
embedding_dim: int = None
num_heads: int = None
max_sequence_length: int = None
group_size: int = None
max_steps: int = None

# Set in main
optimizer = None
tokenizer = None
full_vocab_size: int = None
pad_token = None
tt_model = None


transformer_config = None

# Constants not changeable in the yaml
tile_size: int = 32
seed = 42

SYSTEM_PROMPT = """You are a careful math tutor solving grade-school word problems.
Show your reasoning step by step.
End with exactly one final line in this format:
#### <number>
Do not output any text after the #### line."""

Token: TypeAlias = int
Answer: TypeAlias = str
Prompt: TypeAlias = List[int]
Completion: TypeAlias = List[int]
Reward: TypeAlias = float


def round_up(x: int) -> int:
    return ((x + tile_size - 1) // tile_size) * tile_size


def deallocate_tensors(tensors) -> None:
    """
    Deallocate both TTML autograd tensors and raw TTNN tensors.
    """
    if tensors is None:
        return
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, ttml.autograd.Tensor):
            ttnn.deallocate(t.get_value(), force=True)
        elif isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t, force=True)


def get_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def debug_print_prompt_completion(
    prompt_tokens: List[Token], completion_tokens: Completion, idx: int | None = None
):
    prompt_text = tokenizer.decode(prompt_tokens)
    completion_text = tokenizer.decode(completion_tokens)

    prefix = f"[{idx}] " if idx is not None else ""
    print(f"{prefix}prompt_text: {prompt_text!r}")
    print(f"{prefix}completion_text: {completion_text!r}")


def load_configs():
    global temperature, max_tokens_to_complete
    global num_layers, num_groups, embedding_dim, num_heads, max_sequence_length
    global tile_size, group_size
    global max_steps

    yaml_config = load_config(
        CONFIG, f"{get_tt_metal_home()}/tt-train/configs/training_configs"
    )

    # training_config -> model_config path
    model_config = load_config(yaml_config["training_config"]["model_config"])

    max_steps = int(yaml_config["training_config"]["max_steps"])
    temperature = float(yaml_config["eval_config"]["temperature"])

    # GRPO runtime knobs from training yaml
    grpo_cfg = yaml_config.get("grpo_config", {})
    max_tokens_to_complete = int(grpo_cfg["max_tokens_to_complete"])
    group_size = int(grpo_cfg["group_size"])

    # model architecture knobs from model yaml
    tcfg = model_config.get("transformer_config", {})
    num_layers = int(tcfg["num_blocks"])
    num_groups = int(tcfg["num_groups"])
    embedding_dim = int(tcfg["embedding_dim"])
    num_heads = int(tcfg["num_heads"])
    max_sequence_length = int(tcfg["max_sequence_length"])

    return yaml_config, model_config


def create_model(model_config):
    print("Setting up model...")

    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = len(tokenizer)
    print("Created Model Factory")

    global transformer_config
    transformer_config = tt_model_factory.transformer_config

    print("Creating model...")
    tt_model = tt_model_factory.create_model()

    if LOAD_PRETRAINED:
        model_repo_path = snapshot_download(
            repo_id=HF_MODEL_ID,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
        print(f"Model snapshot path: {model_repo_path}")
        print("Loading from safetensors...")
        tt_model.load_from_safetensors(model_repo_path)

    return tt_model


def setup_tokenizer():
    global tokenizer, full_vocab_size, pad_token

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    full_vocab_size = len(tokenizer)
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id


def extract_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k(split="train") -> Tuple[List[Prompt], List[Answer]]:
    data = load_dataset("openai/gsm8k", "main")[split]
    dataset = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_answer(x["answer"]),
        }
    )

    questions = [ex["question"] for ex in dataset]
    answers = [ex["answer"] for ex in dataset]

    prompts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in questions
    ]

    return prompts, answers


def create_causal_mask(prompt_len: int, query_len: int) -> ttml.autograd.Tensor:
    whole_len = prompt_len + query_len
    padded_q = round_up(query_len)
    padded_w = round_up(whole_len)

    mask = np.zeros((padded_q, padded_w), dtype=np.float32)
    mask[:query_len, :padded_w] = np.tri(
        query_len, padded_w, k=prompt_len, dtype=np.float32
    )

    return ttml.autograd.Tensor.from_numpy(
        mask.reshape(1, 1, padded_q, padded_w),
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
    )


def tokens_to_tensor(tokens_np, B) -> ttml.autograd.Tensor:
    # tokens_np is of shape (B, N) or (B, 1)
    padded_len = round_up(tokens_np.shape[1])

    padded = np.full((B, padded_len), pad_token, dtype=np.uint32)
    padded[:, : tokens_np.shape[1]] = tokens_np

    return ttml.autograd.Tensor.from_numpy(
        padded.reshape(B, 1, 1, padded_len), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
    )


def build_logits_mask(vocab_size: int, padded_vocab_size: int) -> ttml.autograd.Tensor:
    logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
    logits_mask[:, :, :, vocab_size:] = 1e4

    return ttml.autograd.Tensor.from_numpy(
        logits_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
    )


def get_stop_ids():
    stop_ids = set()
    # Core stops
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    if tokenizer.pad_token_id is not None:
        stop_ids.add(int(tokenizer.pad_token_id))
    # Common Llama/chat terminators
    for tok in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            stop_ids.add(int(tid))

    return stop_ids


def completion_batched(
    prompt_tokens: List[int], transformer_config, sample_seed=42, batch_size=1
):
    ttml.autograd.AutoContext.get_instance().set_gradient_mode(
        ttml.autograd.GradMode.DISABLED
    )
    tt_model.eval()

    B = batch_size
    N = len(prompt_tokens)
    V = len(tokenizer)
    padded_V = round_up(V)

    head_dim = getattr(transformer_config, "head_dim", None) or (
        transformer_config.embedding_dim // transformer_config.num_heads
    )
    kv_cache = ttml.models.KvCache(
        transformer_config.num_blocks,
        B,
        transformer_config.num_groups,
        transformer_config.max_sequence_length,
        head_dim,
    )
    kv_cache.reset()

    prompt_tokens_np = np.tile(prompt_tokens, (B, 1))
    logits_mask_tensor = build_logits_mask(V, padded_V) if padded_V != V else None

    tokens_to_complete = min(
        max_tokens_to_complete,
        transformer_config.max_sequence_length - len(prompt_tokens),
    )

    generated = []

    for token in range(tokens_to_complete):
        if kv_cache.get_cache_position() == 0:
            processed = 0
            new_tokens = prompt_tokens_np.shape[1]
            token_tensor = tokens_to_tensor(prompt_tokens_np, B)
        else:
            processed = N - 1
            new_tokens = 1
            # last_token_column has shape [B, 1, 1, 1]
            token_tensor = ttnn.pad(
                last_token_column,
                [(0, 0), (0, 0), (0, 0), (0, tile_size - 1)],
                pad_token,
            )
            token_tensor = ttml.autograd.Tensor(token_tensor, False)

        mask = create_causal_mask(processed, new_tokens)
        logits = tt_model(token_tensor, mask, kv_cache=kv_cache, new_tokens=new_tokens)

        next_token_tensor = ttml.ops.sample.sample_op(
            logits, temperature, np.random.randint(low=1e7), logits_mask_tensor
        )

        last_token_column = ttnn.slice(
            next_token_tensor.get_value(),
            [0, 0, new_tokens - 1, 0],
            [B, 1, new_tokens, 1],
        )  # B 1 1 1

        generated.append(last_token_column)

        N += 1

        deallocate_tensors([token_tensor, mask, logits, next_token_tensor])

        print(f"{token=}")

    completions_np = np.empty((B, tokens_to_complete), dtype=np.int32)
    for j, column in enumerate(generated):
        completions_np[:, j] = column.to_numpy().reshape(
            B,
        )

    stop_ids = get_stop_ids()

    print("completions start:")

    completions = []
    for i in range(B):
        to = max_tokens_to_complete
        for j, token in enumerate(completions_np[i]):
            if token in stop_ids:
                to = j
                break

        completions.append(completions_np[i, :to])

    return completions


def generate_answers(prompt_str: str) -> List[Answer]:
    prompt = tokenizer.encode(prompt_str)

    import time

    start_time = time.time()

    completions = completion_batched(prompt, transformer_config, batch_size=4)

    elapsed_time = time.time() - start_time

    print(f"Tokens generated in {elapsed_time} s")

    for i in range(4):
        output = tokenizer.decode(completions[i], skip_special_tokens=False)
        print(f"{i=}, {output=}")

    pass


if __name__ == "__main__":
    start_time = time.perf_counter()

    setup_tokenizer()

    set_seed(42)
    yaml_config, model_config = load_configs()

    initialize_device(yaml_config)

    tt_model = create_model(model_config)

    print(f"Init time: {time.perf_counter() - start_time} s")

    prompts, answers = get_gsm8k()
    print(prompts[0], answers[0])

    generate_answers(prompts[0])
