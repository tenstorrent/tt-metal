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
import datasets
import time
from typing import List, TypeAlias, Any

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

# Constants not changeable in the yaml
tile_size: int = 32
seed = 42

system_prompt = """You are a careful math tutor solving grade-school word problems.
Show your reasoning step by step.
End with exactly one final line in this format:
#### <number>
Do not output any text after the #### line."""

Tokens: TypeAlias = List[int]
Completion: TypeAlias = List[int]
Completions: TypeAlias = List[List[int]]
Reward: TypeAlias = float


def round_to_tile(x: int) -> int:
    return ((x + tile_size - 1) // tile_size) * tile_size


def iter_pass(total: int, chunk: int):
    full, rem = divmod(total, chunk)
    for _ in range(full):
        yield chunk
    if rem:
        yield rem


def get_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def debug_print_prompt_completion(
    prompt_tokens: Tokens, completion_tokens: Completion, idx: int | None = None
):
    prompt_text = tokenizer.decode(prompt_tokens)
    completion_text = tokenizer.decode(completion_tokens)

    prefix = f"[{idx}] " if idx is not None else ""
    print(f"{prefix}prompt_text: {prompt_text!r}")
    print(f"{prefix}completion_text: {completion_text!r}")


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
        try:
            if isinstance(t, ttml.autograd.Tensor):
                ttnn.deallocate(t.get_value(), force=True)
            elif isinstance(t, ttnn.Tensor):
                ttnn.deallocate(t, force=True)
        except Exception:
            print("pass:(")
            # best-effort
            pass


def tokenize_dataset(data, tokenizer: AutoTokenizer):
    """
    Tokenizes the questions and answers in the dataset using the provided tokenizer.

    data: dataset with "question" and "answer" fields
    tokenizer: HuggingFace tokenizer
    """
    X = [sample["question"] for sample in data]
    y = [sample["answer"] for sample in data]

    tok = lambda texts: tokenizer(texts, add_special_tokens=False)["input_ids"]
    return tok(X), tok(y)


def tokens_to_input_tensor(tokens, padded_n=None):
    batch_size, n = tokens.shape
    if padded_n is None:
        padded_n = round_to_tile(n)

    tokens = ttnn.pad(
        tokens, [(0, 0), (0, padded_n - n)], pad_token
    )  # assumes pad does a copy
    # tokens now has a shape [batch_size, padded_n]

    tokens = ttnn.reshape(tokens, (batch_size, 1, 1, padded_n))

    return tokens


def generate_causal_mask(query_len: int, processed_tokens: int, padded_n: int = None):
    """
    Builds the causal attention mask for decode.
    - Prefill mode (processed_tokens == 0): returns a padded lower-triangular mask
      of shape [1, 1, padded_n, padded_n] so token i can attend only to <= i.
    - Incremental mode (processed_tokens > 0, query_len == 1): returns one query row
      of shape [1, 1, padded_n, padded_n], where only the first row is active and
      can attend to all tokens seen so far (0..processed_tokens).
    Shapes are tile-padded to satisfy TT kernel layout requirements.
    """
    assert (query_len > 0 and processed_tokens == 0) or (
        query_len == 1 and processed_tokens > 0
    )

    if processed_tokens == 0:
        n = query_len
        if padded_n is None:
            padded_n = round_to_tile(n)

        m = np.zeros((padded_n, padded_n), dtype=np.float32)
        m[:n, :n] = np.tril(np.ones((n, n), dtype=np.float32))

        result = ttml.autograd.Tensor.from_numpy(
            m.reshape(1, 1, padded_n, padded_n),
            layout=ttnn.Layout.TILE,
            new_type=ttnn.DataType.BFLOAT16,
        )

        assert result.shape() == [1, 1, padded_n, padded_n]
        return result.get_value()
    else:
        n = processed_tokens + 1
        if padded_n is None:
            padded_n = round_to_tile(n)

        m = np.zeros((padded_n, padded_n), dtype=np.float32)
        m[0, :n] = 1

        result = ttml.autograd.Tensor.from_numpy(
            m.reshape(1, 1, padded_n, padded_n),
            layout=ttnn.Layout.TILE,
            new_type=ttnn.DataType.BFLOAT16,
        )

        assert result.shape() == [1, 1, padded_n, padded_n]
        return result.get_value()


class DecodeState:
    def _init_tokens(self, prompt_tokens: List[int], batch_size: int):
        from torch import tensor, int32

        tokens_torch = tensor(prompt_tokens, dtype=int32)  # shape: (len(prompt_tokens))

        tokens = ttnn.from_torch(
            tokens_torch,
            layout=ttnn.Layout.TILE,
            dtype=ttnn.DataType.UINT32,
            device=get_device(),
        )  # shape: (len(prompt_tokens))

        tokens = ttnn.repeat(
            tokens, (batch_size, 1)
        )  # [batch_size, len(prompt_tokens)]

        self.tokens = tokens

    def __init__(self, prompt_tokens: List[int], sample_seed, batch_size=1):
        head_dim = embedding_dim // num_heads
        cfg = ttml.models.KvCacheConfig(
            num_layers, batch_size, num_groups, max_sequence_length, head_dim
        )
        self.kv_cache = ttml.models.KvCache(cfg)
        self.kv_cache.reset()

        self._init_tokens(prompt_tokens, batch_size)
        self.tokens_last_column = None

        # The token matrix is padded to the same size every time to reduce number of compilations
        self.pad_len = round_to_tile(len(prompt_tokens) + max_tokens_to_complete)

        self.batch_size = batch_size
        self.prefilled = False
        self.sample_seed = sample_seed


def complete_token(state: DecodeState) -> ttnn.Tensor:
    # step_tokens_np are tokens we are about to compute on in this call of complete_token.
    if not state.prefilled:
        step_tokens = (
            state.tokens
        )  # shape: [batch_size, len(prompt_tokens)], where prompt_tokens is from DecodeState constructor
        state.prefilled = True

        processed_tokens = 0
        step_tokens_len = state.tokens.shape[1]
    else:
        step_tokens = state.tokens_last_column  # shape: [batch_size, 1]

        processed_tokens = state.tokens.shape[1] - 1
        step_tokens_len = 1

    input_tensor = tokens_to_input_tensor(
        step_tokens, padded_n=state.pad_len
    )  # [batch_size, 1, 1, padded_n]
    mask_tensor = generate_causal_mask(
        step_tokens_len, processed_tokens, padded_n=state.pad_len
    )  # [1, 1, padded_n, padded_n]

    logits = tt_model(
        input_tensor, mask_tensor, kv_cache=state.kv_cache, new_tokens=step_tokens_len
    ).get_value()

    n, m, k, V = logits.shape()
    assert n == state.batch_size and m == 1 and k == state.pad_len

    sliced = ttnn.slice(
        logits,
        [0, 0, step_tokens_len - 1, 0],
        [state.batch_size, 1, step_tokens_len, V],
    )

    assert sliced.shape == [state.batch_size, 1, 1, V]

    last_logits_ttml = ttml.autograd.Tensor(sliced, False)  # no grad
    next_tokens_ttml = sample_tokens(
        last_logits_ttml, state.sample_seed
    )  # shape: [B, 1]

    state.tokens_last_column = next_tokens_ttml
    state.tokens = ttnn.concat((state.tokens, next_tokens_ttml), dim=1)

    state.sample_seed += state.batch_size

    deallocate_tensors([input_tensor, mask_tensor, logits, last_logits_ttml])

    return next_tokens_ttml


def sample_tokens(logits_ttml, sample_seed: int):
    B, m, k, V = logits_ttml.shape()
    assert m == 1 and k == 1
    assert V == full_vocab_size
    if temperature < 0.01:
        # Greedy for all rows in one call
        ids_tt = ttnn.argmax(logits_ttml.get_value(), dim=3, keepdim=True)  # [B,1,1,1]
        ids_tt = ttnn.reshape(ids_tt, (B, 1))
    else:
        # Stochastic for all rows in one call
        ids_tt = ttml.ops.sample.sample_op(
            logits_ttml, temperature, sample_seed, None
        )  # [B,1,1,1]
        ids_tt = ids_tt.get_value()  # ttnn tensor now
        ids_tt = ttnn.reshape(ids_tt, (B, 1))

    return ids_tt  # [B, 1]


def complete_tokens(
    input_tokens: List[int], sample_seed: int, batch_size: int = 1
) -> Completions:
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    for tok in ["<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0:
            stop_ids.add(int(tid))

    state = DecodeState(input_tokens, sample_seed, batch_size=batch_size)

    tt_model.eval()

    with no_grad():
        for _ in range(max_tokens_to_complete):
            complete_token(state)

    return state.tokens_np

    # with no_grad():
    #     for _ in range(max_tokens_to_complete):
    #         token = complete_token(state)
    #         if token in stop_ids:
    #             break

    #     return state.tokens[len(input_tokens) :]


def compute_nlog_probs(inputs_np, targets_np, B, T) -> Any:
    """
    Takes np.arrays 'inputs_np', 'targets_np', returns a ttml tensor 'tokens_nlog', where
    for every i, j \in [0, B-1]x[0, T-2]
    tokens_nlog[i,j] = -log(prob(token[i,j])), where
    token[i,j] = vocab[targets_np[i,j]]
    """
    assert inputs_np.shape == (B, T - 1)
    assert targets_np.shape == (B, T - 1)

    X = inputs_np.astype(np.uint32)

    X_tt = ttml.autograd.Tensor.from_numpy(
        X.reshape(B, 1, 1, T - 1),
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    mask_tensor = generate_causal_mask(T - 1, 0)  # [1, 1, T-1, T-1]
    logits = tt_model(X_tt, mask_tensor)  # [B, 1, T-1, V]

    targets_tt = ttml.autograd.Tensor.from_numpy(
        targets_np,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    tokens_nlog = ttml.ops.loss.cross_entropy_loss(
        logits, targets_tt, ttml.ops.ReduceType.NONE
    )

    tokens_nlog = ttml.ops.reshape.reshape(tokens_nlog, [B, T - 1])

    assert tokens_nlog.shape() == [B, T - 1]
    return tokens_nlog


def generate_sequences(prompt: Tokens, completions: Completions, start, B, T):
    batch_completions = completions[start : start + B]
    sequences = [prompt + c for c in batch_completions]

    sequences_np = np.full((B, T), pad_token, dtype=np.int32)
    lengths_np = np.zeros((B,), dtype=np.int32)

    for i, seq in enumerate(sequences):
        sequences_np[i, : len(seq)] = np.asarray(seq, dtype=np.int32)
        lengths_np[i] = len(seq)

    assert sequences_np.shape == (B, T)
    assert lengths_np.shape == (B,)
    return sequences_np, lengths_np


def generate_inputs_targets(sequences_np):
    """Prefix of inputs = <<state>>, target = <<action>>."""
    inputs_np = sequences_np[:, :-1]
    targets_np = sequences_np[:, 1:]

    return inputs_np, targets_np


def ignore_probs(probs_tt, l_np, r_np, B, T):
    """
    Takes a ttml tensor probs_tt of shape [B, T-1].
    For every i in [0, B-1], and j in [0, T-1], such that j < l_np[i] or j > r_np[i],
    set 0 on probabilities probs_tt[i][j].
    Doesn't modify the probs_tt itself, returns a new tensor.
    """
    assert l_np.shape == (B,) and r_np.shape == (B,)
    assert probs_tt.shape() == [B, T - 1]

    # Build keep-mask on host (fast/simple), then apply with TTML mul
    j = np.arange(T - 1, dtype=np.int32)[None, :]  # [1, T-1]
    l = l_np.astype(np.int32).reshape(B, 1)  # [B, 1]
    r = r_np.astype(np.int32).reshape(B, 1)  # [B, 1]

    keep_np = ((j >= l) & (j <= r)).astype(np.float32)  # [B, T-1], 1 inside, 0 outside

    keep_tt = ttml.autograd.Tensor.from_numpy(
        keep_np,
        layout=ttnn.Layout.TILE,
        new_type=ttnn.DataType.BFLOAT16,
    )

    # zero-out outside [l, r] per row
    result = ttml.ops.binary.mul(probs_tt, keep_tt)
    assert result.shape() == [B, T - 1]

    return result


def calculate_loss(
    nlog_probs_tt, advantages_np, lengths_np, prompt_len: int, B: int, T: int
):
    """
    Loss formula:
    L = (1/B) * sum[i=1 to B] (A_i * 1/C_i * sum[j=1 to T-1] nlog_probs[i][j]),
    where A_i = advantages_tt[i]
    C_i = max(1, lengths_np[i] - prompt_len)
    C_i prevents over-scaling gradients for longer completions.

    Multiplying and dividing by T-1
    L = 1/[B(T-1)] * sum[i=1 to B] (A_i * [T-1]/C_i * sum[j=1 to T-1] nlog_probs[i][j])
    comp_lens_np[i] = C_i
    row_scale_np[i] = (T-1)/C_i

    L = 1/[B(T-1)](sum[i=1 to B] (A_i * row_scale_np[i] * sum[j=1 to T-1] nlog_probs[i][j]))
    L = 1/[B(T-1)](sum[i=1 to B] sum[j=1 to T-1] (A_i * row_scale_np[i] * nlog_probs[i][j]))
    L = mean[i,j] (A_i * row_scale_np[i] * nlog_probs[i][j])
    L = mean[i,j] (advantages_scaled[i] * nlog_probs[i][j])
    """
    assert nlog_probs_tt.shape() == [B, T - 1]
    assert advantages_np.shape == (B,)

    # completion token counts per row (avoid divide-by-zero)
    comp_lens_np = np.maximum(
        lengths_np.astype(np.float32) - float(prompt_len), 1.0
    )  # [B]

    row_scale_np = (float(T - 1) / comp_lens_np).reshape(B, 1)  # [B,1]
    row_scale_2d_np = np.repeat(row_scale_np, T - 1, axis=1).astype(
        np.float32
    )  # [B, T-1]

    row_scale_2d_tt = ttml.autograd.Tensor.from_numpy(
        row_scale_2d_np,
        layout=ttnn.Layout.TILE,
        new_type=ttnn.DataType.BFLOAT16,
    )  # [B, T-1]

    advantages_np_reshaped = advantages_np.reshape(B, 1)  # [B,1]
    advantages_2d_np = np.repeat(advantages_np_reshaped, T - 1, axis=1).astype(
        np.float32
    )  # [B, T-1]

    advantages_2d_tt = ttml.autograd.Tensor.from_numpy(
        advantages_2d_np,
        layout=ttnn.Layout.TILE,
        new_type=ttnn.DataType.BFLOAT16,
    )  # [B, T-1]

    advantages_scaled_2d_tt = ttml.ops.binary.mul(
        advantages_2d_tt, row_scale_2d_tt
    )  # [B, T-1]
    weighted_tt = ttml.ops.binary.mul(nlog_probs_tt, advantages_scaled_2d_tt)  # [B,T-1]
    weighted_tt_4d = ttml.ops.reshape.reshape(
        weighted_tt, [B, 1, T - 1, 1]
    )  # otherwise loss.backward() doesn't work

    result = ttml.ops.unary.mean(weighted_tt_4d)
    assert result.shape() == [1, 1, 1, 1]

    return result


def get_reward(c: Completion, golden: Tokens) -> Reward:
    return 0.0


def train_grpo():
    print("Loading dataset...")
    train_data = datasets.load_dataset("openai/gsm8k", "main", split="train")
    X, y = tokenize_dataset(train_data, tokenizer)
    print("Loaded the dataset!")

    for step in range(min(max_steps, len(X))):
        global temperature

        if step < 150:
            temperature = 1.1
        elif step < 400:
            temperature = 0.8
        else:
            temperature = 0.6

        print(f"{step=}, {temperature=}")
        start_time = time.perf_counter()
        prompt: Tokens = X[step]

        # -------------------------
        # PHASE 1: sample + rewards
        # -------------------------
        completions = []
        rewards = []

        for i in range(group_size):
            c: Completion = complete_tokens(
                prompt, sample_seed=seed + step * 1000000 + i * 1000
            )
            r = get_reward(c, golden=y[step])
            completions.append(c)
            rewards.append(r)

            debug_print_prompt_completion(prompt, c)

        rewards_np = np.asarray(rewards, dtype=np.float32)
        print(f"{step=}, {rewards_np.mean()=}")
        advantages_np = rewards_np - rewards_np.mean()

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"Phase 1 done! Elapsed time: {elapsed_ms:.2f} ms")

        # ------------------------------------
        # PHASE 2: differentiable policy update
        # ------------------------------------

        tt_model.train()
        optimizer.zero_grad()
        start = 0
        for pass_size in iter_pass(group_size, 4):
            print(f"Pass, {start=}, {pass_size=}")

            B = pass_size

            ## 1. Computing probabilities for all tokens in the pass sequences together.

            # Requirements for T:
            # T >= sequence_length for all sequences
            # (T-1) is divisible by tile_size (limitation of the tt_model call)
            T = max(len(prompt) + len(c) for c in completions)
            T = round_to_tile(T - 1) + 1

            # sequences is of shape BxT, length is of shape (B,)
            sequences_np, lengths_np = generate_sequences(
                prompt, completions, start, B, T
            )

            # shape of inputs_np, and targets_np is (B, T-1)
            inputs_np, targets_np = generate_inputs_targets(sequences_np)

            # shape of nlog_probs is (B, T-1)
            nlog_probs = compute_nlog_probs(inputs_np, targets_np, B, T)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"nlog_probs computed! Elapsed time: {elapsed_ms:.2f} ms")

            ## 2. Ignoring padded tokens and tokens from the prompt

            l_np = np.full((B,), len(prompt) - 1, dtype=np.uint32)
            r_np = lengths_np - 2
            nlog_probs = ignore_probs(
                nlog_probs, l_np, r_np, B, T
            )  # shape of nlog_probs still (B, T - 1)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"ignore_probs done! Elapsed time: {elapsed_ms:.2f} ms")

            ## 3. Calculate the loss for all sequences in the pass
            advantages_pass = advantages_np[start : start + B]
            loss = calculate_loss(
                nlog_probs, advantages_pass, lengths_np, len(prompt), B, T
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"Loss computed! Elapsed time: {elapsed_ms:.2f} ms")

            ## 4. Backward pass. Gradients accumulate over passes
            loss.backward(retain_graph=False)

            start += B

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            print(f"Pass done! Elapsed time: {elapsed_ms:.2f} ms")

        optimizer.step()


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


def inference_example():
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    for tok in ["<|eot_id|>", "<|end_of_text|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0:
            stop_ids.add(int(tid))

    start_time = time.perf_counter()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        },
    ]
    # Build prompt tokens using tokenizer chat template.
    # add_generation_prompt=True appends assistant header, so model continues as assistant.
    input_tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    completed_tokens_np = complete_tokens(input_tokens, sample_seed=seed, batch_size=4)

    print(f"Tokens completed! Time: {time.perf_counter() - start_time} s")

    for i in range(completed_tokens_np.shape[0]):
        tokens_row = completed_tokens_np[i]
        stop = len(tokens_row)
        for j in range(len(input_tokens), len(tokens_row)):
            if tokens_row[j] in stop_ids:
                stop = j
                break

        text = tokenizer.decode(tokens_row.tolist()[:stop], skip_special_tokens=False)
        print(f"{i=}")
        print(text)

    # print(f"{len(completed_tokens)=}")
    # print("Prompt + Generated = ")
    # print(tokenizer.decode(input_tokens + completed_tokens))

    print(f"Inference time: {(time.perf_counter() - start_time)} s")


def setup_tokenizer():
    global tokenizer, full_vocab_size, pad_token

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    full_vocab_size = len(tokenizer)
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id


if __name__ == "__main__":
    start_time = time.perf_counter()

    setup_tokenizer()

    set_seed(42)
    yaml_config, model_config = load_configs()

    initialize_device(yaml_config)

    tt_model = create_model(model_config)

    print(f"Init time: {time.perf_counter() - start_time} s")

    # Uncomment this line to try a sample inference
    inference_example()

    optimizer = create_optimizer(tt_model, yaml_config)
    # train_grpo()
