from transformers import AutoTokenizer, AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import initialize_device, set_seed
import ttnn
import ttml
import numpy as np
import datasets
import time
import re
from typing import List, TypeAlias, Any

CONFIG = "training_gsm8k_rl_llama.yaml"
HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
LOAD_PRETRAINED = True

from ttml.common.config import (
    load_config,
)

from ttml.common.utils import (
    create_optimizer,
    get_tt_metal_home,
    no_grad,
)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
vocab_size: int = tokenizer.vocab_size
temperature: float = 0.5
max_tokens_to_complete: int = 100
num_layers: int = 30
num_groups: int = 3
embedding_dim: int = 576
num_heads: int = 9
max_sequence_length: int = 768
seed = 42
tile_size: int = 32
group_size: int = 4
optimizer = None

Tokens: TypeAlias = List[int]
Completion: TypeAlias = List[int]
Completions: TypeAlias = List[List[int]]
Reward: TypeAlias = float

pad_token = tokenizer.pad_token_id
if pad_token is None:
    pad_token = tokenizer.eos_token_id


def round_to_tile(x: int) -> int:
    return ((x + tile_size - 1) // tile_size) * tile_size


def iter_pass(total: int, chunk: int):
    full, rem = divmod(total, chunk)
    for _ in range(full):
        yield chunk
    if rem:
        yield rem


def debug_print_prompt_completion(
    prompt_tokens: Tokens, completion_tokens: Completion, idx: int | None = None
):
    prompt_text = tokenizer.decode(prompt_tokens)
    completion_text = tokenizer.decode(completion_tokens)

    prefix = f"[{idx}] " if idx is not None else ""
    print(f"{prefix}prompt_text: {prompt_text!r}")
    print(f"{prefix}completion_text: {completion_text!r}")


def get_device():
    ctx = ttml.autograd.AutoContext.get_instance()
    return ctx.get_device()


def tokenize_dataset(data, tokenizer: AutoTokenizer):
    """
    Tokenizes the questions and answers in the dataset using the provided tokenizer.

    data: dataset with "question" and "answer" fields
    tokenizer: HuggingFace tokenizer
    """
    X = [sample["question"] for sample in data]
    y = [sample["answer"] for sample in data]

    tok = lambda texts: tokenizer(texts, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ]
    return tok(X), tok(y)


def tokens_to_model_tensor(tokens: List[int]):
    tokens_len = len(tokens)
    padded_len = round_to_tile(tokens_len)

    arr = np.full((padded_len,), pad_token, dtype=np.uint32)
    arr[:tokens_len] = np.asarray(tokens, dtype=np.uint32)

    t = ttml.autograd.Tensor.from_numpy(
        arr.reshape(1, 1, 1, padded_len),
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )

    return t


# Builds the causal attention mask for decode.
# - Prefill mode (processed_tokens == 0): returns a padded lower-triangular mask
#   of shape [1, 1, padded_n, padded_n] so token i can attend only to <= i.
# - Incremental mode (processed_tokens > 0, query_len == 1): returns one query row
#   of shape [1, 1, tile_size, padded_n], where only the first row is active and
#   can attend to all tokens seen so far (0..processed_tokens).
# Shapes are tile-padded to satisfy TT kernel layout requirements.
def generate_causal_mask(query_len: int, processed_tokens: int):
    assert (query_len > 0 and processed_tokens == 0) or (
        query_len == 1 and processed_tokens > 0
    )

    if processed_tokens == 0:
        n = query_len
        padded_n = round_to_tile(n)

        m = np.zeros((padded_n, padded_n), dtype=np.uint32)
        m[:n, :n] = np.tril(np.ones((n, n), dtype=np.uint32))

        result = ttml.autograd.Tensor.from_numpy(
            m.reshape(1, 1, padded_n, padded_n),
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.BFLOAT16,
        )

        assert result.shape() == [1, 1, padded_n, padded_n]
        return result
    else:
        n = processed_tokens + 1
        padded_n = round_to_tile(n)

        m = np.zeros((tile_size, padded_n), dtype=np.uint32)
        m[0, :n] = 1

        result = ttml.autograd.Tensor.from_numpy(
            m.reshape(1, 1, tile_size, padded_n),
            layout=ttnn.Layout.ROW_MAJOR,
            new_type=ttnn.DataType.BFLOAT16,
        )

        assert result.shape() == [1, 1, tile_size, padded_n]
        return result


class DecodeState:
    def __init__(self, tokens: List[int], sample_seed):
        head_dim = embedding_dim // num_heads
        cfg = ttml.models.KvCacheConfig(
            num_layers, 1, num_groups, max_sequence_length, head_dim
        )
        self.kv_cache = ttml.models.KvCache(cfg)
        self.kv_cache.reset()
        self.tokens = tokens[:]
        self.prefilled = False
        self.sample_seed = sample_seed


def complete_token(state: DecodeState) -> int:
    # step_tokens == tokens we are about to compute on in this call of complete_token.
    if not state.prefilled:
        step_tokens = state.tokens
        state.prefilled = True

        processed_tokens = 0
        step_tokens_len = len(step_tokens)
    else:
        step_tokens = [state.tokens[-1]]

        processed_tokens = len(state.tokens) - 1
        step_tokens_len = 1

    padded_step_len = round_to_tile(step_tokens_len)
    input_tensor = tokens_to_model_tensor(step_tokens)
    mask_tensor = generate_causal_mask(step_tokens_len, processed_tokens)

    logits = tt_model(
        input_tensor, mask_tensor, kv_cache=state.kv_cache, new_tokens=len(step_tokens)
    )

    n, m, k, V = logits.shape()
    assert n == 1 and m == 1 and k == padded_step_len

    sliced = ttnn.slice(
        logits.get_value(), [0, 0, step_tokens_len - 1, 0], [1, 1, step_tokens_len, V]
    )

    assert sliced.shape == [1, 1, 1, V]

    last_logits = ttml.autograd.Tensor(sliced, False)  # no grad

    next_token = sample_token(last_logits, state.sample_seed)
    state.tokens.append(next_token)
    state.sample_seed += 1

    return next_token


# Returns a token from raw logits
def sample_token(logits, sample_seed):
    n, m, k, V = logits.shape()
    assert n == 1 and m == 1 and k == 1

    if temperature < 0.01:
        argmax_result = ttnn.argmax(logits.get_value(), dim=3, keepdim=True)
        next_token = int(argmax_result.item())
        next_token = min(next_token, vocab_size - 1)
    else:
        sampled = ttml.ops.sample.sample_op(logits, temperature, sample_seed, None)
        next_token = int(sampled.get_value().item())

    return next_token


def complete_tokens(input_tokens: List[int], sample_seed: int) -> Completion:
    state = DecodeState(input_tokens, sample_seed)

    tt_model.eval()
    with no_grad():
        for _ in range(max_tokens_to_complete):
            token = complete_token(state)
            if token == tokenizer.eos_token_id:
                break

        return state.tokens[len(input_tokens) :]


# Takes np.arrays 'inputs_np', 'targets_np', returns a ttml tensor 'tokens_nlog', where
# for every i, j \in [0, B-1]x[0, T-2]
# tokens_nlog[i,j] = -log(prob(token[i,j])), where
# token[i,j] = vocab[targets_np[i,j]]
def compute_nlog_probs(inputs_np, targets_np, B, T) -> Any:
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


# Prefix of inputs = <<state>>
# Target = <<action>>
def generate_inputs_targets(sequences_np):
    inputs_np = sequences_np[:, :-1]
    targets_np = sequences_np[:, 1:]

    return inputs_np, targets_np


# Takes a ttml tensor probs_tt of shape [B, T-1].
# For every i in [0, B-1], and j in [0, T-1], such that j < l_np[i] or j > r_np[i],
# set 0 on probabilities probs_tt[i][j].
# Doesn't modify the probs_tt itself, returns a new tensor.
def ignore_probs(probs_tt, l_np, r_np, B, T):
    assert l_np.shape == (B,) and r_np.shape == (B,)
    assert probs_tt.shape() == [B, T - 1]

    # Build keep-mask on host (fast/simple), then apply with TTML mul
    j = np.arange(T - 1, dtype=np.int32)[None, :]  # [1, T-1]
    l = l_np.astype(np.int32).reshape(B, 1)  # [B, 1]
    r = r_np.astype(np.int32).reshape(B, 1)  # [B, 1]

    keep_np = ((j >= l) & (j <= r)).astype(np.float32)  # [B, T-1], 1 inside, 0 outside

    keep_tt = ttml.autograd.Tensor.from_numpy(
        keep_np,
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.BFLOAT16,
    )

    # zero-out outside [l, r] per row
    result = ttml.ops.binary.mul(probs_tt, keep_tt)
    assert result.shape() == [B, T - 1]

    return result


# Loss formula:
# L = (1/B) * sum[i=1 to B] (A_i * 1/C_i * sum[j=1 to T-1] nlog_probs[i][j]),
# where A_i = advantages_tt[i]
# C_i = max(1, lengths_np[i] - prompt_len)
# C_i prevents over-scaling gradients for longer completions.
#
# Multiplying and dividing by T-1
# L = 1/[B(T-1)] * sum[i=1 to B] (A_i * [T-1]/C_i * sum[j=1 to T-1] nlog_probs[i][j])
# comp_lens_np[i] = C_i
# row_scale_np[i] = (T-1)/C_i
#
# L = 1/[B(T-1)](sum[i=1 to B] (A_i * row_scale_np[i] * sum[j=1 to T-1] nlog_probs[i][j]))
# L = 1/[B(T-1)](sum[i=1 to B] sum[j=1 to T-1] (A_i * row_scale_np[i] * nlog_probs[i][j]))
# L = mean[i,j] (A_i * row_scale_np[i] * nlog_probs[i][j])
# L = mean[i,j] (advantages_scaled[i] * nlog_probs[i][j])


def calculate_loss(
    nlog_probs_tt, advantages_np, lengths_np, prompt_len: int, B: int, T: int
):
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
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.BFLOAT16,
    )  # [B, T-1]

    advantages_np_reshaped = advantages_np.reshape(B, 1)  # [B,1]
    advantages_2d_np = np.repeat(advantages_np_reshaped, T - 1, axis=1).astype(
        np.float32
    )  # [B, T-1]

    advantages_2d_tt = ttml.autograd.Tensor.from_numpy(
        advantages_2d_np,
        layout=ttnn.Layout.ROW_MAJOR,
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


# The reward range is [-1.0, 1.0]
# Reward meaning:
# -1.0  -> invalid/degenerate output (empty text or empty first sentence, e.g. starts with punctuation)
# (0,1] -> valid first sentence; higher is better (shorter first sentence)
#         ~1.0 means very short first sentence, values near 0 mean long first sentence
# Formula for valid first sentence: reward = exp(-first_len / tau), where first_len is word count.


def get_reward(c: Completion) -> Reward:
    text = tokenizer.decode(c, skip_special_tokens=True).strip()
    if not text:
        return -1.0
    first = re.split(r"[.!?]+", text, maxsplit=1)[0].strip()
    # Key fix: empty first sentence is bad
    if not first:
        return -1.0
    first_len = len(first.split())
    # Higher reward for shorter first sentence, in (0, 1]
    tau = 10.0
    return float(np.exp(-first_len / tau))


def train_gsm8k(max_steps: int = 1000):
    print("Loading GSM8K dataset...")
    train_data = datasets.load_dataset("openai/gsm8k", "main", split="train")
    X, _ = tokenize_dataset(train_data, tokenizer)
    print("Loaded GSM8K dataset!")

    for step in range(min(max_steps, len(X))):
        print(f"{step=}")
        start_time = time.perf_counter()
        prompt: Tokens = X[step].tolist()

        # -------------------------
        # PHASE 1: sample + rewards
        # -------------------------
        completions = []
        rewards = []

        for i in range(group_size):
            c: Completion = complete_tokens(
                prompt, sample_seed=seed + step * 1000000 + i * 1000
            )
            r = get_reward(c)
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


def load_model_config():
    yaml_config = load_config(
        CONFIG, f"{get_tt_metal_home()}/tt-train/configs/training_configs"
    )

    print(f"YAML config: {yaml_config}")
    model_config = load_config(yaml_config["training_config"]["model_config"])

    return model_config


def create_model(model_config):
    print("Setting up model...")
    orig_vocab_size = tokenizer.vocab_size

    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
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
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    input_tokens = tokenizer.encode(prompt)

    completed_tokens = complete_tokens(input_tokens, sample_seed=seed)
    print(f"{len(completed_tokens)=}")
    print("Prompt + Generated = ")
    print(tokenizer.decode(input_tokens + completed_tokens))


if __name__ == "__main__":
    set_seed(42)
    model_config = load_model_config()
    print(model_config)

    initialize_device(model_config)

    tt_model = create_model(model_config)

    # inference_example()

    optimizer = create_optimizer(tt_model, model_config)
    train_gsm8k(max_steps=100)
