from dataclasses import dataclass
from typing import List, TypeAlias, Tuple

import ttnn
import ttml
import numpy as np


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
    _B: int = None
    _N: int = None


def round_up(ctx: InferenceCtx, x: int) -> int:
    return ((x + ctx.tile_size - 1) // ctx.tile_size) * ctx.tile_size


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
    ctx: InferenceCtx,
    prompt_tokens: List[int],
    completion_tokens: List[int],
):
    prompt_text = ctx.tokenizer.decode(prompt_tokens)
    completion_text = ctx.tokenizer.decode(completion_tokens)

    print(f"prompt_text: {prompt_text!r}")
    print(f"completion_text: {completion_text!r}")


def create_causal_mask(
    ctx: InferenceCtx, prompt_len: int, query_len: int, pad_lengths: List[int]
) -> ttml.autograd.Tensor:
    B = ctx._B
    assert len(pad_lengths) == B

    whole_len = prompt_len + query_len
    padded_q = round_up(ctx, query_len)
    padded_w = round_up(ctx, whole_len)

    mask_one_token = np.zeros((padded_q, padded_w), dtype=np.float32)
    mask_one_token[:query_len, :padded_w] = np.tri(query_len, padded_w, k=prompt_len, dtype=np.float32)

    mask_3d = np.tile(mask_one_token, (B, 1, 1))
    for i in range(B):
        mask_3d[i, :, 0 : pad_lengths[i]] = 0  # don't attend to pad tokens

    mask_4d = mask_3d[:, np.newaxis, :, :]

    assert mask_4d.shape == (B, 1, padded_q, padded_w)

    return ttml.autograd.Tensor.from_numpy(
        mask_4d,
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
    )


def tokens_to_tensor(ctx: InferenceCtx, tokens_np, B) -> ttml.autograd.Tensor:
    # tokens_np is of shape (B, N) or (B, 1)
    padded_len = round_up(ctx, tokens_np.shape[1])

    padded = np.full((B, padded_len), ctx.pad_token, dtype=np.uint32)
    padded[:, : tokens_np.shape[1]] = tokens_np

    return ttml.autograd.Tensor.from_numpy(
        padded.reshape(B, 1, 1, padded_len), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
    )


def build_logits_mask(vocab_size: int, padded_vocab_size: int) -> ttml.autograd.Tensor:
    logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
    logits_mask[:, :, :, vocab_size:] = 1e4

    return ttml.autograd.Tensor.from_numpy(logits_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)


def get_stop_ids(ctx: InferenceCtx):
    stop_ids = set()
    # Core stops
    if ctx.tokenizer.eos_token_id is not None:
        stop_ids.add(int(ctx.tokenizer.eos_token_id))
    if ctx.tokenizer.pad_token_id is not None:
        stop_ids.add(int(ctx.tokenizer.pad_token_id))
    # Common Llama/chat terminators
    for tok in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]:
        tid = ctx.tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0 and tid != ctx.tokenizer.unk_token_id:
            stop_ids.add(int(tid))

    return stop_ids


def _completion_batched_impl(ctx: InferenceCtx, prompt_tokens_np, pad_lengths: List[int]):
    B, N = ctx._B, ctx._N
    assert prompt_tokens_np.shape == (B, N)
    assert len(pad_lengths) == B

    ttml.autograd.AutoContext.get_instance().set_gradient_mode(ttml.autograd.GradMode.DISABLED)
    ctx.tt_model.eval()

    V = len(ctx.tokenizer)
    padded_V = round_up(ctx, V)

    head_dim = getattr(ctx.transformer_config, "head_dim", None) or (
        ctx.transformer_config.embedding_dim // ctx.transformer_config.num_heads
    )
    kv_cache = ttml.models.KvCache(
        ctx.transformer_config.num_blocks,
        B,
        ctx.transformer_config.num_groups,
        ctx.transformer_config.max_sequence_length,
        head_dim,
    )
    kv_cache.reset()

    logits_mask_tensor = build_logits_mask(V, padded_V) if padded_V != V else None

    tokens_to_complete = min(
        ctx.max_tokens_to_complete,
        ctx.transformer_config.max_sequence_length - N,
    )

    generated = []

    for token in range(tokens_to_complete):
        if kv_cache.get_cache_position() == 0:
            processed = 0
            new_tokens = prompt_tokens_np.shape[1]
            token_tensor = tokens_to_tensor(ctx, prompt_tokens_np, B)
        else:
            processed = N - 1
            new_tokens = 1
            # last_token_column has shape [B, 1, 1, 1]
            token_tensor = ttnn.pad(
                last_token_column,
                [(0, 0), (0, 0), (0, 0), (0, ctx.tile_size - 1)],
                ctx.pad_token,
            )
            token_tensor = ttml.autograd.Tensor(token_tensor, False)

        mask = create_causal_mask(ctx, processed, new_tokens, pad_lengths)
        logits = ctx.tt_model(token_tensor, mask, kv_cache=kv_cache, new_tokens=new_tokens)

        next_token_tensor = ttml.ops.sample.sample_op(
            logits, ctx.temperature, np.random.randint(low=1e7), logits_mask_tensor
        )

        last_token_column = ttnn.slice(
            next_token_tensor.get_value(),
            [0, 0, new_tokens - 1, 0],
            [B, 1, new_tokens, 1],
        )  # B 1 1 1

        generated.append(last_token_column)

        N += 1

        deallocate_tensors([token_tensor, mask, logits, next_token_tensor])

    completions_np = np.empty((B, tokens_to_complete), dtype=np.int32)
    for j, column in enumerate(generated):
        completions_np[:, j] = column.to_numpy().reshape(
            B,
        )

    stop_ids = get_stop_ids(ctx)

    completions = []
    for i in range(B):
        to = ctx.max_tokens_to_complete
        for j, token in enumerate(completions_np[i]):
            if token in stop_ids:
                to = j
                break

        completions.append(completions_np[i, :to].tolist())

    return completions


def completion_batched_one_prompt(ctx: InferenceCtx, prompt_tokens: List[int]) -> List[List[int]]:
    B = ctx._B = ctx.group_size
    N = ctx._N = len(prompt_tokens)
    prompt_tokens_np = np.tile(prompt_tokens, (B, 1))

    pad_lengths = [0]  # no padding
    return _completion_batched_impl(ctx, prompt_tokens_np, pad_lengths)


def completion_batched_multiple_prompts(ctx: InferenceCtx, prompts: List[List[int]]) -> List[List[int]]:
    max_len = max(len(row) for row in prompts)
    pad_lengths = [max_len - len(row) for row in prompts]
    prompts_cnt = len(prompts)
    B = ctx._B = ctx.group_size * prompts_cnt
    N = ctx._N = max_len

    # add the pad_token to the left of the shorter prompts, so that all prompts end at the same column
    prompt_tokens_np = np.full((B, max_len), ctx.pad_token)
    for i, row in enumerate(prompts):
        prompt_tokens_np[i * ctx.group_size : (i + 1) * ctx.group_size, max_len - len(row) :] = np.asarray(row)

    return _completion_batched_impl(ctx, prompt_tokens_np, pad_lengths)


def generate_answers_one_prompt(ctx: InferenceCtx, prompt_str: str) -> List[str]:
    prompt = ctx.tokenizer.encode(prompt_str)

    completions = completion_batched_one_prompt(ctx, prompt)

    completions_strs = [ctx.tokenizer.decode(c, skip_special_tokens=False) for c in completions]

    return completions_strs


def generate_answers_multiple_prompts(ctx: InferenceCtx, prompt_strs: List[str]) -> List[str]:
    prompts = [ctx.tokenizer.encode(s) for s in prompt_strs]

    completions = completion_batched_multiple_prompts(ctx, prompts)

    completions_strs = [ctx.tokenizer.decode(c, skip_special_tokens=False) for c in completions]

    return completions_strs


def compute_nlog_probs(
    ctx: InferenceCtx,
    prompts: List[List[int]],
    completions: List[List[int]],
) -> tuple[ttml.autograd.Tensor, ttml.autograd.Tensor, int]:
    assert len(completions) == len(prompts) * ctx.group_size

    B = len(completions)
    ctx._B = B  # create_causal_mask() reads this

    row_prompts = [prompts[i // ctx.group_size] for i in range(B)]
    lengths = [len(p) + len(c) for p, c in zip(row_prompts, completions)]
    T = max(lengths) - 1
    assert T >= 1

    inputs_np = np.full((B, T), ctx.pad_token, dtype=np.uint32)
    targets_np = np.full((B, T), ctx.pad_token, dtype=np.uint32)
    loss_mask_np = np.zeros((B, T), dtype=np.float32)
    pad_lengths = []

    for i, (p, c) in enumerate(zip(row_prompts, completions)):
        sequence = p + c
        L = len(sequence) - 1
        shift = T - L  # left-padding amount in full sequence
        pad_lengths.append(shift)  # left pads in inputs/targets

        if len(p) < 2:
            raise ValueError("Prompt is too short")

        if len(sequence) < 2:
            raise ValueError("Sequence is too short")

        inputs_np[i, -L:] = np.asarray(sequence[:-1], dtype=np.uint32)
        targets_np[i, -L:] = np.asarray(sequence[1:], dtype=np.uint32)

        if c:
            start = -1 + shift + len(p)  # because of the one-token shift we subtract 1
            end = min(start + len(c), T)
            if start < end:
                loss_mask_np[i, start:end] = 1.0

    x_tt = tokens_to_tensor(ctx, inputs_np, B)  # [B,1,1,padded(T)]
    mask_tt = create_causal_mask(ctx, prompt_len=0, query_len=T, pad_lengths=pad_lengths)
    logits = ctx.tt_model(x_tt, mask_tt)  # [B,1,T,V]

    Tp = round_up(ctx, T)
    targets_pad = np.full((B, Tp), ctx.pad_token, dtype=np.uint32)
    targets_pad[:, :T] = targets_np  # or align however your sequence axis is arranged

    targets_tt = ttml.autograd.Tensor.from_numpy(targets_pad, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)

    nlog = ttml.ops.loss.cross_entropy_loss(logits, targets_tt, ttml.ops.ReduceType.NONE)
    nlog = ttml.ops.reshape.reshape(nlog, [B, Tp])

    loss_mask_pad = np.zeros((B, Tp), dtype=np.float32)
    loss_mask_pad[:, :T] = loss_mask_np

    loss_mask_tt = ttml.autograd.Tensor.from_numpy(loss_mask_pad, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16)

    assert nlog.shape() == [B, Tp]
    assert loss_mask_tt.shape() == [B, Tp]
    return nlog, loss_mask_tt, Tp
