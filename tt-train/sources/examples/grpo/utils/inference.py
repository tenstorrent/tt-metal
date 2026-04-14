import copy
import logging
import os
from dataclasses import dataclass
from typing import List
import ttnn
import ttml
import numpy as np
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.utils import no_grad
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from .llama_overrides import LlamaCompositeKV


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
    dp_mapper: object = None
    dp_composer: object = None
    total_devices: int = 1
    _kv_cache: object = None
    _B: int | None = None
    _N: int | None = None


def load_checkpoint(model, checkpoint_path, dp_mapper=None):
    from safetensors.numpy import load_file
    import numpy as np
    import ml_dtypes

    checkpoint = load_file(checkpoint_path)
    parameters = model.parameters()
    loaded, missing = 0, []

    for name, param in parameters.items():
        if name in checkpoint:
            arr = checkpoint[name].astype(ml_dtypes.bfloat16)
            if arr.ndim == 1:
                arr = arr.reshape(1, 1, 1, -1)
            elif arr.ndim == 2:
                arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
            restored = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)
            param.assign(restored)
            loaded += 1
        else:
            missing.append(name)

    print(f"Loaded {loaded}/{len(parameters)} parameters from {checkpoint_path}")
    if missing:
        print(f"Warning: {len(missing)} parameters not found in checkpoint:")
        for n in missing:
            print(f"  - {n}")


def setup_inference(
    temperature, max_completion_length, num_generations, transformer_config, device_config, model_source: str
) -> InferenceCtx:
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    pad_token = tokenizer.pad_token_id
    if pad_token is None:
        pad_token = tokenizer.eos_token_id

    transformer_config = copy.deepcopy(transformer_config)
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

    dp_mapper = None
    dp_composer = None
    total_devices = 1
    if device_config.enable_ddp:
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        autograd_ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False))
        device = autograd_ctx.get_device()
        dp_mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 0)
        dp_composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        total_devices = device_config.total_devices()

    local_safetensors = os.path.isdir(model_source) and any(f == "model.safetensors" for f in os.listdir(model_source))
    if local_safetensors:
        checkpoint_path = os.path.join(model_source, "model.safetensors")
        logging.info("Loading model from local safetensors: %s", model_source)
        logging.info(f"load_checkpoint({checkpoint_path})")
        load_checkpoint(tt_model, checkpoint_path, dp_mapper=dp_mapper)
    else:
        logging.info("Downloading model from HuggingFace: %s", model_source)
        logging.info(f"snapshot_download({model_source})")
        model_repo_path = snapshot_download(
            repo_id=model_source,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
        load_from_safetensors(tt_model, model_repo_path, llama_cfg)

    return InferenceCtx(
        tt_model=tt_model,
        tokenizer=tokenizer,
        pad_token=pad_token,
        temperature=temperature,
        max_tokens_to_complete=max_completion_length,
        transformer_config=transformer_config,
        group_size=num_generations,
        tile_size=32,
        sample_seed=42,
        dp_mapper=dp_mapper,
        dp_composer=dp_composer,
        total_devices=total_devices,
    )


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


def _get_kv_cache(ctx: InferenceCtx, B: int) -> ttml.models.KvCache:
    head_dim = getattr(ctx.transformer_config, "head_dim", None) or (
        ctx.transformer_config.embedding_dim // ctx.transformer_config.num_heads
    )
    if ctx._kv_cache is None or ctx._kv_cache_B != B:
        ctx._kv_cache = ttml.models.KvCache(
            ctx.transformer_config.num_blocks,
            B,
            ctx.transformer_config.num_groups,
            ctx.transformer_config.max_sequence_length,
            head_dim,
        )
        ctx._kv_cache_B = B
    ctx._kv_cache.reset()
    return ctx._kv_cache


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

    return ttml.autograd.Tensor.from_numpy(mask_4d, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, ctx.dp_mapper)


def tokens_to_tensor(ctx: InferenceCtx, tokens_np, B) -> ttml.autograd.Tensor:
    # tokens_np is of shape (B, N) or (B, 1)
    padded_len = round_up(ctx, tokens_np.shape[1])

    padded = np.full((B, padded_len), ctx.pad_token, dtype=np.uint32)
    padded[:, : tokens_np.shape[1]] = tokens_np

    return ttml.autograd.Tensor.from_numpy(
        padded.reshape(B, 1, 1, padded_len), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, ctx.dp_mapper
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
    assert B % ctx.total_devices == 0

    B_local = B // ctx.total_devices  # batch per device

    V = len(ctx.tokenizer)
    padded_V = round_up(ctx, V)

    kv_cache = _get_kv_cache(ctx, B_local)

    logits_mask_tensor = build_logits_mask(V, padded_V) if padded_V != V else None

    tokens_to_complete = min(
        ctx.max_tokens_to_complete,
        ctx.transformer_config.max_sequence_length - N,
    )

    generated_columns = []
    chunk_columns = []

    def to_np(column_list):
        arr = np.empty((B, len(column_list)), dtype=np.int32)
        for j, column in enumerate(column_list):
            arr[:, j] = column.to_numpy(ctx.dp_composer).reshape(
                B,
            )

        return arr

    for i in range(tokens_to_complete):
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
            [B_local, 1, new_tokens, 1],
        )  # B_local 1 1 1 per device = after composing its B 1 1 1

        generated_columns.append(last_token_column)
        chunk_columns.append(last_token_column)

        N += 1

        deallocate_tensors([token_tensor, mask, logits, next_token_tensor])

    completions_np = to_np(generated_columns)
    deallocate_tensors(generated_columns)

    deallocate_tensors([logits_mask_tensor])
    kv_cache.reset()

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
    ctx._N = len(prompt_tokens)
    prompt_tokens_np = np.tile(prompt_tokens, (B, 1))

    pad_lengths = [0] * ctx.group_size  # no padding

    ctx.tt_model.eval()
    with no_grad():
        return _completion_batched_impl(ctx, prompt_tokens_np, pad_lengths)


def completion_batched_multiple_prompts(ctx: InferenceCtx, prompts: List[List[int]]) -> List[List[int]]:
    max_len = max(len(row) for row in prompts)
    pad_lengths = [max_len - len(row) for row in prompts for _ in range(ctx.group_size)]
    prompts_cnt = len(prompts)
    B = ctx._B = ctx.group_size * prompts_cnt
    ctx._N = max_len

    # add the pad_token to the left of the shorter prompts, so that all prompts end at the same column
    prompt_tokens_np = np.full((B, max_len), ctx.pad_token)
    for i, row in enumerate(prompts):
        prompt_tokens_np[i * ctx.group_size : (i + 1) * ctx.group_size, max_len - len(row) :] = np.asarray(row)

    ctx.tt_model.eval()
    with no_grad():
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
