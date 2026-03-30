# Inference using tt_transformers library

import torch
from typing import List

import ttnn
from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
    create_tt_model,
    preprocess_inputs_prefill,
    sample_host,
)
from models.tt_transformers.tt.generator import Generator, create_submeshes
from models.tt_transformers.tt.model_config import DecodersPrecision
from dataclasses import dataclass


@dataclass
class TrInferenceCtx:
    generator: Generator
    model_args: object  # ModelArgs instance
    model: object  # Transformer instance
    tokenizer: object  # HF tokenizer
    tt_kv_cache: list  # list of (k_cache, v_cache) per layer
    page_table: torch.Tensor  # [max_batch_size, blocks_per_user]
    paged_attention_config: PagedAttentionConfig
    mesh_device: object  # ttnn.MeshDevice
    # GRPO / generation knobs
    max_tokens_to_complete: int = 256
    temperature: float = 0.6
    top_p: float = 0.9
    group_size: int = 1
    max_seq_len: int = 1024
    max_batch_size: int = 32
    instruct: bool = True


def setup_tt_transformers_inference(
    mesh_device,
    max_seq_len=1024,
    max_batch_size=32,
    max_tokens_to_complete=256,
    temperature=0.6,
    top_p=0.9,
    group_size=1,
    instruct=True,
) -> TrInferenceCtx:
    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

    optimizations = lambda model_args: DecodersPrecision.performance(model_args.n_layers, model_args.model_name)

    model_args, model, tt_kv_cache, _ = create_tt_model(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        optimizations=optimizations,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_attention_config,
        dtype=ttnn.bfloat8_b,
    )

    tokenizer = model_args.tokenizer

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        max_batch_size,
        paged_attention_config.max_num_blocks // max_batch_size,
    )

    generator = Generator([model], [model_args], mesh_device, tokenizer=tokenizer)

    return TrInferenceCtx(
        generator=generator,
        model_args=model_args,
        model=model,
        tokenizer=tokenizer,
        tt_kv_cache=[tt_kv_cache],
        page_table=page_table,
        paged_attention_config=paged_attention_config,
        mesh_device=mesh_device,
        max_tokens_to_complete=max_tokens_to_complete,
        temperature=temperature,
        top_p=top_p,
        group_size=group_size,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        instruct=instruct,
    )


def completion_batched_multiple_prompts_tr(
    ctx: TrInferenceCtx,
    prompts: List[List[int]],
) -> List[List[int]]:
    """
    Equivalent of completion_batched_multiple_prompts but using tt-transformers.
    Each prompt is repeated `group_size` times (for GRPO).
    Returns a flat list of completion token lists (len = len(prompts) * group_size).
    Completions contain only the generated tokens (no prompt tokens).
    """
    expanded_prompts = [p for p in prompts for _ in range(ctx.group_size)]
    global_batch_size = len(expanded_prompts)

    assert global_batch_size <= ctx.max_batch_size, (
        f"Total batch ({global_batch_size} = {len(prompts)} prompts * {ctx.group_size} group_size) "
        f"exceeds model max_batch_size ({ctx.max_batch_size})"
    )

    prompt_texts = [ctx.tokenizer.decode(p) for p in expanded_prompts]

    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        prompt_texts,
        ctx.tokenizer,
        [ctx.model_args],
        instruct=ctx.instruct,
        max_generated_tokens=ctx.max_tokens_to_complete,
        max_prefill_len=ctx.max_seq_len,
    )

    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(global_batch_size, -1)

    # Zero the KV cache to avoid context leaking from previous calls
    for layer in ctx.model.layers:
        k_cache, v_cache = layer.attention.layer_past
        ttnn.mul(k_cache, 0, output_tensor=k_cache)
        ttnn.mul(v_cache, 0, output_tensor=v_cache)
    ctx.generator.prev_page_table = None

    # --- Prefill: process all prompt tokens at once, get first generated token ---
    prefill_out = ctx.generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=ctx.page_table,
        kv_cache=ctx.tt_kv_cache,
        prompt_lens=decoding_pos,
    )
    out_tok = torch.argmax(prefill_out, dim=-1)  # [B, 1]

    # --- Decode loop: generate tokens one at a time ---
    all_generated = [[] for _ in range(global_batch_size)]
    for user in range(global_batch_size):
        all_generated[user].append(int(out_tok[user].item()))

    current_pos = torch.tensor(decoding_pos)
    user_done = [False] * global_batch_size

    for iteration in range(ctx.max_tokens_to_complete):
        logits, _ = ctx.generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=True,
            page_table=ctx.page_table,
            kv_cache=ctx.tt_kv_cache,
            reset_batch=(iteration == 0),
        )

        if ctx.temperature > 0:
            _, out_tok = sample_host(logits, temperature=ctx.temperature, top_p=ctx.top_p, on_host=True)
        else:
            out_tok = torch.argmax(logits, dim=-1)
            if out_tok.dim() == 1:
                out_tok = out_tok.unsqueeze(0)

        current_pos += 1

        for user in range(global_batch_size):
            tok = int(out_tok[user].item())
            if user_done[user]:
                continue
            if tok in ctx.tokenizer.stop_tokens:
                user_done[user] = True
            else:
                all_generated[user].append(tok)

        if all(user_done):
            break

    return all_generated
