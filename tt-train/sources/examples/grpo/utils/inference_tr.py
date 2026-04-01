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
from models.tt_transformers.tt.model_config import ModelArgs, DecodersPrecision
from models.tt_transformers.tt.model import Transformer
import uuid
from pathlib import Path


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


# sets up a dummy model
def setup_tt_transformers_inference(
    mesh_device,
    tokenizer,  # pass ttml_ctx.tokenizer — dummy_weights=True gives None
    max_seq_len=1024,
    max_batch_size=32,
    max_tokens_to_complete=256,
    temperature=0.6,
    top_p=0.9,
    group_size=1,
    instruct=True,
) -> TrInferenceCtx:
    paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)

    model_args = ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        optimizations=lambda args: DecodersPrecision.performance(args.n_layers, args.model_name),
    )
    state_dict = model_args.load_state_dict()
    empty_state_dict = {k: torch.zeros_like(v) for k, v in state_dict.items()}

    model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=empty_state_dict,
        weight_cache_path=Path(f"/tmp/tt_zero_cache_{uuid.uuid4().hex}"),  # hack to make the cache not found always
        paged_attention_config=paged_attention_config,
    )

    tt_kv_cache = [l.attention.layer_past for l in model.layers]

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


def get_stop_tokens(tokenizer):
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for tok_str in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            stop_ids.add(int(tid))

    return stop_ids


def completion_batched_multiple_prompts_tr(
    ctx: TrInferenceCtx,
    prompt_texts: List[str],
) -> List[List[int]]:
    """
    Equivalent of completion_batched_multiple_prompts but using tt-transformers.
    Each prompt is repeated `group_size` times (for GRPO).
    Returns a flat list of completion token lists (len = len(prompts) * group_size).
    Completions contain only the generated tokens (no prompt tokens).
    """
    expanded_prompt_texts = [p for p in prompt_texts for _ in range(ctx.group_size)]
    global_batch_size = len(expanded_prompt_texts)

    assert global_batch_size <= ctx.max_batch_size, (
        f"Total batch ({global_batch_size} = {len(prompt_texts)} prompts * {ctx.group_size} group_size) "
        f"exceeds model max_batch_size ({ctx.max_batch_size})"
    )

    (
        input_tokens_prefill_pt,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    ) = preprocess_inputs_prefill(
        expanded_prompt_texts,
        ctx.tokenizer,
        [ctx.model_args],
        instruct=False,
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

    stop_ids = get_stop_tokens(ctx.tokenizer)
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
            if tok in stop_ids:
                user_done[user] = True
            else:
                all_generated[user].append(tok)

        if all(user_done):
            break

    return all_generated


def sync_ttml_to_tt_transformers(ttml_model, tr_model):
    params = ttml_model.parameters()
    D = tr_model.args.dim  # hidden_size = 2048

    def _sync_weight(src_tensor, dst_tensor, transpose=False):
        """Sync TILE DRAM INTERLEAVED → DRAM WIDTH_SHARDED weight."""
        w = src_tensor.get_value()
        if transpose:
            w = ttnn.transpose(w, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.typecast(w, dst_tensor.dtype)
        return ttnn.copy_to_memory_config(w, dst_tensor.memory_config(), preallocated_output=dst_tensor)

    def _sync_norm(src_tensor, dst_weight):
        """Sync norm gamma [1,1,1,D] TILE → [1,1,D//32,32] ROW_MAJOR DRAM."""
        w = src_tensor.get_value()
        w = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        w = ttnn.reshape(w, [1, 1, D // 32, 32])
        w = ttnn.typecast(w, dst_weight.dtype)
        ttnn.copy_to_memory_config(w, dst_weight.memory_config(), preallocated_output=dst_weight)

    # ── Token Embedding ────────────────────────────────────────────────────────
    # fc and tok_emb are weight-tied; only fc/weight is in params.
    # fc weight shape is [1,1,hidden,vocab]; embedding needs [1,1,vocab,hidden].
    w_emb = params["Llama/fc/weight"].get_value()
    if w_emb.shape[-2] < w_emb.shape[-1]:  # [1,1,hidden,vocab] → transpose
        w_emb = ttnn.transpose(w_emb, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w_emb = ttnn.to_layout(w_emb, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w_emb = ttnn.typecast(w_emb, tr_model.embd.weights.dtype)
    ttnn.copy_to_memory_config(w_emb, tr_model.embd.weights.memory_config(), preallocated_output=tr_model.embd.weights)

    # ── Final Norm ─────────────────────────────────────────────────────────────
    _sync_norm(params["Llama/ln_fc/gamma"], tr_model.norm.norm.weight)

    # ── LM Head ────────────────────────────────────────────────────────────────
    # ttml fc/weight shape: [1, 1, vocab_size, hidden]
    # LMHead expects weights as [hidden, vocab], device-interleaved and sharded.
    w_lm = params["Llama/fc/weight"].get_value()  # [1, 1, vocab_size, hidden]
    w_lm = ttnn.transpose(w_lm, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, hidden, vocab_size]
    w_lm = ttnn.typecast(w_lm, tr_model.lm_head.dtype)

    # Pad vocab dim to padded_vocab_size so every device slice is in-bounds.
    # (e.g. 128256 → 131072; padding amount is tile-aligned so TILE_LAYOUT is fine.)
    padded_vocab_size = tr_model.lm_head.padded_vocab_size
    real_vocab_cols = w_lm.shape[3]
    if real_vocab_cols < padded_vocab_size:
        w_lm = ttnn.pad(
            w_lm,
            [(0, 0), (0, 0), (0, 0), (0, padded_vocab_size - real_vocab_cols)],
            0,
        )

    num_devices = tr_model.lm_head.num_devices  # 2
    size_per_dev = padded_vocab_size // num_devices  # 65536 for 1B

    for idx, out_shard in enumerate(tr_model.lm_head.output_weights_dram_sharded):
        split_size = tr_model.lm_head.split_sizes_dram_sharded[idx]
        col_start = sum(tr_model.lm_head.split_sizes_dram_sharded[:idx])
        col_end = col_start + split_size
        chunk = ttnn.slice(
            w_lm,
            [0, 0, 0, col_start],
            [1, 1, w_lm.shape[2], col_end],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunk = ttnn.reshape(chunk, [chunk.shape[2], chunk.shape[3]])
        ttnn.copy_to_memory_config(chunk, out_shard.memory_config(), preallocated_output=out_shard)

    # ── Per-Layer Weights ──────────────────────────────────────────────────────
    for i, block in enumerate(tr_model.layers):
        p = f"Llama/blocks/{i}"
        attn = block.attention
        ffn = block.feed_forward

        # Norm weights (attention pre-norm and FFN pre-norm)
        _sync_norm(params[f"{p}/attention_norm/gamma"], block.attention_norm.norm.weight)
        _sync_norm(params[f"{p}/mlp_norm/gamma"], block.ff_norm.norm.weight)

        # wqkv: fuse Q + K + V and sync
        wq = params[f"{p}/attention/q_linear/weight"].get_value()
        kv = params[f"{p}/attention/kv_linear/weight"].get_value()
        half = kv.shape[2] // 2
        wk = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, half, kv.shape[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wv = ttnn.slice(kv, [0, 0, half, 0], [1, 1, kv.shape[2], kv.shape[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wqkv = ttnn.concat(
            [
                ttnn.transpose(wq, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                ttnn.transpose(wk, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                ttnn.transpose(wv, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ],
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wqkv = ttnn.typecast(wqkv, attn.wqkv.dtype)
        wqkv = ttnn.to_memory_config(wqkv, memory_config=attn.wqkv.memory_config())
        ttnn.copy(input_a=wqkv, input_b=attn.wqkv)

        # wo, w1, w3, w2
        attn.wo = _sync_weight(params[f"{p}/attention/out_linear/weight"], attn.wo, transpose=True)
        ffn.w1 = _sync_weight(params[f"{p}/mlp/w1/weight"], ffn.w1, transpose=True)
        ffn.w3 = _sync_weight(params[f"{p}/mlp/w3/weight"], ffn.w3, transpose=True)
        ffn.w2 = _sync_weight(params[f"{p}/mlp/w2/weight"], ffn.w2, transpose=True)
