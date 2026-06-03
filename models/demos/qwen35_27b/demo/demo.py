# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5-27B demo — proper batch prefill + decode, reports TTFT and tok/s.

Uses model.prefill_layer_chunked() for prefill (calls gdn_prefill_fused on GDN
layers, processes all tokens in one pass) then model.ttnn_decode_forward() for
decode.

GDN attention implementation selected via:
  GDN_USE_TTNN_OPS=1  → TTNN ops (4D matmuls + L1 memory)
  (default)           → Fused C++ kernel

Run:
    # 4K ISL, fused kernel
    MESH_DEVICE=P150x4 HF_MODEL=<path> \\
        pytest models/demos/qwen35_27b/demo/demo.py -k isl4k -v -s

    # 4K ISL, TTNN ops
    GDN_USE_TTNN_OPS=1 MESH_DEVICE=P150x4 HF_MODEL=<path> \\
        pytest models/demos/qwen35_27b/demo/demo.py -k isl4k -v -s
"""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import requests
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.demos.qwen35_27b.tt.model import allocate_paged_kv_caches, create_qwen35_model
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig, copy_host_to_device

HF_MODEL_DEFAULT = (
    "/home/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
)
SAMPLE_PROMPT_FILE = "models/tt_transformers/demo/sample_prompts/input_data_long_4k.json"
CONTEXT_CACHE_DIR = Path("models/tt_transformers/demo/context_cache")


def _get_model_path():
    return os.environ.get("HF_MODEL", HF_MODEL_DEFAULT)


def _load_sample_prompt() -> str:
    """Load the Frankenstein long-context prompt from the sample prompts file."""
    with open(SAMPLE_PROMPT_FILE) as f:
        entry = json.load(f)[0]
    prompt_instruction = entry["prompt"]
    context_url = entry["context"]
    max_length = entry.get("max_length")

    CONTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CONTEXT_CACHE_DIR / hashlib.md5(context_url.encode()).hexdigest()
    if cache_file.exists():
        context_text = cache_file.read_text()
    else:
        resp = requests.get(context_url, timeout=30)
        context_text = resp.text if resp.status_code == 200 else ""
        if context_text:
            cache_file.write_text(context_text)

    if max_length:
        context_text = context_text[:max_length]

    return "```" + context_text + "```\n\n" + prompt_instruction


def _build_prompt_tokens(tokenizer, target_len: int) -> list:
    """Build a prompt of exactly target_len tokens from the Frankenstein long-context sample."""
    full_prompt = _load_sample_prompt()
    tokens = tokenizer.encode(full_prompt)
    if len(tokens) >= target_len:
        # Take the last target_len tokens so the instruction stays at the end
        return tokens[-target_len:]
    # Pad front with repeated filler if context is shorter than needed
    filler = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    while len(tokens) < target_len:
        tokens = filler + tokens
    return tokens[-target_len:]


_MESH_SHAPE_MAP = {"N150": (1, 1), "N300": (1, 2), "N150x4": (1, 4), "P150x4": (1, 4), "T3K": (1, 8), "TG": (8, 4)}
_MESH_SHAPE = _MESH_SHAPE_MAP.get(os.environ.get("MESH_DEVICE"))
if _MESH_SHAPE is None:
    _MESH_SHAPE = (1, min(len(ttnn.get_device_ids()), 8))


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": True, "num_command_queues": 2, "trace_region_size": 400_000_000}],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_seq_len, max_gen_tokens",
    [
        pytest.param(128, 20, id="isl128"),
        pytest.param(4096, 20, id="isl4k"),
        pytest.param(8192, 20, id="isl8k"),
        pytest.param(16384, 20, id="isl16k"),
        pytest.param(32768, 20, id="isl32k"),
        pytest.param(65536, 20, id="isl64k"),
        pytest.param(131072, 20, id="isl128k"),
        pytest.param(262144, 20, id="isl256k"),
    ],
)
def test_demo_text(mesh_device, reset_seeds, ensure_gc, input_seq_len, max_gen_tokens):
    """Qwen3.5-27B demo: proper batch prefill + decode with TTFT and tok/s metrics.

    GDN kernel path selected by GDN_USE_TTNN_OPS env var (0=fused, 1=ttnn_ops).
    """
    use_ttnn_ops = bool(os.environ.get("GDN_USE_TTNN_OPS"))
    gdn_path = "ttnn_ops" if use_ttnn_ops else "parallel_scan"
    # Always use paged KV + chunked-SDPA prefill: a single code path scales from
    # ISL=128 up to ISL=128k without per-layer static [B, 1, max_seq_len, HD]
    # KV allocations. Prefill is not traced under this path (chunked layers do
    # ttnn.copy writes between Python calls, forbidden during trace capture).
    use_paged = True
    use_prefill_trace = False
    model_path = _get_model_path()
    batch_size = 32
    max_seq_len = max(1024, input_seq_len + max_gen_tokens + 128)
    max_seq_len = ((max_seq_len + 127) // 128) * 128  # align to 128 for SDPA decode

    if mesh_device.get_num_devices() < 4:
        pytest.skip("Full model requires TP>=4")

    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = model_path

    logger.info(f"=== Qwen3.5-27B Demo | ISL={input_seq_len} | GDN={gdn_path} | paged={use_paged} ===")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Paged capacity must cover max_seq_len: block_size * max_num_blocks >= max_seq_len.
    # block_size=64 → max_num_blocks = ceil(max_seq_len / 64). At 128k → 2048, 256k → 4096.
    _page_block_size = 64
    _page_num_blocks = (max_seq_len + _page_block_size - 1) // _page_block_size
    paged_attention_config = (
        PagedAttentionConfig(block_size=_page_block_size, max_num_blocks=_page_num_blocks) if use_paged else None
    )

    logger.info("Loading model...")
    t0 = time.time()
    model = create_qwen35_model(
        mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=ttnn.bfloat8_b,
        use_ttnn_ops=use_ttnn_ops,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=use_paged,
    )
    logger.info(f"Model loaded in {time.time()-t0:.1f}s")

    # Paged KV caches and page tables for prefill (B=1) and decode (B=batch_size).
    kv_caches = None
    page_table_tt = None
    page_table_torch = None
    page_table_decode_torch = None
    page_table_decode_tt = None
    if use_paged:
        kv_caches = allocate_paged_kv_caches(model.args, paged_attention_config, mesh_device)
        page_table_row = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32)
        page_table_torch = page_table_row.unsqueeze(0)
        page_table_tt = ttnn.from_torch(
            page_table_torch,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        page_table_decode_torch = page_table_row.unsqueeze(0).repeat(batch_size, 1)
        page_table_decode_tt = ttnn.from_torch(
            page_table_decode_torch,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    args = model.args
    prompt_tokens = _build_prompt_tokens(tokenizer, input_seq_len)
    logger.info(f"Prompt: {len(prompt_tokens)} tokens (target ISL={input_seq_len})")

    # --- PREFILL ---
    seq_len = len(prompt_tokens)
    tok_tensor = torch.tensor(prompt_tokens, dtype=torch.int32).reshape(1, 1, 1, seq_len)
    tt_token_ids = ttnn.from_torch(
        tok_tensor,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    if use_paged:
        # Paged path: layer-at-a-time chunked prefill with chunked SDPA + paged KV.
        # No prefill trace (chunking writes ttnn.copy between layers, forbidden in trace).
        # Skip reset_state() for full_attention layers — their KV lives in external paged
        # caches, not in static [B, 1, max_seq_len, HD] tensors.
        def _reset_paged_prefill_states():
            for layer_idx, layer in enumerate(model.layers):
                attn = layer.attention
                layer_type = model.args.layer_types[layer_idx]
                is_paged_attn = layer_type == "full_attention"
                if hasattr(attn, "reset_state") and not is_paged_attn:
                    attn.reset_state()
                if hasattr(attn, "_init_prefill_states"):
                    attn._init_prefill_states()

        tokens_for_chunked = torch.tensor([prompt_tokens], dtype=torch.long)

        logger.info("Prefill warmup (compile, chunked + paged)...")
        _reset_paged_prefill_states()
        x_normed_warmup = model.prefill_layer_chunked(
            tokens_for_chunked,
            use_paged=True,
            page_table=page_table_tt,
            page_table_torch=page_table_torch,
            kv_caches=kv_caches,
            paged_attention_config=paged_attention_config,
            user_id=0,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(x_normed_warmup)

        logger.info("Prefill timed run (chunked + paged)...")
        _reset_paged_prefill_states()
        ttnn.synchronize_device(mesh_device)
        t_prefill = time.time()
        x_normed_trace = model.prefill_layer_chunked(
            tokens_for_chunked,
            use_paged=True,
            page_table=page_table_tt,
            page_table_torch=page_table_torch,
            kv_caches=kv_caches,
            paged_attention_config=paged_attention_config,
            user_id=0,
        )
        ttnn.synchronize_device(mesh_device)
        prefill_time = time.time() - t_prefill

    else:
        # Warmup / compile run — compiles all kernels before the timed run.
        logger.info("Prefill warmup (compile)...")
        model._reset_all_prefill_states(seq_len)
        x_normed_warmup = model._prefill_forward_device(tt_token_ids)
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(x_normed_warmup)

    if use_paged:
        # Paged path: prefill_time and x_normed_trace are already set above.
        # No trace and no _apply_all_trace_prefill_states (KV writes happen
        # directly into paged caches via prefill_layer_chunked).
        pass
    elif use_prefill_trace:
        # Trace path: capture then execute.
        # forward_prefill stores conv/rec states in Python attributes (_trace_qkv_states,
        # _trace_rec_state) as trace-internal tensors instead of ttnn.copy to pre-existing
        # buffers (writes are forbidden during trace capture).  After execute, we call
        # _apply_all_trace_prefill_states() to copy them to the persistent buffers.
        logger.info("Capturing prefill trace...")
        model._zero_all_prefill_states_inplace()
        trace_id_prefill = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        x_normed_trace = model._prefill_forward_device(tt_token_ids)
        ttnn.end_trace_capture(mesh_device, trace_id_prefill, cq_id=0)
        logger.info("Prefill trace captured.")

        model._zero_all_prefill_states_inplace()
        ttnn.synchronize_device(mesh_device)

        t_prefill = time.time()
        ttnn.execute_trace(mesh_device, trace_id_prefill, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        prefill_time = time.time() - t_prefill

        # Copy trace-derived states to persistent buffers (outside trace, ttnn.copy allowed).
        model._apply_all_trace_prefill_states()
    else:
        # Direct path: reset states and run (chunk_gated_delta_rule parallel scan).
        logger.info("Prefill timed run (no trace — parallel scan)...")
        model._reset_all_prefill_states(seq_len)
        ttnn.synchronize_device(mesh_device)

        t_prefill = time.time()
        x_normed_trace = model._prefill_forward_device(tt_token_ids)
        ttnn.synchronize_device(mesh_device)
        prefill_time = time.time() - t_prefill

        # Copy trace-internal state tensors to persistent buffers.
        model._apply_all_trace_prefill_states()

    ttft_ms = prefill_time * 1000

    # Replicate prefill states to all batch slots NOW (after timed run).
    # For paged: KV already in paged caches (shared across decode slots via page_table);
    # only GDN states (conv + recurrence) need replicating.
    model._replicate_all_prefill_states()

    # Apply LM head to get first generated token
    lm_input_mem_cfg = model.args.get_lm_head_input_mem_config(Mode.PREFILL, None)
    if lm_input_mem_cfg.is_sharded():
        x_normed_trace = ttnn.interleaved_to_sharded(x_normed_trace, lm_input_mem_cfg)
    logits_tt = model.lm_head(x_normed_trace)
    ttnn.deallocate(x_normed_trace)
    logits_torch = ttnn.to_torch(logits_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    ttnn.deallocate(logits_tt)
    logits_torch = logits_torch[0, 0, -1, : args.vocab_size]
    current_token = logits_torch.argmax().item()
    first_token_text = tokenizer.decode([current_token])
    logger.info(f"Prefill done in {prefill_time:.2f}s (TTFT={ttft_ms:.0f}ms), first token='{first_token_text}'")

    # --- DECODE (traced) ---
    logger.info(f"Decoding {max_gen_tokens} tokens...")
    start_pos = len(prompt_tokens)
    generated_tokens = [current_token]

    # Compile step at start_pos (warms up decode kernels)
    logger.info("Decode compile step...")
    tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
    cur_pos = torch.full((batch_size,), start_pos, dtype=torch.long)
    host_inputs = model.prepare_decode_inputs_host(
        tok_batch, cur_pos, page_table=page_table_decode_torch if use_paged else None
    )
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    compile_out = model.ttnn_decode_forward(
        *device_inputs,
        kv_cache=kv_caches if use_paged else None,
        sampling_on_device=True,
    )
    tt_compile_tok = compile_out[0] if isinstance(compile_out, tuple) else compile_out
    toks_cpu = ttnn.to_torch(tt_compile_tok, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    current_token = toks_cpu[0].flatten()[0].int().item()
    generated_tokens.append(current_token)
    ttnn.synchronize_device(mesh_device)

    # Trace capture at start_pos + 1
    logger.info("Capturing decode trace...")
    pos = start_pos + 1
    tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
    cur_pos = torch.full((batch_size,), pos, dtype=torch.long)
    host_inputs = model.prepare_decode_inputs_host(
        tok_batch, cur_pos, page_table=page_table_decode_torch if use_paged else None
    )
    copy_host_to_device(host_tensors=host_inputs, device_tensors=device_inputs)

    trace_id_decode = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = model.ttnn_decode_forward(
        *device_inputs,
        kv_cache=kv_caches if use_paged else None,
        sampling_on_device=True,
    )
    ttnn.end_trace_capture(mesh_device, trace_id_decode, cq_id=0)

    tt_toks_out = trace_out[0] if isinstance(trace_out, tuple) else trace_out
    toks_cpu = ttnn.to_torch(tt_toks_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    current_token = toks_cpu[0].flatten()[0].int().item()
    generated_tokens.append(current_token)
    logger.info("Decode trace captured.")

    # Traced decode loop
    decode_times = []
    for step in range(max_gen_tokens - 3):
        t_step = time.time()
        pos = start_pos + step + 2
        tok_batch = torch.full((batch_size,), current_token, dtype=torch.long)
        cur_pos = torch.full((batch_size,), pos, dtype=torch.long)

        host_inputs = model.prepare_decode_inputs_host(
            tok_batch, cur_pos, page_table=page_table_decode_torch if use_paged else None
        )
        copy_host_to_device(host_tensors=host_inputs, device_tensors=device_inputs)

        ttnn.execute_trace(mesh_device, trace_id_decode, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        toks_cpu = ttnn.to_torch(tt_toks_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        current_token = toks_cpu[0].flatten()[0].int().item()
        generated_tokens.append(current_token)
        decode_times.append(time.time() - t_step)

    # Cleanup traces and persistent buffers
    if use_prefill_trace and not use_paged:
        ttnn.release_trace(mesh_device, trace_id_prefill)
    ttnn.release_trace(mesh_device, trace_id_decode)
    ttnn.deallocate(tt_token_ids)

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    prompt_tail = tokenizer.decode(prompt_tokens[-8:])
    logger.info(f"Output: '...{prompt_tail}{generated_text}'")

    # --- PERF REPORT ---
    steady = decode_times if decode_times else [0]
    avg_ms = sum(steady) / len(steady) * 1000
    tok_s_user = 1000 / avg_ms if avg_ms > 0 else 0
    tok_s_total = tok_s_user * batch_size

    logger.info("")
    pf_mode = "traced" if use_prefill_trace else "direct"
    logger.info(f"=== Performance | ISL={input_seq_len} | GDN={gdn_path} | prefill={pf_mode} ===")
    logger.info(f"  TTFT (traced):            {ttft_ms:.0f} ms")
    logger.info(f"  Decode (traced steady):   {avg_ms:.0f} ms/tok")
    logger.info(f"  Throughput:               {tok_s_user:.2f} tok/s/user  ({tok_s_total:.1f} tok/s aggregate)")

    assert len(generated_text.strip()) > 0, "Model produced empty output"
    logger.info("PASSED")
