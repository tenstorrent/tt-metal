# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-9 follow-up — 64L decode coherency on the Llama-70B-Galaxy ISL=128
prompt #0, **trace-by-default**.

Goal
----
Confirm that the qwen3.6-27B v2 model produces a coherent, English-language
continuation when prompted with the standard Llama-70B-Galaxy demo prompt
(``input_data_questions_prefill_128.json`` index 0) — and that the decode
loop runs entirely from a single captured metal trace (replay only, no
re-capture per token).

How "trace by default" is wired
-------------------------------
The qwen3.6 v2 decode forward bakes ``cur_pos`` (Python int) into the
trace when used with ``page_table=None`` (the non-paged path slices the
KV cache + ``ttnn.update_cache`` with a Python int write index).  For a
multi-step decode loop a single non-paged trace is invalid past the
captured cur_pos.

The fix used here is the same as v1 / 70B / demo_qwen_decode.py: switch
to the **paged decode path** (``page_table != None``).  Paged decode:
  - Uses ``ttnn.experimental.paged_update_cache(..., update_idxs_tensor=cur_pos_tensor)``
    so the cache write index is a device tensor (refreshable via
    ``copy_host_to_device_tensor`` outside the trace boundary or via
    ``ttnn.plus_one`` inside the trace).
  - Uses ``ttnn.transformer.paged_scaled_dot_product_attention_decode(..., cur_pos_tensor=cur_pos_tensor)``
    so the SDPA decode kernel reads cur_pos at runtime, never bakes it.
  - Does NOT need the V2-9 ``_decode_mask_buf`` (the kernel uses
    cur_pos_tensor to drive its own mask).

The result: **one captured trace serves all decode positions**.  Per-step
work outside the trace boundary:
  1. ``copy_host_to_device_tensor(input_emb_host, input_emb_buf)``
  2. ``execute_trace(...)``  (trace contains the model forward + a
     ``ttnn.plus_one`` on cur_pos_tensor at the end)
  3. ``ttnn.to_torch(logits_buf)`` outside the trace boundary
  4. Argmax → next token id

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_coherency_isl128.py \\
            -v -s
"""
from __future__ import annotations

import json
import pathlib
import time

import pytest
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_LLAMA70B_PROMPT_FILE = pathlib.Path(
    "models/demos/llama3_70b_galaxy/demo/sample_prompts/input_data_questions_prefill_128.json"
)

_B = 1
_T_PREFILL = 128
_H = 5120
_N_LAYERS = 64
_DECODE_STEPS = 32  # generate this many tokens via trace replay
_PATTERN = (["linear_attention"] * 3 + ["full_attention"]) * 16

# Paged-attention config — block_size=32 is the tile-aligned default; max_num_blocks
# sized so block_size * max_num_blocks / batch >= max_seq_len for a 1024-token horizon.
_PAGED_BLOCK_SIZE = 32
_PAGED_MAX_NUM_BLOCKS = 32  # 32 blocks * 32 tokens / 1 user = 1024-token horizon


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _load_full_state_dict(snapshot_dir: pathlib.Path) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    files = sorted(set(weight_map.values()))
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        sd.update(shard)
    return sd


def _build_paged_page_table(mesh, args, paged_attention_config):
    """Build a permutation-based page table (mirrors demo_qwen_decode.py).

    Returns (page_table_tt, page_table_torch) where page_table_torch is the
    host-side reverse-permutation reshaped to [B, max_blocks_per_seq].
    """
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=args.cluster_shape),
    )
    return page_table_tt, page_table


def _build_tt_model_paged(mesh, state_dict, pattern: list[str], n_layers: int, paged_attention_config):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = n_layers
    args.linear_attention_pattern = pattern
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,  # keep model-owned KV cache (paged shape)
    )
    return model, args


def _build_partial_rope_cos_sin_torch(positions: torch.Tensor):
    """CPU-only partial-RoPE cos/sin build (host tensors of shape [1, T, 64])."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    # build_mrope_cos_sin returns shape [T, rope_dim].  Unsqueeze to [1, T, rope_dim].
    return cos_ref.unsqueeze(0), sin_ref.unsqueeze(0)


def _build_partial_rope_cos_sin_tt(mesh, positions: torch.Tensor):
    """Convenience: build cos/sin device tensors (one-shot allocation).

    Used for eager prefill (where each call gets fresh device tensors).
    For decode trace replay, use ``_build_cos_sin_buf`` + ``_refresh_cos_sin_buf``
    instead so the same buffer persists across trace replays.
    """
    cos_torch, sin_torch = _build_partial_rope_cos_sin_torch(positions)
    cos_tt = ttnn.from_torch(
        cos_torch,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin_torch,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


def _build_cos_sin_buf(mesh, positions: torch.Tensor):
    """Allocate persistent cos/sin device buffers + matching host tensors.

    Returns (cos_buf, sin_buf, cos_host, sin_host) — each *host* tensor is
    a host-side ``ttnn.Tensor`` (no device=) with the same dtype/layout/
    mesh_mapper as its device counterpart, so ``copy_host_to_device_tensor``
    can do an in-place refresh per decode step.

    NB: ``cos_buf`` / ``sin_buf`` get their ADDRESSES baked into the trace;
    ``copy_host_to_device_tensor`` is the trace-OUTSIDE write that updates
    the buffer contents before each ``execute_trace``.
    """
    cos_torch, sin_torch = _build_partial_rope_cos_sin_torch(positions)
    replicate = ttnn.ReplicateTensorToMesh(mesh)
    cos_buf = ttnn.from_torch(
        cos_torch,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    sin_buf = ttnn.from_torch(
        sin_torch,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=replicate,
    )
    cos_host = ttnn.from_torch(
        cos_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
    )
    sin_host = ttnn.from_torch(
        sin_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
    )
    return cos_buf, sin_buf, cos_host, sin_host


def _refresh_cos_sin_buf(mesh, cur_pos_int: int, cos_buf, sin_buf):
    """Refresh persistent cos/sin device buffers in place for the given cur_pos."""
    cos_torch, sin_torch = _build_partial_rope_cos_sin_torch(torch.tensor([cur_pos_int], dtype=torch.long))
    replicate = ttnn.ReplicateTensorToMesh(mesh)
    cos_host = ttnn.from_torch(
        cos_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
    )
    sin_host = ttnn.from_torch(
        sin_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=replicate,
    )
    ttnn.copy_host_to_device_tensor(cos_host, cos_buf)
    ttnn.copy_host_to_device_tensor(sin_host, sin_buf)


def _embed_tokens_cpu(state_dict_hf: dict, token_ids: torch.Tensor) -> torch.Tensor:
    """CPU embedding lookup.  Returns [B, T, H] bf16."""
    emb_w = state_dict_hf["model.language_model.embed_tokens.weight"]
    return emb_w[token_ids].to(torch.bfloat16)


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args, on_host: bool = False):
    """[B, T, H] torch → col-sharded H/4 ttnn tensor (device or host)."""
    B, T, H = t.shape
    t_4d = t.reshape(1, 1, T, H)
    return ttnn.from_torch(
        t_4d,
        device=mesh if not on_host else None,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if not on_host else None,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _do_prefill_paged(model, mesh, args, x_prefill: torch.Tensor, page_table_tt):
    """Prefill through the model with paged KV cache."""
    x_tt = _send_col_sharded_hidden(x_prefill, mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(mesh, torch.arange(x_prefill.shape[1], dtype=torch.long))
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return model.forward(
        x_tt,
        current_pos=None,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )


def _gather_logits_to_cpu(tt_logits_list, mesh, args):
    out0 = tt_logits_list[0] if isinstance(tt_logits_list, list) else tt_logits_list
    logits_torch = ttnn.to_torch(
        out0,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(3, 0), mesh_shape=args.cluster_shape),
    )
    n_cols = args.cluster_shape[1]
    logits_torch = logits_torch[: logits_torch.shape[0] // n_cols]
    while logits_torch.dim() > 3 and logits_torch.shape[0] == 1:
        logits_torch = logits_torch.squeeze(0)
    if logits_torch.dim() == 3:
        logits_decode = logits_torch[:, 0:1, : args.vocab_size]
    else:
        logits_decode = logits_torch[..., : args.vocab_size]
    return logits_decode


def _gather_prefill_logits_to_cpu(tt_prefill_out, mesh, args, model, last_token_idx: int):
    x = tt_prefill_out
    x_norm, _ = model.norm(x, res=None, mode="prefill")
    if last_token_idx >= 0:
        x_norm_last = x_norm[:, :, last_token_idx : last_token_idx + 1, :]
    else:
        x_norm_last = x_norm
    lm_head_out = model.lm_head(x_norm_last, None, mode="prefill")
    return _gather_logits_to_cpu(lm_head_out, mesh, args)


def _build_cur_pos_tensor(mesh, args, cur_pos_int: int):
    """Build the persistent device cur_pos tensor.  Replicated across the
    mesh (the qwen3.6 single-user contract treats this as a scalar)."""
    cur_pos_torch = torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32)
    return ttnn.from_torch(
        cur_pos_torch,
        device=mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _copy_cur_pos(cur_pos_int: int, args, cur_pos_tt):
    """Refresh cur_pos device tensor in place (trace-safe — happens outside trace)."""
    host = ttnn.from_torch(
        torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32),
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(args.mesh_device),
    )
    ttnn.copy_host_to_device_tensor(host, cur_pos_tt)


@pytest.mark.hardware
def test_qwen36_64L_decode_coherency_isl128_trace(bh_glx_mesh):
    """Trace-by-default coherency: prefill 128 tokens, then trace-replay 32 decode steps."""
    from transformers import AutoTokenizer

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    # ---- Tokenize the Llama-70B-Galaxy demo prompt #0 ----
    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    with open(_LLAMA70B_PROMPT_FILE) as f:
        prompts = json.load(f)
    prompt = prompts[0]["prompt"]
    print(f"[coherency] prompt: {prompt!r}")
    ids = tok(prompt, return_tensors="pt").input_ids
    T_prompt = int(ids.shape[-1])
    print(f"[coherency] prompt token count = {T_prompt}")

    if T_prompt > _T_PREFILL:
        ids = ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
    pad_len = _T_PREFILL - T_prompt
    ids_padded = torch.cat([ids, torch.zeros(1, pad_len, dtype=ids.dtype)], dim=1) if pad_len > 0 else ids
    print(f"[coherency] padded to T={ids_padded.shape[-1]}; real prompt ends at index {T_prompt - 1}")

    # ---- Load FULL 64L state dict ----
    print(f"[coherency] loading full state dict ...")
    t0 = time.time()
    state_dict = _load_full_state_dict(_SNAPSHOT)
    print(f"[coherency] state dict loaded in {time.time() - t0:.1f}s; {len(state_dict)} keys")

    # ---- Build paged-attention config + page table ----
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    # ---- Build TT model with paged KV cache ----
    model, args = _build_tt_model_paged(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS, paged_attention_config)
    print(f"[coherency] 64-layer TT model built (paged KV cache)")
    page_table_tt, _ = _build_paged_page_table(bh_glx_mesh, args, paged_attention_config)

    # ---- Embed prompt tokens on CPU ----
    x_prefill = _embed_tokens_cpu(state_dict, ids_padded[:, :_T_PREFILL])
    print(f"[coherency] CPU embed shape: {list(x_prefill.shape)}")

    # ---- TT PREFILL with paged KV cache ----
    t0 = time.time()
    prefill_hidden_tt = _do_prefill_paged(model, bh_glx_mesh, args, x_prefill, page_table_tt)
    ttnn.synchronize_device(bh_glx_mesh)
    prefill_ms = (time.time() - t0) * 1000
    print(f"[coherency] prefill done in {prefill_ms:.1f} ms")

    # ---- First decode token comes from the prefill output's last-real-token logits ----
    last_prompt_logits = _gather_prefill_logits_to_cpu(
        prefill_hidden_tt, bh_glx_mesh, args, model, last_token_idx=T_prompt - 1
    )
    last_prompt_logits_flat = last_prompt_logits.reshape(-1)[: args.vocab_size].float()
    first_decode_token = int(last_prompt_logits_flat.argmax().item())
    print(
        f"[coherency] first decode token (greedy from prefill) = {first_decode_token} "
        f"({tok.decode([first_decode_token])!r})"
    )

    # ---- Set up persistent decode buffers ----
    # cur_pos_tensor (refreshed externally via copy_h2d before each execute_trace
    # OR incremented in-trace via ttnn.plus_one).
    cur_pos_int = T_prompt  # First decode is at index T_prompt (KV cache holds [0..T_prompt-1]).
    cur_pos_tt = _build_cur_pos_tensor(bh_glx_mesh, args, cur_pos_int)

    # Persistent input embedding buffer: shape [B=1, 1, T=1, H] col-sharded H/4.
    # This is the buffer that ``copy_host_to_device_tensor`` refreshes per step.
    init_emb = _embed_tokens_cpu(state_dict, torch.tensor([[first_decode_token]], dtype=torch.long))  # [1,1,H]
    input_emb_buf = _send_col_sharded_hidden(init_emb, bh_glx_mesh, args)
    print(f"[coherency] input_emb_buf shape: {list(input_emb_buf.shape)}")

    # ---- Compile pass (warm program cache before trace capture) ----
    # NOTE: the model.forward(mode='decode') with page_table != None takes
    # current_pos as a device tensor (cur_pos_tt) and uses paged_update_cache
    # + paged_scaled_dot_product_attention_decode.  Single-trace-friendly.
    print(f"[coherency] compile-pass decode (warm program cache)...")
    # Persistent cos/sin buffers (address baked into the trace; values refreshed
    # per step via copy_host_to_device_tensor OUTSIDE the trace).
    cos_buf, sin_buf, _, _ = _build_cos_sin_buf(bh_glx_mesh, torch.tensor([cur_pos_int], dtype=torch.long))

    # Set trace mode ON (skip in-forward mask refresh; paged path doesn't need
    # it but the flag also gates any other inline-host-write paths added later).
    model.set_trace_decode_mode(True)

    def _run_decode_once():
        # V2-9 trace-default: clone ``input_emb_buf`` so the model's first-layer
        # ``x.deallocate(True)`` (llama_decoder.py:458) doesn't free the persistent
        # buffer.  The clone is freed inside the trace; ``input_emb_buf`` stays
        # alive for the next ``copy_host_to_device_tensor`` refresh.
        x_in = ttnn.clone(input_emb_buf, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return model.forward(
            x_in,
            current_pos=cur_pos_tt,
            rot_mats=(cos_buf, sin_buf),
            user_id=0,
            mode="decode",
            page_table=page_table_tt,
            chunk_page_table=None,
            chunk_start_idx=None,
            start_pos=0,
            get_last_token=-1,
            kv_cache=None,
            batch_size=1,
        )

    # 1 compile pass.
    compile_out = _run_decode_once()
    ttnn.synchronize_device(bh_glx_mesh)
    if isinstance(compile_out, list):
        for t in compile_out:
            try:
                t.deallocate(True)
            except Exception:
                pass
    print(f"[coherency] compile-pass done")

    # ---- Capture trace ----
    if hasattr(model, "tt_ccl") and hasattr(model.tt_ccl, "reset_gather_and_buffer_idx"):
        model.tt_ccl.reset_gather_and_buffer_idx()
    ttnn.synchronize_device(bh_glx_mesh)

    print(f"[coherency] capturing trace ...")
    trace_id = ttnn.begin_trace_capture(bh_glx_mesh, cq_id=0)
    traced_out = _run_decode_once()
    ttnn.end_trace_capture(bh_glx_mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[coherency] trace captured: trace_id = {trace_id}")

    # ---- DECODE LOOP: replay trace N times.  Per-step:
    #     1. Set cur_pos device tensor (via copy_h2d) — outside trace.
    #     2. Set input_emb_buf (via copy_h2d) — outside trace.
    #     3. execute_trace — single command, runs the entire forward.
    #     4. ttnn.to_torch the output — outside trace.
    #     5. Argmax → next token.
    # ----
    generated_ids = [first_decode_token]
    nan_inf_steps = []
    decode_t0 = time.time()

    for step in range(_DECODE_STEPS):
        cur_pos_int_this_step = T_prompt + step
        # (1) Refresh cur_pos device tensor.
        _copy_cur_pos(cur_pos_int_this_step, args, cur_pos_tt)
        # (1b) Refresh persistent cos/sin buffers for this cur_pos.
        _refresh_cos_sin_buf(bh_glx_mesh, cur_pos_int_this_step, cos_buf, sin_buf)
        # (2) Refresh input_emb_buf (the most recently generated token's embedding).
        next_id = generated_ids[-1]
        x_step_cpu = _embed_tokens_cpu(state_dict, torch.tensor([[next_id]], dtype=torch.long))
        x_step_host = _send_col_sharded_hidden(x_step_cpu, bh_glx_mesh, args, on_host=True)
        ttnn.copy_host_to_device_tensor(x_step_host, input_emb_buf)
        # (3) execute trace.
        ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=True)
        # (4) gather logits.
        step_logits = _gather_logits_to_cpu(traced_out, bh_glx_mesh, args)
        step_logits_flat = step_logits.reshape(-1)[: args.vocab_size].float()
        if torch.isnan(step_logits_flat).any():
            nan_inf_steps.append(("nan", step))
        if torch.isinf(step_logits_flat).any():
            nan_inf_steps.append(("inf", step))
        # (5) argmax → next token.
        tok_id = int(step_logits_flat.argmax().item())
        generated_ids.append(tok_id)

    decode_ms = (time.time() - decode_t0) * 1000
    mean_decode_ms = decode_ms / _DECODE_STEPS
    print(
        f"[coherency] decode loop done — {_DECODE_STEPS} steps in {decode_ms:.1f} ms "
        f"(mean {mean_decode_ms:.2f} ms/step, "
        f"{1000.0 / mean_decode_ms:.2f} tok/s/user TRACED)"
    )

    # Release the trace.
    try:
        ttnn.release_trace(bh_glx_mesh, trace_id)
    except Exception as e:
        print(f"[coherency] release_trace failed (ignored): {e}")

    # ---- Detokenize ----
    output_text = tok.decode(generated_ids, skip_special_tokens=False)
    print()
    print("=" * 80)
    print(f"PROMPT (last 200 chars):  ...{prompt[-200:]!r}")
    print("=" * 80)
    print(f"GENERATED ({len(generated_ids)} tokens):  {output_text!r}")
    print("=" * 80)
    print(f"GENERATED token ids:  {generated_ids}")
    print("=" * 80)

    # ---- Coherency assertions ----
    assert not nan_inf_steps, f"NaN/Inf detected at decode steps: {nan_inf_steps}"
    assert first_decode_token != 0, "first decode token is the pad token (likely a prefill bug)"
    n_alpha = sum(c.isalpha() for c in output_text)
    assert n_alpha >= 5, f"generated text has <5 alpha chars: {output_text!r}"
    print(f"[coherency] PASSED — {n_alpha} alpha chars in generated text, no NaN/Inf in logits")
    print(f"[coherency] PASSED — trace-by-default: 1 capture, {_DECODE_STEPS} replays")
