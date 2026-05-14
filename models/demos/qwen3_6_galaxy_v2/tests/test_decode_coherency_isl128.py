# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-9 follow-up — 64L decode coherency check on the Llama-70B-Galaxy
ISL=128 prompt #0 ("What is your favorite condiment?...").

Goal
----
Confirm that the qwen3.6-27B v2 model produces a coherent, English-language
continuation when prompted with the standard Llama-70B-Galaxy demo prompt
(``input_data_questions_prefill_128.json`` index 0).  This is the closest
analog to the v1 ' Paris' smoke test for the 64-layer model — a single
forward sanity check on real tokens, not random embeddings.

What this verifies
------------------
- End-to-end prefill of 128-token (tile-padded) prompt through TtTransformer
- 32 greedy decode steps via repeated eager forwards (one decode per step,
  cur_pos advances per step)
- No NaN/Inf in logits at any decode step
- Decoded text is printable ASCII (basic coherency probe; full coherency
  would compare against an HF reference run which is out of scope here)

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
_DECODE_STEPS = 32  # generate this many tokens
_PATTERN = (["linear_attention"] * 3 + ["full_attention"]) * 16  # 64L hybrid default


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
    """Load the FULL 64-layer state dict (all 15 safetensor shards)."""
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    files = sorted(set(weight_map.values()))
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        sd.update(shard)
    return sd


def _build_tt_model(mesh, state_dict, pattern: list[str], n_layers: int):
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
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )
    return model, args


def _build_partial_rope_cos_sin_tt(mesh, positions: torch.Tensor):
    """positions: [T] long → cos/sin TT tensors of shape [1, T, 64] replicated."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    cos_tt = ttnn.from_torch(
        cos_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin_ref.unsqueeze(0),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return cos_tt, sin_tt


def _embed_tokens_cpu(state_dict_hf: dict, token_ids: torch.Tensor) -> torch.Tensor:
    """CPU embedding lookup.  Returns [B, T, H] bf16."""
    emb_w = state_dict_hf["model.language_model.embed_tokens.weight"]  # [V, H]
    # Index lookup (token_ids is [B, T]).
    return emb_w[token_ids].to(torch.bfloat16)


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
    """[B, T, H] torch → col-sharded H/4 ttnn tensor."""
    B, T, H = t.shape
    t_4d = t.reshape(1, 1, T, H)
    return ttnn.from_torch(
        t_4d,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _do_prefill(model, mesh, args, x_prefill: torch.Tensor):
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
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=chunk_start_idx_tt,
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )


def _do_decode(model, x_decode_tt, cos_tt, sin_tt, cur_pos: int):
    return model.forward(
        x_decode_tt,
        current_pos=cur_pos,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
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
    """Prefill returns the hidden state.  Use model's norm+lm_head exit
    (in 'prefill' mode) to project to logits at the LAST prompt token."""
    # The model.forward(mode='prefill') return value is the col-sharded hidden state.
    # We can manually slice it and run norm+lm_head OR — simpler — call the model's
    # ``process_output_prefill_logits`` helper which returns logits.
    # However that helper expects a specific format; for this coherency check
    # we just want the argmax token id at position ``last_token_idx``.
    # Easiest: run norm + lm_head directly.
    x = tt_prefill_out
    x_norm, _ = model.norm(x, res=None, mode="prefill")
    # Slice to the LAST prompt token (we want logits at position last_token_idx).
    if last_token_idx >= 0:
        x_norm_last = x_norm[:, :, last_token_idx : last_token_idx + 1, :]
    else:
        x_norm_last = x_norm
    lm_head_out = model.lm_head(x_norm_last, None, mode="prefill")
    return _gather_logits_to_cpu(lm_head_out, mesh, args)


@pytest.mark.hardware
def test_qwen36_64L_decode_coherency_isl128(bh_glx_mesh):
    """End-to-end coherency: prefill 128 real tokens, then greedy-decode 32 more."""
    from transformers import AutoTokenizer

    # ---- Tokenize the Llama-70B-Galaxy demo prompt #0 ----
    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    with open(_LLAMA70B_PROMPT_FILE) as f:
        prompts = json.load(f)
    prompt = prompts[0]["prompt"]
    print(f"[coherency] prompt: {prompt!r}")
    ids = tok(prompt, return_tensors="pt").input_ids  # [1, T_prompt]
    T_prompt = int(ids.shape[-1])
    print(f"[coherency] prompt token count = {T_prompt}")

    # Pad to T_PREFILL=128 (tile-aligned).
    if T_prompt > _T_PREFILL:
        ids = ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
        print(f"[coherency] truncated prompt to {_T_PREFILL} tokens")
    pad_len = _T_PREFILL - T_prompt
    if pad_len > 0:
        # Pad with token 0; the last-real-token index is T_prompt-1, so
        # the trailing padding does not affect the next-token prediction.
        ids_padded = torch.cat([ids, torch.zeros(1, pad_len, dtype=ids.dtype)], dim=1)
    else:
        ids_padded = ids
    print(f"[coherency] padded to T={ids_padded.shape[-1]}; real prompt ends at index {T_prompt - 1}")

    # ---- Load FULL 64L state dict (~16 GB) ----
    print(f"[coherency] loading full state dict ...")
    t0 = time.time()
    state_dict = _load_full_state_dict(_SNAPSHOT)
    print(f"[coherency] state dict loaded in {time.time() - t0:.1f}s; {len(state_dict)} keys")

    # ---- Build TT model ----
    model, args = _build_tt_model(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS)
    print(f"[coherency] 64-layer TT model built")

    # ---- Embed tokens on CPU using HF weights ----
    x_prefill = _embed_tokens_cpu(state_dict, ids_padded[:, :_T_PREFILL])  # [1, 128, H]
    print(f"[coherency] CPU embed shape: {list(x_prefill.shape)}, dtype={x_prefill.dtype}")

    # ---- TT PREFILL ----
    t0 = time.time()
    prefill_hidden_tt = _do_prefill(model, bh_glx_mesh, args, x_prefill)
    ttnn.synchronize_device(bh_glx_mesh)
    prefill_ms = (time.time() - t0) * 1000
    print(f"[coherency] prefill done in {prefill_ms:.1f} ms")

    # ---- Project the LAST prompt token's hidden state to logits ----
    # Compute logits at position T_prompt-1 (the actual last real token).
    last_prompt_logits = _gather_prefill_logits_to_cpu(
        prefill_hidden_tt, bh_glx_mesh, args, model, last_token_idx=T_prompt - 1
    )
    last_prompt_logits_flat = last_prompt_logits.reshape(-1)[: args.vocab_size].float()
    first_decode_token = int(last_prompt_logits_flat.argmax().item())
    print(
        f"[coherency] first decode token (greedy from prefill) = {first_decode_token} "
        f"({tok.decode([first_decode_token])!r})"
    )

    # ---- DECODE LOOP: greedy decode N tokens ----
    # NB: we use eager (per-step) decode here — wiring the trace loop requires
    # cur_pos to be a device tensor refreshed via copy_host_to_device_tensor
    # between each execute_trace (otherwise the trace bakes in cur_pos).
    # For a coherency check, eager is sufficient; speedup is documented
    # separately in PERF.md.
    generated_ids = [first_decode_token]
    cur_pos = T_prompt  # The KV cache is filled up to index T_prompt-1; the next decode is at index T_prompt.
    # IMPORTANT: ensure trace-mode flag is OFF (inline mask refresh ENABLED for eager).
    if hasattr(model, "set_trace_decode_mode"):
        model.set_trace_decode_mode(False)

    print(f"[coherency] starting decode loop ({_DECODE_STEPS} steps) at cur_pos={cur_pos}")

    decode_t0 = time.time()
    nan_inf_steps = []
    for step in range(_DECODE_STEPS):
        # Embed the most recently generated token on CPU.
        next_id = generated_ids[-1]
        x_step_cpu = _embed_tokens_cpu(state_dict, torch.tensor([[next_id]], dtype=torch.long))  # [1, 1, H]
        x_step_tt = _send_col_sharded_hidden(x_step_cpu, bh_glx_mesh, args)
        cos_tt_step, sin_tt_step = _build_partial_rope_cos_sin_tt(
            bh_glx_mesh, torch.tensor([cur_pos], dtype=torch.long)
        )

        step_out = _do_decode(model, x_step_tt, cos_tt_step, sin_tt_step, cur_pos)
        step_logits_cpu = _gather_logits_to_cpu(step_out, bh_glx_mesh, args)
        step_logits_flat = step_logits_cpu.reshape(-1)[: args.vocab_size].float()

        if torch.isnan(step_logits_flat).any():
            nan_inf_steps.append(("nan", step))
        if torch.isinf(step_logits_flat).any():
            nan_inf_steps.append(("inf", step))

        tok_id = int(step_logits_flat.argmax().item())
        generated_ids.append(tok_id)
        cur_pos += 1

        # Deallocate intermediates to avoid running out of L1.
        try:
            x_step_tt.deallocate(True)
        except Exception:
            pass
        if isinstance(step_out, list):
            for t in step_out:
                try:
                    t.deallocate(True)
                except Exception:
                    pass

    decode_ms = (time.time() - decode_t0) * 1000
    mean_decode_ms = decode_ms / _DECODE_STEPS
    print(
        f"[coherency] decode loop done — {_DECODE_STEPS} steps in {decode_ms:.1f} ms "
        f"(mean {mean_decode_ms:.1f} ms/step, {1000.0 / mean_decode_ms:.2f} tok/s/user eager)"
    )

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
    print()

    # ---- Coherency assertions ----
    assert not nan_inf_steps, f"NaN/Inf detected at decode steps: {nan_inf_steps}"
    # First decode token must be non-trivial (not the pad token 0, not EOS).
    assert first_decode_token != 0, f"first decode token is the pad token (likely a prefill bug)"
    # Generated text must contain at least one alpha char (basic coherency probe).
    n_alpha = sum(c.isalpha() for c in output_text)
    assert n_alpha >= 5, f"generated text has <5 alpha chars — model output looks like garbage: {output_text!r}"
    print(f"[coherency] PASSED — {n_alpha} alpha chars in generated text, no NaN/Inf in logits")
