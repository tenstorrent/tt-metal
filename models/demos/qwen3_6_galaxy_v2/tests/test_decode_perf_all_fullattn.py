# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-perf-gap probe — measure qwen3.6 decode at all 64 layers = full_attention.

Hypothesis under test: "If we replace all 48 DeltaNet layers with full-attention
layers, we'd be close to Qwen3-32B's 50 tok/s/user."

This test forces ``pattern = ['full_attention'] * 64`` on the qwen3.6 v2 model.
Since the HF safetensors only has full-attention weights for 16 layer slots
(indices 3, 7, 11, ..., 63), we synthesize a state dict that round-robins
those 16 real FA layers' weights into all 64 layer slots:

    model layer 0..3   ← HF layer 3
    model layer 4..7   ← HF layer 7
    ...
    model layer 60..63 ← HF layer 63

Output text will be semantically meaningless (each block of 4 layers shares
weights), but the per-step latency = real 64-layer all-FA forward pass.

Driver mirrors ``test_decode_perf_intrace.py``: in-trace decode loop with
sampling + embedding + cos/sin + cur_pos all inside the trace boundary.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_perf_all_fullattn.py \\
            -v -s
"""
from __future__ import annotations

import json
import os
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
_T_PREFILL = int(os.environ.get("QWEN36_PERF_T_PREFILL", "128"))
_H = 5120
_N_LAYERS = 64
_DECODE_STEPS = 32
_PATTERN = ["full_attention"] * _N_LAYERS  # ALL full-attention — the experiment

_PAGED_BLOCK_SIZE = 32
_PAGED_MAX_NUM_BLOCKS = max(32, (_T_PREFILL + _DECODE_STEPS + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE + 4)

# Layer indices in the real qwen3.6 model where layer_types[i] == "full_attention".
_FA_LAYER_INDICES = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]


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


def _synthesize_all_fa_state_dict(orig_sd: dict) -> dict:
    """Build a state dict where every layer slot has FA weights.

    For each target layer ``t`` in [0, 64): find the nearest real FA layer
    index ``src`` (use ``_FA_LAYER_INDICES[t // 4]``) and copy its
    ``model.language_model.layers.{src}.*`` keys to
    ``model.language_model.layers.{t}.*`` — but ONLY the ``self_attn.*``
    and ``input_layernorm.*`` / ``post_attention_layernorm.*`` /
    ``mlp.*`` keys (drop any ``linear_attn.*`` keys that don't exist on
    FA layers anyway).

    The original target-layer's ``linear_attn.*`` keys are removed so the
    weight loader doesn't get confused.
    """
    new_sd: dict[str, torch.Tensor] = {}
    layer_prefix = "model.language_model.layers."

    # Keep all non-layer keys verbatim (embeddings, norm, lm_head, etc.).
    for k, v in orig_sd.items():
        if not k.startswith(layer_prefix):
            new_sd[k] = v

    # For each target layer slot, pull weights from the nearest real FA layer.
    for tgt in range(_N_LAYERS):
        src = _FA_LAYER_INDICES[tgt // 4]  # 0..3 → 3, 4..7 → 7, ...
        src_prefix = f"{layer_prefix}{src}."
        tgt_prefix = f"{layer_prefix}{tgt}."
        for k, v in orig_sd.items():
            if not k.startswith(src_prefix):
                continue
            sub = k[len(src_prefix) :]
            # Skip any linear_attn keys (real FA layers shouldn't have these,
            # but skip defensively).
            if sub.startswith("linear_attn."):
                continue
            new_sd[tgt_prefix + sub] = v

    return new_sd


def _build_paged_page_table(mesh, args, paged_attention_config):
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
    return page_table_tt


def _build_tt_model_paged(mesh, state_dict, pattern, n_layers, paged_attention_config):
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    args = TtQwen36ModelArgs(mesh)
    args.n_layers = n_layers
    args.linear_attention_pattern = pattern
    # Force fresh weight cache to avoid pulling stale hybrid-layout caches.
    weight_cache_path = args.weight_cache_path(ttnn.bfloat8_b)
    weight_cache_path = weight_cache_path.parent / (weight_cache_path.name + "_all_fa")
    weight_cache_path.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        paged_attention_config=paged_attention_config,
        use_paged_kv_cache=False,
    )
    return model, args


def _build_partial_rope_cos_sin_torch(positions: torch.Tensor):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos_ref, sin_ref = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    return cos_ref.unsqueeze(0), sin_ref.unsqueeze(0)


def _build_partial_rope_cos_sin_tt(mesh, positions: torch.Tensor):
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


def _embed_tokens_cpu(state_dict_hf: dict, token_ids: torch.Tensor) -> torch.Tensor:
    emb_w = state_dict_hf["model.language_model.embed_tokens.weight"]
    return emb_w[token_ids].to(torch.bfloat16)


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
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


def _do_prefill_paged(model, mesh, args, x_prefill: torch.Tensor, page_table_tt):
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


@pytest.mark.hardware
def test_qwen36_64L_all_fullattn_decode_perf(bh_glx_mesh):
    """Headline: tok/s/user for 64-layer all-FA decode (the experiment)."""
    from transformers import AutoTokenizer

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    with open(_LLAMA70B_PROMPT_FILE) as f:
        prompts = json.load(f)
    prompt = prompts[0]["prompt"]
    print(f"[all-fa-perf] prompt: {prompt!r}")
    ids = tok(prompt, return_tensors="pt").input_ids
    T_prompt = int(ids.shape[-1])
    print(f"[all-fa-perf] prompt token count = {T_prompt}")
    if T_prompt > _T_PREFILL:
        ids = ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
    pad_len = _T_PREFILL - T_prompt
    ids_padded = torch.cat([ids, torch.zeros(1, pad_len, dtype=ids.dtype)], dim=1) if pad_len > 0 else ids
    print(f"[all-fa-perf] padded to T={ids_padded.shape[-1]}")

    print(f"[all-fa-perf] loading HF state dict ...")
    t0 = time.time()
    orig_sd = _load_full_state_dict(_SNAPSHOT)
    print(f"[all-fa-perf] HF state dict loaded in {time.time() - t0:.1f}s; {len(orig_sd)} keys")

    print(f"[all-fa-perf] synthesizing all-FA state dict (round-robin layers {_FA_LAYER_INDICES})...")
    t0 = time.time()
    sd = _synthesize_all_fa_state_dict(orig_sd)
    print(f"[all-fa-perf] synthesized in {time.time() - t0:.1f}s; {len(sd)} keys")
    del orig_sd  # free memory

    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged(bh_glx_mesh, sd, _PATTERN, _N_LAYERS, paged_attention_config)
    print(f"[all-fa-perf] 64-layer all-FA TT model built (paged KV cache)")
    page_table_tt = _build_paged_page_table(bh_glx_mesh, args, paged_attention_config)

    # ---- TT PREFILL ----
    x_prefill = _embed_tokens_cpu(sd, ids_padded[:, :_T_PREFILL])
    print(f"[all-fa-perf] CPU embed shape: {list(x_prefill.shape)}")
    t0 = time.time()
    _ = _do_prefill_paged(model, bh_glx_mesh, args, x_prefill, page_table_tt)
    ttnn.synchronize_device(bh_glx_mesh)
    prefill_ms = (time.time() - t0) * 1000
    print(f"[all-fa-perf] prefill done in {prefill_ms:.1f} ms")

    # ---- DECODE in trace ----
    cur_pos_int = T_prompt
    cur_pos_torch = torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32)
    cur_pos_tt = ttnn.from_torch(
        cur_pos_torch,
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    rot_idxs_tt = model.rope_setup.get_qwen36_rm_rot_idxs(cur_pos_int, on_host=False)
    first_decode_token = 198  # arbitrary; semantics don't matter
    init_tok = torch.full((1, 1, 1, 32), first_decode_token, dtype=torch.int32)
    tt_out_tok = ttnn.from_torch(
        init_tok,
        device=bh_glx_mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    model.set_trace_decode_mode(True)

    def _run_decode_intrace():
        cos, sin = model.rope_setup.get_qwen36_rm_rot_mats(rot_idxs_tt)
        x_emb_flat = ttnn.embedding(
            tt_out_tok,
            model.embd.weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        x_emb_3d = ttnn.slice(
            x_emb_flat,
            [0, 0, 0],
            [1, 1, x_emb_flat.shape[-1]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_emb_flat.deallocate(True)
        x_emb = ttnn.unsqueeze_to_4D(x_emb_3d)
        lm_head_out = model.forward(
            x_emb,
            current_pos=cur_pos_tt,
            rot_mats=(cos, sin),
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
        logits = lm_head_out[0] if isinstance(lm_head_out, list) else lm_head_out
        num_links = min(3, model.model_config["GALAXY_NUM_LINKS"])
        logits_bf16 = ttnn.typecast(logits, dtype=ttnn.bfloat16)
        logits_full = model.tt_ccl.line_all_gather(
            logits_bf16,
            dim=3,
            num_links=num_links,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits_bf16.deallocate(True)
        logits_untiled = ttnn.untilize(logits_full, use_multicore=True)
        logits_full.deallocate(True)
        V_gathered = logits_untiled.shape[-1]
        logits_row0 = ttnn.slice(
            logits_untiled,
            [0, 0, 0, 0],
            [1, 1, 1, V_gathered],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits_untiled.deallocate(True)
        new_tok = ttnn.argmax(logits_row0, dim=3, keepdim=True, use_multicore=False)
        logits_row0.deallocate(True)
        new_tok_u32 = ttnn.typecast(new_tok, dtype=ttnn.uint32)
        new_tok.deallocate(True)
        ttnn.copy(new_tok_u32, tt_out_tok)
        new_tok_u32.deallocate(True)
        ttnn.plus_one(cur_pos_tt)
        ttnn.plus_one(rot_idxs_tt)

    # --- Compile pass (untraced) to warm caches ---
    print(f"[all-fa-perf] compile pass (untraced) ...")
    t0 = time.time()
    _run_decode_intrace()
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[all-fa-perf] compile pass done in {(time.time() - t0) * 1000:.1f} ms")

    # --- Trace capture ---
    print(f"[all-fa-perf] capturing decode trace ...")
    t0 = time.time()
    tid = ttnn.begin_trace_capture(bh_glx_mesh, cq_id=0)
    _run_decode_intrace()
    ttnn.end_trace_capture(bh_glx_mesh, tid, cq_id=0)
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[all-fa-perf] trace captured in {(time.time() - t0) * 1000:.1f} ms")

    # --- Timed trace replay loop ---
    print(f"[all-fa-perf] running {_DECODE_STEPS} decode steps via trace replay ...")
    step_ms = []
    for i in range(_DECODE_STEPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(bh_glx_mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(bh_glx_mesh)
        step_ms.append((time.perf_counter() - t0) * 1000)
    ttnn.release_trace(bh_glx_mesh, tid)

    # --- Report ---
    import statistics

    mean_ms = statistics.mean(step_ms[2:])  # discard first 2 as warmup
    p50 = statistics.median(step_ms[2:])
    tok_per_s = 1000.0 / mean_ms
    print(f"\n=== ALL-FULL-ATTENTION 64L DECODE PERF ===")
    print(f"  steps timed       : {len(step_ms[2:])}")
    print(f"  mean ms/step      : {mean_ms:.2f}")
    print(f"  median ms/step    : {p50:.2f}")
    print(f"  tok/s/user        : {tok_per_s:.2f}")
    print(f"  raw ms list       : {[round(x, 2) for x in step_ms]}")
    print(f"\nComparisons:")
    print(f"  Qwen3-32B target  : 20.0 ms/step (50 tok/s/u)")
    print(f"  qwen3.6 hybrid V2-DN-TP : 52.6 ms/step (19.0 tok/s/u)")
    print(f"  this run (all-FA) : {mean_ms:.1f} ms/step ({tok_per_s:.2f} tok/s/u)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
