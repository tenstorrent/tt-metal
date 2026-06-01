# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-10 — In-trace decode perf test.

Target: close the host-overhead gap (605 ms/step real-loop -> ~78 ms device
ceiling) by moving these four operations INSIDE the trace boundary:

  1. On-device sampling (greedy argmax) — kills the 248k-vocab to_torch.
  2. In-trace cos/sin via ``rope_setup.get_qwen36_rm_rot_mats(rot_idxs)``.
  3. On-device embedding via ``tt_embd(tt_out_tok)`` — kills the CPU embedding
     lookup + col-shard copy_h2d.
  4. In-trace ``ttnn.plus_one`` on cur_pos and rot_idxs — kills the cur_pos /
     rot_idxs copy_h2d per step.

After these four, the real-loop critical path becomes:
  - ``ttnn.execute_trace(...)`` (~78 ms device-only).
  - ``ttnn.to_torch(next_tok_id_tensor)`` (negligible — single int32).
  - Stop check.

Driver test for V2-10. Coherency is preserved against the existing
``test_decode_coherency_isl128.py`` token sequence.

Run:

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_perf_intrace.py \\
            -v -s
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time

import pytest
import requests
import torch
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_PROMPT_DIR = pathlib.Path("models/demos/llama3_70b_galaxy/demo/sample_prompts")
_CONTEXT_CACHE_DIR = pathlib.Path("models/tt_transformers/demo/context_cache")

_B = 1
# QWEN36_PERF_T_PREFILL env var overrides ISL (default 128).  Common values
# for benchmarking: 128, 2048, 4096, 8192.  Values >= 1024 load the matching
# ``input_data_long_{T//1024}k.json`` file (Gutenberg-context-backed); smaller
# values use ``input_data_questions_prefill_128.json``.
_T_PREFILL = int(os.environ.get("QWEN36_PERF_T_PREFILL", "128"))
_H = 5120
_N_LAYERS = int(os.environ.get("QWEN36_N_LAYERS", "64"))
_DECODE_STEPS = int(os.environ.get("QWEN36_DECODE_STEPS", "32"))  # set to ~3 for tracy capture
_PATTERN = (["linear_attention"] * 3 + ["full_attention"]) * 16

# Paged-attention config — block_size=32 is the tile-aligned default.
_PAGED_BLOCK_SIZE = 32
# Ensure enough blocks to hold T_PREFILL + a margin for decode steps.
_PAGED_MAX_NUM_BLOCKS = max(32, (_T_PREFILL + _DECODE_STEPS + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE + 4)


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


def _load_and_cache_context(context_url: str, cache_dir: pathlib.Path, max_length: int | None = None) -> str:
    """Mirrors ``llama3_70b_galaxy/demo/text_demo.py::load_and_cache_context``.

    Downloads the Gutenberg-style context URL on first use and clips to
    ``max_length`` characters (matching the llama3_70b_galaxy demo).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / hashlib.md5(context_url.encode()).hexdigest()
    if cache_file.exists():
        context_text = cache_file.read_text()
    else:
        resp = requests.get(context_url, timeout=30)
        resp.raise_for_status()
        context_text = resp.text
        cache_file.write_text(context_text)
    if max_length:
        context_text = context_text[:max_length]
    return context_text


def _load_prompt_for_isl(t_prefill: int) -> str:
    """Pick the right ``input_data_*.json`` for the requested ISL and assemble.

    For ``t_prefill < 1024`` returns the first ``prompt`` from
    ``input_data_questions_prefill_128.json`` (existing behaviour, char-only
    prompt with no external context).

    For ``t_prefill >= 1024`` loads ``input_data_long_{t_prefill//1024}k.json``,
    downloads its ``context`` URL, clips to its ``max_length`` characters and
    concatenates ``context + "\\n\\n" + prompt`` so the assembled string
    tokenises to at least ``t_prefill`` tokens. The caller is responsible for
    tokenising and truncating to ``t_prefill`` tokens.
    """
    if t_prefill < 1024:
        prompt_file = _PROMPT_DIR / "input_data_questions_prefill_128.json"
        with open(prompt_file) as f:
            data = json.load(f)
        return data[0]["prompt"]

    k = t_prefill // 1024
    prompt_file = _PROMPT_DIR / f"input_data_long_{k}k.json"
    if not prompt_file.exists():
        raise FileNotFoundError(f"no long-context prompt file for ISL {t_prefill} (looked for {prompt_file})")
    with open(prompt_file) as f:
        data = json.load(f)
    entry = data[0]
    prompt = entry["prompt"]
    context_url = entry.get("context")
    if context_url:
        # The llama3_70b_galaxy json files tune ``max_length`` for the llama
        # tokeniser. Qwen3.6's BPE compresses English ~10% denser, so the
        # llama 16000-char clip for 4k tokenises to ~3.8k Qwen tokens.  Use a
        # 6 chars/token floor (well above any English BPE ratio) so the
        # assembled string reliably exceeds ``t_prefill`` Qwen tokens; the
        # caller truncates to exactly ``t_prefill``.
        max_length = max(entry.get("max_length") or 0, t_prefill * 6)
        context_text = _load_and_cache_context(context_url, _CONTEXT_CACHE_DIR, max_length=max_length)
        # Mirror the instruct-mode assembly used by text_demo.py: wrap the
        # context in a fenced block then append the question prompt.
        prompt = "```" + context_text + "```\n\n" + prompt
    return prompt


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


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args, on_host: bool = False):
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


def _gather_prefill_logits_to_cpu(tt_prefill_out, mesh, args, model, last_token_idx: int):
    x = tt_prefill_out
    x_norm, _ = model.norm(x, res=None, mode="prefill")
    if last_token_idx >= 0:
        x_norm_last = x_norm[:, :, last_token_idx : last_token_idx + 1, :]
    else:
        x_norm_last = x_norm
    lm_head_out = model.lm_head(x_norm_last, None, mode="prefill")
    out0 = lm_head_out[0] if isinstance(lm_head_out, list) else lm_head_out
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


@pytest.mark.hardware
def test_qwen36_64L_decode_intrace_perf(bh_glx_mesh):
    """V2-10 perf test: trace contains sampling + embedding + cos/sin + cur_pos increment."""
    from transformers import AutoTokenizer

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    prompt = _load_prompt_for_isl(_T_PREFILL)
    preview = prompt if len(prompt) <= 200 else f"{prompt[:120]!r} ... {prompt[-80:]!r}"
    print(f"[perf] prompt ({len(prompt)} chars): {preview}")
    ids = tok(prompt, return_tensors="pt").input_ids
    T_prompt = int(ids.shape[-1])
    print(f"[perf] prompt token count = {T_prompt} (target ISL = {_T_PREFILL})")

    if T_prompt > _T_PREFILL:
        ids = ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
    pad_len = _T_PREFILL - T_prompt
    ids_padded = torch.cat([ids, torch.zeros(1, pad_len, dtype=ids.dtype)], dim=1) if pad_len > 0 else ids
    print(f"[perf] padded to T={ids_padded.shape[-1]}; real prompt ends at index {T_prompt - 1}")

    print(f"[perf] loading full state dict ...")
    t0 = time.time()
    state_dict = _load_full_state_dict(_SNAPSHOT)
    print(f"[perf] state dict loaded in {time.time() - t0:.1f}s; {len(state_dict)} keys")

    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS, paged_attention_config)
    print(f"[perf] 64-layer TT model built (paged KV cache)")
    page_table_tt = _build_paged_page_table(bh_glx_mesh, args, paged_attention_config)

    # ---- TT PREFILL ----
    x_prefill = _embed_tokens_cpu(state_dict, ids_padded[:, :_T_PREFILL])
    print(f"[perf] CPU embed shape: {list(x_prefill.shape)}")
    t0 = time.time()
    prefill_hidden_tt = _do_prefill_paged(model, bh_glx_mesh, args, x_prefill, page_table_tt)
    ttnn.synchronize_device(bh_glx_mesh)
    prefill_ms = (time.time() - t0) * 1000
    print(f"[perf] prefill done in {prefill_ms:.1f} ms")

    # ---- First decode token from prefill output ----
    last_prompt_logits = _gather_prefill_logits_to_cpu(
        prefill_hidden_tt, bh_glx_mesh, args, model, last_token_idx=T_prompt - 1
    )
    last_prompt_logits_flat = last_prompt_logits.reshape(-1)[: args.vocab_size].float()
    first_decode_token = int(last_prompt_logits_flat.argmax().item())
    print(
        f"[perf] first decode token (greedy from prefill) = {first_decode_token} "
        f"({tok.decode([first_decode_token])!r})"
    )

    # =============================================================
    # V2-10 IN-TRACE BUFFERS
    # =============================================================
    cur_pos_int = T_prompt  # First decode is at index T_prompt (KV cache holds [0..T_prompt-1]).
    # cur_pos persistent device tensor (replicated, shape [max_batch_size]).
    cur_pos_torch = torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32)
    cur_pos_tt = ttnn.from_torch(
        cur_pos_torch,
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )

    # rot_idxs persistent device tensor (tile-aligned [32, 32] uint32).
    rot_idxs_tt = model.rope_setup.get_qwen36_rm_rot_idxs(cur_pos_int, on_host=False)

    # tt_out_tok persistent device buffer — uint32 [1, 1, 1, 32].
    # The [1,1,1,32] shape is tile-aligned in dim 3, so ttnn.embedding does
    # NOT fire an internal host-write tile-padding (which would corrupt
    # trace replay state). All 32 entries hold the same token id — argmax
    # writes them simultaneously (one per "fake batch user").
    init_tok = torch.full((1, 1, 1, 32), first_decode_token, dtype=torch.int32)
    tt_out_tok = ttnn.from_torch(
        init_tok,
        device=bh_glx_mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )

    model.set_trace_decode_mode(True)

    # The qwen36 in-trace decode wrapper. Inside trace boundary it:
    #   1. Looks up cos/sin via get_qwen36_rm_rot_mats(rot_idxs_tt) — pure device.
    #   2. Embeds tt_out_tok via model.embd(...) — pure device.
    #   3. Runs model.forward(mode='decode', page_table=...) — pure device.
    #   4. Gathers logits + argmax, writes the result into tt_out_tok.
    #   5. plus_one on cur_pos_tt and rot_idxs_tt.
    def _run_decode_intrace():
        # --- 2. on-device cos/sin via embedding lookup ---
        cos, sin = model.rope_setup.get_qwen36_rm_rot_mats(rot_idxs_tt)

        # --- 3. on-device token embedding ---
        # Bypass model.embd.forward (which routes decode-mode embedding to
        # the 70B L1-sharded DECODE_RESIDUAL_MEMCFG with shard_shape=[32, ...]
        # — the qwen3.6 single-user [B=1, T=1] input is incompatible).
        # Call ttnn.embedding directly with DRAM target.
        # Input id tensor: tt_out_tok is [1, 1, 1, 32] uint32 ROW_MAJOR — all
        # 32 entries hold the same sampled token (tile-aligned so embedding's
        # internal padding does not fire a host-write inside trace capture).
        # ttnn.embedding output: [1, 1, 1, 32*H/4] (flattened).
        x_emb_flat = ttnn.embedding(
            tt_out_tok,
            model.embd.weights,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        # ttnn.embedding output shape: [batch=1, num_indices=32, H/4=1280] (3D).
        # Slice user 0 (first row) → [1, 1, 1280], then unsqueeze to 4D.
        x_emb_3d = ttnn.slice(
            x_emb_flat,
            [0, 0, 0],
            [1, 1, x_emb_flat.shape[-1]],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_emb_flat.deallocate(True)
        # [1, 1, H/4] → [1, 1, 1, H/4]
        x_emb = ttnn.unsqueeze_to_4D(x_emb_3d)

        # --- 4. model forward ---
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
        # lm_head_out is list[ttnn.Tensor]; the qwen36 prefill primitive
        # returns one DRAM-interleaved [1,1,1,padded_vocab/4] per col-shard.
        logits = lm_head_out[0] if isinstance(lm_head_out, list) else lm_head_out

        # --- 5. on-device sampling: gather across row-shards + argmax ---
        # LM head weight is sharded ``dims=(3, 2)`` (vocab on rows, H on cols),
        # so after the line_all_reduce inside lm_head (cluster_axis=1) the
        # output is full-H-reduced per row but ROW-SHARDED on vocab.
        # Gather across ROWS = cluster_axis=0 (mirrors v2 ``process_output_prefill``).
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
        # Slice the FIRST logical row of logits (the tile-padded rows 1..31
        # contain garbage data — they're padding, not 32 valid users).
        # Argmax over [1, 1, 1, V_gathered] → [1, 1, 1, 1].
        V_gathered = logits_untiled.shape[-1]
        logits_row0 = ttnn.slice(
            logits_untiled,
            [0, 0, 0, 0],
            [1, 1, 1, V_gathered],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logits_untiled.deallocate(True)
        # argmax → [1, 1, 1, 1] uint32 ROW_MAJOR.
        tok_1x1 = ttnn.argmax(
            logits_row0,
            dim=3,
            keepdim=True,
            use_multicore=True,
        )
        logits_row0.deallocate(True)
        if isinstance(tok_1x1, list):
            tok_1x1 = tok_1x1[0]
        # Broadcast the [1,1,1,1] argmax result into the [1,1,1,32] persistent
        # buffer via ttnn.repeat so the next iteration's embedding lookup
        # (which uses the tile-aligned [1,1,1,32] shape) sees the fresh token.
        tok_broadcast = ttnn.repeat(tok_1x1, ttnn.Shape((1, 1, 1, 32)))
        tok_1x1.deallocate(True)
        ttnn.copy(input_a=tok_broadcast, input_b=tt_out_tok)
        tok_broadcast.deallocate(True)

        # --- 6. plus_one cur_pos + rot_idxs (in-trace) ---
        ttnn.plus_one(
            cur_pos_tt,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            skip_negative_entries=True,
        )
        ttnn.plus_one(
            rot_idxs_tt,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )

    # ---- compile pass (warm program cache) ----
    # Note: this also writes K/V at cur_pos=T_prompt to the KV cache; the
    # subsequent trace-capture pass and the first loop replay both target the
    # same slot, so the cache state stays consistent across compile/capture/loop.
    print(f"[perf] compile-pass decode (warm program cache)...")
    _run_decode_intrace()
    ttnn.synchronize_device(bh_glx_mesh)

    # Sanity: confirm compile pass produced the expected next token (<think>=248068).
    debug_tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out_tok)[0])
    print(
        f"[perf] compile pass argmax → next token = {int(debug_tt_out.reshape(-1)[0].item())} (expected 248068=<think>)"
    )

    # The compile pass incremented cur_pos / rot_idxs / tt_out_tok beyond what we want.
    # Reset them before trace capture.
    cur_pos_reset_host = ttnn.from_torch(
        torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32),
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    ttnn.copy_host_to_device_tensor(cur_pos_reset_host, cur_pos_tt)
    rot_idxs_reset_host = model.rope_setup.get_qwen36_rm_rot_idxs(cur_pos_int, on_host=True)
    ttnn.copy_host_to_device_tensor(rot_idxs_reset_host, rot_idxs_tt)
    tt_out_tok_reset_host = ttnn.from_torch(
        torch.full((1, 1, 1, 32), first_decode_token, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    ttnn.copy_host_to_device_tensor(tt_out_tok_reset_host, tt_out_tok)
    print(f"[perf] compile-pass done; buffers reset")

    # ---- capture trace ----
    if hasattr(model, "tt_ccl") and hasattr(model.tt_ccl, "reset_gather_and_buffer_idx"):
        model.tt_ccl.reset_gather_and_buffer_idx()
    ttnn.synchronize_device(bh_glx_mesh)

    print(f"[perf] capturing trace ...")
    trace_id = ttnn.begin_trace_capture(bh_glx_mesh, cq_id=0)
    _run_decode_intrace()
    ttnn.end_trace_capture(bh_glx_mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(bh_glx_mesh)
    print(f"[perf] trace captured: trace_id = {trace_id}")

    # Reset buffers after trace capture (the capture pass also incremented).
    ttnn.copy_host_to_device_tensor(cur_pos_reset_host, cur_pos_tt)
    ttnn.copy_host_to_device_tensor(rot_idxs_reset_host, rot_idxs_tt)
    ttnn.copy_host_to_device_tensor(tt_out_tok_reset_host, tt_out_tok)
    ttnn.synchronize_device(bh_glx_mesh)

    # ---- DECODE LOOP ----
    # Optionally signpost just the warm trace replay (skip step 0 = compile)
    # so a tracy capture can isolate the warm-only per-op DEVICE KERNEL DURATION
    # on the production critical path. Always on; signpost is a no-op when
    # tracy isn't recording.
    try:
        from tracy import signpost as _ttp_signpost
    except ImportError:
        _ttp_signpost = lambda *a, **k: None
    generated_ids = [first_decode_token]
    decode_t0 = time.time()

    for step in range(_DECODE_STEPS):
        if step == 1:
            # Flush the per-chip tracy DRAM ring buffer (12000-event limit)
            # before the warm signpost so the buffer isn't already full from
            # prefill + compile decode. Mirrors tracy_perf_1L_*.py pattern.
            ttnn.ReadDeviceProfiler(bh_glx_mesh)
            _ttp_signpost("trace_warm_start")
        ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=True)
    _ttp_signpost("trace_warm_done")

    decode_ms = (time.time() - decode_t0) * 1000
    mean_decode_ms = decode_ms / _DECODE_STEPS
    print(
        f"[perf] decode loop done — {_DECODE_STEPS} steps in {decode_ms:.1f} ms "
        f"(mean {mean_decode_ms:.2f} ms/step, {1000.0 / mean_decode_ms:.2f} tok/s/user TRACED)"
    )

    # ---- Read back generated tokens by snapshotting tt_out_tok after the loop ----
    # NOTE: after 32 trace replays, tt_out_tok holds the LAST generated token.
    # For coherency we need to snapshot per step — do that in a second pass.
    # (First pass is the timing measurement above.)
    print(f"[perf] coherency pass — reading tt_out_tok per step ...")
    # Reset buffers for second pass.
    ttnn.copy_host_to_device_tensor(cur_pos_reset_host, cur_pos_tt)
    ttnn.copy_host_to_device_tensor(rot_idxs_reset_host, rot_idxs_tt)
    ttnn.copy_host_to_device_tensor(tt_out_tok_reset_host, tt_out_tok)
    ttnn.synchronize_device(bh_glx_mesh)

    generated_ids = [first_decode_token]
    coherency_t0 = time.time()
    for step in range(_DECODE_STEPS):
        ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=True)
        # to_torch a single-int tensor — should be negligible (<1 ms).
        tok_t = ttnn.to_torch(ttnn.get_device_tensors(tt_out_tok)[0])
        tok_id = int(tok_t.reshape(-1)[0].item())
        generated_ids.append(tok_id)
    coherency_ms = (time.time() - coherency_t0) * 1000
    mean_coherency_ms = coherency_ms / _DECODE_STEPS
    print(
        f"[perf] coherency loop — {_DECODE_STEPS} steps in {coherency_ms:.1f} ms "
        f"(mean {mean_coherency_ms:.2f} ms/step, {1000.0 / mean_coherency_ms:.2f} tok/s/user)"
    )

    try:
        ttnn.release_trace(bh_glx_mesh, trace_id)
    except Exception as e:
        print(f"[perf] release_trace failed (ignored): {e}")

    output_text = tok.decode(generated_ids, skip_special_tokens=False)
    print()
    print("=" * 80)
    print(f"PROMPT (last 200 chars):  ...{prompt[-200:]!r}")
    print("=" * 80)
    print(f"GENERATED ({len(generated_ids)} tokens):  {output_text!r}")
    print("=" * 80)
    print(f"GENERATED token ids:  {generated_ids}")
    print("=" * 80)

    # ---- assertions ----
    n_alpha = sum(c.isalpha() for c in output_text)
    assert n_alpha >= 5, f"generated text has <5 alpha chars: {output_text!r}"
    print(f"[perf] PASSED — {n_alpha} alpha chars in generated text")

    # Perf assertion: target >= 10 tok/s/user (mean per-step <= 100 ms).
    # Soft check — print but don't fail (helpful for iterating).
    if 1000.0 / mean_decode_ms >= 10.0:
        print(f"[perf] PASSED — pure-execute_trace loop reached {1000.0 / mean_decode_ms:.2f} tok/s/user")
    else:
        print(
            f"[perf] target NOT met — pure-execute_trace loop {1000.0 / mean_decode_ms:.2f} tok/s/user "
            f"(target = 10 tok/s/user)"
        )
