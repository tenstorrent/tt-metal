# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B (Galaxy BH_GLX) batch-1 multi-ISL demo + profiler.

Modeled on ``models/demos/llama3_70b_galaxy/demo/text_demo.py``: runs a single
user (batch 1) through prefill + decode at a parametrized input sequence
length (ISL) and reports a BenchmarkProfiler table — time-to-first-token
(TTFT), prefill throughput, and decode throughput — plus the generated text
for a coherence eyeball.

Why a dedicated demo (vs the existing demo/ files):
  - ``demo/text_demo.py``       builds the llama3_70b model.
  - ``demo/text_qwen_demo.py``  builds the qwen3-32B dense model.
  Neither builds the qwen3.6-27B *DeltaNet* model. This demo builds the real
  qwen3.6 model (hybrid linear/full attention) via the proven construction
  from ``tests/test_decode_perf_intrace.py``.

Prefill runs ONCE (a 2nd in-place ``model.forward(mode="prefill")`` corrupts
state — see the note at the prefill call site), so the reported TTFT is COLD
(includes one-time kernel compilation); a true warm/served TTFT would route
through ``tt/generator.py::prefill_forward_text`` (warmup-compiles every ISL
bucket, then traces). The decode loop IS captured as a trace and replayed (the
production decode path), so decode throughput is the warm/traced steady-state
number.

ISL is selected with ``QWEN36_PERF_T_PREFILL`` (default 128). Values >= 1024
load the matching ``input_data_long_{N}k.json`` Gutenberg-context prompt;
smaller values use ``input_data_questions_prefill_128.json``.

Run (single ISL):

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) HF_MODEL=Qwen/Qwen3.6-27B \\
        MESH_DEVICE=BH_GLX QWEN36_FULLATTN_WO_TUNED=1 QWEN36_DELTA_OP_TUNED=1 \\
        QWEN36_CCL_NUM_LINKS_DELTA=2 QWEN36_PERF_T_PREFILL=4096 \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/demo/text_demo_qwen36.py -v -s
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib

import pytest
import requests
import torch
from safetensors.torch import load_file as load_st

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_PROMPT_DIR = pathlib.Path("models/demos/llama3_70b_galaxy/demo/sample_prompts")
_CONTEXT_CACHE_DIR = pathlib.Path("models/tt_transformers/demo/context_cache")

_B = 1
_T_PREFILL = int(os.environ.get("QWEN36_PERF_T_PREFILL", "128"))
_H = 5120
_N_LAYERS = 64
# Match the llama3_70b_galaxy demo's max_generated_tokens (128).
_DECODE_STEPS = int(os.environ.get("QWEN36_DECODE_STEPS", "128"))
_PATTERN = (["linear_attention"] * 3 + ["full_attention"]) * 16

_PAGED_BLOCK_SIZE = 32
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


# ---------------------------------------------------------------------------
# Prompt loading (shared with tests/test_decode_perf_intrace.py)
# ---------------------------------------------------------------------------
def _load_and_cache_context(context_url: str, cache_dir: pathlib.Path, max_length: int | None = None) -> str:
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
    if t_prefill < 1024:
        with open(_PROMPT_DIR / "input_data_questions_prefill_128.json") as f:
            return json.load(f)[0]["prompt"]
    k = t_prefill // 1024
    prompt_file = _PROMPT_DIR / f"input_data_long_{k}k.json"
    if not prompt_file.exists():
        raise FileNotFoundError(f"no long-context prompt file for ISL {t_prefill} (looked for {prompt_file})")
    with open(prompt_file) as f:
        entry = json.load(f)[0]
    prompt = entry["prompt"]
    context_url = entry.get("context")
    if context_url:
        max_length = max(entry.get("max_length") or 0, t_prefill * 6)
        context_text = _load_and_cache_context(context_url, _CONTEXT_CACHE_DIR, max_length=max_length)
        prompt = "```" + context_text + "```\n\n" + prompt
    return prompt


# ---------------------------------------------------------------------------
# Model construction + prefill/decode helpers (from test_decode_perf_intrace.py)
# ---------------------------------------------------------------------------
def _load_full_state_dict(snapshot_dir: pathlib.Path) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    files = sorted(set(weight_map.values()))
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        sd.update(load_st(str(snapshot_dir / fn)))
    return sd


def _build_paged_page_table(mesh, args, paged_attention_config):
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )
    return ttnn.from_torch(
        page_table,
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, None), mesh_shape=args.cluster_shape),
    )


def _build_tt_model_paged(mesh, state_dict, pattern, n_layers, paged_attention_config):
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


def _build_partial_rope_cos_sin_tt(mesh, positions: torch.Tensor):
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
    emb_w = state_dict_hf["model.language_model.embed_tokens.weight"]
    return emb_w[token_ids].to(torch.bfloat16)


def _send_col_sharded_hidden(t: torch.Tensor, mesh, args):
    B, T, H = t.shape
    return ttnn.from_torch(
        t.reshape(1, 1, T, H),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=args.cluster_shape),
    )


def _build_prefill_inputs(model, mesh, args, x_prefill, page_table_tt):
    """Build PERSISTENT device inputs for prefill (so a trace can replay them)."""
    x_tt = _send_col_sharded_hidden(x_prefill, mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(mesh, torch.arange(x_prefill.shape[1], dtype=torch.long))
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return {"x_tt": x_tt, "cos_tt": cos_tt, "sin_tt": sin_tt, "chunk_start_idx_tt": chunk_start_idx_tt}


def _run_prefill(model, page_table_tt, prefill_inputs):
    """Run one prefill forward over the persistent inputs. Idempotent for a
    fixed prompt (writes user_id=0 / chunk_start_idx=0 KV + DeltaNet state), so
    compile / capture / replay all target the same slots — safe to repeat."""
    # Chunked-prefill path (long context): GDN layers processed in sequence-chunks
    # of QWEN36_PREFILL_CHUNK tokens, carrying conv+recurrent state; full-attn single-pass.
    _pf_chunk = os.environ.get("QWEN36_PREFILL_CHUNK")
    if _pf_chunk:
        return model.prefill_chunked(
            prefill_inputs["x_tt"],
            (prefill_inputs["cos_tt"], prefill_inputs["sin_tt"]),
            gdn_chunk_size=int(_pf_chunk),
            user_id=0,
            page_table=page_table_tt,
            kv_cache=None,
            chunk_start_idx_tensor=prefill_inputs["chunk_start_idx_tt"],
            batch_size=1,
        )
    return model.forward(
        prefill_inputs["x_tt"],
        current_pos=None,
        rot_mats=(prefill_inputs["cos_tt"], prefill_inputs["sin_tt"]),
        user_id=0,
        mode="prefill",
        page_table=page_table_tt,
        chunk_page_table=None,
        chunk_start_idx=prefill_inputs["chunk_start_idx_tt"],
        start_pos=0,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
    )


def _gather_prefill_logits_to_cpu(tt_prefill_out, mesh, args, model, last_token_idx: int):
    x_norm, _ = model.norm(tt_prefill_out, res=None, mode="prefill")
    x_norm_last = x_norm[:, :, last_token_idx : last_token_idx + 1, :] if last_token_idx >= 0 else x_norm
    lm_head_out = model.lm_head(x_norm_last, None, mode="prefill")
    out0 = lm_head_out[0] if isinstance(lm_head_out, list) else lm_head_out
    logits_torch = ttnn.to_torch(
        out0, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(3, 0), mesh_shape=args.cluster_shape)
    )
    n_cols = args.cluster_shape[1]
    logits_torch = logits_torch[: logits_torch.shape[0] // n_cols]
    while logits_torch.dim() > 3 and logits_torch.shape[0] == 1:
        logits_torch = logits_torch.squeeze(0)
    if logits_torch.dim() == 3:
        return logits_torch[:, 0:1, : args.vocab_size]
    return logits_torch[..., : args.vocab_size]


@pytest.mark.hardware
def test_qwen36_demo_batch1(bh_glx_mesh):
    """Batch-1 prefill+decode demo with BenchmarkProfiler (TTFT / prefill / decode)."""
    from transformers import AutoTokenizer

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    profiler = BenchmarkProfiler()
    profiler.start("run")

    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)

    profiler.start("loading_inputs")
    prompt = _load_prompt_for_isl(_T_PREFILL)
    ids = tok(prompt, return_tensors="pt").input_ids
    T_prompt = int(ids.shape[-1])
    if T_prompt > _T_PREFILL:
        ids = ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
    pad_len = _T_PREFILL - T_prompt
    ids_padded = torch.cat([ids, torch.zeros(1, pad_len, dtype=ids.dtype)], dim=1) if pad_len > 0 else ids
    profiler.end("loading_inputs")
    print(f"[demo] ISL={_T_PREFILL}  real prompt tokens={T_prompt}  padded T={ids_padded.shape[-1]}")

    state_dict = _load_full_state_dict(_SNAPSHOT)
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS, paged_attention_config)
    page_table_tt = _build_paged_page_table(bh_glx_mesh, args, paged_attention_config)
    x_prefill = _embed_tokens_cpu(state_dict, ids_padded[:, :_T_PREFILL])

    # ---- PREFILL (single pass — see note) ----
    # We run prefill ONCE. A second in-place ``model.forward(mode="prefill")``
    # corrupts state non-deterministically (verified: 1st pass coherent, 2nd
    # pass garbage), because the raw forward path — unlike the generator's
    # ``prefill_forward_text`` — does not re-copy inputs into fixed device
    # buffers, reset CCL indices, and manage KV/DeltaNet state between passes.
    # So this single pass includes one-time kernel compilation; the reported
    # TTFT is therefore COLD (compile + execute). For a true warm/served TTFT
    # (compile amortized via a warmup sweep), route prefill through
    # ``tt/generator.py::prefill_forward_text`` (which warmup-compiles every
    # ISL bucket then traces) — that is the production path. Decode below IS
    # traced (compile pass + capture + replay), so decode tok/s is warm.
    if hasattr(model, "tt_ccl") and hasattr(model.tt_ccl, "reset_gather_and_buffer_idx"):
        model.tt_ccl.reset_gather_and_buffer_idx()
    prefill_inputs = _build_prefill_inputs(model, bh_glx_mesh, args, x_prefill, page_table_tt)

    # Profiler switch (QWEN36_PREFILL_PROFILE=1): wrap prefill in tracy signposts for a
    # device-trace capture of the prefill region (mirrors llama3_70b_galaxy text_qwen_demo).
    # Run under `python -m tracy ...` to collect; no-op signpost if tracy unavailable.
    _prefill_profile = os.environ.get("QWEN36_PREFILL_PROFILE", "0") == "1"
    if _prefill_profile:
        try:
            from tracy import signpost
        except ImportError:
            signpost = lambda *_a, **_k: None  # noqa: E731
    profiler.start("inference_prefill")
    if _prefill_profile:
        signpost("prefill_start")
    prefill_hidden_tt = _run_prefill(model, page_table_tt, prefill_inputs)
    ttnn.synchronize_device(bh_glx_mesh)
    if _prefill_profile:
        signpost("prefill_stop")
    profiler.end("inference_prefill")

    last_prompt_logits = _gather_prefill_logits_to_cpu(
        prefill_hidden_tt, bh_glx_mesh, args, model, last_token_idx=T_prompt - 1
    )
    first_decode_token = int(last_prompt_logits.reshape(-1)[: args.vocab_size].float().argmax().item())
    print(f"[demo] first decode token = {first_decode_token} ({tok.decode([first_decode_token])!r})")

    # ---- DECODE: capture one trace, replay per step ----
    cur_pos_int = T_prompt
    cur_pos_tt = ttnn.from_torch(
        torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    rot_idxs_tt = model.rope_setup.get_qwen36_rm_rot_idxs(cur_pos_int, on_host=False)
    tt_out_tok = ttnn.from_torch(
        torch.full((1, 1, 1, 32), first_decode_token, dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    # On-device sampling (Qwen3.6 thinking-mode defaults: temp=1.0, top_p=0.95, top_k=20).
    # Replaces greedy argmax in the decode loop — greedy degenerates into repetition on long
    # context. Disable with QWEN36_SAMPLE=0. Params overridable via QWEN36_TEMP/TOP_P/TOP_K.
    _use_sampling = os.environ.get("QWEN36_SAMPLE", "1") != "0"
    tt_sampling = None
    if _use_sampling:
        from models.common.sampling.tt_sampling import TTSampling

        # TTSampling/ttnn.sampling expects k/p/temp as per-user torch tensors of length 32
        # (decode is 32-user packed; the op asserts k.shape == [32]), NOT scalars.
        _bs = 32
        _k = int(os.environ.get("QWEN36_TOP_K", "20"))
        _p = float(os.environ.get("QWEN36_TOP_P", "0.95"))
        _t = float(os.environ.get("QWEN36_TEMP", "1.0"))
        tt_sampling = TTSampling(
            mesh_device=bh_glx_mesh,
            tt_ccl=model.tt_ccl,
            args=args,
            k=torch.tensor([_k] * _bs, dtype=torch.int32),
            p=torch.tensor([_p] * _bs, dtype=torch.float32),
            temp=torch.tensor([_t] * _bs, dtype=torch.float32),
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
        if os.environ.get("QWEN36_DECODE_L1_RESIDUAL", "0") == "1" or os.environ.get("QWEN36_DECODE_32ROW", "0") == "1":
            # 32-row carry (llama70b batch-1 contract): keep all 32 tile-padded rows
            # ([1,1,32,H/4], row 0 = real token) so the decode-mode L1 norm (rms_allgather)
            # has its required (1,1,32,M) shape. NOTE: ttnn.reshape is a VIEW aliasing
            # x_emb_flat — do NOT deallocate x_emb_flat (it would free the shared buffer →
            # use-after-free → NaN). Mirrors the perf harness 32-row branch.
            x_emb = ttnn.reshape(x_emb_flat, ttnn.Shape([1, 1, x_emb_flat.shape[-2], x_emb_flat.shape[-1]]))
        else:
            x_emb_3d = ttnn.slice(
                x_emb_flat, [0, 0, 0], [1, 1, x_emb_flat.shape[-1]], memory_config=ttnn.DRAM_MEMORY_CONFIG
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
        if tt_sampling is not None:
            # On-device top-k/top-p/temperature sampling; writes the sampled token in-place
            # into tt_out_tok (the next decode step's embedding input), exactly like
            # llama3_70b_galaxy/demo/demo_qwen_decode.py.
            tt_sampling(logits, tt_out_tok=tt_out_tok)
        else:
            # Greedy fallback (QWEN36_SAMPLE=0): gather full vocab + argmax.
            num_links = min(3, model.model_config["GALAXY_NUM_LINKS"])
            logits_bf16 = ttnn.typecast(logits, dtype=ttnn.bfloat16)
            logits_full = model.tt_ccl.line_all_gather(
                logits_bf16, dim=3, num_links=num_links, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            logits_bf16.deallocate(True)
            logits_untiled = ttnn.untilize(logits_full, use_multicore=True)
            logits_full.deallocate(True)
            V_gathered = logits_untiled.shape[-1]
            logits_row0 = ttnn.slice(
                logits_untiled, [0, 0, 0, 0], [1, 1, 1, V_gathered], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            logits_untiled.deallocate(True)
            tok_1x1 = ttnn.argmax(logits_row0, dim=3, keepdim=True, use_multicore=True)
            logits_row0.deallocate(True)
            if isinstance(tok_1x1, list):
                tok_1x1 = tok_1x1[0]
            tok_broadcast = ttnn.repeat(tok_1x1, ttnn.Shape((1, 1, 1, 32)))
            tok_1x1.deallocate(True)
            ttnn.copy(input_a=tok_broadcast, input_b=tt_out_tok)
            tok_broadcast.deallocate(True)
        ttnn.plus_one(
            cur_pos_tt,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            skip_negative_entries=True,
        )
        ttnn.plus_one(
            rot_idxs_tt, sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))])
        )

    # compile pass + reset buffers
    _run_decode_intrace()
    ttnn.synchronize_device(bh_glx_mesh)
    cur_pos_reset = ttnn.from_torch(
        torch.tensor([cur_pos_int] * args.max_batch_size, dtype=torch.int32),
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    ttnn.copy_host_to_device_tensor(cur_pos_reset, cur_pos_tt)
    rot_idxs_reset = model.rope_setup.get_qwen36_rm_rot_idxs(cur_pos_int, on_host=True)
    ttnn.copy_host_to_device_tensor(rot_idxs_reset, rot_idxs_tt)
    tt_out_tok_reset = ttnn.from_torch(
        torch.full((1, 1, 1, 32), first_decode_token, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)

    if hasattr(model, "tt_ccl") and hasattr(model.tt_ccl, "reset_gather_and_buffer_idx"):
        model.tt_ccl.reset_gather_and_buffer_idx()
    ttnn.synchronize_device(bh_glx_mesh)
    trace_id = ttnn.begin_trace_capture(bh_glx_mesh, cq_id=0)
    _run_decode_intrace()
    ttnn.end_trace_capture(bh_glx_mesh, trace_id, cq_id=0)
    ttnn.synchronize_device(bh_glx_mesh)
    ttnn.copy_host_to_device_tensor(cur_pos_reset, cur_pos_tt)
    ttnn.copy_host_to_device_tensor(rot_idxs_reset, rot_idxs_tt)
    ttnn.copy_host_to_device_tensor(tt_out_tok_reset, tt_out_tok)
    ttnn.synchronize_device(bh_glx_mesh)

    generated_ids = [first_decode_token]
    profiler.start("inference_decode")
    for step in range(_DECODE_STEPS):
        ttnn.execute_trace(bh_glx_mesh, trace_id, cq_id=0, blocking=True)
        tok_t = ttnn.to_torch(ttnn.get_device_tensors(tt_out_tok)[0])
        generated_ids.append(int(tok_t.reshape(-1)[0].item()))
    profiler.end("inference_decode")
    try:
        ttnn.release_trace(bh_glx_mesh, trace_id)
    except Exception:
        pass

    # ---- Warm prefill TTFT (compile EXCLUDED), matching the llama3_70b_galaxy demo ----
    # The timed prefill above was COLD (first run compiles every kernel). Re-run prefill once
    # more with warm kernels for a compile-excluded TTFT comparable to traced demos. Done AFTER
    # decode so this (state-overwriting) re-run cannot affect decode coherence; output discarded.
    ttft_warm_s = None
    try:
        if hasattr(model, "tt_ccl") and hasattr(model.tt_ccl, "reset_gather_and_buffer_idx"):
            model.tt_ccl.reset_gather_and_buffer_idx()
        warm_inputs = _build_prefill_inputs(model, bh_glx_mesh, args, x_prefill, page_table_tt)
        profiler.start("inference_prefill_warm")
        warm_hidden = _run_prefill(model, page_table_tt, warm_inputs)
        ttnn.synchronize_device(bh_glx_mesh)
        profiler.end("inference_prefill_warm")
        ttnn.deallocate(warm_hidden)
        ttft_warm_s = profiler.get_duration("inference_prefill_warm")
    except Exception as _e:
        print(f"[demo] warm prefill timing skipped: {str(_e)[:140]}")

    profiler.end("run")

    # ---- Profiler summary (llama70b-style derivations) ----
    ttft_s = profiler.get_duration("inference_prefill")  # cold prefill (incl. compile) = TTFT
    decode_total_s = profiler.get_duration("inference_decode")
    mean_decode_s = decode_total_s / _DECODE_STEPS
    prefill_tok_s = _T_PREFILL / ttft_s
    decode_tok_s_user = _DECODE_STEPS / decode_total_s

    output_text = tok.decode(generated_ids, skip_special_tokens=False)
    print()
    print("=" * 80)
    print(f"QWEN3.6-27B BH_GLX DEMO — batch 1, ISL={_T_PREFILL}, decode_steps={_DECODE_STEPS}")
    print("=" * 80)
    print(f"  real prompt tokens          : {T_prompt}")
    print(f"  TTFT (cold, incl. compile)  : {ttft_s * 1000:9.1f} ms")
    if ttft_warm_s is not None:
        print(f"  TTFT (warm, compile excl.)  : {ttft_warm_s * 1000:9.1f} ms   <- compare vs traced demos")
        print(f"  prefill throughput (warm)   : {_T_PREFILL / ttft_warm_s:9.1f} tok/s")
    print(f"  prefill throughput          : {prefill_tok_s:9.1f} tok/s")
    print(f"  decode latency / step       : {mean_decode_s * 1000:9.2f} ms")
    print(f"  decode throughput / user    : {decode_tok_s_user:9.2f} tok/s/user")
    print(f"  full demo runtime           : {profiler.get_duration('run'):9.1f} s")
    print("=" * 80)
    print(f"GENERATED ({len(generated_ids)} tokens): {output_text!r}")
    print("=" * 80)

    measurements = {
        "isl": _T_PREFILL,
        "real_prompt_tokens": T_prompt,
        "ttft_s_cold_incl_compile": ttft_s,
        "ttft_s_warm_compile_excl": ttft_warm_s,
        "prefill_tok_s": prefill_tok_s,
        "decode_ms_per_step": mean_decode_s * 1000,
        "decode_tok_s_user": decode_tok_s_user,
    }
    out_path = pathlib.Path(f"/tmp/qwen36_demo_isl_{_T_PREFILL}.json")
    out_path.write_text(json.dumps(measurements, indent=2))
    print(f"[demo] measurements written to {out_path}")

    n_alpha = sum(c.isalpha() for c in output_text)
    assert n_alpha >= 5, f"generated text has <5 alpha chars: {output_text!r}"
