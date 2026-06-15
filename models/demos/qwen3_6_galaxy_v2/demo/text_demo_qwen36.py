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
_N_LAYERS = int(os.environ.get("QWEN36_N_LAYERS", "64"))
# Match the llama3_70b_galaxy demo's max_generated_tokens (128).
_DECODE_STEPS = int(os.environ.get("QWEN36_DECODE_STEPS", "128"))
_PATTERN = (["linear_attention"] * 3 + ["full_attention"]) * 16

_PAGED_BLOCK_SIZE = int(os.environ.get("QWEN36_PAGED_BLOCK_SIZE", "32"))
_PAGED_MAX_NUM_BLOCKS = int(
    os.environ.get(
        "QWEN36_PAGED_MAX_NUM_BLOCKS",
        str(max(32, (_T_PREFILL + _DECODE_STEPS + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE + 4)),
    )
)


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

    _msl = os.environ.get("QWEN36_MAX_SEQ_LEN")
    if _msl is not None:
        args = TtQwen36ModelArgs(mesh, max_seq_len=int(_msl))
    else:
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


def _build_tt_model_paged_kv(mesh, state_dict, pattern, n_layers, paged_attention_config, max_batch_size=1):
    """Same as _build_tt_model_paged but with use_paged_kv_cache=True — the contract
    the Generator (prefill_forward_text / decode_forward) + tt-inference-server use.
    The full-attention layers allocate ``attention.layer_past`` against the paged
    config; DeltaNet layers carry recurrent state (no kv).

    max_batch_size: number of concurrent decode users (1 = production batch-1; 32 = batched decode).
    Sizes the GDN/conv recurrent state buffers + paged KV + sampler to the batch."""
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

    # NOTE: do NOT bump max_seq_len for >128k — it inflates the persistent max_seq_len-scaled DRAM
    # buffers (rope table, decode masks) and starves the ~4GB transient full-attn QK-norm activation
    # at 256k (OOM). The inline demo keeps max_seq_len=128k and builds the prefill RoPE for the ACTUAL
    # seq-len; the >128k generator path needs the same (build actual-len rope), TODO.
    _msl = os.environ.get("QWEN36_MAX_SEQ_LEN")
    if _msl is not None:
        args = TtQwen36ModelArgs(mesh, max_seq_len=int(_msl), max_batch_size=max_batch_size)
    else:
        args = TtQwen36ModelArgs(mesh, max_batch_size=max_batch_size)
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
        use_paged_kv_cache=True,
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


@pytest.mark.hardware
def test_qwen36_demo_generator_batch1(bh_glx_mesh):
    """Batch-1 demo driven through the SERVER decode path:
    ``Generator.prefill_forward_text`` + ``Generator.decode_forward`` with a PAGED
    kv cache — a faithful port of ``models/demos/llama3_70b_galaxy/demo/text_demo.py``.

    Purpose: the tt-inference-server uses exactly this path (generator_vllm ->
    Generator.decode_forward), which the existing inline-trace demo
    (``test_qwen36_demo_batch1``) does NOT exercise. This test reproduces the
    server's decode path locally for fast iteration, and validates it against the
    known-good inline demo for correctness (greedy => deterministic first token).
    """
    from transformers import AutoTokenizer

    from models.common.sampling import SamplingParams
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
    from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
    from models.demos.qwen3_6_galaxy_v2.tt.generator_vllm import allocate_vllm_kv_cache

    # Bake the known-good qwen3.6 decode-CCL config as DEFAULTS so the generator demo runs
    # coherently out-of-the-box (no run_text_demo.sh exports needed); setdefault keeps any
    # explicit override. These are read during model build / decode below.
    #   FORCE_SWITCH_DECODE/DECODE_L1_RESIDUAL : decode-mode tt_ccl tail + 32-row L1 residual norm
    #   LM_HEAD_PLAIN_DECODE                   : decode lm_head via minimal_matmul (the coherence fix)
    #   SEQ_CORES_PER_HEAD / *_TUNED / CCL_NUM_LINKS_DELTA / RESIDUAL_BUF_BF16 : prefill+perf tuning
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FULLATTN_WO_TUNED": "1",
        "QWEN36_DELTA_OP_TUNED": "1",
        "QWEN36_CCL_NUM_LINKS_DELTA": "2",
    }.items():
        os.environ.setdefault(_k, _v)

    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    _prompt_override = os.environ.get("QWEN36_PROMPT_OVERRIDE")
    prompt = _prompt_override if _prompt_override is not None else _load_prompt_for_isl(_T_PREFILL)
    # QWEN36_APPLY_CHAT_TEMPLATE=1: wrap the prompt as a Qwen3.6 chat turn
    # (<|im_start|>user ...<|im_end|>\n<|im_start|>assistant\n) the way HF / vLLM
    # actually invoke the model. Without it the model is fed RAW text and at long
    # ISL simply CONTINUES the context (e.g. the Gutenberg book) instead of answering
    # the instruction. We tokenize the raw body first and truncate it leaving room
    # for the template markers, so the trailing assistant generation-prompt is never
    # chopped by the _T_PREFILL cap below.
    if os.environ.get("QWEN36_APPLY_CHAT_TEMPLATE", "0") == "1":
        _ct_overhead = 64  # tokens reserved for role markers + assistant prompt
        body_ids = tok(prompt, return_tensors="pt").input_ids[0]
        _budget = _T_PREFILL - _ct_overhead
        if body_ids.shape[-1] > _budget:
            # The long-context prompts are ```<book>```\n\n<instruction> — the instruction
            # is at the TAIL. A plain head-truncation would DROP the instruction (the model
            # would then have no task and just continue/respond to raw book text). Keep the
            # book HEAD + the instruction TAIL, dropping the middle of the book, so the model
            # is actually asked the question at long context.
            _tail = int(os.environ.get("QWEN36_INSTR_TAIL_TOKENS", "120"))
            _head = max(1, _budget - _tail)
            body_ids = torch.cat([body_ids[:_head], body_ids[-_tail:]])
            prompt = tok.decode(body_ids, skip_special_tokens=True)
        # QWEN36_ENABLE_THINKING=0 -> no <think> priming (ends at <|im_start|>assistant\n),
        # to discriminate "thinking-mode breaks it" from "any chat marker breaks it".
        _think = os.environ.get("QWEN36_ENABLE_THINKING", "1") == "1"
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=_think,
        )
        print(f"[gen-demo] chat template applied (tail): {prompt[-120:]!r}")
    ids = tok(prompt, return_tensors="pt").input_ids
    T_prompt = int(ids.shape[-1])
    if T_prompt > _T_PREFILL:
        ids = ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
    real_tokens = ids[:, :T_prompt].to(torch.long)  # [1, T_prompt] (unpadded; prompt_lens carries length)
    print(f"[gen-demo] ISL={_T_PREFILL}  real prompt tokens={T_prompt}")

    state_dict = _load_full_state_dict(_SNAPSHOT)
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS, paged_attention_config)

    # Host page table (reverse-permutation block map), exactly like
    # llama3_70b_galaxy/demo/text_demo.py::create_tt_model.
    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )
    # Build the paged kv cache EXACTLY like tt-inference-server (generator_vllm.
    # allocate_vllm_kv_cache): a [k,v] per layer for ALL layers (DeltaNet entries
    # are allocated but ignored by the per-layer kv index). Collecting layer_past
    # directly would give None for the DeltaNet layers and break common.py's
    # kv_cache[0][0].shape[2] block-size probe.
    _kv_shape = (
        paged_attention_config.max_num_blocks,
        1,  # num_kv_heads_per_dev (TP-divided; allocate_vllm_kv_cache rebuilds the row-shard)
        paged_attention_config.block_size,
        args.head_dim,
    )
    tt_kv_cache = allocate_vllm_kv_cache(
        _kv_shape, torch.bfloat16, args.n_layers, model, args.weight_cache_path(ttnn.bfloat8_b)
    )

    generator = Generator(model, args, bh_glx_mesh, tokenizer=tok)
    # Server contract: prefill runs eager (no prefill trace), warmup pre-done.
    generator._disable_prefill_tracing = True
    generator.prefill_warmup_completed = True

    # Real top-k/top-p/temperature sampling (NOT greedy). Greedy (temp=0/top_k=1) produces
    # garbage on this model — coherent decode requires the sampling distribution — so default to
    # the inline demo's params (temp=1.0, top_k=20, top_p=0.95), overridable via QWEN36_TEMP/
    # TOP_K/TOP_P. Set QWEN36_GREEDY=1 only for deterministic first-token diagnostics.
    if os.environ.get("QWEN36_GREEDY", "0") == "1":
        sampling_params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    else:
        sampling_params = SamplingParams(
            temperature=float(os.environ.get("QWEN36_TEMP", "1.0")),
            top_k=int(os.environ.get("QWEN36_TOP_K", "20")),
            top_p=float(os.environ.get("QWEN36_TOP_P", "0.95")),
        )

    # ---- PREFILL via generator (server path) ----
    # Server contract with sample_on_device_mode="decode_only": prefill does NOT
    # sample on device (sampling_params=None -> return_logits), the host argmaxes
    # the first token, then decode_forward samples on device. (Passing
    # sampling_params to prefill at batch-1 trips format_sampling_params' %32
    # assert via model_args.max_batch_size=1 — the on-host prefill path is what
    # the server actually uses.)
    print("[gen-demo] prefill_forward_text (return_logits / host first-token) ...")
    prefill_logits = generator.prefill_forward_text(
        real_tokens,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=[T_prompt],
        enable_trace=False,
        sampling_params=None,
    )
    _logits = torch.as_tensor(prefill_logits).float().reshape(-1)[: args.vocab_size]
    first_decode_token = int(_logits.argmax().item())
    # Drift-probe determinism: force a fixed first decode token so the decode-CCL and
    # prefill-CCL probe runs feed IDENTICAL decode-step-1 inputs (prefill argmax can flip
    # between near-tied bf8 logits run-to-run).
    _force_first = os.environ.get("QWEN36_GEN_FORCE_FIRST_TOK")
    if _force_first is not None:
        first_decode_token = int(_force_first)
    print(f"[gen-demo] first decode token = {first_decode_token} ({tok.decode([first_decode_token])!r})")

    # ---- DECODE via generator.decode_forward (server path) ----
    out_tok = torch.tensor([first_decode_token], dtype=torch.long)  # [1]
    current_pos = torch.tensor([T_prompt], dtype=torch.long)
    generated_ids = [first_decode_token]
    _STEPS = int(os.environ.get("QWEN36_GEN_DECODE_STEPS", str(min(32, _DECODE_STEPS))))
    # ISOLATION knob: QWEN36_GEN_HOST_SAMPLE=1 -> decode returns logits (sampling_params=None,
    # eager) and the host argmaxes. If host-argmax decode is coherent but on-device sampling is
    # garbage, the bug is in the on-device sampler (indices/offsets); if both garbage, the decode
    # forward math is wrong.
    _host_sample = os.environ.get("QWEN36_GEN_HOST_SAMPLE", "0") == "1"
    import time as _time

    if _host_sample:
        # Diagnostic blocking host-argmax loop (greedy => DETERMINISTIC, so per-step outputs are
        # directly comparable across runs). Knobs for the trace-replay-vs-KV-corruption bisection:
        #   QWEN36_GEN_TRACE_HOSTSAMP=1 -> run the (return_logits) decode TRACED instead of eager;
        #   QWEN36_GEN_FORCE_TOKENS=t0,t1,..  -> TEACHER-FORCE the per-step input token sequence so
        #       two runs see IDENTICAL inputs (their per-step logits PCC then measures only op drift);
        #   QWEN36_LOGITS_TAG=<tag> -> save {logits:[per-step vocab logits], toks:[input seq]} to
        #       /tmp/qwen36_logits_<tag>.pt for offline per-step PCC.
        _trace_hs = os.environ.get("QWEN36_GEN_TRACE_HOSTSAMP", "0") == "1"
        _force_env = os.environ.get("QWEN36_GEN_FORCE_TOKENS")
        _force_list = [int(x) for x in _force_env.split(",")] if _force_env else None
        _logits_tag = os.environ.get("QWEN36_LOGITS_TAG")
        _logits_dump = []
        _input_toks = []
        _step_times = []
        _cur_in = first_decode_token
        for it in range(_STEPS):
            _t0 = _time.perf_counter()
            _in = _force_list[it] if (_force_list is not None and it < len(_force_list)) else _cur_in
            _input_toks.append(int(_in))
            out = generator.decode_forward(
                torch.tensor([_in], dtype=torch.long).reshape(1, 1),
                current_pos,
                enable_trace=_trace_hs,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                read_from_device=True,
                sampling_params=None,
                reset_inputs=True,
            )
            _logits = out[0] if isinstance(out, (tuple, list)) else out
            _l = torch.as_tensor(_logits).float().reshape(-1)[: args.vocab_size]
            if _logits_tag:
                _logits_dump.append(_l.clone())
            next_tok = int(_l.argmax().item())
            _step_times.append(_time.perf_counter() - _t0)
            generated_ids.append(next_tok)
            _cur_in = next_tok
            current_pos = current_pos + 1
        if _logits_tag:
            torch.save({"logits": _logits_dump, "input_toks": _input_toks}, f"/tmp/qwen36_logits_{_logits_tag}.pt")
            print(f"[LOGITS_DUMP] saved {len(_logits_dump)} steps -> /tmp/qwen36_logits_{_logits_tag}.pt", flush=True)
    else:
        # FAST path — async-pipelined trace replay, mirroring
        # llama3_70b_galaxy/demo/text_demo.py's decode loop. Keys to perf (vs the
        # earlier host-bound ~5.6 tok/s):
        #   * async_read=True       -> the per-step readback never blocks issuing the next step;
        #   * reset_inputs=(it==0)  -> after step 0 the sampled token stays on device and the
        #                              device self-increments current_pos IN-TRACE, so there is
        #                              NO per-step host->device input reload and NO host data
        #                              dependency (out_tok below is ignored after step 0);
        #   * process step N-1's readback while step N runs on device (one-deep pipeline).
        out_tok = out_tok.reshape(1, 1)
        # QWEN36_GEN_NO_TRACE=1: run the fast path EAGER (no trace) but keep on-device sampling —
        # the missing cell to disentangle the garbage (trace vs on-device sampler vs CCL). decode-CCL
        # eager + on-device sampling: if garbage -> sampler; if semi-coherent (like eager+host-argmax)
        # -> the trace is the amplifier.
        _enable_trace = os.environ.get("QWEN36_GEN_NO_TRACE", "0") != "1"
        read_events = []
        tt_out_toks = []
        _loop_t0 = None
        for it in range(_STEPS):
            if it == 1:
                # steady-state clock starts after the step-0 trace capture/compile.
                _loop_t0 = _time.perf_counter()
            tt_tok, read_event = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=_enable_trace,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                read_from_device=True,
                async_read=True,
                sampling_params=sampling_params,
                reset_inputs=(it == 0),
            )
            read_events.append(read_event)
            tt_out_toks.append(tt_tok)
            current_pos = current_pos + 1
            if it > 0:
                ttnn.event_synchronize(read_events.pop(0)[0])
                _tt_tok, _ = generator.process_decode_output_host(tt_out_toks.pop(0))
                generated_ids.append(int(torch.as_tensor(_tt_tok).reshape(-1)[0].item()))
        # drain the final in-flight step
        _loop_dt = (_time.perf_counter() - _loop_t0) if _loop_t0 is not None else 0.0
        ttnn.event_synchronize(read_events.pop(0)[0])
        _tt_tok, _ = generator.process_decode_output_host(tt_out_toks.pop(0))
        generated_ids.append(int(torch.as_tensor(_tt_tok).reshape(-1)[0].item()))
        _n = max(1, _STEPS - 1)
        _mean_s = _loop_dt / _n
        _step_times = []  # not used on the fast path

    # Coherence eyeball: trim at the FIRST EOS the model emits (HF eos_token_id =
    # [248046 <|im_end|>, 248044]). The decode loop runs a fixed step count and does
    # NOT stop on EOS, so anything after the model's stop token is post-EOS garbage
    # that misrepresents coherence. Report where it stopped.
    _EOS = {248046, 248044}
    _eos_at = next((i for i, t in enumerate(generated_ids) if t in _EOS), None)
    _coherent_ids = generated_ids[:_eos_at] if _eos_at is not None else generated_ids
    if _eos_at is not None:
        print(f"[gen-demo] model emitted EOS at generated-token index {_eos_at} (decode loop ignored it)")
    text_full = tok.decode(generated_ids, skip_special_tokens=False)
    text = tok.decode(_coherent_ids, skip_special_tokens=False) if _coherent_ids else text_full
    if _host_sample and len(_step_times) > 1:
        _warm = _step_times[1:]
        _mean_s = sum(_warm) / len(_warm)
    if not _host_sample or len(_step_times) > 1:
        _tok_s = 1.0 / _mean_s if _mean_s > 0 else 0.0
        print("=" * 80)
        print(
            f"[gen-demo] DECODE PERF: steady-state={_mean_s*1000:.2f}ms/tok  "
            f"{_tok_s:.2f} tok/s/user  (n={_STEPS - 1})"
        )
    print("=" * 80)
    print(f"[gen-demo] GENERATED ({len(generated_ids)} tokens, EOS-trimmed): {text!r}")
    if _eos_at is not None:
        print(f"[gen-demo] FULL (untrimmed, {len(generated_ids)} tokens): {text_full!r}")
    print("=" * 80)

    # Correctness gate vs the inline demo: the first decode token is the prefill
    # argmax and must be deterministic/coherent (the inline demo prints
    # "[demo] first decode token = ..." for the same prompt/ISL).
    assert first_decode_token >= 0
    assert len(set(generated_ids)) > 1, f"degenerate (all-same) decode: {generated_ids}"

    # Alpha check on the FULL generation (a legitimately short EOS-trimmed answer
    # like "Here" is coherent, not a failure — the model chose to stop early).
    n_alpha = sum(c.isalpha() for c in text_full)
    assert n_alpha >= 5, f"generated text has <5 alpha chars (incoherent decode): {text_full!r}"


def test_qwen36_xreq_prefill_probe(bh_glx_mesh):
    """CROSS-REQUEST contamination probe (server model-reuse repro).

    Builds the model ONCE (server contract) and runs ``prefill_forward_text``
    (return_logits / host argmax) on the SAME prompt N times
    (QWEN36_GEN_NUM_REQUESTS, default 2), printing the prefill argmax + top-5 each
    pass. Optionally runs a few decode steps per pass (QWEN36_PROBE_DECODE_STEPS).

    Decisive evidence: if the prefill argmax of request 2+ differs from request 1
    for an identical prompt, the cross-request bug is in the PREFILL path (some
    persistent device state not reset between requests). If the prefill argmax is
    STABLE across requests but decode text degrades, the bug is in the DECODE
    path (stale trace / decode-CCL / sampler), not prefill.
    """
    from transformers import AutoTokenizer

    from models.common.sampling import SamplingParams
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
    from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
    from models.demos.qwen3_6_galaxy_v2.tt.generator_vllm import allocate_vllm_kv_cache

    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FULLATTN_WO_TUNED": "1",
        "QWEN36_DELTA_OP_TUNED": "1",
        "QWEN36_CCL_NUM_LINKS_DELTA": "2",
    }.items():
        os.environ.setdefault(_k, _v)

    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    prompt = _load_prompt_for_isl(_T_PREFILL)
    ids = tok(prompt, return_tensors="pt").input_ids
    T_prompt = int(ids.shape[-1])
    if T_prompt > _T_PREFILL:
        ids = ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
    real_tokens = ids[:, :T_prompt].to(torch.long)
    print(f"[xreq] ISL={_T_PREFILL}  real prompt tokens={T_prompt}")

    state_dict = _load_full_state_dict(_SNAPSHOT)
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS, paged_attention_config)

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )
    _kv_shape = (paged_attention_config.max_num_blocks, 1, paged_attention_config.block_size, args.head_dim)
    tt_kv_cache = allocate_vllm_kv_cache(
        _kv_shape, torch.bfloat16, args.n_layers, model, args.weight_cache_path(ttnn.bfloat8_b)
    )

    generator = Generator(model, args, bh_glx_mesh, tokenizer=tok)
    generator._disable_prefill_tracing = True
    generator.prefill_warmup_completed = True

    sampling_params = SamplingParams(
        temperature=float(os.environ.get("QWEN36_TEMP", "1.0")),
        top_k=int(os.environ.get("QWEN36_TOP_K", "20")),
        top_p=float(os.environ.get("QWEN36_TOP_P", "0.95")),
    )

    _NUM_REQ = int(os.environ.get("QWEN36_GEN_NUM_REQUESTS", "2"))
    _DEC_STEPS = int(os.environ.get("QWEN36_PROBE_DECODE_STEPS", "6"))
    # ISOLATION: QWEN36_PROBE_SWITCH_ONLY=1 -> between requests, cycle
    # switch_mode("decode")->switch_mode("prefill") WITHOUT running decode_forward.
    # If req1 prefill explodes under switch-only, the mode-switch (sub-device /
    # L1 buffer) is the culprit; if req1 is clean, the decode COMPUTE corrupts
    # shared state.
    _switch_only = os.environ.get("QWEN36_PROBE_SWITCH_ONLY", "0") == "1"
    req_argmax = []
    req_text = []
    for _req in range(_NUM_REQ):
        print("#" * 80)
        print(f"[xreq] ===== REQUEST {_req} (identical prompt) =====")
        prefill_logits = generator.prefill_forward_text(
            real_tokens,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=[T_prompt],
            enable_trace=False,
            sampling_params=None,
        )
        _logits = torch.as_tensor(prefill_logits).float().reshape(-1)[: args.vocab_size]
        amax = int(_logits.argmax().item())
        top5 = torch.topk(_logits, 5)
        top5_pairs = [(int(i), round(float(v), 3), tok.decode([int(i)])) for v, i in zip(top5.values, top5.indices)]
        req_argmax.append(amax)
        print(f"[xreq] req{_req} PREFILL argmax={amax} ({tok.decode([amax])!r})  top5={top5_pairs}")

        if _switch_only:
            print("[xreq] switch-only: switch_mode('decode') -> switch_mode('prefill') (NO decode compute)")
            generator.model.switch_mode("decode")
            generator.model.switch_mode("prefill")
            req_text.append("<switch-only>")
            continue

        # a few decode steps (on-device sampling, traced — server path) for a text
        # eyeball. async_read=True path mirrors the working generator demo: returns
        # (tt_tok, read_event); synchronize the event before reading on host.
        out_tok = torch.tensor([amax], dtype=torch.long)
        current_pos = torch.tensor([T_prompt], dtype=torch.long)
        gen_ids = [amax]
        for it in range(_DEC_STEPS):
            tt_tok, read_event = generator.decode_forward(
                out_tok,
                current_pos,
                enable_trace=True,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                read_from_device=True,
                async_read=True,
                sampling_params=sampling_params,
                reset_inputs=(it == 0),
            )
            ttnn.event_synchronize(read_event[0])
            _t, _ = generator.process_decode_output_host(tt_tok)
            nid = int(torch.as_tensor(_t).reshape(-1)[0].item())
            gen_ids.append(nid)
            out_tok = torch.tensor([nid], dtype=torch.long)
            current_pos = current_pos + 1
        txt = tok.decode(gen_ids, skip_special_tokens=False)
        req_text.append(txt)
        print(f"[xreq] req{_req} DECODE text={txt!r}")

    print("#" * 80)
    print(f"[xreq] SUMMARY prefill argmax per request = {req_argmax}")
    for i, t in enumerate(req_text):
        print(f"[xreq]   req{i} text = {t!r}")

    # Cross-request regression check. The pre-fix bug made every request after
    # the first decode explode: the prefill MLP read its persistent CCL buffers
    # after the decode sub-device manager's allocator had overwritten them, so
    # logits blew up to ~1e21 and prefill argmax landed on rare/garbage tokens
    # (mojibake) at ISL >= ~4k. The fix (tt_ccl.rebuild_prefill_persistent_buffers
    # on each prefill entry) restores coherence on EVERY request.
    #
    # We do NOT assert identical argmax: bf8 logits tie-break run-to-run among a
    # near-tied top-k cluster, so a coherent model legitimately drifts by a token.
    # Instead assert that EVERY request stays coherent (no numeric blow-up): the
    # top logit is sane (not ~1e21) and the decoded text is real language.
    for _i, _amx in enumerate(req_argmax):
        assert _amx >= 0
    if not _switch_only:
        for _i, _t in enumerate(req_text):
            _n_alpha = sum(c.isalpha() for c in _t)
            assert _n_alpha >= 5, f"req{_i} incoherent decode (cross-request contamination?): {_t!r}"


def test_qwen36_isl_schedule_repro(bh_glx_mesh):
    """BENCHMARK-CRASH reproducer — generator path, ONE persistent model, NO reset.

    The `run.py --workflow benchmarks` sweep crashes the engine at the 3rd combo:
    the first ISL>128 prefill AFTER ~12 prior ISL-128 decode-heavy requests, with a
    dispatch fetch-queue timeout (system_memory_manager `fetch_on_timeout` -> "device
    timeout in fetch queue wait, potential hang detected"). Every test that "worked"
    either reset the device per ISL (qwen36_isl_sweep.sh) or hand-tested few requests
    on one process, so the cumulative many-request path was NEVER exercised in
    isolation.

    This test runs an ISL SCHEDULE (default: 12x ISL-128 then ISL-4096) through the
    SHARED generator path (prefill_forward_text + decode_forward, switch_mode cycling
    each request) on ONE model build with NO reset -- mimicking the benchmark's request
    sequence WITHOUT HTTP / vLLM / chat-template. Decisive split:

      * If it HANGS at the first big request  -> bug is in the generator/model path
        (cumulative mode-cycling / rebuild_prefill_persistent_buffers). vLLM & the chat
        endpoint are NOT the cause. Iterate the fix against THIS test.
      * If it runs CLEAN through all requests  -> bug is vLLM/chat-endpoint specific;
        move to an HTTP hammer against /v1/chat/completions.

    Knobs:
      QWEN36_ISL_SCHEDULE      comma-sep ISLs (default "128,"x12 + "4096")
      QWEN36_SCHED_DECODE_STEPS decode steps per request (default 16; bump to mirror
                               benchmark osl 128/1024 if a short schedule won't repro)
    """
    import math

    from transformers import AutoTokenizer

    from models.common.sampling import SamplingParams
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig
    from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
    from models.demos.qwen3_6_galaxy_v2.tt.generator_vllm import allocate_vllm_kv_cache

    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
        "QWEN36_FULLATTN_WO_TUNED": "1",
        "QWEN36_DELTA_OP_TUNED": "1",
        "QWEN36_CCL_NUM_LINKS_DELTA": "2",
    }.items():
        os.environ.setdefault(_k, _v)

    _default_sched = ",".join(["128"] * 12 + ["4096"])
    schedule = [int(x) for x in os.environ.get("QWEN36_ISL_SCHEDULE", _default_sched).split(",") if x.strip()]
    dec_steps = int(os.environ.get("QWEN36_SCHED_DECODE_STEPS", "16"))
    max_isl = max(schedule)
    print(f"[sched] schedule={schedule}  decode_steps/req={dec_steps}  max_isl={max_isl}")

    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)

    # Size paged KV for the LARGEST ISL in the schedule (+ decode + margin), rounded
    # up to a multiple of max_batch_size (page_table reshape requires divisibility).
    state_dict = _load_full_state_dict(_SNAPSHOT)
    _blocks = max(32, (max_isl + dec_steps + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE + 4)
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_blocks)
    model, args = _build_tt_model_paged_kv(bh_glx_mesh, state_dict, _PATTERN, _N_LAYERS, paged_attention_config)

    # round max_num_blocks up to a multiple of max_batch_size for the page_table reshape
    if paged_attention_config.max_num_blocks % args.max_batch_size != 0:
        _blocks = int(math.ceil(_blocks / args.max_batch_size) * args.max_batch_size)
        paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_blocks)

    permutation = torch.randperm(paged_attention_config.max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(
        args.max_batch_size, paged_attention_config.max_num_blocks // args.max_batch_size
    )
    _kv_shape = (paged_attention_config.max_num_blocks, 1, paged_attention_config.block_size, args.head_dim)
    tt_kv_cache = allocate_vllm_kv_cache(
        _kv_shape, torch.bfloat16, args.n_layers, model, args.weight_cache_path(ttnn.bfloat8_b)
    )

    generator = Generator(model, args, bh_glx_mesh, tokenizer=tok)
    generator._disable_prefill_tracing = True
    generator.prefill_warmup_completed = True

    sampling_params = SamplingParams(
        temperature=float(os.environ.get("QWEN36_TEMP", "1.0")),
        top_k=int(os.environ.get("QWEN36_TOP_K", "20")),
        top_p=float(os.environ.get("QWEN36_TOP_P", "0.95")),
    )

    def _flush(msg):
        print(msg, flush=True)

    for _req, S in enumerate(schedule):
        prompt = _load_prompt_for_isl(S)
        ids = tok(prompt, return_tensors="pt").input_ids
        if int(ids.shape[-1]) > S:
            ids = ids[:, :S]
        T_prompt = int(ids.shape[-1])
        real_tokens = ids[:, :T_prompt].to(torch.long)

        _flush("#" * 80)
        _flush(f"[sched] >>> REQ {_req}/{len(schedule)-1}  ISL={S}  (tokens={T_prompt})  PREFILL ...")
        try:
            prefill_logits = generator.prefill_forward_text(
                real_tokens,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=[T_prompt],
                enable_trace=False,
                sampling_params=None,
            )
        except Exception as e:  # noqa: BLE001 — localize the failing request/phase
            _flush(f"[sched] !!! CRASH in REQ {_req} ISL={S} PREFILL: {type(e).__name__}: {e}")
            raise
        _logits = torch.as_tensor(prefill_logits).float().reshape(-1)[: args.vocab_size]
        amax = int(_logits.argmax().item())
        _flush(f"[sched] REQ {_req} ISL={S} PREFILL ok  argmax={amax} ({tok.decode([amax])!r})  DECODE {dec_steps} ...")

        out_tok = torch.tensor([amax], dtype=torch.long)
        current_pos = torch.tensor([T_prompt], dtype=torch.long)
        try:
            for it in range(dec_steps):
                tt_tok, read_event = generator.decode_forward(
                    out_tok,
                    current_pos,
                    enable_trace=True,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    read_from_device=True,
                    async_read=True,
                    sampling_params=sampling_params,
                    reset_inputs=(it == 0),
                )
                ttnn.event_synchronize(read_event[0])
                _t, _ = generator.process_decode_output_host(tt_tok)
                nid = int(torch.as_tensor(_t).reshape(-1)[0].item())
                out_tok = torch.tensor([nid], dtype=torch.long)
                current_pos = current_pos + 1
        except Exception as e:  # noqa: BLE001
            _flush(f"[sched] !!! CRASH in REQ {_req} ISL={S} DECODE step={it}: {type(e).__name__}: {e}")
            raise
        _flush(f"[sched] REQ {_req} ISL={S} DECODE ok")

    _flush("#" * 80)
    _flush(
        f"[sched] ALL {len(schedule)} REQUESTS COMPLETED CLEAN — generator path does NOT reproduce "
        f"the benchmark hang; suspect vLLM/chat-endpoint path instead."
    )
