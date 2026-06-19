# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""V2-7b — Per-layer PCC sweep for 64-layer qwen3.6-27B prefill.

Instruments ``TtTransformer.forward(mode="prefill")`` to capture hidden state
after every layer, runs the CPU reference also capturing per-layer hidden
states, and prints a PCC table showing exactly which layer the dtype leak
begins to compound at.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_64layer_per_layer_pcc.py \\
            -v -s
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

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_PROMPT_DIR = pathlib.Path("models/demos/llama3_70b_galaxy/demo/sample_prompts")
_CONTEXT_CACHE_DIR = pathlib.Path("models/tt_transformers/demo/context_cache")

# Override via ``QWEN36_PCC_T_PREFILL=4096`` to find the layer at which long-T
# trajectory diverges from CPU reference.
_T_PREFILL = int(os.environ.get("QWEN36_PCC_T_PREFILL", "128"))
_N_LAYERS = 64


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


def _load_state_dict_all_layers(snapshot_dir: pathlib.Path) -> dict:
    with open(snapshot_dir / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    needed_prefixes = [
        "model.language_model.embed_tokens.",
        "model.language_model.norm.",
        "lm_head.",
        "model.language_model.layers.",
    ]
    needed_keys = [k for k in weight_map if any(k.startswith(p) for p in needed_prefixes)]
    files = sorted({weight_map[k] for k in needed_keys})
    sd: dict[str, torch.Tensor] = {}
    for fn in files:
        shard = load_st(str(snapshot_dir / fn))
        for k in needed_keys:
            if k in shard:
                sd[k] = shard[k]
    return sd


def _load_and_cache_context(context_url: str, cache_dir: pathlib.Path, max_length: int | None = None) -> str:
    """Identical to test_decode_perf_intrace.py — Gutenberg URL cache + char clip."""
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
    """Same prompt-loader as test_decode_perf_intrace.py so the per-layer PCC
    runs the SAME input the production perf test would see at this ISL."""
    if t_prefill < 1024:
        return "The capital of France is"
    k = t_prefill // 1024
    prompt_file = _PROMPT_DIR / f"input_data_long_{k}k.json"
    with open(prompt_file) as f:
        data = json.load(f)
    entry = data[0]
    prompt = entry["prompt"]
    context_url = entry.get("context")
    if context_url:
        # 6 chars/token floor — see test_decode_perf_intrace.py for rationale.
        max_length = max(entry.get("max_length") or 0, t_prefill * 6)
        context_text = _load_and_cache_context(context_url, _CONTEXT_CACHE_DIR, max_length=max_length)
        prompt = "```" + context_text + "```\n\n" + prompt
    return prompt


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _cpu_reference_per_layer(state_dict_hf, x, capture_intermediates: bool = False):
    """Returns list of per-layer hidden states (64 tensors), each [1, T, H].

    When ``capture_intermediates=True``, also returns a parallel list of dicts
    (one per layer) with intermediate hidden states at each sub-block boundary:

        "post_attn_norm"  — hidden state after attention RMSNorm (input_layernorm)
        "post_attn"       — attention output (before post-attention residual add)
        "post_attn_res"   — hidden state after post-attention residual add
        "post_mlp_norm"   — hidden state after MLP RMSNorm (post_attention_layernorm)
        "post_mlp"        — MLP output (before post-MLP residual add)
        "post_mlp_res"    — final layer output (post-MLP residual add, same as per_layer_hidden[i])

    The intermediates are captured by monkey-patching each HybridDecoderLayer's
    forward method in-place (no production code is modified).
    """
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer, Qwen36Config, build_mrope_cos_sin

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)

    T = x.shape[1]
    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    causal_mask = torch.zeros(1, 1, T, T)
    causal_mask = causal_mask.masked_fill(torch.triu(torch.ones(T, T), diagonal=1).bool(), float("-inf"))

    hidden = x.float()
    per_layer_hidden: list[torch.Tensor] = []
    per_layer_intermediates: list[dict[str, torch.Tensor]] = []
    for layer_idx in range(_N_LAYERS):
        layer = HybridDecoderLayer(config, layer_idx).eval()
        pfx = f"model.language_model.layers.{layer_idx}."
        layer_sd = {}
        for k, v in state_dict_hf.items():
            if k.startswith(pfx):
                short = k[len(pfx) :]
                if short.startswith("self_attn."):
                    layer_sd["attention." + short[len("self_attn.") :]] = v.float()
                elif short.startswith("linear_attn."):
                    layer_sd["attention." + short[len("linear_attn.") :]] = v.float()
                else:
                    layer_sd[short] = v.float()
        layer.load_state_dict(layer_sd, strict=False)

        if capture_intermediates:
            # Monkey-patch the layer's forward to capture sub-block states without
            # modifying any production code.  We wrap the method on the *instance*
            # so it is automatically discarded with the layer object after this loop.
            _captured: dict[str, torch.Tensor] = {}

            _orig_forward = layer.forward

            def _instrumented_forward(
                _x,
                _cos=None,
                _sin=None,
                _attention_mask=None,
                _kv_cache=None,
                _conv_state=None,
                _recurrent_state=None,
                *,
                _layer=layer,
                _cap=_captured,
                _orig=_orig_forward,
            ):
                """Instrumented forward: capture intermediates at each sub-step.

                Replicates the HybridDecoderLayer.forward logic step-by-step so we
                can snapshot hidden states at each boundary without modifying the
                production reference module.
                """

                residual = _x
                # Step 1: pre-attention norm
                x_normed = _layer.input_layernorm(_x)
                _cap["post_attn_norm"] = x_normed.clone()

                # Step 2: attention
                if _layer.layer_type == "full_attention":
                    attn_out, kv_cache_new = _layer.attention(x_normed, _cos, _sin, _kv_cache, _attention_mask)
                    conv_state_new = _conv_state
                    recurrent_state_new = _recurrent_state
                else:
                    attn_out, conv_state_new, recurrent_state_new = _layer.attention(
                        x_normed, _conv_state, _recurrent_state
                    )
                    kv_cache_new = None
                _cap["post_attn"] = attn_out.clone()

                # Step 3: post-attention residual add
                x_post_attn = residual + attn_out
                _cap["post_attn_res"] = x_post_attn.clone()

                # Step 4: pre-MLP norm
                residual2 = x_post_attn
                x_mlp_normed = _layer.post_attention_layernorm(x_post_attn)
                _cap["post_mlp_norm"] = x_mlp_normed.clone()

                # Step 5: MLP
                mlp_out = _layer.mlp(x_mlp_normed)
                _cap["post_mlp"] = mlp_out.clone()

                # Step 6: post-MLP residual add (= layer output)
                x_out = residual2 + mlp_out
                _cap["post_mlp_res"] = x_out.clone()

                return x_out, kv_cache_new, conv_state_new, recurrent_state_new

            layer.forward = _instrumented_forward  # type: ignore[method-assign]

        with torch.no_grad():
            hidden, _, _, _ = layer(hidden, cos, sin, attention_mask=causal_mask)
        per_layer_hidden.append(hidden.clone())
        if capture_intermediates:
            per_layer_intermediates.append(dict(_captured))
        del layer
    return per_layer_hidden, config, per_layer_intermediates


def _build_tt_model(mesh, state_dict, pattern, n_layers):
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


def _build_partial_rope_cos_sin_tt(mesh, T):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
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


def _send_col_sharded_hidden(t, mesh, args):
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


def _gather_col_sharded_to_full(tt_tensor, mesh, args, T):
    out = ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 3), mesh_shape=args.cluster_shape),
    )
    out = out[0:1]
    while out.dim() > 3 and out.shape[0] == 1:
        out = out.squeeze(0)
    if out.dim() == 3:
        out = out[:, :T, :]
    return out


@pytest.mark.hardware
def test_qwen36_64_layer_per_layer_pcc(bh_glx_mesh):
    """Per-layer hidden-state PCC sweep — find where compounding error begins."""
    print("[per-layer] loading HF state_dict ...")
    state_dict = _load_state_dict_all_layers(_SNAPSHOT)

    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        cfg_dict = json.load(f)
    config = Qwen36Config(cfg_dict)
    pattern = list(config.layer_types)
    assert len(pattern) == _N_LAYERS

    print("[per-layer] building TT 64-layer model ...")
    model, args = _build_tt_model(bh_glx_mesh, state_dict, pattern, _N_LAYERS)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    prompt = _load_prompt_for_isl(_T_PREFILL)
    preview = prompt if len(prompt) <= 200 else f"{prompt[:120]!r}...{prompt[-80:]!r}"
    print(f"[per-layer] prompt ({len(prompt)} chars): {preview}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    T_prompt = int(input_ids.shape[-1])
    if T_prompt > _T_PREFILL:
        input_ids = input_ids[:, :_T_PREFILL]
        T_prompt = _T_PREFILL
    input_ids_padded = torch.zeros(1, _T_PREFILL, dtype=input_ids.dtype)
    input_ids_padded[0, :T_prompt] = input_ids[0]
    print(f"[per-layer] T_prompt={T_prompt} T_PREFILL={_T_PREFILL}")

    # Sub-block instrumentation flag: set QWEN36_PCC_SUBBLOCK=1 to enable capture
    # of intermediates at each sub-step within every decoder layer.
    # Default OFF — existing layer-level PCC table is unchanged.
    _subblock_mode = os.environ.get("QWEN36_PCC_SUBBLOCK", "0") == "1"

    embed_w = state_dict["model.language_model.embed_tokens.weight"].float()
    x_cpu_torch = embed_w[input_ids_padded[0]].unsqueeze(0)
    print(f"[per-layer] CPU reference: 64 layers (capture_intermediates={_subblock_mode}) ...")
    per_layer_ref, _, per_layer_ref_intermediates = _cpu_reference_per_layer(
        state_dict, x_cpu_torch, capture_intermediates=_subblock_mode
    )
    print(f"[per-layer] CPU ref done, captured {len(per_layer_ref)} layers")

    x_tt = _send_col_sharded_hidden(x_cpu_torch.to(torch.bfloat16), bh_glx_mesh, args)
    cos_tt, sin_tt = _build_partial_rope_cos_sin_tt(bh_glx_mesh, _T_PREFILL)
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )

    # -----------------------------------------------------------------------
    # Sub-block TT instrumentation (QWEN36_PCC_SUBBLOCK=1 only)
    #
    # Monkey-patch each TtTransformerBlock.forward to yield intermediate TT
    # tensors at the same sub-step boundaries captured by the CPU reference.
    # The patch is applied on *instances* (not the class) so it is fully local
    # to this test run and does not touch any production code.
    #
    # Captured sub-steps (qwen36 prefill path in llama_decoder.py):
    #   post_attn_norm  — output of self.attention_norm (DistributedNorm)
    #   post_attn       — output of self.attention.forward (col-sharded)
    #   post_attn_res   — h_new = x + attn_out (after first residual add)
    #   post_mlp_norm   — output of self.ff_norm (DistributedNorm)
    #   post_mlp        — output of MLP forward_prefill
    #   post_mlp_res    — final layer output (after second residual add)
    #
    # NOTE: the patch only instruments the ``is_qwen36_path`` (prefill) branch.
    # The non-qwen36 decode path is not affected and falls through unchanged.
    # -----------------------------------------------------------------------
    _tt_subblock_store: list[dict[str, torch.Tensor]] = []  # per-layer, populated when flag is on

    if _subblock_mode:

        def _make_patched_forward(orig_layer, store_list, layer_idx):
            """Return a replacement forward() that records sub-block outputs."""
            orig_forward = orig_layer.forward

            def _patched_forward(
                x,
                h,
                current_pos,
                rot_mats=None,
                user_id=0,
                mode="decode",
                page_table=None,
                chunk_page_table=None,
                chunk_start_idx=None,
                chunk_start_idx_tensor=None,
                kv_cache=None,
                batch_size=1,
            ):
                # ---- replicate the is_qwen36_path == True / prefill branch ----
                # (mirrors llama_decoder.TtTransformerBlock.forward exactly for
                # the prefill path; any mismatch would show up as a PCC anomaly)
                pass

                is_qwen36_path = orig_layer.is_qwen36 and mode in ("prefill", "decode")
                if not is_qwen36_path or mode != "prefill":
                    # Non-qwen36 or decode: fall through to original forward, no capture.
                    return orig_forward(
                        x,
                        h,
                        current_pos,
                        rot_mats,
                        user_id,
                        mode,
                        page_table,
                        chunk_page_table=chunk_page_table,
                        chunk_start_idx=chunk_start_idx,
                        chunk_start_idx_tensor=chunk_start_idx_tensor,
                        kv_cache=kv_cache,
                        batch_size=batch_size,
                    )

                # --- prefill qwen36 path: instrument each sub-step ---
                mc = orig_layer.model_config
                mesh = orig_layer.mesh_device
                # Determine cluster_shape from args (needed by _gather_col_sharded_to_full).
                # We close over the outer 'args' variable set up before this loop.
                _cap: dict[str, torch.Tensor] = {}

                def _snap(name, tt_t):
                    """Gather tt_t to CPU and store in _cap (non-destructive)."""
                    try:
                        cpu_t = _gather_col_sharded_to_full(tt_t, mesh, args, T=_T_PREFILL)
                        _cap[name] = cpu_t.reshape(1, _T_PREFILL, -1).float()
                    except Exception as _e:  # noqa: BLE001
                        # Gathering may fail if the tensor shape differs (e.g. the
                        # attn_out before the layer output which is not full-T).
                        # Store None so the downstream PCC prints "N/A".
                        _cap[name] = None

                # Step 1: attention norm (col-sharded in → col-sharded out)
                attn_in_sharded, _ = orig_layer.attention_norm(x, None, "prefill")
                _snap("post_attn_norm", attn_in_sharded)

                # Step 2: attention forward
                attn_out = orig_layer.attention.forward(
                    attn_in_sharded,
                    current_pos,
                    rot_mats,
                    user_id,
                    mode,
                    page_table=page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx=chunk_start_idx,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    kv_cache=kv_cache,
                    batch_size=batch_size,
                )
                attn_in_sharded.deallocate(True)
                if len(list(attn_out.shape)) == 3:
                    _B_a, _T_a, _H_a = list(attn_out.shape)
                    attn_out = ttnn.reshape(attn_out, ttnn.Shape([_B_a, 1, _T_a, _H_a]))
                _snap("post_attn", attn_out)

                # Step 3: post-attention residual add
                h_new = ttnn.add(x, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                x.deallocate(True)
                attn_out.deallocate(True)
                _snap("post_attn_res", h_new)

                # Step 4: MLP norm (col-sharded in → col-sharded out)
                ff_in_sharded, _ = orig_layer.ff_norm(h_new, None, "prefill")
                _snap("post_mlp_norm", ff_in_sharded)

                # Step 5: MLP forward
                ff_out_sharded = orig_layer.feed_forward.forward(ff_in_sharded, "prefill", batch_size=batch_size)
                ff_in_sharded.deallocate(True)
                _snap("post_mlp", ff_out_sharded)

                # Step 6: post-MLP residual add (= layer output)
                out_sharded = ttnn.add(ff_out_sharded, h_new, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ff_out_sharded.deallocate(True)
                h_new.deallocate(True)
                _snap("post_mlp_res", out_sharded)

                store_list.append(_cap)
                return out_sharded, None

            return _patched_forward

        for _li, _blk in enumerate(model.layers):
            _blk.forward = _make_patched_forward(_blk, _tt_subblock_store, _li)  # type: ignore[method-assign]

    # Instrument: replace TtTransformer.forward layer loop to capture per-layer outputs.
    print("[per-layer] running TT 64-layer prefill (instrumented) ...")

    # Manually replicate the forward loop, snapping x at each step.
    rot_mats = (cos_tt, sin_tt)
    x = x_tt
    h = None
    per_layer_pcc: list[float] = []
    per_layer_dtype: list[str] = []
    for i, layer in enumerate(model.layers):
        x, h = layer(
            x,
            h,
            None,
            rot_mats,
            0,
            "prefill",
            None,
            chunk_page_table=None,
            chunk_start_idx=0,
            chunk_start_idx_tensor=chunk_start_idx_tt,
            kv_cache=None,
            batch_size=1,
        )
        # Don't deallocate x — it's the layer output going into next layer.
        # Clone via to_torch (gathers data; non-destructive).
        tt_hidden_cpu = _gather_col_sharded_to_full(x, bh_glx_mesh, args, T=_T_PREFILL)
        tt_hidden_cpu = tt_hidden_cpu.reshape(1, _T_PREFILL, -1).float()
        ref_hidden = per_layer_ref[i][:, :_T_PREFILL, :].float()
        pcc_full = _pcc(tt_hidden_cpu, ref_hidden)
        # PCC over only the real prompt tokens (positions 0..T_prompt-1) and
        # over only the last prompt position (the one that matters for logits).
        pcc_prompt = _pcc(tt_hidden_cpu[:, :T_prompt, :], ref_hidden[:, :T_prompt, :])
        pcc_last = _pcc(tt_hidden_cpu[:, T_prompt - 1 : T_prompt, :], ref_hidden[:, T_prompt - 1 : T_prompt, :])
        per_layer_pcc.append(pcc_last)
        per_layer_dtype.append(str(x.dtype))

        if _subblock_mode and i < len(_tt_subblock_store) and i < len(per_layer_ref_intermediates):
            # Print sub-block PCC for this layer.
            _tt_caps = _tt_subblock_store[i]
            _ref_caps = per_layer_ref_intermediates[i]
            _sub_steps = [
                "post_attn_norm",
                "post_attn",
                "post_attn_res",
                "post_mlp_norm",
                "post_mlp",
                "post_mlp_res",
            ]
            _step_labels = {
                "post_attn_norm": "attn_norm",
                "post_attn": "post_attn",
                "post_attn_res": "post_attn_res",
                "post_mlp_norm": "mlp_norm",
                "post_mlp": "post_mlp",
                "post_mlp_res": "full",
            }
            _pcc_parts = []
            for _step in _sub_steps:
                _tt_t = _tt_caps.get(_step)
                _ref_t = _ref_caps.get(_step)
                if _tt_t is None or _ref_t is None:
                    _pcc_parts.append(f"{_step_labels[_step]}=N/A")
                else:
                    # Compare at the last real prompt position (same as pcc_last).
                    _tt_slice = _tt_t[:, T_prompt - 1 : T_prompt, :].float()
                    _ref_slice = _ref_t[:, T_prompt - 1 : T_prompt, :].float()
                    _p = _pcc(_tt_slice, _ref_slice)
                    _pcc_parts.append(f"{_step_labels[_step]}={_p:.4f}")
            print(f"[per-layer] L{i:02d} ({pattern[i][:3]}): " + " ".join(_pcc_parts))
        else:
            print(
                f"[per-layer] L{i:02d} ({pattern[i][:3]}): "
                f"PCC_full={pcc_full:.4f} PCC_prompt={pcc_prompt:.4f} PCC_last={pcc_last:.4f} | "
                f"tt_std={tt_hidden_cpu.std().item():.3f} ref_std={ref_hidden.std().item():.3f} | "
                f"x.dtype={str(x.dtype).split('.')[-1]}"
            )

    print("\n[per-layer] SUMMARY")
    print(f"  First layer with PCC<0.99: ", end="")
    fail_idx = next((i for i, p in enumerate(per_layer_pcc) if p < 0.99), None)
    print(fail_idx)
    print(f"  Final layer PCC: {per_layer_pcc[-1]:.6f}")

    # -----------------------------------------------------------------------
    # Sub-block summary (QWEN36_PCC_SUBBLOCK=1 only)
    # For each sub-step, compute the per-layer PCC drop vs the previous step
    # and print the average drop + the step with the LARGEST average drop.
    # -----------------------------------------------------------------------
    if _subblock_mode and _tt_subblock_store and per_layer_ref_intermediates:
        _sub_steps = [
            "post_attn_norm",
            "post_attn",
            "post_attn_res",
            "post_mlp_norm",
            "post_mlp",
            "post_mlp_res",
        ]
        _step_labels = {
            "post_attn_norm": "attn_norm",
            "post_attn": "post_attn",
            "post_attn_res": "post_attn_res",
            "post_mlp_norm": "mlp_norm",
            "post_mlp": "post_mlp",
            "post_mlp_res": "full",
        }
        # Build per-layer, per-step PCC table (indexed [layer][step]).
        _pcc_table: list[dict[str, float | None]] = []
        for _i in range(len(_tt_subblock_store)):
            _tt_caps = _tt_subblock_store[_i]
            _ref_caps = per_layer_ref_intermediates[_i]
            _row: dict[str, float | None] = {}
            for _step in _sub_steps:
                _tt_t = _tt_caps.get(_step)
                _ref_t = _ref_caps.get(_step)
                if _tt_t is None or _ref_t is None:
                    _row[_step] = None
                else:
                    _tt_s = _tt_t[:, T_prompt - 1 : T_prompt, :].float()
                    _ref_s = _ref_t[:, T_prompt - 1 : T_prompt, :].float()
                    _row[_step] = _pcc(_tt_s, _ref_s)
            _pcc_table.append(_row)

        # Compute per-step average PCC drop vs the immediately preceding step.
        # "drop" = PCC[prev_step] − PCC[this_step] (positive = degradation).
        # For the first step (post_attn_norm) the reference is 1.0 (input = CPU fp32).
        print("\n[per-layer] SUB-BLOCK PCC DROP SUMMARY (avg across all layers):")
        _prev_step_label = "input (1.0)"
        _step_avg_drop: dict[str, float] = {}
        for _si, _step in enumerate(_sub_steps):
            _drops = []
            for _i, _row in enumerate(_pcc_table):
                _cur = _row.get(_step)
                if _cur is None:
                    continue
                if _si == 0:
                    _prev = 1.0
                else:
                    _prev_step = _sub_steps[_si - 1]
                    _prev = _row.get(_prev_step)
                    if _prev is None:
                        continue
                _drops.append(_prev - _cur)
            if _drops:
                _avg_drop = sum(_drops) / len(_drops)
                _step_avg_drop[_step] = _avg_drop
                print(f"  {_step_labels[_step]:18s}: avg_drop={_avg_drop:+.6f}  " f"(over {len(_drops)} layers)")
            else:
                print(f"  {_step_labels[_step]:18s}: no data")

        if _step_avg_drop:
            _worst_step = max(_step_avg_drop, key=_step_avg_drop.__getitem__)
            print(
                f"\n  LARGEST per-layer PCC drop: {_step_labels[_worst_step]!r}  "
                f"(avg {_step_avg_drop[_worst_step]:+.6f}/layer)"
            )

    # Don't assert — this is diagnostic.
