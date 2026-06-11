# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-layer prefill PCC harness (TT vs fp32 CPU reference) WITHOUT the manual-loop
L1 clash.

The existing test_mm_per_layer_pcc.py manually loops model.layers, which bypasses
TtTransformer.forward's prefill setup (sub-device manager / CCL buffer allocation)
and hits a `program 966` static-CB-vs-L1 clash on this build. This harness instead
runs the REAL model.forward (correct prefill setup, like the passing test_mm_prefill)
and captures each layer's output via a forward hook that copies the hidden to HOST
immediately — so no per-layer L1 accumulation.

Text-only by default (cleanest signal; the ~0.83 prefill PCC is the same for text and
VL). Knobs:
  QWEN36_PERLAYER_PROMPT  : prompt text (default short).
  QWEN36_PERLAYER_BUCKET  : prefill bucket (default 128; set 4096 to test the pad bucket).

Run:
  export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) HF_MODEL=Qwen/Qwen3.6-27B MESH_DEVICE=BH_GLX
  source python_env/bin/activate
  python -m pytest models/demos/qwen3_6_galaxy_v2/tests/test_qwen36_perlayer_pcc.py -x -s
"""
import json
import os
import pathlib

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file as load_st

import ttnn

_SNAPSHOT = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_N_LAYERS = 64


def _load_full_state_dict(snap: pathlib.Path) -> dict:
    with open(snap / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    sd: dict = {}
    for fn in sorted(set(weight_map.values())):
        sd.update(load_st(str(snap / fn)))
    return sd


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


class _Passthrough:
    """No-op stand-in for layers L..63 in the cumulative-passthrough mode. Keeps
    model.layers at 64 (so the prefill setup's L1 layout matches the PASSING full
    forward), while the forward output stays layer-(L-1)'s hidden. Matches the layer
    call/return contract: returns (x, h) unchanged, no device ops, no dealloc."""

    def __call__(self, x, h, *args, **kwargs):
        return x, h


def _build_ref_layer(state_dict_hf, config, layer_idx):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import HybridDecoderLayer

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
    return layer


def _mrope_cos_sin(config, position_ids_3d):
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import build_mrope_cos_sin

    return build_mrope_cos_sin(
        position_ids_3d,
        rope_theta=config.rope_theta,
        partial_rotary_dim=int(config.head_dim * config.partial_rotary_factor),
        mrope_section=config.mrope_section,
        attention_scaling=1.0,
        dtype=torch.float32,
    )


def _cpu_reference_per_layer(state_dict_hf: dict, fused_x: torch.Tensor, position_ids_3d: torch.Tensor):
    """fp32 64-layer reference; returns list of per-layer hidden states [1, S, H]."""
    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config

    with open(_SNAPSHOT / "config.json") as f:
        config = Qwen36Config(json.load(f))
    T = fused_x.shape[1]
    cos, sin = _mrope_cos_sin(config, position_ids_3d)
    causal = torch.zeros(1, 1, T, T).masked_fill(torch.triu(torch.ones(T, T), 1).bool(), float("-inf"))
    hidden = fused_x.float()
    per_layer = []
    for layer_idx in range(_N_LAYERS):
        layer = _build_ref_layer(state_dict_hf, config, layer_idx)
        with torch.no_grad():
            hidden, _, _, _ = layer(hidden, cos, sin, attention_mask=causal)
        per_layer.append(hidden.clone())
        del layer
        if (layer_idx + 1) % 16 == 0:
            logger.info(f"[cpu-ref] {layer_idx + 1}/{_N_LAYERS}")
    return per_layer


def _cpu_reference_oplevel(state_dict_hf, config, layer_idx, hidden_in, position_ids_3d):
    """fp32 reference for ONE layer; returns the 4 sub-block outputs (full H) so the
    TT sub-blocks can be PCC'd op-by-op: attn_norm, attn_out, ff_norm, mlp_out."""
    T = hidden_in.shape[1]
    cos, sin = _mrope_cos_sin(config, position_ids_3d)
    causal = torch.zeros(1, 1, T, T).masked_fill(torch.triu(torch.ones(T, T), 1).bool(), float("-inf"))
    layer = _build_ref_layer(state_dict_hf, config, layer_idx)
    x = hidden_in.float()
    with torch.no_grad():
        xn1 = layer.input_layernorm(x)
        if layer.layer_type == "full_attention":
            attn_out, _ = layer.attention(xn1, cos, sin, None, causal)
        else:
            attn_out, _, _ = layer.attention(xn1, None, None)
        h1 = x + attn_out
        xn2 = layer.post_attention_layernorm(h1)
        mlp_out = layer.mlp(xn2)
    # ff_norm_in (h1) lets us split "error already in the residual feeding the norm" from "the norm op".
    # Return the layer too so the harness can apply the CPU fp32 norm to the DEVICE h1.
    return {"attn_norm": xn1, "attn_out": attn_out, "ff_norm_in": h1, "ff_norm": xn2, "mlp_out": mlp_out}, layer


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "BH_GLX": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_qwen36_perlayer_pcc(mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"
    from transformers import AutoTokenizer

    from models.demos.qwen3_6_galaxy.reference.qwen36 import Qwen36Config
    from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs
    from models.demos.qwen3_6_galaxy_v2.tt.qwen36_mrope import build_mrope_tt_tensors, get_rope_index

    prompt = os.environ.get("QWEN36_PERLAYER_PROMPT", "The capital of France is Paris, and the city is known for")
    bucket = int(os.environ.get("QWEN36_PERLAYER_BUCKET", "128"))

    state_dict = _load_full_state_dict(_SNAPSHOT)
    embed_w = state_dict["model.language_model.embed_tokens.weight"].float()

    tok = AutoTokenizer.from_pretrained(str(_SNAPSHOT), trust_remote_code=True)
    input_ids = tok(prompt, return_tensors="pt").input_ids  # [1, T_prompt]
    T_prompt = input_ids.shape[-1]
    S = bucket
    assert T_prompt <= S, f"prompt {T_prompt} > bucket {S}"
    logger.info(f"[perlayer] prompt tokens={T_prompt} bucket S={S}")

    # Embed (HF weights) + pad to bucket with zeros.
    ids_padded = torch.cat([input_ids, torch.zeros(1, S - T_prompt, dtype=input_ids.dtype)], dim=1)
    fused = embed_w[ids_padded[0]].unsqueeze(0)  # [1, S, H]
    # Degenerate (text) 3D positions for M-RoPE.
    position_ids_3d, _ = get_rope_index(ids_padded, image_grid_thw=None, image_token_id=248056)

    # --- CPU reference (per-layer) ---
    logger.info("Running fp32 CPU reference per-layer (~5-10 min)...")
    per_layer_ref = _cpu_reference_per_layer(state_dict, fused, position_ids_3d)

    # --- Build TT model (same as test_mm_prefill / test_64layer) ---
    with open(_SNAPSHOT / "config.json") as f:
        config = Qwen36Config(json.load(f))
    args = TtQwen36ModelArgs(mesh_device)
    args.n_layers = _N_LAYERS
    args.linear_attention_pattern = list(config.layer_types)
    wcp = args.weight_cache_path(ttnn.bfloat8_b)
    wcp.mkdir(parents=True, exist_ok=True)
    model = TtTransformer(
        args=args,
        dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=wcp,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    )

    # Inputs for the REAL forward.
    x_tt = ttnn.from_torch(
        fused.reshape(1, 1, S, args.dim).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
    )
    cos_tt, sin_tt = build_mrope_tt_tensors(
        position_ids_3d,
        rope_theta=config.rope_theta,
        partial_rotary_dim=int(config.head_dim * config.partial_rotary_factor),
        mrope_section=config.mrope_section,
        mesh_device=mesh_device,
    )
    chunk_start_idx_tt = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def _gather(xt):
        out = ttnn.to_torch(
            xt, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=args.cluster_shape)
        )
        out = out[0:1]
        while out.dim() > 3 and out.shape[0] == 1:
            out = out.squeeze(0)
        return out.reshape(1, S, -1)[:, :S, :].float()

    def _reset_state():
        # Light reset between forwards: clear GDN recurrent/conv state + CCL gather idx.
        # (rebuild_prefill_persistent_buffers FREES a live buffer -> "Buffer not allocated"; omit it.)
        for _layer in full_layers:
            _attn = getattr(_layer, "attention", None)
            if _attn is not None and hasattr(_attn, "clear_state"):
                _attn.clear_state()
        _ccl = getattr(model, "tt_ccl", None)
        if _ccl is not None and hasattr(_ccl, "reset_gather_and_buffer_idx"):
            _ccl.reset_gather_and_buffer_idx()

    def _upload_hidden(h_torch):
        # h_torch [1, S, H] fp32 -> col-sharded bf16 device tensor (same layout as x_tt).
        return ttnn.from_torch(
            h_torch.reshape(1, 1, S, args.dim).to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=args.cluster_shape),
        )

    full_layers = list(model.layers)
    pattern = list(config.layer_types)

    def _run_single(layer_idx, x_in):
        # Run exactly ONE TT layer (model.layers=[layer_k]) on teacher-forced input x_in; return
        # post-layer hidden. One layer == low program count == dodges the program-966 L1 clash.
        model.layers = [full_layers[layer_idx]]
        out = model.forward(
            x_in,
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
        ttnn.synchronize_device(mesh_device)
        return _gather(out)

    if os.environ.get("QWEN36_OP_PROBE") == "1":
        # === OP-LEVEL PCC ===  teacher-force ONE layer (clean fp32 input ref[k-1]) and capture
        # its 4 sub-block outputs (attn_norm, attn_out, ff_norm, mlp_out); PCC each vs the fp32
        # reference sub-block. The first sub-block to drop = the op injecting the error. Single
        # layer == low program count, so mid-layer DRAM-clone capture dodges the program-966 clash.
        op_layers = [int(x) for x in os.environ.get("QWEN36_OP_LAYER", "0").split(",") if x.strip()]
        logger.info(f"OP-LEVEL PROBE over layers {op_layers}")
        logger.info("=" * 60)
        for j, k in enumerate(op_layers):
            if j > 0:
                _reset_state()
            hidden_in = fused.float() if k == 0 else per_layer_ref[k - 1][:, :S, :].float()
            ref_sub, ref_layer = _cpu_reference_oplevel(state_dict, config, k, hidden_in, position_ids_3d)
            layer_k = full_layers[k]
            cap = {}

            def _clone4d(t):
                if len(list(t.shape)) == 3:
                    _b, _t, _h = list(t.shape)
                    t = ttnn.reshape(t, ttnn.Shape([_b, 1, _t, _h]))
                return ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            def _wrap(obj, key, take0, cap_in=False):
                orig = obj.forward

                def w(*a, **kw):
                    # DeltaNet attention returns 3D; decoder reshapes to 4D AFTER. _gather wants 4D.
                    if cap_in and len(a) > 0 and isinstance(a[0], ttnn.Tensor):
                        cap[key + "_in"] = _clone4d(a[0])
                    r = orig(*a, **kw)
                    t = r[0] if (take0 and isinstance(r, tuple)) else r
                    cap[key] = _clone4d(t)
                    return r

                obj.forward = w
                return orig

            o1 = _wrap(layer_k.attention_norm, "attn_norm", True)
            o2 = _wrap(layer_k.attention, "attn_out", False)
            o3 = _wrap(layer_k.ff_norm, "ff_norm", True, cap_in=True)  # also capture h1 (norm input)
            o4 = _wrap(layer_k.feed_forward, "mlp_out", False)
            x_in = x_tt if k == 0 else _upload_hidden(hidden_in[:, :S, :])
            model.layers = [layer_k]
            model.forward(
                x_in,
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
            ttnn.synchronize_device(mesh_device)
            layer_k.attention_norm.forward, layer_k.attention.forward = o1, o2
            layer_k.ff_norm.forward, layer_k.feed_forward.forward = o3, o4
            ltype = pattern[k][:3] if k < len(pattern) else "?"
            logger.info(f"--- layer {k} ({ltype}) op-level ---")
            for key in ("attn_norm", "attn_out", "ff_norm_in", "ff_norm", "mlp_out"):
                tt_sub = _gather(cap[key])
                rf = ref_sub[key][:, :S, :].float()
                pp = _pcc(tt_sub[:, :T_prompt, :], rf[:, :T_prompt, :])
                pl = _pcc(tt_sub[:, T_prompt - 1 : T_prompt, :], rf[:, T_prompt - 1 : T_prompt, :])
                logger.info(f"  {key:10s}: PCC_prompt={pp:.4f} PCC_last={pl:.4f}")
            # CPU-norm test: apply the fp32 reference norm to the DEVICE h1. If this recovers to
            # ~0.999 while the device ff_norm output is ~0.98, the on-device norm OP is the culprit;
            # if it stays ~0.98, the error is already in h1 (residual/attention feeding the norm).
            h1_dev = _gather(cap["ff_norm_in"])[:, :S, :].float()
            cpu_norm_out = ref_layer.post_attention_layernorm(h1_dev)
            rf = ref_sub["ff_norm"][:, :S, :].float()
            pp = _pcc(cpu_norm_out[:, :T_prompt, :], rf[:, :T_prompt, :])
            pl = _pcc(cpu_norm_out[:, T_prompt - 1 : T_prompt, :], rf[:, T_prompt - 1 : T_prompt, :])
            logger.info(f"  CPUnorm(h1): PCC_prompt={pp:.4f} PCC_last={pl:.4f}  (vs device ff_norm above)")
        model.layers = full_layers
    elif os.environ.get("QWEN36_TF") == "1":
        # === TEACHER FORCING ===  feed the fp32 reference hidden ref[k-1] into TT layer k, run ONLY
        # layer k, PCC its output vs ref[k]. Isolates per-layer error (no accumulation). High PCC at
        # EVERY layer => the 0.83 is pure cumulative drift; a drop at layer k => layer k is the culprit.
        tf_layers = [
            int(x) for x in os.environ.get("QWEN36_TF_LAYERS", "0,3,16,31,32,47,48,63").split(",") if x.strip()
        ]
        logger.info(f"TEACHER FORCING over layers {tf_layers}")
        logger.info("=" * 60)
        for j, k in enumerate(tf_layers):
            if j > 0:
                _reset_state()
            x_in = x_tt if k == 0 else _upload_hidden(per_layer_ref[k - 1][:, :S, :].float())
            tt_h = _run_single(k, x_in)
            ref_h = per_layer_ref[k][:, :S, :].float()
            pcc_prompt = _pcc(tt_h[:, :T_prompt, :], ref_h[:, :T_prompt, :])
            pcc_last = _pcc(tt_h[:, T_prompt - 1 : T_prompt, :], ref_h[:, T_prompt - 1 : T_prompt, :])
            ltype = pattern[k][:3] if k < len(pattern) else "?"
            logger.info(f"TF L{k:02d} ({ltype}): PCC_prompt={pcc_prompt:.4f} PCC_last={pcc_last:.4f}")
        model.layers = full_layers
    else:
        # --- Truncated cumulative forward: model.layers[:L], output = layer-(L-1) hidden. One L per
        # process (QWEN36_L_VALUES); L=1 and L=64 pass, 8<=L<=63 hit the program-966 clash. ---
        L_values = [int(x) for x in os.environ.get("QWEN36_L_VALUES", "1").split(",") if x.strip()]
        # QWEN36_PASSTHRU=1: pad to 64 layers with no-ops (keeps the passing full-forward L1
        # layout) so 8<=L<=63 dodge the truncation program-966 clash. Output = layer-(L-1) hidden.
        passthru = os.environ.get("QWEN36_PASSTHRU") == "1"
        logger.info(f"{'Passthrough' if passthru else 'Truncated'}-forward sweep over L = {L_values}")
        logger.info("=" * 60)
        for idx, L in enumerate(L_values):
            if idx > 0:
                _reset_state()
            model.layers = (full_layers[:L] + [_Passthrough()] * (_N_LAYERS - L)) if passthru else full_layers[:L]
            tt_out = model.forward(
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
            ttnn.synchronize_device(mesh_device)
            tt_h = _gather(tt_out)
            ref_h = per_layer_ref[L - 1][:, :S, :].float()
            pcc_prompt = _pcc(tt_h[:, :T_prompt, :], ref_h[:, :T_prompt, :])
            pcc_last = _pcc(tt_h[:, T_prompt - 1 : T_prompt, :], ref_h[:, T_prompt - 1 : T_prompt, :])
            ltype = pattern[L - 1][:3] if (L - 1) < len(pattern) else "?"
            logger.info(f"L={L:02d} (last layer {L-1}, {ltype}): PCC_prompt={pcc_prompt:.4f} PCC_last={pcc_last:.4f}")
        model.layers = full_layers
