# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end correctness test for Qwen2.5-VL on TG (8,4).

Tests vision, text prefill, and text decode independently against HuggingFace
reference, so failures can be isolated to the specific stage.

Usage:
    MESH_DEVICE=TG HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
        pytest models/demos/qwen25_vl/tests/test_e2e_correctness.py -xvs
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen25_vl.tt.common import PagedAttentionConfig, multimodal_rope_from_hf, preprocess_inputs_prefill
from models.demos.qwen25_vl.tt.generator import Generator
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer, Transformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), None)


def _load_hf_model():
    from transformers import AutoProcessor
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

    model_name = os.environ.get("HF_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    logger.info(f"Loading HF model: {model_name}")
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    ref_model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return ref_model, processor


def _build_text_prompt():
    return [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]


def _build_image_prompt():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Vision encoder
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_vision_encoder(mesh_device, reset_seeds, ensure_gc):
    """Compare TT vision encoder output to HuggingFace reference."""
    from qwen_vl_utils import process_vision_info

    ref_model, processor = _load_hf_model()
    prompt = _build_image_prompt()

    text = processor.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([prompt])
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    ref_image_embeds = ref_model.visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
    logger.info(f"HF vision output shape: {ref_image_embeds.shape}")

    vision_args = VisionModelArgs(
        mesh_device,
        max_batch_size=1,
        max_seq_len=4096,
        optimizations=DecodersPrecision.accuracy(
            ref_model.config.vision_config.depth,
            ref_model.config._name_or_path,
        ),
    )
    vision_args.hf_config.vision_config.depth = ref_model.config.vision_config.depth
    tt_vision = DropInVisionTransformer(ref_model.visual, vision_args)

    tt_image_embeds = tt_vision(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
    logger.info(f"TT vision output shape: {tt_image_embeds.shape}")

    passing, pcc_msg = comp_pcc(ref_image_embeds.float(), tt_image_embeds.float(), 0.90)
    logger.info(f"Vision PCC: {pcc_msg}")
    assert passing, f"Vision encoder PCC below 0.90: {pcc_msg}"
    logger.info("PASSED: Vision encoder")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Text prefill (with KV cache fill)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_text_prefill(mesh_device, reset_seeds, ensure_gc):
    """Prefill one text-only user, compare last-token logits to HuggingFace."""
    ref_model, processor = _load_hf_model()
    prompt = _build_text_prompt()
    batch_size = 1

    text = processor.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids

    text_embeds = ref_model.model.language_model.embed_tokens(input_ids)

    ref_out = ref_model(input_ids, attention_mask=inputs.attention_mask)
    ref_logits = ref_out.logits[0, -1, :]
    ref_argmax = ref_logits.argmax().item()
    logger.info(f"HF prefill argmax={ref_argmax} ({processor.tokenizer.decode([ref_argmax])})")

    page_params = {"page_block_size": 32, "page_max_num_blocks": 1024}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=batch_size,
        optimizations=DecodersPrecision.performance(28, "Qwen2.5-VL-7B-Instruct"),
        max_seq_len=4096,
    )
    state_dict = model_args.load_state_dict()
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [l.attention.layer_past for l in tt_model.layers]
    generator = Generator(tt_model, model_args, mesh_device, processor=processor, tokenizer=processor.tokenizer)

    pad_token_id = processor.tokenizer.pad_token_id or 151643
    page_table = torch.randperm(paged_attention_config.max_num_blocks).reshape(
        batch_size, paged_attention_config.max_num_blocks // batch_size
    )

    input_prefill_pt, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        text_embeds,
        model_args,
        inputs.attention_mask,
        pad_embedding=ref_model.model.language_model.embed_tokens(torch.tensor(pad_token_id)),
    )
    cos, sin, rope_deltas_tt = multimodal_rope_from_hf(
        inputs,
        text_embeds,
        ref_model,
        model_args,
        pad_token_id=pad_token_id,
    )

    logits = generator.prefill_forward_text(
        input_prefill_pt,
        rot_mats=(cos, sin),
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=torch.tensor(decoding_pos),
    )

    tt_argmax = logits[0].argmax().item()
    logger.info(f"TT prefill argmax={tt_argmax} ({processor.tokenizer.decode([tt_argmax])})")

    assert ref_argmax == tt_argmax, f"Prefill argmax mismatch: HF={ref_argmax} TT={tt_argmax}"
    logger.info("PASSED: Text prefill")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Text decode (single step after prefill)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_text_decode(mesh_device, reset_seeds, ensure_gc):
    """Prefill one user, then run 1 decode step; compare to HuggingFace."""
    ref_model, processor = _load_hf_model()
    prompt = _build_text_prompt()
    batch_size = 1

    text = processor.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    pad_token_id = processor.tokenizer.pad_token_id or 151643

    text_embeds = ref_model.model.language_model.embed_tokens(input_ids)

    # HF: full prefill + 1 decode step
    ref_out = ref_model(input_ids, attention_mask=inputs.attention_mask, use_cache=True)
    prefill_token = ref_out.logits[0, -1, :].argmax().item()
    logger.info(f"HF prefill token: {prefill_token} ({processor.tokenizer.decode([prefill_token])})")

    new_input_ids = torch.tensor([[prefill_token]])
    new_attn_mask = torch.cat([inputs.attention_mask, torch.ones(1, 1, dtype=inputs.attention_mask.dtype)], dim=1)
    ref_decode_out = ref_model(
        new_input_ids,
        attention_mask=new_attn_mask,
        past_key_values=ref_out.past_key_values,
        use_cache=True,
    )
    ref_decode_token = ref_decode_out.logits[0, -1, :].argmax().item()
    logger.info(f"HF decode token: {ref_decode_token} ({processor.tokenizer.decode([ref_decode_token])})")

    # TT: prefill + 1 decode step
    page_params = {"page_block_size": 32, "page_max_num_blocks": 1024}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=batch_size,
        optimizations=DecodersPrecision.performance(28, "Qwen2.5-VL-7B-Instruct"),
        max_seq_len=4096,
    )
    state_dict = model_args.load_state_dict()
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [l.attention.layer_past for l in tt_model.layers]
    generator = Generator(tt_model, model_args, mesh_device, processor=processor, tokenizer=processor.tokenizer)

    page_table = torch.randperm(paged_attention_config.max_num_blocks).reshape(
        batch_size, paged_attention_config.max_num_blocks // batch_size
    )

    input_prefill_pt, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        text_embeds,
        model_args,
        inputs.attention_mask,
        pad_embedding=ref_model.model.language_model.embed_tokens(torch.tensor(pad_token_id)),
    )
    cos, sin, rope_deltas_tt = multimodal_rope_from_hf(
        inputs,
        text_embeds,
        ref_model,
        model_args,
        pad_token_id=pad_token_id,
    )

    tt_logits = generator.prefill_forward_text(
        input_prefill_pt,
        rot_mats=(cos, sin),
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=torch.tensor(decoding_pos),
    )
    tt_prefill_token = tt_logits[0].argmax().item()
    logger.info(f"TT prefill token: {tt_prefill_token} ({processor.tokenizer.decode([tt_prefill_token])})")
    assert tt_prefill_token == prefill_token, f"Prefill token mismatch: HF={prefill_token} TT={tt_prefill_token}"

    # Diagnostic: check bfloat4_b selection matrices for quantization errors
    if model_args.is_galaxy:
        for name in ("slice_mat", "user_selection_matrix"):
            mat = getattr(tt_model.layers[0].attention, name, None)
            if mat is not None:
                devs = ttnn.get_device_tensors(mat)
                t = ttnn.to_torch(devs[0]).float()
                # These should be 0/1 matrices; check max error from nearest integer
                err = (t - t.round()).abs().max().item()
                logger.info(
                    f"DIAG {name} dev0: shape={t.shape}, max_quant_err={err:.6f}, "
                    f"unique_vals={torch.unique(t).tolist()[:10]}"
                )

    # Diagnostic: check bias vs weight shapes AND content on device
    layer0_attn = tt_model.layers[0].attention
    if layer0_attn.wqkv_bias_decode:
        bias_devs = ttnn.get_device_tensors(layer0_attn.wqkv_bias_decode[0])
        b0 = ttnn.to_torch(bias_devs[0]).float()
        b1 = ttnn.to_torch(bias_devs[1]).float()
        b4 = ttnn.to_torch(bias_devs[4]).float()
        logger.info(f"DIAG wqkv_bias shape dev0: {b0.shape}")
        logger.info(f"DIAG wqkv_bias dev0==dev1 (same row, diff col): {torch.allclose(b0, b1, atol=1e-6)}")
        logger.info(f"DIAG wqkv_bias dev0==dev4 (diff row, same col): {torch.allclose(b0, b4, atol=1e-6)}")
        logger.info(f"DIAG wqkv_bias dev0 norm={b0.norm():.4e}, dev1 norm={b1.norm():.4e}, dev4 norm={b4.norm():.4e}")
        logger.info(f"DIAG wqkv_bias dev0[:5]={b0[0,0,0,:5].tolist()}, dev1[:5]={b1[0,0,0,:5].tolist()}")
    wqkv_devs = ttnn.get_device_tensors(layer0_attn.wqkv)
    logger.info(f"DIAG wqkv weight shape dev0: {wqkv_devs[0].shape}, dev4: {wqkv_devs[4].shape}")
    wo_devs = ttnn.get_device_tensors(layer0_attn.wo)
    logger.info(f"DIAG wo weight shape dev0: {wo_devs[0].shape}")

    # Decode — disable on-device sampling so we get raw logits for PCC check
    was_sampling = tt_model._supports_on_device_sampling
    old_sampling = tt_model.sampling
    tt_model._supports_on_device_sampling = False
    tt_model.sampling = None

    cols = model_args.cluster_shape[1] if model_args.is_galaxy else 1

    def _extract_user0_full(tt_tensor):
        devs = ttnn.get_device_tensors(tt_tensor)
        parts = []
        for c in range(cols):
            tc = ttnn.to_torch(devs[c]).float()
            parts.append(tc[0, 0, 0, :].clone())
        return torch.cat(parts, dim=0)

    # Hook strategic layers (first, middle, last) and embedding to capture user0 hidden state
    captured = {}
    n_layers = len(tt_model.layers)
    diag_layer_indices = [0, n_layers // 2, n_layers - 1]

    orig_transform = tt_model._transform_decode_inputs_device

    def _capture_embed(x):
        result = orig_transform(x)
        captured["embed"] = _extract_user0_full(result)
        return result

    tt_model._transform_decode_inputs_device = _capture_embed

    orig_layer_fns = {}
    for li in diag_layer_indices:
        layer = tt_model.layers[li]
        orig_layer_fns[li] = layer.forward

        def _make_layer_hook(layer_idx, original_fn):
            def _hooked(*args, **kwargs):
                result = original_fn(*args, **kwargs)
                captured[f"layer{layer_idx}"] = _extract_user0_full(result)
                return result

            return _hooked

        layer.forward = _make_layer_hook(li, layer.forward)

    # Sublayer hooks for layer 0: capture norm and attention outputs
    orig_attn_norm_fn_0 = tt_model.layers[0].attention_norm.forward

    def _capture_attn_norm_0(x, mode, norm_config=None):
        result = orig_attn_norm_fn_0(x, mode, norm_config=norm_config)
        captured["layer0_attn_norm"] = _extract_user0_full(result)
        return result

    tt_model.layers[0].attention_norm.forward = _capture_attn_norm_0

    # Hook tt_all_reduce in attention module to capture QKV and Wo all-reduce outputs for layer 0
    import models.tt_transformers.tt.attention as _attn_module

    _orig_attn_ar = _attn_module.tt_all_reduce
    _ar_call_idx = [0]
    rows = model_args.cluster_shape[0] if model_args.is_galaxy else 1

    def _diag_all_reduce(input_tensor, mesh_device, tt_ccl, cluster_axis=0, dim=0, **kwargs):
        # Capture the INPUT to all-reduce (before reduction)
        if model_args.is_galaxy and _ar_call_idx[0] < 2:
            in_devs = ttnn.get_device_tensors(input_tensor)
            in0 = ttnn.to_torch(in_devs[0]).float()
            n_inf_in = torch.isinf(in0).sum().item()
            n_nan_in = torch.isnan(in0).sum().item()
            logger.info(
                f"DIAG AR#{_ar_call_idx[0]} INPUT shape={list(in0.shape)}, "
                f"norm={in0.norm():.4e}, #inf={n_inf_in}, #nan={n_nan_in}, "
                f"max={in0.abs().max():.4e}, cluster_axis={cluster_axis}"
            )
        result = _orig_attn_ar(input_tensor, mesh_device, tt_ccl, cluster_axis, dim, **kwargs)
        idx = _ar_call_idx[0]
        _ar_call_idx[0] += 1
        if model_args.is_galaxy and idx < 2:
            devs = ttnn.get_device_tensors(result)
            if cluster_axis == 1:
                d0 = ttnn.to_torch(devs[0]).float()
                d1 = ttnn.to_torch(devs[1]).float()
                pcc_val = torch.corrcoef(torch.stack([d0.flatten(), d1.flatten()]))[0, 1].item()
                logger.info(f"DIAG AR#{idx} (QKV, cluster=1) cross-col PCC row0 col0 vs col1: {pcc_val:.6f}")
                logger.info(
                    f"DIAG AR#{idx} shape={list(d0.shape)}, norm_col0={d0.norm():.4e}, norm_col1={d1.norm():.4e}"
                )
                captured["layer0_qkv_after_ar_row0"] = d0[0, 0, 0, :].clone()
                d_r1 = ttnn.to_torch(devs[cols]).float()
                pcc_r = torch.corrcoef(torch.stack([d0[0, 0, 0, :], d_r1[0, 0, 0, :]]))[0, 1].item()
                logger.info(f"DIAG AR#{idx} QKV cross-row PCC row0 vs row1 col0 user0: {pcc_r:.6f}")
                logger.info(f"DIAG AR#{idx} QKV row0 first5: {d0[0,0,0,:5].tolist()}")
                logger.info(f"DIAG AR#{idx} QKV row1 first5: {d_r1[0,0,0,:5].tolist()}")
            else:
                d0 = ttnn.to_torch(devs[0]).float()
                d_r1 = ttnn.to_torch(devs[cols]).float()
                n_inf = torch.isinf(d0).sum().item()
                n_nan = torch.isnan(d0).sum().item()
                pcc_val = torch.corrcoef(torch.stack([d0.flatten(), d_r1.flatten()]))[0, 1].item()
                logger.info(f"DIAG AR#{idx} (Wo, cluster=0) cross-row PCC row0 vs row1, col0: {pcc_val:.6f}")
                logger.info(
                    f"DIAG AR#{idx} shape={list(d0.shape)}, norm_row0={d0.norm():.4e}, norm_row1={d_r1.norm():.4e}"
                )
                logger.info(f"DIAG AR#{idx} Wo OUTPUT #inf={n_inf}, #nan={n_nan}")
                logger.info(f"DIAG AR#{idx} Wo row0 user0 first5: {d0[0,0,0,:5].tolist()}")
                logger.info(f"DIAG AR#{idx} Wo row0 user31 first5: {d0[0,0,31,:5].tolist()}")
                logger.info(f"DIAG AR#{idx} Wo row1 user0 first5: {d_r1[0,0,0,:5].tolist()}")
                # Check per-user norms
                for u in [0, 7, 8, 15, 16, 23, 24, 31]:
                    u_norm = d0[0, 0, u, :].norm().item()
                    logger.info(f"DIAG AR#{idx} Wo row0 user{u} norm={u_norm:.4e}")
        return result

    if model_args.is_galaxy:
        _attn_module.tt_all_reduce = _diag_all_reduce

    orig_attn_fwd_0 = tt_model.layers[0].attention.forward

    def _capture_attn_fwd_0(*args, **kwargs):
        # Hook ttnn.linear to capture Wo matmul input (SDPA output after user_selection_matrix)
        _orig_linear = ttnn.linear
        _linear_count = [0]

        def _hook_linear(*largs, **lkwargs):
            _linear_count[0] += 1
            if _linear_count[0] == 2 and model_args.is_galaxy:
                # Second linear call = Wo matmul; largs[0] = attn_output (Wo input)
                devs = ttnn.get_device_tensors(largs[0])
                d0 = ttnn.to_torch(devs[0]).float()
                captured["layer0_wo_input_row0"] = d0[0, 0, 0, :].clone()
                logger.info(
                    f"DIAG layer0 Wo INPUT row0 user0: shape={list(d0.shape)}, "
                    f"norm={d0[0,0,0,:].norm():.4e}, first5={d0[0,0,0,:5].tolist()}"
                )
            return _orig_linear(*largs, **lkwargs)

        ttnn.linear = _hook_linear
        try:
            result = orig_attn_fwd_0(*args, **kwargs)
        finally:
            ttnn.linear = _orig_linear

        captured["layer0_attn_out"] = _extract_user0_full(result)
        if model_args.is_galaxy:
            devs = ttnn.get_device_tensors(result)
            d_r0c0 = ttnn.to_torch(devs[0]).float()[0, 0, 0, :]
            d_r1c0 = ttnn.to_torch(devs[cols]).float()[0, 0, 0, :]
            pcc = torch.corrcoef(torch.stack([d_r0c0, d_r1c0]))[0, 1].item()
            logger.info(f"DIAG attn_out cross-row PCC (row0 vs row1, col0, user0): {pcc:.6f}")
            logger.info(f"DIAG attn_out norm row0={d_r0c0.norm():.4e}, row1={d_r1c0.norm():.4e}")
        return result

    tt_model.layers[0].attention.forward = _capture_attn_fwd_0

    orig_ff_norm_fn_0 = tt_model.layers[0].ff_norm.forward

    def _capture_ff_norm_0(x, mode, norm_config=None):
        captured["layer0_res_after_attn"] = _extract_user0_full(x)
        result = orig_ff_norm_fn_0(x, mode, norm_config=norm_config)
        captured["layer0_ff_norm"] = _extract_user0_full(result)
        return result

    tt_model.layers[0].ff_norm.forward = _capture_ff_norm_0

    # On TG with QwenLMHead, the base process_output_decode only reads device 0
    # (1/8 of vocab). Override to concatenate all row devices for full vocab.
    if model_args.is_galaxy:
        _orig_pod = tt_model.process_output_decode

        def _full_vocab_pod(tt_out, B, S=1, is_tokens=False, is_log_probs=False):
            if is_tokens or is_log_probs:
                return _orig_pod(tt_out, B, S, is_tokens, is_log_probs)
            tt_out = tt_model.concat_host_output(tt_out).float()
            tt_out = tt_out[:, 0:1, :B, : tt_model.vocab_size].view(B, S, -1)
            return tt_out

        tt_model.process_output_decode = _full_vocab_pod

    rope_delta_val = rope_deltas_tt[0].item()
    logger.info(f"DIAG rope_deltas[0] = {rope_delta_val}")
    generator.update_rope_deltas([rope_delta_val])
    current_pos = torch.tensor([decoding_pos[0]])
    logger.info(
        f"DIAG current_pos = {current_pos.tolist()}, effective RoPE pos = {current_pos[0].item() + rope_delta_val}"
    )
    out_tok = torch.tensor([[tt_prefill_token]])

    # Hook layer 0 attention to capture Q, K after RoPE and before SDPA

    _orig_paged_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode
    _sdpa_captures = {}

    def _hook_paged_sdpa(q, keys, values, **kwargs):
        ckc = kwargs.get("compute_kernel_config")
        if ckc is not None:
            logger.info(
                f"DIAG SDPA compute_kernel_config: math_fidelity={ckc.math_fidelity}, fp32_dest_acc={ckc.fp32_dest_acc_en}"
            )
        else:
            logger.info("DIAG SDPA compute_kernel_config: None")
        q_devs = ttnn.get_device_tensors(q)
        q0 = ttnn.to_torch(q_devs[0]).float()
        _sdpa_captures["q_shape"] = list(q0.shape)
        _sdpa_captures["q_user0_head0"] = q0[0, 0, 0, :].clone()
        _sdpa_captures["q_user0_norm"] = q0[0, :, 0, :].norm().item()
        for h in range(min(q0.shape[1], 4)):
            _sdpa_captures[f"q_user0_head{h}_norm"] = q0[0, h, 0, :].norm().item()
        k_devs = ttnn.get_device_tensors(keys)
        k0 = ttnn.to_torch(k_devs[0]).float()
        _sdpa_captures["k_cache_shape"] = list(k0.shape)
        block_size_val = page_params["page_block_size"]
        # Paged cache shape: [max_num_blocks, n_kv_heads, block_size, head_dim]
        pos_26_phys_block = page_table[0][decoding_pos[0] // block_size_val].item()
        pos_26_offset = decoding_pos[0] % block_size_val
        k_at_26 = k0[pos_26_phys_block, 0, pos_26_offset, :].clone()
        _sdpa_captures["k_at_pos26"] = k_at_26
        _sdpa_captures["k_at_pos26_norm"] = k_at_26.norm().item()
        pos_0_phys_block = page_table[0][0].item()
        k_at_0 = k0[pos_0_phys_block, 0, 0, :].clone()
        _sdpa_captures["k_at_pos0"] = k_at_0
        _sdpa_captures["k_at_pos0_norm"] = k_at_0.norm().item()
        logger.info(f"DIAG SDPA Q shape={list(q0.shape)}, K cache shape={list(k0.shape)}")
        logger.info(
            f"DIAG SDPA Q user0 norms: "
            + ", ".join(f"h{h}={q0[0, h, 0, :].norm():.4e}" for h in range(min(q0.shape[1], 8)))
        )
        logger.info(f"DIAG SDPA K at pos26 (block={pos_26_phys_block}, off={pos_26_offset}) norm={k_at_26.norm():.4e}")
        logger.info(f"DIAG SDPA K at pos0 (block={pos_0_phys_block}, off=0) norm={k_at_0.norm():.4e}")
        logger.info(f"DIAG SDPA cur_pos_tensor={kwargs.get('cur_pos_tensor', 'N/A')}")
        result = _orig_paged_sdpa(q, keys, values, **kwargs)
        r_devs = ttnn.get_device_tensors(result)
        r0 = ttnn.to_torch(r_devs[0]).float()
        _sdpa_captures["sdpa_out_user0"] = r0[0, :, 0, :].clone()
        _sdpa_captures["sdpa_out_user0_norm"] = r0[0, :, 0, :].norm().item()
        logger.info(f"DIAG SDPA output shape={list(r0.shape)}, user0 norm={r0[0, :, 0, :].norm():.4e}")
        for h in range(min(r0.shape[1], 4)):
            logger.info(f"DIAG SDPA out user0 head{h} norm={r0[0, h, 0, :].norm():.4e}")
        for u in [0, 1, 7]:
            if u < r0.shape[2]:
                logger.info(f"DIAG SDPA out user{u} norm={r0[0, :, u, :].norm():.4e}")
        return result

    _sdpa_call_count = [0]

    def _hook_sdpa_once(q, keys, values, **kwargs):
        _sdpa_call_count[0] += 1
        if _sdpa_call_count[0] == 1:
            return _hook_paged_sdpa(q, keys, values, **kwargs)
        return _orig_paged_sdpa(q, keys, values, **kwargs)

    ttnn.transformer.paged_scaled_dot_product_attention_decode = _hook_sdpa_once

    # Compare TT 1D RoPE cos/sin with M-RoPE cos/sin at the decode position
    rope_setup = tt_model.rope_setup
    decode_pos_val = decoding_pos[0]
    effective_pos = int(decode_pos_val + rope_delta_val)
    logger.info(f"DIAG cos_matrix_pt shape={list(rope_setup.cos_matrix_pt.shape)}")
    if rope_setup.cos_matrix_pt.dim() == 4:
        tt_cos_at_pos = rope_setup.cos_matrix_pt[0, 0, effective_pos, :].float()
    elif rope_setup.cos_matrix_pt.dim() == 2:
        tt_cos_at_pos = rope_setup.cos_matrix_pt[effective_pos, :].float()
    else:
        tt_cos_at_pos = rope_setup.cos_matrix_pt.view(-1, rope_setup.head_dim)[effective_pos, :].float()
    if cos.dim() == 4:
        mrope_cos_at_pos = cos[0, 0, decode_pos_val, :].float()
    elif cos.dim() == 3:
        mrope_cos_at_pos = cos[0, decode_pos_val, :].float()
    else:
        mrope_cos_at_pos = cos[decode_pos_val, :].float()
    logger.info(f"DIAG 1D cos shape={list(tt_cos_at_pos.shape)}, M-RoPE cos shape={list(mrope_cos_at_pos.shape)}")
    min_rope_dim = min(tt_cos_at_pos.shape[0], mrope_cos_at_pos.shape[0])
    rope_pcc = torch.corrcoef(torch.stack([tt_cos_at_pos[:min_rope_dim], mrope_cos_at_pos[:min_rope_dim]]))[0, 1].item()
    logger.info(f"DIAG RoPE cos PCC (1D at eff_pos={effective_pos} vs M-RoPE at pos={decode_pos_val}): {rope_pcc:.6f}")
    logger.info(f"DIAG RoPE 1D cos norm={tt_cos_at_pos.norm():.4e}, M-RoPE cos norm={mrope_cos_at_pos.norm():.4e}")
    logger.info(
        f"DIAG RoPE cos max_diff={(tt_cos_at_pos[:min_rope_dim] - mrope_cos_at_pos[:min_rope_dim]).abs().max():.6f}"
    )
    logger.info(f"DIAG RoPE 1D cos first5: {tt_cos_at_pos[:5].tolist()}")
    logger.info(f"DIAG RoPE M-RoPE cos first5: {mrope_cos_at_pos[:5].tolist()}")

    decode_logits, _ = generator.decode_forward(
        out_tok,
        current_pos,
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )
    ttnn.transformer.paged_scaled_dot_product_attention_decode = _orig_paged_sdpa

    # Restore hooks
    tt_model._supports_on_device_sampling = was_sampling
    tt_model.sampling = old_sampling
    tt_model._transform_decode_inputs_device = orig_transform
    for li in diag_layer_indices:
        tt_model.layers[li].forward = orig_layer_fns[li]
    tt_model.layers[0].attention_norm.forward = orig_attn_norm_fn_0
    tt_model.layers[0].attention.forward = orig_attn_fwd_0
    tt_model.layers[0].ff_norm.forward = orig_ff_norm_fn_0
    if model_args.is_galaxy:
        _attn_module.tt_all_reduce = _orig_attn_ar
    if model_args.is_galaxy:
        tt_model.process_output_decode = _orig_pod

    tt_decode_token = decode_logits[0, 0, :].argmax().item()
    logger.info(f"TT decode token: {tt_decode_token} ({processor.tokenizer.decode([tt_decode_token])})")

    ref_logits = ref_decode_out.logits[0, -1, :].float()
    tt_logits_1d = decode_logits[0, 0, :].float()
    logger.info(f"TT logits shape: {tt_logits_1d.shape}, HF logits shape: {ref_logits.shape}")
    min_vocab = min(ref_logits.shape[0], tt_logits_1d.shape[0])
    passing, pcc_msg = comp_pcc(ref_logits[:min_vocab].unsqueeze(0), tt_logits_1d[:min_vocab].unsqueeze(0), 0.80)
    logger.info(f"Decode logits PCC (full): {pcc_msg}")

    chunk = min_vocab // 8
    for row_idx in range(8):
        s, e = row_idx * chunk, (row_idx + 1) * chunk
        _, pcc_chunk = comp_pcc(ref_logits[s:e].unsqueeze(0), tt_logits_1d[s:e].unsqueeze(0), 0.0)
        logger.info(f"DIAG PCC row_chunk[{row_idx}] ({s}:{e}): {pcc_chunk}")

    # Per-layer PCC diagnostic: compare TT layer outputs vs HF
    # Register HF hooks to capture attention output for sublayer comparison
    hf_intermediates = {}

    def _hf_attn_hook(module, input, output):
        hf_intermediates["layer0_attn_out"] = output[0][:, -1, :].detach().float()

    hf_hook_handle = ref_model.model.language_model.layers[0].self_attn.register_forward_hook(_hf_attn_hook)

    def _hf_oproj_hook(module, input, output):
        hf_intermediates["layer0_before_oproj"] = input[0][:, -1, :].detach().float()

    hf_oproj_handle = ref_model.model.language_model.layers[0].self_attn.o_proj.register_forward_hook(_hf_oproj_hook)

    # Hook HF layer 0 self_attn o_proj to capture SDPA output (before o_proj)
    # Also hook q_proj/k_proj/v_proj to capture Q/K/V pre-projection
    hf_layer0_sa = ref_model.model.language_model.layers[0].self_attn
    _hf_orig_q_proj = hf_layer0_sa.q_proj.forward
    _hf_orig_k_proj = hf_layer0_sa.k_proj.forward

    def _hf_q_hook(x):
        result = _hf_orig_q_proj(x)
        hf_intermediates["layer0_q_proj_out"] = result[:, -1, :].detach().float()
        return result

    def _hf_k_hook(x):
        result = _hf_orig_k_proj(x)
        hf_intermediates["layer0_k_proj_out"] = result[:, -1, :].detach().float()
        return result

    hf_layer0_sa.q_proj.forward = _hf_q_hook
    hf_layer0_sa.k_proj.forward = _hf_k_hook

    ref_decode_hs = ref_model(
        new_input_ids,
        attention_mask=new_attn_mask,
        past_key_values=ref_out.past_key_values,
        use_cache=False,
        output_hidden_states=True,
    )
    hf_hook_handle.remove()
    hf_oproj_handle.remove()
    hf_layer0_sa.q_proj.forward = _hf_orig_q_proj
    hf_layer0_sa.k_proj.forward = _hf_orig_k_proj

    # HF hidden_states: [0]=embed, [1]=after layer 0, ..., [n_layers]=after last layer (pre-norm)
    hf_hs = ref_decode_hs.hidden_states
    logger.info(f"DIAG HF hidden_states count: {len(hf_hs)} (embed + {len(hf_hs)-1} layers)")

    # Compare TT Q/K with HF Q/K projections (layer 0)
    from models.tt_transformers.tt.load_checkpoints import reverse_permute

    tt_q_user0 = _sdpa_captures.get("q_user0_head0")
    tt_k_at_26 = _sdpa_captures.get("k_at_pos26")
    tt_k_at_0 = _sdpa_captures.get("k_at_pos0")

    if "layer0_q_proj_out" in hf_intermediates:
        hf_q_flat = hf_intermediates["layer0_q_proj_out"].squeeze()  # [n_heads * head_dim]
        hf_k_flat = hf_intermediates.get("layer0_k_proj_out", torch.zeros(1)).squeeze()
        logger.info(f"DIAG HF Q proj out shape={list(hf_q_flat.shape)}, norm={hf_q_flat.norm():.4e}")
        logger.info(f"DIAG HF K proj out shape={list(hf_k_flat.shape)}, norm={hf_k_flat.norm():.4e}")
        hd = 128
        n_q = hf_q_flat.shape[0] // hd
        n_kv = hf_k_flat.shape[0] // hd
        for h in range(min(n_q, 8)):
            logger.info(f"DIAG HF Q head{h} norm={hf_q_flat[h*hd:(h+1)*hd].norm():.4e}")
        if tt_q_user0 is not None:
            logger.info(f"DIAG TT Q user0 head0 norm={tt_q_user0.norm():.4e}")

    # Compare KV cache at position 26 (decode token) with HF's decode K
    if tt_k_at_26 is not None:
        logger.info(f"DIAG TT K cache@pos26 norm={tt_k_at_26.norm():.4e}, first5={tt_k_at_26[:5].tolist()}")
        hf_decode_kv = ref_decode_out.past_key_values
        if hasattr(hf_decode_kv, "key_cache"):
            hf_k_at_26 = hf_decode_kv.key_cache[0][0, 0, decoding_pos[0], :].float()
        else:
            hf_k_at_26 = hf_decode_kv[0][0][0, 0, decoding_pos[0], :].float()
        hf_k_at_26_meta = reverse_permute(hf_k_at_26.unsqueeze(-1), 1, hf_k_at_26.shape[-1], 1).squeeze(-1)
        k26_pcc = torch.corrcoef(torch.stack([hf_k_at_26_meta, tt_k_at_26]))[0, 1].item()
        logger.info(f"DIAG K head0 PCC (HF cache@pos26 vs TT cache@pos26): {k26_pcc:.6f}")
        logger.info(f"DIAG K head0 HF@26 norm={hf_k_at_26_meta.norm():.4e}, first5={hf_k_at_26_meta[:5].tolist()}")
        k26_raw_pcc = torch.corrcoef(torch.stack([hf_k_at_26, tt_k_at_26]))[0, 1].item()
        logger.info(f"DIAG K head0 PCC (HF@26 raw vs TT@26): {k26_raw_pcc:.6f}")

    # Compare KV cache at position 0 (from prefill) with HF past_key_values
    if tt_k_at_0 is not None:
        hf_kv = ref_out.past_key_values
        if hasattr(hf_kv, "key_cache"):
            hf_k_cache_0 = hf_kv.key_cache[0][0, 0, 0, :].float()
        else:
            hf_k_cache_0 = hf_kv[0][0][0, 0, 0, :].float()
        hf_k_cache_0_meta = reverse_permute(hf_k_cache_0.unsqueeze(-1), 1, hf_k_cache_0.shape[-1], 1).squeeze(-1)
        k0_pcc = torch.corrcoef(torch.stack([hf_k_cache_0_meta, tt_k_at_0]))[0, 1].item()
        logger.info(f"DIAG K head0 PCC (HF cache@pos0 vs TT cache@pos0): {k0_pcc:.6f}")
        logger.info(f"DIAG K head0 max_diff@pos0: {(hf_k_cache_0_meta - tt_k_at_0).abs().max():.6f}")
        logger.info(f"DIAG K head0 HF first5: {hf_k_cache_0_meta[:5].tolist()}")
        logger.info(f"DIAG K head0 TT first5: {tt_k_at_0[:5].tolist()}")

        # Manual attention computation on CPU for head 0 using TT's Q and KV cache
        if "sdpa_out_user0" in _sdpa_captures and tt_k_at_26 is not None and tt_q_user0 is not None:
            q_h0 = _sdpa_captures["q_user0_head0"]
            k_devs = ttnn.get_device_tensors(tt_kv_cache[0][0])
            k0_full = ttnn.to_torch(k_devs[0]).float()
            v_devs = ttnn.get_device_tensors(tt_kv_cache[0][1])
            v0_full = ttnn.to_torch(v_devs[0]).float()
            block_size_val = page_params["page_block_size"]
            n_tokens = decoding_pos[0] + 1
            k_seq = []
            v_seq = []
            for p in range(n_tokens):
                blk = page_table[0][p // block_size_val].item()
                off = p % block_size_val
                k_seq.append(k0_full[blk, 0, off, :])
                v_seq.append(v0_full[blk, 0, off, :])
            k_seq_t = torch.stack(k_seq)  # [n_tokens, head_dim]
            v_seq_t = torch.stack(v_seq)  # [n_tokens, head_dim]
            scores = (q_h0 @ k_seq_t.T) / (128**0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            manual_out = attn_weights @ v_seq_t
            tt_sdpa_h0 = _sdpa_captures["sdpa_out_user0"][0, :]
            manual_pcc = torch.corrcoef(torch.stack([manual_out, tt_sdpa_h0]))[0, 1].item()
            logger.info(f"DIAG Manual attention PCC vs TT SDPA (head0, user0): {manual_pcc:.6f}")
            logger.info(f"DIAG Manual attn norm={manual_out.norm():.4e}, TT SDPA h0 norm={tt_sdpa_h0.norm():.4e}")
            logger.info(f"DIAG Manual attn first5: {manual_out[:5].tolist()}")
            logger.info(f"DIAG TT SDPA h0 first5: {tt_sdpa_h0[:5].tolist()}")
            hf_sdpa_h0 = hf_intermediates.get("layer0_before_oproj")
            if hf_sdpa_h0 is not None:
                hf_sdpa_h0 = hf_sdpa_h0.squeeze()
                hd = 128
                hf_sdpa_head0_raw = hf_sdpa_h0[:hd]
                hf_vs_manual_raw = torch.corrcoef(torch.stack([hf_sdpa_head0_raw, manual_out]))[0, 1].item()
                logger.info(f"DIAG HF SDPA head0 vs Manual attention PCC (raw): {hf_vs_manual_raw:.6f}")
                hf_sdpa_head0_meta = reverse_permute(hf_sdpa_head0_raw.unsqueeze(-1), 1, hd, 1).squeeze(-1)
                hf_vs_manual_perm = torch.corrcoef(torch.stack([hf_sdpa_head0_meta, manual_out]))[0, 1].item()
                logger.info(f"DIAG HF SDPA head0 vs Manual attention PCC (permuted): {hf_vs_manual_perm:.6f}")
                hf_vs_tt_raw = torch.corrcoef(torch.stack([hf_sdpa_head0_raw, tt_sdpa_h0]))[0, 1].item()
                logger.info(f"DIAG HF SDPA head0 vs TT SDPA PCC (raw): {hf_vs_tt_raw:.6f}")
                logger.info(f"DIAG HF SDPA head0 first5: {hf_sdpa_head0_raw[:5].tolist()}")
                logger.info(f"DIAG HF SDPA head0 norm={hf_sdpa_head0_raw.norm():.4e}")

    if "embed" in captured:
        hf_embed = hf_hs[0][0, -1, :].float()
        tt_embed = captured["embed"]
        dim = min(hf_embed.shape[0], tt_embed.shape[0])
        _, epcc = comp_pcc(hf_embed[:dim].unsqueeze(0), tt_embed[:dim].unsqueeze(0), 0.0)
        logger.info(f"DIAG embed PCC: {epcc}  (HF std={hf_embed.std():.4e}, TT std={tt_embed.std():.4e})")

    for li in diag_layer_indices:
        key = f"layer{li}"
        if key not in captured:
            continue
        hf_layer_out = hf_hs[li + 1][0, -1, :].float()
        tt_layer_out = captured[key]
        dim = min(hf_layer_out.shape[0], tt_layer_out.shape[0])
        _, lpcc = comp_pcc(hf_layer_out[:dim].unsqueeze(0), tt_layer_out[:dim].unsqueeze(0), 0.0)
        logger.info(
            f"DIAG layer{li} PCC: {lpcc}  " f"(HF std={hf_layer_out.std():.4e}, TT std={tt_layer_out.std():.4e})"
        )
        h_chunk = dim // cols
        for c in range(cols):
            s, e = c * h_chunk, (c + 1) * h_chunk
            _, cpcc = comp_pcc(hf_layer_out[s:e].unsqueeze(0), tt_layer_out[s:e].unsqueeze(0), 0.0)
            logger.info(f"DIAG   layer{li}_col{c} ({s}:{e}): PCC={cpcc}")

    # Sublayer diagnostics: distributed RMSNorm and attention output
    if "layer0_attn_norm" in captured:
        hf_layer0 = ref_model.model.language_model.layers[0]
        hf_embed_last = hf_hs[0][:, -1:, :].float()
        hf_attn_norm_out = hf_layer0.input_layernorm(hf_embed_last).squeeze().float()
        tt_attn_norm = captured["layer0_attn_norm"]
        ndim = min(hf_attn_norm_out.shape[0], tt_attn_norm.shape[0])
        _, norm_pcc = comp_pcc(hf_attn_norm_out[:ndim].unsqueeze(0), tt_attn_norm[:ndim].unsqueeze(0), 0.0)
        logger.info(
            f"DIAG layer0_attn_norm PCC: {norm_pcc}  "
            f"(HF std={hf_attn_norm_out.std():.4e}, TT std={tt_attn_norm.std():.4e})"
        )
        h_chunk = ndim // cols
        for c in range(cols):
            s, e = c * h_chunk, (c + 1) * h_chunk
            _, cpcc = comp_pcc(hf_attn_norm_out[s:e].unsqueeze(0), tt_attn_norm[s:e].unsqueeze(0), 0.0)
            logger.info(f"DIAG   layer0_attn_norm_col{c} ({s}:{e}): PCC={cpcc}")

    if "layer0_attn_out" in captured and "layer0_attn_out" in hf_intermediates:
        hf_attn_out = hf_intermediates["layer0_attn_out"].squeeze().float()
        tt_attn_out = captured["layer0_attn_out"]
        ndim = min(hf_attn_out.shape[0], tt_attn_out.shape[0])
        _, attn_pcc = comp_pcc(hf_attn_out[:ndim].unsqueeze(0), tt_attn_out[:ndim].unsqueeze(0), 0.0)
        logger.info(
            f"DIAG layer0_attn_out PCC: {attn_pcc}  "
            f"(HF std={hf_attn_out.std():.4e}, TT std={tt_attn_out.std():.4e})"
        )
        h_chunk = ndim // cols
        for c in range(cols):
            s, e = c * h_chunk, (c + 1) * h_chunk
            _, cpcc = comp_pcc(hf_attn_out[s:e].unsqueeze(0), tt_attn_out[s:e].unsqueeze(0), 0.0)
            logger.info(f"DIAG   layer0_attn_out_col{c} ({s}:{e}): PCC={cpcc}")

    if "layer0_res_after_attn" in captured:
        hf_embed_1d = hf_hs[0][0, -1, :].float()
        hf_attn_contribution = hf_intermediates.get("layer0_attn_out", torch.zeros(1)).squeeze().float()
        hf_res_after_attn = hf_embed_1d + hf_attn_contribution
        tt_res = captured["layer0_res_after_attn"]
        ndim = min(hf_res_after_attn.shape[0], tt_res.shape[0])
        _, res_pcc = comp_pcc(hf_res_after_attn[:ndim].unsqueeze(0), tt_res[:ndim].unsqueeze(0), 0.0)
        logger.info(
            f"DIAG layer0_res_after_attn PCC: {res_pcc}  "
            f"(HF std={hf_res_after_attn.std():.4e}, TT std={tt_res.std():.4e})"
        )

    if "layer0_ff_norm" in captured:
        hf_embed_1d = hf_hs[0][0, -1, :].float()
        hf_attn_contribution = hf_intermediates.get("layer0_attn_out", torch.zeros(1)).squeeze().float()
        hf_res_after_attn = hf_embed_1d + hf_attn_contribution
        hf_ff_norm_out = (
            ref_model.model.language_model.layers[0]
            .post_attention_layernorm(hf_res_after_attn.unsqueeze(0).unsqueeze(0))
            .squeeze()
            .float()
        )
        tt_ff_norm = captured["layer0_ff_norm"]
        ndim = min(hf_ff_norm_out.shape[0], tt_ff_norm.shape[0])
        _, ff_pcc = comp_pcc(hf_ff_norm_out[:ndim].unsqueeze(0), tt_ff_norm[:ndim].unsqueeze(0), 0.0)
        logger.info(
            f"DIAG layer0_ff_norm PCC: {ff_pcc}  " f"(HF std={hf_ff_norm_out.std():.4e}, TT std={tt_ff_norm.std():.4e})"
        )

    # QKV reference: compare TT QKV (after all-reduce) with manually computed HF QKV for row 0
    if "layer0_qkv_after_ar_row0" in captured and "layer0_attn_norm" in captured:
        from models.tt_transformers.tt.load_checkpoints import reverse_permute

        hf_layer0 = ref_model.model.language_model.layers[0].self_attn
        hf_input = hf_hs[0][0, -1, :].float()
        hf_norm_w = ref_model.model.language_model.layers[0].input_layernorm.weight.float()
        hf_norm_input = torch.nn.functional.rms_norm(hf_input, (hf_input.shape[-1],), hf_norm_w)
        hf_q_out = hf_layer0.q_proj.weight.float() @ hf_norm_input + hf_layer0.q_proj.bias.float()
        hf_k_out = hf_layer0.k_proj.weight.float() @ hf_norm_input + hf_layer0.k_proj.bias.float()
        hf_v_out = hf_layer0.v_proj.weight.float() @ hf_norm_input + hf_layer0.v_proj.bias.float()

        attn0 = tt_model.layers[0].attention
        hd = attn0.head_dim
        n_q_real = hf_q_out.shape[0] // hd
        n_kv_real = hf_k_out.shape[0] // hd
        hf_q_meta = reverse_permute(hf_q_out.unsqueeze(-1), n_q_real, hf_q_out.shape[0], 1).squeeze(-1)
        hf_k_meta = reverse_permute(hf_k_out.unsqueeze(-1), n_kv_real, hf_k_out.shape[0], 1).squeeze(-1)
        hf_q_bias_meta = reverse_permute(
            hf_layer0.q_proj.bias.float().unsqueeze(-1), n_q_real, hf_q_out.shape[0], 1
        ).squeeze(-1)
        hf_k_bias_meta = reverse_permute(
            hf_layer0.k_proj.bias.float().unsqueeze(-1), n_kv_real, hf_k_out.shape[0], 1
        ).squeeze(-1)

        q_order = attn0._build_q_head_order()
        kv_order = attn0._build_kv_head_order()

        hf_q_rearranged = torch.cat(
            [hf_q_meta[idx * hd : (idx + 1) * hd] if idx is not None else torch.zeros(hd) for idx in q_order]
        )
        hf_k_rearranged = torch.cat([hf_k_meta[idx * hd : (idx + 1) * hd] for idx in kv_order])
        hf_v_rearranged = torch.cat([hf_v_out[idx * hd : (idx + 1) * hd] for idx in kv_order])

        hf_qkv_full = torch.cat(
            [
                torch.cat(
                    [
                        hf_q_rearranged[i * (attn0.n_local_heads * hd) : (i + 1) * (attn0.n_local_heads * hd)],
                        hf_k_rearranged[i * (attn0.n_local_kv_heads * hd) : (i + 1) * (attn0.n_local_kv_heads * hd)],
                        hf_v_rearranged[i * (attn0.n_local_kv_heads * hd) : (i + 1) * (attn0.n_local_kv_heads * hd)],
                    ]
                )
                for i in range(rows)
            ]
        )

        hf_qkv_row0 = hf_qkv_full[:768]
        tt_qkv_row0 = captured["layer0_qkv_after_ar_row0"]
        _, qkv_pcc = comp_pcc(hf_qkv_row0.unsqueeze(0), tt_qkv_row0.unsqueeze(0), 0.0)
        logger.info(f"DIAG layer0 QKV row0 PCC (TT vs HF meta-fmt ref): {qkv_pcc}")
        logger.info(f"DIAG layer0 QKV row0 HF(meta) first5: {hf_qkv_row0[:5].tolist()}")
        logger.info(f"DIAG layer0 QKV row0 TT first5: {tt_qkv_row0[:5].tolist()}")
        logger.info(f"DIAG layer0 QKV row0 HF norm={hf_qkv_row0.norm():.4e}, TT norm={tt_qkv_row0.norm():.4e}")

        # Also check: Wo output inf analysis
        if "layer0_attn_out" in captured:
            tt_attn_full = captured["layer0_attn_out"]
            n_inf = torch.isinf(tt_attn_full).sum().item()
            n_nan = torch.isnan(tt_attn_full).sum().item()
            logger.info(
                f"DIAG layer0_attn_out (user0 full) #inf={n_inf}, #nan={n_nan}, max={tt_attn_full.abs().max():.4e}"
            )

    # Compare SDPA output (before o_proj/Wo) between TT and HF
    if "layer0_wo_input_row0" in captured and "layer0_before_oproj" in hf_intermediates:
        hf_before_oproj = hf_intermediates["layer0_before_oproj"].squeeze().float()
        tt_wo_input = captured["layer0_wo_input_row0"]
        hd = tt_model.layers[0].attention.head_dim
        n_local = tt_model.layers[0].attention.n_local_heads
        q_order = tt_model.layers[0].attention._build_q_head_order()
        row0_heads = q_order[:n_local]
        hf_row0 = torch.cat([hf_before_oproj[h * hd : (h + 1) * hd] for h in row0_heads if h is not None])
        ndim = min(hf_row0.shape[0], tt_wo_input.shape[0])
        _, wo_in_pcc = comp_pcc(hf_row0[:ndim].unsqueeze(0), tt_wo_input[:ndim].unsqueeze(0), 0.0)
        logger.info(f"DIAG layer0 Wo INPUT (SDPA output) row0 PCC: {wo_in_pcc}")
        logger.info(f"DIAG layer0 Wo INPUT HF first5: {hf_row0[:5].tolist()}")
        logger.info(f"DIAG layer0 Wo INPUT TT first5: {tt_wo_input[:5].tolist()}")
        logger.info(f"DIAG layer0 Wo INPUT HF norm={hf_row0.norm():.4e}, TT norm={tt_wo_input.norm():.4e}")

    ref_top5 = ref_logits.topk(5).indices.tolist()
    tt_top5 = tt_logits_1d.topk(5).indices.tolist()
    logger.info(f"HF decode top5: {ref_top5}")
    logger.info(f"TT decode top5: {tt_top5}")

    if ref_decode_token == tt_decode_token:
        logger.info("PASSED: Text decode (exact match)")
    elif ref_decode_token in tt_top5:
        logger.warning(f"SOFT PASS: HF token {ref_decode_token} in TT top5 {tt_top5}")
    else:
        logger.error(f"FAILED: Decode token mismatch HF={ref_decode_token} TT={tt_decode_token}")

    assert passing, f"Decode logits PCC {pcc_msg} below threshold"
    logger.info("PASSED: Text decode")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Multi-step decode (5 tokens)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_text_decode_multistep(mesh_device, reset_seeds, ensure_gc):
    """Prefill one user, then decode 5 tokens; compare each to HuggingFace."""
    n_decode_steps = 5
    ref_model, processor = _load_hf_model()
    prompt = _build_text_prompt()
    batch_size = 1

    text = processor.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    pad_token_id = processor.tokenizer.pad_token_id or 151643

    text_embeds = ref_model.model.language_model.embed_tokens(input_ids)

    # HF: prefill + N decode steps
    hf_tokens = []
    attn_mask = inputs.attention_mask.clone()
    past_kv = None
    cur_ids = input_ids
    for step in range(n_decode_steps + 1):
        ref_out = ref_model(cur_ids, attention_mask=attn_mask, past_key_values=past_kv, use_cache=True)
        past_kv = ref_out.past_key_values
        tok = ref_out.logits[0, -1, :].argmax().item()
        hf_tokens.append(tok)
        cur_ids = torch.tensor([[tok]])
        attn_mask = torch.cat([attn_mask, torch.ones(1, 1, dtype=attn_mask.dtype)], dim=1)
    logger.info(f"HF tokens: {hf_tokens} = '{processor.tokenizer.decode(hf_tokens)}'")

    # TT: prefill + N decode steps
    page_params = {"page_block_size": 32, "page_max_num_blocks": 1024}
    paged_attention_config = PagedAttentionConfig(
        block_size=page_params["page_block_size"],
        max_num_blocks=page_params["page_max_num_blocks"],
    )
    model_args = ModelArgs(
        mesh_device,
        instruct=True,
        max_batch_size=batch_size,
        optimizations=DecodersPrecision.performance(28, "Qwen2.5-VL-7B-Instruct"),
        max_seq_len=4096,
    )
    state_dict = model_args.load_state_dict()
    tt_model = Transformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
    )
    tt_kv_cache = [l.attention.layer_past for l in tt_model.layers]
    generator = Generator(tt_model, model_args, mesh_device, processor=processor, tokenizer=processor.tokenizer)

    page_table = torch.randperm(paged_attention_config.max_num_blocks).reshape(
        batch_size, paged_attention_config.max_num_blocks // batch_size
    )

    input_prefill_pt, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        text_embeds,
        model_args,
        inputs.attention_mask,
        pad_embedding=ref_model.model.language_model.embed_tokens(torch.tensor(pad_token_id)),
    )
    cos, sin, rope_deltas_tt = multimodal_rope_from_hf(
        inputs,
        text_embeds,
        ref_model,
        model_args,
        pad_token_id=pad_token_id,
    )

    tt_logits = generator.prefill_forward_text(
        input_prefill_pt,
        rot_mats=(cos, sin),
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=torch.tensor(decoding_pos),
    )
    tt_prefill_token = tt_logits[0].argmax().item()
    assert tt_prefill_token == hf_tokens[0], f"Prefill mismatch: HF={hf_tokens[0]} TT={tt_prefill_token}"

    generator.update_rope_deltas([rope_deltas_tt[0].item()])
    current_pos = torch.tensor([decoding_pos[0]])
    out_tok = torch.tensor([[tt_prefill_token]])

    tt_tokens = [tt_prefill_token]
    for step in range(n_decode_steps):
        decode_logits, _ = generator.decode_forward(
            out_tok,
            current_pos,
            enable_trace=False,
            page_table=page_table,
            kv_cache=tt_kv_cache,
        )
        tok = decode_logits[0, 0, :].argmax().item()
        tt_tokens.append(tok)
        out_tok = torch.tensor([[tok]])
        current_pos += 1

    logger.info(f"TT tokens: {tt_tokens} = '{processor.tokenizer.decode(tt_tokens)}'")

    mismatches = [(i, h, t) for i, (h, t) in enumerate(zip(hf_tokens, tt_tokens)) if h != t]
    if mismatches:
        for step, hf_tok, tt_tok in mismatches:
            logger.error(
                f"Step {step}: HF={hf_tok}({processor.tokenizer.decode([hf_tok])}) "
                f"TT={tt_tok}({processor.tokenizer.decode([tt_tok])})"
            )
    assert len(mismatches) == 0, f"{len(mismatches)} token mismatches in {n_decode_steps + 1} steps"
    logger.info(f"PASSED: Multi-step decode ({n_decode_steps + 1} tokens match)")
