# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Per-layer and per-sublayer PCC breakdown for Qwen2.5-VL decode on TG.

Runs prefill + one decode step and compares TT hidden states against
HuggingFace at every layer boundary and at key sublayers (attention norm,
attention output, residual-after-attention, ff norm).

Outputs a sorted table showing:
  - Per-layer output PCC and the drop from the previous layer
  - Per-sublayer PCC within each layer
  - The layer/sublayer with the maximum PCC failure

Usage:
    MESH_DEVICE=TG HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
        pytest models/demos/qwen25_vl/tests/test_decode_pcc_breakdown.py -xvs
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen25_vl.tt.common import PagedAttentionConfig, multimodal_rope_from_hf, preprocess_inputs_prefill
from models.demos.qwen25_vl.tt.generator import Generator
from models.demos.qwen25_vl.tt.model import Transformer
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), None)

# Which layers get sublayer hooks. None = all layers.
# Set to e.g. [0, 7, 14, 21, 27] for faster runs.
SUBLAYER_LAYERS = None

PCC_THRESHOLD = 0.80


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


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two float tensors (flattened to 1-D)."""
    a, b = a.flatten(), b.flatten()
    n = min(a.shape[0], b.shape[0])
    if n < 2:
        return float("nan")
    _, val = comp_pcc(a[:n].unsqueeze(0), b[:n].unsqueeze(0), 0.0)
    return float(val)


def _make_extractor(cols: int):
    """Return a function that extracts user 0's full hidden state from a TT tensor."""

    def _extract_user0(tt_tensor):
        devs = ttnn.get_device_tensors(tt_tensor)
        parts = []
        for c in range(cols):
            tc = ttnn.to_torch(devs[c]).float()
            parts.append(tc[0, 0, 0, :].clone())
        return torch.cat(parts, dim=0)

    return _extract_user0


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_decode_pcc_breakdown(mesh_device, reset_seeds, ensure_gc):
    """Measure per-layer and per-sublayer PCC for a single decode step."""
    ref_model, processor = _load_hf_model()
    batch_size = 1

    # ── Tokenize ────────────────────────────────────────────────────────
    prompt = [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]
    text = processor.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, padding=True, return_tensors="pt")
    input_ids = inputs.input_ids
    pad_token_id = processor.tokenizer.pad_token_id or 151643
    text_embeds = ref_model.model.language_model.embed_tokens(input_ids)

    # ── HF: prefill + 1 decode ──────────────────────────────────────────
    ref_out = ref_model(input_ids, attention_mask=inputs.attention_mask, use_cache=True)
    prefill_token = ref_out.logits[0, -1, :].argmax().item()
    logger.info(f"HF prefill token: {prefill_token} ({processor.tokenizer.decode([prefill_token])})")

    new_input_ids = torch.tensor([[prefill_token]])
    new_attn_mask = torch.cat([inputs.attention_mask, torch.ones(1, 1, dtype=inputs.attention_mask.dtype)], dim=1)

    hf_layers = ref_model.model.language_model.layers
    n_layers = len(hf_layers)
    sublayer_indices = SUBLAYER_LAYERS if SUBLAYER_LAYERS is not None else list(range(n_layers))

    # HF sublayer hooks: capture attention output at each hooked layer
    hf_sublayer = {}
    hf_handles = []
    for li in sublayer_indices:

        def _make_attn_hook(idx):
            def hook(module, inp, out):
                hf_sublayer[f"layer{idx}_attn_out"] = out[0][0, -1, :].detach().float()

            return hook

        hf_handles.append(hf_layers[li].self_attn.register_forward_hook(_make_attn_hook(li)))

    ref_decode_out = ref_model(
        new_input_ids,
        attention_mask=new_attn_mask,
        past_key_values=ref_out.past_key_values,
        use_cache=False,
        output_hidden_states=True,
    )
    for h in hf_handles:
        h.remove()

    ref_logits = ref_decode_out.logits[0, -1, :].float()
    hf_hs = ref_decode_out.hidden_states
    logger.info(f"HF hidden states: {len(hf_hs)} (embed + {len(hf_hs) - 1} layers)")

    # Compute HF sublayer references (attn_norm, res_after_attn, ff_norm) from captured states
    for li in sublayer_indices:
        hf_layer = hf_layers[li]
        hf_input = hf_hs[li][:, -1:, :].float()
        hf_sublayer[f"layer{li}_attn_norm"] = hf_layer.input_layernorm(hf_input).squeeze().float()
        if f"layer{li}_attn_out" in hf_sublayer:
            hf_res = hf_hs[li][0, -1, :].float() + hf_sublayer[f"layer{li}_attn_out"].squeeze()
            hf_sublayer[f"layer{li}_res_after_attn"] = hf_res
            hf_sublayer[f"layer{li}_ff_norm"] = (
                hf_layer.post_attention_layernorm(hf_res.unsqueeze(0).unsqueeze(0)).squeeze().float()
            )

    # ── TT: build model + prefill ───────────────────────────────────────
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
    tt_kv_cache = [layer.attention.layer_past for layer in tt_model.layers]
    generator = Generator(
        tt_model,
        model_args,
        mesh_device,
        processor=processor,
        tokenizer=processor.tokenizer,
    )

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
    assert tt_prefill_token == prefill_token, f"Prefill mismatch: HF={prefill_token} TT={tt_prefill_token}"

    # ── TT decode with hooks ────────────────────────────────────────────
    was_sampling = tt_model._supports_on_device_sampling
    old_sampling = tt_model.sampling
    tt_model._supports_on_device_sampling = False
    tt_model.sampling = None

    cols = model_args.cluster_shape[1] if model_args.is_galaxy else 1
    extract = _make_extractor(cols)
    captured = {}

    # Hook embedding
    orig_transform = tt_model._transform_decode_inputs_device

    def _cap_embed(x):
        result = orig_transform(x)
        captured["embed"] = extract(result)
        return result

    tt_model._transform_decode_inputs_device = _cap_embed

    # Hook all layer outputs
    orig_layer_fns = {}
    for li in range(n_layers):
        layer = tt_model.layers[li]
        orig_layer_fns[li] = layer.forward

        def _make_layer_hook(idx, fn):
            def _h(*args, **kwargs):
                r = fn(*args, **kwargs)
                captured[f"layer{idx}"] = extract(r)
                return r

            return _h

        layer.forward = _make_layer_hook(li, layer.forward)

    # Hook sublayer components for selected layers
    sublayer_orig = {}
    for li in sublayer_indices:
        layer = tt_model.layers[li]

        # attention_norm
        sublayer_orig[f"{li}_attn_norm"] = layer.attention_norm.forward

        def _make_norm_hook(idx, fn):
            def _h(x, mode, norm_config=None):
                r = fn(x, mode, norm_config=norm_config)
                captured[f"layer{idx}_attn_norm"] = extract(r)
                return r

            return _h

        layer.attention_norm.forward = _make_norm_hook(li, layer.attention_norm.forward)

        # attention
        sublayer_orig[f"{li}_attn"] = layer.attention.forward

        def _make_attn_hook(idx, fn):
            def _h(*args, **kwargs):
                r = fn(*args, **kwargs)
                captured[f"layer{idx}_attn_out"] = extract(r)
                return r

            return _h

        layer.attention.forward = _make_attn_hook(li, layer.attention.forward)

        # ff_norm: captures both input (res_after_attn) and output (ff_norm)
        sublayer_orig[f"{li}_ff_norm"] = layer.ff_norm.forward

        def _make_ff_hook(idx, fn):
            def _h(x, mode, norm_config=None):
                captured[f"layer{idx}_res_after_attn"] = extract(x)
                r = fn(x, mode, norm_config=norm_config)
                captured[f"layer{idx}_ff_norm"] = extract(r)
                return r

            return _h

        layer.ff_norm.forward = _make_ff_hook(li, layer.ff_norm.forward)

    # Override process_output_decode for full vocab on TG
    _orig_pod = None
    if model_args.is_galaxy:
        _orig_pod = tt_model.process_output_decode

        def _full_vocab_pod(tt_out, B, S=1, is_tokens=False, is_log_probs=False):
            if is_tokens or is_log_probs:
                return _orig_pod(tt_out, B, S, is_tokens, is_log_probs)
            tt_out = tt_model.concat_host_output(tt_out).float()
            tt_out = tt_out[:, 0:1, :B, : tt_model.vocab_size].view(B, S, -1)
            return tt_out

        tt_model.process_output_decode = _full_vocab_pod

    # Run decode
    generator.update_rope_deltas([rope_deltas_tt[0].item()])
    current_pos = torch.tensor([decoding_pos[0]])
    out_tok = torch.tensor([[tt_prefill_token]])

    decode_logits, _ = generator.decode_forward(
        out_tok,
        current_pos,
        enable_trace=False,
        page_table=page_table,
        kv_cache=tt_kv_cache,
    )

    # ── Restore hooks ───────────────────────────────────────────────────
    tt_model._supports_on_device_sampling = was_sampling
    tt_model.sampling = old_sampling
    tt_model._transform_decode_inputs_device = orig_transform
    for li in range(n_layers):
        tt_model.layers[li].forward = orig_layer_fns[li]
    for li in sublayer_indices:
        tt_model.layers[li].attention_norm.forward = sublayer_orig[f"{li}_attn_norm"]
        tt_model.layers[li].attention.forward = sublayer_orig[f"{li}_attn"]
        tt_model.layers[li].ff_norm.forward = sublayer_orig[f"{li}_ff_norm"]
    if _orig_pod is not None:
        tt_model.process_output_decode = _orig_pod

    # ── Compute PCC at every point ──────────────────────────────────────
    layer_pccs = []  # (name, pcc)

    # Embedding
    if "embed" in captured:
        hf_embed = hf_hs[0][0, -1, :].float()
        layer_pccs.append(("embed", _pcc(hf_embed, captured["embed"])))

    # Per-layer outputs
    for li in range(n_layers):
        key = f"layer{li}"
        if key in captured:
            hf_out = hf_hs[li + 1][0, -1, :].float()
            layer_pccs.append((f"L{li:02d}", _pcc(hf_out, captured[key])))

    # Final logits
    tt_logits_1d = decode_logits[0, 0, :].float()
    min_vocab = min(ref_logits.shape[0], tt_logits_1d.shape[0])
    logits_pcc = _pcc(ref_logits[:min_vocab], tt_logits_1d[:min_vocab])
    layer_pccs.append(("logits", logits_pcc))

    # Sublayer PCCs: (layer_name, sublayer_name, pcc)
    sublayer_pccs = []
    for li in sublayer_indices:
        for stage in ("attn_norm", "attn_out", "res_after_attn", "ff_norm"):
            tt_key = f"layer{li}_{stage}"
            hf_key = f"layer{li}_{stage}"
            if tt_key in captured and hf_key in hf_sublayer:
                sublayer_pccs.append((f"L{li:02d}", stage, _pcc(hf_sublayer[hf_key], captured[tt_key])))

    # ── Report: layer output PCC table ──────────────────────────────────
    logger.info("")
    logger.info("=" * 62)
    logger.info("  PER-LAYER OUTPUT PCC  (TT vs HF at each layer boundary)")
    logger.info("=" * 62)
    logger.info(f"  {'Stage':<8} {'PCC':>10}  {'Drop':>10}  {'HF std':>10}  {'TT std':>10}")
    logger.info("-" * 62)

    prev_pcc = 1.0
    worst_layer_drop = 0.0
    worst_layer_drop_name = ""
    worst_layer_pcc = 1.0
    worst_layer_pcc_name = ""

    for name, pcc in layer_pccs:
        drop = prev_pcc - pcc
        drop_str = f"{drop:+.6f}" if name != "embed" else ""

        # Std dev info for layer outputs
        hf_std_str = ""
        tt_std_str = ""
        if name == "embed" and "embed" in captured:
            hf_std_str = f"{hf_hs[0][0, -1, :].float().std():.3e}"
            tt_std_str = f"{captured['embed'].std():.3e}"
        elif name.startswith("L"):
            li = int(name[1:])
            hf_std_str = f"{hf_hs[li + 1][0, -1, :].float().std():.3e}"
            tt_std_str = f"{captured[f'layer{li}'].std():.3e}"
        elif name == "logits":
            hf_std_str = f"{ref_logits[:min_vocab].std():.3e}"
            tt_std_str = f"{tt_logits_1d[:min_vocab].std():.3e}"

        logger.info(f"  {name:<8} {pcc:>10.6f}  {drop_str:>10}  {hf_std_str:>10}  {tt_std_str:>10}")

        if name != "embed" and drop > worst_layer_drop:
            worst_layer_drop = drop
            worst_layer_drop_name = name
        if pcc < worst_layer_pcc:
            worst_layer_pcc = pcc
            worst_layer_pcc_name = name
        prev_pcc = pcc

    logger.info("-" * 62)
    logger.info(f"  Worst drop:  {worst_layer_drop:.6f}  at {worst_layer_drop_name}")
    logger.info(f"  Worst PCC:   {worst_layer_pcc:.6f}  at {worst_layer_pcc_name}")

    # ── Report: sublayer PCC table ──────────────────────────────────────
    logger.info("")
    logger.info("=" * 62)
    logger.info("  PER-SUBLAYER PCC  (within-layer breakdown)")
    logger.info("=" * 62)
    logger.info(f"  {'Layer':<6} {'Sublayer':<18} {'PCC':>10}  {'Intra-drop':>12}")
    logger.info("-" * 62)

    worst_sublayer_pcc = 1.0
    worst_sublayer_name = ""
    worst_intra_drop = 0.0
    worst_intra_drop_name = ""

    for li in sublayer_indices:
        layer_subs = [(s, p) for (l, s, p) in sublayer_pccs if l == f"L{li:02d}"]
        if not layer_subs:
            continue

        # Input PCC to this layer (output of previous layer, or embed)
        if li == 0:
            input_pcc = next((p for n, p in layer_pccs if n == "embed"), 1.0)
        else:
            input_pcc = next((p for n, p in layer_pccs if n == f"L{li - 1:02d}"), 1.0)

        prev_sub_pcc = input_pcc
        for stage, pcc in layer_subs:
            intra_drop = prev_sub_pcc - pcc
            drop_str = f"{intra_drop:+.6f}"
            logger.info(f"  L{li:02d}   {stage:<18} {pcc:>10.6f}  {drop_str:>12}")

            if pcc < worst_sublayer_pcc:
                worst_sublayer_pcc = pcc
                worst_sublayer_name = f"L{li:02d}/{stage}"
            if intra_drop > worst_intra_drop:
                worst_intra_drop = intra_drop
                worst_intra_drop_name = f"L{li:02d}/{stage}"
            prev_sub_pcc = pcc

        # Also show layer output PCC in the sublayer table
        layer_out_pcc = next((p for n, p in layer_pccs if n == f"L{li:02d}"), None)
        if layer_out_pcc is not None:
            intra_drop = prev_sub_pcc - layer_out_pcc
            logger.info(f"  L{li:02d}   {'output':<18} {layer_out_pcc:>10.6f}  {intra_drop:+.6f}")

        logger.info("")

    logger.info("-" * 62)
    logger.info(f"  Worst sublayer PCC:    {worst_sublayer_pcc:.6f}  at {worst_sublayer_name}")
    logger.info(f"  Worst intra-drop:      {worst_intra_drop:.6f}  at {worst_intra_drop_name}")
    logger.info(f"  Final logits PCC:      {logits_pcc:.6f}")

    # ── Token comparison ────────────────────────────────────────────────
    ref_decode_token = ref_logits.argmax().item()
    tt_decode_token = tt_logits_1d.argmax().item()
    ref_top5 = ref_logits.topk(5).indices.tolist()
    tt_top5 = tt_logits_1d.topk(5).indices.tolist()
    logger.info(f"\n  HF decode token: {ref_decode_token} ({processor.tokenizer.decode([ref_decode_token])})")
    logger.info(f"  TT decode token: {tt_decode_token} ({processor.tokenizer.decode([tt_decode_token])})")
    logger.info(f"  HF top5: {ref_top5}")
    logger.info(f"  TT top5: {tt_top5}")

    # ── Assert ──────────────────────────────────────────────────────────
    assert logits_pcc > PCC_THRESHOLD, (
        f"Decode logits PCC {logits_pcc:.6f} below {PCC_THRESHOLD}. "
        f"Worst layer drop: {worst_layer_drop:.6f} at {worst_layer_drop_name}. "
        f"Worst intra-layer drop: {worst_intra_drop:.6f} at {worst_intra_drop_name}."
    )
