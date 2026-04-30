"""Verify reference functional.py against HuggingFace Molmo2-8B outputs.

Run with:
    cd /home/ttuser/ssinghal/PR-fix/molmo2/tt-metal
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    python models/demos/molmo2/reference/test_functional.py

Golden outputs are saved to models/demos/molmo2/reference/golden/.
PCC > 0.99 is required for all blocks.
"""

import os
import sys
from pathlib import Path

import torch
from PIL import Image

# --------------------------------------------------------------------------
# Paths and imports
# --------------------------------------------------------------------------

HF_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--allenai--Molmo2-8B/snapshots/" "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
)

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_DIR.mkdir(exist_ok=True)

sys.path.insert(0, HF_MODEL_PATH)

from functional import (
    build_prefill_mask,
    build_rope_cache,
    dual_embedding,
    image_pooling_2d,
    image_projector,
    load_decoder_block_weights,
    load_vit_resblock_weights,
    rmsnorm,
    text_attention,
    text_decoder_block,
    text_mlp,
    vit_encode,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def assert_pcc(ref: torch.Tensor, hf: torch.Tensor, tag: str, threshold: float = 0.99) -> float:
    val = pcc(ref, hf)
    status = "PASS" if val >= threshold else "FAIL"
    print(f"  [{status}] {tag}: PCC={val:.6f}")
    assert val >= threshold, f"PCC {val:.6f} < {threshold} for {tag}"
    return val


# --------------------------------------------------------------------------
# Load model and processor
# --------------------------------------------------------------------------


def load_model_and_processor():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(HF_MODEL_PATH, trust_remote_code=True)
    print("Loading model (float32, CPU)...")
    model = AutoModelForImageTextToText.from_pretrained(
        HF_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float32, device_map="cpu"
    )
    model.eval()
    return model, processor


# --------------------------------------------------------------------------
# Step 0: End-to-end generation (verify model works)
# --------------------------------------------------------------------------


def test_end_to_end(model, processor):
    print("\n=== Step 0: End-to-end generation ===")
    torch.manual_seed(42)

    # Use the bundled logo image for a deterministic test
    logo_path = Path(HF_MODEL_PATH) / "molmo_2_logo_RGB.png"
    if logo_path.exists():
        image = Image.open(logo_path).convert("RGB")
    else:
        # Create a simple test image
        image = Image.fromarray((torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8)).numpy())

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image briefly."}]}
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    response = processor.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"  Generated text: {response!r}")
    assert len(response) > 0, "Model produced empty output — end-to-end check FAILED"
    print("  [PASS] End-to-end generation produced non-empty text")

    # Save the golden
    torch.save(
        {
            "input_ids": inputs["input_ids"],
            "output_ids": output_ids,
            "response": response,
        },
        GOLDEN_DIR / "e2e_generation.pt",
    )
    return inputs, response


# --------------------------------------------------------------------------
# Step 1: Text decoder blocks (text-only)
# --------------------------------------------------------------------------


def test_text_decoder_blocks(model, processor):
    print("\n=== Step 1: Text decoder blocks ===")
    torch.manual_seed(0)
    state_dict = model.state_dict()
    text_cfg = model.config.text_config

    # Text-only inputs (no images)
    prompt = "The capital of France is"
    inputs = processor(text=prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    B, S = input_ids.shape

    with torch.no_grad():
        # --- RMSNorm ---
        x = torch.randn(1, 16, text_cfg.hidden_size, dtype=torch.float32)
        w = state_dict["model.transformer.blocks.0.attn_norm.weight"]
        ref_out = rmsnorm(x, w)
        hf_norm = model.model.transformer.blocks[0].attn_norm
        hf_out = hf_norm(x)
        assert_pcc(ref_out, hf_out, "RMSNorm block 0")

        # --- Dual embedding ---
        emb_w = state_dict["model.transformer.wte.embedding"]
        new_emb_w = state_dict["model.transformer.wte.new_embedding"]
        ref_emb = dual_embedding(input_ids, emb_w, new_emb_w)
        hf_emb = model.model.transformer.wte(input_ids)
        assert_pcc(ref_emb, hf_emb, "Dual embedding")

        # --- Text attention block 0 ---
        # text_attention takes pre-normed input (norm is applied outside by text_decoder_block)
        hidden = ref_emb.to(torch.float32)
        w0 = load_decoder_block_weights(state_dict, 0)
        cos, sin = build_rope_cache(S, text_cfg.head_dim, text_cfg.rope_theta, input_ids.device)
        # Causal mask
        mask = torch.full((1, 1, S, S), float("-inf"))
        mask = torch.triu(mask, diagonal=1)

        normed = rmsnorm(hidden, w0["attn_norm_weight"])
        pos_emb = model.model.transformer.rotary_emb(hidden, torch.arange(S).unsqueeze(0))

        ref_attn = text_attention(
            normed,
            w0["att_proj_weight"],
            w0["attn_out_weight"],
            w0["q_norm_weight"],
            w0["k_norm_weight"],
            cos,
            sin,
            mask,
            num_heads=text_cfg.num_attention_heads,
            num_kv_heads=text_cfg.num_key_value_heads,
            head_dim=text_cfg.head_dim,
        )
        # HF: self_attn also expects pre-normed input
        hf_attn_out, _ = model.model.transformer.blocks[0].self_attn(
            normed, position_embeddings=pos_emb, attention_mask=mask
        )
        assert_pcc(ref_attn, hf_attn_out, "Text attention block 0 (pre-normed)")

        # --- Text MLP block 0 ---
        normed_ff = rmsnorm(hidden, w0["ff_norm_weight"])
        ref_mlp_out = text_mlp(normed_ff, w0["ff_proj_weight"], w0["ff_out_weight"])
        hf_mlp_out = model.model.transformer.blocks[0].mlp(normed_ff)
        assert_pcc(ref_mlp_out, hf_mlp_out, "Text MLP block 0")

        # --- Full text decoder block 0 ---
        ref_block_out = text_decoder_block(
            hidden,
            w0["attn_norm_weight"],
            w0["ff_norm_weight"],
            w0["att_proj_weight"],
            w0["attn_out_weight"],
            w0["q_norm_weight"],
            w0["k_norm_weight"],
            w0["ff_proj_weight"],
            w0["ff_out_weight"],
            cos,
            sin,
            mask,
            num_heads=text_cfg.num_attention_heads,
            num_kv_heads=text_cfg.num_key_value_heads,
            head_dim=text_cfg.head_dim,
        )
        hf_block_out = model.model.transformer.blocks[0](hidden, position_embeddings=pos_emb, attention_mask=mask)[0]
        assert_pcc(ref_block_out, hf_block_out, "Full decoder block 0")

        # Save golden
        torch.save(
            {
                "input": hidden,
                "block0_output": ref_block_out,
                "config": {
                    "hidden_size": text_cfg.hidden_size,
                    "num_heads": text_cfg.num_attention_heads,
                    "num_kv_heads": text_cfg.num_key_value_heads,
                    "head_dim": text_cfg.head_dim,
                },
            },
            GOLDEN_DIR / "decoder_block0.pt",
        )
        print("  Saved golden: decoder_block0.pt")


# --------------------------------------------------------------------------
# Step 2: Prefill mask
# --------------------------------------------------------------------------


def test_prefill_mask(model, processor):
    print("\n=== Step 2: Prefill attention mask ===")
    torch.manual_seed(0)

    # Simulate token_type_ids: 0=text, 1=image
    B, S = 1, 20
    token_type_ids = torch.zeros(B, S, dtype=torch.long)
    token_type_ids[0, 5:15] = 1  # image tokens at positions 5..14

    ref_mask = build_prefill_mask(S, token_type_ids)  # [B, 1, S, S]

    # Verify causal property for text tokens (pos 3 is text → can only see pos ≤ 3)
    assert ref_mask[0, 0, 3, 4].item() == float("-inf"), "Text token should not see future"
    assert ref_mask[0, 0, 3, 3].item() == 0.0, "Text token should see itself"

    # Verify bidirectional for image tokens (pos 5 should see pos 14 and vice versa)
    assert ref_mask[0, 0, 5, 14].item() == 0.0, "Image token q=5 should see image token kv=14"
    assert ref_mask[0, 0, 14, 5].item() == 0.0, "Image token q=14 should see image token kv=5"

    # Verify text token cannot see future image token
    assert ref_mask[0, 0, 3, 10].item() == float("-inf"), "Text token q=3 should not see image kv=10 (future)"

    # Verify image token cannot see future text token
    assert ref_mask[0, 0, 8, 17].item() == float("-inf"), "Image token q=8 should not see text kv=17 (future)"

    torch.save({"token_type_ids": token_type_ids, "mask": ref_mask}, GOLDEN_DIR / "prefill_mask.pt")
    print("  [PASS] Prefill mask: all invariants hold")
    print("  Saved golden: prefill_mask.pt")


# --------------------------------------------------------------------------
# Step 3: ViT encoder
# --------------------------------------------------------------------------


def test_vit_encoder(model, processor):
    print("\n=== Step 3: ViT encoder ===")
    torch.manual_seed(0)
    state_dict = model.state_dict()
    vit_cfg = model.config.vit_config

    n_blocks = 25
    resblock_weights = [load_vit_resblock_weights(state_dict, i) for i in range(n_blocks)]

    # Single crop test input
    B_crops = 2
    N_patches = 729
    patch_dim = 588  # 14×14×3
    pixel_values = torch.randn(B_crops, N_patches, patch_dim, dtype=torch.float32)

    patch_embed_weight = state_dict["model.vision_backbone.image_vit.patch_embedding.weight"]
    patch_embed_bias = state_dict["model.vision_backbone.image_vit.patch_embedding.bias"]
    pos_embedding = state_dict["model.vision_backbone.image_vit.positional_embedding"]

    ref_features = vit_encode(
        pixel_values,
        patch_embed_weight,
        patch_embed_bias,
        pos_embedding,
        resblock_weights,
        patch_num=(27, 27),
        num_heads=vit_cfg.num_attention_heads,
        head_dim=vit_cfg.head_dim,
        norm_eps=vit_cfg.layer_norm_eps,
        n_blocks=n_blocks,
        capture_layers=(24, 18),  # HF order: vit_layers=[-3,-9] → [24, 18]
    )
    assert ref_features.shape == (B_crops, N_patches, 2304), f"Unexpected shape: {ref_features.shape}"

    # HF reference: run encode_image on same input
    images_4d = pixel_values.unsqueeze(0)  # [1, B_crops, 729, 588]
    with torch.no_grad():
        hf_features = model.model.vision_backbone.encode_image(images_4d).squeeze(0)  # [B_crops, 729, 2304]

    assert_pcc(ref_features, hf_features, "ViT encode_image (2 crops, layers 18+24 concat)")

    torch.save(
        {
            "pixel_values": pixel_values,
            "ref_features": ref_features,
            "hf_features": hf_features,
        },
        GOLDEN_DIR / "vit_encode.pt",
    )
    print("  Saved golden: vit_encode.pt")


# --------------------------------------------------------------------------
# Step 4: Image pooling 2D + projector
# --------------------------------------------------------------------------


def test_vision_adapter(model, processor):
    print("\n=== Step 4: Vision adapter (pooling + projector) ===")
    torch.manual_seed(0)
    state_dict = model.state_dict()

    logo_path = Path(HF_MODEL_PATH) / "molmo_2_logo_RGB.png"
    if not logo_path.exists():
        print("  SKIP: logo image not found")
        return

    image = Image.open(logo_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")

    pixel_values = inputs["pixel_values"].float()
    pooled_patches_idx = inputs["image_token_pooling"]
    image_grids = inputs["image_grids"]
    image_num_crops = inputs["image_num_crops"]

    n_crops, n_patches, px_dim = pixel_values.shape
    B = 1

    with torch.no_grad():
        # First get the batched tensors from HF's merge_visual_inputs
        images_4d_hf, pool_idx_4d_hf = model.model.merge_visual_inputs(
            input_ids=inputs["input_ids"],
            pixel_values=pixel_values,
            image_token_pooling=pooled_patches_idx,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
        )
        # images_4d_hf: [B, max_crops, 729, 588]; pool_idx_4d_hf: [B, N_pooled, pool_window]

        # Run reference ViT encode on the batched pixel_values [B*max_crops, 729, 588]
        n_vit_blocks = 25
        resblock_weights = [load_vit_resblock_weights(state_dict, i) for i in range(n_vit_blocks)]
        vit_cfg = model.config.vit_config
        B_hf, max_crops_hf = images_4d_hf.shape[:2]
        ref_vit_out = vit_encode(
            images_4d_hf.view(B_hf * max_crops_hf, 729, 588),
            state_dict["model.vision_backbone.image_vit.patch_embedding.weight"],
            state_dict["model.vision_backbone.image_vit.patch_embedding.bias"],
            state_dict["model.vision_backbone.image_vit.positional_embedding"],
            resblock_weights,
            patch_num=(27, 27),
            num_heads=vit_cfg.num_attention_heads,
            head_dim=vit_cfg.head_dim,
            norm_eps=vit_cfg.layer_norm_eps,
            n_blocks=n_vit_blocks,
        )  # [B*max_crops, 729, 2304]

        image_features_4d = ref_vit_out.view(B_hf, max_crops_hf, 729, 2304)

        # Run reference pooling
        ref_pooled = image_pooling_2d(
            image_features_4d,
            pool_idx_4d_hf,
            state_dict["model.vision_backbone.image_pooling_2d.wq.weight"],
            state_dict["model.vision_backbone.image_pooling_2d.wq.bias"],
            state_dict["model.vision_backbone.image_pooling_2d.wk.weight"],
            state_dict["model.vision_backbone.image_pooling_2d.wk.bias"],
            state_dict["model.vision_backbone.image_pooling_2d.wv.weight"],
            state_dict["model.vision_backbone.image_pooling_2d.wv.bias"],
            state_dict["model.vision_backbone.image_pooling_2d.wo.weight"],
            state_dict["model.vision_backbone.image_pooling_2d.wo.bias"],
            num_heads=model.config.adapter_config.num_attention_heads,
            head_dim=model.config.adapter_config.head_dim,
        )  # [B, N_pooled, 1152]

        valid_token = (pool_idx_4d_hf[0] >= 0).any(dim=-1)  # [N_pooled]
        ref_pooled_valid = ref_pooled[0][valid_token]  # [N_valid, 1152]
        ref_proj_out = image_projector(
            ref_pooled_valid,
            state_dict["model.vision_backbone.image_projector.w1.weight"],
            state_dict["model.vision_backbone.image_projector.w2.weight"],
            state_dict["model.vision_backbone.image_projector.w3.weight"],
        )  # [N_valid, 4096]

        # HF reference: run full vision_backbone
        hf_img_features = model.model.vision_backbone(images_4d_hf, pool_idx_4d_hf)  # [N_valid, 4096]

        assert_pcc(ref_proj_out, hf_img_features, "Vision backbone (pooling + projector)")

        torch.save(
            {
                "image_features_4d": image_features_4d,
                "ref_proj_out": ref_proj_out,
                "hf_img_features": hf_img_features,
            },
            GOLDEN_DIR / "vision_adapter.pt",
        )
        print("  Saved golden: vision_adapter.pt")


# --------------------------------------------------------------------------
# Step 5: Full prefill forward pass (image input)
# --------------------------------------------------------------------------


def test_full_prefill(model, processor):
    print("\n=== Step 5: Full prefill forward pass ===")
    torch.manual_seed(42)
    state_dict = model.state_dict()

    logo_path = Path(HF_MODEL_PATH) / "molmo_2_logo_RGB.png"
    if not logo_path.exists():
        print("  SKIP: logo image not found")
        return

    image = Image.open(logo_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")

    with torch.no_grad():
        # Run HF model full forward to get reference hidden states and logits
        outputs = model.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
        hf_last_hidden = outputs.last_hidden_state  # [B, S, 4096]
        hf_image_features = outputs.image_hidden_states  # [N_valid, 4096]

        # Run CG logits via lm_head
        hf_logits = model.lm_head(model.model.transformer.ln_f(hf_last_hidden))

    print(f"  HF last hidden shape: {hf_last_hidden.shape}")
    print(f"  HF image features shape: {hf_image_features.shape}")
    print(f"  HF logits shape: {hf_logits.shape}")
    print(
        f"  Top-1 next token: {hf_logits[0, -1].argmax().item()} "
        f"({processor.decode([hf_logits[0, -1].argmax().item()])!r})"
    )

    torch.save(
        {
            "inputs": {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)},
            "last_hidden_state": hf_last_hidden,
            "image_features": hf_image_features,
            "logits": hf_logits,
        },
        GOLDEN_DIR / "full_prefill.pt",
    )
    print("  Saved golden: full_prefill.pt")
    print("  [PASS] Full prefill produces valid logits")


# --------------------------------------------------------------------------
# Step 6: Per-block hidden state comparison (block 0 with image)
# --------------------------------------------------------------------------


def test_block0_with_image(model, processor):
    print("\n=== Step 6: Decoder block 0 with image embeddings ===")
    torch.manual_seed(42)
    state_dict = model.state_dict()
    text_cfg = model.config.text_config

    logo_path = Path(HF_MODEL_PATH) / "molmo_2_logo_RGB.png"
    if not logo_path.exists():
        print("  SKIP: logo image not found")
        return

    image = Image.open(logo_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")

    with torch.no_grad():
        input_ids = inputs["input_ids"]
        token_type_ids = inputs.get("token_type_ids")
        B, S = input_ids.shape

        # Build input embeddings (text + image features injected)
        # merge_visual_inputs handles batching pixel_values into [B, max_crops, 729, 588]
        images_4d, pool_idx_4d = model.model.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=inputs["pixel_values"].float(),
            image_token_pooling=inputs["image_token_pooling"],
            image_grids=inputs["image_grids"],
            image_num_crops=inputs["image_num_crops"],
        )
        img_features, _ = model.model.build_input_embeddings(input_ids, images_4d, pool_idx_4d)
        # img_features is already the full embed with image injected; shape [B, S, 4096]

        # Build combined prefill mask
        if token_type_ids is not None:
            ref_mask = build_prefill_mask(S, token_type_ids.long())  # [B, 1, S, S]
        else:
            ref_mask = torch.triu(torch.full((1, 1, S, S), float("-inf")), diagonal=1)

        # Build RoPE
        cos, sin = build_rope_cache(S, text_cfg.head_dim, text_cfg.rope_theta, input_ids.device)

        # Reference: run block 0
        w0 = load_decoder_block_weights(state_dict, 0)
        ref_out = text_decoder_block(
            img_features.to(torch.float32),
            w0["attn_norm_weight"],
            w0["ff_norm_weight"],
            w0["att_proj_weight"],
            w0["attn_out_weight"],
            w0["q_norm_weight"],
            w0["k_norm_weight"],
            w0["ff_proj_weight"],
            w0["ff_out_weight"],
            cos,
            sin,
            ref_mask,
            num_heads=text_cfg.num_attention_heads,
            num_kv_heads=text_cfg.num_key_value_heads,
            head_dim=text_cfg.head_dim,
        )

        # HF: run full forward to get hidden state after block 0
        outputs = model.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )
        hf_hidden_states = outputs.hidden_states  # tuple of 37 tensors
        # hidden_states[0] = embeddings, hidden_states[1] = after block 0
        hf_block0_out = hf_hidden_states[1].to(torch.float32)

        assert_pcc(ref_out, hf_block0_out, "Decoder block 0 with image embeddings")

        torch.save(
            {
                "input_embeddings": img_features,
                "ref_block0_out": ref_out,
                "hf_block0_out": hf_block0_out,
            },
            GOLDEN_DIR / "block0_with_image.pt",
        )
        print("  Saved golden: block0_with_image.pt")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading Molmo2-8B model...")
    model, processor = load_model_and_processor()

    test_end_to_end(model, processor)
    test_text_decoder_blocks(model, processor)
    test_prefill_mask(model, processor)
    test_vit_encoder(model, processor)
    test_vision_adapter(model, processor)
    test_full_prefill(model, processor)
    test_block0_with_image(model, processor)

    print("\n=== All reference tests passed. Goldens saved to reference/golden/ ===")
