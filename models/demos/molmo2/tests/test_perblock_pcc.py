#!/usr/bin/env python3
"""Per-block PCC comparison between HuggingFace and TTNN for Molmo2.

Compares outputs at each transformer block during multimodal prefill to find
where divergence occurs.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from transformers import AutoProcessor

# Add repo to path
repo_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

os.environ["TT_METAL_HOME"] = str(repo_root)

import ttnn
from models.demos.molmo2.tt.model_loader import create_model, load_model_weights


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    a_mean = a_flat.mean()
    b_mean = b_flat.mean()

    a_centered = a_flat - a_mean
    b_centered = b_flat - b_mean

    cov = (a_centered * b_centered).sum()
    std_a = (a_centered**2).sum().sqrt()
    std_b = (b_centered**2).sum().sqrt()

    if std_a < 1e-8 or std_b < 1e-8:
        return 1.0 if torch.allclose(a_flat, b_flat) else 0.0

    pcc = cov / (std_a * std_b)
    return pcc.item()


def load_hf_model():
    """Load HuggingFace Molmo2 model."""
    logger.info("Loading HuggingFace model...")

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)

    # Import the custom model class
    model_class = get_class_from_dynamic_module(
        "modeling_molmo2.Molmo2ForConditionalGeneration",
        "allenai/Molmo2-8B",
        trust_remote_code=True,
    )
    hf_model = model_class.from_pretrained(
        "allenai/Molmo2-8B",
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.eval()

    processor = AutoProcessor.from_pretrained("allenai/Molmo2-8B", trust_remote_code=True)

    logger.info("HuggingFace model loaded")
    return hf_model, processor


def hook_hf_blocks(hf_model):
    """Add hooks to capture intermediate outputs from HF model."""
    hf_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            # Handle tuple outputs (transformers often return (hidden_states, ...))
            if isinstance(output, tuple):
                hf_outputs[name] = output[0].detach().clone()
            else:
                hf_outputs[name] = output.detach().clone()

        return hook

    # Hook embedding output
    hf_model.model.transformer.wte.register_forward_hook(make_hook("embedding"))

    # Hook each transformer block
    for i, block in enumerate(hf_model.model.transformer.blocks):
        block.register_forward_hook(make_hook(f"block_{i}"))

    # Hook final layer norm
    hf_model.model.transformer.ln_f.register_forward_hook(make_hook("ln_f"))

    return hf_outputs


def run_hf_multimodal(hf_model, processor, image_path, prompt):
    """Run HF model on image and return intermediate outputs."""
    logger.info(f"Running HF model with image: {image_path}")

    # Add hooks to capture outputs
    hf_outputs = hook_hf_blocks(hf_model)

    # Load and process image
    image = Image.open(image_path)
    inputs = processor(
        images=image,
        text=f"<|image|> {prompt}",
        return_tensors="pt",
    )

    logger.info(f"Input IDs shape: {inputs['input_ids'].shape}")
    logger.info(f"Pixel values shape: {inputs['pixel_values'].shape}")

    # Run forward pass
    with torch.no_grad():
        output = hf_model(**inputs)

    logger.info(f"HF captured {len(hf_outputs)} outputs")

    return hf_outputs, inputs


def run_ttnn_multimodal(ttnn_model, mesh_device, processor, image_path, prompt):
    """Run TTNN model on image and return intermediate outputs."""
    logger.info(f"Running TTNN model with image: {image_path}")

    # Load and process image
    image = Image.open(image_path)
    inputs = processor(
        images=image,
        text=f"<|image|> {prompt}",
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    image_token_pooling = inputs["image_token_pooling"]  # [n_tokens, k_pool]
    # embed_image expects [n_images, n_out, k_pool]; for a single image n_images=1
    if image_token_pooling.dim() == 2:
        n_tokens, k_pool = image_token_pooling.shape
        # n_out = n_tokens for single image (each patch is one query)
        image_token_pooling = image_token_pooling.unsqueeze(0)  # [1, n_tokens, k_pool]

    logger.info(f"Input IDs shape: {input_ids.shape}")
    logger.info(f"Pixel values shape: {pixel_values.shape}")

    # Dictionary to store outputs at each stage
    ttnn_outputs = {}

    # 1. Get text embeddings
    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    mesh_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None

    input_ids_ttnn = ttnn.from_torch(
        input_ids,
        dtype=ttnn.uint32,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
    )

    text_embeddings = ttnn_model.text_model.embed_tokens(input_ids_ttnn)
    text_emb_full = ttnn.to_torch(text_embeddings, mesh_composer=mesh_composer)
    # mesh_composer concats on dim 0, so [8, 1, seq, hidden] - take first device
    text_emb_torch = text_emb_full[0:1]  # Keep 4D: [1, 1, seq, hidden]

    logger.info(f"Text embeddings shape: {text_emb_torch.shape}")
    logger.info(f"Text embeddings: mean={text_emb_torch.mean():.6f}, std={text_emb_torch.std():.6f}")

    # 2. Get visual embeddings
    visual_embeddings, _ = ttnn_model.embed_image(
        pixel_values,
        image_token_pooling,
    )
    visual_emb_full = ttnn.to_torch(visual_embeddings, mesh_composer=mesh_composer)
    visual_emb_torch = visual_emb_full[0:1]  # Keep 4D: [1, 1, num_patches, hidden]

    logger.info(f"Visual embeddings shape: {visual_emb_torch.shape}")
    logger.info(f"Visual embeddings: mean={visual_emb_torch.mean():.6f}, std={visual_emb_torch.std():.6f}")

    # 3. Combine embeddings (same as prepare_inputs_for_multimodal)
    image_patch_id = 151938
    image_positions = (input_ids[0] == image_patch_id).nonzero(as_tuple=True)[0]

    logger.info(f"Image positions: {image_positions.shape[0]} positions")

    combined_emb = text_emb_torch.clone()  # [1, 1, seq, hidden]
    # Visual embeddings are [1, 1, num_patches, hidden], add at image positions
    combined_emb[0, 0, image_positions, :] += visual_emb_torch[0, 0, :, :]

    ttnn_outputs["embedding"] = combined_emb  # [1, 1, seq, hidden]
    logger.info(f"Combined embedding: mean={combined_emb.mean():.6f}, std={combined_emb.std():.6f}")

    # 4. Run through transformer blocks one by one with debug output
    combined_ttnn = ttnn.from_torch(
        combined_emb,  # Already [1, 1, seq, hidden]
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
    )

    seq_len = combined_emb.shape[2]
    rot_mats = ttnn_model.text_model.rotary_setup.get_rot_mats_prefill(seq_len, 0)
    transformation_mats = ttnn_model.text_model.transformation_mats

    x = combined_ttnn
    for layer_idx, block in enumerate(ttnn_model.text_model.blocks):
        x, _ = block(
            x,
            rot_mats,
            transformation_mats,
            attn_mask=None,  # Uses causal masking internally
            start_pos=0,
            kv_cache=None,
            page_table=None,
            user_id=0,
        )

        x_full = ttnn.to_torch(x, mesh_composer=mesh_composer)
        x_torch = x_full[0:1]  # Keep 4D: [1, 1, seq, hidden]
        ttnn_outputs[f"block_{layer_idx}"] = x_torch

        if layer_idx < 5 or layer_idx >= 31:  # First 5 and last 5 blocks
            logger.info(f"Block {layer_idx}: mean={x_torch.mean():.6f}, std={x_torch.std():.6f}")

    # 5. Final layer norm
    x = ttnn_model.text_model.ln_f(x)
    x_full = ttnn.to_torch(x, mesh_composer=mesh_composer)
    x_torch = x_full[0:1]  # Keep 4D: [1, 1, seq, hidden]
    ttnn_outputs["ln_f"] = x_torch
    logger.info(f"ln_f: mean={x_torch.mean():.6f}, std={x_torch.std():.6f}")

    return ttnn_outputs, inputs


def compare_outputs(hf_outputs, ttnn_outputs):
    """Compare outputs between HF and TTNN at each stage."""
    logger.info("\n" + "=" * 60)
    logger.info("PCC Comparison: HuggingFace vs TTNN")
    logger.info("=" * 60)

    results = []

    for name in ["embedding"] + [f"block_{i}" for i in range(36)] + ["ln_f"]:
        if name not in hf_outputs or name not in ttnn_outputs:
            logger.warning(f"Missing output for {name}")
            continue

        hf_out = hf_outputs[name]
        ttnn_out = ttnn_outputs[name]

        # Handle shape differences
        if hf_out.dim() == 3:  # [batch, seq, hidden]
            hf_out = hf_out.unsqueeze(1)  # [batch, 1, seq, hidden]

        if hf_out.shape != ttnn_out.shape:
            logger.warning(f"{name}: Shape mismatch - HF {hf_out.shape} vs TTNN {ttnn_out.shape}")
            # Try to align
            min_seq = min(hf_out.shape[2], ttnn_out.shape[2])
            hf_out = hf_out[:, :, :min_seq, :]
            ttnn_out = ttnn_out[:, :, :min_seq, :]

        pcc = compute_pcc(hf_out, ttnn_out)

        # Get stats
        hf_mean = hf_out.mean().item()
        hf_std = hf_out.std().item()
        ttnn_mean = ttnn_out.mean().item()
        ttnn_std = ttnn_out.std().item()

        status = "PASS" if pcc > 0.99 else "FAIL"

        result = {
            "name": name,
            "pcc": pcc,
            "hf_mean": hf_mean,
            "hf_std": hf_std,
            "ttnn_mean": ttnn_mean,
            "ttnn_std": ttnn_std,
            "status": status,
        }
        results.append(result)

        if pcc < 0.99 or name in ["embedding", "block_0", "block_35", "ln_f"]:
            logger.info(
                f"{name:12s}: PCC={pcc:.6f} [{status}] | "
                f"HF(mean={hf_mean:.4f}, std={hf_std:.4f}) vs "
                f"TTNN(mean={ttnn_mean:.4f}, std={ttnn_std:.4f})"
            )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)

    passing = sum(1 for r in results if r["status"] == "PASS")
    failing = sum(1 for r in results if r["status"] == "FAIL")

    logger.info(f"Passing: {passing}/{len(results)}")
    logger.info(f"Failing: {failing}/{len(results)}")

    if failing > 0:
        logger.info("\nFailing blocks:")
        for r in results:
            if r["status"] == "FAIL":
                logger.info(f"  {r['name']}: PCC={r['pcc']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="models/demos/molmo2/demo/dog.jpg")
    parser.add_argument("--prompt", default="What is in this image?")
    args = parser.parse_args()

    # Load HF model
    hf_model, processor = load_hf_model()

    # Run HF model
    hf_outputs, inputs = run_hf_multimodal(hf_model, processor, args.image, args.prompt)

    # Load TTNN model
    logger.info("Loading TTNN model...")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))
    state_dict = load_model_weights()
    ttnn_model = create_model(
        mesh_device=mesh_device,
        state_dict=state_dict,
    )
    logger.info("TTNN model loaded")

    # Run TTNN model
    ttnn_outputs, _ = run_ttnn_multimodal(ttnn_model, mesh_device, processor, args.image, args.prompt)

    # Compare outputs
    results = compare_outputs(hf_outputs, ttnn_outputs)

    # Cleanup
    ttnn.close_mesh_device(mesh_device)

    return 0 if all(r["status"] == "PASS" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
