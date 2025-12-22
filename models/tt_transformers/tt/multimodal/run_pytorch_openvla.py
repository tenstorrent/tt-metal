#!/usr/bin/env python3
"""
Standalone script to run PyTorch OpenVLA and save outputs.

Run this in a separate environment with timm==0.9.16:

    # Create venv
    python3 -m venv /tmp/openvla_pt_env
    source /tmp/openvla_pt_env/bin/activate
    pip install torch transformers timm==0.9.16 safetensors pillow numpy

    # Run this script
    python run_pytorch_openvla.py --output /tmp/pytorch_openvla_outputs.pt

Then the main test can load these outputs for comparison.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run PyTorch OpenVLA and save outputs")
    parser.add_argument(
        "--output", type=str, default="/tmp/pytorch_openvla_outputs.pt", help="Output file path for saved tensors"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to input image (default: LeRobot sample or synthetic)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In: What action should the robot take to pick up the red block?\nOut:",
        help="Input prompt",
    )
    args = parser.parse_args()

    # Check timm version
    import timm

    print(f"timm version: {timm.__version__}")

    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # Ensure numpy is available globally in this scope
    globals()["np"] = np

    print("=" * 60)
    print("PyTorch OpenVLA Inference")
    print("=" * 60)

    # Load image
    LEROBOT_IMAGES_DIR = os.path.expanduser("~/teja/smolvla/demo/images")
    if args.image:
        image_path = args.image
    else:
        image_path = os.path.join(LEROBOT_IMAGES_DIR, "lerobot_sample_1.png")

    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
        print(f"✅ Using image: {image_path}")
    else:
        image = Image.new("RGB", (224, 224), color=(128, 64, 32))
        print(f"⚠️  Image not found, using synthetic image")

    prompt = args.prompt
    print(f"📝 Prompt: {prompt}")

    # Load processor
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # Load model
    print("Loading PyTorch OpenVLA model (this may take a while)...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print(f"✅ Model loaded: {type(model).__name__}")

    # Get inputs
    inputs = processor(prompt, image)
    print(f"\nInput shapes:")
    print(f"  pixel_values: {inputs['pixel_values'].shape}")
    print(f"  input_ids: {inputs['input_ids'].shape}")

    # Run inference with detailed debug outputs
    print("\nRunning PyTorch inference with debug outputs...")
    with torch.no_grad():
        # Get pixel values
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        input_ids = inputs["input_ids"]

        # ============================================
        # VISION BACKBONE
        # ============================================
        # DEBUG: Check LLM embedding for token 31872
        try:
            embd = model.language_model.model.embed_tokens
            token_embd = embd.weight[31872].float()
            print(
                f"DEBUG PT LLM embedding[31872]: mean={token_embd.mean():.6f}, std={token_embd.std():.6f}, first5={token_embd[:5].tolist()}"
            )
        except Exception as e:
            print(f"DEBUG PT LLM embedding check failed: {e}")

        print("\n--- Vision Backbone ---")

        # Get individual encoder outputs for comparison
        img, img_fused = torch.split(pixel_values.to(torch.bfloat16), [3, 3], dim=1)
        dinov2_output = model.vision_backbone.featurizer(img)
        siglip_output = model.vision_backbone.fused_featurizer(img_fused)
        print(
            f"DEBUG PT DINOv2: shape={dinov2_output.shape}, mean={dinov2_output.float().mean():.6f}, std={dinov2_output.float().std():.6f}"
        )
        print(
            f"DEBUG PT SigLIP: shape={siglip_output.shape}, mean={siglip_output.float().mean():.6f}, std={siglip_output.float().std():.6f}"
        )

        vision_output = model.vision_backbone(pixel_values)
        print(
            f"DEBUG PT vision_features: shape={vision_output.shape}, mean={vision_output.float().mean():.6f}, std={vision_output.float().std():.6f}"
        )

        # ============================================
        # PROJECTOR
        # ============================================
        print("\n--- Projector ---")
        projector_output = model.projector(vision_output)
        print(
            f"DEBUG PT projector_output: shape={projector_output.shape}, mean={projector_output.float().mean():.6f}, std={projector_output.float().std():.6f}"
        )

        # ============================================
        # MULTIMODAL EMBEDDINGS (Vision + Text)
        # ============================================
        print("\n--- Multimodal Embeddings ---")
        # Get text embeddings
        text_embeddings = model.language_model.get_input_embeddings()(input_ids)
        print(
            f"DEBUG PT text_embeddings: shape={text_embeddings.shape}, mean={text_embeddings.float().mean():.6f}, std={text_embeddings.float().std():.6f}"
        )

        # Combine (simplified - actual method may differ)
        # The actual combination happens in the forward method

        # ============================================
        # GENERATE TOKENS (with logits capture)
        # ============================================
        print("\n--- Token Generation ---")

        # Use generate with output_scores to get logits
        gen_outputs = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=8,  # 7 action tokens + 1
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        generated_ids = gen_outputs.sequences
        scores = gen_outputs.scores  # List of logits for each step

        print(f"Generated {len(scores)} tokens")

        all_tokens = []
        for i, score in enumerate(scores):
            # Get top tokens
            top_vals, top_ids = torch.topk(score[0], 5)
            top1_id = top_ids[0].item()
            top1_val = top_vals[0].item()
            top2_val = top_vals[1].item()
            gap = top1_val - top2_val

            all_tokens.append(top1_id)
            step_name = "prefill" if i == 0 else f"decode[{i-1}]"
            print(
                f"DEBUG PT {step_name}: top1={top1_id} ({top1_val:.4f}), top2={top_ids[1].item()} ({top2_val:.4f}), GAP={gap:.4f}"
            )

        print(f"DEBUG PT all_tokens={all_tokens}")

        # ============================================
        # COMPUTE ACTION FROM GENERATED TOKENS
        # ============================================
        print("\n--- Action Computation ---")
        # Use the tokens we already generated (skip first token which is from prefill)
        action_tokens = torch.tensor(all_tokens[1:8], dtype=torch.long)  # 7 action tokens
        print(f"Action tokens: {action_tokens.tolist()}")

        # Get action from tokens using model's method
        unnorm_key = "bridge_orig"
        action_token_ids = action_tokens.cpu().numpy()

        # Map tokens back to action values
        # Token IDs for actions are in range [31744, 31999] mapping to 256 bins
        n_bins = 256
        bin_centers = np.linspace(-1, 1, n_bins)

        # Convert token IDs to bin indices
        bin_indices = action_token_ids - 31744
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Get normalized actions from bin centers
        normalized_actions = bin_centers[bin_indices]
        print(f"Bin indices: {bin_indices}")
        print(f"Normalized actions: {normalized_actions}")

        # Try to unnormalize using model's norm_stats
        if hasattr(model, "norm_stats") and unnorm_key in model.norm_stats:
            norm_stats = model.norm_stats[unnorm_key]
            print(f"Norm stats keys: {norm_stats.keys()}")

            # Check for different possible key formats
            if "q01" in norm_stats:
                q01 = np.array(norm_stats["q01"])
                q99 = np.array(norm_stats["q99"])
            elif "action" in norm_stats:
                q01 = np.array(norm_stats["action"]["q01"])
                q99 = np.array(norm_stats["action"]["q99"])
            elif "mean" in norm_stats:
                # Different format: use mean/std
                mean = np.array(norm_stats["mean"])
                std = np.array(norm_stats["std"])
                action = normalized_actions * std + mean
                print(f"DEBUG PT action (mean/std unnorm): {action}")
            else:
                # Just use normalized values
                action = normalized_actions
                print(f"DEBUG PT action (normalized, no unnorm stats): {action}")

            if "q01" in norm_stats or "action" in norm_stats:
                action = (normalized_actions + 1) / 2 * (q99 - q01) + q01
                print(f"DEBUG PT action: {action}")
        else:
            action = normalized_actions
            print(f"DEBUG PT action (normalized): {action}")

    print(f"\n✅ Inference complete!")
    print(f"Action: {action}")
    print(f"Vision output shape: {vision_output.shape}")
    print(f"Projector output shape: {projector_output.shape}")

    # Save outputs with all debug info
    outputs = {
        # Main outputs
        "action": torch.tensor(action, dtype=torch.float32),
        "vision_output": vision_output.to(torch.float32).cpu(),
        "projector_output": projector_output.to(torch.float32).cpu(),
        # Inputs
        "pixel_values": pixel_values.to(torch.float32).cpu(),
        "input_ids": inputs["input_ids"].cpu(),
        "prompt": prompt,
        "image_path": image_path if os.path.exists(image_path) else "synthetic",
        # Token generation info
        "generated_tokens": torch.tensor(all_tokens, dtype=torch.long),
        "logits_scores": [s.cpu().to(torch.float32) for s in scores],
        # Stats for quick comparison
        "vision_mean": vision_output.float().mean().item(),
        "vision_std": vision_output.float().std().item(),
        "projector_mean": projector_output.float().mean().item(),
        "projector_std": projector_output.float().std().item(),
    }

    torch.save(outputs, args.output)
    print(f"\n✅ Outputs saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY - Compare these with TTNN outputs")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"\n--- Vision Backbone ---")
    print(f"Shape: {vision_output.shape}")
    print(f"Mean: {vision_output.float().mean():.6f}")
    print(f"Std: {vision_output.float().std():.6f}")
    print(f"\n--- Projector ---")
    print(f"Shape: {projector_output.shape}")
    print(f"Mean: {projector_output.float().mean():.6f}")
    print(f"Std: {projector_output.float().std():.6f}")
    print(f"\n--- Generated Tokens ---")
    print(f"Tokens: {all_tokens}")
    print(f"\n--- Action ---")
    print(f"Action: {action}")

    return 0


def test_pytorch_llm_with_ttnn_vision(ttnn_vision_path: str):
    """
    Test: Feed TTNN vision encoder output to PyTorch LLM.
    This isolates whether the issue is in TTNN's vision or LLM.

    Usage:
        python run_pytorch_openvla.py --ttnn-vision /tmp/ttnn_vision_output.pt
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("=" * 70)
    print("TEST: PyTorch LLM with TTNN Vision Encoder Output")
    print("=" * 70)

    # Load TTNN vision output
    print(f"\nLoading TTNN vision output from: {ttnn_vision_path}")
    ttnn_data = torch.load(ttnn_vision_path)

    # Handle different key names
    if "vision_output" in ttnn_data:
        ttnn_vision = ttnn_data["vision_output"]
    elif "vision_features" in ttnn_data:
        ttnn_vision = ttnn_data["vision_features"]
    else:
        raise KeyError(f"Expected 'vision_output' or 'vision_features' in {ttnn_vision_path}")

    ttnn_projector = ttnn_data.get("projector_output", None)

    print(
        f"TTNN vision: shape={ttnn_vision.shape}, mean={ttnn_vision.float().mean():.4f}, std={ttnn_vision.float().std():.4f}"
    )
    if ttnn_projector is not None:
        print(
            f"TTNN projector: shape={ttnn_projector.shape}, mean={ttnn_projector.float().mean():.4f}, std={ttnn_projector.float().std():.4f}"
        )

    # Load PyTorch model
    print("\nLoading PyTorch OpenVLA model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    # Get prompt tokens
    prompt = "In: What action should the robot take to pick up the red block?\nOut:"
    dummy_image = Image.new("RGB", (224, 224))  # Dummy image, we'll replace vision
    inputs = processor(prompt, dummy_image)
    input_ids = inputs["input_ids"]

    print(f"\nPrompt: {prompt}")
    print(f"input_ids shape: {input_ids.shape}")

    # Get text embeddings from LLM
    text_embeds = model.language_model.get_input_embeddings()(input_ids)
    print(f"Text embeddings: shape={text_embeds.shape}")

    # Use TTNN projector output if available, otherwise run PyTorch projector on TTNN vision
    if ttnn_projector is not None:
        projected = ttnn_projector.to(torch.float32)
        print(f"Using TTNN projector output directly")
    else:
        # Run PyTorch projector on TTNN vision output
        print(f"Running PyTorch projector on TTNN vision output...")
        projected = model.projector(ttnn_vision.to(torch.float32))
    print(f"Projected: shape={projected.shape}, mean={projected.mean():.4f}, std={projected.std():.4f}")

    # Build multimodal embeddings: [BOS] + projected_vision + text[1:]
    bos_embed = text_embeds[:, :1, :]
    text_rest = text_embeds[:, 1:, :]
    multimodal_embeds = torch.cat([bos_embed, projected, text_rest], dim=1)
    print(f"Multimodal embeddings: shape={multimodal_embeds.shape}")

    # Run LLM with TTNN vision features
    print("\n--- Running PyTorch LLM with TTNN vision features ---")
    all_tokens = []
    with torch.no_grad():
        # Use forward pass with KV cache for token-by-token generation
        past_key_values = None
        current_embeds = multimodal_embeds

        for step in range(8):  # Generate 8 tokens
            outputs = model.language_model(
                inputs_embeds=current_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            # Get next token
            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1).item()
            all_tokens.append(next_token)

            # Get top tokens for debug
            top_vals, top_ids = torch.topk(next_logits[0], 5)
            top1_val = top_vals[0].item()
            top2_val = top_vals[1].item()
            gap = top1_val - top2_val

            print(f"Step {step}: token={next_token}, top1={top1_val:.4f}, top2={top2_val:.4f}, GAP={gap:.4f}")

            # Prepare next input (embed the token)
            next_embed = model.language_model.get_input_embeddings()(torch.tensor([[next_token]]))
            current_embeds = next_embed

    print(f"\n✅ All generated tokens: {all_tokens}")

    # Compare with expected PyTorch behavior
    print("\n--- Analysis ---")
    unique_tokens = set(all_tokens)
    if len(unique_tokens) == 1:
        print(f"⚠️  All tokens are the same ({all_tokens[0]}) - matches TTNN behavior")
        print("   This suggests the issue is in vision/projector, not LLM!")
    else:
        print(f"✅ Different tokens generated: {unique_tokens}")
        print("   This suggests the issue is in TTNN LLM, not vision!")

    return all_tokens


if __name__ == "__main__":
    # Check for --ttnn-vision argument
    if "--ttnn-vision" in sys.argv:
        idx = sys.argv.index("--ttnn-vision")
        if idx + 1 < len(sys.argv):
            ttnn_vision_path = sys.argv[idx + 1]
            test_pytorch_llm_with_ttnn_vision(ttnn_vision_path)
            sys.exit(0)
        else:
            print("Error: --ttnn-vision requires a path argument")
            sys.exit(1)

    sys.exit(main())
