#!/usr/bin/env python3
"""
Standalone script to run PyTorch OpenVLA and save outputs.

Run this in a separate environment with compatible versions:

    # Create venv
    python3 -m venv /tmp/openvla_pt_env
    source /tmp/openvla_pt_env/bin/activate
    pip install torch transformers==4.40.0 timm==0.9.16 accelerate safetensors pillow numpy

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

    # Load image - check multiple possible locations
    LEROBOT_IMAGES_DIRS = [
        os.path.expanduser("~/teja/demo_images"),
        os.path.expanduser("~/demo/images"),
        os.path.expanduser("~/teja/smolvla/demo/images"),
        os.path.expanduser("~/lerobot/images"),
    ]
    if args.image:
        image_path = args.image
    else:
        image_path = None
        for img_dir in LEROBOT_IMAGES_DIRS:
            candidate = os.path.join(img_dir, "lerobot_sample_1.png")
            if os.path.exists(candidate):
                image_path = candidate
                break
        if image_path is None:
            image_path = os.path.join(LEROBOT_IMAGES_DIRS[0], "lerobot_sample_1.png")  # Will use synthetic

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
        # FIX: Add empty token 29871 to match TTNN behavior
        # This is what predict_action() does internally
        # ============================================
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)), dim=1
            )
            print(f"✅ Added empty token 29871. New input_ids shape: {input_ids.shape}")
        else:
            print(f"ℹ️  Empty token already present. input_ids shape: {input_ids.shape}")

        # GUARDRAIL: Print prompt length info for verification
        text_len = input_ids.shape[1]
        expected_multimodal_len = 1 + 256 + (text_len - 1)  # BOS + vision + text_without_BOS
        print(f"\n=== PROMPT LENGTH GUARDRAIL ===")
        print(f"TEXT_LEN={text_len}, TEXT_TAIL={input_ids[0, -5:].tolist()}")
        print(f"Expected multimodal_len={expected_multimodal_len}")
        assert input_ids[0, -1].item() == 29871, f"Empty token 29871 must be at end! Got {input_ids[0, -1].item()}"

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


def test_pytorch_llm_with_ttnn_vision(ttnn_vision_path: str, use_tt_multimodal: bool = False):
    """
    Test: Feed TTNN outputs to PyTorch LLM.

    TEST A (default): TT Vision + TT Projector + PT Text → PT LLM
    TEST B (--use-tt-mm): TT Full Multimodal Embeddings → PT LLM

    Usage:
        # Test A: TT vision/projector → PT LLM
        python run_pytorch_openvla.py --ttnn-vision /tmp/tt_outputs_pick.pt

        # Test B: TT full multimodal → PT LLM
        python run_pytorch_openvla.py --ttnn-vision /tmp/tt_outputs_pick.pt --use-tt-mm
    """
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("=" * 70)
    if use_tt_multimodal:
        print("TEST B: PyTorch LLM with TT FULL Multimodal Embeddings")
    else:
        print("TEST A: PyTorch LLM with TT Vision/Projector + PT Text")
    print("=" * 70)

    # Load TT outputs
    print(f"\nLoading TT outputs from: {ttnn_vision_path}")
    tt_data = torch.load(ttnn_vision_path)

    # Print what we have
    print(f"Available keys: {list(tt_data.keys())}")

    # Get prompt from saved data
    prompt = tt_data.get("prompt", "In: What action should the robot take to pick up the block?\nOut:")
    print(f"\nPrompt: {prompt}")

    # Get TT action for comparison
    tt_action = tt_data.get("tt_action", None)
    tt_tokens = tt_data.get("tt_tokens", None)
    if tt_action is not None:
        print(f"TT Action: {tt_action.tolist()}")
    if tt_tokens is not None:
        print(f"TT Tokens: {tt_tokens.tolist()}")

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

    if use_tt_multimodal:
        # ============================================
        # TEST B: Use TT full multimodal embeddings directly
        # ============================================
        print("\n--- Using TT Full Multimodal Embeddings ---")

        if "multimodal_embeddings" not in tt_data:
            raise KeyError("multimodal_embeddings not found in saved data")

        multimodal_embeds = tt_data["multimodal_embeddings"].to(torch.float32)
        print(
            f"TT Multimodal: shape={multimodal_embeds.shape}, mean={multimodal_embeds.mean():.4f}, std={multimodal_embeds.std():.4f}"
        )
    else:
        # ============================================
        # TEST A: Use TT vision/projector + PT text
        # ============================================
        print("\n--- Using TT Vision/Projector + PyTorch Text ---")

        # Get TT projector output
        tt_projector = tt_data.get("projector_output", None)
        if tt_projector is None:
            # Fall back to running PT projector on TT vision
            tt_vision = tt_data.get("vision_output")
            if tt_vision is None:
                raise KeyError("Neither projector_output nor vision_output found")
            print(f"TT Vision: shape={tt_vision.shape}, mean={tt_vision.float().mean():.4f}")
            print("Running PyTorch projector on TT vision...")
            tt_projector = model.projector(tt_vision.to(torch.float32))

        print(
            f"TT Projector: shape={tt_projector.shape}, mean={tt_projector.float().mean():.4f}, std={tt_projector.float().std():.4f}"
        )

        # Get PyTorch text embeddings
        dummy_image = Image.new("RGB", (224, 224))
        inputs = processor(prompt, dummy_image)
        input_ids = inputs["input_ids"]

        # Add empty token 29871 if not present
        if input_ids[0, -1].item() != 29871:
            input_ids = torch.cat((input_ids, torch.tensor([[29871]])), dim=1)
            print(f"Added empty token 29871")

        print(f"input_ids: {input_ids[0].tolist()}")

        text_embeds = model.language_model.get_input_embeddings()(input_ids)
        print(f"PT Text Emb: shape={text_embeds.shape}, mean={text_embeds.float().mean():.4f}")

        # Build multimodal: [BOS] + [TT vision 256] + [PT text after BOS]
        multimodal_embeds = torch.cat(
            [
                text_embeds[:, :1, :],  # BOS from PyTorch
                tt_projector.to(torch.float32),  # Vision from TT
                text_embeds[:, 1:, :],  # Text from PyTorch
            ],
            dim=1,
        )
        print(f"Hybrid Multimodal: shape={multimodal_embeds.shape}")

    # ============================================
    # Run PyTorch LLM
    # ============================================
    print("\n--- Running PyTorch LLM ---")
    all_tokens = []
    with torch.no_grad():
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

            next_logits = logits[:, -1, :]
            next_token = torch.argmax(next_logits, dim=-1).item()
            all_tokens.append(next_token)

            top_vals, top_ids = torch.topk(next_logits[0], 5)
            top1_val = top_vals[0].item()
            top2_val = top_vals[1].item()
            gap = top1_val - top2_val

            print(f"Step {step}: token={next_token}, top1={top1_val:.4f}, top2={top2_val:.4f}, GAP={gap:.4f}")

            next_embed = model.language_model.get_input_embeddings()(torch.tensor([[next_token]]))
            current_embeds = next_embed

    print(f"\n✅ PyTorch LLM generated tokens: {all_tokens}")

    # Convert to action
    action_tokens = all_tokens[1:8]  # 7 action tokens
    bin_centers = np.linspace(-1, 1, 256)
    bin_indices = [t - 31744 for t in action_tokens]
    bin_indices = [max(0, min(255, i)) for i in bin_indices]
    normalized_actions = [bin_centers[i] for i in bin_indices]
    print(f"PyTorch LLM action tokens: {action_tokens}")
    print(f"PyTorch LLM normalized actions: {[f'{a:.4f}' for a in normalized_actions]}")

    # ============================================
    # Analysis
    # ============================================
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    unique_tokens = set(all_tokens)
    if len(unique_tokens) == 1:
        print(f"⚠️  All tokens are SAME ({all_tokens[0]})")
        if use_tt_multimodal:
            print("   → TT multimodal embeddings cause collapse")
            print("   → Issue is in TT Vision/Projector/Text embeddings")
        else:
            print("   → TT vision/projector causes collapse")
            print("   → Issue is in TT Vision or Projector")
    else:
        print(f"✅ Different tokens generated: {unique_tokens}")
        if use_tt_multimodal:
            print("   → TT multimodal embeddings are good")
            print("   → Issue is in TT LLM forward pass")
        else:
            print("   → TT vision/projector are good")

    # Compare with TT tokens if available
    if tt_tokens is not None:
        tt_tokens_list = tt_tokens.tolist() if hasattr(tt_tokens, "tolist") else list(tt_tokens)
        print(f"\nComparison:")
        print(f"  TT tokens:      {tt_tokens_list}")
        print(f"  PT LLM tokens:  {all_tokens}")
        if tt_tokens_list == all_tokens:
            print("  ✅ MATCH!")
        else:
            print("  ❌ DIFFERENT - TT LLM behaves differently than PT LLM")

    return all_tokens


def capture_llm_layer_outputs(output_path="/tmp/pytorch_llm_layers.pt"):
    """
    Capture layer-by-layer LLM outputs during prefill.
    This helps identify where TTNN diverges from PyTorch.

    Usage:
        python run_pytorch_openvla.py --llm-layers
    """
    import torch
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("=" * 70)
    print("LLM LAYER-BY-LAYER DEBUG")
    print("=" * 70)

    # Load model
    print("\nLoading PyTorch OpenVLA model...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    # Use synthetic RED image (matches test_image_sensitivity)
    image = Image.new("RGB", (224, 224), color=(200, 50, 50))
    prompt = "In: What action should the robot take to pick up the red block?\nOut:"

    print(f"Image: Synthetic RED (200, 50, 50)")
    print(f"Prompt: {prompt}")

    # Get inputs
    inputs = processor(prompt, image)
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    input_ids = inputs["input_ids"]

    print(f"\nInput shapes:")
    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  input_ids: {input_ids.shape} = {input_ids.tolist()}")

    # Storage for layer outputs
    layer_outputs = {}

    with torch.no_grad():
        # ============================================
        # STEP 1: Get vision features
        # ============================================
        vision_output = model.vision_backbone(pixel_values)
        layer_outputs["vision_output"] = vision_output.float().cpu()
        print(f"\nVision: shape={vision_output.shape}, mean={vision_output.float().mean():.4f}")

        # ============================================
        # STEP 2: Project to LLM dimension
        # ============================================
        projected = model.projector(vision_output)
        layer_outputs["projector_output"] = projected.float().cpu()
        print(f"Projector: shape={projected.shape}, mean={projected.float().mean():.4f}")

        # ============================================
        # STEP 3: Build multimodal embeddings
        # ============================================
        # Get text embeddings
        text_embeds = model.language_model.get_input_embeddings()(input_ids)
        layer_outputs["text_embeddings"] = text_embeds.float().cpu()
        print(f"Text embeddings: shape={text_embeds.shape}, mean={text_embeds.float().mean():.4f}")

        # Build combined embeddings (vision + text)
        # In OpenVLA, vision tokens replace the image placeholder token
        # The model concatenates: [BOS] + vision_tokens + text_tokens
        batch_size = 1
        num_vision_tokens = projected.shape[1]  # 256 tokens

        # Create combined embeddings
        # Format: vision_tokens (256) + text_tokens
        combined_embeds = torch.cat([projected, text_embeds], dim=1)
        layer_outputs["combined_embeddings"] = combined_embeds.float().cpu()
        print(f"Combined embeddings: shape={combined_embeds.shape}, mean={combined_embeds.float().mean():.4f}")

        # ============================================
        # STEP 4: Run through LLM layers with hooks
        # ============================================
        print(f"\n--- LLM Layer-by-Layer ---")

        llm = model.language_model.model

        # Hook to capture layer outputs
        hooks = []
        captured_outputs = {}

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                captured_outputs[name] = hidden.float().cpu()

            return hook

        # Register hooks on each transformer layer
        for i, layer in enumerate(llm.layers):
            hook = layer.register_forward_hook(make_hook(f"layer_{i}"))
            hooks.append(hook)

        # Also capture output of embed_tokens and final norm
        embed_hook = llm.embed_tokens.register_forward_hook(make_hook("embed_tokens"))
        hooks.append(embed_hook)
        norm_hook = llm.norm.register_forward_hook(make_hook("final_norm"))
        hooks.append(norm_hook)

        # Run forward pass with combined embeddings
        # We need to create the right attention mask and position ids
        seq_len = combined_embeds.shape[1]
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        # Run the LLM forward (just the model part, not generate)
        try:
            # Use the model's _prepare_inputs_for_generation or similar
            # Actually, let's just call the language model directly
            outputs = model.language_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get logits
            logits = outputs.logits
            layer_outputs["logits"] = logits.float().cpu()

            # Get last token's logits (for prefill output)
            last_logits = logits[0, -1, :]
            layer_outputs["last_token_logits"] = last_logits.float().cpu()

            # Top tokens
            top_vals, top_ids = torch.topk(last_logits, 10)
            print(f"\nPrefill output (last token logits):")
            print(f"  Top 10 tokens: {top_ids.tolist()}")
            print(f"  Top 10 values: {[f'{v:.2f}' for v in top_vals.tolist()]}")
            print(
                f"  Top1: {top_ids[0].item()}, Top2: {top_ids[1].item()}, GAP: {(top_vals[0]-top_vals[1]).item():.4f}"
            )

        finally:
            # Remove hooks
            for h in hooks:
                h.remove()

        # Add captured layer outputs
        for name, tensor in captured_outputs.items():
            layer_outputs[name] = tensor
            if "layer_" in name:
                layer_idx = int(name.split("_")[1])
                if layer_idx % 8 == 0 or layer_idx == 31:  # Print every 8th layer
                    print(f"  {name}: shape={tensor.shape}, mean={tensor.mean():.4f}, std={tensor.std():.4f}")

        # Print first and last layer stats
        if "layer_0" in captured_outputs:
            l0 = captured_outputs["layer_0"]
            print(f"\n  Layer 0 output: mean={l0.mean():.6f}, std={l0.std():.6f}")
            print(f"    First token: mean={l0[0,0,:].mean():.6f}, last token: mean={l0[0,-1,:].mean():.6f}")

        if "layer_31" in captured_outputs:
            l31 = captured_outputs["layer_31"]
            print(f"  Layer 31 output: mean={l31.mean():.6f}, std={l31.std():.6f}")
            print(f"    First token: mean={l31[0,0,:].mean():.6f}, last token: mean={l31[0,-1,:].mean():.6f}")

    # Save all outputs
    torch.save(layer_outputs, output_path)
    print(f"\n✅ Layer outputs saved to: {output_path}")
    print(f"   Keys: {list(layer_outputs.keys())}")

    return layer_outputs


if __name__ == "__main__":
    # Check for --llm-layers argument
    if "--llm-layers" in sys.argv:
        capture_llm_layer_outputs()
        sys.exit(0)

    # Check for --ttnn-vision argument (Test A or Test B)
    if "--ttnn-vision" in sys.argv:
        idx = sys.argv.index("--ttnn-vision")
        if idx + 1 < len(sys.argv):
            ttnn_vision_path = sys.argv[idx + 1]
            use_tt_mm = "--use-tt-mm" in sys.argv  # Test B: use full TT multimodal
            test_pytorch_llm_with_ttnn_vision(ttnn_vision_path, use_tt_multimodal=use_tt_mm)
            sys.exit(0)
        else:
            print("Error: --ttnn-vision requires a path argument")
            sys.exit(1)

    sys.exit(main())
