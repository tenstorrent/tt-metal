"""
Standalone script to run PyTorch OpenVLA and save outputs.

Run this in a separate environment with compatible versions:

    # Create venv
    python3 -m venv /tmp/openvla_pt_env
    source /tmp/openvla_pt_env/bin/activate
    pip install torch transformers==4.40.0 timm==0.9.16 accelerate safetensors pillow numpy

    # Run this script
    python run_pytorch_openvla.py --output /tmp/pytorch_openvla_outputs.pt

"""

import argparse
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def main():
    parser = argparse.ArgumentParser(description="Run PyTorch OpenVLA and save outputs")
    parser.add_argument("--output", type=str, default="/tmp/pytorch_openvla_outputs.pt", help="Output file path")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (default: synthetic)")
    parser.add_argument(
        "--prompt",
        type=str,
        default="In: What action should the robot take to pick up the red block?\nOut:",
        help="Input prompt",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PyTorch OpenVLA Inference")
    print("=" * 60)

    # Load image
    if args.image:
        image = Image.open(args.image).convert("RGB")
        print(f"Using image: {args.image}")
    else:
        image = Image.new("RGB", (224, 224), color=(128, 64, 32))
        print("Using synthetic image (224x224, brown)")

    prompt = args.prompt
    print(f"Prompt: {prompt}")

    # Load processor and model
    print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)

    print("Loading PyTorch OpenVLA model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded: {type(model).__name__}")

    # Get inputs
    inputs = processor(prompt, image)
    print(f"\nInput shapes:")
    print(f"  pixel_values: {inputs['pixel_values'].shape}")
    print(f"  input_ids: {inputs['input_ids'].shape}")

    with torch.no_grad():
        pixel_values = inputs["pixel_values"].to(torch.bfloat16)
        input_ids = inputs["input_ids"]

        # Add empty token 29871 to match TTNN behavior (what predict_action() does)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat((input_ids, torch.tensor([[29871]], dtype=input_ids.dtype)), dim=1)

        # Vision backbone
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        dinov2_output = model.vision_backbone.featurizer(img)
        siglip_output = model.vision_backbone.fused_featurizer(img_fused)
        vision_output = model.vision_backbone(pixel_values)

        print(f"\nVision backbone:")
        print(f"  DINOv2: shape={dinov2_output.shape}, mean={dinov2_output.float().mean():.6f}")
        print(f"  SigLIP: shape={siglip_output.shape}, mean={siglip_output.float().mean():.6f}")
        print(f"  Fused:  shape={vision_output.shape}, mean={vision_output.float().mean():.6f}")

        # Projector
        projector_output = model.projector(vision_output)
        print(f"\nProjector: shape={projector_output.shape}, mean={projector_output.float().mean():.6f}")

        # Generate tokens
        print("\nGenerating tokens...")
        gen_outputs = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=8,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )

        scores = gen_outputs.scores
        all_tokens = []
        for i, score in enumerate(scores):
            top_vals, top_ids = torch.topk(score[0], 5)
            all_tokens.append(top_ids[0].item())

        print(f"Generated tokens: {all_tokens}")

        # Compute action from tokens
        action_tokens = torch.tensor(all_tokens[1:8], dtype=torch.long)
        action_token_ids = action_tokens.cpu().numpy()

        # Map tokens to action values (tokens 31744-31999 -> 256 bins -> [-1, 1])
        n_bins = 256
        bin_centers = np.linspace(-1, 1, n_bins)
        bin_indices = np.clip(action_token_ids - 31744, 0, n_bins - 1)
        normalized_actions = bin_centers[bin_indices]

        # Try to unnormalize using model's norm_stats
        action = normalized_actions
        if hasattr(model, "norm_stats") and "bridge_orig" in model.norm_stats:
            norm_stats = model.norm_stats["bridge_orig"]
            if "q01" in norm_stats:
                q01 = np.array(norm_stats["q01"])
                q99 = np.array(norm_stats["q99"])
                action = (normalized_actions + 1) / 2 * (q99 - q01) + q01

    print(f"\nAction: {action}")

    # Save outputs
    outputs = {
        "action": torch.tensor(action, dtype=torch.float32),
        "vision_output": vision_output.float().cpu(),
        "dinov2_output": dinov2_output.float().cpu(),
        "siglip_output": siglip_output.float().cpu(),
        "projector_output": projector_output.float().cpu(),
        "pixel_values": pixel_values.float().cpu(),
        "input_ids": inputs["input_ids"].cpu(),
        "prompt": prompt,
        "generated_tokens": torch.tensor(all_tokens, dtype=torch.long),
        "logits_scores": [s.cpu().float() for s in scores],
    }

    torch.save(outputs, args.output)
    print(f"\nOutputs saved to: {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Vision shape: {vision_output.shape}")
    print(f"Projector shape: {projector_output.shape}")
    print(f"Generated tokens: {all_tokens}")
    print(f"Action: {action}")

    return 0


def capture_layer_outputs(output_path="/tmp/pytorch_llm_layers.pt"):
    """
    Capture layer-by-layer LLM outputs during prefill.
    Useful for debugging where TTNN diverges from PyTorch.

    Usage:
        python run_pytorch_openvla.py --llm-layers
    """
    print("=" * 70)
    print("LLM LAYER-BY-LAYER DEBUG")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    # Use synthetic image
    image = Image.new("RGB", (224, 224), color=(200, 50, 50))
    prompt = "In: What action should the robot take to pick up the red block?\nOut:"

    inputs = processor(prompt, image)
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    input_ids = inputs["input_ids"]

    print(f"Input shapes: pixel_values={pixel_values.shape}, input_ids={input_ids.shape}")

    layer_outputs = {}
    captured_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured_outputs[name] = hidden.float().cpu()

        return hook

    with torch.no_grad():
        # Vision and projector
        vision_output = model.vision_backbone(pixel_values)
        layer_outputs["vision_output"] = vision_output.float().cpu()

        projected = model.projector(vision_output)
        layer_outputs["projector_output"] = projected.float().cpu()

        # Text embeddings
        text_embeds = model.language_model.get_input_embeddings()(input_ids)
        layer_outputs["text_embeddings"] = text_embeds.float().cpu()

        # Combined embeddings
        combined_embeds = torch.cat([projected, text_embeds], dim=1)
        layer_outputs["combined_embeddings"] = combined_embeds.float().cpu()

        # Register hooks
        llm = model.language_model.model
        hooks = []

        for i, layer in enumerate(llm.layers):
            hooks.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
        hooks.append(llm.embed_tokens.register_forward_hook(make_hook("embed_tokens")))
        hooks.append(llm.norm.register_forward_hook(make_hook("final_norm")))

        try:
            seq_len = combined_embeds.shape[1]
            attention_mask = torch.ones(1, seq_len, dtype=torch.long)

            outputs = model.language_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            layer_outputs["logits"] = outputs.logits.float().cpu()
            layer_outputs["last_token_logits"] = outputs.logits[0, -1, :].float().cpu()

            # Top tokens
            top_vals, top_ids = torch.topk(outputs.logits[0, -1, :], 10)
            print(f"\nPrefill top 10 tokens: {top_ids.tolist()}")
            print(f"Top 10 values: {[f'{v:.2f}' for v in top_vals.tolist()]}")

        finally:
            for h in hooks:
                h.remove()

        # Add captured layer outputs
        layer_outputs.update(captured_outputs)

    torch.save(layer_outputs, output_path)
    print(f"\nLayer outputs saved to: {output_path}")
    print(f"Keys: {list(layer_outputs.keys())}")

    return layer_outputs


def benchmark_fps(iterations=10, warmup=2, device="cpu"):
    """
    Benchmark PyTorch OpenVLA FPS.

    Usage:
        python run_pytorch_openvla.py --benchmark [--iterations 10] [--device cuda]
    """
    import time

    print("=" * 70)
    print("PyTorch OpenVLA FPS BENCHMARK")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
        print("Model on CUDA")
    else:
        device = "cpu"
        print("Model on CPU")

    model.eval()

    # Prepare input
    image = Image.new("RGB", (224, 224), color=(128, 64, 32))
    prompt = "In: What action should the robot take to pick up the red block?\nOut:"
    inputs = processor(prompt, image)

    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    input_ids = inputs["input_ids"]
    if device == "cuda":
        pixel_values = pixel_values.cuda()
        input_ids = input_ids.cuda()

    # Warmup
    print(f"\nWarmup ({warmup} iterations)...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=8,
                do_sample=False,
            )
        if device == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    for i in range(iterations):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=8,
                do_sample=False,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t_start) * 1000)

    times_arr = np.array(times)
    fps = 1000.0 / times_arr.mean()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Mean: {times_arr.mean():.2f}ms, Std: {times_arr.std():.2f}ms")
    print(f"Min: {times_arr.min():.2f}ms, Max: {times_arr.max():.2f}ms")
    print(f"FPS: {fps:.2f}")

    return {"fps": fps, "mean_time_ms": times_arr.mean(), "device": device}


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        iterations = 10
        device = "cpu"
        if "--iterations" in sys.argv:
            idx = sys.argv.index("--iterations")
            if idx + 1 < len(sys.argv):
                iterations = int(sys.argv[idx + 1])
        if "--device" in sys.argv:
            idx = sys.argv.index("--device")
            if idx + 1 < len(sys.argv):
                device = sys.argv[idx + 1]
        benchmark_fps(iterations=iterations, device=device)
        sys.exit(0)

    if "--llm-layers" in sys.argv:
        capture_layer_outputs()
        sys.exit(0)

    sys.exit(main())
