import os
import sys
import torch
from types import SimpleNamespace
import argparse
from pathlib import Path
import numpy as np
from models.experimental.lingbot_va.reference.lingbot import Lingbot


# Keys the server uses (va_robotwin_cfg.obs_cam_keys)
OBS_CAM_HIGH = "observation.images.cam_high"
OBS_CAM_LEFT_WRIST = "observation.images.cam_left_wrist"
OBS_CAM_RIGHT_WRIST = "observation.images.cam_right_wrist"
OBS_STATE = "observation.state"


def build_infer_obs(
    cam_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
    prompt: str = "",
    state: np.ndarray | None = None,
):
    """Build the single observation dict that the server expects for an infer (get-action) call."""
    out = {
        OBS_CAM_HIGH: _as_rgb_uint8(cam_high),
        OBS_CAM_LEFT_WRIST: _as_rgb_uint8(cam_left_wrist),
        OBS_CAM_RIGHT_WRIST: _as_rgb_uint8(cam_right_wrist),
        "task": prompt,
    }
    if state is not None:
        out[OBS_STATE] = np.asarray(state, dtype=np.float64)
    return out


def build_infer_message(
    cam_high: np.ndarray,
    cam_left_wrist: np.ndarray,
    cam_right_wrist: np.ndarray,
    prompt: str,
    video_guidance_scale: float = 5.0,
    action_guidance_scale: float = 1.0,
    state: np.ndarray | None = None,
):
    """Build the full message dict for model.infer(...) to get one action chunk."""
    obs = build_infer_obs(cam_high, cam_left_wrist, cam_right_wrist, prompt=prompt, state=state)
    return {
        "obs": obs,
        "prompt": prompt,
        "video_guidance_scale": video_guidance_scale,
        "action_guidance_scale": action_guidance_scale,
    }


def build_reset_message(prompt: str):
    """Build the message for model.infer(...) to reset (start of episode)."""
    return {"reset": True, "prompt": prompt}


def build_kv_cache_message(key_frame_obs_list: list[dict], state: np.ndarray):
    """Build the message for model.infer(...) to update KV cache after executing a chunk."""
    return {
        "obs": key_frame_obs_list,
        "compute_kv_cache": True,
        "imagine": False,
        "state": np.asarray(state, dtype=np.float64),
    }


def _as_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Ensure (H, W, 3) and uint8 for server (it will resize and normalize)."""
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {img.shape}")
    if img.dtype != np.uint8:
        if img.max() <= 1.0 + 1e-6:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def load_message_from_files(
    path_cam_high: str,
    path_cam_left_wrist: str,
    path_cam_right_wrist: str,
    prompt: str = "Lift the cup from the table",
) -> dict:
    """Load three images from paths and return the infer message dict."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL/Pillow required. pip install Pillow")
    cam_high = np.array(Image.open(path_cam_high).convert("RGB"))
    cam_left = np.array(Image.open(path_cam_left_wrist).convert("RGB"))
    cam_right = np.array(Image.open(path_cam_right_wrist).convert("RGB"))
    return build_infer_message(cam_high, cam_left, cam_right, prompt)


def run_inference(
    message: dict,
    checkpoint_path: str | Path,
    save_dir: str | Path | None = None,
) -> dict:
    """
    Run Lingbot-VA inference on the input dict using Lingbot class.

    Loads Lingbot with checkpoint at checkpoint_path, then:
    1. infer(reset_message) with message["prompt"]
    2. infer(message) to get action.

    save_dir: Where to save internal files (currently unused but kept for compatibility).

    Returns the result dict from the second infer (e.g. {"action": np.ndarray}).
    """

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")

    # Create a simple config object for Lingbot
    # Assuming checkpoint_path contains subdirectories: vae, text_encoder, tokenizer, transformer
    config = SimpleNamespace()
    config.vae_path = str(checkpoint_path / "vae")
    config.text_encoder_path = str(checkpoint_path / "text_encoder")
    config.tokenizer_path = str(checkpoint_path / "tokenizer")
    config.transformer_path = str(checkpoint_path / "transformer")
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.torch_dtype = torch.bfloat16
    config.patch_size = [1, 2, 2]  # Default patch size
    config.frame_chunk_size = 2
    config.height = 224
    config.width = 224

    # Initialize Lingbot model
    model = Lingbot(config)

    # Convert message format to obs format expected by Lingbot.infer
    # message has: {'obs': {...}, 'prompt': '...'}
    # Lingbot.infer expects: {'image': {...}, 'prompt': '...', 'state': ...}

    prompt = message.get("prompt", "")

    # Build reset message (empty observation with just prompt)
    reset_obs = {"image": {}, "prompt": prompt, "state": None}

    # Convert message['obs'] to the image format expected by Lingbot
    # Assuming message['obs'] has keys like 'cam_high', 'cam_left_wrist', 'cam_right_wrist'
    # or similar, and we need to map them to 'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'
    obs = message.get("obs", {})
    obs_image = {}

    # Convert message['obs'] to the image format expected by Lingbot
    # message['obs'] uses keys like "observation.images.cam_high"
    # Lingbot.infer expects keys like "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"
    obs = message.get("obs", {})
    obs_image = {}

    # Map observation keys to Lingbot image keys using the constants
    if OBS_CAM_HIGH in obs:
        obs_image["base_0_rgb"] = obs[OBS_CAM_HIGH]

    if OBS_CAM_LEFT_WRIST in obs:
        obs_image["left_wrist_0_rgb"] = obs[OBS_CAM_LEFT_WRIST]

    if OBS_CAM_RIGHT_WRIST in obs:
        obs_image["right_wrist_0_rgb"] = obs[OBS_CAM_RIGHT_WRIST]

    # Build the observation dict for inference
    inference_obs = {"image": obs_image, "prompt": prompt, "state": message.get("state", None)}

    # Run reset inference (with empty images)
    # model.infer(reset_obs)

    # Run actual inference
    result = model.infer(inference_obs, action_mode=True)
    return result


def _print_dict_shapes(d: dict, prefix: str = "") -> None:
    """Print dict keys and shapes of numpy arrays (for debugging)."""
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  {prefix}{k}: shape {v.shape}, dtype {v.dtype}")
        elif isinstance(v, dict):
            print(f"  {prefix}{k}: (dict)")
            _print_dict_shapes(v, prefix=prefix + "    ")
        elif isinstance(v, (int, float, str, bool)) or v is None:
            print(f"  {prefix}{k}: {v!r}")
        else:
            print(f"  {prefix}{k}: type {type(v).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build input dict and run Lingbot-VA inference.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("LINGBOT_VA_CHECKPOINT", ""),
        help="Path to checkpoint dir (vae, tokenizer, text_encoder, transformer). Default: env LINGBOT_VA_CHECKPOINT.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="",
        help="Dir with observation.images.cam_high.png, cam_left_wrist.png, cam_right_wrist.png. Default: example/robotwin/.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Lift the cup from the table",
        help="Task instruction string.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save the action array (e.g. action.npy).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Dir for VA_Server internal saves (latents_*.pt, actions_*.pt) or demo.mp4 when --generate. Default: evaluation/robotwin/out_inference.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run model.generate() instead of infer(): multi-chunk video generation, decode to RGB, save demo.mp4. Do not run infer().",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=2,
        help="Number of chunks for generate() (only used with --generate). Default: 10.",
    )
    args = parser.parse_args()

    save_dir = args.save_dir or str(Path(__file__).parent / "out_inference")
    images_dir = Path(args.images_dir) if args.images_dir else Path(__file__).parent / "example" / "robotwin"

    # import pdb; pdb.set_trace()

    if args.generate:
        if not args.checkpoint:
            print("--generate requires --checkpoint (or LINGBOT_VA_CHECKPOINT).")
            sys.exit(1)
        for key in (
            "observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist",
        ):
            if not (images_dir / f"{key}.png").exists():
                print(f"Missing {images_dir / f'{key}.png'} for generate(). Use --images-dir.")
                sys.exit(1)
        print("=" * 60)
        print("Running generate() (no infer): multi-chunk → decode → demo.mp4")
        print("=" * 60)
        print("Checkpoint:", args.checkpoint)
        print("Images dir:", images_dir)
        print("Prompt:", repr(args.prompt))
        print("Num chunks:", args.num_chunks)
        print("Save dir:", save_dir)
        print("=" * 60)
        out_path = run_generate(
            args.checkpoint,
            images_dir,
            args.prompt,
            save_dir,
            num_chunks=args.num_chunks,
        )
        print("Generated video saved to:", out_path)
        return

    # Infer mode: build message, run reset + infer one chunk
    cam_high_path = images_dir / "observation.images.cam_high.png"
    cam_left_path = images_dir / "observation.images.cam_left_wrist.png"
    cam_right_path = images_dir / "observation.images.cam_right_wrist.png"
    for p in (cam_high_path, cam_left_path, cam_right_path):
        if not p.exists():
            print(f"Missing image: {p}")
            print(f"  Use --images-dir to specify another dir.")
            sys.exit(1)

    message = load_message_from_files(
        str(cam_high_path),
        str(cam_left_path),
        str(cam_right_path),
        prompt=args.prompt,
    )

    print("=" * 60)
    print("Input dict (message for model.infer)")
    print("=" * 60)
    print("Top-level keys:", list(message.keys()))
    print("Observation keys (message['obs']):", list(message["obs"].keys()))
    print("Observation array shapes:")
    for k in (OBS_CAM_HIGH, OBS_CAM_LEFT_WRIST, OBS_CAM_RIGHT_WRIST):
        arr = message["obs"][k]
        print(f"  {k}: {arr.shape} {arr.dtype}")
    print("Prompt:", repr(message["prompt"]))
    print("=" * 60)

    if not args.checkpoint:
        print("No --checkpoint (or LINGBOT_VA_CHECKPOINT) set. Skipping inference.")
        print("Set checkpoint path to run inference on the above dict.")
        return

    print("\nRunning inference (reset + infer one chunk)...")
    print("VA_Server internal saves (latents_*.pt, actions_*.pt):", save_dir)
    result = run_inference(message, args.checkpoint, save_dir=save_dir)
    print("=" * 60)
    print("Inference result")
    print(result)
    print("=" * 60)
    if "action" in result:
        action = result["action"]
        print("action shape:", action.shape, "dtype:", action.dtype)
        if args.output:
            np.save(args.output, action)
            print("action saved to:", args.output)
    else:
        print("Keys:", list(result.keys()))
        _print_dict_shapes(result)
    print("=" * 60)


if __name__ == "__main__":
    main()
