# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Parity harness between PyTorch reference MaskFormer and TT-NN implementation.

Responsibilities:
* Load the reference HuggingFace model in PyTorch and capture golden activations.
* Provide utilities to dump / load golden tensors to ``data/goldens`` (or user-specified).
* Expose helpers that compare TT outputs vs PyTorch outputs using PCC / max-abs metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union, List, Optional, TYPE_CHECKING
import argparse

import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - guard optional torch dependency for utility helpers
    torch = None

try:
    from transformers import AutoImageProcessor
except ModuleNotFoundError:  # pragma: no cover - fallback handled by caller
    AutoImageProcessor = None

if TYPE_CHECKING:  # pragma: no cover - only evaluated by type checkers
    from .weights import WeightConversionConfig


@dataclass
class ParityConfig:
    """Runtime options for parity collection and comparison."""

    golden_dir: Path = Path("models/experimental/maskformer_swin/data/goldens")
    pcc_threshold: float = float(os.environ.get("MASKFORMER_PCC_THRESHOLD", "0.98"))
    max_abs_threshold: float = float(os.environ.get("MASKFORMER_MAX_ABS_TOL", "0.001"))


GOLDEN_TAPS: Tuple[str, ...] = (
    # Swin stages (C2–C5)
    "backbone.stage1",
    "backbone.stage2",
    "backbone.stage3",
    "backbone.stage4",
    # Pixel decoder fusion outputs
    "pixel_decoder.high_res",
    # Transformer decoder taps (after selected blocks)
    "transformer_decoder.block0",
    "transformer_decoder.block3",
    # Final outputs
    "class_logits",
    "mask_logits",
)


def _require_torch(action: str) -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch is required to {action}. Install the `torch` package to continue.")


def _require_weight_api():
    _require_torch("work with MaskFormer reference weights")
    from . import weights as weights_module  # local import to avoid mandatory torch dependency at import time

    return weights_module


def _require_fallback_pipeline():
    _require_torch("run the MaskFormer fallback pipeline")
    from .fallback import MaskFormerFallbackPipeline

    return MaskFormerFallbackPipeline


def dump_golden_activations(
    images: Iterable[Union[np.ndarray, Image.Image, str]],
    config: ParityConfig,
    *,
    weight_cfg: Optional["WeightConversionConfig"] = None,
    max_images: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Run the PyTorch reference MaskFormer and persist golden activations.

    Parameters
    ----------
    images:
        Iterable of PIL images, numpy arrays, or filesystem paths.
    config:
        Parity configuration controlling thresholds and output directories.
    weight_cfg:
        Optional weight conversion configuration. Defaults to the canonical
        ``facebook/maskformer-swin-base-coco`` checkpoint when omitted.
    max_images:
        Optional cap on how many images from ``images`` are processed.

    Returns
    -------
    dict
        Mapping from tap name to file path for the stored tensors.
    """
    if AutoImageProcessor is None:
        raise RuntimeError("transformers is required to dump golden activations.")

    golden_dir = config.golden_dir
    golden_dir.mkdir(parents=True, exist_ok=True)

    _require_torch("dump golden activations")
    weights_mod = _require_weight_api()
    MaskFormerFallbackPipeline = _require_fallback_pipeline()

    weight_cfg = weight_cfg or weights_mod.WeightConversionConfig()
    processor = AutoImageProcessor.from_pretrained(weight_cfg.pretrained_model_name)
    reference = weights_mod.download_reference_weights(weight_cfg)
    tt_state = weights_mod.convert_state_dict_to_tt(reference.state_dict, weight_cfg)
    pipeline = MaskFormerFallbackPipeline.from_reference(reference, tt_state)

    tap_paths: Dict[str, Path] = {}

    for idx, raw_image in enumerate(images):
        if max_images is not None and idx >= max_images:
            break

        pil_image = _ensure_pil(raw_image)
        inputs = processor(images=pil_image, return_tensors="pt")
        pixel_values: torch.Tensor = inputs["pixel_values"]

        with torch.no_grad():
            outputs = pipeline.forward(pixel_values)

        image_root = golden_dir / f"image_{idx}"
        image_root.mkdir(parents=True, exist_ok=True)

        # Backbone feature maps (stage outputs)
        backbone_dir = image_root / "backbone"
        backbone_dir.mkdir(exist_ok=True)
        for stage_idx, fmap in enumerate(outputs.encoder_feature_maps):
            path = backbone_dir / f"stage_{stage_idx + 1}.npy"
            np.save(path, fmap.detach().cpu().numpy())
            tap_paths[f"image_{idx}/backbone/stage_{stage_idx + 1}"] = path

        # Pixel decoder features
        pixel_dir = image_root / "pixel_decoder"
        pixel_dir.mkdir(exist_ok=True)
        mask_features_path = pixel_dir / "mask_features.npy"
        np.save(mask_features_path, outputs.mask_features.detach().cpu().numpy())
        tap_paths[f"image_{idx}/pixel_decoder/mask_features"] = mask_features_path
        for level_idx, fmap in enumerate(outputs.pixel_decoder_hidden_states):
            path = pixel_dir / f"hidden_{level_idx + 1}.npy"
            np.save(path, fmap.detach().cpu().numpy())
            tap_paths[f"image_{idx}/pixel_decoder/hidden_{level_idx + 1}"] = path

        # Transformer decoder hidden states
        transformer_dir = image_root / "transformer_decoder"
        transformer_dir.mkdir(exist_ok=True)
        np.save(transformer_dir / "last_hidden_state.npy", outputs.transformer_hidden_states[-1].detach().cpu().numpy())
        tap_paths[f"image_{idx}/transformer_decoder/last_hidden_state"] = transformer_dir / "last_hidden_state.npy"
        # Store first and last layer for quick parity; include all for completeness.
        for layer_idx, hidden in enumerate(outputs.transformer_hidden_states):
            path = transformer_dir / f"layer_{layer_idx + 1}.npy"
            np.save(path, hidden.detach().cpu().numpy())
            tap_paths[f"image_{idx}/transformer_decoder/layer_{layer_idx + 1}"] = path

        # Final logits
        head_dir = image_root / "heads"
        head_dir.mkdir(exist_ok=True)
        class_path = head_dir / "class_logits.npy"
        mask_path = head_dir / "mask_logits.npy"
        np.save(class_path, outputs.class_logits.detach().cpu().numpy())
        np.save(mask_path, outputs.mask_logits.detach().cpu().numpy())
        tap_paths[f"image_{idx}/heads/class_logits"] = class_path
        tap_paths[f"image_{idx}/heads/mask_logits"] = mask_path

    return tap_paths


def compare_tensors(
    ref: np.ndarray,
    test: np.ndarray,
    *,
    pcc_threshold: float,
    max_abs_threshold: float,
) -> Tuple[float, float]:
    """
    Compare tensors and return (PCC, max_abs).  Raises AssertionError on failure.
    """

    if ref.shape != test.shape:
        raise AssertionError(f"Shape mismatch: ref {ref.shape} vs test {test.shape}")

    ref_flat = ref.astype(np.float32).ravel()
    test_flat = test.astype(np.float32).ravel()

    # Optional quantization mode to align with TTNN BF16 numerics.
    quantize_mode = os.environ.get("MASKFORMER_COMPARE_QUANTIZE", "").lower()
    if quantize_mode in {"bf16", "bfloat16"}:
        try:
            import torch  # type: ignore

            ref_flat = torch.from_numpy(ref_flat).to(dtype=torch.bfloat16).to(dtype=torch.float32).numpy()
            test_flat = torch.from_numpy(test_flat).to(dtype=torch.bfloat16).to(dtype=torch.float32).numpy()
        except Exception:
            pass

    diff = ref_flat - test_flat
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0

    ref_centered = ref_flat - ref_flat.mean()
    test_centered = test_flat - test_flat.mean()
    denom = float(np.linalg.norm(ref_centered) * np.linalg.norm(test_centered))
    if denom == 0.0:
        pcc = 1.0 if max_abs <= max_abs_threshold else 0.0
    else:
        pcc = float(np.dot(ref_centered, test_centered) / denom)

    if not np.isfinite(pcc):
        raise AssertionError("Non-finite PCC encountered.")

    if pcc < pcc_threshold and max_abs > max_abs_threshold:
        raise AssertionError(f"PCC {pcc:.6f} < {pcc_threshold} and max_abs {max_abs:.6f} > {max_abs_threshold}")

    return pcc, max_abs


def _ensure_pil(image: Union[np.ndarray, Image.Image, str]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] in (3, 4):
            arr = image
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr[:, :, :3], mode="RGB")
        raise ValueError(f"Unsupported numpy image shape: {image.shape}")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def load_golden_tensors(base_dir: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load every `.npy` file under ``base_dir`` and return a flat dictionary.

    Keys are forward-slash-separated relative paths without the `.npy` suffix.
    """

    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Golden directory does not exist: {base_dir}")

    tensors: Dict[str, np.ndarray] = {}
    for npy_file in sorted(base_path.rglob("*.npy")):
        rel_parts = npy_file.relative_to(base_path).with_suffix("").parts
        key = "/".join(rel_parts)
        tensors[key] = np.load(npy_file)
    return tensors


def compare_with_golden(
    test_tensors: Dict[str, np.ndarray],
    golden_tensors: Dict[str, np.ndarray],
    *,
    config: ParityConfig,
) -> Dict[str, Tuple[float, float]]:
    """
    Compare ``test_tensors`` against ``golden_tensors`` tap-by-tap.

    Returns a dictionary mapping tap names to (PCC, max_abs).
    """

    metrics: Dict[str, Tuple[float, float]] = {}
    for tap, golden in golden_tensors.items():
        if tap not in test_tensors:
            raise KeyError(f"Missing tap in test tensors: {tap}")
        test = test_tensors[tap]
        pcc, max_abs = compare_tensors(
            golden,
            test,
            pcc_threshold=config.pcc_threshold,
            max_abs_threshold=config.max_abs_threshold,
        )
        metrics[tap] = (pcc, max_abs)
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump HuggingFace reference activations for MaskFormer Swin-B.")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Paths to images (files or directories). Directories are scanned recursively for common image extensions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ParityConfig().golden_dir,
        help="Directory to store golden activations (default: models/experimental/maskformer_swin/data/goldens).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="facebook/maskformer-swin-base-coco",
        help="HuggingFace repo id or local checkpoint to load for goldens.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Preferred dtype for weight conversion (auto/bf16/fp16/fp32).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on the number of images processed.",
    )
    return parser.parse_args()


def _expand_image_paths(items: List[str]) -> List[Union[str, Image.Image]]:
    outputs: List[Union[str, Image.Image]] = []
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    for item in items:
        path = Path(item)
        if path.is_file():
            outputs.append(str(path))
        elif path.is_dir():
            for sub in sorted(path.rglob("*")):
                if sub.suffix.lower() in valid_exts and sub.is_file():
                    outputs.append(str(sub))
        else:
            raise FileNotFoundError(f"Image path not found: {item}")
    return outputs


def main() -> None:
    args = _parse_args()
    image_list = _expand_image_paths(args.images)
    config = ParityConfig(golden_dir=args.output_dir)
    weights_mod = _require_weight_api()
    weight_cfg = weights_mod.WeightConversionConfig(
        pretrained_model_name=args.weights,
        dtype=weights_mod.resolve_cli_dtype(args.dtype),
    )
    tap_paths = dump_golden_activations(
        image_list,
        config,
        weight_cfg=weight_cfg,
        max_images=args.max_images,
    )
    print(f"Saved {len(tap_paths)} activation tensors to {config.golden_dir}")


if __name__ == "__main__":
    main()
