# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI runner for MaskFormer Swin-Base demo.

This file will expose a ``python -m models.experimental.maskformer_swin.demo.runner``
entrypoint that:

1. Loads / converts weights (using ``weights.py`` helpers).
2. Builds TT-NN modules for backbone, pixel decoder, transformer decoder, and heads.
3. Executes the pipeline on a single image, produces overlay visualisations, and
   optionally dumps performance metrics in the TT-NN perf header format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, List
import json
import time
import os

import numpy as np
import torch

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover - only needed for preview mode
    Image = None

from ..tt.backbone_swin import MaskFormerSwinBackbone
from ..tt.heads import MaskFormerHeads, MaskFormerHeadsConfig
from ..tt.pixel_decoder import MaskFormerPixelDecoder, PixelDecoderConfig
from ..tt.transformer_decoder import MaskFormerTransformerDecoder, TransformerDecoderConfig
from ..tt.fallback import MaskFormerFallbackPipeline
from ..tt.weights import (
    ReferenceWeights,
    WeightConversionConfig,
    convert_state_dict_to_tt,
    download_reference_weights,
)
from ..tt.ttnn_compat import ttnn, require_ttnn
from ..tt.eval_utils import default_prediction_path, dump_predictions_json, run_coco_eval


def build_argparser() -> argparse.ArgumentParser:
    """Define CLI arguments for the demo."""

    parser = argparse.ArgumentParser(description="MaskFormer Swin-B TT-NN demo")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument(
        "--weights",
        type=str,
        default="facebook/maskformer-swin-base-coco",
        help="HuggingFace weights identifier or local path.",
    )
    parser.add_argument("--device", type=str, default="wormhole_n300", help="Target Tenstorrent device type.")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save overlay visualisation.")
    parser.add_argument("--dump-perf", type=Path, default=None, help="Optional path to write perf JSON.")
    parser.add_argument(
        "--dump-predictions",
        nargs="?",
        const=Path("generated/predictions_tt.json"),
        type=Path,
        default=None,
        help="Write per-query predictions JSON (optional path overrides default near --dump-perf).",
    )
    parser.add_argument("--coco-eval", action="store_true", help="Run COCO evaluation hooks (if available).")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional height to force via the image processor (e.g., 320 or 640).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional width to force via the image processor (e.g., 320 or 640).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Activation dtype (bf16/fp16/fp32/auto). Auto picks the first supported dtype.",
    )
    parser.add_argument("--enable-profiling", action="store_true", help="Enable TT-NN profiling hooks.")
    parser.add_argument(
        "--preview-backbone",
        action="store_true",
        help="Run the HuggingFace fallback backbone on the provided image and print feature map shapes.",
    )
    parser.add_argument(
        "--fallback-run",
        action="store_true",
        help="Execute the HuggingFace fallback pipeline end-to-end (until TT kernels are in place).",
    )
    parser.add_argument(
        "--tt-run",
        action="store_true",
        help="Run hybrid TT pipeline (TT heads and optional TT mask projection) and produce overlay/perf.",
    )
    parser.add_argument(
        "--fallback-overlay",
        type=Path,
        default=None,
        help="Optional path to save a semantic segmentation overlay using the fallback pipeline.",
    )
    parser.add_argument(
        "--describe-backbone",
        action="store_true",
        help="Print patch embedding spec and stage plan, then exit.",
    )
    parser.add_argument(
        "--patch-embed-parity",
        action="store_true",
        help="Compare TT vs HF patch embedding outputs (requires TT device).",
    )
    parser.add_argument(
        "--stage1-parity",
        action="store_true",
        help="Compare TT Stage 1 activations against HuggingFace fallback (requires TT device).",
    )
    parser.add_argument(
        "--stage2-parity",
        action="store_true",
        help="Compare TT Stage 2 activations against HuggingFace fallback (requires TT device).",
    )
    parser.add_argument(
        "--decoder-parity",
        action="store_true",
        help="Compare TT vs HF transformer decoder last_hidden_state (BF16-quantized compare recommended).",
    )
    # COCO evaluation options
    parser.add_argument("--coco-dir", type=Path, default=None, help="Path to COCO root with val2017 + annotations/")
    parser.add_argument(
        "--coco-panoptic-json",
        type=Path,
        default=None,
        help="Path to panoptic_val2017.json (enables PQ if panopticapi is available).",
    )
    parser.add_argument(
        "--coco-panoptic-root",
        type=Path,
        default=None,
        help="Directory containing panoptic_val2017/*.png ground-truth annotations.",
    )
    parser.add_argument(
        "--coco-max-images",
        type=int,
        default=50,
        help="Max images to evaluate for COCO metrics (default: 50).",
    )
    parser.add_argument(
        "--coco-report",
        type=Path,
        default=None,
        help="Path to write compact COCO metrics JSON (defaults next to --dump-perf).",
    )
    # Perf polish
    parser.add_argument(
        "--tt-repeats",
        type=int,
        default=2,
        help="Number of timed TT passes to average for perf (default: 2).",
    )
    return parser


def _resolve_dtype(name: str) -> Optional[object]:
    """Map common dtype strings to TT-NN dtypes."""

    normalized = name.replace("-", "").replace("_", "").lower()
    candidates = [
        ({"auto", "default"}, getattr(ttnn, "bfloat16", None)),
        ({"auto", "default"}, getattr(ttnn, "float16", None)),
        ({"auto", "default"}, getattr(ttnn, "float32", None)),
        ({"bf16", "bfloat16"}, getattr(ttnn, "bfloat16", None)),
        ({"fp16", "float16"}, getattr(ttnn, "float16", None)),
        ({"fp32", "float32"}, getattr(ttnn, "float32", None)),
    ]

    if normalized in {"auto", "default"}:
        for aliases, dtype in candidates:
            if dtype is not None and "auto" in aliases:
                return dtype
        return None

    for aliases, dtype in candidates:
        if normalized in aliases and dtype is not None:
            return dtype
        if normalized in aliases and dtype is None:
            raise ValueError(f"dtype '{name}' requested but not available in this TT-NN build.")

    valid = sorted(
        {
            alias
            for aliases, dtype in candidates
            if dtype is not None
            for alias in aliases
            if alias not in {"auto", "default"}
        }
    )
    raise ValueError(f"Unsupported dtype '{name}'. Expected one of: {', '.join(valid)}")


def main(argv: Optional[list[str]] = None) -> None:
    """Command-line entrypoint (download weights + prepare configs)."""

    args = build_argparser().parse_args(argv)
    try:
        dtype = _resolve_dtype(args.dtype)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    weight_cfg = WeightConversionConfig(
        pretrained_model_name=args.weights,
        cache_dir=None,
        dtype=dtype,
    )

    print(f"[maskformer] Resolving weights for '{weight_cfg.pretrained_model_name}' ...")
    ref_weights = download_reference_weights(weight_cfg)
    print(f"[maskformer] Loaded checkpoint from {ref_weights.checkpoint_path}")

    try:
        tt_state_dict = convert_state_dict_to_tt(ref_weights.state_dict, weight_cfg)
    except NotImplementedError as exc:
        print("[maskformer] Weight conversion not implemented yet:")
        print(f"  → {exc}")
        print("Aborting before TT module construction; implement conversion to proceed.")
        return

    backbone_config_payload = (
        ref_weights.config.get("backbone_config", {}) if isinstance(ref_weights.config, dict) else {}
    )
    device = None
    try:
        if args.patch_embed_parity or args.stage1_parity or args.stage2_parity or args.decoder_parity:
            require_ttnn("run patch embedding parity")
            device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)

        backbone = MaskFormerSwinBackbone.from_huggingface(
            tt_state_dict,
            device=device,
            config_dict=backbone_config_payload,
        )
        assert backbone is not None  # silence linters
        print("[maskformer] Backbone weights loaded (device construction pending).")

        if args.describe_backbone:
            _describe_backbone(backbone)
            return

        if args.patch_embed_parity:
            _run_patch_embed_parity(args, backbone)
            return

        if args.stage1_parity:
            _run_stage1_parity(args, backbone)
            return
        if args.stage2_parity:
            _run_stage2_parity(args, backbone)
            return
        if args.decoder_parity:
            _run_decoder_parity(args, ref_weights, tt_state_dict)
            return

        if args.preview_backbone or args.fallback_overlay:
            _run_backbone_preview(args, backbone, tt_state_dict, ref_weights)
            return

        # COCO evaluation can run on CPU or TT depending on --tt-run.
        if args.coco_eval:
            _run_coco_eval(args, ref_weights, tt_state_dict)
            return

        if args.fallback_run:
            _run_fallback_inference(args, ref_weights, tt_state_dict)
            return
        if args.tt_run:
            _run_tt_inference(args, ref_weights, tt_state_dict)
            return

        # No explicit mode selected. The demo supports:
        #  - --fallback-run (CPU), --tt-run (TT decoder + heads),
        #  - --coco-eval, and parity/preview helpers.
        # This branch is intentionally non-operative to avoid surprising work;
        # pass one of the flags above to run. (Bounty #30876 requires --tt-run.)
        print("[maskformer] No action specified. Try --tt-run or --fallback-run.\n" "Use --help for all options.")
        return
    finally:
        if device is not None:
            ttnn.close_device(device)


def _describe_backbone(backbone: MaskFormerSwinBackbone) -> None:
    spec = backbone.get_patch_embed_spec()
    print("[maskformer] Patch embedding spec:")
    if spec is None:
        print("  (not available)")
    else:
        print(
            f"  in_channels={spec.in_channels} out_channels={spec.out_channels}"
            f" kernel={spec.kernel_size} stride={spec.stride}"
        )
    plans = backbone.describe_stage_plan()
    print("[maskformer] Stage plan:")
    for plan in plans:
        task_str = ", ".join(plan.tasks)
        print(
            f"  {plan.name}: depth={plan.depth} heads={plan.num_heads} "
            f"dim={plan.input_dim}->{plan.output_dim} tasks=[{task_str}]"
        )


def _run_patch_embed_parity(args: argparse.Namespace, backbone: MaskFormerSwinBackbone) -> None:
    import numpy as np

    if Image is None:
        raise SystemExit("Patch embedding parity requires Pillow; install via `pip install pillow`.")
    try:
        from transformers import AutoImageProcessor
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "Patch embedding parity requires transformers; install via `pip install transformers`."
        ) from exc

    if not backbone._can_run_patch_embed_tt():
        raise SystemExit("Patch embedding TT path unavailable; ensure TT runtime + fallback ops exist.")

    processor = AutoImageProcessor.from_pretrained(args.weights)
    image = Image.open(args.image).convert("RGB")
    proc_kwargs = {}
    if args.height is not None and args.width is not None:
        proc_kwargs.update({"size": {"height": int(args.height), "width": int(args.width)}, "do_resize": True})
    pixel_values = processor(images=image, return_tensors="pt", **proc_kwargs)["pixel_values"]

    hf_embeds, _ = backbone.run_patch_embedding_hf(pixel_values)
    tt_embeds, _ = backbone.run_patch_embedding_tt(pixel_values)

    from . import parity as parity_mod

    pcc, max_abs = parity_mod.compare_tensors(
        hf_embeds.numpy().astype(np.float32),
        tt_embeds.to(dtype=torch.float32).numpy().astype(np.float32),
        pcc_threshold=parity_mod.ParityConfig().pcc_threshold,
        max_abs_threshold=parity_mod.ParityConfig().max_abs_threshold,
    )
    print(f"[maskformer] Patch embed parity: PCC={pcc:.6f} max_abs={max_abs:.3e}")


def _run_stage1_parity(args: argparse.Namespace, backbone: MaskFormerSwinBackbone) -> None:
    import numpy as np

    if Image is None:
        raise SystemExit("Stage 1 parity requires Pillow; install via `pip install pillow`.")
    try:
        from transformers import AutoImageProcessor
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("Stage 1 parity requires transformers; install via `pip install transformers`.") from exc

    if backbone._hf_backbone_model is None:
        raise SystemExit("HuggingFace backbone not initialized; load weights before running parity.")

    processor = AutoImageProcessor.from_pretrained(args.weights)
    image = Image.open(args.image).convert("RGB")
    proc_kwargs = {}
    if args.height is not None and args.width is not None:
        proc_kwargs.update({"size": {"height": int(args.height), "width": int(args.width)}, "do_resize": True})
    pixel_values = processor(images=image, return_tensors="pt", **proc_kwargs)["pixel_values"]

    # Compare end-of-stage feature first
    with torch.no_grad():
        hf_outputs = backbone._hf_backbone_model(pixel_values.to(backbone._torch_device))
    # TT stage pre-merge and taps from a single execution to avoid nondeterminism across runs
    tt_stage_feature, tt_taps, (height, width) = backbone.run_stage1_tt_with_taps(pixel_values)
    tt_pre_np = tt_stage_feature.numpy().astype(np.float32)
    tt_post_np = None

    # HF post-merge feature map
    hf_post_np = hf_outputs.feature_maps[0].detach().cpu().numpy().astype(np.float32)

    # HF pre-merge final hidden state from stage 1
    hf_model = backbone._hf_backbone_model.model
    with torch.no_grad():
        embeddings, dims = hf_model.embeddings(pixel_values.to(backbone._torch_device), interpolate_pos_encoding=False)
        stage = hf_model.encoder.layers[0]
        _, _, hidden_states_seq = stage(embeddings, dims, output_hidden_states=True)
    hf_pre_np = (
        hidden_states_seq[-1]
        .view(1, dims[0], dims[1], hidden_states_seq[-1].shape[-1])
        .permute(0, 3, 1, 2)
        .contiguous()
        .cpu()
        .float()
        .numpy()
        .astype(np.float32)
    )

    from . import parity as parity_mod

    # Compare pre-merge parity
    pcc_pre, max_abs_pre = parity_mod.compare_tensors(
        hf_pre_np,
        tt_pre_np,
        pcc_threshold=parity_mod.ParityConfig().pcc_threshold,
        max_abs_threshold=parity_mod.ParityConfig().max_abs_threshold,
    )
    print(f"[maskformer] Stage 1 pre-merge parity: PCC={pcc_pre:.6f} max_abs={max_abs_pre:.3e}")
    perf_payload: Dict[str, object] = {
        "stage1_pre_merge": {"pcc": float(pcc_pre), "max_abs": float(max_abs_pre)},
        "post_merge": None,
        "tap_metrics": [],
    }

    # Compare post-merge parity if TT post is available
    if tt_post_np is not None and hf_post_np.shape == tt_post_np.shape:
        pcc_post, max_abs_post = parity_mod.compare_tensors(
            hf_post_np,
            tt_post_np,
            pcc_threshold=parity_mod.ParityConfig().pcc_threshold,
            max_abs_threshold=parity_mod.ParityConfig().max_abs_threshold,
        )
        print(f"[maskformer] Stage 1 post-merge parity: PCC={pcc_post:.6f} max_abs={max_abs_post:.3e}")
        perf_payload["post_merge"] = {"pcc": float(pcc_post), "max_abs": float(max_abs_post)}
    else:
        print(
            f"[maskformer] Stage 1 post-merge parity: unavailable (HF={hf_post_np.shape} vs TT={None if tt_post_np is None else tt_post_np.shape})"
        )

    # Per-block taps: compare TT taps and HF stage block outputs via HF internals
    print("[maskformer] Collecting per-block taps for Stage 1...")
    try:
        hf_model = backbone._hf_backbone_model.model
        with torch.no_grad():
            # Use the same padding convention as the TT path to align dims
            embeddings, dims = backbone.run_patch_embedding_hf(pixel_values)
            embeddings = embeddings.to(backbone._torch_device)
            stage = hf_model.encoder.layers[0]
            _, _, hidden_states_seq = stage(embeddings, dims, output_hidden_states=True)
        # Choose HF taps after each block (post-residual). For 2 blocks, use indices [1, 3].
        hf_taps_all: List[torch.Tensor] = []
        for idx, seq in enumerate(hidden_states_seq):
            b, s, c = seq.shape
            h, w = dims
            assert s == h * w, "HF tap sequence length mismatch"
            fmap = seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous().cpu().float()
            hf_taps_all.append(fmap)
        hf_taps: List[torch.Tensor] = [hf_taps_all[i] for i in range(len(hf_taps_all)) if i % 2 == 1]
        print(f"[maskformer] Stage1 taps collected: TT={len(tt_taps)} HF_post={len(hf_taps)}")
        # Compare corresponding TT vs HF post-block taps
        for i in range(min(len(tt_taps), len(hf_taps))):
            pcc_i, max_abs_i = parity_mod.compare_tensors(
                hf_taps[i].numpy().astype(np.float32),
                tt_taps[i].numpy().astype(np.float32),
                pcc_threshold=parity_mod.ParityConfig().pcc_threshold,
                max_abs_threshold=parity_mod.ParityConfig().max_abs_threshold,
            )
            print(f"[maskformer] Stage1 block[{i}] parity: PCC={pcc_i:.6f} max_abs={max_abs_i:.3e}")
            perf_payload["tap_metrics"].append({"block": i, "pcc": float(pcc_i), "max_abs": float(max_abs_i)})
    except Exception as e:
        print(f"[maskformer] Stage1 per-block taps failed: {e}")

    if args.dump_perf is not None:
        try:
            with args.dump_perf.open("w", encoding="utf-8") as f:
                json.dump(perf_payload, f, indent=2)
            print(f"[maskformer] Wrote perf summary to {args.dump_perf}")
        except Exception as e:
            print(f"[maskformer] Failed to write perf summary: {e}")


def _run_stage2_parity(args: argparse.Namespace, backbone: MaskFormerSwinBackbone) -> None:
    import numpy as np

    if Image is None:
        raise SystemExit("Stage 2 parity requires Pillow; install via `pip install pillow`.")
    try:
        from transformers import AutoImageProcessor
    except ModuleNotFoundError as exc:
        raise SystemExit("Stage 2 parity requires transformers; install via `pip install transformers`.") from exc

    if backbone._hf_backbone_model is None:
        raise SystemExit("HuggingFace backbone not initialized; load weights before running parity.")

    processor = AutoImageProcessor.from_pretrained(args.weights)
    image = Image.open(args.image).convert("RGB")
    proc_kwargs = {}
    if args.height is not None and args.width is not None:
        proc_kwargs.update({"size": {"height": int(args.height), "width": int(args.width)}, "do_resize": True})
    pixel_values = processor(images=image, return_tensors="pt", **proc_kwargs)["pixel_values"]

    # TT Stage 2 (pre-merge + taps)
    tt_stage2_pre, tt_stage2_taps, (h2, w2) = backbone.run_stage2_tt_with_taps(pixel_values)
    tt_stage2_pre_np = tt_stage2_pre.numpy().astype(np.float32)

    # HF Stage 2 reference via stage internals to avoid extra per-stage LN
    try:
        hf_model = backbone._hf_backbone_model.model
        with torch.no_grad():
            # Use HF embeddings helper to mirror TT padding behavior
            embeddings, dims1 = backbone.run_patch_embedding_hf(pixel_values)
            embeddings = embeddings.to(backbone._torch_device)
            stage1 = hf_model.encoder.layers[0]
            stage1_out, output_dims1, _ = stage1(embeddings, dims1, output_hidden_states=False)
            # Next stage spatial dims are the last two entries of output_dims1
            dims2 = (output_dims1[-2], output_dims1[-1])
            stage2 = hf_model.encoder.layers[1]
            _, _, hidden_states_seq2 = stage2(stage1_out, dims2, output_hidden_states=True)
        # Last element corresponds to post-residual output of the last block (pre-merge)
        hf_seq2 = hidden_states_seq2[-1]
        b, s, c = hf_seq2.shape
        assert s == h2 * w2, "HF Stage 2 sequence length mismatch"
        hf_stage2_pre = (
            hf_seq2.view(b, h2, w2, c).permute(0, 3, 1, 2).contiguous().cpu().float().numpy().astype(np.float32)
        )
    except Exception as e:
        raise SystemExit(f"Failed to compute HF Stage 2 internals for parity: {e}")

    from . import parity as parity_mod

    pcc_pre, max_abs_pre = parity_mod.compare_tensors(
        hf_stage2_pre,
        tt_stage2_pre_np,
        pcc_threshold=parity_mod.ParityConfig().pcc_threshold,
        max_abs_threshold=parity_mod.ParityConfig().max_abs_threshold,
    )
    print(f"[maskformer] Stage 2 pre-merge parity: PCC={pcc_pre:.6f} max_abs={max_abs_pre:.3e}")

    # Optional per-block tap parity for Stage 2 to localize drift
    tap_metrics: List[Dict[str, float]] = []
    try:
        hf_taps_all2: List[torch.Tensor] = []
        for idx, seq in enumerate(hidden_states_seq2):
            bb, ss, cc = seq.shape
            assert ss == h2 * w2, "HF Stage 2 tap sequence length mismatch"
            fmap = seq.view(bb, h2, w2, cc).permute(0, 3, 1, 2).contiguous().cpu().float()
            hf_taps_all2.append(fmap)
        hf_taps2: List[torch.Tensor] = [hf_taps_all2[i] for i in range(len(hf_taps_all2)) if i % 2 == 1]
        print(f"[maskformer] Stage2 taps collected: TT={len(tt_stage2_taps)} HF_post={len(hf_taps2)}")
        for i in range(min(len(tt_stage2_taps), len(hf_taps2))):
            pcc_i, max_abs_i = parity_mod.compare_tensors(
                hf_taps2[i].numpy().astype(np.float32),
                tt_stage2_taps[i].numpy().astype(np.float32),
                pcc_threshold=parity_mod.ParityConfig().pcc_threshold,
                max_abs_threshold=parity_mod.ParityConfig().max_abs_threshold,
            )
            print(f"[maskformer] Stage2 block[{i}] parity: PCC={pcc_i:.6f} max_abs={max_abs_i:.3e}")
            tap_metrics.append({"block": i, "pcc": float(pcc_i), "max_abs": float(max_abs_i)})
    except Exception as e:
        print(f"[maskformer] Stage2 per-block taps failed: {e}")

    # Per-block taps for Stage 2 (optional)
    perf_payload: Dict[str, object] = {
        "stage2_pre_merge": {"pcc": float(pcc_pre), "max_abs": float(max_abs_pre)},
        "tap_metrics": tap_metrics,
    }

    if args.dump_perf is not None:
        try:
            with args.dump_perf.open("w", encoding="utf-8") as f:
                json.dump(perf_payload, f, indent=2)
            print(f"[maskformer] Wrote Stage 2 perf summary to {args.dump_perf}")
        except Exception as e:
            print(f"[maskformer] Failed to write Stage 2 perf summary: {e}")


def _run_backbone_preview(
    args: argparse.Namespace,
    backbone: MaskFormerSwinBackbone,
    state_dict: Dict[str, object],
    ref_weights: ReferenceWeights,
) -> None:
    """Execute the HuggingFace fallback backbone and report feature map shapes."""

    if Image is None:
        raise SystemExit("Preview mode requires Pillow; install via `pip install pillow`.")

    try:
        pass
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Preview mode requires transformers; install via `pip install transformers`.") from exc

    processor, image, pixel_values = _prepare_inputs(args)

    feature_maps, encoder_hidden = backbone.forward(pixel_values)
    print("[maskformer] Backbone feature maps:")
    for idx, fmap in enumerate(feature_maps):
        print(f"  stage{idx + 1}: {tuple(fmap.shape)}")

    pixel_decoder_cfg = PixelDecoderConfig()
    pixel_decoder = MaskFormerPixelDecoder.from_huggingface(
        state_dict,
        config=pixel_decoder_cfg,
        device=None,
    )
    mask_features, decoder_hidden = pixel_decoder.forward(feature_maps)
    print("[maskformer] Pixel decoder output:")
    print(f"  mask_features: {tuple(mask_features.shape)}")
    for idx, fmap in enumerate(decoder_hidden):
        print(f"  decoder_hidden[{idx}]: {tuple(fmap.shape)}")

    transformer_cfg = TransformerDecoderConfig(
        num_layers=ref_weights.config.get("num_hidden_layers", 6),
        num_attention_heads=ref_weights.config.get("num_attention_heads", 8),
        hidden_dim=ref_weights.config.get("fpn_feature_size", 256),
        dim_feedforward=ref_weights.config["decoder_config"].get("decoder_ffn_dim", 2048),
        dropout=ref_weights.config["decoder_config"].get("dropout", 0.0),
        activation=ref_weights.config["decoder_config"].get("activation_function", "relu"),
        in_features=pixel_decoder_cfg.input_channels[-1],
        maskformer_config=ref_weights.config,
    )
    transformer_decoder = MaskFormerTransformerDecoder.from_huggingface(
        state_dict,
        config=transformer_cfg,
        device=None,
    )
    image_features = feature_maps[-1]
    transformer_output, decoder_hidden_states, _ = transformer_decoder.forward(
        image_features,
        output_hidden_states=True,
        output_attentions=False,
    )
    print("[maskformer] Transformer decoder output:")
    print(f"  last_hidden_state: {tuple(transformer_output.shape)}")
    for idx, fmap in enumerate(decoder_hidden_states):
        print(f"  transformer_hidden[{idx}]: {tuple(fmap.shape)}")

    num_classes = len(ref_weights.config.get("id2label", {}))
    heads = MaskFormerHeads.from_huggingface(
        state_dict,
        config=MaskFormerHeadsConfig(
            num_classes=num_classes,
            hidden_dim=transformer_cfg.hidden_dim,
            mask_dim=pixel_decoder_cfg.fpn_dim,
        ),
        device=None,
    )
    class_logits, mask_logits = heads.forward(transformer_output, mask_features)
    print("[maskformer] Heads output:")
    print(f"  class_logits: {tuple(class_logits.shape)}")
    print(f"  mask_logits:  {tuple(mask_logits.shape)}")

    if args.fallback_overlay:
        pipeline = MaskFormerFallbackPipeline.from_reference(ref_weights, state_dict)
        _save_overlay(args.fallback_overlay, processor, image, pipeline, pixel_values)


def _run_fallback_inference(
    args: argparse.Namespace,
    ref_weights: ReferenceWeights,
    state_dict: Dict[str, object],
) -> None:
    if Image is None:
        raise SystemExit("Fallback inference requires Pillow; install via `pip install pillow`.")

    processor, image, pixel_values = _prepare_inputs(args)
    # If a TT device type is provided and TTNN is available, open it to enable TT paths in pixel decoder / heads.
    tt_device = None
    try:
        if (
            args.device
            and ttnn is not None
            and isinstance(args.device, str)
            and args.device.lower().startswith(("wormhole", "blackhole"))
        ):
            tt_device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)
    except Exception:
        tt_device = None
    pipeline = MaskFormerFallbackPipeline.from_reference(ref_weights, state_dict, device=tt_device)

    with torch.no_grad():
        _ = pipeline.forward(pixel_values)
        start = time.perf_counter()
        outputs = pipeline.forward(pixel_values)
        latency_ms = (time.perf_counter() - start) * 1000.0

    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    print(f"[maskformer] Fallback inference latency: {latency_ms:.2f} ms ({fps:.2f} FPS)")

    _print_class_summary(outputs, ref_weights)

    if args.dump_predictions is not None:
        pred_path = args.dump_predictions
        if args.dump_perf is not None and pred_path == Path("generated/predictions_tt.json"):
            pred_path = default_prediction_path(args.dump_perf, mode="fallback")
        dump_predictions_json(
            class_logits=outputs.class_logits,
            id2label=ref_weights.config.get("id2label", {}),
            output_path=pred_path,
            task_type="instance",
        )
        print(f"[maskformer] Wrote predictions to {pred_path}")

    overlay_path = args.save or args.fallback_overlay
    if overlay_path:
        _save_overlay(overlay_path, processor, image, pipeline, pixel_values)

    if tt_device is not None:
        try:
            ttnn.close_device(tt_device)
        except Exception:
            pass

    target_hw = [int(args.height), int(args.width)] if args.height and args.width else list(image.size[::-1])

    if args.dump_perf:
        perf_payload = {
            "mode": "fallback",
            "device": "cpu",
            "dtype": "fp32",
            "latency_ms": latency_ms,
            "fps": fps,
            "image_size_hw": target_hw,
            "num_queries": int(outputs.class_logits.shape[1]),
            "tt_submodules": [],
        }
        args.dump_perf.parent.mkdir(parents=True, exist_ok=True)
        args.dump_perf.write_text(json.dumps(perf_payload, indent=2))
        _emit_perf_header(args.dump_perf.with_name(args.dump_perf.stem + "_header.json"), perf_payload)
        print(
            f"[maskformer] Wrote fallback perf to {args.dump_perf} and header to "
            f"{args.dump_perf.with_name(args.dump_perf.stem + '_header.json')}"
        )


def _run_tt_inference(
    args: argparse.Namespace,
    ref_weights: ReferenceWeights,
    state_dict: Dict[str, object],
) -> None:
    if Image is None:
        raise SystemExit("TT run requires Pillow; install via `pip install pillow`.")

    processor, image, pixel_values = _prepare_inputs(args)
    # Open TT device and enable TT pixel mask projection
    tt_device = None
    try:
        if ttnn is not None:
            tt_device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)
            os.environ["MASKFORMER_TT_MASK_PROJ"] = "1"
            os.environ["MASKFORMER_TT_DECODER"] = "1"
    except Exception:
        tt_device = None

    pipeline = MaskFormerFallbackPipeline.from_reference(ref_weights, state_dict, device=tt_device)

    # Warmup + timed repeats
    latencies: List[float] = []
    with torch.no_grad():
        _ = pipeline.forward(pixel_values)
        repeats = max(int(args.tt_repeats or 1), 1)
        for i in range(repeats):
            start = time.perf_counter()
            outputs = pipeline.forward(pixel_values)
            latency_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(latency_ms)

    latency_ms = float(np.mean(latencies)) if latencies else float("nan")
    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    print(
        f"[maskformer] TT run latency: mean={latency_ms:.2f} ms std={np.std(latencies):.2f} ms "
        f"({fps:.2f} FPS over {len(latencies)} runs)"
    )
    _print_class_summary(outputs, ref_weights)

    if args.dump_predictions is not None:
        pred_path = args.dump_predictions
        if args.dump_perf is not None and pred_path == Path("generated/predictions_tt.json"):
            pred_path = default_prediction_path(args.dump_perf, mode="tt")
        dump_predictions_json(
            class_logits=outputs.class_logits,
            id2label=ref_weights.config.get("id2label", {}),
            output_path=pred_path,
            task_type="instance",
        )
        print(f"[maskformer] Wrote predictions to {pred_path}")

    overlay_path = args.save or args.fallback_overlay
    if overlay_path:
        _save_overlay(overlay_path, processor, image, pipeline, pixel_values)

    # Emit perf JSON + header for review
    target_hw = [int(args.height), int(args.width)] if args.height and args.width else list(image.size[::-1])

    if args.dump_perf:
        dtype_str = _dtype_to_str(_resolve_dtype(args.dtype)) if args.dtype else "auto"
        perf_payload = {
            "mode": "tt_run",
            "device": str(args.device),
            "dtype": dtype_str,
            "latency_ms": latency_ms,
            "latency_ms_repeats": latencies,
            "latency_ms_std": float(np.std(latencies)) if latencies else None,
            "fps": fps,
            "image_size_hw": target_hw,
            "num_queries": int(outputs.class_logits.shape[1]),
            "tt_submodules": ["decoder", "heads", "mask_projection"],
        }
        args.dump_perf.parent.mkdir(parents=True, exist_ok=True)
        # Write aggregate perf
        args.dump_perf.write_text(json.dumps(perf_payload, indent=2))
        # Write per-run snapshots for traceability
        for i, l in enumerate(latencies):
            snap = dict(perf_payload)
            snap.update({"latency_ms": float(l), "latency_ms_repeats": [float(l)], "latency_ms_std": 0.0})
            snap_path = args.dump_perf.with_name(args.dump_perf.stem + f"_run{i+1}.json")
            snap_path.write_text(json.dumps(snap, indent=2))
        _emit_perf_header(args.dump_perf.with_name(args.dump_perf.stem + "_header.json"), perf_payload)
        print(
            f"[maskformer] Wrote TT perf (avg) to {args.dump_perf}, header to "
            f"{args.dump_perf.with_name(args.dump_perf.stem + '_header.json')} and {len(latencies)} snapshots"
        )

    if tt_device is not None:
        try:
            ttnn.close_device(tt_device)
        except Exception:
            pass


def _save_overlay(
    output_path: Path,
    image_processor,
    image,
    pipeline: MaskFormerFallbackPipeline,
    pixel_values: torch.Tensor,
) -> None:
    outputs = pipeline.forward(pixel_values, output_hidden_states=False, output_attentions=False)
    segmentation = pipeline.post_process_semantic(
        outputs,
        image_processor=image_processor,
        target_sizes=[image.size[::-1]],
    )[0]

    segmentation = segmentation.cpu().numpy().astype(np.int32)
    num_classes = getattr(pipeline.config, "num_labels", len(getattr(pipeline.config, "id2label", {})))
    palette = _build_palette(num_classes + 1)
    colored = palette[segmentation % len(palette)]
    overlay = Image.fromarray(colored.astype(np.uint8)).resize(image.size, Image.NEAREST)
    blended = Image.blend(image.convert("RGBA"), overlay.convert("RGBA"), alpha=0.5)
    blended.save(output_path)
    print(f"[maskformer] Saved fallback overlay to {output_path}")


def _build_palette(num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(max(num_classes, 1), 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


def _prepare_inputs(args: argparse.Namespace):
    if Image is None:
        raise SystemExit("Image handling requires Pillow; install via `pip install pillow`.")

    try:
        from transformers import AutoImageProcessor
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("Image preprocessing requires transformers; install via `pip install transformers`.") from exc

    processor = AutoImageProcessor.from_pretrained(args.weights)
    image = Image.open(args.image).convert("RGB")
    proc_kwargs = {}
    if args.height is not None and args.width is not None:
        proc_kwargs.update({"size": {"height": int(args.height), "width": int(args.width)}, "do_resize": True})
    pixel_values = processor(images=image, return_tensors="pt", **proc_kwargs)["pixel_values"]
    return processor, image, pixel_values


def _emit_perf_header(path: Path, perf: Dict[str, object]) -> None:
    header = {
        "model": "maskformer-swin-base-coco",
        "mode": perf.get("mode", "unknown"),
        "device": perf.get("device", "unknown"),
        "dtype": perf.get("dtype", "auto"),
        "image_h": perf.get("image_size_hw", [None, None])[0],
        "image_w": perf.get("image_size_hw", [None, None])[1],
        "batch_size": 1,
        "num_queries": perf.get("num_queries", None),
        "latency_ms": perf.get("latency_ms", None),
        "latency_ms_std": perf.get("latency_ms_std", None),
        "fps": perf.get("fps", None),
        "tt_submodules": perf.get("tt_submodules", []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header, indent=2))


def _run_coco_eval(args: argparse.Namespace, ref_weights: ReferenceWeights, state_dict: Dict[str, object]) -> None:
    # Dependencies
    try:
        import pycocotools.mask as mask_utils  # noqa: F401
    except Exception:
        print("[maskformer] COCO eval requires 'pycocotools'. Install via `pip install pycocotools`.")
        return

    try:
        from transformers import AutoImageProcessor  # type: ignore
    except Exception as exc:
        print(f"[maskformer] transformers not available for COCO eval: {exc}")
        return

    have_panoptic = False
    try:
        import panopticapi  # type: ignore  # noqa: F401

        have_panoptic = True
    except Exception:
        have_panoptic = False

    images_dir = None
    pan_json = args.coco_panoptic_json
    pan_root = args.coco_panoptic_root
    if args.coco_dir and args.coco_dir.exists():
        candidate_images = args.coco_dir / "val2017"
        if candidate_images.exists():
            images_dir = candidate_images
        candidate_ann = args.coco_dir / "annotations" / "panoptic_val2017.json"
        candidate_root = args.coco_dir / "annotations" / "panoptic_val2017"
        if pan_json is None and candidate_ann.exists():
            pan_json = candidate_ann
        if pan_root is None and candidate_root.exists():
            pan_root = candidate_root

    if images_dir is None or not images_dir.exists():
        print("[maskformer] COCO images not found. Provide --coco-dir pointing to a COCO root with val2017/.")
        return

    if have_panoptic and (pan_json is None or pan_root is None):
        print(
            "[maskformer] panopticapi available but GT missing; continuing with mIoU only. "
            "Provide --coco-panoptic-json and --coco-panoptic-root for PQ."
        )
        have_panoptic = False

    # Build pipeline (TT vs CPU decided by --tt-run)
    tt_device = None
    try:
        if args.tt_run and ttnn is not None:
            tt_device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)
            os.environ["MASKFORMER_TT_DECODER"] = "1"
            os.environ.setdefault("MASKFORMER_TT_MASK_PROJ", "1")
    except Exception:
        tt_device = None

    pipeline = MaskFormerFallbackPipeline.from_reference(ref_weights, state_dict, device=tt_device)
    processor = AutoImageProcessor.from_pretrained(args.weights)

    image_list = sorted([p for p in images_dir.glob("*.jpg")])
    if args.coco_max_images:
        image_list = image_list[: int(args.coco_max_images)]
    if not image_list:
        print(f"[maskformer] No images found in {images_dir}")
        return

    device_label = str(args.device if args.tt_run else "cpu")
    report_path = args.coco_report
    if report_path is None and args.dump_perf:
        report_path = args.dump_perf.with_name(args.dump_perf.stem + "_coco_tt.json")
    report_path = report_path or Path("generated/coco_eval_tt.json")

    result = run_coco_eval(
        images=image_list,
        pipeline=pipeline,
        processor=processor,
        panoptic_json=pan_json if have_panoptic else None,
        panoptic_root=pan_root if have_panoptic else None,
        max_images=args.coco_max_images,
        device_label=device_label,
        report_path=report_path,
    )
    print(f"[maskformer] COCO eval → mIoU={result.miou} PQ={result.pq}; wrote {report_path}")

    if tt_device is not None:
        try:
            ttnn.close_device(tt_device)
        except Exception:
            pass


def _dtype_to_str(dtype_obj: Optional[object]) -> str:
    try:
        import builtins as _b
    except Exception:
        _b = None
    if dtype_obj is None:
        return "auto"
    name = str(dtype_obj)
    # Common TT-NN dtype string forms
    if hasattr(ttnn, "bfloat16") and dtype_obj == getattr(ttnn, "bfloat16"):
        return "bfloat16"
    if hasattr(ttnn, "float16") and dtype_obj == getattr(ttnn, "float16"):
        return "float16"
    if hasattr(ttnn, "float32") and dtype_obj == getattr(ttnn, "float32"):
        return "float32"
    # Fallback to best-effort parsing
    for key in ("bfloat16", "float16", "float32", "bf16", "fp16", "fp32"):
        if key in name:
            return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(key, key)
    return name


def _run_decoder_parity(
    args: argparse.Namespace,
    ref_weights: ReferenceWeights,
    state_dict: Dict[str, object],
) -> None:
    if Image is None:
        raise SystemExit("Decoder parity requires Pillow; install via `pip install pillow`.")

    processor, image, pixel_values = _prepare_inputs(args)

    # Open TT device and build pipeline with TT-capable decoder
    tt_device = None
    try:
        if ttnn is not None:
            tt_device = ttnn.open_device(device_id=0, l1_small_size=64 * 1024)
    except Exception:
        tt_device = None

    pipeline = MaskFormerFallbackPipeline.from_reference(ref_weights, state_dict, device=tt_device)

    # 1) HF reference decoder
    os.environ["MASKFORMER_TT_DECODER"] = "0"
    with torch.no_grad():
        ref_out = pipeline.forward(pixel_values, output_hidden_states=True, output_attentions=False)
    ref_last = ref_out.transformer_hidden_states[-1].detach().cpu().numpy().astype(np.float32)

    # 2) TT decoder
    os.environ["MASKFORMER_TT_DECODER"] = "1"
    with torch.no_grad():
        tt_out = pipeline.forward(pixel_values, output_hidden_states=True, output_attentions=False)
    tt_last = tt_out.transformer_hidden_states[-1].detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32)

    # 3) Compare (BF16 quantization respected via MASKFORMER_COMPARE_QUANTIZE)
    from . import parity as parity_mod

    pcc, max_abs = parity_mod.compare_tensors(
        ref_last,
        tt_last,
        pcc_threshold=parity_mod.ParityConfig().pcc_threshold,
        max_abs_threshold=parity_mod.ParityConfig().max_abs_threshold,
    )
    print(
        f"[maskformer] Decoder parity (last_hidden_state): shape={ref_last.shape} PCC={pcc:.6f} max_abs={max_abs:.3e}"
    )
    # Reset to avoid side effects
    os.environ["MASKFORMER_TT_DECODER"] = "0"
    if tt_device is not None:
        try:
            ttnn.close_device(tt_device)
        except Exception:
            pass


def _print_class_summary(outputs, ref_weights: ReferenceWeights, top_k: int = 5) -> None:
    logits = outputs.class_logits
    probs = torch.softmax(logits, dim=-1)
    scores, class_ids = torch.max(probs[..., :-1], dim=-1)
    scores = scores[0]
    class_ids = class_ids[0]
    if scores.numel() == 0:
        print("[maskformer] No decoder queries available for summary.")
        return
    top_k = min(top_k, scores.numel())
    top_scores, top_indices = torch.topk(scores, k=top_k)
    id2label = ref_weights.config.get("id2label", {}) if isinstance(ref_weights.config, dict) else {}

    def _lookup(idx: int) -> str:
        if isinstance(id2label, dict):
            return id2label.get(str(idx), id2label.get(idx, f"class_{idx}"))
        return f"class_{idx}"

    print("[maskformer] Top queries by confidence:")
    for rank, (score, query_idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
        class_id = int(class_ids[query_idx].item())
        label = _lookup(class_id)
        print(f"  #{rank:<2} query {query_idx:<3} → class {class_id:<3} ({label}) prob {score:.3f}")


if __name__ == "__main__":
    main()
