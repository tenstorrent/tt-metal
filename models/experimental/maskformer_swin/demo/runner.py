# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MaskFormer Swin-B TTNN demo runner."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import torch

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None

from ..tt.backbone_swin import MaskFormerSwinBackbone
from ..tt.heads import MaskFormerHeads, MaskFormerHeadsConfig
from ..tt.pixel_decoder import MaskFormerPixelDecoder, PixelDecoderConfig
from ..tt.transformer_decoder import MaskFormerTransformerDecoder, TransformerDecoderConfig
from ..tt.ttnn_compat import require_ttnn
from ..tt.weights import (
    WeightConversionConfig,
    convert_state_dict_to_tt,
    download_reference_weights,
    resolve_hf_cache_dir,
    resolve_hf_token,
)

_OPT_STAGE_ENV = {
    "stage1": {
        "MASKFORMER_TT_FUSE_LINEAR_ACT": "0",
        "MASKFORMER_TT_USE_LINEAR": "0",
        "MASKFORMER_TT_ENABLE_SDPA": "0",
        "MASKFORMER_TT_ENABLE_FUSED_QKV": "0",
        "MASKFORMER_TT_ENABLE_L1_SEQ": "0",
        "MASKFORMER_TT_DISABLE_CORE_GRID": "1",
        "MASKFORMER_TT_DISABLE_MATMUL_PC": "1",
        "MASKFORMER_TT_DISABLE_DECODER_TT_CACHE": "1",
        "MASKFORMER_TT_RETURN_TT_DECODER": "0",
    },
    "stage2": {
        "MASKFORMER_TT_FUSE_LINEAR_ACT": "1",
        "MASKFORMER_TT_USE_LINEAR": "1",
        "MASKFORMER_TT_ENABLE_SDPA": "0",
        "MASKFORMER_TT_ENABLE_FUSED_QKV": "0",
        "MASKFORMER_TT_ENABLE_L1_SEQ": "1",
        "MASKFORMER_TT_DISABLE_CORE_GRID": "0",
        "MASKFORMER_TT_DISABLE_MATMUL_PC": "0",
        "MASKFORMER_TT_DISABLE_DECODER_TT_CACHE": "0",
        "MASKFORMER_TT_RETURN_TT_DECODER": "0",
    },
    "stage3": {
        "MASKFORMER_TT_FUSE_LINEAR_ACT": "1",
        "MASKFORMER_TT_USE_LINEAR": "1",
        "MASKFORMER_TT_ENABLE_SDPA": "1",
        "MASKFORMER_TT_ENABLE_FUSED_QKV": "1",
        "MASKFORMER_TT_ENABLE_L1_SEQ": "1",
        "MASKFORMER_TT_DISABLE_CORE_GRID": "0",
        "MASKFORMER_TT_DISABLE_MATMUL_PC": "0",
        "MASKFORMER_TT_DISABLE_DECODER_TT_CACHE": "0",
        "MASKFORMER_TT_RETURN_TT_DECODER": "1",
    },
}


@dataclass
class InferenceOutputs:
    class_logits: torch.Tensor
    mask_logits: torch.Tensor


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MaskFormer Swin-B TTNN demo")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument(
        "--weights",
        type=str,
        default="facebook/maskformer-swin-base-coco",
        help="Hugging Face model id or local checkpoint directory.",
    )
    parser.add_argument("--device", type=str, default="wormhole_n300", help="Target Tenstorrent device label.")
    parser.add_argument(
        "--l1-small-size",
        type=int,
        default=32768,
        help="Device L1 small allocation size; lower values can reduce CB/L1 clashes.",
    )
    parser.add_argument("--height", type=int, default=None, help="Resize height (e.g. 320).")
    parser.add_argument("--width", type=int, default=None, help="Resize width (e.g. 320).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--optimization-stage",
        choices=("stage1", "stage2", "stage3"),
        default="stage1",
        help="Optimization profile to apply via runtime env flags.",
    )

    parser.add_argument("--save", type=Path, default=None, help="Optional semantic overlay PNG path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for demo artifacts.")
    parser.add_argument(
        "--topk-instance-masks",
        type=int,
        default=8,
        help="Top-K binary instance masks to export under output-dir.",
    )

    parser.add_argument("--dump-perf", type=Path, default=None, help="Path to write perf JSON.")
    parser.add_argument("--dump-perf-header", type=Path, default=None, help="Path to write perf header JSON.")
    parser.add_argument("--tt-repeats", type=int, default=1, help="Number of timed TT passes (after warmup).")

    parser.add_argument("--coco-eval", action="store_true", help="Run COCO eval (expects panoptic annotations).")
    parser.add_argument("--coco-dir", type=Path, default=None, help="COCO root with val2017/ and annotations/.")
    parser.add_argument("--coco-max-images", type=int, default=50, help="Max images to evaluate.")
    parser.add_argument("--coco-report", type=Path, default=None, help="Path to write COCO report JSON.")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    if Image is None:
        raise SystemExit("This demo requires Pillow; install via `pip install pillow`.")

    _set_seed(args.seed)
    opt_env = _configure_optimization_stage(args.optimization_stage)

    processor, image, pixel_values = _prepare_inputs(args)

    print(f"[maskformer] Resolving weights for '{args.weights}' ...")
    weight_cfg = WeightConversionConfig(pretrained_model_name=args.weights)
    ref_weights = download_reference_weights(weight_cfg)
    tt_state_dict = convert_state_dict_to_tt(ref_weights.state_dict, weight_cfg)
    id2label = _normalize_id2label(ref_weights.config.get("id2label", {}))

    tt = require_ttnn("run MaskFormer Swin-B on device")
    device = None
    try:
        open_kwargs = {"device_id": 0}
        if args.l1_small_size is not None:
            open_kwargs["l1_small_size"] = int(args.l1_small_size)
        try:
            device = tt.open_device(**open_kwargs)
        except TypeError:
            device = tt.open_device(device_id=0)
        runner = _build_runner(ref_weights.config, tt_state_dict, device=device)

        outputs, perf = _timed_inference(
            runner=runner,
            pixel_values=pixel_values,
            repeats=max(int(args.tt_repeats), 1),
            device=device,
            perf_device_label=str(args.device),
            image_hw=_resolve_image_hw(args, pixel_values),
            optimization_stage=args.optimization_stage,
            optimization_env=opt_env,
        )
        _print_prediction_summary(outputs.class_logits, id2label)

        if args.output_dir is not None:
            _save_demo_artifacts(
                output_dir=args.output_dir,
                processor=processor,
                image=image,
                outputs=outputs,
                id2label=id2label,
                topk_instance_masks=max(int(args.topk_instance_masks), 0),
            )

        if args.save is not None:
            _save_overlay(args.save, processor, image, outputs)

        if args.dump_perf is not None or args.dump_perf_header is not None:
            if args.dump_perf is not None:
                args.dump_perf.parent.mkdir(parents=True, exist_ok=True)
                args.dump_perf.write_text(json.dumps(perf, indent=2))

            header_path = (
                args.dump_perf_header
                if args.dump_perf_header is not None
                else args.dump_perf.with_name(args.dump_perf.stem + "_header.json")
            )
            assert header_path is not None
            _emit_perf_header(header_path, perf)
            if args.dump_perf is not None:
                print(f"[maskformer] Wrote perf to {args.dump_perf} and header to {header_path}")
            else:
                print(f"[maskformer] Wrote perf header to {header_path}")

        if args.coco_eval:
            if args.coco_dir is None:
                raise SystemExit("--coco-eval requires --coco-dir")

            report_path = args.coco_report
            if report_path is None and args.output_dir is not None:
                report_path = args.output_dir / "coco_eval" / "report.json"

            _run_coco_eval(
                coco_dir=args.coco_dir,
                max_images=int(args.coco_max_images),
                report_path=report_path,
                processor=processor,
                forward_fn=lambda pv: runner.forward(pv, device=device),
                sync_fn=(lambda: tt.synchronize_device(device)) if hasattr(tt, "synchronize_device") else None,
                device_label=str(args.device),
            )
    finally:
        if device is not None:
            try:
                tt.close_device(device)
            except Exception:
                pass


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_optimization_stage(stage: str) -> Dict[str, str]:
    settings = dict(_OPT_STAGE_ENV[stage])
    for key, value in settings.items():
        os.environ[key] = value
    print(f"[maskformer] Optimization stage={stage} with env flags: {json.dumps(settings, sort_keys=True)}")
    return settings


def _prepare_inputs(args: argparse.Namespace):
    try:
        from transformers import AutoImageProcessor
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("Image preprocessing requires transformers; install via `pip install transformers`.") from exc

    hf_kwargs: Dict[str, object] = {}
    token = resolve_hf_token()
    if token:
        hf_kwargs["token"] = token
    cache_dir = resolve_hf_cache_dir(None)
    if cache_dir is not None:
        hf_kwargs["cache_dir"] = str(cache_dir)

    processor = AutoImageProcessor.from_pretrained(args.weights, **hf_kwargs)
    image = Image.open(args.image).convert("RGB")

    proc_kwargs = {}
    if args.height is not None and args.width is not None:
        proc_kwargs.update({"size": {"height": int(args.height), "width": int(args.width)}, "do_resize": True})
    pixel_values = processor(images=image, return_tensors="pt", **proc_kwargs)["pixel_values"]
    return processor, image, pixel_values


def _resolve_image_hw(args: argparse.Namespace, pixel_values: torch.Tensor) -> list[int]:
    if args.height is not None and args.width is not None:
        return [int(args.height), int(args.width)]
    return [int(pixel_values.shape[2]), int(pixel_values.shape[3])]


class _MaskFormerRunner:
    def __init__(
        self,
        *,
        backbone: MaskFormerSwinBackbone,
        pixel_decoder: MaskFormerPixelDecoder,
        transformer_decoder: MaskFormerTransformerDecoder,
        heads: MaskFormerHeads,
        decoder_return_tt: bool,
    ) -> None:
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder
        self.heads = heads
        self.decoder_return_tt = decoder_return_tt

    def forward(self, pixel_values: torch.Tensor, *, device: object) -> InferenceOutputs:
        _ = device
        features, _ = self.backbone.forward(pixel_values)
        mask_features, _ = self.pixel_decoder.forward(features)
        decoder_last, _, _ = self.transformer_decoder.forward_tt(features[-1], return_tt_tensor=self.decoder_return_tt)
        class_logits, mask_logits = self.heads.forward(decoder_last, mask_features)
        return InferenceOutputs(class_logits=class_logits, mask_logits=mask_logits)


def _build_runner(ref_config: Dict[str, object], state_dict: Dict[str, object], *, device: object) -> _MaskFormerRunner:
    backbone_cfg = ref_config.get("backbone_config", {}) if isinstance(ref_config, dict) else {}
    decoder_cfg = ref_config.get("decoder_config", {}) if isinstance(ref_config, dict) else {}

    backbone = MaskFormerSwinBackbone.from_huggingface(state_dict, device=device, config_dict=backbone_cfg)

    pixel_cfg = PixelDecoderConfig(
        fpn_dim=int(ref_config.get("fpn_feature_size", 256)),
        mask_dim=int(ref_config.get("mask_feature_size", 256)),
    )
    pixel_decoder = MaskFormerPixelDecoder.from_huggingface(state_dict, config=pixel_cfg, device=device)

    transformer_cfg = TransformerDecoderConfig(
        num_layers=int(ref_config.get("num_hidden_layers", 6)),
        num_attention_heads=int(ref_config.get("num_attention_heads", 8)),
        hidden_dim=int(ref_config.get("fpn_feature_size", 256)),
        dim_feedforward=int(decoder_cfg.get("decoder_ffn_dim", 2048)) if isinstance(decoder_cfg, dict) else 2048,
        dropout=float(decoder_cfg.get("dropout", 0.0)) if isinstance(decoder_cfg, dict) else 0.0,
        activation=str(decoder_cfg.get("activation_function", "relu")) if isinstance(decoder_cfg, dict) else "relu",
        in_features=pixel_cfg.input_channels[-1],
        maskformer_config=dict(ref_config) if isinstance(ref_config, dict) else None,
    )
    transformer_decoder = MaskFormerTransformerDecoder.from_huggingface(
        state_dict, config=transformer_cfg, device=device
    )

    heads_cfg = MaskFormerHeadsConfig(
        num_classes=len(ref_config.get("id2label", {})),
        hidden_dim=transformer_cfg.hidden_dim,
        mask_dim=pixel_cfg.fpn_dim,
    )
    heads = MaskFormerHeads(config=heads_cfg, device=device)
    heads.load_weights(state_dict)

    decoder_return_tt = os.environ.get("MASKFORMER_TT_RETURN_TT_DECODER", "1") == "1"
    return _MaskFormerRunner(
        backbone=backbone,
        pixel_decoder=pixel_decoder,
        transformer_decoder=transformer_decoder,
        heads=heads,
        decoder_return_tt=decoder_return_tt,
    )


def _timed_inference(
    *,
    runner: _MaskFormerRunner,
    pixel_values: torch.Tensor,
    repeats: int,
    device: object,
    perf_device_label: str,
    image_hw: list[int],
    optimization_stage: str,
    optimization_env: Dict[str, str],
) -> Tuple[InferenceOutputs, Dict[str, object]]:
    tt = require_ttnn("sync device")

    with torch.no_grad():
        _ = runner.forward(pixel_values, device=device)  # warmup
        if hasattr(tt, "synchronize_device"):
            tt.synchronize_device(device)

        latencies_ms = []
        outputs = None
        for _ in range(repeats):
            start = time.perf_counter()
            outputs = runner.forward(pixel_values, device=device)
            if hasattr(tt, "synchronize_device"):
                tt.synchronize_device(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    assert outputs is not None
    latency_ms = float(statistics.mean(latencies_ms)) if latencies_ms else float("nan")
    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    latency_std = float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0
    print(
        f"[maskformer] TT run latency: mean={latency_ms:.2f} ms std={latency_std:.2f} ms "
        f"({fps:.2f} FPS over {len(latencies_ms)} runs)"
    )

    perf_payload = {
        "mode": "tt_run",
        "device": perf_device_label,
        "dtype": "bfloat16",
        "latency_ms": latency_ms,
        "latency_ms_repeats": [float(x) for x in latencies_ms],
        "latency_ms_std": latency_std if latencies_ms else None,
        "fps": fps,
        "image_size_hw": image_hw,
        "num_queries": int(outputs.class_logits.shape[1]),
        "tt_submodules": ["backbone", "pixel_decoder", "decoder", "heads"],
        "optimization_stage": optimization_stage,
        "optimization_env": dict(optimization_env),
    }
    return outputs, perf_payload


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
        "optimization_stage": perf.get("optimization_stage", "unknown"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header, indent=2))


def _print_prediction_summary(class_logits: torch.Tensor, id2label: Mapping[int, str], top_k: int = 5) -> None:
    probs = torch.softmax(class_logits, dim=-1)
    scores, class_ids = torch.max(probs[..., :-1], dim=-1)  # drop no-object
    flat_scores = scores.flatten()
    if flat_scores.numel() == 0:
        return
    top = min(int(top_k), int(flat_scores.numel()))
    best = torch.topk(flat_scores, k=top)
    print("[maskformer] Top predictions (query, class, score):")
    for idx, score in zip(best.indices.tolist(), best.values.tolist()):
        q = idx % int(scores.shape[1])
        cid = int(class_ids[0, q].item())
        label = id2label.get(cid, f"class_{cid}")
        print(f"  q={q:3d} class={cid:3d} ({label}) score={float(score):.3f}")


def _make_hf_output(outputs: InferenceOutputs):
    try:
        from transformers.models.maskformer.modeling_maskformer import MaskFormerForInstanceSegmentationOutput
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("Post-processing requires transformers; install via `pip install transformers`.") from exc
    return MaskFormerForInstanceSegmentationOutput(
        class_queries_logits=outputs.class_logits,
        masks_queries_logits=outputs.mask_logits,
    )


def _save_overlay(output_path: Path, processor, image, outputs: InferenceOutputs) -> None:
    mf_output = _make_hf_output(outputs)
    segmentation = processor.post_process_semantic_segmentation(mf_output, target_sizes=[image.size[::-1]])[0]
    segmentation = segmentation.cpu().to(torch.int64)
    num_classes = int(outputs.class_logits.shape[-1])
    palette = _build_palette(num_classes + 1)
    overlay = _segmentation_to_image(segmentation, palette).resize(image.size, Image.NEAREST)
    blended = Image.blend(image.convert("RGBA"), overlay.convert("RGBA"), alpha=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    blended.save(output_path)
    print(f"[maskformer] Saved overlay to {output_path}")


def _save_demo_artifacts(
    *,
    output_dir: Path,
    processor,
    image,
    outputs: InferenceOutputs,
    id2label: Mapping[int, str],
    topk_instance_masks: int,
) -> None:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit("Demo artifact export requires numpy; install via `pip install numpy`.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    mf_output = _make_hf_output(outputs)
    target_size = [image.size[::-1]]

    semantic = processor.post_process_semantic_segmentation(mf_output, target_sizes=target_size)[0]
    semantic = semantic.cpu().to(torch.int64)
    palette = _build_palette(int(outputs.class_logits.shape[-1]) + 1)
    semantic_color = _segmentation_to_image(semantic, palette).resize(image.size, Image.NEAREST)
    semantic_overlay = Image.blend(image.convert("RGBA"), semantic_color.convert("RGBA"), alpha=0.5)
    semantic_overlay_path = output_dir / "semantic_overlay.png"
    semantic_overlay.save(semantic_overlay_path)

    panoptic = processor.post_process_panoptic_segmentation(mf_output, target_sizes=target_size)[0]
    pan_seg = panoptic["segmentation"].cpu().to(torch.int64)
    pan_palette = _build_palette(max(int(pan_seg.max().item()) + 1, 1))
    panoptic_path = output_dir / "panoptic_segmentation.png"
    _segmentation_to_image(pan_seg, pan_palette).save(panoptic_path)

    segments_json = []
    for seg in panoptic.get("segments_info", []):
        seg_id = int(seg.get("id", -1))
        label_id = int(seg.get("label_id", seg.get("category_id", -1)))
        area = int((pan_seg == seg_id).sum().item()) if seg_id >= 0 else 0
        segments_json.append(
            {
                "segment_id": seg_id,
                "category_id": label_id,
                "label": id2label.get(label_id, f"class_{label_id}"),
                "score": float(seg["score"]) if seg.get("score") is not None else None,
                "area": area,
                "isthing": bool(seg["isthing"]) if "isthing" in seg else None,
                "was_fused": bool(seg.get("was_fused", False)),
            }
        )

    segments_payload = {
        "image": str(image.filename) if getattr(image, "filename", None) else None,
        "height": int(image.size[1]),
        "width": int(image.size[0]),
        "segments": segments_json,
    }
    segments_json_path = output_dir / "panoptic_segments.json"
    segments_json_path.write_text(json.dumps(segments_payload, indent=2))

    instance_manifest = []
    if topk_instance_masks > 0:
        inst = processor.post_process_instance_segmentation(
            mf_output,
            target_sizes=target_size,
            threshold=0.0,
            return_binary_maps=True,
        )[0]
        instance_masks = inst.get("segmentation", None)
        segments = list(inst.get("segments_info", []))
        segments.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

        masks_dir = output_dir / "instance_masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for rank, seg in enumerate(segments[:topk_instance_masks]):
            seg_id = int(seg.get("id", -1))
            if instance_masks is None or seg_id < 0 or seg_id >= int(instance_masks.shape[0]):
                continue
            label_id = int(seg.get("label_id", -1))
            label = id2label.get(label_id, f"class_{label_id}")
            score = float(seg.get("score", 0.0))
            mask = instance_masks[seg_id].detach().cpu().to(torch.uint8).numpy() * 255
            mask_path = masks_dir / f"{rank:03d}_{_safe_name(label)}_{score:.4f}.png"
            Image.fromarray(mask.astype(np.uint8), mode="L").save(mask_path)
            instance_manifest.append(
                {
                    "rank": rank,
                    "segment_id": seg_id,
                    "category_id": label_id,
                    "label": label,
                    "score": score,
                    "mask_path": str(mask_path),
                }
            )
        (output_dir / "instance_masks.json").write_text(json.dumps(instance_manifest, indent=2))

    print(f"[maskformer] Saved semantic overlay: {semantic_overlay_path}")
    print(f"[maskformer] Saved panoptic map: {panoptic_path}")
    print(f"[maskformer] Saved panoptic segments JSON: {segments_json_path}")
    if instance_manifest:
        print(f"[maskformer] Saved {len(instance_manifest)} top instance masks: {output_dir / 'instance_masks'}")


def _segmentation_to_image(segmentation: torch.Tensor, palette: list[tuple[int, int, int]]) -> Image.Image:
    height, width = int(segmentation.shape[0]), int(segmentation.shape[1])
    flat = segmentation.flatten().tolist()
    colors = [palette[int(idx) % len(palette)] for idx in flat]
    img = Image.new("RGB", (width, height))
    img.putdata(colors)
    return img


def _build_palette(num_classes: int) -> list[tuple[int, int, int]]:
    rng = random.Random(0)
    size = max(int(num_classes), 1)
    palette = [(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size)]
    palette[0] = (0, 0, 0)
    return palette


def _normalize_id2label(id2label: Mapping[object, object]) -> Dict[int, str]:
    normalized: Dict[int, str] = {}
    for key, value in id2label.items():
        try:
            normalized[int(key)] = str(value)
        except Exception:
            continue
    return normalized


def _safe_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._-") or "label"


def _run_coco_eval(
    *,
    coco_dir: Path,
    max_images: int,
    report_path: Optional[Path],
    processor,
    forward_fn,
    sync_fn,
    device_label: str,
) -> None:
    images_dir = coco_dir / "val2017"
    panoptic_json = coco_dir / "annotations" / "panoptic_val2017.json"
    panoptic_root = coco_dir / "annotations" / "panoptic_val2017"
    if not images_dir.exists():
        raise SystemExit(f"COCO images not found: {images_dir}")
    if not panoptic_json.exists() or not panoptic_root.exists():
        raise SystemExit("COCO panoptic GT not found under coco_dir/annotations; required for --coco-eval.")

    from PIL import Image as _Image

    try:
        import numpy as _np
    except ModuleNotFoundError as exc:
        raise SystemExit("COCO eval requires numpy; install via `pip install numpy`.") from exc

    try:
        from panopticapi.utils import rgb2id, id2rgb  # type: ignore

        have_panoptic = True
    except Exception:
        have_panoptic = False
        rgb2id = None
        id2rgb = None

    with panoptic_json.open("r", encoding="utf-8") as fh:
        gt_payload = json.load(fh)
    ann_by_image_id = {ann["image_id"]: ann for ann in gt_payload.get("annotations", [])}
    file_to_image_id = {img["file_name"]: img["id"] for img in gt_payload.get("images", [])}
    pan_gt_by_file = {
        fn: ann_by_image_id[file_to_image_id[fn]] for fn in file_to_image_id if file_to_image_id[fn] in ann_by_image_id
    }

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])[:max_images]
    intersections: Dict[int, int] = {}
    unions: Dict[int, int] = {}
    inference_latencies_ms = []
    num_with_gt = 0

    pred_dir = None
    pred_json = None
    pred_annotations = []
    if report_path is not None and have_panoptic:
        pred_dir = report_path.parent / "panoptic_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_json = report_path.parent / "panoptic_predictions.json"

    for img_path in images:
        image = _Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]

        start = time.perf_counter()
        outputs = forward_fn(pixel_values)
        if sync_fn is not None:
            sync_fn()
        inference_latencies_ms.append((time.perf_counter() - start) * 1000.0)

        mf_output = _make_hf_output(outputs)
        pred_sem = processor.post_process_semantic_segmentation(mf_output, target_sizes=[image.size[::-1]])[0]
        pred_sem = pred_sem.cpu().numpy().astype(_np.int32)

        gt_entry = pan_gt_by_file.get(img_path.name)
        if gt_entry is None:
            continue
        num_with_gt += 1

        if have_panoptic and rgb2id is not None:
            gt_png = panoptic_root / gt_entry["file_name"]
            gt_seg = _np.array(_Image.open(gt_png), dtype=_np.uint8)
            gt_seg = rgb2id(gt_seg)
            id_to_cat = {int(s["id"]): int(s["category_id"]) for s in gt_entry.get("segments_info", [])}
            gt_sem = _np.vectorize(lambda sid: id_to_cat.get(int(sid), 0), otypes=[_np.int32])(gt_seg)
        else:
            gt_sem = _np.zeros_like(pred_sem)

        classes = _np.union1d(_np.unique(gt_sem), _np.unique(pred_sem)).astype(_np.int64).tolist()
        for cid in classes:
            gt_mask = gt_sem == cid
            pr_mask = pred_sem == cid
            inter = int(_np.logical_and(gt_mask, pr_mask).sum())
            uni = int(_np.logical_or(gt_mask, pr_mask).sum())
            if uni == 0:
                continue
            intersections[cid] = intersections.get(cid, 0) + inter
            unions[cid] = unions.get(cid, 0) + uni

        if have_panoptic and pred_dir is not None and pred_json is not None and id2rgb is not None:
            pan_pred = processor.post_process_panoptic_segmentation(mf_output, target_sizes=[image.size[::-1]])[0]
            seg = pan_pred["segmentation"].cpu().numpy().astype(_np.int32)
            rgb = id2rgb(seg)
            out_name = img_path.with_suffix(".png").name
            _Image.fromarray(rgb).save(pred_dir / out_name)  # type: ignore[arg-type]

            segments_info = []
            for seg_meta in pan_pred.get("segments_info", []):
                seg_id = int(seg_meta.get("id", -1))
                label_id = int(seg_meta.get("label_id", seg_meta.get("category_id", 0)))
                area = int((seg == seg_id).sum()) if seg_id >= 0 else 0
                segments_info.append({"id": seg_id, "category_id": label_id, "area": area, "iscrowd": 0})
            pred_annotations.append(
                {"image_id": int(gt_entry["image_id"]), "file_name": out_name, "segments_info": segments_info}
            )

    miou = None
    if unions:
        ious = [intersections[c] / unions[c] for c in intersections if unions[c] > 0]
        miou = float(sum(ious) / len(ious)) if ious else None

    pq = None
    if have_panoptic and pred_dir is not None and pred_json is not None and pred_annotations:
        try:
            from panopticapi.evaluation import pq_compute  # type: ignore

            pred_json.write_text(json.dumps({"annotations": pred_annotations}, indent=2))
            pq_res = pq_compute(
                gt_json_file=str(panoptic_json),
                gt_folder=str(panoptic_root),
                pred_json_file=str(pred_json),
                pred_folder=str(pred_dir),
            )
            pq = float(pq_res["All"]["pq"])  # type: ignore[index]
        except Exception:
            pq = None

    avg_latency_ms = float(sum(inference_latencies_ms) / len(inference_latencies_ms)) if inference_latencies_ms else None
    throughput = (1000.0 / avg_latency_ms) if avg_latency_ms and avg_latency_ms > 0 else None

    result = {
        "dataset": str(images_dir),
        "num_images_requested": int(max_images),
        "num_images_evaluated": int(len(images)),
        "num_images_with_gt": int(num_with_gt),
        "panopticapi_available": bool(have_panoptic),
        "miou": miou,
        "pq": pq,
        "avg_inference_latency_ms": avg_latency_ms,
        "images_per_second": throughput,
        "device": device_label,
    }
    print("[maskformer] COCO eval:", json.dumps(result, indent=2))
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2))
        print(f"[maskformer] Wrote COCO report to {report_path}")


if __name__ == "__main__":
    main()
