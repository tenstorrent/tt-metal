# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for evaluation outputs and reporting.

This module stays dependency-light so that unit tests can import it without a
TT runtime.  All heavy imports (pycocotools, panopticapi, PIL) are deferred to
call-sites inside runner hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional
import json

import torch


PredictionDict = Dict[str, Any]


@dataclass
class CocoEvalResult:
    dataset: str
    num_images: int
    miou: Optional[float]
    pq: Optional[float]
    device: str
    report_path: Optional[Path] = None


def _lookup_label(id2label: Mapping[str | int, str], class_id: int) -> str:
    """Resolve a class id to a readable label."""

    if isinstance(id2label, Mapping):
        if str(class_id) in id2label:
            return str(id2label[str(class_id)])
        if class_id in id2label:
            return str(id2label[class_id])
    return f"class_{class_id}"


def summarize_predictions(
    *,
    class_logits: torch.Tensor,
    id2label: Mapping[str | int, str],
    task_type: str = "instance",
) -> list[PredictionDict]:
    """Convert class logits to a compact prediction summary per query.

    Returns:
        List of dictionaries, one per query index, containing:
        mask_index, class_id, class_label, confidence, task_type.
    """

    probs = torch.softmax(class_logits, dim=-1)
    scores, class_ids = torch.max(probs[..., :-1], dim=-1)  # drop no-object
    records: list[PredictionDict] = []
    # Expect B=1; iterate defensively
    for b in range(scores.shape[0]):
        for q in range(scores.shape[1]):
            cid = int(class_ids[b, q].item())
            records.append(
                {
                    "mask_index": int(q),
                    "class_id": cid,
                    "class_label": _lookup_label(id2label, cid),
                    "confidence": float(scores[b, q].item()),
                    "task_type": task_type,
                }
            )
    return records


def dump_predictions_json(
    *,
    class_logits: torch.Tensor,
    id2label: Mapping[str | int, str],
    output_path: Path,
    task_type: str = "instance",
) -> Path:
    """Write per-query predictions to JSON and return the path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_type": task_type,
        "num_queries": int(class_logits.shape[1]),
        "predictions": summarize_predictions(class_logits=class_logits, id2label=id2label, task_type=task_type),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def default_prediction_path(dump_perf: Optional[Path], *, mode: str) -> Path:
    """Choose a default prediction path near perf JSON (or generated/)."""

    if dump_perf is not None:
        return dump_perf.with_name(dump_perf.stem + f"_{mode}_predictions.json")
    return Path("generated") / f"predictions_{mode}.json"


def write_coco_report(report_path: Path, result: CocoEvalResult) -> Path:
    """Serialize a CocoEvalResult to JSON."""

    payload: MutableMapping[str, Any] = {
        "dataset": result.dataset,
        "num_images": result.num_images,
        "miou": result.miou,
        "pq": result.pq,
        "device": result.device,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))
    return report_path


def run_coco_eval(
    *,
    images: Iterable[Path],
    pipeline,
    processor,
    panoptic_json: Optional[Path] = None,
    panoptic_root: Optional[Path] = None,
    max_images: int = 50,
    device_label: str = "cpu",
    report_path: Optional[Path] = None,
) -> CocoEvalResult:
    """Execute COCO semantic mIoU (and optional PQ) over a list of images.

    Args:
        images: iterable of image paths (e.g., val2017/*.jpg).
        pipeline: MaskFormerFallbackPipeline (CPU or TT) with forward/post-process methods.
        processor: transformers AutoImageProcessor matching the checkpoint.
        panoptic_json/root: enable PQ when provided and panopticapi is available.
        max_images: clamp evaluation size for quick smoke tests.
        device_label: string describing the execution device used.
        report_path: optional destination for a compact JSON report.
    """

    from PIL import Image  # local import to keep module lightweight
    import numpy as np

    try:
        import pycocotools.mask as _  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency optional
        raise RuntimeError("COCO eval requires pycocotools; install it to enable --coco-eval.") from exc

    have_panoptic = False
    if panoptic_json is not None and panoptic_root is not None:
        try:
            from panopticapi.utils import rgb2id, id2rgb  # type: ignore
            from panopticapi.evaluation import pq_compute  # type: ignore

            have_panoptic = True
        except Exception:
            have_panoptic = False

    image_list = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    image_list = sorted(image_list)[: int(max_images) if max_images else None]
    if not image_list:
        raise RuntimeError("No images provided for COCO evaluation.")

    intersections: Dict[int, int] = {}
    unions: Dict[int, int] = {}
    pred_dir = None
    pred_json = None
    pred_annotations = []
    pan_gt_by_file = {}

    if have_panoptic:
        import json as _json

        with panoptic_json.open("r", encoding="utf-8") as fh:
            gt_payload = _json.load(fh)
        ann_by_image_id = {ann["image_id"]: ann for ann in gt_payload.get("annotations", [])}
        file_to_image_id = {img["file_name"]: img["id"] for img in gt_payload.get("images", [])}
        pan_gt_by_file = {
            fn: ann_by_image_id[file_to_image_id[fn]]
            for fn in file_to_image_id
            if file_to_image_id[fn] in ann_by_image_id
        }
        pred_dir = (report_path or Path("generated/coco_eval_tt.json")).with_suffix("").with_name("panoptic_pred")
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_json = pred_dir / "pred.json"

    for img_path in image_list:
        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            outputs = pipeline.forward(pixel_values, output_hidden_states=False, output_attentions=False)
        pred_sem = pipeline.post_process_semantic(outputs, image_processor=processor, target_sizes=[image.size[::-1]])[
            0
        ]
        pred_sem = pred_sem.cpu().numpy().astype(np.int32)

        gt_sem = None
        if have_panoptic and img_path.name in pan_gt_by_file:
            gt_entry = pan_gt_by_file[img_path.name]
            gt_png = panoptic_root / gt_entry["file_name"]
            gt_seg = np.array(Image.open(gt_png), dtype=np.uint8)
            gt_seg = rgb2id(gt_seg)
            id_to_cat = {s["id"]: s["category_id"] for s in gt_entry.get("segments_info", [])}
            gt_sem = np.vectorize(lambda sid: id_to_cat.get(int(sid), 0), otypes=[np.int32])(gt_seg)

        if gt_sem is not None:
            classes = np.union1d(np.unique(gt_sem), np.unique(pred_sem)).astype(np.int64).tolist()
            for cid in classes:
                gt_mask = gt_sem == cid
                pr_mask = pred_sem == cid
                inter = int(np.logical_and(gt_mask, pr_mask).sum())
                uni = int(np.logical_or(gt_mask, pr_mask).sum())
                if uni == 0:
                    continue
                intersections[cid] = intersections.get(cid, 0) + inter
                unions[cid] = unions.get(cid, 0) + uni

        if have_panoptic and img_path.name in pan_gt_by_file:
            pan_pred = pipeline.post_process_panoptic(
                outputs, image_processor=processor, target_sizes=[image.size[::-1]]
            )[0]
            seg = pan_pred["segmentation"].cpu().numpy().astype(np.int32)
            seg_ids = np.unique(seg)
            rgb = id2rgb(seg)
            out_name = img_path.with_suffix(".png").name
            Image.fromarray(rgb).save(pred_dir / out_name)  # type: ignore[arg-type]
            segments_info = []
            for sid in seg_ids:
                area = int((seg == sid).sum())
                cat = 0
                for s in pan_pred.get("segments_info", []):
                    if int(s.get("id", -1)) == int(sid):
                        cat = int(s.get("category_id", 0))
                        break
                segments_info.append({"id": int(sid), "category_id": int(cat), "area": area, "iscrowd": 0})
            pred_annotations.append(
                {
                    "image_id": int(pan_gt_by_file[img_path.name]["image_id"]),
                    "file_name": out_name,
                    "segments_info": segments_info,
                }
            )

    miou = None
    if unions:
        ious = [intersections[c] / unions[c] for c in intersections if unions[c] > 0]
        miou = float(sum(ious) / len(ious)) if ious else None

    pq = None
    if have_panoptic and pred_dir is not None and pred_json is not None and pred_annotations:
        import json as _json
        from panopticapi.evaluation import pq_compute  # type: ignore

        with pred_json.open("w", encoding="utf-8") as fh:
            _json.dump({"annotations": pred_annotations}, fh)
        try:
            pq_res = pq_compute(
                gt_json_file=str(panoptic_json),
                gt_folder=str(panoptic_root),
                pred_json_file=str(pred_json),
                pred_folder=str(pred_dir),
            )
            pq = float(pq_res["All"]["pq"])  # type: ignore[index]
        except Exception:
            pq = None

    dataset_root = str(image_list[0].parent) if image_list else ""
    result = CocoEvalResult(
        dataset=dataset_root,
        num_images=len(image_list),
        miou=miou,
        pq=pq,
        device=device_label,
        report_path=report_path,
    )
    if report_path is not None:
        write_coco_report(report_path, result)
    return result
