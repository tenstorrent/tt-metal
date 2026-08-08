# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load DINO config+checkpoint and run optional inference.")
    parser.add_argument(
        "--config",
        default="checkpoints/dino_5scale_swin_l/dino_5scale_swin_l.py",
        help="DINO config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/dino_5scale_swin_l/dino_5scale_swin_l.pth",
        help="DINO checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device (e.g. cuda:0 or cpu).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional image path or directory. If omitted, script only checks model loading.",
    )
    parser.add_argument("--ext", default="jpg,jpeg,png", help="Extensions when --input is a directory.")
    parser.add_argument("--score-thr", type=float, default=0.15, help="Score threshold for saved predictions.")
    parser.add_argument("--output-dir", default="results/standalone_dino", help="Output directory.")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization images.")
    parser.add_argument("--dry-run", action="store_true", help="Load model and exit.")
    return parser.parse_args()


def parse_exts(exts: str) -> List[str]:
    out = []
    for e in exts.split(","):
        e = e.strip().lower()
        if not e:
            continue
        out.append(e if e.startswith(".") else f".{e}")
    return out


def check_file(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_file():
        raise SystemExit(f"Missing file: {p}")
    return p


def load_model(cfg_path: Path, ckpt_path: Path, device: str):
    try:
        from mmdet.apis import init_detector
    except Exception as exc:
        raise SystemExit(f"Failed to import MMDetection APIs: {exc}. " "Use an env with mmengine/mmcv/mmdet installed.")
    print(f"[load] DINO: cfg={cfg_path} ckpt={ckpt_path} device={device}")
    return init_detector(str(cfg_path), str(ckpt_path), device=device)


def get_class_names(model) -> Sequence[str]:
    meta = getattr(model, "dataset_meta", None)
    if isinstance(meta, dict) and meta.get("classes") is not None:
        return list(meta["classes"])
    classes = getattr(model, "CLASSES", None)
    if classes is not None:
        return list(classes)
    return []


def run_inference(model, image_bgr, score_thr: float) -> List[Dict]:
    from mmdet.apis import inference_detector

    data_sample = inference_detector(model, image_bgr)
    pred = data_sample.pred_instances
    if pred is None or len(pred) == 0:
        return []

    class_names = get_class_names(model)
    bboxes = pred.bboxes.detach().cpu().numpy()
    scores = pred.scores.detach().cpu().numpy()
    labels = pred.labels.detach().cpu().numpy()

    out = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < score_thr:
            continue
        label_id = int(labels[i])
        label = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        out.append(
            {
                "label": str(label),
                "label_id": label_id,
                "score": score,
                "bbox": [float(x) for x in bboxes[i].tolist()],
            }
        )
    out.sort(key=lambda x: (-x["score"], x["label"], x["bbox"][0], x["bbox"][1]))
    return out


def draw_predictions(image_bgr, preds: List[Dict], title: str):
    vis = image_bgr.copy()
    for p in preds:
        x1, y1, x2, y2 = [int(round(v)) for v in p["bbox"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{p['label']} {p['score']:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(vis, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def collect_images(input_path: Path, exts: List[str]) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        allowed = set(exts)
        return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in allowed])
    raise SystemExit(f"--input not found: {input_path}")


def main() -> None:
    args = parse_args()
    if args.score_thr < 0:
        raise SystemExit("--score-thr must be >= 0")

    cfg = check_file(args.config)
    ckpt = check_file(args.checkpoint)
    model = load_model(cfg, ckpt, args.device)
    print("[ok] DINO loaded successfully.")

    if args.dry_run:
        return
    if args.input is None:
        print("[info] No --input provided. Load-only check completed.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(Path(args.input), parse_exts(args.ext))
    if not image_paths:
        raise SystemExit("No input images found.")

    all_results = []
    for img_path in tqdm(image_paths, desc="Running DINO"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warn] Could not read: {img_path}")
            continue
        h, w = img.shape[:2]
        preds = run_inference(model, img, score_thr=float(args.score_thr))
        all_results.append(
            {
                "file": img_path.name,
                "path": str(img_path),
                "width": int(w),
                "height": int(h),
                "predictions": preds,
            }
        )
        if args.save_vis:
            vis = draw_predictions(img, preds, "DINO")
            cv2.imwrite(str(vis_dir / f"{img_path.stem}.dino.jpg"), vis)

    json_path = output_dir / "predictions_dino.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"[ok] Processed images: {len(all_results)}")
    print(f"[ok] Wrote predictions: {json_path}")
    if args.save_vis:
        print(f"[ok] Wrote visualizations: {vis_dir}")


if __name__ == "__main__":
    main()
