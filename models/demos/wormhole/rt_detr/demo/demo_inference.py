# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

import ttnn

this_dir = Path(__file__).parent.resolve()
project = this_dir.parent
repo_path = project / "RT-DETR" / "rtdetr_pytorch"

sys.path.insert(0, str(repo_path))
sys.path.insert(0, str(project))

from src.core import YAMLConfig
from tt.hybrid_encoder import hybrid_encoder
from tt.resnet_backbone import presnet50
from tt.rtdetr_decoder import run_decoder
from tt.weight_utils import get_backbone_parameters, get_decoder_parameters, get_encoder_parameters, get_head_parameters

# COCO category ID mapping (80 classes, non-contiguous IDs)
coco_class_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

coco_class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

idx_to_name = {i: name for i, name in enumerate(coco_class_names)}

score_threshold = 0.4
input_size = (640, 640)
demo_images_dir = this_dir / "demo_images"
output_dir = this_dir / "demo_outputs"


def _to_device(tensor, device, nchw_to_nhwc=False, mem_config=ttnn.DRAM_MEMORY_CONFIG):
    t = tensor.permute(0, 2, 3, 1).contiguous() if nchw_to_nhwc else tensor.contiguous()
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _pull(tt_tensor, device):
    return ttnn.to_torch(
        tt_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0),
    )[0:1].float()


# Full TTNN forward pass


def run_ttnn_rtdetr_inference(img, torch_model, tt_params, device):
    """End-to-End TTNN forward pass."""

    # 1. TTNN Backbone
    t0 = time.perf_counter()
    x_tt = _to_device(img, device, nchw_to_nhwc=True)
    s3_tt, s4_tt, s5_tt = presnet50(x_tt, tt_params["backbone"], device)
    ttnn.deallocate(x_tt)
    t_backbone = time.perf_counter()

    # 2. TTNN Encoder
    p3_tt, p4_tt, p5_tt = hybrid_encoder(s3_tt, s4_tt, s5_tt, tt_params["encoder"], device)
    ttnn.deallocate(s3_tt)
    ttnn.deallocate(s4_tt)
    ttnn.deallocate(s5_tt)
    t_encoder = time.perf_counter()

    # 3. Pull and reshape for the Decoder handoff (PyTorch bridge)
    p3_pt = _pull(p3_tt, device).squeeze(1).reshape(1, 80, 80, 256).permute(0, 3, 1, 2)
    p4_pt = _pull(p4_tt, device).squeeze(1).reshape(1, 40, 40, 256).permute(0, 3, 1, 2)
    p5_pt = _pull(p5_tt, device).squeeze(1).reshape(1, 20, 20, 256).permute(0, 3, 1, 2)
    ttnn.deallocate(p3_tt)
    ttnn.deallocate(p4_tt)
    ttnn.deallocate(p5_tt)

    with torch.no_grad():
        # Apply encoder projection and get spatial shapes
        memory_pt, spatial_shapes, level_start_index = torch_model.decoder._get_encoder_input([p3_pt, p4_pt, p5_pt])

        spatial_shapes_tensor = torch.tensor(spatial_shapes)

        # Generate initial queries and reference points
        tgt, init_ref_unact, _, _ = torch_model.decoder._get_decoder_input(memory_pt, spatial_shapes_tensor)

    # 4. Convert queries to TTNN
    query_tt = _to_device(tgt.reshape(1, 1, 300, 256), device, mem_config=ttnn.L1_MEMORY_CONFIG)

    # 5. Run TTNN Decoder (memory_pt stays as PyTorch tensor)
    query_out, final_ref_points = run_decoder(
        query_tt,
        torch_decoder=torch_model.decoder,
        tt_layer_params=tt_params["decoder"],
        memory_torch=memory_pt,
        ref_points=init_ref_unact,
        spatial_shapes=spatial_shapes_tensor,
        device=device,
    )
    t_decoder = time.perf_counter()

    # 6. Pull final query output for prediction heads
    query_torch = _pull(query_out, device).view(1, 300, 256)
    ttnn.deallocate(query_out)

    # 7. Prediction heads
    with torch.no_grad():
        logits = torch_model.decoder.dec_score_head[-1](query_torch)
        boxes = final_ref_points.view(1, 300, 4)

    # 8. Clean up remaining TTNN tensors
    ttnn.deallocate(query_tt)
    t_end = time.perf_counter()

    timings = {
        "backbone_ms":  (t_backbone - t0)       * 1000,
        "encoder_ms":   (t_encoder  - t_backbone) * 1000,
        "decoder_ms":   (t_decoder  - t_encoder)  * 1000,
        "heads_ms":     (t_end      - t_decoder)  * 1000,
        "e2e_ms":       (t_end      - t0)         * 1000,
    }

    return logits, boxes, timings


# Visualisation


def draw_detections(image_path: Path, detections: list[dict], out_path: Path):
    """Draw bounding boxes with label + score on the original image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=14)
    except IOError:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, w, h = det["bbox"]
        x2, y2 = x1 + w, y1 + h
        label = det["label_name"]
        score = det["score"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        text = f"{label} {score:.2f}"

        bbox_text = draw.textbbox((x1, y1 - 16), text, font=font)
        draw.rectangle(bbox_text, fill="red")
        draw.text((x1, y1 - 16), text, fill="white", font=font)

    img.save(out_path)
    print(f"  Saved annotated image → {out_path}")


# Main


def main():
    config_path = project / "RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    ckpt_path = project / "weights/rtdetr_r50vd.pth"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in demo_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {demo_images_dir}")

    print(f"Found {len(image_paths)} demo image(s): {[p.name for p in image_paths]}")

    #  Device
    mesh_shape = ttnn.MeshShape(1, 2)
    device = ttnn.open_mesh_device(mesh_shape, l1_small_size=16384)

    try:
        #  Load PyTorch model
        print("\nLoading PyTorch model...")
        cfg = YAMLConfig(str(config_path))
        model = cfg.model
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["ema"]["module"])
        model.eval()
        print("  Model loaded and in eval mode.")

        #  Push weights to TTNN
        print("\nPushing weights to TTNN device...")
        tt_params = {
            "backbone": get_backbone_parameters(model, device),
            "encoder": get_encoder_parameters(model, device),
            "decoder": get_decoder_parameters(model, device),
            "head": get_head_parameters(model, device),
        }
        print("  Weights loaded successfully.")

        #  Image transform
        transform = T.Compose(
            [
                T.Resize(input_size),
                T.ToTensor(),
            ]
        )

        all_results = {}

        #  Per-image inference
        for img_path in image_paths:
            print(f"\n{'='*60}")
            print(f"Processing: {img_path.name}")

            orig_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = orig_img.size

            img_tensor = transform(orig_img).unsqueeze(0)

            with torch.no_grad():
                logits, boxes, timings = run_ttnn_rtdetr_inference(img_tensor, model, tt_params, device)

            # Post-process
            scores_all, labels_all = torch.max(logits[0].sigmoid(), dim=-1)
            keep = scores_all > score_threshold

            detections = []
            for score, label, box in zip(scores_all[keep], labels_all[keep], boxes[0][keep]):
                cx, cy, w, h = box.tolist()
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                w_abs = w * orig_w
                h_abs = h * orig_h
                cls_id = int(label.item())

                det = {
                    "label_idx": cls_id,
                    "label_name": idx_to_name.get(cls_id, f"cls_{cls_id}"),
                    "coco_cat_id": coco_class_ids[cls_id],
                    "score": round(float(score.item()), 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(w_abs, 2), round(h_abs, 2)],
                }
                detections.append(det)

            detections.sort(key=lambda d: d["score"], reverse=True)
            all_results[img_path.name] = detections

            # output in terminal
            print(f"  Detections ({len(detections)} above threshold {score_threshold}):")
            if detections:
                for det in detections:
                    print(
                        f"    [{det['label_name']:20s}]  "
                        f"score={det['score']:.4f}  "
                        f"bbox=[{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, "
                        f"{det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}]"
                    )
            else:
                print("    (no detections above threshold)")

            # Timing summary
            print(f"\n  Timing:")
            print(f"    backbone  : {timings['backbone_ms']:7.2f} ms")
            print(f"    encoder   : {timings['encoder_ms']:7.2f} ms")
            print(f"    decoder   : {timings['decoder_ms']:7.2f} ms")
            print(f"    heads     : {timings['heads_ms']:7.2f} ms")
            print(f"    {'─'*26}")
            print(f"    end-to-end: {timings['e2e_ms']:7.2f} ms  ({1000/timings['e2e_ms']:.1f} FPS)")

            # save
            out_img_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"
            draw_detections(img_path, detections, out_img_path)

            torch.cuda.empty_cache()

        #  Save combined JSON
        json_path = output_dir / "detections.json"
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved all detections → {json_path}")

    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()