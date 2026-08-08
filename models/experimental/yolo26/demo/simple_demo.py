# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Simple YOLO26 Demo - Compare TTNN vs PyTorch inference.

This demo:
1. Loads a sample image
2. Runs inference on both PyTorch and TTNN
3. Compares detection outputs
4. Prints results
"""

import torch
import ttnn
import numpy as np
from PIL import Image
import urllib.request
import os
from loguru import logger

from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE


def download_sample_image():
    """Download a sample image for testing."""
    url = "https://ultralytics.com/images/bus.jpg"
    filepath = "/tmp/bus.jpg"
    if not os.path.exists(filepath):
        logger.info(f"Downloading sample image from {url}")
        urllib.request.urlretrieve(url, filepath)
    return filepath


def preprocess_image(image_path, size=640):
    """Preprocess image for YOLO inference."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size))
    img_np = np.array(img).astype(np.float32) / 255.0
    # HWC -> CHW -> NCHW
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    return img_tensor


def to_nhwc(t, batch_size, h, w, ch):
    """Convert TTNN tensor to NHWC format."""
    if t.memory_config().is_sharded():
        t = ttnn.sharded_to_interleaved(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.reshape(t, [batch_size, h, w, ch])
    return t


def run_ttnn_inference(device, weight_loader, x_torch):
    """Run full YOLO26 inference on TTNN."""
    from models.experimental.yolo26.tt.ttnn_yolo26 import (
        TtConvBNSiLU,
        TtC2f,
        TtC3k2,
        TtSPPF,
        TtC2PSA,
        TtC3k2PSA,
        TtUpsample,
        TtYOLO26Head,
    )

    batch_size = 1
    input_size = 640

    # Create backbone layers
    backbone_layers = [
        TtConvBNSiLU(device, 3, 16, kernel_size=3, stride=2, padding=1, name="model.0"),
        TtConvBNSiLU(device, 16, 32, kernel_size=3, stride=2, padding=1, name="model.1"),
        TtC2f(device, 32, 64, hidden_channels=16, n=1, name="model.2"),
        TtConvBNSiLU(device, 64, 64, kernel_size=3, stride=2, padding=1, name="model.3"),
        TtC2f(device, 64, 128, hidden_channels=32, n=1, name="model.4"),
        TtConvBNSiLU(device, 128, 128, kernel_size=3, stride=2, padding=1, name="model.5"),
        TtC3k2(device, 128, 128, hidden_channels=64, n=1, name="model.6"),
        TtConvBNSiLU(device, 128, 256, kernel_size=3, stride=2, padding=1, name="model.7"),
        TtC3k2(device, 256, 256, hidden_channels=128, n=1, name="model.8"),
        TtSPPF(device, 256, 256, kernel_size=5, name="model.9"),
    ]

    for i, layer in enumerate(backbone_layers):
        if isinstance(layer, TtConvBNSiLU):
            w, b = weight_loader.get_conv_bn(f"model.{i}")
            layer.load_weights(w, b)
        else:
            layer.load_weights(weight_loader, f"model.{i}")

    # Neck layers
    c2psa_10 = TtC2PSA(device, 256, 256, n=1, name="model.10")
    c2psa_10.load_weights(weight_loader, "model.10")

    upsample = TtUpsample(scale_factor=2)

    c3k2_13 = TtC3k2(device, 384, 128, hidden_channels=64, n=1, name="model.13")
    c3k2_13.load_weights(weight_loader, "model.13")

    c3k2_16 = TtC3k2(device, 256, 64, hidden_channels=32, n=1, name="model.16")
    c3k2_16.load_weights(weight_loader, "model.16")

    conv_17 = TtConvBNSiLU(device, 64, 64, kernel_size=3, stride=2, padding=1, name="model.17")
    w, b = weight_loader.get_conv_bn("model.17")
    conv_17.load_weights(w, b)

    c3k2_19 = TtC3k2(device, 192, 128, hidden_channels=64, n=1, name="model.19")
    c3k2_19.load_weights(weight_loader, "model.19")

    conv_20 = TtConvBNSiLU(device, 128, 128, kernel_size=3, stride=2, padding=1, name="model.20")
    w, b = weight_loader.get_conv_bn("model.20")
    conv_20.load_weights(w, b)

    c3k2_22 = TtC3k2PSA(device, 384, 256, hidden_channels=128, n=1, name="model.22")
    c3k2_22.load_weights(weight_loader, "model.22")

    # Detection head
    detect_head = TtYOLO26Head(device, "yolo26n", num_classes=80)
    detect_head.load_weights(weight_loader)

    # Forward pass
    x_nhwc = x_torch.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)
    tt_x = ttnn.from_torch(x_nhwc, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_intermediates = {}
    h, w = input_size, input_size
    out_channels = [16, 32, 64, 64, 128, 128, 128, 256, 256, 256]

    # Backbone
    for i, layer in enumerate(backbone_layers):
        tt_x, h, w = layer(tt_x, batch_size, h, w)
        tt_x_conv = to_nhwc(tt_x, batch_size, h, w, out_channels[i])
        tt_intermediates[i] = (ttnn.to_torch(tt_x_conv), h, w, out_channels[i])
        tt_x = ttnn.from_torch(tt_intermediates[i][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Neck
    tt_x, h, w = c2psa_10(tt_x, batch_size, h, w)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_intermediates[10] = (ttnn.to_torch(tt_x_conv), h, w, 256)

    tt_x = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = upsample(tt_x, batch_size, 20, 20, 256)
    tt_x6 = ttnn.from_torch(tt_intermediates[6][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x6], dim=3)

    tt_x, h, w = c3k2_13(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_intermediates[13] = (ttnn.to_torch(tt_x_conv), h, w, 128)

    tt_x = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = upsample(tt_x, batch_size, 40, 40, 128)
    tt_x4 = ttnn.from_torch(tt_intermediates[4][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x4], dim=3)

    tt_x, h, w = c3k2_16(tt_x, batch_size, 80, 80)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_n3 = ttnn.to_torch(tt_x_conv)
    tt_intermediates[16] = (tt_n3, 80, 80, 64)

    tt_x = ttnn.from_torch(tt_n3, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = conv_17(tt_x, batch_size, 80, 80)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 64)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x13 = ttnn.from_torch(tt_intermediates[13][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x13], dim=3)

    tt_x, h, w = c3k2_19(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_n4 = ttnn.to_torch(tt_x_conv)
    tt_intermediates[19] = (tt_n4, 40, 40, 128)

    tt_x = ttnn.from_torch(tt_n4, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x, h, w = conv_20(tt_x, batch_size, 40, 40)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 128)
    tt_x = ttnn.from_torch(ttnn.to_torch(tt_x_conv), dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x10 = ttnn.from_torch(tt_intermediates[10][0], dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.concat([tt_x, tt_x10], dim=3)

    tt_x, h, w = c3k2_22(tt_x, batch_size, 20, 20)
    tt_x_conv = to_nhwc(tt_x, batch_size, h, w, 256)
    tt_n5 = ttnn.to_torch(tt_x_conv)

    # Detection head
    n3_tensor = ttnn.from_torch(tt_n3, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    n4_tensor = ttnn.from_torch(tt_n4, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    n5_tensor = ttnn.from_torch(tt_n5, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_detect_out = detect_head((n3_tensor, 80, 80), (n4_tensor, 40, 40), (n5_tensor, 20, 20), batch_size)

    return tt_n3, tt_n4, tt_n5, tt_detect_out


def postprocess_detections(boxes_list, cls_list, conf_thresh=0.25, img_size=640):
    """
    Post-process raw detection outputs to get bounding boxes.

    YOLO26 one2one head outputs:
    - boxes: [B, H, W, 4] where 4 = [left, top, right, bottom] distances from anchor
    - cls: [B, H, W, 80] class logits (need sigmoid)

    Decoding (dist2bbox):
    - x1 = anchor_x - left
    - y1 = anchor_y - top
    - x2 = anchor_x + right
    - y2 = anchor_y + bottom

    NO complex DFL decoding needed! Just simple anchor-based conversion.
    """
    detections = []
    strides = [8, 16, 32]  # N3, N4, N5 strides

    for scale_idx, (boxes, cls, stride) in enumerate(zip(boxes_list, cls_list, strides)):
        h, w = boxes.shape[1], boxes.shape[2]

        # Get class scores and find max
        cls_scores = torch.sigmoid(cls[0])  # [H, W, 80]
        max_scores, max_cls = cls_scores.max(dim=-1)  # [H, W]

        # Find positions above threshold
        mask = max_scores > conf_thresh

        for y in range(h):
            for x in range(w):
                if mask[y, x]:
                    conf = max_scores[y, x].item()
                    cls_id = max_cls[y, x].item()

                    # Anchor point (center of grid cell)
                    anchor_x = (x + 0.5) * stride
                    anchor_y = (y + 0.5) * stride

                    # Raw box = [left, top, right, bottom] distances
                    box_raw = boxes[0, y, x]  # [4]
                    left = box_raw[0].item() * stride
                    top = box_raw[1].item() * stride
                    right = box_raw[2].item() * stride
                    bottom = box_raw[3].item() * stride

                    # dist2bbox: convert distances to coordinates
                    x1 = max(0, anchor_x - left)
                    y1 = max(0, anchor_y - top)
                    x2 = min(img_size, anchor_x + right)
                    y2 = min(img_size, anchor_y + bottom)

                    detections.append({"box": [x1, y1, x2, y2], "class": cls_id, "conf": conf})

    # Sort by confidence
    detections.sort(key=lambda x: x["conf"], reverse=True)
    return detections[:10]  # Return top 10


def draw_detections(image_path, detections, class_names, output_path, title="Detections"):
    """Draw bounding boxes on image and save."""
    from PIL import ImageDraw

    img = Image.open(image_path).convert("RGB")
    img = img.resize((640, 640))
    draw = ImageDraw.Draw(img)

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for i, det in enumerate(detections[:5]):
        box = det["box"]
        cls_name = class_names[det["class"]]
        conf = det["conf"]
        color = colors[i % len(colors)]

        # Draw box
        draw.rectangle(box, outline=color, width=3)

        # Draw label
        label = f"{cls_name}: {conf:.1%}"
        draw.text((box[0], box[1] - 15), label, fill=color)

    # Add title
    draw.text((10, 10), title, fill=(255, 255, 255))

    img.save(output_path)
    return output_path


def main():
    """Run the demo."""
    from ultralytics import YOLO
    from models.experimental.yolo26.tt.model_preprocessing import YOLO26WeightLoader

    logger.info("=" * 60)
    logger.info("YOLO26 TTNN Demo - Full Model Inference")
    logger.info("=" * 60)

    # Load PyTorch model
    logger.info("\n1. Loading YOLO26n model...")
    torch_model = YOLO("yolo26n.pt")
    state_dict = torch_model.model.state_dict()
    weight_loader = YOLO26WeightLoader(state_dict)
    class_names = torch_model.names

    # Download and preprocess image
    logger.info("\n2. Loading sample image...")
    image_path = download_sample_image()
    x_torch = preprocess_image(image_path, size=640)
    logger.info(f"   Input shape: {x_torch.shape}")

    # PyTorch inference
    logger.info("\n3. Running PyTorch inference...")
    with torch.no_grad():
        pt_results = torch_model(image_path, verbose=False)

    pt_boxes = pt_results[0].boxes
    logger.info(f"   PyTorch detections: {len(pt_boxes)} objects")

    # Get intermediate outputs for comparison
    pt_intermediates = {}
    with torch.no_grad():
        x_pt = x_torch.float()
        for i in range(23):
            layer = torch_model.model.model[i]
            if layer.__class__.__name__ == "Concat":
                tensors = [pt_intermediates[idx] if idx != -1 else x_pt for idx in layer.f]
                x_pt = torch.cat(tensors, dim=1)
            else:
                x_pt = layer(x_pt)
            pt_intermediates[i] = x_pt.clone()

    pt_n3 = pt_intermediates[16]
    pt_n4 = pt_intermediates[19]
    pt_n5 = pt_intermediates[22]

    # Get PyTorch detection head raw outputs for comparison
    detect = torch_model.model.model[23]
    with torch.no_grad():
        pt_box_raw = [detect.one2one_cv2[i]([pt_n3, pt_n4, pt_n5][i]) for i in range(3)]
        pt_cls_raw = [detect.one2one_cv3[i]([pt_n3, pt_n4, pt_n5][i]) for i in range(3)]

    # TTNN inference
    logger.info("\n4. Running TTNN inference...")
    device = ttnn.open_device(device_id=0, l1_small_size=YOLO26_L1_SMALL_SIZE)

    try:
        tt_n3, tt_n4, tt_n5, tt_detect_out = run_ttnn_inference(device, weight_loader, x_torch)

        # Extract TTNN detection outputs
        tt_box_list = []
        tt_cls_list = []
        scale_dims = [(80, 80), (40, 40), (20, 20)]

        for i, (h, w) in enumerate(scale_dims):
            tt_bbox, _, _ = tt_detect_out["boxes"][i]
            tt_cls, _, _ = tt_detect_out["scores"][i]

            tt_bbox_nhwc = to_nhwc(tt_bbox, 1, h, w, 4)
            tt_cls_nhwc = to_nhwc(tt_cls, 1, h, w, 80)

            tt_box_list.append(ttnn.to_torch(tt_bbox_nhwc).float())
            tt_cls_list.append(ttnn.to_torch(tt_cls_nhwc).float())

        # Compare outputs
        logger.info("\n5. Comparing Detection Head Outputs...")

        def calc_pcc(pt, tt):
            pt_flat = pt.flatten()
            tt_flat = tt.flatten()
            return torch.corrcoef(torch.stack([pt_flat, tt_flat]))[0, 1].item()

        # Neck PCC
        def calc_pcc_nchw_nhwc(pt_nchw, tt_nhwc):
            pt_nhwc = pt_nchw.permute(0, 2, 3, 1).contiguous().flatten()
            tt_flat = tt_nhwc.float().flatten()
            return torch.corrcoef(torch.stack([pt_nhwc, tt_flat]))[0, 1].item()

        pcc_n3 = calc_pcc_nchw_nhwc(pt_n3, tt_n3)
        pcc_n4 = calc_pcc_nchw_nhwc(pt_n4, tt_n4)
        pcc_n5 = calc_pcc_nchw_nhwc(pt_n5, tt_n5)

        logger.info(f"   Backbone+Neck PCC:")
        logger.info(f"      N3 (80x80): {pcc_n3:.4f}")
        logger.info(f"      N4 (40x40): {pcc_n4:.4f}")
        logger.info(f"      N5 (20x20): {pcc_n5:.4f}")

        # Detection head PCC
        box_pccs = []
        cls_pccs = []
        for i in range(3):
            pt_box_nhwc = pt_box_raw[i].permute(0, 2, 3, 1).contiguous()
            pt_cls_nhwc = pt_cls_raw[i].permute(0, 2, 3, 1).contiguous()

            box_pcc = calc_pcc(pt_box_nhwc, tt_box_list[i])
            cls_pcc = calc_pcc(pt_cls_nhwc, tt_cls_list[i])
            box_pccs.append(box_pcc)
            cls_pccs.append(cls_pcc)

        logger.info(f"\n   Detection Head PCC:")
        logger.info(f"      Box (all scales): {sum(box_pccs)/3:.4f}")
        logger.info(f"      Cls (all scales): {sum(cls_pccs)/3:.4f}")

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DETECTION RESULTS COMPARISON")
        logger.info("=" * 60)

        logger.info(f"\nImage: {image_path}")

        # PyTorch detections
        logger.info(f"\n--- PyTorch Detections ({len(pt_boxes)} objects) ---")
        for i, box in enumerate(pt_boxes[:5]):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = class_names[cls_id]
            xyxy = box.xyxy[0].tolist()
            logger.info(
                f"   {i+1}. {cls_name}: {conf:.1%} @ [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]"
            )

        # TTNN detections (post-processed)
        tt_detections = postprocess_detections(tt_box_list, tt_cls_list, conf_thresh=0.25)
        logger.info(f"\n--- TTNN Detections ({len(tt_detections)} objects above 25% conf) ---")
        for i, det in enumerate(tt_detections[:5]):
            cls_name = class_names[det["class"]]
            conf = det["conf"]
            box = det["box"]
            logger.info(f"   {i+1}. {cls_name}: {conf:.1%} @ [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

        # Save output images
        logger.info("\n" + "=" * 60)
        logger.info("OUTPUT IMAGES")
        logger.info("=" * 60)

        # Get original image size for scaling PyTorch boxes
        orig_img = Image.open(image_path)
        orig_w, orig_h = orig_img.size
        scale_x = 640 / orig_w
        scale_y = 640 / orig_h

        # Output directory (same as demo script)
        demo_dir = os.path.dirname(os.path.abspath(__file__))

        # PyTorch output - scale boxes to 640x640
        pt_detections = []
        for box in pt_boxes[:5]:
            orig_box = box.xyxy[0].tolist()
            scaled_box = [orig_box[0] * scale_x, orig_box[1] * scale_y, orig_box[2] * scale_x, orig_box[3] * scale_y]
            pt_detections.append({"box": scaled_box, "class": int(box.cls[0]), "conf": float(box.conf[0])})
        pt_output_path = os.path.join(demo_dir, "output_pytorch.jpg")
        pt_output = draw_detections(image_path, pt_detections, class_names, pt_output_path, "PyTorch Detections")
        logger.info(f"   PyTorch output: {pt_output}")

        # TTNN output
        tt_output_path = os.path.join(demo_dir, "output_ttnn.jpg")
        tt_output = draw_detections(image_path, tt_detections, class_names, tt_output_path, "TTNN Detections")
        logger.info(f"   TTNN output:    {tt_output}")

        logger.info("\n" + "=" * 60)
        logger.info("ALL OPERATIONS RUNNING ON TTNN ✅")
        logger.info("=" * 60)

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
