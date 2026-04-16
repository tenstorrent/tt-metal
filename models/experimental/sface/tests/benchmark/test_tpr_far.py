# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TPR@FAR Benchmark for SFace Model (Full Pipeline: YuNet + SFace)

This script evaluates the TTNN SFace model against the LFW (Labeled Faces in the Wild)
benchmark dataset to measure True Positive Rate at various False Acceptance Rates.

Uses the SAME pipeline as the live demo:
  Image → YuNet (detection + keypoints) → Align face → SFace → Embedding

Customer Requirement: >98% TPR @ 0.1% FAR

Usage:
    # Full LFW benchmark (uses scikit-learn to download)
    python -m models.experimental.sface.tests.benchmark.test_tpr_far

    # Quick test with registered demo faces (no download)
    python -m models.experimental.sface.tests.benchmark.test_tpr_far --quick

The script will:
1. Download LFW pairs dataset via scikit-learn (has working mirrors)
2. Load YuNet (face detection) and SFace (recognition) models
3. Run full pipeline: detect → align → embed for all pairs
4. Calculate ROC curve and report TPR at FAR=0.1%, 0.01%, 1%
"""

import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

import torch
import ttnn

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"

# Global models
yunet_model = None


def setup_lfw_dataset_sklearn():
    """
    Download LFW pairs dataset using scikit-learn (has working mirrors).
    Returns the pairs data directly as numpy arrays.
    """
    try:
        from sklearn.datasets import fetch_lfw_pairs
    except ImportError:
        logger.error("scikit-learn not installed. Install with: pip install scikit-learn")
        return None, None

    logger.info("Downloading LFW pairs via scikit-learn (this may take a few minutes)...")
    logger.info("  Source: scikit-learn mirrors (more reliable than UMass server)")

    # Fetch the pairs dataset - scikit-learn handles download/caching
    # subset='test' gives us 1000 pairs (500 positive, 500 negative)
    # subset='10_folds' gives all 6000 pairs for full benchmark
    lfw_pairs = fetch_lfw_pairs(subset="test", color=True, resize=1.0, data_home=str(DATA_DIR))

    # lfw_pairs.pairs shape: (n_pairs, 2, H, W, C)
    # lfw_pairs.target: 1 = same person, 0 = different
    pairs_images = lfw_pairs.pairs  # Shape: (n_pairs, 2, 62, 47, 3) or similar
    labels = lfw_pairs.target

    logger.info(f"Loaded {len(labels)} pairs ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")

    return pairs_images, labels


def setup_celeba_dataset(num_pairs: int = 1000):
    """
    Download CelebA dataset using TensorFlow Datasets and create verification pairs.
    CelebA has higher resolution (178x218) and includes landmarks.

    Returns:
        pairs_images: numpy array of shape (num_pairs, 2, H, W, C)
        labels: numpy array of shape (num_pairs,) - 1 for same person, 0 for different
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        logger.error("tensorflow-datasets not installed. Install with: pip install tensorflow-datasets")
        return None, None

    logger.info("Loading CelebA dataset via TensorFlow Datasets...")
    logger.info("  (This may take a while on first run - downloading ~1.4GB)")

    # Load CelebA with identity labels
    ds = tfds.load("celeb_a", split="test", data_dir=str(DATA_DIR), with_info=False)

    # Group images by identity
    identity_to_images = {}
    logger.info("Grouping images by identity...")

    for example in tfds.as_numpy(ds):
        identity = example["attributes"]["identity"] if "identity" in example["attributes"] else None
        if identity is None:
            continue

        image = example["image"]  # Shape: (218, 178, 3), values 0-255

        if identity not in identity_to_images:
            identity_to_images[identity] = []

        # Store image and landmarks if available
        identity_to_images[identity].append(image)

    logger.info(f"Found {len(identity_to_images)} identities")

    # Filter identities with at least 2 images (for positive pairs)
    multi_image_identities = {k: v for k, v in identity_to_images.items() if len(v) >= 2}
    logger.info(f"Identities with 2+ images: {len(multi_image_identities)}")

    if len(multi_image_identities) < 10:
        logger.error("Not enough identities with multiple images")
        return None, None

    # Create pairs
    import random

    random.seed(42)

    pairs_list = []
    labels_list = []

    identity_list = list(multi_image_identities.keys())

    # Create positive pairs (same identity)
    num_positive = num_pairs // 2
    for _ in range(num_positive):
        identity = random.choice(identity_list)
        images = multi_image_identities[identity]
        if len(images) >= 2:
            idx1, idx2 = random.sample(range(len(images)), 2)
            pairs_list.append([images[idx1], images[idx2]])
            labels_list.append(1)

    # Create negative pairs (different identities)
    num_negative = num_pairs - len(labels_list)
    for _ in range(num_negative):
        id1, id2 = random.sample(identity_list, 2)
        img1 = random.choice(multi_image_identities[id1])
        img2 = random.choice(multi_image_identities[id2])
        pairs_list.append([img1, img2])
        labels_list.append(0)

    # Convert to numpy arrays
    pairs_images = np.array(pairs_list)
    labels = np.array(labels_list)

    logger.info(f"Created {len(labels)} pairs ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")
    logger.info(f"Image resolution: {pairs_images.shape[2]}x{pairs_images.shape[3]}")

    return pairs_images, labels


def load_yunet_model(device):
    """Load YuNet face detection model (same as server)."""
    global yunet_model

    from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model
    from models.experimental.yunet.common import (
        load_torch_model as load_yunet_torch_model,
        get_default_weights_path as get_yunet_weights_path,
    )

    weights_path = get_yunet_weights_path()
    logger.info(f"Loading YuNet model from {weights_path}")

    torch_model = load_yunet_torch_model(weights_path)
    torch_model = torch_model.to(torch.bfloat16)
    yunet_model = create_yunet_model(device, torch_model)

    # Warmup
    logger.info("Warming up YuNet...")
    dummy_input = torch.randn(1, 640, 640, 3, dtype=torch.bfloat16)
    dummy_ttnn = ttnn.from_torch(dummy_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = yunet_model(dummy_ttnn)

    return yunet_model


def decode_yunet_detections(
    cls_outs, box_outs, obj_outs, kpt_outs, input_size: int = 640, threshold: float = 0.5
) -> List[dict]:
    """Decode YuNet outputs to face detections (same as server)."""
    from models.experimental.yunet.common import STRIDES, DEFAULT_NMS_IOU_THRESHOLD

    detections = []

    for scale_idx in range(3):
        cls_out = ttnn.to_torch(cls_outs[scale_idx]).float().permute(0, 3, 1, 2)
        box_out = ttnn.to_torch(box_outs[scale_idx]).float().permute(0, 3, 1, 2)
        obj_out = ttnn.to_torch(obj_outs[scale_idx]).float().permute(0, 3, 1, 2)
        kpt_out = ttnn.to_torch(kpt_outs[scale_idx]).float().permute(0, 3, 1, 2)

        stride = STRIDES[scale_idx]
        score = cls_out.sigmoid() * obj_out.sigmoid()

        high_conf = score > threshold
        if high_conf.any():
            indices = torch.where(high_conf)
            for i in range(len(indices[0])):
                b, c, h, w = indices[0][i], indices[1][i], indices[2][i], indices[3][i]
                conf = score[b, c, h, w].item()
                anchor_x, anchor_y = w.item() * stride, h.item() * stride

                dx, dy = box_out[b, 0, h, w].item(), box_out[b, 1, h, w].item()
                dw, dh = box_out[b, 2, h, w].item(), box_out[b, 3, h, w].item()

                cx, cy = dx * stride + anchor_x, dy * stride + anchor_y
                bw, bh = np.exp(dw) * stride, np.exp(dh) * stride

                # Normalized coordinates (0-1)
                x1 = (cx - bw / 2) / input_size
                y1 = (cy - bh / 2) / input_size
                x2 = (cx + bw / 2) / input_size
                y2 = (cy + bh / 2) / input_size

                keypoints = []
                for k in range(5):
                    kpt_dx = kpt_out[b, k * 2, h, w].item()
                    kpt_dy = kpt_out[b, k * 2 + 1, h, w].item()
                    kx = (kpt_dx * stride + anchor_x) / input_size
                    ky = (kpt_dy * stride + anchor_y) / input_size
                    keypoints.append([kx, ky])

                detections.append({"box": [x1, y1, x2, y2], "conf": conf, "keypoints": keypoints})

    # NMS
    detections = sorted(detections, key=lambda x: x["conf"], reverse=True)
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        remaining = []
        for det in detections:
            bx1 = max(best["box"][0], det["box"][0])
            by1 = max(best["box"][1], det["box"][1])
            bx2 = min(best["box"][2], det["box"][2])
            by2 = min(best["box"][3], det["box"][3])
            inter = max(0, bx2 - bx1) * max(0, by2 - by1)
            area1 = (best["box"][2] - best["box"][0]) * (best["box"][3] - best["box"][1])
            area2 = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])
            if inter / max(area1 + area2 - inter, 1e-6) < DEFAULT_NMS_IOU_THRESHOLD:
                remaining.append(det)
        detections = remaining

    return keep


def align_face_keypoints(image: np.ndarray, keypoints: List, target_size: int = 112) -> np.ndarray:
    """
    Align face using 5-point keypoints (same as server).
    Standard alignment: left_eye, right_eye, nose, left_mouth, right_mouth
    """
    if keypoints is None or len(keypoints) < 5:
        # Fallback: just resize
        img = Image.fromarray(image.astype(np.uint8))
        return np.array(img.resize((target_size, target_size), Image.BILINEAR))

    # Standard destination points for 112x112 aligned face
    dst_pts = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041],  # right mouth
        ],
        dtype=np.float32,
    )

    # Source points from keypoints
    src_pts = np.array(keypoints[:5], dtype=np.float32)

    # Estimate similarity transform
    tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

    if tform is None:
        img = Image.fromarray(image.astype(np.uint8))
        return np.array(img.resize((target_size, target_size), Image.BILINEAR))

    # Apply transform
    aligned = cv2.warpAffine(image.astype(np.uint8), tform, (target_size, target_size))

    return aligned


def detect_and_align_face(image: np.ndarray, device, input_size: int = 640) -> Optional[np.ndarray]:
    """
    Run full detection + alignment pipeline on an image.
    Returns aligned 112x112 face or None if no face detected.
    """
    global yunet_model

    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Resize to YuNet input size
    h, w = image.shape[:2]
    img_resized = cv2.resize(image, (input_size, input_size))

    # Run YuNet (returns 4 outputs: cls, box, obj, kpt)
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float()
    img_ttnn = ttnn.from_torch(img_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cls_out, box_out, obj_out, kpt_out = yunet_model(img_ttnn)

    # Decode detections
    detections = decode_yunet_detections(cls_out, box_out, obj_out, kpt_out, input_size, threshold=0.3)

    if not detections:
        # No face detected - fallback to simple resize
        return None

    # Get best detection - keypoints are normalized (0-1), convert to pixel coords
    best = detections[0]
    keypoints_norm = best.get("keypoints")

    # Convert normalized keypoints to pixel coordinates
    keypoints = [[kp[0] * input_size, kp[1] * input_size] for kp in keypoints_norm]

    # Align using keypoints on the resized image
    aligned = align_face_keypoints(img_resized, keypoints, target_size=112)

    return aligned


def create_quick_test_pairs():
    """
    Create test pairs from registered demo faces for quick testing.
    Returns (pairs_images, labels) in same format as sklearn loader.
    """
    demo_dir = Path(__file__).parent.parent.parent / "web_demo" / "server" / "registered_faces"

    if not demo_dir.exists():
        logger.error(f"Demo faces directory not found: {demo_dir}")
        return None, None

    # Find all face images
    faces = []
    for person_dir in demo_dir.iterdir():
        if person_dir.is_dir():
            face_img = person_dir / "face.jpg"
            if face_img.exists():
                img = np.array(Image.open(face_img).convert("RGB"))
                faces.append((person_dir.name, img))

    if len(faces) < 2:
        logger.error("Need at least 2 registered faces for quick test")
        return None, None

    pairs_list = []
    labels_list = []

    # Create positive pairs (same image compared to itself - should be ~1.0)
    for name, img in faces:
        pairs_list.append([img, img])
        labels_list.append(1)

    # Create negative pairs (different people)
    for i, (name1, img1) in enumerate(faces):
        for j, (name2, img2) in enumerate(faces):
            if i < j:
                pairs_list.append([img1, img2])
                labels_list.append(0)

    pairs_images = np.array(pairs_list)
    labels = np.array(labels_list)

    logger.info(f"Created {len(labels)} test pairs from {len(faces)} registered faces")
    logger.info(f"  Positive (self-comparison): {sum(labels)}")
    logger.info(f"  Negative (cross-comparison): {len(labels) - sum(labels)}")
    logger.info("\n⚠️  Note: This is a QUICK TEST, not a proper benchmark.")
    logger.info("    For accurate TPR@FAR, use the full LFW dataset.\n")

    return pairs_images, labels


def preprocess_face_for_sface(image: np.ndarray, target_size: int = 112) -> np.ndarray:
    """
    Preprocess a face image for SFace.

    Args:
        image: Input image as numpy array (H, W, C)
               - Can be 0-1 float (sklearn LFW) or 0-255 uint8
        target_size: Target size for SFace input (112x112)

    Returns:
        Preprocessed image as numpy array (target_size, target_size, C) with values 0-255
    """
    # Handle 0-1 float range (sklearn LFW format)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    img = Image.fromarray(image)

    # Resize to SFace input size (handles non-square images like 125x94)
    img = img.resize((target_size, target_size), Image.BILINEAR)

    return np.array(img)


def load_ttnn_sface_model(device):
    """Load the TTNN SFace model."""
    from models.experimental.sface.tt.ttnn_sface import create_sface_model
    from models.experimental.sface.reference.sface_model import load_sface_from_onnx
    from models.experimental.sface.common import get_sface_onnx_path

    onnx_path = get_sface_onnx_path()
    logger.info(f"Loading SFace model from {onnx_path}")

    # Load PyTorch reference model
    pytorch_model = load_sface_from_onnx(onnx_path)

    # Create TTNN model using factory function (same as server)
    ttnn_model = create_sface_model(device, pytorch_model)

    # Warmup (NHWC format like server)
    logger.info("Warming up model...")
    dummy_input = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)  # NHWC
    dummy_ttnn = ttnn.from_torch(dummy_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _ = ttnn_model(dummy_ttnn)

    return ttnn_model


def compute_embedding(model, image: np.ndarray, device) -> np.ndarray:
    """Compute embedding for a single face image (same as server)."""
    # Convert to tensor - NHWC format for TTNN (same as server)
    img_tensor = torch.from_numpy(image).unsqueeze(0).float()  # [1, 112, 112, 3] NHWC

    # Convert to TTNN
    img_ttnn = ttnn.from_torch(img_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # Run inference
    output = model(img_ttnn)

    # Convert back to numpy
    embedding = ttnn.to_torch(output).float().numpy().flatten()

    return embedding


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    # Embeddings should already be L2 normalized
    return float(np.dot(emb1, emb2))


def compute_tpr_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float) -> Tuple[float, float]:
    """
    Compute TPR at a specific FAR.

    Args:
        scores: Similarity scores for all pairs
        labels: Ground truth labels (1=same, 0=different)
        target_far: Target False Acceptance Rate

    Returns:
        (tpr, threshold): TPR at target FAR and the corresponding threshold
    """
    # Sort thresholds
    thresholds = np.sort(np.unique(scores))[::-1]  # High to low

    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    num_positive = len(positive_scores)
    num_negative = len(negative_scores)

    best_tpr = 0.0
    best_threshold = 0.0

    for thresh in thresholds:
        # FAR = FP / N (false positives among negatives)
        fp = np.sum(negative_scores >= thresh)
        far = fp / num_negative

        # TPR = TP / P (true positives among positives)
        tp = np.sum(positive_scores >= thresh)
        tpr = tp / num_positive

        if far <= target_far:
            if tpr > best_tpr:
                best_tpr = tpr
                best_threshold = thresh

    return best_tpr, best_threshold


def compute_roc_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Compute comprehensive ROC metrics."""
    metrics = {}

    # TPR at various FAR levels
    for far in [0.001, 0.01, 0.1]:  # 0.1%, 1%, 10%
        tpr, thresh = compute_tpr_at_far(scores, labels, far)
        metrics[f"TPR@FAR={far*100:.1f}%"] = tpr
        metrics[f"Threshold@FAR={far*100:.1f}%"] = thresh

    # Overall accuracy at optimal threshold (EER point approximation)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    # Find threshold that maximizes accuracy
    thresholds = np.linspace(scores.min(), scores.max(), 1000)
    best_acc = 0
    best_thresh = 0

    for thresh in thresholds:
        tp = np.sum(positive_scores >= thresh)
        tn = np.sum(negative_scores < thresh)
        acc = (tp + tn) / len(scores)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    metrics["Best_Accuracy"] = best_acc
    metrics["Best_Threshold"] = best_thresh

    # Mean scores
    metrics["Mean_Positive_Score"] = float(np.mean(positive_scores))
    metrics["Mean_Negative_Score"] = float(np.mean(negative_scores))
    metrics["Score_Separation"] = metrics["Mean_Positive_Score"] - metrics["Mean_Negative_Score"]

    return metrics


def run_benchmark(dataset: str = "lfw", num_pairs: int = 1000):
    """
    Run the full TPR@FAR benchmark.

    Args:
        dataset: "lfw", "celeba", or "quick"
        num_pairs: Number of pairs for CelebA (default 1000)
    """
    logger.info("=" * 60)
    logger.info("SFace TPR@FAR Benchmark (Full Pipeline: YuNet + SFace)")
    logger.info("=" * 60)

    # Setup dataset
    logger.info(f"\n[1/4] Loading dataset: {dataset.upper()}...")

    if dataset == "celeba":
        try:
            pairs_images, labels = setup_celeba_dataset(num_pairs)
        except Exception as e:
            logger.error(f"Failed to load CelebA: {e}")
            logger.info("\nInstall tensorflow-datasets: pip install tensorflow-datasets")
            return None
    elif dataset == "lfw":
        try:
            pairs_images, labels = setup_lfw_dataset_sklearn()
        except Exception as e:
            logger.error(f"Failed to download LFW: {e}")
            logger.info("\nFailed to download. Try: --quick for quick test using registered faces")
            return None
    else:  # quick
        logger.info("Quick test mode - using registered demo faces...")
        pairs_images, labels = create_quick_test_pairs()

    if pairs_images is None or labels is None:
        logger.error("No pairs to evaluate. Exiting.")
        return None

    # Initialize TTNN device with proper L1 size (same as server)
    from models.experimental.sface.common import SFACE_L1_SMALL_SIZE
    from models.experimental.yunet.common import YUNET_L1_SMALL_SIZE

    l1_size = max(SFACE_L1_SMALL_SIZE, YUNET_L1_SMALL_SIZE)

    logger.info("\n[2/4] Initializing TTNN device and models...")
    logger.info(f"Using l1_small_size={l1_size}")
    device = ttnn.open_device(device_id=0, l1_small_size=l1_size)
    ttnn.CONFIG.enable_model_cache = True

    try:
        # Load both YuNet and SFace (same as server)
        yunet = load_yunet_model(device)
        model = load_ttnn_sface_model(device)

        # Compute embeddings and scores using full pipeline
        logger.info("\n[3/4] Computing embeddings for all pairs (YuNet → Align → SFace)...")
        scores = []
        inference_times = []
        skipped = 0
        fallback_count = 0

        for i in tqdm(range(len(labels)), desc="Processing pairs"):
            # Get image pair - pairs_images shape: (n_pairs, 2, H, W, C)
            img1_raw = pairs_images[i, 0]
            img2_raw = pairs_images[i, 1]

            # Run full pipeline: YuNet detection + alignment
            start_time = time.time()

            img1_aligned = detect_and_align_face(img1_raw, device)
            img2_aligned = detect_and_align_face(img2_raw, device)

            # Handle cases where face detection fails
            if img1_aligned is None:
                img1_aligned = preprocess_face_for_sface(img1_raw)
                fallback_count += 1
            if img2_aligned is None:
                img2_aligned = preprocess_face_for_sface(img2_raw)
                fallback_count += 1

            # Compute embeddings
            emb1 = compute_embedding(model, img1_aligned, device)
            emb2 = compute_embedding(model, img2_aligned, device)

            inference_times.append((time.time() - start_time) * 1000 / 2)  # Per image

            # Compute similarity
            sim = cosine_similarity(emb1, emb2)
            scores.append(sim)

        scores = np.array(scores)

        logger.info(f"\nProcessed {len(scores)} pairs")
        logger.info(f"  Face detection fallbacks: {fallback_count} (used simple resize)")

        # Compute metrics
        logger.info("\n" + "=" * 60)
        logger.info("BENCHMARK RESULTS")
        logger.info("=" * 60)

        metrics = compute_roc_metrics(scores, labels)

        # Key metrics
        logger.info("\n📊 KEY METRICS (Customer Requirements):")
        logger.info("-" * 40)

        tpr_at_01 = metrics.get("TPR@FAR=0.1%", 0)
        customer_pass = "✅ PASS" if tpr_at_01 >= 0.98 else "❌ FAIL"
        logger.info(f"  TPR @ 0.1% FAR: {tpr_at_01*100:.2f}% (Required: >98%) {customer_pass}")
        logger.info(f"  Threshold @ 0.1% FAR: {metrics.get('Threshold@FAR=0.1%', 0):.4f}")

        logger.info("\n📈 Additional Metrics:")
        logger.info("-" * 40)
        logger.info(f"  TPR @ 1.0% FAR: {metrics.get('TPR@FAR=1.0%', 0)*100:.2f}%")
        logger.info(f"  TPR @ 10.0% FAR: {metrics.get('TPR@FAR=10.0%', 0)*100:.2f}%")
        logger.info(f"  Best Accuracy: {metrics.get('Best_Accuracy', 0)*100:.2f}%")
        logger.info(f"  Optimal Threshold: {metrics.get('Best_Threshold', 0):.4f}")

        logger.info("\n📉 Score Distribution:")
        logger.info("-" * 40)
        logger.info(f"  Mean Positive Score (same person): {metrics.get('Mean_Positive_Score', 0):.4f}")
        logger.info(f"  Mean Negative Score (different): {metrics.get('Mean_Negative_Score', 0):.4f}")
        logger.info(f"  Score Separation: {metrics.get('Score_Separation', 0):.4f}")

        logger.info("\n⚡ Latency:")
        logger.info("-" * 40)
        logger.info(f"  Mean inference time: {np.mean(inference_times):.2f} ms/image")
        logger.info(f"  P95 inference time: {np.percentile(inference_times, 95):.2f} ms/image")
        logger.info(f"  P99 inference time: {np.percentile(inference_times, 99):.2f} ms/image")

        logger.info("\n" + "=" * 60)

        return metrics

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFace TPR@FAR Benchmark (Full Pipeline)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lfw",
        choices=["lfw", "celeba", "quick"],
        help="Dataset to use: lfw (default), celeba (higher res), quick (demo faces)",
    )
    parser.add_argument(
        "--num-pairs", type=int, default=1000, help="Number of pairs for CelebA dataset (default: 1000)"
    )
    parser.add_argument("--quick", action="store_true", help="Shortcut for --dataset quick")

    args = parser.parse_args()

    dataset = "quick" if args.quick else args.dataset
    run_benchmark(dataset=dataset, num_pairs=args.num_pairs)
