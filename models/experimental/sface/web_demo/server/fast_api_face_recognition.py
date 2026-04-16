# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI server for Face Recognition pipeline on Tenstorrent hardware.

Pipeline:
1. YuNet (TTNN) - Face detection
2. SFace (TTNN) - Face recognition (embedding extraction)
3. Cosine similarity - Match against known faces database
"""

import os

# Suppress verbose TTNN/TT_METAL logging (must be set before importing ttnn)
os.environ["TT_METAL_LOGGER_LEVEL"] = "ERROR"
os.environ["LOGURU_LEVEL"] = "WARNING"

import base64
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

import ttnn

# Directory to persist registered faces
FACES_DIR = Path(__file__).parent / "registered_faces"
FACES_DIR.mkdir(exist_ok=True)

# Mock Database directory for reference ID images (POC)
MOCK_DB_DIR = Path(__file__).parent / "mock_database"
MOCK_DB_DIR.mkdir(exist_ok=True)


# ============== Pydantic Models for /api/v1/verify ==============


class VerifyRequestJSON(BaseModel):
    """JSON request format with Base64 encoded images."""

    live_selfie: str  # Base64 encoded image
    user_id: Optional[str] = None  # Fetch reference from mock DB
    reference_image: Optional[str] = None  # OR provide Base64 reference directly


class VerifyResponse(BaseModel):
    """Standard verification response."""

    match_status: bool
    confidence_score: float  # 0.00 - 1.00
    latency_ms: int


# YuNet imports
from models.experimental.yunet.common import (
    YUNET_L1_SMALL_SIZE,
    STRIDES,
    DEFAULT_NMS_IOU_THRESHOLD,
    load_torch_model as load_yunet_torch_model,
    get_default_weights_path as get_yunet_weights_path,
)
from models.experimental.yunet.tt.ttnn_yunet import create_yunet_model

# SFace imports
from models.experimental.sface.common import get_sface_onnx_path, SFACE_L1_SMALL_SIZE
from models.experimental.sface.reference.sface_model import load_sface_from_onnx
from models.experimental.sface.tt.ttnn_sface import create_sface_model

app = FastAPI(
    title="Face Recognition Pipeline",
    description="YuNet (detection) + SFace (recognition) on Tenstorrent hardware",
    version="1.0",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Global variables
yunet_model = None
sface_model = None  # TTNN model
device = None
face_database: Dict[str, np.ndarray] = {}  # name -> embedding


def save_face_to_disk(name: str, embedding: np.ndarray, image: Image.Image):
    """Save face embedding and image to disk."""
    person_dir = FACES_DIR / name
    person_dir.mkdir(exist_ok=True)

    # Save embedding
    np.save(person_dir / "embedding.npy", embedding)

    # Save face image
    image.save(person_dir / "face.jpg")

    logging.info(f"Saved face for '{name}' to {person_dir}")


def load_faces_from_disk():
    """Load all saved faces from disk."""
    global face_database

    if not FACES_DIR.exists():
        return

    for person_dir in FACES_DIR.iterdir():
        if person_dir.is_dir():
            embedding_path = person_dir / "embedding.npy"
            if embedding_path.exists():
                name = person_dir.name
                embedding = np.load(embedding_path)
                face_database[name] = embedding
                logging.info(f"Loaded face for '{name}'")

    logging.info(f"Loaded {len(face_database)} faces from disk")


def delete_face_from_disk(name: str):
    """Delete face from disk."""
    person_dir = FACES_DIR / name
    if person_dir.exists():
        import shutil

        shutil.rmtree(person_dir)
        logging.info(f"Deleted face for '{name}' from disk")


@app.get("/")
async def root():
    return {"message": "Face Recognition API - YuNet + SFace on Tenstorrent"}


@app.on_event("startup")
async def startup():
    global yunet_model, sface_model, device

    # Enable model/program caching to avoid recompilation
    ttnn.CONFIG.enable_model_cache = True
    logging.info("Enabled TTNN model cache")

    # Use larger L1 size for both models
    l1_size = max(YUNET_L1_SMALL_SIZE, SFACE_L1_SMALL_SIZE)
    logging.info(f"Initializing Tenstorrent device with l1_small_size={l1_size}...")
    device = ttnn.open_device(device_id=0, l1_small_size=l1_size)

    # Load YuNet
    logging.info("Loading YuNet model...")
    yunet_weights = get_yunet_weights_path()
    yunet_torch = load_yunet_torch_model(yunet_weights)
    yunet_torch = yunet_torch.to(torch.bfloat16)
    yunet_model = create_yunet_model(device, yunet_torch)
    logging.info("YuNet loaded!")

    # Load SFace (TTNN only)
    logging.info("Loading SFace model...")
    sface_onnx = get_sface_onnx_path()
    sface_torch = load_sface_from_onnx(sface_onnx)
    sface_torch.eval()
    sface_model = create_sface_model(device, sface_torch)
    logging.info("SFace loaded! (TTNN, 0.95+ PCC)")

    # Load previously registered faces from disk
    load_faces_from_disk()

    # Warmup both models to compile ALL kernel variants (prevents runtime lag)
    logging.info("Warming up models (multiple iterations)...")

    # Warmup YuNet (detection) - multiple iterations with real-like data
    for i in range(3):
        warmup_yunet = torch.randint(0, 256, (1, 640, 640, 3), dtype=torch.float32).to(torch.bfloat16)
        warmup_yunet_tt = ttnn.from_torch(
            warmup_yunet, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        _ = yunet_model(warmup_yunet_tt)
        ttnn.synchronize_device(device)
    logging.info("  YuNet warmup done (3 iterations)")

    # Warmup SFace (recognition) - multiple iterations
    for i in range(3):
        warmup_sface = torch.randint(0, 256, (1, 112, 112, 3), dtype=torch.float32)
        warmup_sface_tt = ttnn.from_torch(
            warmup_sface, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        _ = sface_model(warmup_sface_tt)
        ttnn.synchronize_device(device)
    logging.info("  SFace warmup done (3 iterations)")

    # Load mock database for POC
    load_mock_database()

    logging.info("Warmup complete! Face Recognition pipeline ready!")


@app.on_event("shutdown")
async def shutdown():
    global device
    if device is not None:
        ttnn.close_device(device)
        logging.info("Device closed.")


def decode_yunet_detections(cls_outs, box_outs, obj_outs, kpt_outs, input_size, threshold=0.5):
    """Decode YuNet outputs to face detections."""
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
            x1 = max(best["box"][0], det["box"][0])
            y1 = max(best["box"][1], det["box"][1])
            x2 = min(best["box"][2], det["box"][2])
            y2 = min(best["box"][3], det["box"][3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (best["box"][2] - best["box"][0]) * (best["box"][3] - best["box"][1])
            area2 = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])
            if inter / max(area1 + area2 - inter, 1e-6) < DEFAULT_NMS_IOU_THRESHOLD:
                remaining.append(det)
        detections = remaining

    return keep


# Toggle face alignment (set to False to disable)
ENABLE_FACE_ALIGNMENT = True


def align_face_keypoints(image: np.ndarray, keypoints: List, target_size: int = 112) -> np.ndarray:
    """Align face using 5 keypoints (eyes, nose, mouth corners).

    Uses affine transformation to:
    1. Make eyes horizontal
    2. Center the face
    3. Scale to target size

    Args:
        image: numpy array (H, W, 3)
        keypoints: List of 5 [x, y] pairs (normalized 0-1):
                   [[left_eye], [right_eye], [nose], [left_mouth], [right_mouth]]
        target_size: output size (default 112x112 for SFace)

    Returns:
        Aligned face image (target_size x target_size x 3)
    """
    h, w = image.shape[:2]

    # Convert normalized keypoints to pixel coordinates
    # Keypoints format: [[x,y], [x,y], [x,y], [x,y], [x,y]]
    left_eye = np.array([keypoints[0][0] * w, keypoints[0][1] * h])
    right_eye = np.array([keypoints[1][0] * w, keypoints[1][1] * h])
    nose = np.array([keypoints[2][0] * w, keypoints[2][1] * h])
    left_mouth = np.array([keypoints[3][0] * w, keypoints[3][1] * h])
    right_mouth = np.array([keypoints[4][0] * w, keypoints[4][1] * h])

    # Source points (detected keypoints)
    src_pts = np.float32([left_eye, right_eye, nose])

    # Reference destination points for 112x112 aligned face
    # These are standard positions for SFace/ArcFace alignment
    dst_pts = np.float32(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
        ]
    )

    # Compute affine transform
    M = cv2.getAffineTransform(src_pts, dst_pts)

    # Apply transform
    aligned = cv2.warpAffine(image, M, (target_size, target_size), borderMode=cv2.BORDER_REPLICATE)

    return aligned


def crop_and_align_face(
    image: Image.Image, box: List[float], keypoints: List[float] = None, target_size: int = 112
) -> tuple:
    """Crop and optionally align face from image.

    Args:
        image: PIL Image
        box: [x1, y1, x2, y2] normalized coordinates
        keypoints: Optional 10 values [lx, ly, rx, ry, nx, ny, lmx, lmy, rmx, rmy] normalized
        target_size: Output size (112 for SFace)

    Returns: (face_image, face_size) where face_size is original crop size in pixels
    """
    w, h = image.size
    x1 = int(box[0] * w)
    y1 = int(box[1] * h)
    x2 = int(box[2] * w)
    y2 = int(box[3] * h)

    # Original face size (before margin)
    face_width = x2 - x1
    face_height = y2 - y1
    face_size = min(face_width, face_height)

    # Try keypoint alignment if enabled and keypoints available (5 keypoints)
    if ENABLE_FACE_ALIGNMENT and keypoints is not None and len(keypoints) >= 5:
        try:
            img_np = np.array(image)
            aligned = align_face_keypoints(img_np, keypoints, target_size)
            return Image.fromarray(aligned), face_size
        except Exception as e:
            logging.warning(f"Face alignment failed, falling back to crop: {e}")

    # Fallback: simple crop and resize
    margin_x = int((x2 - x1) * 0.1)
    margin_y = int((y2 - y1) * 0.1)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    face_crop = image.crop((x1, y1, x2, y2))
    face_resized = face_crop.resize((target_size, target_size))
    return face_resized, face_size


def get_face_embedding(face_image: Image.Image) -> np.ndarray:
    """Extract face embedding using SFace (TTNN).

    Args:
        face_image: PIL Image of cropped face (112x112)
    """
    global sface_model, device

    # Convert to tensor - NHWC format for TTNN
    face_np = np.array(face_image).astype(np.float32)
    face_tensor = torch.from_numpy(face_np).unsqueeze(0)  # [1, 112, 112, 3] NHWC

    # Run TTNN SFace
    tt_input = ttnn.from_torch(face_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = sface_model(tt_input)
    embedding = ttnn.to_torch(tt_output).float().numpy().flatten()

    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def match_face(embedding: np.ndarray, threshold: float = 0.5) -> tuple:
    """Match embedding against known faces database.

    Simple threshold matching - returns best match if above threshold.

    Returns: (matched_name or None, best_similarity_score)
    """
    global face_database

    if not face_database:
        logging.debug("No faces in database")
        return None, 0.0

    # Calculate similarity to all registered faces
    best_match = None
    best_score = 0.0

    scores = {}
    for name, db_embedding in face_database.items():
        score = cosine_similarity(embedding, db_embedding)
        scores[name] = score
        if score > best_score:
            best_score = score
            best_match = name

    # Log all scores for debugging
    score_str = ", ".join([f"{n}: {s:.3f}" for n, s in sorted(scores.items(), key=lambda x: -x[1])])
    logging.info(f"Scores: [{score_str}] | Best: '{best_match}' @ {best_score:.3f} (thresh: {threshold:.2f})")

    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


@app.post("/register")
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    """Register a new face to the database."""
    global face_database, yunet_model, device

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Detect face using YuNet
    input_size = 640  # Customer requirement: 640x640 input
    image_resized = image.resize((input_size, input_size))
    tensor = torch.from_numpy(np.array(image_resized)).float().unsqueeze(0).to(torch.bfloat16)

    tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cls_out, box_out, obj_out, kpt_out = yunet_model(tt_input)
    ttnn.synchronize_device(device)

    detections = decode_yunet_detections(cls_out, box_out, obj_out, kpt_out, input_size, threshold=0.5)

    if not detections:
        return JSONResponse(status_code=400, content={"error": "No face detected in image"})

    # Use the highest confidence face
    best_face = max(detections, key=lambda x: x["conf"])
    face_crop, face_size = crop_and_align_face(image, best_face["box"], best_face.get("keypoints"))

    if face_size < 50:
        return JSONResponse(status_code=400, content={"error": "Face too small. Please use a closer photo."})

    # Get embedding (use PyTorch for high accuracy)
    embedding = get_face_embedding(face_crop)
    face_database[name] = embedding

    # Save to disk for persistence
    save_face_to_disk(name, embedding, face_crop)

    logging.info(f"Registered face for '{name}' with embedding shape {embedding.shape}")
    return {"message": f"Face registered for '{name}'", "num_registered": len(face_database)}


@app.get("/registered")
async def get_registered_faces():
    """Get list of registered faces."""
    return {"registered_faces": list(face_database.keys()), "count": len(face_database)}


@app.delete("/registered/{name}")
async def delete_registered_face(name: str):
    """Delete a registered face."""
    global face_database
    if name in face_database:
        del face_database[name]
        delete_face_from_disk(name)  # Also delete from disk
        return {"message": f"Deleted '{name}'", "num_registered": len(face_database)}
    return JSONResponse(status_code=404, content={"error": f"Face '{name}' not found"})


@app.post("/recognize")
async def recognize_faces(
    file: UploadFile = File(...), input_size: int = 640, detection_thresh: float = 0.5, recognition_thresh: float = 0.6
):
    """
    Full face recognition pipeline:
    1. Detect faces with YuNet
    2. Extract embeddings with SFace
    3. Match against registered faces
    """
    global yunet_model, sface_model, device

    t_start = time.time()

    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    orig_w, orig_h = image.size

    # ===== STEP 1: Face Detection (YuNet) =====
    t1 = time.time()
    image_resized = image.resize((input_size, input_size))
    tensor = torch.from_numpy(np.array(image_resized)).float().unsqueeze(0).to(torch.bfloat16)

    tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    cls_out, box_out, obj_out, kpt_out = yunet_model(tt_input)
    ttnn.synchronize_device(device)
    t2 = time.time()

    detections = decode_yunet_detections(cls_out, box_out, obj_out, kpt_out, input_size, threshold=detection_thresh)
    detection_time_ms = (t2 - t1) * 1000

    # ===== STEP 2 & 3: Face Recognition (SFace) for each detected face =====
    results = []
    recognition_time_ms = 0

    # Minimum face size for recognition (pixels) - skip tiny faces
    MIN_FACE_SIZE = 50  # Lowered for better range (was 100)

    for det in detections:
        # Crop and align face (with keypoint alignment if available)
        face_crop, face_size = crop_and_align_face(image, det["box"], det.get("keypoints"))

        # Skip recognition for very small faces (unreliable)
        if face_size < MIN_FACE_SIZE:
            results.append(
                {
                    "box": det["box"],
                    "detection_conf": det["conf"],
                    "keypoints": det["keypoints"],
                    "identity": "Too Small",
                    "similarity": 0.0,
                    "is_known": False,
                }
            )
            continue

        t3 = time.time()

        # Get embedding (use PyTorch for high accuracy)
        embedding = get_face_embedding(face_crop)

        t4 = time.time()
        recognition_time_ms += (t4 - t3) * 1000

        # Match against database
        matched_name, similarity = match_face(embedding, threshold=recognition_thresh)

        results.append(
            {
                "box": det["box"],
                "detection_conf": det["conf"],
                "keypoints": det["keypoints"],
                "identity": matched_name if matched_name else "Unknown",
                "similarity": similarity,
                "is_known": matched_name is not None,
            }
        )

    t_end = time.time()
    total_time_ms = (t_end - t_start) * 1000

    logging.info(
        f"Detection: {detection_time_ms:.1f}ms | "
        f"Recognition: {recognition_time_ms:.1f}ms | "
        f"Total: {total_time_ms:.1f}ms | "
        f"Faces: {len(results)}"
    )

    return {
        "faces": results,
        "num_faces": len(results),
        "detection_time_ms": detection_time_ms,
        "recognition_time_ms": recognition_time_ms,
        "total_time_ms": total_time_ms,
        "registered_faces": len(face_database),
    }


# ============== Mock Database for Reference Images (POC) ==============

# In-memory mock database: user_id -> embedding
mock_database: Dict[str, np.ndarray] = {}


def load_mock_database():
    """Load reference images from mock_database directory into memory."""
    global mock_database
    mock_database = {}

    if not MOCK_DB_DIR.exists():
        logging.info("Mock database directory not found, creating...")
        MOCK_DB_DIR.mkdir(exist_ok=True)
        return

    for user_dir in MOCK_DB_DIR.iterdir():
        if user_dir.is_dir():
            user_id = user_dir.name
            embedding_path = user_dir / "embedding.npy"
            if embedding_path.exists():
                mock_database[user_id] = np.load(embedding_path)
                logging.info(f"Loaded mock DB entry: {user_id}")

    logging.info(f"Mock database loaded: {len(mock_database)} reference identities")


def save_to_mock_database(user_id: str, embedding: np.ndarray, image: Image.Image):
    """Save a reference identity to the mock database."""
    global mock_database

    user_dir = MOCK_DB_DIR / user_id
    user_dir.mkdir(exist_ok=True)

    # Save embedding
    np.save(user_dir / "embedding.npy", embedding)

    # Save reference image
    image.save(user_dir / "reference.jpg")

    # Update in-memory database
    mock_database[user_id] = embedding

    logging.info(f"Saved to mock DB: {user_id}")


def get_reference_embedding(user_id: str) -> Optional[np.ndarray]:
    """Get reference embedding from mock database."""
    return mock_database.get(user_id)


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode a Base64 string to PIL Image."""
    # Handle data URL format (e.g., "data:image/jpeg;base64,...")
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")


def extract_face_embedding_from_image(image: Image.Image) -> Optional[np.ndarray]:
    """
    Extract face embedding from an image.
    Uses YuNet for detection and SFace for embedding.
    Returns None if no face detected.
    """
    global yunet_model, sface_model, device

    # Detect face
    img_np = np.array(image)
    input_size = 640

    # Resize for detection
    img_resized = image.resize((input_size, input_size))
    img_np_resized = np.array(img_resized).astype(np.float32)
    tensor = torch.from_numpy(img_np_resized).unsqueeze(0).to(torch.bfloat16)
    tt_input = ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    cls_out, box_out, obj_out, kpt_out = yunet_model(tt_input)
    ttnn.synchronize_device(device)

    detections = decode_yunet_detections(cls_out, box_out, obj_out, kpt_out, input_size, threshold=0.5)

    if not detections:
        return None

    # Use the largest/most confident face
    det = max(detections, key=lambda d: d["conf"])

    # Crop face
    orig_w, orig_h = image.size
    x1 = int(det["box"][0] * orig_w)
    y1 = int(det["box"][1] * orig_h)
    x2 = int(det["box"][2] * orig_w)
    y2 = int(det["box"][3] * orig_h)

    # Add padding
    face_w, face_h = x2 - x1, y2 - y1
    pad = int(max(face_w, face_h) * 0.1)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(orig_w, x2 + pad), min(orig_h, y2 + pad)

    face_crop = image.crop((x1, y1, x2, y2)).resize((112, 112))

    # Get embedding
    embedding = get_face_embedding(face_crop)
    return embedding


# ============== Production API: /api/v1/verify ==============


@app.post("/api/v1/verify", response_model=VerifyResponse)
async def verify_faces_json(request: VerifyRequestJSON):
    """
    Verify if live selfie matches reference identity.

    Accepts JSON with Base64 encoded images.

    Args:
        live_selfie: Base64 encoded live selfie image
        user_id: User ID to fetch reference from mock database (optional)
        reference_image: Base64 encoded reference image (optional)

    Must provide either user_id OR reference_image.

    Returns:
        match_status: Boolean indicating if faces match
        confidence_score: Float 0.00-1.00 (cosine similarity)
        latency_ms: Processing time in milliseconds
    """
    t_start = time.time()

    # Validate input
    if not request.user_id and not request.reference_image:
        raise HTTPException(status_code=400, detail="Must provide either user_id or reference_image")

    try:
        # Decode live selfie
        live_image = decode_base64_image(request.live_selfie)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid live_selfie image: {str(e)}")

    # Get reference embedding
    if request.user_id:
        # Fetch from mock database
        reference_embedding = get_reference_embedding(request.user_id)
        if reference_embedding is None:
            raise HTTPException(status_code=404, detail=f"User ID '{request.user_id}' not found in database")
    else:
        # Decode and extract from provided reference image
        try:
            reference_image = decode_base64_image(request.reference_image)
            reference_embedding = extract_face_embedding_from_image(reference_image)
            if reference_embedding is None:
                raise HTTPException(status_code=400, detail="No face detected in reference_image")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid reference_image: {str(e)}")

    # Extract embedding from live selfie
    live_embedding = extract_face_embedding_from_image(live_image)
    if live_embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in live_selfie")

    # Calculate similarity
    similarity = cosine_similarity(live_embedding, reference_embedding)

    # Determine match (with correct preprocessing, 0.5 threshold works well)
    MATCH_THRESHOLD = 0.5
    match_status = similarity >= MATCH_THRESHOLD

    t_end = time.time()
    latency_ms = int((t_end - t_start) * 1000)

    logging.info(f"[/api/v1/verify] Match: {match_status} | Score: {similarity:.3f} | Latency: {latency_ms}ms")

    return VerifyResponse(
        match_status=match_status,
        confidence_score=round(float(similarity), 2),
        latency_ms=latency_ms,
    )


@app.post("/api/v1/verify/multipart", response_model=VerifyResponse)
async def verify_faces_multipart(
    live_selfie: UploadFile = File(..., description="Live selfie image (JPEG/PNG/WEBP)"),
    user_id: Optional[str] = Form(None, description="User ID to fetch reference from mock DB"),
    reference_image: Optional[UploadFile] = File(None, description="Reference image (if not using user_id)"),
):
    """
    Verify if live selfie matches reference identity.

    Accepts Multipart/Form-data with image files.

    Args:
        live_selfie: Live selfie image file
        user_id: User ID to fetch reference from mock database (optional)
        reference_image: Reference image file (optional)

    Must provide either user_id OR reference_image.

    Returns:
        match_status: Boolean indicating if faces match
        confidence_score: Float 0.00-1.00 (cosine similarity)
        latency_ms: Processing time in milliseconds
    """
    t_start = time.time()

    # Validate input
    if not user_id and not reference_image:
        raise HTTPException(status_code=400, detail="Must provide either user_id or reference_image")

    try:
        # Load live selfie
        live_bytes = await live_selfie.read()
        live_image = Image.open(BytesIO(live_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid live_selfie image: {str(e)}")

    # Get reference embedding
    if user_id:
        # Fetch from mock database
        reference_embedding = get_reference_embedding(user_id)
        if reference_embedding is None:
            raise HTTPException(status_code=404, detail=f"User ID '{user_id}' not found in database")
    else:
        # Extract from provided reference image
        try:
            ref_bytes = await reference_image.read()
            ref_image = Image.open(BytesIO(ref_bytes)).convert("RGB")
            reference_embedding = extract_face_embedding_from_image(ref_image)
            if reference_embedding is None:
                raise HTTPException(status_code=400, detail="No face detected in reference_image")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid reference_image: {str(e)}")

    # Extract embedding from live selfie
    live_embedding = extract_face_embedding_from_image(live_image)
    if live_embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in live_selfie")

    # Calculate similarity
    similarity = cosine_similarity(live_embedding, reference_embedding)

    # Determine match (with correct preprocessing, 0.5 threshold works well)
    MATCH_THRESHOLD = 0.5
    match_status = similarity >= MATCH_THRESHOLD

    t_end = time.time()
    latency_ms = int((t_end - t_start) * 1000)

    logging.info(
        f"[/api/v1/verify/multipart] Match: {match_status} | Score: {similarity:.3f} | Latency: {latency_ms}ms"
    )

    return VerifyResponse(
        match_status=match_status,
        confidence_score=round(float(similarity), 2),
        latency_ms=latency_ms,
    )


@app.post("/api/v1/mock-db/register")
async def register_mock_user(
    user_id: str = Form(..., description="Unique user identifier"),
    reference_image: UploadFile = File(..., description="Reference ID image"),
):
    """
    Register a new user in the mock database.

    Args:
        user_id: Unique identifier for the user
        reference_image: Reference ID image file

    Returns:
        Success message with user_id
    """
    try:
        ref_bytes = await reference_image.read()
        ref_image = Image.open(BytesIO(ref_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    # Extract embedding
    embedding = extract_face_embedding_from_image(ref_image)
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in reference image")

    # Save to mock database
    save_to_mock_database(user_id, embedding, ref_image)

    return {"status": "success", "user_id": user_id, "message": f"User '{user_id}' registered in mock database"}


@app.get("/api/v1/mock-db/users")
async def list_mock_users():
    """List all users in the mock database."""
    return {
        "users": list(mock_database.keys()),
        "count": len(mock_database),
    }


@app.delete("/api/v1/mock-db/users/{user_id}")
async def delete_mock_user(user_id: str):
    """Delete a user from the mock database."""
    global mock_database

    if user_id not in mock_database:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")

    # Remove from memory
    del mock_database[user_id]

    # Remove from disk
    user_dir = MOCK_DB_DIR / user_id
    if user_dir.exists():
        import shutil

        shutil.rmtree(user_dir)

    return {"status": "success", "message": f"User '{user_id}' deleted from mock database"}
