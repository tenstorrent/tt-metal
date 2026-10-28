# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real inputs for the full-model tests and the demo (no ttnn import; usable from any python env).

Every item is in the sentence-transformers input format: a `str` (text only) or a `dict` whose
keys are modalities ("text", "image", "audio", "video"). The dict key order is the order of the
content in the rendered prompt, as in `SentenceTransformer.encode` (`base/modality.py`
`batch_to_message`). A text that contains `<|image|>` / `<|audio|>` / `<|video|>` places the
media at those markers (chat_template.jinja `has_manual_placeholders`).

Text prompts (`config_sentence_transformers.json` "prompts") are already prefixed. ST passes a
prompt as a system message, and the chat template renders the system text directly before the
user content, so the rendered string is the same.

Media (all in-tree):
    images  models/sample_data, resnet50 demo images, retinanet resources (same files as E2)
    audio   16 kHz speech clips of the Whisper demo (same cuts as E3) + one SYNTHETIC tone
    video   8 frames at 1 fps of tech_reports/ttnn/images/ttnn-dashboard.mp4 (same as E2)
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]  # tt-metal checkout

IMAGE_FILES = {
    "house": "models/sample_data/house_in_field_1080p.jpg",  # 1600 x 900
    "pajamas": "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images/"
    "ILSVRC2012_val_00040007_n03877472.JPEG",  # 332 x 500
    "dog": "models/experimental/retinanet/resources/dog_800x800.jpg",  # 800 x 800
    "handwriting": "models/sample_data/iam_ocr_image.jpg",  # 248 x 76
}
SPEECH_DIR = "models/demos/audio/whisper/demo/dataset/conditional_generation"
VIDEO_FILE = "tech_reports/ttnn/images/ttnn-dashboard.mp4"
VIDEO_FRAMES = 8
SR = 16000

QUERY = "task: search result | query: "
DOC = "title: none | text: "
STS = "task: sentence similarity | query: "


@functools.lru_cache(maxsize=None)
def image(name: str):
    from PIL import Image

    return Image.open(_ROOT / IMAGE_FILES[name]).convert("RGB")


@functools.lru_cache(maxsize=None)
def _wav(path: str) -> np.ndarray:
    import soundfile as sf

    w, sr = sf.read(path, dtype="float32")
    if sr != SR or w.ndim != 1:
        raise ValueError(f"{path}: expected 16 kHz mono, got sr={sr} shape={w.shape}")
    return w


def audio(name: str) -> np.ndarray:
    """16 kHz mono float32 waveform (cuts identical to tests/pcc/audio/audio_common.py)."""
    files = sorted((_ROOT / SPEECH_DIR).glob("*.wav"))
    if name == "speech2s":
        return _wav(str(files[1]))[int(1.00 * SR) : int(3.00 * SR)]
    if name == "speech10s":
        out, n = [], 10 * SR
        for f in files:
            out.append(_wav(str(f)))
            if sum(len(x) for x in out) >= n:
                return np.concatenate(out)[:n]
        raise ValueError("in-tree speech totals less than 10 s")
    if name == "tone3s":  # SYNTHETIC: 440 Hz tone + 200 -> 4000 Hz linear chirp, 3 s
        t = np.arange(3 * SR) / SR
        chirp = np.sin(2 * np.pi * (200 * t + (4000 - 200) / (2 * 3.0) * t**2))
        return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * chirp).astype(np.float32)
    raise KeyError(name)


@functools.lru_cache(maxsize=1)
def video_frames(n: int = VIDEO_FRAMES) -> np.ndarray:
    """`n` RGB frames at 1 fps (every fps-th frame), [n, H, W, 3] uint8. Needs OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(_ROOT / VIDEO_FILE))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(i * fps)))
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {i} of {VIDEO_FILE}")
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)


def long_text(min_chars: int = 7000) -> str:
    """A > 1025-token English document: the head of tech_reports/LLMs/llms.md."""
    text = (_ROOT / "tech_reports/LLMs/llms.md").read_text()
    return text[:min_chars]


def video(frames: np.ndarray | None = None) -> dict:
    """Video input as ST passes it to the processor: frames + metadata at 1 fps (so the processor's
    1-fps sampling keeps every frame)."""
    frames = video_frames() if frames is None else frames
    n = int(frames.shape[0])
    return {"array": frames, "video_metadata": {"fps": 1.0, "total_num_frames": n, "duration": float(n)}}


# ------------------------------------------------------------------ catalogue
IMAGE_CAPTIONS = {
    "house": "a house in a green field",
    "pajamas": "a man in pajamas sitting on a bed",
    "dog": "a photo of a dog",
    "handwriting": "a handwritten word",
}
AUDIO_CAPTIONS = {
    "speech2s": "a short clip of a person speaking",
    "speech10s": "a long recording of several people talking",
    "tone3s": "a synthetic beep tone with a rising frequency sweep",
}


def items() -> dict[str, object]:
    """name -> item. Built lazily (media decoding)."""
    out: dict[str, object] = {
        # text
        "text_query": QUERY + "What causes the northern lights?",
        "text_doc": DOC + "The northern lights are caused by charged particles from the sun colliding with "
        "gases in the upper atmosphere.",
        "text_de": STS + "Der schnelle braune Fuchs springt über den faulen Hund.",
        "text_ja": STS + "東京は日本の首都であり、世界有数の大都市です。",
        "text_es": QUERY + "¿Cuál es la capital de Francia?",
        "text_long": DOC + long_text(),
        # image only (retrieval columns)
        **{f"image_{k}": {"image": image(k)} for k in IMAGE_FILES},
        # image + caption
        "image_house_caption": {"text": "A photo of a house in a field.", "image": image("house")},
        "image_dog_caption": {"image": image("dog"), "text": "A dog looking at the camera."},
        "image_pajamas_caption": {"text": "A man wearing pajamas.", "image": image("pajamas")},
        # video + text
        "video_text": {"text": "A screen recording of a software dashboard.", "video": video()},
        # audio only (retrieval columns)
        **{f"audio_{k}": {"audio": {"array": audio(k), "sampling_rate": SR}} for k in AUDIO_CAPTIONS},
        # audio + text
        "audio_speech2s_text": {"text": "Someone says:", "audio": {"array": audio("speech2s"), "sampling_rate": SR}},
        "audio_speech10s_text": {
            "audio": {"array": audio("speech10s"), "sampling_rate": SR},
            "text": "A recording of English speech.",
        },
        # mixed
        "mixed_image_audio": {
            "text": "A dog <|image|> and a spoken note <|audio|> about it.",
            "image": [image("dog")],
            "audio": [{"array": audio("speech2s"), "sampling_rate": SR}],
        },
        "mixed_two_images": {
            "text": "Two photos: <|image|> then <|image|>.",
            "image": [image("house"), image("pajamas")],
        },
    }
    return out


# names of the text items used as retrieval rows
IMAGE_RETRIEVAL_TEXTS = {f"caption_{k}": v for k, v in IMAGE_CAPTIONS.items()}
AUDIO_RETRIEVAL_TEXTS = {f"caption_{k}": v for k, v in AUDIO_CAPTIONS.items()}
