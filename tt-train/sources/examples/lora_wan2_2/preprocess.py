# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 1 -- OmniConsistency <style>/tar + caption -> local dataset."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from PIL import Image

from pipeline_config import Config
from timing import phase


def preprocess(cfg: Config) -> Path:
    from huggingface_hub import snapshot_download

    out = Path(cfg.DATA_DIR)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pre] downloading {cfg.STYLE}/tar + {cfg.STYLE}/caption from {cfg.DATASET_ID} ...")
    with phase("hf download"):
        local = snapshot_download(
            repo_id=cfg.DATASET_ID,
            repo_type="dataset",
            allow_patterns=[f"{cfg.STYLE}/tar/*.png", f"{cfg.STYLE}/caption/*.txt"],
        )
    style_root = Path(local) / cfg.STYLE
    tar_dir, cap_dir = style_root / "tar", style_root / "caption"
    if not tar_dir.is_dir():
        raise FileNotFoundError(f"no {cfg.STYLE}/tar/ in {cfg.DATASET_ID} — check the style name")

    metadata: list[dict] = []
    skipped = 0
    with phase("copy images + captions"):
        for img_path in sorted(tar_dir.glob("*.png")):
            stem = img_path.stem
            cap_path = cap_dir / f"{stem}.txt"
            if not cap_path.exists():
                skipped += 1
                continue
            caption = cap_path.read_text(encoding="utf-8").strip()
            if not caption:
                skipped += 1
                continue
            shutil.copyfile(img_path, images_dir / f"{stem}.png")
            metadata.append({"idx": int(stem), "image": f"images/{stem}.png", "caption": caption})

    metadata.sort(key=lambda m: m["idx"])
    (out / "metadata.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in metadata) + "\n", encoding="utf-8"
    )
    print(
        f"[pre] wrote {len(metadata)} (image, caption) pairs -> {out.resolve()} "
        f"(skipped {skipped} without a caption)"
    )
    if not metadata:
        raise RuntimeError("no usable pairs produced")
    return out


_ART = r"(?:a\s+|an\s+|the\s+)?"  # optional article
_STYLE_PHRASES = [
    r"inspired by[^,.]*",  # "Inspired by LEGO minifigure art style"
    rf"in\s+{_ART}blocky\s+lego\s+form",
    r"blocky\s+lego\s+form",
    rf"in\s+{_ART}lego\s+style",
    r"lego\s+minifigure\s+art\s+style",
    r"lego\s+figure\s+style",
    r"lego\s+minifigure\s+style",
    r"lego\s+art\s+style",
    r"lego\s+minifigure",
    r"lego\s+style",
    r"blocky\s+shapes",
]
_PHRASE_RE = re.compile("|".join(_STYLE_PHRASES), re.I)
_STYLE_WORD_RE = re.compile(r"\b(lego|blocky|minifigure)\b", re.I)


def strip_style_words(caption: str) -> str:
    text = _PHRASE_RE.sub("", caption)
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if _STYLE_WORD_RE.search(s):  # pure style-directive sentence
            continue
        body, end = re.match(r"(.*?)([.!?]*)\s*$", s.strip()).groups()
        clauses = [c.strip() for c in body.split(",") if c.strip()]
        if clauses:
            out.append(", ".join(clauses) + end)
    return re.sub(r"\s{2,}", " ", " ".join(out)).strip(" ,")


def load_samples(data_dir: str) -> list[tuple[Image.Image, str]]:
    out = Path(data_dir)
    rows = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines() if line.strip()]
    return [(Image.open(out / r["image"]).convert("RGB"), r["caption"]) for r in rows]
