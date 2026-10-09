"""Voice-reference store: <dir>/<id>/{audio.wav, text.lab, codes.pt}. Codes are encoded once at `add`."""
from __future__ import annotations

import hashlib
import re
import threading
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch

from models.autoports.fishaudio_s2_pro.config import SAMPLE_RATE

ID_RE = re.compile(r"^[a-zA-Z0-9\-_ ]{1,255}$")


class ReferenceStore:
    def __init__(self, root: Path, encode_fn):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.encode = encode_fn  # float32 wav -> codes (10, T)
        self._lock = threading.Lock()
        self._hash_cache: Dict[str, Tuple[torch.Tensor, str]] = {}

    @staticmethod
    def valid_id(ref_id: str) -> bool:
        return bool(ID_RE.match(ref_id or ""))

    def list(self) -> List[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "codes.pt").exists())

    def exists(self, ref_id: str) -> bool:
        return (self.root / ref_id / "codes.pt").exists()

    def add(self, ref_id: str, wav: np.ndarray, text: str, overwrite: bool = False):
        d = self.root / ref_id
        with self._lock:
            if d.exists() and not overwrite:
                raise FileExistsError(ref_id)
            d.mkdir(parents=True, exist_ok=True)
            codes = self.encode(wav)
            sf.write(d / "audio.wav", wav, SAMPLE_RATE, subtype="PCM_16")
            (d / "text.lab").write_text(text)
            torch.save(codes, d / "codes.pt")
        return codes

    def get(self, ref_id: str) -> Tuple[torch.Tensor, str]:
        d = self.root / ref_id
        if not (d / "codes.pt").exists():
            raise FileNotFoundError(ref_id)
        return torch.load(d / "codes.pt"), (d / "text.lab").read_text() if (d / "text.lab").exists() else ""

    def delete(self, ref_id: str):
        d = self.root / ref_id
        if not d.exists():
            raise FileNotFoundError(ref_id)
        for f in d.iterdir():
            f.unlink()
        d.rmdir()

    def rename(self, old: str, new: str):
        (self.root / old).rename(self.root / new)

    def inline(self, audio: bytes, text: str, cache: bool = True) -> Tuple[torch.Tensor, str]:
        """Reference sent with the request: sha256-keyed in-memory cache (fish's load_by_hash)."""
        from models.autoports.fishaudio_s2_pro.server.audio_io import decode_audio

        key = hashlib.sha256(audio + text.encode()).hexdigest()
        if cache and key in self._hash_cache:
            return self._hash_cache[key]
        codes = self.encode(decode_audio(audio))
        if cache:
            self._hash_cache[key] = (codes, text)
        return codes, text
