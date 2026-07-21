# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Download and resolve golden reference audio from the Microsoft VibeVoice demo site."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from models.experimental.vibevoice.common.config import RESOURCES_DIR, TEXT_EXAMPLES_DIR

PathLike = Union[str, Path]

# https://microsoft.github.io/VibeVoice/
WEBSITE_BASE = "https://microsoft.github.io/VibeVoice"
GOLDEN_DIR = RESOURCES_DIR / "golden"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
TARGET_SAMPLE_RATE = 24000

_USER_AGENT = "tt-metal-vibevoice-golden-audio-sync"


@dataclass(frozen=True)
class GoldenDemoEntry:
    """One audio sample from the public VibeVoice demo page."""

    id: str
    website_section: str
    website_title: str
    audio_mp3_url: str
    wav_filename: str
    text_file: Optional[str] = None
    transcript_json_url: Optional[str] = None

    @property
    def wav_path(self) -> Path:
        return GOLDEN_DIR / self.wav_filename


# Catalog aligned with https://microsoft.github.io/VibeVoice/ sections and local text scripts.
GOLDEN_DEMOS: tuple[GoldenDemoEntry, ...] = (
    GoldenDemoEntry(
        id="2p_argument",
        website_section="context_aware_expression",
        website_title="Spontaneous Emotion",
        audio_mp3_url=f"{WEBSITE_BASE}/assets/audio/2p_argument.mp3",
        wav_filename="2p_argument.wav",
        transcript_json_url=f"{WEBSITE_BASE}/assets/text/2p_argument_gt_timestamp.json",
    ),
    GoldenDemoEntry(
        id="2p_see_u_again",
        website_section="context_aware_expression",
        website_title="Spontaneous Singing",
        audio_mp3_url=f"{WEBSITE_BASE}/assets/audio/2p_see_u_again.mp3",
        wav_filename="2p_see_u_again.wav",
        transcript_json_url=f"{WEBSITE_BASE}/assets/text/2p_see_u_again_gt_timestamp.json",
    ),
    GoldenDemoEntry(
        id="3p_gpt5",
        website_section="podcast_background_music",
        website_title="Podcast with Background Music (3-speaker)",
        audio_mp3_url=f"{WEBSITE_BASE}/assets/audio/3p_gpt5.mp3",
        wav_filename="3p_gpt5.wav",
        text_file="3p_gpt5.txt",
        transcript_json_url=f"{WEBSITE_BASE}/assets/text/3p_gpt5_gt_timestamp.json",
    ),
    GoldenDemoEntry(
        id="2p_goat",
        website_section="podcast_background_music",
        website_title="Podcast with Background Music (2-speaker)",
        audio_mp3_url=f"{WEBSITE_BASE}/assets/audio/2p_goat.mp3",
        wav_filename="2p_goat.wav",
        text_file="2p_goat.txt",
        transcript_json_url=f"{WEBSITE_BASE}/assets/text/2p_goat_gt_timestamp.json",
    ),
    GoldenDemoEntry(
        id="1p_CH2EN",
        website_section="cross_lingual",
        website_title="Mandarin to English",
        audio_mp3_url=f"{WEBSITE_BASE}/assets/audio/1p_CH2EN.mp3",
        wav_filename="1p_CH2EN.wav",
        text_file="1p_Ch2EN.txt",
        transcript_json_url=f"{WEBSITE_BASE}/assets/text/1p_CH2EN_gt_timestamp.json",
    ),
    GoldenDemoEntry(
        id="1p_EN2CH",
        website_section="cross_lingual",
        website_title="English to Mandarin",
        audio_mp3_url=f"{WEBSITE_BASE}/assets/audio/1p_EN2CH.mp3",
        wav_filename="1p_EN2CH.wav",
        transcript_json_url=f"{WEBSITE_BASE}/assets/text/1p_EN2CH_gt_timestamp.json",
    ),
    GoldenDemoEntry(
        id="4p_climate_45min",
        website_section="long_conversational_speech",
        website_title="Long 4-speaker conversation (~45 min)",
        audio_mp3_url=f"{WEBSITE_BASE}/assets/audio/4p_climate_45min.mp3",
        wav_filename="4p_climate_45min.wav",
        text_file="4p_climate_45min.txt",
        transcript_json_url=f"{WEBSITE_BASE}/assets/text/4p_climate_45min_gt_timestamp.json",
    ),
    GoldenDemoEntry(
        id="4p_climate_100min",
        website_section="long_conversational_speech",
        website_title="Long 4-speaker conversation (~90–100 min)",
        audio_mp3_url=f"{WEBSITE_BASE}/assets/audio/4p_climate_100min.mp3",
        wav_filename="4p_climate_100min.wav",
        text_file="4p_climate_100min.txt",
        transcript_json_url=f"{WEBSITE_BASE}/assets/text/4p_climate_100min_gt_timestamp.json",
    ),
)

# Shortest website golden clip that also has a local text script (quick TT smoke + compare).
MINIMAL_GOLDEN_DEMO_ID = "1p_CH2EN"
_TEXT_TO_GOLDEN_ID: dict[str, str] = {}
for _entry in GOLDEN_DEMOS:
    if _entry.text_file:
        _TEXT_TO_GOLDEN_ID[Path(_entry.text_file).stem.lower()] = _entry.id


def _download_bytes(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        dest.write_bytes(response.read())


def _mp3_to_wav_ffmpeg(mp3_path: Path, wav_path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; install ffmpeg to convert MP3 golden audio to WAV")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(mp3_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _mp3_to_wav_torchaudio(mp3_path: Path, wav_path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    import torchaudio

    wav, sr = torchaudio.load(str(mp3_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(wav_path), wav, sample_rate)


def convert_mp3_to_wav(mp3_path: Path, wav_path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    if shutil.which("ffmpeg"):
        _mp3_to_wav_ffmpeg(mp3_path, wav_path, sample_rate=sample_rate)
        return
    _mp3_to_wav_torchaudio(mp3_path, wav_path, sample_rate=sample_rate)


def write_manifest(entries: Optional[list[GoldenDemoEntry]] = None, manifest_path: Path = MANIFEST_PATH) -> Path:
    entries = entries or list(GOLDEN_DEMOS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": WEBSITE_BASE,
        "sample_rate_hz": TARGET_SAMPLE_RATE,
        "demos": [
            {
                **asdict(entry),
                "wav_path": str(entry.wav_path),
                "local_text_path": str(TEXT_EXAMPLES_DIR / entry.text_file) if entry.text_file else None,
            }
            for entry in entries
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def download_golden_audios(
    golden_dir: Optional[PathLike] = None,
    *,
    download_transcripts: bool = True,
    force: bool = False,
) -> list[Path]:
    """Download demo MP3s from microsoft.github.io/VibeVoice and convert to 24 kHz mono WAV."""
    root = Path(golden_dir) if golden_dir is not None else GOLDEN_DIR
    root.mkdir(parents=True, exist_ok=True)
    transcripts_dir = root / "transcripts"
    downloaded_wavs: list[Path] = []

    for entry in GOLDEN_DEMOS:
        wav_path = root / entry.wav_filename
        if wav_path.is_file() and wav_path.stat().st_size > 0 and not force:
            logger.info(f"Golden audio already present: {wav_path}")
            downloaded_wavs.append(wav_path)
            continue

        mp3_path = root / f"{entry.id}.mp3"
        logger.info(f"Downloading {entry.audio_mp3_url}")
        try:
            _download_bytes(entry.audio_mp3_url, mp3_path)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Failed to download {entry.audio_mp3_url}: {exc}") from exc

        logger.info(f"Converting {mp3_path.name} → {wav_path.name} ({TARGET_SAMPLE_RATE} Hz mono)")
        convert_mp3_to_wav(mp3_path, wav_path)
        downloaded_wavs.append(wav_path)

        if download_transcripts and entry.transcript_json_url:
            json_dest = transcripts_dir / f"{entry.id}_gt_timestamp.json"
            if force or not json_dest.is_file():
                logger.info(f"Downloading transcript {entry.transcript_json_url}")
                _download_bytes(entry.transcript_json_url, json_dest)

    write_manifest(manifest_path=root / "manifest.json")
    return downloaded_wavs


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Golden manifest not found at {manifest_path}. "
            "Call ensure_golden_audios(download=True) to fetch golden audio."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_golden_demo(demo_id: str) -> GoldenDemoEntry:
    for entry in GOLDEN_DEMOS:
        if entry.id == demo_id:
            return entry
    raise KeyError(f"Unknown golden demo id: {demo_id}")


def minimal_golden_demo() -> GoldenDemoEntry:
    """Return the shortest golden demo that has a paired local text script."""
    candidates = [e for e in GOLDEN_DEMOS if e.text_file and e.wav_path.is_file()]
    if not candidates:
        return get_golden_demo(MINIMAL_GOLDEN_DEMO_ID)
    return min(candidates, key=lambda e: e.wav_path.stat().st_size)


def text_path_for_demo(demo_id: str) -> Path:
    entry = get_golden_demo(demo_id)
    if not entry.text_file:
        raise ValueError(f"Demo {demo_id} has no local text script; use transcript JSON under golden/transcripts/")
    return TEXT_EXAMPLES_DIR / entry.text_file


def download_golden_demo(demo_id: str, golden_dir: Optional[PathLike] = None, *, force: bool = False) -> Path:
    """Download and convert a single golden demo clip."""
    root = Path(golden_dir) if golden_dir is not None else GOLDEN_DIR
    entry = get_golden_demo(demo_id)
    wav_path = root / entry.wav_filename
    if wav_path.is_file() and wav_path.stat().st_size > 0 and not force:
        return wav_path

    mp3_path = root / f"{entry.id}.mp3"
    logger.info(f"Downloading {entry.audio_mp3_url}")
    _download_bytes(entry.audio_mp3_url, mp3_path)
    convert_mp3_to_wav(mp3_path, wav_path)
    if entry.transcript_json_url:
        json_dest = root / "transcripts" / f"{entry.id}_gt_timestamp.json"
        if force or not json_dest.is_file():
            _download_bytes(entry.transcript_json_url, json_dest)
    write_manifest(manifest_path=root / "manifest.json")
    return wav_path


def resolve_golden_wav_for_text(text_path: PathLike, golden_dir: Path = GOLDEN_DIR) -> Optional[Path]:
    """Return golden WAV for a local text script when a website demo mapping exists."""
    stem = Path(text_path).stem.lower()
    demo_id = _TEXT_TO_GOLDEN_ID.get(stem)
    if demo_id is None:
        # Common alias: 1p_ch2en.txt ↔ website 1p_CH2EN
        if stem == "1p_ch2en":
            demo_id = "1p_CH2EN"
        else:
            return None
    wav_path = golden_dir / get_golden_demo(demo_id).wav_filename
    return wav_path if wav_path.is_file() else None


def ensure_golden_audios(
    golden_dir: Optional[PathLike] = None,
    *,
    download: bool = True,
    download_transcripts: bool = True,
) -> Path:
    root = Path(golden_dir) if golden_dir is not None else GOLDEN_DIR
    missing = [entry for entry in GOLDEN_DEMOS if not (root / entry.wav_filename).is_file()]
    if missing and download:
        download_golden_audios(root, download_transcripts=download_transcripts)
    elif missing:
        names = ", ".join(e.wav_filename for e in missing)
        raise FileNotFoundError(
            f"Missing golden audio under {root}: {names}. " "Call ensure_golden_audios(download=True) to fetch them."
        )
    return root.resolve()
