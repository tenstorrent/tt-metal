"""Download the FunAudioLLM/CosyVoice reference repo into ``model_data/CosyVoice_src/``.

Phase-0 setup step (BRINGUP_PLAN.md §7 Phase 0 Task 2). The reference repo is
read-only and is used only to extract op-inventory and to run
``example.py::cosyvoice2_example`` for golden fixtures.

Idempotent: if the target dir already exists at the pinned SHA, this is a no-op.
Downloads pinned tarballs via HTTPS (no subprocess / OS commands).

Run inside the tt-metal env:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal/models/demos/cosyvoice
    python scripts/clone_reference.py
"""
from __future__ import annotations

import io
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

COSYVOICE_SHA = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
COSYVOICE_TARBALL = f"https://github.com/FunAudioLLM/CosyVoice/archive/{COSYVOICE_SHA}.tar.gz"

MATCHA_SHA = "dd9105b34bf2be2230f4aa1e4769fb586a3c824e"
MATCHA_TARBALL = f"https://github.com/shivammehta25/Matcha-TTS/archive/{MATCHA_SHA}.tar.gz"

DEMO_ROOT = Path(__file__).resolve().parent.parent
TARGET = DEMO_ROOT / "model_data" / "CosyVoice_src"
MATCHA_TARGET = TARGET / "third_party" / "Matcha-TTS"
PIN_FILE = TARGET / ".pinned_sha"
PLAN = DEMO_ROOT / "BRINGUP_PLAN.md"


def _download_tarball(url: str) -> bytes:
    print(f"[clone_reference] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "cosyvoice-bringup/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def _extract_tarball(data: bytes, dest: Path, strip_prefix: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.startswith(strip_prefix):
                continue
            rel = member.name[len(strip_prefix) :]
            if not rel or rel.startswith("/"):
                continue
            if ".." in Path(rel).parts:
                continue
            member.name = rel
            tf.extract(member, dest)


def record_sha_in_plan(sha: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"Invalid git SHA format: {sha!r}")
    if not PLAN.exists():
        print(f"[clone_reference] {PLAN.name} not found, skipping SHA recording.")
        return
    marker = "- Repo (CosyVoice pin): https://github.com/FunAudioLLM/CosyVoice @ commit"
    new_line = f"{marker} {sha}"
    text = PLAN.read_text()
    if marker in text:
        text = re.sub(rf"^{re.escape(marker)}.*$", new_line, text, flags=re.MULTILINE)
    else:
        anchor = "- Repo: https://github.com/FunAudioLLM/CosyVoice"
        replacement = f"{anchor}\n{new_line}"
        text = text.replace(anchor, replacement, 1)
    PLAN.write_text(text)


def main() -> int:
    if TARGET.exists() and PIN_FILE.exists():
        stored = PIN_FILE.read_text().strip()
        if stored == COSYVOICE_SHA:
            print(f"[clone_reference] already present at {TARGET}  (SHA {stored[:12]}).")
            record_sha_in_plan(stored)
            return 0

    data = _download_tarball(COSYVOICE_TARBALL)
    _extract_tarball(data, TARGET, f"CosyVoice-{COSYVOICE_SHA}/")
    print(f"[clone_reference] extracted CosyVoice @ {COSYVOICE_SHA[:12]}")

    matcha_data = _download_tarball(MATCHA_TARBALL)
    _extract_tarball(matcha_data, MATCHA_TARGET, f"Matcha-TTS-{MATCHA_SHA}/")
    print(f"[clone_reference] extracted Matcha-TTS @ {MATCHA_SHA[:12]}")

    PIN_FILE.write_text(COSYVOICE_SHA + "\n")
    record_sha_in_plan(COSYVOICE_SHA)

    for rel in ("example.py", "cosyvoice/cli/cosyvoice.py"):
        if not (TARGET / rel).exists():
            print(f"[clone_reference] WARNING: expected file missing: {rel}", file=sys.stderr)
    if not (MATCHA_TARGET / "matcha").exists():
        print("[clone_reference] WARNING: matcha/ package missing in Matcha-TTS", file=sys.stderr)

    print(f"[clone_reference] done. HEAD = {COSYVOICE_SHA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
