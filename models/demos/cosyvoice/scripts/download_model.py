"""Download the CosyVoice2-0.5B checkpoint into a repo-local cache.

Phase-0 setup step (BRINGUP_PLAN.md §7 Phase 0 Task 2). Uses
``huggingface_hub.snapshot_download`` to materialize the full
``FunAudioLLM/CosyVoice2-0.5B`` snapshot under
``model_data/cosyvoice2-0.5B/`` (repo-local, git-ignored). The pulled dir is
usable directly as the ``model_dir`` argument to ``AutoModel(model_dir=...)``
since its layout matches what ``cosyvoice/cli/cosyvoice.py::CosyVoice2`` expects
(``cosyvoice2.yaml``, ``llm.pt``, ``flow.pt``, ``hift.pt``,
``campplus.onnx``, ``speech_tokenizer_v2.onnx``, ``flow.decoder.estimator.fp32.onnx``,
``CosyVoice-BlankEN/``).

Idempotent: if the snapshot is already complete on disk, no-op (the user must
pass ``--force`` to re-download).

Records the HF snapshot revision (commit SHA) into ``BRINGUP_PLAN.md`` §10 so the
next agent has a frozen checkpoint pin.

Run inside the tt-metal env:
    source /root/tt-metal/python_env/bin/activate
    cd /root/tt-metal/models/demos/cosyvoice
    python scripts/download_model.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

REPO_ID = "FunAudioLLM/CosyVoice2-0.5B"
DEMO_ROOT = Path(__file__).resolve().parent.parent
TARGET = DEMO_ROOT / "model_data" / "cosyvoice2-0.5B"
PLAN = DEMO_ROOT / "BRINGUP_PLAN.md"

# Files we expect from the HF snapshot — used for the completeness probe.
EXPECTED_FILES = [
    "cosyvoice2.yaml",
    "llm.pt",
    "flow.pt",
    "hift.pt",
    "campplus.onnx",
    "speech_tokenizer_v2.onnx",
    "flow.decoder.estimator.fp32.onnx",
]

HF_REVISION_LINE = "- HF checkpoint pin: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B @ revision"


def record_revision_in_plan(revision: str) -> None:
    text = PLAN.read_text()
    marker_line = HF_REVISION_LINE
    new_line = f"{marker_line} {revision}"
    if marker_line in text:
        text = re.sub(rf"^{re.escape(marker_line)}.*$", new_line, text, flags=re.MULTILINE)
    else:
        anchor = "- HF: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B"
        replacement = f"{anchor}\n{new_line}"
        text = text.replace(anchor, replacement, 1)
    PLAN.write_text(text)


def snapshot_is_complete() -> bool:
    if not TARGET.exists():
        return False
    for rel in EXPECTED_FILES:
        if not (TARGET / rel).exists():
            return False
    return True


def main() -> int:
    force = "--force" in sys.argv
    if snapshot_is_complete() and not force:
        print(f"[download_model] checkpoint already complete at {TARGET}.")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        TARGET.mkdir(parents=True, exist_ok=True)

    print(f"[download_model] snapshot_download -> {TARGET}")
    # NOTE: snapshot_download with local_dir set returns the local path str,
    # NOT the revision SHA. Fetch the real pinned commit SHA via model_info.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(TARGET),
    )
    print(f"[download_model] snapshot_dir = {local_path}")
    info = model_info(repo_id=REPO_ID)
    revision = info.sha
    print(f"[download_model] snapshot_revision = {revision}")
    record_revision_in_plan(revision)

    # Confirm completeness.
    missing = [rel for rel in EXPECTED_FILES if not (TARGET / rel).exists()]
    if missing:
        print(
            f"[download_model] WARNING: missing expected files after download: {missing}",
            file=sys.stderr,
        )
        return 1
    # Report the SFT-mode open item (BRINGUP_PLAN.md §1.1): whether spk2info.pt shipped.
    spk = TARGET / "spk2info.pt"
    print(f"[download_model] spk2info.pt present: {spk.exists()} (resolves §1.1 SFT-mode open item)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
