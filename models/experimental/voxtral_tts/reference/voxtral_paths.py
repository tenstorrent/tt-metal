# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where the Voxtral-TTS model files live. Standard library only, so the tokenizer can import it.

The model directory holds consolidated.safetensors, params.json, tekken.json and voice_embedding/.
Resolved as: explicit argument > $VOXTRAL_CKPT (the directory, or its .safetensors) > the Hugging
Face repo, downloaded into the local HF cache. Importing never downloads; `resolve_model_dir` does.
"""

import os

HF_REPO = os.environ.get("VOXTRAL_HF_MODEL", "mistralai/Voxtral-4B-TTS-2603")
MODEL_FILES = ["consolidated.safetensors", "params.json", "tekken.json", "voice_embedding/*.pt"]
CKPT_NAME = "consolidated.safetensors"


def _as_dir(path):
    return os.path.dirname(path) if path.endswith(".safetensors") else path


def local_model_dir(path=None):
    """-> the model directory from `path`, $VOXTRAL_CKPT or an existing HF download, else None.
    Never downloads."""
    for p in (path, os.environ.get("VOXTRAL_CKPT")):
        if p:
            return _as_dir(p)
    try:
        from huggingface_hub import snapshot_download

        d = snapshot_download(HF_REPO, allow_patterns=MODEL_FILES, local_files_only=True)
    except Exception:
        return None
    return d if os.path.exists(os.path.join(d, CKPT_NAME)) else None


def resolve_model_dir(path=None):
    """-> the model directory: `path` > $VOXTRAL_CKPT > download HF_REPO into the local HF cache."""
    d = local_model_dir(path)
    if d is not None:
        return d
    from huggingface_hub import snapshot_download
    from loguru import logger

    logger.info(f"downloading {HF_REPO} (CC BY-NC 4.0, non-commercial) into the Hugging Face cache")
    return snapshot_download(HF_REPO, allow_patterns=MODEL_FILES)


# Import-time defaults: local only, so a missing checkpoint makes these paths not exist rather
# than triggering a download (the tests skip on that).
MODEL_DIR = local_model_dir() or "<no local Voxtral checkpoint: set VOXTRAL_CKPT>"

DOWNLOAD_HINT = """Point $VOXTRAL_CKPT at a directory holding the (CC BY-NC 4.0, non-commercial) model:
    hf download mistralai/Voxtral-4B-TTS-2603 consolidated.safetensors params.json tekken.json \\
        "voice_embedding/*" --local-dir voxtral_model
    export VOXTRAL_CKPT=$PWD/voxtral_model
or construct TtVoxtralPipeline() with neither, which downloads it into the Hugging Face cache."""
