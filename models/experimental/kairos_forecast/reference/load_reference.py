# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Load the pinned Kairos reference (CPU fp32). Runs only in the reference venv, never in python_env.

Pins: GitHub foundation-model-research/Kairos @ 16eb4b98, HF mldi-lab/Kairos_50m @ b788434d.
"""
import hashlib
import importlib.util
import os
from pathlib import Path

REPO_COMMIT = "16eb4b984eb547023187813056a5f289734b9cf6"
HF_REPO = "mldi-lab/Kairos_50m"
HF_REVISION = "b788434dd95f9893be002e6858bfb2afb0e3f360"
SAFETENSORS_SHA256 = "c68503d9a3a3141d25ef40fb71f182556de566989c95331efdab4eeb1b48f177"
SRC = Path("/home/andy/bringup/venvs/kairos/src/Kairos")
SOURCE_FILES = [
    "modeling_kairos.py",
    "moe.py",
    "patch_utils.py",
    "layers.py",
    "tunable_rope_utils.py",
    "modeling_t5_instance_rope.py",
    "utils.py",
    "configuration_kairos.py",
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def snapshot_dir():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(HF_REPO, revision=HF_REVISION, local_files_only=True))


def source_hashes():
    return {f: sha256(SRC / "tsfm/model/kairos" / f) for f in SOURCE_FILES}


def stored_dtypes(snap=None):
    """dtype of every safetensors entry as stored on disk (for MANIFEST; D2 fix 1)."""
    from safetensors import safe_open

    with safe_open(str((snap or snapshot_dir()) / "model.safetensors"), framework="pt") as f:
        return {k: str(f.get_slice(k).get_dtype()) for k in f.keys()}


def manifest_fields(m=None):
    """What MANIFEST records at golden time: torch version, source hashes, stored and loaded dtypes."""
    import torch

    out = {"torch": torch.__version__, "source_hashes": source_hashes(), "stored_dtypes": stored_dtypes()}
    if m is not None:
        out["loaded_dtypes"] = sorted({str(p.dtype) for p in m.parameters()})
    return out


def load(num_threads=8):
    """Returns (model, bias0). bias0 is the router bias right after loading; restore it before every call."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # apex swaps T5LayerNorm for FusedRMSNorm.
    assert importlib.util.find_spec("apex") is None, "apex must not be importable"
    import subprocess

    head = subprocess.check_output(["git", "-C", str(SRC), "rev-parse", "HEAD"], text=True).strip()
    assert head == REPO_COMMIT, head
    snap = snapshot_dir()
    got = sha256(snap / "model.safetensors")
    assert got == SAFETENSORS_SHA256, got

    import torch
    import tsfm.model.kairos  # noqa: F401  registers the kairos config (confirm the path at stage A)
    from tsfm.model.kairos import AutoModel

    torch.set_num_threads(num_threads)
    m = AutoModel.from_pretrained(str(snap), trust_remote_code=True).eval()
    # D2 fix 1: check the dtypes as loaded, before any cast, so a non-fp32 load cannot be cast up silently.
    bad = {n: str(p.dtype) for n, p in m.named_parameters() if p.dtype != torch.float32}
    bad.update({n: str(b.dtype) for n, b in m.named_buffers() if b.is_floating_point() and b.dtype != torch.float32})
    assert not bad, f"non-fp32 after load: {bad}"
    # All fp32 already, so .float() is a no-op kept only as a guard against a later edit of the check above.
    m = m.float()
    bias0 = m.patch.moe.gate.bias.detach().clone()
    return m, bias0


def restore_bias(m, bias0):
    import torch

    with torch.no_grad():
        m.patch.moe.gate.bias.copy_(bias0)
