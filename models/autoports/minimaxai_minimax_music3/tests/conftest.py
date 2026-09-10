# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the MiniMax-Music3 tests.

One 1x1 mesh and one ``MusicLLM`` per session: building the 8B backbone on device takes minutes,
so every test shares it. The HF reference model (16 GB bf16 on CPU) is loaded lazily, once.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc"
GENERATED_DIR = MODEL_DIR / "generated"

# Stage 07: backbone decode trace + depth traces + one 36-layer DiT step trace per window shape (two shapes per song
# at most). 90 MB sufficed for the AR traces alone (stages 02-06); 200 MB is the stage-07 default (context contract).
TRACE_REGION_SIZE = int(os.environ.get("MM3_TRACE_REGION_SIZE", 200_000_000))

# Stage 05: the DiT's converted bf16 weights (4.6 GB) are cached by models.tt_dit.utils.cache.load_model
# under TT_DIT_CACHE_DIR (cache hit: 0.7 s instead of 7 s of safetensors -> device conversion).
os.environ.setdefault("TT_DIT_CACHE_DIR", str(GENERATED_DIR / "tt_dit_cache"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long sweeps, excluded from the stage gate (-m 'not slow')")
    config.addinivalue_line("markers", "hardware: needs a Tenstorrent device")


def _device_available() -> bool:
    try:
        return ttnn.get_num_devices() > 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def mm3_mesh_device():
    """A 1x1 mesh on the single visible chip (TT_METAL_VISIBLE_DEVICES pins it), program cache on."""
    if not _device_available():
        pytest.skip("no Tenstorrent device visible")
    # No fabric: enabling FABRIC_1D on this multi-chip host makes tt-metal handshake the ethernet
    # routers with the *other* (unused) chips and times out ("Fabric Router Sync: Timeout ... Device 3").
    # tt_transformers' single-device CCL wrapper does not need fabric on a 1x1 mesh.
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=TRACE_REGION_SIZE)
    mesh.enable_program_cache()
    yield mesh
    ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="session")
def music_llm(mm3_mesh_device):
    """The backbone in the stage-02 "functional" policy (the stage 02-04 tests' reference bars were set for it).

    ``MM3_LLM_POLICY`` selects another ``tt/llm.py`` policy and ``MM3_LLM_NUM_LAYERS`` a reduced-layer variant
    (stage 07 profiling harness: tt-perf-report on one decoder layer + norm + LM head; not for correctness tests)."""
    from models.autoports.minimaxai_minimax_music3.tt.llm import MusicLLM

    kwargs = {}
    if os.environ.get("MM3_LLM_POLICY"):
        kwargs["dtype_policy"] = os.environ["MM3_LLM_POLICY"]
    if os.environ.get("MM3_LLM_NUM_LAYERS"):
        kwargs["num_layers"] = int(os.environ["MM3_LLM_NUM_LAYERS"])
    llm = MusicLLM(mm3_mesh_device, **kwargs)
    yield llm
    llm.release()


@pytest.fixture(scope="session")
def hf_model():
    """``Qwen3ForCausalLM`` in bf16 on CPU (loaded once, about 16 GB)."""
    from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R

    torch.set_num_threads(max(8, os.cpu_count() or 8))
    return R.load_hf_qwen3()


@pytest.fixture(scope="session")
def audio_embeddings():
    from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R

    return R.load_audio_embeddings()


@pytest.fixture(scope="session")
def golden():
    """The stage-01 golden tensors that this stage compares against."""
    from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R

    root = R.reference_dir()
    if not (root / "text_ids.pt").is_file():
        pytest.skip(f"golden reference missing under {root}")
    return {
        "text_ids": torch.load(root / "text_ids.pt"),
        "sampled_codes": torch.load(root / "sampled_codes.pt"),
        "llm_layer0": torch.load(root / "llm_layer0.pt"),
    }


@pytest.fixture(scope="session")
def evidence_dir():
    d = DOC_DIR / "llm"
    d.mkdir(parents=True, exist_ok=True)
    return d
