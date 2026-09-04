# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The connector cache the LTX pipeline actually loads, on the mesh it actually loads it on.

These are the three modules whose tensorbins once held weights that were not the checkpoint's.
`tests/unit/test_cache_key.py` covers the key and manifest logic on a stub; this covers the real
modules, the real checkpoint and the real cache directory — that the pipeline writes a content-
keyed, verifiable artifact, that a second process hits it, and that the quarantined poison cannot
be served in its place.
"""

from __future__ import annotations

import os

import pytest

import ttnn

from ....encoders.gemma.encoder_pair import _read_connector_checkpoint
from ....pipelines.ltx.pipeline_ltx import LTXPipeline
from ....utils import cache

POISONED = os.path.expanduser("~/.cache/tt-dit-ltxrt/ltx-2.3-22b-distilled-1.1")


def _ckpt() -> str | None:
    path = os.environ.get("LTX_CHECKPOINT")
    return path if path and os.path.exists(path) else None


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((4, 8), {"l1_small_size": 8192, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="4x8")],
    indirect=["mesh_device", "device_params"],
)
def test_connector_cache_is_content_keyed(*, mesh_device):
    ckpt = _ckpt()
    gemma = os.environ.get("GEMMA_PATH", "")
    if not ckpt:
        pytest.skip("LTX checkpoint not found")

    # checkpoint_name=None keeps the 22B transformer out of this test; the encoder pair is then
    # pointed at the real checkpoint so it keys exactly what the pipeline keys.
    pipe = LTXPipeline.create_pipeline(
        mesh_device, checkpoint_name=None, gemma_path=gemma, mode="av", dynamic_load=False
    )
    pair = pipe.gemma_encoder_pair
    pair.checkpoint_name = ckpt
    pair.load_embeddings_connectors(lambda: _read_connector_checkpoint(ckpt))

    name = os.path.basename(ckpt).removesuffix(".safetensors")
    modules = {
        ("feature_extractor", "bf16"): pair.feature_extractor,
        ("video_connector", "float32"): pair.video_connector,
        ("audio_connector", "float32"): pair.audio_connector,
    }

    for (subfolder, dtype), module in modules.items():
        key = cache.content_key(module, [ckpt])
        cache_dir = cache.model_cache_dir(
            model_name=name,
            subfolder=subfolder,
            parallel_config=pair.parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            dtype=dtype,
            content=key,
        )

        # What the pipeline just wrote is what the next process will read.
        assert cache_dir.name.endswith(f"_c{key}"), f"{subfolder} is not content-keyed"
        assert cache._cache_is_complete(cache_dir, module, key), f"{subfolder} would not hit on reload"

        # The poisoned artifact is a different directory, and would be refused even if it weren't.
        poisoned = os.path.join(POISONED, f"{subfolder}.stale-jul7", cache_dir.name.removesuffix(f"_c{key}"))
        assert str(cache_dir) != poisoned
        if os.path.isdir(poisoned):
            assert not cache._cache_is_complete(poisoned, module, key), f"{subfolder} poison is still servable"
