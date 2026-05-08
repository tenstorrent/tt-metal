# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal glue for Mistral-Small-4-119B via ``create_multimodal_model`` (``simple_vision_demo``).

Exercises ModelArgs + checkpoint standardization (vision tower, projector, outer norm/embed/lm_head)
and builds ``MistralTransformer`` with ``n_layers=0`` when the checkpoint has no text decoder tensors
(vision-only safetensors slice used when full ``from_pretrained`` fails on FP8 hubs).

Requires ``HF_MODEL`` pointing at a 119B hub (e.g. ``mistralai/Mistral-Small-4-119B-2603``), ``TT_CACHE_PATH``,
and a device mesh (see ``test_pixtral_transformer.py``).
"""

import os
import re

import pytest

import ttnn
from models.tt_transformers.demo.simple_vision_demo import MISTRAL_SMALL_4_119B_BASE, create_multimodal_model
from models.tt_transformers.tt.common import get_base_model_name
from models.tt_transformers.tt.multimodal.mistral_24b.mistral_e2e_model import MistralTransformer


def _hf_points_at_mistral_small_4_119b():
    hf = os.environ.get("HF_MODEL", "")
    if not hf:
        return False
    tail = hf.strip("/").split("/")[-1]
    return get_base_model_name(tail) == MISTRAL_SMALL_4_119B_BASE


pytestmark = pytest.mark.skipif(
    not _hf_points_at_mistral_small_4_119b(),
    reason="Set HF_MODEL to a Mistral-Small-4-119B checkpoint (e.g. mistralai/Mistral-Small-4-119B-2603)",
)


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "P150x8": (1, 8)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_create_multimodal_model_mistral_small_4_119b(mesh_device):
    tt_model_args, model, checkpoint = create_multimodal_model(
        mesh_device=mesh_device,
        max_batch_size=1,
        max_seq_len=4096,
        dtype=ttnn.bfloat16,
        use_paged_kv_cache=False,
        checkpoint=None,
    )

    assert isinstance(model, MistralTransformer)
    assert hasattr(model, "vision_model")
    assert tt_model_args.is_multimodal
    assert tt_model_args.is_mistral3_style_pixtral

    has_tt_text_decoder = any(re.match(r"^layers\.\d+\.(attention|feed_forward)", k) for k in checkpoint)
    if not has_tt_text_decoder:
        assert tt_model_args.n_layers == 0, "Vision-only glue should trim instantiated decoder count to 0"

    assert any(k.startswith("vision_tower.") for k in checkpoint) or any(
        "vision_tower" in k for k in checkpoint
    ), "Expected vision_tower keys in standardized checkpoint"
