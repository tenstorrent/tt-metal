# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision fixtures. The device (`eg2_device`, module scope) and snapshot come from the model conftest."""

import pytest

from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import LAYER_CASES, load_case
from models.demos.blackhole.embedding_gemma2.tt.vision.config import VisionConfig
from models.demos.blackhole.embedding_gemma2.tt.vision.weights import load_vision_config_dicts, load_vision_state_dict


@pytest.fixture(scope="session")
def vision_state_dict(eg2_snapshot):
    return load_vision_state_dict(eg2_snapshot)


@pytest.fixture(scope="session")
def vision_config(eg2_snapshot):
    vcfg, text_hidden = load_vision_config_dicts(eg2_snapshot)
    return VisionConfig.from_hf(vcfg, text_hidden)


@pytest.fixture(scope="session")
def hf_vision(eg2_snapshot):
    from models.demos.blackhole.embedding_gemma2.reference.hf_vision_reference import load_reference_vision

    tower, embed, _ = load_reference_vision(eg2_snapshot)
    return tower, embed


@pytest.fixture(scope="session")
def vision_case(eg2_snapshot):
    """vision_case(name) -> (pixel_values [B,N,768], position_ids [B,N,2]) from the HF processor."""

    def get(case):
        return load_case(str(eg2_snapshot), case)

    return get


@pytest.fixture(scope="session")
def vision_reference_for(hf_vision, vision_case):
    """vision_reference_for(case) -> (pixel_values, position_ids, VisionReferenceOutputs).

    Records every layer's in/out; attention/MLP in/out for layers 0, 7 and 15 on LAYER_CASES.
    """
    from models.demos.blackhole.embedding_gemma2.reference.hf_vision_reference import run_vision_reference

    tower, embed = hf_vision
    cache = {}

    def get(case):
        if case not in cache:
            if len(cache) >= 3:
                cache.pop(next(iter(cache)))
            pv, pid = vision_case(case)
            sub = (0, 7, 15) if case in LAYER_CASES else ()
            cache[case] = (pv, pid, run_vision_reference(tower, embed, pv, pid, capture_sublayers=sub))
        return cache[case]

    return get


@pytest.fixture(scope="module")
def tt_vision_model(eg2_device, vision_state_dict, vision_config, eg2_cache_dir):
    from models.demos.blackhole.embedding_gemma2.tt.vision.vision_model import VisionModel

    return VisionModel.from_state_dict(vision_state_dict, vision_config, eg2_device, cache_dir=eg2_cache_dir)


@pytest.fixture(scope="module")
def tt_vision_encoder(eg2_device, tt_vision_model):
    from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder

    return VisionEncoder(tt_vision_model, eg2_device)
