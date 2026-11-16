# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""checkpoint_name() must give one name for every way a checkpoint is addressed.

The warm weight-cache marker records ``model_name``; the demo seeds it from the HF id while vLLM
(under HF_HUB_OFFLINE) hands over the resolved hub snapshot directory. If the two spell the name
differently the vLLM server never sees the marker and cold-loads the full checkpoint every start.
"""

import pytest

from models.common.weight_cache import checkpoint_name

HUB = "/mnt/MLPerf/huggingface/hub"


@pytest.mark.parametrize(
    "model_path, expected",
    [
        ("google/gemma-4-31B-it", "gemma-4-31B-it"),
        ("google/gemma-4-31B-it/", "gemma-4-31B-it"),
        (f"{HUB}/models--google--gemma-4-31B-it/snapshots/842da37", "gemma-4-31B-it"),
        (f"{HUB}/models--google--gemma-4-31B-it/snapshots/842da37/", "gemma-4-31B-it"),
        (f"{HUB}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f", "Llama-3.1-8B-Instruct"),
        ("/mnt/MLPerf/tt_dnn-models/google/gemma-4-26B-A4B-it", "gemma-4-26B-A4B-it"),
        ("gemma-4-E2B-it", "gemma-4-E2B-it"),
    ],
)
def test_checkpoint_name(model_path, expected):
    assert checkpoint_name(model_path) == expected


def test_hf_id_and_hub_snapshot_agree():
    hf_id = "google/gemma-4-31B-it"
    snapshot = f"{HUB}/models--google--gemma-4-31B-it/snapshots/842da37"
    assert checkpoint_name(hf_id) == checkpoint_name(snapshot)
