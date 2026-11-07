# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests that model weights can be fetched from the Hugging Face Hub.

tt-train tests pull weights a shared CI cache may lack, so the Hub has to be reachable and
HF_HUB_CACHE writable. CI arranges both in .github/actions/setup-tt-train.
"""

import json

from huggingface_hub import constants, hf_hub_download
from safetensors.numpy import load_file

MODEL_ID = "hf-internal-testing/tiny-random-gpt2"
REVISION = "71034c5d8bde858ff824298bdedc65515b97d2b9"


def _download(filename):
    # A copy already in the cache would otherwise pass even when offline or read-only.
    return hf_hub_download(MODEL_ID, filename, revision=REVISION, force_download=True)


def test_download_tiny_model_from_hub():
    print(f"HF_HUB_CACHE={constants.HF_HUB_CACHE} HF_HUB_OFFLINE={constants.HF_HUB_OFFLINE}")
    with open(_download("config.json")) as f:
        config = json.load(f)
    weights = load_file(_download("model.safetensors"))

    assert weights["transformer.wte.weight"].shape == (config["vocab_size"], config["n_embd"])
    assert weights["transformer.wpe.weight"].shape == (config["n_positions"], config["n_embd"])
