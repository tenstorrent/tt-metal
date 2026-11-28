# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.blackhole.llama_nemotron_rerank_1b.reference.hf_reference import load_reference, load_tokenizer
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import MODEL_ID
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.weights import load_state_dict


@pytest.fixture(scope="session")
def hf_model():
    return load_reference()


@pytest.fixture(scope="session")
def tokenizer():
    return load_tokenizer()


@pytest.fixture(scope="session")
def state_dict():
    return load_state_dict(MODEL_ID)
