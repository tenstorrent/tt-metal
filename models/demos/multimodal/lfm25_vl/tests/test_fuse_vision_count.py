# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""CPU-only check of the placeholder/feature count guard in the vision merge of lfm25_vl and gemma3 (ttnn is stubbed)."""

import importlib
import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

_COMMON_STUBS = [
    "ttnn",
    "models.tt_transformers",
    "models.tt_transformers.tt",
    "models.tt_transformers.tt.model",
    "models.tt_transformers.tt.generator",
    "models.tt_transformers.tt.common",
]

_MODELS = {
    "lfm25_vl": (
        "models.demos.multimodal.lfm25_vl.tt.e2e_model",
        ["models.demos.multimodal.lfm25_vl.tt.decoder", "models.demos.multimodal.lfm25_vl.tt.vision_model"],
    ),
    "gemma3": (
        "models.demos.multimodal.gemma3.tt.gemma_e2e_model",
        [
            "models.common.sampling.generator",
            "models.common.utility_functions",
            "models.demos.multimodal.gemma3.tt.gemma_vision_model",
        ],
    ),
}


@pytest.fixture(params=list(_MODELS))
def e2e_cls(request):
    module_name, extra_stubs = _MODELS[request.param]
    # patch.dict restores sys.modules on exit, including every module imported inside the block.
    with mock.patch.dict("sys.modules"):
        for name in _COMMON_STUBS + extra_stubs:
            sys.modules[name] = mock.MagicMock()
        sys.modules["models.tt_transformers.tt.model"].Transformer = type("Transformer", (), {})
        sys.modules["models.tt_transformers.tt.generator"].Generator = type("Generator", (), {})
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
        for obj in vars(module).values():
            if isinstance(obj, type) and "_fuse_vision_into_text_embeddings" in vars(obj):
                yield obj
                return
        raise AssertionError("no class defines _fuse_vision_into_text_embeddings")


def _fuse(cls, tokens, features, **kwargs):
    self = SimpleNamespace(args=SimpleNamespace(image_token_index=7))
    embd = torch.zeros(1, tokens.shape[-1], 4)
    return cls._fuse_vision_into_text_embeddings(self, tokens, embd, features, **kwargs)


def test_matching_counts_are_spliced(e2e_cls):
    tokens = torch.tensor([[1, 7, 7, 2]])
    out = _fuse(e2e_cls, tokens, torch.arange(8, dtype=torch.float32).reshape(2, 4) + 1)
    assert out[0, 1, 0] == 1 and out[0, 2, 0] == 5


@pytest.mark.parametrize("n_features", [1, 3])
def test_count_mismatch_raises(e2e_cls, n_features, expect_error):
    tokens = torch.tensor([[1, 7, 7, 2]])
    with expect_error(ValueError, "Image features and image tokens do not match"):
        _fuse(e2e_cls, tokens, torch.ones(n_features, 4))


def test_count_check_can_be_skipped_for_a_partial_prompt(e2e_cls):
    # A cached-prefix or later chunk holds only some placeholders while the features cover the whole prompt.
    tokens = torch.tensor([[1, 7, 2, 2]])
    out = _fuse(e2e_cls, tokens, torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1, check_counts=False)
    assert out[0, 1, 0] == 1
