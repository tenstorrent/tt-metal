# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for TrOCR generation config handling.

`models.experimental.trocr.trocr_generate_utils` parses the id-based decoding fields out of the generation
config and then builds no logits processors at all, so before this change a config that set
`repetition_penalty`, `no_repeat_ngram_size`, `suppress_tokens`, `begin_suppress_tokens`,
`forced_decoder_ids` or `min_length` produced different text than HF `generate()` with no warning, and a
non-greedy mode (`do_sample`, `num_beams > 1`) fell off the end of `generate()` and returned `None`.
`generate()` must now raise for those settings and keep working for the default greedy config.

No device and no download: `generate()` is driven with a stub model and the module's TTNN imports are filled in
with stubs when the host build has no ttnn Python package (a real package is left untouched).
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _ttnn_is_usable():
    """True only when a built TTNN Python package is importable on this host."""
    try:
        import ttnn
        from ttnn.device import Arch  # noqa: F401
    except Exception:
        return False
    # The source tree carries a bare `ttnn/` directory that resolves as a namespace package without the
    # compiled API, so check a name every real build defines.
    return hasattr(ttnn, "TILE_LAYOUT")


def _import_trocr_generate_utils(monkeypatch):
    """Import the module, standing in for TTNN pieces the host build does not ship."""
    import importlib

    if not _ttnn_is_usable():
        for name in ("ttnn", "ttnn.device", "ttnn.comparison", "tt_lib", "tt_lib.fallback_ops"):
            stub = MagicMock(name=name)
            monkeypatch.setitem(sys.modules, name, stub)
            if "." in name:
                parent_name, child = name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    monkeypatch.setattr(parent, child, stub, raising=False)
        monkeypatch.delitem(sys.modules, "models.common.utility_functions", raising=False)
    return importlib.import_module("models.experimental.trocr.trocr_generate_utils")


class _FakeDecoderModel:
    """Just enough of a decoder model for `generate()` to reach its config validation."""

    main_input_name = "pixel_values"

    def __init__(self):
        self.config = SimpleNamespace(is_encoder_decoder=True)
        self.generation_config = None

    def prepare_inputs_for_generation(self, input_ids=None, **kwargs):
        return {"input_ids": input_ids}

    def forward(self, pixel_values=None, **kwargs):
        return None


def _loader(monkeypatch_module):
    model = _FakeDecoderModel()
    mixin = monkeypatch_module.GenerationMixin(
        config=model.config,
        state_dict={},
        model=model,
        device=torch.device("cpu"),
    )
    return mixin


def test_default_greedy_config_is_accepted(monkeypatch):
    module = _import_trocr_generate_utils(monkeypatch)
    mixin = _loader(module)
    module.GenerationMixin._validate_supported_generation_config(mixin, module.GenerationConfig())


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"repetition_penalty": 1.5}, "repetition_penalty"),
        ({"no_repeat_ngram_size": 2}, "no_repeat_ngram_size"),
        ({"suppress_tokens": [11]}, "suppress_tokens"),
        ({"begin_suppress_tokens": [11]}, "begin_suppress_tokens"),
        ({"min_length": 5}, "min_length"),
        ({"encoder_repetition_penalty": 1.5}, "encoder_repetition_penalty"),
        ({"encoder_no_repeat_ngram_size": 2}, "encoder_no_repeat_ngram_size"),
    ],
)
def test_dropped_logits_fields_raise_instead_of_being_ignored(monkeypatch, kwargs, expected, expect_error):
    """These fields are parsed but never applied, so they must not be accepted silently."""
    module = _import_trocr_generate_utils(monkeypatch)
    mixin = _loader(module)
    config = module.GenerationConfig(**kwargs)
    with expect_error(NotImplementedError, expected):
        module.GenerationMixin._validate_supported_generation_config(mixin, config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"do_sample": True},
        {"num_beams": 4},
        {"num_beam_groups": 2, "num_beams": 4},
        {"penalty_alpha": 0.6},
    ],
)
def test_unsupported_generation_modes_raise(monkeypatch, kwargs, expect_error):
    module = _import_trocr_generate_utils(monkeypatch)
    mixin = _loader(module)
    config = module.GenerationConfig(**kwargs)
    with expect_error(NotImplementedError, "greedy"):
        module.GenerationMixin._validate_supported_generation_config(mixin, config)


def test_forced_decoder_ids_raise(monkeypatch, expect_error):
    """`forced_decoder_ids` is not a transformers 5 attribute, so it arrives through the config dict only."""
    module = _import_trocr_generate_utils(monkeypatch)
    mixin = _loader(module)
    config = module.GenerationConfig(forced_decoder_ids=[[1, 7]])
    with expect_error(NotImplementedError, "forced_decoder_ids"):
        module.GenerationMixin._validate_supported_generation_config(mixin, config)


def test_generate_raises_for_a_kwarg_it_would_have_dropped(monkeypatch, expect_error):
    """The user-visible path: the kwarg reaches `generate()`, which used to accept it and ignore it."""
    module = _import_trocr_generate_utils(monkeypatch)
    mixin = _loader(module)
    pixel_values = torch.zeros(1, 3, 4, 4)
    with expect_error(NotImplementedError, "repetition_penalty"):
        module.GenerationMixin.generate(
            mixin, pixel_values, generation_config=module.GenerationConfig(), repetition_penalty=1.5
        )


def test_generate_raises_for_do_sample(monkeypatch, expect_error):
    """`do_sample=True` used to fall through generate() and return None."""
    module = _import_trocr_generate_utils(monkeypatch)
    mixin = _loader(module)
    pixel_values = torch.zeros(1, 3, 4, 4)
    with expect_error(NotImplementedError, "do_sample"):
        module.GenerationMixin.generate(
            mixin, pixel_values, generation_config=module.GenerationConfig(), do_sample=True
        )
