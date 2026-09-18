# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests: submodule is present and the Amazon Chronos package imports."""

import torch

from models.experimental.chronos_forecast.common.chronos_src import CHRONOS_SRC, ensure_chronos_on_path
from models.experimental.chronos_forecast.common.configs import ChronosModelConfig
from models.experimental.chronos_forecast.reference.pytorch_chronos import (
    ChronosConfig,
    MeanScaleUniformBins,
    create_chronos_config,
)
from models.experimental.chronos_forecast.tt.model import TtChronos


def test_submodule_src_is_checked_out():
    ensure_chronos_on_path()
    assert (CHRONOS_SRC / "chronos" / "__init__.py").is_file()


def test_chronos_package_imports_from_submodule():
    ensure_chronos_on_path()
    import chronos

    assert chronos.__file__.startswith(str(CHRONOS_SRC))
    assert hasattr(chronos, "ChronosPipeline")
    assert hasattr(chronos, "ChronosBoltPipeline")
    assert hasattr(chronos, "Chronos2Pipeline")


def test_tokenizer_roundtrip(reset_seeds):
    config = create_chronos_config(ChronosModelConfig())
    tokenizer = config.create_tokenizer()
    assert isinstance(config, ChronosConfig)
    assert isinstance(tokenizer, MeanScaleUniformBins)

    context = tokenizer.centers.unsqueeze(0)
    scale = torch.ones((1,))
    token_ids, _, _ = tokenizer._input_transform(context, scale=scale)
    samples = tokenizer.output_transform(token_ids.unsqueeze(1), scale=scale)
    assert torch.equal(samples[0, 0, :], context[0])


def test_tt_chronos_stub_constructs():
    model = TtChronos(device=None)
    assert model.config.prediction_length == 64
