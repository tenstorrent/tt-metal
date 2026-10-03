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


def test_tt_chronos_weights_from_dummy():
    from models.experimental.chronos_forecast.reference.chronos2.model import Chronos2Model as RefModel
    from models.experimental.chronos_forecast.tests.golden_helpers import DUMMY_MODEL_PATH
    from models.experimental.chronos_forecast.tt.model import (
        TtChronosConfig,
        TtChronosWeights,
        tt_chronos_config_from_torch_model,
    )

    model = RefModel.from_pretrained(DUMMY_MODEL_PATH).eval()
    weights = TtChronosWeights.from_torch_model(model)
    config = tt_chronos_config_from_torch_model(model)
    assert isinstance(config, TtChronosConfig)
    assert config.d_model == model.config.d_model
    assert config.num_quantiles == len(model.chronos_config.quantiles)
    assert weights.shared_weight.shape[0] == model.config.vocab_size
    assert len(weights.encoder.blocks) == model.config.num_layers
