# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import transformers

import ttnn


@pytest.mark.requires_fast_runtime_mode_off
def test_codegen_traced_torch_bert():
    model_name = "bert-base-uncased"
    config = transformers.BertConfig.from_pretrained(model_name)
    config.num_hidden_layers = 3
    model = transformers.BertModel.from_pretrained(model_name, config=config).eval()
    input_tensor = torch.randint(0, 1000, (1, 128))

    with ttnn.tracer.trace():
        input_tensor = torch.randint(0, 1000, (1, 128))
        outputs = model(input_tensor)

    ttnn.tracer.codegen(outputs)


@pytest.mark.requires_fast_runtime_mode_off
def test_codegen_traced_torch_bloom():
    model_name = "bigscience/bloom-560m"
    config = transformers.BloomConfig.from_pretrained(model_name)
    config.n_layer = 3
    model = transformers.BloomModel.from_pretrained(model_name, config=config).eval()
    input_tensor = torch.randint(0, 1000, (1, 128))

    with ttnn.tracer.trace():
        input_tensor = torch.randint(0, 1000, (1, 128))
        outputs = model(input_tensor)

    ttnn.tracer.codegen(outputs)
