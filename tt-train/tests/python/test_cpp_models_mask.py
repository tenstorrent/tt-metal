# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The C++ GPT-2 and Llama models accept mask=None; the fused SDPA then applies the causal mask itself."""

import numpy as np
import pytest

import ttnn
import ttml
from ttml.common.utils import build_causal_mask

VOCAB_SIZE = 64
SEQ_LEN = 128


def create_gpt2():
    config = ttml.models.gpt2.GPT2TransformerConfig()
    config.num_heads = 2
    config.embedding_dim = 64
    config.num_blocks = 2
    config.vocab_size = VOCAB_SIZE
    config.max_sequence_length = SEQ_LEN
    config.dropout_prob = 0.0
    return ttml.models.gpt2.create_gpt2_model(config)


def create_llama():
    config = ttml.models.llama.CppLlamaConfig()
    config.num_heads = 2
    config.num_groups = 1
    config.embedding_dim = 64
    config.num_blocks = 2
    config.vocab_size = VOCAB_SIZE
    config.max_sequence_length = SEQ_LEN
    config.dropout_prob = 0.0
    return ttml.models.llama.create_cpp_llama_model(config)


@pytest.mark.requires_device
@pytest.mark.parametrize("create_model", [create_gpt2, create_llama], ids=["gpt2", "llama"])
def test_cpp_model_without_mask_matches_causal_mask(create_model):
    np.random.seed(0)
    model = create_model()
    model.eval()

    tokens = np.random.randint(0, VOCAB_SIZE, size=(2, 1, 1, SEQ_LEN)).astype(np.uint32)
    inputs = ttml.autograd.Tensor.from_numpy(tokens, layout=ttnn.Layout.ROW_MAJOR, new_type=ttnn.DataType.UINT32)
    mask = ttml.autograd.Tensor.from_numpy(
        build_causal_mask(SEQ_LEN), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )

    logits_with_mask = model(inputs, mask).to_numpy().astype(np.float32)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    logits_without_mask = model(inputs, None).to_numpy().astype(np.float32)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    logits_default = model(inputs).to_numpy().astype(np.float32)
    ttml.autograd.AutoContext.get_instance().reset_graph()

    assert logits_without_mask.shape == logits_with_mask.shape
    np.testing.assert_allclose(logits_without_mask, logits_with_mask, rtol=0.05, atol=0.05)
    np.testing.assert_array_equal(logits_default, logits_without_mask)
