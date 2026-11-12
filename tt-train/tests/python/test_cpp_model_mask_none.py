# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import ttnn
import ttml
from ttml.common.utils import build_causal_mask

SEQ = 64
VOCAB = 96


def _gpt2():
    cfg = ttml.models.gpt2.GPT2TransformerConfig()
    cfg.num_heads = 2
    cfg.embedding_dim = 64
    cfg.num_blocks = 1
    cfg.vocab_size = VOCAB
    cfg.max_sequence_length = SEQ
    cfg.dropout_prob = 0.0
    return ttml.models.gpt2.create_gpt2_model(cfg)


def _llama():
    cfg = ttml.models.llama.CppLlamaConfig()
    cfg.num_heads = 2
    cfg.num_groups = 1
    cfg.embedding_dim = 64
    cfg.num_blocks = 1
    cfg.vocab_size = VOCAB
    cfg.max_sequence_length = SEQ
    cfg.dropout_prob = 0.0
    return ttml.models.llama.create_cpp_llama_model(cfg)


@pytest.mark.requires_device
@pytest.mark.parametrize("make_model", [_gpt2, _llama], ids=["gpt2", "llama"])
def test_mask_none_matches_dense_causal_mask(make_model):
    model = make_model()
    model.eval()
    tokens = np.random.RandomState(0).randint(0, VOCAB, size=(2, 1, 1, SEQ)).astype(np.uint32)
    x = ttml.autograd.Tensor.from_numpy(tokens, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    mask = ttml.autograd.Tensor.from_numpy(
        build_causal_mask(SEQ), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16
    )

    dense = model(x, mask).to_numpy(ttnn.DataType.FLOAT32)
    causal = model(x, None).to_numpy(ttnn.DataType.FLOAT32)
    ttml.autograd.AutoContext.get_instance().reset_graph()

    assert dense.shape == causal.shape
    pcc = np.corrcoef(dense.ravel(), causal.ravel())[0, 1]
    assert pcc > 0.999, f"PCC {pcc}"
