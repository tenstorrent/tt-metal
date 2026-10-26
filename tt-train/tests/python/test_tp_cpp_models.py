# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The C++ tensor-parallel models must be callable from Python.

``DistributedLlama`` and ``DistributedGPT2Transformer`` only override
``operator()(tensor, optional<mask>)``. Without their own ``__call__`` binding,
``model(x, mask)`` fell through to ``ModuleBase.__call__(tensor, other)`` and raised
"Not implemented", and ``model(x, None)`` matched no overload at all.
"""

import numpy as np
import pytest
import ttml
import ttnn
from ttml.common.utils import build_causal_mask

pytestmark = pytest.mark.requires_device

# At TP=2 every shard stays tile-aligned: one 32-wide head and KV group per device,
# and 32 vocab columns of the LM head.
HIDDEN = VOCAB = 64
SEQ = 32


def _llama():
    cfg = ttml.models.llama.CppLlamaConfig()
    cfg.num_heads = 2
    cfg.num_groups = 2
    cfg.embedding_dim = HIDDEN
    cfg.num_blocks = 1
    cfg.vocab_size = VOCAB
    cfg.max_sequence_length = SEQ
    return ttml.models.distributed.llama.create_llama_model(cfg)


def _gpt2():
    cfg = ttml.models.gpt2.GPT2TransformerConfig()
    cfg.num_heads = 2
    cfg.embedding_dim = HIDDEN
    cfg.num_blocks = 1
    cfg.vocab_size = VOCAB
    cfg.max_sequence_length = SEQ
    cfg.dropout_prob = 0.0
    return ttml.models.distributed.gpt2.create_gpt2_model(cfg)


@pytest.mark.parametrize("make_model", [_llama, _gpt2], ids=["llama", "gpt2"])
def test_mask_none_matches_dense_causal_mask(tp_mesh, make_model):
    model = make_model()
    model.eval()
    tokens = np.random.RandomState(0).randint(0, VOCAB, size=(1, 1, 1, SEQ)).astype(np.uint32)
    x = ttml.autograd.Tensor.from_numpy(tokens, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
    # The LM head output stays vocab-sharded; concatenating the shards gives the full logits.
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 3)

    dense = model(x, build_causal_mask(SEQ, device=True)).to_numpy(composer=composer)
    causal = model(x, None).to_numpy(composer=composer)
    ttml.autograd.AutoContext.get_instance().reset_graph()

    assert dense.shape == causal.shape == (1, 1, SEQ, VOCAB)
    pcc = np.corrcoef(dense.ravel(), causal.ravel())[0, 1]
    assert pcc > 0.999, f"PCC {pcc}"
