# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One decoder layer on device vs the HF layer with bidirectional attention and a padding mask."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model import NemotronReranker
from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import RerankConfig


@pytest.mark.parametrize("batch, seq_len, n_real", [(1, 128, 100), (2, 512, 300)])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_decoder_layer(device, hf_model, state_dict, batch, seq_len, n_real):
    cfg = RerankConfig(n_layers=1)
    model = NemotronReranker(device, cfg, state_dict, cache_dir=None)
    torch.manual_seed(0)
    x = torch.randn(batch, seq_len, cfg.dim) * 0.5
    attn_mask = torch.zeros(batch, 1, seq_len, seq_len)
    attn_mask[..., n_real:] = torch.finfo(torch.float32).min
    pos = torch.arange(seq_len).unsqueeze(0)
    hf = hf_model.model
    with torch.no_grad():
        pe = hf.rotary_emb(x, pos)
        ref = hf.layers[0](x, attention_mask=attn_mask, position_ids=pos, position_embeddings=pe)
        ref = ref[0] if isinstance(ref, tuple) else ref

    host, _ = model.prepare_host_inputs([[0] * n_real] * batch, seq_len=seq_len)
    _, valid, block_map = model.to_device(host)
    mask = model._segment_mask(valid)
    rope = model.rot_mats(seq_len)
    tt_x = ttnn.from_torch(
        x.reshape(1, 1, batch * seq_len, cfg.dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    layer = model.layers[0]
    a = model._attention(layer, model._norm(tt_x, layer.attn_norm), batch, seq_len, mask, block_map, rope)
    h = ttnn.add(tt_x, a)
    out = ttnn.add(h, model._mlp(layer, model._norm(h, layer.ffn_norm)))
    got = ttnn.to_torch(out).float().reshape(batch, seq_len, cfg.dim)
    # Padded query rows are not used downstream; compare the real tokens only.
    ok, pcc = comp_pcc(ref[:, :n_real], got[:, :n_real], 0.99)
    print(f"layer PCC {pcc}")
    assert ok, pcc
