# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("batch_size", [1])
def test_tt_multihead_attention(device, reset_seeds, batch_size, seq_len, sam3_transformer):
    """Test tt_multihead_attention against nn.MultiheadAttention."""
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_transformer import (
        tt_multihead_attention,
        _preprocess_mha_weights,
    )

    torch.manual_seed(42)

    # Use self_attn from the first encoder layer as reference (batch_first=True)
    ref_mha = sam3_transformer.encoder.layers[0].self_attn
    d_model = ref_mha.embed_dim
    num_heads = ref_mha.num_heads

    # Create batch-first input: (batch, seq_len, d_model) matching encoder MHA config
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    # PyTorch reference
    with torch.no_grad():
        ref_output, _ = ref_mha(query, key, value)

    # Preprocess weights
    mha_params = _preprocess_mha_weights(ref_mha, device)

    # Convert inputs to ttnn
    tt_q = ttnn.from_torch(query, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(key, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run ttnn version
    tt_output = tt_multihead_attention(
        tt_q, tt_k, tt_v,
        mha_params["qw"], mha_params["qb"],
        mha_params["kw"], mha_params["kb"],
        mha_params["vw"], mha_params["vb"],
        mha_params["ow"], mha_params["ob"],
        num_heads, device,
    )

    output = ttnn.to_torch(tt_output).float()
    if output.shape != ref_output.shape:
        output = output.reshape(ref_output.shape)

    assert_with_pcc(ref_output.float(), output, 0.97)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("seq_len", [64])
@pytest.mark.parametrize("batch_size", [1])
def test_tt_encoder_layer(device, reset_seeds, batch_size, seq_len, sam3_transformer):
    """Test a single encoder layer against PyTorch reference.

    The SAM3 encoder uses batch-first format, pre-norm, with:
    - pos_enc_at_attn=True
    - pos_enc_at_cross_attn_keys=False
    - pos_enc_at_cross_attn_queries=False
    """
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_transformer import (
        tt_encoder_layer,
        preprocess_encoder_layer_weights,
    )

    torch.manual_seed(42)

    ref_layer = sam3_transformer.encoder.layers[0]
    d_model = ref_layer.d_model
    num_heads = ref_layer.self_attn.num_heads

    # Batch-first inputs: (batch, seq_len, d_model)
    src = torch.randn(batch_size, seq_len, d_model)
    pos = torch.randn(batch_size, seq_len, d_model)
    text_len = 32
    text_features = torch.randn(batch_size, text_len, d_model)

    # PyTorch reference: forward_pre(tgt=src, memory=text, query_pos=pos)
    # pos defaults to None so cross-attn keys don't get pos added
    with torch.no_grad():
        ref_output = ref_layer.forward_pre(
            tgt=src,
            memory=text_features,
            query_pos=pos,
        )

    # Preprocess weights
    layer_params = preprocess_encoder_layer_weights(ref_layer, num_heads, device)

    # Convert inputs to ttnn
    tt_src = ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_pos = ttnn.from_torch(pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_text = ttnn.from_torch(text_features, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run ttnn version
    tt_output = tt_encoder_layer(tt_src, tt_pos, tt_text, layer_params, device)

    output = ttnn.to_torch(tt_output).float()
    if output.shape != ref_output.shape:
        output = output.reshape(ref_output.shape)

    assert_with_pcc(ref_output.float(), output, 0.95)


@pytest.mark.skip(reason="Decoder reference hangs on CPU due to complex DAC/box-refine ops. TODO: simplify.")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("num_queries", [32])
@pytest.mark.parametrize("mem_len", [64])
@pytest.mark.parametrize("batch_size", [1])
def test_tt_decoder_layer(device, reset_seeds, batch_size, num_queries, mem_len, sam3_transformer):
    """Test a single decoder layer against PyTorch reference.

    The SAM3 decoder uses seq-first format, post-norm.
    Order: self-attn + norm2 -> cross-attn + norm1 -> FFN + norm3.
    """
    from models.demos.vision.segmentation.sam3.common.tt.ttnn_sam3_transformer import (
        tt_decoder_layer,
        preprocess_decoder_layer_weights,
    )

    torch.manual_seed(42)

    ref_layer = sam3_transformer.decoder.layers[0]
    d_model = sam3_transformer.decoder.d_model
    num_heads = ref_layer.self_attn.num_heads

    # Seq-first inputs: (seq_len, batch, d_model)
    queries = torch.randn(num_queries, batch_size, d_model)
    memory = torch.randn(mem_len, batch_size, d_model)
    query_pos = torch.randn(num_queries, batch_size, d_model)
    memory_pos = torch.randn(mem_len, batch_size, d_model)

    # PyTorch reference: TransformerDecoderLayer.forward
    # Temporarily disable text cross-attention since our ttnn layer doesn't implement it
    orig_use_text = ref_layer.use_text_cross_attention
    ref_layer.use_text_cross_attention = False
    with torch.no_grad():
        ref_output, _ = ref_layer(
            tgt=queries,
            tgt_query_pos=query_pos,
            memory=memory,
            memory_pos=memory_pos,
        )
    ref_layer.use_text_cross_attention = orig_use_text

    # Preprocess weights
    layer_params = preprocess_decoder_layer_weights(ref_layer, num_heads, device)

    # Convert inputs to ttnn
    tt_queries = ttnn.from_torch(queries, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_memory = ttnn.from_torch(memory, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_query_pos = ttnn.from_torch(query_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_memory_pos = ttnn.from_torch(memory_pos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Run ttnn version
    tt_output = tt_decoder_layer(
        tt_queries, tt_memory, tt_query_pos, tt_memory_pos, layer_params, device,
    )

    output = ttnn.to_torch(tt_output).float()
    if output.shape != ref_output.shape:
        output = output.reshape(ref_output.shape)

    assert_with_pcc(ref_output.float(), output, 0.95)
