# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end encoder parity: TT ``DiffusionGemmaEncoderTextModel`` vs the actual
HF ``DiffusionGemmaEncoderTextModel``.

Tiny config (256 hidden, 6 layers, 8 experts) so the test stays bounded.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_text_encoder.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma._state_utils import per_layer_moe_substates
from ....models.transformers.diffusion_gemma.text_encoder import DiffusionGemmaEncoderTextModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import local_device_to_torch
from ....utils.test import line_params, ring_params

# Text encoder = embed + N × DiffusionGemmaLayer + final norm. Extrapolating from one
# layer (PCC 99.94%, max abs 0.53) over 6 layers, drift compounds; also PCC-of-composition
# should stay above 0.999 as long as each layer is >=0.999. Not yet run — set to match
# the layer-test tolerance and tighten with observed data.
PCC_THRESHOLD = 0.999
ALLCLOSE_ATOL = 5.5e-1
ALLCLOSE_RTOL = 5e-2


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_tp2"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [128])
def test_encoder_text_model(
    mesh_device: ttnn.MeshDevice, tp_axis: int, num_links: int, topology: ttnn.Topology, seq_len: int
) -> None:
    """TT encoder text model vs HF DiffusionGemmaEncoderTextModel."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaEncoderTextModel as HFEncoderTextModel,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextConfig

    torch.manual_seed(0)
    dtype = torch.float32

    # Real Gemma4 hyperparameters — the demos/gemma4 sparse_matmul kernel is compiled for
    # these shapes; substantially smaller (e.g. hidden=256) causes it to hang on shape-
    # mismatched compiled binaries. Weights are still random-init — no HF checkpoint needed.
    hf_config = DiffusionGemmaTextConfig(
        vocab_size=1024,
        hidden_size=2816,
        intermediate_size=2816,
        moe_intermediate_size=2112,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_experts=8,
        top_k_experts=4,
        num_hidden_layers=6,
        sliding_window=1024,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=8192,
        pad_token_id=0,
    )
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if hf_config.num_global_key_value_heads % tp_factor != 0:
        pytest.skip(f"num_global_kv_heads={hf_config.num_global_key_value_heads} != tp_factor={tp_factor}")

    # HF reference encoder.
    hf_encoder = HFEncoderTextModel(hf_config).to(dtype).eval()

    # Boost router.proj.weight in each layer so softmax is peaked. Default init N(0, 0.02)
    # gives near-uniform softmax → bf16 vs fp32 disagree on near-tie topk selections. Different
    # experts get picked, and MoE outputs diverge significantly. This is a random-init
    # artifact — real trained Gemma4 routing is peaked. Same trick as
    # models/demos/gemma4/tests/unit/test_moe.py:51 and models/tt_dit/tests/models/diffusion_gemma/test_moe.py.
    with torch.no_grad():
        for layer in hf_encoder.layers:
            layer.router.proj.weight.normal_(0, 1.0)

    # Inputs.
    input_ids = torch.randint(low=1, high=hf_config.vocab_size, size=(B, seq_len), dtype=torch.long)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        hf_out = hf_encoder(
            input_ids=input_ids,
            attention_mask=None,  # HF auto-builds the sliding+full mask dict from the config.
            position_ids=position_ids,
            past_key_values=None,
        )
        hf_last = hf_out.last_hidden_state  # [B, S, hidden]

    # Re-build the exact same mask dict HF used (so TT consumes identical inputs).
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    inputs_embeds_for_mask = hf_encoder.embed_tokens(input_ids)
    mask_kwargs = {
        "config": hf_config,
        "inputs_embeds": inputs_embeds_for_mask,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    hf_masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

    # TT setup.
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    hf_state = hf_encoder.state_dict()
    moe_state_dicts = per_layer_moe_substates(hf_state, num_layers=hf_config.num_hidden_layers)

    tt_encoder = DiffusionGemmaEncoderTextModel(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        layer_types=list(hf_config.layer_types),
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        num_global_key_value_heads=hf_config.num_global_key_value_heads,
        head_dim=hf_config.head_dim,
        global_head_dim=hf_config.global_head_dim,
        sliding_window=hf_config.sliding_window,
        num_experts=hf_config.num_experts,
        top_k_experts=hf_config.top_k_experts,
        moe_intermediate_size=hf_config.moe_intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        max_position_embeddings=hf_config.max_position_embeddings,
        sliding_rope_theta=hf_config.rope_parameters["sliding_attention"]["rope_theta"],
        full_rope_theta=hf_config.rope_parameters["full_attention"]["rope_theta"],
        full_partial_rotary_factor=hf_config.rope_parameters["full_attention"]["partial_rotary_factor"],
        pad_token_id=hf_config.pad_token_id,
        moe_state_dicts=moe_state_dicts,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        num_links=num_links,
        topology=topology,
    )
    tt_encoder.load_state_dict(hf_state)

    # Upload inputs / masks. HF's create_causal_mask returns a BOOLEAN mask under the SDPA
    # attention path (True=attend, False=masked). ttnn.transformer.scaled_dot_product_attention
    # expects an ADDITIVE mask (0.0=attend, -inf=masked). Cast boolean → additive before upload;
    # for float masks already in additive form (0/-inf), just pass through as bfloat16.
    tt_input_ids = ttnn.from_torch(input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_masks = {}
    for layer_type, m in hf_masks.items():
        if m is None:
            tt_masks[layer_type] = None
            continue
        if m.dtype == torch.bool:
            m = torch.where(m, torch.tensor(0.0), torch.tensor(float("-inf")))
        tt_masks[layer_type] = ttnn.from_torch(
            m.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )

    tt_h, _per_layer_kv = tt_encoder(tt_input_ids, position_ids, tt_masks)
    tt_h_torch = local_device_to_torch(tt_h).squeeze(0)

    # Diagnostic: run the TT encoder layer-by-layer, comparing each layer's output vs the
    # HF equivalent. Localizes which layer introduces divergence.
    def _pcc_pair(a, b):
        a = a.detach().flatten().to(torch.float64)
        b = b.detach().flatten().to(torch.float64)
        cov = torch.cov(torch.stack([a, b])).numpy()
        return (cov[0, 1] / (math.sqrt(cov[0, 0]) * math.sqrt(cov[1, 1]))).item()

    import math

    with torch.no_grad():
        hf_full = hf_encoder(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
            output_hidden_states=True,
        )
    hf_hidden_all = hf_full.hidden_states  # tuple: [initial_embed, after_layer0, ..., after_layerN]
    logger.info(f"HF hidden_states count = {len(hf_hidden_all)}")

    # Run TT layer by layer and log per-layer PCC.
    tt_h_layer = tt_encoder.embed_tokens(tt_input_ids)
    tt_embed_torch = local_device_to_torch(tt_h_layer).float()
    if tt_embed_torch.ndim == 4:
        tt_embed_torch = tt_embed_torch.squeeze(0)
    logger.info(
        f"[embed] TT shape={tuple(tt_embed_torch.shape)}, HF shape={tuple(hf_hidden_all[0].shape)}, PCC={_pcc_pair(hf_hidden_all[0], tt_embed_torch)*100:.4f}%"
    )
    if len(tt_h_layer.shape) == 3:
        tt_h_layer = ttnn.unsqueeze(tt_h_layer, 0)
    cos_sin = {lt: tt_encoder.rope.get_cos_sin(lt, position_ids) for lt in set(tt_encoder.layer_types)}
    for i in range(tt_encoder.num_hidden_layers):
        lt = tt_encoder.layer_types[i]
        cos, sin = cos_sin[lt]
        tt_h_layer, _, _ = tt_encoder.layers[i](
            tt_h_layer,
            cos,
            sin,
            attention_mask=tt_masks.get(lt),
            encoder_kv=None,
        )
        tt_h_i = local_device_to_torch(tt_h_layer).float()
        if tt_h_i.ndim == 4:
            tt_h_i = tt_h_i.squeeze(0)
        hf_h_i = hf_hidden_all[i + 1].float()
        logger.info(
            f"[chained layer {i} type={lt}] TT shape={tuple(tt_h_i.shape)}, HF shape={tuple(hf_h_i.shape)}, PCC={_pcc_pair(hf_h_i, tt_h_i)*100:.4f}%"
        )

    # Intrinsic per-layer PCC: feed each TT layer with HF's ideal input (isolates each layer's
    # forward from compounding drift). If intrinsic PCC stays high, the chained drift is
    # "natural" bf16 compounding; if intrinsic PCC decays, there's a genuine per-layer bug.
    from ....utils.tensor import bf16_tensor

    for i in range(tt_encoder.num_hidden_layers):
        lt = tt_encoder.layer_types[i]
        cos, sin = cos_sin[lt]
        tt_ideal_input = bf16_tensor(hf_hidden_all[i].unsqueeze(0), device=mesh_device)
        tt_out, _, _ = tt_encoder.layers[i](
            tt_ideal_input,
            cos,
            sin,
            attention_mask=tt_masks.get(lt),
            encoder_kv=None,
        )
        tt_out_torch = local_device_to_torch(tt_out).float()
        if tt_out_torch.ndim == 4:
            tt_out_torch = tt_out_torch.squeeze(0)
        hf_out_i = hf_hidden_all[i + 1].float()
        logger.info(f"[intrinsic layer {i} type={lt}] PCC={_pcc_pair(hf_out_i, tt_out_torch)*100:.4f}%")

    logger.info(f"hf_last: {hf_last.shape}, tt_h: {tt_h_torch.shape}")
    assert_quality(hf_last, tt_h_torch, pcc=PCC_THRESHOLD)

    abs_diff = (hf_last - tt_h_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(hf_last, tt_h_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
