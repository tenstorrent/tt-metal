# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Qwen3-VL-8B text encoder for Ideogram 4.0. Ported from the Qwen2.5-VL encoder
# (encoders/qwen25vl) + the two Qwen3 deltas: per-head QK-RMSNorm and no qkv bias.
# Adds the multi-layer feature tap (raw hidden states after the 13 activation
# layers, no final norm) that Ideogram concatenates and feeds to the DiT.
#
# The HF reference is built from CONFIG (AutoModel.from_config) so the full 8B base
# checkpoint is never pulled, then the shipped Ideogram text_encoder weights
# (dequantized fp8) — the weights the pipeline actually uses — are overlaid.
# For text tokens the 3 MRoPE axes share the same position, so MRoPE collapses to
# 1D RoPE and the qwen25vl rope path is exact.
# =============================================================================

import os

import pytest
import torch
import transformers
from loguru import logger
from safetensors.torch import load_file

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, create_rope_tensors
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....reference.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from ....reference.ideogram4.dequant import dequant_fp8_state_dict
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

REPO = "Qwen/Qwen3-VL-8B-Instruct"
FP8 = os.environ.get("IDEOGRAM4_WEIGHTS")
# The real-weight case needs the gated fp8 checkpoint; skip (don't error) when it isn't set.
_NEEDS_WEIGHTS = pytest.mark.skipif(not FP8, reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)")


def _reference_lm(weights: str):
    """HF Qwen3-VL language model built from config (no base-checkpoint pull). weights="real"
    overlays the shipped Ideogram text_encoder weights (dequantized fp8); weights="random" keeps
    the fresh init, giving the encoder (incl. the new per-head QK-norm path) weight-free CI
    coverage. Both sides load the SAME state_dict, so the port is checked structurally."""
    cfg = transformers.AutoConfig.from_pretrained(REPO)
    hf = transformers.AutoModel.from_config(cfg).to(torch.bfloat16)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    if weights == "real":
        sd = dequant_fp8_state_dict(load_file(f"{FP8}/text_encoder/model.safetensors"))
        sd = {k[len("language_model.") :]: v for k, v in sd.items() if k.startswith("language_model.")}
        lm.load_state_dict(sd, strict=False)  # load the Ideogram-shipped (dequantized) weights
    return lm.eval()


# The production encoder always runs is_fsdp=True (weights FSDP-sharded on the non-TP axis), so
# both configs keep a non-TP axis of size > 1 for FSDP to be active.
@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [
        pytest.param((2, 4), (2, 4), 1, id="tp4_fsdp2"),  # TP=4 (axis 1), FSDP on axis 0 (size 2)
        pytest.param((4, 2), (4, 2), 1, id="tp2_fsdp4"),  # SP4xTP2 denoiser: encoder TP=2, FSDP on axis 0 (size 4)
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
@pytest.mark.parametrize(
    "weights",
    [pytest.param("random", id="random"), pytest.param("real", id="real", marks=_NEEDS_WEIGHTS)],
)
# masked=True exercises the explicit-mask path (prepare_attention_bias -> tensor.tril -> is_causal=False);
# masked=False uses the internal causal path (is_causal=True). Both are full-causal here so both match
# the causal HF golden — and the masked case is what covers the tril/bias branch (else zero coverage).
@pytest.mark.parametrize("masked", [pytest.param(False, id="nomask"), pytest.param(True, id="masked")])
@pytest.mark.parametrize("seq_len", [128])
def test_qwen3vl_text_encoder(
    *, mesh_device: ttnn.MeshDevice, submesh_shape, tp_axis, weights: str, masked: bool, seq_len: int
) -> None:
    """Qwen3-VL encoder under TP (the pipeline configs). weights="random" is weight-free CI coverage
    of the port (incl. per-head QK-norm); weights="real" is the shipped-weight fidelity check.
    Output is replicated across TP."""
    torch.manual_seed(0)
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    tp_factor = tuple(submesh.shape)[tp_axis]

    lm = _reference_lm(weights)
    cfg = lm.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    mrope_section = cfg.rope_scaling["mrope_section"]
    rope_theta = cfg.rope_scaling.get("rope_theta", cfg.rope_theta)  # top-level in transformers >=4.57

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    caps: dict[int, torch.Tensor] = {}
    handles = [
        lm.layers[i].register_forward_hook(
            lambda m, i_, o, i=i: caps.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())
        )
        for i in QWEN3_VL_ACTIVATION_LAYERS
    ]
    with torch.no_grad():
        lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for h in handles:
        h.remove()
    golden = [caps[i].float() for i in QWEN3_VL_ACTIVATION_LAYERS]

    enc = Qwen3VlTextEncoder(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act="silu",
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        activation_layers=QWEN3_VL_ACTIVATION_LAYERS,
        device=submesh,
        parallel_config=EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis)),
        ccl_manager=CCLManager(submesh, num_links=1, topology=ttnn.Topology.Linear),
        is_fsdp=True,  # production always FSDP-shards encoder weights on the non-TP axis
    )
    enc.load_torch_state_dict(lm.state_dict())

    cos, sin = create_rope_tensors(1, seq_len, None, head_dim, rope_theta, mrope_section)
    tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=submesh)
    # All-valid mask -> the encoder builds a full causal bias (prepare_attention_bias + tril),
    # matching the causal HF golden; None uses the internal causal path.
    attn_mask = tensor.from_torch(torch.ones(1, seq_len, dtype=torch.bool), device=submesh) if masked else None
    tt_caps = enc.forward(
        tt_ids,
        attention_mask=attn_mask,
        pos_embeds=(bf16_tensor(cos, device=submesh), bf16_tensor(sin, device=submesh)),
    )

    # Untrained (random) weights over Qwen's 36 layers accumulate more bf16 error than the trained
    # checkpoint, so allow a little more slack there; real weights hold 0.99.
    pcc = 0.99 if weights == "real" else 0.98
    for layer_idx, g, tt_t in zip(QWEN3_VL_ACTIVATION_LAYERS, golden, tt_caps):
        logger.info(f"qwen3vl [{weights}] TP={tp_factor} layer {layer_idx}:")
        assert_quality(g, tensor.to_torch(tt_t, mesh_axes=[None, None, None]), pcc=pcc)
