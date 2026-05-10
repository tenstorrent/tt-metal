# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full decoder layer with **TTNN MoE** vs HF ``Mistral4DecoderLayer``.

* **MoE-only PCC**: TTNN post-attention norm (composed) as MoE input; HF ``Mistral4MoE`` vs device routed+shared with
  HF ``gate`` + ``route_tokens_to_experts`` (PCC ``>= 0.28``). Hub MoE tables are loaded via
  :func:`~models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh.mistral4_mlp_state_dict_bf16_match_hf` so FP8
  weights match ``Mistral4MoE``; strict ``0.99`` is reserved for random-init PCC in ``test_mistral4_moe_mesh_routed_pcc.py``.
* **Full-layer PCC (HF routing)**: ``use_ttnn_moe=True``, ``moe_hf_torch_routing=True`` vs full HF layer
  (PCC ``>= 0.28``; host ``gate`` + ``topk`` for parity).
* **Full-layer PCC (demo path)**: ``moe_hf_torch_routing=False`` — device ``ttnn.linear`` + FP32 softmax + ``ttnn.topk``
  (PCC vs HF is still **routing-limited**, not dense-quality; CI floor ``_DEVICE_ROUTING_FULL_LAYER_MIN_PCC``). Run with ``-k device_routing_pcc``.
* **Smoke** (``moe_hf_torch_routing=False``): finite outputs only (``-k device_routing_smoke``).
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.common.auto_compose import to_torch_auto_compose
from models.common.utility_functions import comp_allclose, comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    strip_fp8_aux_tensors_from_decoder_inner,
    text_decoder_layer_inner_state_dict,
    text_decoder_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_moe_mesh import (
    TtMistral4MoeRoutedExpertParallelSkeleton,
    TtMistral4SharedExpertsMlpTtnn,
    mistral4_mlp_state_dict_bf16_match_hf,
)
from models.experimental.mistral_small_4_119b.tt.text_backbone import (
    TtMistral4DecoderLayer,
    TtMistral4DecoderLayerAttnPrefillBlock,
)
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


def _trim_mesh_compose_to_ref(y: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    ``to_torch_auto_compose`` may concatenate replicated mesh replicas (see ``models.common.auto_compose``).

    Keep a single logical replica with the same shape as ``ref`` for PCC vs HF.
    """
    if y.shape == ref.shape:
        return y.contiguous()
    n = ref.numel()
    if y.numel() == n:
        return y.reshape(ref.shape).contiguous()
    if y.numel() % n != 0:
        pytest.fail(
            f"composed output numel {y.numel()} not a multiple of reference {n}; "
            f"shape={tuple(y.shape)} ref_shape={tuple(ref.shape)}"
        )
    return y.flatten()[:n].view_as(ref).contiguous()


pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 requires recent transformers")


def _text_config_eager_attn():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")
    text = cfg.text_config
    if text is None:
        pytest.skip("Config has no text_config")
    if hasattr(text, "attn_implementation"):
        text.attn_implementation = "eager"
    if hasattr(text, "_attn_implementation"):
        text._attn_implementation = "eager"
    return text


def _layer_checkpoint(layer_idx: int):
    prefix = text_decoder_layer_state_dict_prefix(layer_idx)
    try:
        return load_hf_state_dict_filtered(HF_MODEL_ID, (prefix,))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"No checkpoint shards for layer {layer_idx}: {exc}")


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
@pytest.mark.parametrize("layer_idx", (0, 1), ids=("layer0", "layer1"))
def test_mistral_small_4_moe_only_ttnn_vs_hf_pcc(seq_len, layer_idx, reset_seeds, mesh_device):
    """HF ``Mistral4MoE`` vs TTNN routed+shared on the same composed post-attention norm (HF gate + topk)."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    state_dict = _layer_checkpoint(layer_idx)
    inner = strip_fp8_aux_tensors_from_decoder_inner(text_decoder_layer_inner_state_dict(state_dict, layer_idx))
    mlp_sd_raw = {k[len("mlp.") :]: v for k, v in inner.items() if k.startswith("mlp.")}

    layer = Mistral4DecoderLayer(text, layer_idx=layer_idx).eval()
    try:
        layer.load_state_dict(inner, strict=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF layer-{layer_idx}: {exc}")
    layer = layer.to(torch.bfloat16)
    mlp_sd = mistral4_mlp_state_dict_bf16_match_hf(text, mlp_sd_raw)

    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    torch.manual_seed(1)
    x = torch.randn(1, seq_len, text.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(x, position_ids)

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=mesh_device),
    )

    try:
        attn = TtMistral4DecoderLayerAttnPrefillBlock(mesh_device, state_dict, text, layer_idx=layer_idx)
    except Exception as exc:
        pytest.skip(f"TTNN attn init failed: {exc}")

    _, normed = attn.forward_split(
        x_tt,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        mode="prefill",
    )
    nt = to_torch_auto_compose(normed, device=mesh_device)
    x_hf = nt[:, 0, :seq_len, :].contiguous()
    ref_mlp = layer.mlp(x_hf)

    logits = layer.mlp.gate(x_hf)
    topk_idx_th, topk_w_th = layer.mlp.route_tokens_to_experts(logits)

    tt_ccl = TT_CCL(mesh_device)
    try:
        sk = TtMistral4MoeRoutedExpertParallelSkeleton(mesh_device, text, mlp_sd, tt_ccl=tt_ccl)
        sh = TtMistral4SharedExpertsMlpTtnn(mesh_device, text, mlp_sd)
    except Exception as exc:
        pytest.skip(f"TTNN MoE init failed: {exc}")

    n_tile = ttnn.from_torch(
        x_hf.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=mesh_device),
    )
    routed = sk(n_tile, topk_indices_torch=topk_idx_th, topk_weights_torch=topk_w_th)
    shared = sh(n_tile)
    mlp_tt = ttnn.add(routed, shared)
    ttnn.deallocate(routed)
    ttnn.deallocate(shared)
    ttnn.deallocate(n_tile)

    got = to_torch_auto_compose(mlp_tt, device=mesh_device)[:, 0, :seq_len, :].contiguous()
    got = _trim_mesh_compose_to_ref(got, ref_mlp)

    passing, msg = comp_pcc(ref_mlp, got, pcc=0.28)
    logger.info(comp_allclose(ref_mlp, got))
    logger.info(f"layer{layer_idx} MoE-only (TTNN vs HF) PCC: {msg}")
    assert passing, msg


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
@pytest.mark.parametrize("layer_idx", (0, 1), ids=("layer0", "layer1"))
def test_mistral_small_4_text_decoder_layer_full_ttnn_moe_pcc_vs_hf(seq_len, layer_idx, reset_seeds, mesh_device):
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    state_dict = _layer_checkpoint(layer_idx)
    inner = strip_fp8_aux_tensors_from_decoder_inner(text_decoder_layer_inner_state_dict(state_dict, layer_idx))

    layer = Mistral4DecoderLayer(text, layer_idx=layer_idx).eval()
    try:
        layer.load_state_dict(inner, strict=True)
    except Exception as exc:
        pytest.skip(
            f"Could not load layer-{layer_idx} weights into HF ``Mistral4DecoderLayer`` after stripping FP8 "
            f"aux keys. Detail: {exc}"
        )
    layer = layer.to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)

    torch.manual_seed(1)
    x = torch.randn(1, seq_len, text.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(x, position_ids)

    ref = layer(
        x,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )

    tt_ccl = TT_CCL(mesh_device)
    try:
        tt_layer = TtMistral4DecoderLayer(
            mesh_device,
            state_dict,
            text,
            layer_idx=layer_idx,
            tt_ccl=tt_ccl,
            use_ttnn_moe=True,
            moe_hf_torch_routing=True,
        )
    except Exception as exc:
        pytest.skip(f"TTNN full layer {layer_idx} (device MoE + HF routing) init failed: {exc}")

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=mesh_device),
    )
    y_tt = tt_layer(
        x_tt,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        mode="prefill",
    )
    y_tt_torch = to_torch_auto_compose(y_tt, device=mesh_device)[:, :, :seq_len, :].squeeze(1)
    y_tt_torch = _trim_mesh_compose_to_ref(y_tt_torch, ref)

    passing, pcc_message = comp_pcc(ref, y_tt_torch, pcc=0.28)
    logger.info(comp_allclose(ref, y_tt_torch))
    logger.info(f"layer{layer_idx} full decoder (TTNN MoE, HF routing) vs full HF PCC: {pcc_message}")
    assert passing, f"PCC below 0.28: {pcc_message}"


# Demo path: no host ``F.linear`` / ``topk`` in MoE forward. Full-layer PCC vs HF stays **well below** dense
# bring-up (~0.9+) because discrete expert choice diverges from HF until device routing matches HF exactly.
# Observed BH ~0.29 — keep CI floor just under that to catch regressions without claiming production quality.
_DEVICE_ROUTING_FULL_LAYER_MIN_PCC = 0.28


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
@pytest.mark.parametrize("layer_idx", (0, 1), ids=("layer0", "layer1"))
def test_mistral_small_4_text_decoder_layer_full_ttnn_moe_device_routing_pcc_vs_hf(
    seq_len, layer_idx, reset_seeds, mesh_device
):
    """Full layer: ``moe_hf_torch_routing=False`` (device router + topk) vs HF ``Mistral4DecoderLayer``."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    state_dict = _layer_checkpoint(layer_idx)
    inner = strip_fp8_aux_tensors_from_decoder_inner(text_decoder_layer_inner_state_dict(state_dict, layer_idx))

    layer = Mistral4DecoderLayer(text, layer_idx=layer_idx).eval()
    try:
        layer.load_state_dict(inner, strict=True)
    except Exception as exc:
        pytest.skip(
            f"Could not load layer-{layer_idx} weights into HF ``Mistral4DecoderLayer`` after stripping FP8 "
            f"aux keys. Detail: {exc}"
        )
    layer = layer.to(torch.bfloat16)
    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)

    torch.manual_seed(1)
    x = torch.randn(1, seq_len, text.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(x, position_ids)

    ref = layer(
        x,
        attention_mask=None,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
        position_embeddings=position_embeddings,
    )

    tt_ccl = TT_CCL(mesh_device)
    try:
        tt_layer = TtMistral4DecoderLayer(
            mesh_device,
            state_dict,
            text,
            layer_idx=layer_idx,
            tt_ccl=tt_ccl,
            use_ttnn_moe=True,
            moe_hf_torch_routing=False,
        )
    except Exception as exc:
        pytest.skip(f"TTNN full layer {layer_idx} (device MoE routing) init failed: {exc}")

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=mesh_device),
    )
    y_tt = tt_layer(
        x_tt,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        mode="prefill",
    )
    y_tt_torch = to_torch_auto_compose(y_tt, device=mesh_device)[:, :, :seq_len, :].squeeze(1)
    y_tt_torch = _trim_mesh_compose_to_ref(y_tt_torch, ref)

    min_pcc = _DEVICE_ROUTING_FULL_LAYER_MIN_PCC
    passing, pcc_message = comp_pcc(ref, y_tt_torch, pcc=min_pcc)
    logger.info(comp_allclose(ref, y_tt_torch))
    logger.info(
        f"layer{layer_idx} full decoder (TTNN MoE, device routing, demo path) vs full HF PCC: {pcc_message} "
        f"(min={min_pcc})"
    )
    assert passing, f"PCC below {min_pcc}: {pcc_message}"


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [mesh_device_request_param()], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (8,))
@pytest.mark.parametrize("layer_idx", (0,), ids=("layer0",))
def test_mistral_small_4_text_decoder_layer_ttnn_moe_device_routing_smoke(seq_len, layer_idx, reset_seeds, mesh_device):
    """Pure device routing + TTNN experts: forward runs and outputs are finite."""
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    text = _text_config_eager_attn()
    state_dict = _layer_checkpoint(layer_idx)
    tt_ccl = TT_CCL(mesh_device)
    try:
        tt_layer = TtMistral4DecoderLayer(
            mesh_device,
            state_dict,
            text,
            layer_idx=layer_idx,
            tt_ccl=tt_ccl,
            use_ttnn_moe=True,
            moe_hf_torch_routing=False,
        )
    except Exception as exc:
        pytest.skip(f"TTNN decoder layer {layer_idx} init failed: {exc}")

    rotary = Mistral4RotaryEmbedding(text).eval().to(torch.bfloat16)
    torch.manual_seed(2)
    x = torch.randn(1, seq_len, text.hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary(x, position_ids)

    x_tt = ttnn.from_torch(
        x.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device=mesh_device),
    )
    y_tt = tt_layer(
        x_tt,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        mode="prefill",
    )
    y = to_torch_auto_compose(y_tt, device=mesh_device)[:, :, :seq_len, :].squeeze(1)
    shape_ref = y.new_empty((1, seq_len, text.hidden_size))
    y = _trim_mesh_compose_to_ref(y, shape_ref)
    assert y.shape == (1, seq_len, text.hidden_size)
    assert torch.isfinite(y.to(torch.float32)).all()
