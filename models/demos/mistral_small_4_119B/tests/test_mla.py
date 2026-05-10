# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for :class:`TtMistral4MLA1D` (Mistral-4 latent attention, PyTorch bring-up).

DeepSeek reference layout: ``models/demos/deepseek_v3/tt/mla/mla1d.py`` (`link
<https://github.com/tenstorrent/tt-metal/blob/main/models/demos/deepseek_v3/tt/mla/mla1d.py>`_).
That stack is ``ttnn``/mesh-centric; here we compare against HF :class:`Mistral4Attention` on CPU.
:class:`TtMistral4MLA2D` (`DeepSeek MLA2D <https://github.com/tenstorrent/tt-metal/blob/main/models/demos/deepseek_v3/tt/mla/mla2d.py>`__)
extends :class:`TtMistral4MLA1D` for API parity; eager math matches MLA1D until seq-parallel ``ttnn`` is added.

Run: ``PYTHONPATH=. pytest models/demos/mistral_small_4_119B/tests/test_mla.py -v``
with ``--confcutdir=models/demos/mistral_small_4_119B`` if root ``conftest`` needs ``ttnn``.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger
from transformers.masking_utils import create_causal_mask
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention, Mistral4RotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.mla import (
    MistralSmall4MLA1D,
    MistralSmall4MLA2D,
    TtMistral4MLA1D,
    TtMistral4MLA2D,
    load_ttmistral4_mla2d_from_sharded_safetensors,
    load_ttmistral4_mla_from_sharded_safetensors,
    mistral4_hf_config_for_mla,
    read_mla_tensors_from_sharded_checkpoint,
)

PCC_REQUIRED = 0.99
PCC_REQUIRED_CHECKPOINT = 0.999


def _prepare_config(config: Mistral4Config) -> None:
    if getattr(config, "_attn_implementation", None) in (None, ""):
        try:
            config._attn_implementation = "sdpa"
        except (AttributeError, TypeError):
            pass


def _mla_inputs(config: Mistral4Config, x: torch.Tensor):
    b, t, _ = x.shape
    position_ids = torch.arange(t, device=x.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
    rope = Mistral4RotaryEmbedding(config).to(device=x.device, dtype=torch.bfloat16).eval()
    position_embeddings = rope(x, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=config,
        inputs_embeds=x,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    return position_ids, position_embeddings, causal_mask


def _assert_pcc(a: torch.Tensor, b: torch.Tensor, *, pcc_required: float, msg: str = "") -> None:
    assert a.shape == b.shape, (msg, a.shape, b.shape)
    passing, pcc = comp_pcc(b, a, pcc_required)
    status = "PASS" if passing else "FAIL"
    logger.info(f"[{status}] {msg} | PCC={pcc} | required>={pcc_required}")
    assert passing, f"{msg} PCC={pcc} below required threshold {pcc_required}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("mla_cls", [TtMistral4MLA1D, TtMistral4MLA2D])
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_forward_random_weights_tt_mla_matches_hf(
    mistral_text_config: Mistral4Config,
    mla_cls: type,
    layer_idx: int,
    mesh_device,
):
    _prepare_config(mistral_text_config)
    torch.manual_seed(101)
    ref = Mistral4Attention(mistral_text_config, layer_idx=layer_idx).eval()
    dut = mla_cls(mistral_text_config, layer_idx=layer_idx).eval()
    dut.attn.load_state_dict(ref.state_dict())

    ref = ref.to(torch.bfloat16)
    dut = dut.to(torch.bfloat16)

    b, t, h = 1, 5, mistral_text_config.hidden_size
    x = torch.randn(b, t, h, dtype=torch.bfloat16)
    position_ids, position_embeddings, causal_mask = _mla_inputs(mistral_text_config, x)

    # Route input through TT device (upload → readback)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_device = ttnn.to_torch(tt_x)

    with torch.no_grad():
        y_ref, _ = ref(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )
        y_dut, _ = dut(
            hidden_states=x_device,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )

    # Route DUT output through TT device (upload → readback)
    tt_y = ttnn.from_torch(
        y_dut,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_from_device = ttnn.to_torch(tt_y)

    _assert_pcc(
        y_from_device,
        y_ref,
        pcc_required=PCC_REQUIRED,
        msg=f"random {mla_cls.__name__} layer {layer_idx}",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("mla_cls", [TtMistral4MLA1D, TtMistral4MLA2D])
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_forward_sharded_checkpoint_matches_hf(
    mistral_text_config: Mistral4Config,
    mistral_sharded_checkpoint,
    mla_cls: type,
    layer_idx: int,
    mesh_device,
):
    _prepare_config(mistral_text_config)
    ref = Mistral4Attention(mistral_text_config, layer_idx=layer_idx).eval()
    dut = mla_cls(mistral_text_config, layer_idx=layer_idx).eval()

    raw, _ = read_mla_tensors_from_sharded_checkpoint(mistral_sharded_checkpoint, layer_idx)
    ref.load_state_dict(raw, strict=False)
    dut.attn.load_state_dict(raw, strict=False)

    ref = ref.to(torch.bfloat16)
    dut = dut.to(torch.bfloat16)

    torch.manual_seed(202)
    b, t, h = 1, 4, mistral_text_config.hidden_size
    x = torch.randn(b, t, h, dtype=torch.bfloat16)
    position_ids, position_embeddings, causal_mask = _mla_inputs(mistral_text_config, x)

    # Route input through TT device (upload → readback)
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_device = ttnn.to_torch(tt_x)

    with torch.no_grad():
        y_ref, _ = ref(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )
        y_dut, _ = dut(
            hidden_states=x_device,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
        )

    # Route DUT output through TT device (upload → readback)
    tt_y = ttnn.from_torch(
        y_dut,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_from_device = ttnn.to_torch(tt_y)

    _assert_pcc(
        y_from_device,
        y_ref,
        pcc_required=PCC_REQUIRED_CHECKPOINT,
        msg=f"checkpoint {mla_cls.__name__} layer {layer_idx}",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_read_mla_checkpoint_has_attention_keys(mistral_sharded_checkpoint, mesh_device):
    raw, prefix = read_mla_tensors_from_sharded_checkpoint(mistral_sharded_checkpoint, 0)
    assert "o_proj.weight" in raw
    assert prefix.endswith("self_attn.")

    # Verify a checkpoint weight tensor survives device round-trip
    weight = raw["o_proj.weight"].to(torch.bfloat16)
    tt_w = ttnn.from_torch(
        weight,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_back = ttnn.to_torch(tt_w)
    assert w_back.shape == weight.shape


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_load_ttmistral4_mla_from_sharded_safetensors_smoke(
    mistral_text_config: Mistral4Config,
    mistral_sharded_checkpoint,
    mesh_device,
):
    """``load_ttmistral4_mla_from_sharded_safetensors`` loads into the wrapper without double-key prefix issues."""
    _prepare_config(mistral_text_config)
    mla = TtMistral4MLA1D(mistral_text_config, layer_idx=0).eval()
    result = load_ttmistral4_mla_from_sharded_safetensors(mla, mistral_sharded_checkpoint, 0, strict=False)
    assert result.keys_loaded > 0
    assert result.prefix_used is not None

    # Verify a loaded weight survives device round-trip
    weight = next(mla.parameters()).to(torch.bfloat16)
    tt_w = ttnn.from_torch(
        weight,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_back = ttnn.to_torch(tt_w)
    assert w_back.shape == weight.shape


def test_mistral_small4_mla1d_subclasses_abstract_module():
    from models.demos.mistral_small_4_119B.tt_utils.abstract_module import AbstractModule

    assert issubclass(MistralSmall4MLA1D, AbstractModule)


def test_mistral_small4_mla2d_subclasses_mistral_mla1d():
    assert issubclass(MistralSmall4MLA2D, MistralSmall4MLA1D)


def test_mistral4_hf_config_for_mla_rope_scaling_passthrough():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    inner = Mistral4Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
        rope_scaling={"mscale": 1.0, "factor": 2.0},
    )
    assert mistral4_hf_config_for_mla(inner) is inner


def test_mistral4_hf_config_for_mla_maps_rope_parameters():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    inner = Mistral4Config(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
    )
    setattr(inner, "rope_parameters", {"mscale": 1.25, "factor": 32.0})
    view = mistral4_hf_config_for_mla(inner)
    assert view.rope_scaling["mscale"] == 1.25
    assert view.rope_scaling["factor"] == 32.0
    assert view.hidden_size == 64


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mistral_small4_mla1d_prefill_decode_config_smoke(mistral_text_config: Mistral4Config, mesh_device):
    """Device-side MLA1D configs (same keys as DeepSeek ``MLA1D``). Requires a TT mesh."""
    _prepare_config(mistral_text_config)
    pre = MistralSmall4MLA1D.prefill_model_config(mistral_text_config, mesh_device, batch_size_per_row=1)
    assert "wq_kv_a" in pre and "flash_mla" in pre and "batch_size_per_row" in pre
    dec = MistralSmall4MLA1D.decode_model_config(mistral_text_config, mesh_device, batch_size_per_row=1)
    assert "wq_kv_a" in dec and "flash_mla" in dec


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mistral_small4_mla2d_prefill_wraps_inner_config(mistral_text_config: Mistral4Config, mesh_device):
    _prepare_config(mistral_text_config)
    pre = MistralSmall4MLA2D.prefill_model_config(mistral_text_config, mesh_device, batch_size_per_row=1)
    assert "mla1d" in pre and "seq_ag_prefill" in pre and "seq_rs_prefill" in pre
    dec = MistralSmall4MLA2D.decode_model_config(mistral_text_config, mesh_device, batch_size_per_row=1)
    assert "mla1d" in dec


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_load_ttmistral4_mla2d_from_sharded_safetensors_smoke(
    mistral_text_config: Mistral4Config,
    mistral_sharded_checkpoint,
    mesh_device,
):
    """DeepSeek-style MLA2D loader alias fills ``TtMistral4MLA2D.attn``."""
    _prepare_config(mistral_text_config)
    mla2d = TtMistral4MLA2D(mistral_text_config, layer_idx=0).eval()
    result = load_ttmistral4_mla2d_from_sharded_safetensors(mla2d, mistral_sharded_checkpoint, 0, strict=False)
    assert result.keys_loaded > 0

    # Verify a loaded weight survives device round-trip
    weight = next(mla2d.parameters()).to(torch.bfloat16)
    tt_w = ttnn.from_torch(
        weight,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_back = ttnn.to_torch(tt_w)
    assert w_back.shape == weight.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
