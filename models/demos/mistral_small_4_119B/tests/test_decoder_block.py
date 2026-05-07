# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decoder-block parity: HF :class:`Mistral4DecoderLayer` vs :class:`TtMistral4DecoderLayer`.

Follows the structure of DeepSeek-V3 ``test_decoder_block.py`` (`link
<https://github.com/tenstorrent/tt-metal/blob/main/models/demos/deepseek_v3/tests/test_decoder_block.py>`_):
build a reference layer, optionally load real sharded weights, run both implementations with identical
RoPE and causal mask, compare hidden states. This demo uses PyTorch eager modules (no ``ttnn`` mesh).

Run from ``tt-metal`` with ``PYTHONPATH=.``; use ``--confcutdir=models/demos/mistral_small_4_119B`` if
root ``conftest`` requires ``ttnn`` (see ``test_moe.py``).
"""

from __future__ import annotations

import inspect
from copy import deepcopy

import pytest
import torch
from loguru import logger
from transformers.masking_utils import create_causal_mask
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer, Mistral4RotaryEmbedding

from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.mistral4_decoder_layer import TtMistral4DecoderLayer

PCC_REQUIRED_RANDOM = 0.988
PCC_REQUIRED_CHECKPOINT = 0.9899


def _prepare_decoder_config(config: Mistral4Config) -> None:
    if getattr(config, "_experts_implementation", None) in (None, ""):
        try:
            config._experts_implementation = "grouped_mm"
        except (AttributeError, TypeError):
            pass
    # Avoid noisy "AttentionInterface … _attn_implementation is None" when layers run standalone.
    if getattr(config, "_attn_implementation", None) in (None, ""):
        try:
            config._attn_implementation = "sdpa"
        except (AttributeError, TypeError):
            pass


def _assert_hidden_pcc(a: torch.Tensor, b: torch.Tensor, *, pcc_required: float, msg: str = "") -> None:
    assert a.shape == b.shape, (msg, a.shape, b.shape)
    af, bf = a.float(), b.float()
    non_finite_a = ~torch.isfinite(af)
    non_finite_b = ~torch.isfinite(bf)
    assert torch.equal(non_finite_a, non_finite_b), f"{msg} non-finite masks differ"
    finite = torch.isfinite(af) & torch.isfinite(bf)
    if not finite.any():
        pytest.fail(f"{msg}: no finite values to compare")
    a_finite = af[finite]
    b_finite = bf[finite]
    passing, pcc = comp_pcc(a_finite, b_finite, pcc_required)
    status = "PASS" if passing else "FAIL"
    logger.info(
        f"{msg} | PCC={pcc:.6f} required={pcc_required:.6f} [{status}]",
    )
    assert passing, f"{msg} PCC {pcc:.6f} < required {pcc_required:.6f}"


def _layer_forward_inputs(config: Mistral4Config, x: torch.Tensor):
    """Match ``Mistral4Model`` wiring: RoPE + causal mask."""
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


def _get_model_config(module_class, mode: str, *args, batch_size_per_row: int | None = None, **kwargs):
    if mode == "prefill":
        config_fn = module_class.prefill_model_config
    elif mode == "decode":
        config_fn = module_class.decode_model_config
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if batch_size_per_row is not None and "batch_size_per_row" in inspect.signature(config_fn).parameters:
        kwargs.setdefault("batch_size_per_row", batch_size_per_row)

    return config_fn(*args, **kwargs)


@pytest.mark.parametrize("layer_idx", [5])
def test_forward_random_weights_moe_layer_matches_hf(mistral_text_config: Mistral4Config, layer_idx: int):
    """Same ``state_dict`` on reference and TT layer (MoE path when ``first_k_dense_replace`` is 0).

    Uses scaled BF16 inputs and a mid-stack ``layer_idx`` (here 5). Random init + grouped-MM MoE
    under BF16 is seed-sensitive once ``ttnn`` is loaded (root ``conftest``); we scan a small seed
    range for the first draw that yields finite activations, then assert DUT matches HF. Real
    weights are covered by ``test_forward_sharded_checkpoint_matches_hf``.
    """
    cfg = deepcopy(mistral_text_config)
    _prepare_decoder_config(cfg)
    b, t, h = 1, 4, cfg.hidden_size

    for seed in range(128):
        torch.manual_seed(seed)
        ref = Mistral4DecoderLayer(cfg, layer_idx=layer_idx).eval()
        dut = TtMistral4DecoderLayer(cfg, layer_idx=layer_idx).eval()
        missing = dut.load_state_dict(ref.state_dict(), strict=False)
        assert not missing.missing_keys, missing
        ref_bf = ref.to(torch.bfloat16)
        dut_bf = dut.to(torch.bfloat16)

        x = (0.1 * torch.randn(b, t, h, dtype=torch.float32)).to(torch.bfloat16)
        position_ids, position_embeddings, causal_mask = _layer_forward_inputs(cfg, x)
        with torch.no_grad():
            y_ref = ref_bf(
                x,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        if not torch.isfinite(y_ref).all():
            continue
        with torch.no_grad():
            y_dut = dut_bf(
                x,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        _assert_hidden_pcc(
            y_dut,
            y_ref,
            pcc_required=PCC_REQUIRED_RANDOM,
            msg=f"random MoE decoder layer {layer_idx} seed={seed}",
        )
        return

    pytest.fail(
        f"random MoE decoder layer {layer_idx}: no seed in 0..127 produced finite HF output "
        "(BF16 grouped-MM + init); rely on checkpoint parity tests or adjust range."
    )


def test_forward_random_weights_dense_layer_matches_hf(mistral_text_config: Mistral4Config):
    """First layer dense MLP when ``first_k_dense_replace`` forces it."""
    cfg = deepcopy(mistral_text_config)
    cfg.first_k_dense_replace = 1
    _prepare_decoder_config(cfg)
    layer_idx = 0
    torch.manual_seed(11)
    ref = Mistral4DecoderLayer(cfg, layer_idx=layer_idx).eval()
    dut = TtMistral4DecoderLayer(cfg, layer_idx=layer_idx).eval()
    missing = dut.load_state_dict(ref.state_dict(), strict=False)
    assert not missing.missing_keys, missing
    ref = ref.to(torch.bfloat16)
    dut = dut.to(torch.bfloat16)

    b, t, h = 1, 3, cfg.hidden_size
    x = (0.1 * torch.randn(b, t, h, dtype=torch.float32)).to(torch.bfloat16)
    position_ids, position_embeddings, causal_mask = _layer_forward_inputs(cfg, x)
    with torch.no_grad():
        y_ref = ref(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        y_dut = dut(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    _assert_hidden_pcc(
        y_dut,
        y_ref,
        pcc_required=PCC_REQUIRED_RANDOM,
        msg="random dense decoder layer 0",
    )


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_forward_sharded_checkpoint_matches_hf(
    mistral_text_config: Mistral4Config,
    mistral_sharded_checkpoint,
    layer_idx: int,
):
    """Load full layer tensors from sharded ``safetensors`` into both modules; forwards must match."""
    from models.demos.mistral_small_4_119B.tt.decoder_checkpoint import (
        read_decoder_layer_tensors_from_sharded_checkpoint,
    )

    _prepare_decoder_config(mistral_text_config)
    raw, prefix = read_decoder_layer_tensors_from_sharded_checkpoint(mistral_sharded_checkpoint, layer_idx)
    assert prefix, "checkpoint prefix should be non-empty"

    ref = Mistral4DecoderLayer(mistral_text_config, layer_idx=layer_idx).eval()
    dut = TtMistral4DecoderLayer(mistral_text_config, layer_idx=layer_idx).eval()
    ref.load_state_dict(raw, strict=False)
    dut.load_state_dict(raw, strict=False)
    ref = ref.to(torch.bfloat16)
    dut = dut.to(torch.bfloat16)

    torch.manual_seed(13)
    b, t, h = 1, 3, mistral_text_config.hidden_size
    x = (0.1 * torch.randn(b, t, h, dtype=torch.float32)).to(torch.bfloat16)
    position_ids, position_embeddings, causal_mask = _layer_forward_inputs(mistral_text_config, x)
    with torch.no_grad():
        y_ref = ref(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        y_dut = dut(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    _assert_hidden_pcc(
        y_dut,
        y_ref,
        pcc_required=PCC_REQUIRED_CHECKPOINT,
        msg=f"real weights decoder layer {layer_idx}",
    )


def test_read_decoder_checkpoint_has_core_keys(mistral_sharded_checkpoint):
    from models.demos.mistral_small_4_119B.tt.decoder_checkpoint import (
        read_decoder_layer_tensors_from_sharded_checkpoint,
    )

    raw, _prefix = read_decoder_layer_tensors_from_sharded_checkpoint(mistral_sharded_checkpoint, 0)
    assert "input_layernorm.weight" in raw
    assert "post_attention_layernorm.weight" in raw
    assert any(k.startswith("self_attn.") for k in raw)
    assert any(k.startswith("mlp.") for k in raw)


@pytest.mark.parametrize(
    "decoder_kind, layer_idx",
    [
        ("dense", 0),
        ("moe", 0),
    ],
)
def test_mode_decode_forward_pass_batch_8_users_per_row(
    mistral_text_config: Mistral4Config,
    decoder_kind: str,
    layer_idx: int,
):
    """Decode-path parity at batch=8, seq=1 (adapted from DeepSeek decoder test intent)."""
    cfg = deepcopy(mistral_text_config)
    if decoder_kind == "dense":
        cfg.first_k_dense_replace = 1
    elif decoder_kind == "moe":
        cfg.first_k_dense_replace = 0
    else:
        raise ValueError(f"Unsupported decoder kind: {decoder_kind}")

    _prepare_decoder_config(cfg)

    torch.manual_seed(19)
    ref = Mistral4DecoderLayer(cfg, layer_idx=layer_idx).eval()
    dut = TtMistral4DecoderLayer(cfg, layer_idx=layer_idx).eval()
    missing = dut.load_state_dict(ref.state_dict(), strict=False)
    assert not missing.missing_keys, missing
    ref = ref.to(torch.bfloat16)
    dut = dut.to(torch.bfloat16)

    b, t, h = 8, 1, cfg.hidden_size
    x = (0.1 * torch.randn(b, t, h, dtype=torch.float32)).to(torch.bfloat16)
    position_ids = torch.full((b, t), 17, dtype=torch.long, device=x.device)
    rope = Mistral4RotaryEmbedding(cfg).to(device=x.device, dtype=torch.bfloat16).eval()
    position_embeddings = rope(x, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=cfg,
        inputs_embeds=x,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    with torch.no_grad():
        y_ref = ref(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        y_dut = dut(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )

    _assert_hidden_pcc(
        y_dut,
        y_ref,
        pcc_required=PCC_REQUIRED_RANDOM,
        msg=f"decode batch8 kind={decoder_kind} layer={layer_idx}",
    )


@pytest.mark.timeout(900)
def test_mode_decode_forward_pass_batch_8_users_per_row_tt_mesh_native(
    mistral_text_config: Mistral4Config,
    tmp_path,
    request: pytest.FixtureRequest,
):
    """TT-mesh-native decode parity for row width 8 using ``DecoderBlock2D.forward_decode``."""
    ttnn = pytest.importorskip("ttnn")
    try:
        mesh_device = request.getfixturevalue("mesh_device")
    except pytest.FixtureLookupError:
        pytest.skip("TT mesh-native decoder test requires the root 'mesh_device' fixture/device harness")

    from models.demos.deepseek_v3.utils.test_utils import (
        assert_hidden_dim_pcc,
        get_rope_tensors,
        paged_cache_from_torch,
        run_reference_with_attention,
        torch_cache_from_transformers_single_layer,
    )
    from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_2d import DecoderBlock2D
    from models.demos.mistral_small_4_119B.tt.mla.mla2d import MistralSmall4MLA2D
    from models.demos.mistral_small_4_119B.tt_utils.config_helpers import USERS_PER_ROW, get_fabric_config
    from models.demos.mistral_small_4_119B.tt_utils.run_config import (
        create_run_config,
        deallocate_weight_config_tensors,
    )

    cfg = deepcopy(mistral_text_config)
    cfg.first_k_dense_replace = 1
    _prepare_decoder_config(cfg)

    batch_size_per_row = 8
    seq_len = 1
    decode_position_id = 17
    layer_idx = 0
    reference_batch_size = batch_size_per_row * mesh_device.shape[0]

    torch.manual_seed(23)
    reference_model = Mistral4DecoderLayer(cfg, layer_idx=layer_idx).eval().to(torch.bfloat16)
    state_dict = {k: v.detach().clone() for k, v in reference_model.state_dict().items()}
    torch_input = (0.1 * torch.randn(reference_batch_size, seq_len, cfg.hidden_size, dtype=torch.float32)).to(
        torch.bfloat16
    )
    position_ids = torch.full((reference_batch_size,), decode_position_id, dtype=torch.long)

    reference_output, input_cache, _ = run_reference_with_attention(
        reference_model,
        torch_input,
        position_ids,
        layer_idx,
        cfg,
        "decode",
        False,
    )
    reference_output = reference_output.permute(1, 0, 2)
    input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
    torch_input = torch_input.permute(1, 0, 2)

    paged_config = MistralSmall4MLA2D.get_valid_paged_config(cfg.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache,
        tuple(mesh_device.shape),
        paged_config,
        user_id=None,
    )

    weight_config = DecoderBlock2D.convert_weights(
        cfg,
        (state_dict,),
        tmp_path / "decoder_block_2d_mesh_native_decode",
        mesh_device,
    )
    model_config = _get_model_config(
        DecoderBlock2D,
        "decode",
        cfg,
        mesh_device,
        get_fabric_config(),
        batch_size_per_row=batch_size_per_row,
    )
    model_state = DecoderBlock2D.create_state(
        cfg,
        paged_config,
        mesh_device,
        ccl=None,
        mla_cache=paged_input_cache,
    )
    model_shared_state = DecoderBlock2D.create_shared_state(cfg, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    tt_input = None
    position_ids_tensor = None
    tt_page_table = None
    tt_output = None
    try:
        tt_input = ttnn.from_torch(
            torch_input.unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        position_ids_tensor = ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        tt_page_table = MistralSmall4MLA2D.create_page_table(
            page_table=torch_page_table,
            paged_config=paged_config,
            mesh_device=mesh_device,
        )
        rope_tensors = get_rope_tensors(cfg, batch_size_per_row, seq_len, position_ids, mesh_device)

        tt_output = DecoderBlock2D.forward_decode(
            tt_input,
            position_ids_tensor,
            run_config,
            rope_tensors,
            tt_page_table,
        )
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )
        assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED_RANDOM)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if position_ids_tensor is not None:
            ttnn.deallocate(position_ids_tensor)
        if tt_page_table is not None:
            ttnn.deallocate(tt_page_table)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        deallocate_weight_config_tensors(weight_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
