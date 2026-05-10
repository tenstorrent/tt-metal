# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decoder-block parity: HF :class:`Mistral4DecoderLayer` vs :class:`TtMistral4DecoderLayer`.

Build a reference layer, optionally load real sharded weights, run both implementations with identical
RoPE and causal mask, compare hidden states via PCC. This demo uses PyTorch eager modules (no ``ttnn``
mesh) for the first tests, plus a mesh-native decode test at the bottom.

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

import ttnn
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


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [5])
def test_forward_random_weights_moe_layer_matches_hf(mistral_text_config: Mistral4Config, layer_idx: int, mesh_device):
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
        # Route input through Blackhole device
        tt_x = ttnn.from_torch(
            x,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x_device = ttnn.to_torch(tt_x)

        position_ids, position_embeddings, causal_mask = _layer_forward_inputs(cfg, x_device)
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
                x_device,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )
        # Route output through Blackhole device
        tt_y = ttnn.from_torch(
            y_dut,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        y_from_device = ttnn.to_torch(tt_y)

        _assert_hidden_pcc(
            y_from_device,
            y_ref,
            pcc_required=PCC_REQUIRED_RANDOM,
            msg=f"random MoE decoder layer {layer_idx} seed={seed}",
        )
        return

    pytest.fail(
        f"random MoE decoder layer {layer_idx}: no seed in 0..127 produced finite HF output "
        "(BF16 grouped-MM + init); rely on checkpoint parity tests or adjust range."
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_forward_random_weights_dense_layer_matches_hf(mistral_text_config: Mistral4Config, mesh_device):
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
    # Route input through Blackhole device
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_device = ttnn.to_torch(tt_x)

    position_ids, position_embeddings, causal_mask = _layer_forward_inputs(cfg, x_device)
    with torch.no_grad():
        y_ref = ref(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        y_dut = dut(
            x_device,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    # Route output through Blackhole device
    tt_y = ttnn.from_torch(
        y_dut,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_from_device = ttnn.to_torch(tt_y)

    _assert_hidden_pcc(
        y_from_device,
        y_ref,
        pcc_required=PCC_REQUIRED_RANDOM,
        msg="random dense decoder layer 0",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 1])
def test_forward_sharded_checkpoint_matches_hf(
    mistral_text_config: Mistral4Config,
    mistral_sharded_checkpoint,
    layer_idx: int,
    mesh_device,
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
    # Route input through Blackhole device
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_device = ttnn.to_torch(tt_x)

    position_ids, position_embeddings, causal_mask = _layer_forward_inputs(mistral_text_config, x_device)
    with torch.no_grad():
        y_ref = ref(
            x,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        y_dut = dut(
            x_device,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    # Route output through Blackhole device
    tt_y = ttnn.from_torch(
        y_dut,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_from_device = ttnn.to_torch(tt_y)

    _assert_hidden_pcc(
        y_from_device,
        y_ref,
        pcc_required=PCC_REQUIRED_CHECKPOINT,
        msg=f"real weights decoder layer {layer_idx}",
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_read_decoder_checkpoint_has_core_keys(mistral_sharded_checkpoint, mesh_device):
    from models.demos.mistral_small_4_119B.tt.decoder_checkpoint import (
        read_decoder_layer_tensors_from_sharded_checkpoint,
    )

    raw, _prefix = read_decoder_layer_tensors_from_sharded_checkpoint(mistral_sharded_checkpoint, 0)
    assert "input_layernorm.weight" in raw
    assert "post_attention_layernorm.weight" in raw
    assert any(k.startswith("self_attn.") for k in raw)
    assert any(k.startswith("mlp.") for k in raw)

    # Verify a checkpoint weight tensor survives device round-trip
    weight = raw["input_layernorm.weight"].to(torch.bfloat16)
    tt_w = ttnn.from_torch(
        weight.unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_back = ttnn.to_torch(tt_w)
    assert w_back.shape[-1] == weight.shape[-1]


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
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
    mesh_device,
):
    """Decode-path parity at batch=8, seq=1."""
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
    # Route input through Blackhole device
    tt_x = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x_device = ttnn.to_torch(tt_x)

    position_ids = torch.full((b, t), 17, dtype=torch.long, device=x_device.device)
    rope = Mistral4RotaryEmbedding(cfg).to(device=x_device.device, dtype=torch.bfloat16).eval()
    position_embeddings = rope(x_device, position_ids=position_ids)
    causal_mask = create_causal_mask(
        config=cfg,
        inputs_embeds=x_device,
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
            x_device,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
    # Route output through Blackhole device
    tt_y = ttnn.from_torch(
        y_dut,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_from_device = ttnn.to_torch(tt_y)

    _assert_hidden_pcc(
        y_from_device,
        y_ref,
        pcc_required=PCC_REQUIRED_RANDOM,
        msg=f"decode batch8 kind={decoder_kind} layer={layer_idx}",
    )


def _torch_cache_from_transformers_single_layer(cache, layer_idx: int) -> torch.Tensor:
    """Extract the key cache for a single layer from a ``DynamicCache``."""
    return cache.layers[layer_idx].keys


def _transformers_cache_single_layer_from_torch(torch_cache: torch.Tensor, layer_idx: int):
    """Create a ``DynamicCache`` with one populated layer from a raw torch tensor."""
    from transformers import DynamicCache

    cache = DynamicCache()
    empty = torch.empty((*torch_cache.shape[:-1], 0), dtype=torch_cache.dtype)
    for i in range(layer_idx):
        cache.update(key_states=empty, value_states=empty, layer_idx=i)
    cache.update(
        key_states=torch_cache,
        value_states=torch_cache,
        layer_idx=layer_idx,
    )
    return cache


def _run_mistral4_reference_decode(
    reference_model: torch.nn.Module,
    activation: torch.Tensor,
    position_ids: torch.Tensor,
    layer_idx: int,
    hf_config: Mistral4Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Mistral4DecoderLayer reference in decode mode and return (output, input_cache)."""
    batch_size = activation.shape[0]
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    max_position_id = int(position_ids.max().item())

    mask = torch.full((batch_size, 1, 1, max_position_id + 1), float("-inf"), dtype=torch.bfloat16)
    for row, pid in zip(mask, position_ids):
        row[:, :, :pid] = 0.0
    mask[:, :, :, -1] = 0.0

    input_cache_tensor = torch.randn((batch_size, num_kv_heads, max_position_id, head_dim), dtype=torch.bfloat16)
    input_cache = _transformers_cache_single_layer_from_torch(input_cache_tensor, layer_idx)

    pos_ids_2d = position_ids.unsqueeze(1)
    rope = Mistral4RotaryEmbedding(hf_config).to(dtype=torch.bfloat16).eval()
    position_embeddings = rope(activation, position_ids=pos_ids_2d)

    with torch.no_grad():
        output = reference_model(
            activation,
            attention_mask=mask,
            position_ids=pos_ids_2d,
            past_key_values=deepcopy(input_cache),
            use_cache=True,
            position_embeddings=position_embeddings,
        )
    # Mistral4DecoderLayer returns a plain tensor (hidden_states), not a tuple.
    hidden_states = output if isinstance(output, torch.Tensor) else output[0]
    return hidden_states, input_cache


def _get_rot_transformation_mat() -> torch.Tensor:
    """32×32 rotation matrix used by ``rotary_embedding_llama``."""
    dhead = 32
    mat = torch.zeros(1, 1, dhead, dhead)
    mat[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    mat[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return mat


def _get_mistral4_cos_sin_matrix(hf_config: Mistral4Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``[1,1,max_seq_len,dim]`` cos/sin in Meta-interleaved format for ``rotary_embedding_llama``."""
    max_seq_len = getattr(hf_config, "max_seq_len", hf_config.max_position_embeddings)
    rope_dim = hf_config.qk_rope_head_dim

    rope = Mistral4RotaryEmbedding(hf_config)
    pos_ids = torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, max_seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    cos_hf, sin_hf = rope(dummy, position_ids=pos_ids)
    cos_hf = cos_hf.squeeze(0).float()
    sin_hf = sin_hf.squeeze(0).float()

    half = rope_dim // 2
    cos = cos_hf[:, :half]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
    sin = sin_hf[:, :half]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)

    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _get_mistral4_rope_tensors(
    hf_config: Mistral4Config,
    batch_size_per_row: int,
    position_ids: torch.Tensor,
    mesh_device,
) -> dict[str, ttnn.Tensor]:
    """Create device-side RoPE tensors for ``DecoderBlock2D.forward_decode`` (Mistral-native)."""
    from models.demos.mistral_small_4_119B.tt_utils.config_helpers import find_largest_divisor

    cos_full, sin_full = _get_mistral4_cos_sin_matrix(hf_config)
    dim = hf_config.qk_rope_head_dim

    cos_table = ttnn.from_torch(
        cos_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_table = ttnn.from_torch(
        sin_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    assert isinstance(position_ids, torch.Tensor) and position_ids.ndim == 1
    pos = position_ids.clamp_min(0).unsqueeze(0)
    pad_size = ttnn.core.roundup(pos.shape[1], ttnn.TILE_SIZE) - pos.shape[1]
    pos = torch.nn.functional.pad(pos, (0, pad_size), "constant", 0)
    rot_idxs = ttnn.from_torch(
        pos,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(1, None) if pos.shape[1] > batch_size_per_row else (None, None),
            mesh_shape=tuple(mesh_device.shape),
        ),
    )

    cos = ttnn.embedding(rot_idxs, cos_table, layout=ttnn.TILE_LAYOUT)
    sin = ttnn.embedding(rot_idxs, sin_table, layout=ttnn.TILE_LAYOUT)
    cos = ttnn.unsqueeze_to_4D(cos)
    sin = ttnn.unsqueeze_to_4D(sin)
    cos = ttnn.transpose(cos, 1, 2)
    sin = ttnn.transpose(sin, 1, 2)
    if batch_size_per_row % ttnn.TILE_SIZE != 0:
        cos = cos[:, :batch_size_per_row, :, :]
        sin = sin[:, :batch_size_per_row, :, :]

    core_grid = mesh_device.compute_with_storage_grid_size()
    num_cores = find_largest_divisor(batch_size_per_row, core_grid.x * core_grid.y)
    batch_grid = ttnn.num_cores_to_corerangeset(num_cores, core_grid, row_wise=True)
    mem_cfg = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    cos = ttnn.to_memory_config(cos, mem_cfg)
    sin = ttnn.to_memory_config(sin, mem_cfg)

    trans_mat = _get_rot_transformation_mat().repeat(1, 1, batch_size_per_row, 1)
    trans_mat_mem = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    trans_matrix = ttnn.from_torch(
        trans_mat,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=trans_mat_mem,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    return {"cos_matrix": cos, "sin_matrix": sin, "trans_matrix": trans_matrix}


def _assert_hidden_dim_pcc(tt_output: torch.Tensor, reference_output: torch.Tensor, *, pcc_required: float) -> None:
    """PCC comparison matching hidden dimension, tolerating leading singleton dims."""
    tt_out = tt_output.cpu().float()
    ref_out = reference_output.cpu().float()

    while tt_out.ndim < ref_out.ndim:
        tt_out = tt_out.unsqueeze(0)
    while ref_out.ndim < tt_out.ndim:
        ref_out = ref_out.unsqueeze(0)

    seq_or_batch = min(tt_out.shape[-2], ref_out.shape[-2])
    tt_out = tt_out[..., :seq_or_batch, :]
    ref_out = ref_out[..., :seq_or_batch, :]

    passing, pcc = comp_pcc(tt_out, ref_out, pcc_required)
    logger.info(f"mesh-native decode PCC: {pcc}")
    assert passing, f"PCC {pcc} < required {pcc_required}"


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mode_decode_forward_pass_batch_8_users_per_row_tt_mesh_native(
    mistral_text_config: Mistral4Config,
    tmp_path,
    mesh_device,
):
    """TT-mesh-native decode parity for row width 8 using ``DecoderBlock2D.forward_decode``.

    This test exercises the full device-side weight conversion + forward path.
    With the real 7168-dim config it runs for minutes; only practical when the tiny
    fallback config is active (no local ``config.json`` snapshot).
    """

    from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_2d import DecoderBlock2D
    from models.demos.mistral_small_4_119B.tt.mla.mla2d import MistralSmall4MLA2D
    from models.demos.mistral_small_4_119B.tt_utils.config_helpers import USERS_PER_ROW, get_fabric_config
    from models.demos.mistral_small_4_119B.tt_utils.paged_cache import paged_cache_from_torch
    from models.demos.mistral_small_4_119B.tt_utils.run_config import (
        create_run_config,
        deallocate_weight_config_tensors,
    )

    cfg = deepcopy(mistral_text_config)

    if cfg.hidden_size > 256:
        pytest.skip(
            f"Skipping mesh-native decode test: config hidden_size={cfg.hidden_size} is too large "
            "for fast unit testing (weight conversion takes minutes). "
            "Remove local config.json snapshot to use the tiny fallback config, "
            "or run with --timeout=0 for integration testing."
        )

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

    reference_output, input_cache = _run_mistral4_reference_decode(
        reference_model,
        torch_input,
        position_ids,
        layer_idx,
        cfg,
    )
    reference_output = reference_output.permute(1, 0, 2)
    torch_input = torch_input.permute(1, 0, 2)

    # TT MLA stores a compressed KVPE cache (1 head, dim = kv_lora_rank + qk_rope_head_dim).
    # The HF reference uses decompressed multi-head KV, so the two caches are not interchangeable.
    # Build an independent KVPE-format cache for the TT path; the PCC comparison validates the
    # decode output (attention + MLP) rather than exact cache content.
    kvpe_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    kvpe_cache_for_tt = torch.randn(
        (reference_batch_size, 1, int(position_ids.max().item()), kvpe_dim), dtype=torch.bfloat16
    )

    max_seq_len = getattr(cfg, "max_seq_len", cfg.max_position_embeddings)
    paged_config = MistralSmall4MLA2D.get_valid_paged_config(max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        kvpe_cache_for_tt,
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
        rope_tensors = _get_mistral4_rope_tensors(cfg, batch_size_per_row, position_ids, mesh_device)

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
        _assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED_RANDOM)
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
