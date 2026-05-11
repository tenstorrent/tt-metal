# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-layer decode (and prefill) orchestration for Mistral Small 4 TT decoder blocks.

Stacks :class:`DecoderBlock2D` for early dense layers and :class:`MoEDecoderBlock2D` for MoE layers
(``layer_idx >= hf_config.first_k_dense_replace``), repeating for ``num_hidden_layers`` (or a caller
override).

**Paged KV policy:** one logical page-table mapping (``torch.Tensor`` from
:func:`models.demos.mistral_small_4_119B.tt_utils.paged_cache.paged_cache_from_torch`) is shared
across layers: layer 0 allocates the mapping; subsequent layers pass the same ``mapping=`` into
``paged_cache_from_torch`` so block indices line up. Each layer still gets its own flat
``mla_cache`` tensor (separate KVPE payload). Device-side :meth:`MistralSmall4MLA2D.create_page_table`
is called once; the same ``page_table`` ttnn tensor is passed to every
:meth:`DecoderBlock2DBase.forward_decode` / ``forward_prefill`` call.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer

import ttnn
from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_2d import DecoderBlock2D
from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_2d_base import DecoderBlock2DBase
from models.demos.mistral_small_4_119B.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.mistral_small_4_119B.tt.mla.mla2d import MistralSmall4MLA2D
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.paged_cache import paged_cache_from_torch
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
    create_run_config,
)
from models.tt_transformers.tt.common import PagedAttentionConfig

DecoderBlockTTCls = type[DecoderBlock2DBase]


def decoder_block_orchestration_class(hf_config: Mistral4Config, layer_idx: int) -> DecoderBlockTTCls:
    """Return the TT orchestration class (dense vs MoE) for ``layer_idx``."""
    if layer_idx < hf_config.first_k_dense_replace:
        return DecoderBlock2D
    return MoEDecoderBlock2D


def _model_config_for_mode(
    module_class: DecoderBlockTTCls,
    mode: str,
    hf_config: Mistral4Config,
    mesh_device: ttnn.MeshDevice,
    fabric_config: ttnn.FabricConfig,
    *,
    batch_size_per_row: int,
):
    if mode == "prefill":
        fn = module_class.prefill_model_config
    elif mode == "decode":
        fn = module_class.decode_model_config
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    kwargs: dict = {}
    if "batch_size_per_row" in inspect.signature(fn).parameters:
        kwargs["batch_size_per_row"] = batch_size_per_row
    return fn(hf_config, mesh_device, fabric_config, **kwargs)


def _layer_weight_and_state(
    hf_config: Mistral4Config,
    layer_idx: int,
    mesh_device: ttnn.MeshDevice,
    tmp_path: Path,
    paged_config: PagedAttentionConfig,
    ccl: CCL | None,
    *,
    mode: str,
    batch_size_per_row: int,
    fabric_config: ttnn.FabricConfig,
    kvpe_cache_torch: torch.Tensor,
    shared_page_mapping: torch.Tensor | None,
    state_dict: dict[str, torch.Tensor],
) -> tuple[DecoderBlockTTCls, WeightConfig, dict, dict, dict, torch.Tensor]:
    """One layer: convert weights, model config slice, state, shared state; return updated shared page mapping."""
    BlockCls = decoder_block_orchestration_class(hf_config, layer_idx)
    out_dir = tmp_path / f"layer_{layer_idx}"
    weight_cfg = BlockCls.convert_weights(hf_config, (state_dict,), out_dir, mesh_device)
    model_cfg = _model_config_for_mode(
        BlockCls,
        mode,
        hf_config,
        mesh_device,
        fabric_config,
        batch_size_per_row=batch_size_per_row,
    )
    paged_in, mapping = paged_cache_from_torch(
        kvpe_cache_torch,
        tuple(mesh_device.shape),
        paged_config,
        user_id=None,
        mapping=shared_page_mapping,
    )
    if shared_page_mapping is None:
        shared_page_mapping = mapping
    state = BlockCls.create_state(hf_config, paged_config, mesh_device, ccl, mla_cache=paged_in)
    shared_state = BlockCls.create_shared_state(hf_config, mesh_device)
    return BlockCls, weight_cfg, model_cfg, state, shared_state, shared_page_mapping


def build_mistral4_text_stack_decode_run_config(
    hf_config: Mistral4Config,
    mesh_device: ttnn.MeshDevice,
    fabric_config: ttnn.FabricConfig,
    batch_size_per_row: int,
    tmp_path: Path,
    paged_config: PagedAttentionConfig,
    ccl: CCL | None,
    *,
    num_layers: int | None = None,
    kvpe_seq_len: int,
    reference_batch_size: int,
    layer_state_dicts: Sequence[dict[str, torch.Tensor]] | None = None,
) -> tuple[RunDecodeConfig, tuple[DecoderBlockTTCls, ...], WeightConfig, torch.Tensor]:
    """Build a merged decode :func:`create_run_config` with ``{"layers": [...]}`` zip structure.

    Args:
        hf_config: HF config (dense vs MoE split uses ``first_k_dense_replace``).
        mesh_device: Target mesh.
        fabric_config: Fabric config for MLP/MoE decode paths.
        batch_size_per_row: Users per row (decode).
        tmp_path: Root directory for per-layer converted weights.
        paged_config: Shared paged attention layout for all layers.
        ccl: Optional CCL handle; if ``None``, each block's ``create_state`` builds ``CCL(mesh_device)``
            (required for RMSNorm / MLA / MLP collectives even on a 1×1 mesh).
        num_layers: Stack depth; default ``hf_config.num_hidden_layers``.
        kvpe_seq_len: Third dimension of per-layer host KVPE cache (match max decode position usage).
        reference_batch_size: Total batch across mesh (``batch_size_per_row * mesh_rows`` typical).
        layer_state_dicts: Optional per-layer HF weights; length must match stack depth. When set,
            TT conversion uses these dicts so parity tests can share weights with a reference model.

    Returns:
        ``run_config``, per-layer TT block classes, ``weight_config`` tree for
        :func:`models.demos.mistral_small_4_119B.tt_utils.run_config.deallocate_weight_config_tensors`,
        and the host **shared** page-table mapping tensor used for :func:`create_text_stack_page_table`.
    """
    n = num_layers if num_layers is not None else hf_config.num_hidden_layers
    if layer_state_dicts is not None and len(layer_state_dicts) != n:
        raise ValueError(f"layer_state_dicts length {len(layer_state_dicts)} != num_layers {n}")
    kvpe_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim

    model_cfgs: list = []
    weight_cfgs: list = []
    states: list = []
    shared_states: list = []
    classes: list[DecoderBlockTTCls] = []
    shared_mapping: torch.Tensor | None = None

    for li in range(n):
        if layer_state_dicts is None:
            ref = Mistral4DecoderLayer(hf_config, layer_idx=li).eval().to(torch.bfloat16)
            sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
        else:
            sd = layer_state_dicts[li]
        kvpe = torch.randn((reference_batch_size, 1, kvpe_seq_len, kvpe_dim), dtype=torch.bfloat16)
        BlockCls, wcfg, mcfg, st, sst, shared_mapping = _layer_weight_and_state(
            hf_config,
            li,
            mesh_device,
            tmp_path,
            paged_config,
            ccl,
            mode="decode",
            batch_size_per_row=batch_size_per_row,
            fabric_config=fabric_config,
            kvpe_cache_torch=kvpe,
            shared_page_mapping=shared_mapping,
            state_dict=sd,
        )
        classes.append(BlockCls)
        weight_cfgs.append(wcfg)
        model_cfgs.append(mcfg)
        states.append(st)
        shared_states.append(sst)

    if not model_cfgs:
        raise ValueError("num_layers must be >= 1")
    assert shared_mapping is not None
    run_cfg = create_run_config(
        {"layers": model_cfgs},
        {"layers": weight_cfgs},
        {"layers": states},
        {"layers": shared_states},
    )
    weight_root: WeightConfig = {"layers": weight_cfgs}
    return run_cfg, tuple(classes), weight_root, shared_mapping


def build_mistral4_text_stack_prefill_run_config(
    hf_config: Mistral4Config,
    mesh_device: ttnn.MeshDevice,
    fabric_config: ttnn.FabricConfig,
    batch_size_per_row: int,
    tmp_path: Path,
    paged_config: PagedAttentionConfig,
    ccl: CCL | None,
    *,
    num_layers: int | None = None,
    kvpe_seq_len: int,
    reference_batch_size: int,
    layer_state_dicts: Sequence[dict[str, torch.Tensor]] | None = None,
) -> tuple[RunPrefillConfig, tuple[DecoderBlockTTCls, ...], WeightConfig, torch.Tensor]:
    """Same as :func:`build_mistral4_text_stack_decode_run_config` but for prefill model configs/state."""
    n = num_layers if num_layers is not None else hf_config.num_hidden_layers
    if layer_state_dicts is not None and len(layer_state_dicts) != n:
        raise ValueError(f"layer_state_dicts length {len(layer_state_dicts)} != num_layers {n}")
    kvpe_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim

    model_cfgs: list = []
    weight_cfgs: list = []
    states: list = []
    shared_states: list = []
    classes: list[DecoderBlockTTCls] = []
    shared_mapping: torch.Tensor | None = None

    for li in range(n):
        if layer_state_dicts is None:
            ref = Mistral4DecoderLayer(hf_config, layer_idx=li).eval().to(torch.bfloat16)
            sd = {k: v.detach().clone() for k, v in ref.state_dict().items()}
        else:
            sd = layer_state_dicts[li]
        kvpe = torch.randn((reference_batch_size, 1, kvpe_seq_len, kvpe_dim), dtype=torch.bfloat16)
        BlockCls, wcfg, mcfg, st, sst, shared_mapping = _layer_weight_and_state(
            hf_config,
            li,
            mesh_device,
            tmp_path,
            paged_config,
            ccl,
            mode="prefill",
            batch_size_per_row=batch_size_per_row,
            fabric_config=fabric_config,
            kvpe_cache_torch=kvpe,
            shared_page_mapping=shared_mapping,
            state_dict=sd,
        )
        classes.append(BlockCls)
        weight_cfgs.append(wcfg)
        model_cfgs.append(mcfg)
        states.append(st)
        shared_states.append(sst)

    if not model_cfgs:
        raise ValueError("num_layers must be >= 1")
    assert shared_mapping is not None
    run_cfg = create_run_config(
        {"layers": model_cfgs},
        {"layers": weight_cfgs},
        {"layers": states},
        {"layers": shared_states},
    )
    weight_root: WeightConfig = {"layers": weight_cfgs}
    return run_cfg, tuple(classes), weight_root, shared_mapping


def forward_mistral4_text_stack_decode(
    x: ttnn.Tensor,
    position_idxs: ttnn.Tensor,
    run_config: RunDecodeConfig,
    layer_block_classes: tuple[DecoderBlockTTCls, ...],
    rope_tensors: dict,
    page_table: ttnn.Tensor,
) -> ttnn.Tensor:
    """Run ``forward_decode`` for each layer, reusing ``rope_tensors`` and ``page_table``."""
    layers_cfg = run_config["layers"]
    if len(layers_cfg) != len(layer_block_classes):
        raise ValueError(f"run_config has {len(layers_cfg)} layers but {len(layer_block_classes)} block classes")
    for BlockCls, layer_cfg in zip(layer_block_classes, layers_cfg, strict=True):
        x = BlockCls.forward_decode(x, position_idxs, layer_cfg, rope_tensors, page_table)
    return x


def forward_mistral4_text_stack_prefill(
    x: ttnn.Tensor,
    user_id: int,
    run_config: RunPrefillConfig,
    layer_block_classes: tuple[DecoderBlockTTCls, ...],
    rope_tensors: dict,
    page_table: ttnn.Tensor,
) -> ttnn.Tensor:
    """Run ``forward_prefill`` for each layer, reusing ``rope_tensors`` and ``page_table``."""
    layers_cfg = run_config["layers"]
    if len(layers_cfg) != len(layer_block_classes):
        raise ValueError(f"run_config has {len(layers_cfg)} layers but {len(layer_block_classes)} block classes")
    for BlockCls, layer_cfg in zip(layer_block_classes, layers_cfg, strict=True):
        x = BlockCls.forward_prefill(x, user_id, layer_cfg, rope_tensors, page_table)
    return x


def create_text_stack_page_table(
    torch_page_table: torch.Tensor,
    paged_config: PagedAttentionConfig,
    mesh_device: ttnn.MeshDevice,
) -> ttnn.Tensor:
    """Device page table for the stack (one tensor, shared by all layers)."""
    return MistralSmall4MLA2D.create_page_table(
        page_table=torch_page_table,
        paged_config=paged_config,
        mesh_device=mesh_device,
    )
