# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Multi-layer TT decoder stack vs chained HF :class:`Mistral4DecoderLayer` (decode path).

Loads helpers from ``test_decoder_block`` via ``importlib`` (``tests/`` is not a regular package).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from copy import deepcopy
from pathlib import Path

_repo = Path(__file__).resolve().parents[4]
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))
try:
    from tests.scripts.ompi_singleton_env import apply_ompi_singleton_workaround_env

    apply_ompi_singleton_workaround_env()
except ImportError:
    if os.environ.get("TT_METAL_OMPI_SINGLETON_WORKAROUND", "1") != "0":
        os.environ.setdefault("OMPI_MCA_plm", "isolated")
        os.environ.setdefault("PRTE_MCA_plm", "isolated")

import pytest
import torch
from loguru import logger
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4DecoderLayer

import ttnn


def _tiny_mistral4_config_for_fast_decode() -> Mistral4Config:
    """Small Mistral4 geometry for fast decode on Blackhole (matches ``mla1d.decode_model_config`` divisibility).

    - ``RMSNorm`` / tilized weights: last dim multiple of 32; use ``qk_nope_head_dim == 32`` so ``wkv_b1`` shard height is tile-sized.
    - ``wq_b``: ``even_int_div(q_lora_rank, 16 * 32)`` → ``q_lora_rank`` multiple of **512** (use 512).
    - ``wq_b`` width: ``even_int_div(wq_b_n_padded, 16 * 32)`` → ``num_heads * (qk_nope + qk_rope)`` padded must be a multiple of **512** (use 128 per head → 4×128 = 512, ``hidden_size = 512``).
    - ``wo``: ``num_heads * v_head_dim`` ≥ **1024** (4×256).
    """
    return Mistral4Config(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=4096,
        kv_lora_rank=32,
        q_lora_rank=512,
        qk_rope_head_dim=64,
        v_head_dim=256,
        qk_nope_head_dim=64,
    )


def _cfg_for_text_stack_decode(base: Mistral4Config) -> Mistral4Config:
    """Use the fixture config when already small; otherwise a tiny config so decode runs in CI time."""
    if base.hidden_size <= 512:
        return deepcopy(deepcopy(base))
    logger.info(
        "test_text_stack: fixture hidden_size={} — using tiny Mistral4Config for fast decode parity.",
        base.hidden_size,
    )
    return _tiny_mistral4_config_for_fast_decode()


def _load_decoder_block_tests():
    path = Path(__file__).resolve().parent / "test_decoder_block.py"
    spec = importlib.util.spec_from_file_location("_mistral4_decoder_block_tests", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_text_stack_decode_matches_hf_chained_layers(
    mistral_text_config: Mistral4Config,
    tmp_path: Path,
    mesh_device: ttnn.MeshDevice,
):
    """Stack ``num_hidden_layers`` TT blocks (dense + MoE per ``first_k_dense_replace``) vs HF chain.

    Requires multi-device mesh (fabric) — MLA decode path uses unconditional all_gather_async
    across the column axis. Skip on single-device (1,1) meshes.
    """
    if mesh_device.get_num_devices() < 2:
        pytest.skip(
            "Text stack decode requires multi-device mesh (MLA all_gather_async needs fabric). "
            "Run on a TG/DUAL/QUAD cluster."
        )

    tdb = _load_decoder_block_tests()

    cfg = _cfg_for_text_stack_decode(mistral_text_config)
    if cfg.num_hidden_layers > 6:
        pytest.skip(
            f"Skipping text stack decode: num_hidden_layers={cfg.num_hidden_layers} "
            "(keep stack test bounded; override in integration runs)."
        )

    cfg.first_k_dense_replace = 1 if cfg.num_hidden_layers >= 2 else 0
    tdb._prepare_decoder_config(cfg)

    from models.demos.mistral_small_4_119B.tt.mistral4_text_stack import (
        build_mistral4_text_stack_decode_run_config,
        create_text_stack_page_table,
        forward_mistral4_text_stack_decode,
    )
    from models.demos.mistral_small_4_119B.tt.mla.mla2d import MistralSmall4MLA2D
    from models.demos.mistral_small_4_119B.tt_utils.config_helpers import USERS_PER_ROW, get_fabric_config
    from models.demos.mistral_small_4_119B.tt_utils.run_config import deallocate_weight_config_tensors

    batch_size_per_row = 8
    seq_len = 1
    decode_position_id = 17
    num_layers = cfg.num_hidden_layers
    reference_batch_size = batch_size_per_row * mesh_device.shape[0]

    torch.manual_seed(29)
    hf_refs = [Mistral4DecoderLayer(cfg, layer_idx=li).eval().to(torch.bfloat16) for li in range(num_layers)]
    layer_sds = [{k: v.detach().clone() for k, v in ref.state_dict().items()} for ref in hf_refs]

    position_ids = torch.full((reference_batch_size,), decode_position_id, dtype=torch.long)
    kvpe_seq_len = int(position_ids.max().item())

    torch_input = (0.1 * torch.randn(reference_batch_size, seq_len, cfg.hidden_size, dtype=torch.float32)).to(
        torch.bfloat16
    )
    x_hf = torch_input
    for li, ref in enumerate(hf_refs):
        x_hf, _ = tdb._run_mistral4_reference_decode(ref, x_hf, position_ids, li, cfg)
    reference_output = x_hf.permute(1, 0, 2)

    max_seq_len = getattr(cfg, "max_seq_len", cfg.max_position_embeddings)
    paged_config = MistralSmall4MLA2D.get_valid_paged_config(max_seq_len, USERS_PER_ROW, mesh_device.shape[1])

    run_cfg, layer_classes, weight_cfg, torch_page_table = build_mistral4_text_stack_decode_run_config(
        cfg,
        mesh_device,
        get_fabric_config(),
        batch_size_per_row,
        tmp_path / "text_stack_decode",
        paged_config,
        ccl=None,
        num_layers=num_layers,
        kvpe_seq_len=kvpe_seq_len,
        reference_batch_size=reference_batch_size,
        layer_state_dicts=layer_sds,
    )

    torch_input_tt = torch_input.permute(1, 0, 2)
    tt_input = None
    position_ids_tensor = None
    tt_page_table = None
    tt_output = None
    try:
        tt_input = ttnn.from_torch(
            torch_input_tt.unsqueeze(0),
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
        tt_page_table = create_text_stack_page_table(torch_page_table, paged_config, mesh_device)
        rope_tensors = tdb._get_mistral4_rope_tensors(cfg, batch_size_per_row, position_ids, mesh_device)

        tt_output = forward_mistral4_text_stack_decode(
            tt_input,
            position_ids_tensor,
            run_cfg,
            layer_classes,
            rope_tensors,
            tt_page_table,
        )
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        )
        tdb._assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=tdb.PCC_REQUIRED_RANDOM)
    finally:
        if tt_input is not None:
            ttnn.deallocate(tt_input)
        if position_ids_tensor is not None:
            ttnn.deallocate(position_ids_tensor)
        if tt_page_table is not None:
            ttnn.deallocate(tt_page_table)
        if tt_output is not None:
            ttnn.deallocate(tt_output)
        deallocate_weight_config_tensors(weight_cfg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
