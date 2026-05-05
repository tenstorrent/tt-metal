# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit test: OLMo3 1-layer prefill — ring SDPA vs standard SDPA PCC + 64K test.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.olmo_galaxy.tt.llama_common import PagedAttentionConfig
from models.demos.olmo_galaxy.tt.llama_model import TtTransformer
from models.demos.olmo_galaxy.tt.olmo_model_config import TtOlmoModelArgs
from models.tt_transformers.tt.common import copy_host_to_device


def run_prefill(model, model_args, mesh_device, tokens, page_table, kv_cache, seq_len):
    """Run 1-layer prefill and return logits."""
    host_inputs = model.prepare_prefill_inputs_host(tokens, user_id=0, page_table=page_table)
    device_inputs = copy_host_to_device(host_inputs, mesh_device=mesh_device)
    transformed = model.transform_prefill_inputs_device(*device_inputs)
    tt_out = model.ttnn_prefill_forward(*transformed, kv_cache=kv_cache, batch_size=1)
    ttnn.synchronize_device(mesh_device)
    tt_logits_saved = torch.zeros(1, model_args.padded_vocab_size)
    model.process_output_prefill(tt_out, last_token_idx=seq_len - 1, tt_out_logits_saved=tt_logits_saved)
    ttnn.synchronize_device(mesh_device)
    return tt_logits_saved[0, : model_args.vocab_size].float()


@pytest.mark.parametrize("num_layers", [1, 64], ids=["1L", "64L"])
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 184915840,
            "num_command_queues": 1,
            "worker_l1_size": 1345000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_olmo_ring_vs_standard_sdpa_pcc(num_layers, mesh_device):
    """Compare ring vs standard SDPA at 8K ISL — check PCC doesn't degrade across layers."""
    torch.manual_seed(42)
    seq_len = 8192

    paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=2048)
    model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=128 * 1024)
    model_args.n_layers = num_layers
    state_dict = model_args.load_state_dict()

    model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
        decode_mode_only=False,
    )

    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)
    page_table = torch.arange(paged_attention_config.max_num_blocks).reshape(1, -1).int()
    page_table = torch.nn.functional.pad(page_table, (0, 0, 0, 31), value=0)
    kv_cache = [layer.attention.layer_past for layer in model.layers]

    # Run standard SDPA
    for layer in model.layers:
        layer.attention.force_ring_sdpa = False
    logits_standard = run_prefill(model, model_args, mesh_device, tokens, page_table, kv_cache, seq_len)

    # Reset KV cache for all layers
    for layer in model.layers:
        k_c, v_c = layer.attention.layer_past
        ttnn.mul(k_c, 0, output_tensor=k_c)
        ttnn.mul(v_c, 0, output_tensor=v_c)

    # Run ring SDPA — also disable sliding window to isolate PCC impact
    for layer in model.layers:
        layer.attention.force_ring_sdpa = True
        layer.attention._saved_sliding_window = layer.attention.sliding_window_size
        layer.attention.sliding_window_size = None  # Full attention for ring SDPA
    logits_ring = run_prefill(model, model_args, mesh_device, tokens, page_table, kv_cache, seq_len)

    pcc_pass, pcc_val = comp_pcc(logits_ring, logits_standard, 0.95)
    top5_std = torch.topk(logits_standard, 5)
    top5_ring = torch.topk(logits_ring, 5)
    top1_match = top5_std.indices[0].item() == top5_ring.indices[0].item()
    top5_overlap = len(set(top5_std.indices.tolist()) & set(top5_ring.indices.tolist()))

    logger.info(f"Ring vs Standard SDPA at {seq_len} ISL, {num_layers} layers:")
    logger.info(f"  PCC: {pcc_val}")
    logger.info(f"  Standard top-5: {top5_std.indices.tolist()}")
    logger.info(f"  Ring top-5:     {top5_ring.indices.tolist()}")
    logger.info(f"  Top-1 match: {top1_match}, Top-5 overlap: {top5_overlap}/5")

    assert pcc_pass, f"Ring vs Standard SDPA PCC {pcc_val} < 0.95 ({num_layers} layers)"


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 184915840,
            "num_command_queues": 1,
            "worker_l1_size": 1345000,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_olmo_64k_ring_sdpa(mesh_device):
    """64K prefill with ring SDPA — verify no hang and valid logits."""
    torch.manual_seed(42)
    seq_len = 65536

    paged_attention_config = PagedAttentionConfig(block_size=64, max_num_blocks=2048)
    model_args = TtOlmoModelArgs(mesh_device, max_batch_size=32, max_seq_len=128 * 1024)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    model = TtTransformer(
        args=model_args,
        mesh_device=mesh_device,
        dtype=ttnn.bfloat8_b,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(ttnn.bfloat8_b),
        paged_attention_config=paged_attention_config,
        decode_mode_only=False,
    )

    for layer in model.layers:
        layer.attention.force_ring_sdpa = True

    tokens = torch.randint(0, model_args.vocab_size, (1, seq_len), dtype=torch.long)
    page_table = torch.arange(paged_attention_config.max_num_blocks).reshape(1, -1).int()
    page_table = torch.nn.functional.pad(page_table, (0, 0, 0, 31), value=0)
    kv_cache = [layer.attention.layer_past for layer in model.layers]

    logits = run_prefill(model, model_args, mesh_device, tokens, page_table, kv_cache, seq_len)

    logger.info(f"64K ring SDPA: top-5={torch.topk(logits, 5).indices.tolist()}")
    assert torch.isfinite(logits).all(), "Logits contain Inf/NaN"
    assert logits.abs().sum() > 0, "Logits are all zero"
