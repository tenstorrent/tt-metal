# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import getpass
import json
import shlex
import shutil
import socket
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

import ttnn
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.model.row_batched_model import RowBatchedModel
from models.demos.deepseek_v3.utils.bitsculpt_trace import BitSculptTraceCollector
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_rope_tensors
from models.demos.deepseek_v3.utils.weight_config import _distributed_barrier, _get_distributed_rank, get_weight_config


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_runtime_input(tokenizer, input_ids: torch.Tensor, mesh_device: ttnn.MeshDevice) -> tuple[torch.Tensor, int]:
    seq_len = int(input_ids.shape[-1])
    ring_multiple = ttnn.TILE_SIZE * int(mesh_device.shape[0])
    runtime_seq_len = max(ttnn.TILE_SIZE, _round_up(seq_len, ring_multiple))
    if runtime_seq_len == seq_len:
        return input_ids, runtime_seq_len

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    padded = torch.full((1, runtime_seq_len), pad_token_id, dtype=input_ids.dtype)
    padded[:, :seq_len] = input_ids
    return padded, runtime_seq_len


def _get_save_dtype(save_dtype: str) -> torch.dtype:
    try:
        return getattr(torch, save_dtype)
    except AttributeError as exc:
        raise ValueError(f"Unsupported save dtype {save_dtype!r}; expected bfloat16 or float32") from exc


def _configure_trace_hooks(run_config, collector: BitSculptTraceCollector, hf_config) -> None:
    run_config["embedding"]["keep_padded_output"] = True

    for layer_idx, block_cfg in enumerate(run_config["mlp_decoder_block"]):
        block_cfg["layer_idx"] = layer_idx
        block_cfg["debug_trace"] = collector
        block_cfg["mla"]["layer_idx"] = layer_idx
        block_cfg["mla"]["debug_trace"] = collector
        block_cfg["mla"]["mla1d"]["layer_idx"] = layer_idx
        block_cfg["mla"]["mla1d"]["debug_trace"] = collector

    for offset, block_cfg in enumerate(run_config["moe_decoder_block"], start=hf_config.first_k_dense_replace):
        block_cfg["layer_idx"] = offset
        block_cfg["debug_trace"] = collector
        block_cfg["mla"]["layer_idx"] = offset
        block_cfg["mla"]["debug_trace"] = collector
        block_cfg["mla"]["mla1d"]["layer_idx"] = offset
        block_cfg["mla"]["mla1d"]["debug_trace"] = collector
        block_cfg["mlp"]["moe"]["layer_idx"] = offset
        block_cfg["mlp"]["moe"]["debug_trace"] = collector


@pytest.mark.timeout(14400)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
def test_bitsculpt_trace(
    request,
    hf_config,
    model_path,
    state_dict,
    cache_path,
    mesh_device,
    ccl,
    force_recalculate_weight_config,
):
    assert not force_recalculate_weight_config, (
        "test_bitsculpt_trace reuses the shared full-model cache at DEEPSEEK_V3_CACHE; "
        "do not run it with --recalculate-weights"
    )

    prompt = request.config.getoption("bitsculpt_trace_prompt")
    run_tag = request.config.getoption("bitsculpt_trace_run_tag")
    output_dir = Path(request.config.getoption("bitsculpt_trace_output_dir"))
    model_id = request.config.getoption("bitsculpt_trace_model_id")
    tokenizer_path = request.config.getoption("bitsculpt_trace_tokenizer") or str(model_path)
    max_tokens = request.config.getoption("bitsculpt_trace_max_tokens")
    save_dtype = _get_save_dtype(request.config.getoption("bitsculpt_trace_save_dtype"))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    token_ids = input_ids[0].tolist()
    token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]
    seq_len = len(token_ids)
    runtime_input_ids, runtime_seq_len = _pad_runtime_input(tokenizer, input_ids, mesh_device)

    assert seq_len <= max_tokens, f"Prompt has {seq_len} tokens, exceeds max_tokens={max_tokens}"

    hf_config_trace = deepcopy(hf_config)
    hf_config_trace.max_seq_len = max(128, runtime_seq_len)

    paged_config = MLA2D.get_valid_paged_config(hf_config_trace.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    tt_page_tables = tuple(
        MLA2D.create_page_table(
            paged_config=paged_config,
            mesh_device=mesh_device,
            identity=True,
        )
        for _ in range(hf_config_trace.num_hidden_layers)
    )

    weight_config = get_weight_config(
        ModuleClass=RowBatchedModel,
        hf_config=hf_config_trace,
        state_dicts=(state_dict,),
        weight_cache_path=cache_path,
        mesh_device=mesh_device,
        force_recalculate=force_recalculate_weight_config,
    )
    model_config = get_model_config(RowBatchedModel, "prefill", hf_config_trace, mesh_device)
    model_state = RowBatchedModel.create_state(hf_config_trace, paged_config, mesh_device, ccl)
    model_shared_state = RowBatchedModel.create_shared_state(hf_config_trace, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    collector = BitSculptTraceCollector(
        hf_config=hf_config_trace,
        mesh_device=mesh_device,
        model_id=model_id,
        prompt=prompt,
        token_ids=token_ids,
        token_strings=token_strings,
        save_dtype=save_dtype,
    )
    _configure_trace_hooks(run_config, collector, hf_config_trace)

    tt_input = ttnn.from_torch(
        runtime_input_ids.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    rope_tensors = get_rope_tensors(
        hf_config_trace,
        batch_size_per_row=1,
        seq_len=runtime_seq_len,
        position_ids=None,
        mesh_device=mesh_device,
    )

    run_dir = output_dir / run_tag
    rank = _get_distributed_rank()
    if rank == 0 and run_dir.exists():
        shutil.rmtree(run_dir)
    _distributed_barrier()

    tt_output = RowBatchedModel.forward_prefill(tt_input, 0, run_config, rope_tensors, tt_page_tables)
    ttnn.synchronize_device(mesh_device)

    collector.validate()
    _distributed_barrier()

    if rank == 0:
        run_dir = collector.save(output_dir, run_tag)
    _distributed_barrier()

    if rank == 0:
        with open(run_dir / "metadata.json") as f:
            metadata = json.load(f)

        routing = load_file(str(run_dir / "expert_routing.safetensors"))
        hidden = load_file(str(run_dir / "hidden_states.safetensors"))
        kv_cache = load_file(str(run_dir / "kv_cache.safetensors"))

        assert metadata["prompt"] == prompt
        assert metadata["token_ids"] == token_ids
        assert metadata["token_strings"] == token_strings
        assert metadata["n_tokens"] == seq_len
        assert metadata["n_layers"] == hf_config_trace.num_hidden_layers
        assert metadata["moe_layer_offset"] == hf_config_trace.first_k_dense_replace
        assert metadata["kv_lora_rank"] == hf_config_trace.kv_lora_rank
        assert metadata["model_id"] == model_id
        assert metadata["hidden_dim"] == hf_config_trace.hidden_size
        assert metadata["n_experts"] == hf_config_trace.n_routed_experts
        assert metadata["top_k"] == hf_config_trace.num_experts_per_tok
        assert metadata["save_dtype"] == str(save_dtype)
        assert metadata["username"] == getpass.getuser()
        assert metadata["hostname"] == socket.gethostname()
        assert metadata["command_line"] == shlex.join(sys.argv)
        assert "git_branch" in metadata
        assert "git_commit" in metadata
        assert metadata["timestamp"]

        assert len(hidden) == hf_config_trace.num_hidden_layers * 3
        assert len(kv_cache) == hf_config_trace.num_hidden_layers
        assert len(routing) == (hf_config_trace.num_hidden_layers - hf_config_trace.first_k_dense_replace) * 2

        for layer_idx in range(hf_config_trace.num_hidden_layers):
            assert hidden[f"decoder_output_layer_{layer_idx}"].shape == (seq_len, hf_config_trace.hidden_size)
            assert hidden[f"post_mla_residual_layer_{layer_idx}"].shape == (seq_len, hf_config_trace.hidden_size)
            assert hidden[f"post_attn_norm_layer_{layer_idx}"].shape == (seq_len, hf_config_trace.hidden_size)
            assert kv_cache[f"compressed_kv_layer_{layer_idx}"].shape == (
                seq_len,
                hf_config_trace.kv_lora_rank + hf_config_trace.qk_rope_head_dim,
            )

        for layer_idx in range(hf_config_trace.first_k_dense_replace, hf_config_trace.num_hidden_layers):
            assert routing[f"expert_ids_layer_{layer_idx}"].shape == (seq_len, hf_config_trace.num_experts_per_tok)
            assert routing[f"expert_weights_layer_{layer_idx}"].shape == (seq_len, hf_config_trace.num_experts_per_tok)
            assert routing[f"expert_ids_layer_{layer_idx}"].dtype == torch.int32

    _distributed_barrier()

    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)

    for page_table in tt_page_tables:
        ttnn.deallocate(page_table)
