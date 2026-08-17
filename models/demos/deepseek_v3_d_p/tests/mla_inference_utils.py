# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared TT MLA single-forward helper.

Run one MLA forward (construct ttMLA, build rope/index caches, shard input, forward,
synchronize) without host comparison. Moved verbatim from the removed dense MLA test file (#53362);
used by the KV-cache-table and sparse-MLA suites.
"""

from loguru import logger
import torch

import ttnn

from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.indexer import num_full_indexer_layers, resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

def run_mla_inference(
    config,
    weights,
    mesh_device,
    seq_len,
    mesh_shape,
    sp_axis,
    tp_axis,
    is_balanced,
    topology,
    tt_kvpe_cache,
    return_indices=False,
    inject_indices=None,
):
    """
    Utility function to run MLA inference without host comparison.

    Args:
        config: Model configuration
        weights: Model weights dictionary
        mesh_device: Mesh device for TT
        seq_len: Sequence length
        mesh_shape: Shape of mesh device
        sp_axis: Sequence parallel axis
        tp_axis: Tensor parallel axis
        is_balanced: Whether to use balanced chunk ordering
        topology: Topology (Linear or Ring)
        tt_kvpe_cache: Initialized KVPE cache on device

    Returns:
        Tuple of (tt_output, hidden_states, chunk_order, shard_dims)
    """
    # Create TT MLA
    logger.info("Creating TT MLA...")

    mla_tt = ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=is_balanced,
        topology=topology,
        # Match the single-layer test cache (num_kvpe_cache_layers=1): the sparse single-shot write now
        # goes through update_padded_kv_cache, which asserts cache_batch % layer_num == 0. Dense is
        # unaffected (its single-shot write uses fill_cache_for_user_, which ignores layer_num).
        layer_num=1,
        sparse_kv_cache_format=tt_kvpe_cache.format,
    )
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=is_balanced)
    # Sparse (DSA) single-shot is folded onto the block-cyclic path (one full-seq chunk at offset 0):
    # it uses the indexed rope tables and a caller-owned indexer key cache, exactly like the chunked
    # path. Dense keeps natural rope + no index cache.
    has_indexer = resolve_has_indexer(config)
    index_kv_cache = None
    if has_indexer:
        rope_tensors = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len, chunk_size_global=seq_len)
        # Layer-slot count mirrors the serving adapter: the indexer strides the folded user-major cache by
        # num_full_indexer_layers (GLM-5.2 cross-layer reuse), so the cache must carry that many slots for
        # update_padded_kv_cache's cache_batch % num_layers check. Falls back to 1 (no indexer_types).
        index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
            num_users=1,
            dtype=ttnn.bfloat8_b,
        )
    else:
        rope_tensors = rope_setup.get_rope_tensors(seq_len)

    # Verify TT MLA exists
    assert mla_tt is not None, "TT MLA should exist"

    # Create test inputs
    batch_size = 1
    hidden_size = config.hidden_size

    logger.info(f"Creating test inputs: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")

    # Create random input tensor (generate in float32, then convert to bfloat16)
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(torch.bfloat16)

    # Reorder hidden_states for balanced ring attention
    sp_factor = mesh_shape[sp_axis]
    chunk_order = create_balanced_chunk_order(sp_factor) if is_balanced else None
    tt_input = hidden_states.unsqueeze(0)  # [1, batch, seq, hidden]
    if is_balanced:
        tt_input = reorder_tensor_chunks(tt_input, chunk_order, seq_dim=2)

    shard_dims = [None, None]
    shard_dims[tp_axis] = -1
    shard_dims[sp_axis] = -2
    tt_hidden_states = ttnn.from_torch(
        tt_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
    )
    # GLM-5.2 indexer reuse (return_indices / inject_indices): capture this layer's top-k selection, or
    # feed a prior layer's to skip the indexer. Defaults leave the single-shot forward unchanged.
    mla_out = mla_tt.forward(
        hidden_states=tt_hidden_states,
        rope_tensors=rope_tensors,
        kvpe_cache=tt_kvpe_cache,
        indexer_indices=inject_indices,
        return_indexer_indices=return_indices,
        index_kv_cache=index_kv_cache,
    )
    indices = None
    if return_indices:
        tt_output, indices = mla_out
    else:
        tt_output = mla_out

    ttnn.synchronize_device(mesh_device)
    ttnn.distributed_context_barrier()

    if return_indices:
        return tt_output, hidden_states, chunk_order, shard_dims, indices
    return tt_output, hidden_states, chunk_order, shard_dims
