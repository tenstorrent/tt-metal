# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-model KV cache regression against a self-contained GPU trace dataset."""

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

import ttnn
from models.demos.gemma4_d_p.demo.text_demo_prefill import (
    TRACE_REGION_SIZE,
    _build_prefill_model,
    _mesh_config,
    _model_path,
    _run_traced_prefill,
    _validate_prefill_shape,
)
from models.demos.gemma4_d_p.tests.kv_pcc.data import KvPccDataset
from models.demos.gemma4_d_p.tests.kv_pcc.report import KvPccRun, digest
from models.demos.gemma4_d_p.tests.test_factory import parametrize_mesh_with_fabric


def _read_ring_cache(tensor, context_len, chunk, cp, tp):
    """Read a populated cache prefix and restore [tokens, heads, width] order.

    Untilizing on device avoids the generic mesh-composer path, and slicing the
    sequence first prevents unused cache capacity from reaching the host. Each
    CP shard stores its local slab of every chunk consecutively; interleave those
    slabs while concatenating TP-local heads.
    """
    if context_len % chunk or chunk % cp:
        raise ValueError("KV readback requires complete chunks divisible by CP")
    local_len = context_len // cp
    selected = ttnn.slice(
        tensor,
        (0, 0, 0, 0),
        (1, tensor.shape[1], local_len, tensor.shape[3]),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        row_major = ttnn.untilize(selected, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    finally:
        ttnn.deallocate(selected)
    try:
        host = ttnn.from_device(row_major, blocking=True)
    finally:
        ttnn.deallocate(row_major)

    shards = [ttnn.to_torch(shard).float() for shard in ttnn.get_device_tensors(host)]
    if len(shards) != cp * tp:
        raise ValueError(f"expected {cp * tp} cache shards, got {len(shards)}")
    local_heads, width = shards[0].shape[1], shards[0].shape[3]
    chunks, local_chunk = context_len // chunk, chunk // cp
    gathered = torch.empty((local_heads * tp, chunks, cp, local_chunk, width), dtype=torch.float32)
    for cp_row in range(cp):
        for tp_column in range(tp):
            shard = shards[cp_row * tp + tp_column].reshape(local_heads, chunks, local_chunk, width)
            gathered[tp_column * local_heads : (tp_column + 1) * local_heads, :, cp_row].copy_(shard)
    return gathered.reshape(local_heads * tp, context_len, width).permute(1, 0, 2)


def _kv_accuracy(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Compute PCC and relative L2 from shared float64 inputs."""
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    centered_a = a - a.mean()
    centered_b = b - b.mean()
    d = centered_a.norm() * centered_b.norm()
    correlation = float("nan") if d == 0 else float(torch.clamp((centered_a @ centered_b) / d, -1.0, 1.0))
    del centered_a, centered_b
    n = b.norm()
    relative_error = float("nan") if n == 0 else float((a - b).norm() / n)
    return correlation, relative_error


def _measure_kv_pcc(model, mesh_config, ref_dir, kv_streams, metadata, tokens, chunk):
    """Compare populated caches against the matching reference after all replays complete.

    Score each reference block separately.
    """
    from models.demos.gemma4_d_p.tt.attention.global_kv_cache import (
        GLOBAL_HEAD_DIM,
        GLOBAL_ROTARY_DIM,
        pack_global_kv_reference,
        sliding_kv_indices,
    )
    from models.demos.gemma4_d_p.tt.attention.ring_prefill import GlobalRingKVCache

    reference_tokens = torch.tensor(metadata["token_ids"][: tokens.shape[-1]], dtype=torch.int64).unsqueeze(0)
    assert torch.equal(
        tokens.to(torch.int64), reference_tokens
    ), "GPU traces and the device model implementation must use the same input"
    assert len(kv_streams) == len(model.layers), "KV reference layer count differs from model"
    for layer_idx, blocks in enumerate(kv_streams):
        next_row = 0
        for block in blocks:
            assert (
                block["row_start"] == next_row and block["row_end"] > next_row
            ), f"layer {layer_idx}: KV reference ranges must be contiguous and start at zero"
            next_row = min(block["row_end"], tokens.shape[-1])
        assert (
            next_row == tokens.shape[-1]
        ), f"layer {layer_idx}: KV reference rows must match the complete input sequence"

    records = []
    context_len = tokens.shape[-1]
    cp, tp = mesh_config.cp_degree, mesh_config.tp_degree
    for layer_idx, layer in enumerate(model.layers):
        attn = layer.self_attn
        assert attn.ring_kv_cache is not None, f"layer {layer_idx}: missing ring cache"
        heads, dim = attn.config.num_key_value_heads, attn.config.head_dim
        packed = isinstance(attn.ring_kv_cache, GlobalRingKVCache)
        if packed:
            actual_packed = _read_ring_cache(attn.ring_kv_cache.kv, context_len, chunk, cp, tp)
            actual_parts = {
                "K": actual_packed[..., :GLOBAL_HEAD_DIM],
                "V": actual_packed[..., GLOBAL_ROTARY_DIM:],
            }
            del actual_packed
        else:
            # Holding K and V together costs no more than the previous K readback
            # plus every reference shard retained until the V pass. It also lets
            # each reference shard in the 381-GB dataset be read once and released.
            actual_parts = {
                "K": _read_ring_cache(attn.ring_kv_cache.k, context_len, chunk, cp, tp),
                "V": _read_ring_cache(attn.ring_kv_cache.v, context_len, chunk, cp, tp),
            }

        for block_idx, block in enumerate(kv_streams[layer_idx]):
            start, end = block["row_start"], min(block["row_end"], context_len)
            path = ref_dir / block["path"]
            golden = load_file(str(path))[f"kv_post_transform_layer_{layer_idx}"]
            assert golden.shape == (block["row_end"] - start, 2 * heads * dim), f"{path}: invalid KV shape"
            golden = golden[: end - start]
            golden_k = golden[:, : heads * dim].reshape(end - start, heads, dim)
            golden_v = golden[:, heads * dim :].reshape(end - start, heads, dim)
            if packed:
                golden_packed = pack_global_kv_reference(golden_k, golden_v)
                golden_parts = {
                    "K": golden_packed[..., :GLOBAL_HEAD_DIM],
                    "V": golden_packed[..., GLOBAL_ROTARY_DIM:],
                }
            else:
                golden_parts = {
                    "K": golden_k.index_select(-1, sliding_kv_indices(dim)),
                    "V": golden_v,
                }
            for part in ("K", "V"):
                actual = actual_parts[part][start:end]
                golden_part = golden_parts[part]
                value, error = _kv_accuracy(actual, golden_part)
                records.append(
                    {
                        "layer": layer_idx,
                        "part": "K" if part == "K_effective" else part,
                        "chunk": block_idx,
                        "row_start": start,
                        "row_end": end,
                        "pcc": value,
                        "rel_l2": error,
                    }
                )
                logger.info(f"[kv_pcc] layer={layer_idx} {part} rows=[{start},{end}) pcc={value:.6f} relL2={error:.6f}")
        del actual, golden_part, golden_parts, golden_k, golden_v, golden, actual_parts
        if packed:
            del golden_packed
    return records


@torch.no_grad()
@pytest.mark.timeout(7200)
@parametrize_mesh_with_fabric([(8, 4)], device_params_extra={"trace_region_size": TRACE_REGION_SIZE})
@pytest.mark.parametrize("context_len", [262144], ids=lambda c: f"ctx_{c // 1024}k")
@pytest.mark.parametrize("chunk_size", [8192], ids=lambda c: f"chunk{c}")
def test_kv_pcc(mesh_device, context_len, chunk_size, reset_seeds, request):
    """Compare every layer's K/V PCC and relative L2 with the dataset baseline.

    Tokenize the dataset's input.txt and require an exact match with the GPU
    reference token IDs (or their prefix for a shorter requested context).
    Save per-chunk JSON measurements and fail on baseline tolerance violations.
    """
    mesh_config = _mesh_config(mesh_device)
    _validate_prefill_shape(mesh_config, context_len, chunk_size)
    model_path = _model_path()
    model_args, model, _kv_cache = _build_prefill_model(
        mesh_config=mesh_config,
        model_path=model_path,
        chunk_size=chunk_size,
        context_len=context_len,
    )

    dataset = KvPccDataset(request.config.getoption("--kv-pcc-data"))
    kv_metadata = dataset.metadata
    kv_streams = dataset.reference_blocks(context_len)
    configuration = {
        "model": model_path,
        "mesh_shape": list(mesh_device.shape),
        "chunk_size": chunk_size,
        "context_len": context_len,
        "token_source": "text",
        "input_text_sha256": dataset.input_text_sha256,
        "token_ids_sha256": digest(kv_metadata["token_ids"][:context_len]),
        "reference_metadata_sha256": digest(kv_metadata),
        "reference_index_sha256": digest(dataset.index),
        "layer_types": kv_metadata["layer_types"],
        "cache_comparison": "global_effective_k_and_packed_v_sliding_reordered_k_v1",
    }
    kv_run = KvPccRun(configuration, request.node.nodeid, baseline_path=dataset.baseline_path)
    tokens_all = torch.tensor(
        dataset.token_ids(model_path, context_len, model_n_layers=len(model.layers)),
        dtype=torch.int32,
    ).unsqueeze(0)
    assert int(tokens_all.min()) >= 0 and int(tokens_all.max()) < model_args.vocab_size, "Input token outside vocab"
    logger.info("[kv_pcc] Performance reporting disabled")

    _run_traced_prefill(model, mesh_config, tokens_all, chunk_size, readback_all=False, report_performance=False)
    records = _measure_kv_pcc(model, mesh_config, dataset.directory, kv_streams, kv_metadata, tokens_all, chunk_size)
    report_path = kv_run.finish(records)
    logger.info("[kv_pcc] PASS: all K/V measurements are within baseline tolerances.")
    logger.info(f"[kv_pcc] JSON report: {report_path}")
