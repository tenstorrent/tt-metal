# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MLA-family (DeepSeek-V3 / Kimi) prefill-runner helpers.

The model-agnostic engine helpers (mesh open, H2D service, trace loading) live in
the common package at ``models.demos.common.prefill.runners.runner_utils``. What
remains here is specific to the MLA kvpe KV layout: chunked-prefill input prep, the
block-cyclic KV-cache PCC check + golden loader, and kvpe-cache diagnostics. The
runtime exposes ``kv_cache_pcc_check`` (which calls the function here); and
``prepare_prefill_input_tensor`` backs ``TtPrefillRuntime.make_chunk_input``.
"""

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks


def prepare_prefill_input_tensor(
    token_ids: list[int],
    mesh_device: ttnn.MeshDevice,
    sp_factor: int,
    is_balanced: bool,
    mesh_shape: tuple,
    sp_axis: int,
) -> ttnn.Tensor:
    """Shard and upload token IDs to device as a prefill input tensor.

    Produces an SP-sharded uint32 ROW_MAJOR DRAM tensor of shape
    [sp_factor, 1, len(token_ids) // sp_factor] — the format expected by
    TtPrefillTransformer.forward.
    """
    isl_per_chip = len(token_ids) // sp_factor
    if is_balanced:
        chunk_order = create_balanced_chunk_order(sp_factor)
        t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    else:
        token_ids_sharded = torch.tensor(token_ids, dtype=torch.int64).reshape(sp_factor, 1, isl_per_chip)
    return ttnn.from_torch(
        token_ids_sharded,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(sp_axis, None)),
    )


def probe_dram_allocatable_base(mesh_device, label: str = "") -> None:
    """Snapshot the DRAM allocator state at the moment this is called.

    Allocates a 1-element DRAM tensor, reads its buffer_address(), then
    deallocates. Shows where the allocator is currently placing new
    buffers, useful for comparing across phases (after-mesh-open vs.
    after-model-build vs. after-compile) to localize where kvpe_cache
    buffer-address divergence enters.

    `label` is optional context to tag the output (e.g. "after-mesh-open").
    """
    tag = f"[{label}] " if label else ""
    try:
        probe = ttnn.empty(
            shape=[1],
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        addr = probe.buffer_address()
        logger.info(f"[buf-trace] {tag}next-allocation buffer_address = {addr} (0x{addr:X})")
        ttnn.deallocate(probe)
    except Exception as e:
        logger.error(f"[buf-trace] {tag}probe ttnn.empty FAILED: {type(e).__name__}: {e}")


def verify_kvpe_cache_layout(mesh_device, kvpe_cache) -> None:
    """Verify that kvpe_cache buffer addresses are consistent across all mesh devices.

    The migration table is built from a single mesh-level buffer_address(), but
    each per-device tensor has its own allocator-assigned address. If those
    diverge, the table reads from the wrong physical location on some devices.
    Also dumps the first 16 physical pages via ttnn._ttnn.reports.get_buffer_pages
    for cross-checking bank/offset encoding against the migration table entries.
    """
    try:
        cache_addr = kvpe_cache.buffer_address()
        logger.info(
            f"[verify-layout] kvpe_cache shape={kvpe_cache.shape} "
            f"buffer_address={cache_addr} "
            f"dtype={kvpe_cache.dtype} layout={kvpe_cache.layout}"
        )
        logger.info(f"[verify-layout] kvpe_cache memory_config={kvpe_cache.memory_config()}")

        try:
            device_tensors = ttnn.get_device_tensors(kvpe_cache)
            addr_set = set()
            for i, dt in enumerate(device_tensors):
                try:
                    dt_addr = dt.buffer_address()
                    try:
                        dev_id = dt.device().id() if dt.device() is not None else None
                    except Exception:
                        dev_id = None
                    addr_set.add(dt_addr)
                    if i < 4 or i >= len(device_tensors) - 4 or dt_addr != cache_addr:
                        logger.info(
                            f"[verify-layout] device_tensors[{i}] device_id={dev_id} "
                            f"buffer_address={dt_addr} (delta vs mesh={dt_addr - cache_addr})"
                        )
                except Exception as e:
                    logger.error(f"[verify-layout] device_tensors[{i}] buffer_address FAILED: {type(e).__name__}: {e}")
            consistent = addr_set == {cache_addr}
            logger.info(
                f"[verify-layout] PER-DEVICE buffer_address: "
                f"{len(addr_set)} unique value(s) across {len(device_tensors)} device tensors. "
                f"Mesh-level cache.buffer_address()={cache_addr}. "
                f"{'CONSISTENT' if consistent else 'MISMATCH — migration table built from mesh address but per-device addresses differ!'}"
            )
        except Exception as e:
            logger.error(f"[verify-layout] per-device address dump FAILED: {type(e).__name__}: {e}")

        try:
            all_pages = ttnn._ttnn.reports.get_buffer_pages(mesh_device)
            cache_pages = [p for p in all_pages if p.address == cache_addr]
            logger.info(f"[verify-layout] kvpe_cache has {len(cache_pages)} pages across mesh; sampling first 16:")
            cache_pages.sort(key=lambda p: (p.device_id, p.bank_id, p.page_index))
            for i, p in enumerate(cache_pages[:16]):
                logger.info(
                    f"[verify-layout] page[{i}]: device_id={p.device_id} "
                    f"bank_id={p.bank_id} core=({p.core_x},{p.core_y}) "
                    f"page_index={p.page_index} page_address={p.page_address} "
                    f"page_size={p.page_size}"
                )
        except Exception as e:
            logger.error(f"[verify-layout] page dump FAILED: {type(e).__name__}: {e}")
    except Exception as e:
        logger.error(f"[verify-layout] dump FAILED: {type(e).__name__}: {e}")


def dump_kv_cache_shard_readback(layer_idx: int, kvpe_cache, sample_positions=None) -> None:
    """Dump KV cache bytes via the ttnn shard-spec path (host pull on device 0).

    Reads `device_tensors[0]` (which holds global positions 0..seq_len_local-1)
    and prints the first 16 bytes at each sample position. Use this to verify
    what the cache actually contains for the SP=0 shard — compare against the
    migration table's raw-NOC reads at the same positions (see the blaze-side
    `dump_migration_table_at_layer` helper) to detect table-vs-cache address
    mismatches.

    Args:
        layer_idx: which layer's KV slice to dump (cache shape is
            [num_layers, 1, seq_len_local, head_dim]).
        kvpe_cache: the live KVPE cache mesh tensor.
        sample_positions: list of global token positions to inspect. Defaults
            to early-layer (0/32/64/96) + a pad-region sample (1024/1056/...).
    """
    import torch

    if sample_positions is None:
        sample_positions = [0, 32, 64, 96, 128, 1024, 1056, 1088, 1120]

    try:
        device_tensors = ttnn.get_device_tensors(kvpe_cache)
        try:
            dev0_phys_id = device_tensors[0].device().id() if device_tensors[0].device() is not None else None
        except Exception:
            dev0_phys_id = None
        try:
            dev0_buf_addr = device_tensors[0].buffer_address()
        except Exception:
            dev0_buf_addr = None
        try:
            mesh_buf_addr = kvpe_cache.buffer_address()
        except Exception:
            mesh_buf_addr = None
        delta = (dev0_buf_addr - mesh_buf_addr) if (dev0_buf_addr is not None and mesh_buf_addr is not None) else "n/a"
        logger.info(
            f"[verify-readback] device_tensors[0]: device_id={dev0_phys_id} "
            f"buffer_address={dev0_buf_addr} (mesh.buffer_address={mesh_buf_addr}, delta={delta})"
        )
        dev0 = ttnn.to_torch(device_tensors[0])  # [num_layers, 1, seq_len_local, head_dim]
        seq_len_local = dev0.shape[2]
        for global_pos in sample_positions:
            if global_pos >= seq_len_local:
                continue
            row = dev0[layer_idx, 0, global_pos, :]
            head_bytes = row.contiguous().view(torch.uint8)[:16].tolist()
            head_uint32 = row.contiguous().view(torch.uint32)[:4].tolist()
            logger.info(
                f"[verify-readback] layer={layer_idx} dev=0 global_pos={global_pos} "
                f"bytes[0..16]={head_bytes} uint32[0..4]={head_uint32}"
            )
    except Exception as e:
        logger.error(f"[verify-readback] FAILED layer={layer_idx}: {type(e).__name__}: {e}")


def _load_golden_kv_post(trace_dir, layer_idx: int, total_len: int):
    """[total_len, 576] golden kv_post_transform for one layer, format-agnostic:
    - DeepSeek: a single kv_cache/layer_N.safetensors holding the full tensor.
    - Kimi (vllm): kv_cache/layer_N/rows_<start>_<end>.safetensors shards, concatenated by start row.
    """
    import torch
    from safetensors import safe_open

    key = f"kv_post_transform_layer_{layer_idx}"
    single = Path(trace_dir) / "kv_cache" / f"layer_{layer_idx}.safetensors"
    if single.exists():
        with safe_open(single, framework="pt") as f:
            return f.get_slice(key)[:total_len].to(torch.float32)
    layer_dir = Path(trace_dir) / "kv_cache" / f"layer_{layer_idx}"
    shards = sorted(layer_dir.glob("rows_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
    rows, have = [], 0
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            t = f.get_tensor(key)
        rows.append(t)
        have += t.shape[0]
        if have >= total_len:
            break
    return torch.cat(rows, dim=0)[:total_len].to(torch.float32)


def kv_cache_pcc_check(
    pipeline, kvpe_cache, slot_id: int, n_chunks: int, trace_dir=None, first_layer_idx: int = 0
) -> float:
    """Gather `kvpe_cache` (the engine-owned KV cache) for `slot_id`, un-rotate the block-cyclic
    layout to natural order, and PCC-compare each layer against the golden DeepSeek-R1
    `kv_post_transform` trace. Returns the min per-layer PCC and asserts (unless
    PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1) when any layer is below threshold.

    `trace_dir` defaults to the resolved PREFILL_TRACE_DIR env (caller passes the adapter's
    prefill_trace_default). The golden is loaded format-agnostically (DeepSeek single-file or Kimi vllm
    row-shards) via _load_golden_kv_post.

    `first_layer_idx` offsets the golden layer index for a pipeline-parallel rank: the device cache
    holds this rank's `num_layers` slice at local indices, but the golden trace is indexed by global
    layer, so golden layer = first_layer_idx + local_idx. Defaults to 0 for single-rank.

    Env:
      PREFILL_STANDALONE_CHUNKED_PCC          min per-layer KV-cache PCC threshold (default 0.88)
      PREFILL_STANDALONE_CHUNKED_RECORD_ONLY  "1" -> log PCC only, do not assert
    """
    import torch

    from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
    from tests.ttnn.utils_for_testing import comp_pcc

    cfg = pipeline.config
    mesh_device = pipeline.mesh_device
    sp = cfg.sp_factor
    chunk_size = cfg.chunk_size
    num_layers = cfg.num_layers
    seq_len_cache = cfg.max_seq_len
    total_len = n_chunks * chunk_size

    trace_dir = resolve_trace_dir(trace_dir if trace_dir is not None else os.environ["PREFILL_TRACE_DIR"])

    threshold = float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88"))
    record_only = os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0") == "1"
    kv_lora = pipeline.hf_config.kv_lora_rank
    kvpe_dim = pipeline.hf_config.qk_rope_head_dim + kv_lora

    # One gather: [num_users*num_layers, tp_replicas, seq_len_cache, kvpe] -> collapse TP via [:, :1].
    cache_full = ttnn.to_torch(
        kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[:, :1]

    p = blockcyclic_positions(sp, chunk_size, seq_len_cache)
    logger.info(f"[kv-pcc] device KV cache vs golden kv_post_transform (slot={slot_id}, per layer):")
    min_pcc = 1.0
    failures = []
    for i in range(num_layers):
        # user-major slot layout: cache batch index = slot_id * num_layers + local_layer_idx
        batch_idx = slot_id * num_layers + i
        global_layer = first_layer_idx + i  # golden trace is indexed by global layer
        nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
        nat[p] = cache_full[batch_idx, 0]  # un-rotate block-cyclic -> natural order
        dev_cache = nat[:total_len]

        g_post = _load_golden_kv_post(trace_dir, global_layer, total_len)
        # nope (kv_lora) compares directly; the RoPE (pe) slice uses the Meta-interleaved basis while
        # the golden stores the HF half-split, so re-interleave the golden before comparing.
        _, pcc_nope = comp_pcc(g_post[:, :kv_lora], dev_cache[:, :kv_lora])
        ref_pe = g_post[:, kv_lora:]
        d = ref_pe.shape[-1]
        ref_pe_int = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(-1, d)  # HF -> Meta
        _, pcc_pe = comp_pcc(ref_pe_int, dev_cache[:, kv_lora:])
        layer_pcc = min(pcc_nope, pcc_pe)
        min_pcc = min(min_pcc, layer_pcc)
        logger.info(
            f"  cache layer local={i} global={global_layer} PCC: "
            f"nope={pcc_nope:.6f} pe(interleaved)={pcc_pe:.6f} -> {layer_pcc:.6f}"
        )
        if layer_pcc < threshold:
            failures.append((i, layer_pcc))

    logger.info(f"[kv-pcc] KV cache min PCC across {num_layers} layers: {min_pcc:.6f} (threshold {threshold})")
    # stdout, not a log line: callers (tests / orchestrators) parse this.
    print(
        f"[standalone-chunked] kv_cache_pcc_complete slot={slot_id} n_chunks={n_chunks} "
        f"total_len={total_len} min_pcc={min_pcc:.6f}"
    )
    if failures:
        msg = "; ".join(f"layer {layer} PCC {pcc:.6f} < {threshold}" for layer, pcc in failures)
        if record_only:
            logger.warning(f"[kv-pcc] sub-threshold PCC (record-only, not asserted): {msg}")
        else:
            raise AssertionError(f"[kv-pcc] KV cache PCC below {threshold}: {msg}")
    else:
        logger.success(f"[kv-pcc] KV cache PCC PASSED (min {min_pcc:.6f} >= {threshold})")
    return min_pcc
