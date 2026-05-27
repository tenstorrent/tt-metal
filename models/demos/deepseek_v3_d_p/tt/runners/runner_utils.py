# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Debug/diagnostic utilities for the prefill runner.

Only the metal-native helpers live here — pure ttnn, no upward dependencies
on blaze (`_migration`, `_mpi_test_helpers`). Migration-coupled diagnostics
live in blaze at `disaggregation/migration/python/prefill_runner_util.py`.

Import selectively — all helpers are gated on `PREFILL_DEBUG=1` at the
caller; importing this module is cheap and safe in any context.
"""

from loguru import logger

import ttnn


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
