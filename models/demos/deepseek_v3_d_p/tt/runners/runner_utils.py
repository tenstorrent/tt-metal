# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Utilities for the prefill runner.

Only metal-native helpers live here — pure ttnn, no upward dependencies
on blaze (`_migration`, `_mpi_test_helpers`). Migration-coupled diagnostics
live in blaze at `disaggregation/migration/python/prefill_runner_util.py`.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config


# ---------------------------------------------------------------------------
# Model-variant registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunnerVariant:
    """Per-model knobs the runner needs to build a DeepSeek-V3-family model.

    The TT layer code is variant-agnostic — it takes `model_config` (the static
    dimension constants) + the HF `config`. Everything here is the runner-side
    plumbing that differs per model: where to find the HF config and the TTNN
    weight cache, and the sensible defaults for input layout / gate mode.
    """

    name: str  # matches the pytest weight-cache dir prefix: {name}_{arch}_{N}dev
    model_config: type  # DeepSeekV3Config | KimiK26Config
    hf_model_env: str  # env var holding the HF model dir (for config.json)
    hf_model_default: str
    ttnn_cache_env: str  # env var holding the TTNN weight-cache root
    ttnn_cache_default: str
    default_is_balanced: bool
    default_gate_mode: str  # GateComputeMode name


VARIANTS = {
    "deepseek_v3_d_p": RunnerVariant(
        name="deepseek_v3_d_p",
        model_config=DeepSeekV3Config,
        hf_model_env="DEEPSEEK_V3_HF_MODEL",
        hf_model_default="models/demos/deepseek_v3/reference",
        ttnn_cache_env="TT_DS_PREFILL_TTNN_CACHE",
        ttnn_cache_default="/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure",
        default_is_balanced=True,
        default_gate_mode="DEVICE_FP32",
    ),
    "kimi_k2_6": RunnerVariant(
        name="kimi_k2_6",
        model_config=KimiK26Config,
        hf_model_env="KIMI_K2_6_HF_MODEL",
        # Repo-local config (dot-free, in-tree) — same default the pytest
        # `kimi_k2_6` variant uses (model_variants.py default_local_path). The
        # runner only needs config dims; real weights come from the TTNN cache.
        # To use a different checkpoint, set KIMI_K2_6_HF_MODEL to a dot-free
        # path (transformers' trust_remote_code import chokes on the "." in the
        # canonical /mnt/models/moonshotai/Kimi-K2.6-dequantized dir name).
        hf_model_default="models/demos/deepseek_v3_d_p/reference/kimi_k2_6",
        ttnn_cache_env="TT_KIMI_PREFILL_TTNN_CACHE",
        ttnn_cache_default="/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill",
        default_is_balanced=False,  # Kimi is validated only in non_balanced layout
        default_gate_mode="DEVICE_FP32",  # Kimi (1 expert group) routes through the fp32 grouped-topk device kernel (handles n_groups == 1)
    ),
}


def get_variant(name: str) -> RunnerVariant:
    """Resolve a RunnerVariant by name; raises KeyError with the valid set."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise KeyError(f"Unknown PREFILL_MODEL_VARIANT={name!r}; valid: {sorted(VARIANTS)}")


# ---------------------------------------------------------------------------
# HF config loading
# ---------------------------------------------------------------------------
def unwrap_multimodal_config(cfg):
    """Unwrap Kimi K2.5/K2.6's multimodal wrapper config to the inner text_config.

    The LM fields the rest of the code reads (hidden_size, n_routed_experts, ...)
    live under `text_config`. Also stubs `quantization_config.weight_block_size`
    when missing so DSv3's dequant helper's eager read doesn't fail on the
    pre-dequantized Kimi checkpoint. Mirrors the test-side helper in conftest.py.
    """
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        logger.info(f"Unwrapping multimodal wrapper config (inner model_type={cfg.text_config.model_type})")
        cfg = cfg.text_config
    qc = getattr(cfg, "quantization_config", None)
    if isinstance(qc, dict) and not qc.get("weight_block_size"):
        qc["weight_block_size"] = [128, 128]
        logger.info("Stubbed quantization_config.weight_block_size for pre-dequantized checkpoint")
    return cfg


def load_hf_config(variant: RunnerVariant):
    """Load (and unwrap) the HF config for a variant from its `hf_model_env`
    (falling back to the variant's repo-local default)."""
    model_path = os.environ.get(variant.hf_model_env) or variant.hf_model_default
    logger.info(f"Loading HF config for variant={variant.name!r} from {model_path}")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return unwrap_multimodal_config(cfg)


# ---------------------------------------------------------------------------
# Device / weight-cache / H2D-service setup
# ---------------------------------------------------------------------------
def open_mesh_device(mesh_shape: tuple, model_cfg: type) -> ttnn.MeshDevice:
    """Configure fabric (1D for sp<=8, else 2D) and open the mesh device."""
    sp = mesh_shape[0]
    fabric_config = ttnn.FabricConfig.FABRIC_1D if sp <= 8 else ttnn.FabricConfig.FABRIC_2D

    fabric_router_config = create_fabric_router_config(
        max_payload_size=model_cfg.FABRIC_PAYLOAD_SIZE,
    )

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    # PERF EXPERIMENT (PREFILL_NUM_CQ): the H2D service imposes a per-op-launch dispatch tax on the
    # request loop (tracy: on-device time identical, op-to-op latency +33%). Opening >1 command queue
    # lets the service claim its own CQ so the model's CQ0 dispatch is uncontended. Default 1 (upstream).
    _num_cq = int(os.environ.get("PREFILL_NUM_CQ", "1"))
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), num_command_queues=_num_cq)


def resolve_weight_cache_path(variant: RunnerVariant, mesh_shape: tuple) -> Optional[Path]:
    """Mirror the layout produced by the pytest weight_cache_path fixture so
    we read the same files the cache-populate run wrote:
      $<variant.ttnn_cache_env> / {variant.name}_{arch}_{N}dev / {sp}x{tp}
    Defaults to the variant's cache root; returns None only if explicitly empty."""
    env_cache = os.environ.get(variant.ttnn_cache_env, variant.ttnn_cache_default)
    if not env_cache:
        return None
    arch = "bh" if is_blackhole() else "wh"
    num_devices = ttnn.get_num_devices()
    sp, tp = mesh_shape
    path = Path(env_cache) / f"{variant.name}_{arch}_{num_devices}dev" / f"{sp}x{tp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_global_spec(mesh_shape: tuple, max_seq_len: int) -> ttnn.TensorSpec:
    """Per-iter input spec used by `build_h2d_service` to set the service's
    global tensor shape (the producer matches it on the host side).
    Shape `(sp_factor, 1, isl_per_chip)` uint32 ROW_MAJOR DRAM."""
    sp_factor = mesh_shape[0]
    isl_per_chip = max_seq_len // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def build_h2d_service(
    mesh_device: ttnn.MeshDevice,
    *,
    mesh_shape: tuple,
    max_seq_len: int,
    mapper_config: ttnn.MeshMapperConfig,
    worker_cores: ttnn.CoreRange,
    metadata_size_bytes: int,
) -> ttnn.H2DStreamService:
    """Construct an H2DStreamService whose per-shard backing tensor matches
    what `prepare_prefill_input_tensor` would have produced.

    Per-shard target: `(1, 1, isl_per_chip)` uint32 ROW_MAJOR DRAM.
    Achieved by setting global_spec.shape = `(sp_factor, 1, isl_per_chip)` and
    mapping `[Shard(0), Replicate]` on a `(sp, tp)` mesh — first axis of the
    tensor is sharded across mesh rows (sp), nothing else is split.
    """
    sp_factor, tp_factor = mesh_shape
    assert max_seq_len % sp_factor == 0, f"max_seq_len={max_seq_len} must be divisible by sp_factor={sp_factor}"
    isl_per_chip = max_seq_len // sp_factor
    per_chip_bytes = isl_per_chip * 4  # uint32

    global_spec = make_global_spec(mesh_shape, max_seq_len)
    mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)
    # worker_cores set so the service-core kernel multicasts a data-ready inc
    # after each transfer; h2d_socket_sync() waits on that on-device, which
    # avoids the host-side barrier() round-trip per iteration.
    # metadata_size_bytes set so the producer can ship per-iter control bytes
    # (slot_id, actual_start, actual_end) inline with the token push.
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_chip_bytes,  # 8 in-flight pages of headroom
        scratch_cb_size_bytes=per_chip_bytes,  # one page; service requires >= page_size
        mapper=mapper,
        worker_cores=worker_cores,
        metadata_size_bytes=metadata_size_bytes,
    )
    logger.info(
        f"[h2d] H2DStreamService built: global_shape=({sp_factor},1,{isl_per_chip}) "
        f"uint32 ROW_MAJOR DRAM, per_chip_bytes={per_chip_bytes}, worker_cores={worker_cores}"
    )
    return service


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
