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
    hf_model_default: str  # HF model dir for config.json; PREFILL_HF_MODEL overrides
    ttnn_cache_default: str  # TTNN weight-cache root; PREFILL_TTNN_CACHE overrides
    default_gate_mode: str  # GateComputeMode name
    prefill_trace_default: (
        str  # golden trace dir (input token_ids + kv_post_transform); resolve_trace_dir descends one level if needed
    )


VARIANTS = {
    "deepseek_v3_d_p": RunnerVariant(
        name="deepseek_v3_d_p",
        model_config=DeepSeekV3Config,
        hf_model_default="models/demos/deepseek_v3/reference",
        ttnn_cache_default="/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure",
        default_gate_mode="DEVICE_FP32",
        prefill_trace_default="/mnt/models/deepseek-prefill-cache/golden/longbook_qa_eng_prefill_56320_nopad",
    ),
    "kimi_k2_6": RunnerVariant(
        name="kimi_k2_6",
        model_config=KimiK26Config,
        # Repo-local config (dot-free, in-tree). The runner only needs config dims; real weights come
        # from the TTNN cache. To use a different checkpoint, set PREFILL_HF_MODEL to a dot-free path
        # (transformers' trust_remote_code import chokes on the "." in the canonical
        # /mnt/models/moonshotai/Kimi-K2.6-dequantized dir name).
        hf_model_default="models/demos/deepseek_v3_d_p/reference/kimi_k2_6",
        ttnn_cache_default="/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill",
        default_gate_mode="DEVICE_FP32",  # Kimi (1 expert group)
        # vllm-traced golden: metadata.json + kv_cache live under a single run-hash subdir, and the
        # per-layer KV is row-sharded into layer_N/rows_*.safetensors. resolve_trace_dir descends to
        # the subdir; kv_cache_pcc_check reassembles the shards.
        prefill_trace_default="/mnt/models/deepseek-prefill-cache/golden/kimi-26/kimi_longbook_56320",
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
    """Load (and unwrap) the HF config for a variant from PREFILL_HF_MODEL
    (falling back to the variant's repo-local default)."""
    model_path = os.environ.get("PREFILL_HF_MODEL") or variant.hf_model_default
    logger.info(f"Loading HF config for variant={variant.name!r} from {model_path}")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return unwrap_multimodal_config(cfg)


# ---------------------------------------------------------------------------
# Device / weight-cache / H2D-service setup
# ---------------------------------------------------------------------------
def open_mesh_device(mesh_shape: tuple, model_cfg: type, l1_small_size: int = 0) -> ttnn.MeshDevice:
    """Configure fabric (1D for sp<=8, else 2D) and open the mesh device. `l1_small_size` > 0 carves an
    L1_SMALL region (needed when an op routes its semaphores there, e.g. the Kimi MoE routing all-gather
    with use_l1_small_for_semaphores)."""
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
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=l1_small_size)


def resolve_weight_cache_path(variant: RunnerVariant, mesh_shape: tuple) -> Optional[Path]:
    """Mirror the layout produced by the pytest weight_cache_path fixture so
    we read the same files the cache-populate run wrote:
      $PREFILL_TTNN_CACHE / {variant.name}_{arch}_{N}dev / {sp}x{tp}
    Defaults to the variant's cache root; returns None only if explicitly empty."""
    env_cache = os.environ.get("PREFILL_TTNN_CACHE", variant.ttnn_cache_default)
    if not env_cache:
        return None
    arch = "bh" if is_blackhole() else "wh"
    num_devices = ttnn.get_num_devices()
    sp, tp = mesh_shape
    path = Path(env_cache) / f"{variant.name}_{arch}_{num_devices}dev" / f"{sp}x{tp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_global_spec(mesh_shape: tuple, chunk_size: int) -> ttnn.TensorSpec:
    """Per-push input spec used by `build_h2d_service` to set the service's
    global tensor shape (the producer matches it on the host side). One push carries one
    chunk_size-token chunk. Shape `(sp_factor, 1, chunk_size // sp_factor)` uint32 ROW_MAJOR DRAM."""
    sp_factor = mesh_shape[0]
    isl_per_chip = chunk_size // sp_factor
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
    chunk_size: int,
    mapper_config: ttnn.MeshMapperConfig,
    worker_cores: ttnn.CoreRange,
    metadata_size_bytes: int,
) -> ttnn.H2DStreamService:
    """Construct an H2DStreamService whose per-shard backing tensor matches
    what `prepare_prefill_input_tensor` would have produced. Each push carries one chunk_size-token
    chunk (chunked prefill streams one chunk per push), not the full sequence.

    Per-shard target: `(1, 1, chunk_size // sp_factor)` uint32 ROW_MAJOR DRAM.
    Achieved by setting global_spec.shape = `(sp_factor, 1, chunk_size // sp_factor)` and
    mapping `[Shard(0), Replicate]` on a `(sp, tp)` mesh — first axis of the
    tensor is sharded across mesh rows (sp), nothing else is split.
    """
    sp_factor, tp_factor = mesh_shape
    assert chunk_size % sp_factor == 0, f"chunk_size={chunk_size} must be divisible by sp_factor={sp_factor}"
    isl_per_chip = chunk_size // sp_factor
    per_chip_bytes = isl_per_chip * 4  # uint32

    global_spec = make_global_spec(mesh_shape, chunk_size)
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


def resolve_trace_dir(path) -> Path:
    """Resolve a trace dir to the one holding metadata.json. vllm traces nest metadata.json + kv_cache
    under a single run-hash subdir, so if `path` itself has no metadata.json, descend into the sole
    subdir that does."""
    path = Path(path)
    if (path / "metadata.json").exists():
        return path
    subs = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]
    if len(subs) != 1:
        raise FileNotFoundError(f"no metadata.json in {path} or a unique subdir (found {len(subs)} candidates)")
    return subs[0]


def load_trace_token_ids(trace_dir, total_len=None) -> list:
    """Input token_ids from a resolved trace's metadata.json (optionally truncated to total_len)."""
    import json

    with open(Path(trace_dir) / "metadata.json") as f:
        md = json.load(f)
    tids = list(md["token_ids"])
    return tids[:total_len] if total_len is not None else tids


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


def kv_cache_pcc_check(pipeline, slot_id: int, n_chunks: int, trace_dir=None) -> float:
    """Gather the device KV cache for `slot_id`, un-rotate the block-cyclic layout to natural order,
    and PCC-compare each layer against the golden DeepSeek-R1 `kv_post_transform` trace. Returns the
    min per-layer PCC and asserts (unless PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1) when any layer is
    below threshold.

    `trace_dir` defaults to the resolved PREFILL_TRACE_DIR env (caller passes the variant's
    prefill_trace_default). The golden is loaded format-agnostically (DeepSeek single-file or Kimi vllm
    row-shards) via _load_golden_kv_post.

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
        pipeline.kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[:, :1]

    p = blockcyclic_positions(sp, chunk_size, seq_len_cache)
    logger.info(f"[kv-pcc] device KV cache vs golden kv_post_transform (slot={slot_id}, per layer):")
    min_pcc = 1.0
    failures = []
    for i in range(num_layers):
        # user-major slot layout: cache batch index = slot_id * num_layers + layer_idx
        batch_idx = slot_id * num_layers + i
        nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
        nat[p] = cache_full[batch_idx, 0]  # un-rotate block-cyclic -> natural order
        dev_cache = nat[:total_len]

        g_post = _load_golden_kv_post(trace_dir, i, total_len)
        # nope (kv_lora) compares directly; the RoPE (pe) slice uses the Meta-interleaved basis while
        # the golden stores the HF half-split, so re-interleave the golden before comparing.
        _, pcc_nope = comp_pcc(g_post[:, :kv_lora], dev_cache[:, :kv_lora])
        ref_pe = g_post[:, kv_lora:]
        d = ref_pe.shape[-1]
        ref_pe_int = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(-1, d)  # HF -> Meta
        _, pcc_pe = comp_pcc(ref_pe_int, dev_cache[:, kv_lora:])
        layer_pcc = min(pcc_nope, pcc_pe)
        min_pcc = min(min_pcc, layer_pcc)
        logger.info(f"  cache layer {i} PCC: nope={pcc_nope:.6f} pe(interleaved)={pcc_pe:.6f} -> {layer_pcc:.6f}")
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
