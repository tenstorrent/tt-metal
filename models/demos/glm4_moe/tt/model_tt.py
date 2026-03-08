# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Top-level model runner for GLM-4.7-REAP-218B on TT hardware.

Provides Glm4MoeTT: a frozen dataclass with create() factory that loads all 92 layers
of weights, creates decoder layers, embedding, final norm, and LM head, and exposes
decode() and prefill() entry points.

Key architecture:
- 92 layers (num_hidden_layers=92)
- Layers 0-2: dense MLP (first_k_dense_replace=3)
- Layers 3-91: MoE (96 routed experts EP=32 + 1 shared expert TP=8)
- Standard GQA attention (96Q/8KV heads, head_dim=128, NOT MLA)
- Galaxy Wormhole: Mesh(8,4), TP=8 (axis 0), EP=32, DP=4 (axis 1)
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

import ttnn
from loguru import logger

from models.common.rmsnorm import RMSNorm
from models.demos.glm4_moe.tt.config import Glm4MoeHParams
from models.demos.glm4_moe.tt.decoder_layer_tt import Glm4MoeDecoderLayer, _sharded_rms_norm
from models.demos.glm4_moe.tt.layer_weights import (
    DecoderLayerTTWeights,
    _env_dense_dtype,
    _tp_axis_and_size,
    _tp_mesh_mapper,
    _linear_weight_tt,
    convert_decoder_layer_weights,
)
from models.demos.glm4_moe.tt.ccl import CCL
from models.demos.glm4_moe.tt.moe_tt import create_moe_runtime, Glm4MoeMoERuntime
from models.demos.glm4_moe.tt.tt_embedding import (
    convert_embedding_weight_to_tt,
    run_tt_embedding,
)
from models.demos.glm4_moe.tt.weights import (
    load_glm_lazy_state_dict,
)


# ---------------------------------------------------------------------------
# RoPE utilities (adapted from glm4_moe_lite/layer0_tt.py for GQA head_dim)
# ---------------------------------------------------------------------------

def _rot_transformation_mat_torch() -> torch.Tensor:
    """Transformation matrix for ttnn.experimental.rotary_embedding_llama."""
    dhead = 32
    rot = torch.zeros(1, 1, dhead, dhead, dtype=torch.float32)
    rot[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1.0
    rot[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1.0
    return rot


def _rope_cos_sin_torch(*, seq_len: int, dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin matrices in NeoX half-rotation form: [1,1,seq_len,dim].

    Last dim layout: [t0, t1, ..., t_{d/2-1}, t0, t1, ..., t_{d/2-1}]
    to match NeoX-style rotate_half(x) = cat(-x[..., d//2:], x[..., :d//2]).
    GLM-4.7 uses NeoX-style RoPE (confirmed from HuggingFace transformers glm4_moe).
    """
    if dim % 2 != 0:
        raise ValueError(f"rope dim must be even, got dim={dim}")
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * (2.0 / dim)))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def _make_rope_tensors(
    *,
    device,
    seq_len: int,
    rope_dim: int,
    rope_theta: float,
) -> dict[str, object]:
    """Create RoPE cos/sin/trans tensors for GQA with partial RoPE.

    For GLM-4.7-REAP: partial_rotary_factor=0.5, so rope_dim=64 (half of head_dim=128).
    RoPE tables are created at rope_dim (the portion that gets rotated).
    """
    cos_t, sin_t = _rope_cos_sin_torch(seq_len=seq_len, dim=rope_dim, base=rope_theta)
    trans_t = _rot_transformation_mat_torch().to(dtype=torch.bfloat16)

    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    cos_host = cos_t.to(dtype=torch.bfloat16).cpu()
    sin_host = sin_t.to(dtype=torch.bfloat16).cpu()

    cos = ttnn.from_torch(
        cos_t.to(dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    sin = ttnn.from_torch(
        sin_t.to(dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    trans = ttnn.from_torch(
        trans_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
    )
    return {
        "cos_matrix": cos,
        "sin_matrix": sin,
        "trans_matrix": trans,
        "cos_matrix_host": cos_host,
        "sin_matrix_host": sin_host,
    }


# ---------------------------------------------------------------------------
# Decode RoPE input preparation (per-step cos/sin for batch of positions)
# ---------------------------------------------------------------------------

def _prepare_decode_rope_and_positions_tt(
    *, device, rope: dict, positions: torch.Tensor, dp_shard_axis: int | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Prepare per-step RoPE cos/sin and position tensors for decode.

    positions: [B] int32 tensor of current positions.
    dp_shard_axis: If set, shard cos/sin along dim=1 (batch) across this mesh axis
        instead of replicating.  Used on TG (Galaxy) where DP groups each process
        a subset of the batch.  The mesh axis must evenly divide the batch size.
    Returns: (tt_positions, cos_batch, sin_batch) on device.
    """
    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    batch = int(positions.shape[0])

    # Upload positions to device.
    # When dp_shard_axis is set, shard positions across DP groups (each group gets
    # only its subset of positions).
    if is_mesh_device and dp_shard_axis is not None:
        mesh_shape = list(device.shape)
        pos_dims = [None, None]
        pos_dims[dp_shard_axis] = 0  # shard tensor dim=0 across dp_shard_axis
        pos_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(pos_dims), mesh_shape=mesh_shape)
    elif is_mesh_device:
        pos_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        pos_mapper = None
    tt_positions = ttnn.from_torch(
        positions.view(-1).contiguous().to(torch.int32),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=pos_mapper,
    )

    # Gather per-position cos/sin from host-side RoPE tables.
    cos_host = rope["cos_matrix_host"]  # [1, 1, max_seq_len, rope_dim]
    sin_host = rope["sin_matrix_host"]
    rope_dim = int(cos_host.shape[3])

    positions_cpu = positions.to(torch.long).clamp(min=0, max=int(cos_host.shape[2]) - 1)
    cos_batch_t = cos_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)
    sin_batch_t = sin_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)

    # Determine mesh mapper for cos/sin.
    if is_mesh_device and dp_shard_axis is not None:
        # Shard batch dim across DP groups so each group gets only its positions' cos/sin.
        mesh_shape = list(device.shape)
        dims = [None, None]
        dims[dp_shard_axis] = 1  # shard tensor dim=1 (batch) across dp_shard_axis
        rope_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(dims), mesh_shape=mesh_shape)
    elif is_mesh_device:
        rope_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        rope_mapper = None

    cos_batch = ttnn.from_torch(
        cos_batch_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rope_mapper,
    )
    sin_batch = ttnn.from_torch(
        sin_batch_t,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=rope_mapper,
    )

    return tt_positions, cos_batch, sin_batch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_mesh_device(device: Any) -> bool:
    return device.__class__.__name__ == "MeshDevice"


def _tt_to_torch_for_vllm_output(*, tensor: ttnn.Tensor, device: Any) -> torch.Tensor:
    """Convert a TT tensor to torch for vLLM outputs.

    On a mesh, reads back device 0 (replicated model bring-up mode).
    """
    if not _is_mesh_device(device):
        return ttnn.to_torch(tensor.cpu())
    device_tensors = ttnn.get_device_tensors(tensor)
    if not device_tensors:
        raise RuntimeError("ttnn.get_device_tensors returned an empty list for a mesh tensor")
    return ttnn.to_torch(device_tensors[0].cpu())


def _load_hparams_from_snapshot(snapshot_dir: Path) -> Glm4MoeHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


def _torch_dtype_to_ttnn(dtype: torch.dtype) -> ttnn.DataType:
    """Map vLLM torch dtype to TT KV cache dtype, with env override."""
    override = os.environ.get("GLM4_MOE_KV_CACHE_TT_DTYPE", "").strip().lower()
    if override:
        if override in {"bf8", "bfloat8_b"}:
            return ttnn.bfloat8_b
        if override in {"bf16", "bfloat16"}:
            return ttnn.bfloat16
        if override in {"f32", "fp32", "float32"}:
            return ttnn.float32
        raise ValueError(f"Invalid GLM4_MOE_KV_CACHE_TT_DTYPE={override!r}")
    return ttnn.bfloat8_b


@dataclass
class _DecodeTraceState:
    """Per-bucket state for batch-bucketed decode traces."""
    trace_id: Any | None = None
    batch: int = 0
    page_table_width: int = 0
    tokens_tt: ttnn.Tensor | None = None
    positions_tt: ttnn.Tensor | None = None
    cos_batch_tt: ttnn.Tensor | None = None
    sin_batch_tt: ttnn.Tensor | None = None
    trans_matrix_tt: ttnn.Tensor | None = None
    page_table_tt: ttnn.Tensor | None = None
    logits_tt: ttnn.Tensor | None = None
    top1_values_tt: ttnn.Tensor | None = None
    top1_indices_tt: ttnn.Tensor | None = None
    embed_tt: ttnn.Tensor | None = None


# ---------------------------------------------------------------------------
# Model Runner
# ---------------------------------------------------------------------------

@dataclass
class Glm4MoeTT:
    """TT runner for GLM-4.7-REAP-218B (92 layers, GQA attention, MoE).

    Loaded via `Glm4MoeTT.create(...)` factory.
    """

    device: Any
    snapshot_dir: Path
    cache_dir: Path
    max_seq_len: int

    hparams: Glm4MoeHParams
    state: Any  # LazyStateDict
    embed_w: Optional[ttnn.Tensor]
    embed_w_cpu: torch.Tensor
    rope: dict[str, Any]
    final_norm: RMSNorm
    lm_head_w: ttnn.Tensor
    lm_head_sharded_vocab: bool
    lm_head_tp_axis: int | None
    lm_head_tp_size: int
    lm_head_vocab_per_shard: int
    layer_weights: dict[int, DecoderLayerTTWeights]
    decoder_layers: dict[int, Glm4MoeDecoderLayer]
    num_layers_to_run: int
    enable_moe: bool
    moe_runtime: Glm4MoeMoERuntime | None
    configuration: dict[str, Any]
    tt_ccl: Any | None

    _decode_trace_states: dict[int, _DecodeTraceState] = field(init=False, default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        device: Any,
        snapshot_dir: Path,
        cache_dir: Path,
        max_seq_len: int,
        max_batch_size: int = 32,
        hparams: Optional[Glm4MoeHParams] = None,
    ) -> "Glm4MoeTT":
        snapshot_dir = Path(snapshot_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        hparams = _load_hparams_from_snapshot(snapshot_dir) if hparams is None else hparams

        state = load_glm_lazy_state_dict(snapshot_dir, num_layers=int(hparams.num_hidden_layers))

        # Embedding.
        embed_w_cpu = state["model.embed_tokens.weight"].clone().to(torch.bfloat16)
        # Skip device embedding — TG mesh uses host-side lookup (embed_w_cpu).
        # Saves ~1.49 GB DRAM per device.
        embed_w = None

        # RoPE for partial rotary (rope_dim = head_dim * partial_rotary_factor = 64).
        rope_dim = int(hparams.head_dim * hparams.partial_rotary_factor)
        rope = _make_rope_tensors(
            device=device,
            seq_len=int(max_seq_len),
            rope_dim=rope_dim,
            rope_theta=float(hparams.rope_theta),
        )

        # Final norm + LM head.
        dense_dtype = _env_dense_dtype()
        final_norm = RMSNorm(
            device=device,
            dim=int(hparams.hidden_size),
            eps=float(hparams.rms_norm_eps),
            state_dict=state,
            state_dict_prefix="model.",
            weight_key="norm",
            weight_cache_path=cache_dir,
            weight_dtype=ttnn.bfloat16,
            is_distributed=False,
        )

        # LM head sharding: shard vocab across TP devices (axis 0), replicate across DP.
        lm_head_mapper = None
        lm_head_variant = ""
        lm_head_sharded_vocab = False
        lm_head_tp_axis = None
        lm_head_tp_size = 1
        lm_head_vocab_per_shard = int(hparams.vocab_size)
        num_devices = 1
        if _is_mesh_device(device):
            num_devices = int(device.get_num_devices())

        tp_axis, tp_size_detected = _tp_axis_and_size(device)
        if tp_size_detected > 1 and num_devices > 1:
            vocab = int(hparams.vocab_size)
            if vocab % tp_size_detected == 0:
                # Shard vocab across TP axis only, replicate across DP.
                lm_head_mapper = _tp_mesh_mapper(device, shard_dim=3)
                lm_head_variant = f"tp{tp_size_detected}_shard_v1"
                lm_head_sharded_vocab = True
                lm_head_tp_axis = tp_axis
                lm_head_tp_size = tp_size_detected
                lm_head_vocab_per_shard = vocab // tp_size_detected
            else:
                logger.warning(
                    "LM head vocab {} not divisible by TP={} devices, replicating instead of sharding",
                    vocab, tp_size_detected,
                )

        lm_head_w = _linear_weight_tt(
            device=device,
            torch_weight_out_in=state["lm_head.weight"],
            cache_file=cache_dir / f"lm_head_w_{lm_head_variant}" if lm_head_variant else cache_dir / "lm_head_w",
            dtype=dense_dtype,
            mesh_mapper=lm_head_mapper,
        )

        # Layers.
        num_layers_env = os.environ.get("GLM4_MOE_NUM_LAYERS", "").strip()
        if num_layers_env and os.environ.get("GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS", "").strip() != "1":
            raise ValueError(
                "GLM4_MOE_NUM_LAYERS is debug-only. "
                "Set GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS=1 to run a partial model."
            )
        num_layers_to_run = int(num_layers_env) if num_layers_env else int(hparams.num_hidden_layers)
        num_layers_to_run = max(1, min(num_layers_to_run, int(hparams.num_hidden_layers)))

        enable_moe = os.environ.get("GLM4_MOE_ENABLE_MOE", "1").strip() != "0"
        moe_runtime = None
        if enable_moe:
            moe_runtime = create_moe_runtime(device=device, hparams=hparams)

        # Configuration dict for attention and decoder layers.
        mesh_rows = int(device.shape[0]) if _is_mesh_device(device) else 1
        mesh_cols = int(device.shape[1]) if _is_mesh_device(device) else 1
        configuration = {
            "num_devices": mesh_rows * mesh_cols,
            "max_batch_size": max_batch_size,
            "MAX_QKV_MM_SEQ_LEN": 4096,
            "ccl_dtype": ttnn.bfloat16,
        }

        # CCL for attention all-reduce/all-gather (async semaphore management).
        if _is_mesh_device(device):
            logger.info("Glm4MoeTT.create: initializing CCL semaphores for mesh {}", list(device.shape))
            tt_ccl = CCL(device)
        else:
            tt_ccl = None

        # Load decoder layer weights lazily (one at a time to save host memory).
        layer_weights_dict: dict[int, DecoderLayerTTWeights] = {}
        decoder_layers_dict: dict[int, Glm4MoeDecoderLayer] = {}

        logger.info(
            "Glm4MoeTT.create: loading {} layers (total={}), moe={}, devices={}, max_seq_len={}",
            num_layers_to_run, hparams.num_hidden_layers, enable_moe, num_devices, max_seq_len,
        )

        # DEBUG: test device sync before loading layers
        if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
            import sys
            print("  [DEBUG MODEL] synchronizing before layer loading ...", flush=True, file=sys.stderr)
            ttnn.synchronize_device(device)
            print("  [DEBUG MODEL] synchronize before layers OK", flush=True, file=sys.stderr)

        for layer_idx in range(num_layers_to_run):
            t0 = time.perf_counter()
            logger.info("  [DEBUG] Starting layer {} weight conversion", layer_idx)
            lw = convert_decoder_layer_weights(
                device=device,
                state=state,
                layer_idx=layer_idx,
                hparams=hparams,
                cache_dir=cache_dir / "layers",
                enable_moe=enable_moe and (layer_idx >= int(hparams.first_k_dense_replace)),
            )
            layer_weights_dict[layer_idx] = lw
            logger.info("  [DEBUG] Layer {} weights converted, creating decoder layer", layer_idx)

            dl = Glm4MoeDecoderLayer(
                mesh_device=device,
                tt_ccl=tt_ccl,
                hparams=hparams,
                layer_weights=lw,
                configuration=configuration,
                paged_attention_config=None,
                moe_runtime=moe_runtime,
            )
            decoder_layers_dict[layer_idx] = dl

            elapsed = time.perf_counter() - t0
            if layer_idx == 0 or (layer_idx + 1) % 10 == 0 or (layer_idx + 1) == num_layers_to_run:
                logger.info("  Layer {}/{} loaded ({:.1f}s)", layer_idx + 1, num_layers_to_run, elapsed)

            # DEBUG: sync after each layer to find which one hangs the device
            if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
                print(f"  [DEBUG MODEL] sync after layer {layer_idx} ...", flush=True, file=sys.stderr)
                ttnn.synchronize_device(device)
                print(f"  [DEBUG MODEL] sync after layer {layer_idx} OK", flush=True, file=sys.stderr)

        # DEBUG: test device sync after model init
        if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
            import sys
            print("  [DEBUG MODEL] synchronizing after all layers loaded ...", flush=True, file=sys.stderr)
            ttnn.synchronize_device(device)
            print("  [DEBUG MODEL] synchronize after layers OK", flush=True, file=sys.stderr)

        return cls(
            device=device,
            snapshot_dir=snapshot_dir,
            cache_dir=cache_dir,
            max_seq_len=int(max_seq_len),
            hparams=hparams,
            state=state,
            embed_w=embed_w,
            embed_w_cpu=embed_w_cpu,
            rope=rope,
            final_norm=final_norm,
            lm_head_w=lm_head_w,
            lm_head_sharded_vocab=lm_head_sharded_vocab,
            lm_head_tp_axis=lm_head_tp_axis,
            lm_head_tp_size=lm_head_tp_size,
            lm_head_vocab_per_shard=lm_head_vocab_per_shard,
            layer_weights=layer_weights_dict,
            decoder_layers=decoder_layers_dict,
            num_layers_to_run=num_layers_to_run,
            enable_moe=enable_moe,
            moe_runtime=moe_runtime,
            configuration=configuration,
            tt_ccl=tt_ccl,
        )

    # -------------------------------------------------------------------
    # Decode
    # -------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
        enable_trace: bool = False,
    ) -> Any:
        """Run a single-token decode step through all layers.

        Args:
            tokens: [B, 1] int32
            start_pos: [B] int32 (padded with -1 for inactive slots)
            page_table: [B, W] int32
            kv_cache: list of [cache_k, cache_v] per layer
            sampling_params: if not None, do on-device greedy sampling
            enable_trace: use trace capture/replay for decode

        Returns:
            logits [active, 1, vocab] as torch float32 when sampling_params is None,
            or next token ids [active] as torch int32 when sampling_params is not None.
        """
        if tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError(f"expected tokens [B,1], got {tuple(tokens.shape)}")
        if start_pos.ndim != 1:
            raise ValueError(f"expected start_pos [B], got {tuple(start_pos.shape)}")

        start_pos = start_pos.to(torch.int32)
        active = int((start_pos >= 0).sum().item())
        if active <= 0:
            return torch.zeros((0, 1, int(self.hparams.vocab_size)), dtype=torch.float32)

        if enable_trace:
            return self._decode_trace(
                tokens=tokens[:active].to(torch.int32),
                positions=start_pos[:active].to(torch.int32),
                page_table=page_table[:active].to(torch.int32),
                kv_cache=kv_cache,
                sampling_params=sampling_params,
            )

        return self._decode_eager(
            tokens=tokens[:active].to(torch.int32),
            positions=start_pos[:active].to(torch.int32),
            page_table=page_table[:active].to(torch.int32),
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    def _decode_eager(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
    ) -> Any:
        """Eager (non-traced) decode step."""
        active = int(tokens.shape[0])
        hidden = int(self.hparams.hidden_size)
        is_mesh = _is_mesh_device(self.device)

        # Prepare inputs.
        tokens = tokens.contiguous().clone()
        positions = positions.contiguous().clone()
        page_table = page_table.contiguous().clone()

        # On TG (2D mesh) with multi-user batch, shard batch-dimension inputs
        # across DP groups (page_table, positions, cos/sin).
        dp_shard_axis = None
        dp_batch_mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if active > 1 and active % dp_size == 0:
                dp_shard_axis = 1
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=(None, 0), mesh_shape=mesh_shape,
                )

        page_table_tt = ttnn.from_torch(
            page_table,
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=dp_batch_mapper,
        )

        tt_positions, cos_batch, sin_batch = _prepare_decode_rope_and_positions_tt(
            device=self.device, rope=self.rope, positions=positions,
            dp_shard_axis=dp_shard_axis,
        )

        # Embedding: host-side lookup to avoid device-side tile conversion hang on TG mesh.
        embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
        x = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
        )

        # RoPE mats for attention (cos, sin, trans_matrix).
        rot_mats = (cos_batch, sin_batch, self.rope["trans_matrix"])

        # Decoder stack.
        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(
                x, tt_positions, rot_mats, page_table_tt, kv_cache[layer_idx], mode="decode",
                active_batch=active,
            )
            ttnn.deallocate(x, force=False)
            x = x_next

        # Final norm + LM head (sharded norm to avoid L1 overflow with hidden=5120).
        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size))
        logits_tt = ttnn.linear(x, self.lm_head_w)  # [1, 1, B, vocab]

        if sampling_params is not None:
            return self._sample_greedy(logits_tt, active, x, tt_positions, cos_batch, sin_batch, page_table_tt)

        # Return full logits on host.
        vocab = int(self.hparams.vocab_size)
        result = self._logits_to_host(logits_tt, active, vocab)

        ttnn.deallocate(logits_tt, force=False)
        ttnn.deallocate(x, force=False)
        ttnn.deallocate(tt_positions, force=False)
        ttnn.deallocate(cos_batch, force=False)
        ttnn.deallocate(sin_batch, force=False)
        ttnn.deallocate(page_table_tt, force=False)

        # Reset CCL semaphore counters after eager decode.
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()

        return result

    def _sample_greedy(
        self,
        logits_tt: ttnn.Tensor,
        active: int,
        x: ttnn.Tensor,
        tt_positions: ttnn.Tensor,
        cos_batch: ttnn.Tensor,
        sin_batch: ttnn.Tensor,
        page_table_tt: ttnn.Tensor,
    ) -> torch.Tensor:
        """On-device greedy sampling (argmax) from logits."""
        vocab = int(self.hparams.vocab_size)

        if not (self.lm_head_sharded_vocab and _is_mesh_device(self.device)):
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, next_ids_tt = max_out
                ttnn.deallocate(local_max_tt, force=False)
            else:
                next_ids_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
            ttnn.deallocate(logits_rm_tight, force=False)

            next_ids_torch = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
            next_ids_flat = next_ids_torch.reshape(-1).to(dtype=torch.int32).cpu()
            ttnn.deallocate(logits_rm, force=False)
            ttnn.deallocate(logits_tt, force=False)
            ttnn.deallocate(next_ids_tt, force=False)
        else:
            # Vocab-sharded: per-device max+argmax, reduce on host.
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            vocab_per_shard = int(self.lm_head_vocab_per_shard)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab_per_shard])
            max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, local_argmax_tt = max_out
            else:
                local_max_tt = max_out
                local_argmax_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)

            num_devices = int(self.device.get_num_devices())
            local_max_dts = ttnn.get_device_tensors(local_max_tt)
            local_idx_dts = ttnn.get_device_tensors(local_argmax_tt)

            next_ids = torch.empty((active,), dtype=torch.int32)
            for b in range(active):
                best_val = None
                best_global = None
                for shard_idx in range(num_devices):
                    max_val = float(ttnn.to_torch(local_max_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    local_idx = int(ttnn.to_torch(local_idx_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    global_idx = shard_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global
            next_ids_flat = next_ids

            ttnn.deallocate(local_max_tt, force=False)
            ttnn.deallocate(local_argmax_tt, force=False)
            ttnn.deallocate(logits_rm, force=False)
            ttnn.deallocate(logits_tt, force=False)

        ttnn.deallocate(x, force=False)
        ttnn.deallocate(tt_positions, force=False)
        ttnn.deallocate(cos_batch, force=False)
        ttnn.deallocate(sin_batch, force=False)
        ttnn.deallocate(page_table_tt, force=False)

        # Reset CCL semaphore counters after eager decode (sampling path).
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()

        return next_ids_flat

    def _sample_from_trace_logits(self, logits_tt: ttnn.Tensor, active: int) -> torch.Tensor:
        """Device-side greedy sampling from trace-owned logits (OUTSIDE trace).

        Called after execute_trace + synchronize_device. The logits_tt tensor
        is trace-owned and must NOT be deallocated.

        For TP-sharded vocab: per-shard max+argmax on device, then pick global
        best on host. Transfers only ~64 bytes instead of 9.6 MB.
        """
        vocab = int(self.hparams.vocab_size)

        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            # Vocab-sharded: per-device max+argmax, reduce on host.
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            vocab_per_shard = int(self.lm_head_vocab_per_shard)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab_per_shard])
            max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                local_max_tt, local_argmax_tt = max_out
            else:
                local_max_tt = max_out
                local_argmax_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)

            num_devices = int(self.device.get_num_devices())
            tp_size = self.lm_head_tp_size
            local_max_dts = ttnn.get_device_tensors(local_max_tt)
            local_idx_dts = ttnn.get_device_tensors(local_argmax_tt)

            # Skip DP duplicates: pick one shard per TP position
            dp_stride = num_devices // tp_size if num_devices > tp_size else 1

            next_ids = torch.empty((active,), dtype=torch.int32)
            for b in range(active):
                best_val = None
                best_global = None
                for tp_idx in range(tp_size):
                    shard_idx = tp_idx * dp_stride
                    max_val = float(ttnn.to_torch(local_max_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    local_idx = int(ttnn.to_torch(local_idx_dts[shard_idx].cpu()).reshape(-1)[b].item())
                    global_idx = tp_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global

            # Clean up temporaries (NOT logits_tt — trace-owned)
            ttnn.deallocate(local_max_tt, force=True)
            ttnn.deallocate(local_argmax_tt, force=True)
            ttnn.deallocate(logits_rm_view, force=False)
            ttnn.deallocate(logits_rm, force=True)
        else:
            # Non-sharded: single argmax
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                _, next_ids_tt = max_out
            else:
                next_ids_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)

            next_ids = _tt_to_torch_for_vllm_output(tensor=next_ids_tt, device=self.device)
            next_ids = next_ids.reshape(-1).to(dtype=torch.int32).cpu()

            ttnn.deallocate(next_ids_tt, force=True)
            ttnn.deallocate(logits_rm_tight, force=True)
            ttnn.deallocate(logits_rm_view, force=False)
            ttnn.deallocate(logits_rm, force=True)

        return next_ids

    def _host_argmax_from_trace_logits(
        self, logits_tt: ttnn.Tensor, active: int, vocab: int,
    ) -> torch.Tensor:
        """Host-side argmax from trace-owned logits, optimized for minimal work.

        Instead of concatenating all TP shards into full vocab then running argmax,
        does per-shard argmax on host and picks global best. Avoids the full
        torch.cat allocation.

        Does NOT deallocate logits_tt (trace-owned).
        """
        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            shards = ttnn.get_device_tensors(logits_tt)
            num_shards = len(shards)
            tp_size = self.lm_head_tp_size
            vocab_per_shard = int(self.lm_head_vocab_per_shard)

            # Select one shard per TP position (skip DP duplicates).
            dp_stride = num_shards // tp_size if num_shards > tp_size else 1

            next_ids = torch.empty((active,), dtype=torch.int32)

            # Transfer shards and do per-shard argmax on host
            shard_maxvals = []
            shard_argmaxes = []
            for tp_idx in range(tp_size):
                shard_idx = tp_idx * dp_stride
                shard_torch = ttnn.to_torch(shards[shard_idx].cpu())
                # Slice to valid batch and vocab range
                shard_torch = shard_torch[..., :active, :vocab_per_shard]
                shard_flat = shard_torch.reshape(active, -1).to(torch.float32)
                shard_maxvals.append(shard_flat.max(dim=-1))
                shard_argmaxes.append(shard_flat.argmax(dim=-1))

            # Pick global best across shards
            for b in range(active):
                best_val = None
                best_global = None
                for tp_idx in range(tp_size):
                    max_val = float(shard_maxvals[tp_idx].values[b].item())
                    local_idx = int(shard_argmaxes[tp_idx][b].item())
                    global_idx = tp_idx * vocab_per_shard + local_idx
                    if global_idx >= vocab:
                        continue
                    if best_val is None or max_val > best_val:
                        best_val = max_val
                        best_global = global_idx
                if best_global is None:
                    best_global = max(0, vocab - 1)
                next_ids[b] = best_global

            return next_ids
        else:
            # Non-sharded: simple host transfer + argmax
            logits_host = self._logits_to_host(logits_tt, active, vocab)
            return logits_host.reshape(active, -1)[:, :vocab].argmax(dim=-1).to(torch.int32)

    def _logits_to_host(
        self,
        logits_tt: ttnn.Tensor,
        active: int,
        vocab: int,
    ) -> torch.Tensor:
        """Convert logits TT tensor to host torch [active, 1, vocab]."""
        if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
            shards = ttnn.get_device_tensors(logits_tt)
            num_shards = len(shards)
            tp_size = self.lm_head_tp_size
            # Select one shard per TP position (skip DP duplicates).
            if num_shards == tp_size:
                # Already one per TP device (ShardTensor2dMesh collapsed DP).
                tp_shards = list(shards)
            elif num_shards > tp_size:
                # All mesh devices returned; pick every (num_shards/tp_size)-th shard.
                dp_stride = num_shards // tp_size
                tp_shards = [shards[i * dp_stride] for i in range(tp_size)]
            else:
                tp_shards = list(shards)
            logits_shards = [ttnn.to_torch(t.cpu())[..., :int(t.shape[-1])] for t in tp_shards]
            logits_full = torch.cat(logits_shards, dim=-1)[..., :vocab]
            # Slice off TILE_SIZE padding — decode pads batch to 32 but we only need `active`.
            logits_full = logits_full[..., :active, :]
            logits_flat = logits_full.reshape(-1, vocab)
        else:
            logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
            logits_torch = logits_torch[..., :vocab]
            # Slice off TILE_SIZE padding — decode pads batch to 32 but we only need `active`.
            logits_torch = logits_torch[..., :active, :]
            logits_flat = logits_torch.reshape(-1, vocab)

        if logits_flat.shape[0] != active:
            raise RuntimeError(
                f"decode logits shape mismatch: expected {active} rows, got {int(logits_flat.shape[0])}"
            )

        return logits_flat.reshape(active, 1, vocab).to(dtype=torch.float32).cpu()

    # -------------------------------------------------------------------
    # Decode Trace (capture/replay)
    # -------------------------------------------------------------------

    def _decode_trace(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None = None,
    ) -> Any:
        """Decode using trace capture/replay for performance.

        On first call with a given batch size, captures a trace.
        On subsequent calls, replays the trace with updated inputs.
        """
        active = int(tokens.shape[0])
        page_table_w = int(page_table.shape[1])
        bucket = active  # Use batch size as the trace bucket key.

        state = self._decode_trace_states.get(bucket)

        if state is None or state.trace_id is None:
            # Capture a new trace for this batch bucket.
            try:
                state = self._capture_decode_trace(
                    tokens=tokens,
                    positions=positions,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    sampling_params=sampling_params,
                    active=active,
                )
                self._decode_trace_states[bucket] = state
            except RuntimeError as e:
                if "trace_region_size" in str(e) or "trace buffers" in str(e).lower():
                    logger.warning(
                        "Trace capture failed for batch={} (trace region too small). "
                        "Falling back to eager decode for this bucket.", active,
                    )
                    return self._decode_eager(
                        tokens=tokens,
                        positions=positions,
                        page_table=page_table,
                        kv_cache=kv_cache,
                        sampling_params=sampling_params,
                    )
                raise
        else:
            # Update persistent inputs and replay.
            self._update_trace_inputs(state, tokens, positions, page_table)
            if self.tt_ccl is not None:
                self.tt_ccl.reset_sem_counters()
            ttnn.synchronize_device(self.device)
            ttnn.execute_trace(self.device, state.trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.device)

        # Read outputs.
        if sampling_params is not None and state.top1_indices_tt is not None:
            next_ids_torch = _tt_to_torch_for_vllm_output(
                tensor=state.top1_indices_tt, device=self.device
            ).reshape(-1).to(dtype=torch.int32).cpu()
            return next_ids_torch

        if state.logits_tt is not None:
            if sampling_params is not None:
                if _is_mesh_device(self.device):
                    # TG mesh: device-side sampling ops hang. Use optimized host transfer.
                    vocab = int(self.hparams.vocab_size)
                    return self._host_argmax_from_trace_logits(state.logits_tt, active, vocab)
                else:
                    # Non-TG: device-side sampling works fine outside trace.
                    return self._sample_from_trace_logits(state.logits_tt, active)
            # Non-sampling path: still need full logits on host
            vocab = int(self.hparams.vocab_size)
            logits_host = self._logits_to_host(state.logits_tt, active, vocab)
            return logits_host

        return torch.zeros((active, 1, int(self.hparams.vocab_size)), dtype=torch.float32)

    def _capture_decode_trace(
        self,
        *,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list,
        sampling_params: Any | None,
        active: int,
    ) -> _DecodeTraceState:
        """Capture a decode trace for the given batch size."""
        hidden = int(self.hparams.hidden_size)
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None

        # On TG (2D mesh) with multi-user batch, shard batch-dimension inputs (cos/sin,
        # page_table, positions) across DP groups so each group gets only its subset.
        # Required because attention slices QKV per DP group: Q/K have
        # logical batch = batch_per_group (not full batch).
        dp_shard_axis = None
        dp_batch_mapper = mapper  # replicate by default
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if active > 1 and active % dp_size == 0:
                dp_shard_axis = 1
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=(None, 0), mesh_shape=mesh_shape,
                )

        # Create persistent input tensors (BEFORE trace capture AND warm-up).
        page_table_tt = ttnn.from_torch(
            page_table.contiguous().clone(),
            device=self.device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=dp_batch_mapper,
        )

        tt_positions, cos_batch, sin_batch = _prepare_decode_rope_and_positions_tt(
            device=self.device, rope=self.rope, positions=positions,
            dp_shard_axis=dp_shard_axis,
        )

        # Pre-allocate persistent token buffer on device (outside trace).
        tokens_tt = ttnn.from_torch(
            tokens.contiguous().clone().to(torch.int32),
            device=self.device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

        # Pre-allocate embedding tensor BEFORE compile warm-up!
        # This ensures the memory allocator state exactly matches during trace capture.
        embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
        embed_tt = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
        )

        rot_mats = (cos_batch, sin_batch, self.rope["trans_matrix"])

        # Run forward once (compile warm-up) using the persistent tensors.
        x = embed_tt
        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(x, tt_positions, rot_mats, page_table_tt, kv_cache[layer_idx], mode="decode",
                                active_batch=active)
            # Match trace capture exactly: skip deallocation of embed_tt!
            if layer_idx > 0:
                ttnn.deallocate(x, force=False)
            x = x_next

        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size))
        logits_tt = ttnn.linear(x, self.lm_head_w)

        # On TG mesh, skip sampling in compile warm-up (matches trace capture pattern).
        # On non-TG, include sampling to compile those programs too.
        top1_values_tt = None
        top1_indices_tt = None
        if not is_mesh and sampling_params is not None:
            vocab = int(self.hparams.vocab_size)
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                top1_values_tt, top1_indices_tt = max_out
            else:
                top1_values_tt = max_out
                top1_indices_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
            ttnn.deallocate(logits_rm_tight, force=False)
            ttnn.deallocate(logits_rm, force=False)

        # Synchronize device to drain all async ops from compile-forward
        # before starting trace capture.
        ttnn.synchronize_device(self.device)

        # Explicitly free compile-forward outputs.
        ttnn.deallocate(x, force=True)
        ttnn.deallocate(logits_tt, force=True)
        x = logits_tt = top1_values_tt = top1_indices_tt = None
        ttnn.synchronize_device(self.device)

        # Re-copy persistent inputs before trace capture (like GLM4-Flash)
        # to ensure no inadvertent in-place mutation broke them.
        host_embed = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_embed, embed_tt)

        # Now capture trace.
        logger.info("Capturing decode trace for batch={}", active)
        if self.tt_ccl is not None:
            self.tt_ccl.reset_sem_counters()
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)

        # Trace reads from embed_tt; we must NOT deallocate it after first layer
        # so it survives for copy_host_to_device_tensor during replay.
        x = embed_tt

        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next = dl.forward(x, tt_positions, rot_mats, page_table_tt, kv_cache[layer_idx], mode="decode",
                                active_batch=active)
            # Skip deallocate on first iteration: x IS embed_tt which must survive
            # for copy_host_to_device_tensor during trace replay.
            if layer_idx > 0:
                ttnn.deallocate(x, force=False)
            x = x_next

        x = _sharded_rms_norm(x, self.final_norm, int(self.hparams.hidden_size))
        logits_tt = ttnn.linear(x, self.lm_head_w)

        # On TG mesh, sampling ops (to_layout, slice, max, argmax) inside trace
        # produce wrong results. Do sampling on host instead (read logits from trace output).
        if not _is_mesh_device(self.device) and sampling_params is not None:
            vocab = int(self.hparams.vocab_size)
            logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
            logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, active, vocab])
            logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
            if isinstance(max_out, tuple):
                top1_values_tt, top1_indices_tt = max_out
            else:
                top1_values_tt = max_out
                top1_indices_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
            ttnn.deallocate(logits_rm_tight, force=False)
            ttnn.deallocate(logits_rm, force=False)

        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        logger.info("Decode trace captured for batch={}", active)

        # TG mesh: logits_tt for host-side sampling; non-TG: use on-device sampling results
        use_logits_output = _is_mesh_device(self.device)
        return _DecodeTraceState(
            trace_id=trace_id,
            batch=active,
            page_table_width=int(page_table.shape[1]),
            tokens_tt=tokens_tt,
            positions_tt=tt_positions,
            cos_batch_tt=cos_batch,
            sin_batch_tt=sin_batch,
            trans_matrix_tt=self.rope["trans_matrix"],
            page_table_tt=page_table_tt,
            logits_tt=logits_tt if use_logits_output else (logits_tt if sampling_params is None else None),
            top1_values_tt=top1_values_tt if not use_logits_output else None,
            top1_indices_tt=top1_indices_tt if not use_logits_output else None,
            embed_tt=embed_tt,
        )

    def _update_trace_inputs(
        self,
        state: _DecodeTraceState,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        page_table: torch.Tensor,
    ) -> None:
        """Update persistent trace input tensors with new values."""
        is_mesh = _is_mesh_device(self.device)
        mapper = ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None
        batch = int(tokens.shape[0])

        # On TG with multi-user batch, batch-dimension inputs (positions, cos/sin,
        # page_table) were sharded across DP groups during trace capture.
        # Must use the same sharding here for copy_host_to_device_tensor.
        dp_batch_mapper = mapper  # for page_table (dim=0), positions (dim=0)
        rope_mapper = mapper      # for cos/sin (dim=1)
        if is_mesh and int(self.device.shape[1]) > 1:
            dp_size = int(self.device.shape[1])
            if batch > 1 and batch % dp_size == 0:
                mesh_shape = list(self.device.shape)
                dp_batch_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=(None, 0), mesh_shape=mesh_shape,
                )
                rope_mapper = ttnn.ShardTensor2dMesh(
                    self.device, dims=(None, 1), mesh_shape=mesh_shape,
                )

        # Update embedding (host lookup + copy to pre-allocated device buffer).
        embed_torch = self.embed_w_cpu[tokens[:, 0].long()]  # [B, hidden]
        host_embed = ttnn.from_torch(
            embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, B, hidden]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_embed, state.embed_tt)


        # Update positions (host tensor, then copy).
        host_positions = ttnn.from_torch(
            positions.view(-1).contiguous().to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=dp_batch_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_positions, state.positions_tt)

        # Update cos/sin for new positions.
        cos_host = self.rope["cos_matrix_host"]
        sin_host = self.rope["sin_matrix_host"]
        rope_dim = int(cos_host.shape[3])
        positions_cpu = positions.to(torch.long).clamp(min=0, max=int(cos_host.shape[2]) - 1)
        cos_batch_t = cos_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)
        sin_batch_t = sin_host[0, 0, positions_cpu, :].reshape(1, batch, 1, rope_dim).to(torch.bfloat16)

        host_cos = ttnn.from_torch(
            cos_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rope_mapper,
        )
        host_sin = ttnn.from_torch(
            sin_batch_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rope_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_cos, state.cos_batch_tt)
        ttnn.copy_host_to_device_tensor(host_sin, state.sin_batch_tt)

        # Update page table (host tensor, then copy).
        host_pt = ttnn.from_torch(
            page_table.contiguous().clone(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=dp_batch_mapper,
        )
        ttnn.copy_host_to_device_tensor(host_pt, state.page_table_tt)

    def _release_all_decode_traces(self) -> None:
        """Release ALL bucket decode traces and deallocate trace output tensors."""
        if not self._decode_trace_states:
            return
        ttnn.synchronize_device(self.device)
        for bucket, state in list(self._decode_trace_states.items()):
            if state.trace_id is not None:
                try:
                    ttnn.release_trace(self.device, state.trace_id)
                except Exception:
                    pass
                state.trace_id = None
            for t in (
                state.logits_tt, state.top1_values_tt, state.top1_indices_tt,
                state.tokens_tt, state.positions_tt, state.cos_batch_tt,
                state.sin_batch_tt, state.trans_matrix_tt, state.page_table_tt
            ):
                if t is not None:
                    try:
                        ttnn.deallocate(t, force=True)
                    except Exception:
                        pass
        self._decode_trace_states.clear()
        ttnn.synchronize_device(self.device)

    # -------------------------------------------------------------------
    # Prefill
    # -------------------------------------------------------------------

    @torch.no_grad()
    def prefill(
        self,
        *,
        tokens: torch.Tensor,
        prompt_lens: list[int],
        page_table: torch.Tensor,
        kv_cache: list,
        seq_pad_multiple: int = 128,
    ) -> torch.Tensor:
        """Compute logits for the last prompt token for each request and fill KV caches.

        Processes each request independently (B loop). For long prompts (>16K tokens),
        uses chunked prefill to avoid activation memory OOM.

        Args:
            tokens: [B, S] int32
            prompt_lens: length B, actual prompt lengths
            page_table: [B, W] int32
            kv_cache: list of [cache_k, cache_v] per layer
            seq_pad_multiple: pad prompt lengths to this multiple

        Returns:
            Logits [B, 1, vocab] as torch float32
        """
        # Release decode traces before prefill to avoid buffer corruption.
        # Prefill allocates device buffers that can overlap with trace-owned buffers,
        # producing garbled output on subsequent trace replays.
        # With synchronize_device before trace capture (line ~931), re-capture is safe.
        self._release_all_decode_traces()  # Required: prefill buffers overlap with trace-owned buffers

        if tokens.ndim != 2:
            raise ValueError(f"expected tokens [B,S], got {tuple(tokens.shape)}")
        if page_table.ndim != 2:
            raise ValueError(f"expected page_table [B,W], got {tuple(page_table.shape)}")
        batch, seq_total = tokens.shape
        if len(prompt_lens) != int(batch):
            raise ValueError(f"prompt_lens length {len(prompt_lens)} != batch {batch}")

        hidden = int(self.hparams.hidden_size)
        vocab = int(self.hparams.vocab_size)
        rope_dim = int(self.hparams.head_dim * self.hparams.partial_rotary_factor)
        pad_multiple = max(128, int(seq_pad_multiple))
        is_mesh = _is_mesh_device(self.device)

        # Prefill chunk size (16K tokens, matching DSv3 pattern for activation memory).
        PREFILL_CHUNK_SIZE = int(os.environ.get("GLM4_MOE_PREFILL_CHUNK_SIZE", "16384"))

        out_logits: list[torch.Tensor] = []

        for i in range(int(batch)):
            prompt_len = int(prompt_lens[i])
            if prompt_len <= 0:
                out_logits.append(torch.zeros((1, vocab), dtype=torch.float32))
                continue

            padded_len = ((prompt_len + pad_multiple - 1) // pad_multiple) * pad_multiple
            padded_len = min(padded_len, int(self.max_seq_len))

            prompt_ids = tokens[i, :prompt_len].to(torch.int32).cpu()
            input_padded = torch.zeros((1, padded_len), dtype=torch.int32)
            input_padded[0, :prompt_len] = prompt_ids

            # Page table for this request.
            page_row = page_table[i : i + 1, :].to(torch.int32)
            page_table_tt = ttnn.from_torch(
                page_row,
                device=self.device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )

            # Slice RoPE tables.
            rope_slices_owned = True
            if (
                int(padded_len) == int(self.rope["cos_matrix"].shape[2])
                and int(rope_dim) == int(self.rope["cos_matrix"].shape[3])
            ):
                cos_matrix = self.rope["cos_matrix"]
                sin_matrix = self.rope["sin_matrix"]
                rope_slices_owned = False
            else:
                cos_matrix = ttnn.slice(
                    self.rope["cos_matrix"], [0, 0, 0, 0], [1, 1, padded_len, rope_dim]
                )
                sin_matrix = ttnn.slice(
                    self.rope["sin_matrix"], [0, 0, 0, 0], [1, 1, padded_len, rope_dim]
                )

            # DEBUG: sync checkpoints for prefill pipeline
            _dbg = os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0"
            def _psync(label):
                if _dbg:
                    import sys
                    print(f"  [DEBUG PREFILL-MODEL] {label} ...", flush=True, file=sys.stderr)
                    ttnn.synchronize_device(self.device)
                    print(f"  [DEBUG PREFILL-MODEL] {label} OK", flush=True, file=sys.stderr)

            _psync("after page_table + rope_slice")

            # Embedding: do host-side lookup to avoid device-side tile conversion hang on TG.
            # ttnn.embedding with TILE_LAYOUT and ttnn.to_layout both hang on 32-device TG mesh.
            embed_torch = self.embed_w_cpu[input_padded[0].long()]  # [padded_len, hidden]
            x = ttnn.from_torch(
                embed_torch.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),  # [1, 1, padded_len, hidden]
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if is_mesh else None,
            )
            _psync("after embedding (host)")

            rot_mats = (cos_matrix, sin_matrix, self.rope["trans_matrix"])

            _psync("before layer loop")

            # Chunked prefill: process PREFILL_CHUNK_SIZE tokens at a time.
            if padded_len > PREFILL_CHUNK_SIZE:
                x = self._prefill_chunked(
                    x=x, padded_len=padded_len, prompt_len=prompt_len,
                    page_table_tt=page_table_tt, kv_cache=kv_cache,
                    rot_mats=rot_mats, user_id=i,
                    chunk_size=PREFILL_CHUNK_SIZE,
                )
            else:
                # Single-pass prefill.
                for layer_idx in range(self.num_layers_to_run):
                    dl = self.decoder_layers[layer_idx]
                    x_next = dl.forward(
                        x, None, rot_mats, page_table_tt, kv_cache[layer_idx], mode="prefill",
                    )
                    ttnn.deallocate(x, force=False)
                    x = x_next

            x_last = ttnn.slice(x, [0, 0, prompt_len - 1, 0], [1, 1, prompt_len, hidden])
            ttnn.deallocate(x, force=False)

            x_last = _sharded_rms_norm(x_last, self.final_norm, int(self.hparams.hidden_size))
            logits_tt = ttnn.linear(x_last, self.lm_head_w)

            # Assemble vocab-sharded logits on host (NOT all_gather — all_gather
            # corrupts shard ordering on 2D mesh, inflating Chinese token scores).
            if self.lm_head_sharded_vocab and _is_mesh_device(self.device):
                shards = ttnn.get_device_tensors(logits_tt)
                num_shards = len(shards)
                tp_size = self.lm_head_tp_size
                if num_shards == tp_size:
                    tp_shards = list(shards)
                elif num_shards > tp_size:
                    dp_stride = num_shards // tp_size
                    tp_shards = [shards[i * dp_stride] for i in range(tp_size)]
                else:
                    tp_shards = list(shards)
                logits_shards = [ttnn.to_torch(t.cpu())[..., :int(t.shape[-1])] for t in tp_shards]
                logits_torch = torch.cat(logits_shards, dim=-1)[..., :vocab]
                logits_flat = logits_torch.reshape(-1, vocab)
            else:
                logits_torch = _tt_to_torch_for_vllm_output(tensor=logits_tt, device=self.device)
                logits_torch = logits_torch[..., :vocab]
                logits_flat = logits_torch.reshape(-1, vocab)
            logits_i = logits_flat.to(dtype=torch.float32).cpu()
            out_logits.append(logits_i)

            ttnn.deallocate(logits_tt, force=False)
            ttnn.deallocate(x_last, force=False)
            if rope_slices_owned:
                ttnn.deallocate(cos_matrix, force=False)
                ttnn.deallocate(sin_matrix, force=False)
            ttnn.deallocate(page_table_tt, force=False)

            # Reset CCL semaphore counters after each prefill to keep state consistent.
            if self.tt_ccl is not None:
                self.tt_ccl.reset_sem_counters()

        return torch.stack(out_logits, dim=0)  # [B, 1, vocab]

    def _prefill_chunked(
        self,
        *,
        x: ttnn.Tensor,
        padded_len: int,
        prompt_len: int,
        page_table_tt: ttnn.Tensor,
        kv_cache: list,
        rot_mats: tuple,
        user_id: int,
        chunk_size: int,
    ) -> ttnn.Tensor:
        """Run prefill in chunks for long sequences to avoid activation OOM.

        Processes chunk_size tokens at a time through all layers, writing KV cache
        incrementally. Returns the full hidden state [1,1,padded_len,hidden].
        """
        hidden = int(self.hparams.hidden_size)
        num_chunks = (padded_len + chunk_size - 1) // chunk_size

        for layer_idx in range(self.num_layers_to_run):
            dl = self.decoder_layers[layer_idx]
            x_next_chunks = []

            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, padded_len)

                x_chunk = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, hidden])
                x_chunk_out = dl.forward(
                    x_chunk, None, rot_mats, page_table_tt, kv_cache[layer_idx], mode="prefill",
                )
                x_next_chunks.append(x_chunk_out)
                ttnn.deallocate(x_chunk, force=False)

            ttnn.deallocate(x, force=False)
            if len(x_next_chunks) == 1:
                x = x_next_chunks[0]
            else:
                x = ttnn.concat(x_next_chunks, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for chunk_t in x_next_chunks:
                    ttnn.deallocate(chunk_t, force=False)

        return x
