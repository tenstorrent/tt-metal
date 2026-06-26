# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ttnn helpers shared by the DeepSeek-V3.2 TT layers (MLA + Indexer):

  * RoPE matrices — cos/sin/transformation tensors for
    ``ttnn.experimental.rotary_embedding_llama``, derived from the *reference*
    ``precompute_freqs_cis`` so the YaRN scaling matches the CPU reference bit
    for bit (modulo bf16).  See ``rope_matrices`` for the interleaved-rope proof.
  * Weight conversion — turn a CPU-module ``state_dict`` (torch tensors) into
    cached ttnn tensors (``ttnn.as_tensor`` with ``cache_file_name``), single
    device (replicated; ``world_size == 1``).

Everything here is single-device: weights and rope tensors are *replicated*
across the (1, 1) mesh.  Head/TP sharding is intentionally not implemented (see
spec §3); the head dimension is kept explicit so it can be sharded later.
"""

from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis


def default_compute_kernel_config(mesh_device: ttnn.MeshDevice):
    """HiFi4 + fp32 accumulation — the functional-accuracy config used everywhere."""
    return ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def replicate_to_device(t: torch.Tensor, mesh_device: ttnn.MeshDevice, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    """Upload a host tensor to the (single-device) mesh, replicated, bf16."""
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=layout,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _meta_style_cos_sin(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert the reference complex ``freqs_cis`` ``[seq, rope/2]`` into Meta-style
    cos/sin tables ``[1, 1, seq, rope]`` with each angle duplicated per
    interleaved pair: ``[c0, c0, c1, c1, ...]``.

    With this layout, ``rotary_embedding_llama`` (which computes
    ``x * cos + rotate(x) * sin`` where ``rotate`` is the per-pair (-x1, x0)
    swap encoded by ``rope_transformation_matrix``) reproduces the reference
    ``apply_rotary_emb(interleaved=True)`` exactly:
        y[2k]   = x[2k]*cos_k - x[2k+1]*sin_k
        y[2k+1] = x[2k+1]*cos_k + x[2k]*sin_k
    """
    parts = torch.view_as_real(freqs_cis)  # [seq, rope/2, 2] -> real=cos, imag=sin
    cos_half, sin_half = parts[..., 0], parts[..., 1]
    cos = torch.stack((cos_half, cos_half), dim=-1).flatten(-2)  # [seq, rope]
    sin = torch.stack((sin_half, sin_half), dim=-1).flatten(-2)
    return cos[None, None], sin[None, None]  # [1, 1, seq, rope]


def rope_transformation_matrix() -> torch.Tensor:
    """Single-tile (-x1, x0) pair-swap matrix used by ``rotary_embedding_llama``."""
    n = ttnn.TILE_SIZE
    m = torch.zeros(1, 1, n, n)
    m[..., torch.arange(0, n, 2), torch.arange(1, n, 2)] = 1
    m[..., torch.arange(1, n, 2), torch.arange(0, n, 2)] = -1
    return m


class RopeTables:
    """
    Host-side YaRN-scaled cos/sin tables (full ``max_seq_len``) plus the device
    transformation matrix.  ``rope_tensors(start, seqlen)`` slices the tables to
    the absolute position range and uploads the per-call ttnn tensors for
    ``rotary_embedding_llama`` (prefill mode).
    """

    def __init__(self, args, mesh_device: ttnn.MeshDevice):
        self.mesh_device = mesh_device
        cos, sin = _meta_style_cos_sin(precompute_freqs_cis(args))
        self._cos, self._sin = cos, sin  # [1, 1, max_seq, rope]
        self._trans = replicate_to_device(rope_transformation_matrix(), mesh_device)

    def rope_tensors(self, start_pos: int, seqlen: int) -> dict:
        end = start_pos + seqlen
        return {
            "cos_matrix": replicate_to_device(self._cos[:, :, start_pos:end, :], self.mesh_device),
            "sin_matrix": replicate_to_device(self._sin[:, :, start_pos:end, :], self.mesh_device),
            "trans_matrix": self._trans,
        }


def apply_noninterleaved_rope_host(x: ttnn.Tensor, freqs_cis: torch.Tensor, start_pos: int, mesh_device) -> ttnn.Tensor:
    """
    Non-interleaved RoPE for the Indexer (CPU FALLBACK, spec §6/§10).

    ``rotary_embedding_llama`` implements the interleaved convention only; the
    Indexer uses the half-split (non-interleaved) convention, so we apply the
    reference ``apply_rotary_emb(interleaved=False)`` on the host and re-upload.
    Numerically exact (bf16 round-trip is lossless); perf is out of scope (§9).
    """
    from models.demos.deepseek_v32.reference_cpu.utils import apply_rotary_emb

    t = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    b, h, s, d = t.shape
    t = t[:b].permute(0, 2, 1, 3)  # [B, S, H, d] for apply_rotary_emb
    t = apply_rotary_emb(t, freqs_cis[start_pos : start_pos + s], interleaved=False)
    t = t.permute(0, 2, 1, 3).contiguous()  # back to [B, H, S, d]
    return replicate_to_device(t, mesh_device)


def apply_interleaved_rope(x: ttnn.Tensor, rope: dict) -> ttnn.Tensor:
    """
    Interleaved RoPE on ``x`` of shape ``[B, n_heads, seqlen, rope_dim]`` via
    ``rotary_embedding_llama``.  Batch is folded into the head axis so a single
    ``[1, 1, seqlen, rope_dim]`` cos/sin table broadcasts across all batches
    (rope depends only on position, identical across the batch).
    """
    b, h, s, d = x.shape
    x = ttnn.reshape(x, (1, b * h, s, d))
    x = ttnn.experimental.rotary_embedding_llama(
        x, rope["cos_matrix"], rope["sin_matrix"], rope["trans_matrix"], is_decode_mode=False
    )
    return ttnn.reshape(x, (b, h, s, d))


# ===== Weight conversion =====

# CPU-module state_dict key -> (transpose?, is_norm?) for the MLA projections.
_MLA_LINEARS = ("wq_a", "wq_b", "wkv_a", "wkv_b", "wo")
_MLA_NORMS = ("q_norm", "kv_norm")


def convert_mla_weights(
    state_dict: dict,
    config,
    mesh_device: ttnn.MeshDevice,
    layer_idx: int = 0,
    cache_path: Optional[Path] = None,
) -> dict:
    """
    Convert an ``MLACPU`` ``state_dict`` (torch tensors) to replicated ttnn
    tensors, cached on disk via ``cache_file_name`` so conversion happens once.

    Linear weights are stored transposed to ``[in, out]`` for ``ttnn.linear``.
    Norm weights are reshaped to ``[1, 1, dim/32, 32]`` (tile rows) per the v3
    convention.  Returns a dict keyed by the CPU parameter prefix.
    """
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def _name(key):
        return str(cache_path / f"layer_{layer_idx}.mla.{key}") if cache_path else None

    def _linear(key):
        return ttnn.as_tensor(
            state_dict[f"{key}.weight"].transpose(-2, -1).contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=_name(key),
        )

    def _norm(key):
        return ttnn.as_tensor(
            state_dict[f"{key}.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=_name(key),
        )

    def _tile_tensor(key, torch_tensor):
        return ttnn.as_tensor(
            torch_tensor.contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=_name(key),
        )

    weights = {k: _linear(k) for k in _MLA_LINEARS}
    weights.update({k: _norm(k) for k in _MLA_NORMS})

    # Decode (MQA) absorption weights, derived from wkv_b = Linear(c -> H*(nope+v)).
    # wkv_b.weight is [H*(nope+v), c]; view per head as [H, nope+v, c] and split:
    #   wkv_b1 [1, H, nope, c] absorbs the nope half into the query  (q_nope @ wkv_b1)
    #   wkv_b2 [1, H, c, v]    absorbs the value half at the output   (x @ wkv_b2)
    H = config.n_heads
    nope, v, c = config.qk_nope_head_dim, config.v_head_dim, config.kv_lora_rank
    wkv_b = state_dict["wkv_b.weight"].view(H, nope + v, c)
    weights["wkv_b1"] = _tile_tensor("wkv_b1", wkv_b[:, :nope, :].unsqueeze(0))  # [1,H,nope,c]
    weights["wkv_b2"] = _tile_tensor("wkv_b2", wkv_b[:, nope:, :].transpose(-2, -1).unsqueeze(0))  # [1,H,c,v]
    return weights


def convert_indexer_weights(
    state_dict: dict,
    mesh_device: ttnn.MeshDevice,
    layer_idx: int = 0,
    cache_path: Optional[Path] = None,
) -> dict:
    """
    Convert an ``IndexerCPU`` ``state_dict`` to replicated ttnn tensors.

    ``state_dict`` keys may be bare (standalone ``IndexerCPU``) or carry an
    ``indexer.`` prefix (nested in ``MLACPU``); both are accepted.  Linear
    weights are stored transposed; ``k_norm`` (a full LayerNorm) keeps weight
    and bias, reshaped to tile rows.  ``weights_proj`` is fp32 in the reference;
    it is cast to bf16 here (functional port).
    """
    sd = {k[len("indexer.") :] if k.startswith("indexer.") else k: v for k, v in state_dict.items()}
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)

    def _name(key):
        return str(cache_path / f"layer_{layer_idx}.indexer.{key}") if cache_path else None

    def _linear(key):
        return ttnn.as_tensor(
            sd[f"{key}.weight"].to(torch.bfloat16).transpose(-2, -1).contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=_name(key),
        )

    def _norm(key, field):
        return ttnn.as_tensor(
            sd[f"{key}.{field}"].reshape(1, 1, -1, ttnn.TILE_SIZE),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_file_name=_name(f"{key}.{field}"),
        )

    return {
        "wq_b": _linear("wq_b"),
        "wk": _linear("wk"),
        "weights_proj": _linear("weights_proj"),
        "k_norm_weight": _norm("k_norm", "weight"),
        "k_norm_bias": _norm("k_norm", "bias"),
    }
