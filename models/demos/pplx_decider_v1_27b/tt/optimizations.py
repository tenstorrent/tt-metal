# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The one place for precision policy and compute / memory / program configs.

Modules never pick a dtype, fidelity or matmul config themselves. They receive a
resolved ``Optimizations`` (built once per device) and read their group from it.

The default policy is the stage-8 selection, read from
``doc/datatype_sweep/selected_precision_config.json``. ``$PPLX_DECIDER_PRECISION_CONFIG`` points
the default at another policy JSON (the sweep candidates live in ``doc/datatype_sweep/candidates/``;
``C0_bfp8_all_hifi2.json`` is the safe all-BFP8 baseline).

Usage::

    opts = Optimizations.build(mesh_device)  # PrecisionPolicy.default()
    layer = PplxDecoderLayer.from_state_dict(state_dict, args=args, layer_idx=3, optimizations=opts)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import ttnn

# Matmul roles. Each projection weight belongs to exactly one role.
ROLES = ("attention_qkvg", "attention_out", "delta_in", "delta_out", "mlp_gate_up", "mlp_down", "readout")

SELECTED_CONFIG_PATH = Path(__file__).resolve().parents[1] / "doc" / "datatype_sweep" / "selected_precision_config.json"
PRECISION_CONFIG_ENV = "PPLX_DECIDER_PRECISION_CONFIG"


@dataclass(frozen=True)
class PrecisionPolicy:
    """Dtypes and fidelities. ``name`` is recorded next to every PCC number."""

    name: str
    activation_dtype: str = "bfloat16"
    weight_dtypes: dict = field(default_factory=dict)  # role -> dtype name
    fidelities: dict = field(default_factory=dict)  # role -> MathFidelity name
    embedding_dtype: str = "bfloat16"
    norm_weight_dtype: str = "bfloat16"
    kv_cache_dtype: str = "bfloat16"
    recurrent_dtype: str = "float32"  # HF mamba_ssm_dtype; pinned
    conv_state_dtype: str = "bfloat16"
    norm_fidelity: str = "HiFi4"

    @classmethod
    def bfp8_weights(cls) -> "PrecisionPolicy":
        """Stage-1 policy: BF16 activations, BFP8 weights for every projection, HiFi2 matmuls, fp32 acc."""
        return cls(
            name="act_bf16__w_bfp8_all__hifi2",
            weight_dtypes={role: "bfloat8_b" for role in ROLES},
            fidelities={role: "HiFi2" for role in ROLES},
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PrecisionPolicy":
        """Policy from the ``policy`` object of a precision config JSON. Every role needs a dtype and a fidelity."""
        data = json.loads(Path(path).read_text())
        values = data["policy"]
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"{path}: unknown policy fields {sorted(unknown)}")
        policy = cls(**values)
        for table in ("weight_dtypes", "fidelities"):
            missing = [r for r in ROLES if r not in getattr(policy, table)]
            if missing:
                raise ValueError(f"{path}: policy.{table} misses roles {missing}")
        return policy

    @classmethod
    def default(cls) -> "PrecisionPolicy":
        """The selected stage-8 policy, or the JSON named by ``$PPLX_DECIDER_PRECISION_CONFIG``."""
        return cls.from_json(os.environ.get(PRECISION_CONFIG_ENV) or SELECTED_CONFIG_PATH)

    def weight_dtype(self, role: str) -> ttnn.DataType:
        return getattr(ttnn, self.weight_dtypes.get(role, "bfloat8_b"))

    def describe(self) -> dict:
        return asdict(self)


@dataclass
class LinearOptimizations:
    """Per-role matmul configs for prefill-shaped [B, S, K] x [K, N] projections.

    Program configs are per projection, keyed by the stored weight shape (K, N); every text-layer
    projection has a distinct shape except attention o_proj and the GDN out_proj, which share one.
    A shape missing from the tables (the decision-head readout) keeps the stage-2 behaviour.
    The configs were selected in stage 3A (doc/optimized_decoder/work_log.md).
    """

    compute_kernel_cfg: dict  # role -> compute kernel config
    output_dtype: ttnn.DataType
    output_memcfg: ttnn.MemoryConfig
    core_grid: ttnn.CoreGrid
    # minimal_matmul for long prefill rows; below the threshold ttnn.linear auto-configures.
    minimal_min_rows: int
    minimal_config: object
    # (K, N) -> MinimalMatmulConfig for rows >= minimal_min_rows (the 1024 / 2048 chunks).
    minimal_configs: dict = field(default_factory=dict)
    # Short rows (the 128 bucket): at most this many rows use the short-row tables below.
    short_max_rows: int = 0
    # (K, N) -> (per_core_N, in0_block_w) of a 1D in0-multicast ttnn.linear on the interleaved weight.
    short_mcast1d: dict = field(default_factory=dict)
    # (M_block, K_block, N_block, subblock_h, subblock_w) of the fused-SwiGLU minimal_matmul for short
    # rows; its grid height is cut to the row tiles (fewer idle receivers of each weight multicast).
    short_swiglu_blocks: tuple | None = None

    def minimal_config_for(self, k: int, n: int):
        return self.minimal_configs.get((k, n), self.minimal_config)

    def short_program_config(self, k: int, n: int, m_tiles: int):
        """1D in0-multicast program config for a short-row projection, or None for ttnn.linear auto."""
        entry = self.short_mcast1d.get((k, n))
        if entry is None:
            return None
        per_core_n, in0_block_w = entry
        subblock_w = max(d for d in range(1, 5) if per_core_n % d == 0)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(self.core_grid.x, self.core_grid.y),
            in0_block_w=in0_block_w,
            out_subblock_h=1,
            out_subblock_w=subblock_w,
            out_block_h=m_tiles,
            out_block_w=per_core_n,
            per_core_M=m_tiles,
            per_core_N=per_core_n,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

    def short_swiglu_config(self, m_tiles: int):
        m_block, k_block, n_block, sub_h, sub_w = self.short_swiglu_blocks
        return ttnn.MinimalMatmulConfig(
            M_block_size=m_block,
            K_block_size=k_block,
            N_block_size=n_block,
            subblock_h=sub_h,
            subblock_w=sub_w,
            compute_with_storage_grid_size=(self.core_grid.x, max(2, min(self.core_grid.y, m_tiles))),
        )


@dataclass
class NormOptimizations:
    compute_kernel_cfg: object
    output_memcfg: ttnn.MemoryConfig


@dataclass
class AttentionOptimizations:
    kv_cache_dtype: ttnn.DataType
    page_size: int
    sdpa_q_chunk: int
    sdpa_k_chunk: int
    sdpa_grid: tuple[int, int]
    compute_kernel_cfg: object
    # rotary_embedding_llama rejects fp32 dest accumulation for head_dim > 128.
    rope_compute_kernel_cfg: object


@dataclass
class DeltaOptimizations:
    recurrent_dtype: ttnn.DataType
    conv_state_dtype: ttnn.DataType
    conv_channel_chunk: int
    scan_chunk: int
    elementwise_compute_kernel_cfg: object


@dataclass
class Optimizations:
    mesh_device: object
    policy: PrecisionPolicy
    max_seq_len: int
    prefill_chunk: int  # bounded physical chunk for every layer; logical lengths are arbitrary
    linear: LinearOptimizations
    norm: NormOptimizations
    attention: AttentionOptimizations
    delta: DeltaOptimizations

    @classmethod
    def build(
        cls,
        mesh_device,
        *,
        policy: PrecisionPolicy | None = None,
        max_seq_len: int = 8192,
        prefill_chunk: int = 2048,
    ) -> "Optimizations":
        policy = policy or PrecisionPolicy.default()
        grid = mesh_device.compute_with_storage_grid_size()
        if prefill_chunk % 128:
            raise ValueError("prefill_chunk must be a multiple of the 128-token SDPA chunk")
        linear = LinearOptimizations(
            compute_kernel_cfg={role: _compute_cfg(policy.fidelities[role]) for role in ROLES},
            output_dtype=getattr(ttnn, policy.activation_dtype),
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(x=grid.x, y=grid.y),
            minimal_min_rows=512,
            minimal_config=ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=8,
                N_block_size=16,
                subblock_h=1,
                subblock_w=4,
                compute_with_storage_grid_size=(grid.x, grid.y),
            ),
            minimal_configs={shape: _minimal_cfg(blocks, grid) for shape, blocks in _MINIMAL_BLOCKS.items()},
            short_max_rows=128,
            short_mcast1d=dict(_SHORT_MCAST1D),
            short_swiglu_blocks=(4, 32, 8, 1, 4),
        )
        norm = NormOptimizations(
            compute_kernel_cfg=_compute_cfg(policy.norm_fidelity),
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        )
        attention = AttentionOptimizations(
            kv_cache_dtype=getattr(ttnn, policy.kv_cache_dtype),
            page_size=32,
            sdpa_q_chunk=128,
            sdpa_k_chunk=128,
            sdpa_grid=(grid.x, grid.y),
            compute_kernel_cfg=_compute_cfg("HiFi4"),
            rope_compute_kernel_cfg=_compute_cfg("HiFi4", fp32_dest_acc=False),
        )
        delta = DeltaOptimizations(
            recurrent_dtype=getattr(ttnn, policy.recurrent_dtype),
            conv_state_dtype=getattr(ttnn, policy.conv_state_dtype),
            conv_channel_chunk=256,
            scan_chunk=32,
            elementwise_compute_kernel_cfg=_compute_cfg("HiFi4"),
        )
        return cls(
            mesh_device=mesh_device,
            policy=policy,
            max_seq_len=max_seq_len,
            prefill_chunk=prefill_chunk,
            linear=linear,
            norm=norm,
            attention=attention,
            delta=delta,
        )


# ----------------------------------------------------------------------------------------------
# Vision tower (stage 12A). Kept apart from ``ROLES`` / ``PrecisionPolicy`` so the text precision
# JSONs (which must name every text role) are unchanged. Person decision 2026-10-09: the vision
# tower is BF16 end to end (weights and activations), fp32 matmul accumulation, no BFP8/BFP4.
# ----------------------------------------------------------------------------------------------

# Matmul roles of the vision tower. Each vision projection weight belongs to exactly one role.
VISION_ROLES = ("vision_patch", "vision_qkv", "vision_proj", "vision_fc1", "vision_fc2", "merger_fc1", "merger_fc2")


@dataclass(frozen=True)
class VisionPrecisionPolicy:
    name: str = "vision_bf16__w_bf16__hifi4"
    activation_dtype: str = "bfloat16"
    weight_dtypes: dict = field(default_factory=lambda: {role: "bfloat16" for role in VISION_ROLES})
    fidelities: dict = field(default_factory=lambda: {role: "HiFi4" for role in VISION_ROLES})
    norm_weight_dtype: str = "bfloat16"
    bias_dtype: str = "bfloat16"
    norm_fidelity: str = "HiFi4"

    def __post_init__(self):
        for table in ("weight_dtypes", "fidelities"):
            missing = [r for r in VISION_ROLES if r not in getattr(self, table)]
            if missing:
                raise ValueError(f"vision policy.{table} misses roles {missing}")
        low = {r: d for r, d in self.weight_dtypes.items() if d != "bfloat16"}
        if low or self.activation_dtype != "bfloat16":
            raise ValueError(f"The vision tower is BF16 only (person decision); got {low or self.activation_dtype}")

    def weight_dtype(self, role: str) -> ttnn.DataType:
        return getattr(ttnn, self.weight_dtypes[role])

    def describe(self) -> dict:
        return asdict(self)


@dataclass
class VisionOptimizations:
    """Compute / memory / program configs for the vision tower on one device."""

    mesh_device: object
    policy: VisionPrecisionPolicy
    compute_kernel_cfg: dict  # role -> compute kernel config
    output_dtype: ttnn.DataType
    output_memcfg: ttnn.MemoryConfig
    core_grid: ttnn.CoreGrid
    norm_compute_kernel_cfg: object
    sdpa_q_chunk: int
    sdpa_k_chunk: int
    sdpa_grid: tuple[int, int]
    sdpa_compute_kernel_cfg: object
    rope_compute_kernel_cfg: object

    @classmethod
    def build(cls, mesh_device, *, policy: VisionPrecisionPolicy | None = None) -> "VisionOptimizations":
        policy = policy or VisionPrecisionPolicy()
        grid = mesh_device.compute_with_storage_grid_size()
        return cls(
            mesh_device=mesh_device,
            policy=policy,
            compute_kernel_cfg={role: _compute_cfg(policy.fidelities[role]) for role in VISION_ROLES},
            output_dtype=getattr(ttnn, policy.activation_dtype),
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=ttnn.CoreGrid(x=grid.x, y=grid.y),
            norm_compute_kernel_cfg=_compute_cfg(policy.norm_fidelity),
            sdpa_q_chunk=128,
            sdpa_k_chunk=128,
            sdpa_grid=(grid.x, grid.y),
            sdpa_compute_kernel_cfg=_compute_cfg("HiFi4"),
            rope_compute_kernel_cfg=_compute_cfg("HiFi4"),
        )

    def sdpa_program_config(self):
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_grid,
            q_chunk_size=self.sdpa_q_chunk,
            k_chunk_size=self.sdpa_k_chunk,
            exp_approx_mode=False,
        )


# Stage 3A matmul geometry (one p150a, 13x10 grid, BFP8 weights, HiFi2 + fp32 accumulation; the
# fp32 dest holds 4 tiles, so subblock_h * subblock_w <= 4). minimal_matmul puts N on the 13 grid
# columns when M <= N, so an N block equal to the per-core N tiles (or half of it) avoids a ragged
# last block. Measured per shape at M = 1024 and 2048: work_log.md, "Matmul sweep".
_MINIMAL_BLOCKS = {
    # (K, N): (M_block, K_block, N_block, subblock_h, subblock_w)
    (5120, 8192): (4, 8, 10, 4, 1),  # attention q|k|v (20 N tiles per core)
    (5120, 6144): (4, 8, 8, 4, 1),  # attention output gate
    (6144, 5120): (4, 8, 13, 4, 1),  # attention o_proj and GDN out_proj (13 N tiles per core)
    (5120, 10240): (4, 8, 13, 4, 1),  # GDN in-projection q|k|v (25 N tiles per core)
    (5120, 6272): (4, 8, 8, 1, 4),  # GDN in-projection z|b|a
    (5120, 34816): (4, 32, 8, 1, 4),  # MLP gate|up, fused SwiGLU
    (17408, 5120): (4, 16, 7, 4, 1),  # MLP down (13 N tiles per core)
}
# Short rows (128 bucket): ttnn.linear auto picks a small in0 block and is weight-read bound at
# 42-68 % DRAM. A 1D in0-multicast config with a wider K block reads the same interleaved weight
# faster. (K, N): (per_core_N, in0_block_w). The GDN q|k|v projection keeps ttnn.linear auto (no
# 1D config beat it).
_SHORT_MCAST1D = {
    (5120, 8192): (4, 4),
    (5120, 6144): (3, 8),
    (6144, 5120): (2, 8),
    (5120, 6272): (2, 4),
    (17408, 5120): (2, 8),
}


def _minimal_cfg(blocks, grid):
    m_block, k_block, n_block, sub_h, sub_w = blocks
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=k_block,
        N_block_size=n_block,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=(grid.x, grid.y),
    )


def _compute_cfg(fidelity: str, *, fp32_dest_acc: bool = True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=True,
    )
