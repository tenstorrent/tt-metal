# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The one place for precision policy and compute / memory / program configs.

Modules never pick a dtype, fidelity or matmul config themselves. They receive a
resolved ``Optimizations`` (built once per device) and read their group from it.

Usage::

    opts = Optimizations.build(mesh_device, policy=PrecisionPolicy.bfp8_weights())
    layer = PplxDecoderLayer.from_state_dict(state_dict, args=args, layer_idx=3, optimizations=opts)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import ttnn

# Matmul roles. Each projection weight belongs to exactly one role.
ROLES = ("attention_qkvg", "attention_out", "delta_in", "delta_out", "mlp_gate_up", "mlp_down", "readout")


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

    def weight_dtype(self, role: str) -> ttnn.DataType:
        return getattr(ttnn, self.weight_dtypes.get(role, "bfloat8_b"))

    def describe(self) -> dict:
        return asdict(self)


@dataclass
class LinearOptimizations:
    """Per-role matmul configs for prefill-shaped [B, S, K] x [K, N] projections."""

    compute_kernel_cfg: dict  # role -> compute kernel config
    output_dtype: ttnn.DataType
    output_memcfg: ttnn.MemoryConfig
    core_grid: ttnn.CoreGrid
    # minimal_matmul for long prefill rows; below the threshold ttnn.linear auto-configures.
    minimal_min_rows: int
    minimal_config: object


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
        policy = policy or PrecisionPolicy.bfp8_weights()
        grid = mesh_device.compute_with_storage_grid_size()
        if prefill_chunk % 128:
            raise ValueError("prefill_chunk must be a multiple of the 128-token SDPA chunk")
        linear = LinearOptimizations(
            compute_kernel_cfg={role: _compute_cfg(policy.fidelities.get(role, "HiFi2")) for role in ROLES},
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


def _compute_cfg(fidelity: str, *, fp32_dest_acc: bool = True):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=True,
    )
