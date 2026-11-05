# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phi-3.5-mini MLP: ``MLP1D`` with a model-owned single-device decode path and decode weights.

As in :mod:`models.common.models.phi35_mini.attention`: ``MLP1D`` holds ``w1`` / ``w2`` / ``w3`` DRAM-interleaved
for its inherited prefill path (1D-multicast matmuls for the 128-row bucket via the ``prefill_128_*`` configs).
Prefill above 128 rows and decode both run W1, W3 and SiLU(W1) * W3 as one fused-SwiGLU matmul over ``[W1 | W3]``
laid out as gate / up column tile pairs: prefill on ``minimal_matmul`` with an interleaved copy, decode on the
DRAM-sharded pipeline matmul with a DRAM width-sharded copy (``w13_swiglu_decode``). W2 decode reads its own
DRAM width-sharded copy (``w2_decode``).
"""

from __future__ import annotations

from dataclasses import dataclass

import ttnn
from models.common.models.phi35_mini.attention import PREFILL_1D_SEQ_LEN, _packed_prefill_mm_config, _residual_gate
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig, _load_input_device_tensor


@dataclass
class Phi35MiniMLPConfig(MLP1DConfig):
    # Decode: [W1 | W3] gate / up tile pairs, DRAM width-sharded, for the fused-SwiGLU DRAM-sharded matmul
    w13_swiglu_decode: LazyWeight | None = None
    decode_w13_swiglu_prg_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig | None = None
    w2_decode: LazyWeight | None = None
    prefill_128_w1_w3_prg_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    # Prefill above 128 rows: [W1|W3] tile-pair interleaved for minimal_matmul's fused SwiGLU
    w13_swiglu_prefill: LazyWeight | None = None
    prefill_swiglu_minimal_matmul_config: ttnn.MinimalMatmulConfig | None = None
    prefill_128_w2_prg_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    # 128-row W2 of _prefill_128 on the multi-row DRAM-streamed program (dram_stream_workers_per_bank), which reads
    # w2_decode, the DRAM width-sharded copy; None runs prefill_128_w2_prg_config on w2.
    prefill_128_w2_stream_prg_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    # Compute config of that streamed W2 (None: ff2_compute_kernel_cfg).
    prefill_128_w2_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
    # 128-row bucket: dtype of the W1 / W3 outputs that the SiLU-multiply reads (None: linear_dtype)
    prefill_128_w1_w3_dtype: ttnn.DataType | None = None
    # 128-row bucket: one 1D-multicast matmul with fuse_swiglu over the gate / up tile-pair interleaved [W1|W3] instead
    # of W1, W3 and the SiLU-multiply; None keeps the three ops. It reads the DRAM width-sharded decode copy
    # (w13_swiglu_decode): each core's columns lie contiguously in one bank, so a K row of them is one NOC read.
    prefill_128_w13_swiglu_prg_config: ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig | None = None
    # Prefill above 128 rows: W2 with the residual add fused in (dit_minimal_matmul_addcmul_fused); None keeps an add.
    prefill_w2_residual_minimal_matmul_config: ttnn.MinimalMatmulConfig | None = None
    # Compute config of that fused W2 (None: ff2_compute_kernel_cfg); above 128 rows W2 is compute-bound.
    prefill_w2_residual_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None


class Phi35MiniMLP(MLP1D):
    @classmethod
    def from_config(cls, config: Phi35MiniMLPConfig) -> Phi35MiniMLP:
        if not isinstance(config, Phi35MiniMLPConfig):
            raise TypeError("Phi35MiniMLP.from_config expects Phi35MiniMLPConfig")
        if config.w13_swiglu_decode is None or config.w2_decode is None:
            raise ValueError("Phi35MiniMLPConfig needs w13_swiglu_decode and w2_decode")
        if config.decode_w13_swiglu_prg_config is None or not config.decode_w13_swiglu_prg_config.fuse_swiglu:
            raise ValueError("Phi35MiniMLPConfig needs a fuse_swiglu decode_w13_swiglu_prg_config")
        if config.w13_swiglu_prefill is None or config.prefill_swiglu_minimal_matmul_config is None:
            raise ValueError("Phi35MiniMLPConfig needs w13_swiglu_prefill and prefill_swiglu_minimal_matmul_config")
        if not config.prefill_w2_minimal_matmul:
            raise ValueError("Phi35MiniMLP prefill above 128 rows runs W2 on minimal_matmul")
        if config.mlp_activation_type != ttnn.UnaryOpType.SILU:
            raise ValueError("Phi-3.5-mini's MLP gate activation is SiLU")
        instance = super().from_config(config)
        cfg = instance.config
        default_w1_w3, default_w2 = cfg.prefill_w1_w3_prg_config, cfg.prefill_w2_prg_config
        w1_w3_128, w2_128 = cfg.prefill_128_w1_w3_prg_config, cfg.prefill_128_w2_prg_config
        cfg.prefill_w1_w3_prg_config = lambda seq_len: w1_w3_128 if seq_len == PREFILL_1D_SEQ_LEN else default_w1_w3(seq_len)
        cfg.prefill_w2_prg_config = lambda seq_len: w2_128 if seq_len == PREFILL_1D_SEQ_LEN else default_w2(seq_len)
        return instance

    def load_device_weights(self):
        if self._device_weights_loaded:
            return
        super().load_device_weights()
        self.w13_swiglu_decode = self.config.w13_swiglu_decode.get_device_weight()
        self.w2_decode = self.config.w2_decode.get_device_weight()
        self.w13_swiglu_prefill = self.config.w13_swiglu_prefill.get_device_weight()
        self.residual_gate = _residual_gate(self.config.mesh_device, self.config.dim)

    def prefill_forward_with_residual(
        self, x: ttnn.Tensor, residual: ttnn.Tensor, *, residual_memcfg: ttnn.MemoryConfig
    ) -> ttnn.Tensor:
        """``residual + mlp(x)``. Above 128 rows the residual add runs inside the W2 matmul (residual + 1 * (h @ W2) *
        ones-row): P150, 4096 rows, 764 us against 863 us for W2 and a separate add."""
        self.load_device_weights()
        cfg = self.config
        if x.shape[-2] == PREFILL_1D_SEQ_LEN or cfg.prefill_w2_residual_minimal_matmul_config is None:
            out = self.prefill_forward(x)
            h = ttnn.add(residual, out, memory_config=residual_memcfg)
            ttnn.deallocate(out)
            return h
        w2_in = self._prefill_w13_swiglu(x)
        out = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            w2_in,
            self.w2,
            1.0,
            residual,
            self.residual_gate,
            config=_packed_prefill_mm_config("w2", w2_in.shape[-2], cfg.prefill_w2_residual_minimal_matmul_config),
            compute_kernel_config=cfg.prefill_w2_residual_compute_kernel_cfg or cfg.ff2_compute_kernel_cfg,
            dtype=residual.dtype,
        )
        ttnn.deallocate(w2_in)
        return out

    def _prefill_w13_swiglu(self, x: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        if not isinstance(x, ttnn.Tensor):  # a device tensor may be in L1 (the model's norm output)
            x = _load_input_device_tensor(x, cfg, mode="prefill")
        w2_in = ttnn.experimental.minimal_matmul(
            x,
            self.w13_swiglu_prefill,
            fuse_swiglu=True,
            dtype=cfg.mul_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            config=_packed_prefill_mm_config("w13", x.shape[-2], cfg.prefill_swiglu_minimal_matmul_config),
        )
        ttnn.deallocate(x)
        return w2_in

    def prefill_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """Prefill MLP.

        128-row bucket: see ``_prefill_128``.
        Longer prefills: one minimal_matmul with fused SwiGLU over the tile-pair interleaved [W1|W3] weight, then
        W2 on minimal_matmul. ``MLP1D`` runs W1 and W3 as two 2D matmuls (1.27 ms each at 4096 rows on a P150) and
        then a separate SiLU-multiply that reads both 4096 x 8192 outputs back from DRAM (0.45 ms).
        """
        self.load_device_weights()
        cfg = self.config
        if isinstance(x, ttnn.Tensor) and x.shape[-2] == PREFILL_1D_SEQ_LEN:
            return self._prefill_128(x)
        seq_len = x.shape[-2]
        w2_in = self._prefill_w13_swiglu(x)
        w2_out = ttnn.experimental.minimal_matmul(
            w2_in,
            self.w2,
            compute_kernel_config=cfg.ff2_compute_kernel_cfg,
            config=cfg.prefill_w2_minimal_matmul_config(seq_len),
        )
        ttnn.deallocate(w2_in)
        return w2_out

    def _prefill_128(self, x: ttnn.Tensor, out_memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor:
        """128-row bucket: 1D-multicast W1 and W3, SiLU(W1) * W3 as one multiply, W2.

        The input (L1 or DRAM), the W1 / W3 outputs (2 MB each) and the W2 output stay in L1: W1 + W3 + multiply take
        128.5 us on a P150 against 154.2 us with the W1 / W3 outputs in DRAM. The SiLU runs in the multiply rather than
        fused into W1, where it held up the matmul's output by 13 us (W1 64.6 us vs W3 51.7 us); time to first token
        at batch 1 went 14.2 -> 14.0 ms.
        """
        cfg = self.config
        if cfg.prefill_128_w13_swiglu_prg_config is not None:
            w2_in = ttnn.linear(
                x,
                self.w13_swiglu_decode,
                dtype=cfg.mul_dtype,
                compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
                program_config=cfg.prefill_128_w13_swiglu_prg_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x)
            return self._prefill_128_w2(w2_in, out_memory_config)
        w1_out = ttnn.linear(
            x,
            self.w1,
            dtype=cfg.prefill_128_w1_w3_dtype or cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.prefill_128_w1_w3_prg_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        w3_out = ttnn.linear(
            x,
            self.w3,
            dtype=cfg.prefill_128_w1_w3_dtype or cfg.linear_dtype,
            compute_kernel_config=cfg.ff1_3_compute_kernel_cfg,
            program_config=cfg.prefill_128_w1_w3_prg_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        w2_in = ttnn.mul(
            w1_out,
            w3_out,
            input_tensor_a_activations=[cfg.mlp_activation_type],
            dtype=cfg.mul_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w3_out)
        ttnn.deallocate(w1_out)
        return self._prefill_128_w2(w2_in, out_memory_config)

    def prefill_128_to(self, x: ttnn.Tensor, out_memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
        """128-row MLP with the W2 output in ``out_memory_config`` (needs prefill_128_w2_stream_prg_config, whose
        program writes any layout), e.g. the residual stream's block-sharded one for the next norm's in-place add."""
        self.load_device_weights()
        if self.config.prefill_128_w2_stream_prg_config is None:
            raise ValueError("prefill_128_to needs prefill_128_w2_stream_prg_config")
        return self._prefill_128(x, out_memory_config)

    def _prefill_128_w2(self, w2_in: ttnn.Tensor, out_memory_config: ttnn.MemoryConfig | None = None) -> ttnn.Tensor:
        cfg = self.config
        stream = cfg.prefill_128_w2_stream_prg_config
        w2_out = ttnn.linear(
            w2_in,
            self.w2_decode if stream else self.w2,
            compute_kernel_config=(cfg.prefill_128_w2_compute_kernel_cfg if stream else None)
            or cfg.ff2_compute_kernel_cfg,
            dtype=cfg.linear_dtype,
            program_config=stream or cfg.prefill_128_w2_prg_config,
            memory_config=(out_memory_config if stream else None) or ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)
        return w2_out

    def decode_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        """SiLU(x @ W1) * (x @ W3) as one fused-SwiGLU DRAM-sharded matmul -> W2 (DRAM-sharded) on its output layout."""
        self.load_device_weights()
        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config
        if cfg.mesh_device.get_num_devices() != 1:
            raise ValueError("Phi35MiniMLP decode is single-device only")

        w2_in = ttnn.linear(
            x,
            self.w13_swiglu_decode,
            dtype=cfg.mul_dtype,
            compute_kernel_config=cfg.decode_ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w13_swiglu_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)

        # W2's program config reads its input on the fused matmul's output layout, so there is no reshard.
        w2_out = ttnn.linear(
            w2_in,
            self.w2_decode,
            compute_kernel_config=cfg.decode_ff2_compute_kernel_cfg,
            dtype=cfg.linear_dtype,
            program_config=cfg.decode_w2_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(w2_in)
        return ttnn.to_memory_config(w2_out, cfg.decode_residual_memcfg)
