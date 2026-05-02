# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Voxtral-4B-TTS-2603 TTNN implementation.

Target device: N150 (single Wormhole B0, 12GB DRAM).
Model ~8GB BF16, fits comfortably.  No tensor parallelism needed.
"""

import ttnn


class VoxtralTTSConfig:
    # ── Text decoder / acoustic transformer ──────────────────────────────
    dim = 3072
    n_heads = 32
    n_kv_heads = 8
    head_dim = 128
    padded_head_dim = 128
    norm_eps = 1e-5
    rope_theta = 1_000_000.0
    vocab_size = 131072
    n_layers = 26
    intermediate_size = 9216
    # TTS sequences are short: max ~500 voice frames + ~1000 text tokens = ~1500 positions.
    # 4096 is comfortably sufficient and saves ~6.5GB vs the 65536 default.
    max_seq_len = 4096

    # ── Codec decoder ─────────────────────────────────────────────────────
    codec_dim = 1024
    codec_n_heads = 8
    codec_head_dim = 128
    codec_intermediate = 4096
    codec_norm_eps = 0.01

    # ── Acoustic token spaces ─────────────────────────────────────────────
    semantic_codebook_size = 8192
    acoustic_codebook_size = 21
    n_acoustic_codebooks = 36

    # ── N150 single-device settings ───────────────────────────────────────
    num_devices = 1
    cluster_shape = [1, 1]
    is_multichip = False
    max_batch_size = 1
    dummy_weights = False
    tile_size = 32
    MAX_QKV_MM_SEQ_LEN = 2048
    min_kv_prefill_shard_seqlen = 1024

    # Derived (no TP)
    n_local_heads = n_heads
    n_local_kv_heads = n_kv_heads
    qkv_size = head_dim * (n_heads + 2 * n_kv_heads)  # 6144

    def __init__(self, mesh_device=None):
        self.mesh_device = mesh_device

        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self._model_config = self._build_model_config()

    def _build_model_config(self):
        cfg = {}
        cfg["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            exp_approx_mode=False,
            q_chunk_size=256,
            k_chunk_size=256,
        )
        cfg["SDPA_DECODE_COMPUTE_PROGCFG"] = self.compute_kernel_config_hifi4
        # N150 has 7×8=56 cores; use (8, 4)=32 cores for prefill SDPA
        cfg["SDPA_PROGCFG"] = lambda seq_len: ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            q_chunk_size=256,
            k_chunk_size=256,
        )
        cfg["XQKV_PREFILL_PROGCFG"] = lambda seq_len: None
        cfg["WO_PREFILL_PROGCFG"] = lambda seq_len: None
        cfg["ATTN_OUTPUT_PROGCFG"] = None
        return cfg

    def get_model_config(self):
        return self._model_config
