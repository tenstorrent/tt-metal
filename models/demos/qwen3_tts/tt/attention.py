# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
Attention implementation for Qwen3-TTS.

Note: Qwen3-TTS uses non-interleaved RoPE (pairs dims i and i+64),
while TTNN rotary_embedding_llama uses interleaved format (pairs dims 2i and 2i+1).
This module handles the necessary dimension rearrangement.

Supports both prefill mode (full sequence) and decode mode (single token with KV cache).
"""

from typing import Optional, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    build_dram_sharded_weight,
    dram_sharded_program_config,
    find_grid_k_n,
    width_sharded_l1_memcfg,
)
from models.demos.qwen3_tts.tt.linear_1d_program_config import make_linear_1d_program_config

# Prefill bucket sizes for which we pre-build sharded NLP head op memcfgs.
# Matches SUPPORTED_PREFILL_LENS in demo_full_ttnn_tts.py.
_PREFILL_SEQS = (32, 64, 96, 128, 192, 256)


class Attention(LightweightModule):
    """
    Multi-head attention with GQA and QK-norm for Qwen3-TTS.

    Features:
    - Grouped Query Attention (GQA) with 16 Q heads and 8 KV heads
    - QK-normalization (q_norm, k_norm) for stable training
    - RoPE positional embeddings
    - Float32 attention computation (required — bfloat16 SDPA loses precision)
    """

    def __init__(
        self,
        device,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        state_dict: dict,
        layer_prefix: str,
        rms_norm_eps: float = 1e-6,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.rms_norm_eps = rms_norm_eps

        def _permute_rope_head_dim_rows(weight_2d, local_heads: int, head_dim: int):
            # Convert each head block from non-interleaved [..., d0..d63, d64..d127]
            # to interleaved [..., d0, d64, d1, d65, ...] once at load time.
            hidden_size_in = int(weight_2d.shape[1])
            half_dim = head_dim // 2
            w = weight_2d.reshape(local_heads, head_dim, hidden_size_in)
            first = w[:, :half_dim, :]
            second = w[:, half_dim:, :]
            out = w.clone()
            out[:, 0::2, :] = first
            out[:, 1::2, :] = second
            return out.reshape(local_heads * head_dim, hidden_size_in)

        def _permute_rope_head_dim_vector(weight_1d, head_dim: int):
            half_dim = head_dim // 2
            out = weight_1d.clone()
            out[0::2] = weight_1d[:half_dim]
            out[1::2] = weight_1d[half_dim:]
            return out

        is_mesh_device = device.__class__.__name__ == "MeshDevice"

        def get_cache_name(name):
            if weight_cache_path is None:
                return None
            return weight_cache_path / f"{layer_prefix}_{name}".replace(".", "_")

        # Fuse QKV weights: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim]
        q_proj_weight = _permute_rope_head_dim_rows(
            state_dict[f"{layer_prefix}.self_attn.q_proj.weight"], num_heads, head_dim
        )
        k_proj_weight = _permute_rope_head_dim_rows(
            state_dict[f"{layer_prefix}.self_attn.k_proj.weight"], num_kv_heads, head_dim
        )
        v_proj_weight = state_dict[f"{layer_prefix}.self_attn.v_proj.weight"]
        o_proj_weight = state_dict[f"{layer_prefix}.self_attn.o_proj.weight"]

        # Torch reference (fused QKV on CPU, then as_tensor):
        # qkv_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
        # qkv_weight = torch.transpose(qkv_weight, -2, -1).unsqueeze(0).unsqueeze(0)
        # self.wqkv = ttnn.as_tensor(
        #     qkv_weight,
        #     device=device,
        #     dtype=weight_dtype,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     cache_file_name=get_cache_name("wqkv"),
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        # )

        _mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None
        _dram = ttnn.DRAM_MEMORY_CONFIG
        _fused_qkv = (num_heads + 2 * num_kv_heads) * head_dim
        # Fused QKV on host, single upload (same pattern as MLP): stable [1,1,hidden,fused] DRAM weights for traces.
        qkv_2d = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
        assert int(qkv_2d.shape[0]) == _fused_qkv
        _wqkv_host = qkv_2d.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
        _wqkv_cache = get_cache_name("wqkv")
        if _wqkv_cache is not None:
            self.wqkv = ttnn.as_tensor(
                _wqkv_host,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                cache_file_name=_wqkv_cache,
                mesh_mapper=_mesh_mapper,
            )
        else:
            self.wqkv = ttnn.from_torch(
                _wqkv_host,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_dram,
                mesh_mapper=_mesh_mapper,
            )

        # Torch reference (o_proj transpose on CPU, then as_tensor):
        # o_proj_weight = torch.transpose(o_proj_weight, -2, -1).unsqueeze(0).unsqueeze(0)
        # self.wo = ttnn.as_tensor(
        #     o_proj_weight,
        #     device=device,
        #     dtype=weight_dtype,
        #     layout=ttnn.TILE_LAYOUT,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     cache_file_name=get_cache_name("wo"),
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        # )

        _o_rows = num_heads * head_dim
        _wo_host = o_proj_weight.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous()
        _wo_cache = get_cache_name("wo")
        if _wo_cache is not None:
            self.wo = ttnn.as_tensor(
                _wo_host,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=_dram,
                cache_file_name=_wo_cache,
                mesh_mapper=_mesh_mapper,
            )
        else:
            self.wo = ttnn.from_torch(
                _wo_host,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_dram,
                mesh_mapper=_mesh_mapper,
            )

        # QK-norm weights (per-head RMSNorm)
        q_norm_weight = _permute_rope_head_dim_vector(state_dict[f"{layer_prefix}.self_attn.q_norm.weight"], head_dim)
        k_norm_weight = _permute_rope_head_dim_vector(state_dict[f"{layer_prefix}.self_attn.k_norm.weight"], head_dim)

        TILE = 32
        q_norm_torch = q_norm_weight.unsqueeze(0).view(1, 1, head_dim).reshape([1, 1, head_dim // TILE, TILE])
        k_norm_torch = k_norm_weight.unsqueeze(0).view(1, 1, head_dim).reshape([1, 1, head_dim // TILE, TILE])

        _qk_norm_gamma_memcfg = ttnn.L1_MEMORY_CONFIG

        _qn_cache = get_cache_name("q_norm")
        if _qn_cache is not None:
            self.q_norm_weight = ttnn.as_tensor(
                q_norm_torch,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=_qk_norm_gamma_memcfg,
                cache_file_name=_qn_cache,
                mesh_mapper=_mesh_mapper,
            )
        else:
            self.q_norm_weight = ttnn.from_torch(
                q_norm_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=_qk_norm_gamma_memcfg,
                mesh_mapper=_mesh_mapper,
            )

        _kn_cache = get_cache_name("k_norm")
        if _kn_cache is not None:
            self.k_norm_weight = ttnn.as_tensor(
                k_norm_torch,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=_qk_norm_gamma_memcfg,
                cache_file_name=_kn_cache,
                mesh_mapper=_mesh_mapper,
            )
        else:
            self.k_norm_weight = ttnn.from_torch(
                k_norm_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=_qk_norm_gamma_memcfg,
                mesh_mapper=_mesh_mapper,
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Keep fp32 accumulation for attention matmuls. HiFi2 is the current
        # quality/speed tradeoff for this path.
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Prefill SDPA via the fused ttnn.transformer.scaled_dot_product_attention.
        # Bumped to HiFi4 (vs the manual chain's HiFi2) because the fused op runs
        # on bf16 Q/K/V — the K-amplification from k_norm (gain ≈ 68 → values
        # ≈ ±260) needs full bf16 multiply mantissa to preserve attention scores
        # well enough for the AR sampling trajectory. fp32_dest_acc_en keeps the
        # softmax + matmul accumulation in fp32. Together this gives parity-ish
        # numerics with the manual fp32 path while avoiding the
        # repeat_interleave→DRAM bounce + 4 typecasts that cost ~5.5 ms/prefill.
        self.sdpa_prefill_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Explicit SDPA program config (matches tt_transformers for seq<2048).
        # q_chunk/k_chunk=64 covers all our prefill buckets (32/64/96/128/192/256).
        self.sdpa_prefill_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=64,
            k_chunk_size=64,
        )

        # RoPE (P3): default kernel was HiFi4; LoFi + explicit L1 matches linears and avoids DRAM spill.
        self.rope_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.short_seq_limit = 32
        _grid = device.compute_with_storage_grid_size()
        _fp32_linear = self.compute_kernel_config.fp32_dest_acc_en
        self._decode_wqkv_progcfg = make_linear_1d_program_config(
            m=1,
            k=hidden_size,
            n=_fused_qkv,
            grid_x=_grid.x,
            grid_y=_grid.y,
            fp32_dest_acc_en=_fp32_linear,
        )
        self._short_seq_wqkv_progcfg = make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=hidden_size,
            n=_fused_qkv,
            grid_x=_grid.x,
            grid_y=_grid.y,
            fp32_dest_acc_en=_fp32_linear,
        )
        self._decode_wo_progcfg = make_linear_1d_program_config(
            m=1,
            k=_o_rows,
            n=hidden_size,
            grid_x=_grid.x,
            grid_y=_grid.y,
            fp32_dest_acc_en=_fp32_linear,
        )
        self._short_seq_wo_progcfg = make_linear_1d_program_config(
            m=self.short_seq_limit,
            k=_o_rows,
            n=hidden_size,
            grid_x=_grid.x,
            grid_y=_grid.y,
            fp32_dest_acc_en=_fp32_linear,
        )

        # === Decode-only DRAM-sharded QKV / O projections ===
        # qkv_2d rows are reordered into KV-group-interleaved layout (shard i =
        # [q_{2i}, q_{2i+1}, k_i, v_i]) so the sharded nlp_create_qkv_heads kernel
        # can split the matmul output without an intermediate L1 copy.
        self._fused_qkv = _fused_qkv
        num_q_per_kv = num_heads // num_kv_heads

        # Helper: pick num_cores so in0_block_w (= k_tiles / num_cores) ≥ 2.
        # Halves the DRAM-sharded inner-loop iteration count (and per-iter overhead).
        # Reduces MLP/QKV/O decode matmul time by ~30% on K=2048 shapes.
        def _pick_grid_block_w2(k_tiles, n_tiles, max_rows=10, max_cols=13):
            max_cores = max_rows * max_cols
            cands = [
                c for c in range(1, max_cores + 1) if k_tiles % c == 0 and n_tiles % c == 0 and (k_tiles // c) >= 2
            ]
            if not cands:
                return find_grid_k_n(k_tiles, n_tiles)
            cands.sort(reverse=True)
            for cores in cands:
                for rows in range(1, max_rows + 1):
                    if cores % rows == 0 and (cores // rows) <= max_cols:
                        return rows, cores // rows
            return find_grid_k_n(k_tiles, n_tiles)

        row_perm = []
        for i in range(num_kv_heads):
            for q_in_group in range(num_q_per_kv):
                q_head_idx = i * num_q_per_kv + q_in_group
                row_perm.extend(range(q_head_idx * head_dim, (q_head_idx + 1) * head_dim))
            k_off = num_heads * head_dim
            row_perm.extend(range(k_off + i * head_dim, k_off + (i + 1) * head_dim))
            v_off = (num_heads + num_kv_heads) * head_dim
            row_perm.extend(range(v_off + i * head_dim, v_off + (i + 1) * head_dim))
        qkv_2d_kvgi = qkv_2d.index_select(0, torch.tensor(row_perm, dtype=torch.long)).contiguous()
        wqkv_kn = qkv_2d_kvgi.transpose(-2, -1).contiguous()
        self.wqkv_dram_sharded, k_q, n_padded_q = build_dram_sharded_weight(wqkv_kn, device, dtype=weight_dtype)
        self._decode_wqkv_n_padded = n_padded_q
        k_tiles_q, n_tiles_q = k_q // 32, n_padded_q // 32
        rows_q, cols_q = _pick_grid_block_w2(k_tiles_q, n_tiles_q)
        self._decode_wqkv_dramshard_progcfg = dram_sharded_program_config(
            m=32, k=k_q, n=n_padded_q, num_cores=rows_q * cols_q
        )
        self._decode_wqkv_in0_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=k_tiles_q, num_cores_x=cols_q, num_cores_y=rows_q
        )
        self._decode_wqkv_out_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=n_tiles_q, num_cores_x=cols_q, num_cores_y=rows_q
        )

        wo_kn = o_proj_weight.transpose(-2, -1).contiguous()
        self.wo_dram_sharded, k_o, n_padded_o = build_dram_sharded_weight(wo_kn, device, dtype=weight_dtype)
        self._decode_wo_n_padded = n_padded_o
        k_tiles_o, n_tiles_o = k_o // 32, n_padded_o // 32
        rows_o, cols_o = _pick_grid_block_w2(k_tiles_o, n_tiles_o)
        self._decode_wo_dramshard_progcfg = dram_sharded_program_config(
            m=32, k=k_o, n=n_padded_o, num_cores=rows_o * cols_o
        )
        self._decode_wo_in0_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=k_tiles_o, num_cores_x=cols_o, num_cores_y=rows_o
        )
        self._decode_wo_out_memcfg = width_sharded_l1_memcfg(
            m_tiles=1, k_tiles=n_tiles_o, num_cores_x=cols_o, num_cores_y=rows_o
        )

        # === Prefill bucket=128 — width-sharded IN0 + 1D-mcast (mcast_in0=True) ===
        # Reuses the same in0_block_w=2 trick from decode (halve num_cores, double
        # per-core K shard, fewer inner-loop iterations). For QKV: 32 cores grid,
        # IN0 shard [4 M-tiles, 2 K-tiles] per core, per_core_M=4, per_core_N=N_tiles/32.
        _pf128_num_cores = 32
        if k_tiles_q % _pf128_num_cores == 0 and n_tiles_q % _pf128_num_cores == 0:
            _pf128_grid_x, _pf128_grid_y = 8, 4
            self._prefill128_wqkv_in0_memcfg = width_sharded_l1_memcfg(
                m_tiles=128 // 32, k_tiles=k_tiles_q, num_cores_x=_pf128_grid_x, num_cores_y=_pf128_grid_y
            )
            self._prefill128_wqkv_out_memcfg = width_sharded_l1_memcfg(
                m_tiles=128 // 32, k_tiles=n_tiles_q, num_cores_x=_pf128_grid_x, num_cores_y=_pf128_grid_y
            )
            _pf_in0_block_w = k_tiles_q // _pf128_num_cores  # 2
            _pf_per_core_N = n_tiles_q // _pf128_num_cores
            _pf_per_core_M = 128 // 32  # 4
            _sb_lim = 4 if _fp32_linear else 8
            _pf_sbw = max(i for i in range(1, _sb_lim + 1) if _pf_per_core_N % i == 0)
            _pf_sbh = max(i for i in range(1, _sb_lim + 1) if _pf_per_core_M % i == 0 and i * _pf_sbw <= _sb_lim)
            self._prefill128_wqkv_progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(_pf128_grid_x, _pf128_grid_y),
                in0_block_w=_pf_in0_block_w,
                out_subblock_h=_pf_sbh,
                out_subblock_w=_pf_sbw,
                per_core_M=_pf_per_core_M,
                per_core_N=_pf_per_core_N,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        else:
            self._prefill128_wqkv_progcfg = None

        # Same for O proj
        if k_tiles_o % _pf128_num_cores == 0 and n_tiles_o % _pf128_num_cores == 0:
            self._prefill128_wo_in0_memcfg = width_sharded_l1_memcfg(
                m_tiles=128 // 32, k_tiles=k_tiles_o, num_cores_x=_pf128_grid_x, num_cores_y=_pf128_grid_y
            )
            self._prefill128_wo_out_memcfg = width_sharded_l1_memcfg(
                m_tiles=128 // 32, k_tiles=n_tiles_o, num_cores_x=_pf128_grid_x, num_cores_y=_pf128_grid_y
            )
            _po_in0_block_w = k_tiles_o // _pf128_num_cores
            _po_per_core_N = n_tiles_o // _pf128_num_cores
            _po_sbw = max(i for i in range(1, _sb_lim + 1) if _po_per_core_N % i == 0)
            _po_sbh = max(i for i in range(1, _sb_lim + 1) if _pf_per_core_M % i == 0 and i * _po_sbw <= _sb_lim)
            self._prefill128_wo_progcfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(_pf128_grid_x, _pf128_grid_y),
                in0_block_w=_po_in0_block_w,
                out_subblock_h=_po_sbh,
                out_subblock_w=_po_sbw,
                per_core_M=_pf_per_core_M,
                per_core_N=_po_per_core_N,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
        else:
            self._prefill128_wo_progcfg = None

        # === Sharded NLP head op memcfgs (decode m=32 + prefill m=128) ===
        # nlp_concat_heads HEIGHT_SHARDED input over num_heads cores (1 head/shard).
        # nlp_create_qkv_heads WIDTH_SHARDED input with shard_width=(Q/KV+2)*head_dim
        # over fused_qkv/shard_width cores (= num_kv_heads).
        _compute_grid = device.compute_with_storage_grid_size()
        concat_grid = ttnn.num_cores_to_corerangeset(num_heads, _compute_grid, True)
        qkv_shard_width = (num_q_per_kv + 2) * head_dim
        qkv_num_cores = _fused_qkv // qkv_shard_width
        assert _fused_qkv % qkv_shard_width == 0, "fused_qkv must divide qkv_shard_width"
        qkv_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(qkv_num_cores - 1, 0))})
        q_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_heads - 1, 0))})
        kv_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_kv_heads - 1, 0))})

        def _build_sharded_nlp_memcfgs(m: int):
            return {
                "concat_in": ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(concat_grid, (m, head_dim), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                "concat_out": ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(concat_grid, (m, head_dim), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                "qkv_in": ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(qkv_grid, (m, qkv_shard_width), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                "q_out": ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(q_grid, (m, head_dim), ttnn.ShardOrientation.ROW_MAJOR),
                ),
                "k_out": ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(kv_grid, (m, head_dim), ttnn.ShardOrientation.ROW_MAJOR),
                ),
            }

        _dec = _build_sharded_nlp_memcfgs(32)
        self._decode_concat_heads_in_memcfg = _dec["concat_in"]
        self._decode_concat_heads_out_memcfg = _dec["concat_out"]
        self._decode_qkv_split_in_memcfg = _dec["qkv_in"]
        self._decode_qkv_split_q_out_memcfg = _dec["q_out"]
        self._decode_qkv_split_k_out_memcfg = _dec["k_out"]
        self._decode_qkv_split_v_out_memcfg = self._decode_qkv_split_k_out_memcfg

        # Per-bucket prefill sharded NLPConcat memcfgs.
        self._prefill_concat_configs = {m: _build_sharded_nlp_memcfgs(m) for m in _PREFILL_SEQS}

        # Pre-compute HEIGHT_SHARDED memory configs for paged_update_cache inputs.
        # paged_update_cache requires input in [1, batch, kv_heads, head_dim] HEIGHT_SHARDED on batch cores.
        # For batch=1: tensor [1, 1, num_kv_heads, head_dim] padded to [1, 1, tile(kv_heads), head_dim].
        # Physical height = tile(kv_heads), width = head_dim, 1 shard on 1 core.
        # The fused variant `paged_fused_update_cache` writes K and V together but
        # requires their input shards to live on DISJOINT cores (it parallelizes across
        # them). We give K core (0,0) and V core (1,0) so the fused kernel can run.
        TILE = 32
        kv_shard_height = ((num_kv_heads + TILE - 1) // TILE) * TILE  # ceil to tile: 8→32
        _k_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
        _v_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))})
        self.paged_k_input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_k_grid, [kv_shard_height, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.paged_v_input_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(_v_grid, [kv_shard_height, head_dim], ttnn.ShardOrientation.ROW_MAJOR),
        )
        # Backwards-compat alias for any callers still expecting the merged name.
        self.paged_input_mem_config = self.paged_k_input_mem_config

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        transformation_mat: ttnn.Tensor,
        attention_mask: ttnn.Tensor = None,
        kv_cache: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        start_pos: int = 0,
        mode: str = "prefill",
        cur_pos_tensor: Optional[ttnn.Tensor] = None,
        decode_attn_mask: Optional[ttnn.Tensor] = None,
        cp_prefill_mask: Optional[ttnn.Tensor] = None,
        prefill_attn_mask: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Apply multi-head attention with QK-norm and RoPE.

        Args:
            x: Input tensor [batch, 1, seq_len, hidden_size]
            cos, sin: RoPE frequencies [1, 1, seq_len, head_dim]
            transformation_mat: TTNN RoPE transformation matrix
            attention_mask: Optional attention mask
            kv_cache: (k_cache, v_cache) each [batch, num_kv_heads, max_seq, head_dim]
            start_pos: KV cache write position (decode, used only when cur_pos_tensor is None)
            mode: "prefill" or "decode"
            cur_pos_tensor: Optional int32 device tensor [1] for trace-compatible decode.
                When provided, uses paged_update_cache and attends over full cache.
            decode_attn_mask: Optional float32 device tensor [1,1,1,max_seq] for decode.
                Pre-allocated; caller updates it each step (0 for valid, -inf for future).
            cp_prefill_mask: Optional float32 device tensor [1,1,seq,max_seq] for trace-
                compatible CP prefill. When provided, writes K/V at positions 0 and 1
                using update_cache (constant scalars, trace-safe) and attends over the
                full cache masked by this tensor.
            prefill_attn_mask: Optional float32 device tensor [1,heads,padded_seq,max_seq]
                for trace-compatible Talker prefill. When provided, writes the full K/V
                sequence to cache at position 0 using update_cache (trace-safe) and
                attends over the full cache masked by this tensor. The mask encodes
                both causal constraints and padding.

        Returns:
            (output [batch, 1, seq_len, hidden_size], updated_kv_cache)
        """
        batch_size = x.shape[0]
        is_decode = mode == "decode"
        seq_len = x.shape[2]
        if is_decode or seq_len == 1:
            wqkv_progcfg = self._decode_wqkv_progcfg
            wo_progcfg = self._decode_wo_progcfg
        elif seq_len <= self.short_seq_limit:
            wqkv_progcfg = self._short_seq_wqkv_progcfg
            wo_progcfg = self._short_seq_wo_progcfg
        else:
            wqkv_progcfg = wo_progcfg = None

        # QKV projection — DRAM-sharded matmul path. Originally gated to
        # decode (seq=1); relaxed to all seq_len <= 32 (one tile in M) so
        # CP_prefill (seq=2) also takes the sharded fast path → 16-core
        # nlp_create_qkv_heads instead of single-core. Larger prefill buckets
        # (64, 128) need separate per-m shard configs to engage — TODO.
        use_dram_shard_qkv = seq_len <= 32
        use_prefill128_qkv = (not is_decode) and (seq_len == 128) and (self._prefill128_wqkv_progcfg is not None)
        # Sharded nlp_create_qkv_heads engages downstream of the DRAM-sharded QKV
        # since wqkv was rearranged to KV-group-interleaved layout.
        sharded_qkv_split = use_dram_shard_qkv
        if use_prefill128_qkv:
            x_sharded = ttnn.to_memory_config(x, self._prefill128_wqkv_in0_memcfg)
            xqkv_sharded = ttnn.linear(
                x_sharded,
                self.wqkv,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self._prefill128_wqkv_progcfg,
                memory_config=self._prefill128_wqkv_out_memcfg,
            )
            ttnn.deallocate(x_sharded)
            xqkv = ttnn.to_memory_config(xqkv_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(xqkv_sharded)
            xqkv_already_sharded_for_split = False
        elif use_dram_shard_qkv:
            # Skip the I→S if x is already in the matching width-sharded layout
            # (e.g. piped through from a sharded layernorm in decoder_layer).
            if x.memory_config() == self._decode_wqkv_in0_memcfg:
                x_sharded = x
                _own_x_sharded = False
            else:
                x_sharded = ttnn.to_memory_config(x, self._decode_wqkv_in0_memcfg)
                _own_x_sharded = True
            xqkv_sharded = ttnn.linear(
                x_sharded,
                self.wqkv_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self._decode_wqkv_dramshard_progcfg,
                memory_config=self._decode_wqkv_out_memcfg,
            )
            if _own_x_sharded:
                ttnn.deallocate(x_sharded)
            if sharded_qkv_split and self._decode_wqkv_n_padded == self._fused_qkv:
                # Direct sharded→sharded reshard (64-core matmul out → 8-core nlp_create in).
                # Skips the L1_INTERLEAVED intermediate (S→I + I→S = 2 ops, ~3.3 µs).
                xqkv = ttnn.to_memory_config(xqkv_sharded, self._decode_qkv_split_in_memcfg)
                ttnn.deallocate(xqkv_sharded)
                xqkv_already_sharded_for_split = True
            else:
                xqkv_padded = ttnn.to_memory_config(xqkv_sharded, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(xqkv_sharded)
                if self._decode_wqkv_n_padded != self._fused_qkv:
                    xqkv = ttnn.slice(
                        xqkv_padded,
                        [0, 0, 0, 0],
                        [
                            xqkv_padded.shape[0],
                            xqkv_padded.shape[1],
                            xqkv_padded.shape[2],
                            self._fused_qkv,
                        ],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    ttnn.deallocate(xqkv_padded)
                else:
                    xqkv = xqkv_padded
                xqkv_already_sharded_for_split = False
        else:
            xqkv = ttnn.linear(
                x,
                self.wqkv,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=wqkv_progcfg,
            )
            xqkv_already_sharded_for_split = False

        # Split: Q [b, num_heads, seq, head_dim], K/V [b, num_kv_heads, seq, head_dim]
        if sharded_qkv_split:
            # xqkv may already be in the 8-core split layout (no extra reshard needed).
            if xqkv_already_sharded_for_split:
                xqkv_for_split = xqkv
                _own_xqkv_for_split = False
            else:
                xqkv_for_split = ttnn.to_memory_config(xqkv, self._decode_qkv_split_in_memcfg)
                ttnn.deallocate(xqkv)
                _own_xqkv_for_split = True
            q_sharded, k_sharded, v_sharded = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_for_split,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                transpose_k_heads=False,
                memory_config=self._decode_qkv_split_q_out_memcfg,
            )
            if _own_xqkv_for_split:
                ttnn.deallocate(xqkv_for_split)
            else:
                ttnn.deallocate(xqkv_for_split)
            # q_norm / k_norm / rotary_embedding_llama expect L1_INTERLEAVED.
            q = ttnn.to_memory_config(q_sharded, ttnn.L1_MEMORY_CONFIG)
            k = ttnn.to_memory_config(k_sharded, ttnn.L1_MEMORY_CONFIG)
            v = ttnn.to_memory_config(v_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q_sharded)
            ttnn.deallocate(k_sharded)
            ttnn.deallocate(v_sharded)
        else:
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                xqkv,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(xqkv)

        # QK-norm (per-head RMSNorm to stabilize attention with large logit scales)
        q = ttnn.rms_norm(
            q,
            epsilon=self.rms_norm_eps,
            weight=self.q_norm_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k = ttnn.rms_norm(
            k,
            epsilon=self.rms_norm_eps,
            weight=self.k_norm_weight,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        if q.dtype != ttnn.bfloat16:
            q = ttnn.typecast(q, dtype=ttnn.bfloat16)
        if k.dtype != ttnn.bfloat16:
            k = ttnn.typecast(k, dtype=ttnn.bfloat16)

        # RoPE:
        # Q/K projection rows + Q/K norm weights were pre-permuted at init time so
        # Q/K are already in interleaved head-dim layout for rotary_embedding_llama.
        # This avoids per-call reshape/permute/reshape churn in decode/prefill.
        # Use is_decode_mode=False to work with DRAM layout for all sequence lengths.
        q = ttnn.experimental.rotary_embedding_llama(
            q,
            cos,
            sin,
            transformation_mat,
            is_decode_mode=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.rope_compute_kernel_config,
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k,
            cos,
            sin,
            transformation_mat,
            is_decode_mode=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.rope_compute_kernel_config,
        )
        # Keep Q/K in interleaved layout after RoPE.

        k_seq = k.shape[2]

        # Default: k/v are temporary tensors owned by this call, safe to free after typecast.
        # The trace-compatible decode path overrides k_for_attn/v_for_attn to alias k_cache
        # and sets k_is_cache_alias=True to prevent accidental deallocation.
        k_for_attn = k
        v_for_attn = v
        k_is_cache_alias = False

        # KV cache: store bfloat16, read for attention (then typecast to float32 below)
        updated_kv_cache = None
        if kv_cache is not None:
            k_cache, v_cache = kv_cache

            if is_decode:
                if cur_pos_tensor is not None:
                    # Trace-compatible path: paged_fused_update_cache uses a device tensor for position.
                    # Reshape K/V from [batch, kv_heads, 1, dim] → [1, batch, kv_heads, dim]
                    # for paged_update_cache's expected input format.
                    # Have transpose write straight into the HEIGHT_SHARDED layout that
                    # paged_update_cache requires; eliminates a separate I→S per K/V.
                    # K and V land on different cores (paged_fused_update_cache parallelizes
                    # across them and rejects overlapping shard grids).
                    k_paged_hs = ttnn.transpose(k, 1, 2, memory_config=self.paged_k_input_mem_config)
                    v_paged_hs = ttnn.transpose(v, 1, 2, memory_config=self.paged_v_input_mem_config)
                    ttnn.deallocate(k)
                    ttnn.deallocate(v)
                    # Typecast K/V to cache dtype if cache is higher precision (e.g. fp32).
                    # paged_fused_update_cache requires matching dtypes between input and cache.
                    if k_paged_hs.dtype != k_cache.dtype:
                        k_paged_typed = ttnn.typecast(k_paged_hs, dtype=k_cache.dtype)
                        ttnn.deallocate(k_paged_hs)
                        k_paged_hs = k_paged_typed
                    if v_paged_hs.dtype != v_cache.dtype:
                        v_paged_typed = ttnn.typecast(v_paged_hs, dtype=v_cache.dtype)
                        ttnn.deallocate(v_paged_hs)
                        v_paged_hs = v_paged_typed
                    # Write K + V to cache at cur_pos_tensor position. The fused variant
                    # writes both in one kernel launch — saves 1× dispatch overhead per
                    # layer compared to two separate paged_update_cache calls. Same op
                    # signature; trace-compatible.
                    ttnn.experimental.paged_fused_update_cache(
                        k_cache, k_paged_hs, v_cache, v_paged_hs, update_idxs_tensor=cur_pos_tensor
                    )
                    ttnn.deallocate(k_paged_hs)
                    ttnn.deallocate(v_paged_hs)
                    # Read FULL cache for attention — fixed shape, trace-compatible.
                    # k_for_attn = k_cache is an ALIAS (not a copy). The code below
                    # must NOT deallocate it; k_is_cache_alias=True guards that.
                    k_for_attn = k_cache  # [batch, kv_heads, max_seq, dim]
                    v_for_attn = v_cache
                    k_is_cache_alias = True
                    k_seq = k_cache.shape[2]  # max_seq (constant)
                else:
                    # update_cache with Python scalar position (baked into trace as constant).
                    # Use this path when start_pos is a fixed int (e.g. 13 separate CP decode
                    # traces, one per position 2..14 — each bakes its own constant position).
                    ttnn.update_cache(k_cache, k, update_idx=start_pos)
                    ttnn.update_cache(v_cache, v, update_idx=start_pos)
                    ttnn.deallocate(k)
                    ttnn.deallocate(v)
                    if decode_attn_mask is not None:
                        # Trace-compatible full-cache attention: decode_attn_mask masks future
                        # positions.  k_seq = max_seq is a constant — trace-safe.
                        k_for_attn = k_cache
                        v_for_attn = v_cache
                        k_is_cache_alias = True
                        k_seq = k_cache.shape[2]
                    else:
                        # Non-trace path: slice cache to the valid prefix only.
                        cache_len = start_pos + 1
                        k_for_attn = ttnn.slice(
                            k_cache, [0, 0, 0, 0], [batch_size, self.num_kv_heads, cache_len, self.head_dim]
                        )
                        v_for_attn = ttnn.slice(
                            v_cache, [0, 0, 0, 0], [batch_size, self.num_kv_heads, cache_len, self.head_dim]
                        )
                        k_is_cache_alias = False
                        k_seq = cache_len
            elif prefill_attn_mask is not None:
                # Trace-compatible Talker prefill: write full padded K/V sequence to
                # cache at position 0. update_cache with update_idx=0 is a Python
                # constant baked into the trace — trace-safe.
                # k shape: [batch, kv_heads, padded_seq_len, head_dim]
                ttnn.update_cache(k_cache, k, update_idx=0)
                ttnn.update_cache(v_cache, v, update_idx=0)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                # Full-cache attention with prefill_attn_mask handles both causal
                # masking and padding: real positions only attend to prior real
                # positions; padding + empty cache positions are masked to -inf.
                k_for_attn = k_cache
                v_for_attn = v_cache
                k_is_cache_alias = True
                k_seq = k_cache.shape[2]
            elif cp_prefill_mask is not None:
                # Trace-compatible CP prefill: write 2 K/V positions to cache at batch=0.
                # ttnn.fill_cache(cache, input, batch_idx) writes input.shape[2] positions
                # in one kernel launch — replaces the prior 4-slice + 4-update pattern.
                # batch_idx is a Python constant captured in the trace → trace-safe.
                ttnn.fill_cache(k_cache, k, 0)
                ttnn.fill_cache(v_cache, v, 0)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                # Read full cache for attention — fixed shape, trace-compatible.
                # Positions 2..max_seq-1 may contain stale data from prior frames,
                # but cp_prefill_mask masks those positions to -inf.
                k_for_attn = k_cache
                v_for_attn = v_cache
                k_is_cache_alias = True
                k_seq = k_cache.shape[2]
            else:
                # Standard prefill: write the fresh K/V into the persistent KV cache
                # in-place via ttnn.fill_cache (matches tt_transformers prefill). The
                # cache buffer was allocated once at init (allocate_kv_cache) at shape
                # [B, n_kv, max_seq, D]; fill_cache writes positions [0:k_seq] without
                # reallocating. Attention itself reads from the freshly projected k/v
                # (small, fits L1) — the cache is purely a write-through for later decode.
                # Typecast K/V to cache dtype only for the cache write (do not mutate
                # the in-flight bf16 K/V used by the prefill SDPA below).
                k_to_write = k if k.dtype == k_cache.dtype else ttnn.typecast(k, dtype=k_cache.dtype)
                v_to_write = v if v.dtype == v_cache.dtype else ttnn.typecast(v, dtype=v_cache.dtype)
                ttnn.fill_cache(k_cache, k_to_write, 0)
                ttnn.fill_cache(v_cache, v_to_write, 0)
                if k_to_write is not k:
                    ttnn.deallocate(k_to_write)
                if v_to_write is not v:
                    ttnn.deallocate(v_to_write)

            updated_kv_cache = (k_cache, v_cache)

        # ─── Fused prefill SDPA path ──────────────────────────────────────────
        # ttnn.transformer.scaled_dot_product_attention handles GQA natively
        # (k/v keep num_kv_heads, no repeat_interleave) and fuses scale + mask +
        # softmax + matmul. Replaces the ~30-line manual fp32 chain below for
        # standard causal prefill (Talker prefill). bf16 inputs + fp32 dest acc
        # + HiFi4 multiply preserves enough precision through k_norm's K
        # amplification.
        # Falls back to manual chain for: decode, cp_prefill_mask, custom
        # prefill_attn_mask — those need explicit handling.
        _q_seq = int(q.shape[2])
        _k_seq_inner = int(k_for_attn.shape[2])
        _use_fused_prefill_sdpa = (
            not is_decode
            and decode_attn_mask is None
            and cp_prefill_mask is None
            and prefill_attn_mask is None
            and _q_seq == _k_seq_inner
            and _q_seq > 1
        )
        if _use_fused_prefill_sdpa:
            attn_output = ttnn.transformer.scaled_dot_product_attention(
                q,
                k_for_attn,
                v_for_attn,
                is_causal=True,
                scale=self.scale,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_config,
                program_config=self.sdpa_prefill_program_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            if not k_is_cache_alias:
                ttnn.deallocate(k_for_attn)
                ttnn.deallocate(v_for_attn)
            # Skip the manual fp32 SDPA chain below; jump to o_proj path.
            attn_output_pre_oproj = attn_output  # already bf16, [B, num_heads, S, D]

        # ─── Manual fp32 SDPA path (decode + special-mask prefill) ────────────
        # Typecast Q/K/V to float32 for precise attention.
        # k_norm gamma up to 68 amplifies K to ~260; bfloat16 SDPA loses enough
        # precision to cause completely wrong token predictions (no EOS, model loops).
        if not _use_fused_prefill_sdpa:
            q_f32 = ttnn.typecast(q, dtype=ttnn.float32)
            ttnn.deallocate(q)

            # GQA expansion: replicate each KV head num_kv_groups times.
            # Order: repeat_interleave on bf16 first (half the bandwidth of fp32),
            # then typecast the expanded tensor. Same math as cast-then-expand but
            # the layout-bound repeat_interleave moves 2-byte bf16 instead of
            # 4-byte fp32 elements.
            if self.num_kv_groups > 1:
                k_exp_bf16 = ttnn.repeat_interleave(k_for_attn, self.num_kv_groups, dim=1)
                v_exp_bf16 = ttnn.repeat_interleave(v_for_attn, self.num_kv_groups, dim=1)
                if not k_is_cache_alias:
                    ttnn.deallocate(k_for_attn)
                    ttnn.deallocate(v_for_attn)
                k_exp = ttnn.typecast(k_exp_bf16, dtype=ttnn.float32)
                v_exp = ttnn.typecast(v_exp_bf16, dtype=ttnn.float32)
                ttnn.deallocate(k_exp_bf16)
                ttnn.deallocate(v_exp_bf16)
            else:
                k_exp = ttnn.typecast(k_for_attn, dtype=ttnn.float32)
                v_exp = ttnn.typecast(v_for_attn, dtype=ttnn.float32)
                if not k_is_cache_alias:
                    ttnn.deallocate(k_for_attn)
                    ttnn.deallocate(v_for_attn)

        if not _use_fused_prefill_sdpa:
            # Float32 scaled dot-product attention via ttnn.matmul + ttnn.softmax
            q_seq = q_f32.shape[2]
            scores = ttnn.matmul(
                q_f32,
                k_exp,
                transpose_b=True,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q_f32)
            scores = ttnn.mul(scores, self.scale, memory_config=ttnn.L1_MEMORY_CONFIG)

            if decode_attn_mask is not None:
                scores = ttnn.add(scores, decode_attn_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif cp_prefill_mask is not None:
                scores = ttnn.add(scores, cp_prefill_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif prefill_attn_mask is not None:
                scores = ttnn.add(scores, prefill_attn_mask, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif q_seq == k_seq and q_seq > 1:
                mask_cpu = torch.triu(
                    torch.full((q_seq, k_seq), float("-inf"), dtype=torch.float32),
                    diagonal=1,
                ).reshape(1, 1, q_seq, k_seq)
                mask_tt = ttnn.from_torch(
                    mask_cpu,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                scores = ttnn.add(scores, mask_tt, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(mask_tt)

            attn_weights = ttnn.softmax(scores, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(scores)

            attn_output_f32 = ttnn.matmul(
                attn_weights,
                v_exp,
                compute_kernel_config=self.sdpa_compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_weights)
            ttnn.deallocate(v_exp)

            # Cast back to bfloat16 for output projection.
            attn_output = ttnn.typecast(attn_output_f32, dtype=ttnn.bfloat16)
            ttnn.deallocate(attn_output_f32)

        # Hoist use_dram_shard_o so it can also gate the direct concat→wo reshard below.
        use_dram_shard_o = is_decode and seq_len == 1
        use_prefill128_o = (
            (not is_decode) and (seq_len == 128) and getattr(self, "_prefill128_wo_progcfg", None) is not None
        )
        # Pick decode (m=32) vs prefill bucket-size sharded NLPConcat memcfgs.
        sharded_concat_decode = is_decode
        sharded_concat_prefill = not is_decode and seq_len in self._prefill_concat_configs
        use_sharded_concat = sharded_concat_decode or sharded_concat_prefill
        if sharded_concat_prefill:
            _pre = self._prefill_concat_configs[seq_len]
            concat_in_memcfg = _pre["concat_in"]
            concat_out_memcfg = _pre["concat_out"]
        else:
            concat_in_memcfg = self._decode_concat_heads_in_memcfg
            concat_out_memcfg = self._decode_concat_heads_out_memcfg
        # Reshape: [b, num_heads, seq, head_dim] → [b, 1, seq, hidden_size]
        if use_sharded_concat:
            if attn_output.memory_config() == concat_in_memcfg:
                attn_sharded = attn_output
                _own_attn_sharded_pre = False
            else:
                attn_sharded = ttnn.to_memory_config(attn_output, concat_in_memcfg)
                ttnn.deallocate(attn_output)
                _own_attn_sharded_pre = True
            attn_concat_sharded = ttnn.experimental.nlp_concat_heads(attn_sharded, memory_config=concat_out_memcfg)
            if _own_attn_sharded_pre:
                ttnn.deallocate(attn_sharded)
            # Direct sharded→sharded reshard 16c → 64c into wo's expected in0 layout,
            # skipping the L1_INTERLEAVED intermediate (saves S→I + I→S = 2 ops).
            # Only applicable in decode where wo is DRAM-sharded; prefill goes back to L1.
            if sharded_concat_decode and use_dram_shard_o and self._decode_wo_n_padded == self.hidden_size:
                attn_output = ttnn.to_memory_config(attn_concat_sharded, self._decode_wo_in0_memcfg)
                ttnn.deallocate(attn_concat_sharded)
                _attn_already_in_wo_in0 = True
            else:
                attn_output = ttnn.to_memory_config(attn_concat_sharded, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(attn_concat_sharded)
                _attn_already_in_wo_in0 = False
        else:
            attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
            _attn_already_in_wo_in0 = False

        if use_prefill128_o:
            attn_sharded = ttnn.to_memory_config(attn_output, self._prefill128_wo_in0_memcfg)
            ttnn.deallocate(attn_output)
            out_sharded = ttnn.linear(
                attn_sharded,
                self.wo,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self._prefill128_wo_progcfg,
                memory_config=self._prefill128_wo_out_memcfg,
            )
            ttnn.deallocate(attn_sharded)
            output = ttnn.to_memory_config(out_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(out_sharded)
        elif use_dram_shard_o:
            if _attn_already_in_wo_in0:
                attn_sharded = attn_output  # already 64-core width-sharded
                _own_attn_sharded = False
            else:
                attn_sharded = ttnn.to_memory_config(attn_output, self._decode_wo_in0_memcfg)
                ttnn.deallocate(attn_output)
                _own_attn_sharded = True
            # When N is unpadded, return the width-sharded matmul output directly so the
            # caller (decoder_layer) can do a sharded residual add — saves a S→I plus an
            # op-dispatch.
            wo_n_unpadded = self._decode_wo_n_padded == self.hidden_size
            out_sharded = ttnn.linear(
                attn_sharded,
                self.wo_dram_sharded,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self._decode_wo_dramshard_progcfg,
                memory_config=self._decode_wo_out_memcfg,
            )
            if _own_attn_sharded:
                ttnn.deallocate(attn_sharded)
            if wo_n_unpadded:
                output = out_sharded  # caller does the S→I via residual add
            else:
                output_padded = ttnn.to_memory_config(out_sharded, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(out_sharded)
                if self._decode_wo_n_padded != self.hidden_size:
                    output = ttnn.slice(
                        output_padded,
                        [0, 0, 0, 0],
                        [
                            output_padded.shape[0],
                            output_padded.shape[1],
                            output_padded.shape[2],
                            self.hidden_size,
                        ],
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    ttnn.deallocate(output_padded)
                else:
                    output = output_padded
        else:
            output = ttnn.linear(
                attn_output,
                self.wo,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=wo_progcfg,
            )
            ttnn.deallocate(attn_output)

        return output, updated_kv_cache
