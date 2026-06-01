# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Mistral4 Multi-head Latent Attention (MLA) — prefill mode.

Architecture (all on device, no PyTorch fallback for compute):

  Q path:
    hidden  →[q_a_proj]→ q_latent (Q_LORA_RANK)
            →[q_a_norm]→
            →[q_b_proj]→ [N_HEADS, HEAD_DIM]
            split → q_nope [N_HEADS, QK_NOPE_HEAD_DIM]
                  + q_rope [N_HEADS, QK_ROPE_HEAD_DIM]
            RoPE(q_rope) → q_rope_rotated
            q = concat(q_nope, q_rope_rotated)

  KV path:
    hidden  →[kv_a_proj]→ kv_combined (KV_LORA_RANK + QK_ROPE_HEAD_DIM)
            split → kv_latent [KV_LORA_RANK]
                  + k_rope_raw [QK_ROPE_HEAD_DIM]
            kv_latent →[kv_a_norm]→
                      →[kv_b_proj]→ [N_HEADS, KV_B_PER_HEAD]
                      split → k_nope [N_HEADS, QK_NOPE_HEAD_DIM]
                            + v      [N_HEADS, V_HEAD_DIM]
            RoPE(k_rope_raw) → k_rope_rotated
            k_rope_expanded = broadcast k_rope_rotated to N_HEADS
            k = concat(k_nope, k_rope_expanded)

  Attention:
    SDPA(q, k, v, is_causal=True)
    concat_heads → [seq, N_HEADS * V_HEAD_DIM]
    →[o_proj]→ hidden

Sharding strategy for 2-device mesh [1, 2] (P300 × 2):
  All attention weights are *replicated* for the initial bring-up.
  Both devices compute identical attention outputs; only device-0
  output is used for logit accuracy, but both are in sync for the
  residual stream used by the MoE layer.
"""

import math
import os

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.mistral_small_4_119b.constants import (
    HEAD_DIM,
    HIDDEN_SIZE,
    KV_A_PROJ_OUT,
    KV_B_PROJ_OUT_PER_HEAD,
    KV_LORA_RANK,
    N_HEADS,
    NORM_EPS,
    Q_LORA_RANK,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    V_HEAD_DIM,
)


def _torch_for_ttnn_upload(w: torch.Tensor, scale_inv: torch.Tensor | None = None) -> torch.Tensor:
    """Convert a weight tensor to bfloat16 for TTNN upload, dequantizing FP8 if needed.

    scale_inv may be scalar () or per-expert [N]; reshaped to broadcast correctly.
    """
    if w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        w = w.to(torch.float32)
        if scale_inv is not None:
            s = scale_inv.to(torch.float32)
            while s.dim() < w.dim():
                s = s.unsqueeze(-1)
            w = w * s
    return w.to(torch.bfloat16).contiguous()


def _load_weight(
    state_dict: dict,
    key: str,
    transpose: bool,
    dtype: ttnn.DataType,
    mesh_device: ttnn.MeshDevice,
    mesh_mapper=None,
    transform_fn=None,
    cache_file_name=None,
    memory_config=None,
) -> ttnn.Tensor:
    """Load a weight from state_dict to TTNN device with optional transpose.
    Automatically dequantizes FP8 weights using the companion weight_scale_inv key."""
    scale_inv = state_dict.get(key.replace(".weight", ".weight_scale_inv"))
    w = _torch_for_ttnn_upload(state_dict[key], scale_inv)
    if transpose:
        w = w.T.contiguous()
    if transform_fn is not None:
        w = transform_fn(w)
    # Ensure 4D for TILE layout
    while w.dim() < 2:
        w = w.unsqueeze(0)
    mapper = mesh_mapper if mesh_mapper is not None else ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
        cache_file_name=cache_file_name,
    )


def _load_norm_weight(
    state_dict: dict,
    key: str,
    dim: int,
    mesh_device: ttnn.MeshDevice,
    cache_file_name=None,
) -> ttnn.Tensor:
    """
    Load RMSNorm ``weight`` for ``ttnn.rms_norm`` with TILE activations.

    ROW_MAJOR gamma must end in tile width (``ttnn.TILE_SIZE``); see
    ``LayerNormDeviceOperation::validate_on_program_cache_miss`` (gamma last dim
    == tile width, volume aligns with input last dim).
    """
    tw = ttnn.TILE_SIZE
    if dim % tw != 0:
        raise ValueError(f"RMS norm hidden dim {dim} must be divisible by ttnn.TILE_SIZE ({tw})")
    w = _torch_for_ttnn_upload(state_dict[key]).reshape(1, 1, dim // tw, tw)
    return ttnn.as_tensor(
        w,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=cache_file_name,
    )


def _deinterleave_q_b_proj(w: torch.Tensor) -> torch.Tensor:
    """
    Convert rope columns of q_b_proj from interleaved [r0,i0,r1,i1,…] to
    half-split [r0,r1,…,i0,i1,…] to match apply_rotary_pos_emb_interleave.

    w: [Q_LORA_RANK, N_HEADS * HEAD_DIM]  (already transposed)
    """
    in_dim = w.shape[0]
    w3 = w.reshape(in_dim, N_HEADS, HEAD_DIM)
    nope = w3[:, :, :QK_NOPE_HEAD_DIM]
    rope = w3[:, :, QK_NOPE_HEAD_DIM:]  # [in, H, rope_dim]
    rope = rope.reshape(in_dim, N_HEADS, QK_ROPE_HEAD_DIM // 2, 2)
    rope = rope.permute(0, 1, 3, 2).contiguous()  # [in, H, 2, rope_dim//2]
    rope = rope.reshape(in_dim, N_HEADS, QK_ROPE_HEAD_DIM)
    return torch.cat([nope, rope], dim=-1).reshape(in_dim, N_HEADS * HEAD_DIM).contiguous()


def _deinterleave_kv_a_proj(w: torch.Tensor) -> torch.Tensor:
    """
    Convert k_rope columns of kv_a_proj from interleaved to half-split.

    w: [HIDDEN_SIZE, KV_A_PROJ_OUT]  (already transposed)
    The last QK_ROPE_HEAD_DIM columns produce k_rope_raw.
    """
    nope_cols = w[:, :KV_LORA_RANK]
    rope_cols = w[:, KV_LORA_RANK:]  # [H, rope_dim]
    rope_cols = rope_cols.reshape(-1, QK_ROPE_HEAD_DIM // 2, 2)
    rope_cols = rope_cols.permute(0, 2, 1).contiguous()  # [H, 2, rope_dim//2]
    rope_cols = rope_cols.reshape(-1, QK_ROPE_HEAD_DIM)
    return torch.cat([nope_cols, rope_cols], dim=-1).contiguous()


def _apply_rope_ttnn(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    seq_len: int,
    n_heads: int,
    rope_dim: int,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """
    Apply rotary position embeddings in pure TTNN (standard half-split variant).

    Args:
        x:       [1, n_heads, seq_len, rope_dim]  — already in interleaved layout
        cos/sin: [1, 1, seq_len, rope_dim]  (broadcast over heads)
        Returns: [1, n_heads, seq_len, rope_dim]
    """
    half = rope_dim // 2
    # rotate_half: concat([-x2, x1]) where x1=x[..,:half], x2=x[..,half:]
    x1 = ttnn.slice(x, [0, 0, 0, 0], [1, n_heads, seq_len, half], memory_config=memory_config)
    x2 = ttnn.slice(x, [0, 0, 0, half], [1, n_heads, seq_len, rope_dim], memory_config=memory_config)
    neg_x2 = ttnn.neg(x2, memory_config=memory_config)
    ttnn.deallocate(x2)
    x_rot = ttnn.concat([neg_x2, x1], dim=-1, memory_config=memory_config)
    ttnn.deallocate(x1)
    ttnn.deallocate(neg_x2)

    out = ttnn.add(
        ttnn.multiply(x, cos, memory_config=memory_config),
        ttnn.multiply(x_rot, sin, memory_config=memory_config),
        memory_config=memory_config,
    )
    ttnn.deallocate(x_rot)
    return out


class TtMistral4Attention(LightweightModule):
    """
    MLA attention for Mistral-Small-4 (prefill + decode, replicated weights).

    All weights are replicated across all mesh devices; both devices
    execute identical computation.  This is intentional for the initial
    bring-up: the dominant memory cost is the expert weights (MoE), not
    the relatively small attention projections.

    KV cache convention (for decode): [1, N_HEADS, max_seq_len, dim] — batch-first.
    Both update_cache_for_token_ and fill_cache_for_user_ validate padded_shape()[0]==1.
    SDPA decode expects Q=[1, B, NH, DH] and K/V=[1, NH, S, DH] in the same batch-first layout.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_prefix: str,
        compute_kernel_config=None,
        cache_dir=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.n_heads = N_HEADS
        self.head_dim = HEAD_DIM
        self.qk_nope_head_dim = QK_NOPE_HEAD_DIM
        self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        self.v_head_dim = V_HEAD_DIM
        self.kv_lora_rank = KV_LORA_RANK
        self.kv_a_proj_out = KV_A_PROJ_OUT
        self.kv_b_per_head = KV_B_PROJ_OUT_PER_HEAD
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.prefill_l1_reshape = os.environ.get("MISTRAL4_ATTN_L1_RESHAPE", "1") == "1"
        # Paged latent KV cache: tokens-per-block. Must be a multiple of TILE_SIZE
        # and of the flash-MLA q/k chunk size (128) so chunk_start_idx alignment holds.
        self.block_size = int(os.environ.get("MISTRAL4_KV_BLOCK_SIZE", "128"))

        if compute_kernel_config is None:
            compute_kernel_config = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
        self.compute_kernel_config = compute_kernel_config
        self.lofi_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # High-fidelity config for the flash-MLA attention ops. The previous
        # (expanded-cache) path called scaled_dot_product_attention with no
        # compute_kernel_config → SDPA's high-fidelity default; matching that here
        # (HiFi4 + fp32 accumulation) preserves fine image detail read during
        # prefill. Attention is a tiny fraction of FLOPs, so the fidelity is cheap.
        self.attn_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        # Flash-MLA decode reduces over a single (latent) KV head, so every core in
        # the grid reduces for that one head. The op caps tree-reduction at 64
        # cores/head (6 rounds); the full device grid yields ~110 cores/head and
        # fails. Bound the decode grid to ≤ 8×8 = 64 cores.
        # k_chunk_size must divide the padded cache length (always a multiple of
        # 32), so use 32 — works for any max_seq_len.
        # Decode runs single-token, batch=1, over ONE latent KV head. flash-MLA
        # decode parallelises by splitting the K-sequence across cores and
        # recombining per-core partial softmaxes — and that cross-core combine is
        # numerically wrong for this MQA-in-latent-space case: PCC vs a fresh
        # prefill collapses from 0.9997 (1 core) to ~0.89 (any multi-core grid),
        # which compounds to garbage over 36 layers. Pinning the decode SDPA to a
        # single core removes the cross-core reduction entirely → exact.
        # NOTE: single-core decode attention is fine at low/moderate context but
        # becomes a bandwidth bottleneck at very long context — revisit if/when the
        # op's multi-core latent reduction is fixed upstream. (env-overridable for
        # experiments.)
        _gx = int(os.environ.get("MLA_GRID_X", "1"))
        _gy = int(os.environ.get("MLA_GRID_Y", "1"))
        _kc = int(os.environ.get("MLA_KCHUNK", "32"))
        self._mla_decode_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(_gx, _gy),
            q_chunk_size=128,
            k_chunk_size=_kc,
            exp_approx_mode=False,
        )

        # 1D-mcast program configs for decode matmuls (M=1 tile for all).
        self._decode_pcs = self._build_decode_program_configs(grid)

        # Save grid for the prefill o_proj PC helper; build lazily and cache by m_tiles.
        self._compute_grid = grid
        self._o_proj_prefill_pc_cache: dict = {}

        # Height-sharded input mem config required by paged_update_cache (the
        # tensor-indexed, trace-compatible latent-cache write used in decode).
        self._kvpe_update_mem_cfg = self._build_kv_update_mem_cfg(grid, KV_LORA_RANK + QK_ROPE_HEAD_DIM)

        p = layer_prefix + "self_attn."
        _cf = (lambda key: str(cache_dir / key)) if cache_dir is not None else (lambda _: None)

        # ── Replicated weights (small bottleneck projections) ──────────────
        # HF stores [out, in]; we transpose → [in, out] for TTNN matmul
        self.q_a_proj = _load_weight(
            state_dict,
            p + "q_a_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat4_b,
            mesh_device=mesh_device,
            cache_file_name=_cf(p + "q_a_proj.weight"),
        )  # [HIDDEN_SIZE, Q_LORA_RANK]

        self.q_a_norm = _load_norm_weight(
            state_dict,
            p + "q_a_layernorm.weight",
            Q_LORA_RANK,
            mesh_device,
            cache_file_name=_cf(p + "q_a_layernorm.weight"),
        )  # [1, 1, Q_LORA_RANK / TILE, TILE]

        # Pre-split q_b_proj into nope/rope sub-weights, used by both prefill and
        # decode. The full q_b_proj is never materialised on device — latent
        # caching only needs the per-head q_nope/q_rope projections.
        _q_b_scale = state_dict.get((p + "q_b_proj.weight").replace(".weight", ".weight_scale_inv"))
        _q_b = _deinterleave_q_b_proj(
            _torch_for_ttnn_upload(state_dict[p + "q_b_proj.weight"], _q_b_scale).T.contiguous()
        )
        _q_b3 = _q_b.reshape(Q_LORA_RANK, N_HEADS, HEAD_DIM)
        _q_nope = _q_b3[:, :, :QK_NOPE_HEAD_DIM].reshape(Q_LORA_RANK, N_HEADS * QK_NOPE_HEAD_DIM).contiguous()
        _q_rope = _q_b3[:, :, QK_NOPE_HEAD_DIM:].reshape(Q_LORA_RANK, N_HEADS * QK_ROPE_HEAD_DIM).contiguous()
        self.q_nope_w = ttnn.as_tensor(
            _q_nope,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cf(p + "q_nope_w"),
        )
        self.q_rope_w = ttnn.as_tensor(
            _q_rope,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cf(p + "q_rope_w"),
        )
        del _q_b, _q_b3, _q_nope, _q_rope

        self.kv_a_proj = _load_weight(
            state_dict,
            p + "kv_a_proj_with_mqa.weight",
            transpose=True,
            dtype=ttnn.bfloat4_b,
            mesh_device=mesh_device,
            transform_fn=_deinterleave_kv_a_proj,
            cache_file_name=_cf(p + "kv_a_proj_with_mqa.weight"),
        )  # [HIDDEN_SIZE, KV_A_PROJ_OUT]

        self.kv_a_norm = _load_norm_weight(
            state_dict,
            p + "kv_a_layernorm.weight",
            KV_LORA_RANK,
            mesh_device,
            cache_file_name=_cf(p + "kv_a_layernorm.weight"),
        )  # [1, 1, KV_LORA_RANK / TILE, TILE]

        # ── MLA absorption weights (latent caching) ────────────────────────
        # We cache the compressed latent (kvpe = [kv_latent ‖ k_rope]) instead of
        # expanding it to full per-head K/V. kv_b_proj is folded into the query
        # and output paths (DeepSeek-MLA "weight absorption"):
        #   q_score[h]   = q_nope[h]    @ wkv_b1[h]   # K side, [QK_NOPE]→[KV_LORA]
        #   v_out[h]     = attn_latent[h] @ wkv_b2[h] # V side, [KV_LORA]→[V_HEAD]
        # wkv_b1/wkv_b2 are the two per-head column-halves of kv_b_proj. The full
        # kv_b_proj is never materialised on device.
        _kv_b_scale = state_dict.get((p + "kv_b_proj.weight").replace(".weight", ".weight_scale_inv"))
        _kv_b = _torch_for_ttnn_upload(state_dict[p + "kv_b_proj.weight"], _kv_b_scale).T.contiguous()
        _kv_b3 = _kv_b.reshape(KV_LORA_RANK, N_HEADS, KV_B_PROJ_OUT_PER_HEAD)  # [latent, head, k_nope|v]
        # wkv_b1: per-head [QK_NOPE_HEAD_DIM, KV_LORA_RANK]  (q_nope @ wkv_b1 → latent score)
        _wkv_b1 = _kv_b3[:, :, :QK_NOPE_HEAD_DIM].permute(1, 2, 0).unsqueeze(0).contiguous()
        # wkv_b2: per-head [KV_LORA_RANK, V_HEAD_DIM]        (attn_latent @ wkv_b2 → v)
        _wkv_b2 = _kv_b3[:, :, QK_NOPE_HEAD_DIM:].permute(1, 0, 2).unsqueeze(0).contiguous()
        self.wkv_b1 = ttnn.as_tensor(
            _wkv_b1,  # [1, N_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cf(p + "wkv_b1"),
        )
        self.wkv_b2 = ttnn.as_tensor(
            _wkv_b2,  # [1, N_HEADS, KV_LORA_RANK, V_HEAD_DIM]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cf(p + "wkv_b2"),
        )
        del _kv_b, _kv_b3, _wkv_b1, _wkv_b2

        self.o_proj = _load_weight(
            state_dict,
            p + "o_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat4_b,
            mesh_device=mesh_device,
            cache_file_name=_cf(p + "o_proj.weight"),
        )  # [N_HEADS * V_HEAD_DIM, HIDDEN_SIZE]

        # DS variant of o_proj for the decode path (M=1 tile).  Same data, DRAM
        # bank-sharded across all banks so each compute core reads one bank at
        # full BW.  Adds ~8 MB DRAM per layer (BFP4).  Sweep on this shape:
        #   K=N=4096, BFP8 weights → DS gx=4 in0_block_w=8 was the winner.
        _dram = mesh_device.dram_grid_size()
        self._num_banks = _dram.x
        _K = N_HEADS * V_HEAD_DIM  # 4096
        _N = HIDDEN_SIZE  # 4096
        _shard_w = math.ceil(math.ceil(_N / self._num_banks) / ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        _o_proj_decode_memcfg = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.DRAM,
            shard_spec=ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(_dram.x - 1, 0))}),
                [_K, _shard_w],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self.o_proj_decode = _load_weight(
            state_dict,
            p + "o_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat4_b,
            mesh_device=mesh_device,
            memory_config=_o_proj_decode_memcfg,
            cache_file_name=_cf(p + "o_proj.weight.ds"),
        )
        self._o_proj_decode_gx = 4  # DS compute-grid width (sweep winner)
        _NT = _N // ttnn.TILE_SIZE  # 128
        self.o_proj_decode_pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=8,
            per_core_M=1,
            per_core_N=_NT // self._o_proj_decode_gx,  # 128/4 = 32
            fused_activation=None,
        )

    # ------------------------------------------------------------------
    def _build_decode_program_configs(self, grid):
        """Pre-build 1D-mcast program configs for each decode matmul."""
        gx, gy = grid.x, grid.y
        num_cores = gx * gy

        def _pc(k_tiles, n_tiles):
            gy_eff = gy
            nc = num_cores
            if n_tiles < nc:
                gy_eff = max(1, n_tiles // gx)
                nc = gx * gy_eff
            per_core_N = max(1, (n_tiles + nc - 1) // nc)
            in0_block_w = 1
            for c in (8, 4, 2):
                if k_tiles % c == 0:
                    in0_block_w = c
                    break
            out_subblock_w = 1
            for c in (4, 2, 1):
                if per_core_N % c == 0:
                    out_subblock_w = c
                    break
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(gx, gy_eff),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=1,
                per_core_N=per_core_N,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

        # Latent caching folds kv_b into the absorption matmuls and the o_proj uses a
        # DRAM-sharded PC, so only the q_a / kv_a / q_nope / q_rope decode projections
        # need a 1D-mcast config here.
        H = HIDDEN_SIZE
        return {
            "q_a": _pc(H // 32, Q_LORA_RANK // 32),  # [4096,1024]: K=128, N=32
            "kv_a": _pc(H // 32, KV_A_PROJ_OUT // 32),  # [4096,320]: K=128, N=10
            "q_nope": _pc(Q_LORA_RANK // 32, N_HEADS * QK_NOPE_HEAD_DIM // 32),  # [1024,2048]: K=32, N=64
            "q_rope": _pc(Q_LORA_RANK // 32, N_HEADS * QK_ROPE_HEAD_DIM // 32),  # [1024,2048]: K=32, N=64
        }

    # ------------------------------------------------------------------
    def _o_proj_prefill_pc(self, m_tiles: int):
        """1D-mcast program config for the prefill o_proj matmul (K=4096, N=4096).

        Without an explicit PC the default falls to ~12 active cores and ~117 GB/s
        DRAM (vs ~240 GB/s achievable with a 64-core 1D mcast). Same approach the
        routed-expert PCs use in mistral4_moe.py; cached by m_tiles so each unique
        seq_len builds the program once.
        """
        cached = self._o_proj_prefill_pc_cache.get(m_tiles)
        if cached is not None:
            return cached
        grid = self._compute_grid
        gx, gy = grid.x, grid.y
        K_TILES = (N_HEADS * V_HEAD_DIM) // 32  # 128
        N_TILES = HIDDEN_SIZE // 32  # 128

        # Find the largest grid (px * py) that exactly divides N_TILES, so every
        # selected core has a full N-tile share.
        best_nc, best_x, best_y = 1, 1, 1
        for py in range(1, gy + 1):
            for px in range(1, gx + 1):
                nc = px * py
                if N_TILES % nc == 0 and nc > best_nc:
                    best_nc, best_x, best_y = nc, px, py
        per_core_N = N_TILES // best_nc
        # in0_block_w: largest divisor of K up to 8.
        in0_block_w = 1
        for c in (8, 4, 2):
            if K_TILES % c == 0:
                in0_block_w = c
                break
        # out_subblock_h * out_subblock_w ≤ 8 (DST capacity, fp32_dest_acc=False).
        out_subblock_w = 1
        for c in (4, 2, 1):
            if per_core_N % c == 0:
                out_subblock_w = c
                break
        out_subblock_h = 1
        for c in (4, 2, 1):
            if m_tiles % c == 0 and c * out_subblock_w <= 8:
                out_subblock_h = c
                break
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(best_x, best_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=m_tiles,
            per_core_N=per_core_N,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        self._o_proj_prefill_pc_cache[m_tiles] = pc
        return pc

    # ------------------------------------------------------------------
    def _build_kv_update_mem_cfg(self, grid, head_width: int):
        """Height-sharded ``[1, 1, N_HEADS, head_width]`` config for paged_update_cache.

        Decode batch is 1 (single token, KV cache replicated across the mesh), so the
        op sees one update index and all ``N_HEADS`` rows land on a single core.  The
        shard width must equal the input's last dim (``head_width``).
        """
        shard_h = ((self.n_heads + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        return ttnn.create_sharded_memory_config(
            shape=(shard_h, head_width),
            core_grid=ttnn.num_cores_to_corerangeset(1, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        kv_cache: tuple = None,
        chunk_start_idx: int = 0,
    ) -> ttnn.Tensor:
        """
        Prefill forward (one chunk of the prompt).

        Args:
            x:        [1, 1, seq_len, HIDDEN_SIZE] replicated on all devices
            cos/sin:  [1, 1, seq_len, QK_ROPE_HEAD_DIM]  (from HF rotary, on device)
            kv_cache: optional (kvpe_cache, page_table) — paged latent cache, filled in-place
            chunk_start_idx: absolute position of this chunk's first token (0 for the
                whole-prompt / first-chunk case). When > 0, the chunk's queries attend
                causally over the full prefix [0, chunk_start_idx+seq_len) in the cache.
        Returns:
            [1, 1, seq_len, HIDDEN_SIZE] replicated on all devices
        """
        seq_len = x.shape[2]
        # L1 intermediates speed up short prefill, but the per-head activations
        # ([1, 1, seq, N_HEADS*dim]) overflow L1 for long prefill (~16k tokens →
        # 64 MB > L1). Fall back to DRAM above a safe length so single-pass prefill
        # scales; short prefill keeps the L1 fast path.
        reshape_mem = (
            ttnn.L1_MEMORY_CONFIG if (self.prefill_l1_reshape and seq_len <= 4096) else ttnn.DRAM_MEMORY_CONFIG
        )

        # ── Q projection (pre-split q_b → per-head q_nope / q_rope) ─────────
        q_latent = ttnn.linear(
            x,
            self.q_a_proj,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, Q_LORA_RANK]
        q_latent = ttnn.rms_norm(q_latent, weight=self.q_a_norm, epsilon=NORM_EPS, memory_config=ttnn.L1_MEMORY_CONFIG)

        q_nope_flat = ttnn.linear(
            q_latent,
            self.q_nope_w,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=reshape_mem,
        )  # [1, 1, seq, N_HEADS * QK_NOPE_HEAD_DIM]
        q_rope_flat = ttnn.linear(
            q_latent,
            self.q_rope_w,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=reshape_mem,
        )  # [1, 1, seq, N_HEADS * QK_ROPE_HEAD_DIM]
        ttnn.deallocate(q_latent)

        q_nope = ttnn.transpose(
            ttnn.reshape(q_nope_flat, [1, seq_len, self.n_heads, self.qk_nope_head_dim]),
            1,
            2,
            memory_config=reshape_mem,
        )  # [1, N_HEADS, seq, QK_NOPE_HEAD_DIM]
        ttnn.deallocate(q_nope_flat)
        q_rope = ttnn.transpose(
            ttnn.reshape(q_rope_flat, [1, seq_len, self.n_heads, self.qk_rope_head_dim]),
            1,
            2,
            memory_config=reshape_mem,
        )  # [1, N_HEADS, seq, QK_ROPE_HEAD_DIM]
        ttnn.deallocate(q_rope_flat)

        q_rope_rotated = ttnn.experimental.rotary_embedding_hf(
            q_rope, cos, sin, is_decode_mode=False, compute_kernel_config=self.lofi_compute_kernel_config
        )
        ttnn.deallocate(q_rope)

        # Absorb the K side of kv_b: q_nope @ wkv_b1 → latent-space query score.
        q_score = ttnn.matmul(
            q_nope,
            self.wkv_b1,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, N_HEADS, seq, KV_LORA_RANK]
        ttnn.deallocate(q_nope)
        q_mla = ttnn.concat([q_score, q_rope_rotated], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_score)
        ttnn.deallocate(q_rope_rotated)
        # q_mla: [1, N_HEADS, seq, KV_LORA_RANK + QK_ROPE_HEAD_DIM]

        # ── KV projection → compressed latent (kvpe), no per-head expansion ─
        kv_combined = ttnn.linear(
            x,
            self.kv_a_proj,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, 1, seq, KV_A_PROJ_OUT]
        kv_latent = ttnn.slice(kv_combined, [0, 0, 0, 0], [1, 1, seq_len, self.kv_lora_rank])
        k_rope_raw = ttnn.slice(kv_combined, [0, 0, 0, self.kv_lora_rank], [1, 1, seq_len, self.kv_a_proj_out])
        ttnn.deallocate(kv_combined)

        kv_latent_normed = ttnn.rms_norm(
            kv_latent, weight=self.kv_a_norm, epsilon=NORM_EPS, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(kv_latent)

        k_rope_rotated = ttnn.experimental.rotary_embedding_hf(
            k_rope_raw, cos, sin, is_decode_mode=False, compute_kernel_config=self.lofi_compute_kernel_config
        )  # [1, 1, seq, QK_ROPE_HEAD_DIM]
        ttnn.deallocate(k_rope_raw)

        kvpe = ttnn.concat([kv_latent_normed, k_rope_rotated], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(kv_latent_normed)
        ttnn.deallocate(k_rope_rotated)
        # kvpe: [1, 1, seq, KV_LORA_RANK + QK_ROPE_HEAD_DIM]  (single shared KV head)

        # ── Write this chunk's latent into the paged cache ─────────────────
        end_block = 0
        if kv_cache is not None:
            kvpe_cache, page_table = kv_cache
            bs = self.block_size
            start_block = chunk_start_idx // bs
            end_block = (chunk_start_idx + seq_len + bs - 1) // bs
            chunk_pt = ttnn.slice(page_table, [0, start_block], [1, end_block])
            ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe, chunk_pt, batch_idx=0)
            ttnn.deallocate(chunk_pt)

        # ── Attention (V = first KV_LORA_RANK dims of the latent) ──────────
        if kv_cache is not None and chunk_start_idx > 0:
            # Later chunk: attend causally over the full prefix [0, chunk_start_idx+seq)
            # using the paged cache. q/k chunk sizes (128) divide block_size (128) and
            # chunk_start_idx, so the alignment constraints hold.
            attend_pt = ttnn.slice(page_table, [0, 0], [1, end_block])
            attn_latent = ttnn.transformer.chunked_flash_mla_prefill(
                q_mla,
                kvpe_cache,
                self.kv_lora_rank,  # head_dim_v
                attend_pt,
                chunk_start_idx,
                scale=self.scale,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.attn_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, N_HEADS, seq, KV_LORA_RANK]
            ttnn.deallocate(attend_pt)
        else:
            # First chunk (or no cache): square causal over the chunk's own latent.
            attn_latent = ttnn.transformer.flash_mla_prefill(
                q_mla,
                kvpe,
                self.kv_lora_rank,  # head_dim_v
                is_causal=True,
                scale=self.scale,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.attn_compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # [1, N_HEADS, seq, KV_LORA_RANK]
        ttnn.deallocate(q_mla)
        ttnn.deallocate(kvpe)

        # Expand the V side of kv_b: attn_latent @ wkv_b2 → per-head value output.
        attn_out = ttnn.matmul(
            attn_latent,
            self.wkv_b2,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, N_HEADS, seq, V_HEAD_DIM]
        ttnn.deallocate(attn_latent)

        # ── Output projection ──────────────────────────────────────────────
        attn_out_t = ttnn.transpose(attn_out, 1, 2, memory_config=reshape_mem)
        ttnn.deallocate(attn_out)
        attn_flat = ttnn.reshape(
            attn_out_t, [1, 1, seq_len, self.n_heads * self.v_head_dim]
        )  # [1, 1, seq, N_HEADS * V_HEAD_DIM = 4096]
        ttnn.deallocate(attn_out_t)

        m_tiles = (seq_len + 31) // 32
        # The custom 1D-mcast o_proj PC assumes a single M-block per core (per_core_M
        # = m_tiles) and fails to build past ~32 M-tiles (seq ≳ 1024). For longer
        # prefill fall back to the default matmul config, which tiles M and scales
        # (slower per the helper's note, but correct at any length).
        o_proj_pc = self._o_proj_prefill_pc(m_tiles) if m_tiles <= 32 else None
        out = ttnn.linear(
            attn_flat,
            self.o_proj,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=o_proj_pc,
        )  # [1, 1, seq, HIDDEN_SIZE]
        ttnn.deallocate(attn_flat)

        return out

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def allocate_kv_cache(self, max_seq_len: int) -> tuple:
        """
        Pre-allocate the PAGED MLA latent cache (zeroed, DRAM, replicated).

        Latent caching: a single ``kvpe`` cache holds the compressed
        ``[kv_latent ‖ k_rope]`` shared across all heads (one "KV head"). It is
        block-paged so prefill can run in bounded chunks (``chunked_flash_mla_prefill``)
        — the key to context beyond the ~32k single-pass limit.

        Returns ``(kvpe_cache, page_table)``:
          - kvpe_cache: ``[num_blocks, 1, BLOCK_SIZE, KV_LORA_RANK + QK_ROPE_HEAD_DIM]``
          - page_table: ``[1, num_blocks]`` int32, identity mapping (single stream)
        """
        kvpe_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # +1 guard block: the prefill attends with page_table[:, :end_block]. When the
        # last chunk reaches the final block, end_block == num_blocks makes that a
        # full-extent ttnn.slice, which ALIASES the parent — deallocating the slice
        # would then free the real page_table and crash decode. One extra (unused)
        # block keeps every slice a strict sub-slice (a true copy), so dealloc is safe.
        num_blocks = (max_seq_len + self.block_size - 1) // self.block_size + 1
        kvpe_cache = ttnn.as_tensor(
            torch.zeros(num_blocks, 1, self.block_size, kvpe_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        page_table = ttnn.as_tensor(
            torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return (kvpe_cache, page_table)

    # ------------------------------------------------------------------
    # Decode forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        kv_cache: tuple,
        current_pos: int,
        cur_pos_tensor: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """
        Single-token decode step.

        Args:
            x:           [1, 1, 1, HIDDEN_SIZE] replicated on all devices
            cos/sin:     [1, 1, 1, QK_ROPE_HEAD_DIM] for position current_pos
            kv_cache:    (kvpe_cache,) — single [1, 1, max_seq_len, KV_LORA_RANK+QK_ROPE_HEAD_DIM] latent cache
            current_pos: cache slot to write the new latent token into
        Returns:
            [1, 1, 1, HIDDEN_SIZE]
        """
        kvpe_cache, page_table = kv_cache
        _mem = ttnn.L1_MEMORY_CONFIG

        # ── Q path (seq_len=1, pre-split weights → no slice) ──────────
        pcs = self._decode_pcs
        q_latent = ttnn.matmul(
            x,
            self.q_a_proj,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
            program_config=pcs["q_a"],
        )
        q_latent = ttnn.rms_norm(q_latent, weight=self.q_a_norm, epsilon=NORM_EPS, memory_config=_mem)

        q_nope_flat = ttnn.matmul(
            q_latent,
            self.q_nope_w,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
            program_config=pcs["q_nope"],
        )  # [1, 1, 1, N_HEADS * QK_NOPE_HEAD_DIM]
        q_rope_flat = ttnn.matmul(
            q_latent,
            self.q_rope_w,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
            program_config=pcs["q_rope"],
        )  # [1, 1, 1, N_HEADS * QK_ROPE_HEAD_DIM]
        ttnn.deallocate(q_latent)

        q_nope = ttnn.reshape(q_nope_flat, [1, self.n_heads, 1, self.qk_nope_head_dim])
        ttnn.deallocate(q_nope_flat)
        q_rope = ttnn.reshape(q_rope_flat, [1, self.n_heads, 1, self.qk_rope_head_dim])
        ttnn.deallocate(q_rope_flat)

        q_rope_rotated = _apply_rope_ttnn(q_rope, cos, sin, 1, self.n_heads, self.qk_rope_head_dim, _mem)
        ttnn.deallocate(q_rope)

        # Absorb the K side of kv_b: q_nope @ wkv_b1 → latent-space query score.
        q_score = ttnn.matmul(
            q_nope,
            self.wkv_b1,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
        )  # [1, N_HEADS, 1, KV_LORA_RANK]
        ttnn.deallocate(q_nope)
        q_mla = ttnn.concat([q_score, q_rope_rotated], dim=-1, memory_config=_mem)
        ttnn.deallocate(q_score)
        ttnn.deallocate(q_rope_rotated)
        # q_mla: [1, N_HEADS, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM]

        # ── KV path → compressed latent (kvpe), no per-head expansion ─
        kv_combined = ttnn.matmul(
            x,
            self.kv_a_proj,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
            program_config=pcs["kv_a"],
        )
        kv_latent = ttnn.slice(kv_combined, [0, 0, 0, 0], [1, 1, 1, self.kv_lora_rank], memory_config=_mem)
        k_rope_raw = ttnn.slice(
            kv_combined, [0, 0, 0, self.kv_lora_rank], [1, 1, 1, self.kv_a_proj_out], memory_config=_mem
        )
        ttnn.deallocate(kv_combined)

        kv_latent_normed = ttnn.rms_norm(kv_latent, weight=self.kv_a_norm, epsilon=NORM_EPS, memory_config=_mem)
        ttnn.deallocate(kv_latent)

        k_rope_rotated = _apply_rope_ttnn(k_rope_raw, cos, sin, 1, 1, self.qk_rope_head_dim, _mem)
        ttnn.deallocate(k_rope_raw)

        kvpe = ttnn.concat([kv_latent_normed, k_rope_rotated], dim=-1, memory_config=_mem)
        ttnn.deallocate(kv_latent_normed)
        ttnn.deallocate(k_rope_rotated)
        # kvpe: [1, 1, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM]  (single shared KV head)

        # ── Ensure an INT32 device position tensor ─────────────────────
        # Both paged_update_cache (latent write) and flash-MLA-decode (attend) read
        # the position from this device tensor rather than a Python int — that is
        # what lets the whole decode step be captured as a replayable trace later.
        # paged_update_cache requires INT32 specifically.
        _free_pos_tensor = False
        if cur_pos_tensor is None:
            cur_pos_tensor = ttnn.as_tensor(
                torch.tensor([current_pos], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            _free_pos_tensor = True

        # ── Update latent cache at current_pos (tensor-indexed → trace-safe) ─
        # kvpe is [1, 1, 1, kvpe_dim] (single KV head); height-shard it and write
        # kvpe → kvpe_cache[0, 0, pos, :].
        kvpe_upd = ttnn.to_memory_config(kvpe, self._kvpe_update_mem_cfg)
        ttnn.deallocate(kvpe)
        ttnn.experimental.paged_update_cache(
            kvpe_cache, kvpe_upd, update_idxs_tensor=cur_pos_tensor, page_table=page_table
        )
        ttnn.deallocate(kvpe_upd)

        # ── Flash-MLA decode over the paged latent cache ────────────────
        # q_mla [1, N_HEADS, 1, dh] → [1, 1, N_HEADS, dh] (op wants [1, b, nh, dh]).
        q_decode = ttnn.transpose(q_mla, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_mla)
        attn_latent = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_decode,
            kvpe_cache,
            None,  # V reuses the kvpe cache (first head_dim_v dims)
            self.kv_lora_rank,  # head_dim_v
            page_table,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            program_config=self._mla_decode_pc,
            compute_kernel_config=self.attn_compute_kernel_config,
            memory_config=_mem,
        )  # [1, 1, N_HEADS, KV_LORA_RANK]
        ttnn.deallocate(q_decode)
        if _free_pos_tensor:
            ttnn.deallocate(cur_pos_tensor)

        # Expand the V side of kv_b: attn_latent @ wkv_b2 → per-head value output.
        # [1, 1, N_HEADS, KV_LORA_RANK] → [1, N_HEADS, 1, KV_LORA_RANK] for batched matmul.
        attn_latent = ttnn.transpose(attn_latent, 1, 2, memory_config=_mem)
        attn_out = ttnn.matmul(
            attn_latent,
            self.wkv_b2,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
        )  # [1, N_HEADS, 1, V_HEAD_DIM]
        ttnn.deallocate(attn_latent)

        # [1, N_HEADS, 1, V_HEAD_DIM] → transpose(1,2) → [1, 1, N_HEADS, V_HEAD_DIM]
        # → reshape → [1, 1, 1, N_HEADS * V_HEAD_DIM] for o_proj
        attn_out_t = ttnn.transpose(attn_out, 1, 2, memory_config=_mem)
        ttnn.deallocate(attn_out)

        # ── Output projection ──────────────────────────────────────────
        # M=1 tile → DRAM-bank-sharded weight + DS program config.
        # Width-shard in0 across the DS compute row (gx=4), output L1
        # width-sharded, then convert back to L1 interleaved so downstream
        # ops see the same layout as before.
        attn_flat = ttnn.reshape(attn_out_t, [1, 1, 1, self.n_heads * self.v_head_dim])
        ttnn.deallocate(attn_out_t)
        _in0_memcfg = ttnn.create_sharded_memory_config(
            (1, 1, ttnn.TILE_SIZE, self.n_heads * self.v_head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=self._o_proj_decode_gx),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        attn_flat = ttnn.to_memory_config(attn_flat, _in0_memcfg)
        out = ttnn.matmul(
            attn_flat,
            self.o_proj_decode,
            program_config=self.o_proj_decode_pc,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                buffer_type=ttnn.BufferType.L1,
            ),
        )
        ttnn.deallocate(attn_flat)
        # Downstream (residual add) expects interleaved L1 — match the prior contract.
        out = ttnn.to_memory_config(out, _mem)
        return out
