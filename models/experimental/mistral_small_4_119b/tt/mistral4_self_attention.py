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
from models.experimental.mistral_small_4_119b.tt.prefill_matmul_config import (
    build_kv_a_proj_preset,
    build_kv_b_proj_preset,
    build_o_proj_preset,
    build_q_a_proj_preset,
    build_q_b_proj_preset,
    prefill_linear,
)
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


def _sharded_rms_norm(x: ttnn.Tensor, weight: ttnn.Tensor, epsilon: float, out_memory_config) -> ttnn.Tensor:
    """rms_norm that forces width-sharded input so the sharded multi-core
    kernel dispatches (~16-32 cores) instead of the default ~1-core path.

    For attention's q_a_norm (K=1024) and kv_a_norm (K=512) this drops the
    norm from ~16/6 µs on 1 core to ~3 µs on 16/32 cores."""
    TILE = 32
    in_mem = x.memory_config()
    if in_mem.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        ws_cfg = in_mem
        x_ws = x
    else:
        M_padded = (int(x.shape[-2]) + TILE - 1) // TILE * TILE
        K_padded = (int(x.shape[-1]) + TILE - 1) // TILE * TILE
        Kt = K_padded // TILE
        cores = 1
        for cand in (32, 16, 8, 4, 2):
            if Kt % cand == 0:
                cores = cand
                break
        if cores >= 32:
            gx, gy = 8, 4
        elif cores >= 16:
            gx, gy = 8, 2
        elif cores >= 8:
            gx, gy = 8, 1
        elif cores >= 4:
            gx, gy = 4, 1
        else:
            gx, gy = cores, 1
        ws_cfg = ttnn.create_sharded_memory_config(
            (1, 1, M_padded, K_padded),
            core_grid=ttnn.CoreGrid(y=gy, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        x_ws = ttnn.to_memory_config(x, ws_cfg)
    out = ttnn.rms_norm(x_ws, weight=weight, epsilon=epsilon, memory_config=ws_cfg)
    if out_memory_config.memory_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        out = ttnn.to_memory_config(out, out_memory_config)
    return out


def _load_weight(
    state_dict: dict,
    key: str,
    transpose: bool,
    dtype: ttnn.DataType,
    mesh_device: ttnn.MeshDevice,
    mesh_mapper=None,
    transform_fn=None,
    cache_file_name=None,
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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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

        # Sweep-tuned matmul presets (test_prefill_matmul_sweep.py).
        self.q_a_proj_preset = build_q_a_proj_preset(mesh_device)
        self.q_b_proj_preset = build_q_b_proj_preset(mesh_device)
        self.kv_a_proj_preset = build_kv_a_proj_preset(mesh_device)
        self.kv_b_proj_preset = build_kv_b_proj_preset(mesh_device)
        self.o_proj_preset = build_o_proj_preset(mesh_device)

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

        grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        # 1D-mcast program configs for decode matmuls (M=1 tile for all).
        self._decode_pcs = self._build_decode_program_configs(grid)

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

        self.q_b_proj = _load_weight(
            state_dict,
            p + "q_b_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            transform_fn=_deinterleave_q_b_proj,
            cache_file_name=_cf(p + "q_b_proj.weight"),
        )  # [Q_LORA_RANK, N_HEADS * HEAD_DIM]

        # Pre-split q_b_proj into nope/rope sub-weights for decode (avoids 2 slices/step).
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

        self.kv_b_proj = _load_weight(
            state_dict,
            p + "kv_b_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat16,
            mesh_device=mesh_device,
            cache_file_name=_cf(p + "kv_b_proj.weight"),
        )  # [KV_LORA_RANK, KV_B_PROJ_OUT_TOTAL]

        # Pre-split kv_b_proj into k_nope/v sub-weights for decode (avoids 2 slices/step).
        _kv_b_scale = state_dict.get((p + "kv_b_proj.weight").replace(".weight", ".weight_scale_inv"))
        _kv_b = _torch_for_ttnn_upload(state_dict[p + "kv_b_proj.weight"], _kv_b_scale).T.contiguous()
        _kv_b3 = _kv_b.reshape(KV_LORA_RANK, N_HEADS, KV_B_PROJ_OUT_PER_HEAD)
        _k_nope = _kv_b3[:, :, :QK_NOPE_HEAD_DIM].reshape(KV_LORA_RANK, N_HEADS * QK_NOPE_HEAD_DIM).contiguous()
        _v = _kv_b3[:, :, QK_NOPE_HEAD_DIM:].reshape(KV_LORA_RANK, N_HEADS * V_HEAD_DIM).contiguous()
        self.k_nope_w = ttnn.as_tensor(
            _k_nope,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cf(p + "k_nope_w"),
        )
        self.v_w = ttnn.as_tensor(
            _v,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=_cf(p + "v_w"),
        )
        del _kv_b, _kv_b3, _k_nope, _v

        self.o_proj = _load_weight(
            state_dict,
            p + "o_proj.weight",
            transpose=True,
            dtype=ttnn.bfloat4_b,
            mesh_device=mesh_device,
            cache_file_name=_cf(p + "o_proj.weight"),
        )  # [N_HEADS * V_HEAD_DIM, HIDDEN_SIZE]

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

        H = HIDDEN_SIZE
        return {
            "q_a": _pc(H // 32, Q_LORA_RANK // 32),  # [4096,1024]: K=128, N=32
            "kv_a": _pc(H // 32, KV_A_PROJ_OUT // 32),  # [4096,320]: K=128, N=10
            "q_nope": _pc(Q_LORA_RANK // 32, N_HEADS * QK_NOPE_HEAD_DIM // 32),  # [1024,2048]: K=32, N=64
            "q_rope": _pc(Q_LORA_RANK // 32, N_HEADS * QK_ROPE_HEAD_DIM // 32),  # [1024,2048]: K=32, N=64
            "k_nope": _pc(KV_LORA_RANK // 32, N_HEADS * QK_NOPE_HEAD_DIM // 32),  # [256,2048]: K=8, N=64
            "v": _pc(KV_LORA_RANK // 32, N_HEADS * V_HEAD_DIM // 32),  # [256,4096]: K=8, N=128
            "o": _pc((N_HEADS * V_HEAD_DIM) // 32, H // 32),  # [4096,4096]: K=128, N=128
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        kv_cache: tuple = None,
    ) -> ttnn.Tensor:
        """
        Prefill forward.

        Args:
            x:        [1, 1, seq_len, HIDDEN_SIZE] replicated on all devices
            cos/sin:  [1, 1, seq_len, QK_ROPE_HEAD_DIM]  (from HF rotary, on device)
            kv_cache: optional (k_cache, v_cache) — if provided, filled in-place
        Returns:
            [1, 1, seq_len, HIDDEN_SIZE] replicated on all devices
        """
        seq_len = x.shape[2]
        reshape_mem = ttnn.L1_MEMORY_CONFIG if self.prefill_l1_reshape else ttnn.DRAM_MEMORY_CONFIG

        # ── Q projection ──────────────────────────────────────────────────
        # Sweep-tuned 1D l1/dram/ws 4×4 w=16 (20.3 TFLOPs, 4.3× over default).
        # Output to L1 so the downstream q_a_norm reads L1 instead of DRAM.
        q_latent = prefill_linear(
            x,
            self.q_a_proj,
            self.q_a_proj_preset,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
            in0_bf8=True,
        )  # [1, 1, seq, Q_LORA_RANK]

        # rms_norm output to L1: small tensor (Q_LORA_RANK ≪ HIDDEN_SIZE) and the
        # downstream q_b_proj matmul gets an L1 in0 for cheaper reads.
        # _sharded_rms_norm dispatches the multi-core sharded kernel (~32 cores)
        # instead of the 1-core default.
        q_latent = _sharded_rms_norm(
            q_latent,
            self.q_a_norm,
            NORM_EPS,
            ttnn.L1_MEMORY_CONFIG,
        )

        # Sweep-tuned: 1D dram/dram/ws on 8×8 grid, in0_block_w=2 (13.4 TFLOPs).
        q = prefill_linear(
            q_latent,
            self.q_b_proj,
            self.q_b_proj_preset,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            in0_bf8=True,
        )  # [1, 1, seq, N_HEADS * HEAD_DIM]
        ttnn.deallocate(q_latent)

        q = ttnn.reshape(q, [1, seq_len, self.n_heads, self.head_dim])
        q = ttnn.transpose(q, 1, 2, memory_config=reshape_mem)
        # [1, N_HEADS, seq, HEAD_DIM]

        q_nope = ttnn.slice(
            q, [0, 0, 0, 0], [1, self.n_heads, seq_len, self.qk_nope_head_dim], memory_config=reshape_mem
        )
        q_rope = ttnn.slice(
            q,
            [0, 0, 0, self.qk_nope_head_dim],
            [1, self.n_heads, seq_len, self.head_dim],
            memory_config=reshape_mem,
        )
        ttnn.deallocate(q)

        q_rope_rotated = ttnn.experimental.rotary_embedding_hf(
            q_rope, cos, sin, is_decode_mode=False, compute_kernel_config=self.lofi_compute_kernel_config
        )
        ttnn.deallocate(q_rope)

        q_full = ttnn.concat([q_nope, q_rope_rotated], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_rope_rotated)
        # q_full: [1, N_HEADS, seq, HEAD_DIM]

        # ── KV projection ─────────────────────────────────────────────────
        # Sweep-tuned: 1D l1/dram/ws on 1×10 grid (forced by Nt=10), in0_block_w=16.
        # Output to L1 so the downstream kv_a_norm reads L1 instead of DRAM.
        kv_combined = prefill_linear(
            x,
            self.kv_a_proj,
            self.kv_a_proj_preset,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
            in0_bf8=True,
        )  # [1, 1, seq, KV_A_PROJ_OUT]

        kv_latent = ttnn.slice(kv_combined, [0, 0, 0, 0], [1, 1, seq_len, self.kv_lora_rank])
        k_rope_raw = ttnn.slice(kv_combined, [0, 0, 0, self.kv_lora_rank], [1, 1, seq_len, self.kv_a_proj_out])
        ttnn.deallocate(kv_combined)

        # rms_norm output to L1: KV_LORA_RANK ≪ HIDDEN_SIZE, downstream kv_b_proj
        # matmul reads from L1. _sharded_rms_norm dispatches the multi-core
        # sharded kernel (~16 cores at K=512) instead of the 1-core default.
        kv_latent_normed = _sharded_rms_norm(
            kv_latent,
            self.kv_a_norm,
            NORM_EPS,
            ttnn.L1_MEMORY_CONFIG,
        )
        # Keep kv_latent alive; will be reused for kv_b_proj

        # Sweep-tuned: 1D dram/dram/ws on 8×4 grid, in0_block_w=1 (10.7 TFLOPs).
        kv = prefill_linear(
            kv_latent_normed,
            self.kv_b_proj,
            self.kv_b_proj_preset,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            in0_bf8=True,
        )  # [1, 1, seq, KV_B_PROJ_OUT_TOTAL]
        ttnn.deallocate(kv_latent)
        ttnn.deallocate(kv_latent_normed)

        kv = ttnn.reshape(kv, [1, seq_len, self.n_heads, self.kv_b_per_head])
        kv = ttnn.transpose(kv, 1, 2, memory_config=reshape_mem)
        # [1, N_HEADS, seq, KV_B_PER_HEAD=192]

        k_nope = ttnn.slice(
            kv, [0, 0, 0, 0], [1, self.n_heads, seq_len, self.qk_nope_head_dim], memory_config=reshape_mem
        )
        v = ttnn.slice(
            kv,
            [0, 0, 0, self.qk_nope_head_dim],
            [1, self.n_heads, seq_len, self.kv_b_per_head],
            memory_config=reshape_mem,
        )
        ttnn.deallocate(kv)

        k_rope_rotated = ttnn.experimental.rotary_embedding_hf(
            k_rope_raw, cos, sin, is_decode_mode=False, compute_kernel_config=self.lofi_compute_kernel_config
        )
        # [1, 1, seq, QK_ROPE_HEAD_DIM]
        ttnn.deallocate(k_rope_raw)

        # Broadcast k_rope to all heads by repeating along dim=1
        k_rope_expanded = ttnn.repeat(
            k_rope_rotated,
            ttnn.Shape([1, self.n_heads, 1, 1]),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, N_HEADS, seq, QK_ROPE_HEAD_DIM]
        ttnn.deallocate(k_rope_rotated)

        k_full = ttnn.concat([k_nope, k_rope_expanded], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(k_nope)
        ttnn.deallocate(k_rope_expanded)
        # k_full: [1, N_HEADS, seq, HEAD_DIM]

        # Fill KV cache if provided (prefill with cache)
        if kv_cache is not None:
            self.fill_kv_cache(k_full, v, kv_cache)

        # ── Scaled dot-product attention ───────────────────────────────────
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_full,
            k_full,
            v,
            is_causal=True,
            scale=self.scale,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, N_HEADS, seq, V_HEAD_DIM]
        ttnn.deallocate(q_full)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v)

        # ── Output projection ──────────────────────────────────────────────
        # attn_out is [1, N_HEADS, seq, V_HEAD_DIM]; transpose to [1, seq, N_HEADS, V_HEAD_DIM]
        # before reshaping so head features for the same position are contiguous.
        attn_out_t = ttnn.transpose(attn_out, 1, 2, memory_config=reshape_mem)
        ttnn.deallocate(attn_out)
        attn_flat = ttnn.reshape(
            attn_out_t, [1, 1, seq_len, self.n_heads * self.v_head_dim]
        )  # [1, 1, seq, N_HEADS * V_HEAD_DIM = 4096]
        ttnn.deallocate(attn_out_t)

        # Sweep-tuned 1D l1/dram/dram 8×4 w=8 (33.3 TFLOPs, 2.3× over default).
        out = prefill_linear(
            attn_flat,
            self.o_proj,
            self.o_proj_preset,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            in0_bf8=True,
        )  # [1, 1, seq, HIDDEN_SIZE]
        ttnn.deallocate(attn_flat)

        return out

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def allocate_kv_cache(self, max_seq_len: int) -> tuple:
        """
        Pre-allocate K and V cache tensors on device (zeroed, DRAM, replicated).

        Cache shape: [1, N_HEADS, padded_seq, dim] — batch-first.
        padded_seq is max_seq_len rounded up to the nearest 32 because
        scaled_dot_product_attention_decode requires k_chunk_size % 32 == 0,
        and get_chunk_size(S) can only return a multiple of 32 if S itself is.
        """
        padded_seq = ((max_seq_len + 31) // 32) * 32
        k_cache = ttnn.as_tensor(
            torch.zeros(1, self.n_heads, padded_seq, self.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        v_cache = ttnn.as_tensor(
            torch.zeros(1, self.n_heads, padded_seq, self.v_head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return k_cache, v_cache

    def fill_kv_cache(self, k_full: ttnn.Tensor, v: ttnn.Tensor, kv_cache: tuple) -> None:
        """
        Fill KV cache in-place from prefill K/V tensors.

        k_full and v are already [1, N_HEADS, seq, dim] (batch-first),
        which is exactly what fill_cache_for_user_ expects.
        """
        k_cache, v_cache = kv_cache
        ttnn.kv_cache.fill_cache_for_user_(k_cache, k_full, 0)
        ttnn.kv_cache.fill_cache_for_user_(v_cache, v, 0)

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
            kv_cache:    (k_cache, v_cache) each [1, N_HEADS, max_seq_len, dim] (batch-first)
            current_pos: cache slot to write the new K/V token into
        Returns:
            [1, 1, 1, HIDDEN_SIZE]
        """
        k_cache, v_cache = kv_cache
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
        q_full = ttnn.concat([q_nope, q_rope_rotated], dim=-1, memory_config=_mem)
        ttnn.deallocate(q_nope)
        ttnn.deallocate(q_rope_rotated)
        # q_full: [1, N_HEADS, 1, HEAD_DIM]

        # ── KV path (seq_len=1, pre-split weights → no slice) ─────────
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

        k_nope_flat = ttnn.matmul(
            kv_latent_normed,
            self.k_nope_w,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
            program_config=pcs["k_nope"],
        )  # [1, 1, 1, N_HEADS * QK_NOPE_HEAD_DIM]
        v_new_flat = ttnn.matmul(
            kv_latent_normed,
            self.v_w,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
            program_config=pcs["v"],
        )  # [1, 1, 1, N_HEADS * V_HEAD_DIM]
        ttnn.deallocate(kv_latent)
        ttnn.deallocate(kv_latent_normed)

        k_nope = ttnn.reshape(k_nope_flat, [1, self.n_heads, 1, self.qk_nope_head_dim])
        ttnn.deallocate(k_nope_flat)
        v_new = ttnn.reshape(v_new_flat, [1, self.n_heads, 1, self.v_head_dim])
        ttnn.deallocate(v_new_flat)

        k_rope_rotated = _apply_rope_ttnn(k_rope_raw, cos, sin, 1, 1, self.qk_rope_head_dim, _mem)
        ttnn.deallocate(k_rope_raw)
        k_rope_expanded = ttnn.repeat(k_rope_rotated, ttnn.Shape([1, self.n_heads, 1, 1]), memory_config=_mem)
        ttnn.deallocate(k_rope_rotated)
        k_full = ttnn.concat([k_nope, k_rope_expanded], dim=-1, memory_config=_mem)
        ttnn.deallocate(k_nope)
        ttnn.deallocate(k_rope_expanded)
        # k_full: [1, N_HEADS, 1, HEAD_DIM],  v_new: [1, N_HEADS, 1, V_HEAD_DIM]

        # ── Update KV cache at current_pos ─────────────────────────────
        ttnn.kv_cache.update_cache_for_token_(k_cache, k_full, current_pos)
        ttnn.kv_cache.update_cache_for_token_(v_cache, v_new, current_pos)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_new)

        # ── Decode SDPA ────────────────────────────────────────────────
        # scaled_dot_product_attention_decode requires Q in DRAM (kernel constraint).
        # q_full is [1, N_HEADS, 1, HEAD_DIM]; transpose dims 1&2 → [1, 1, N_HEADS, HEAD_DIM].
        q_decode = ttnn.transpose(q_full, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_full)

        _free_pos_tensor = False
        if cur_pos_tensor is None:
            cur_pos_tensor = ttnn.as_tensor(
                torch.tensor([current_pos], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            _free_pos_tensor = True

        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q_decode,
            k_cache,
            v_cache,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            memory_config=_mem,
        )  # [1, B=1, NH=N_HEADS, V_HEAD_DIM]
        ttnn.deallocate(q_decode)
        if _free_pos_tensor:
            ttnn.deallocate(cur_pos_tensor)

        # [1, 1, N_HEADS, V_HEAD_DIM] → transpose(1,2) → [1, N_HEADS, 1, V_HEAD_DIM]
        # → reshape → [1, 1, 1, N_HEADS * V_HEAD_DIM] for o_proj
        attn_out_t = ttnn.transpose(attn_out, 1, 2, memory_config=_mem)
        ttnn.deallocate(attn_out)

        # ── Output projection ──────────────────────────────────────────
        attn_flat = ttnn.reshape(attn_out_t, [1, 1, 1, self.n_heads * self.v_head_dim])
        ttnn.deallocate(attn_out_t)
        attn_flat = ttnn.to_memory_config(attn_flat, _mem)
        out = ttnn.matmul(
            attn_flat,
            self.o_proj,
            compute_kernel_config=self.lofi_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=_mem,
            program_config=pcs["o"],
        )
        ttnn.deallocate(attn_flat)
        return out
