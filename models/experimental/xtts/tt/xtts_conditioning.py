# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ttnn.operations.normalization import dram_group_norm_virtual_columns

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.config import (  # noqa: F401 — re-exported for callers
    ENC_HEAD_DIM,
    GROUP_NORM_EPS,
    GROUP_NORM_GROUPS,
    HIDDEN_SIZE,
    NUM_ATTN_HEADS,
    NUM_LATENTS,
    PERCEIVER_DEPTH,
    PERCEIVER_HEAD_DIM,
    PERCEIVER_HEADS,
    PERCEIVER_INNER,
    TILE,
)

L1 = ttnn.L1_MEMORY_CONFIG

COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

STATS_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

ATTN_QKV_GX, ATTN_QKV_IBW = 12, 4
ATTN_PROJ_GX, ATTN_PROJ_IBW = 13, 4
PERC_QKV_GX, PERC_QKV_IBW = 12, 4
FF_GATE_IBW = 8
ATTN_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

INIT_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


def _lin(torch_tensor, device):
    """Upload a linear weight transposed to device tiles."""
    w = torch_tensor
    if w.dim() == 3:
        w = w.squeeze(-1)
    return ttnn.from_torch(
        w.t().contiguous().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )


def _vec(torch_tensor, device):
    """Upload a bias/vector tensor to device tiles."""
    return ttnn.from_torch(torch_tensor.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)


def _perm_qkv_out(t):
    """Reorder packed QKV weights to head-major layout."""
    idx = torch.arange(3 * HIDDEN_SIZE).reshape(NUM_ATTN_HEADS, 3, ENC_HEAD_DIM).permute(1, 0, 2).reshape(-1)
    return t[idx]


def _clamp_gx(preferred_gx, grid_x, nt):
    """Clamp preferred GX to a divisor of Nt within grid."""
    max_gx = max(1, min(int(preferred_gx), int(grid_x)))
    for gx in range(max_gx, 0, -1):
        if nt % gx == 0:
            return gx
    return max_gx


def _1d_grid_covering(n_tiles, grid):
    """Pick a compact 1D grid covering at least n_tiles cores."""
    max_x, max_y = int(grid.x), int(grid.y)
    best = None
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            cores = gx * gy
            if cores < n_tiles:
                continue
            key = (cores, abs(gx - gy), gx * gy)
            if best is None or key < best[0]:
                best = (key, gx, gy)
    if best is None:
        return max_x, max_y
    return best[1], best[2]


def _mm_2d(grid, mt, kt, nt, gx=8, ibw=None, fp32_acc=False):
    """Build a 2D multicast matmul program config."""
    gx = _clamp_gx(gx, grid.x, nt)
    gy = max(1, min(mt, grid.y))
    per_core_m, per_core_n = -(-mt // gy), -(-nt // gx)
    # DEST budget 8 tiles (4 with fp32_dest_acc_en); pass fp32_acc=True or the op fatals.
    cap = 4 if fp32_acc else 8
    sub_w = max(w for w in range(1, per_core_n + 1) if per_core_n % w == 0 and w <= cap)
    sub_h = max(h for h in range(1, per_core_m + 1) if per_core_m % h == 0 and h * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        in0_block_w=min(ibw or kt, kt),
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


class TtXttsConditioning(LightweightModule):
    def __init__(self, state_dict, device):
        """Load conditioning encoder and perceiver weights."""
        super().__init__()
        self.device = device
        e = "gpt.conditioning_encoder."
        p = "gpt.conditioning_perceiver."

        _ff_inner = state_dict[p + "layers.0.1.0.weight"].shape[0] // 2
        _ff_nt = -(-_ff_inner // 32)
        _ff_gx, _ff_gy = _1d_grid_covering(_ff_nt, self.device.compute_with_storage_grid_size())
        # activation="gelu" on ttnn.linear does NOT fuse unless program_config pins the grid.
        self._ff_gate_mm = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(_ff_gx, _ff_gy),
            in0_block_w=FF_GATE_IBW,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False),  # False = exact erf GELU
            mcast_in0=True,
        )
        self._ff_val_mm = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(_ff_gx, _ff_gy),
            in0_block_w=FF_GATE_IBW,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.init_w = _lin(state_dict[e + "init.weight"], device)
        self.init_b = _vec(state_dict[e + "init.bias"], device)
        self._grid = device.compute_with_storage_grid_size()
        self._pc_cache = {}
        self._perc_qkv_pc = {}
        self._gn_masks = {}

        self.blocks = []
        i = 0
        while (e + f"attn.{i}.qkv.weight") in state_dict:
            self.blocks.append(
                {
                    "gn_host": (
                        state_dict[e + f"attn.{i}.norm.weight"].float(),
                        state_dict[e + f"attn.{i}.norm.bias"].float(),
                    ),
                    "qkv_w": _lin(_perm_qkv_out(state_dict[e + f"attn.{i}.qkv.weight"]), device),
                    "qkv_b": _vec(_perm_qkv_out(state_dict[e + f"attn.{i}.qkv.bias"]), device),
                    "proj_w": _lin(state_dict[e + f"attn.{i}.proj_out.weight"], device),
                    "proj_b": _vec(state_dict[e + f"attn.{i}.proj_out.bias"], device),
                }
            )
            i += 1

        # Keep in DRAM: L1 residency clashes GPT circular buffers in the full demo.
        self.latents = ttnn.from_torch(
            state_dict[p + "latents"].reshape(1, NUM_LATENTS, HIDDEN_SIZE).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=device,
            dtype=ttnn.bfloat16,
        )
        self.layers = []
        for j in range(PERCEIVER_DEPTH):
            qkv = torch.cat([state_dict[p + f"layers.{j}.0.to_q.weight"], state_dict[p + f"layers.{j}.0.to_kv.weight"]])
            ff0_w, ff0_b = state_dict[p + f"layers.{j}.1.0.weight"], state_dict[p + f"layers.{j}.1.0.bias"]
            inner = ff0_w.shape[0] // 2
            self.layers.append(
                {
                    "qkv_w": _lin(qkv, device),
                    "to_out": _lin(state_dict[p + f"layers.{j}.0.to_out.weight"], device),
                    "ff_val_w": _lin(ff0_w[:inner], device),
                    "ff_val_b": _vec(ff0_b[:inner], device),
                    "ff_gate_w": _lin(ff0_w[inner:], device),
                    "ff_gate_b": _vec(ff0_b[inner:], device),
                    "ff2_w": _lin(state_dict[p + f"layers.{j}.1.2.weight"], device),
                    "ff2_b": _vec(state_dict[p + f"layers.{j}.1.2.bias"], device),
                }
            )
        self.perc_norm_gamma = _vec(state_dict[p + "norm.gamma"], device)

    def _gn_operands(self, s, blk):
        # Cache DRAM mask/affine: host->device writes are fatal inside a trace capture.
        """Cache group-norm mask and affine operands for length S."""
        cached = blk.setdefault("_gn", {}).get(s)
        if cached is None:
            grid = ttnn.determine_expected_group_norm_dram_grid_size(
                device=self.device,
                num_channels=HIDDEN_SIZE,
                num_groups=GROUP_NORM_GROUPS,
                input_nhw=s,
                num_batches=1,
            )
            cols = dram_group_norm_virtual_columns(grid, HIDDEN_SIZE, GROUP_NORM_GROUPS)
            # A non-tile-aligned S needs the doubled mask (second set with rows >= S % TILE zeroed,
            # selected on the final row-tile). Build it here: given a single-set mask, group_norm
            # derives the second set itself with a host build + upload per call, which is fatal
            # inside a trace capture.
            rows_in_last_tile = s % TILE
            mask = self._gn_masks.get((cols, rows_in_last_tile))
            if mask is None:
                mask = ttnn.to_device(
                    ttnn.create_group_norm_input_mask(
                        HIDDEN_SIZE,
                        GROUP_NORM_GROUPS,
                        cols,
                        ttnn.bfloat16,
                        rows_in_last_tile=rows_in_last_tile,
                    ),
                    self.device,
                )
                self._gn_masks[(cols, rows_in_last_tile)] = mask
            affine = [
                ttnn.from_torch(
                    ttnn.create_group_norm_weight_bias_rm(t, HIDDEN_SIZE, cols),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                )
                for t in blk["gn_host"]
            ]
            cached = (grid, mask, *affine)
            blk["_gn"][s] = cached
        return cached

    def _group_norm(self, x, blk):
        """Apply group norm over the sequence channel axis."""
        s = x.shape[1]
        grid, mask, gamma, beta = self._gn_operands(s, blk)
        y = ttnn.group_norm(
            ttnn.reshape(x, (1, 1, s, HIDDEN_SIZE)),
            num_groups=GROUP_NORM_GROUPS,
            epsilon=GROUP_NORM_EPS,
            input_mask=mask,
            weight=gamma,
            bias=beta,
            memory_config=L1,
            core_grid=grid,
            inplace=False,
            output_layout=ttnn.TILE_LAYOUT,
            compute_kernel_config=STATS_KERNEL_CONFIG,
        )
        ttnn.deallocate(x)
        return ttnn.reshape(y, (1, s, HIDDEN_SIZE))

    def _attn_pcs(self, s):
        """Cache QKV and proj matmul configs for sequence length S."""
        pcs = self._pc_cache.get(s)
        if pcs is None:
            mt, kt = -(-s // 32), HIDDEN_SIZE // 32
            pcs = (
                _mm_2d(self._grid, mt, kt, 3 * HIDDEN_SIZE // 32, ATTN_QKV_GX, ATTN_QKV_IBW, fp32_acc=True),
                _mm_2d(self._grid, mt, kt, HIDDEN_SIZE // 32, ATTN_PROJ_GX, ATTN_PROJ_IBW, fp32_acc=True),
            )
            self._pc_cache[s] = pcs
        return pcs

    def _attn_block(self, x, blk):
        """Run one conditioning attention block with residual."""
        y = self._group_norm(x, blk)
        qkv_pc, proj_pc = self._attn_pcs(y.shape[1])
        qkv = ttnn.linear(
            y,
            blk["qkv_w"],
            bias=blk["qkv_b"],
            memory_config=L1,
            compute_kernel_config=ATTN_KERNEL_CONFIG,
            program_config=qkv_pc,
        )
        b, s, _ = qkv.shape
        qkv = ttnn.reshape(qkv, (b, 1, s, 3 * HIDDEN_SIZE))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NUM_ATTN_HEADS, transpose_k_heads=False, memory_config=L1
        )
        ttnn.deallocate(qkv)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, memory_config=L1)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)
        ttnn.deallocate(attn)
        h = ttnn.linear(
            out,
            blk["proj_w"],
            bias=blk["proj_b"],
            memory_config=L1,
            compute_kernel_config=ATTN_KERNEL_CONFIG,
            program_config=proj_pc,
        )
        ttnn.deallocate(out)
        res = ttnn.add(y, h, memory_config=L1)
        ttnn.deallocate(y)
        ttnn.deallocate(h)
        return res

    @staticmethod
    def _tile_concat(latents, context):
        """Concat latents and context with tile-aligned padding."""
        n_lat, n_ctx = latents.shape[1], context.shape[1]
        ctx_pad = context.padded_shape[1]
        c = context.shape[-1]
        aligned = ttnn.reshape(context, ttnn.Shape([1, ctx_pad, c]), ttnn.Shape([1, ctx_pad, c]))
        cat = ttnn.concat([latents, aligned], dim=1, memory_config=L1)
        return ttnn.reshape(cat, ttnn.Shape([1, n_lat + n_ctx, c]), ttnn.Shape([1, n_lat + ctx_pad, c]))

    def _perceiver_attn(self, latents, context, layer):
        """Run one perceiver cross-attention layer."""
        ctx = self._tile_concat(latents, context)
        n = ctx.shape[1]
        pc = self._perc_qkv_pc.get(n)
        if pc is None:
            pc = _mm_2d(
                self._grid, -(-n // 32), HIDDEN_SIZE // 32, 3 * PERCEIVER_INNER // 32, PERC_QKV_GX, PERC_QKV_IBW
            )
            self._perc_qkv_pc[n] = pc
        qkv = ttnn.linear(
            ctx, layer["qkv_w"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG, program_config=pc
        )
        ttnn.deallocate(ctx)
        n = qkv.shape[1]
        qkv = ttnn.reshape(qkv, (1, 1, n, 3 * PERCEIVER_INNER))
        q_all, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=PERCEIVER_HEADS, transpose_k_heads=False, memory_config=L1
        )
        ttnn.deallocate(qkv)
        q = ttnn.slice(q_all, [0, 0, 0, 0], [1, PERCEIVER_HEADS, NUM_LATENTS, PERCEIVER_HEAD_DIM], memory_config=L1)
        ttnn.deallocate(q_all)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False, memory_config=L1)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)
        ttnn.deallocate(attn)
        proj = ttnn.linear(out, layer["to_out"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG)
        ttnn.deallocate(out)
        return proj

    def _perceiver_ff(self, x, layer):
        """Run gated perceiver feed-forward layer."""
        val = ttnn.linear(
            x,
            layer["ff_val_w"],
            bias=layer["ff_val_b"],
            memory_config=L1,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG,
            program_config=self._ff_val_mm,
        )
        gate = ttnn.linear(
            x,
            layer["ff_gate_w"],
            bias=layer["ff_gate_b"],
            memory_config=L1,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG,
            program_config=self._ff_gate_mm,
        )
        h = ttnn.multiply(gate, val, memory_config=L1)
        ttnn.deallocate(gate)
        ttnn.deallocate(val)
        out = ttnn.linear(
            h, layer["ff2_w"], bias=layer["ff2_b"], memory_config=L1, compute_kernel_config=COMPUTE_KERNEL_CONFIG
        )
        ttnn.deallocate(h)
        return out

    def mel_to_device(self, mel):
        """Upload mel spectrogram to device L1 tiles."""
        return ttnn.from_torch(
            mel.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device, dtype=ttnn.bfloat16, memory_config=L1
        )

    def forward(self, mel):
        """Encode host mel into conditioning latents."""
        return self.forward_dev(self.mel_to_device(mel))

    def forward_dev(self, mel_tt):
        """Encode device mel through encoder and perceiver."""
        x = ttnn.permute(mel_tt, (0, 2, 1), memory_config=L1)
        s = x.shape[1]
        h = ttnn.linear(
            x,
            self.init_w,
            bias=self.init_b,
            memory_config=L1,
            compute_kernel_config=INIT_KERNEL_CONFIG,
            program_config=_mm_2d(self._grid, -(-s // 32), -(-x.shape[2] // 32), HIDDEN_SIZE // 32, fp32_acc=True),
        )
        ttnn.deallocate(x)
        x = h

        for blk in self.blocks:
            x = self._attn_block(x, blk)

        latents = self.latents
        for layer in self.layers:
            attn = self._perceiver_attn(latents, x, layer)
            latents = ttnn.add(attn, latents, memory_config=L1)
            ttnn.deallocate(attn)
            ff = self._perceiver_ff(latents, layer)
            nxt = ttnn.add(ff, latents, memory_config=L1)
            ttnn.deallocate(ff)
            ttnn.deallocate(latents)
            latents = nxt
        ttnn.deallocate(x)
        normed = ttnn.rms_norm(latents, weight=self.perc_norm_gamma, epsilon=1e-12, memory_config=L1)
        ttnn.deallocate(latents)
        out = ttnn.permute(normed, (0, 2, 1), memory_config=L1)
        ttnn.deallocate(normed)
        return out
