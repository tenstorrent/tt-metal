# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn

from models.experimental.xtts.config import (  # noqa: F401 — re-exported for callers
    FFN_SIZE,
    HEAD_DIM,
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    NEG_INF,
    NUM_HEADS,
)
from models.common.lightweightmodule import LightweightModule

_BFP4_WEIGHTS: set[str] = set()  # empty: all weights bfloat8_b (bfp4 fails e2e spectrogram PCC)
L1 = ttnn.L1_MEMORY_CONFIG


def _to_device(torch_tensor, device):
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


def _to_device_w8(torch_tensor, device, dtype=ttnn.bfloat8_b):
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        dtype=dtype,
    )


def _to_device_bias(torch_tensor, device):
    return _to_device(torch_tensor.reshape(1, -1), device)


_LN_SHARD_CACHE = {}
_PREFILL_LN_CACHE = {}
_PREFILL_MLP_CPROJ_CACHE = {}
_PREFILL_LN_MAX_CORES = 16  # matches mlp/attn c_proj shard grid (skip post-residual ITS)
_PREFILL_MLP_CPROJ_CORES = 16


def _fit_core_grid(nc, max_x, max_y):
    for gx in range(min(nc, max_x), 0, -1):
        if nc % gx == 0 and nc // gx <= max_y:
            return gx, nc // gx
    return None


def _decode_ln_cfg(device):
    key = id(device)
    if key not in _LN_SHARD_CACHE:
        nc = 8
        bw = HIDDEN_SIZE // nc // 32
        mc = ttnn.create_sharded_memory_config(
            shape=(32, HIDDEN_SIZE // nc),
            core_grid=ttnn.CoreGrid(x=nc, y=1),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[nc, 1], subblock_w=bw, block_h=1, block_w=bw, inplace=False
        )
        _LN_SHARD_CACHE[key] = (mc, pc)
    return _LN_SHARD_CACHE[key]


def _prefill_ln_cfg(device, mt):
    key = (id(device), mt)
    if key not in _PREFILL_LN_CACHE:
        grid = device.compute_with_storage_grid_size()
        nc = _PREFILL_LN_MAX_CORES
        placed = _fit_core_grid(nc, grid.x, grid.y)
        if placed is None:
            for cand in (8, 4, 2, 1):
                placed = _fit_core_grid(cand, grid.x, grid.y)
                if placed is not None:
                    nc = cand
                    break
        gx, gy = placed
        bw = (HIDDEN_SIZE // 32) // nc
        mc = ttnn.create_sharded_memory_config(
            shape=(mt * 32, HIDDEN_SIZE // nc),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[gx, gy],
            subblock_w=min(bw, 4),
            block_h=mt,
            block_w=bw,
            inplace=False,
        )
        _PREFILL_LN_CACHE[key] = (mc, pc)
    return _PREFILL_LN_CACHE[key]


def sharded_decode_ln(x, weight, bias, device):
    mc, pc = _decode_ln_cfg(device)
    xs = ttnn.to_memory_config(x, mc)
    h = ttnn.layer_norm(xs, weight=weight, bias=bias, epsilon=LAYER_NORM_EPS, program_config=pc, memory_config=mc)
    ttnn.deallocate(xs)
    out = ttnn.to_memory_config(h, L1)
    ttnn.deallocate(h)
    return out


def sharded_prefill_ln(x, weight, bias, device):
    mt = -(-x.shape[-2] // 32)
    mc, pc = _prefill_ln_cfg(device, mt)
    if x.is_sharded() and x.memory_config() == mc:
        h = ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=LAYER_NORM_EPS, program_config=pc, memory_config=mc)
    else:
        xs = ttnn.to_memory_config(x, mc)
        h = ttnn.layer_norm(xs, weight=weight, bias=bias, epsilon=LAYER_NORM_EPS, program_config=pc, memory_config=mc)
        ttnn.deallocate(xs)
    out = ttnn.to_memory_config(h, L1)
    ttnn.deallocate(h)
    return out


def _prefill_mlp_cproj_cfg(device, m):
    key = (id(device), m)
    if key not in _PREFILL_MLP_CPROJ_CACHE:
        grid = device.compute_with_storage_grid_size()
        nc = _PREFILL_MLP_CPROJ_CORES
        placed = _fit_core_grid(nc, grid.x, grid.y)
        if placed is None:
            for cand in (8, 4, 2, 1):
                placed = _fit_core_grid(cand, grid.x, grid.y)
                if placed is not None:
                    nc = cand
                    break
        gx, gy = placed
        in_mc = ttnn.create_sharded_memory_config(
            shape=(m, FFN_SIZE // nc),
            core_grid=ttnn.CoreGrid(x=gx, y=gy),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        out_mc, _ = _prefill_ln_cfg(device, math.ceil(m / 32))
        mt, kt, nt = math.ceil(m / 32), FFN_SIZE // 32, HIDDEN_SIZE // 32
        pcn = math.ceil(nt / (gx * gy))
        ibw = next(b for b in (8, 4, 2, 1) if kt % b == 0)
        osw = next(w for w in (4, 2, 1) if pcn % w == 0)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=osw,
            per_core_M=mt,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        _PREFILL_MLP_CPROJ_CACHE[key] = (in_mc, out_mc, pc)
    return _PREFILL_MLP_CPROJ_CACHE[key]


def _mm_1d_config(device, m, k, n, fused_activation=None):
    grid = device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    mt, kt, nt = math.ceil(m / 32), math.ceil(k / 32), math.ceil(n / 32)
    if mt == 1:
        # Decode: ibw=8 flips greedy argmax; pcn from decode sweep.
        ibw = next(b for b in (4, 2, 1) if kt % b == 0)
        pcn = 3 if nt <= 32 else (4 if nt <= 96 else 6)
        osw = 2 if pcn % 2 == 0 else 1
        ncols = math.ceil(nt / pcn)
        cx = min(gx, ncols)
        cy = math.ceil(ncols / cx)
        if cy > gy:
            cy = gy
            cx = min(gx, math.ceil(ncols / cy))
            pcn = math.ceil(nt / (cx * cy))
            osw = 2 if pcn % 2 == 0 else 1
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(cx, cy),
            in0_block_w=ibw,
            out_subblock_h=1,
            out_subblock_w=osw,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=True,
            fused_activation=fused_activation,
            mcast_in0=True,
        )
    # Prefill: keep ibw=8 with fused GELU (ibw=4 regresses).
    ibw = next(b for b in (8, 4, 2, 1) if kt % b == 0)
    if mt < gy:
        cx = cy = pcn = None
        for trial_pcn in (1, 2, 3, 4, 6, 8, 12, 16):
            ncols = math.ceil(nt / trial_pcn)
            placed = _fit_core_grid(ncols, gx, gy)
            if placed is None:
                continue
            cx, cy = placed
            pcn = math.ceil(nt / (cx * cy))
            break
        if cx is not None:
            osw = next(w for w in (4, 2, 1) if pcn % w == 0)
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(cx, cy),
                in0_block_w=ibw,
                out_subblock_h=1,
                out_subblock_w=osw,
                per_core_M=mt,
                per_core_N=pcn,
                fuse_batch=True,
                fused_activation=fused_activation,
                mcast_in0=True,
            )
    pcm = max(1, math.ceil(mt / gy))
    pcn = max(1, math.ceil(nt / gx))
    osw = next(w for w in (4, 2, 1) if pcn % w == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=osw,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


class TtXttsGptBlock(LightweightModule):
    def __init__(
        self,
        state_dict,
        device,
        layer_idx=0,
    ):
        super().__init__()
        self.device = device
        self.layer_idx = layer_idx

        prefix = f"gpt.gpt.h.{layer_idx}."

        self.ln_1_weight = _to_device(state_dict[prefix + "ln_1.weight"], device)
        self.ln_1_bias = _to_device(state_dict[prefix + "ln_1.bias"], device)
        self.ln_2_weight = _to_device(state_dict[prefix + "ln_2.weight"], device)
        self.ln_2_bias = _to_device(state_dict[prefix + "ln_2.bias"], device)

        def _w(name):
            dtype = ttnn.bfloat4_b if name in _BFP4_WEIGHTS else ttnn.bfloat8_b
            return _to_device_w8(state_dict[prefix + name], device, dtype=dtype)

        self.attn_c_attn_weight = _w("attn.c_attn.weight")
        self.attn_c_attn_bias = _to_device_bias(state_dict[prefix + "attn.c_attn.bias"], device)
        self.attn_c_proj_weight = _w("attn.c_proj.weight")
        self.attn_c_proj_bias = _to_device_bias(state_dict[prefix + "attn.c_proj.bias"], device)

        self.mlp_c_fc_weight = _w("mlp.c_fc.weight")
        self.mlp_c_fc_bias = _to_device_bias(state_dict[prefix + "mlp.c_fc.bias"], device)
        self.mlp_c_proj_weight = _w("mlp.c_proj.weight")
        self.mlp_c_proj_bias = _to_device_bias(state_dict[prefix + "mlp.c_proj.bias"], device)

    def _qkv(self, x):
        qkv = ttnn.linear(
            x,
            self.attn_c_attn_weight,
            bias=self.attn_c_attn_bias,
            program_config=_mm_1d_config(self.device, x.shape[-2], x.shape[-1], self.attn_c_attn_weight.shape[-1]),
            memory_config=L1,
        )
        b, s, three_h = qkv.shape
        qkv = ttnn.reshape(qkv, (b, 1, s, three_h))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=NUM_HEADS, transpose_k_heads=False, memory_config=L1
        )
        ttnn.deallocate(qkv)
        return q, k, v

    def _attn_out(self, attn, shard_out=False):
        out = ttnn.transformer.concatenate_heads(attn, memory_config=L1)
        ttnn.deallocate(attn)
        m, k, n = out.shape[-2], out.shape[-1], self.attn_c_proj_weight.shape[-1]
        if shard_out:
            grid = self.device.compute_with_storage_grid_size()
            mt = math.ceil(m / 32)
            if mt < int(grid.y):
                mem, _ = _prefill_ln_cfg(self.device, mt)
                nc = _PREFILL_LN_MAX_CORES
                placed = _fit_core_grid(nc, int(grid.x), int(grid.y))
                gx, gy = placed
                nt = math.ceil(n / 32)
                pcn = math.ceil(nt / (gx * gy))
                kt = math.ceil(k / 32)
                ibw = next(b for b in (8, 4, 2, 1) if kt % b == 0)
                osw = next(w for w in (4, 2, 1) if pcn % w == 0)
                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=ibw,
                    out_subblock_h=1,
                    out_subblock_w=osw,
                    per_core_M=mt,
                    per_core_N=pcn,
                    fuse_batch=True,
                    fused_activation=None,
                    mcast_in0=True,
                )
                proj = ttnn.linear(
                    out, self.attn_c_proj_weight, bias=self.attn_c_proj_bias, program_config=pc, memory_config=mem
                )
                ttnn.deallocate(out)
                return proj
        proj = ttnn.linear(
            out,
            self.attn_c_proj_weight,
            bias=self.attn_c_proj_bias,
            program_config=_mm_1d_config(self.device, m, k, n),
            memory_config=L1,
        )
        ttnn.deallocate(out)
        return proj

    def _mlp(self, x, decode=False):
        h = ttnn.linear(
            x,
            self.mlp_c_fc_weight,
            bias=self.mlp_c_fc_bias,
            program_config=_mm_1d_config(
                self.device,
                x.shape[-2],
                x.shape[-1],
                self.mlp_c_fc_weight.shape[-1],
                fused_activation=(ttnn.UnaryOpType.GELU, False),
            ),
            memory_config=L1,
        )
        ttnn.deallocate(x)
        if decode:
            out = ttnn.linear(
                h,
                self.mlp_c_proj_weight,
                bias=self.mlp_c_proj_bias,
                program_config=_mm_1d_config(self.device, h.shape[-2], h.shape[-1], self.mlp_c_proj_weight.shape[-1]),
                memory_config=L1,
            )
            ttnn.deallocate(h)
            return out
        grid = self.device.compute_with_storage_grid_size()
        mt = math.ceil(h.shape[-2] / 32)
        if mt < int(grid.y):
            in_mc, out_mc, pc = _prefill_mlp_cproj_cfg(self.device, h.shape[-2])
            hs = ttnn.to_memory_config(h, in_mc)
            ttnn.deallocate(h)
            out = ttnn.linear(
                hs, self.mlp_c_proj_weight, bias=self.mlp_c_proj_bias, program_config=pc, memory_config=out_mc
            )
            ttnn.deallocate(hs)
            return out
        out = ttnn.linear(
            h,
            self.mlp_c_proj_weight,
            bias=self.mlp_c_proj_bias,
            program_config=_mm_1d_config(self.device, h.shape[-2], h.shape[-1], self.mlp_c_proj_weight.shape[-1]),
            memory_config=L1,
        )
        ttnn.deallocate(h)
        return out

    def _ln(self, x, weight, bias):
        return sharded_decode_ln(x, weight, bias, self.device)

    def _residual_ffn(self, x, decode=False):
        h = (
            self._ln(x, self.ln_2_weight, self.ln_2_bias)
            if decode
            else sharded_prefill_ln(x, self.ln_2_weight, self.ln_2_bias, self.device)
        )
        m = self._mlp(h, decode=decode)
        if decode or not m.is_sharded():
            y = ttnn.add(x, m, memory_config=L1)
        else:
            y = ttnn.add(x, m, memory_config=m.memory_config())
        ttnn.deallocate(x)
        ttnn.deallocate(m)
        return y

    def forward_prefill(self, x):
        h = sharded_prefill_ln(x, self.ln_1_weight, self.ln_1_bias, self.device)
        q, k, v = self._qkv(h)
        ttnn.deallocate(h)
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, memory_config=L1)
        ttnn.deallocate(q)
        ao = self._attn_out(attn, shard_out=True)
        xa = ttnn.add(x, ao, memory_config=ao.memory_config() if ao.is_sharded() else L1)
        ttnn.deallocate(x)
        ttnn.deallocate(ao)
        return self._residual_ffn(xa), k, v

    def forward_decode(self, x, k_cache, v_cache, onehot, add_mask, write_idx=None):
        h = self._ln(x, self.ln_1_weight, self.ln_1_bias)
        q, k, v = self._qkv(h)
        ttnn.deallocate(h)
        if write_idx is not None:
            ttnn.update_cache(k_cache, k, write_idx)
            ttnn.update_cache(v_cache, v, write_idx)
        else:
            ttnn.where(onehot, k, k_cache, output_tensor=k_cache)
            ttnn.where(onehot, v, v_cache, output_tensor=v_cache)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k_cache, v_cache, attn_mask=add_mask, is_causal=False, scale=1.0 / math.sqrt(HEAD_DIM), memory_config=L1
        )
        ttnn.deallocate(q)
        ao = self._attn_out(attn)
        xa = ttnn.add(x, ao, memory_config=L1)
        ttnn.deallocate(x)
        ttnn.deallocate(ao)
        return self._residual_ffn(xa, decode=True)
