# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_hifigan import (
    LRELU_SLOPE,
    RESBLOCK_DILATION_SIZES,
    RESBLOCK_KERNEL_SIZES,
    UPSAMPLE_RATES,
    get_padding,
)
from models.experimental.xtts.tt.xtts_conv import (
    TtConv1d,
    TtConvTranspose1d,
    block_chain_fits_l1,
    block_shard_l1,
    height_shard_l1,
    sharded_chain_fits_l1,
)

from models.experimental.xtts.config import FINAL_LRELU_SLOPE, TILE  # noqa: F401 — re-exported

_SHARD_RESBLOCKS = True

_BLOCK_SHARD_STAGES = {0}

_CONV_FIDELITY = ttnn.MathFidelity.HiFi2

_INTERLEAVED_CONV_DB = {"enable_act_double_buffer": True, "enable_weights_double_buffer": True}

# Do NOT force HEIGHT on ups[0]: HEIGHT_SHARDED fails the DRAM slicer on this shape.
_UPS_SHARD_OVERRIDE = {0: ttnn.TensorMemoryLayout.BLOCK_SHARDED}


def _ups_conv_overrides(i):
    """Build upsample conv config overrides for stage i."""
    ov = dict(_INTERLEAVED_CONV_DB)
    if i in _UPS_SHARD_OVERRIDE:
        ov["shard_layout"] = _UPS_SHARD_OVERRIDE[i]
    return ov


_COND_MM_CFG = {
    512: (8, 1, 2, 2, False, "HiFi2"),
    256: (8, 1, 1, 1, False, "HiFi2"),
    128: (4, 1, 1, 1, False, "HiFi2"),
    64: (2, 1, 1, 1, False, "HiFi2"),
    32: (1, 1, 1, 1, False, "HiFi2"),
}


class TtCondProj(LightweightModule):
    def __init__(self, device, weight, bias, dtype=ttnn.float32):
        """Load a 1x1 speaker conditioning projection matmul."""
        super().__init__()
        self.device = device
        self.dtype = dtype
        out_ch, in_ch, k = weight.shape
        assert k == 1, f"cond proj expects a 1x1 conv weight, got k={k}"
        assert out_ch in _COND_MM_CFG, f"no tuned cond-matmul config for N={out_ch}"
        assert in_ch % TILE == 0, f"K={in_ch} must be tile-aligned"
        self.n = out_ch
        self.k = in_ch
        w = weight.squeeze(-1).transpose(0, 1).contiguous()
        self.tt_weight = ttnn.from_torch(
            w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.tt_bias = None
        if bias is not None:
            self.tt_bias = ttnn.from_torch(
                bias.reshape(1, -1).float(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        gx, gy, per_core_N, osw, fp32_acc, fid = _COND_MM_CFG[out_ch]
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            in0_block_w=in_ch // TILE,
            out_subblock_h=1,
            out_subblock_w=osw,
            out_block_h=1,
            out_block_w=per_core_N,
            per_core_M=1,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=False,
            fp32_dest_acc_en=fp32_acc,
            packer_l1_acc=True,
        )

    def forward(self, g_mm):
        """Project speaker embedding to conditioning channels."""
        return ttnn.linear(
            g_mm,
            self.tt_weight,
            bias=self.tt_bias,
            program_config=self.program_config,
            compute_kernel_config=self.compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
        )


_FUSED_PRE_ACT_MIN_STAGE = 0

_SHARDED_STAGE_HANDOFF = True


def _is_l1_clash(exc):
    """Return whether an exception indicates an L1 CB clash."""
    msg = str(exc).lower()
    return "circular buffer" in msg or "clash" in msg


def _fused_pre_act_plan(stage_i, sharded):
    """Decide whether to fuse pre-activation into residual add."""
    return sharded and stage_i >= _FUSED_PRE_ACT_MIN_STAGE


def _shard_plan(stage_i, kernel_size):
    """Decide sharding and act double-buffer for a resblock."""
    if not _SHARD_RESBLOCKS:
        return False, True
    if stage_i in _BLOCK_SHARD_STAGES:
        return True, True
    if kernel_size <= 3:
        return True, True
    return stage_i >= 1, False


class TtResBlock1(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        prefix,
        kernel_size,
        dilation,
        activations_dtype=ttnn.float32,
        sharded=False,
        act_double_buffer=True,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        conv_config_overrides=None,
        fused_pre_act=False,
        block_shard=False,
    ):
        """Build dilated residual conv pairs for one MRF branch."""
        super().__init__()
        self.device = device
        self.block_shard = block_shard
        self.fused_pre_act = fused_pre_act
        self._pre_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, LRELU_SLOPE)
        self.sharded = sharded
        self._blocked_lengths = set()
        adb = act_double_buffer if sharded else None
        mid_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, LRELU_SLOPE)
        self.convs1 = [
            TtConv1d(
                device,
                state_dict[f"{prefix}convs1.{j}.weight"],
                state_dict[f"{prefix}convs1.{j}.bias"],
                padding=get_padding(kernel_size, d),
                dilation=d,
                activation=mid_act,
                activations_dtype=activations_dtype,
                act_double_buffer=adb,
                math_fidelity=math_fidelity,
                conv_config_overrides=conv_config_overrides,
            )
            for j, d in enumerate(dilation)
        ]
        self.convs2 = [
            TtConv1d(
                device,
                state_dict[f"{prefix}convs2.{j}.weight"],
                state_dict[f"{prefix}convs2.{j}.bias"],
                padding=get_padding(kernel_size, 1),
                dilation=1,
                activations_dtype=activations_dtype,
                act_double_buffer=adb,
                math_fidelity=math_fidelity,
                conv_config_overrides=conv_config_overrides,
            )
            for j in range(len(dilation))
        ]

    def shard(self, x, channels):
        """Shard activations height- or block-wise in L1."""
        return (block_shard_l1 if self.block_shard else height_shard_l1)(self.device, x, channels)

    def chain_fits_l1(self, length, channels):
        """Return whether the sharded residual chain fits L1."""
        return (block_chain_fits_l1 if self.block_shard else sharded_chain_fits_l1)(self.device, length, channels)

    def will_shard(self, length, channels):
        """Return whether this length will use the sharded path."""
        return self.sharded and length not in self._blocked_lengths and self.chain_fits_l1(length, channels)

    def forward(self, x, pre_act=None):
        """Run residual block, falling back if L1 clashes."""
        length = x.shape[1]
        if self.sharded and length not in self._blocked_lengths and self.chain_fits_l1(length, x.shape[2]):
            try:
                return self._forward_sharded(x, pre_act=pre_act)
            except RuntimeError as e:
                if "circular buffer" not in str(e).lower() and "clash" not in str(e).lower():
                    raise
                self._blocked_lengths.add(length)
        return self._forward_interleaved(x, pre_act=pre_act)

    def _forward_interleaved(self, x, pre_act=None):
        """Run residual block on interleaved DRAM activations."""
        for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            if idx == 0 and pre_act is not None:
                b = c1(pre_act)
            else:
                a = ttnn.leaky_relu(x, negative_slope=LRELU_SLOPE)
                b = c1(a)
                ttnn.deallocate(a)
            d = c2(b)
            ttnn.deallocate(b)
            nxt = ttnn.add(d, x)
            ttnn.deallocate(d)
            if idx > 0:
                ttnn.deallocate(x)
            x = nxt
        return x

    def _forward_sharded(self, x, return_sharded=False, pre_sharded=False, pre_act=None):
        """Run residual block on sharded L1 activations."""
        _, length, channels = x.shape
        b = d = nxt = None
        xs = x if pre_sharded else self.shard(x, channels)
        entry = xs if pre_sharded else None
        act, own_act = pre_act, False
        n_iters = len(self.convs1)
        try:
            for idx, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
                if act is None:
                    act = ttnn.leaky_relu(xs, negative_slope=LRELU_SLOPE)
                    own_act = True
                b = c1(act, keep_sharded=True)
                if own_act:
                    ttnn.deallocate(act)
                act, own_act = None, False
                d = c2(b, keep_sharded=True)
                ttnn.deallocate(b)
                b = None
                nxt = ttnn.add(d, xs)
                if self.fused_pre_act and idx + 1 < n_iters:
                    act = ttnn.add(d, xs, activations=[self._pre_act])
                    own_act = True
                ttnn.deallocate(d)
                d = None
                if xs is not entry:
                    ttnn.deallocate(xs)
                xs = nxt
                nxt = None
        except Exception:
            for t in (b, d, nxt, act if own_act else None):
                if isinstance(t, ttnn.Tensor) and t.is_allocated():
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            if xs is not entry and isinstance(xs, ttnn.Tensor) and xs.is_allocated():
                try:
                    ttnn.deallocate(xs)
                except Exception:
                    pass
            raise
        if return_sharded:
            return xs
        out = ttnn.to_memory_config(xs, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xs)
        return out


class TtHifiganGenerator(LightweightModule):
    def __init__(self, device, state_dict):
        """Load HiFi-GAN upsamplers, residuals, and cond layers."""
        super().__init__()
        self.device = device
        self.num_kernels = len(RESBLOCK_KERNEL_SIZES)
        self.num_upsamples = len(UPSAMPLE_RATES)
        self.inv_num_kernels = 1.0 / self.num_kernels

        self.conv_pre = TtConv1d(
            device,
            state_dict["conv_pre.weight"],
            state_dict["conv_pre.bias"],
            padding=3,
            activations_dtype=ttnn.bfloat16,
            math_fidelity=_CONV_FIDELITY,
            conv_config_overrides=_INTERLEAVED_CONV_DB,
        )
        self.cond_layer = TtCondProj(
            device, state_dict["cond_layer.weight"], state_dict["cond_layer.bias"], dtype=ttnn.bfloat16
        )

        self.ups = [
            TtConvTranspose1d(
                device,
                state_dict[f"ups.{i}.weight"],
                state_dict[f"ups.{i}.bias"],
                stride=UPSAMPLE_RATES[i],
                activations_dtype=ttnn.bfloat16,
                math_fidelity=_CONV_FIDELITY,
                weight_scale=self.inv_num_kernels if i >= 1 else 1.0,
                conv_config_overrides=_ups_conv_overrides(i),
            )
            for i in range(self.num_upsamples)
        ]
        self.conds = [
            TtCondProj(device, state_dict[f"conds.{i}.weight"], state_dict[f"conds.{i}.bias"])
            for i in range(self.num_upsamples)
        ]
        self._g_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
                [TILE, self.cond_layer.k],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        self._cond = {}

        self.resblocks = []
        for i in range(self.num_upsamples):
            for j, (k, d) in enumerate(zip(RESBLOCK_KERNEL_SIZES, RESBLOCK_DILATION_SIZES)):
                sharded, act_double_buffer = _shard_plan(i, k)
                db_ov = None if (sharded and i not in _BLOCK_SHARD_STAGES) else _INTERLEAVED_CONV_DB
                self.resblocks.append(
                    TtResBlock1(
                        device,
                        state_dict,
                        f"resblocks.{i * self.num_kernels + j}.",
                        k,
                        d,
                        activations_dtype=ttnn.bfloat16,
                        sharded=sharded,
                        act_double_buffer=act_double_buffer,
                        math_fidelity=_CONV_FIDELITY,
                        conv_config_overrides=db_ov,
                        fused_pre_act=_fused_pre_act_plan(i, sharded),
                        block_shard=i in _BLOCK_SHARD_STAGES,
                    )
                )

        self.conv_post = TtConv1d(
            device,
            state_dict["conv_post.weight"],
            None,
            padding=3,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH),
            activations_dtype=ttnn.bfloat16,
            math_fidelity=_CONV_FIDELITY,
            weight_scale=self.inv_num_kernels,
            conv_config_overrides=_INTERLEAVED_CONV_DB,
        )
        self._pre_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, LRELU_SLOPE)
        self._final_act = ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, FINAL_LRELU_SLOPE)

    def _conditioning(self, g):
        """Cache global and per-stage speaker conditioning biases."""
        hit = self._cond.get(id(g))
        if hit is not None and hit[0] is g:
            return hit[1], hit[2]
        g_rm = ttnn.reshape(g, [1, g.shape[-1]])
        g_pad = ttnn.pad(g_rm, [(0, TILE - 1), (0, 0)], 0.0)
        g_tiled = ttnn.tilize(g_pad)
        ttnn.deallocate(g_pad)
        g_mm = ttnn.slice(g_tiled, [0, 0], [1, g.shape[-1]])
        ttnn.deallocate(g_tiled)
        g_l1 = ttnn.to_memory_config(g_mm, self._g_mem_config)
        ttnn.deallocate(g_mm)
        cond_global = ttnn.reshape(self.cond_layer(g_l1), [1, 1, self.cond_layer.n])
        cond_biases = [ttnn.reshape(c(g_l1), [1, 1, 1, c.n]) for c in self.conds]
        ttnn.deallocate(g_l1)
        self._cond[id(g)] = (g, cond_global, cond_biases)
        return cond_global, cond_biases

    def release_conditioning(self):
        """Free cached conditioning tensors and upsample caches."""
        for entry in self._cond.values():
            for t in (entry[1], *entry[2]):
                if t.is_allocated():
                    ttnn.deallocate(t)
        self._cond.clear()
        for u in self.ups:
            u.release_cond_cache()

    def _mrf(self, i, o, post_act):
        """Sum multi-receptive-field residual branches for a stage."""
        nk = self.num_kernels
        rbs = self.resblocks[i * nk : (i + 1) * nk]
        length, channels = o.shape[1], o.shape[2]
        if all(rb.will_shard(length, channels) for rb in rbs):
            try:
                out = self._mrf_sharded(rbs, o, post_act, keep_sharded=_SHARDED_STAGE_HANDOFF)
                ttnn.deallocate(o)
                return out
            except RuntimeError as e:
                if "circular buffer" not in str(e).lower() and "clash" not in str(e).lower():
                    raise
                for rb in rbs:
                    rb._blocked_lengths.add(length)
        if o.memory_config().is_sharded():
            gathered = ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(o)
            o = gathered
        out = self._mrf_interleaved(rbs, o, post_act)
        ttnn.deallocate(o)
        return out

    def _mrf_sharded(self, rbs, o, post_act, keep_sharded=False):
        """Run MRF residual sum on sharded activations."""
        channels = o.shape[2]
        o_shard = rbs[0].shard(o, channels)
        pre_act = z_sum = None
        try:
            pre_act = ttnn.leaky_relu(o_shard, negative_slope=LRELU_SLOPE)
            for n, rb in enumerate(rbs):
                try:
                    res = rb._forward_sharded(o_shard, return_sharded=True, pre_sharded=True, pre_act=pre_act)
                except Exception:
                    if isinstance(z_sum, ttnn.Tensor) and z_sum.is_allocated():
                        ttnn.deallocate(z_sum)
                    raise
                if z_sum is None:
                    z_sum = res
                else:
                    z_new = ttnn.add(z_sum, res, activations=[post_act] if n == len(rbs) - 1 else [])
                    ttnn.deallocate(z_sum)
                    ttnn.deallocate(res)
                    z_sum = z_new
        finally:
            ttnn.deallocate(o_shard)
            if isinstance(pre_act, ttnn.Tensor) and pre_act.is_allocated():
                ttnn.deallocate(pre_act)
        if keep_sharded:
            return z_sum
        out = ttnn.to_memory_config(z_sum, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(z_sum)
        return out

    def _mrf_interleaved(self, rbs, o, post_act):
        """Run MRF residual sum on interleaved activations."""
        length, channels = o.shape[1], o.shape[2]
        pre_act = z_sum = None
        for n, rb in enumerate(rbs):
            if rb.will_shard(length, channels):
                res = rb(o)
            else:
                if pre_act is None:
                    pre_act = ttnn.leaky_relu(o, negative_slope=LRELU_SLOPE)
                res = rb(o, pre_act=pre_act)
            if z_sum is None:
                z_sum = res
            else:
                z_new = ttnn.add(z_sum, res, activations=[post_act] if n == len(rbs) - 1 else [])
                ttnn.deallocate(z_sum)
                ttnn.deallocate(res)
                z_sum = z_new
        if pre_act is not None:
            ttnn.deallocate(pre_act)
        return z_sum

    def forward(self, x, g):
        """Generate waveform from upsampled latents and speaker emb."""
        cond_global, cond_biases = self._conditioning(g)
        pre = self.conv_pre(x)
        ttnn.deallocate(x)
        a = ttnn.add(pre, cond_global, activations=[self._pre_act])
        ttnn.deallocate(pre)

        for i in range(self.num_upsamples):
            try:
                o = self.ups[i](a, cond_bias=cond_biases[i])
            except RuntimeError as e:
                if not (_is_l1_clash(e) and a.memory_config().is_sharded()):
                    raise
                a_dram = ttnn.to_memory_config(a, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(a)
                a = a_dram
                o = self.ups[i](a, cond_bias=cond_biases[i])
            ttnn.deallocate(a)
            last = i == self.num_upsamples - 1
            a = self._mrf(i, o, self._final_act if last else self._pre_act)
        out = self.conv_post(a)
        ttnn.deallocate(a)
        return out
