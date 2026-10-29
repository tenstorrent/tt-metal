# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA (Kimi Delta Attention) layer for GLM-5.3-Flash on one device.

Prefill runs ``models.demos.glm53_flash.tt.ttkda.ttKDA`` (the vendored deepseek_v3_d_p/tt/kda driver) (chunked delta-rule kernels, proven
against the Kimi-K3 torch reference, which ``reference/check_kda_reference.py`` shows is the same
function as HF's ``Glm5NextTextLinearAttention`` with GLM weights at PCC 0.99999).

Decode (T=1, B users) is the design's v1 composite on the same device-owned state:

    S *= exp(g)[.., :, None]       (per-channel decay, g = -5 sigmoid(exp(A_log) (f + dt_bias)))
    kv = k^T S ; delta = beta (v - kv) ; S += k^T delta ; o = q S

with q, k L2-normalized per head and q additionally scaled by K^-0.5 (FLA semantics), followed by
the gated RMSNorm (sigmoid(g_b(g_a(x)))) and o_proj. Every elementwise step is a broadcast
multiply plus a reduction so no tile padding ever enters a contraction.

``GLM53_KDA_FUSED=1`` (design 8.1) runs the whole decode block as 5 ops: the packed qkv matmul,
two gate matmuls (``[f_a | g_a | b]`` then the block-diagonal ``[f_b, g_b, I]``), ONE
``custom_ops.kda_decode`` program (a model-local ``ttnn.generic_op``: conv + SiLU from the
device-owned history, L2 norms, gates, the per-key-channel delta-rule recurrence, the gated
RMSNorm; state and conv history updated in place; bf16 TILE output), then o_proj. The block path
keeps the conv history as tap-major TILES ``[kernel-1, B, Q+K+V]`` bf16 (``conv_tiled``) instead
of ttKDA's row-major ``[B, kernel-1, Q+K+V]``: the program reads whole tiles and shifts them in
place; ``prefill_forward`` converts at its boundary and ``user_conv_state`` /
``write_user_conv_state`` / ``conv_history_torch`` / ``upload_conv_history`` serve the per-user
prefill hand-offs and the tests.

``kda_decode`` roofline knobs (``_agents/kda_roofline``; every default is the block path above, read once at
construction so a captured trace bakes them in):

* ``GLM53_KDA_DECODE_V2=1``: the bundle = ``GATES=act`` + ``PLACEMENT=rect``.
* ``GLM53_KDA_DECODE_GATES=act``: the channel constants (``exp(A_log)``, ``dt_bias``) are folded into the second gate
  matmul (``prearrange_gate_weights``: f columns scaled per head, an fp32 bias row ``a dt_bias``) and the sigmoid is
  fused into that matmul, so ``g2 = [sigmoid(a (f + dt)) | sigmoid(og) | sigmoid(b)]`` and the program skips its
  three sigmoid passes and the two channel-constant broadcasts (``KdaDecode(gates="act")``).
* ``GLM53_KDA_DECODE_PLACEMENT=rect``: 2-column slot rectangles when they give more parts (96 cores / 6 parts for
  16 local heads on a 12x10 grid instead of 80 / 5: 6 users per part at B = 32 instead of 7).
* ``GLM53_KDA_DECODE_FIDELITY=hifi2``: the program's FPU fidelity (default HiFi4; the state matmuls' precision).
* ``GLM53_KDA_MM_{QKV,G1,G2,O}=cx,cy,per_core_N,in0_block_w``: a 1D mcast program config for that projection
  (the sweep table of ``tt/matmul_configs.py`` otherwise).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.reference.kda import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict
from models.demos.glm53_flash.custom_ops.kda_decode.op import (
    GATES_MODES,
    PLACEMENTS,
    KdaDecode,
    kda_selector_tiles,
    prearrange_gate_weights,
)
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.matmul_configs import DecodeLinear
from models.demos.glm53_flash.tt.mesh import GLMMesh, from_torch_host_layout, interleave_blocks, single_chip
from models.demos.glm53_flash.tt.ttkda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.glm53_flash.tt.ttkda.kda import KdaState, ttKDA
from models.demos.glm53_flash.tt.ttkda.weights import KDAWeights, load_kda_weights

L2_NORM_EPS = 1e-6  # FLA l2norm: x / sqrt(sum(x^2) + 1e-6)
# decode path: "composite" (GLM53_KDA_FUSED=0) | "block" (GLM53_KDA_FUSED=1: one kda_decode program for the whole block)
_FUSED_MODES = {"0": "composite", "1": "block", "composite": "composite", "block": "block"}
KDA_FUSED_DEFAULT = _FUSED_MODES.get(os.environ.get("GLM53_KDA_FUSED", "1"), "block")  # block = design 8.1 (6 ops)
KDA_DECODE_FIDELITIES = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi2": ttnn.MathFidelity.HiFi2}
KDA_MM_NAMES = ("QKV", "G1", "G2", "O")


def kda_decode_options(env=None) -> dict:
    """The ``kda_decode`` program options from the environment (``gates``, ``placement``, ``fidelity``):
    ``GLM53_KDA_DECODE_V2=1`` selects the bundle (gates ``act`` + placement ``rect``), the per-option knobs
    ``GLM53_KDA_DECODE_{GATES,PLACEMENT,FIDELITY}`` override it; unset = today's path."""
    env = os.environ if env is None else env
    v2 = env.get("GLM53_KDA_DECODE_V2", "0") == "1"
    opts = dict(
        gates=env.get("GLM53_KDA_DECODE_GATES", "act" if v2 else "pre").lower(),
        placement=env.get("GLM53_KDA_DECODE_PLACEMENT", "rect" if v2 else "col").lower(),
        fidelity=env.get("GLM53_KDA_DECODE_FIDELITY", "hifi4").lower(),
    )
    if opts["gates"] not in GATES_MODES:
        raise ValueError(f"GLM53_KDA_DECODE_GATES={opts['gates']!r}: expected one of {GATES_MODES}")
    if opts["placement"] not in PLACEMENTS:
        raise ValueError(f"GLM53_KDA_DECODE_PLACEMENT={opts['placement']!r}: expected one of {PLACEMENTS}")
    if opts["fidelity"] not in KDA_DECODE_FIDELITIES:
        raise ValueError(
            f"GLM53_KDA_DECODE_FIDELITY={opts['fidelity']!r}: expected one of {tuple(KDA_DECODE_FIDELITIES)}"
        )
    return opts


def kda_mm_override(name: str, env=None) -> tuple[int, int, int, int] | None:
    """``GLM53_KDA_MM_<NAME>=cx,cy,per_core_N,in0_block_w`` (NAME in ``KDA_MM_NAMES``) -> the tuple, else ``None``."""
    env = os.environ if env is None else env
    assert name in KDA_MM_NAMES, name
    raw = env.get(f"GLM53_KDA_MM_{name}")
    if not raw:
        return None
    vals = tuple(int(v) for v in raw.split(","))
    if len(vals) != 4 or min(vals) < 1:
        raise ValueError(f"GLM53_KDA_MM_{name}={raw!r}: expected cx,cy,per_core_N,in0_block_w (positive)")
    return vals


def mcast1d_config(cx: int, cy: int, pcn: int, ibw: int, activation=None):
    """A 1D ``mcast_in0`` program config (the shape ``tt/matmul_configs.py`` builds) from the four knob values."""
    sub_w = max(d for d in (1, 2, 4) if pcn % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cx, cy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=sub_w,
        out_block_h=1,
        out_block_w=pcn,
        per_core_M=1,
        per_core_N=pcn,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=True,
    )


def _exact_compute(device):
    """HiFi4 + fp32 accumulation: a one-hot matmul is then exact (used for the per-user state reads)."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


class BiasedDecodeLinear(DecodeLinear):
    """``DecodeLinear`` with a persistent bias (``[1, 1, 1, N]`` TILE) and the fused activation of its decode
    config; the default dispatch (M > 32) applies the same activation as a composite."""

    _ACT_NAMES = {ttnn.UnaryOpType.SIGMOID: "sigmoid"}

    def __init__(self, device, weight: ttnn.Tensor, bias: ttnn.Tensor, activation=None):
        super().__init__(device, weight, activation=activation)
        self.bias = bias
        self._act_name = self._ACT_NAMES.get(activation) if activation is not None else None

    def __call__(self, x: ttnn.Tensor, **kw) -> ttnn.Tensor:
        shape = x.shape
        if len(shape) == 4 and shape[0] == 1 and shape[1] == 1 and shape[-2] <= 32 and "program_config" not in kw:
            return ttnn.linear(x, self.w, bias=self.bias, program_config=self.config(), **kw)
        return ttnn.linear(x, self.w, bias=self.bias, activation=self._act_name, **kw)


def decode_linear_for(device, weight: ttnn.Tensor, name: str, *, activation=None, bias=None, env=None):
    """``DecodeLinear`` (``BiasedDecodeLinear`` with ``bias``) of one projection whose program config honours the
    ``GLM53_KDA_MM_<name>`` override when set (the cached config of ``DecodeLinear.config``)."""
    lin = (
        BiasedDecodeLinear(device, weight, bias, activation=activation)
        if bias is not None
        else DecodeLinear(device, weight, activation=activation)
    )
    override = kda_mm_override(name, env)
    if override is not None:
        lin._cfg = mcast1d_config(*override, activation=activation)
    return lin


def kda_config(cfg: GLM53FlashConfig) -> KDAConfig:
    return KDAConfig(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.kda_num_heads,
        head_k_dim=cfg.kda_head_dim,
        head_v_dim=cfg.kda_head_dim,
        conv_kernel_size=cfg.kda_conv_kernel,
        norm_eps=cfg.rms_norm_eps,
        use_full_rank_gate=False,
        gate_lower_bound=cfg.kda_gate_lower_bound,
    )


def device_scalar_u32(device, value: int) -> ttnn.Tensor:
    """Replicated UINT32 row-major [1,1,1,1] device scalar (ttKDA's actual_start / actual_end)."""
    return ttnn.from_torch(
        torch.tensor([[[[int(value)]]]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _fp32_compute(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _hifi2_compute(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _upload(device, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


class TtKDA(LightweightModule):
    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        attn_state_dict: dict[str, torch.Tensor],
        *,
        layer_idx: int,
        weight_cache_path: Path | None = None,
        weight_dtype=ttnn.bfloat8_b,
        mesh: GLMMesh | None = None,
        fused_decode: str | bool | None = None,
        decode_opts: dict | None = None,
    ):
        """``attn_state_dict``: the raw checkpoint tensors of ``self_attn.`` (prefix stripped).
        ``fused_decode``: "composite" | "block" (default: the ``GLM53_KDA_FUSED`` environment knob);
        ``True`` means "block". ``decode_opts``: overrides of ``kda_decode_options()`` (``gates`` /
        ``placement`` / ``fidelity`` of the block path's program; the environment otherwise).

        On a mesh the heads are split over the columns (``mesh.tp``): every projection that is
        per head is column-sharded head-major, the state holds the local heads, the decode output
        projection is row-parallel and all-reduced over the columns (ttKDA's prefill weights use
        the same placement through ``tensor_parallel_axis=1``)."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        tp = self.mesh.tp
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.kcfg = kda_config(cfg)
        assert self.kcfg.num_heads % tp == 0, (self.kcfg.num_heads, tp)
        self.H_total = self.kcfg.num_heads
        self.H, self.K, self.V = self.kcfg.num_heads // tp, self.kcfg.head_k_dim, self.kcfg.head_v_dim
        self.qkv_dim = self.kcfg.q_dim // tp  # local heads x K
        self.conv_width = (self.kcfg.q_dim + self.kcfg.k_dim + self.kcfg.v_dim) // tp  # [q_c | k_c | v_c]
        raw = normalize_kda_state_dict(attn_state_dict, self.kcfg)
        self.weight_cache_path = weight_cache_path
        # ---- prefill (ttKDA) weights, materialized once and shared across sequence lengths
        self.prefill_weights: KDAWeights = load_kda_weights(
            device,
            self.kcfg,
            raw,
            weight_cache_path,
            cache_name_prefix=f"layer_{layer_idx}.kda",
            tensor_parallel_axis=1,
        )
        self._prefill_layers: dict[tuple[int, bool], ttKDA] = {}  # (active_seq_len, local)
        # ---- decode weights (per-head tensors column-sharded head-major: [q_c | k_c | v_c] per chip)
        up = self.mesh.upload
        self.w_qkv = up(
            interleave_blocks([raw["q_proj.weight"].t(), raw["k_proj.weight"].t(), raw["v_proj.weight"].t()], tp),
            weight_dtype,
            cols=-1,
        )
        taps = [
            interleave_blocks(
                [raw["q_conv1d.weight"][:, 0, j], raw["k_conv1d.weight"][:, 0, j], raw["v_conv1d.weight"][:, 0, j]], tp
            ).reshape(1, 1, 1, -1)
            for j in range(self.kcfg.conv_kernel_size)
        ]
        self.taps = [up(t.float(), ttnn.bfloat16, cols=-1) for t in taps]
        self.w_fa = up(raw["f_a_proj.weight"].t(), ttnn.bfloat16)
        self.w_fb = up(raw["f_b_proj.weight"].t(), ttnn.bfloat16, cols=-1)
        self.w_ga = up(raw["g_a_proj.weight"].t(), ttnn.bfloat16)
        self.w_gb = up(raw["g_b_proj.weight"].t(), ttnn.bfloat16, cols=-1)
        self.w_b = up(raw["b_proj.weight"].t(), ttnn.bfloat16, cols=-1)
        a_exp = raw["A_log"].float().reshape(self.H_total, 1).exp().expand(self.H_total, self.K).reshape(1, 1, 1, -1)
        self.a_exp = up(a_exp, ttnn.float32, cols=-1)
        self.dt_bias = up(raw["dt_bias"].float().reshape(1, 1, 1, -1), ttnn.float32, cols=-1)
        self.o_norm_w = up(raw["o_norm.weight"].float().reshape(1, 1, 1, -1), ttnn.bfloat16)
        self.w_o = up(raw["o_proj.weight"].t(), weight_dtype, cols=-2)
        self.fp32 = _fp32_compute(device)
        self.hifi2 = _hifi2_compute(device)
        # decode-shaped (M <= 32) projections use the 1D mcast configs of tt/matmul_configs.py (GLM53_KDA_MM_* override)
        self.lin_qkv = decode_linear_for(device, self.w_qkv, "QKV")
        self.lin_o = decode_linear_for(device, self.w_o, "O")
        self.lin_fa = DecodeLinear(device, self.w_fa)
        self.lin_fb = DecodeLinear(device, self.w_fb)
        self.lin_ga = DecodeLinear(device, self.w_ga)
        self.lin_gb = DecodeLinear(device, self.w_gb)
        self.lin_b = DecodeLinear(device, self.w_b)
        self.gate_lower_bound = float(self.kcfg.gate_lower_bound)
        self._scalars: dict[int, ttnn.Tensor] = {}
        self._const_cache: dict = {}
        self.chrono_scalar(0, "start")  # allocate the mesh's chronology pair before any trace capture
        self.chrono_scalar(0, "end")
        # the traced prefill's constant start (functional_decoder._prefill_forward_traced): allocated here so a
        # harness that captures the decode trace before the prefill traces (tests/test_mesh_model.py) never allocates
        # it after a capture (the e2e_8x4_B4_h1h2_default run, 2026-10-29); serving captures the prefill traces first
        self.scalar(0)
        if fused_decode is None:
            self.fused_decode = KDA_FUSED_DEFAULT
        elif isinstance(fused_decode, bool):
            self.fused_decode = "block" if fused_decode else "composite"
        else:
            self.fused_decode = _FUSED_MODES[str(fused_decode)]
        opts = dict(kda_decode_options())
        opts.update(decode_opts or {})
        self.decode_opts = opts
        K, V, H = self.K, self.V, self.H
        if self.fused_decode == "block":
            # the kda_decode program's constants (its tensor contract is in the op's docstring)
            self.taps_blk = up(
                torch.stack([t.reshape(-1).float() for t in taps]).reshape(1, 1, 4, -1), ttnn.bfloat16, cols=-1
            )  # the conv weight is bf16 in the checkpoint: exact
            self.chan_consts = up(
                torch.stack([raw["dt_bias"].float().reshape(-1), a_exp.reshape(-1)]).reshape(1, 1, 2, -1),
                ttnn.float32,
                cols=-1,
            )
            self.o_norm_w32 = up(raw["o_norm.weight"].float().reshape(1, 1, 1, -1), ttnn.float32)
            self.selectors = up(kda_selector_tiles(), ttnn.bfloat16)
            self.kda_decode = KdaDecode(
                num_heads=H,
                head_dim=K,
                gate_lower_bound=self.gate_lower_bound,
                l2_eps=L2_NORM_EPS,
                norm_eps=self.kcfg.norm_eps,
                math_fidelity=KDA_DECODE_FIDELITIES[opts["fidelity"]],
                gates=opts["gates"],
                placement=opts["placement"],
            )
            # gates in two matmuls: x -> [f_a | g_a | b_logits] (bf16, as HF), then the block-diagonal
            # [f_b, g_b, I] -> [f | og | b_logits] (fp32 accumulate); the identity block carries the logits
            Hb = -(-H // 32) * 32
            # per chip: [f_a | g_a | b columns of its heads | zero pad to Hb] (an even column split over the TP
            # columns hands every chip its own block)
            w_fa_t, w_ga_t, w_b_t = (raw[n].t() for n in ("f_a_proj.weight", "g_a_proj.weight", "b_proj.weight"))
            parts = [
                torch.cat([w_fa_t, w_ga_t, w_b_t[:, c * H : (c + 1) * H], torch.zeros(w_b_t.shape[0], Hb - H)], dim=1)
                for c in range(tp)
            ]
            self.w_g1 = up(torch.cat(parts, dim=1), ttnn.bfloat16, cols=-1)
            d_a = raw["f_a_proj.weight"].shape[0]
            # per TP column c: [f_b | g_b] columns of ITS heads (head-major, heads c*H .. (c+1)*H) and the identity
            # on its b logits; concatenated so the even column split hands every chip its own block
            fb_t = raw["f_b_proj.weight"].t().float()  # [d_a, H_total*K]
            gb_t = raw["g_b_proj.weight"].t().float()  # [d_a, H_total*V]
            gates_act = opts["gates"] == "act"
            a_flat, dt_flat = a_exp.reshape(-1), raw["dt_bias"].float().reshape(-1)  # [H_total*K] each
            blocks, biases = [], []
            for c in range(tp):
                w_g2 = torch.zeros(2 * d_a + Hb, H * K + H * V + Hb)
                w_g2[:d_a, : H * K] = fb_t[:, c * H * K : (c + 1) * H * K]
                w_g2[d_a : 2 * d_a, H * K : H * K + H * V] = gb_t[:, c * H * V : (c + 1) * H * V]
                w_g2[2 * d_a : 2 * d_a + H, H * K + H * V : H * K + H * V + H] = torch.eye(H)
                if gates_act:
                    # gates="act": a (f + dt) = (a w_fb)^T f_a + a dt per key channel of this chip's heads, the sigmoid
                    # fused into the matmul (fp32 bias; the scaled f columns round to bf16 like the checkpoint weight)
                    sl = slice(c * H * K, (c + 1) * H * K)
                    w_g2, bias = prearrange_gate_weights(w_g2, a_flat[sl], dt_flat[sl], H * K)
                    biases.append(bias)
                blocks.append(w_g2)
            self.w_g2 = up(torch.cat(blocks, dim=1).to(torch.bfloat16), ttnn.bfloat16, cols=-1)
            # decode-shaped (M <= 32) gate matmuls use the 1D mcast configs of tt/matmul_configs.py
            self.lin_g1 = decode_linear_for(device, self.w_g1, "G1")
            if gates_act:
                self.g2_bias = up(torch.cat(biases).reshape(1, 1, 1, -1), ttnn.float32, cols=-1)
                self.lin_g2 = decode_linear_for(
                    device, self.w_g2, "G2", activation=ttnn.UnaryOpType.SIGMOID, bias=self.g2_bias
                )
            else:
                self.g2_bias = None
                self.lin_g2 = decode_linear_for(device, self.w_g2, "G2")

    def scalar(self, value: int) -> ttnn.Tensor:
        """Cached replicated UINT32 device scalar for a CONSTANT the traced prefill keeps (``scalar(0)``). Never
        allocated after a trace capture: the 2026-10-27 served hang was scalar(49152) allocated after the captures and
        clobbered by ~1200 decode replays (ttKDA then saw an interval longer than its capacity and wedged the mesh).
        Per-chunk values go through ``chrono_scalar``."""
        if value not in self._scalars:
            if self.mesh.allocations_frozen[0]:
                raise RuntimeError(
                    f"KDA scalar({value}) would allocate a device constant after a trace capture (replays clobber it): "
                    "use chrono_scalar for per-chunk values or allocate it in the warm-up"
                )
            self._scalars[value] = self.mesh.upload(
                torch.tensor([[[[int(value)]]]], dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        return self._scalars[value]

    def chrono_scalar(self, value: int, role: str) -> ttnn.Tensor:
        """The eager prefill's per-chunk ``actual_start`` / ``actual_end``: ONE persistent replicated uint32 tensor per
        role, shared by every KDA layer of the mesh (``GLMMesh.chrono``; allocated by the first layer's construction,
        before any capture) and rewritten in place with ``copy_host_to_device_tensor`` (in order on the command queue
        ahead of the chunk's ops) only when the value changes: two tiny writes per chunk instead of a fresh buffer per
        new value. The traced prefill keeps its own ``trace.end`` tensors and ``scalar(0)``."""
        if role not in ("start", "end"):
            raise ValueError(f"chrono_scalar role {role!r} is not 'start' / 'end'")
        ch = self.mesh.chrono
        if "tensors" not in ch:
            if self.mesh.allocations_frozen[0]:
                raise RuntimeError("KDA chronology scalars must be allocated before the first trace capture")
            ch["tensors"] = {
                r: self.mesh.upload(
                    torch.zeros((1, 1, 1, 1), dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                for r in ("start", "end")
            }
            ch["values"] = {"start": 0, "end": 0}
        if ch["values"][role] != int(value):
            host = ttnn.from_torch(
                torch.tensor([[[[int(value)]]]], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.mesh.shard_2d(None, None),
            )
            ttnn.copy_host_to_device_tensor(host, ch["tensors"][role])
            ch["values"][role] = int(value)
        return ch["tensors"][role]

    def scalar_rows(self, values: tuple[int, ...], role: str) -> ttnn.Tensor:
        """A wave's per-row ``actual_start`` / ``actual_end`` (``doc/mesh/row_ownership.md``): one value per mesh row,
        ``[1, 1, 1, 1]`` uint32 on every chip of that row. ONE persistent row-sharded tensor per role on the mesh
        (``GLMMesh.chrono``, next to ``chrono_scalar``'s replicated pair), allocated on first use (before the first
        trace capture: the wave warm-up) and rewritten in place only when the values change."""
        if role not in ("wave_start", "wave_end"):
            raise ValueError(f"scalar_rows role {role!r} is not 'wave_start' / 'wave_end'")
        rows = self.mesh.rows
        vals = tuple(int(v) for v in values)
        if len(vals) != rows:
            raise ValueError(f"{len(vals)} per-row values for a {rows}-row mesh")
        ch = self.mesh.chrono
        if "row_tensors" not in ch:
            if self.mesh.allocations_frozen[0]:
                raise RuntimeError("KDA wave chronology scalars must be allocated before the first trace capture")
            ch["row_tensors"] = {
                r: self.mesh.upload(
                    torch.zeros((rows, 1, 1, 1), dtype=torch.int32), ttnn.uint32, rows=0, layout=ttnn.ROW_MAJOR_LAYOUT
                )
                for r in ("wave_start", "wave_end")
            }
            ch["row_values"] = {"wave_start": None, "wave_end": None}
        if ch["row_values"][role] != vals:
            host = ttnn.from_torch(
                torch.tensor(vals, dtype=torch.int32).reshape(rows, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.mesh.shard_rows(0),
            )
            ttnn.copy_host_to_device_tensor(host, ch["row_tensors"][role])
            ch["row_values"][role] = vals
        return ch["row_tensors"][role]

    # ------------------------------------------------------------------ state
    def allocate_state(self, batch_size: int = 1) -> KdaState:
        """Zero state for ``batch_size`` users (ttKDA layout)."""
        return KdaState(
            recurrent=ttnn.zeros(
                (batch_size, self.H, self.K, self.V),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            convolution=self._zero_conv(batch_size),
        )

    @property
    def conv_tiled(self) -> bool:
        """The block path keeps the conv history as tap-major tiles ``[kernel-1, B, W]`` bf16."""
        return self.fused_decode == "block"

    def _zero_conv(self, batch_size: int) -> ttnn.Tensor:
        taps = self.kcfg.conv_kernel_size - 1
        shape = (taps, batch_size, self.conv_width) if self.conv_tiled else (batch_size, taps, self.conv_width)
        return ttnn.zeros(
            shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT if self.conv_tiled else ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------- conv history layout helpers
    def conv_to_rm(self, conv: ttnn.Tensor) -> ttnn.Tensor:
        """Tap-major tiles ``[taps, B, W]`` -> ttKDA's ``[B, taps, W]`` bf16 ROW_MAJOR (a fresh tensor)."""
        rm = ttnn.to_layout(conv, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.permute(rm, [1, 0, 2])
        ttnn.deallocate(rm)
        return out

    def conv_to_tiles(self, conv_rm: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, taps, W]`` bf16 ROW_MAJOR -> tap-major tiles ``[taps, B, W]`` (a fresh tensor)."""
        t = ttnn.permute(conv_rm, [1, 0, 2])
        out = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
        ttnn.deallocate(t)
        return out

    def conv_history_torch(self, state: KdaState) -> torch.Tensor:
        """The conv history as ``[B, taps, W]`` on the host, whatever the device layout."""
        t = ttnn.to_torch(state.convolution)
        return t.permute(1, 0, 2).contiguous() if state.convolution.layout == ttnn.TILE_LAYOUT else t

    def upload_conv_history(self, state: KdaState, hist: torch.Tensor):
        """Write a host ``[B, taps, W]`` history into ``state.convolution`` (its layout)."""
        if state.convolution.layout == ttnn.TILE_LAYOUT:
            src = ttnn.from_torch(hist.permute(1, 0, 2).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        else:
            src = ttnn.from_torch(hist, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(src, state.convolution)

    def user_conv_state(self, state: KdaState, user: int) -> ttnn.Tensor:
        """One user's conv history as ttKDA's ``[1, taps, W]`` bf16 ROW_MAJOR (a fresh tensor)."""
        taps = self.kcfg.conv_kernel_size - 1
        if state.convolution.layout != ttnn.TILE_LAYOUT:
            return ttnn.slice(state.convolution, [user, 0, 0], [user + 1, taps, self.conv_width])
        # the tap-major tiles [taps, B, W] read at row ``user`` by an exact one-hot matmul [taps, 1, B] @ [taps, B, W]
        # (every product x * 1.0 or x * 0.0, fp32 accumulation): ONE program whatever ``user`` is. The earlier
        # untilize + slice at the user's index was a program per user, which the first continued chunk of a slot the
        # warm-up's continuation prompts had not used compiled under the sealed cache (2026-10-28 served APC, 8x4).
        row = self.user_conv_tiles(state, user)
        rm = ttnn.to_layout(row, ttnn.ROW_MAJOR_LAYOUT)  # [taps, 1, W]
        ttnn.deallocate(row)
        out = ttnn.permute(rm, [1, 0, 2])  # [1, taps, W]
        ttnn.deallocate(rm)
        return out

    def user_conv_tiles(self, state: KdaState, user: int) -> ttnn.Tensor:
        """One user's conv history as tap-major tiles ``[taps, 1, W]`` bf16 (a fresh tensor; the block path's
        batched layout): the one-hot matmul of ``user_conv_state``, which ``prefill_forward`` also accepts as the
        input history (B == 1) and returns as the carry (``conv_tiled``)."""
        taps = self.kcfg.conv_kernel_size - 1
        if state.convolution.layout != ttnn.TILE_LAYOUT:
            raise NotImplementedError("user_conv_tiles needs the block KDA decode (tiled conv history)")
        B = int(state.convolution.shape[1])
        sel = torch.zeros(taps, 1, B)
        sel[:, 0, user] = 1.0
        sel_dev = from_torch_host_layout(sel, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device)
        row = ttnn.matmul(
            sel_dev, state.convolution, dtype=ttnn.bfloat16, compute_kernel_config=_exact_compute(self.device)
        )
        ttnn.deallocate(sel_dev)
        return row

    def write_user_conv_state(self, state: KdaState, user: int, conv: ttnn.Tensor):
        """Write one user's history (``[taps, 1, W]`` tiles, as ``prefill_forward`` returns it in block mode) into
        slot ``user`` of the batched tap-major tile history: ``dst * keep + new * mask`` with host-built one-hot
        user masks ``[taps, B, 1]`` (every product is x * 1.0 or x * 0.0, exact in bf16: the traced prefill's
        blend, ``functional_decoder._write_user_kda_state_slot``). One program whatever ``user`` is; the earlier
        row-major slice / concat round trip built a slice program per user index, the first prefill into a slot
        the warm-up had not used compiling it at request time under live traces (the 2026-10-28 served seal hit)."""
        taps = self.kcfg.conv_kernel_size - 1
        B = int(state.convolution.shape[1])
        mask = torch.zeros(taps, B, 1)
        mask[:, user, 0] = 1.0
        up = lambda t: from_torch_host_layout(t, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device)  # no Tilize program
        mask_dev, keep_dev = up(mask), up(1.0 - mask)
        ttnn.multiply(state.convolution, keep_dev, output_tensor=state.convolution)  # zero the slot's rows
        new_rows = ttnn.multiply(conv, mask_dev)  # [taps,1,W] x [taps,B,1] -> [taps,B,W]
        ttnn.add(state.convolution, new_rows, output_tensor=state.convolution)
        for t in (new_rows, mask_dev, keep_dev):
            ttnn.deallocate(t)

    # ---------------------------------------------------------------- prefill
    def summary_group_chunks(self, local_chunks: int) -> int:
        """Chunks per summary group of the sequence-parallel grouped scan for ``local_chunks`` 32-row chunks per
        row: the largest divisor of ``local_chunks`` up to 20 (the Kimi-K3 production range, one serial scan of at
        most 20 chunks per owner core) whose group count keeps ``local heads x groups`` summary owners within
        the worker grid, else the smallest divisor that does (a 4 K-row slab = 128 chunks with 16 local heads
        needs 32: 4 groups x 16 heads = 64 owners; 16 would ask for 128 of the 120 cores).
        ``GLM53_KDA_GROUP_CHUNKS`` overrides (A/B of the group size)."""
        override = os.environ.get("GLM53_KDA_GROUP_CHUNKS")
        if override:
            g = int(override)
            if local_chunks % g:
                raise ValueError(f"GLM53_KDA_GROUP_CHUNKS={g} does not divide the local chunk count {local_chunks}")
            return g
        grid = self.device.compute_with_storage_grid_size()
        capacity = int(grid.x) * int(grid.y)
        divisors = [d for d in range(1, local_chunks + 1) if local_chunks % d == 0]
        fits = [d for d in divisors if (local_chunks // d) * self.H <= capacity]
        if not fits:
            raise ValueError(
                f"no summary group size fits {self.H} local heads x {local_chunks} chunks on {capacity} cores"
            )
        small = [d for d in fits if d <= 20]
        return max(small) if small else min(fits)

    def prefill_layer(self, active_seq_len: int, local: bool = False) -> ttKDA:
        """ttKDA instance for one physical (32-aligned) chunk length; weights are shared. ``local``: a row-local
        scan on a multi-row mesh (every row runs its own sequence of ``active_seq_len`` rows: a wave of per-row
        prompts, doc/mesh/galaxy_throughput_plan.md A1) through ttKDA's ``sequence_parallel_size=1``; the default
        is the sequence-parallel scan over the rows (one prompt, ``active_seq_len`` global rows)."""
        if active_seq_len % ttnn.TILE_SIZE:
            raise ValueError(f"active_seq_len must be a multiple of {ttnn.TILE_SIZE}, got {active_seq_len}")
        local = bool(local) and self.mesh.rows > 1
        layer = self._prefill_layers.get((active_seq_len, local))
        if layer is None:
            if self.mesh.allocations_frozen[0]:
                logger.warning(
                    f"KDA prefill layer for {active_seq_len} rows built after a trace capture: its device buffers can be "
                    "clobbered by replays; warm this chunk length up before capturing (bucket the chunk lengths)"
                )
            kw = {}
            sp_kw = {"sequence_parallel_size": 1} if local else {}
            if self.mesh.rows > 1 and not local:
                # sequence-parallel over the rows: the grouped scan needs summary_group_chunks to divide the
                # LOCAL chunk count (active_seq_len / rows / 32) and one summary-owner core per (local head,
                # group): heads * groups <= the worker grid (120 on P150); see summary_group_chunks
                local_chunks = active_seq_len // self.mesh.rows // ttnn.TILE_SIZE
                kw["recurrence"] = KDARecurrenceProgramConfig(
                    summary_group_chunks=self.summary_group_chunks(local_chunks)
                )
            program_config = KDAProgramConfig(
                gated_rms_output_dtype=ttnn.bfloat16,
                input_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
                output_projection_math_fidelity=ttnn.MathFidelity.HiFi2,
                **kw,
            )
            tt_ccl = None
            if self.mesh.tp > 1:
                try:  # tt-metal main keeps tt_ccl as models.common.tt_ccl (TTTv2 moved to tt-transformers)
                    from models.common.tt_ccl import get_tt_ccl
                except ImportError:  # the 2026-10 base still has it under models.common.modules
                    from models.common.modules.tt_ccl import get_tt_ccl

                tt_ccl = get_tt_ccl(self.device)  # ttKDA's TP reduce-scatter semaphores (one set per mesh)
            layer = ttKDA(
                self.device,
                self.kcfg,
                None,
                layer_idx=self.layer_idx,
                weights=self.prefill_weights,
                tt_ccl=tt_ccl,
                sp_axis=0,
                tp_axis=1,
                program_config=program_config,
                active_seq_len=active_seq_len,
                **sp_kw,
            )
            self._prefill_layers[(active_seq_len, local)] = layer
        return layer

    def prefill_forward(
        self,
        x: ttnn.Tensor,
        state: KdaState,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None = None,
        *,
        local: bool = False,
    ) -> tuple[ttnn.Tensor, KdaState]:
        """x: [1, T_phys, hidden] bf16 TILE (T_phys % 32 == 0; rows past ``actual_end`` are padding).
        ``actual_start`` / ``actual_end`` are CHUNK-RELATIVE device scalars (0 and the valid row count): ttKDA keys
        its sequence-parallel placement on them (block-cyclic by the absolute start), and the contiguous slab upload
        this model makes (rank r holds rows [r T/rows, (r+1) T/rows) of the chunk) is that placement only for a
        start of 0 mod T; the carried state supplies the history, so the absolute position is never needed.
        Returns y [1, T_phys, hidden] bf16 and the new state (the old one is only read).
        On a multi-row mesh ttKDA is sequence-parallel over the rows: ``x`` is this row's slab
        (row r holds rows [r*T/rows, (r+1)*T/rows) of the chunk, T = rows * T_phys_local) and the
        returned carries are replicated over the rows (TP-sharded heads as in decode). ``local``: every row
        runs its own chunk of ``x.shape[1]`` rows (a wave of per-row prompts; row-sharded ``actual_start`` /
        ``actual_end``) and returns its own carry."""
        layer = self.prefill_layer(x.shape[1] if local else x.shape[1] * self.mesh.rows, local=local)
        tiled_in = state.convolution.layout == ttnn.TILE_LAYOUT
        if tiled_in:  # ttKDA reads / writes the row-major [B, taps, W] history
            conv_rm = self.conv_to_rm(state.convolution)
            state = KdaState(recurrent=state.recurrent, convolution=conv_rm)
        y, new_state = layer.forward(x, state, actual_start, actual_end)
        if tiled_in:
            ttnn.deallocate(conv_rm)
        if self.conv_tiled:  # the returned carry in the block path's layout
            tiles = self.conv_to_tiles(new_state.convolution)
            ttnn.deallocate(new_state.convolution)
            new_state = KdaState(recurrent=new_state.recurrent, convolution=tiles)
        if self.mesh.tp > 1:
            # ttKDA reduce-scatters the TP partials on the hidden dim; make the output replicated
            y = self.mesh.all_gather_cols(y, dim=-1)
        return y, new_state

    # ----------------------------------------------------------------- decode
    def decode_forward(self, x: ttnn.Tensor, state: KdaState, *, inplace: bool = True) -> tuple[ttnn.Tensor, KdaState]:
        """x: [1, 1, B, hidden] bf16 TILE, one token per user. Returns y [1, 1, B, hidden] and the
        state after the token: ``inplace`` (default) updates ``state``'s tensors in place and returns
        the same object (trace-safe); otherwise returns a fresh ``KdaState``. Composite v1 (design 5.3)."""
        B = x.shape[-2]
        H, K, V = self.H, self.K, self.V
        BH = B * H
        if state.recurrent.shape[0] != B:
            raise ValueError(f"state batch {state.recurrent.shape[0]} != input batch {B}")
        if self.fused_decode == "block":
            return self._block_decode(x, state, inplace)

        # --- projections + causal conv (kernel 4) over the 3-row history + current token
        qkv = self.lin_qkv(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,3*QKV]
        hist = ttnn.to_layout(state.convolution, ttnn.TILE_LAYOUT)  # [B, 3, W]
        hist4 = ttnn.reshape(hist, [1, B, self.kcfg.conv_kernel_size - 1, self.conv_width])
        conv = ttnn.multiply(qkv, self.taps[3])
        for j in range(self.kcfg.conv_kernel_size - 1):
            row = ttnn.slice(hist4, [0, 0, j, 0], [1, B, j + 1, self.conv_width])  # [1,B,1,W]
            row = ttnn.reshape(row, [1, 1, B, self.conv_width])
            conv = ttnn.add(conv, ttnn.multiply(row, self.taps[j]), output_tensor=conv)
        conv = ttnn.silu(conv, output_tensor=conv)  # [1,1,B,3*QKV] bf16
        # new history = rows 1..2 of the old history + the raw (pre-conv) projection of this token
        qkv_rm = ttnn.reshape(ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT), [B, 1, self.conv_width])
        tail = ttnn.slice(state.convolution, [0, 1, 0], [B, self.kcfg.conv_kernel_size - 1, self.conv_width])
        new_conv = ttnn.concat([tail, qkv_rm], dim=1)  # [B, 3, W] row-major bf16
        ttnn.deallocate(qkv)
        ttnn.deallocate(hist)

        q = ttnn.slice(conv, [0, 0, 0, 0], [1, 1, B, self.qkv_dim])
        k = ttnn.slice(conv, [0, 0, 0, self.qkv_dim], [1, 1, B, 2 * self.qkv_dim])
        v = ttnn.slice(conv, [0, 0, 0, 2 * self.qkv_dim], [1, 1, B, 3 * self.qkv_dim])
        ttnn.deallocate(conv)

        # --- gates (fp32, as HF)
        f = self.lin_fa(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        f = self.lin_fb(f, dtype=ttnn.float32, compute_kernel_config=self.fp32)  # [1,1,B,H*K]
        f = ttnn.add(f, self.dt_bias, output_tensor=f)
        f = ttnn.multiply(f, self.a_exp, output_tensor=f)
        g = ttnn.sigmoid(f, output_tensor=f)
        g = ttnn.multiply(g, self.gate_lower_bound, output_tensor=g)  # [1,1,B,H*K] fp32 log decay
        decay = ttnn.exp(g, output_tensor=g)  # [1,1,B,H*K] fp32
        beta = ttnn.sigmoid(self.lin_b(x, dtype=ttnn.float32, compute_kernel_config=self.fp32))  # [1,1,B,H]
        og = self.lin_ga(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        og = self.lin_gb(og, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,H*V]

        # --- per-head L2 norm of q, k (FLA: x / sqrt(sum x^2 + 1e-6)); q also scaled by K^-0.5
        qh = ttnn.reshape(ttnn.typecast(q, ttnn.float32), [1, B, H, K])
        kh = ttnn.reshape(ttnn.typecast(k, ttnn.float32), [1, B, H, K])
        vh = ttnn.reshape(ttnn.typecast(v, ttnn.float32), [1, B, H, V])
        qh = ttnn.multiply(ttnn.rms_norm(qh, epsilon=L2_NORM_EPS / K, compute_kernel_config=self.fp32), 1.0 / K)
        kh = ttnn.multiply(ttnn.rms_norm(kh, epsilon=L2_NORM_EPS / K, compute_kernel_config=self.fp32), K**-0.5)

        # --- recurrence on S [1, B*H, K, V] fp32
        S = ttnn.reshape(state.recurrent, [1, BH, K, V])
        decay_col = ttnn.transpose(ttnn.reshape(decay, [1, BH, 1, K]), -2, -1)  # [1,BH,K,1]
        S = ttnn.multiply(S, decay_col)  # new tensor (the input state is only read)
        k_row = ttnn.reshape(kh, [1, BH, 1, K])
        k_col = ttnn.transpose(k_row, -2, -1)  # [1,BH,K,1]
        kv = ttnn.sum(ttnn.multiply(S, k_col), dim=-2, keepdim=True)  # [1,BH,1,V]
        v_row = ttnn.reshape(vh, [1, BH, 1, V])
        beta_col = ttnn.reshape(beta, [1, BH, 1, 1])
        delta = ttnn.multiply(ttnn.subtract(v_row, kv), beta_col)  # [1,BH,1,V]
        S = ttnn.add(S, ttnn.multiply(k_col, delta), output_tensor=S)  # outer product k^T delta
        q_col = ttnn.transpose(ttnn.reshape(qh, [1, BH, 1, K]), -2, -1)
        o = ttnn.sum(ttnn.multiply(S, q_col), dim=-2, keepdim=True)  # [1,BH,1,V]
        new_recurrent = ttnn.reshape(S, [B, H, K, V])
        return self._output(x, ttnn.reshape(o, [1, B, H, V]), og, new_conv, state, new_recurrent, inplace)

    def _block_decode(self, x: ttnn.Tensor, state: KdaState, inplace: bool) -> tuple[ttnn.Tensor, KdaState]:
        """The 5-op decode block: qkv matmul, two gate matmuls, ONE ``kda_decode`` program (conv, norms,
        gates, recurrence, gated RMSNorm; state + conv history in place; bf16 TILE output), o_proj.
        ``inplace=False`` runs on copies of the state tensors (the program always updates its inputs in place)."""
        rec, conv = state.recurrent, state.convolution
        if not inplace:
            rec, conv = ttnn.clone(rec), ttnn.clone(conv)
        qkv = self.lin_qkv(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,3*H*K]
        g1 = self.lin_g1(x, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [f_a | g_a | b]
        g2 = self.lin_g2(
            g1, dtype=ttnn.float32, compute_kernel_config=self.fp32
        )  # [f | og | b] (gates="act": activated)
        B = x.shape[-2]
        o = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, B, self.H * self.V]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self.device,
            ttnn.DRAM_MEMORY_CONFIG,
        )  # [1,1,B,H*V] bf16 TILE: the program's pre-allocated output (the o_proj input)
        self.kda_decode(qkv, g2, conv, rec, self.taps_blk, self.chan_consts, self.o_norm_w32, self.selectors, o)
        ttnn.deallocate(qkv)
        ttnn.deallocate(g1)
        ttnn.deallocate(g2)
        y = self.lin_o(o, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,hidden]
        ttnn.deallocate(o)
        y = self.mesh.all_reduce_cols(y)
        if inplace:
            return y, state
        return y, KdaState(recurrent=rec, convolution=conv)

    def _output(self, x, oh, og, new_conv, state: KdaState, new_recurrent, inplace: bool):
        """Gated RMSNorm per head (rmsnorm(o) * w * sigmoid(gate)), o_proj, TP all-reduce and the state
        hand-back. ``oh`` [1,B,H,V] fp32."""
        B, H, V = oh.shape[1], self.H, self.V
        oh = ttnn.rms_norm(oh, epsilon=self.kcfg.norm_eps, weight=self.o_norm_w, compute_kernel_config=self.fp32)
        gate = ttnn.sigmoid(ttnn.reshape(ttnn.typecast(og, ttnn.float32), [1, B, H, V]))
        oh = ttnn.multiply(oh, gate, output_tensor=oh)
        o_flat = ttnn.typecast(ttnn.reshape(oh, [1, 1, B, H * V]), ttnn.bfloat16)
        y = self.lin_o(o_flat, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)  # [1,1,B,hidden]
        y = self.mesh.all_reduce_cols(y)  # TP partial -> replicated over the row's columns
        if inplace:
            # write the new carries back into the caller's persistent tensors so a captured decode
            # trace reads the updated state on the next replay (device-owned autoregressive state);
            ttnn.copy(new_recurrent, state.recurrent)
            ttnn.deallocate(new_recurrent)
            ttnn.copy(new_conv, state.convolution)
            ttnn.deallocate(new_conv)
            return y, state
        return y, KdaState(recurrent=new_recurrent, convolution=new_conv)
