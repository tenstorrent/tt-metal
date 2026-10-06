# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: MiniMax-H3 attention / token refiner under the opt-in SDPA recipes.

Random weights, no checkpoint. Each test builds one torch reference (fp32, written here because the
pinned diffusers MiniMax-H3 reference is not installed everywhere; it mirrors the reference ops the
bringup tests compare against: bias-free to_q/k/v/to_out, per-head RMSNorm with one shared head_dim
affine, half-split partial RoPE on the leading rotary_dim channels, unmasked SDPA) and runs the tt
module once per variant of sdpa_smoke.py with the SAME state dict and inputs.

Gates (sdpa_smoke.gate_sdpa_core), per recipe variant:
- SDPA core (always): the SDPA op call inside the model is captured -- the inputs it received (after
  norm/RoPE and, for FAST, prepare_sdpa_input) and its output -- and compared against exact
  fp64 attention on those inputs (keys limited to logical_n): relative L2 (100*||a-b||/||b||) <= the
  absolute bound (3%). For FAST the same L2 against the
  pre-preparation bf16 Q/K/V (i.e. including the bfp8 KV quantization) is recorded as
  `*_sdpa_core_l2_incl_input_prep`; the end-to-end no-regression gate below covers that cost.
- end to end vs torch: L2 <= legacy L2 + margin (no regression) always; the absolute bound too on the
  token refiner, where legacy sits well inside it (~0.8-1.1%). The ring case's output is the bare
  attention module (no residual): legacy is already ~2.6-3.0% vs torch there, of which ~2.3-2.6% is
  the legacy SDPA itself (HiFi2, no fp32 dest acc) and the rest bf16 weights/activations and
  matmuls; L2 vs the legacy tt output would mostly measure legacy's own error, so the SDPA-core gate
  is the one that isolates the recipe there. All numbers are recorded with record_property.

Paths covered:
- ring joint SDPA: MiniMaxH3Attention(is_sequence_parallel=True) on a 1x2 mesh, SP=2, TP=1, incl. a
  logical_n pad tail. The exp ring path is NOT taken there: the model gates it on TP=4 and SP=32
  (4x32 Galaxy), and the op itself requires num_links == 2 and Ring topology.
- dense SDPA: MiniMaxH3TokenRefiner (attention is_sequence_parallel=False) on 1x1 and 1x2.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from ...models.transformers.minimax_h3.attention_minimax_h3 import MiniMaxH3Attention, prepare_rope_tables
from ...models.transformers.minimax_h3.token_refiner_minimax_h3 import MiniMaxH3TokenRefiner
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...utils.tensor import bf16_tensor, bf16_tensor_2dshard, from_torch
from .sdpa_smoke import LINE_1D, SdpaCapture, gate_sdpa_core, run_variants

HEAD_DIM = 128
ROTARY_DIM = 96  # MiniMax-H3: 3 axes x 2 x 16 freqs; channels [96, 128) pass through
EPS = 1e-5

BOUNDS = {"default": (3.0, 1.0), "FAST_bfp8": (3.0, 1.5)}  # variant -> (absolute bound %, margin % points)


# ---------------------------------------------------------------- torch reference


def _rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rope_half_split(x, cos, sin):
    """Rotate-half RoPE on the leading rotary_dim channels (pairs i with i + rot/2)."""
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., : rot // 2], xr[..., rot // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return torch.cat([xr * cos + rotated * sin, xp], dim=-1)


def _torch_attention(x, sd, prefix, num_heads, cos=None, sin=None):
    B, N, _ = x.shape

    def heads(t):
        return t.view(B, N, num_heads, HEAD_DIM).transpose(1, 2)

    q = heads(x @ sd[f"{prefix}to_q.weight"].T)
    k = heads(x @ sd[f"{prefix}to_k.weight"].T)
    v = heads(x @ sd[f"{prefix}to_v.weight"].T)
    q = _rms(q, sd[f"{prefix}norm_q.weight"])
    k = _rms(k, sd[f"{prefix}norm_k.weight"])
    if cos is not None:
        q, k = _rope_half_split(q, cos, sin), _rope_half_split(k, cos, sin)
    o = F.scaled_dot_product_attention(q, k, v)
    o = o.transpose(1, 2).reshape(B, N, num_heads * HEAD_DIM)
    return o @ sd[f"{prefix}to_out.0.weight"].T


def _random_attention_state(hidden, num_heads, prefix=""):
    inner = num_heads * HEAD_DIM
    sd = {}
    for name, shape in (("to_q", (inner, hidden)), ("to_k", (inner, hidden)), ("to_v", (inner, hidden))):
        sd[f"{prefix}{name}.weight"] = torch.randn(shape) / math.sqrt(hidden)
    sd[f"{prefix}to_out.0.weight"] = torch.randn(hidden, inner) / math.sqrt(inner)
    for name in ("norm_q", "norm_k"):
        sd[f"{prefix}{name}.weight"] = 1.0 + 0.5 * torch.randn(HEAD_DIM)
    return sd


def _random_rope(N):
    theta = torch.rand(N, ROTARY_DIM // 2) * 2 * math.pi
    cos = torch.cat([theta.cos(), theta.cos()], dim=-1)
    sin = torch.cat([theta.sin(), theta.sin()], dim=-1)
    return cos, sin


def _attn_path(attn: MiniMaxH3Attention, seq_local: int) -> str:
    if attn._exp_sdpa_program_config(seq_local) is not None:
        return "exp_ring"
    if attn.use_ring:
        return "ring_joint"
    return "dense"


# ---------------------------------------------------------------- ring joint (SP=2) attention


@pytest.mark.parametrize("device_params", [LINE_1D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    ("N", "hidden", "num_heads"),
    [
        pytest.param(2200, 768, 4, id="n2200_pad2240_local1120"),  # logical_n masks a 40-row pad tail
        pytest.param(2304, 512, 2, id="n2304_local1152"),
    ],
)
def test_minimax_h3_attention_ring_sp2_recipes(mesh_device, N, hidden, num_heads, record_property, monkeypatch):
    torch.manual_seed(0)
    sp_axis, tp_axis = 1, 0
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    alignment = sp_factor * ttnn.TILE_SIZE
    padded = ((N + alignment - 1) // alignment) * alignment
    seq_local = padded // sp_factor

    sd = _random_attention_state(hidden, num_heads)
    x = torch.randn(1, N, hidden)
    cos, sin = _random_rope(N)
    with torch.no_grad():
        torch_out = _torch_attention(x, sd, "", num_heads, cos, sin)

    # Pad rows: zero input, identity rope (logical_n masks them as keys; their rows are dropped).
    x_pad = torch.cat([x, torch.zeros(1, padded - N, hidden)], dim=1)
    tcos, tsin = prepare_rope_tables(cos, sin, HEAD_DIM)
    tcos = torch.cat([tcos, torch.ones(padded - N, HEAD_DIM)])
    tsin = torch.cat([tsin, torch.zeros(padded - N, HEAD_DIM)])

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )

    def upload_table(t):
        return from_torch(
            t.reshape(1, 1, *t.shape), device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None]
        )

    concat_dims = [None, None]
    concat_dims[sp_axis] = 2
    concat_dims[tp_axis] = 3

    def run_tt(sdpa_precision, kv_dtype):
        capture = SdpaCapture(monkeypatch, mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
        tt_model = MiniMaxH3Attention(
            hidden_size=hidden,
            num_heads=num_heads,
            head_dim=HEAD_DIM,
            rotary_dim=ROTARY_DIM,
            qk_norm_eps=EPS,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_sequence_parallel=True,
            sdpa_precision=sdpa_precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_model.load_torch_state_dict({k: v.clone() for k, v in sd.items()})
        path = _attn_path(tt_model, seq_local)
        logger.info(f"seq_local={seq_local} path={path}")
        assert path == "ring_joint", f"expected ring joint SDPA on 1x2, got {path}"

        tt_x = bf16_tensor_2dshard(x_pad.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
        capture.logical_n = N
        logical_n = from_torch(
            torch.tensor([N], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.Layout.ROW_MAJOR,
            mesh_axes=[..., None, None],
        )
        tt_out = tt_model(tt_x, logical_n=logical_n, rope_cos=upload_table(tcos), rope_sin=upload_table(tsin))
        out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
        )
        assert len(capture.calls) == 1 and capture.calls[0]["logical_n"] == N
        monkeypatch.undo()
        return out[0, :, :N, :].float(), (capture.core_l2(), capture.core_l2(unprepared=True))

    results = run_variants(run_tt)
    gate_sdpa_core(
        record_property,
        f"ring_N{N}",
        torch_out,
        {vid: out for vid, (out, _) in results.items()},
        {vid: core for vid, (_, core) in results.items()},
        BOUNDS,
    )


# ---------------------------------------------------------------- token refiner (dense SDPA)


def _random_refiner_state(hidden, num_heads, ffn_dim, num_layers):
    sd = {}
    for i in range(num_layers):
        p = f"refiner_blocks.{i}."
        sd.update(_random_attention_state(hidden, num_heads, prefix=f"{p}attn."))
        sd[f"{p}norm1.weight"] = 1.0 + 0.5 * torch.randn(hidden)
        sd[f"{p}norm2.weight"] = 1.0 + 0.5 * torch.randn(hidden)
        sd[f"{p}ff.net.0.proj.weight"] = torch.randn(2 * ffn_dim, hidden) / math.sqrt(hidden)
        sd[f"{p}ff.net.2.weight"] = torch.randn(hidden, ffn_dim) / math.sqrt(ffn_dim)
    sd["final_norm.weight"] = 1.0 + 0.5 * torch.randn(hidden)
    return sd


def _torch_refiner(x, sd, num_heads, num_layers):
    for i in range(num_layers):
        p = f"refiner_blocks.{i}."
        x = x + _torch_attention(_rms(x, sd[f"{p}norm1.weight"]), sd, f"{p}attn.", num_heads)
        h = _rms(x, sd[f"{p}norm2.weight"]) @ sd[f"{p}ff.net.0.proj.weight"].T
        up, gate = h.chunk(2, dim=-1)  # diffusers SwiGLU: hidden, gate = chunk; hidden * silu(gate)
        x = x + (up * F.silu(gate)) @ sd[f"{p}ff.net.2.weight"].T
    return _rms(x, sd["final_norm.weight"])


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [
        pytest.param((1, 1), {}, id="1x1"),
        pytest.param((1, 2), LINE_1D, id="1x2"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompt_len",
    [
        pytest.param(512, id="l512"),
        pytest.param(96, id="l96"),  # short text stream: legacy clamps chunks to 96, recipes fall back to Q256/K512
    ],
)
def test_minimax_h3_token_refiner_dense_recipes(mesh_device, prompt_len, record_property, monkeypatch):
    torch.manual_seed(0)
    hidden, num_heads, ffn_dim, num_layers = 768, 4, 1024, 1
    sp_axis, tp_axis = 1, 0
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    sd = _random_refiner_state(hidden, num_heads, ffn_dim, num_layers)
    x = torch.randn(1, prompt_len, hidden)
    with torch.no_grad():
        torch_out = _torch_refiner(x, sd, num_heads, num_layers)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=sp_factor),
        cfg_parallel=None,
    )
    concat_dims = [None, None]
    concat_dims[sp_axis] = 0
    concat_dims[tp_axis] = 3

    def run_tt(sdpa_precision, kv_dtype):
        capture = SdpaCapture(monkeypatch, mesh_device, sp_axis=sp_axis, tp_axis=tp_axis, replicated=True)
        tt_model = MiniMaxH3TokenRefiner(
            hidden_size=hidden,
            num_heads=num_heads,
            head_dim=HEAD_DIM,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            norm_eps=EPS,
            qk_norm_eps=EPS,
            final_norm_eps=EPS,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            sdpa_precision=sdpa_precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_model.load_torch_state_dict({k: v.clone() for k, v in sd.items()})
        path = _attn_path(tt_model.refiner_blocks[0].attn, prompt_len)
        assert path == "dense", f"token refiner must use dense SDPA, got {path}"

        tt_x = bf16_tensor(x.unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3)
        out = ttnn.to_torch(
            tt_model(tt_x),
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)),
        )
        assert out.shape[0] == sp_factor
        for d in range(1, sp_factor):
            torch.testing.assert_close(out[0], out[d], rtol=0, atol=0, msg=f"SP replica {d} diverged")
        assert len(capture.calls) == num_layers
        monkeypatch.undo()
        return out[0].float(), (capture.core_l2(), capture.core_l2(unprepared=True))

    results = run_variants(run_tt)
    gate_sdpa_core(
        record_property,
        f"refiner_L{prompt_len}",
        torch_out[0],
        {vid: out for vid, (out, _) in results.items()},
        {vid: core for vid, (_, core) in results.items()},
        BOUNDS,
        e2e_abs=True,
    )
