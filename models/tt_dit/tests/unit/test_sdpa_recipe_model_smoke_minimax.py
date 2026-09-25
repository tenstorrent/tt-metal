# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: MiniMax-H3 attention / token refiner under the opt-in SDPA recipes.

Random weights, no checkpoint. Each test builds one torch reference (fp32, written here because the
pinned diffusers MiniMax-H3 reference is not installed everywhere; it mirrors the reference ops the
bringup tests compare against: bias-free to_q/k/v/to_out, per-head RMSNorm with one shared head_dim
affine, half-split partial RoPE on the leading rotary_dim channels, unmasked SDPA) and runs the tt
module once per variant -- legacy (the
module's legacy SDPA config, via tests/unit/sdpa_legacy.py), the default recipe (sdpa_precision=None),
FAST, ACCURATE, LOW_PRECISION(bfp8 KV) --
with the SAME state dict and inputs.

Gates, per recipe variant:
- SDPA core (always): the SDPA op call inside the model is captured -- the inputs it received (after
  norm/RoPE and, for LOW_PRECISION, prepare_sdpa_input) and its output -- and compared against exact
  fp64 attention on those inputs (keys limited to logical_n): relative L2 (100*||a-b||/||b||) <= the
  absolute bound (3% FAST / LOW_PRECISION, 1% ACCURATE). For LOW_PRECISION the same L2 against the
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
from .sdpa_legacy import LEGACY, sdpa_variant

HEAD_DIM = 128
ROTARY_DIM = 96  # MiniMax-H3: 3 axes x 2 x 16 freqs; channels [96, 128) pass through
EPS = 1e-5

VARIANTS = [
    pytest.param(LEGACY, None, id="legacy"),  # the module's legacy SDPA config (tests/unit/sdpa_legacy.py)
    pytest.param(None, None, id="default"),  # the module's default recipe (sdpa_precision_default)
    pytest.param(ttnn.SDPAPrecision.FAST, None, id="fast"),
    pytest.param(ttnn.SDPAPrecision.ACCURATE, None, id="accurate"),
    pytest.param(ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat8_b, id="low_bfp8"),
]
VARIANT_LIST = [(p.id, *p.values) for p in VARIANTS]
# The default recipe is FAST (sdpa_precision_default), so "default" uses FAST's gates.
ABS_BOUND = {"default": 3.0, "fast": 3.0, "accurate": 1.0, "low_bfp8": 3.0}
MARGIN = {"default": 1.0, "fast": 1.0, "accurate": 0.25, "low_bfp8": 1.5}  # percentage points over legacy

LINE_1D = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return 100.0 * (a - b).norm().item() / b.norm().item()


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


# ---------------------------------------------------------------- gating


class _SdpaCapture:
    """Wraps the ttnn SDPA entry points the model calls; records (q, k, v, out) as host fp32 tensors."""

    def __init__(self, monkeypatch, mesh_device, *, seq_dim_axis: int | None, sp_axis: int, tp_axis: int):
        self.calls = []
        self.logical_n = None
        concat = [None, None]
        concat[tp_axis] = 1  # heads
        concat[sp_axis] = 2 if seq_dim_axis is not None else 0  # fractured sequence, or SP replicas
        self._composer = lambda: ttnn.ConcatMesh2dToTensor(
            mesh_device, dims=concat, mesh_shape=tuple(mesh_device.shape)
        )
        self._replicated = seq_dim_axis is None
        for name in ("ring_joint_scaled_dot_product_attention", "scaled_dot_product_attention"):
            orig = getattr(ttnn.transformer, name)
            monkeypatch.setattr(ttnn.transformer, name, self._wrap(orig))
        # LOW_PRECISION: remember the bf16 tensors BEFORE preparation, so the core reference charges
        # the recipe with its own KV quantization instead of starting from the already-packed K/V.
        self._unprepared = []
        orig_prepare = ttnn.transformer.prepare_sdpa_input

        def prepare(t, *args, **kwargs):
            self._unprepared.append(self._host(t))
            return orig_prepare(t, *args, **kwargs)

        monkeypatch.setattr(ttnn.transformer, "prepare_sdpa_input", prepare)

    def _host(self, t):
        out = ttnn.to_torch(t, mesh_composer=self._composer()).float()
        return out[:1] if self._replicated else out

    def _wrap(self, orig):
        def wrapped(q, k, v, *args, **kwargs):
            result = orig(q, k, v, *args, **kwargs)
            out = result[0] if isinstance(result, tuple) else result
            call = dict(q=self._host(q), k=self._host(k), v=self._host(v), out=self._host(out))
            call["logical_n"] = kwargs.get("logical_n")
            if self._unprepared:  # LOW_PRECISION: also keep the pre-preparation bf16 Q/K/V
                assert len(self._unprepared) == 3
                call["q0"], call["k0"], call["v0"] = self._unprepared
                self._unprepared = []
            self.calls.append(call)
            return result

        return wrapped

    def core_l2(self, *, unprepared: bool = False) -> float:
        """L2 of the captured SDPA output vs exact attention on the captured inputs (all calls pooled).

        unprepared=True references LOW_PRECISION against the bf16 Q/K/V before prepare_sdpa_input, i.e.
        charges the recipe with its own KV quantization too (same as the default for other variants)."""
        diffs, refs = [], []
        for c in self.calls:
            n = c["logical_n"] or c["q"].shape[2]
            pre = "0" if unprepared and "q0" in c else ""
            q, k, v = (c[f"{name}{pre}"][:, :, :n].double() for name in "qkv")
            ref = F.scaled_dot_product_attention(q, k, v)
            diffs.append((c["out"][:, :, :n].double() - ref).flatten())
            refs.append(ref.flatten())
        return 100.0 * torch.cat(diffs).norm().item() / torch.cat(refs).norm().item()


def _gate(
    results: dict[str, torch.Tensor],
    cores: dict[str, float],
    torch_out: torch.Tensor,
    record_property,
    tag: str,
    *,
    e2e_abs: bool,
):
    legacy = results["legacy"]
    legacy_l2 = _l2(legacy, torch_out)
    record_property(f"{tag}_legacy_l2_vs_torch", round(legacy_l2, 4))
    record_property(f"{tag}_legacy_sdpa_core_l2", round(cores["legacy"][0], 4))
    logger.info(f"{tag} legacy: L2 vs torch {legacy_l2:.4f}%, SDPA core {cores['legacy'][0]:.4f}%")
    failures = []
    for vid, out in results.items():
        if vid == "legacy":
            continue
        l2_ref, l2_leg, (core, core_kvq) = _l2(out, torch_out), _l2(out, legacy), cores[vid]
        record_property(f"{tag}_{vid}_l2_vs_torch", round(l2_ref, 4))
        record_property(f"{tag}_{vid}_l2_vs_legacy", round(l2_leg, 4))
        record_property(f"{tag}_{vid}_sdpa_core_l2", round(core, 4))
        record_property(f"{tag}_{vid}_sdpa_core_l2_incl_input_prep", round(core_kvq, 4))
        bound, margin = ABS_BOUND[vid], MARGIN[vid]
        checks = {
            f"vs torch <= legacy+{margin}": l2_ref <= legacy_l2 + margin,
            f"SDPA core <= {bound}": core <= bound,
        }
        if e2e_abs:
            checks[f"vs torch <= {bound}"] = l2_ref <= bound
        bad = [k for k, ok in checks.items() if not ok]
        msg = (
            f"{tag} {vid}: L2 vs torch {l2_ref:.4f}%, vs legacy {l2_leg:.4f}%, SDPA core {core:.4f}% "
            f"(incl. input prep {core_kvq:.4f}%)"
        )
        logger.info(f"{msg} -> {'OK' if not bad else 'FAIL ' + str(bad)}")
        if bad:
            failures.append(f"{msg} failed {bad}")
    return failures


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
        pytest.param(1024, 768, 4, id="n1024_local512"),
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

    results, cores = {}, {}
    for vid, precision, kv_dtype in VARIANT_LIST:
        with sdpa_variant(precision) as sdpa_precision:
            capture = _SdpaCapture(monkeypatch, mesh_device, seq_dim_axis=2, sp_axis=sp_axis, tp_axis=tp_axis)
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
            pc = tt_model._attn_program_config(seq_local, ring=True)
            record_property(f"{vid}_path", f"{path} q{pc.q_chunk_size} k{pc.k_chunk_size}")
            logger.info(f"{vid}: seq_local={seq_local} path={path} q={pc.q_chunk_size} k={pc.k_chunk_size}")
            assert path == "ring_joint", f"expected ring joint SDPA on 1x2, got {path}"

            tt_x = bf16_tensor_2dshard(x_pad.unsqueeze(0), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3})
            tt_out = tt_model(tt_x, N=N, rope_cos=upload_table(tcos), rope_sin=upload_table(tsin))
            out = ttnn.to_torch(
                tt_out,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)
                ),
            )
            results[vid] = out[0, :, :N, :].float()
            assert len(capture.calls) == 1 and capture.calls[0]["logical_n"] == N
            cores[vid] = (capture.core_l2(), capture.core_l2(unprepared=True))
            monkeypatch.undo()
            del tt_model

    failures = _gate(results, cores, torch_out, record_property, f"ring_N{N}", e2e_abs=False)
    assert not failures, "\n".join(failures)


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

    results, cores = {}, {}
    for vid, precision, kv_dtype in VARIANT_LIST:
        with sdpa_variant(precision) as sdpa_precision:
            capture = _SdpaCapture(monkeypatch, mesh_device, seq_dim_axis=None, sp_axis=sp_axis, tp_axis=tp_axis)
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
            attn = tt_model.refiner_blocks[0].attn
            path = _attn_path(attn, prompt_len)
            pc = attn._attn_program_config(prompt_len, ring=False)
            record_property(f"{vid}_path", f"{path} q{pc.q_chunk_size} k{pc.k_chunk_size}")
            assert path == "dense", f"token refiner must use dense SDPA, got {path}"

            tt_x = bf16_tensor(x.unsqueeze(0), device=mesh_device, mesh_axis=tp_axis, shard_dim=3)
            out = ttnn.to_torch(
                tt_model(tt_x),
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=concat_dims, mesh_shape=tuple(mesh_device.shape)
                ),
            )
            assert out.shape[0] == sp_factor
            for d in range(1, sp_factor):
                torch.testing.assert_close(out[0], out[d], rtol=0, atol=0, msg=f"{vid}: SP replica {d} diverged")
            results[vid] = out[0].float()
            assert len(capture.calls) == num_layers
            cores[vid] = (capture.core_l2(), capture.core_l2(unprepared=True))
            monkeypatch.undo()
            del tt_model

    failures = _gate(results, cores, torch_out[0], record_property, f"refiner_L{prompt_len}", e2e_abs=True)
    assert not failures, "\n".join(failures)
