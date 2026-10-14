# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the SDPA recipe model smoke tests (test_sdpa_recipe_model_smoke_*.py).

Each smoke case builds the tt module once per variant (``VARIANTS``: legacy, the module's default recipe, FAST
with BFP8 K/V) from the same torch state dict and inputs and gates it against a torch reference and the legacy
tt output. ``sdpa_variant(LEGACY)`` is a test hook (models have no legacy flag): it strips ``precision`` from
every SDPA call and requests the HiFi2 / BF16-dest Q128/K256 setup the models used before the recipes. Non-ring
joint SDPA has no legacy path (it runs STANDARD without precision), so on joint cases legacy equals the default
recipe and only the torch bound gates.
"""

from __future__ import annotations

import contextlib
import functools
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager

LEGACY = "legacy"  # smoke-test precision marker: the op's no-recipe path (STANDARD for non-ring joint)
LEGACY_CHUNKS = (128, 256)
_SDPA_OPS = (
    "scaled_dot_product_attention",
    "joint_scaled_dot_product_attention",
    "ring_joint_scaled_dot_product_attention",
    "exp_ring_joint_scaled_dot_product_attention",
)

# name -> (sdpa_precision, sdpa_kv_dtype). Every denoiser default is STANDARD, so "default" covers it; FAST is
# the only variant with model-side wiring of its own (prepare_recipe_inputs, BFP8 K/V gather buffers).
VARIANTS = {
    "legacy": (LEGACY, None),
    "default": (None, None),
    "FAST_bfp8": (ttnn.SDPAPrecision.FAST, ttnn.bfloat8_b),
}

# 1x2 runs on 2-chip hosts only, like the other tt_dit fabric tests (a 1x2 submesh with FABRIC_1D times out in the
# fabric router handshake on a 4-chip QB2). 1x1 runs without fabric (a 1x1 submesh with FABRIC_1D fails the router
# handshake on a 2-chip host).
LINE_1D = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True}
MESHES = pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "device_params"),
    [
        pytest.param((1, 1), 0, 1, {}, id="1x1"),
        pytest.param((1, 2), 1, 0, LINE_1D, id="1x2sp1"),
    ],
    indirect=["mesh_device", "device_params"],
)


def _without_recipe(op):
    @functools.wraps(op)
    def call(q, *args, **kwargs):
        kwargs.pop("precision", None)
        grid = kwargs["program_config"].compute_with_storage_grid_size
        kwargs["program_config"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=LEGACY_CHUNKS[0],
            k_chunk_size=LEGACY_CHUNKS[1],
            exp_approx_mode=False,
        )
        kwargs["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
            q.device().arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False
        )
        return op(q, *args, **kwargs)

    return call


@contextlib.contextmanager
def sdpa_variant(precision):
    """Yield the ``sdpa_precision`` constructor argument for a smoke variant (``LEGACY``: ``None``, with every
    SDPA call run without a recipe)."""
    if precision != LEGACY:
        yield precision
        return
    with contextlib.ExitStack() as stack:
        for name in _SDPA_OPS:
            stack.enter_context(
                mock.patch.object(ttnn.transformer, name, _without_recipe(getattr(ttnn.transformer, name)))
            )
        yield None


def run_variants(run_tt) -> dict:
    """``run_tt(sdpa_precision, sdpa_kv_dtype)`` once per variant -> {variant: its result}."""
    results = {}
    for variant, (precision, kv_dtype) in VARIANTS.items():
        with sdpa_variant(precision) as precision:
            results[variant] = run_tt(precision, kv_dtype)
    return results


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """100 * ||a - b|| / ||b||."""
    a, b = a.double().flatten(), b.double().flatten()
    return (100.0 * torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)).item()


def parallel(mesh_device, sp_axis, tp_axis):
    """(ccl_manager, parallel_config, sp_factor) for a 2D mesh with SP and TP on the given axes."""
    shape = tuple(mesh_device.shape)
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=shape[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=shape[sp_axis]),
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    return ccl_manager, parallel_config, shape[sp_axis]


def gather(mesh_device, tt_out, sp_axis, tp_axis, sp_dim=2, tp_dim=3):
    dims = [None, None]
    dims[sp_axis] = sp_dim
    dims[tp_axis] = tp_dim
    return ttnn.to_torch(
        tt_out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    )


def randn_bf16(*shape) -> torch.Tensor:
    """Random input already representable in bf16, so the torch reference sees the device input."""
    return torch.randn(*shape).bfloat16().float()


def bf16_weights(module: torch.nn.Module) -> torch.nn.Module:
    """Round the random weights to bf16 in place: torch reference and tt module share identical weights."""
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(param.bfloat16().float())
    return module.eval()


# ------------------------------------------------------------------------------------------------ LTX-2


def _diffusers_qk_to_split(t: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """diffusers (interleaved-rotation) Q/K -> Lightricks SPLIT convention the TT loader expects."""
    inv = torch.empty(head_dim, dtype=torch.long)
    inv[: head_dim // 2] = torch.arange(0, head_dim, 2)
    inv[head_dim // 2 :] = torch.arange(1, head_dim, 2)
    rest = t.shape[1:]
    return t.reshape(num_heads, head_dim, *rest).index_select(1, inv).reshape(num_heads * head_dim, *rest)


def convert_ltx_state(state: dict, num_heads: int, head_dim: int) -> dict:
    out = dict(state)
    for k in ("to_q.weight", "to_q.bias", "to_k.weight", "to_k.bias", "norm_q.weight", "norm_k.weight"):
        if k in out:
            out[k] = _diffusers_qk_to_split(out[k], num_heads, head_dim)
    return out


def ltx_video_rope(dim, heads, head_dim, F, H, W):
    from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed

    grid = torch.meshgrid(torch.arange(F), torch.arange(H), torch.arange(W), indexing="ij")
    indices = torch.stack([g.flatten() for g in grid], dim=0).float().unsqueeze(0)
    rope = LTX2AudioVideoRotaryPosEmbed(
        dim=dim,
        base_num_frames=20,
        base_height=2048,
        base_width=2048,
        theta=10000.0,
        modality="video",
        double_precision=False,
        rope_type="interleaved",
        num_attention_heads=heads,
    )
    return rope(indices)  # (B, N, dim) interleaved cos/sin


# ------------------------------------------------------------------------------------------------ gates


def gate_vs_legacy(record_property, label: str, torch_outs: dict, tt_outs: dict, bounds: dict) -> None:
    """Joint-attention gate (d64, flux). bounds: variant -> (absolute L2 bound %, margin over legacy in % points
    or None, legacy scale). Each output must satisfy L2 vs torch <= max(bound, scale x legacy L2 vs torch) and,
    with a margin, L2 vs torch <= legacy + margin. The cap accepts a legacy tt output that itself misses the
    bound (bf16 weights, HiFi2 legacy SDPA); FAST's scale > 1 lets its input rounding (RNE7 Q, RNE5+BFP8 KV)
    track a larger legacy error."""
    failures = []
    legacy = tt_outs["legacy"]
    for key, ref in torch_outs.items():
        base = rel_l2(legacy[key], ref)
        record_property(f"{label}.legacy.{key}.l2_vs_torch", round(base, 4))
        for variant, (bound, margin, scale) in bounds.items():
            vs_torch, vs_legacy = rel_l2(tt_outs[variant][key], ref), rel_l2(tt_outs[variant][key], legacy[key])
            record_property(f"{label}.{variant}.{key}.l2_vs_torch", round(vs_torch, 4))
            record_property(f"{label}.{variant}.{key}.l2_vs_legacy", round(vs_legacy, 4))
            ok = vs_torch <= max(bound, scale * base) and (margin is None or vs_torch <= base + margin)
            logger.info(
                f"{label} {variant} {key}: vs torch {vs_torch:.4f}% (legacy {base:.4f}%), vs legacy {vs_legacy:.4f}% "
                f"{'OK' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(f"{variant}/{key}: vs torch {vs_torch:.3f}% (legacy {base:.3f}%), bound {bound}%")
    assert not failures, f"{label}: " + "; ".join(failures)


def gate_vs_torch(record_property, label: str, torch_outs: dict, tt_outs: dict, bounds: dict) -> None:
    """Video-attention gate (Mochi, Wan, LTX-2). bounds: variant -> (absolute L2 bound %, margin over legacy in %
    points). L2 vs torch <= legacy L2 vs torch + margin, and L2 vs torch <= the bound; if the legacy tt output
    itself misses the bound, the bound may be met vs the legacy tt output instead."""
    failures = []
    for key, ref in torch_outs.items():
        legacy = tt_outs["legacy"][key]
        legacy_l2 = rel_l2(legacy, ref)
        record_property(f"{label}.{key}.legacy.l2_vs_torch", round(legacy_l2, 4))
        logger.info(f"{label}.{key} legacy: L2 vs torch {legacy_l2:.4f}%")
        for variant, (bound, margin) in bounds.items():
            out = tt_outs[variant][key]
            l2_torch, l2_legacy = rel_l2(out, ref), rel_l2(out, legacy)
            record_property(f"{label}.{key}.{variant}.l2_vs_torch", round(l2_torch, 4))
            record_property(f"{label}.{key}.{variant}.l2_vs_legacy", round(l2_legacy, 4))
            ok_rel = l2_torch <= legacy_l2 + margin
            if variant.startswith("FAST"):
                # FAST also rounds its inputs (RNE7 Q, RNE5+BFP8 KV), so its error may scale with
                # the legacy error (short-KV cross attention has a small reference norm): allow 1.35x. A single
                # short K chunk (Wan text cross attention, 77 keys) gains nothing from the FP32 state, which
                # reads back through the BF16 dest: 3.02% vs 1.35 x 2.31% legacy on Wan N1024/L77.
                bound = max(bound, 1.35 * legacy_l2)
            if legacy_l2 <= bound:
                ok_abs = l2_torch <= bound
                gate = f"vs torch <= legacy {legacy_l2:.3f} + {margin} and <= {bound}"
            else:
                ok_abs = l2_torch <= bound or l2_legacy <= bound
                gate = f"vs torch <= legacy {legacy_l2:.3f} + {margin}; vs torch or vs legacy tt <= {bound}"
            ok = ok_rel and ok_abs
            logger.info(
                f"{label}.{key} {variant}: L2 vs torch {l2_torch:.4f}%, vs legacy {l2_legacy:.4f}% "
                f"[{gate}] {'OK' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(f"{key}/{variant}: vs torch {l2_torch:.4f}%, vs legacy {l2_legacy:.4f}%, gate {gate}")
    assert not failures, f"{label}: " + "; ".join(failures)


class SdpaCapture:
    """Wraps the ttnn SDPA entry points; records (q, k, v, out) of every call as host fp32 tensors.

    Heads are concatenated over ``tp_axis``; the sequence over ``sp_axis``, or (``replicated``) the SP replicas
    are stacked and the first is kept. FAST: the bf16 Q/K/V before ``prepare_sdpa_input`` are kept too, so the
    core reference can charge the recipe with its own input quantization. A device-tensor ``logical_n`` is
    recorded as ``self.logical_n`` (set by the test)."""

    def __init__(self, monkeypatch, mesh_device, *, sp_axis: int, tp_axis: int, replicated: bool = False):
        self.calls = []
        self.logical_n = None
        dims = [None, None]
        dims[tp_axis] = 1
        dims[sp_axis] = 0 if replicated else 2
        self._composer = lambda: ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
        self._replicated = replicated
        for name in ("ring_joint_scaled_dot_product_attention", "scaled_dot_product_attention"):
            monkeypatch.setattr(ttnn.transformer, name, self._wrap(getattr(ttnn.transformer, name)))
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
            logical_n = kwargs.get("logical_n")
            call["logical_n"] = self.logical_n if isinstance(logical_n, ttnn.Tensor) else logical_n
            mask = kwargs.get("attn_mask")
            call["mask"] = None if mask is None else self._host(mask)
            call["dtypes"] = (q.dtype, k.dtype, v.dtype)
            call["kwargs"] = {key: kwargs[key] for key in ("precision",) if key in kwargs}
            if self._unprepared:
                assert len(self._unprepared) == 3
                call["q0"], call["k0"], call["v0"] = self._unprepared
                self._unprepared = []
            self.calls.append(call)
            return result

        return wrapped

    def core_l2(self, *, unprepared: bool = False) -> float:
        """L2 of the captured SDPA outputs vs exact fp64 attention on the captured inputs (all calls pooled; keys
        and queries limited to logical_n). ``unprepared``: FAST against its bf16 Q/K/V before preparation."""
        diffs, refs = [], []
        for c in self.calls:
            n = c["logical_n"] or c["q"].shape[2]
            pre = "0" if unprepared and "q0" in c else ""
            q, k, v = (c[f"{name}{pre}"][:, :, :n].double() for name in "qkv")
            mask = None if c["mask"] is None else c["mask"][:, :, :n, :n].double()
            ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            diffs.append((c["out"][:, :, :n].double() - ref).flatten())
            refs.append(ref.flatten())
        return 100.0 * torch.cat(diffs).norm().item() / torch.cat(refs).norm().item()


def gate_sdpa_core(
    record_property, tag: str, torch_out: torch.Tensor, outs: dict, cores: dict, bounds: dict, *, e2e_abs=False
) -> None:
    """Captured-SDPA gate (Ideogram4, MiniMax-H3). bounds: variant -> (absolute bound %, margin % points). The
    SDPA core L2 (``SdpaCapture.core_l2``) <= bound and the end-to-end L2 vs torch <= legacy + margin (and <=
    bound with ``e2e_abs``). cores: variant -> (core L2, core L2 incl. FAST input preparation)."""
    legacy = outs["legacy"]
    legacy_l2 = rel_l2(legacy, torch_out)
    record_property(f"{tag}_legacy_l2_vs_torch", round(legacy_l2, 4))
    record_property(f"{tag}_legacy_sdpa_core_l2", round(cores["legacy"][0], 4))
    logger.info(f"{tag} legacy: L2 vs torch {legacy_l2:.4f}%, SDPA core {cores['legacy'][0]:.4f}%")
    failures = []
    for vid, (bound, margin) in bounds.items():
        l2_ref, l2_leg, (core, core_prep) = rel_l2(outs[vid], torch_out), rel_l2(outs[vid], legacy), cores[vid]
        record_property(f"{tag}_{vid}_l2_vs_torch", round(l2_ref, 4))
        record_property(f"{tag}_{vid}_l2_vs_legacy", round(l2_leg, 4))
        record_property(f"{tag}_{vid}_sdpa_core_l2", round(core, 4))
        record_property(f"{tag}_{vid}_sdpa_core_l2_incl_input_prep", round(core_prep, 4))
        checks = {f"vs torch <= legacy+{margin}": l2_ref <= legacy_l2 + margin, f"SDPA core <= {bound}": core <= bound}
        if e2e_abs:
            checks[f"vs torch <= {bound}"] = l2_ref <= bound
        bad = [k for k, ok in checks.items() if not ok]
        msg = (
            f"{tag} {vid}: L2 vs torch {l2_ref:.4f}%, vs legacy {l2_leg:.4f}%, SDPA core {core:.4f}% "
            f"(incl. input prep {core_prep:.4f}%)"
        )
        logger.info(f"{msg} -> {'OK' if not bad else 'FAIL ' + str(bad)}")
        if bad:
            failures.append(f"{msg} failed {bad}")
    assert not failures, "\n".join(failures)
