# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests: LTX-2.5's stage-1 ancestral Euler keeps image conditioning pinned (task #346).

The ancestral branch of ``_denoise_no_guidance`` stepped the raw x0 estimate, so a conditioned stage 1
(i2v frame 0, keyframes, an IC-LoRA reference) drifted off its image and stage 2's re-pin crossfaded
back over pixel frames 1-8. Upstream pins the x0 estimate (``post_process_latent``) before every
sampler step. These tests run the real sources with a torch-backed ``ttnn`` stand-in (step math) and
the IC length tests' shim (the loop's wiring), so no device is needed.

Run: PYTHONPATH=<worktree> python -m pytest <file> -v -p no:cacheprovider
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

HERE = Path(__file__).resolve().parent
DISTILLED_PATH = HERE.parents[2] / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"


def _torch_ttnn() -> SimpleNamespace:
    """The few ttnn ops the step uses, on host tensors (in-place ops mutate their first argument)."""

    def _inplace(op):
        def f(a, b):
            a.copy_(op(a, b))

        return f

    return SimpleNamespace(
        bfloat16=torch.float32,
        typecast=lambda t, dtype: t.clone(),
        multiply=lambda a, b: a * b,
        subtract=lambda a, b: a - b,
        add=lambda a, b: a + b,
        multiply_=_inplace(lambda a, b: a * b),
        copy=lambda src, dst: dst.copy_(src),
        deallocate=lambda t: None,
        full_like=lambda t, v, dtype=None: torch.full_like(t, v),
    )


def _step_namespace() -> dict:
    """Exec the real step functions and ``_post_process_latent_tt`` against the torch ttnn."""
    src = DISTILLED_PATH.read_text()
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    top = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    cls = top["LTXDistilledPipeline"]
    post = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_post_process_latent_tt")
    take = lambda n: "".join(lines[n.lineno - 1 : n.end_lineno])  # noqa: E731
    shim = "from __future__ import annotations\n\n" + "\n\n".join(
        take(top[n]) for n in ("_euler_ancestral_step", "_ancestral_step_coeffs", "_euler_ancestral_step_tt")
    )
    shim += (
        "\n\nclass _P:\n    @staticmethod\n" + take(post) + "\n_post_process_latent_tt = _P._post_process_latent_tt\n"
    )
    ns = {"torch": torch, "ttnn": _torch_ttnn(), "ANCESTRAL_ETA": 1.0, "ANCESTRAL_S_NOISE": 1.0}
    exec(compile(shim, str(DISTILLED_PATH), "exec"), ns)
    return ns


def _pinned_case(seed=0, n=12, c=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, n, c, generator=g)
    vel = torch.randn(1, 1, n, c, generator=g)
    noise = torch.randn(1, 1, n, c, generator=g)
    clean = torch.randn(1, 1, n, c, generator=g)
    mask = torch.ones(1, 1, n, 1)
    mask[:, :, :4] = 0.0  # rows 0-3 pinned at strength 1.0
    pad = torch.ones(1, 1, n, 1)
    return x, vel, noise, clean, mask, pad


def test_ancestral_step_pins_x0_on_device_and_host():
    """The device step with a pin equals the host reference with the same pin, and at sigma_next=0 the
    pinned rows land exactly on the clean latent whatever the velocity."""
    ns = _step_namespace()
    x, vel, noise, clean, mask, pad = _pinned_case()
    pin = lambda x0: ns["_post_process_latent_tt"](x0, mask, clean)  # noqa: E731
    for sigma, sigma_next in ((0.9, 0.7), (0.4, 0.0)):
        tt_x = x.clone()
        ns["_euler_ancestral_step_tt"](tt_x, vel, pad, sigma, sigma_next, noise, pin=pin)
        host = ns["_euler_ancestral_step"](x, vel, sigma, sigma_next, noise, pin=pin)
        torch.testing.assert_close(tt_x, host)
    assert torch.equal(tt_x[:, :, :4], clean[:, :, :4]), "final ancestral step must land pinned rows on the image"
    assert not torch.equal(tt_x[:, :, 4:], clean[:, :, 4:])


def test_ancestral_step_unpinned_is_unchanged():
    """No pin (T2V): the step is the plain ancestral Euler, host and device alike."""
    ns = _step_namespace()
    x, vel, noise, _clean, _mask, pad = _pinned_case(seed=1)
    tt_x = x.clone()
    ns["_euler_ancestral_step_tt"](tt_x, vel, pad, 0.9, 0.7, noise)
    torch.testing.assert_close(tt_x, ns["_euler_ancestral_step"](x, vel, 0.9, 0.7, noise))


def _ic_shim():
    spec = importlib.util.spec_from_file_location("_ic_lengths", HERE / "test_ic_denoise_lengths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_ancestral(image_conds, ref_latent=None):
    ic = _ic_shim()
    ns = ic._build_shim_namespace()
    obj, _cap, _inner = ic._make_pipeline(ns, in_channels=ic.IN_CH, sp_factor=ic.SP, image_conditioning=True)
    obj._assert_trace_family_warmed = lambda *a, **k: None
    step = mock.MagicMock(name="_euler_ancestral_step_tt")
    ns.update(
        _euler_ancestral_step_tt=step,
        _env_on=lambda name, default="0": False,
        ANCESTRAL_ETA=1.0,
        ANCESTRAL_NOISE_SEED_OFFSET=10000,
    )
    obj._post_process_latent_tt = mock.MagicMock(name="_post_process_latent_tt", side_effect=lambda x0, m, c: x0)
    obj._denoise_no_guidance(
        torch.zeros(4, 16),
        torch.zeros(4, 16),
        num_frames=ic.NUM_FRAMES,
        height=ic.HEIGHT,
        width=ic.WIDTH,
        sigma_values=[1.0, 0.0],
        seed=0,
        image_conds=image_conds,
        ref_latent=ref_latent,
        traced=False,
        ancestral=True,
        ancestral_noise_seed=10000,
    )
    return obj, step


def test_ancestral_loop_pins_i2v_frame0():
    ic = _ic_shim()
    obj, step = _run_ancestral([ic._cond(0, 1.0)])
    video_call, audio_call = step.call_args_list
    pin = video_call.kwargs.get("pin")
    assert pin is not None, "an i2v stage-1 ancestral step must pin the x0 estimate to the image"
    pin("x0")
    assert obj._post_process_latent_tt.call_args.args[0] == "x0"
    assert audio_call.kwargs.get("pin") is None, "audio carries no image conditioning"


def test_ancestral_loop_pins_reference_sheet():
    ic = _ic_shim()
    _obj, step = _run_ancestral(None, ref_latent=torch.randn(1, ic.IN_CH, 2, 3, 3))
    assert step.call_args_list[0].kwargs.get("pin") is not None, "reference frames must stay pinned in stage 1"
