# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only: LTX25_S2_ANCESTRAL gating and one stage-2 ancestral Euler step against upstream math.

The pipeline functions are AST-extracted and exec'd with torch stand-ins for the ttnn ops, so the test
needs no device or ttnn build. The reference is ltx_core ``EulerAncestralDiffusionStep.step`` with
ltx_pipelines.distilled's stage-2 noise seed (seed + 20000), video noise drawn before audio.
"""

import __future__

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

LTX_DIR = Path(__file__).resolve().parents[2] / "pipelines" / "ltx"
DISTILLED = LTX_DIR / "pipeline_ltx_distilled.py"
LTX25 = LTX_DIR / "pipeline_ltx25_distilled.py"

S2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
UPSTREAM_S2_SEED_OFFSET = 20000


def _exec_defs(path, names, ns):
    src = path.read_text()
    lines = src.splitlines(keepends=True)
    found = set()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            target = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            target = node.targets[0].id
        else:
            continue
        found.add(target)
        code = compile(
            "".join(lines[node.lineno - 1 : node.end_lineno]), str(path), "exec", __future__.annotations.compiler_flag
        )
        exec(code, ns)
    assert found == set(names), set(names) - found


def _torch_ttnn():
    def copy(src, dst):
        dst.copy_(src)

    return SimpleNamespace(
        Tensor=torch.Tensor,
        bfloat16=torch.bfloat16,
        typecast=lambda t, dtype: t.clone(),
        multiply_=lambda t, other: t.mul_(other),
        multiply=torch.mul,
        subtract=torch.sub,
        add=torch.add,
        deallocate=lambda t: None,
        copy=copy,
    )


@pytest.fixture(scope="module")
def ns():
    ns = {"torch": torch, "os": __import__("os"), "ttnn": _torch_ttnn()}
    _exec_defs(
        DISTILLED,
        (
            "_env_on",
            "ANCESTRAL_ETA",
            "ANCESTRAL_S_NOISE",
            "ANCESTRAL_NOISE_SEED_OFFSET",
            "ANCESTRAL_S2_NOISE_SEED_OFFSET",
            "_ancestral_step_coeffs",
            "_euler_ancestral_step_tt",
        ),
        ns,
    )
    _exec_defs(LTX25, ("_ancestral_noise_seed",), ns)
    return ns


def _upstream_step(sample, denoised_sample, sigmas, step_index, noise, eta=1.0, s_noise=1.0):
    """ltx_core.components.diffusion_steps.EulerAncestralDiffusionStep.step."""
    sigma = sigmas[step_index].to(torch.float32)
    sigma_next = sigmas[step_index + 1].to(torch.float32)
    if sigma_next == 0:
        return denoised_sample.to(sample.dtype)
    x = sample.to(torch.float32)
    denoised = denoised_sample.to(torch.float32)
    downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
    sigma_down = sigma_next * downstep_ratio
    sigma_down_ratio = sigma_down / sigma
    x_next = sigma_down_ratio * x + (1.0 - sigma_down_ratio) * denoised
    if eta > 0:
        alpha_next = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_coeff = (sigma_next**2 - sigma_down**2 * alpha_next**2 / alpha_down**2).clamp(min=0) ** 0.5
        x_next = (alpha_next / alpha_down) * x_next + noise.to(torch.float32) * s_noise * renoise_coeff
    return x_next.to(sample.dtype)


def test_s2_ancestral_gate(ns, monkeypatch):
    seed_fn = ns["_ancestral_noise_seed"]
    monkeypatch.delenv("LTX25_S2_ANCESTRAL", raising=False)
    assert seed_fn("s1", 7, True) == 7 + 10000
    assert seed_fn("s2", 7, True) is None
    assert seed_fn("s2_ref", 7, True) is None

    monkeypatch.setenv("LTX25_S2_ANCESTRAL", "1")
    assert seed_fn("s1", 7, True) == 7 + 10000
    assert seed_fn("s2", 7, True) == 7 + UPSTREAM_S2_SEED_OFFSET
    assert seed_fn("s2_ref", 7, True) is None
    assert seed_fn("s2", 7, False) is None

    monkeypatch.setenv("LTX25_S2_ANCESTRAL", "0")
    assert seed_fn("s2", 7, True) is None


@pytest.mark.parametrize("step_index", range(len(S2_SIGMAS) - 1))
def test_s2_ancestral_step_matches_upstream(ns, monkeypatch, step_index):
    monkeypatch.setenv("LTX25_S2_ANCESTRAL", "1")
    seed, video_n, audio_n, channels = 3, 96, 32, 128
    noise_seed = ns["_ancestral_noise_seed"]("s2", seed, True)
    sigma, sigma_next = S2_SIGMAS[step_index], S2_SIGMAS[step_index + 1]

    data = torch.Generator().manual_seed(1)
    x_v = torch.randn(1, 1, video_n, channels, generator=data)
    vel_v = torch.randn(1, 1, video_n, channels, generator=data)
    x_a = torch.randn(1, 1, audio_n, channels, generator=data)
    vel_a = torch.randn(1, 1, audio_n, channels, generator=data)

    # The pipeline's draw: one generator, video then audio, only when sigma_next != 0.
    gen = torch.Generator().manual_seed(noise_seed)
    noise_v = noise_a = None
    if sigma_next != 0.0:
        noise_v = torch.randn(1, 1, video_n, channels, generator=gen, dtype=torch.float32)
        noise_a = torch.randn(1, 1, audio_n, channels, generator=gen, dtype=torch.float32)

    ref_gen = torch.Generator().manual_seed(seed + UPSTREAM_S2_SEED_OFFSET)
    sigmas = torch.tensor(S2_SIGMAS)
    refs = []
    for x, vel, n in ((x_v, vel_v, video_n), (x_a, vel_a, audio_n)):
        ref_noise = torch.randn(1, 1, n, channels, generator=ref_gen) if sigma_next != 0.0 else None
        denoised = x - sigmas[step_index].to(torch.float32) * vel
        refs.append(_upstream_step(x, denoised, sigmas, step_index, ref_noise))

    for x, vel, noise, ref in ((x_v, vel_v, noise_v, refs[0]), (x_a, vel_a, noise_a, refs[1])):
        tt_x = x.clone()
        pad_mask = torch.ones(1, 1, x.shape[2], 1)
        ns["_euler_ancestral_step_tt"](tt_x, vel, pad_mask, sigma, sigma_next, noise)
        torch.testing.assert_close(tt_x, ref, rtol=1e-5, atol=1e-5)
