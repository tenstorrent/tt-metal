# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the distilled pipeline's seeded latent noise (explicit generator, optionally prefetched on a
thread) must be bit-identical to the global ``torch.manual_seed`` + bf16 ``torch.randn`` draws the
reference and every golden output were made with.

The methods are AST-extracted from the pipeline source and exec'd, so
the test needs no device build. Run: python -m pytest <file> -v -p no:cacheprovider
"""
from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

DISTILLED_PATH = Path(__file__).resolve().parents[3] / "pipelines" / "ltx" / "pipeline_ltx_distilled.py"
METHODS = ("_noise_video_latent", "_draw_seeded_noise", "_prefetch_noise", "_seeded_noise")


def _pipeline_cls():
    src = DISTILLED_PATH.read_text()
    lines = src.splitlines(keepends=True)
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "LTXDistilledPipeline")
    blocks = []
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            start = min([d.lineno for d in node.decorator_list] + [node.lineno])
            blocks.append("".join(lines[start - 1 : node.end_lineno]))
    assert len(blocks) == len(METHODS)
    ns = {"torch": torch, "ThreadPoolExecutor": ThreadPoolExecutor}
    exec("from __future__ import annotations\n\nclass LTXDistilledPipeline:\n" + "\n".join(blocks), ns)
    return ns["LTXDistilledPipeline"]


def _legacy_stage1(seed, video_shape, audio_shape):
    """The pre-generator draws: video in the noiser, then a reseed + video-sized skip for audio."""
    torch.manual_seed(seed)
    video = torch.randn(video_shape, dtype=torch.bfloat16)
    torch.manual_seed(seed)
    _ = torch.randn(video_shape, dtype=torch.bfloat16)
    audio = torch.randn(audio_shape, dtype=torch.bfloat16)
    return video, audio


# (1, 9728, 128) / (1, 151, 128): 1080p/145f stage-1 video and audio; (1, 37, 5): a draw that is not a
# multiple of 16 elements, where torch's vectorized normal fill takes its tail path.
@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("video_shape,audio_shape", [((1, 9728, 128), (1, 151, 128)), ((1, 37, 5), (1, 3, 5))])
@pytest.mark.parametrize("prefetch", [False, True])
def test_seeded_noise_matches_global_seed_draws(seed, video_shape, audio_shape, prefetch):
    pipe = _pipeline_cls()()
    if prefetch:
        pipe._prefetch_noise(seed, [(video_shape, audio_shape)])
    video, audio = pipe._seeded_noise(seed, video_shape, audio_shape)
    ref_video, ref_audio = _legacy_stage1(seed, video_shape, audio_shape)
    assert video.dtype == audio.dtype == torch.bfloat16
    assert torch.equal(video, ref_video) and torch.equal(audio, ref_audio)
    assert not getattr(pipe, "_noise_prefetch", {}), "a served prefetch must be consumed"


def test_prefetch_miss_draws_inline():
    pipe = _pipeline_cls()()
    pipe._prefetch_noise(0, [((1, 64, 8),)])
    (other_seed,) = pipe._seeded_noise(1, (1, 64, 8))
    (other_shape,) = pipe._seeded_noise(0, (1, 32, 8))
    torch.manual_seed(1)
    assert torch.equal(other_seed, torch.randn(1, 64, 8, dtype=torch.bfloat16))
    torch.manual_seed(0)
    assert torch.equal(other_shape, torch.randn(1, 32, 8, dtype=torch.bfloat16))
    assert (0, ((1, 64, 8),)) in pipe._noise_prefetch


def test_noise_video_latent_mix_unchanged():
    Pipe = _pipeline_cls()
    base, sigma = torch.randn(1, 64, 8), torch.tensor(0.7)
    (noise,) = Pipe._draw_seeded_noise(3, ((1, 64, 8),))
    mask = torch.rand(1, 64, 1)
    for m in (None, mask):
        scaled = sigma if m is None else m * sigma
        expect = noise.float() * scaled + base * (1.0 - scaled)
        assert torch.equal(Pipe._noise_video_latent(base, m, sigma, noise), expect)
