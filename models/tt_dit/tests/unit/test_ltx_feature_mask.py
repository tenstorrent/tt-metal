# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU padding semantics for production feature-extractor methods.

Run directly without device conftest. The adapter exercises the actual methods
with BF16 rounding; it does not establish native TT matmul bias or trace behavior.
"""

import ast
import math
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


class Operations:
    def __init__(self):
        self.experimental = SimpleNamespace(dit_rms_norm_unary_fused=self.rms)
        self.mask_widths = []
        self.select_widths = []

    @staticmethod
    def rms(value, *, epsilon, **kwargs):
        value = value.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + epsilon)).bfloat16()

    def multiply(self, value, other):
        if isinstance(other, torch.Tensor):
            self.mask_widths.append(value.shape[-1])
        return (value * other).bfloat16()

    def where(self, mask, value, bias):
        self.select_widths.append(value.shape[-1])
        return torch.where(mask.bool(), value, bias)

    concat = staticmethod(torch.cat)
    deallocate = staticmethod(lambda value: None)
    unsqueeze = staticmethod(torch.unsqueeze)
    squeeze = staticmethod(torch.squeeze)


class Projection:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = SimpleNamespace(data=bias)

    def __call__(self, value):
        return (value.float() @ self.weight.float() + self.bias.data.float()).bfloat16()


def extractor(ops, enabled, video, audio, tp):
    path = Path(__file__).resolve().parents[2] / "encoders" / "gemma" / "feature_extractor.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GemmaFeatureExtractor")
    methods = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_normed_concat", "_aggregate", "forward"}
    ]
    ns = {"ttnn": ops, "math": math}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])), str(path), "exec"), ns
    )
    instance = SimpleNamespace(
        _mask_after_projection=enabled,
        embedding_dim=32,
        rmsnorm_cc=None,
        video_dim=video.weight.shape[1] * tp,
        audio_dim=audio.weight.shape[1] * tp if audio else None,
        video_aggregate_embed=video,
        audio_aggregate_embed=audio,
        tp_factor=tp,
        tp_mesh_axis=1,
        ccl_manager=SimpleNamespace(all_gather=lambda value, dim, **kwargs: torch.cat([value] * tp, dim=dim)),
    )
    for name in ("_normed_concat", "_aggregate", "forward"):
        setattr(instance, name, MethodType(ns[name], instance))
    return instance


class FeatureMaskTest(unittest.TestCase):
    def test_padding_bias_valid_tokens_and_changed_mask_restoration(self):
        rng = torch.Generator().manual_seed(13013)
        states = [torch.randn(2, 65, 32, generator=rng).bfloat16() for _ in range(3)]
        for tp in (1, 4, 8):
            video = Projection(torch.randn(96, 32, generator=rng).bfloat16(), torch.linspace(-2, 2, 32).bfloat16())
            audio = Projection(torch.randn(96, 16, generator=rng).bfloat16(), torch.linspace(2, -1, 16).bfloat16())
            for with_audio in (False, True):
                baseline_ops, candidate_ops = Operations(), Operations()
                baseline = extractor(baseline_ops, False, video, audio if with_audio else None, tp)
                candidate = extractor(candidate_ops, True, video, audio if with_audio else None, tp)
                all_valid = torch.ones(2, 65, 1).bfloat16()
                all_pad = torch.zeros_like(all_valid)
                mixed = all_valid.clone()
                mixed[0, 31:] = 0
                mixed[1, :33] = 0
                first = None
                for mask in (mixed, all_valid, all_pad, mixed):
                    with self.subTest(tp=tp, audio=with_audio, valid=int(mask.sum())):
                        expected = baseline.forward(states, mask)
                        actual = candidate.forward(states, mask)
                        for index, projection in enumerate((video, audio) if with_audio else (video,)):
                            self.assertTrue(torch.equal(expected[index], actual[index]))
                            padded = ~mask[..., 0].bool()
                            bias = projection.bias.data.repeat(tp)
                            self.assertTrue(torch.equal(actual[index][padded], bias.expand(int(padded.sum()), -1)))
                        if first is None:
                            first = actual
                for a, b in zip(first, actual):
                    self.assertTrue(a is b is None or torch.equal(a, b))
                self.assertEqual(candidate_ops.mask_widths, [])
                self.assertEqual(set(candidate_ops.select_widths), {32, 16} if with_audio else {32})
                self.assertEqual(set(baseline_ops.mask_widths), {96})


if __name__ == "__main__":
    unittest.main()
