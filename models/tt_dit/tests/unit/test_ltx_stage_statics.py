# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU FPS/ownership regression; run directly to avoid device-aware conftest.

Execute production stage preparation, StateTensor and RoPE math. Only device
uploads/copies and unrelated mask builders are adapted; no native trace claim.
"""

import __future__

import ast
import functools
import math
import runpy
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
VIDEO_BUFFERS = ("_tt_video_rope_cos", "_tt_video_rope_sin", "_tt_video_cross_pe_cos", "_tt_video_cross_pe_sin")


def definitions(path, namespace, names=None):
    nodes = [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (names is None or node.name in names)
    ]
    if names is not None:
        assert len(nodes) == len(names)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )


class Buffer:
    def __init__(self, value, **kwargs):
        self.value = value.to(torch.bfloat16).clone()


class StageStaticsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

        def copy(source, destination):
            destination.value.copy_(source.value)

        self.ops = SimpleNamespace(TILE_SIZE=32, bfloat16=torch.bfloat16, copy=Mock(side_effect=copy))
        self.ns = runpy.run_path(str(ROOT / "utils/patchifiers.py"))
        self.ns.update(
            torch=torch,
            np=np,
            math=math,
            functools=functools,
            Enum=Enum,
            ttnn=self.ops,
            tensor=SimpleNamespace(from_torch=Buffer),
            bf16_tensor=Buffer,
            bf16_tensor_2dshard=Buffer,
        )
        definitions(ROOT / "models/transformers/ltx/rope_ltx.py", self.ns)
        self.builders = {name: self.ns[name] for name in ("prepare_video_rope", "prepare_av_cross_pe")}
        for name in (*self.builders, "prepare_audio_rope"):
            self.ns[name] = Mock(wraps=self.ns[name])
        self.ns["build_audio_masks"] = Mock(side_effect=lambda *a, **kw: tuple(Buffer(torch.ones(1)) for _ in range(3)))
        self.ns["build_video_pad_mask"] = Mock(side_effect=lambda *a, **kw: Buffer(torch.ones(1)))
        definitions(ROOT / "utils/tracing.py", self.ns, {"StateTensor"})
        definitions(ROOT / "pipelines/ltx/pipeline_ltx.py", self.ns, {"LTXTransformerState"})
        definitions(ROOT / "pipelines/ltx/pipeline_ltx_distilled.py", self.ns, {"_prepare_stage_statics"})

    def test_initial_changed_aba_and_unchanged_anchors_at_pipeline_fps(self):
        for fps in (24.0, 25.0, 30.0):
            for hw in (2, 4):  # Two stage resolutions, same temporal grid.
                with self.subTest(fps=fps, hw=hw):
                    self.check_sequence(fps, hw)

    def check_sequence(self, fps, hw):
        pipe = SimpleNamespace(
            fps=fps,
            inner_dim=4096,
            in_channels=128,
            num_attention_heads=32,
            positional_embedding_theta=10000.0,
            positional_embedding_max_pos=[20, 2048, 2048],
            mesh_device=object(),
            parallel_config=SimpleNamespace(
                sequence_parallel=SimpleNamespace(factor=2, mesh_axis=0),
                tensor_parallel=SimpleNamespace(factor=4, mesh_axis=1),
            ),
            _prepare_trans_mat=Mock(side_effect=lambda: Buffer(torch.ones(1))),
        )
        audio_real = self.ns["AudioLatentShape"].from_duration(1, 153 / fps).frames
        dims = dict(
            latent_frames=20,
            latent_h=hw,
            latent_w=hw,
            video_N=math.ceil(21 * hw * hw / 64) * 64,
            video_N_real=21 * hw * hw,
            video_N_grid=20 * hw * hw,
            audio_N=math.ceil(audio_real / 64) * 64,
            audio_N_real=audio_real,
            sp_axis=0,
        )
        state = self.ns["LTXTransformerState"]()
        prepare = self.ns["_prepare_stage_statics"]
        first = owners = None
        for index, anchors in enumerate(([2], [3], [2])):
            copies = self.ops.copy.call_count
            prepare(pipe, state, anchor_frames=anchors, **dims)
            self.assertEqual(self.ops.copy.call_count - copies, 4 if index else 0)
            buffers = {
                name: value.data for name, value in vars(state).items() if isinstance(value, self.ns["StateTensor"])
            }
            values = {name: buf.value.clone() for name, buf in buffers.items() if buf is not None}
            common = dict(
                theta=pipe.positional_embedding_theta,
                mesh_device=pipe.mesh_device,
                parallel_config=pipe.parallel_config,
                fps=fps,
                anchor_frames=anchors,
            )
            expected = (
                self.builders["prepare_video_rope"](
                    20,
                    hw,
                    hw,
                    inner_dim=4096,
                    num_attention_heads=32,
                    max_pos=pipe.positional_embedding_max_pos,
                    **common,
                )
                + self.builders["prepare_av_cross_pe"](20, hw, hw, dims["audio_N"], audio_real, **common)[:2]
            )
            for name, fresh in zip(VIDEO_BUFFERS, expected):
                self.assertTrue(torch.equal(values[name], fresh.value), (fps, hw, anchors, name))
            self.assertEqual(state._anchor_frames, anchors)
            self.assertIsNot(state._anchor_frames, anchors)
            if index == 0:
                first, owners = values, buffers
            else:
                for name, buf in buffers.items():
                    self.assertIs(buf, owners[name], name)
                for name, value in values.items():
                    if name not in VIDEO_BUFFERS or index == 2:
                        self.assertTrue(torch.equal(value, first[name]), name)
                    else:
                        self.assertFalse(torch.equal(value, first[name]), name)

            # A repeated value (including a different list object) must do no work.
            counts = {name: value.call_count for name, value in self.ns.items() if isinstance(value, Mock)}
            copies = self.ops.copy.call_count
            prepare(pipe, state, anchor_frames=list(anchors), **dims)
            self.assertEqual(
                counts, {name: value.call_count for name, value in self.ns.items() if isinstance(value, Mock)}
            )
            self.assertEqual(self.ops.copy.call_count, copies)
            for name, value in values.items():
                self.assertIs(getattr(state, name).data, owners[name])
                self.assertTrue(torch.equal(getattr(state, name).data.value, value), name)
        pipe._prepare_trans_mat.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
