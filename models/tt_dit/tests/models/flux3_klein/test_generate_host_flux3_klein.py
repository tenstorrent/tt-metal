# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only run of the pipeline's real warmup() and generate() through the transformer's real forward(), with the
device ops, the encoders, the VAE and the blocks stubbed: the plan / plan-parts path (helper-thread build, parts cache,
upload, forward's use of them) runs as on the server. No device needed."""

from __future__ import annotations

import threading
import types

import pytest
import torch
from PIL import Image

import ttnn

from ....models.transformers.flux3_klein import transformer_flux3_klein as tm
from ....pipelines.flux3_klein import pipeline_flux3_klein as pl

SP = 8


class _DeviceTensor:
    def __init__(self, shape=(1,)) -> None:
        self.shape = tuple(shape)


def _shape_of(args) -> tuple:
    for a in args:
        if isinstance(a, (_DeviceTensor, torch.Tensor)):
            return tuple(a.shape)
    return (1,)


class _Ttnn:
    """ttnn with every function a no-op returning a stand-in device tensor (also its Tensor type); constants and
    types stay ttnn's."""

    def __getattr__(self, name):
        if name == "Tensor":
            return _DeviceTensor
        attr = getattr(ttnn, name)
        if name == "using_distributed_env":
            return lambda: False
        if callable(attr) and not isinstance(attr, type):
            return lambda *args, **kwargs: _DeviceTensor(_shape_of(args))
        return attr


class _Layer:
    def __call__(self, *args, **kwargs):
        return _DeviceTensor(_shape_of(args))


class _Blocks(list):
    """Single blocks that check what forward hands them: the request's RoPE and length from its plan parts."""

    def __init__(self, record) -> None:
        def block(h, *, rope, mod, logical_length):
            record.append((rope, mod, logical_length))
            return h

        super().__init__([block, block])


@pytest.fixture
def pipe(monkeypatch):
    stub = _Ttnn()
    monkeypatch.setattr(pl, "ttnn", stub)
    monkeypatch.setattr(tm, "ttnn", stub)
    # host uploads: the torch tensor itself
    monkeypatch.setattr(tm.tensor, "from_torch", lambda t, **kwargs: t.clone())

    gathers = []

    def gather(table, indices, **kwargs):
        assert isinstance(indices, torch.Tensor) and indices.dtype == torch.int32, type(indices)
        gathers.append(indices)
        return _DeviceTensor()

    monkeypatch.setattr(tm, "gather_rows", gather)
    monkeypatch.setattr(tm, "gather_rows_over_sp", lambda x, indices, **kwargs: gather(None, indices))

    blocks = []
    transformers = {}
    for model in ("distilled", "base"):
        t = tm.Flux3KleinTransformer.__new__(tm.Flux3KleinTransformer)
        t.__dict__.update(
            flow_map=model == "distilled",
            sp_factor=SP,
            _device=None,
            _sp_axis=1,
            _tp_axis=0,
            _parallel_config=None,
            _ccl_manager=types.SimpleNamespace(all_gather=lambda x, **kwargs: x),
            _rope_rows_cache={},
            _rope_rows_lock=threading.Lock(),
            modulation=lambda t_embed, t_ref: (_DeviceTensor(), (_DeviceTensor(), _DeviceTensor())),
            emb_in_image=_Layer(),
            emb_in_image_cond=_Layer(),
            txt_in=_Layer(),
            early=lambda mode, x, **kwargs: x,
            single_blocks=_Blocks(blocks),
            norm_out=_Layer(),
            proj_out=_Layer(),
        )
        transformers[model] = t

    def encode_text(prompt, rows):
        return _DeviceTensor((1, rows, 7680)), torch.zeros(rows, 4, dtype=torch.int64), min(rows, 3 + len(prompt) % 7)

    p = pl.Flux3KleinPipeline.__new__(pl.Flux3KleinPipeline)
    p.__dict__.update(
        _device=None,
        _text_length=256,
        _text_buckets=None,
        _sp_axis=1,
        _tp_axis=0,
        _transformers=transformers,
        _prompt_encoder=types.SimpleNamespace(warmup=lambda bucket: None, encode=encode_text),
        _null_text=encode_text("", 256),
        _vae=types.SimpleNamespace(
            encode_canvas=lambda image: torch.zeros((image.width // 16) * (image.height // 16), 128),
            decode=lambda state, size: _DeviceTensor(),
        ),
        _plan_cache={},
        _plan_parts_cache={},
        _mask_cache={},
    )
    p._read_image = lambda decoded: None
    p.blocks = blocks
    p.gathers = gathers
    yield p
    if getattr(p, "_prestep_executor", None) is not None:
        p._prestep_executor.shutdown(wait=True)


def _count_builds(monkeypatch):
    builds = []
    original = tm.Flux3KleinTransformer.plan_parts

    def plan_parts(self, layout, ids):
        builds.append(threading.current_thread().name)
        return original(self, layout, ids)

    monkeypatch.setattr(tm.Flux3KleinTransformer, "plan_parts", plan_parts)
    return builds


def test_warmup_and_requests_run_the_plan_path(pipe, monkeypatch):
    builds = _count_builds(monkeypatch)
    pipe.warmup(sizes=((1024, 1024), (2304, 4096)))
    assert pipe.blocks and all(isinstance(r[0][0], torch.Tensor) for r in pipe.blocks)
    # every request's plan parts are built on the helper thread
    assert builds and all(name.startswith("klein-prestep") for name in builds), builds

    def request(model, size, ref, prompt="a mug on a desk"):
        del builds[:], pipe.blocks[:]
        image = Image.new("RGB", ref) if ref else None
        pipe.generate(
            pl.Flux3KleinRequest(
                prompt, model=model, width=size[0], height=size[1], seed=1, images=(image,) if image else ()
            )
        )
        return len(builds)

    for model in ("distilled", "base"):
        branches = 2 if model == "base" else 1
        assert request(model, (2304, 4096), (2496, 1664)) == branches  # a new reference size: built
        assert request(model, (2304, 4096), (2496, 1664)) == 0  # repeat: nothing built
        assert request(model, (2304, 4096), (640, 480)) == branches
        assert request(model, (2304, 4096), (2496, 1664)) == 0  # the earlier one is still cached
        # a new prompt length: the conditional branch only (the null text is unchanged)
        assert request(model, (2304, 4096), (2496, 1664), prompt="another prompt") == 1
        assert request(model, (1024, 1024), None, prompt="warmup") == 0  # warmed: no reference at 1024^2

    # forward handed the blocks the request's single-stage RoPE and attended length
    rope, mod, length = pipe.blocks[-1]
    assert rope[0].shape[2] % (SP * 32) == 0 and isinstance(length, torch.Tensor)
