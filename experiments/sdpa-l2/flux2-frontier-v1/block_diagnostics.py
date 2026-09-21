# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read-only intermediate capture for the existing FLUX.2 reference test."""

import json
import os
from pathlib import Path

import torch

from models.tt_dit.utils import tensor


def install(monkeypatch, existing):
    reference = {}
    actual = {}
    rows = []
    output = Path(os.environ["FLUX2_DIAGNOSTICS"])
    output.mkdir(parents=True, exist_ok=False)

    def compare(name, value):
        actual[name] = value.detach().float().clone()
        if name not in reference:
            return
        a, b = actual[name].double().flatten(), reference[name].double().flatten()
        row = dict(name=name, actual_shape=list(value.shape), reference_shape=list(reference[name].shape))
        if a.numel() == b.numel():
            row.update(
                l2_pct=100 * float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b)),
                pcc=float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
                actual_rms=float(a.square().mean().sqrt()),
                reference_rms=float(b.square().mean().sqrt()),
            )
        rows.append(row)
        print("BLOCK_DIAGNOSTIC", json.dumps(row), flush=True)
        (output / "metrics.json").write_text(json.dumps(rows, indent=2))

    def torch_hook(name, select=lambda x: x):
        def hook(module, args, value):
            reference[name] = select(value).detach().float().clone()
            if name in actual:
                compare(name, actual[name])

        return hook

    original_pretrained = existing.diffusers.Flux2Transformer2DModel.from_pretrained

    def pretrained(*args, **kwargs):
        model = original_pretrained(*args, **kwargs)
        for name in ("x_embedder", "context_embedder", "time_guidance_embed", "norm_out", "proj_out"):
            getattr(model, name).register_forward_hook(torch_hook(name))
        model.transformer_blocks[0].register_forward_hook(torch_hook("dual.spatial", lambda x: x[1]))
        model.transformer_blocks[0].register_forward_hook(torch_hook("dual.prompt", lambda x: x[0]))
        model.single_transformer_blocks[0].register_forward_hook(torch_hook("single.spatial", lambda x: x[:, 512:]))
        for source, start, count, scale_indices in (
            (model.double_stream_modulation_img.linear, 0, 6, (1, 4)),
            (model.double_stream_modulation_txt.linear, 6, 6, (1, 4)),
            (model.single_stream_modulation.linear, 12, 3, (1,)),
            (model.norm_out.linear, 15, 2, (0,)),
        ):
            for i in range(count):
                source.register_forward_hook(
                    torch_hook(
                        f"mod.{start+i}",
                        lambda x, i=i, count=count, scales=scale_indices: x.chunk(count, dim=-1)[i]
                        + (1 if i in scales else 0),
                    )
                )
        return model

    monkeypatch.setattr(existing.diffusers.Flux2Transformer2DModel, "from_pretrained", pretrained)
    original_load = existing.cache.load_model

    def cpu(value, sp=False, tp=True):
        axes = [None] * len(value.shape)
        if sp:
            axes[-2] = 0
        if tp:
            axes[-1] = 1
        return tensor.to_torch(value, mesh_axes=axes)

    def wrap(module, capture):
        original = module.forward

        def forward(*args, **kwargs):
            result = original(*args, **kwargs)
            capture(result)
            return result

        module.forward = forward

    def load(model, *args, **kwargs):
        result = original_load(model, *args, **kwargs)
        wrap(model.context_embedder, lambda x: compare("context_embedder", cpu(x)))
        wrap(model.x_embedder, lambda x: compare("x_embedder", cpu(x, sp=True)))
        wrap(model.time_guidance_embed, lambda x: compare("time_guidance_embed", cpu(x, tp=False)))
        wrap(model.time_mod, lambda xs: [compare(f"mod.{i}", cpu(x)) for i, x in enumerate(xs)])
        wrap(
            model.transformer_blocks[0],
            lambda xs: (compare("dual.spatial", cpu(xs[0], sp=True)), compare("dual.prompt", cpu(xs[1]))),
        )
        wrap(model.single_transformer_blocks[0], lambda x: compare("single.spatial", cpu(x, sp=True)))
        wrap(model.norm_out, lambda x: compare("norm_out", cpu(x, sp=True)))
        wrap(model.proj_out, lambda x: compare("proj_out", cpu(x, sp=True, tp=False)))
        return result

    monkeypatch.setattr(existing.cache, "load_model", load)
