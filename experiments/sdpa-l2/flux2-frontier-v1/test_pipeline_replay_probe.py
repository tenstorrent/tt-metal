# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic repeated full-pipeline calls, without the isolated block benchmark."""

import hashlib
import functools
import json
import os
from pathlib import Path

import pytest
import torch
import ttnn

from model_attention import FrontierAttention
from model_fixes import install as repair
from test_pipeline import PROMPTS, device_params


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["bh_lb"], indirect=True)
@pytest.mark.timeout(1800)
def test_pipeline_replays(mesh_device, monkeypatch):
    from models.tt_dit.models.vae.vae import VaeAttention
    from models.tt_dit.pipelines.flux2.pipeline_flux2 import Flux2Pipeline
    from models.tt_dit.utils.tensor import to_torch

    output = Path(os.environ["FLUX2_PROBE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = os.environ["FLUX2_CHECKPOINT"]
    steps = int(os.environ.get("FLUX2_PROBE_STEPS", "2"))
    variant = os.environ.get("FLUX2_VARIANT", "B")
    metadata = dict(
        variant=variant,
        steps=steps,
        checkpoint=checkpoint,
        seed=0,
        model_repair="main_fused",
        conditioning="stock",
        converted_weight_cache=os.environ.get("TT_DIT_CACHE_DIR"),
        pinned_memory_cache_limit_bytes=os.environ.get("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES"),
        hostname=os.uname().nodename,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    adapter = None if variant == "stock" else FrontierAttention(variant)
    boundary_mode = os.environ.get("FLUX2_PROBE_BOUNDARIES", "0")
    boundary_enabled = boundary_mode in ("1", "retain")
    boundary_active = False
    boundaries = {}
    boundary_reference = {}
    boundary_rows = []
    torch.set_num_threads(16)
    monkeypatch.setattr(VaeAttention, "sdpa_chunk_size_map", {(True, 2, 1, 4): (64, 64)})

    def setup(transformer):
        repair(transformer, residual=True, per_head_norm=True, fused=True)
        if adapter is not None:
            adapter.install(transformer)
        if boundary_enabled:
            from block_bench import map_tensors

            def wrap(name, original):
                @functools.wraps(original)
                def forward(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if boundary_active and name not in boundaries:
                        boundaries[name] = result if boundary_mode == "retain" else map_tensors(result, ttnn.clone)
                    return result

                return forward

            for family, blocks in (
                ("dual", transformer.transformer_blocks),
                ("single", transformer.single_transformer_blocks),
            ):
                for block_index, block in enumerate(blocks):
                    block.forward = wrap(f"{family}.{block_index}", block.forward)

    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=checkpoint,
        sp_axis=0,
        tp_axis=1,
        encoder_tp_axis=1,
        vae_tp_axis=1,
        vae_h_axis=0,
        vae_w_axis=None,
        num_links=2,
        topology=ttnn.Topology.Linear,
        width=1024,
        height=1024,
        is_fsdp=False,
        dynamic_load=False,
        trace_warmup=False,
        shard_prompt=True,
        transformer_setup=setup,
    )
    cache = Path(os.environ["FLUX2_EMBEDDING_CACHE"])

    def encode(prompts, *, num_images_per_prompt, sequence_length, traced):
        key = dict(checkpoint=checkpoint, prompts=prompts, num_images=num_images_per_prompt, length=sequence_length)
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return torch.load(cache / f"{digest}.pt", weights_only=True)

    pipeline._prompt_encoder.encode = encode
    values, rows = [], []
    # StateTensor untraced updates rebind buffers, which is not supported after
    # capturing the pipeline trace. Finish all untraced repetitions first.
    for index, mode in enumerate(("untraced", "untraced", "untraced", "capture", "traced", "traced", "traced")):
        boundary_active = boundary_enabled and mode == "untraced"
        boundaries.clear()
        pipeline(
            prompts=[PROMPTS[0]],
            num_inference_steps=steps,
            seed=0,
            guidance_scale=4.0,
            prompt_upsample_temperature=None,
            traced=mode != "untraced",
        )
        value = to_torch(pipeline.ts.tt_latents_step, mesh_axes=[None, 0, None]).clone()
        assert bool(torch.isfinite(value).all())
        torch.save(value, output / f"{index}-{mode}.pt")
        values.append(value)
        if boundary_active:
            from block_bench import first_device_outputs

            for name, tensors in boundaries.items():
                hosts = first_device_outputs(tensors)
                if index == 0:
                    boundary_reference[name] = hosts
                for part, (actual, ref) in enumerate(zip(hosts, boundary_reference[name], strict=True)):
                    difference = actual.float() - ref.float()
                    boundary_rows.append(
                        dict(
                            run=index,
                            block=name,
                            part=part,
                            exact=torch.equal(actual, ref),
                            l2_pct=100
                            * float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(ref.float())),
                            max_abs=float(difference.abs().max()),
                        )
                    )
            (output / "boundaries.json").write_text(json.dumps(boundary_rows, indent=2) + "\n")
            print(
                "FIRST_CHANGED_BLOCK",
                index,
                next((r for r in boundary_rows if r["run"] == index and not r["exact"]), None),
                flush=True,
            )
            boundaries.clear()
        delta = value.float() - values[0].float()
        row = dict(
            index=index,
            mode=mode,
            exact_vs_first=torch.equal(value, values[0]),
            l2_vs_first_pct=100 * float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(values[0].float())),
            max_abs=float(delta.abs().max()),
            unequal=int(torch.count_nonzero(value != values[0])),
        )
        rows.append(row)
        print("PIPELINE_REPLAY_PROBE", json.dumps(row), flush=True)
        (output / "report.json").write_text(json.dumps(dict(status="running", **metadata, rows=rows), indent=2))
    comparisons = [
        dict(a=i, b=j, exact=torch.equal(values[i], values[j]))
        for i in range(len(values))
        for j in range(i + 1, len(values))
        if "capture" not in (rows[i]["mode"], rows[j]["mode"])
    ]
    (output / "report.json").write_text(
        json.dumps(dict(status="completed", **metadata, rows=rows, comparisons=comparisons), indent=2) + "\n"
    )
    assert all(r["exact"] for r in comparisons), "Full-pipeline replay dependence; see report"
