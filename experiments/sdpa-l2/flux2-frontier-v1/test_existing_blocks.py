# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run the existing FLUX.2 reference test using the pinned local checkpoint."""

import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.tt_dit.tests.models.flux2 import test_transformer_flux2 as existing


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [existing.line_params_flux2_transformer], indirect=True)
@pytest.mark.timeout(1200)
def test_existing_single_blocks(mesh_device, monkeypatch):
    # Only checkpoint resolution and host thread count differ from the original
    # case. No attention, conditioning, inputs, or acceptance thresholds change.
    torch.set_num_threads(16)
    checkpoint = Path(os.environ["FLUX2_CHECKPOINT"])
    assert (checkpoint / "transformer" / "config.json").is_file()
    block_variant = os.environ.get("FLUX2_BLOCK_VARIANT")
    if block_variant is not None:
        from model_attention import FrontierAttention

        original_constructor = existing.Flux2Transformer

        def constructor(**kwargs):
            # The upstream case replicates its prompt. The frontier integration
            # requires the pipeline's SP-sharded prompt; preserve the exact
            # global inputs/reference while adapting only their placement.
            kwargs["shard_prompt"] = True
            model = original_constructor(**kwargs)
            adapter = FrontierAttention(block_variant)
            adapter.install(model)
            original_forward = model.forward

            def forward(**inputs):
                inputs["embedded_prompt"] = ttnn.mesh_partition(
                    inputs["embedded_prompt"],
                    dim=1,
                    cluster_axis=0,
                    memory_config=inputs["embedded_prompt"].memory_config(),
                )
                inputs["prompt_rope"] = tuple(
                    ttnn.mesh_partition(value, dim=2, cluster_axis=0, memory_config=value.memory_config())
                    for value in inputs["prompt_rope"]
                )
                inputs["combined_rope"] = tuple(
                    ttnn.concat([spatial, prompt], dim=2)
                    for spatial, prompt in zip(inputs["spatial_rope"], inputs["prompt_rope"], strict=True)
                )
                result = original_forward(**inputs)
                assert len(adapter.transport) == 2, "Both blocks must use the selected attention"
                return result

            model.forward = forward
            return model

        monkeypatch.setattr(existing, "Flux2Transformer", constructor)
    repair = os.environ.get("FLUX2_MODEL_REPAIR", "none")
    assert repair in ("none", "residual", "headnorm", "both", "both_conditioning", "main_fused")
    if repair != "none":
        from model_fixes import install as install_fixes

        original_load = existing.cache.load_model

        def load(model, *args, **kwargs):
            result = original_load(model, *args, **kwargs)
            install_fixes(
                model,
                residual=repair in ("residual", "both", "both_conditioning", "main_fused"),
                per_head_norm=repair in ("headnorm", "both", "both_conditioning", "main_fused"),
                fused=repair == "main_fused",
            )
            if repair == "both_conditioning":
                from conditioning import install

                install(model)
            return result

        monkeypatch.setattr(existing.cache, "load_model", load)
    if os.environ.get("FLUX2_DIAGNOSTICS"):
        from block_diagnostics import install

        install(monkeypatch, existing)

    def local_checkpoint(model_version, **kwargs):
        assert model_version == "black-forest-labs/FLUX.2-dev"
        assert kwargs.get("model_subdir") == "transformer"
        return str(checkpoint)

    existing.test_transformer(
        mesh_device=mesh_device,
        sp_axis=0,
        tp_axis=1,
        topology=ttnn.Topology.Linear,
        num_links=1,
        batch_size=1,
        height=1024,
        width=1024,
        prompt_seq_len=512,
        skip_layers=7,
        skip_single_layers=47,
        model_location_generator=local_checkpoint,
    )
