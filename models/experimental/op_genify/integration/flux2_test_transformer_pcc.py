# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FLUX.2-dev DiT against diffusers with random weights: stock matmuls vs. genify_matmul (#65150).

Adapted from `models/tt_dit/tests/models/flux2/test_transformer_flux2.py::test_transformer` (row
`bh_2x2_linear`: (2, 2) mesh, SP on axis 0, TP on axis 1, Linear, 2 links, FSDP) with 1 double and 2 single
blocks, so both single-block variants run (a regular one, and the final one, which skips the prompt's MLP and
proj_out). The FLUX.2-dev checkpoint is gated, so the weights are random at its config, built the way
`models/experimental/op_genify/matmul_shape_set/test_flux2_profile.py` builds them (seed 0, no weight cache).
`prompt_replicated` is the original row (shard_prompt=False); `prompt_sharded` is the QB2 pipeline / profiling
setting (shard_prompt=True), i.e. the matmul shapes the profile measures.

Run once per mode (TT_GENIFY_MATMUL selects genify_matmul in models/tt_dit/layers/linear.py):
    scripts/run_safe_pytest.sh models/experimental/op_genify/integration/flux2_test_transformer_pcc.py -s
    TT_GENIFY_MATMUL=1 scripts/run_safe_pytest.sh models/experimental/op_genify/integration/flux2_test_transformer_pcc.py -s

With FLUX2_PCC_DIR=<dir> the torch reference (the same in both modes) is cached there, each mode's TT output is
saved there, a run that finds the other mode's output also reports genify vs. stock directly, and every result
is appended to <dir>/results.jsonl.
"""

import json
import os
from pathlib import Path

import diffusers
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.op_genify import genify_matmul_adapter as genify
from models.tt_dit.models.transformers.transformer_flux2 import Flux2Transformer
from models.tt_dit.parallel.config import DiTGParallelConfigNoCFG, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.flux2.device_params import line_params_flux2_transformer
from models.tt_dit.utils import cache, tensor
from models.tt_dit.utils.check import assert_quality

NUM_LAYERS = 1
NUM_SINGLE_LAYERS = 2


def _prepare_ids(*, height: int = 1, width: int = 1, text_sequence_length: int = 1) -> torch.Tensor:
    return torch.cartesian_prod(
        torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(text_sequence_length)
    )


def _metrics(reference: torch.Tensor, output: torch.Tensor) -> tuple[float, float]:
    """(PCC, RMSE / std(reference)), as models/tt_dit/utils/check.py::assert_quality computes them."""
    a = reference.detach().flatten().to(torch.float64)
    b = output.detach().flatten().to(torch.float64)
    cov = torch.cov(torch.stack([a, b]))
    pcc = (cov[0, 1] / (cov[0, 0].sqrt() * cov[1, 1].sqrt())).item()
    relative_rmse = (torch.nn.functional.mse_loss(a, b).sqrt() / cov[0, 0].sqrt()).item()
    return pcc, relative_rmse


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, topology, num_links, fsdp, device_params",
    [[(2, 2), 0, 1, ttnn.Topology.Linear, 2, True, line_params_flux2_transformer]],
    ids=["bh_2x2_linear"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "shard_prompt", [pytest.param(False, id="prompt_replicated"), pytest.param(True, id="prompt_sharded")]
)
@pytest.mark.parametrize(("batch_size", "height", "width", "prompt_seq_len"), [(1, 1024, 1024, 512)])
def test_flux2_transformer_pcc(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    topology: ttnn.Topology,
    num_links: int,
    fsdp: bool,
    shard_prompt: bool,
    batch_size: int,
    height: int,
    width: int,
    prompt_seq_len: int,
) -> None:
    os.environ.pop("TT_DIT_CACHE_DIR", None)  # random weights: never read or write a weight cache
    mode = "genify" if genify.enabled() else "stock"
    case = f"{'prompt_sharded' if shard_prompt else 'prompt_replicated'}_L{NUM_LAYERS}_S{NUM_SINGLE_LAYERS}"
    out_dir = Path(os.environ["FLUX2_PCC_DIR"]) if os.environ.get("FLUX2_PCC_DIR") else None
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    logger.info(f"mode={mode} case={case} mesh={tuple(mesh_device.shape)} sp={sp_factor} tp={tp_factor}")

    torch.manual_seed(0)
    torch_model = diffusers.Flux2Transformer2DModel(num_layers=NUM_LAYERS, num_single_layers=NUM_SINGLE_LAYERS)
    torch_model = torch_model.to(torch.bfloat16).eval()
    config = torch_model.config
    assert config.num_attention_heads % tp_factor == 0

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTGParallelConfigNoCFG(
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    tt_model = Flux2Transformer(
        in_channels=torch_model.in_channels,
        num_layers=NUM_LAYERS,
        num_single_layers=NUM_SINGLE_LAYERS,
        attention_head_dim=config.attention_head_dim,
        num_attention_heads=config.num_attention_heads,
        joint_attention_dim=config.joint_attention_dim,
        out_channels=torch_model.out_channels,
        device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        padding_config=None,
        is_fsdp=fsdp,
        shard_prompt=shard_prompt,
    )
    cache.load_model(
        tt_model,
        get_torch_state_dict=torch_model.state_dict,
        model_name="FLUX.2-dev-random-weights-pcc",
        subfolder=f"transformer_L{NUM_LAYERS}_S{NUM_SINGLE_LAYERS}",
        parallel_config=parallel_config,
        mesh_shape=tuple(mesh_device.shape),
        mesh_device=mesh_device,
        is_fsdp=fsdp,
        create_cache=False,
    )

    spatial_seq_len = height * width // 16**2
    torch.manual_seed(0)
    spatial = torch.randn([batch_size, spatial_seq_len, torch_model.in_channels])
    prompt = torch.randn([batch_size, prompt_seq_len, config.joint_attention_dim])
    timestep = torch.full([batch_size], fill_value=500)
    guidance = torch.full([batch_size], fill_value=3)

    text_ids = _prepare_ids(text_sequence_length=prompt_seq_len)
    image_ids = _prepare_ids(height=height // 16, width=width // 16)
    prompt_rope_cos, prompt_rope_sin = torch_model.pos_embed.forward(text_ids)
    spatial_rope_cos, spatial_rope_sin = torch_model.pos_embed.forward(image_ids)

    ref_path = out_dir / f"torch_ref_L{NUM_LAYERS}_S{NUM_SINGLE_LAYERS}_seed0.pt" if out_dir else None
    if ref_path is not None and ref_path.exists():
        logger.info(f"torch reference from {ref_path}")
        torch_output = torch.load(ref_path)
    else:
        logger.info("running Torch model...")
        with torch.no_grad():
            torch_output = torch_model.forward(
                hidden_states=spatial.to(torch.bfloat16),
                encoder_hidden_states=prompt.to(torch.bfloat16),
                timestep=timestep / 1000,
                guidance=guidance,
                img_ids=image_ids,
                txt_ids=text_ids,
            ).sample.float()
        if ref_path is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(torch_output, ref_path)
    del torch_model

    tt_spatial = tensor.from_torch(spatial, device=mesh_device, mesh_axes=[None, sp_axis, None])
    prompt_mesh_axes = [None, sp_axis, None] if shard_prompt else None
    prompt_rope_mesh_axes = [None, None, sp_axis, None] if shard_prompt else None
    tt_prompt = tensor.from_torch(prompt, device=mesh_device, mesh_axes=prompt_mesh_axes)
    tt_embedded_prompt = tt_model.context_embedder(tt_prompt)
    tt_timestep = tensor.from_torch(timestep.unsqueeze(-1), dtype=ttnn.float32, device=mesh_device)
    tt_guidance = tensor.from_torch(guidance.unsqueeze(-1), device=mesh_device)

    def rope(t, axes):
        return tensor.from_torch(t.unsqueeze(0).unsqueeze(0), device=mesh_device, mesh_axes=axes)

    spatial_rope = (
        rope(spatial_rope_cos, [None, None, sp_axis, None]),
        rope(spatial_rope_sin, [None, None, sp_axis, None]),
    )
    prompt_rope = (rope(prompt_rope_cos, prompt_rope_mesh_axes), rope(prompt_rope_sin, prompt_rope_mesh_axes))
    combined_rope = (
        ttnn.concat([spatial_rope[0], prompt_rope[0]], dim=2),
        ttnn.concat([spatial_rope[1], prompt_rope[1]], dim=2),
    )

    logger.info("running TT model...")
    tt_output = tt_model.forward(
        spatial=tt_spatial,
        embedded_prompt=tt_embedded_prompt,
        timestep=tt_timestep,
        guidance=tt_guidance,
        spatial_rope=spatial_rope,
        prompt_rope=prompt_rope,
        combined_rope=combined_rope,
        spatial_sequence_length=spatial_seq_len,
        prompt_sequence_length=prompt_seq_len,
    )
    tt_output_torch = tensor.to_torch(tt_output, mesh_axes=[None, sp_axis, None]).float()

    pcc, relative_rmse = _metrics(torch_output, tt_output_torch)
    result = {"mode": mode, "case": case, "pcc": pcc, "relative_rmse": relative_rmse}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tt_output_torch, out_dir / f"tt_{mode}_{case}.pt")
        other = out_dir / f"tt_{'stock' if mode == 'genify' else 'genify'}_{case}.pt"
        if other.exists():
            stock_out, genify_out = (
                (torch.load(other), tt_output_torch) if mode == "genify" else (tt_output_torch, torch.load(other))
            )
            result["genify_vs_stock_pcc"], result["genify_vs_stock_relative_rmse"] = _metrics(stock_out, genify_out)
        with open(out_dir / "results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")
    logger.info(f"FLUX2_PCC_RESULT {json.dumps(result)}")

    assert_quality(torch_output, tt_output_torch, pcc=0.996, relative_rmse=0.09)
