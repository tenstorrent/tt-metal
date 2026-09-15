# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the Llama-3.1-8B dense SwiGLU MLP (tt-blaze#4140).

Three device cases, covering the two acceptance criteria that need hardware:

  1. ``single-card-full-14336``  — one chip, TP=1, the whole 4096 -> 14336 -> 4096 MLP.
     Validates the SwiGLU math and the fused-SiLU multiply. Takes the ``tp == 1`` branch, so it
     does **not** exercise the reduce-scatter.

  2. ``single-card-prod-width-1792`` — one chip, TP=1, but at the *per-chip* intermediate width
     production runs (14336/8 = 1792 = 56 tiles). #4140 asks for the production width explicitly,
     not only the single-card shape: 1792 is the matmul geometry the galaxy will actually issue,
     and it is reachable on one card because the per-chip shapes of a TP shard are a valid MLP in
     their own right.

  3. ``tp8-1x8-reduce-scatter`` — eight chips as a 1x8 mesh, real TP=8: 1792/chip intermediate and
     a genuine ``reduce_scatter`` over the TP axis, output 512/chip. This is the only case that
     exercises the collective and the full layout contract. Auto-skipped on machines that cannot
     allocate a 1x8 mesh (the ``mesh_device`` fixture skips on a capacity mismatch), so cases 1-2
     still run on a single card.

Case 3 earns its keep because the PCC floor passing on one card proves nothing about the CCL path:
at TP=1 ``forward`` returns before the collective. The immediately preceding Qwen3-32B work on
Blackhole hit a machine whose fabric had no column-axis connection at all, where every
``cluster_axis=1`` collective raised ``IndexError: map::at`` — a hardware/fabric limitation that a
single-card test cannot distinguish from a healthy build.

Run (single card — cases 1 and 2):
    pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_mlp_vs_ref.py
Eight-chip loudbox (all three):
    pytest models/demos/llama_3p1_8b_d_p/tests/unit/test_mlp_vs_ref.py -k tp8
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.reference.model import Llama31MLP
from models.demos.llama_3p1_8b_d_p.tests.mesh_profiles import drop_sp_replicas, galaxy_torus_xy_device_params
from models.demos.llama_3p1_8b_d_p.tt.config import MeshConfig
from models.demos.llama_3p1_8b_d_p.tt.mlp import TtLlamaMLP

EMB_DIM = Llama31_8BConfig.EMB_SIZE  # 4096
FULL_INTERMEDIATE = Llama31_8BConfig.INTERMEDIATE_SIZE  # 14336
TARGET_TP = 8
PROD_INTERMEDIATE_PER_CHIP = FULL_INTERMEDIATE // TARGET_TP  # 1792 = 56 tiles

# #4140's acceptance floor. bf16 activations *and* bf16 weights; bfloat8_b weights (the DeepSeek FFN
# default, graded at 0.97 there) do not reliably clear this on the 14336 reduction.
PCC_REQUIRED = 0.99


@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
@pytest.mark.parametrize(
    "mesh_device, device_params, hidden_dim",
    [
        pytest.param(
            (1, 1),
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            FULL_INTERMEDIATE,
            id="single-card-full-14336",
        ),
        pytest.param(
            (1, 1),
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            PROD_INTERMEDIATE_PER_CHIP,
            id="single-card-prod-width-1792",
        ),
        pytest.param(
            (1, TARGET_TP),
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
            FULL_INTERMEDIATE,
            id="tp8-1x8-reduce-scatter",
        ),
        # 4. ``galaxy-tp8-4x8`` — the production geometry: TP=8 on the column axis with the four SP
        #    rows replicating, i.e. case 3's collective on the mesh that ships. A Galaxy cannot open
        #    the 1x8 of case 3 at all (see ``galaxy_torus_xy_device_params``), so this is the only
        #    arm that exercises the real reduce-scatter there.
        pytest.param(
            (4, TARGET_TP),
            galaxy_torus_xy_device_params(),
            FULL_INTERMEDIATE,
            id="galaxy-tp8-4x8-reduce-scatter",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_mlp_vs_ref(mesh_device, device_params, hidden_dim, seq_len, reset_seeds):
    """TtLlamaMLP vs the torch reference, PCC >= 0.99."""
    torch.manual_seed(0)

    rows, cols = mesh_device.shape
    tp = cols  # TP lives on the column axis, matching MeshConfig's default tp_axis=1
    mesh_config = MeshConfig((rows, cols), tp=tp, tp_axis=1)

    reference = Llama31MLP(emb_dim=EMB_DIM, hidden_dim=hidden_dim).eval()

    tt_mlp = TtLlamaMLP(
        mesh_device=mesh_device,
        mesh_config=mesh_config,
        torch_weights=reference.torch_weights(),
        emb_dim=EMB_DIM,
        hidden_dim=hidden_dim,
        activations_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
    )

    # The production per-chip intermediate must be what the module actually allocated, otherwise
    # "runs at the production width" is only true of the parametrization and not of the device.
    assert tt_mlp.hidden_dim_per_chip == hidden_dim // tp
    assert tt_mlp.hidden_dim_per_chip % ttnn.TILE_SIZE == 0
    logger.info(
        f"mesh={rows}x{cols} tp={tp} hidden_dim={hidden_dim} "
        f"-> {tt_mlp.hidden_dim_per_chip}/chip ({tt_mlp.hidden_dim_per_chip // ttnn.TILE_SIZE} tiles), "
        f"out {tt_mlp.emb_dim_per_chip}/chip"
    )

    torch_input = torch.randn(1, 1, seq_len, EMB_DIM, dtype=torch.float32)
    with torch.no_grad():
        torch_output = reference(torch_input)

    # Replicated full-width input: what the distributed ffn_norm emits.
    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_output = tt_mlp(tt_input)
    ttnn.synchronize_device(mesh_device)

    if tp > 1:
        # Output is reduce-scattered on the last dim: emb_dim/tp per chip, concat back to full width.
        assert tt_output.shape[-1] == EMB_DIM // tp, (
            f"reduce_scatter should leave {EMB_DIM // tp}/chip on the hidden dim (the layout the "
            f"residual stream is in), got {tt_output.shape[-1]}"
        )
        tt_output_torch = drop_sp_replicas(
            ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(0, -1)),
            ),
            rows,
        )
    else:
        assert tt_output.shape[-1] == EMB_DIM
        tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0])

    tt_output_torch = tt_output_torch.reshape(torch_output.shape).to(torch.float32)

    assert not torch.isnan(tt_output_torch).any(), "NaN in MLP output"
    assert not torch.isinf(tt_output_torch).any(), "Inf in MLP output"

    passing, pcc = comp_pcc(torch_output, tt_output_torch, PCC_REQUIRED)
    logger.info(f"MLP PCC: {pcc}")
    assert passing, f"MLP PCC {pcc} below {PCC_REQUIRED}"


def test_mlp_module_is_import_light():
    """Importing ``tt.mlp`` must not drag in reference modelling or checkpoint loading (#4140).

    The module is on the adapter's dependency path, and the prefill engine's H2D producers import
    that path only to read the registry. torch and ttnn are expected here — what must stay out is
    anything that reads a checkpoint or builds the torch reference.

    A subprocess, because pytest has already imported most of these by collection time, so
    inspecting ``sys.modules`` in-process would prove nothing.
    """
    forbidden = ("safetensors", "transformers", "models.demos.llama_3p1_8b_d_p.reference.model")
    probe = (
        "import sys;"
        "import models.demos.llama_3p1_8b_d_p.tt.mlp;"
        f"print(','.join(m for m in {forbidden!r} if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"tt.mlp import pulled in {out}"


def test_reference_mlp_is_biasfree_and_unclamped():
    """The reference must be Llama's SwiGLU, not a neighbour's.

    Pins the two properties #4140 calls out by name, since both are what "strip biases and the OAI
    activation" means in practice, and a reference that quietly grew either would still produce
    plausible PCC at small activation magnitudes. Checked against a hand-written SwiGLU rather
    than by re-reading the module's own parameters.
    """
    torch.manual_seed(0)
    mlp = Llama31MLP(emb_dim=64, hidden_dim=128).eval()

    for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        assert proj.bias is None, "Llama-3.1-8B's MLP projections carry no biases"

    # Large inputs: a gpt-oss-style clamp at 7.0 would bind here and change the result.
    x = torch.randn(4, 64) * 20.0
    with torch.no_grad():
        got = mlp(x)
        gate = x @ mlp.gate_proj.weight.T
        up = x @ mlp.up_proj.weight.T
        want = (torch.nn.functional.silu(gate) * up) @ mlp.down_proj.weight.T
    # Loose enough to absorb BLAS reassociation between F.linear and the explicit matmul at these
    # magnitudes (outputs are O(100)); far tighter than any clamp or bias would be.
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-3)
