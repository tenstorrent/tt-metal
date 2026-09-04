# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Every module's multi-device output vs its own single-device output. Gate: `G-TP-PARITY`.

**HF anchor:** none directly — each module's anchor is its own file. What this gate proves is not
model math but that the **TP collectives are exact**: the same module, the same weights, the same
input, run at TP=1 and at TP∈{2,4,8}, must agree. Collectives are mathematically exact up to
reduction order, so a large drop here is a **sharding bug, not precision**
(`BRINGUP_RECIPE.md:1827-1829`).

**Device-vs-device, not device-vs-torch**, and that is the point: comparing the two device runs
removes the reference's own error from both sides, so the residual is entirely the sharding and the
collective. It is a strictly sharper instrument than either arm's PCC against torch.

## Shapes, and why every one of them is a submesh

`(1,1)`, `(1,2)`, `(1,4)`, `(1,8)`, `(2,8)`, `(4,8)` — all carved from **one** open `(4,8)` mesh
(`G-FABRIC-MATRIX`: a top-level partial mesh dies in fabric bring-up). `(2,8)` is not optional:
`get_default_num_links` returns **1** for any single-row mesh
(`models/demos/gpt_oss_d_p/utils/general_utils.py:33`), so every `(1,N)` shape runs `num_links=1`
and never touches the deployment link count; `(2,8)` is the cheapest shape that exercises 2 links
(`BRINGUP_RECIPE.md:1766-1771`).

**This gate holds two overlapping submeshes in one process, which is the machine-hanging landmine.**
Every hand-out goes through `SubmeshPool`, which calls `parent.quiesce_devices()` on both sides of
each phase, so "two live submeshes with no barrier between their phases" is unreachable through the
API rather than merely remembered (`DEC-077`; `G-FABRIC-MATRIX` case
`overlap_1x2_then_1x8_no_quiesce` measures the hang, 246 s and a `tt-smi -r`).

## The one deviation from the recipe's wording, and its reason

`BRINGUP_RECIPE.md:1832-1834` says: "At SP > 1 the multi-device output is a token slice, so compare
it against the corresponding slice of the `(1,1)` output". **That is true only for the token-wise
modules.** `RMSNorm` and `MLP` act on each token row independently, so a sequence-sharded input does
produce exactly the corresponding slice — and comparing it that way is the direct proof of the
CCL plan's central claim, that collectives go on the **TP axis only** and every module is therefore
SP-safe (`bringup_log/04_CCL_PLAN.md` §4). `Attention` and `DecoderLayer` are **not** token-wise:
the dense causal SDPA mixes tokens, so a sequence-sharded input makes each row block attend only
itself and the output is *not* a slice of the single-device output. For those two the sequence is
**replicated** across the SP rows instead, which keeps the comparison a direct device-vs-device one
and still runs the TP collective on a 2-row and 4-row mesh at `num_links=2` — the transport the
recipe's sentence is reaching for (`DEC-083`). The genuine SP attention core is
`G-SP-RING`'s and `G-CHUNK-ATTN`'s, not this gate's.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_tp_parity.py -x -q
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links
from models.demos.llama31_8b_d_p.tests.test_factory import (  # noqa: F401 — submesh_pool is a fixture
    GALAXY_MESH_SHAPE,
    galaxy_device_params,
    llama_config_dims,
    prefill_topology,
    requires_galaxy,
    requires_ring_fabric,
    submesh_pool,
)
from models.demos.llama31_8b_d_p.tt.attention import Attention, ProgramConfig
from models.demos.llama31_8b_d_p.tt.ccl import CCLManager
from models.demos.llama31_8b_d_p.tt.config import MeshConfig, derive_head_dim
from models.demos.llama31_8b_d_p.tt.layer import DecoderLayer, build_attention_config
from models.demos.llama31_8b_d_p.tt.mlp import MLP
from models.demos.llama31_8b_d_p.tt.rms_norm import RMSNorm
from models.demos.llama31_8b_d_p.tt.rope import build_prefill_rope, build_transformation_mat

WEIGHT_DTYPE = ttnn.bfloat8_b  # `DEC-022`
ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022`

PARITY_PCC_THRESHOLD = 0.999  # `BRINGUP_RECIPE.md:2082`
CONTROL_PCC_CEILING = 0.95  # `BRINGUP_RECIPE.md:2082`

SEQ_LEN = 128  # 128 / 4 rows = 32, so the SP shard stays tile-aligned at every shape
SHAPES = [(1, 2), (1, 4), (1, 8), (2, 8), (4, 8)]
REFERENCE_SHAPE = (1, 1)

# Which modules are token-wise, i.e. row `s` of the output depends only on row `s` of the input.
# See the module docstring's "one deviation" section — this is the whole difference between the
# sequence-sharded arm and the replicated one.
TOKEN_WISE = {"rms_norm": True, "mlp": True, "attention": False, "layer": False}


def _random_weights(hf, head_dim):
    """One torch state_dict, generated once and fed to **both** sides of every comparison."""
    torch.manual_seed(0)
    hidden, inter = hf["hidden_size"], hf["intermediate_size"]
    n_q, n_kv = hf["num_attention_heads"], hf["num_key_value_heads"]
    return {
        "input_layernorm.weight": torch.randn(hidden),
        "post_attention_layernorm.weight": torch.randn(hidden),
        "self_attn.q_proj.weight": torch.randn(n_q * head_dim, hidden) * 0.02,
        "self_attn.k_proj.weight": torch.randn(n_kv * head_dim, hidden) * 0.02,
        "self_attn.v_proj.weight": torch.randn(n_kv * head_dim, hidden) * 0.02,
        "self_attn.o_proj.weight": torch.randn(hidden, n_q * head_dim) * 0.02,
        "mlp.gate_proj.weight": torch.randn(inter, hidden) * 0.02,
        "mlp.up_proj.weight": torch.randn(inter, hidden) * 0.02,
        "mlp.down_proj.weight": torch.randn(hidden, inter) * 0.02,
    }


def _substate(weights, prefix):
    return {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}


def _build(module_name, mesh, hf, weights, head_dim):
    """Build one module on `mesh`, with TP taken from the mesh's own column count."""
    shape = tuple(mesh.shape)
    mesh_config = MeshConfig(shape, tp=shape[1])
    ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=prefill_topology())
    if module_name == "rms_norm":
        return RMSNorm(mesh, hf, _substate(weights, "input_layernorm."), mesh_config=mesh_config), mesh_config
    if module_name == "mlp":
        return (
            MLP(
                mesh,
                hf,
                _substate(weights, "mlp."),
                mesh_config=mesh_config,
                ccl_manager=ccl,
                weight_dtype=WEIGHT_DTYPE,
                activation_dtype=ACTIVATION_DTYPE,
            ),
            mesh_config,
        )
    if module_name == "attention":
        return (
            Attention(
                mesh,
                build_attention_config(hf, max_seq_len=SEQ_LEN),
                _substate(weights, "self_attn."),
                ccl_manager=ccl,
                mesh_config=mesh_config,
                program_config=ProgramConfig(),
                layer_idx=0,
                transformation_mats={"prefill": build_transformation_mat(mesh)},
                weight_dtype=WEIGHT_DTYPE,
            ),
            mesh_config,
        )
    assert module_name == "layer", f"unknown module {module_name!r}"
    return (
        DecoderLayer(
            mesh,
            hf,
            weights,
            0,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            transformation_mats={"prefill": build_transformation_mat(mesh)},
            weight_dtype=WEIGHT_DTYPE,
            activation_dtype=ACTIVATION_DTYPE,
            max_seq_len=SEQ_LEN,
        ),
        mesh_config,
    )


def _call(module_name, module, x, mesh, hf):
    if module_name == "rms_norm":
        return module(x)
    if module_name == "mlp":
        return module(x)
    rope = build_prefill_rope(mesh, hf, SEQ_LEN)
    out = module(x, rope) if module_name == "attention" else module(x, rope)
    for t in rope:
        t.deallocate(True)
    return out


def _upload(host, mesh, *, token_wise):
    """`[1, 1, S, hidden]` -> device. Sequence-sharded over the SP rows iff `token_wise`."""
    rows, cols = tuple(mesh.shape)
    if token_wise and rows > 1:
        mapper = ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(2, None))
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh)
    return ttnn.from_torch(
        host,
        device=mesh,
        dtype=ACTIVATION_DTYPE,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _run_on(module_name, mesh, hf, weights, head_dim, host_input):
    """-> a list of per-device host tensors, in `(row, col)` device order."""
    module, _ = _build(module_name, mesh, hf, weights, head_dim)
    x = _upload(host_input, mesh, token_wise=TOKEN_WISE[module_name])
    out = _call(module_name, module, x, mesh, hf)
    ttnn.synchronize_device(mesh)
    per_device = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    out.deallocate(True)
    return per_device


@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("module_name", list(TOKEN_WISE), ids=list(TOKEN_WISE))
def test_multi_device_matches_single_device(submesh_pool, module_name, reset_seeds):
    """**`G-TP-PARITY`.** Five multi-device shapes vs `(1,1)`, per module.

    * **Input distribution:** **standard-normal** `[1, 1, 128, 4096]`, with weights scaled 0.02 so
      the residual blocks stay in a plausible activation range. Distribution is not load-bearing
      here in the way §2.2.1 warns about: this is a *device-vs-device* comparison, so both sides see
      the identical input and there is no reference-scale dilution to worry about — the standard
      caveat about synthetic input measuring an easier problem applies to floor comparisons, not to
      an exactness claim about a collective.
    * **Reference dtype policy:** none. The reference is **the same module on one chip**, at the
      same dtypes (`bfloat8_b` weights, `bfloat16` activations), so no torch precision enters the
      comparison. That is exactly why this instrument is sharper than PCC-vs-torch.
    * **Computed noise floor:** not applicable and deliberately so — a collective is exact up to
      reduction order, so the *expected* value is 1.0 and there is no dtype floor to sit at. What is
      recorded instead is the worst PCC across the five shapes and the fact that column-parallel
      sharding cannot move `bfloat8_b`'s exponent blocks (the shard boundary, `4096/8 = 512`, is a
      multiple of the 16-element block), so the only source of disagreement is the reduction order
      of the row-parallel matmul plus the collective.
    * **Negative control:** `test_shard_rotation_control_fails` below — rotate the reference by one
      TP shard and the same assertion must reject it.
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    weights = _random_weights(hf, head_dim)
    torch.manual_seed(1)
    host_input = torch.randn(1, 1, SEQ_LEN, hf["hidden_size"])

    with submesh_pool.use(REFERENCE_SHAPE) as ref_mesh:
        reference = _run_on(module_name, ref_mesh, hf, weights, head_dim, host_input)[0]
    logger.info(
        f"[G-TP-PARITY] {module_name}: reference on {REFERENCE_SHAPE} (TP=1), output "
        f"{tuple(reference.shape)}, token_wise={TOKEN_WISE[module_name]}"
    )

    results = {}
    for shape in SHAPES:
        rows, cols = shape
        with submesh_pool.use(shape) as mesh:
            per_device = _run_on(module_name, mesh, hf, weights, head_dim, host_input)
        assert len(per_device) == rows * cols
        worst, worst_at = 1.0, None
        for r in range(rows):
            for c in range(cols):
                got = per_device[r * cols + c]
                if TOKEN_WISE[module_name] and rows > 1:
                    seq_local = SEQ_LEN // rows
                    expected = reference[:, :, r * seq_local : (r + 1) * seq_local, :]
                else:
                    expected = reference
                assert tuple(got.shape) == tuple(expected.shape), (
                    f"{module_name} on {shape} device ({r},{c}) returned {tuple(got.shape)}, "
                    f"expected {tuple(expected.shape)}"
                )
                _, pcc = comp_pcc(expected, got, 0.0)
                if float(pcc) < worst:
                    worst, worst_at = float(pcc), (r, c)
        results[shape] = worst
        logger.info(
            f"[G-TP-PARITY] {module_name} on {shape} (TP={cols}, SP={rows}, "
            f"num_links={1 if rows == 1 else 2}): worst device PCC = {worst:.7f} at device "
            f"{worst_at} (threshold {PARITY_PCC_THRESHOLD})"
        )

    overall = min(results.values())
    logger.info(
        f"[G-TP-PARITY] {module_name}: worst over {len(SHAPES)} shapes = {overall:.7f}; "
        + ", ".join(f"{s}={v:.7f}" for s, v in results.items())
    )
    for shape, worst in results.items():
        assert worst >= PARITY_PCC_THRESHOLD, (
            f"{module_name} on {shape} disagrees with its own (1,1) run at PCC {worst:.7f} < "
            f"{PARITY_PCC_THRESHOLD}. Collectives are exact up to reduction order, so a drop this "
            f"size is a sharding bug, not precision."
        )


@requires_galaxy
@requires_ring_fabric
@torch.no_grad()
@pytest.mark.parametrize("device_params", [galaxy_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH_SHAPE], indirect=True)
def test_shard_rotation_control_fails(submesh_pool, reset_seeds):
    """Rotate the reference by one TP shard; the parity assertion must reject it (<= 0.95).

    A control on the **instrument**: it shows that the comparison above would notice a module whose
    output features landed one TP shard out of place. Run on the MLP, whose `down_proj` is
    row-parallel and whose output therefore comes back through the TP all-reduce — the path a shard
    misplacement would travel.
    """
    hf = llama_config_dims()
    head_dim = derive_head_dim(hf)
    weights = _random_weights(hf, head_dim)
    torch.manual_seed(1)
    host_input = torch.randn(1, 1, SEQ_LEN, hf["hidden_size"])
    shape = (1, 8)
    shard = hf["hidden_size"] // shape[1]

    with submesh_pool.use(REFERENCE_SHAPE) as ref_mesh:
        reference = _run_on("mlp", ref_mesh, hf, weights, head_dim, host_input)[0]
    with submesh_pool.use(shape) as mesh:
        got = _run_on("mlp", mesh, hf, weights, head_dim, host_input)[0]

    _, pcc_correct = comp_pcc(reference, got, 0.0)
    _, pcc_rotated = comp_pcc(torch.roll(reference, shifts=shard, dims=-1), got, 0.0)
    logger.info(
        f"[G-TP-PARITY] control: MLP on {shape} vs its (1,1) run scores PCC "
        f"{float(pcc_correct):.7f}; against the reference rolled by one TP shard ({shard} "
        f"features) it scores {float(pcc_rotated):.5f} (ceiling {CONTROL_PCC_CEILING})"
    )
    assert float(pcc_correct) >= PARITY_PCC_THRESHOLD, "the correct arm of the control must itself pass"
    assert float(pcc_rotated) <= CONTROL_PCC_CEILING, (
        f"the reference rolled by a whole TP shard still scores {float(pcc_rotated):.7f} > "
        f"{CONTROL_PCC_CEILING}; then this gate could not tell a correctly sharded module from one "
        f"whose features are a shard out of place"
    )
