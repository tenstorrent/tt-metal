# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-TP-PARITY` — every module's multi-device output equals its single-device output.

`BRINGUP_RECIPE.md:847-852`: "run the same module with the same weights on `(1,1)` and on `(1,TP)`
and compare device outputs **to each other** (not just each to torch) — this is a sharper test than
PCC-vs-torch because it removes the reference's own error." Collectives are mathematically exact up
to reduction order, so anything below `0.999` here is a **sharding bug**, not precision.

What each shape adds:

| submesh | TP | `num_links` | what it is for |
|---|---|---|---|
| `(1,2)` | 2 | 1 | the smallest shard that can be mis-ordered |
| `(1,4)` | 4 | 1 | 4096/4 = 1024, 14336/4 = 3584 — both tile-aligned |
| `(1,8)` | 8 | 1 | the deployment TP: 4 Q heads and **1 KV head** per chip |
| `(2,8)` | 8 | **2** | `R-012`: the first parity shape with two fabric links and `sp > 1` |
| `(4,8)` | 8 | **2** | the deployment mesh itself |

Three things this file does differently from the recipe's sketch, all forced by the machine:

* every shape is a **submesh** of the open `(4,8)` galaxy, because a top-level partial mesh cannot
  bring the fabric up here (`DEC-080`);
* `Topology.Ring` everywhere (`DEC-081`), so the parity ladder and the deployment run the same
  transport — which also settles `R-012`'s worry that parity would never touch the ring;
* the single-device reference and the multi-device run **overlap on device 0**, so
  `parent_mesh.quiesce_devices()` separates the two phases. Without that barrier the second phase's
  first collective hangs the whole machine (`DEC-081`, measured).

**Input distribution:** `randn`, seed 0, for both the activations and the weights — identical tensors
on both meshes, built once on the host and mapped twice.
**Reference dtype policy:** there is no torch reference. Both sides are the *same* device code at the
*same* dtypes (bf8_b weights, bf16 activations); only the mesh differs. That is the whole point: the
comparison sees sharding, not arithmetic.
**Negative control:** the reference tensor with its hidden dimension rotated by one TP shard —
exactly what an off-by-one shard order produces — must score far below the gate.

Run::

    export TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/\
single_bh_galaxy_torus_xy_graph_descriptor.textproto
    pytest models/demos/llama31_8b_d_p/tests/unit/test_tp_parity.py -x -q
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import TestFactory, llama_config_dims, parametrize_galaxy_submeshes
from models.demos.llama31_8b_d_p.tt.attention import Attention, ProgramConfig, attention_config_from_hf
from models.demos.llama31_8b_d_p.tt.layer import DecoderLayer
from models.demos.llama31_8b_d_p.tt.mlp import MLP
from models.demos.llama31_8b_d_p.tt.model_config import llama_hf_config
from models.demos.llama31_8b_d_p.tt.rms_norm import RMSNorm
from models.demos.llama31_8b_d_p.tt.rope import build_prefill_rope, build_transformation_mat

# BRINGUP_RECIPE.md:848 — collectives are exact up to reduction order.
PARITY_PCC = 0.999
# An off-by-one shard order must destroy the correlation, not dent it.
NEGATIVE_CONTROL_MAX_PCC = 0.95
SEQ_LEN = 512
PARITY_SHAPES = [(1, 2), (1, 4), (1, 8), (2, 8), (4, 8)]

# Weights are drawn small (x0.02) so a 32-head attention block and a 14336-wide SwiGLU stay in a
# sane numeric range at bf16; the parity comparison itself is scale-free.
WEIGHT_SCALE = 0.02


def _random_layer_state(dims, generator):
    """One decoder layer's worth of HF-layout weights. The same host tensors feed both meshes."""
    hidden, inter = dims["hidden_size"], dims["intermediate_size"]
    n_heads, n_kv, head_dim = dims["num_attention_heads"], dims["num_key_value_heads"], dims["head_dim"]

    def rnd(*shape):
        return torch.randn(*shape, generator=generator) * WEIGHT_SCALE

    return {
        "input_layernorm.weight": torch.rand(hidden, generator=generator) + 0.5,
        "post_attention_layernorm.weight": torch.rand(hidden, generator=generator) + 0.5,
        "self_attn.q_proj.weight": rnd(n_heads * head_dim, hidden),
        "self_attn.k_proj.weight": rnd(n_kv * head_dim, hidden),
        "self_attn.v_proj.weight": rnd(n_kv * head_dim, hidden),
        "self_attn.o_proj.weight": rnd(hidden, n_heads * head_dim),
        "mlp.gate_proj.weight": rnd(inter, hidden),
        "mlp.up_proj.weight": rnd(inter, hidden),
        "mlp.down_proj.weight": rnd(hidden, inter),
    }


def _substate(state, prefix):
    cut = len(prefix) + 1
    return {k[cut:]: v for k, v in state.items() if k.startswith(prefix + ".")}


def _to_device(mesh, host):
    return ttnn.from_torch(
        host,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _run_modules(objs, hf_config, state, host_x):
    """Run all four modules on one mesh and return `{name: host tensor from device 0}`.

    Scheme A (`DEC-018`): every module closes with a TP all-reduce, so the output is full width and
    identical on every device. Reading device 0 is therefore reading the module's answer, not a
    shard of it — and if that stopped being true, the parity PCC is what would catch it.
    """
    mesh = objs["mesh_device"]
    mesh_config = objs["mesh_config"]
    ccl = objs["ccl_manager"]
    out = {}

    norm = RMSNorm(mesh, hf_config, _substate(state, "input_layernorm"), mesh_config=mesh_config)
    tt_x = _to_device(mesh, host_x)
    tt_norm = norm.forward(tt_x)
    out["rms_norm"] = ttnn.to_torch(ttnn.get_device_tensors(tt_norm)[0]).float()
    tt_norm.deallocate(True)

    mlp = MLP(mesh, hf_config, _substate(state, "mlp"), mesh_config=mesh_config, ccl_manager=ccl)
    tt_x = _to_device(mesh, host_x)
    tt_mlp = mlp(tt_x)
    out["mlp"] = ttnn.to_torch(ttnn.get_device_tensors(tt_mlp)[0]).float()
    tt_mlp.deallocate(True)

    program_config = ProgramConfig()
    attn = Attention(
        mesh,
        attention_config_from_hf(hf_config, max_seq_len=SEQ_LEN),
        _substate(state, "self_attn"),
        mesh_config=mesh_config,
        ccl_manager=ccl,
        program_config=program_config,
        layer_idx=0,
        transformation_mats={"prefill": build_transformation_mat(mesh)},
    )
    rope_mats = build_prefill_rope(mesh, hf_config, seq_len=SEQ_LEN, start_pos=0)
    tt_x = _to_device(mesh, host_x)
    tt_attn = attn(tt_x, rope_mats=rope_mats, kv_cache=None)
    out["attention"] = ttnn.to_torch(ttnn.get_device_tensors(tt_attn)[0]).float()
    tt_attn.deallocate(True)

    layer = DecoderLayer(
        mesh,
        hf_config,
        state,
        0,
        mesh_config=mesh_config,
        ccl_manager=ccl,
        program_config=program_config,
        transformation_mats={"prefill": build_transformation_mat(mesh)},
        max_seq_len=SEQ_LEN,
    )
    tt_x = _to_device(mesh, host_x)
    tt_layer = layer(tt_x, position_embeddings=build_prefill_rope(mesh, hf_config, seq_len=SEQ_LEN, start_pos=0))
    out["decoder_layer"] = ttnn.to_torch(ttnn.get_device_tensors(tt_layer)[0]).float()
    tt_layer.deallocate(True)

    return out


@parametrize_galaxy_submeshes(PARITY_SHAPES)
@torch.no_grad()
def test_module_outputs_match_single_device(mesh_device, device_params, submesh_shape, reset_seeds):
    """Same weights, same input, `(1,1)` vs `(1,TP)` / `(sp,TP)`: device output vs device output."""
    rows, cols = submesh_shape
    dims = llama_config_dims()
    hf_config = llama_hf_config(dims)
    generator = torch.Generator().manual_seed(0)
    state = _random_layer_state(dims, generator)
    host_x = torch.randn(1, 1, SEQ_LEN, dims["hidden_size"], generator=generator) * 0.5

    # --- phase 1: the single-device reference. TP=1, so no collective is entered at all. ---
    single = TestFactory.setup_submesh(mesh_device, (1, 1), tp=1)
    reference = _run_modules(single, hf_config, state, host_x)

    # The two phases share device 0. `mesh_device.hpp:296` requires a barrier between phases that use
    # overlapping submeshes, and without it the next collective hangs the MACHINE (`DEC-081`).
    mesh_device.quiesce_devices()

    # --- phase 2: the multi-device run. ---
    multi = TestFactory.setup_submesh(mesh_device, submesh_shape)
    assert multi["ccl_manager"].topology == ttnn.Topology.Ring, "DEC-081: submeshes run Ring"
    measured = _run_modules(multi, hf_config, state, host_x)

    num_links = multi["ccl_manager"].num_links
    worst_name, worst_pcc = None, 1.0
    for name in ("rms_norm", "mlp", "attention", "decoder_layer"):
        ref, got = reference[name], measured[name]
        assert ref.shape == got.shape, f"{name}: single-device {tuple(ref.shape)} vs {rows}x{cols} {tuple(got.shape)}"
        _, pcc = comp_pcc(ref, got, 0.0)
        if float(pcc) < worst_pcc:
            worst_name, worst_pcc = name, float(pcc)
        logger.info(
            f"[G-TP-PARITY] {rows}x{cols} TP={cols} num_links={num_links} {name:>14}: "
            f"single-device vs multi-device PCC = {float(pcc):.6f} (threshold {PARITY_PCC})"
        )

    # Negative control: rotate the reference's hidden dim by one TP shard — the exact shape an
    # off-by-one shard order produces — and require the comparison to reject it.
    shard = dims["hidden_size"] // cols
    rotated = torch.roll(reference["decoder_layer"], shifts=shard, dims=-1)
    _, control_pcc = comp_pcc(rotated, measured["decoder_layer"], 0.0)
    logger.info(
        f"[G-TP-PARITY] {rows}x{cols} negative control: reference rotated by one TP shard "
        f"({shard} columns) scores {float(control_pcc):.6f} (must be <= {NEGATIVE_CONTROL_MAX_PCC})"
    )

    assert worst_pcc >= PARITY_PCC, (
        f"[G-TP-PARITY] {rows}x{cols} (TP={cols}, num_links={num_links}): {worst_name} differs "
        f"between one device and {rows * cols} — PCC {worst_pcc:.6f} < {PARITY_PCC}. Collectives are "
        f"exact up to reduction order, so this is a sharding bug (a wrong mapper axis, a shard order, "
        f"or a collective on the wrong cluster_axis), not precision."
    )
    assert float(control_pcc) <= NEGATIVE_CONTROL_MAX_PCC, (
        f"[G-TP-PARITY] NEGATIVE CONTROL FAILED at {rows}x{cols}: a reference rotated by a whole TP "
        f"shard still scores {float(control_pcc):.6f}, so this gate cannot see a shard-order bug."
    )


@parametrize_galaxy_submeshes([(1, 8)])
def test_a_sub_axis_tp_is_refused(mesh_device, device_params, submesh_shape, expect_error):
    """`MeshConfig` refuses `TP < cols` rather than sharding across the whole axis anyway.

    The parity numbers above are only meaningful because `tp` and the mapper agree: `shard_mapper`
    always shards across the ENTIRE `tp_axis` (`tt/config.py:54-57`), so a `MeshConfig((1,8), tp=4)`
    would build head counts from 4 while the mapper split across 8 — inconsistent per-device shapes,
    with a plausible-looking tensor at the end. `G-MESH` gates this at `(1,1)`; it is re-asserted here
    on a real 8-wide mesh, where the mapper would actually do the wrong thing.
    """
    from models.demos.llama31_8b_d_p.tt.config import MeshConfig

    with expect_error(ValueError, "sub-axis TP is unsupported"):
        MeshConfig(tuple(submesh_shape), tp=4)
    logger.info("[G-TP-PARITY] MeshConfig((1,8), tp=4) refuses: sub-axis TP is unsupported")
