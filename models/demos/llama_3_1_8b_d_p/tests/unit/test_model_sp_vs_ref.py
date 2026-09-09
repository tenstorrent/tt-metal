# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The WHOLE model at target SP x TP vs a composed torch reference.

Target mesh (8, 4), random weights, identical on both sides. Structure follows
`minimax_m3/tests/unit/test_model_sp_vs_ref.py`.

Sequence sharded across the SP rows, residual stream SP-sharded through every layer. What this
catches that the single-layer tests cannot:

* **per-layer weight slicing** — that layer `i` gets layer `i`'s weights and not layer 0's. Every
  layer here is structurally identical, so a slicing bug produces a perfectly plausible model.
* **the layer-to-cache index mapping** — `layer_idx` selects weights, `cache_layer_idx` selects the
  cache slot, and conflating them makes every layer read layer 0's KV.
* **the seams around the stack** — embedding output layout into the first residual, and the gather
  before the tail norm.

It also pins the KV-cache CONVENTION: the device stores K in Meta head-dim order (because q/k are
permuted HF -> Meta for the rope ops), so every comparison against an HF-order golden — including
the golden trace P1/P2 are graded on — must permute one side. Getting this wrong reads as
uncorrelated (PCC ~0.008), not as a near miss.

**Depth is REDUCED here (4 layers), and that is a diagnostic, not a result.** A full-depth
random-weight model would have to be materialised identically on the host and the mesh, and the
numbers that grade this bring-up come from P1/P2 against the golden trace with REAL weights at the
full 32 layers. Every number this file produces is labelled reduced wherever it is quoted.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefModel
from models.demos.llama_3_1_8b_d_p.tt.model import Model
from models.demos.llama_3_1_8b_d_p.utils.weight_conversion import hf_to_meta_head_dim

from ..test_factory import (
    CHUNK_SIZE,
    KV_DTYPE,
    WEIGHT_DTYPE,
    assert_pcc,
    parametrize_target_mesh,
    sp_shard_rope,
)
from .test_attention_vs_ref import meta_rope_tables

SEQ = 1024
REDUCED_LAYERS = 4
"""A REDUCED depth. See the module docstring — this is a diagnostic, never a bring-up grade."""


def _reference(config, num_layers, seq_len, seed=0):
    """A reduced-depth reference model, its token ids, its logits and its per-layer K/V."""
    torch.manual_seed(seed)
    model = RefModel(config, num_layers=num_layers).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
    with torch.no_grad():
        logits, per_layer_kv = model(input_ids, return_kv=True)
    return model, input_ids, logits, per_layer_kv


def _device_state_dict(ref_model, num_layers):
    """The reference's parameters under the HF checkpoint's own key names.

    `tt/model.py` slices with `substate(state_dict, "model.layers.N")`, so the test has to hand it
    checkpoint-shaped keys — the same mapping P1's real loader produces.
    """
    sd = {}
    for name, tensor in ref_model.state_dict().items():
        key = name if name.startswith("lm_head") else f"model.{name}"
        sd[key] = tensor.detach().clone()
    return sd


@parametrize_target_mesh()
@pytest.mark.parametrize("sharded_residual", [True, False], ids=["sharded_residual", "replicated_residual"])
def test_model_sp_vs_ref(
    mesh_device, device_params, config, hf_config, mesh_config, ccl_manager, topology_name,
    sharded_residual, monkeypatch,
):
    """Whole-model logits at sp8 x tp4 vs the composed torch reference. REDUCED depth."""
    monkeypatch.setenv("LLAMA31_8B_SHARDED_RESIDUAL", "1" if sharded_residual else "0")

    ref_model, input_ids, golden_logits, _ = _reference(config, REDUCED_LAYERS, SEQ)
    state_dict = _device_state_dict(ref_model, REDUCED_LAYERS)

    model = Model(
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        max_seq_len=SEQ,
        chunk_size=CHUNK_SIZE,
        num_layers=REDUCED_LAYERS,
        weight_dtype=WEIGHT_DTYPE,
    )

    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    tt_tokens = ttnn.from_torch(
        input_ids.reshape(1, 1, SEQ).to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )
    cos, sin = meta_rope_tables(mesh_device, model.rope_setup, config.head_dim, 0, SEQ)

    # The rope tables must be sharded on the SP axis EXACTLY like the tokens: SP row r holds
    # positions [r*seq_local, (r+1)*seq_local), so it needs those rows of cos/sin. Replicating the
    # full-length table gives every row positions [0, seq_local) — the op does not (and cannot)
    # know a device's SP offset, and it does not validate the table length against the input, so a
    # replicated table is silently wrong for 7 of the 8 rows.
    out = model.forward(tt_tokens, rope_mats=(sp_shard_rope(mesh_device, mesh_config, cos), sp_shard_rope(mesh_device, mesh_config, sin)), kv_cache=None, logical_n=SEQ)
    ttnn.synchronize_device(mesh_device)

    # Logits: sequence over SP rows, vocab over TP cols.
    got = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))
    ).to(REF_DTYPE)
    residual = "sharded" if sharded_residual else "replicated"
    logger.warning(f"model_sp[{residual}] is a REDUCED run: {REDUCED_LAYERS} of {config.num_hidden_layers} layers")
    assert_pcc(
        f"model_sp[{residual}, REDUCED {REDUCED_LAYERS}L]",
        golden_logits.reshape(1, 1, SEQ, config.vocab_size),
        got,
        topology_name,
    )


@parametrize_target_mesh()
def test_model_per_layer_kv_vs_ref(
    mesh_device, device_params, config, hf_config, mesh_config, ccl_manager, topology_name, monkeypatch
):
    """Per-layer KV after a whole-model forward vs the reference's. REDUCED depth.

    This is the shape of the P1/P2 grade, rehearsed on random weights: it is what catches a stack
    that reads the wrong layer's cache, which the logits check above can survive when the layers are
    structurally identical.
    """
    monkeypatch.setenv("LLAMA31_8B_SHARDED_RESIDUAL", "1")

    ref_model, input_ids, _, golden_kv = _reference(config, REDUCED_LAYERS, SEQ)
    state_dict = _device_state_dict(ref_model, REDUCED_LAYERS)

    model = Model(
        mesh_device, hf_config, state_dict, ccl_manager=ccl_manager, mesh_config=mesh_config,
        max_seq_len=SEQ, chunk_size=SEQ, num_layers=REDUCED_LAYERS, weight_dtype=WEIGHT_DTYPE,
    )
    kv = model.allocate_kv_cache(cache_dtype=KV_DTYPE)

    dims = [None, None]
    dims[mesh_config.sp_axis] = 2
    tt_tokens = ttnn.from_torch(
        input_ids.reshape(1, 1, SEQ).to(torch.int32), device=mesh_device, dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )
    cos, sin = meta_rope_tables(mesh_device, model.rope_setup, config.head_dim, 0, SEQ)

    model.forward(
        tt_tokens,
        rope_mats=(sp_shard_rope(mesh_device, mesh_config, cos), sp_shard_rope(mesh_device, mesh_config, sin)),
        kv_cache=kv,
        logical_n=SEQ,
        return_logits=False,
    )
    ttnn.synchronize_device(mesh_device)

    concat = [None, None]
    concat[mesh_config.sp_axis] = 2
    concat[mesh_config.tp_axis] = 1

    def readback(cache):
        return ttnn.to_torch(
            cache,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=tuple(concat), mesh_shape=tuple(mesh_device.shape)),
        ).to(REF_DTYPE)

    host_k, host_v = readback(kv.k), readback(kv.v)

    # Undo the block-cyclic sequence layout: natural position p -> its row in the cache.
    sp = mesh_config.sp
    chunk_local = SEQ // sp
    tokens_per_dev = kv.capacity // sp
    p = torch.arange(SEQ)
    chip = (p % SEQ) // chunk_local
    local_row = (p // SEQ) * chunk_local + (p % chunk_local)
    dim2_idx = chip * tokens_per_dev + local_row

    logger.warning(f"model KV is a REDUCED run: {REDUCED_LAYERS} of {config.num_hidden_layers} layers")
    for layer in range(REDUCED_LAYERS):
        slot = 0 * REDUCED_LAYERS + layer  # user-major packing
        k_ref, v_ref = golden_kv[layer]
        # The cached K is in META head-dim order — q/k projections are permuted HF -> Meta so the
        # device rope ops can consume Meta tables (utils/weight_conversion.py). V is untouched:
        # it is never rotated. Comparing an HF-order golden K against the cache directly reads as
        # UNCORRELATED (measured PCC 0.008), not as a small error, which is the useful shape of
        # failure — but it is a property of the cache, not a bug, so the golden is permuted here.
        assert_pcc(
            f"model_kv.k[L{layer}, REDUCED]", hf_to_meta_head_dim(k_ref[0]), host_k[slot][:, dim2_idx, :], topology_name
        )
        assert_pcc(f"model_kv.v[L{layer}, REDUCED]", v_ref[0], host_v[slot][:, dim2_idx, :], topology_name)
