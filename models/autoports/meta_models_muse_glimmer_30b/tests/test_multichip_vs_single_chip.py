# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The multichip decoder against the single-chip TTNN decoder, on identical inputs.

This is the comparison the multichip stage rests on, and it lives in its own
module because it needs **two meshes in sequence**, which is a different session
shape from every other test in this suite.

Why it cannot share the suite's mesh
------------------------------------

The obvious construction -- carve a ``1x1`` submesh out of the open ``1x4`` mesh
and build the single-chip layer on it -- is not usable on this build.  Creating
the submesh succeeds, and work *on the submesh* succeeds, but every subsequent
collective on the **parent** mesh hangs.  Minimal repro, in order, with a
120-second alarm that never fires because the wait is inside a C++ op that does
not release the GIL:

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    sub  = mesh.create_submesh(ttnn.MeshShape(1, 1))   # ok
    ttnn.linear(<tensor on sub>, <weight on sub>)      # ok
    ttnn.all_reduce(<tensor on mesh>, topology=Ring)   # hangs, forever

The hang also leaves the fabric wedged: a *fresh* process afterwards hangs in its
first `all_reduce` too, and only `tt-smi -r` clears it.  So the comparison runs
two meshes one after the other in one process instead: open ``1x1`` **without**
fabric, run the single-chip layer, copy its outputs to host, close it; then open
``1x4`` with `FABRIC_1D_RING`, run the multichip layer on the same inputs, and
compare.  That is also the more faithful comparison -- each layer runs in exactly
the regime its own stage measured.

Run it as its **own** pytest invocation, not appended to
``test_multichip_decoder.py``.  That module holds a session-scoped ``1x4`` mesh
for its whole run, session fixtures tear down last, and opening the ``1x1`` mesh
this module needs while those four dies are still owned ends in

    Device 0: Timed out while waiting for active ethernet core 29-25 to become
    active again. Try resetting the board.

which costs a ``tt-smi -r``.  ``doc/multichip_decoder/README.md`` runs the two
invocations one after the other.

What the bar means
------------------

Every PCC number in ``test_multichip_decoder.py`` compares TTNN against HF, and
that comparison carries the shared precision policy's floor (BFP4 MLP weights,
BFP8 attention weights, a BFP8 KV cache) -- roughly 1e-2 of headroom on the
synthetic harness.  A tensor-parallel fault worth 1e-3 hides under it: one device
holding the wrong GQA KV head, a reduction over the wrong axis, a dropped padding
column, an ``in0_block_w`` that quietly truncated a K block.

Comparing the two TTNN layers directly removes the floor from both sides --
identical weights, inputs, page table, positions, precision policy and part -- so
what is left is the fracture itself, and the bar is 0.999.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tests import reference as R
from models.autoports.meta_models_muse_glimmer_30b.tests.test_functional_decoder import (
    LAYER_KINDS,
    PAGE_BLOCK_SIZE,
    PREFILL_CHUNK_SIZE,
    layer_idx_for,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_MESH_SHAPE,
    FABRIC_CONFIG,
    MultichipDecoder,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc

#: The bar for multichip-against-single-chip TTNN.  Two decimal orders tighter
#: than the HF bars, because the shared precision policy cancels out.
MULTICHIP_VS_SINGLE_CHIP_PCC = 0.999

MAX_SEQ_LEN = 16384
#: Non-aligned, past the sliding window, mid-page.
PREFILL_SEQ_LEN = 2049
#: Past the 8192-token internal chunk, so the comparison covers the multi-chunk
#: prefill path and (on sliding layers) the per-device single-KV-head tail
#: hand-off between chunks -- neither of which the single-chunk length exercises.
MULTI_CHUNK_SEQ_LEN = 12345
DECODE_STEPS = 4
#: Batch for the second comparison. 4 is enough to exercise per-user cache slots,
#: ragged positions and the batched decode head path; the fracture is per-device,
#: not per-user, so a wider batch adds cost rather than coverage.
BATCH = 4


def _page_table_rows(batch: int, max_seq_len: int, seed: int) -> torch.Tensor:
    blocks = (max_seq_len + PAGE_BLOCK_SIZE - 1) // PAGE_BLOCK_SIZE
    permutation = torch.randperm(batch * blocks, generator=torch.Generator().manual_seed(seed))
    return permutation.reshape(batch, blocks).to(torch.int32)


def _to_mesh(mesh, tensor: torch.Tensor, *, dtype, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        device=mesh,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _first_device(tensor: ttnn.Tensor) -> torch.Tensor:
    """Device 0's copy of a replicated tensor.

    Both layers' public contract is a replicated output, so device 0 is the whole
    answer; ``test_replicas_are_bit_identical`` in the main module pins that the
    other three agree with it bit for bit.
    """
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _run(decoder, mesh, hidden: torch.Tensor, token: torch.Tensor, page_rows: torch.Tensor, *, batch: int = 1) -> dict:
    """Prefill every user, then ``DECODE_STEPS`` decode steps, on the host.

    ``batch`` prefills each user into its own cache slot at its own length and
    then decodes all of them in one call at ragged positions, which is what makes
    the comparison cover the per-user page-table rows and the batched decode head
    path rather than only the batch-1 one.
    """
    page_table = _to_mesh(mesh, page_rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    seq_len = hidden.shape[1]
    out = {}
    for user in range(batch):
        tt_hidden = _to_mesh(mesh, hidden.reshape(1, 1, seq_len, -1), dtype=ttnn.bfloat16)
        tt_out = decoder.prefill_forward(tt_hidden, page_table=page_table, user_id=user)
        if user == 0:
            out["prefill"] = _first_device(tt_out).reshape(1, seq_len, -1).clone()
        ttnn.deallocate(tt_out)
        ttnn.deallocate(tt_hidden)

    tokens = token.reshape(1, 1, 1, -1).repeat(1, 1, batch, 1)
    tt_token = _to_mesh(mesh, tokens, dtype=ttnn.bfloat16)
    for step in range(DECODE_STEPS):
        # Ragged positions: user u decodes at seq_len + step + u.
        position = torch.tensor([seq_len + step + user for user in range(batch)])
        current_pos = _to_mesh(mesh, position.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        rope_pos_ids = _to_mesh(
            mesh, position.reshape(1, -1).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        tt_decode = decoder.decode_forward(
            tt_token, current_pos=current_pos, page_table=page_table, rope_pos_ids=rope_pos_ids
        )
        out[f"decode{step}"] = _first_device(tt_decode).reshape(1, batch, -1).clone()
        ttnn.deallocate(tt_decode)
        ttnn.deallocate(current_pos)
        ttnn.deallocate(rope_pos_ids)
    ttnn.deallocate(tt_token)
    ttnn.deallocate(page_table)
    return out


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("kind", LAYER_KINDS)
@pytest.mark.parametrize(
    "seq_len,batch",
    (
        (PREFILL_SEQ_LEN, 1),
        # Past the internal chunk and at batch 4: multi-chunk prefill, the
        # per-device sliding tail carried between chunks, four cache slots and
        # ragged decode positions.
        (MULTI_CHUNK_SEQ_LEN, BATCH),
    ),
)
def test_multichip_matches_single_chip(kind, seq_len, batch):
    """Prefill and every decode step, fractured against single-chip, at 0.999."""
    if ttnn.get_num_devices() < DEFAULT_MESH_SHAPE[0] * DEFAULT_MESH_SHAPE[1]:  # pragma: no cover
        pytest.skip(f"needs {DEFAULT_MESH_SHAPE[0] * DEFAULT_MESH_SHAPE[1]} devices")

    layer_idx = layer_idx_for(kind)
    state_dict = R.synthetic_state_dict(layer_idx)
    hidden = R.synthetic_hidden_states(1, seq_len, seed=4242 + seq_len)
    token = R.synthetic_hidden_states(1, 1, seed=4343)
    page_rows = _page_table_rows(max(batch, 1), MAX_SEQ_LEN, seed=1717)
    build = dict(
        hf_config=R.hf_config(),
        layer_idx=layer_idx,
        max_batch_size=batch,
        max_seq_len=MAX_SEQ_LEN,
        page_block_size=PAGE_BLOCK_SIZE,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
    )
    label = f"{kind} seq_len={seq_len} batch={batch}"

    # ---- one chip, its own mesh, no fabric ------------------------------------
    single_mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        single = OptimizedDecoder.from_state_dict(state_dict, mesh_device=single_mesh, **build)
        assert single.config.num_attention_heads == 32 and single.config.num_key_value_heads == 2
        expected = _run(single, single_mesh, hidden, token, page_rows, batch=batch)
        del single
    finally:
        ttnn.close_mesh_device(single_mesh)

    # ---- four chips ----------------------------------------------------------
    mesh = open_multichip_mesh(DEFAULT_MESH_SHAPE, trace_region_size=0, fabric_config=FABRIC_CONFIG)
    try:
        multichip = MultichipDecoder.from_state_dict(state_dict, mesh_device=mesh, **build)
        assert multichip.config.num_attention_heads == 8 and multichip.config.num_key_value_heads == 1
        actual = _run(multichip, mesh, hidden, token, page_rows, batch=batch)
        del multichip
    finally:
        close_multichip_mesh(mesh, fabric_config=FABRIC_CONFIG)

    worst = (1.0, "")
    for name, reference in expected.items():
        passed, message = comp_pcc(reference.float(), actual[name].float(), MULTICHIP_VS_SINGLE_CHIP_PCC)
        # comp_pcc returns the value itself for a pass and a diagnostic string for
        # a failure, so the number is parsed rather than assumed.
        logger.info(f"multichip vs single-chip TTNN {name}[{label}]: {message}")
        try:
            value = float(str(message).strip().split(":")[-1])
        except ValueError:
            value = float("nan")
        if value < worst[0]:
            worst = (value, name)
        assert passed, f"multichip vs single-chip {name}[{label}] below {MULTICHIP_VS_SINGLE_CHIP_PCC}: {message}"
    logger.info(f"multichip vs single-chip TTNN worst[{label}]: {worst[0]:.6f} on {worst[1]}")
