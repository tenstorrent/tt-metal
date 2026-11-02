# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic (not for merge): does a chunk's result depend on its position in a packed forward?

Chunk X (user 0, tokens [0, 5120)) runs at position 0 with an empty partner, at position 1 behind user 1's chunk,
and at position 0 ahead of user 1's chunk. X's rows after every layer and its KDA carries are compared across the
three. Run with AttnRes and with the plain residual stream to tell an AttnRes effect from a KDA / dense-FFN one.
Logs only.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.prefill_chunk import PrefillChunk
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import test_k3_prefill_multiuser as mu
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import PlainResidualStream


def _tap_rows(mesh_device, captured, k, n, sp):
    """Chunk k's rows (all of them) of every tapped layer, [layers, rows, emb] on host."""
    out = {}
    chunk_local = mu.CHUNK // sp
    for idx, host in captured.items():
        rows = [host[(c * n + k) * chunk_local : (c * n + k + 1) * chunk_local] for c in range(sp)]
        out[idx] = torch.cat(rows)
    return out


@pytest.mark.parametrize("residual", ["attnres", "plain"])
@mu._common
def test_k3_packed_position_probe(
    variant, config_only, mesh_device, device_params, num_links, weight_cache_path, residual
):
    sp = mesh_device.shape[mu.SP_AXIS]
    mesh_device.enable_program_cache()
    config = mu._config(config_only)
    if residual == "plain":
        orig = mu.TtKimiK3Transformer

        def build(*args, **kwargs):
            return orig(
                *args, residual_factory=lambda hidden, block_residual=None: PlainResidualStream(hidden), **kwargs
            )

        build.check_cache_complete = orig.check_cache_complete
        mu.TtKimiK3Transformer, saved = build, orig
    try:
        model, kvpe = mu._build(mesh_device, config, device_params, num_links, weight_cache_path, 6, forward_chunks=2)
    finally:
        if residual == "plain":
            mu.TtKimiK3Transformer = saved
    try:
        ids = mu._token_ids(config, 2, mu.CHUNK)
        ids_by_slot = {0: ids[0], 1: ids[0], 2: ids[1], 3: ids[0], 4: ids[1]}
        C = mu.CHUNK
        E = PrefillChunk(0, 0, 0)

        def run(chunks, k):
            captured = {}
            tokens = mu._forward_input(mesh_device, chunks, ids_by_slot)
            out = model.forward(
                tokens,
                kvpe,
                actual_isl=C,
                chunks=chunks,
                layer_tap=lambda idx, h: captured.__setitem__(
                    idx,
                    ttnn.to_torch(
                        h,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)
                        ),
                    ).to(torch.float32)[0, 0],
                ),
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(tokens)
            if out is not None:
                ttnn.deallocate(out)
            return _tap_rows(mesh_device, captured, k, len(chunks), sp)

        alone = run([PrefillChunk(0, 0, C), E], 0)
        second = run([PrefillChunk(2, 0, C), PrefillChunk(1, 0, C)], 1)
        first = run([PrefillChunk(3, 0, C), PrefillChunk(4, 0, C)], 0)
        for idx in sorted(alone):
            for name, other in (("pos1-behind-real", second), ("pos0-ahead-of-real", first)):
                d = (alone[idx] - other[idx]).abs()
                bad_rows = (d.amax(dim=-1) > 0).nonzero().flatten()
                logger.info(
                    f"PROBE {residual} layer {idx} {name}: max|diff| {d.max().item():.3e} rows differing "
                    f"{bad_rows.numel()}/{d.shape[0]} first {bad_rows[:8].tolist()}"
                )
        for layer in model.schedule.rank_kda_layer_ids():
            ref = mu._kda_slot(mesh_device, model.kda_states, 0, layer)
            for name, slot in (("pos1-behind-real", 1), ("pos0-ahead-of-real", 3)):
                got = mu._kda_slot(mesh_device, model.kda_states, slot, layer)
                logger.info(
                    f"PROBE {residual} kda layer {layer} {name}: recurrent max|diff| "
                    f"{(ref[0] - got[0]).abs().max().item():.3e} convolution max|diff| "
                    f"{(ref[1] - got[1]).abs().max().item():.3e}"
                )
    finally:
        model.release_sub_device_managers()
