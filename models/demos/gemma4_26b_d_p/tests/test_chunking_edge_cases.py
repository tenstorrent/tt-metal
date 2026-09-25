# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunk schedules tt-d-gen actually produces, driven through the runtime exactly like prefill_runner:

  * unaligned resume: prefix reuse restarts at a 32-aligned ``actual_start`` that is NOT chunk-aligned, so the
    chunk is block-cyclically ROTATED (tt-d-gen ring_sdpa_reshuffle == rotated_chip_positions) and overlaps
    positions already in the cache (rewritten with identical values);
  * short middle chunk: a chunk whose ``actual_end`` stops mid-chunk followed by a resume from inside it;
  * interleaved slots with different lengths / boundaries.

Every schedule must leave the cache equal (vs the fp32 golden KV) on [0, real_len) of each slot.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.gemma4_26b_d_p.bringup.registry import record_result
from models.demos.gemma4_26b_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.gemma4_26b_d_p.tests.test_model_prefill import golden
from models.demos.gemma4_26b_d_p.tt.model import PAD_TOKEN_ID, block_cyclic_index

CHUNK = 4096
N_LAYERS = 6
GOLDEN_SEQ = 8192

# name -> list of (slot, actual_start, actual_end); a slot's real length is its max actual_end.
SCHEDULES = {
    # plain two chunks, reference point
    "aligned": [(0, 0, 4096), (0, 4096, 8192)],
    # chunk 2 stops at 5000 (pad tail), prefix reuse resumes at align_down(5000, 32) = 4992: rotated + overlapping
    "resume_unaligned": [(0, 0, 4096), (0, 4096, 5000), (0, 4992, 8192)],
    # first chunk itself short, resume at 1024 (inside chunk 0), then a rotated tail
    "short_first_resume": [(0, 0, 1000), (0, 992, 5088), (0, 5088, 8192)],
    # two slots interleaved, different lengths and boundaries
    "interleaved_slots": [(0, 0, 4096), (1, 0, 3000), (0, 4096, 6500), (1, 2976, 7072), (0, 6496, 8192), (1, 7072, 7500)],
    # what tt-d-gen issues with kv_block_size == chunk_size: short chunk, then a chunk-aligned re-submit of it
    "resume_aligned": [(0, 0, 4096), (0, 4096, 5000), (0, 4096, 8192)],
    "interleaved_aligned": [(0, 0, 4096), (1, 0, 3000), (1, 0, 4096), (0, 4096, 6500), (1, 4096, 7500), (0, 4096, 8192)],
}
# Ring sliding SDPA needs chunk-aligned starts on SP>1 (see tt_prefill_runtime.prefill_chunk); SP=1 takes any.
UNALIGNED = {"resume_unaligned", "short_first_resume", "interleaved_slots"}


@MESH_PARAMS
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_chunking_edge_cases(mesh_device, device_params, schedule):
    g = golden(N_LAYERS, GOLDEN_SEQ, with_kv=True)
    sp, tp = tuple(mesh_device.shape)
    C_local = CHUNK // sp
    max_seq = 3 * CHUNK  # room for a rotated chunk starting past 8192 - CHUNK
    adapter = get_adapter("gemma4_26b_d_p")
    params = PrefillRunParams(mesh_shape=(sp, tp), num_layers=N_LAYERS, first_layer_idx=0, is_first_rank=True, is_last_rank=True,
                              max_seq_len=max_seq, chunk_size=CHUNK, num_users=2, capacity_factor=0, num_links=1, gate_mode_name="",
                              kv_only_last_layer=False, weight_cache_path=None)
    hf_cfg = adapter.load_hf_config()
    rt = adapter.build_runtime(mesh_device=mesh_device, hf_config=hf_cfg, params=params)
    kv = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_cfg, params=params)
    rt.compile(kv)

    sched = SCHEDULES[schedule]
    if schedule in UNALIGNED and sp > 1:
        with pytest.raises(ValueError, match="kv_block_size == chunk_size"):
            for slot, a0, a1 in sched:
                window = torch.zeros(CHUNK, dtype=torch.int64)
                rt.prefill_chunk(rt.make_chunk_input(window), kv, slot_id=slot, actual_start=a0, actual_end=a1)
        logger.info(f"[{schedule}] mesh={mesh_id(mesh_device)}: unaligned start rejected with a clear error (expected on SP={sp})")
        return
    real_len = {}
    for slot, a0, a1 in sched:
        assert a0 % 32 == 0 and a0 < a1 <= a0 + CHUNK
        window = torch.full((CHUNK,), PAD_TOKEN_ID, dtype=torch.int64)
        n = min(CHUNK, GOLDEN_SEQ - a0)
        window[:n] = g["ids"][a0 : a0 + n]
        window[a1 - a0 :] = PAD_TOKEN_ID  # tokens past actual_end are pad, as tt-d-gen sends them
        dev_order = window[block_cyclic_index(a0, sp, C_local) - a0]  # == tt-d-gen ring_sdpa_reshuffle
        rt.prefill_chunk(rt.make_chunk_input(dev_order), kv, slot_id=slot, actual_start=a0, actual_end=a1)
        real_len[slot] = max(real_len.get(slot, 0), a1)
    ttnn.synchronize_device(mesh_device)

    worst = 1.0
    for slot, L in sorted(real_len.items()):
        for layer in range(N_LAYERS):
            k_ref, v_ref = (t.float()[:, :, :L] for t in g["kv"][layer])
            D = k_ref.shape[-1]
            meta = torch.arange(D).view(2, D // 2).T.reshape(-1)
            k_dev, v_dev = rt.read_layer_kv(kv, slot, layer, L)
            pk, pv = comp_pcc(k_ref[..., meta], k_dev)[1], comp_pcc(v_ref, v_dev)[1]
            worst = min(worst, pk, pv)
            if layer in (0, N_LAYERS - 1):
                logger.info(f"[{schedule}] slot {slot} L{layer}: K {pk:.5f} V {pv:.5f} over [0,{L})")
    logger.info(f"[{schedule}] mesh={mesh_id(mesh_device)}: worst KV PCC {worst:.5f}")
    record_result(f"layer:chunking_{schedule}", mesh_id(mesh_device), float(worst), worst > 0.99, f"{sched}")
    assert worst > 0.99, worst
