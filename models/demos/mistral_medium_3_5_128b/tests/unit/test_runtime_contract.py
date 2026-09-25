# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P2 runtime contract on random weights (REDUCED DEPTH: 2 of 88 layers, full width):

* out-of-contract chunk ranges and inputs fail loudly in ``prefill_chunk`` / ``make_chunk_input``;
* ``compile`` warms every chunk bucket;
* a ragged multi-chunk prefill (last chunk padded) writes the same KV as an equal-length one-shot run,
  and both match the cached whole-model reference on the real (non-pad) tokens.
"""

import pytest

from models.demos.mistral_medium_3_5_128b.reference.golden import golden_forward
from models.demos.mistral_medium_3_5_128b.reference.model import random_state_dict
from models.demos.mistral_medium_3_5_128b.tt.kv_cache import naturalize
from models.demos.mistral_medium_3_5_128b.tt.rope import hf_to_meta_perm
from models.demos.mistral_medium_3_5_128b.tt.runtime import PrefillRuntime, PrefillRuntimeConfig
from models.demos.mistral_medium_3_5_128b.tt.weights import StateDictWeights

from .common import CFG, assert_pcc, spec_dtypes

NUM_LAYERS, CAPACITY, CHUNK, N_REAL = 2, 10240, 5120, 9984


@pytest.mark.timeout(2400)
def test_runtime_contract_and_chunked_equals_one_shot(galaxy_mesh, expect_error):
    cfg = CFG.reduced(num_hidden_layers=NUM_LAYERS)
    _, ids, _, kv = golden_forward(cfg, CAPACITY, seed=0)
    token_ids = ids[0, :N_REAL].tolist()
    weights = StateDictWeights(random_state_dict(cfg, seed=0))
    dt = spec_dtypes()

    def runtime(chunk):
        return PrefillRuntime(
            galaxy_mesh,
            cfg,
            weights,
            PrefillRuntimeConfig(
                num_layers=NUM_LAYERS, max_seq_len=CAPACITY, chunk_size=chunk, dtypes=dt, num_users=2, pad_token_id=11
            ),
        )

    chunked = runtime(CHUNK)
    kv_c = chunked.allocate_kv_cache()
    good = chunked.make_chunk_input([0] * CHUNK)
    for (slot, start, end), msg in [
        ((0, 128, 5248), "must be a multiple of chunk_size"),
        ((0, 10240, 10300), "exceeds the per-user KV capacity"),
        ((0, 5120, 5120), "not a non-empty range"),
        ((0, 0, 5121), "not a non-empty range"),
        ((2, 0, 5120), "slot_id 2 out of range"),
    ]:
        with expect_error(AssertionError, msg):
            chunked.prefill_chunk(good, kv_c, slot, start, end)
    with expect_error(AssertionError, "exactly chunk_size"):
        chunked.make_chunk_input([0] * (CHUNK - 32))

    chunked.compile(kv_c)
    assert chunked.compiled
    assert chunked.prefill(token_ids, kv_c, slot_id=1) == 2

    one_shot = runtime(CAPACITY)
    kv_o = one_shot.allocate_kv_cache()
    assert one_shot.prefill(token_ids, kv_o, slot_id=1) == 1

    kc, vc = chunked.read_slot_kv(kv_c, 1)
    ko, vo = one_shot.read_slot_kv(kv_o, 1)
    perm = hf_to_meta_perm(cfg.head_dim)
    for i in range(NUM_LAYERS):
        k_chunked = naturalize(kc[i], N_REAL, galaxy_mesh.shape[0], CHUNK, CAPACITY)
        v_chunked = naturalize(vc[i], N_REAL, galaxy_mesh.shape[0], CHUNK, CAPACITY)
        k_one = naturalize(ko[i], N_REAL, galaxy_mesh.shape[0], CAPACITY, CAPACITY)
        v_one = naturalize(vo[i], N_REAL, galaxy_mesh.shape[0], CAPACITY, CAPACITY)
        ref_k, ref_v = kv[i][0][0][:, :N_REAL][..., perm], kv[i][1][0][:, :N_REAL]
        assert_pcc(f"runtime_k_chunked_vs_one_shot[layer {i}, reduced 2L]", k_one, k_chunked)
        assert_pcc(f"runtime_v_chunked_vs_one_shot[layer {i}, reduced 2L]", v_one, v_chunked)
        assert_pcc(f"runtime_k_chunked_vs_ref[layer {i}, reduced 2L]", ref_k, k_chunked)
        assert_pcc(f"runtime_v_chunked_vs_ref[layer {i}, reduced 2L]", ref_v, v_chunked)
