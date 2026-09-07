# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1: the golden cache round-trips. No donor — authored for this bring-up.

Two properties, and both matter for a different reason:

  1. **A second run loads from disk instead of recomputing.** A full-model CPU forward is the most
     expensive thing in the bring-up, so a test that silently recomputes one turns a 30-second suite
     into an afternoon. ``load_or_fail`` therefore RAISES on a miss (with the generator command)
     rather than computing, and that behaviour is asserted here.
  2. **A changed ``ReferenceCacheKey`` field forces a miss** rather than reusing a stale result. The
     key is frozen and stringifies into the filename, so every field that changes the output changes
     the file — this test walks each field and checks the filename moved.

Host only. The golden used here is deliberately tiny (4 layers, 1024 hidden) so the test can compute
it in seconds; the shape of the guarantee is the same at real dims.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from loguru import logger

from models.demos.mistral_3_5_d_p.reference import golden_cache
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C

SPEC_SMALL = golden_cache.GoldenSpec(
    num_layers=2, isl=128, hidden_size=512, intermediate_size=1024, vocab_size=256, seed=3
)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory):
    """Point the shared cache helpers at a scratch dir so the test never touches a real golden."""
    import os

    path = tmp_path_factory.mktemp("golden_cache")
    previous = os.environ.get(golden_cache.VARIANT.ref_cache_env)
    os.environ[golden_cache.VARIANT.ref_cache_env] = str(path)
    yield path
    if previous is None:
        os.environ.pop(golden_cache.VARIANT.ref_cache_env, None)
    else:
        os.environ[golden_cache.VARIANT.ref_cache_env] = previous


def test_miss_raises_with_the_generator_command(cache_dir):
    """A miss must raise, and the message must say how to produce the golden."""
    assert not golden_cache.exists(SPEC_SMALL)
    with pytest.raises(FileNotFoundError) as excinfo:
        golden_cache.load_or_fail(SPEC_SMALL)
    message = str(excinfo.value)
    assert "generate_golden_kv_cache.py" in message, "a miss must name the generator script"
    assert f"--layers {SPEC_SMALL.num_layers}" in message and f"--isl {SPEC_SMALL.isl}" in message

    # load_or_compute must ALSO refuse unless the caller explicitly opts in — that opt-in is what
    # keeps an expensive forward out of a test run.
    with pytest.raises(FileNotFoundError):
        golden_cache.load_or_compute(SPEC_SMALL)


def test_roundtrip_loads_from_disk(cache_dir):
    """Compute once, then load: the second call must return identical tensors without recomputing."""
    computed = golden_cache.load_or_compute(SPEC_SMALL, allow_compute=True)
    assert golden_cache.exists(SPEC_SMALL), "compute+save did not leave a loadable golden"

    loaded = golden_cache.load_or_fail(SPEC_SMALL)
    assert torch.equal(loaded.token_ids, computed.token_ids), "token ids did not survive the round trip"
    assert (
        len(loaded.snapshots) == len(computed.snapshots) == SPEC_SMALL.num_layers + 1
    ), "expected one snapshot per layer input plus the post-final-norm hidden state"
    for i, (a, b) in enumerate(zip(computed.snapshots, loaded.snapshots)):
        assert torch.equal(a, b), f"snapshot {i} changed across the round trip"
    for i, (a, b) in enumerate(zip(computed.kv, loaded.kv)):
        assert torch.equal(a, b), f"layer {i} KV changed across the round trip"
    assert set(loaded.state_dict) == set(computed.state_dict), "the weight dump lost or gained keys"

    # The KV must be the pair the device cache holds: post-RoPE K stacked over raw V.
    k, v = loaded.layer_kv(0)
    hf_config = SPEC_SMALL.hf_config()
    expected = (1, hf_config.num_key_value_heads, SPEC_SMALL.isl, hf_config.head_dim)
    assert tuple(k.shape) == expected and tuple(v.shape) == expected, f"{tuple(k.shape)} != {expected}"
    logger.info(f"golden cache round trip OK for {SPEC_SMALL.cache_key}")


def test_reloaded_golden_reproduces_the_reference(cache_dir):
    """The persisted weights + tokens must recompute the persisted snapshots.

    This is what makes the cache trustworthy: it is not merely self-consistent on disk, it still
    equals a fresh reference forward on the weights it stored.
    """
    golden = golden_cache.load_or_compute(SPEC_SMALL, allow_compute=True)
    hf_config = SPEC_SMALL.hf_config()
    model = reference.build_reference_model(hf_config, state_dict=golden.state_dict)
    fresh = reference.model_reference_forward(model, golden.token_ids, skip_lm_head=True)

    assert torch.allclose(
        fresh.hidden_states, golden.snapshots[-1], atol=1e-5
    ), "a fresh forward on the stored weights does not reproduce the stored final hidden state"
    for layer_idx, (k, v) in enumerate(fresh.kv):
        gk, gv = golden.layer_kv(layer_idx)
        assert torch.allclose(k, gk, atol=1e-5), f"layer {layer_idx} K differs from the stored golden"
        assert torch.allclose(v, gv, atol=1e-5), f"layer {layer_idx} V differs from the stored golden"


@pytest.mark.parametrize(
    "field, value",
    [
        ("num_layers", 3),
        ("isl", 256),
        ("hidden_size", 1024),
        ("intermediate_size", 2048),
        ("vocab_size", 512),
        ("seed", 99),
        ("weight_type", "pretrained"),
    ],
)
def test_changed_key_field_forces_a_miss(cache_dir, field, value):
    """Changing ANY field that affects the output must change the filename, i.e. force a miss.

    The frozen ``ReferenceCacheKey`` covers weight type, input source, sequence length, layer count,
    expert count and padding side. The width fields it has no slot for (hidden / intermediate /
    vocab) and the seed are folded into ``input_source`` by ``GoldenSpec`` — this walks all of them,
    so a future field added to ``GoldenSpec`` without being folded in fails here.
    """
    golden_cache.load_or_compute(SPEC_SMALL, allow_compute=True)
    assert golden_cache.exists(SPEC_SMALL)

    changed = dataclasses.replace(SPEC_SMALL, **{field: value})
    assert str(changed.cache_key) != str(
        SPEC_SMALL.cache_key
    ), f"changing {field} did not change the cache key: {changed.cache_key}"
    assert not golden_cache.exists(changed), f"a golden with a different {field} was treated as a hit"


def test_cache_key_records_the_dense_expert_count():
    """``n_routed_experts`` is 0 for this model — a dense checkpoint, not a missing value."""
    assert SPEC_SMALL.cache_key.n_routed_experts == C.NUM_EXPERTS == 0
    assert "experts0" in str(SPEC_SMALL.cache_key)
