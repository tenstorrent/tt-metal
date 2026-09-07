# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only guards on the Gemma 4 pipeline-parallel layer split.

No device, no weights. These cover the arithmetic that a wrong answer to would produce a
model that RUNS and is silently wrong -- a rank building sliding attention where its global
layers are, or picking up another layer's learned scalar -- rather than one that errors.
"""

import json
from pathlib import Path

import pytest

from models.demos.common.prefill.adapter import PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.runner_utils import compute_layer_split

# The 31B layer pattern: 60 layers, every 6th is full_attention.
GLOBAL_LAYERS = tuple(range(5, 60, 6))
LAYER_TYPES = tuple("full_attention" if i in GLOBAL_LAYERS else "sliding_attention" for i in range(60))


def _params(**over):
    base = dict(
        mesh_shape=(8, 1),
        num_layers=15,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=262144,
        chunk_size=8192,
        num_users=2,
        capacity_factor=8,
        num_links=2,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=True,
        weight_cache_path=None,
        use_trace=True,
    )
    base.update(over)
    return PrefillRunParams(**base)


def test_adapter_accepts_a_pipeline_rank_at_8x1():
    """The (8,4)/single-rank rejection is gone, but the CP arithmetic still holds."""
    adapter = get_adapter("gemma4_31b")
    adapter._validate(_params(first_layer_idx=17, num_layers=13, is_first_rank=False, is_last_rank=False))


def test_adapter_rejects_a_chunk_too_small_for_the_sliding_window():
    """chunk 4096 at CP=8 gives a 512-token Q slab, half the 1024-token window."""
    adapter = get_adapter("gemma4_31b")
    with pytest.raises(ValueError, match="1024"):
        adapter._validate(_params(chunk_size=4096))


def test_adapter_rejects_a_tp_that_does_not_divide_the_global_kv_heads():
    """Global layers have 4 KV heads, not 16 -- TP=8 divides sliding but not global."""
    adapter = get_adapter("gemma4_31b")
    with pytest.raises(ValueError, match="global KV"):
        adapter._validate(_params(mesh_shape=(4, 8), chunk_size=8192))


def test_even_split_puts_three_globals_on_two_of_four_stages():
    """The reason PREFILL_PP_LAYER_COUNTS exists for this model.

    10 globals do not divide by 4, and a global layer costs several times a sliding one at
    depth, so an even layer split is NOT an even work split.
    """
    split = compute_layer_split(60, 4)
    counts = [sum(1 for i in range(f, f + n) if LAYER_TYPES[i] == "full_attention") for f, n in split]
    assert [n for _, n in split] == [15, 15, 15, 15]
    assert counts == [2, 3, 2, 3]


def test_the_shipped_layer_counts_take_sliding_layers_off_the_slow_stages():
    """17/13/17/13 keeps the unavoidable 2/3/2/3 global split but rebalances the sliding ones."""
    counts = [int(x) for x in json.loads(_manifest_path().read_text())["env"]["PREFILL_PP_LAYER_COUNTS"].split(",")]
    assert sum(counts) == 60
    windows, start = [], 0
    for c in counts:
        windows.append((start, c))
        start += c
    per_stage = [
        (
            sum(1 for i in range(f, f + n) if LAYER_TYPES[i] == "full_attention"),
            sum(1 for i in range(f, f + n) if LAYER_TYPES[i] == "sliding_attention"),
        )
        for f, n in windows
    ]
    assert per_stage == [(2, 15), (3, 10), (2, 15), (3, 10)]
    # In sliding-equivalents at the ~6.7x global/sliding ratio measured at depth, the spread
    # between the fastest and slowest stage is what caps pipeline throughput (1/max(stage)).
    cost = [6.7 * g + s for g, s in per_stage]
    assert max(cost) / min(cost) < 1.10, f"stage spread too wide: {cost}"


def test_shipped_layer_counts_are_the_optimum_over_every_split():
    """Guards PREFILL_PP_LAYER_COUNTS against being "tidied" back to an even split.

    Pipeline throughput is 1/max(stage), so the split that matters is the one with the smallest
    SLOWEST stage. With a global layer costing ~7.4x a sliding one at depth, that is NOT the most
    even layer count -- 17,13,17,13 is the unique minimum over all 32,509 contiguous splits, and
    the even 15,15,15,15 measures 12.04 s against 11.64 s end to end.
    """
    from models.demos.gemma4.tests.perf.pp4.optimal_layer_split import (
        GLOBAL_COST,
        SLIDING_COST,
        enumerate_splits,
    )

    types = list(LAYER_TYPES)
    results = sorted(enumerate_splits(60, 4, types, SLIDING_COST, GLOBAL_COST))
    shipped = tuple(
        int(x) for x in json.loads(_manifest_path().read_text())["env"]["PREFILL_PP_LAYER_COUNTS"].split(",")
    )
    assert results[0][1] == shipped, f"shipped {shipped} is not optimal; best is {results[0][1]}"
    assert sum(1 for r in results if r[0] == results[0][0]) == 1, "expected a unique optimum"


def test_kv_cache_layer_window_is_sliced_not_truncated():
    """A later rank must allocate PACKED global caches where ITS global layers are.

    Truncating layer_types to a prefix (the single-rank shortcut) gives rank 1 the types of
    layers 0..12 for layers 17..29 -- sliding caches under global attention, which does not
    error, it computes the wrong thing.
    """
    first, count = 17, 13
    window = LAYER_TYPES[first : first + count]
    assert [first + i for i, t in enumerate(window) if t == "full_attention"] == [17, 23, 29]
    prefix = LAYER_TYPES[:count]
    assert [i for i, t in enumerate(prefix) if t == "full_attention"] == [5, 11]
    assert window != prefix


def test_weight_cache_path_is_qualified_by_the_stage_mesh():
    """A [8,1] stage must not silently reuse the TP=4 cache; its failure mode is a deadlock."""
    adapter = get_adapter("gemma4_31b")
    assert adapter.weight_cache_path((8, 1)).name.endswith("_mesh8x1")
    assert adapter.weight_cache_path((8, 4)).name.endswith("_mesh8x4")


def _manifest_path():
    return Path("models/demos/gemma4/tt/runners/manifests/gemma4_31b_pp4.json")


def test_pp4_manifest_matches_the_topology_yaml():
    env = json.loads(_manifest_path().read_text())["env"]
    assert env["PREFILL_MODEL"] == "gemma4_31b"
    assert (int(env["PREFILL_SP"]), int(env["PREFILL_TP"])) == (8, 1)
    assert int(env["PREFILL_NUM_LAYERS"]) == 60
    assert int(env["PREFILL_CHUNK_SIZE"]) == 8192

    import yaml

    topo = yaml.safe_load(
        Path(
            "models/demos/common/prefill/runners/topology_configuration/"
            "gemma4_pipeline_prefill_4rank_8x1.yaml"
        ).read_text()
    )
    genv = topo["global_env"]
    # ttrun applies global_env to EVERY rank while an exported var reaches rank 0 only, so any
    # knob the run depends on has to be here AND has to agree with the manifest.
    for key in ("PREFILL_SP", "PREFILL_TP", "PREFILL_NUM_LAYERS", "PREFILL_CHUNK_SIZE", "PREFILL_PP_LAYER_COUNTS"):
        assert genv[key] == env[key], key
    assert len(topo["rank_bindings"]) == 4
    # Gemma 4's CP collective (ring_joint SDPA) is Topology.Linear and its 8x4 baseline opens plain
    # FABRIC_2D, so there is no wrap to match -- unlike the Mistral binding this is ported from. The
    # fabric mode and the descriptor's dim_types have to agree or a collective hangs.
    assert genv["PREFILL_FABRIC_MODE"] == "2d"
    assert "torus" not in topo["mesh_graph_desc_path"]
    assert Path(topo["mesh_graph_desc_path"]).exists()
