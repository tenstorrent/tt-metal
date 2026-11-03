# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Public GDN dispatch against an independent recurrence.

Forced phased, forced fused, and automatic public dispatch must agree with
the independent recurrence and retain trace/address reuse. The model's rows
recurrences (``Qwen38TTNNGDN._chunk_rows`` and ``_chunk_rows_lanes``) call the
public op on normalized rank-four q/k and must agree with the same recurrence.
"""

import hashlib
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from ttnn.operations.transformer_golden import recurrent_gated_delta_rule
from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.helpers.tp_harness import DEVICE_PARAMS, pcc
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gdn_public_adapter import chunk_public

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")

HEADS, DIM, CHUNK = gdn_module.VALUE_HEADS_PER_DEVICE, gdn_module.HEAD_DIM, gdn_module.CHUNK_SIZE
PCC_FLOOR = 0.99999


def host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], other) for other in replicas[1:]), "replicas differ across the mesh"
    return replicas[0]


def replicated(mesh_device, owned: list, value: torch.Tensor, dtype):
    """``value`` as a replicated DRAM TILE tensor, appended to ``owned``."""

    tensor = ttnn.from_torch(
        value.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    owned.append(tensor)
    return tensor


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 8_000_000}], indirect=True)
@pytest.mark.parametrize(
    "rows,nonzero_state,committed_rows",
    [(rows, nonzero, None) for rows in (32, 128, 2048) for nonzero in (False, True)]
    + [(32, True, committed) for committed in (0, 1, 5, 31)],
)
def test_gdn_merged_public_paths(mesh_device, rows, nonzero_state, committed_rows, record_property):
    heads, dim, chunk = HEADS, DIM, CHUNK
    owned, traces, observations = [], [], []

    def upload(value, dtype):
        return replicated(mesh_device, owned, value, dtype)

    ii, jj = torch.arange(chunk)[:, None], torch.arange(chunk)[None, :]
    low_i, low_j = ii < 16, jj < 16
    masks = torch.cat([(low_i & low_j).float(), (~low_i & ~low_j).float(), (~low_i & low_j).float()], 1)
    constants = tuple(
        upload(x[None, None], ttnn.float32)
        for x in (
            torch.eye(chunk),
            torch.ones(chunk, chunk).tril(),
            torch.ones(chunk, chunk),
            masks,
        )
    )
    configs = {
        "phased": ttnn.ChunkGdnPhasedProgramConfig(),
        "fused": ttnn.ChunkGdnFusedProgramConfig(),
        "default": None,
    }
    cached = None
    addresses = []
    try:
        for seed in (20260929, 20260930):
            rng = torch.Generator().manual_seed(seed)

            def random(shape):
                return torch.randn(shape, generator=rng)

            # Rounded producer bytes define the independent recurrence inputs.
            # Both source q scale folds remain in the query; no normalization occurs in the adapter.
            q = (F.normalize(random((1, rows, heads, dim)), dim=-1) * dim**-1).bfloat16()
            k = F.normalize(random((1, rows, heads, dim)), dim=-1).bfloat16()
            v = random((1, rows, heads, dim)).bfloat16()
            g = -F.softplus(random((1, rows, heads))) * 0.5
            beta = torch.sigmoid(random((1, rows, heads)))
            if committed_rows is not None:
                # The MTP commit masks beta and log-decay, making the suffix an identity update.
                beta[:, committed_rows:] = 0
                g[:, committed_rows:] = 0
            state = random((1, heads, dim, dim)) * 0.1 if nonzero_state else torch.zeros(1, heads, dim, dim)
            q_c, k_c = [
                upload(x.permute(0, 2, 1, 3).reshape(heads, rows // chunk, chunk, dim), ttnn.bfloat16) for x in (q, k)
            ]
            g_c, beta_c = [
                upload(x.permute(0, 2, 1).reshape(heads, rows // chunk, chunk, 1), ttnn.float32) for x in (g, beta)
            ]
            value = upload(v.reshape(1, 1, rows, heads * dim), ttnn.bfloat16)
            initial = upload(state, ttnn.float32)
            addresses.append(q_c.buffer_address())
            assert len(set(addresses)) == len(addresses), "keep prior seed inputs alive to force fresh addresses"
            args = (q_c, k_c, value, beta_c, g_c, initial, constants)

            def invoke(kind):
                return chunk_public(*args, rows_total=rows, program_config=configs[kind])

            outputs, actual_by_kind, path_cache = {}, {}, {}
            for kind in configs:
                before_path = mesh_device.num_program_cache_entries()
                outputs[kind] = invoke(kind)
                owned.extend(outputs[kind])
                actual_by_kind[kind] = tuple(host(t) for t in outputs[kind])
                path_cache[kind] = dict(before=before_path, after=mesh_device.num_program_cache_entries())
            if seed == 20260929:
                assert (
                    path_cache["fused"]["after"] > path_cache["phased"]["after"]
                ), "forced fused must compile a distinct program"
            assert (
                path_cache["default"]["after"] == path_cache["default"]["before"]
            ), "default must reuse one of the explicitly warmed configurations"
            actual = actual_by_kind["phased"]
            reference = recurrent_gated_delta_rule(
                q.float(),
                k.float(),
                v.float(),
                beta,
                g,
                scale=1.0,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm=False,
            )
            expected = (reference[0].permute(0, 2, 1, 3).reshape(heads, rows, dim), reference[1])
            correlations = {kind: [pcc(a, b) for a, b in zip(pair, expected)] for kind, pair in actual_by_kind.items()}
            diagnostic = dict(
                seed=seed,
                rows=rows,
                nonzero_state=nonzero_state,
                committed_rows=committed_rows,
                public_configs={kind: repr(config) for kind, config in configs.items()},
                pcc=correlations,
                path_program_cache=path_cache,
                output_sha256={kind: [digest(x) for x in pair] for kind, pair in actual_by_kind.items()},
            )
            logger.debug("GDN_MERGED_BEFORE_GATES=" + json.dumps(diagnostic, sort_keys=True))
            for kind, pair in actual_by_kind.items():
                assert all(torch.isfinite(t).all() for t in pair), kind
                assert min(correlations[kind]) >= PCC_FLOOR, (kind, correlations[kind])
                assert all(
                    torch.equal(a, b) for a, b in zip(pair, actual)
                ), f"{kind} must be bitwise identical to explicit phased"
            assert torch.equal(host(initial), state), "public op must not mutate the persistent input state"
            before = mesh_device.num_program_cache_entries()
            if cached is not None:
                assert before == cached
            cached = before
            trace_ids, captured = {}, {}
            for kind in configs:
                tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                try:
                    captured[kind] = invoke(kind)
                finally:
                    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                trace_ids[kind] = tid
                traces.append(tid)
                owned.extend(captured[kind])
                # These two tensors are intentional trace destinations. Inputs and
                # constants stay tracked; no broad allocation scope is exempted.
                for destination in captured[kind]:
                    TraceAllocationTracker.acknowledge_corruptible(destination)
                for _ in range(3):
                    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
                    for observed, wanted in zip(captured[kind], actual):
                        assert torch.equal(host(observed), wanted), f"{kind} trace replay differs from the eager run"
            assert mesh_device.num_program_cache_entries() == cached, "the trace replays compiled a program"
            diagnostic.update(cache_entries=cached, fresh_input_addresses=list(addresses))
            observations.append(diagnostic)
            for tid in trace_ids.values():
                ttnn.release_trace(mesh_device, tid)
                traces.remove(tid)
        record_property("gdn_merged_public_paths", json.dumps(observations, sort_keys=True))
        logger.debug("GDN_MERGED_PUBLIC_PATHS=" + json.dumps(observations, sort_keys=True))
    finally:
        for tid in traces:
            ttnn.release_trace(mesh_device, tid)
        for tensor in owned:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)


def unit_rows(generator, shape) -> torch.Tensor:
    """BF16 rows of unit L2 norm over the head dim: q and k as ``_make_chunk_inputs`` lands them."""

    return F.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()


def recurrence(q, k, v, beta, g, state):
    """The independent recurrence on ``[1, T, HV, *]`` rows with the model's scale: the head-major output
    ``[HV, T, V]`` and the final state ``[1, HV, K, V]``."""

    output, final_state = recurrent_gated_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        beta,
        g,
        scale=DIM**-0.5,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm=False,
    )
    return output.permute(0, 2, 1, 3).reshape(HEADS, q.shape[1], DIM), final_state


def chunk_constants(mesh_device, owned: list, rows: int) -> SimpleNamespace:
    """The rows constants ``_chunk_rows`` reads: the tile rows and the eye / tril / ones / quadrant-mask tiles."""

    tiles = {
        name: replicated(mesh_device, owned, tile, ttnn.float32)
        for name, tile in gdn_module.chunk_constant_tiles().items()
    }
    return SimpleNamespace(tile_rows=rows, **tiles)


def assert_matches_recurrence(output, final_state, expected, label: str) -> None:
    for name, got, want in zip(("output", "final state"), (host(output), host(final_state)), expected):
        assert got.shape == want.shape, f"{label} {name}: shape {tuple(got.shape)}, expected {tuple(want.shape)}"
        score = pcc(got, want)
        assert score >= PCC_FLOOR, f"{label} {name}: PCC {score:.7f} against the recurrence"


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "rows,accepted",
    [(32, None), (32, 0), (32, 5), (32, 31), (128, None), (128, 77), (2048, None), (2048, 1500)],
)
def test_gdn_rows_recurrence_is_the_public_op_on_normalized_rows(mesh_device, rows, accepted):
    """``Qwen38TTNNGDN._chunk_rows`` (the 32-row chunk, the 128-row chunk, the slab) runs the public op on normalized
    rank-four q/k with scale ``HEAD_DIM**-0.5``, flat v and token-major beta / g, masked past the committed rows
    (``arange <= accepted``) when a mask is given; it matches the recurrence and leaves the initial state alone."""

    owned = []
    generator = torch.Generator().manual_seed(20261026 + rows)
    q, k = (unit_rows(generator, (1, rows, HEADS, DIM)) for _ in range(2))
    v = torch.randn((1, rows, HEADS, DIM), generator=generator).bfloat16()
    g = -F.softplus(torch.randn((1, rows, HEADS), generator=generator)) * 0.5
    beta = torch.sigmoid(torch.randn((1, rows, HEADS), generator=generator))
    state = torch.randn((1, HEADS, DIM, DIM), generator=generator) * 0.1
    try:
        rows_state = SimpleNamespace(
            constants=chunk_constants(mesh_device, owned, rows),
            q=replicated(mesh_device, owned, q, ttnn.bfloat16),
            k=replicated(mesh_device, owned, k, ttnn.bfloat16),
            v=replicated(mesh_device, owned, v.reshape(1, 1, rows, HEADS * DIM), ttnn.bfloat16),
            beta=replicated(mesh_device, owned, beta.reshape(1, 1, rows, HEADS), ttnn.float32),
            g=replicated(mesh_device, owned, g.reshape(1, 1, rows, HEADS), ttnn.float32),
        )
        initial = replicated(mesh_device, owned, state, ttnn.float32)
        committed_mask = None
        if accepted is not None:
            keep = (torch.arange(rows) <= accepted).float()
            committed_mask = replicated(mesh_device, owned, keep.reshape(1, 1, rows, 1), ttnn.float32)
            beta, g = beta * keep[None, :, None], g * keep[None, :, None]
        gdn = object.__new__(gdn_module.Qwen38TTNNGDN)
        output, final_state = gdn._chunk_rows(rows_state, initial, committed_mask)
        owned += [output, final_state]
        assert_matches_recurrence(
            output, final_state, recurrence(q, k, v, beta, g, state), f"{rows} rows, accepted {accepted}"
        )
        assert torch.equal(host(initial), state), "the rows recurrence wrote its initial state"
    finally:
        for tensor in owned:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_gdn_lane_rows_recurrence_is_the_single_stream_public_op_per_lane(mesh_device):
    """``Qwen38TTNNGDN._chunk_rows_lanes`` on three lanes committing 32, 6 and 1 rows (``arange < committed``): each
    lane matches the recurrence on its own rows and state, and equals ``_chunk_rows`` on that lane bitwise."""

    lanes, committed = 3, (32, 6, 1)
    owned = []
    generator = torch.Generator().manual_seed(20261027)
    q, k = (unit_rows(generator, (lanes, CHUNK, HEADS, DIM)) for _ in range(2))
    v = torch.randn((lanes, CHUNK, HEADS, DIM), generator=generator).bfloat16()
    g = -F.softplus(torch.randn((lanes, CHUNK, HEADS), generator=generator)) * 0.5
    beta = torch.sigmoid(torch.randn((lanes, CHUNK, HEADS), generator=generator))
    state = torch.randn((lanes, HEADS, DIM, DIM), generator=generator) * 0.1
    keep = (torch.arange(CHUNK)[None, :] < torch.tensor(committed)[:, None]).float()  # [lanes, 32]
    gdn = object.__new__(gdn_module.Qwen38TTNNGDN)
    try:
        constants = chunk_constants(mesh_device, owned, CHUNK)
        rows_state = SimpleNamespace(
            lanes=lanes,
            constants=constants,
            q=replicated(mesh_device, owned, q, ttnn.bfloat16),
            k=replicated(mesh_device, owned, k, ttnn.bfloat16),
            v=replicated(mesh_device, owned, v.reshape(1, lanes, CHUNK, HEADS * DIM), ttnn.bfloat16),
            beta=replicated(mesh_device, owned, beta.reshape(1, lanes, CHUNK, HEADS), ttnn.float32),
            g=replicated(mesh_device, owned, g.reshape(1, lanes, CHUNK, HEADS), ttnn.float32),
        )
        initial = replicated(mesh_device, owned, state, ttnn.float32)
        mask = replicated(mesh_device, owned, keep.reshape(1, lanes, CHUNK, 1), ttnn.float32)
        output, final_state = gdn._chunk_rows_lanes(rows_state, initial, mask)
        owned += [output, final_state]
        outputs, states = host(output), host(final_state)
        assert torch.equal(host(initial), state), "the lane recurrence wrote its initial states"
        for lane in range(lanes):
            lane_keep = keep[lane][None, :, None]
            lane_beta, lane_g = beta[lane : lane + 1] * lane_keep, g[lane : lane + 1] * lane_keep
            expected = recurrence(
                q[lane : lane + 1], k[lane : lane + 1], v[lane : lane + 1], lane_beta, lane_g, state[lane : lane + 1]
            )
            got = (outputs[lane * HEADS : (lane + 1) * HEADS], states[lane : lane + 1])
            for name, actual, want in zip(("output", "final state"), got, expected):
                score = pcc(actual, want)
                assert score >= PCC_FLOOR, f"lane {lane} {name}: PCC {score:.7f} against the recurrence"
            single = SimpleNamespace(
                constants=constants,
                q=replicated(mesh_device, owned, q[lane : lane + 1], ttnn.bfloat16),
                k=replicated(mesh_device, owned, k[lane : lane + 1], ttnn.bfloat16),
                v=replicated(mesh_device, owned, v[lane : lane + 1].reshape(1, 1, CHUNK, HEADS * DIM), ttnn.bfloat16),
                beta=replicated(mesh_device, owned, beta[lane : lane + 1].reshape(1, 1, CHUNK, HEADS), ttnn.float32),
                g=replicated(mesh_device, owned, g[lane : lane + 1].reshape(1, 1, CHUNK, HEADS), ttnn.float32),
            )
            single_mask = replicated(mesh_device, owned, keep[lane].reshape(1, 1, CHUNK, 1), ttnn.float32)
            single_state = replicated(mesh_device, owned, state[lane : lane + 1], ttnn.float32)
            single_output, single_final = gdn._chunk_rows(single, single_state, single_mask)
            owned += [single_output, single_final]
            assert torch.equal(host(single_output), got[0]), f"lane {lane} output differs from the single stream's"
            assert torch.equal(host(single_final), got[1]), f"lane {lane} final state differs from the single stream's"
    finally:
        for tensor in owned:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
