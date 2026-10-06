# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""``fused/mtp_accept`` without a device: the registry entry, the kernel's constants against the Python side, the
host mirror's law (the theta rule: a tie rejects, an unkept draft rejects, u = 0 accepts every kept draft, the
resample skips the rejected draft, the bonus row draws with the second uniform) and its agreement with the device
sampler's reference draw."""

import re
from pathlib import Path

import torch

from models.demos.blackhole.qwen38_flash_next.ttnn import device_sampler as ds
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mtp_accept as ma
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import registry

KERNEL = Path(ma.__file__).parent / "kernels" / "accept.cpp"
POLICY = ds.Qwen38DeviceSamplerPolicy(temperature=0.7, top_k=20, top_p=0.8, min_p=0.0)
SENTINEL = 248320


def _rows(k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rows = torch.zeros(k + 1, ma.ROW_LANES)
    for r in range(k + 1):
        packs = rows[r].reshape(4, 2, 32)
        packs[:, 0] = (torch.randn(128, generator=g) * 3).to(torch.bfloat16).float().reshape(4, 32)
        packs[:, 1] = torch.randperm(248320, generator=g)[:128].float().reshape(4, 32)
    return rows


def _lanes(rows: torch.Tensor, r: int):
    packs = rows[r].reshape(4, 2, 32)
    return packs[:, 0].reshape(-1).to(torch.float32), packs[:, 1].reshape(-1).to(torch.int64)


def test_registered_bitwise_with_the_host_decision_as_its_composed_form(expect_error):
    kernel = registry.kernel(ma.NAME)
    assert kernel.tolerance == registry.BITWISE and kernel.fused is ma.mtp_accept
    with expect_error(RuntimeError, match="decides the pass on the host"):  # allow-pytest.raises: reads the exception
        kernel.composed()
    assert ma.NAME not in registry.DEFAULT_ON  # opt-in until the wired chain's law gate


def test_kernel_constants_match_the_python_side():
    source = KERNEL.read_text()
    for name in ("cb_stage", "rows", "lanes", "table_size"):
        assert f'get_named_compile_time_arg_val("{name}")' in source
    for lane_name, value in (
        ("STAT_WEIGHT", ma.STAT_WEIGHT),
        ("STAT_TOTAL", ma.STAT_TOTAL),
        ("STAT_GUARD", ma.STAT_GUARD),
        ("STAT_RESAMPLED", ma.STAT_RESAMPLED),
        ("STAT_ACCEPTED", ma.STAT_ACCEPTED),
        ("STAT_TOKEN", ma.STAT_TOKEN),
        ("STAT_THETA", ma.STAT_THETA),
        ("STAT_KEPT", ma.STAT_KEPT),
    ):
        assert re.search(rf"constexpr uint32_t {lane_name} = {value};", source), lane_name
    assert f"constexpr uint32_t STATS_LANES = {ma.STATS_LANES};" in source
    assert "static_assert(ROWS >= 2 && ROWS <= 6" in source  # 1..5 drafts and the bonus row
    # the stage layout the Python side sizes: tile, rows, three 32-lane rows, policy + scalars grains, stats, table chunks, work
    assert ma.stage_bytes(5) == 4096 + 5 * 1024 + 3 * 128 + 3 * 64 + 64 + 32 * 64 + (3 * 128 + 5 * 32) * 4
    assert "theta_j < static_cast<float>(w_d)" in source  # a tie rejects


def test_u_zero_accepts_every_kept_draft_and_draws_the_bonus_row_with_v():
    k = 4
    rows = _rows(k, 1)
    table = ds.weight_table(POLICY.temperature)
    kept = [ma.row_kept_set(*_lanes(rows, r), POLICY, table) for r in range(k + 1)]
    drafts = [kept[j].ids[kept[j].kept - 1] for j in range(k)]  # the last kept lane: positive weight, accepted at u = 0
    ref = ma.accept_reference(rows, drafts, POLICY, [0.0] * k + [0.5], table=table, sentinel=SENTINEL)
    assert ref.accepted == k and not ref.resampled
    assert ref.token == ma.theta_draw(kept[k], 0.5)[0]
    assert ref.alignment == tuple(drafts) + (ref.token,) + (SENTINEL,) * (32 - k - 1)
    assert ref.weights == tuple(kept[j].weights[kept[j].kept - 1] for j in range(k))
    assert ref.totals == tuple(kept[j].total for j in range(k))


def test_an_unkept_draft_rejects_and_the_resample_never_returns_it():
    k = 3
    rows = _rows(k, 2)
    table = ds.weight_table(POLICY.temperature)
    kept = [ma.row_kept_set(*_lanes(rows, r), POLICY, table) for r in range(k + 1)]
    drafts = [kept[0].ids[0], 7, kept[2].ids[0]]  # id 7 is in no row
    for n in (0, 1 << 20, (1 << 24) - 1):
        v = n * 2.0**-24
        ref = ma.accept_reference(rows, drafts, POLICY, [0.0, 0.0, 0.0, v], table=table, sentinel=SENTINEL)
        assert ref.accepted == 1 and ref.resampled and ref.weights == (kept[0].weights[0], 0)
        assert ref.token in kept[1].ids[: kept[1].kept] and ref.token != 7
        assert ref.alignment[:3] == (drafts[0], ref.token, SENTINEL)


def test_a_tie_rejects_and_the_resample_skips_the_rejected_draft():
    # a row whose kept set is one lane cannot be built (lane 0 always weighs 2**18 and u < 1), so take two kept lanes
    # and a uniform whose product with the total is exactly the draft's weight: u = w / S when S is a power of two
    k = 1
    table = ds.weight_table(1.0)
    policy = ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=2, top_p=1.0, min_p=0.0)
    rows = _rows(k, 3)
    values, ids = _lanes(rows, 0)
    values[:] = -100.0
    values[0] = 20.0
    values[1] = 20.0  # two equal top lanes: weights 2**18 each, S = 2**19
    rows[0].reshape(4, 2, 32)[:, 0] = values.reshape(4, 32)
    kept = ma.row_kept_set(values, ids, policy, table)
    assert kept.kept == 2 and kept.total == 2 * ds.WEIGHT_ONE and kept.weights[0] == kept.weights[1] == ds.WEIGHT_ONE
    top, second = kept.ids[0], kept.ids[1]
    tie = ma.accept_reference(rows, [top], policy, [0.5, 0.25], table=table, sentinel=SENTINEL)  # 0.5 * S == w: reject
    assert tie.accepted == 0 and tie.resampled and tie.token == second  # the resample skips the rejected top lane
    below = ma.accept_reference(rows, [top], policy, [0.5 - 2.0**-24, 0.25], table=table, sentinel=SENTINEL)
    assert below.accepted == 1 and not below.resampled


def test_theta_draw_without_skip_is_the_device_sampler_reference_draw():
    rows = _rows(2, 4)
    table = ds.weight_table(POLICY.temperature)
    for r in range(3):
        values, ids = _lanes(rows, r)
        kept = ma.row_kept_set(values, ids, POLICY, table)
        for n in (0, 1, 12345, 1 << 23, (1 << 24) - 1):
            u = n * 2.0**-24
            reference = ds.device_sampler_reference(values, ids, POLICY, u, table=table)
            token, _theta = ma.theta_draw(kept, u)
            assert (token, kept.kept) == (reference.token_id, reference.kept), (r, n)


def test_reference_refuses_malformed_inputs(expect_error):
    rows = _rows(2, 5)
    with expect_error(ValueError, match="uniforms"):  # allow-pytest.raises: reads the exception
        ma.accept_reference(rows, [1, 2], POLICY, [0.5, 0.5], sentinel=SENTINEL)
    with expect_error(ValueError, match="multiple of 2"):  # allow-pytest.raises: reads the exception
        ma.accept_reference(rows, [1, 2], POLICY, [0.5, 0.5, 1 / 3], sentinel=SENTINEL)
    with expect_error(ValueError, match="candidate rows"):  # allow-pytest.raises: reads the exception
        ma.accept_reference(rows[:2], [1, 2], POLICY, [0.5, 0.5, 0.5], sentinel=SENTINEL)
    with expect_error(ValueError, match="presence"):  # allow-pytest.raises: reads the exception
        ma.row_kept_set(
            *_lanes(rows, 0),
            ds.Qwen38DeviceSamplerPolicy(1.0, 20, 0.9, 0.0, presence_penalty=0.5),
            ds.weight_table(1.0),
        )
