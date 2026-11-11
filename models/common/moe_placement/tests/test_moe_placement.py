# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the expert placement tool (no device, no ttnn). Run with:
pytest --noconftest models/common/moe_placement/tests -q
"""

import json
from pathlib import Path

import numpy as np
import pytest

from models.common.moe_placement import abtest, calibrate, corpus, routing, screen, traffic
from models.common.moe_placement import preset as presetlib
from models.common.moe_placement.spec import MoeSpec, inverse, validate_mapping
from models.common.moe_placement.transport import contributions, linear_contributions, ring_contributions, ring_links

SPECS = [
    MoeSpec("glm_like", 256, 8),
    MoeSpec("k3_like", 896, 16, transport="linear"),
    MoeSpec("small", 128, 4, rows=4, groups=2, chunk_tokens=512, n_chunks=4),
]


# --- layout ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_identity_and_roundtrip(spec):
    ident = spec.identity_owners()
    assert spec.physical_from_owners(ident) == list(range(spec.num_experts))
    rng = np.random.default_rng(0)
    owners = ident.copy()
    for g in range(spec.groups):  # random valid placement: permute rows inside each group
        sl = slice(g * spec.group_size, (g + 1) * spec.group_size)
        owners[sl] = rng.permutation(owners[sl])
    phys = spec.physical_from_owners(owners)
    assert (spec.owners_from_physical(phys) == owners).all()
    validate_mapping(phys, spec)
    assert inverse(inverse(phys)) == phys
    # stability: an expert already on its target row keeps its slot
    for e in range(spec.num_experts):
        if owners[e] == ident[e]:
            assert phys[e] == e


def test_validate_rejects():
    spec = SPECS[0]
    m = list(range(256))
    with pytest.raises(ValueError, match="exactly once"):
        validate_mapping(m[:-1] + [0], spec)
    m2 = m.copy()
    m2[0], m2[64] = m2[64], m2[0]  # crosses a dispatch group
    with pytest.raises(ValueError, match="dispatch groups"):
        validate_mapping(m2, spec)
    with pytest.raises(ValueError):
        MoeSpec("bad", 100, 8)


def test_physical_from_owners_rejects_uneven_rows():
    spec = SPECS[2]
    owners = spec.identity_owners().copy()
    owners[0] = 1  # row 0 of group 0 now one short, row 1 one extra
    with pytest.raises(ValueError, match="needs exactly"):
        spec.physical_from_owners(owners)


def test_agrees_with_runtime_validator():
    rt = pytest.importorskip("models.demos.deepseek_v3_d_p.tt.moe.expert_placement")
    for spec in SPECS[:2]:
        rng = np.random.default_rng(1)
        owners = spec.identity_owners().copy()
        for g in range(spec.groups):
            sl = slice(g * spec.group_size, (g + 1) * spec.group_size)
            owners[sl] = rng.permutation(owners[sl])
        phys = spec.physical_from_owners(owners)
        assert rt.validate_mapping(phys, spec.num_experts) == phys


# --- transport -------------------------------------------------------------------------------------
@pytest.mark.parametrize("rows", [4, 8, 16])
def test_ring_pages_times_distance(rows):
    for n in (0, 1, 2, 3, 31, 640):
        for s in range(rows):
            for d in range(rows):
                assert ring_links(n, s, d, rows).sum() == n * min((d - s) % rows, (s - d) % rows)


@pytest.mark.parametrize("rows", [4, 8])
def test_contributions_hops(rows):
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 50, size=(3, rows, 5))
    for fn, dist in (
        (ring_contributions, lambda s, o: min((o - s) % rows, (s - o) % rows)),
        (lambda c: linear_contributions(c, 2), lambda s, o: abs(o - s)),
    ):
        disp, comb = fn(counts)
        for o in range(rows):
            want = sum(counts[:, s, :] * dist(s, o) for s in range(rows))
            assert (disp[:, :, o, :].sum(-1) == want).all()
            assert (comb[:, :, o, :].sum(-1) == want).all()
    with pytest.raises(ValueError):
        contributions(counts, "mesh")


# --- routing ---------------------------------------------------------------------------------------
def test_routing_roundtrip_and_counts(tmp_path):
    spec = SPECS[2]
    rng = np.random.default_rng(3)
    ids = np.stack(
        [
            np.stack([rng.choice(spec.num_experts, spec.top_k, replace=False) for _ in range(spec.sequence_tokens)])
            for _ in range(2)
        ]
    )
    routing.save(tmp_path / "r.npz", ids, [5, 7])
    x, layers = routing.load(tmp_path / "r.npz")
    assert layers == [5, 7] and (x == ids).all()
    c = routing.chunk_counts(ids[:, : spec.chunk_tokens], spec.rows, spec.num_experts)
    assert c.shape == (2, spec.rows, spec.num_experts) and c.sum() == 2 * spec.chunk_tokens * spec.top_k
    bad = ids.copy()
    bad[0, 0, 1] = bad[0, 0, 0]
    with pytest.raises(ValueError, match="twice"):
        routing.validate(bad, spec.num_experts, spec.top_k)


# --- optimizer -------------------------------------------------------------------------------------
@pytest.mark.parametrize("transport", ["ring", "linear"])
def test_optimize_balances_skewed_load(transport):
    spec = MoeSpec("t", 128, 4, rows=4, groups=2, chunk_tokens=512, transport=transport)
    rng = np.random.default_rng(4)
    hot = np.zeros(spec.num_experts)
    hot[:: spec.experts_per_chip * spec.rows] = 1  # pile hot experts onto row 0 of every group
    p = 0.02 + 0.5 * hot
    counts = rng.poisson(lam=np.broadcast_to(p * 40, (16, spec.rows, spec.num_experts))).astype(np.int32)
    owners = calibrate.optimize(counts, np.random.default_rng(5), spec, steps=3000)
    spec.physical_from_owners(owners)  # valid: every row of every group keeps experts_per_chip experts

    def peak(o):
        tiles = (counts.sum(1) + 31) // 32
        return sum(
            max(
                tiles[f, (np.arange(spec.num_experts) // spec.group_size == g) & (o == r)].sum()
                for g in range(spec.groups)
                for r in range(spec.rows)
            )
            for f in range(counts.shape[0])
        )

    assert peak(owners) < peak(spec.identity_owners())


# --- preset ----------------------------------------------------------------------------------------
def test_preset_roundtrip(tmp_path):
    spec = SPECS[0]
    rng = np.random.default_rng(6)
    ident = spec.identity_owners()
    moved = ident.copy()
    moved[:64] = rng.permutation(moved[:64])
    p, calib, stats = presetlib.build(spec, [3, 4], [moved, ident], "bal128_s1")
    assert list(p["layers"]) == ["3"] and stats["omitted"] == [4]
    path = tmp_path / "p.json"
    presetlib.write(path, spec, p, calib)
    loaded = presetlib.load(path, spec)
    assert loaded == p
    back = presetlib.owners_per_layer(loaded, spec, [3, 4])
    assert (back[0] == moved).all() and (back[1] == ident).all()
    assert json.loads(path.with_suffix(".calibration.json").read_text())["plan"] == "bal128_s1"
    with pytest.raises(ValueError, match="experts"):
        presetlib.load(path, SPECS[1])


# --- end to end on synthetic captures ---------------------------------------------------------------
def _synthetic_workdir(tmp_path, spec):
    """7 domains x 2 sources, each a full sequence of short requests, with domain-skewed routing."""
    rng = np.random.default_rng(7)
    T = spec.sequence_tokens
    for di, dom in enumerate(corpus.DOMAINS):
        for k in range(2):
            name = f"{dom}_src{k}"
            lengths, total = [], 0
            while total < T:
                n = min(int(rng.integers(40, 300)), T - total)
                lengths.append(n)
                total += n
            sample_of = np.repeat(np.arange(len(lengths)), lengths)
            samples = [
                dict(source=name, domain=dom, lang="en" if di % 2 else "de", kind="chat", tokens=n) for n in lengths
            ]
            corpus.write_sequence(tmp_path, name, rng.integers(0, 1000, T).tolist(), sample_of, samples)
            pref = np.full(spec.num_experts, 1.0)
            pref[di * 8 : (di + 1) * 8] = 25.0  # each domain prefers its own experts
            pref /= pref.sum()
            ids = np.stack(
                [
                    np.stack([rng.choice(spec.num_experts, spec.top_k, replace=False, p=pref) for _ in range(T)])
                    for _ in range(2)
                ]
            )
            routing.save(tmp_path / "captures" / name / "routing.npz", ids, [1, 2])
    edge = np.stack(
        [np.stack([rng.choice(spec.num_experts, spec.top_k, replace=False) for _ in range(T)]) for _ in range(2)]
    )
    corpus.write_sequence(
        tmp_path,
        "edge_random_tokens",
        [0] * T,
        [0] * T,
        [dict(source="edge", domain="edge", lang="none", kind="tokens", tokens=T)],
    )
    routing.save(tmp_path / "captures" / "edge_random_tokens" / "routing.npz", edge, [1, 2])


def test_end_to_end_host(tmp_path):
    spec = SPECS[2]
    _synthetic_workdir(tmp_path, spec)
    c = traffic.load_corpus(tmp_path)
    assert c.domains == sorted(corpus.DOMAINS) and {s["split"] for s in c.samples} == {"train", "test"}
    summ = calibrate.run(tmp_path, spec, plans=["bal32_s1", "lodo_code"], steps=400, workers=2, log=lambda *_: None)
    assert summ["peak_change_pct"]["mix_balanced"]["bal32_s1"] < 0
    assert "seq_edge_random_tokens" in summ["tests"]
    layers, owners = calibrate.load_plan(tmp_path, "calibration", "bal32_s1")
    p, calib, _ = presetlib.build(spec, layers, owners, "bal32_s1")
    presetlib.write(tmp_path / "presets" / "p.json", spec, p, calib)
    info = traffic.pack_heldout(tmp_path, tokens=spec.sequence_tokens)
    assert info["requests"] > 0
    held = routing.load(tmp_path / "heldout" / "heldout_mix_bal" / "routing_spliced.npz")[0]
    assert held.shape == (2, spec.sequence_tokens, spec.top_k)
    gain = screen.modeled_gain(held, spec, dict(plan=list(owners)))["plan"]
    assert gain < 0
    assert 0 < screen.imbalance(held, spec)["utilization"] < 1
    # the same split is used everywhere: no held-out token comes from a TRAIN request
    meta = json.loads((tmp_path / "sequences" / "heldout_mix_bal" / "samples.json").read_text())
    train_starts = {(s["seq"], s["start"]) for s in c.samples if s["split"] == "train"}
    assert not any((m["from_seq"], m["from_start"]) in train_starts for m in meta)


# --- A/B gate --------------------------------------------------------------------------------------
def _fake_run(ab, label, seconds, digest="x"):
    d = ab / "runs" / label
    d.mkdir(parents=True)
    (d / "measurement.json").write_text(
        json.dumps(
            dict(status="complete", iteration_totals=[dict(iteration=i, whole_seconds=seconds) for i in range(10)])
        )
    )
    if digest is not None:
        (d / "cache-digests.jsonl").write_text(digest)


def test_ab_plan_order():
    labels = [x[0] for x in abtest.plan(dict(control={}, mix={}))]
    assert labels[:6] == ["control-O1", "control-B1", "control-B2", "control-O2", "control-O3", "control-B3"]
    assert labels[6:] == [l.replace("control", "mix") for l in labels[:6]]


@pytest.mark.parametrize("case", ["pass", "overlap", "digest", "slower"])
def test_ab_gate(tmp_path, case):
    ab = tmp_path / "ab"
    o, b = [2.67, 2.672, 2.673], [2.65, 2.655, 2.652]
    if case == "overlap":
        b[0] = 2.68
    if case == "slower":
        o, b = b, o
    for i in range(3):
        _fake_run(ab, f"control-O{i + 1}", o[i])
        _fake_run(ab, f"control-B{i + 1}", b[i], digest="y" if case == "digest" and i == 0 else "x")
    res = abtest.summarize(ab, dict(control=dict(golden=True)))
    assert res["gate"]["passed"] == (case == "pass"), res["gate"]


def test_cli_help():
    from models.common.moe_placement.cli import build_parser

    p = build_parser()
    a = p.parse_args(["calibrate", "--workdir", "/tmp/x", "--spec", "s.json", "--plans", "bal128_s1"])
    assert a.stage == "calibrate" and a.plans == "bal128_s1"


def test_profile_shares_count_only_routed_expert_stage():
    from models.common.moe_placement.adapters.dsv3dp_bench import stage_shares

    stages = {
        "start->pre_gate": 46.0,
        "pre_gate->gate": 140.0,
        "gate->pre_routing_setup": 42.0,
        "pre_routing_setup->routing_setup": 124.0,
        "routing_setup->pre_routed_expert": 619.0,  # TP gather + dispatch + shared expert, NOT routed experts
        "pre_routed_expert->routed_expert": 793.0,
        "routed_expert->pre_combine_module": 53.0,
        "pre_combine_module->combine_module": 442.0,
        "combine_module->pre_reduce_module": 42.0,
        "pre_reduce_module->reduce_module": 154.0,
        "reduce_module->end": 241.0,
    }
    s = stage_shares(stages, 4171.0)
    assert s["routed_expert_share"] == round(793.0 / 4171.0, 4)
    assert s["moe_share"] == round(sum(stages.values()) / 4171.0, 4)
    fused = stage_shares(
        {"routing_setup->pre_routed_expert": 600.0, "pre_routed_expert->routed_expert_combine": 1200.0}, 4000.0
    )
    assert fused["routed_expert_share"] == 0.3
