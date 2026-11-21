# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The CPU half of the K3.1 teacher-feed harnesses (`teacher_feed.py`, `dsa_teacher_feed.py`, `moe_teacher_feed.py`):
the input replay, the scoring measures, the bars and the run lifecycle, on tiny hand-checkable data. No device."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import HfStream
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST
from models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_teacher_feed import BARS, NRMSE_GROWTH, score
from models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_teacher_feed import HostMoE, owner_positions, route, routing_window
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import (
    FFN,
    Trace,
    check_bars,
    chunk_path,
    finish_run,
    missed_keys,
    replay_live_and_sealed,
    replay_missing,
    rms_norm,
    start_run,
    topk_recall,
)


def _tiny_trace(path: Path, layers: int, rows: int, dim: int, chunk_rows: int) -> dict[str, torch.Tensor]:
    """A chunked_group_a_v1 trace of random bf16 streams (decoder_input_layer_0, attn_output and FFN output of
    every layer below ``layers``), in ``chunk_rows``-row shards."""
    gen = torch.Generator().manual_seed(3)
    keys = ["decoder_input_layer_0"] + [
        f"{s}_layer_{l}" for l in range(layers) for s in ("attn_output", "decoder_output")
    ]
    streams = {k: torch.randn(rows, dim, generator=gen).to(torch.bfloat16) for k in keys}
    index = {}
    for key, value in streams.items():
        chunks = []
        for lo in range(0, rows, chunk_rows):
            hi = min(lo + chunk_rows, rows)
            rel = f"{key}/rows_{lo:08d}_{hi:08d}.safetensors"
            (path / key).mkdir(parents=True, exist_ok=True)
            save_file({key: value[lo:hi].contiguous()}, str(path / rel))
            chunks.append({"row_start": lo, "row_end": hi, "path": rel})
        index[key] = {"row_count": rows, "dtype": "bfloat16", "shape_tail": [dim], "chunks": chunks}
    (path / "index.json").write_text(json.dumps({"tensor_streams": index}))
    (path / "metadata.json").write_text(json.dumps({"n_prompt_tokens": rows}))
    return streams


@pytest.mark.parametrize(
    "layer, sealed_at", [(1, (1, 1)), (12, (1, 2)), (13, (2, 2))], ids=["layer1", "layer12", "layer13"]
)
def test_replay_matches_the_reference_walk(tmp_path: Path, layer: int, sealed_at: tuple[int, int]):
    """The tracer's replay rule equals the reference AttnRes walk (`hf_walk.hf_layer`: seal at every block start,
    the write is `+=`) driven with the trace's own module outputs, bit for bit, at both read sites of a layer -
    before its attention and before its FFN (after its own seal and attention) - for the first MoE layer, a layer
    that seals and one past the seal, in a row window that straddles a shard. ``sealed_at``: the snapshots each site
    reads (attention, FFN): layer 12 seals between its two reads."""
    streams = _tiny_trace(tmp_path, layer + 1, rows=10, dim=8, chunk_rows=4)
    stream = HfStream(streams["decoder_input_layer_0"], block_size=12, eps=1e-5)
    seen = {}
    for l in range(layer + 1):
        if l == layer:
            seen["attention"] = (stream.running_sum, stream.block_residual)
        if l % 12 == 0:
            stream.seal()
        stream.accumulate(streams[f"attn_output_layer_{l}"])
        if l == layer:
            seen[FFN] = (stream.running_sum, stream.block_residual)
        stream.accumulate(streams[f"decoder_output_layer_{l}"])
    for (site, (live_ref, sealed_ref)), count in zip(seen.items(), sealed_at):
        live, sealed = replay_live_and_sealed(Trace(tmp_path), 3, 9, layer, site)
        assert sealed.shape[1] == sealed_ref.shape[1] == count, site
        assert torch.equal(live, live_ref[3:9]), site
        assert torch.equal(sealed, sealed_ref[3:9]), site


def test_rms_norm_rounds_where_the_model_reference_does():
    """x = (3, 4): rms = sqrt(12.5), x / rms = (0.848528, 1.131371). The reference rounds that to bf16 (step 2^-8
    below 1, 2^-7 above): (0.84765625, 1.1328125), then scales by the bf16 weight 1.5 and rounds again (step 2^-7):
    1.271484 -> 1.2734375, and 1.69921875, a tie, -> 1.703125 (even). One rounding of the fp32 product would give
    1.697056 -> 1.6953125 for the second element, so the test sees which order is implemented."""
    out = rms_norm(torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16), torch.tensor([1.5, 1.5]))
    assert out.dtype == torch.bfloat16
    assert out.float().tolist() == [[1.2734375, 1.703125]]


def test_topk_recall_and_missed_keys_by_hand():
    """Row 0: trace keys 5 / 6 / 7 / 8 scored 4 / 3 / 2 / 1, device {5, 7, 9, 8}: recall 3/4, one miss, key 6, at
    rank 1 of 4 (top half) and (3 - 1) / (4 - 1) = 2/3 of the spread. Row 1: two valid trace keys (and two -1), both
    found: recall 1, no miss. A device -1 (no key) matches nothing, and duplicates count once."""
    want = torch.tensor([[5, 6, 7, 8], [1, 0, -1, -1]])
    got = torch.tensor([[5, 7, 9, 8], [0, 1, -1, -1]])
    scores = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 5.0, 0.0, 0.0]])
    assert topk_recall(got, want).tolist() == pytest.approx([3 / 4, 1.0])
    misses = missed_keys(got, want, scores)
    assert misses["row"].tolist() == [0] and misses["rank"].tolist() == [1] and misses["selected"].tolist() == [4]
    assert misses["spread_share"].tolist() == pytest.approx([2 / 3])
    assert topk_recall(torch.tensor([[-1, 0, 0]]), torch.tensor([[0, 1, 2]])).tolist() == pytest.approx([1 / 3])


def test_check_bars_flags_each_breach():
    """A window under a min bar, over a max bar and NRMSE growth past NRMSE_GROWTH are each named; NaN passes."""
    good = {name: bar for name, (_, bar) in BARS.items()}
    good |= {"start": 0, "end": 1, "attn_nrmse_median": 0.04, "recall_min_below_topk": float("nan")}
    growth = ("attn_nrmse_median", NRMSE_GROWTH)
    assert check_bars([good, {**good, "start": 1, "end": 2}], BARS, growth) == []
    bad = {
        **good,
        "start": 1,
        "end": 2,
        "kv_latent_pcc": 0.99,
        "miss_top_half_fraction": 0.5,
        "attn_nrmse_median": 0.05,
    }
    failures = check_bars([good, bad], BARS, growth)
    assert [f.split()[0] for f in failures] == ["kv_latent_pcc", "miss_top_half_fraction", "attn_nrmse_median"]


def test_a_new_run_clears_stale_outputs_and_score_needs_completion(tmp_path: Path, expect_error):
    """A run started in a directory an earlier run used leaves none of its chunks or caches behind, and `score`
    refuses the directory until the run is marked complete."""
    for stale in (chunk_path(tmp_path, 7), tmp_path / "caches.pt", tmp_path / "metrics.json"):
        stale.write_text("old")
    start_run(tmp_path, {"layer": 92, "count": 1, "chunk": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST]
    with expect_error(ValueError, "no complete device run"):
        score(trace=None, out=tmp_path, window=1)
    finish_run(tmp_path, {"layer": 92, "count": 1, "chunk": 1}, 1.0)
    assert json.loads((tmp_path / MANIFEST).read_text())["complete"] is True


def test_a_run_refuses_its_input_dir_or_a_foreign_manifest(tmp_path: Path, expect_error):
    """`start_run` clears nothing in the run's input directory or in a directory whose manifest is not a device
    run's (here an input's: segments, no chunk)."""
    source = tmp_path / "input"
    source.mkdir()
    (source / MANIFEST).write_text(json.dumps({"layer": 92, "segment": 4096, "segments": []}))
    with expect_error(ValueError, "input directory"):
        start_run(source, {"layer": 92, "count": 1, "chunk": 1, "input": str(source)})
    with expect_error(ValueError, "not a device run"):
        start_run(source, {"layer": 92, "count": 1, "chunk": 1, "input": str(tmp_path / "other")})
    assert json.loads((source / MANIFEST).read_text())["segment"] == 4096


def test_replay_names_what_a_layer_input_needs(tmp_path: Path):
    """A trace with every stream below layer 3 (not layer 3's own attention output, as the 1M trace lacks
    attn_output_layer_1): layer 3's attention input is rebuildable, its MoE input is not."""
    trace_dir = tmp_path / "t"
    trace_dir.mkdir()
    _tiny_trace(trace_dir, 3, rows=2, dim=4, chunk_rows=2)
    trace = Trace(trace_dir)
    assert replay_missing(trace, 3) == [] and replay_missing(trace, 3, FFN) == [3]
    assert replay_missing(trace, 4, FFN) == [3, 4]


def _tiny_moe():
    """A structurally exact, scaled-down K3 latent MoE (`modeling_kimi_moe.KimiSparseMoeBlock`, SiTU, latent norm,
    two shared experts) with a non-zero selection bias."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.configuration_kimi_k3 import KimiLinearConfig
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiSparseMoeBlock

    torch.manual_seed(5)
    cfg = KimiLinearConfig(
        hidden_size=64,
        routed_expert_hidden_size=32,
        moe_intermediate_size=16,
        intermediate_size=64,
        num_experts=12,
        num_experts_per_token=4,
        num_shared_experts=2,
        num_expert_group=1,
        topk_group=1,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=KimiK3Config.ROUTE_SCALE,
        latent_moe_use_norm=True,
        hidden_act="situ",
        activation_situ_beta=KimiK3Config.ACTIVATION_SITU_BETA,
        activation_situ_linear_beta=KimiK3Config.ACTIVATION_SITU_LINEAR_BETA,
        rms_norm_eps=KimiK3Config.RMS_NORM_EPS,
    )
    block = KimiSparseMoeBlock(cfg).eval()
    with torch.no_grad():
        block.gate.e_score_correction_bias.normal_(std=0.05)
        block.routed_expert_norm.weight.normal_(1.0, 0.1)
    return block


def test_router_and_host_moe_match_the_model_reference():
    """`route` selects the reference gate's top-k set with its weights, and `HostMoE` (reading each expert only
    when a row picks it) reproduces the reference latent MoE block's output, both in fp32."""
    block = _tiny_moe()
    x = torch.randn(9, 64)
    with torch.no_grad():
        ref_ids, ref_w = block.gate(x.unsqueeze(0))
        ref_out = block(x.unsqueeze(0)).squeeze(0)
    ids, weights, choice = route(x, block.gate.weight, block.gate.e_score_correction_bias, top_k=4)
    assert torch.equal(ids.sort(dim=1).values, ref_ids.sort(dim=1).values)
    assert torch.allclose(weights.gather(1, ids.argsort(dim=1)), ref_w.gather(1, ref_ids.argsort(dim=1)), atol=1e-6)
    assert (choice.gather(1, ids).diff(dim=1) <= 0).all(), "the selection is sorted best first"
    w = {n: p.detach() for n, p in block.named_parameters() if not n.startswith("experts.")}
    experts = lambda e: tuple(getattr(block.experts[e], m).weight.detach() for m in ("w1", "w3", "w2"))  # noqa: E731
    host = HostMoE(w, experts, top_k=4)
    with torch.no_grad():
        got = host(x)
    assert torch.allclose(got, ref_out, atol=1e-5, rtol=1e-5), (got - ref_out).abs().max()
    assert len(host.experts) == int(ids.unique().numel())


def test_routing_window_by_hand():
    """Row 0: trace experts 3 / 5 / 7 / 1 at weights 0.4 / 0.3 / 0.2 / 0.1, fp32 choice order 3 > 5 > 7 > 1; the
    device picks {3, 5, 2, 1} at 0.5 / 0.25 / 0.15 / 0.1: recall 3/4, expert 7 missed at rank 2 of 4 (bottom half),
    dropped trace weight 0.2, largest weight difference on the shared picks |0.4 - 0.5| = 0.1. Row 1: the same set
    in another order: recall 1, nothing dropped."""
    want = torch.tensor([[3, 5, 7, 1], [0, 1, 2, 3]])
    want_w = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.25, 0.25, 0.25, 0.25]])
    got = torch.tensor([[3, 5, 2, 1], [3, 2, 1, 0]])
    got_w = torch.tensor([[0.5, 0.25, 0.15, 0.1], [0.25, 0.25, 0.25, 0.25]])
    choice = torch.tensor([[4.0, 3.0, 2.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    r = routing_window(got, got_w, want, want_w, choice, num_experts=8)
    assert r["recall"].tolist() == pytest.approx([0.75, 1.0])
    assert r["dropped"].tolist() == pytest.approx([0.2, 0.0])
    assert r["weight_err"].tolist() == pytest.approx([0.1, 0.0])
    assert r["misses"]["row"].tolist() == [0] and r["misses"]["rank"].tolist() == [2]


def test_owner_positions_by_hand():
    """SP rows of 8 chunk rows, TP 2. One row block: TP column 1 of SP row 1 owns chunk rows 12-15. Two blocks: it
    owns the second half of each block, rows 8 + (2, 3, 6, 7). Four padding rows (an SP row of 12, one block): column 1
    owns block rows 6-11, of which 6 and 7 are real. Every layout places every chunk row exactly once."""
    kw = dict(rows_per_sp_row=8, tp=2)
    assert owner_positions(1, 1, pad_rows=0, row_blocks=1, **kw)[1].tolist() == [12, 13, 14, 15]
    assert owner_positions(1, 1, pad_rows=0, row_blocks=2, **kw)[1].tolist() == [10, 11, 14, 15]
    owned, pos = owner_positions(1, 1, pad_rows=4, row_blocks=1, **kw)
    assert owned.tolist() == [0, 1] and pos.tolist() == [14, 15]
    for pad, blocks in ((0, 1), (0, 2), (4, 1), (8, 2)):
        rows = torch.cat(
            [owner_positions(r, p, pad_rows=pad, row_blocks=blocks, **kw)[1] for r in range(3) for p in range(2)]
        )
        assert sorted(rows.tolist()) == list(range(24)), (pad, blocks)
