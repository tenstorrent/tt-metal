# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The CPU half of the K3.1 teacher-feed harnesses (`teacher_feed.py`, `dsa_teacher_feed.py`, `moe_teacher_feed.py`):
the input replay, the scoring measures, the bars and the run lifecycle, on tiny hand-checkable data; and the DSA FP64
golden (`dsa_fp64_golden.py`) against independent references and through its resumable steps. No device, no ttnn."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.model import IndexerCPU, ModelArgs
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import HfStream
from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_k3_mla import KimiMLAAttention, KimiRMSNorm
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST, segment_path
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import dsa_fp64_golden as golden
from models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_teacher_feed import BARS, NRMSE_GROWTH, score
from models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_teacher_feed import (
    TOP_K,
    HostMoE,
    owner_positions,
    route,
    routing_window,
    same_rows,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.moe_teacher_feed import score as score_moe
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import (
    FFN,
    Trace,
    build_input,
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


def _write_trace(path: Path, streams: dict[str, torch.Tensor], chunk_rows: int) -> None:
    """``streams`` as a chunked_group_a_v1 trace in ``path``, in ``chunk_rows``-row shards."""
    index = {}
    for key, value in streams.items():
        rows = value.shape[0]
        chunks = []
        for lo in range(0, rows, chunk_rows):
            hi = min(lo + chunk_rows, rows)
            rel = f"{key}/rows_{lo:08d}_{hi:08d}.safetensors"
            (path / key).mkdir(parents=True, exist_ok=True)
            save_file({key: value[lo:hi].contiguous()}, str(path / rel))
            chunks.append({"row_start": lo, "row_end": hi, "path": rel})
        index[key] = {
            "row_count": rows,
            "dtype": str(value.dtype),
            "shape_tail": list(value.shape[1:]),
            "chunks": chunks,
        }
    (path / "index.json").write_text(json.dumps({"tensor_streams": index}))
    (path / "metadata.json").write_text(json.dumps({"n_prompt_tokens": next(iter(streams.values())).shape[0]}))


def _tiny_trace(path: Path, layers: int, rows: int, dim: int, chunk_rows: int) -> dict[str, torch.Tensor]:
    """A chunked_group_a_v1 trace of random bf16 streams (decoder_input_layer_0, attn_output and FFN output of
    every layer below ``layers``), in ``chunk_rows``-row shards."""
    gen = torch.Generator().manual_seed(3)
    keys = ["decoder_input_layer_0"] + [
        f"{s}_layer_{l}" for l in range(layers) for s in ("attn_output", "decoder_output")
    ]
    streams = {k: torch.randn(rows, dim, generator=gen).to(torch.bfloat16) for k in keys}
    _write_trace(path, streams, chunk_rows)
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


def test_same_rows_names_the_chunks_that_differ(tmp_path: Path):
    """Two complete runs of one input agree chunk by chunk, or `same_rows` names each chunk where any of the MoE
    output, expert ids or weights differs (here one bf16 ulp of one output element in chunk 1)."""
    run = {"layer": 1, "input": "in", "count": 3, "chunk": 2}
    chunks = [
        {"moe_out": torch.ones(2, 4, dtype=torch.bfloat16), "expert_ids": torch.zeros(2, 16, dtype=torch.int16),
         "expert_weights": torch.full((2, 16), 1 / 16, dtype=torch.bfloat16)}
        for _ in range(2)
    ]  # fmt: skip
    for name, order in (("a", "forward"), ("b", "reverse")):
        start_run(tmp_path / name, {**run, "order": order})
        for c, payload in enumerate(chunks):
            torch.save(payload, chunk_path(tmp_path / name, c))
        finish_run(tmp_path / name, {**run, "order": order}, 1.0)
    assert same_rows(tmp_path / "a", tmp_path / "b") == {"chunks": 2, "differ": [], "orders": ["forward", "reverse"]}
    bumped = {**chunks[1], "moe_out": chunks[1]["moe_out"].clone()}
    bumped["moe_out"][1, 3] = 1.0078125  # (1 + 2^-7: the next bf16 above 1)
    torch.save(bumped, chunk_path(tmp_path / "b", 1))
    assert same_rows(tmp_path / "a", tmp_path / "b")["differ"] == [1]


def test_moe_score_places_rows_and_misses_by_context_position(tmp_path: Path):
    """A 5-row run in 2-row chunks (the last one partial) scored in 2-row windows: rows 0-3 match the trace
    exactly; row 4, alone in the partial chunk and the last window, has its output scaled by 1.1 (per-row NRMSE
    0.1) and one of the trace's 16 experts replaced by an expert outside the trace's set (recall 15/16, one miss,
    trace weight 1/16 dropped). Every window and the miss must land at row 4's position."""
    dim, experts, count, chunk = 8, 32, 5, 2
    gen = torch.Generator().manual_seed(7)
    out = torch.randn(count, dim, generator=gen).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(experts, generator=gen)[:TOP_K] for _ in range(count)]).to(torch.int32)
    weights = torch.full((count, TOP_K), 1 / TOP_K, dtype=torch.bfloat16)
    (tmp_path / "trace").mkdir()
    _write_trace(
        tmp_path / "trace",
        {"decoder_output_layer_7": out, "expert_ids_layer_7": ids, "expert_weights_layer_7": weights},
        chunk_rows=3,
    )
    x = torch.randn(count, dim, generator=gen).to(torch.bfloat16)
    build_input(
        out=tmp_path / "input",
        expected={"layer": 7},
        count=count,
        segment=4,
        make=lambda lo, hi: (x[lo:hi], {}),
        check_bars={},
    )
    run = {"layer": 7, "input": str(tmp_path / "input"), "count": count, "chunk": chunk}
    start_run(tmp_path / "run", run)
    device_ids = ids.clone()
    outside = next(e for e in range(experts) if e not in ids[4].tolist())
    device_ids[4, 5] = outside
    device_out = out.clone()
    device_out[4] = (out[4].float() * 1.1).to(torch.bfloat16)
    for c, lo in enumerate(range(0, count, chunk)):
        hi = min(lo + chunk, count)
        torch.save(
            {"start": lo, "end": hi, "moe_out": device_out[lo:hi], "expert_ids": device_ids[lo:hi].to(torch.int16),
             "expert_weights": weights[lo:hi]},
            chunk_path(tmp_path / "run", c),
        )  # fmt: skip
    finish_run(tmp_path / "run", run, 1.0)
    gate, bias = torch.randn(experts, dim, generator=gen), torch.zeros(experts)
    result = score_moe(trace=Trace(tmp_path / "trace"), out=tmp_path / "run", window=2, gate=gate, bias=bias)
    windows = result["windows"]
    assert [(w["start"], w["end"]) for w in windows] == [(0, 2), (2, 4), (4, 5)]
    want_nrmse = float((device_out[4].float() - out[4].float()).norm() / out[4].float().norm())  # 0.1, bf16-rounded
    assert [w["moe_nrmse_median"] for w in windows] == pytest.approx([0.0, 0.0, want_nrmse], abs=1e-6)
    assert want_nrmse == pytest.approx(0.1, abs=0.01)
    assert [w["misses"] for w in windows] == [0, 0, 1]
    assert [w["recall_vs_trace_mean"] for w in windows] == pytest.approx([1.0, 1.0, 15 / 16])
    assert [w["dropped_trace_weight_mean"] for w in windows] == pytest.approx([0.0, 0.0, 1 / 16])
    assert result["count"] == count and result["overall"]["misses"] == 1


# --- FP64 golden (dsa_fp64_golden.py) ---------------------------------------------------------------------------

SMALL = golden.Dims(heads=4, nope=8, rope=4, v=6, lora=16, index_heads=4, index_dim=16, topk=6, eps=1e-5)


def _weights(dims: golden.Dims, hidden: int, q_lora: int, seed: int = 5) -> dict[str, torch.Tensor]:
    """Random FP64 weights of the layer's attention (checkpoint names), scaled by fan-in."""
    gen = torch.Generator().manual_seed(seed)
    shapes = {
        "q_a_proj.weight": (q_lora, hidden),
        "q_a_layernorm.weight": (q_lora,),
        "q_b_proj.weight": (dims.heads * (dims.nope + dims.rope), q_lora),
        "kv_a_proj_with_mqa.weight": (dims.lora + dims.rope, hidden),
        "kv_a_layernorm.weight": (dims.lora,),
        "kv_b_proj.weight": (dims.heads * (dims.nope + dims.v), dims.lora),
        "g_proj.weight": (dims.heads * dims.v, hidden),
        "o_proj.weight": (hidden, dims.heads * dims.v),
        "indexer.wq_b.weight": (dims.index_heads * dims.index_dim, q_lora),
        "indexer.wk.weight": (dims.index_dim, hidden),
        "indexer.k_norm.weight": (dims.index_dim,),
        "indexer.k_norm.bias": (dims.index_dim,),
        "indexer.weights_proj.weight": (dims.index_heads, hidden),
    }
    assert set(shapes) == set(golden.WEIGHTS)
    out = {}
    for name, shape in shapes.items():
        t = torch.randn(*shape, generator=gen, dtype=torch.float64)
        out[name] = 1 + 0.1 * t if len(shape) == 1 and not name.endswith("bias") else t / shape[-1] ** 0.5
    return out


def _small_layer(seq: int = 20, hidden: int = 32, q_lora: int = 24):
    w = _weights(SMALL, hidden, q_lora)
    x = torch.randn(seq, hidden, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    return w, x, golden.layer_keys(x, w, SMALL), golden.layer_queries(x, w, SMALL)


def test_fp64_indexer_matches_the_deepseek_v32_reference():
    """The golden's indexer score and top-k equal `cpu_deepseek_v32.IndexerCPU` (NoPE, no Hadamard/fp8: the K3.1
    configuration) on random weights. The reference's LayerNorm and score einsum run in fp32, so scores agree to fp32
    rounding (a semantic difference is O(1)), and the top-k sets match exactly on every row with a full selection."""
    seq, hidden, q_lora = 20, 32, 24
    w, x, keys, q = _small_layer(seq, hidden, q_lora)
    args = ModelArgs(
        max_batch_size=1,
        max_seq_len=seq,
        dim=hidden,
        q_lora_rank=q_lora,
        index_n_heads=SMALL.index_heads,
        index_head_dim=SMALL.index_dim,
        index_topk=SMALL.topk,
        index_rope_dim=0,
        index_k_norm_eps=SMALL.eps,
    )
    ref = IndexerCPU(args, use_fp8_path=False)
    ref.wq_b.weight = torch.nn.Parameter(w["indexer.wq_b.weight"])
    ref.wk.weight = torch.nn.Parameter(w["indexer.wk.weight"])
    ref.k_norm.weight = torch.nn.Parameter(w["indexer.k_norm.weight"].float())
    ref.k_norm.bias = torch.nn.Parameter(w["indexer.k_norm.bias"].float())
    ref.weights_proj.weight = torch.nn.Parameter(w["indexer.weights_proj.weight"].float())
    ref.k_cache = torch.zeros(1, seq, SMALL.index_dim, dtype=torch.float64)  # the bf16 buffer would round the keys
    qr = golden.rms64(x @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"], SMALL.eps)
    mask = torch.full((seq, seq), float("-inf")).triu(1)
    with torch.no_grad():
        ref_top, ref_scores = ref(x.unsqueeze(0), qr.unsqueeze(0), 0, None, mask)
    for r in range(seq):
        got = golden.index_scores(q["index_q"][r], q["index_w"][r], keys["index_key"][: r + 1])
        want = ref_scores[0, r, : r + 1].double()
        assert (got - want).abs().max() <= 1e-5 * want.abs().max(), r
        if r + 1 >= SMALL.topk:
            assert set(got.topk(SMALL.topk).indices.tolist()) == set(ref_top[0, r].tolist()), r


def test_fp64_attention_matches_the_kimi_reference_layer(monkeypatch):
    """The golden's absorbed attention over each row's selection equals the unabsorbed model reference
    (`KimiMLAAttention`: per-head k/v from kv_b_proj, NoPE k_rot expanded, SDPA, gate before o_proj) masked to the
    same selection, in FP64, to 1e-12 NRMSE. The reference's RMSNorm upcasts to fp32; it runs in FP64 here, with the
    model's eps (1e-5, as vLLM and the device; the trimmed reference defaults to 1e-6)."""
    seq, hidden, q_lora = 20, 32, 24
    w, x, keys, q = _small_layer(seq, hidden, q_lora)

    def rms_fp64(self, h):
        return self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

    monkeypatch.setattr(KimiRMSNorm, "forward", rms_fp64)
    config = SimpleNamespace(
        hidden_size=hidden,
        num_attention_heads=SMALL.heads,
        num_key_value_heads=SMALL.heads,
        q_lora_rank=q_lora,
        qk_rope_head_dim=SMALL.rope,
        kv_lora_rank=SMALL.lora,
        v_head_dim=SMALL.v,
        qk_nope_head_dim=SMALL.nope,
        mla_use_nope=True,
        mla_use_output_gate=True,
    )
    ref = KimiMLAAttention(config).double().eval()
    ref.load_state_dict({n: w[n] for n in golden.WEIGHTS if not n.startswith("indexer.")})
    ref.q_a_layernorm.variance_epsilon = ref.kv_a_layernorm.variance_epsilon = SMALL.eps
    mask = torch.full((1, 1, seq, seq), float("-inf"), dtype=torch.float64)
    selections = []
    for r in range(seq):
        scores = golden.index_scores(q["index_q"][r], q["index_w"][r], keys["index_key"][: r + 1])
        selections.append(scores.topk(min(SMALL.topk, r + 1)).indices)
        mask[0, 0, r, selections[-1]] = 0.0
    with torch.no_grad():
        want = ref(x.unsqueeze(0), attention_mask=mask)[0]
    for r, sel in enumerate(selections):
        got = golden.attend(q, r, keys["latent"][sel], keys["kpe"][sel], w, SMALL)
        assert ((got - want[r]).norm() / want[r].norm()) <= 1e-12, r


def test_fp8_e4m3_and_cache_formats_by_hand():
    """Block [100, 1, -1, 0.5]: amax 100, scale 100/448, so 100 -> 448 (the e4m3 maximum, exact) -> 100, and 1 ->
    4.48 -> 4.5 (e4m3 step 0.5 in [4, 8)) -> 1.00446, and 0.5 -> 2.24 -> 2.25 (step 0.25 in [2, 4)). With ue8m0 the scale rounds up to 0.25: 100 -> 400, a tie
    between 384 and 416 (step 32 in [256, 448]) -> 384 (even) -> 96, and 1 -> 4 -> 1 exactly. An all-zero block stays
    zero (amax clamped at 1e-4). The bf16 format rounds latent and rope; the fp8 formats quantize only the latent."""
    x = torch.tensor([[100.0, 1.0, -1.0, 0.5]], dtype=torch.float64)
    assert golden.fp8_e4m3(x, 4, False).tolist()[0] == pytest.approx(
        [100.0, 4.5 * 100 / 448, -4.5 * 100 / 448, 2.25 * 100 / 448]
    )
    assert golden.fp8_e4m3(x, 4, True).tolist() == [[96.0, 1.0, -1.0, 0.5]]
    assert golden.fp8_e4m3(torch.zeros(1, 4, dtype=torch.float64), 4, True).tolist() == [[0.0] * 4]
    latent = torch.full((1, 128), 1.0 + 2**-12, dtype=torch.float64)
    latent[0, 0] = 100.0
    kpe = torch.full((1, 2), 1.0 + 2**-12, dtype=torch.float64)
    assert golden.cache_format(latent, kpe, "exact")[0] is latent
    l16, k16 = golden.cache_format(latent, kpe, "bf16")
    assert l16[0, 1].item() == 1.0 and k16[0, 0].item() == 1.0
    l8, k8 = golden.cache_format(latent, kpe, "fp8_ue8m0")
    assert l8[0, 0].item() == 96.0 and k8[0, 0].item() == 1.0


def test_rank_agreement_by_hand():
    """Golden order 5, 3, 9, 1, 7 (best first), 3 selected: {5, 3, 9}. A selection {5, 9, 7} recalls 2/3 and misses
    key 3 at rank 1 (top half: 2 * 1 < 3); its extra key 7 ranks 4, inside the order. Adding key 42, absent from the
    order, counts one extra beyond it; -1 entries are no key."""
    order = torch.tensor([5, 3, 9, 1, 7])
    r = golden.rank_agreement(torch.tensor([5, 9, 7]), order, 3)
    assert r == {
        "recall": pytest.approx(2 / 3),
        "misses": 1,
        "miss_rank_min": 1,
        "miss_top_half": 1,
        "extra_beyond_order": 0,
    }
    r = golden.rank_agreement(torch.tensor([5, 42, 9, -1]), order, 3)
    assert r["extra_beyond_order"] == 1 and r["misses"] == 1


def _golden_inputs(tmp_path: Path, rows: int = 10, hidden: int = 8, q_lora: int = 8, segment: int = 4):
    """A checkpoint holding layer 92's attention at K3.1's dimensions (tiny hidden size) and a teacher-feed input
    directory of ``rows`` bf16 rows in ``segment``-row segments (the last one shorter)."""
    w = _weights(golden.Dims(), hidden, q_lora)
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    names = {f"model.layers.92.self_attn.{n}": t.to(torch.bfloat16) for n, t in w.items()}
    save_file(names, str(checkpoint / "shard.safetensors"))
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(names, "shard.safetensors")})
    )
    source = tmp_path / "input"
    source.mkdir()
    x = torch.randn(rows, hidden, generator=torch.Generator().manual_seed(11)).to(torch.bfloat16)
    segments = []
    for i, lo in enumerate(range(0, rows, segment)):
        hi = min(lo + segment, rows)
        torch.save({"start": lo, "end": hi, "normed": x[lo:hi]}, source / f"segment_{i:06d}.pt")
        segments.append({"start": lo, "end": hi})
    (source / MANIFEST).write_text(json.dumps({"layer": 92, "count": rows, "segment": segment, "segments": segments}))
    return checkpoint, source, x


def test_fp64_golden_steps_resume_bit_identically(tmp_path: Path):
    """keys and queries over 10 rows (segments 4 / 4 / 2 and 2 / 2 / 1 sampled rows) give the same bytes when the
    last segment is lost and rerun. With fewer rows than top-k every causal key is selected, so the fp8-indexer arm
    equals the golden and the order lists every causal key once."""
    checkpoint, source, _ = _golden_inputs(tmp_path)
    out = tmp_path / "golden"
    golden.build_keys(input_dir=source, checkpoint=checkpoint, out=out, segment=4, threads=1, trace=None)
    rows = golden.sample_rows(10, 5, 0)
    assert rows == [0, 2, 4, 7, 9]
    golden.build_queries(out=out, checkpoint=checkpoint, name="t", rows=rows, segment=2, threads=1)
    for step, last in (("keys", 2), ("queries-t", 2)):
        path = segment_path(out / step, last)
        before = torch.load(path, weights_only=True)
        manifest = json.loads((out / step / MANIFEST).read_text())
        manifest["segments"].pop()
        (out / step / MANIFEST).write_text(json.dumps(manifest))
        path.unlink()
        if step == "keys":
            golden.build_keys(input_dir=source, checkpoint=checkpoint, out=out, segment=4, threads=1, trace=None)
        else:
            golden.build_queries(out=out, checkpoint=checkpoint, name="t", rows=rows, segment=2, threads=1)
        after = torch.load(path, weights_only=True)
        flat = lambda p: {k: v for k, v in p.items() if k != "out"} | {
            f"out.{a}": v for a, v in p.get("out", {}).items()
        }
        assert flat(before).keys() == flat(after).keys()
        for k, v in flat(before).items():  # bit-identical; the order's padding scores are NaN
            torch.testing.assert_close(flat(after)[k], v, rtol=0, atol=0, equal_nan=True) if torch.is_tensor(
                v
            ) else None
            assert torch.is_tensor(v) or v == flat(after)[k]
        assert len(json.loads((out / step / MANIFEST).read_text())["sessions"]) == 2
    g = golden.load_queries(out, "t")
    for i, r in enumerate(rows):
        assert sorted(g["order"][i][g["order"][i] >= 0].tolist()) == list(range(r + 1))
        assert torch.allclose(g["out"]["idx_fp8"][i], g["out"]["fp64"][i], rtol=1e-12, atol=1e-12)
        assert not torch.equal(g["out"]["kv_fp8"][i], g["out"]["fp64"][i])


def test_fp64_compare_on_a_trace_built_from_the_golden(tmp_path: Path):
    """A trace whose streams are the golden's own values in bf16 (keys, attention output, the full causal selection
    and its scores) scores near the bf16 floor against the golden; FP64 attention over the trace's selection equals
    the golden exactly, recall is 1 and the score convention ratio is 1. A device run holding the same outputs and a
    bf16 cache scores as the trace, and its own cache sits closer to the golden than the trace's FP8 one."""
    rows = 10
    checkpoint, source, _ = _golden_inputs(tmp_path, rows=rows)
    out = tmp_path / "golden"
    golden.build_keys(input_dir=source, checkpoint=checkpoint, out=out, segment=4, threads=1, trace=None)
    sampled = golden.sample_rows(rows, 4, 0)
    golden.build_queries(out=out, checkpoint=checkpoint, name="t", rows=sampled, segment=4, threads=1)
    keys = golden.load_keys(out)
    w = golden.layer_weights(checkpoint, 92)
    topk = golden.C.INDEX_TOPK
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    scores = torch.zeros(rows, topk, dtype=torch.float32)
    attn = torch.zeros(rows, w["o_proj.weight"].shape[0], dtype=torch.bfloat16)
    x = golden.input_rows(source, list(range(rows)))
    q = golden.layer_queries(x, w)
    for r in range(rows):
        s = golden.index_scores(q["index_q"][r], q["index_w"][r], keys["index_key"][: r + 1])
        indices[r, : r + 1] = torch.arange(r + 1, dtype=torch.int32)
        scores[r, : r + 1] = s.float()
        attn[r] = golden.attend(q, r, keys["latent"][: r + 1], keys["kpe"][: r + 1], w).to(torch.bfloat16)
    trace_dir = tmp_path / "trace"
    _write_trace(
        trace_dir,
        {
            "kv_post_transform_layer_92": torch.cat([keys["latent"], keys["kpe"]], -1).to(torch.bfloat16),
            "indexer_k_layer_92": keys["index_key"].to(torch.bfloat16),
            "dsa_topk_indices_layer_92": indices,
            "dsa_topk_scores_layer_92": scores,
            "attn_output_layer_92": attn,
        },
        chunk_rows=4,
    )
    device = tmp_path / "device"  # a device run that reproduced the trace, its cache holding the bf16 keys
    device.mkdir()
    for c, lo in enumerate(range(0, rows, 4)):
        payload = {"start": lo, "end": min(lo + 4, rows), "attn_out": attn[lo : lo + 4], "topk": indices[lo : lo + 4]}
        torch.save(payload, chunk_path(device, c))
    torch.save({"kvpe": torch.cat([keys["latent"], keys["kpe"]], -1).to(torch.bfloat16)}, device / "caches.pt")
    finish_run(device, {"layer": 92, "count": rows, "chunk": 4}, 1.0)
    result = golden.compare(
        out=out, name="t", trace=Trace(trace_dir), device=device, checkpoint=checkpoint, report=tmp_path / "r"
    )
    o = result["overall"]
    assert o["trace_vs_fp64_p90"] < 0.01
    assert o["fp64_tracesel_exact_vs_fp64_p90"] < 1e-12
    assert o["fp32_vs_fp64_p90"] < 0.05
    assert o["trace_recall_mean"] == 1.0 and o["trace_misses_sum"] == 0
    assert o["device_vs_fp64_median"] == o["trace_vs_fp64_median"] and o["device_recall_mean"] == 1.0
    assert o["device_vs_trace_p90"] == 0.0
    assert o["fp64_devicesel_owncache_vs_fp64_p90"] < 0.01  # bf16 keys
    assert o["fp64_tracesel_owncache_vs_fp64_p90"] > o["fp64_devicesel_owncache_vs_fp64_p90"]  # FP8 keys
    assert o["trace_score_ratio_median"] == pytest.approx(1.0, abs=1e-6)
    assert (tmp_path / "r" / "rows.tsv").read_text().count("\n") == len(sampled) + 1


def test_summaries_leave_out_rows_below_topk():
    """A row below top-k selects every causal key (recall 1, no selection error), so neither the rows >= top-k
    summary nor any position window counts it: rows 100 / 2100 / 5000 with recall 1 / 0.5 / 0.7 give window recalls
    0.5 and 0.7, not (1 + 0.5) / 2."""
    records = [
        {"row": r, "trace_recall": v, "trace_vs_fp64": e}
        for r, v, e in ((100, 1.0, 0.0), (2100, 0.5, 0.2), (5000, 0.7, 0.4))
    ]
    s = golden.summarize(records, floor=0.0, windows=2)
    assert s["overall"]["rows"] == 3 and s["overall_rows_ge_topk"]["rows"] == 2
    assert [(w["start"], w["end"], w["rows"]) for w in s["windows"]] == [(0, 2500, 1), (2500, 5001, 1)]
    assert [w["trace_recall_mean"] for w in s["windows"]] == pytest.approx([0.5, 0.7])
