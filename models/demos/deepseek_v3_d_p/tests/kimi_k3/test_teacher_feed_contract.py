# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The CPU half of the K3.1 teacher-feed harnesses (`teacher_feed.py`, `dsa_teacher_feed.py`, `moe_teacher_feed.py`):
the input replay, the scoring measures, the bars and the run lifecycle, on tiny hand-checkable data; and the DSA FP64
golden (`dsa_fp64_golden.py`) against independent references and through its resumable steps. No device, no ttnn."""

from __future__ import annotations

import json
from contextlib import contextmanager
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
    load_input,
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
    assert [w["recall_vs_reference_mean"] for w in windows] == pytest.approx([1.0, 1.0, 15 / 16])
    assert [w["dropped_reference_weight_mean"] for w in windows] == pytest.approx([0.0, 0.0, 1 / 16])
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


def test_dsa_missed_weight_by_hand_and_on_a_small_layer():
    """``head_mass`` sums a head's weight on the marked keys: [[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]] on keys {1, 2} is
    [0.5, 0.9]. On a small layer, the same selection on both sides misses nothing; a device that swaps the FP64
    selection's last (lowest-scored, stored best first) key for another loses that key's FP64 weight, the bottom-1
    mass is the same key's, and the intruder's mass is its weight in the softmax over the device's selection.
    ``error_split`` of FP64 output [0, 0], FP64 output over the device's selection [3, 0], device output [3, 4]:
    orthogonal parts, shares 9/25 and 16/25; with device output [6, 0] the parts are parallel (cos 1), shares 9/36
    each and the cross term the other half."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_missed_weight import error_split, head_mass, row_weights

    got = head_mass(torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]), torch.tensor([False, True, True]))
    assert torch.allclose(got, torch.tensor([0.5, 0.9]))
    w, _, keys, q = _small_layer()
    r = 15
    fp64_sel = torch.tensor([1, 4, 7, 9, 15, 12])  # best first
    same = row_weights(q, r, fp64_sel, fp64_sel.sort().values, keys["latent"], keys["kpe"], w, SMALL)
    assert same["missed"] == same["intruders"] == 0 and same["missed_key_max"] == 0.0
    assert float(same["missed_mass"].abs().max()) == float(same["intruder_mass"].abs().max()) == 0.0
    # same keys in another order: equal up to FP64 summation order
    assert torch.allclose(same["out_fp64"], same["out_device_sel"], rtol=1e-12, atol=1e-12)
    device_sel = torch.tensor([1, 4, 7, 9, 10, 15])  # 12 (the lowest FP64 score) missed, 10 intruded
    res = row_weights(q, r, fp64_sel, device_sel, keys["latent"], keys["kpe"], w, SMALL)
    wa = golden.attention_weights(q, r, keys["latent"][fp64_sel], keys["kpe"][fp64_sel], w, SMALL)
    wd = golden.attention_weights(q, r, keys["latent"][device_sel], keys["kpe"][device_sel], w, SMALL)
    assert res["missed"] == res["intruders"] == 1
    assert torch.equal(res["missed_mass"], wa[:, 5]) and torch.equal(res["bottom_mass"], wa[:, 5])
    assert torch.equal(res["intruder_mass"], wd[:, 4]) and res["missed_key_max"] == float(wa[:, 5].max())
    assert torch.allclose(wa.sum(-1), torch.ones(SMALL.heads, dtype=torch.float64))
    assert torch.equal(
        res["out_device_sel"], golden.attend(q, r, keys["latent"][device_sel], keys["kpe"][device_sel], w, SMALL)
    )
    zero, sel = torch.zeros(2, dtype=torch.float64), torch.tensor([3.0, 0.0], dtype=torch.float64)
    assert error_split(zero, sel, torch.tensor([3.0, 4.0], dtype=torch.float64)) == pytest.approx(
        {"selection_rest_cos": 0.0, "selection_share": 9 / 25, "rest_share": 16 / 25}
    )
    assert error_split(zero, sel, torch.tensor([6.0, 0.0], dtype=torch.float64)) == pytest.approx(
        {"selection_rest_cos": 1.0, "selection_share": 0.25, "rest_share": 0.25}
    )


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


# --- FP64 chain of layers 0-3 (fp64_chain.py) ----------------------------------------------------------------------


class _F64Torch:
    """``torch`` as the model reference modules see it, with ``float32`` read as ``float64``: the references hard-code
    fp32 compute, and the FP64 comparison substitutes the dtype without editing their semantics."""

    float32 = torch.float64

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture
def fp64_references(monkeypatch):
    """Run the vendored AttnRes read, `KimiRMSNorm` and the trimmed MoE reference in FP64."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3 import modeling_kimi_moe

    monkeypatch.setattr(modeling_kimi_moe, "torch", _F64Torch())
    monkeypatch.setattr(
        KimiRMSNorm,
        "forward",
        lambda self, h: self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)),
    )
    return monkeypatch


@contextmanager
def _fp64_upcasts(monkeypatch):
    """`Tensor.float` as `double` while the context lasts: the vendored `_apply_attn_res` upcasts with it. Only
    around reference code, so the code under test keeps its own casts."""
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "float", lambda self, *a, **k: self.double())
        yield


def _rel(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got.double() - want.double()).norm() / want.double().norm())


def _tiny_moe_cfg(hidden: int):
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.configuration_kimi_k3 import KimiLinearConfig

    return KimiLinearConfig(
        hidden_size=hidden,
        routed_expert_hidden_size=16,
        moe_intermediate_size=8,
        intermediate_size=24,
        num_experts=6,
        num_experts_per_token=2,
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


def _moe64_of(block, top_k: int):
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import MoE64

    w = {n: p.detach() for n, p in block.named_parameters() if not n.startswith("experts.")}
    experts = [tuple(getattr(x, m).weight.detach() for m in ("w1", "w3", "w2")) for x in block.experts]
    return MoE64(w, experts, top_k, block.routed_expert_norm.variance_epsilon)


def test_fp64_attn_res_matches_the_vendored_read(fp64_references):
    """`attn_res64` (scores as proj . rms_norm(candidate), a softmax mix of the raw candidates) equals the vendored
    HuggingFace read run in FP64 to <= 1e-14 relative, with one and two sealed snapshots; mixing the normalized
    candidates instead (a plausible slip) is O(1) away."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import hf_attn_res
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import attn_res64, rms64

    gen = torch.Generator().manual_seed(1)
    d = 32
    norm = 1 + 0.1 * torch.randn(d, generator=gen, dtype=torch.float64)
    proj = torch.randn(1, d, generator=gen, dtype=torch.float64)
    for s in (1, 2):
        live = torch.randn(5, d, generator=gen, dtype=torch.float64)
        sealed = 3 * torch.randn(5, s, d, generator=gen, dtype=torch.float64)
        with _fp64_upcasts(fp64_references):
            want = hf_attn_res(live, sealed, norm, proj, 1e-5)
        assert want.dtype == torch.float64
        assert _rel(attn_res64(live, sealed, (norm, proj), 1e-5), want) <= 1e-14
        v = torch.cat([sealed, live.unsqueeze(1)], 1)
        probs = (rms64(v, norm, 1e-5) @ proj[0]).softmax(-1)
        assert _rel(torch.einsum("ns,nsd->nd", probs, rms64(v, norm, 1e-5)), want) > 0.1


def test_fp64_dense_ffn_and_moe_match_the_model_reference(fp64_references):
    """`dense_mlp64` equals the reference `KimiMLP` and `MoE64` (experts grouped, each expert's rows at once) the
    reference `KimiSparseMoeBlock` (per expert over sorted tokens, then a weighted sum per row), all in FP64: outputs
    to <= 1e-13 relative, the routing's sets and weights to rounding, the selection best first with a positive margin
    to the next expert. Mutations (shared experts left out; the routing weights not renormalized) are O(0.01-1)."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiMLP, KimiSparseMoeBlock
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import dense_mlp64

    torch.manual_seed(5)
    hidden = 32
    cfg = _tiny_moe_cfg(hidden)
    block = KimiSparseMoeBlock(cfg).double().eval()
    with torch.no_grad():
        block.gate.e_score_correction_bias.normal_(std=0.05)
        block.routed_expert_norm.weight.normal_(1.0, 0.1)
    x = torch.randn(11, hidden, dtype=torch.float64)
    with torch.no_grad():
        ref_ids, ref_w = block.gate(x.unsqueeze(0))
        want = block(x.unsqueeze(0))[0]
    moe = _moe64_of(block, 2)
    got = moe(x)
    assert _rel(got["output"], want) <= 1e-13
    order = ref_ids.argsort(dim=1)
    assert torch.equal(got["ids"].sort(dim=1).values, ref_ids.sort(dim=1).values)
    assert torch.allclose(got["weights"].gather(1, got["ids"].argsort(dim=1)), ref_w.gather(1, order), rtol=1e-14)
    choice = (x @ block.gate.weight.T).sigmoid() + block.gate.e_score_correction_bias
    assert (choice.gather(1, got["ids"]).diff(dim=1) <= 0).all() and (got["margin"] > 0).all()
    shared = {
        f"{n}.weight": getattr(block.shared_experts, n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj")
    }
    assert _rel(got["output"] - dense_mlp64(x, shared), want) > 0.01
    with torch.no_grad():
        assert _rel(block.shared_experts(x), dense_mlp64(x, shared)) <= 1e-14
        dense = KimiMLP(cfg).double().eval()
        w = {f"{n}.weight": getattr(dense, n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj")}
        assert _rel(dense_mlp64(x, w), dense(x)) <= 1e-14
    unnormalized = _moe64_of(block, 2)
    unnormalized.route = lambda h, r=moe.route: (lambda i, w, m: (i, w * 3, m))(*r(h))
    assert _rel(unnormalized(x)["output"], want) > 0.01


def test_index_topk_matches_the_rowwise_golden():
    """The batched indexer (`index_topk`, key blocks of 7 here) scores every causal key as the row-wise
    `dsa_fp64_golden.index_scores` does to <= 1e-14 relative and selects the same top-k, best first; a row with fewer
    causal keys than top-k selects all of them and pads with -1 (score NaN); keys after a row are never selected."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_chain

    gen = torch.Generator().manual_seed(2)
    keys = torch.randn(40, 16, generator=gen, dtype=torch.float64)
    q = torch.randn(6, 4, 16, generator=gen, dtype=torch.float64)
    w = torch.randn(6, 4, generator=gen, dtype=torch.float64)
    rows = torch.tensor([0, 3, 9, 10, 25, 38])
    fp64_chain.KEY_BLOCK, block = 7, fp64_chain.KEY_BLOCK
    try:
        idx, score = fp64_chain.index_topk(q, w, keys, rows, 10)
    finally:
        fp64_chain.KEY_BLOCK = block
    for i, r in enumerate(rows.tolist()):
        want = golden.index_scores(q[i], w[i], keys[: r + 1])
        top = want.topk(min(10, r + 1))
        n = top.indices.numel()
        assert torch.equal(idx[i, :n], top.indices) and (idx[i, n:] == -1).all()
        assert ((score[i, :n] - top.values).abs() <= 1e-14 * want.abs().max()).all() and score[i, n:].isnan().all()
        assert int(idx[i].max()) <= r


def _tiny_chain(hidden: int = 32, vocab: int = 50):
    """A ChainModel with random weights at tiny dimensions, and the reference modules it was built from."""
    from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
    from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import random_weights
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiMLP, KimiSparseMoeBlock
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import ChainModel

    torch.manual_seed(9)
    gen = torch.Generator().manual_seed(9)
    eps = KimiK3Config.RMS_NORM_EPS
    kda_cfg = KDAConfig(hidden, 2, 8, 8, 4, eps, use_full_rank_gate=True, gate_lower_bound=-5.0)
    vec = lambda: 1 + 0.1 * torch.randn(hidden, generator=gen, dtype=torch.float64)  # noqa: E731
    query = lambda: (vec(), torch.randn(1, hidden, generator=gen, dtype=torch.float64))  # noqa: E731
    cfg = _tiny_moe_cfg(hidden)
    refs = {"mlp": KimiMLP(cfg).double().eval()}
    layers = {}
    for l in range(4):
        layers[l] = {
            "input_layernorm": vec(),
            "attn_res": query(),
            "mlp_res": query(),
            "post_attention_layernorm": vec(),
        }
        if l < 3:
            layers[l]["kda"] = {n: v.double() for n, v in random_weights(kda_cfg).items()}
    layers[0]["mlp"] = {
        f"{n}.weight": getattr(refs["mlp"], n).weight.detach() for n in ("gate_proj", "up_proj", "down_proj")
    }
    for l in (1, 2):
        block = KimiSparseMoeBlock(cfg).double().eval()
        with torch.no_grad():
            block.gate.e_score_correction_bias.normal_(std=0.05)
            block.routed_expert_norm.weight.normal_(1.0, 0.1)
        refs[l] = block
        layers[l]["moe"] = _moe64_of(block, 2)
    layers[3]["dsa"] = _weights(SMALL, hidden, 24)
    layers[3]["gate"] = torch.randn(6, hidden, generator=gen, dtype=torch.float64)
    layers[3]["gate_bias"] = 0.05 * torch.randn(6, generator=gen, dtype=torch.float64)
    embedding = torch.randn(vocab, hidden, generator=gen).to(torch.bfloat16)
    model = ChainModel(embedding, layers, kda_cfg, SMALL, eps, top_k=2)
    return model, refs


def _reference_walk(model, refs, token_ids: torch.Tensor, monkeypatch) -> dict:
    """Layers 0-3 over the whole sequence at once, by the vendored AttnRes walk (`hf_walk.hf_layer`), the reference
    norm and MoE modules, one-shot FP64 KDA per layer, and the DSA golden's row-wise indexer and attention."""
    from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import HfStream, hf_layer

    def norm(weight):
        n = KimiRMSNorm(weight.numel(), eps=model.eps).double()
        n.weight = torch.nn.Parameter(weight)
        return n

    got = {}
    stream = HfStream(model.embedding[token_ids].double(), model.block_size, model.eps)
    for l in range(3):
        L = model.layers[l]

        def attn_fn(h, L=L, l=l):
            out = kda_forward_reference(
                norm(L["input_layernorm"])(h)[None], L["kda"], model.kda_config, dtype=torch.float64
            )[0][0]
            got[f"attn_output_{l}"] = out
            return out

        def mlp_fn(h, L=L, l=l):
            x = norm(L["post_attention_layernorm"])(h)
            out = refs[l](x.unsqueeze(0))[0] if l else refs["mlp"](x)
            got[f"moe_input_{l}"], got[f"moe_output_{l}"] = x, out
            return out

        with torch.no_grad(), _fp64_upcasts(monkeypatch):
            hf_layer(stream, l, L["attn_res"], L["mlp_res"], attn_fn, mlp_fn)
        if l:
            got[f"routing_ids_{l}"] = refs[l].gate(got[f"moe_input_{l}"].unsqueeze(0))[0]
    L = model.layers[3]
    with _fp64_upcasts(monkeypatch):
        got["dsa_input_3"] = stream.read(L["attn_res"])
    x = norm(L["input_layernorm"])(got["dsa_input_3"]).detach()
    keys = golden.layer_keys(x, L["dsa"], SMALL)
    q = golden.layer_queries(x, L["dsa"], SMALL)
    got.update(keys)
    got["topk"] = [
        golden.index_scores(q["index_q"][r], q["index_w"][r], keys["index_key"][: r + 1]).topk(min(SMALL.topk, r + 1))
        for r in range(x.shape[0])
    ]
    got["attn_output_3"] = torch.stack(
        [
            golden.attend(q, r, keys["latent"][t.indices], keys["kpe"][t.indices], L["dsa"], SMALL)
            for r, t in enumerate(got["topk"])
        ]
    )
    return got


def test_fp64_chain_segments_match_a_one_shot_reference_walk_and_resume(tmp_path: Path, fp64_references):
    """The segmented chain (10 rows in segments 4 / 4 / 2: KDA state and DSA keys carried across, a last segment
    shorter than the others) equals a one-shot walk of independent forms in FP64 (vendored AttnRes read and seal
    schedule, reference norm / FFN / MoE modules, whole-sequence KDA, row-wise DSA golden) to <= 1e-12 relative on
    every product, with identical routing and top-k; the cross-checks against a trace built from that walk read
    exact embeddings, PCC ~1 and recall 1. Losing the last segment and rerunning rewrites it bit for bit."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_chain

    model, refs = _tiny_chain()
    rows = 10
    token_ids = torch.randint(0, 50, (rows,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = _reference_walk(model, refs, token_ids, fp64_references)
    trace_dir = tmp_path / "trace"
    topk_idx = torch.full((rows, SMALL.topk), -1, dtype=torch.int32)
    for r, t in enumerate(want["topk"]):
        topk_idx[r, : t.indices.numel()] = t.indices.int()
    streams = {
        "decoder_input_layer_0": model.embedding[token_ids],
        "attn_output_layer_0": want["attn_output_0"].to(torch.bfloat16),
        "decoder_output_layer_0": want["moe_output_0"].to(torch.bfloat16),
        "expert_ids_layer_1": want["routing_ids_1"].int(),
        "expert_ids_layer_2": want["routing_ids_2"].int(),
        "expert_ids_layer_3": torch.zeros(rows, 2, dtype=torch.int32),
        "kv_post_transform_layer_3": torch.cat([want["latent"], want["kpe"]], -1).to(torch.bfloat16),
        "indexer_k_layer_3": want["index_key"].to(torch.bfloat16),
        "dsa_topk_indices_layer_3": topk_idx,
    }
    _write_trace(trace_dir, streams, chunk_rows=4)
    out = tmp_path / "chain"
    sampled = golden.sample_rows(rows, 4, 0)
    run = lambda: fp64_chain.run_chain(  # noqa: E731
        run_dir=out,
        model=model,
        token_ids=token_ids,
        segment=4,
        sampled=sampled,
        identity={"t": 1},
        trace=Trace(trace_dir),
    )
    with torch.no_grad():
        manifest = run()
    assert [(s["start"], s["end"]) for s in manifest["segments"]] == [(0, 4), (4, 8), (8, 10)]
    names = ["moe_input_1", "moe_output_1", "moe_input_2", "moe_output_2", "dsa_input_3", "latent", "kpe", "index_key"]
    got = fp64_chain.load_products(out, names + ["routing_ids_1", "routing_ids_2", "topk_indices_3"])
    for n in names:
        assert got[n].dtype == (torch.float64 if n in ("latent", "kpe", "index_key") else torch.float32), n
    for n in ("latent", "kpe", "index_key"):
        assert _rel(got[n], want[n]) <= 1e-12, n
    for n in names[
        :5
    ]:  # stored fp32: equal to the walk at fp32 rounding (the FP64 values themselves are checked below)
        assert _rel(got[n], want[n]) <= 1e-7, n
    for l in (1, 2):
        assert torch.equal(got[f"routing_ids_{l}"].long().sort(1).values, want[f"routing_ids_{l}"].sort(1).values)
    assert torch.equal(got["topk_indices_3"], topk_idx)
    attn = torch.cat(
        [torch.load(fp64_chain.segment_path(out, i), weights_only=True)["attn_output_3"] for i in range(3)]
    )
    assert attn.dtype == torch.float64 and _rel(attn, want["attn_output_3"][sampled]) <= 1e-12
    # FP64 products before the fp32 store: rerun one segment's chain in memory from the stored carried state
    carried = fp64_chain.load_carried(out, 1)
    products, _ = fp64_chain.chain_segment(model, token_ids[4:8], 4, carried, sampled)
    for n in names[:5]:
        assert _rel(products[n], want[n][4:8]) <= 1e-12, n
    checks = manifest["segments"][1]["checks"]
    assert checks["embedding_ulp_mismatches"] == 0 and checks["routing_recall_1"] == 1.0
    assert (
        checks["kv_latent_pcc"] > 0.9999 and checks["attn_output_0_pcc"] > 0.9999 and checks["topk_recall_mean"] == 1.0
    )
    assert manifest["bytes_per_row"]["topk_indices_3"] == 4 * SMALL.topk
    before = {
        p: torch.load(p, weights_only=True) for p in (fp64_chain.segment_path(out, 2), fp64_chain.state_path(out, 2))
    }
    m = json.loads((out / MANIFEST).read_text())
    m["segments"].pop()
    (out / MANIFEST).write_text(json.dumps(m))
    for p in before:
        p.unlink()
    with torch.no_grad():
        resumed = run()
    assert len(resumed["sessions"]) == 2 and len(resumed["segments"]) == 3
    flat = lambda d, pre="": {  # noqa: E731
        f"{pre}{k}": v
        for k0, v0 in d.items()
        for k, v in (flat(v0, f"{k0}.").items() if isinstance(v0, dict) else [(k0, v0)])
    }
    for p, old in before.items():
        new = flat(torch.load(p, weights_only=True))
        old = flat(old)
        assert new.keys() == old.keys()
        for k, v in old.items():
            assert (
                (torch.equal(new[k], v) or (new[k].isnan() == v.isnan()).all()) if torch.is_tensor(v) else new[k] == v
            ), k


def test_fp64_chain_refuses_a_different_run(tmp_path: Path, expect_error):
    """A resume with other tokens (or segment size) is refused, not mixed into the stored run."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_chain

    model, _ = _tiny_chain()
    ids = torch.arange(6) % 50
    with torch.no_grad():
        fp64_chain.run_chain(run_dir=tmp_path, model=model, token_ids=ids, segment=4, sampled=[5], identity={})
        with expect_error(ValueError, "token_ids_sha256"):
            fp64_chain.run_chain(run_dir=tmp_path, model=model, token_ids=ids + 1, segment=4, sampled=[5], identity={})


# --- GPU trace vs the FP64 chain (chain_vs_trace.py) ---------------------------------------------------------------


def test_chain_vs_trace_scores_hand_made_differences_incrementally(tmp_path: Path, expect_error, monkeypatch):
    """A trace made from the chain's own products with three planted differences scores them exactly: row 2's
    layer-1 set loses its FP64 rank-1 expert (recall 1/2, that expert's weight dropped, miss rank 1), the index key is
    twice the FP64 one (PCC 1, NRMSE exactly 1), row 7's selection loses the FP64 best key (recall 5/6, miss rank 0,
    spread share 1); everything else is exact (KV PCC 1, recall 1). Scoring with two of three segments landed and
    then all three reuses the cached segments; each trend's worst follows its metric's direction; the FP64 cut gap;
    a score directory of another chain, or of the same chain path at another producer version, is refused."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import chain_vs_trace, fp64_chain

    model, _ = _tiny_chain()
    rows, chain = 10, tmp_path / "chain"
    ids = torch.randint(0, 50, (rows,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        fp64_chain.run_chain(run_dir=chain, model=model, token_ids=ids, segment=4, sampled=[3, 9], identity={})
    p = fp64_chain.load_products(
        chain,
        [
            "routing_ids_1",
            "routing_weights_1",
            "routing_ids_2",
            "latent",
            "kpe",
            "index_key",
            "topk_indices_3",
            "topk_scores_3",
        ],
    )
    r1 = p["routing_ids_1"].clone()
    spare = next(e for e in range(6) if e not in r1[2].tolist())
    r1[2, 1] = spare
    topk = p["topk_indices_3"].clone()
    topk[7, 0] = 9
    _write_trace(
        tmp_path / "trace",
        {
            "expert_ids_layer_1": r1,
            "expert_ids_layer_2": p["routing_ids_2"],
            "kv_post_transform_layer_3": torch.cat([p["latent"], p["kpe"]], -1),
            "indexer_k_layer_3": 2 * p["index_key"],
            "dsa_topk_indices_layer_3": topk,
        },
        chunk_rows=4,
    )
    trace, out = Trace(tmp_path / "trace"), tmp_path / "score"
    manifest = json.loads((chain / MANIFEST).read_text())
    partial = {**manifest, "segments": manifest["segments"][:2]}
    (chain / MANIFEST).write_text(json.dumps(partial))
    assert [r["end"] for r in chain_vs_trace.score_landed(chain, trace, out)] == [4, 8]
    first = (out / "segment_000000.pt").stat().st_mtime_ns
    (chain / MANIFEST).write_text(json.dumps(manifest))
    results = chain_vs_trace.score_landed(chain, trace, out)
    assert [r["end"] for r in results] == [4, 8, 10] and (out / "segment_000000.pt").stat().st_mtime_ns == first
    s = chain_vs_trace.summarize(results, window=4)
    w0, o = s["windows"][0], s["overall"]
    assert w0["routing_1_recall_mean"] == pytest.approx((3 + 0.5) / 4) and w0["routing_1_set_equal"] == 0.75
    assert w0["routing_1_dropped_mean"] == pytest.approx(float(p["routing_weights_1"][2, 1]) / 4, rel=1e-6)
    assert o["routing_1_miss_rank_median"] == 1 and o["routing_2_recall_mean"] == 1.0
    assert o["index_key_pcc"] == pytest.approx(1.0, abs=1e-12) and o["index_key_nrmse"] == pytest.approx(1.0, rel=1e-12)
    assert o["latent_pcc"] == pytest.approx(1.0, abs=1e-12) and o["kpe_nrmse"] < 1e-7
    assert o["topk_rows"] == 5 and o["topk_recall_mean"] == pytest.approx((4 + 5 / 6) / 5)
    assert o["topk_miss_rank_median"] == 0 and o["topk_miss_top_half_fraction"] == 1.0
    assert o["topk_miss_share_median"] == pytest.approx(0.99)
    assert o["routing_3_sampled_rows"] == 0  # (the chain ran without trace checks, so it recorded none)
    assert s["trend"]["routing_1_recall_mean"]["worst_at"] == 0
    # worst follows each metric's direction: a smaller miss rank is worse; FP64 margins have no worst
    d = chain_vs_trace.direction
    assert d("topk_miss_rank_median") == d("routing_2_miss_rank_median") == d("latent_pcc") == -1
    assert d("routing_2_dropped_p99") == d("index_key_nrmse") == d("topk_miss_top_half_fraction") == 1
    assert d("routing_1_margin_median_all") is None and d("topk_cut_gap_median") is None
    assert "routing_1_margin_median_all" in s["trend"] and "worst" not in s["trend"]["routing_1_margin_median_all"]
    # the cut gap: the FP64 score at rank k-1-CUT_GAP_RANKS minus the cut (rank k-1); the chain stores scores best
    # first. k = 6 here: CUT_GAP_RANKS 100 > k-1 takes the top; CUT_GAP_RANKS 2 (as 100 < 2048 at 1M) takes rank 3.
    scores = p["topk_scores_3"]
    assert (scores[8, :-1] >= scores[8, 1:]).all()
    seg2 = chain_vs_trace.score_segment(chain, 2, trace, {})  # rows 8 and 9
    assert float(seg2["topk"]["cut_gap"][0]) == pytest.approx(float(scores[8, 0] - scores[8, 5]))
    monkeypatch.setattr(chain_vs_trace, "CUT_GAP_RANKS", 2)
    seg2 = chain_vs_trace.score_segment(chain, 2, trace, {})
    assert float(seg2["topk"]["cut_gap"][0]) == pytest.approx(float(scores[8, 3] - scores[8, 5]))
    identity = json.loads((out / "identity.json").read_text())
    assert identity["scorer_version"] == chain_vs_trace.SCORER_VERSION and identity["chain_producer_version"] == 2
    # a chain rebuilt at the same path by another producer version is not mixed with its cached scores
    (chain / MANIFEST).write_text(json.dumps({**manifest, "producer_version": 99}))
    with expect_error(ValueError, "another chain or trace"):
        chain_vs_trace.score_landed(chain, trace, out)
    (chain / MANIFEST).write_text(json.dumps(manifest))
    (out / "identity.json").write_text(json.dumps({"chain": "another"}))
    with expect_error(ValueError, "another chain or trace"):
        chain_vs_trace.score_landed(chain, trace, out)


def test_key_sums_give_pcc_and_nrmse_by_hand():
    """x = [1, 2, 3], y = [1, 2, 4]: PCC 0.98198 and NRMSE 1 / sqrt(21); rows' sums combine to the PCC / NRMSE of all
    their elements."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.chain_vs_trace import key_sums, pcc_nrmse

    p, n = pcc_nrmse(key_sums(torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[1.0, 2.0, 4.0]])))
    assert p == pytest.approx(0.9819805, rel=1e-6) and n == pytest.approx(21**-0.5)
    gen = torch.Generator().manual_seed(0)
    x, y = torch.randn(5, 7, generator=gen, dtype=torch.float64), torch.randn(5, 7, generator=gen, dtype=torch.float64)
    p, n = pcc_nrmse(key_sums(x, y))
    xf, yf = x.flatten() - x.mean(), y.flatten() - y.mean()
    assert p == pytest.approx(float(xf @ yf / (xf.norm() * yf.norm())), rel=1e-12)
    assert n == pytest.approx(float((x - y).norm() / y.norm()), rel=1e-12)


def test_chain_model_reads_each_checkpoint_tensor_into_its_slot(tmp_path: Path):
    """`ChainModel.from_checkpoint` on a tiny checkpoint whose every tensor is distinct puts each one where its
    checkpoint name says: per layer the pre-attention (`self_attention_res_*`) and pre-FFN (`mlp_res_*`) queries, the
    two norms, layer 0-2 KDA, layer 0's dense FFN, layer 1-2 MoE (non-routed weights and every routed expert as
    (w1, w3, w2)), layer 3's DSA attention and gate; the KDA config, eps and expert count come from config.json."""
    from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
    from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import random_weights
    from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_chain

    hidden, experts, eps = 16, 3, 1e-5
    kda_cfg = KDAConfig(hidden, 2, 8, 8, 4, eps, use_full_rank_gate=True, gate_lower_bound=-5.0)
    gen = torch.Generator().manual_seed(3)
    rnd = lambda *shape: torch.randn(*shape, generator=gen).to(torch.bfloat16)  # noqa: E731
    t = {"embed_tokens.weight": rnd(20, hidden)}
    for l in range(4):
        p = f"layers.{l}."
        for n in ("input_layernorm", "post_attention_layernorm", "mlp_res_norm", "self_attention_res_norm"):
            t[f"{p}{n}.weight"] = rnd(hidden)
        t[f"{p}mlp_res_proj.weight"], t[f"{p}self_attention_res_proj.weight"] = rnd(1, hidden), rnd(1, hidden)
        if l < 3:
            t.update(
                {
                    f"{p}self_attn.{n}": (v + torch.randn(v.shape, generator=gen) * 0.01).to(torch.bfloat16)
                    for n, v in random_weights(kda_cfg).items()
                }
            )
        if l == 0:
            t.update({f"{p}mlp.{n}.weight": rnd(8, hidden) for n in ("gate_proj", "up_proj", "down_proj")})
        if l >= 1:
            t[f"{p}block_sparse_moe.gate.weight"], t[f"{p}block_sparse_moe.gate.e_score_correction_bias"] = rnd(
                experts, hidden
            ), rnd(experts)
        if l in (1, 2):
            for n in fp64_chain.MoE_NON_ROUTED[2:]:
                t[f"{p}block_sparse_moe.{n}"] = rnd(4, 4)
            for e in range(experts):
                t.update({f"{p}block_sparse_moe.experts.{e}.{m}.weight": rnd(4, 4) for m in ("w1", "w3", "w2")})
        if l == 3:
            t.update({f"{p}self_attn.{n}": rnd(5, 3) for n in golden.WEIGHTS})
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_file({f"model.{k}": v.contiguous() for k, v in t.items()}, str(ckpt / "shard.safetensors"))
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"model.{k}": "shard.safetensors" for k in t}})
    )
    linear = {
        "num_heads": 2,
        "head_dim": 8,
        "short_conv_kernel_size": 4,
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    }
    config = {"hidden_size": hidden, "rms_norm_eps": eps, "num_experts": experts, "linear_attn_config": linear}
    (ckpt / "config.json").write_text(json.dumps({"text_config": config}))
    model = fp64_chain.ChainModel.from_checkpoint(ckpt, threads=2, log=lambda m: None)

    same = lambda got, name: torch.equal(got, t[name].double())  # noqa: E731
    assert torch.equal(model.embedding, t["embed_tokens.weight"]) and model.eps == eps and model.kda_config == kda_cfg
    for l in range(4):
        L, p = model.layers[l], f"layers.{l}."
        assert same(L["input_layernorm"], f"{p}input_layernorm.weight")
        assert same(L["post_attention_layernorm"], f"{p}post_attention_layernorm.weight")
        assert same(L["attn_res"][0], f"{p}self_attention_res_norm.weight") and same(
            L["attn_res"][1], f"{p}self_attention_res_proj.weight"
        )
        assert same(L["mlp_res"][0], f"{p}mlp_res_norm.weight") and same(L["mlp_res"][1], f"{p}mlp_res_proj.weight")
        if l < 3:
            kda = {n[len(p + "self_attn.") :]: v for n, v in t.items() if n.startswith(p + "self_attn.")}
            want = normalize_kda_state_dict(kda, kda_cfg)
            assert want.keys() == L["kda"].keys() and all(torch.equal(L["kda"][n], want[n].double()) for n in want)
        if l == 0:
            assert all(
                same(L["mlp"][f"{n}.weight"], f"{p}mlp.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj")
            )
        if l in (1, 2):
            moe = L["moe"]
            assert all(same(moe.w[n], f"{p}block_sparse_moe.{n}") for n in fp64_chain.MoE_NON_ROUTED)
            assert len(moe.experts) == experts
            for e in range(experts):
                for got, m in zip(moe.experts[e], ("w1", "w3", "w2")):
                    assert torch.equal(got, t[f"{p}block_sparse_moe.experts.{e}.{m}.weight"]), (l, e, m)
        if l == 3:
            assert all(same(L["dsa"][n], f"{p}self_attn.{n}") for n in golden.WEIGHTS)
            assert same(L["gate"], f"{p}block_sparse_moe.gate.weight")
            assert same(L["gate_bias"], f"{p}block_sparse_moe.gate.e_score_correction_bias")
    assert set(model.digests) == {
        "embed_tokens",
        "layers.0",
        "layers.1",
        "layers.2",
        "layers.3",
        "layers.1.experts",
        "layers.2.experts",
    }


# --- teacher feeds from the FP64 chain (moe_teacher_feed.py / dsa_teacher_feed.py with --chain) ---------------------


def _chain_checkpoint(model, path: Path) -> Path:
    """The layer tensors the teacher-feed input and score steps read, from a tiny ChainModel, as a checkpoint."""
    t = {}
    for l in (1, 2, 3):
        L, p = model.layers[l], f"model.layers.{l}."
        t[p + "input_layernorm.weight"] = L["input_layernorm"]
        t[p + "post_attention_layernorm.weight"] = L["post_attention_layernorm"]
        t[p + "mlp_res_norm.weight"], t[p + "mlp_res_proj.weight"] = L["mlp_res"]
        t[p + "self_attention_res_norm.weight"], t[p + "self_attention_res_proj.weight"] = L["attn_res"]
        if l in (1, 2):
            t[p + "block_sparse_moe.gate.weight"] = L["moe"].w["gate.weight"]
            t[p + "block_sparse_moe.gate.e_score_correction_bias"] = L["moe"].w["gate.e_score_correction_bias"]
        else:
            t.update({f"{p}self_attn.{n}": v for n, v in L["dsa"].items()})
    path.mkdir()
    save_file({k: v.contiguous() for k, v in t.items()}, str(path / "shard.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(t, "shard.safetensors")})
    )
    return path


def _fed_chain(tmp_path: Path, rows: int = 10):
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_chain

    model, _ = _tiny_chain()
    chain = tmp_path / "chain"
    ids = torch.randint(0, 50, (rows,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        fp64_chain.run_chain(
            run_dir=chain, model=model, token_ids=ids, segment=4, sampled=[2, 6, 9], identity={"checkpoint": "c"}
        )
    return model, chain, _chain_checkpoint(model, tmp_path / "ckpt")


def test_moe_teacher_feed_from_the_chain_scores_against_it(tmp_path: Path):
    """The MoE input built from the chain is its gate input rounded once to bf16 and records the chain; a device run
    that reproduces the chain (its MoE output in bf16, its routing) scores against the chain, not the trace: recall
    1, NRMSE at bf16 level; with one planted swap of a row's last expert, recall 1 - 1/2 / 10 per that window and the
    miss at FP64 rank 1. The GPU trace's routing (here the chain's with one row changed) is the third reference."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_chain, moe_teacher_feed

    model, chain, ckpt = _fed_chain(tmp_path)
    layer, src = 1, tmp_path / "moe_in"
    m = moe_teacher_feed.build_moe_input_from_chain(
        chain=chain, checkpoint=ckpt, layer=layer, count=10, out=src, segment=4
    )
    assert m["chain"] == str(chain) and m["checkpoint"] == "c" and all(s["router_recall"] == 1.0 for s in m["segments"])
    p = fp64_chain.load_products(chain, ["moe_input_1", "moe_output_1", "routing_ids_1", "routing_weights_1"])
    assert torch.equal(load_input(src, 0, 10), p["moe_input_1"].to(torch.bfloat16))
    gpu = p["routing_ids_1"].clone()
    gpu[5, 0] = next(e for e in range(6) if e not in gpu[5].tolist())
    _write_trace(tmp_path / "trace", {"expert_ids_layer_1": gpu}, chunk_rows=4)
    out = tmp_path / "dev"
    run = {"layer": layer, "count": 10, "chunk": 4, "input": str(src)}
    start_run(out, run)
    dev_ids = p["routing_ids_1"].clone()
    dev_ids[7, 1] = next(e for e in range(6) if e not in dev_ids[7].tolist())
    for c, lo in enumerate(range(0, 10, 4)):
        hi = min(lo + 4, 10)
        torch.save(
            {
                "start": lo,
                "end": hi,
                "moe_out": p["moe_output_1"][lo:hi].to(torch.bfloat16),
                "expert_ids": dev_ids[lo:hi].to(torch.int16),
                "expert_weights": p["routing_weights_1"][lo:hi].to(torch.bfloat16),
            },
            chunk_path(out, c),
        )
    finish_run(out, run, 1.0)
    w = model.layers[layer]["moe"].w
    result = score_moe(
        trace=Trace(tmp_path / "trace"),
        out=out,
        window=4,
        gate=w["gate.weight"],
        bias=w["gate.e_score_correction_bias"],
    )
    assert result["reference"] == str(chain)
    o = result["overall"]
    assert o["recall_vs_reference_mean"] == pytest.approx(1 - 0.5 / 10) and o["miss_rank_histogram"][1] == 1
    assert o["recall_gpu_vs_reference_mean"] == pytest.approx(1 - 0.5 / 10)
    assert o["moe_nrmse_median"] < 1e-2
    assert result["windows"][1]["recall_vs_reference_mean"] == pytest.approx(1 - 0.5 / 4)


def test_dsa_teacher_feed_from_the_chain_scores_against_it(tmp_path: Path, expect_error):
    """The DSA input built from the chain is input_layernorm of its layer-3 read in FP64, rounded once to bf16, and
    passes the key check against the chain's FP64 keys; a device run that reproduces the chain scores against it:
    attention only on the chain's sampled rows (bf16 level), recall 1 and the bf16 keys' PCC ~1; the GPU trace's
    top-k (one row missing its FP64 best key) is the third reference."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import dsa_teacher_feed, fp64_chain

    model, chain, ckpt = _fed_chain(tmp_path)
    src = tmp_path / "dsa_in"
    m = dsa_teacher_feed.build_dsa_input_from_chain(chain=chain, checkpoint=ckpt, count=10, out=src, segment=4)
    assert m["layer"] == 3 and all(s["kv_latent_pcc"] > 0.9999 and s["index_key_pcc"] > 0.9999 for s in m["segments"])
    p = fp64_chain.load_products(chain, ["dsa_input_3", "latent", "kpe", "index_key", "topk_indices_3"])
    read = p["dsa_input_3"].double()
    norm = model.layers[3]["input_layernorm"]
    want = (read * torch.rsqrt(read.pow(2).mean(-1, keepdim=True) + model.eps) * norm).to(torch.bfloat16)
    assert torch.equal(load_input(src, 0, 10), want)
    sampled = fp64_chain.load_sampled(chain)
    assert sampled["sampled_rows"].tolist() == [2, 6, 9]
    gpu = p["topk_indices_3"].clone()
    gpu[8, 0] = 9
    _write_trace(tmp_path / "trace", {"dsa_topk_indices_layer_3": gpu}, chunk_rows=4)
    attn = torch.zeros(10, sampled["attn_output_3"].shape[1], dtype=torch.bfloat16)
    attn[sampled["sampled_rows"]] = sampled["attn_output_3"].to(torch.bfloat16)
    out = tmp_path / "dev"
    run = {"layer": 3, "count": 10, "chunk": 4, "input": str(src)}
    start_run(out, run)
    for c, lo in enumerate(range(0, 10, 4)):
        hi = min(lo + 4, 10)
        torch.save(
            {"start": lo, "end": hi, "attn_out": attn[lo:hi], "topk": p["topk_indices_3"][lo:hi]}, chunk_path(out, c)
        )
    kvpe = torch.cat([p["latent"], p["kpe"]], -1).to(torch.bfloat16)
    torch.save({"kvpe": kvpe, "index": p["index_key"].to(torch.bfloat16)}, out / "caches.pt")
    finish_run(out, run, 1.0)
    result = dsa_teacher_feed.score(trace=Trace(tmp_path / "trace"), out=out, window=4)
    o = result["overall"]
    assert result["reference"] == str(chain) and o["attn_rows"] == 3 and o["attn_nrmse_median"] < 1e-2
    assert o["recall_mean"] == 1.0 and o["misses"] == 0
    assert o["recall_gpu_vs_reference_mean"] == pytest.approx((3 + 5 / 6) / 4)  # rows 6-9 select fully (top-k 6)
    assert o["kv_latent_pcc"] > 0.9999 and o["index_key_pcc"] > 0.9999
    assert [w["attn_rows"] for w in result["windows"]] == [1, 1, 1]
    with expect_error(ValueError, "sampled-row products"):
        fp64_chain.load_products(chain, ["attn_output_3"])


def test_quantile_by_hand_and_beyond_torch_quantile_size():
    """[1, 2, 3, 4]: median 2.5, p10 1.3 (linear interpolation); empty is NaN; 2^24 + 1 values (torch.quantile's
    limit is 2^24) give their exact median."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import quantile

    t = torch.tensor([4.0, 1.0, 3.0, 2.0])
    assert quantile(t, 0.5) == pytest.approx(2.5) and quantile(t, 0.1) == pytest.approx(1.3)
    assert quantile(t, 0.0) == 1.0 and quantile(t, 1.0) == 4.0
    assert quantile(torch.empty(0), 0.5) != quantile(torch.empty(0), 0.5)  # NaN
    big = torch.arange(2**24 + 1).flip(0)
    assert quantile(big, 0.5) == 2**23


# --- KDA-precision arms of the chain (kda_vllm_emulation.py, fp64_chain --kda-precision) --------------------------


def test_kda_flashkda_emulation_without_rounding_is_the_reference():
    """With rounding off and FP64 arithmetic, the chunked (WY, chunk 16) emulation is the token-ordered FP64 KDA
    reference to <= 1e-12 (output, recurrent and convolution state), across two calls whose split (21 rows) is not a
    chunk multiple and a last chunk shorter than 16. With bf16 rounding on, it moves the output at bf16 level and
    carries an fp32 recurrent state."""
    from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
    from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
    from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import random_weights
    from models.demos.deepseek_v3_d_p.tests.kimi_k3.kda_vllm_emulation import kda_flashkda_emulation

    cfg = KDAConfig(64, 2, 32, 32, 4, 1e-5, use_full_rank_gate=True, gate_lower_bound=-5.0)
    w = {n: v.double() for n, v in random_weights(cfg).items()}
    x = torch.randn(1, 53, 64, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    want, want_state = kda_forward_reference(x, w, cfg, dtype=torch.float64)
    exact = dict(low=lambda t: t, dtype=torch.float64)
    a, state = kda_flashkda_emulation(x[:, :21], w, cfg, **exact)
    b, state = kda_flashkda_emulation(x[:, 21:], w, cfg, state, **exact)
    assert _rel(torch.cat([a, b], 1), want) <= 1e-12
    for name in ("recurrent", "q_convolution", "k_convolution", "v_convolution"):
        assert _rel(getattr(state, name), getattr(want_state, name)) <= 1e-12, name
    got, state = kda_flashkda_emulation(x, w, cfg)
    assert got.dtype == torch.float32 and state.recurrent.dtype == torch.float32
    assert 1e-4 < _rel(got.double(), want) < 3e-2


def test_chain_kda_precision_arm_scores_as_a_candidate_against_the_fp64_chain(tmp_path: Path, expect_error):
    """A chain whose KDA layers run the vLLM emulation records its precision in the manifest (the FP64 golden's
    model identity is unchanged), and its products score against the FP64 chain through ``ChainStreams``: the FP64
    chain scored as its own candidate is exact (recall 1, NRMSE 0); the emulated arm differs at bf16 level; only the
    rows the candidate holds are scored; a candidate on another segment grid is refused."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import chain_vs_trace, fp64_chain

    rows = 10
    ids = torch.randint(0, 50, (rows,), generator=torch.Generator().manual_seed(4))
    identity = {"checkpoint": "c", "token_source": f"vllm trace t metadata token_ids[:{rows}]"}
    runs = {}
    for precision, segment, count in (("fp64", 4, rows), ("vllm", 4, 8), ("fp32", 2, rows)):
        model, _ = _tiny_chain()
        model.kda_precision = precision
        runs[precision] = tmp_path / precision
        with torch.no_grad():
            fp64_chain.run_chain(
                run_dir=runs[precision],
                model=model,
                token_ids=ids[:count],
                segment=segment,
                sampled=[3],
                identity={**identity, "token_source": f"vllm trace t metadata token_ids[:{count}]"},
            )
    manifests = {p: json.loads((r / MANIFEST).read_text()) for p, r in runs.items()}
    assert "kda_precision" not in manifests["fp64"]["model"] and manifests["vllm"]["model"]["kda_precision"] == "vllm"
    chain = runs["fp64"]
    o = chain_vs_trace.summarize(
        chain_vs_trace.score_landed(chain, chain_vs_trace.ChainStreams(chain), tmp_path / "score-self"), window=4
    )["overall"]
    assert o["routing_1_recall_mean"] == 1.0 and o["routing_2_set_equal"] == 1.0 and o["topk_recall_mean"] == 1.0
    assert o["latent_nrmse"] == 0.0 and o["index_key_nrmse"] == 0.0
    results = chain_vs_trace.score_landed(chain, chain_vs_trace.ChainStreams(runs["vllm"]), tmp_path / "score-vllm")
    assert [r["end"] for r in results] == [4, 8]
    o = chain_vs_trace.summarize(results, window=4)["overall"]
    assert 1e-4 < o["latent_nrmse"] < 0.1 and o["routing_3_sampled_rows"] == 0
    with expect_error(ValueError, "segment"):
        chain_vs_trace.score_landed(chain, chain_vs_trace.ChainStreams(runs["fp32"]), tmp_path / "score-fp32")


# --- layer-major deep chain (fp64_deep_chain.py) ---------------------------------------------------------------------

DEEP_KINDS = {l: ("dsa" if l in (3, 7) else "kda", "dense" if l == 0 else "moe") for l in range(8)}


def _tiny_deep(hidden: int = 32):
    """Tiny weights for layers 0-7 (KDA 0-2 and 4-6, DSA 3 and 7, dense FFN 0, MoE 1-7): layers 0-2 and layer 3's
    attention are ``_tiny_chain``'s, so layers 0-3 can be checked against ``fp64_chain``; plus the reference MoE
    modules the walk uses."""
    from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import random_weights
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.modeling_kimi_moe import KimiSparseMoeBlock

    model, refs = _tiny_chain(hidden)
    gen = torch.Generator().manual_seed(11)
    vec = lambda: 1 + 0.1 * torch.randn(hidden, generator=gen, dtype=torch.float64)  # noqa: E731
    query = lambda: (vec(), torch.randn(1, hidden, generator=gen, dtype=torch.float64))  # noqa: E731
    layers = dict(model.layers)
    cfg = _tiny_moe_cfg(hidden)
    for l in range(3, 8):
        if l > 3:
            layers[l] = {
                "input_layernorm": vec(),
                "attn_res": query(),
                "mlp_res": query(),
                "post_attention_layernorm": vec(),
            }
            if DEEP_KINDS[l][0] == "kda":
                w = random_weights(model.kda_config)
                gw = torch.Generator().manual_seed(100 + l)
                layers[l]["kda"] = {n: (v + 0.02 * torch.randn(v.shape, generator=gw)).double() for n, v in w.items()}
            else:
                layers[l]["dsa"] = _weights(SMALL, hidden, 24, seed=7)
        torch.manual_seed(20 + l)
        block = KimiSparseMoeBlock(cfg).double().eval()
        with torch.no_grad():
            block.gate.e_score_correction_bias.normal_(std=0.05)
            block.routed_expert_norm.weight.normal_(1.0, 0.1)
        refs[l] = block
        layers[l] = {**layers[l], "moe": _moe64_of(block, 2)}
    return model, layers, refs


def _deep_load(layers):
    return lambda l, kind: (layers[l], {f"layers.{l}": kind})


def _run_deep(model, layers, ids, out, load=None):
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_deep_chain

    with torch.no_grad():
        return fp64_deep_chain.run_deep_chain(
            run_dir=out,
            load=load or _deep_load(layers),
            embedding=model.embedding,
            kinds=DEEP_KINDS,
            kda_config=model.kda_config,
            dims=SMALL,
            eps=model.eps,
            token_ids=ids,
            segment=4,
            identity={"t": 1},
        )


def _deep_reference_walk(model, layers, refs, token_ids, monkeypatch) -> dict:
    """Layers 0-7 over the whole sequence at once: the vendored AttnRes walk, one-shot FP64 KDA, the reference MoE
    modules, and for DSA layers the golden's row-wise keys, indexer top-k and attention of every row."""
    from models.demos.deepseek_v3_d_p.reference.kda.layer import kda_forward_reference
    from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import hf_layer

    def norm(weight):
        n = KimiRMSNorm(weight.numel(), eps=model.eps).double()
        n.weight = torch.nn.Parameter(weight)
        return n

    got = {}
    stream = HfStream(model.embedding[token_ids].double(), model.block_size, model.eps)
    for l in range(8):
        L = layers[l]

        def attn_fn(h, L=L, l=l):
            x = norm(L["input_layernorm"])(h)
            if "kda" in L:
                return kda_forward_reference(x[None], L["kda"], model.kda_config, dtype=torch.float64)[0][0]
            got[f"dsa_input_{l}"] = h
            keys = golden.layer_keys(x, L["dsa"], SMALL)
            q = golden.layer_queries(x, L["dsa"], SMALL)
            got[f"keys_{l}"] = keys
            top = [
                golden.index_scores(q["index_q"][r], q["index_w"][r], keys["index_key"][: r + 1]).topk(
                    min(SMALL.topk, r + 1)
                )
                for r in range(x.shape[0])
            ]
            got[f"topk_{l}"] = top
            out = torch.stack(
                [
                    golden.attend(q, r, keys["latent"][t.indices], keys["kpe"][t.indices], L["dsa"], SMALL)
                    for r, t in enumerate(top)
                ]
            )
            got[f"attn_output_{l}"] = out
            return out

        def mlp_fn(h, L=L, l=l):
            x = norm(L["post_attention_layernorm"])(h)
            out = refs[l](x.unsqueeze(0))[0] if l else refs["mlp"](x)
            got[f"moe_input_{l}"], got[f"moe_output_{l}"] = x, out
            return out

        with torch.no_grad(), _fp64_upcasts(monkeypatch):
            hf_layer(stream, l, L["attn_res"], L["mlp_res"], attn_fn, mlp_fn)
        if l:
            got[f"routing_ids_{l}"] = refs[l].gate(got[f"moe_input_{l}"].unsqueeze(0))[0]
    return got


def test_fp64_deep_chain_matches_a_one_shot_walk_and_fp64_chain_and_resumes(
    tmp_path: Path, fp64_references, expect_error
):
    """Layer-major passes over 10 rows (segments 4 / 4 / 2) of layers 0-7 equal a one-shot walk of independent forms
    (vendored AttnRes, whole-sequence KDA, reference MoE modules, row-wise DSA golden with every row's attention) to
    <= 1e-12 on FP64 products and fp32 rounding on stored ones, with identical routing and top-k, at both DSA layers;
    layers 0-3 reproduce fp64_chain's stored products bit for bit; a run interrupted in its last pass resumes from
    the live checkpoint to byte-identical files; another token sequence is refused."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_chain, fp64_deep_chain

    model, layers, refs = _tiny_deep()
    rows = 10
    ids = torch.randint(0, 50, (rows,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = _deep_reference_walk(model, layers, refs, ids, fp64_references)
    out = tmp_path / "deep"
    manifest = _run_deep(model, layers, ids, out)
    assert [(p["layer"], p["site"]) for p in manifest["segments"]] == fp64_deep_chain.passes(8)
    for l in range(1, 8):
        p = fp64_deep_chain.load_deep(out, l, "ffn", [f"moe_input_{l}", f"moe_output_{l}", f"routing_ids_{l}"])
        assert _rel(p[f"moe_input_{l}"], want[f"moe_input_{l}"]) <= 1e-7, l
        assert _rel(p[f"moe_output_{l}"], want[f"moe_output_{l}"]) <= 1e-7, l
        assert torch.equal(p[f"routing_ids_{l}"].long().sort(1).values, want[f"routing_ids_{l}"].sort(1).values), l
    for l in (3, 7):
        names = ["latent", "kpe", "index_key", f"dsa_input_{l}", f"topk_indices_{l}", f"attn_output_{l}"]
        p = fp64_deep_chain.load_deep(out, l, "attn", names)
        for k in ("latent", "kpe", "index_key"):
            assert p[k].dtype == torch.float64 and _rel(p[k], want[f"keys_{l}"][k]) <= 1e-12, (l, k)
        assert (
            _rel(p[f"dsa_input_{l}"], want[f"dsa_input_{l}"]) <= 1e-7
            and _rel(p[f"attn_output_{l}"], want[f"attn_output_{l}"]) <= 1e-7
        )
        for r, t in enumerate(want[f"topk_{l}"]):
            assert p[f"topk_indices_{l}"][r, : t.indices.numel()].tolist() == t.indices.tolist(), (l, r)
            assert (p[f"topk_indices_{l}"][r, t.indices.numel() :] == -1).all()
    # layers 0-3 against fp64_chain (segment-major, layer 3's router only)
    chain = tmp_path / "chain"
    with torch.no_grad():
        fp64_chain.run_chain(run_dir=chain, model=model, token_ids=ids, segment=4, sampled=[3], identity={})
    names = ["moe_input_1", "moe_output_1", "routing_ids_1", "moe_input_2", "routing_ids_2"]
    c = fp64_chain.load_products(chain, names + ["dsa_input_3", "latent", "kpe", "index_key", "topk_indices_3"])
    d = {**fp64_deep_chain.load_deep(out, 1, "ffn", names[:3]), **fp64_deep_chain.load_deep(out, 2, "ffn", names[3:])}
    d.update(fp64_deep_chain.load_deep(out, 3, "attn", ["dsa_input_3", "latent", "kpe", "index_key", "topk_indices_3"]))
    for n, v in c.items():
        assert torch.equal(d[n], v), n
    # resume: fail in the last pass, rerun, compare every file with the uninterrupted run
    again = tmp_path / "again"

    def failing(l, kind):
        if l == 7 and kind == "moe":
            raise RuntimeError("interrupted")
        return layers[l], {f"layers.{l}": kind}

    with expect_error(RuntimeError, "interrupted"):
        _run_deep(model, layers, ids, again, load=failing)
    assert len(json.loads((again / MANIFEST).read_text())["segments"]) == 15
    resumed = _run_deep(model, layers, ids, again)
    assert len(resumed["sessions"]) == 2
    for f in sorted(out.rglob("segment_*.pt")) + [out / "live.pt"]:
        a, b = torch.load(f, weights_only=True), torch.load(again / f.relative_to(out), weights_only=True)
        assert a.keys() == b.keys(), f
        for k in a:
            assert (
                (torch.equal(a[k], b[k]) or (a[k].isnan() == b[k].isnan()).all())
                if torch.is_tensor(a[k])
                else a[k] == b[k]
            ), (f, k)
    with expect_error(ValueError, "different run"):
        _run_deep(model, layers, (ids + 1) % 50, again)


def test_fp64_deep_chain_scores_hand_made_differences(tmp_path: Path):
    """A trace made from the deep chain's products, with layer 5's row 2 losing its FP64 best expert and layer 7's index
    key doubled, scores exactly: routing_5 recall 1/2 on that row (its weight dropped), index key PCC 1 / NRMSE 1 at
    layer 7, everything else exact (recall 1, NRMSE 0)."""
    from models.demos.deepseek_v3_d_p.tests.kimi_k3 import fp64_deep_chain

    model, layers, _ = _tiny_deep()
    rows, out = 10, tmp_path / "deep"
    ids = torch.randint(0, 50, (rows,), generator=torch.Generator().manual_seed(4))
    _run_deep(model, layers, ids, out)
    streams = {}
    for l in range(1, 8):
        streams[f"expert_ids_layer_{l}"] = fp64_deep_chain.load_deep(out, l, "ffn", [f"routing_ids_{l}"])[
            f"routing_ids_{l}"
        ]
    for l in (3, 7):
        p = fp64_deep_chain.load_deep(out, l, "attn", ["latent", "kpe", "index_key", f"topk_indices_{l}"])
        streams[f"kv_post_transform_layer_{l}"] = torch.cat([p["latent"], p["kpe"]], -1)
        streams[f"indexer_k_layer_{l}"] = p["index_key"] * (2 if l == 7 else 1)
        streams[f"dsa_topk_indices_layer_{l}"] = p[f"topk_indices_{l}"]
    r5 = streams["expert_ids_layer_5"].clone()
    weights5 = fp64_deep_chain.load_deep(out, 5, "ffn", ["routing_weights_5"])["routing_weights_5"]
    r5[2, 0] = next(e for e in range(6) if e not in r5[2].tolist())
    streams["expert_ids_layer_5"] = r5
    _write_trace(tmp_path / "trace", streams, chunk_rows=4)
    s = fp64_deep_chain.score(out, Trace(tmp_path / "trace"), tmp_path / "score", window=4)
    o = s["overall"]
    assert o["routing_5_recall_mean"] == pytest.approx(1 - 0.5 / rows) and o["routing_5_set_equal"] == pytest.approx(
        0.9
    )
    assert o["routing_5_dropped_mean"] == pytest.approx(float(weights5[2, 0]) / rows, rel=1e-6)
    assert all(o[f"routing_{l}_recall_mean"] == 1.0 for l in (1, 2, 3, 4, 6, 7))
    assert o["index_key_7_pcc"] == pytest.approx(1.0, abs=1e-12) and o["index_key_7_nrmse"] == pytest.approx(
        1.0, rel=1e-12
    )
    assert o["index_key_3_nrmse"] == 0.0 and o["latent_7_nrmse"] == 0.0
    assert o["topk_3_recall_mean"] == 1.0 and o["topk_7_recall_mean"] == 1.0
    assert [w["start"] for w in s["windows"]] == [0, 4, 8]
