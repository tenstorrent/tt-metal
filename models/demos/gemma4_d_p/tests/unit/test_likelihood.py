# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import math

import pytest
import torch
from safetensors.torch import save_file

from models.demos.gemma4_d_p.tests.scripts import likelihood
from models.demos.gemma4_d_p.tests.scripts.likelihood import (
    HiddenSampler,
    Samples,
    check_limits,
    load_samples,
    sample_positions,
    save_samples,
    score_positions,
    summarize,
    topk_kl,
)


class LinearHead:
    def __init__(self, hidden, vocab):
        self.weight = torch.randn(vocab, hidden, generator=torch.Generator().manual_seed(0))

    def logprobs(self, hidden):
        return torch.log_softmax(hidden.float() @ self.weight.T, dim=-1)


def test_sample_positions_keep_stride_and_last_chunk():
    positions = sample_positions(context_len=128, chunk_size=32, stride=16)
    assert positions == [0, 16, 32, 48, 64, 80] + list(range(96, 128))
    assert sample_positions(context_len=32, chunk_size=32, stride=16) == list(range(32))


def test_topk_kl_is_zero_for_equal_and_matches_full_kl_when_k_covers_vocab():
    reference = torch.log_softmax(torch.randn(4, 8), dim=-1)
    candidate = torch.log_softmax(torch.randn(4, 8), dim=-1)
    assert torch.equal(topk_kl(reference, reference, k=3), torch.zeros(4, dtype=torch.float64))
    full = (reference.exp() * (reference - candidate)).sum(-1).double()
    torch.testing.assert_close(topk_kl(reference, candidate, k=8), full, rtol=1e-5, atol=1e-6)
    # Lumping the tail into one bucket can only lose information.
    assert (topk_kl(reference, candidate, k=3) <= full + 1e-6).all()


def test_scores_bins_and_last_position(monkeypatch, expect_error):
    monkeypatch.setattr(likelihood, "DEPTH_BINS", ((0, 8), (8, 64)))
    head = LinearHead(hidden=16, vocab=32)
    positions = torch.tensor(sample_positions(context_len=32, chunk_size=8, stride=4))
    gold = torch.randint(0, 32, positions.shape)
    gold[-1] = -1
    hidden = torch.randn(len(positions), 16).bfloat16()
    reference = Samples(positions, hidden, gold)

    same = summarize(score_positions(head, reference, Samples(positions, hidden.clone(), gold), batch=3))
    assert [(entry["start"], entry["end"], entry["positions"]) for entry in same["bins"]] == [(0, 8, 2), (8, 32, 12)]
    for entry in same["bins"] + [same["all"]]:
        assert entry["delta_nll"] == 0 and entry["top1_agreement"] == 1 and entry["topk_kl"] == 0
    assert same["last_position"]["position"] == 31 and same["last_position"]["delta_nll"] is None

    noisy = Samples(positions, (hidden.float() + 0.5 * torch.randn_like(hidden.float())).bfloat16(), gold)
    report = summarize(score_positions(head, reference, noisy))
    assert report["all"]["mean_abs_delta_nll"] > 0 and report["all"]["topk_kl"] > 0
    assert not math.isnan(report["last_position"]["topk_kl"])

    with expect_error(ValueError, "same positions"):
        score_positions(head, reference, Samples(positions + 1, hidden, gold))


def test_sampler_keeps_sampled_rows_in_position_order(tmp_path, monkeypatch):
    chunk = 8
    monkeypatch.setattr(likelihood, "read_chunk_hidden", lambda output: output)
    sampler = HiddenSampler(context_len=24, chunk_size=chunk, token_ids=list(range(100, 124)))
    rows = torch.arange(24, dtype=torch.float32).unsqueeze(-1).expand(24, 4).bfloat16()
    for start in (16, 0, 8):
        sampler.add_chunk(rows[start : start + chunk], start, start + chunk)
    sampler.save(tmp_path / likelihood.HIDDEN_SAMPLES)

    samples = load_samples(tmp_path)
    assert samples.positions.tolist() == [0] + list(range(16, 24))
    assert samples.hidden[:, 0].tolist() == [float(p) for p in samples.positions]
    assert samples.next_tokens.tolist() == [101 + p for p in samples.positions[:-1]] + [-1]


def test_gpu_trace_samples_follow_the_prepared_copy(tmp_path, expect_error):
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    (source / "decoder_io").mkdir(parents=True)
    prepared.mkdir()
    hidden = torch.randn(12, 4).bfloat16()
    streams = []
    for start in (0, 6):
        path = f"decoder_io/rows_{start}.safetensors"
        save_file({"decoder_output_layer_1": hidden[start : start + 6]}, str(source / path))
        streams.append(dict(row_start=start, row_end=start + 6, path=path))
    stream = dict(row_count=12, shape_tail=[4], chunks=streams)
    (source / "index.json").write_text(json.dumps({"tensor_streams": {"decoder_output_layer_1": stream}}))
    metadata = dict(n_layers=2, token_ids=list(range(12)))
    (source / "metadata.json").write_text(json.dumps(metadata))
    (prepared / "metadata.json").write_text(json.dumps(dict(metadata, source_trace_dir=str(source))))

    samples = load_samples(prepared, positions=[1, 5, 6, 11])
    assert torch.equal(samples.hidden, hidden[[1, 5, 6, 11]])
    assert samples.next_tokens.tolist() == [2, 6, 7, -1]
    with expect_error(ValueError, "no rows for 1 requested positions, from 12"):
        load_samples(prepared, positions=[12])
    # A gap in the stream must fail rather than score uninitialized rows.
    (source / "index.json").write_text(
        json.dumps({"tensor_streams": {"decoder_output_layer_1": dict(stream, chunks=streams[1:])}})
    )
    with expect_error(ValueError, "no rows for 2 requested positions, from 1"):
        load_samples(prepared, positions=[1, 5, 6, 11])

    save_samples(tmp_path / "run.safetensors", samples.positions, samples.hidden, samples.next_tokens)
    assert torch.equal(load_samples(tmp_path / "run.safetensors").hidden, samples.hidden)


def test_compare_scores_positions_both_runs_hold(tmp_path, monkeypatch):
    head = LinearHead(hidden=8, vocab=32)
    for name, chunk in (("a", 8), ("b", 16)):
        positions = torch.tensor(sample_positions(context_len=64, chunk_size=chunk, stride=16))
        hidden = torch.ones(len(positions), 8).bfloat16()
        save_samples(tmp_path / f"{name}.safetensors", positions, hidden, (positions + 1) % 16)
    report = likelihood.compare(tmp_path / "a.safetensors", tmp_path / "b.safetensors", head)
    # Stride positions 0, 16, 32, 48 plus a's last chunk 56..63, all of which b also holds.
    assert report["all"]["positions"] == 4 + 8 and report["last_position"]["position"] == 63
    assert report["all"]["delta_nll"] == 0


def test_limits_fail_a_bin_without_gold_tokens(monkeypatch):
    monkeypatch.setattr(likelihood, "DEPTH_BINS", ((0, 8), (8, 64)))
    head = LinearHead(hidden=16, vocab=32)
    positions = torch.tensor(sample_positions(context_len=32, chunk_size=8, stride=4))
    gold = torch.where(positions < 8, -1, positions % 32)
    hidden = torch.randn(len(positions), 16).bfloat16()
    report = summarize(score_positions(head, Samples(positions, hidden, gold), Samples(positions, hidden, gold)))
    assert report["bins"][0]["delta_nll"] is None

    # Rows are top-1, ΔNLL and KL for the first bin, then the same for the second.
    rows = [
        (achieved, passed) for _, _, achieved, passed in check_limits(report, {0: (0.9, 0.1, 0.1), 8: (0.9, 0.1, 0.1)})
    ]
    assert rows[:3] == [(1.0, True), (None, False), (0.0, True)]
    assert rows[3:] == [(1.0, True), (0.0, True), (0.0, True)]


def _fake_trace(tmp_path, hidden, token_ids, shard_rows):
    """A prepared GPU trace directory whose source capture holds ``hidden`` as its final-layer stream."""
    source, prepared = tmp_path / "source", tmp_path / "prepared"
    (source / "decoder_io").mkdir(parents=True)
    prepared.mkdir()
    chunks = []
    for start in range(0, len(hidden), shard_rows):
        path = f"decoder_io/rows_{start}.safetensors"
        save_file({"decoder_output_layer_1": hidden[start : start + shard_rows].contiguous()}, str(source / path))
        chunks.append(dict(row_start=start, row_end=min(start + shard_rows, len(hidden)), path=path))
    stream = dict(row_count=len(hidden), shape_tail=[hidden.shape[-1]], chunks=chunks)
    (source / "index.json").write_text(json.dumps({"tensor_streams": {"decoder_output_layer_1": stream}}))
    metadata = dict(n_layers=2, token_ids=token_ids)
    (source / "metadata.json").write_text(json.dumps(metadata))
    (prepared / "metadata.json").write_text(json.dumps(dict(metadata, source_trace_dir=str(source))))
    return prepared


def _flatten(report, prefix=""):
    """A report's numbers keyed by their path, for an approximate comparison."""
    if isinstance(report, dict):
        return {
            key: value for name, item in report.items() for key, value in _flatten(item, f"{prefix}{name}.").items()
        }
    if isinstance(report, list):
        return {key: value for i, item in enumerate(report) for key, value in _flatten(item, f"{prefix}{i}.").items()}
    return {prefix: report}


class IdentifiedHead(LinearHead):
    identity = {"model": "fake", "checkpoint": "snapshot-a", "format": str(likelihood.REFERENCE_FORMAT)}


def test_stored_reference_scores_match_on_the_fly_scoring(tmp_path):
    from models.demos.gemma4_d_p.tests.scripts.prepare_likelihood_reference import prepare_likelihood_reference

    head = IdentifiedHead(hidden=16, vocab=64)
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(96, 16, generator=generator).bfloat16()
    token_ids = torch.randint(0, 64, (96,), generator=generator).tolist()
    trace = _fake_trace(tmp_path, hidden, token_ids, shard_rows=32)

    # A candidate at a 64-token context: every 16th position plus the last chunk, as the test samples them.
    positions = torch.tensor(sample_positions(context_len=64, chunk_size=16, stride=16))
    candidate = (hidden[positions].float() + 0.3 * torch.randn(len(positions), 16, generator=generator)).bfloat16()
    save_samples(
        tmp_path / "run.safetensors", positions, candidate, likelihood.next_tokens(token_ids, positions.tolist())
    )

    on_the_fly = likelihood.compare(trace, tmp_path / "run.safetensors", head)
    assert on_the_fly.pop("reference") == "computed"
    stored = prepare_likelihood_reference(trace, head=head)
    assert stored == trace / likelihood.REFERENCE_SCORES
    scores = likelihood.load_reference_scores(trace, positions, head.identity)
    assert scores is not None and torch.equal(scores.positions, positions)
    report = likelihood.compare(trace, tmp_path / "run.safetensors", head)
    assert report.pop("reference") == "stored"
    # The reference logits come from other row batches than on the fly, so fp32 rounding can differ in the last bits.
    assert _flatten(report) == pytest.approx(_flatten(on_the_fly), rel=1e-6, abs=1e-9)
    assert on_the_fly["all"]["mean_abs_delta_nll"] > 0


def _stored_trace(tmp_path, head, rows=32):
    from models.demos.gemma4_d_p.tests.scripts.prepare_likelihood_reference import prepare_likelihood_reference

    hidden = torch.randn(rows, 8, generator=torch.Generator().manual_seed(2)).bfloat16()
    trace = _fake_trace(tmp_path, hidden, list(range(rows)), shard_rows=16)
    assert likelihood.load_reference_scores(trace, list(range(rows)), head.identity) is None  # no file yet
    prepare_likelihood_reference(trace, head=head)
    assert likelihood.load_reference_scores(trace, list(range(rows)), head.identity) is not None
    return trace


@pytest.mark.parametrize("key", sorted(IdentifiedHead.identity))
def test_reference_scores_for_another_head_are_ignored(tmp_path, key):
    head = IdentifiedHead(hidden=8, vocab=32)
    trace = _stored_trace(tmp_path, head)
    changed = dict(head.identity, **{key: head.identity[key] + "-other"})
    assert likelihood.load_reference_scores(trace, list(range(32)), changed) is None


def test_reference_scores_for_another_document_are_ignored(tmp_path):
    head = IdentifiedHead(hidden=8, vocab=32)
    trace = _stored_trace(tmp_path, head)
    for metadata_path in (trace / "metadata.json", tmp_path / "source" / "metadata.json"):
        metadata = json.loads(metadata_path.read_text())
        metadata_path.write_text(json.dumps(dict(metadata, token_ids=list(range(1, 33)))))
    assert likelihood.load_reference_scores(trace, list(range(32)), head.identity) is None


def test_reference_scores_missing_a_position_are_ignored(tmp_path):
    head = IdentifiedHead(hidden=8, vocab=32)
    trace = _stored_trace(tmp_path, head)
    assert likelihood.load_reference_scores(trace, [0, 31, 32], head.identity) is None


def test_head_without_identity_computes_the_reference(tmp_path):
    trace = _stored_trace(tmp_path, IdentifiedHead(hidden=8, vocab=32))
    positions = torch.arange(16)
    hidden = torch.randn(16, 8, generator=torch.Generator().manual_seed(3)).bfloat16()
    save_samples(
        tmp_path / "run.safetensors", positions, hidden, likelihood.next_tokens(list(range(32)), positions.tolist())
    )
    # Same weights, but nothing says so: the stored file must not be trusted.
    report = likelihood.compare(trace, tmp_path / "run.safetensors", LinearHead(hidden=8, vocab=32))
    assert report["reference"] == "computed"
