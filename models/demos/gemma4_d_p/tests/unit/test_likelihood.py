# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import math

import torch
from safetensors.torch import save_file

from models.demos.gemma4_d_p.tt.runners import likelihood
from models.demos.gemma4_d_p.tt.runners.likelihood import (
    HiddenSampler,
    Samples,
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
    with expect_error(ValueError, "covers 12 positions"):
        load_samples(prepared, positions=[12])

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
