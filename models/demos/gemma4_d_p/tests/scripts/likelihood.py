# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Score Gemma4 prefill's next-token predictions from its final hidden states.

The runner keeps the final decoder hidden states at sampled positions. On the host the HF model's own final norm
and tied LM head turn them into logits, and two runs (or a run and a reference) are compared per context-depth bin:
ΔNLL of the gold next token, top-1 agreement, and KL over the reference's top-k plus a tail bucket. The last
prompt position, whose logits are a served request's first output token, is reported on its own.

    python -m models.demos.gemma4_d_p.tests.scripts.likelihood REFERENCE CANDIDATE

Each argument is a directory or file holding ``hidden_samples.safetensors`` (a TT run or a CPU reference). The
reference may instead be a GPU trace directory, whose final-layer stream covers every position; the candidate's saved
positions decide which are scored.

A GPU trace's side of the comparison never changes, so ``prepare_likelihood_reference.py`` stores it once as
``likelihood_reference.safetensors`` in the trace directory. When that file matches the checkpoint and the document,
only the candidate's logits are computed.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

import torch
from loguru import logger
from safetensors import safe_open
from safetensors.torch import save_file

import ttnn
from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, Gemma4ServiceConfig

HIDDEN_SAMPLES = "hidden_samples.safetensors"
SAMPLE_STRIDE = 16
TOP_K = 20
DEPTH_BINS = ((0, 8192), (8192, 65536), (65536, 262144))
SCORE_BATCH = 256
REFERENCE_SCORES = "likelihood_reference.safetensors"
# Bump when ReferenceScores or the scoring math changes, so stored scores are recomputed rather than misread.
REFERENCE_FORMAT = 1
_NORM_KEY = "model.language_model.norm.weight"
_EMBED_KEY = "model.language_model.embed_tokens.weight"


def sample_positions(context_len, chunk_size, stride=SAMPLE_STRIDE):
    """Every ``stride``-th position plus the whole last chunk, which ends at the first output token."""
    return sorted(set(range(0, context_len, stride)) | set(range(max(context_len - chunk_size, 0), context_len)))


def next_tokens(token_ids, positions):
    """The gold next token at each position, or -1 past the end of the document."""
    return torch.tensor([token_ids[p + 1] if p + 1 < len(token_ids) else -1 for p in positions], dtype=torch.int64)


def read_chunk_hidden(output):
    """Gather one chunk's final hidden states from the traced output, in position order."""
    cp, tp = Gemma4ServiceConfig.MESH_SHAPE
    host = ttnn.from_device(output, blocking=True)
    # The output is all-gathered across TP, so one column of the mesh holds each CP rank's rows once.
    shards = ttnn.get_device_tensors(host)
    rows = [ttnn.to_torch(shards[row * tp]) for row in range(cp)]
    return torch.cat(rows, dim=-2).reshape(-1, rows[0].shape[-1])


class HiddenSampler:
    """Keep the sampled rows of each prefilled chunk's final hidden states."""

    def __init__(self, context_len, chunk_size, token_ids):
        self.context_len = context_len
        self.chunk_size = chunk_size
        self.positions = sample_positions(context_len, chunk_size)
        self.next_tokens = next_tokens(token_ids, self.positions)
        self.rows = {}

    def add_chunk(self, output, actual_start, actual_end):
        wanted = [p for p in self.positions if actual_start <= p < actual_end]
        if wanted:
            hidden = read_chunk_hidden(output)
            self.rows[actual_start] = hidden[torch.tensor(wanted) - actual_start].clone()

    def save(self, path):
        hidden = torch.cat([self.rows[start] for start in sorted(self.rows)])
        if hidden.shape[0] != len(self.positions):
            raise ValueError(f"Captured {hidden.shape[0]} of {len(self.positions)} sampled hidden rows")
        save_samples(path, torch.tensor(self.positions), hidden, self.next_tokens, chunk_size=self.chunk_size)


def save_samples(path, positions, hidden, gold, **metadata):
    save_file(
        {"positions": positions.contiguous(), "hidden": hidden.contiguous(), "next_tokens": gold.contiguous()},
        str(path),
        metadata={key: str(value) for key, value in metadata.items()},
    )


class Samples(NamedTuple):
    positions: torch.Tensor
    hidden: torch.Tensor
    next_tokens: torch.Tensor


def _samples_path(source):
    source = Path(source)
    if source.is_file():
        return source
    return source / HIDDEN_SAMPLES if (source / HIDDEN_SAMPLES).is_file() else None


def select_positions(samples, positions):
    positions = torch.as_tensor(positions, dtype=torch.int64)
    index = torch.searchsorted(samples.positions, positions).clamp_max(len(samples.positions) - 1)
    if not torch.equal(samples.positions[index], positions):
        raise ValueError("The samples do not cover every compared position")
    return Samples(positions, samples.hidden[index], samples.next_tokens[index])


def load_samples(source, positions=None):
    """Load a run's sampled hidden states, or a GPU trace's at ``positions``."""
    if (path := _samples_path(source)) is not None:
        with safe_open(str(path), framework="pt") as tensors:
            samples = Samples(*(tensors.get_tensor(name) for name in Samples._fields))
        return samples if positions is None else select_positions(samples, positions)
    if positions is None:
        raise ValueError(f"{source} is a GPU trace; compare it against a run, which supplies the positions")
    return load_gpu_samples(source, positions)


def gpu_final_layer_stream(trace_dir):
    """The capture directory, metadata and final-layer stream index of a GPU trace.

    A prepared KV-only copy names the full capture it came from, which holds the stream.
    """
    trace_dir = Path(trace_dir)
    metadata = json.loads((trace_dir / "metadata.json").read_text())
    if not (trace_dir / "index.json").is_file() and "source_trace_dir" in metadata:
        return gpu_final_layer_stream(metadata["source_trace_dir"])
    name = f"decoder_output_layer_{metadata['n_layers'] - 1}"
    streams = json.loads((trace_dir / "index.json").read_text())["tensor_streams"]
    if name not in streams:
        raise ValueError(f"{trace_dir} has no {name} stream for the likelihood check")
    return trace_dir, metadata, streams[name]


def load_gpu_samples(trace_dir, positions):
    """Read the GPU trace's final-layer (pre-norm) hidden states at ``positions``."""
    trace_dir, metadata, stream = gpu_final_layer_stream(trace_dir)
    positions = torch.as_tensor(positions, dtype=torch.int64)
    hidden = torch.empty((len(positions), stream["shape_tail"][0]), dtype=torch.bfloat16)
    loaded = torch.zeros(len(positions), dtype=torch.bool)
    for chunk in stream["chunks"]:
        start, end = chunk["row_start"], chunk["row_end"]
        selected = ((positions >= start) & (positions < end)).nonzero().flatten()
        if len(selected):
            with safe_open(str(trace_dir / chunk["path"]), framework="pt") as tensors:
                rows = tensors.get_slice(f"decoder_output_layer_{metadata['n_layers'] - 1}")
                first, last = positions[selected[0]].item(), positions[selected[-1]].item()
                hidden[selected] = rows[first - start : last + 1 - start][positions[selected] - first]
            loaded[selected] = True
    # A short or gapped stream would otherwise leave uninitialized rows to be scored.
    if not loaded.all():
        missing = positions[~loaded]
        raise ValueError(f"GPU trace has no rows for {len(missing)} requested positions, from {missing[0].item()}")
    return Samples(positions, hidden, next_tokens(metadata["token_ids"], positions.tolist()))


class FinalHead:
    """The HF model's final norm and tied LM head, with its final logit softcap, in fp32 on the host."""

    def __init__(self, hf_model_id=None):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

        from models.demos.gemma4_d_p.utils.partial_weights import load_state_dict_subset

        adapter = Gemma4PrefillAdapter()
        hf_model_id = hf_model_id or adapter.hf_model_id
        config = adapter.load_hf_config()
        state = load_state_dict_subset(hf_model_id, lambda key: key in (_NORM_KEY, _EMBED_KEY))
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm.weight.data = state[_NORM_KEY]
        # Gemma4 ties the LM head to the token embedding.
        self.lm_head = state.pop(_EMBED_KEY).float()
        self.softcap = config.final_logit_softcapping
        from models.demos.gemma4_d_p.utils.partial_weights import resolve_checkpoint_dir

        # What stored reference scores must match: the same weights (checkpoint snapshot), softcap and top-k.
        self.identity = dict(
            model=str(hf_model_id),
            checkpoint=resolve_checkpoint_dir(hf_model_id).name,
            softcap=str(self.softcap),
            top_k=str(TOP_K),
            format=str(REFERENCE_FORMAT),
        )

    @torch.no_grad()
    def logprobs(self, hidden):
        logits = self.norm(hidden).float() @ self.lm_head.T
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.log_softmax(logits, dim=-1)


def _tail_logprob(logprobs, ids):
    """Log of the mass outside ``ids``, summed directly: 1 - sum(top-k) rounds to zero or below when the top-k holds
    nearly all the mass."""
    return logprobs.scatter(-1, ids, -math.inf).logsumexp(-1)


def _kl(p, q, p_tail, q_tail):
    """KL(p || q) over k buckets (log-probabilities [n, k]) plus one tail bucket each."""
    p, q, p_tail, q_tail = (x.double() for x in (p, q, p_tail, q_tail))
    tail = torch.where(p_tail == -math.inf, 0.0, p_tail.exp() * (p_tail - q_tail))
    return (p.exp() * (p - q)).sum(-1) + tail


def topk_kl(reference, candidate, k=TOP_K):
    """KL(reference || candidate) over the reference's top-k tokens plus one bucket for the rest."""
    ids = reference.topk(k, dim=-1).indices
    return _kl(
        reference.gather(-1, ids),
        candidate.gather(-1, ids),
        _tail_logprob(reference, ids),
        _tail_logprob(candidate, ids),
    )


class ReferenceScores(NamedTuple):
    """The reference's side of the comparison at each position: all scoring needs from its logits."""

    positions: torch.Tensor  # int64 [n]
    next_tokens: torch.Tensor  # int64 [n], -1 past the end of the document
    gold_logprob: torch.Tensor  # float32 [n], log-probability of the gold next token (unused where there is none)
    top1: torch.Tensor  # int64 [n], argmax token
    top_ids: torch.Tensor  # int64 [n, k], the top-k tokens
    top_logprobs: torch.Tensor  # float32 [n, k], their log-probabilities
    tail_logprob: torch.Tensor  # float32 [n], log of the mass outside the top-k (-inf when there is none)


def reference_scores(head, samples, batch=SCORE_BATCH, k=TOP_K):
    """Score a reference's hidden states once, keeping only what comparisons read."""
    parts = []
    for start in range(0, len(samples.positions), batch):
        rows = slice(start, start + batch)
        lp = head.logprobs(samples.hidden[rows])
        top = lp.topk(k, dim=-1)
        gold = samples.next_tokens[rows].clamp_min(0).unsqueeze(-1)
        parts.append(
            (
                lp.gather(-1, gold).squeeze(-1),
                lp.argmax(-1),
                top.indices,
                top.values,
                _tail_logprob(lp, top.indices),
            )
        )
    gold_logprob, top1, top_ids, top_logprobs, tail_logprob = (torch.cat(part) for part in zip(*parts))
    return ReferenceScores(
        samples.positions, samples.next_tokens, gold_logprob, top1, top_ids, top_logprobs, tail_logprob
    )


def select_reference(scores, positions):
    positions = torch.as_tensor(positions, dtype=torch.int64)
    index = torch.searchsorted(scores.positions, positions).clamp_max(len(scores.positions) - 1)
    if not torch.equal(scores.positions[index], positions):
        raise ValueError("The reference scores do not cover every compared position")
    return ReferenceScores(*(field[index] for field in scores))


def document_sha(token_ids):
    return hashlib.sha256(torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes()).hexdigest()


def save_reference_scores(path, scores, identity, token_ids):
    tensors = {name: value.contiguous() for name, value in scores._asdict().items()}
    tensors["top_ids"] = tensors["top_ids"].to(torch.int32)
    save_file(tensors, str(path), metadata=dict(identity, document_sha256=document_sha(token_ids)))


def load_reference_scores(trace_dir, positions, identity):
    """The stored reference scores of a GPU trace at ``positions``, or None when absent or stale."""
    path = Path(trace_dir) / REFERENCE_SCORES
    if not path.is_file():
        return None
    _, metadata, _ = gpu_final_layer_stream(trace_dir)
    expected = dict(identity, document_sha256=document_sha(metadata["token_ids"]))
    with safe_open(str(path), framework="pt") as tensors:
        stored = tensors.metadata() or {}
        if any(stored.get(key) != value for key, value in expected.items()):
            mismatch = sorted(key for key, value in expected.items() if stored.get(key) != value)
            logger.warning(f"Ignoring {path}: it does not match the current {', '.join(mismatch)}")
            return None
        scores = ReferenceScores(*(tensors.get_tensor(name) for name in ReferenceScores._fields))
    scores = scores._replace(top_ids=scores.top_ids.to(torch.int64))
    try:
        return select_reference(scores, positions)
    except ValueError:
        logger.warning(f"Ignoring {path}: it does not cover every compared position")
        return None


def score_positions(head, reference, candidate, batch=SCORE_BATCH):
    """Per-position gold NLL under both runs, top-1 agreement and top-k KL.

    ``reference`` is either the reference's hidden states (Samples) or its stored ReferenceScores.
    """
    if isinstance(reference, Samples):
        if not torch.equal(reference.positions, candidate.positions):
            raise ValueError("Reference and candidate must cover the same positions of the same document")
        reference = reference_scores(head, reference, batch)
    gold = reference.next_tokens
    if not torch.equal(reference.positions, candidate.positions) or not torch.equal(gold, candidate.next_tokens):
        raise ValueError("Reference and candidate must cover the same positions of the same document")
    has_gold = gold >= 0
    nll_cand, top1, kl = [], [], []
    for start in range(0, len(gold), batch):
        rows = slice(start, start + batch)
        lp_cand = head.logprobs(candidate.hidden[rows])
        index = gold[rows].clamp_min(0).unsqueeze(-1)
        nll_cand.append(-lp_cand.gather(-1, index).squeeze(-1).double())
        top1.append(reference.top1[rows] == lp_cand.argmax(-1))
        # KL(reference || candidate) over the reference's top-k tokens plus one bucket for the rest.
        ids = reference.top_ids[rows]
        q, q_tail = lp_cand.gather(-1, ids), _tail_logprob(lp_cand, ids)
        kl.append(_kl(reference.top_logprobs[rows], q, reference.tail_logprob[rows], q_tail))
    nan = torch.tensor(math.nan, dtype=torch.float64)
    return dict(
        positions=reference.positions,
        nll_reference=torch.where(has_gold, -reference.gold_logprob.double(), nan),
        nll_candidate=torch.where(has_gold, torch.cat(nll_cand), nan),
        top1_agree=torch.cat(top1),
        topk_kl=torch.cat(kl),
    )


def _summary(scores, selected):
    with_gold = selected & ~scores["nll_reference"].isnan()
    delta = scores["nll_candidate"][with_gold] - scores["nll_reference"][with_gold]
    has_gold = bool(with_gold.any())
    return dict(
        positions=int(selected.sum()),
        nll_reference=scores["nll_reference"][with_gold].mean().item() if has_gold else None,
        nll_candidate=scores["nll_candidate"][with_gold].mean().item() if has_gold else None,
        delta_nll=delta.mean().item() if has_gold else None,
        mean_abs_delta_nll=delta.abs().mean().item() if has_gold else None,
        top1_agreement=scores["top1_agree"][selected].double().mean().item(),
        topk_kl=scores["topk_kl"][selected].mean().item(),
        max_topk_kl=scores["topk_kl"][selected].max().item(),
    )


def summarize(scores):
    """Per-depth-bin metrics and the last prompt position on its own."""
    positions = scores["positions"]
    context_len = positions.max().item() + 1
    bins = []
    for start, end in DEPTH_BINS:
        selected = (positions >= start) & (positions < end)
        if selected.any():
            bins.append(dict(start=start, end=min(end, context_len), **_summary(scores, selected)))
    last = positions == context_len - 1
    return dict(
        context_len=context_len,
        top_k=TOP_K,
        bins=bins,
        all=_summary(scores, torch.ones_like(last)),
        last_position=dict(position=context_len - 1, **_summary(scores, last)),
    )


def check_limits(report, limits):
    """One (name, required, achieved, passed) row per depth bin and metric.

    ``limits`` maps each bin's first position to (minimum top-1 agreement, maximum ΔNLL, maximum top-k KL). A
    missing value, such as ΔNLL for a bin without gold tokens, fails its criterion.
    """
    rows = []
    for entry in report["bins"]:
        min_top1, max_delta_nll, max_kl = limits[entry["start"]]
        depth = f"{entry['start'] // 1024}-{entry['end'] // 1024}K"
        for name, sign, limit, achieved in (
            (f"Top-1 {depth}", ">=", min_top1, entry["top1_agreement"]),
            (f"ΔNLL {depth}", "<=", max_delta_nll, entry["delta_nll"]),
            (f"Top-{report['top_k']} KL {depth}", "<=", max_kl, entry["topk_kl"]),
        ):
            passed = achieved is not None and (achieved >= limit if sign == ">=" else achieved <= limit)
            rows.append((name, f"{sign} {limit:.6f}", achieved, passed))
    return rows


def compare(reference, candidate, head=None):
    """Compare a candidate run with a reference (a run, a CPU reference or a GPU trace)."""
    candidate_samples = load_samples(candidate)
    head = head or FinalHead()
    if _samples_path(reference) is None:
        # Stored scores are used only for a head that can say which weights they must come from.
        identity = getattr(head, "identity", None)
        stored = load_reference_scores(reference, candidate_samples.positions, identity) if identity else None
        if stored is not None:
            return dict(summarize(score_positions(head, stored, candidate_samples)), reference="stored")
        reference_samples = load_gpu_samples(reference, candidate_samples.positions)
    else:
        # Runs at different chunk sizes sample different last chunks; score the positions both hold.
        reference_samples = load_samples(reference)
        shared = candidate_samples.positions[torch.isin(candidate_samples.positions, reference_samples.positions)]
        reference_samples = select_positions(reference_samples, shared)
        candidate_samples = select_positions(candidate_samples, shared)
    return dict(summarize(score_positions(head, reference_samples, candidate_samples)), reference="computed")


def _tokens(count):
    return f"{count // 1024}k" if count % 1024 == 0 and count else str(count)


def format_report(report):
    def number(value, width, digits):
        return f"{'n/a':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

    kl_label = f"Top-{report['top_k']} KL"
    header = (
        f"{'Depth':>13} {'Positions':>9} {'NLL ref':>8} {'NLL cand':>8} {'ΔNLL':>10} {'|ΔNLL|':>9} "
        f"{'Top-1':>7} {kl_label:>11} {'Max KL':>9}"
    )

    def row(label, entry):
        return (
            f"{label:>13} {entry['positions']:>9} {number(entry['nll_reference'], 8, 4)} "
            f"{number(entry['nll_candidate'], 8, 4)} {number(entry['delta_nll'], 10, 6)} "
            f"{number(entry['mean_abs_delta_nll'], 9, 6)} {entry['top1_agreement']:>7.4f} "
            f"{entry['topk_kl']:>11.3e} {entry['max_topk_kl']:>9.2e}"
        )

    lines = [header]
    lines += [row(f"{_tokens(entry['start'])}-{_tokens(entry['end'])}", entry) for entry in report["bins"]]
    lines.append(row("All", report["all"]))
    lines.append(row(f"Last ({report['last_position']['position']})", report["last_position"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="Run directory, hidden_samples file or GPU trace directory")
    parser.add_argument("candidate", type=Path, help="Run directory or hidden_samples file")
    parser.add_argument("--json", type=Path, help="Also write the report here")
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        logger.info(f"Likelihood report: {args.json}")


if __name__ == "__main__":
    main()
