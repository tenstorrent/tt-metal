"""A batch-N model is measured serving N users, at the SAME sequence length each.

The generated perf test wrote `batch=1` into three places as a literal, so a pipeline emit-e2e had
built to serve 8 users was measured serving one, and its aggregate throughput under-reported
eightfold. Batch is a property of the ARTIFACT under test, not a measurement condition the harness
picks -- unlike ISL and OSL, which are deliberately the tool's choice.

Un-hardcoding it alone would have been worse than leaving it. PipelineStageAdapter built its input as

    torch.tensor(ids).reshape(self.batch, -1)

which SPLITS one prompt across the rows: at batch 8 a 128-token prompt became eight sequences of 16.
ISL silently fell to an eighth of what the test declared while the scorecard still multiplied
throughput by 8 (perf_mcp.py: TS = tsu * batch) -- a batch speedup manufactured out of a shorter
sequence. It also raised outright whenever ISL was not divisible by batch. PipelineDecodeAdapter had
the same defect by another route: it passed the bare prompt to decode_prefill whatever the batch, so
batch 8 built single-user state and then had its throughput multiplied anyway.

Batch N means N users each doing the DECLARED work. The row count is the only thing that varies.

The third piece is the ceiling. perf_target.active_bytes models a KV term but had no batch factor,
so an 8-user step was costed as a 1-user step. Batch scales the KV term and NOTHING else -- the
weights are read once and amortised across the batch, which is the whole reason batching pays -- so
the per-user ceiling falls by the added KV, not by 8x.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import perf_adapter as PA  # noqa: E402
from agent import perf_target as PT  # noqa: E402

GEN = Path(__file__).resolve().parent.parent / "agent" / "perf_test_gen.py"
ISL = 128


class _Pipe:
    """A pipeline that accepts a batched prompt and records what it was given."""

    def __init__(self, declared=None, accepts_batch=True):
        if declared is not None:
            self.max_batch_size = declared
        self._accepts = accepts_batch
        self.seen = None

    def decode_prefill(self, ids):
        import torch

        t = torch.as_tensor(ids)
        if t.dim() > 1 and not self._accepts:
            raise TypeError("this pipeline wants a 1-D prompt")
        self.seen = tuple(t.shape)
        return {"state": 1}

    def decode_step(self, state):
        return state


def _prompt():
    import torch

    return torch.arange(ISL, dtype=torch.long)


# ---------------------------------------------------------------- every user gets the full sequence


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 32])
def test_the_prompt_is_replicated_not_split(batch):
    """THE DEFECT: reshape(batch, -1) turned 128 tokens into `batch` rows of 128/batch."""
    a = PA.PipelineStageAdapter(lambda d: _Pipe(), _prompt(), batch=batch)
    a.batch = batch
    got = a._inputs_dict()["input_ids"]
    assert tuple(got.shape) == (batch, ISL), got.shape
    for row in range(batch):
        assert got[row].tolist() == list(range(ISL)), "row %d is not the full prompt" % row


def test_a_batch_that_does_not_divide_isl_still_works():
    """reshape raised outright here; 128 tokens across 5 users is not a reshape, it is 5 copies."""
    a = PA.PipelineStageAdapter(lambda d: _Pipe(), _prompt(), batch=5)
    a.batch = 5
    assert tuple(a._inputs_dict()["input_ids"].shape) == (5, ISL)


@pytest.mark.parametrize("batch", [2, 4, 8])
def test_the_decode_adapter_prefills_every_user(batch):
    """The other route to the same fake speedup: prefill one sequence, multiply throughput by N."""
    pipe = _Pipe()
    a = PA.PipelineDecodeAdapter(lambda d: pipe, _prompt(), batch=batch)
    a.setup(object())
    assert pipe.seen == (batch, ISL), pipe.seen


# ---------------------------------------------------------------- batch comes from the model


@pytest.mark.parametrize("declared", [1, 4, 8, 64])
def test_the_pipeline_declares_its_own_batch(declared):
    assert PA.resolve_batch(_Pipe(declared=declared), 0) == declared


def test_an_explicit_request_overrides_the_model():
    """So a batch sweep does not need the demo rebuilt."""
    assert PA.resolve_batch(_Pipe(declared=8), 2) == 2


def test_a_pipeline_that_declares_nothing_is_batch_one():
    assert PA.resolve_batch(_Pipe(declared=None), 0) == 1


@pytest.mark.parametrize("attr", ["max_batch_size", "batch_size", "batch", "max_batch"])
def test_every_spelling_a_model_might_use(attr):
    p = _Pipe(declared=None)
    setattr(p, attr, 4)
    assert PA.resolve_batch(p, 0) == 4


def test_junk_declarations_fall_back_rather_than_crash():
    p = _Pipe(declared=None)
    p.max_batch_size = "eight"
    assert PA.resolve_batch(p, 0) == 1


def test_the_adapter_resolves_from_the_pipeline_at_setup():
    """It cannot resolve in __init__ -- the pipeline does not exist until setup(device)."""
    a = PA.PipelineStageAdapter(lambda d: _Pipe(declared=8), _prompt(), batch=0)
    assert a.batch == 1, "before setup it must not claim a batch it has not confirmed"
    a.setup(object())
    assert a.batch == 8


def test_the_generated_test_no_longer_hardcodes_batch():
    src = GEN.read_text()
    assert "batch=1)" not in src, "a literal batch=1 is still written into the generated test"
    assert "TT_PERF_BATCH" in src and "PERF_BATCH" in src


# ---------------------------------------------------------------- a batch that did not happen is not claimed


def test_a_pipeline_that_refuses_a_batched_prompt_corrects_its_batch():
    """trace_replay derives tokens_per_sec from adapter.batch. Leaving it at 8 after serving one user
    would report an 8x aggregate that never ran -- the exact defect this file exists to stop."""
    pipe = _Pipe(accepts_batch=False)
    a = PA.PipelineDecodeAdapter(lambda d: pipe, _prompt(), batch=8)
    a.setup(object())
    assert a.batch == 1, "batch must be corrected to what actually ran"
    assert pipe.seen == (ISL,)


# ---------------------------------------------------------------- the ceiling knows about batch


_MF = {
    "total_params": 8_000_000_000,
    "dominant_dtype": "bfloat8_b",
    "layers": 32,
    "kv_heads": 8,
    "head_dim": 128,
}


def test_weights_are_amortised_and_kv_is_not():
    """The whole reason batching pays. Doubling the batch must NOT double the bytes."""
    b1 = PT.active_bytes(_MF, seq_len=4096, batch=1)
    b8 = PT.active_bytes(_MF, seq_len=4096, batch=8)
    kv1 = b1 - PT.active_bytes(_MF, seq_len=0, batch=1)
    assert b8 == pytest.approx(b1 + 7 * kv1, rel=1e-9)
    assert b8 < 8 * b1, "weights must not be counted once per user"


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
def test_the_kv_term_is_linear_in_batch(batch):
    weights = PT.active_bytes(_MF, seq_len=0)
    kv1 = PT.active_bytes(_MF, seq_len=4096, batch=1) - weights
    assert PT.active_bytes(_MF, seq_len=4096, batch=batch) == pytest.approx(weights + batch * kv1, rel=1e-9)


def test_batch_changes_nothing_without_a_sequence():
    """No seq_len means no KV term modelled, so there is nothing for batch to scale."""
    assert PT.active_bytes(_MF, batch=8) == PT.active_bytes(_MF, batch=1)


def test_the_default_is_unchanged():
    """Every existing caller omits batch; they must get exactly what they got before."""
    assert PT.active_bytes(_MF, seq_len=4096) == PT.active_bytes(_MF, seq_len=4096, batch=1)


def test_a_bad_batch_does_not_corrupt_the_ceiling():
    base = PT.active_bytes(_MF, seq_len=4096, batch=1)
    for bad in (0, -3, None):
        assert PT.active_bytes(_MF, seq_len=4096, batch=bad) == base


def test_the_per_user_ceiling_falls_with_batch_but_not_eightfold():
    """The number a reader would misread. More users means more bytes per step, so per-user
    throughput drops -- but nothing like linearly, because the weights are shared."""
    c1, _ = PT.rate_and_band(PT.active_bytes(_MF, seq_len=4096, batch=1), 512e9, frac=0.8)
    c8, _ = PT.rate_and_band(PT.active_bytes(_MF, seq_len=4096, batch=8), 512e9, frac=0.8)
    assert c8 < c1
    assert c8 > c1 / 8.0, "per-user ceiling must not scale as if weights were re-read per user"
