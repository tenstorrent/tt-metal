# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1 and P2: real weights, full depth, per-layer K/V against the prepared CPU golden trace.

The recipe's row for both stages is ``minimax_m3/tests/galaxy_prefill_kv_pcc.py``, run with
``PREFILL_CHUNKED=0`` for P1 and ``PREFILL_CHUNKED=1`` for P2. This is that file. It is also the
measurement engine ``tests/test_prefill_acceptance.py`` calls — one implementation, so the number
the acceptance report carries is the number this row measured, not a second opinion.

**The one-shot / chunked difference is entirely in the cache period.** ``MistralModel``'s
``chunk_size`` is the block-cyclic addressing period of the KV cache *and* the length of one
prefill call; the two have to agree or the read-back permutation
(``kv_cache.cache_row_index``) inverts a layout nothing wrote. So one-shot builds the model with
``chunk_size = seq_len`` and makes a single 10240-token call, and chunked builds it with the
spec's 5120 and makes two. The acceptance report says ``chunk_size: 5120`` in both modes because
that is the configured chunk size the contract asks for; the cache period one-shot actually used
is in ``cache_period``, which the report does not carry but this module logs.

That difference also selects the SDPA core (see ``tt/attention/prefill.py``): one-shot takes
``gathered_sp_attention`` — the cache is sized exactly to the sequence, so there is no prefix to
read — and chunked takes the cache-backed ``dense_sp_attention`` for both chunks. P1 and P2
therefore measure two genuinely different attention paths against one golden trace, which is what
makes P2 a statement and not a re-run.

**Weights stream.** :class:`~...reference.checkpoint.CheckpointStateDict` reads one layer from the
checkpoint as ``MistralModel`` builds it, so the host never holds more than ~2.8 GB of the ~250 GB
dequantized model. There is no tilized-weight cache: at bfloat8_b the 88 layers are ~130 GB on
disk per layout, which is not a reasonable thing to write for a bring-up, so every run pays the
measured ~9.3 s/layer streaming load instead — about 14 minutes before the first chunk runs.

**Both modes drive the P2 runtime.** The chunk loop goes through
:class:`~...tt.runtime.PrefillRuntime` — ``compile``, then ``make_chunk_input`` / ``prefill_chunk``
per chunk — rather than calling ``MistralModel`` directly, so P2's step-1 deliverable is the code
these numbers were measured through instead of a parallel implementation beside it. One-shot is
then just the degenerate schedule: one chunk, ``chunk_size = seq_len``. The runtime's schedule
assertions are covered on the host by ``tests/unit/test_runtime_contract.py``.

**Comparison layout.** The device caches K Meta-interleaved (Q/K projection rows are permuted at
load, ``tt/attention/weights.py``); the trace stores it HF half-split. ``layer_kv_meta`` is the
seam, and it is the same permutation ``tests/unit/test_model_sp_vs_ref.py`` applies against the
torch reference. V is unrotated and needs none.

Run directly::

    PREFILL_CHUNKED=0 scripts/run_safe_pytest.sh \\
        models/demos/mistral_medium_3_5_128b/tests/galaxy_prefill_kv_pcc.py -q -s
"""

import json
import os
import time
from dataclasses import dataclass, field

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.checkpoint import CheckpointLoader, CheckpointStateDict
from models.demos.mistral_medium_3_5_128b.reference.golden import GoldenTrace
from models.demos.mistral_medium_3_5_128b.tests.device_utils import read_kv_cache
from models.demos.mistral_medium_3_5_128b.tt.model import MistralModel
from models.demos.mistral_medium_3_5_128b.tt.runtime import PrefillRuntime, RuntimeConfig


def chunked_from_env() -> bool:
    """``PREFILL_CHUNKED``: ``1`` selects multi-chunk (P2), ``0`` one-shot (P1). Default one-shot."""
    return os.getenv("PREFILL_CHUNKED", "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class PrefillMeasurement:
    """What one real-weights prefill run measured. Everything the report needs, and nothing derived."""

    mode: str  # "one_shot" | "chunked"
    num_layers: int
    seq_len: int
    cache_period: int  # the block-cyclic period actually used == the length of one call
    layer_pcc: list[dict] = field(default_factory=list)
    load_seconds: float = 0.0
    compile_seconds: float = 0.0
    prefill_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        """Steady-state: the warm-up chunk ``PrefillRuntime.compile`` ran is not in this number."""
        return self.seq_len / self.prefill_seconds if self.prefill_seconds else float("nan")

    def worst(self) -> tuple[float, float]:
        """``(min k PCC, min v PCC)`` over the stack."""
        return min(r["k"] for r in self.layer_pcc), min(r["v"] for r in self.layer_pcc)

    def summary(self) -> str:
        k, v = self.worst()
        return (
            f"{self.mode}: {self.num_layers}L x {self.seq_len} tokens, period {self.cache_period}, "
            f"load {self.load_seconds:.0f}s, compile {self.compile_seconds:.1f}s, "
            f"prefill {self.prefill_seconds:.1f}s ({self.tokens_per_second:.0f} tok/s), "
            f"worst k {k:.6f} v {v:.6f}"
        )


def build_real_model(mesh, cfg, mesh_config, ccl, *, seq_len, cache_period, num_layers, loader=None):
    """The device model on the real checkpoint, weights streamed layer by layer.

    ``with_lm_head=False``: a 10240 x 12288 x 131072 matmul produces nothing the K/V comparison
    reads, and the head's 1.6 G parameters would not fit alongside the stack anyway (the 88 layers
    are already ~30.2 GiB of the ~31.8 GiB a Blackhole chip can allocate).
    """
    loader = loader or CheckpointLoader.from_env(cfg)
    return MistralModel(
        mesh,
        cfg,
        CheckpointStateDict(loader),
        ccl,
        mesh_config,
        max_seq_len=seq_len,
        chunk_size=cache_period,
        num_layers=num_layers,
        with_lm_head=False,
        # No tensor_cache_path: see the module docstring.
    )


def run_prefill_kv_pcc(mesh, cfg, mesh_config, ccl, trace, *, chunked, seq_len, chunk_size, num_layers, loader=None):
    """Load the checkpoint, prefill ``seq_len`` of the trace's tokens, PCC every layer's K and V.

    The chunk loop runs through :class:`~...tt.runtime.PrefillRuntime` in both modes, so the
    schedule is asserted per chunk rather than trusted (see that module).

    Args:
        trace: the prepared :class:`~...reference.golden.GoldenTrace` — the ground truth and the
            source of the token ids, so the device cannot be prefilled on a different prompt than
            the one that was traced.
        chunked: False runs one call of ``seq_len`` tokens; True runs ``seq_len / chunk_size``
            calls of ``chunk_size``, each attending the prefix left in the cache.
        seq_len: tokens to run. Below the trace's length this is a *causal prefix* of it: a query
            at position ``p`` attends exactly keys ``0..p`` either way, so the trace's first
            ``seq_len`` positions are still the right answer. Acceptance runs the full length.
        chunk_size: the spec's chunk size. Used as the cache period only in chunked mode.
        num_layers: stack depth. Acceptance runs ``cfg.num_hidden_layers``.

    Returns:
        A :class:`PrefillMeasurement`. Nothing is asserted here — the caller owns the thresholds,
        because P1/P2 and the acceptance report want the same numbers with different framing.
    """
    assert seq_len <= trace.n_tokens, f"trace has {trace.n_tokens} tokens, asked for {seq_len}"
    assert num_layers <= trace.num_layers, f"trace has {trace.num_layers} layers, asked for {num_layers}"
    if chunked:
        assert (
            seq_len % chunk_size == 0 and seq_len > chunk_size
        ), f"chunked mode needs at least two whole chunks: {seq_len} tokens / chunk {chunk_size}"
    period = chunk_size if chunked else seq_len
    mode = "chunked" if chunked else "one_shot"
    out = PrefillMeasurement(mode=mode, num_layers=num_layers, seq_len=seq_len, cache_period=period)

    logger.info(f"[{mode}] loading {num_layers} layers of real weights (streamed, ~9.3 s/layer)")
    t0 = time.time()
    model = build_real_model(
        mesh, cfg, mesh_config, ccl, seq_len=seq_len, cache_period=period, num_layers=num_layers, loader=loader
    )
    kv = model.allocate_cache()
    ttnn.synchronize_device(mesh)
    out.load_seconds = time.time() - t0
    logger.info(f"[{mode}] weights on mesh in {out.load_seconds:.0f}s")

    runtime = PrefillRuntime(
        model,
        RuntimeConfig(chunk_size=period, max_seq_len=kv.max_seq_len, num_layers=num_layers, num_users=1),
    )
    # Warm up first so the throughput below is steady-state and not dominated by the first chunk's
    # kernel compiles. The warm-up writes slot 0's first chunk, which the real chunk 0 overwrites.
    t0 = time.time()
    runtime.compile(kv)
    ttnn.synchronize_device(mesh)
    out.compile_seconds = time.time() - t0

    ids = trace.token_ids(seq_len)
    t0 = time.time()
    for start in range(0, seq_len, period):
        tokens = runtime.make_chunk_input(ids[0, start : start + period].tolist())
        runtime.prefill_chunk(
            tokens,
            kv,
            slot_id=0,
            actual_start=start,
            actual_end=min(start + period, seq_len),
            request_id=start // period,
        )
        tokens.deallocate(True)
    ttnn.synchronize_device(mesh)
    out.prefill_seconds = time.time() - t0
    logger.info(
        f"[{mode}] prefilled {seq_len} tokens in {out.prefill_seconds:.1f}s ({out.tokens_per_second:.0f} tok/s)"
    )

    # One device-to-host transfer per cache, not one per layer: the composed tensor is every slot.
    got_k = read_kv_cache(mesh, kv.k, cache_global=kv.max_seq_len, chunk_size=period, upto=seq_len)
    got_v = read_kv_cache(mesh, kv.v, cache_global=kv.max_seq_len, chunk_size=period, upto=seq_len)
    for i in range(num_layers):
        k_ref, v_ref = trace.layer_kv_meta(i, seq_len)  # user 0 => slot == layer_idx
        out.layer_pcc.append({"layer": i, "k": _pcc(k_ref, got_k[i : i + 1]), "v": _pcc(v_ref, got_v[i : i + 1])})
    logger.info(f"[{mode}] {out.summary()}")
    return out


def _pcc(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Plain fp32 correlation as a float.

    ``comp_pcc`` returns a (bool, str-or-float) pair shaped for an assertion; the report needs a
    number for every layer whether it passed or not, so the correlation is taken directly here.
    fp64 accumulation over 10 M elements: at 10240 tokens the fp32 sum of squares is within a few
    ULP of saturating the mantissa, and a PCC that is wrong in the sixth digit is a PCC nobody can
    interpret.
    """
    assert ref.shape == got.shape, f"shape {tuple(got.shape)} != reference {tuple(ref.shape)}"
    x, y = ref.double().flatten(), got.double().flatten()
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


def report_lines(m: PrefillMeasurement, *, lower: float, target: float) -> list[str]:
    """Per-layer lines for the log, flagging anything below ``target``. Every layer, in order."""
    lines = [m.summary()]
    for row in m.layer_pcc:
        worst = min(row["k"], row["v"])
        flag = "" if worst >= target else ("  <- below target" if worst >= lower else "  <- BELOW LOWER BOUND")
        lines.append(f"  layer {row['layer']:2d}  k {row['k']:.7f}  v {row['v']:.7f}{flag}")
    return lines


# ---------------------------------------------------------------------------------------------
# The P1 / P2 Testing-table row
# ---------------------------------------------------------------------------------------------
#: Diagnostic overrides. Unset (the default) runs the full stack over the whole trace, which is
#: what P1 and P2 are defined at; ``tests/test_prefill_acceptance.py`` does not read them at all.
#: They exist so a wiring change can be smoke-tested in two minutes instead of forty.
DIAG_LAYERS = os.getenv("MISTRAL_PREFILL_LAYERS")
DIAG_SEQ = os.getenv("MISTRAL_PREFILL_SEQ")

#: Seconds. The repo's ``pytest.ini`` sets ``timeout = 300`` for every test, which a full-depth
#: real-weights run cannot meet: streaming 88 layers off the fp8 checkpoint and tilizing them onto
#: 32 chips is already past it before the first token is embedded. This is deliberately loose —
#: it is a hang guard, not a perf budget, and perf is out of scope for this bring-up.
FULL_RUN_TIMEOUT = 7200


@pytest.fixture(scope="module")
def trace():
    return GoldenTrace.from_env()


@pytest.mark.timeout(FULL_RUN_TIMEOUT)
def test_prefill_kv_pcc(galaxy, cfg, mesh_config, ccl, spec, trace):
    """Real weights, per-layer K/V against the golden trace, in the mode ``PREFILL_CHUNKED`` selects.

    Asserts the spec's ``pcc_lower_bound`` on every layer and reports every value, so a layer that
    clears the bound but misses ``pcc_target`` is visible in the log rather than silently passing.
    """
    num_layers = int(DIAG_LAYERS) if DIAG_LAYERS else cfg.num_hidden_layers
    seq_len = int(DIAG_SEQ) if DIAG_SEQ else trace.n_tokens
    chunked = chunked_from_env()
    if DIAG_LAYERS or DIAG_SEQ:
        logger.warning(f"[diagnostic] reduced run: {num_layers} layers x {seq_len} tokens — NOT a P1/P2 result")

    m = run_prefill_kv_pcc(
        galaxy,
        cfg,
        mesh_config,
        ccl,
        trace,
        chunked=chunked,
        seq_len=seq_len,
        chunk_size=spec.chunk_size,
        num_layers=num_layers,
    )
    for line in report_lines(m, lower=spec.pcc_lower_bound, target=spec.pcc_target):
        logger.info(line)
    print(json.dumps({"mode": m.mode, "tokens_per_second": m.tokens_per_second, "layer_pcc": m.layer_pcc}))

    bad = [r for r in m.layer_pcc if min(r["k"], r["v"]) < spec.pcc_lower_bound]
    assert not bad, f"{len(bad)} layers below pcc_lower_bound {spec.pcc_lower_bound}: {bad[:5]}"
