# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 on-device sampling (greedy + top-k + top-p).

Covers acceptance criteria:
- batch=32 decode runs through the on-device sampling kernel (no host
  argmax fallback) — see ``test_sampling_greedy_batch32``.
- Top-k and top-p code paths actually fire and constrain the sampled tokens
  to the torch reference set — see ``test_sampling_top_k_constrained`` and
  ``test_sampling_top_p_constrained``.
- ``test_sampling`` PCC ≥ 0.999 — see ``test_sampling_log_probs_pcc`` which
  compares the per-sampled-token logprob returned by the sampling kernel
  against a torch ``log_softmax`` reference.

Decoder shape: vocab=262144 with per-device shard = ``V/TP``. The on-device
sampling module only requires ``tp > 1``, so the tests parametrize over both
``(1, 4)`` (Blackhole quietbox, TP=4, shard 65536) and ``(1, 8)`` (T3K, TP=8,
shard 32768). The host-argmax fallback for TP=1 is covered by ``test_full_model``.

    pytest -k "1x4 and sampling"   # Blackhole quietbox (on-device sampling active)
    pytest -k "1x8 and sampling"   # T3K (on-device sampling active)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params

from ...config import MeshConfig, ModeConfig
from ...tests.test_factory import compare_tensors, get_pcc_threshold, parametrize_mesh_with_fabric
from ...tt.model import Gemma4Model, _compute_per_device_vocab

LOG_PROBS_SUPPORTED_DEVICE_COUNTS = (8, 32)

VOCAB_SIZE = 262144
BATCH_SIZE = 32  # TTSampling minimum
MAX_TOP_K = 32


def test_gemma4_make_sampling_args_sets_sampling_dp_from_mesh_rows():
    """Regression test for row-sharded sampling contract.

    `_get_sampling_contract` reads `model.sampling_dp`; Gemma4 must set it
    consistently with the mesh row count so DP=4 (4x8) can use sampling_dp=4.
    This test is pure-Python and does not require hardware.
    """

    class _FakeMeshDevice:
        def __init__(self, shape):
            self.shape = shape

        def get_num_devices(self):
            return self.shape[0] * self.shape[1]

    class _FakeConfig:
        vocab_size = VOCAB_SIZE

    mesh = _FakeMeshDevice((4, 8))
    args = Gemma4Model._make_sampling_args(_FakeConfig(), mesh, tp=mesh.shape[1])
    assert args.sampling_dp == 4


def _make_sampling_args(mesh_device, *, use_topk_logprobs=False):
    """Build the same args ``Gemma4Model._make_sampling_args`` produces.

    Kept local to the test so it stays decoupled from the model wiring —
    test failures point at the sampling module, not at model construction.
    """

    class _Args:
        pass

    tp = mesh_device.shape[1]
    per_device_vocab = _compute_per_device_vocab(VOCAB_SIZE, tp)

    args = _Args()
    args.vocab_size = VOCAB_SIZE
    args.padded_vocab_size = per_device_vocab * tp
    args.cluster_shape = tuple(mesh_device.shape)
    args.sampling_all_gather_axis = 1
    # Keep these unit tests in non-row-sharded mode. Row-sharded sampling
    # (sampling_dp > 1) is exercised separately at the Generator/vLLM contract level.
    args.sampling_dp = 1
    args.num_devices = mesh_device.get_num_devices()
    args.is_galaxy = mesh_device.shape[0] > 1
    args.model_config = {}
    args.max_top_k = MAX_TOP_K
    args.sub_core_grids = None
    args.use_topk_logprobs = use_topk_logprobs
    return args


def _shard_logits(logits_cpu, mesh_device):
    """Column-parallel shard logits across the TP (column) axis.

    Mirrors the ``Gemma4Model`` decode path: each device holds
    ``[1, 1, B, V/TP]`` of the post-softcap logits and the sampling module
    drives the cross-device gather.
    """
    tp = mesh_device.shape[1]
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    col_mapper = mesh_config.column_parallel(mesh_device)
    return ttnn.from_torch(
        logits_cpu,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=col_mapper,
    )


def _make_hot_logits(batch_size, mesh_device, num_hot=8, baseline=0.001, hot_value=10.0, hot_decay=0.25):
    """Build logits with ``num_hot`` elevated tokens spread across TP shards.

    Hot tokens are placed in distinct shards so the cross-device top-k +
    all-gather path is exercised. Out-of-vocab padding columns are set to
    ``-inf`` so the sampler never selects them.
    """
    tp = mesh_device.shape[1]
    per_device_vocab = _compute_per_device_vocab(VOCAB_SIZE, tp)
    padded_vocab = per_device_vocab * tp

    hot_tokens = []
    for i in range(num_hot):
        shard_idx = i % tp
        candidate = shard_idx * per_device_vocab + 100 + i
        if candidate >= VOCAB_SIZE:
            candidate = 200 + i
        hot_tokens.append(candidate)

    logits = torch.full((1, 1, batch_size, padded_vocab), baseline, dtype=torch.bfloat16)
    for idx, tok in enumerate(hot_tokens):
        logits[:, :, :, tok] = hot_value - idx * hot_decay
    if padded_vocab > VOCAB_SIZE:
        logits[:, :, :, VOCAB_SIZE:] = float("-inf")
    return logits, hot_tokens


def _extract_tokens(tt_tokens, batch_size):
    return ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)[:batch_size].tolist()


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4), (1, 8)])
def test_sampling_greedy_batch32(mesh_device, reset_seeds):
    """Greedy on-device sampling at batch=32 matches CPU argmax exactly.

    Each of the 32 users gets a distinct winner token spread across the TP
    shards (so the cross-device top-k path is exercised for every row).
    """
    tp = mesh_device.shape[1]
    per_device_vocab = _compute_per_device_vocab(VOCAB_SIZE, tp)
    args = _make_sampling_args(mesh_device)

    # Distinct winners across users, spread across shards: shard_i has
    # winner = i * per_device_vocab + (1000 + i). This guarantees the
    # per-device top-1 differs from row to row, hitting the gather path.
    winners = []
    logits_cpu = torch.zeros(1, 1, BATCH_SIZE, args.padded_vocab_size, dtype=torch.bfloat16)
    for u in range(BATCH_SIZE):
        shard_idx = u % tp
        winner = shard_idx * per_device_vocab + (1000 + u)
        winners.append(winner)
        logits_cpu[0, 0, u, winner] = 50.0

    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None)
    logits_tt = _shard_logits(logits_cpu, mesh_device)
    tt_tokens, _ = sampling.sample(logits_tt, enable_trace=False)
    tokens = _extract_tokens(tt_tokens, BATCH_SIZE)

    for u, (got, want) in enumerate(zip(tokens, winners)):
        assert got == want, f"Greedy mismatch user {u}: got {got}, expected {want}"
    logger.info(f"Greedy batch=32 sampling matched all {BATCH_SIZE} winners")


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4), (1, 8)])
def test_sampling_top_k_constrained(mesh_device, reset_seeds):
    """Top-k sampling keeps every drawn token inside the reference top-k set.

    Uses ``num_hot`` >> ``top_k`` and a tight gap between hot and baseline,
    so only the top-k hot tokens have appreciable mass. Drawing many samples
    and checking they all land in the reference set verifies (a) the kernel
    fires (no host fallback) and (b) top-k actually restricts the sampler.
    """
    top_k = 4
    num_hot = 8
    num_samples = 32
    args = _make_sampling_args(mesh_device)

    logits_cpu, hot_tokens = _make_hot_logits(BATCH_SIZE, mesh_device, num_hot=num_hot)
    expected_set = set(hot_tokens[:top_k])

    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None)
    params = format_sampling_params(SamplingParams(temperature=1.0, top_k=top_k, top_p=1.0, seed=42), BATCH_SIZE)
    sampling.reset_sampling_params(params)
    sampling.seed_manager.reset_seed([42] * BATCH_SIZE, list(range(BATCH_SIZE)))

    logits_tt = _shard_logits(logits_cpu, mesh_device)

    seen = set()
    for _ in range(num_samples):
        sampling.seed_manager.get_new_values()
        tt_tokens, _ = sampling.sample(logits_tt, enable_trace=False)
        for tok in _extract_tokens(tt_tokens, BATCH_SIZE):
            assert 0 <= tok < VOCAB_SIZE, f"Sampled token {tok} outside vocab range"
            assert tok in expected_set, (
                f"Top-k=4 sampler drew token {tok} outside top-{top_k} set {expected_set}; "
                f"hot tokens were {hot_tokens}"
            )
            seen.add(tok)
    assert len(seen) >= 2, f"Top-k sampler stuck on single token: {seen}"
    logger.info(f"Top-k={top_k} sampling stayed within {expected_set}; sampled {len(seen)} unique tokens")


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4), (1, 8)])
def test_sampling_top_p_constrained(mesh_device, reset_seeds):
    """Top-p (nucleus) sampling keeps every drawn token inside the reference nucleus.

    Hot tokens dominate so the torch-side cumulative softmax nucleus is
    well-defined; we assert sampled tokens are a subset of it.
    """
    top_p = 0.9
    num_hot = 8
    num_samples = 32
    args = _make_sampling_args(mesh_device)

    logits_cpu, hot_tokens = _make_hot_logits(BATCH_SIZE, mesh_device, num_hot=num_hot)

    # Torch reference nucleus: rank all hot tokens by softmax probability and
    # include those whose cumulative probability crosses top_p. All other
    # tokens (and -inf padding) are vanishingly small under softmax, so the
    # nucleus is fully contained in ``hot_tokens``.
    hot_logits = torch.tensor([10.0 - i * 0.25 for i in range(num_hot)])
    sorted_probs, sorted_indices = torch.sort(torch.softmax(hot_logits, dim=-1), descending=True)
    cumprob = torch.cumsum(sorted_probs, dim=-1)
    nucleus_rank = (cumprob < top_p).sum().item() + 1  # include the boundary token
    nucleus_set = {hot_tokens[i.item()] for i in sorted_indices[:nucleus_rank]}

    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None)
    params = format_sampling_params(SamplingParams(temperature=1.0, top_k=num_hot, top_p=top_p, seed=43), BATCH_SIZE)
    sampling.reset_sampling_params(params)
    sampling.seed_manager.reset_seed([43] * BATCH_SIZE, list(range(BATCH_SIZE)))

    logits_tt = _shard_logits(logits_cpu, mesh_device)

    seen = set()
    for _ in range(num_samples):
        sampling.seed_manager.get_new_values()
        tt_tokens, _ = sampling.sample(logits_tt, enable_trace=False)
        for tok in _extract_tokens(tt_tokens, BATCH_SIZE):
            assert 0 <= tok < VOCAB_SIZE, f"Sampled token {tok} outside vocab range"
            assert tok in nucleus_set, (
                f"Top-p={top_p} sampler drew token {tok} outside nucleus {nucleus_set}; "
                f"hot tokens were {hot_tokens}"
            )
            seen.add(tok)
    logger.info(f"Top-p={top_p} sampling stayed within {nucleus_set}; sampled {len(seen)} unique tokens")


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4), (1, 8)])
def test_sampling_log_probs_pcc(mesh_device, reset_seeds, request):
    """PCC ≥ 0.999 between device-computed log-probs and torch log_softmax reference.

    Acceptance criterion §1 of issue #44953 calls for ``test_sampling PCC ≥
    0.999``. The natural surface is the sampled-token log-probability the
    on-device sampling kernel returns when ``enable_log_probs=True``: with
    greedy decoding, the sampled token is deterministic, so we can compare
    the per-user logprob against ``log_softmax(logits, -1).gather(token)``.

    The shared on-device log-probs kernel
    (``models.common.sampling.tt_log_probs.LogProbsCalculator._is_supported``)
    currently only handles 8- and 32-device meshes, so on a 4-chip mesh we
    skip rather than fail. Greedy / top-k / top-p coverage on 1x4 is still
    provided by the three other tests in this module.
    """
    num_devices = mesh_device.get_num_devices()
    if num_devices not in LOG_PROBS_SUPPORTED_DEVICE_COUNTS:
        pytest.skip(
            f"On-device log-probs (LogProbsCalculator) only supports "
            f"{LOG_PROBS_SUPPORTED_DEVICE_COUNTS} device meshes, got {num_devices}."
        )
    args = _make_sampling_args(mesh_device)

    torch.manual_seed(0)
    # Range-bounded random logits so softmax doesn't underflow on bf16. The
    # device-side log_probs path applies a multiply-mask (-inf * 0 → NaN on
    # bf16), so we use a large negative finite floor for the padding region
    # past VOCAB_SIZE — same trick as ``test_gpt_oss_logprobs``.
    logits_cpu = torch.randn(1, 1, BATCH_SIZE, args.padded_vocab_size, dtype=torch.bfloat16) * 2.0
    if args.padded_vocab_size > VOCAB_SIZE:
        logits_cpu[:, :, :, VOCAB_SIZE:] = -1e9

    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None)
    params = format_sampling_params(
        # temperature=0 → top_k clamped to 1 and temperature set to 1.0 by
        # format_sampling_params; this gives deterministic greedy sampling
        # with log-probs enabled.
        SamplingParams(temperature=0.0, top_k=1, top_p=0.0, enable_log_probs=True, seed=7),
        BATCH_SIZE,
    )
    sampling.reset_sampling_params(params)
    sampling.seed_manager.reset_seed([7] * BATCH_SIZE, list(range(BATCH_SIZE)))
    sampling.seed_manager.get_new_values()

    logits_tt = _shard_logits(logits_cpu, mesh_device)
    tt_tokens, tt_log_probs = sampling.sample(logits_tt, enable_trace=False)
    assert tt_log_probs is not None, "enable_log_probs=True but tt_log_probs is None"

    tokens = _extract_tokens(tt_tokens, BATCH_SIZE)
    device_log_probs = ttnn.to_torch(ttnn.get_device_tensors(tt_log_probs)[0]).float().reshape(-1)[:BATCH_SIZE]

    # Torch reference: log_softmax over the active vocab range, then gather
    # the device-sampled token. fp32 keeps numerical noise below the PCC bar.
    ref_logits = logits_cpu[0, 0, :, :VOCAB_SIZE].float()
    ref_log_softmax = torch.nn.functional.log_softmax(ref_logits, dim=-1)
    ref_log_probs = ref_log_softmax.gather(-1, torch.tensor(tokens).unsqueeze(-1)).squeeze(-1)

    passing, pcc_value = compare_tensors(
        device_log_probs, ref_log_probs, pcc_threshold=get_pcc_threshold(request, default=0.999)
    )
    assert passing, f"Sampling log-probs PCC below threshold: {pcc_value}"
