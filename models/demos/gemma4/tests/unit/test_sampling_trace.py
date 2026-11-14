# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GEMMA4_SAMPLING_TRACE (item 3c of the gemma4_12b_qb2 port): the device sampler's replayed trace draws the same
tokens as the eager sampler.

Greedy over several steps of fresh logits (the force-argmax gather and the top-k form; #48037 corrupted from the
second replay), a fixed request seed (eager by contract: the seed tensor is rewritten every token, so the trace is
never taken), and an unseeded sampled request whose replays stay inside the top-k set and keep drawing (the device
PRNG advances under replay).

    pytest -k "1x4" models/demos/gemma4/tests/unit/test_sampling_trace.py
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.demos.gemma4.tt.generator import Gemma4Generator
from models.demos.gemma4.tt.model import SAMPLING_TRACE_ENV, _compute_per_device_vocab, sampling_trace_default
from models.tt_transformers.tt.ccl import TT_CCL

from ...config import MeshConfig, ModeConfig
from ...tests.test_factory import TestFactory, _get_model_path, parametrize_mesh_with_fabric
from .test_prefill_trace import _allocate_fresh_kv_cache, _create_page_table, _page_params

VOCAB = 262144
ROWS = 32
STEPS = 4


def _args(mesh_device, *, allow_force_argmax):
    tp = mesh_device.shape[1]
    per_device = _compute_per_device_vocab(VOCAB, tp)
    args = SimpleNamespace()
    args.vocab_size = VOCAB
    args.padded_vocab_size = per_device * tp
    args.cluster_shape = tuple(mesh_device.shape)
    args.sampling_all_gather_axis = 1
    args.sampling_dp = 1
    args.num_devices = mesh_device.get_num_devices()
    args.is_galaxy = mesh_device.shape[0] > 1
    args.model_config = (
        {"SAMPLING_AG_CONFIG": {"allow_force_argmax": True, "num_links": 1, "topology": ttnn.Topology.Linear}}
        if allow_force_argmax
        else {}
    )
    args.max_top_k = 32
    args.sub_core_grids = None
    args.use_topk_logprobs = False
    args.salt_duplicate_seeds = False
    return args


def _mapper(mesh_device):
    return MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1])).column_parallel(mesh_device)


def _tokens(tt_tokens):
    return ttnn.to_torch(ttnn.get_device_tensors(tt_tokens)[0]).reshape(-1)[:ROWS].to(torch.int64)


def _greedy_logits(padded, step):
    """Row u's winner at shard (u + step) % 4 with a step-dependent offset: every step and row differs."""
    logits = torch.zeros(1, 1, ROWS, padded, dtype=torch.bfloat16)
    per_device = padded // 4
    winners = []
    for u in range(ROWS):
        winner = ((u + step) % 4) * per_device + 1000 + 7 * u + 13 * step
        winners.append(winner)
        logits[0, 0, u, winner] = 50.0
    if padded > VOCAB:
        logits[:, :, :, VOCAB:] = float("-inf")
    return logits, winners


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
@pytest.mark.parametrize("form", ["force_argmax", "topk"])
def test_sampling_trace_greedy_matches_eager(mesh_device, device_params, reset_seeds, form):
    """Greedy, STEPS steps of fresh logits in the captured buffer: replay == eager == the host argmax."""
    args = _args(mesh_device, allow_force_argmax=form == "force_argmax")
    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=TT_CCL(mesh_device))
    sampling.reset_sampling_params(format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), ROWS))
    assert sampling.tt_sampling.force_argmax_sampling == (form == "force_argmax")
    sampling.seed_manager.reset_seed(None, list(range(ROWS)))
    mapper = _mapper(mesh_device)
    first, _ = _greedy_logits(args.padded_vocab_size, 0)
    logits_tt = ttnn.from_torch(
        first, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper
    )
    sampling.set_trace_bucket(ROWS)
    for step in range(STEPS):
        logits_cpu, winners = _greedy_logits(args.padded_vocab_size, step)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(logits_cpu, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper), logits_tt
        )
        sampling.seed_manager.get_new_values()
        eager = _tokens(sampling.sample(logits_tt, enable_trace=False)[0])
        traced = _tokens(sampling.sample(logits_tt, enable_trace=True)[0])
        assert eager.tolist() == winners, f"step {step}: eager greedy != host argmax"
        assert traced.tolist() == winners, f"step {step}: replayed greedy != host argmax ({traced.tolist()[:4]}...)"
    assert any(slot["id"] is not None for slot in sampling._trace_states.values()), "no sampling trace was captured"
    sampling.reset_trace()


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_sampling_trace_seeded_and_unseeded(mesh_device, device_params, reset_seeds):
    """A fixed seed: trace call == eager call (both eager, no trace captured); unseeded top-k: replays stay in the
    top-k set and the draws keep changing."""
    args = _args(mesh_device, allow_force_argmax=True)
    sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=TT_CCL(mesh_device))
    top_k, num_hot = 4, 8
    per_device = args.padded_vocab_size // 4
    hot = [(i % 4) * per_device + 100 + i for i in range(num_hot)]
    logits_cpu = torch.full((1, 1, ROWS, args.padded_vocab_size), 0.001, dtype=torch.bfloat16)
    for i, tok in enumerate(hot):
        logits_cpu[:, :, :, tok] = 10.0 - 0.25 * i
    if args.padded_vocab_size > VOCAB:
        logits_cpu[:, :, :, VOCAB:] = float("-inf")
    logits_tt = ttnn.from_torch(
        logits_cpu, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=_mapper(mesh_device)
    )
    sampling.set_trace_bucket(ROWS)
    sampling.reset_sampling_params(
        format_sampling_params(SamplingParams(temperature=1.0, top_k=top_k, top_p=1.0, seed=42), ROWS)
    )
    # fixed seed: the same seed advance before each call draws the same token eagerly and "traced"
    for _ in range(3):
        sampling.seed_manager.reset_seed([42] * ROWS, list(range(ROWS)))
        sampling.seed_manager.get_new_values()
        eager = _tokens(sampling.sample(logits_tt, enable_trace=False)[0])
        sampling.seed_manager.reset_seed([42] * ROWS, list(range(ROWS)))
        sampling.seed_manager.get_new_values()
        traced = _tokens(sampling.sample(logits_tt, enable_trace=True)[0])
        assert torch.equal(eager, traced)
        assert set(eager.tolist()) <= set(hot[:top_k])
    assert all(slot["id"] is None for slot in sampling._trace_states.values()), "a seeded request took the trace"
    # unseeded: device-advanced randomness under replay
    sampling.seed_manager.reset_seed(None, list(range(ROWS)))
    draws = []
    for _ in range(8):
        sampling.seed_manager.get_new_values()
        draws.append(_tokens(sampling.sample(logits_tt, enable_trace=True)[0]))
    assert any(slot["id"] is not None for slot in sampling._trace_states.values()), "no sampling trace was captured"
    for d in draws:
        assert set(d.tolist()) <= set(hot[:top_k]), f"replay drew outside the top-{top_k} set: {d.tolist()}"
    assert len({tuple(d.tolist()) for d in draws}) > 1, "every replay drew the same tokens (PRNG did not advance)"
    sampling.reset_trace()


@pytest.mark.gemma4_prefill_trace  # loads the checkpoint: skipped under a config-only HF_MODEL (tests/conftest.py)
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_sampling_trace_buckets_through_generator(mesh_device, device_params, reset_seeds, monkeypatch):
    """GEMMA4_SAMPLING_TRACE=1 through the generator: the 1-row and 32-row decode buckets each capture their own
    sampling trace on first use, alternating between the buckets replays each against its own logits and feedback
    buffers (every replay's tokens == the host argmax of the same step's logits), a release of the decode traces after
    the warmup is declared complete re-captures the sampling traces from ``_g4_capture_sampling_traces`` at the next
    decode capture, and the penalties / grammar keys keep the eager sampler."""
    monkeypatch.setenv(SAMPLING_TRACE_ENV, "1")
    hf_config = TestFactory.create_hf_config()
    if int(getattr(hf_config, "hidden_size_per_layer_input", 0) or 0) > 0:
        pytest.skip("PLI models restage their decode inputs from the host and keep the eager sampler")
    batch, max_new = 32, 16
    max_seq_len = 4096
    paged_cfg = _page_params(batch, 128, max_new)
    page_tables = {batch: _create_page_table(batch, paged_cfg)}
    page_tables[1] = page_tables[batch][:1].clone()
    generator, _kv, _tok = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=_get_model_path(),
        max_batch_size=batch,
        max_seq_len=max_seq_len,
        paged_attention_config=paged_cfg,
        bounded_sliding_kv_cache=False,
    )
    model = generator.model[0]
    if model.sampling is None:
        pytest.skip("no device sampler on this mesh")
    assert not model._tt_disable_sampling_trace, "GEMMA4_SAMPLING_TRACE=1 did not enable the replayed sampling trace"
    kv = [
        _allocate_fresh_kv_cache(model, max_batch_size=batch, max_seq_len=max_seq_len, paged_attention_config=paged_cfg)
    ]
    greedy = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    gen = torch.Generator().manual_seed(7)
    vocab = int(hf_config.vocab_size)

    def step(rows, pos):
        """One decode step of ``rows`` users at ``pos``: the host argmax of the logits (host-sampled key) and the
        device sampler's tokens (its traced key) for the same inputs; the K/V write of the step is idempotent."""
        tokens = torch.randint(10, vocab - 10, (rows, 1), generator=gen)
        start = torch.full((rows,), pos, dtype=torch.long)
        common = dict(
            page_table=page_tables[rows],
            kv_cache=kv,
            enable_trace=True,
            read_from_device=True,
            reload_inputs=True,
            reload_page_table=True,
            reload_sampling_params=True,
            reset_sampling_state=True,
        )
        logits = generator.decode_forward(tokens, start, sampling_params=None, **common)
        logits = logits[0] if isinstance(logits, (list, tuple)) else logits
        expect = torch.as_tensor(logits).float().reshape(rows, -1)[:, :vocab].argmax(-1)
        out = generator.decode_forward(tokens, start, sampling_params=greedy, **common)
        out = out[0] if isinstance(out, (list, tuple)) else out
        sampled = torch.as_tensor(out).reshape(-1)[:rows].to(torch.int64)
        return sampled, expect

    def slot_id(bucket):
        sampling = model.sampling
        saved = sampling._active_trace_bucket
        sampling.set_trace_bucket(bucket)
        try:
            _, slot = sampling._trace_slot(False, False, bool(sampling.tt_sampling.force_argmax_sampling), False)
            return slot["id"]
        finally:
            sampling.set_trace_bucket(saved)

    # each bucket captures its sampling trace on first use (the warmup is open: the shared lazy capture), and
    # alternating between the buckets replays each one against its own logits buffer
    for rows, pos in ((batch, 0), (1, 1), (batch, 2), (1, 3), (batch, 4)):
        sampled, expect = step(rows, pos)
        assert torch.equal(sampled, expect), f"bucket {rows} @ {pos}: replayed tokens != host argmax"
        assert slot_id(rows) is not None, f"bucket {rows}: no sampling trace captured"
    assert slot_id(batch) is not None and slot_id(1) is not None
    # a release of the decode traces (the allocation guard, a page-table realloc) drops the sampling traces; once the
    # warmup is complete the next decode capture re-binds each bucket's trace from _g4_capture_sampling_traces
    generator._decode_warmup_complete = True
    generator._g4_release_decode_traces("test: re-capture after the warmup")
    assert slot_id(batch) is None and slot_id(1) is None, "the release should drop the sampling traces"
    for rows, pos in ((batch, 5), (1, 6), (batch, 7), (1, 8)):
        sampled, expect = step(rows, pos)
        assert torch.equal(sampled, expect), f"bucket {rows} @ {pos} after the re-capture: tokens != host argmax"
        assert slot_id(rows) is not None, f"bucket {rows}: not re-captured after the decode-trace release"
    # keys outside the plain ones keep the eager sampler, whatever the bucket's trace state
    sampling = model.sampling
    sampling._penalties_active = True
    try:
        assert not generator._g4_sampling_trace_replay_ok(sampling, None)
    finally:
        sampling._penalties_active = False
    assert not generator._g4_sampling_trace_replay_ok(sampling, object())  # a grammar mask
    assert generator._g4_sampling_trace_replay_ok(sampling, None)
    generator._g4_release_decode_traces("test: done")


def test_sampling_trace_default_policy(monkeypatch):
    """Host: the env var overrides, else the measured-shape policy; never on lanes / fractured meshes."""
    mesh = SimpleNamespace(shape=(1, 4))
    monkeypatch.setenv(SAMPLING_TRACE_ENV, "1")
    assert sampling_trace_default(mesh) is True
    assert sampling_trace_default(mesh, SimpleNamespace(lane_sharded=True, weight_fracture=False)) is False
    assert sampling_trace_default(mesh, SimpleNamespace(lane_sharded=False, weight_fracture=True)) is False
    assert sampling_trace_default(mesh, SimpleNamespace(lane_sharded=False, weight_fracture=False)) is True
    monkeypatch.setenv(SAMPLING_TRACE_ENV, "0")
    assert sampling_trace_default(mesh) is False
    monkeypatch.delenv(SAMPLING_TRACE_ENV)
    from models.demos.gemma4.tt import model as model_module

    monkeypatch.setattr(model_module, "SAMPLING_TRACE_DEFAULT_SHAPES", ((1, 4),))
    monkeypatch.setattr(model_module, "is_blackhole", lambda: True)
    assert sampling_trace_default(mesh) is True
    assert sampling_trace_default(SimpleNamespace(shape=(1, 8))) is False
    monkeypatch.setattr(model_module, "is_blackhole", lambda: False)  # Wormhole: never by default
    assert sampling_trace_default(mesh) is False
