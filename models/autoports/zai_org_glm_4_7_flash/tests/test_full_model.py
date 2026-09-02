# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model tests for zai-org/GLM-4.7-Flash on one Blackhole p150-class chip.

Batch-1 suite over the complete 47-layer stack at the advertised 202752-token
context. One module-scoped generator (weights are 17.4 GiB, so exactly one full
model fits at a time); batch > 1 lives in ``test_full_model_batch.py``, which
must run in its own pytest session.

    pytest models/autoports/zai_org_glm_4_7_flash/tests/test_full_model.py -x -s

Covered: the context/capacity contract, non-aligned prompt lengths through the
public generator, AIME24 chat-template top-1/5/100 for prefill and traced
teacher-forced decode, greedy and logit determinism, the canonical
split-sampling trace contract (token feedback, on-device position advance,
changed/unchanged page tables), the runtime no-host-fallback tripwire, the
host-sampling compatibility mode, and top-k/top-p sampling through the same
path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import GREEDY, build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import source_manifest
from models.common.readiness_check.schema import load_reference
from models.common.sampling import SamplingParams

MODEL_DIR = Path(__file__).resolve().parents[1]
REFERENCE = MODEL_DIR / "readiness_aime24_chat.refpt"
TRACE_REGION_SIZE = 350_000_000
L1_SMALL_SIZE = 32768
#: Measured allocatable DRAM on this board (probe: 512 MiB chunks until OOM).
ALLOCATABLE_DRAM_BYTES = int(31.5 * 2**30)

TOP5_BAR = 0.98
TOP100_BAR = 1.0


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE
    )
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def gen(device):
    generator = build_generator(MODEL_DIR, device)
    yield generator
    generator.teardown()


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE.is_file():
        pytest.skip(f"missing readiness reference {REFERENCE}")
    return load_reference(REFERENCE)


# --------------------------------------------------------------------- capability contract


def test_readiness_reference_is_present():
    """The accuracy gate must not be able to pass by being skipped.

    ``reference`` is a skipping fixture, so a missing reference would drop both
    top-k tests and still report a green suite. This one fails instead.
    """
    assert REFERENCE.is_file(), f"missing readiness reference {REFERENCE}"
    meta = REFERENCE.with_suffix(".meta.json")
    assert meta.is_file(), f"missing reference metadata {meta}"


def test_context_and_capacity_contract(gen):
    """Supported context is the HF-advertised one, and weights + the full-context
    KV cache fit the measured allocatable DRAM."""
    model = gen.model
    assert model.max_seq_len == model.hf_max_position_embeddings == 202752
    assert len(model.layers) == model.hf_num_hidden_layers == 47
    assert [layer.layer_kind for layer in model.layers[:2]] == ["dense", "moe"]

    weights = model.weight_bytes()
    cache = model.kv_cache_bytes()
    total = weights["total"] + cache
    print(
        f"weights {weights['total'] / 2**30:.3f} GiB (layers {weights['layers'] / 2**30:.3f}, "
        f"embed {weights['embedding'] / 2**30:.3f}, lm_head {weights['lm_head'] / 2**30:.3f}, "
        f"rope {weights['rope'] / 2**30:.3f}) + cache {cache / 2**30:.3f} GiB "
        f"= {total / 2**30:.3f} GiB of {ALLOCATABLE_DRAM_BYTES / 2**30:.1f} GiB"
    )
    assert total < ALLOCATABLE_DRAM_BYTES, "full-context model does not fit measured DRAM"
    # the cache really is allocated at the advertised context
    assert model.blocks_per_user * model.paged_config.block_size >= model.max_seq_len


def test_deployment_dtype_policy_preserved(gen):
    """The optimized-decoder deployment policy reaches the built layers."""
    model = gen.model
    moe = next(layer for layer in model.layers if layer.layer_kind == "moe")
    dense = next(layer for layer in model.layers if layer.layer_kind == "dense")
    assert moe.experts_gate_up.dtype == ttnn.bfloat4_b
    assert moe.experts_down.dtype == ttnn.bfloat4_b
    assert moe.shared_gate.dtype == ttnn.bfloat4_b
    assert moe.shared_down_ds.dtype == ttnn.bfloat4_b
    assert dense.mlp_gate.dtype == ttnn.bfloat8_b  # bf4 dense MLP was measured and rejected
    assert dense.dense_down_ds.dtype == ttnn.bfloat8_b
    for layer in (moe, dense):
        for name in ("wqkv_a_ds", "wq_b_ds", "wo_ds", "w_uk", "w_uv_t"):
            assert getattr(layer, name).dtype == ttnn.bfloat4_b, name
        if layer.layer_kind == "moe":
            assert layer.gate_w.dtype == ttnn.float32
        assert layer.ck_attn.math_fidelity == ttnn.MathFidelity.LoFi
        assert layer.ck_expert.math_fidelity == ttnn.MathFidelity.LoFi
        # DRAM width-sharded decode weights, not interleaved fallbacks
        assert layer.wo_ds.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    for cache in gen._kv_cache:
        assert cache.dtype == ttnn.bfloat8_b
    assert model.lm_head_weight.dtype == ttnn.bfloat8_b
    # every layer shares layer 0's RoPE tables (2.4 GiB of duplicates otherwise)
    assert all(layer.rope is model.layers[0].rope for layer in model.layers)


@pytest.mark.parametrize("position", [0, 1, 31, 32, 33, 12345, 65535, 202751])
def test_shared_rope_matches_per_layer_lookup(gen, position):
    """The ROW_MAJOR shared decode RoPE lookup is byte-identical to the
    per-layer TILE-table lookup it replaces, at tile boundaries and at the
    last valid position."""
    model = gen.model
    idx = ttnn.from_torch(
        torch.tensor([[position]], dtype=torch.int32),
        device=model.mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    rope = model.shared_rope
    for rm, tile, name in (
        (model.rope_cos_rm, rope.cos_matrix, "cos"),
        (model.rope_sin_rm, rope.sin_matrix, "sin"),
    ):
        a = ttnn.to_torch(ttnn.embedding(idx, rm, layout=ttnn.TILE_LAYOUT)).float()
        b = ttnn.to_torch(ttnn.embedding(idx, tile, layout=ttnn.TILE_LAYOUT)).float()
        assert torch.equal(a, b), f"{name} table disagrees at position {position}"
    ttnn.deallocate(idx)


def test_decode_rope_index_derived_from_position(gen):
    """The RoPE index is derived from the current position on device: active
    slots track it exactly and an inactive (-1) slot is pinned at 0 rather than
    being incremented past the end of the cos/sin table."""
    import torch as _torch

    model = gen.model
    batch = gen.max_batch_size
    for positions in ([0], [7], [202751], [-1]):
        if len(positions) != batch:
            positions = (positions * batch)[:batch]
        pos = ttnn.from_torch(
            _torch.tensor(positions, dtype=_torch.int32),
            device=model.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        rot = ttnn.to_torch(model.decode_rope_indices(pos, batch)).reshape(-1).tolist()
        expected = [max(int(p), 0) for p in positions]
        assert rot == expected, (positions, rot, expected)
        ttnn.deallocate(pos)


# --------------------------------------------------------------------- prompt-length contract


@pytest.mark.parametrize("seq", [1, 17, 63, 65, 129, 154, 1057, 2049])
def test_non_aligned_prompt_lengths(gen, seq):
    """Any logical prompt length is accepted: not a multiple of the tile (32),
    the paged block (64), the prefill bucket, or the prefill chunk (2048)."""
    ids = _prompt_ids(gen, seq)
    logits = gen.prefill_logits(ids)
    assert logits.shape == (1, seq, gen.model.vocab_size), logits.shape
    assert torch.isfinite(logits).all()
    phys = gen.model.prefill_physical_len(seq)
    print(f"seq={seq} physical={phys} logits={tuple(logits.shape)}")


def test_prefill_last_row_agrees_with_all_logits(gen):
    """The generate() path's device-side last-position logits and the host
    all-positions path pick the same token."""
    seq = 154
    ids = _prompt_ids(gen, seq)
    allp = gen.prefill_logits(ids)
    gen.reset()
    last_only = gen.model.prefill_forward(ids, kv_cache=gen._kv_cache, page_table=gen._page_table_dev, seq_len=seq)
    # same terminal program as the device path (single 32-row tile, decode config)
    host_top1 = int(last_only[0, 0].argmax())
    gen.reset()
    device_token = gen._prefill_and_sample_first(ids)
    assert device_token == host_top1, (device_token, host_top1)
    # the chunked all-positions path must agree on the same row within top-5
    all_top5 = allp[0, -1].topk(5).indices.tolist()
    assert device_token in all_top5, (device_token, all_top5)


def test_prompt_longer_than_one_prefill_chunk(gen):
    """A prompt past the 2048-token prefill chunk still returns logical logits."""
    seq = 2600
    ids = _prompt_ids(gen, seq)
    logits = gen.prefill_logits(ids)
    assert logits.shape == (1, seq, gen.model.vocab_size)
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------- accuracy vs HF reference


def test_prefill_topk_vs_reference(gen, reference):
    entry = reference.entries[0]
    prompt = entry.prompt_tokens[0].tolist()
    gen_tokens = entry.generated_tokens[0].tolist()
    logits = gen.prefill_logits(prompt + gen_tokens)
    stats = _topk_stats(logits[0, len(prompt) - 1 : len(prompt) + len(gen_tokens) - 1], entry.topk_tokens)
    print("prefill", stats)
    _write_stats("prefill", stats)
    assert stats["top5"] >= TOP5_BAR
    assert stats["top100"] >= TOP100_BAR


def test_teacher_forcing_topk_vs_reference(gen, reference):
    entry = reference.entries[0]
    prompt = entry.prompt_tokens[0].tolist()
    gt = entry.generated_tokens[0].tolist()
    gen.reset()
    preds = gen.generate(
        prompt_token_ids=prompt,
        max_new_tokens=len(gt),
        next_input=lambda step, predicted: gt[step],
        enable_trace=True,
    )
    assert len(preds) == len(gt)
    stats = _topk_stats_from_tokens(preds, entry.topk_tokens)
    print("teacher-forced decode", stats)
    _write_stats("teacher_forcing", stats)
    assert stats["top5"] >= TOP5_BAR
    assert stats["top100"] >= TOP100_BAR


# --------------------------------------------------------------------- determinism


def test_greedy_generation_deterministic(gen):
    ids = _prompt_ids(gen, 96)
    gen.reset()
    a = gen.generate(ids, 24, enable_trace=True, stop_on_eos=False)
    gen.reset()
    b = gen.generate(ids, 24, enable_trace=True, stop_on_eos=False)
    assert a == b, (a, b)


def test_traced_decode_matches_eager_decode(gen):
    """The captured traces produce exactly the eager result.

    This is the control for the one warning trace capture emits
    ("Allocating device buffers is potentially unsafe due to the existence of
    an active trace", raised while the *sampling* trace is captured with the
    model trace already recorded): if any buffer the sampling trace owns were
    clobbered by a model-trace replay, the traced and eager token streams would
    part company.
    """
    ids = _prompt_ids(gen, 88)
    gen.reset()
    traced = gen.generate(ids, 16, enable_trace=True, stop_on_eos=False)
    gen.reset()
    eager = gen.generate(ids, 16, enable_trace=False, stop_on_eos=False)
    assert gen.counters["eager_decode_steps"] >= 15
    assert traced == eager, (traced, eager)


def test_prefill_logits_bitwise_reproducible(gen):
    ids = _prompt_ids(gen, 129)
    a = gen.prefill_logits(ids)
    b = gen.prefill_logits(ids)
    assert torch.equal(a, b), "prefill logits are not reproducible across runs"


def test_decode_logits_independent_of_cache_history(gen):
    """The same (token, position, cache prefix) yields the same logits after a
    reset: no stale decode state leaks between prompts."""
    ids = _prompt_ids(gen, 64)
    gen.reset()
    gen._prefill_and_sample_first(ids)
    gen.set_decode_positions([len(ids)])
    gen.decode_step_traced()
    first = gen.read_decode_tokens(1)[0]
    # run an unrelated prompt in between, then repeat
    gen.reset()
    gen.generate(_prompt_ids(gen, 40), 8, enable_trace=True, stop_on_eos=False)
    gen.reset()
    gen._prefill_and_sample_first(ids)
    gen.set_decode_positions([len(ids)])
    gen.decode_step_traced()
    assert gen.read_decode_tokens(1)[0] == first


# --------------------------------------------------------------------- split-sampling trace contract


def test_split_sampling_trace_feedback(gen):
    """Two+ traced decode steps: the sampled token of step N is the token input
    of step N+1 with no host reconstruction, and the current-position and RoPE
    tensors advance on device."""
    ids = _prompt_ids(gen, 80)
    gen.reset()
    first = gen._prefill_and_sample_first(ids)
    gen.set_decode_positions([len(ids)])
    gen.reset_counters()

    seen_inputs = []
    seen_positions = []
    outputs = []
    expected_input = first
    for step in range(4):
        token_in = int(ttnn.to_torch(gen._tokens_dev).reshape(-1)[0].item())
        pos_in = int(ttnn.to_torch(gen._pos_dev).reshape(-1)[0].item())
        rot_in = int(
            ttnn.to_torch(gen.model.decode_rope_indices(gen._pos_dev, gen.max_batch_size)).reshape(-1)[0].item()
        )
        seen_inputs.append(token_in)
        seen_positions.append(pos_in)
        assert token_in == expected_input, f"step {step}: token input {token_in} != sampled {expected_input}"
        assert pos_in == len(ids) + step, f"step {step}: position {pos_in}"
        assert rot_in == pos_in, "rope index diverged from the current position"
        gen.decode_step_traced()
        expected_input = gen.read_decode_tokens(1)[0]
        outputs.append(expected_input)

    print("trace feedback inputs:", seen_inputs, "positions:", seen_positions, "outputs:", outputs)
    assert len(set(seen_inputs)) > 1 or len(set(outputs)) > 1, "decode never changed its inputs"
    counters = gen.counters
    assert counters["model_trace_replays"] == 4
    assert counters["sampling_trace_replays"] == 4
    assert counters["eager_decode_steps"] == 0, "enable_trace did not take the traced path"
    assert counters["token_input_refreshes"] == 0, counters
    assert counters["position_refreshes"] == 0, counters
    assert counters["rope_index_refreshes"] == 0, counters
    assert counters["page_table_refreshes"] == 0, counters
    assert counters["full_logits_readbacks"] == 0, counters
    assert counters["host_argmax_calls"] == 0, counters


def test_unchanged_and_changed_page_table(gen):
    """The steady-state loop copies no page table; an explicit page-table change
    is picked up by the same captured trace."""
    ids = _prompt_ids(gen, 64)
    gen.reset()
    gen._prefill_and_sample_first(ids)
    gen.set_decode_positions([len(ids)])
    gen.reset_counters()
    for _ in range(3):
        gen.decode_step_traced()
        gen.read_decode_tokens(1)
    assert gen.counters["page_table_refreshes"] == 0

    # remap this user's blocks, refill the cache through the new table, and show
    # the same captured trace follows the new mapping.
    identity = gen.model.default_page_table()
    gen.refresh_page_table(identity)
    baseline = _decode_after_prefill(gen, ids)
    permuted = _permuted_page_table(gen.model)
    gen.refresh_page_table(permuted)
    remapped = _decode_after_prefill(gen, ids)
    gen.refresh_page_table(identity)
    assert gen.counters["page_table_refreshes"] == 3
    assert remapped == baseline, "decode changed when only the physical block mapping changed"


def test_no_host_fallback_during_traced_decode(gen, monkeypatch):
    """No host<->device traffic inside a traced token-out decode step."""
    ids = _prompt_ids(gen, 48)
    gen.reset()
    gen._prefill_and_sample_first(ids)
    gen.set_decode_positions([len(ids)])

    calls = []
    for name in ("from_torch", "to_torch", "as_tensor", "copy_host_to_device_tensor"):
        original = getattr(ttnn, name)

        def tripwire(*args, _name=name, _orig=original, **kwargs):
            calls.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(ttnn, name, tripwire)

    gen.decode_step_traced()
    assert calls == [], f"host fallback inside traced decode: {calls}"
    gen.read_decode_tokens(1)
    assert calls == ["to_torch"], f"token readback should be the only host touch, got {calls}"


def test_static_no_torch_in_runtime_modules():
    """model.py / generator.py import torch only inside functions, so the TTNN
    runtime path cannot silently fall back to host tensors."""
    import ast

    for name in ("model.py", "generator.py"):
        tree = ast.parse((MODEL_DIR / "tt" / name).read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(a.name.split(".")[0] != "torch" for a in node.names), f"{name}: module-level import torch"
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "torch", f"{name}: module-level from torch import"


# --------------------------------------------------------------------- sampling modes


def test_host_sampling_compatibility_mode(gen):
    """The explicit host-sampling mode selects the same greedy tokens as the
    on-device traced sampler, and is not what the measured path uses."""
    ids = _prompt_ids(gen, 72)
    gen.reset()
    device_tokens = gen.generate(ids, 12, enable_trace=True, stop_on_eos=False)
    gen.reset()
    gen.reset_counters()
    host_tokens = gen.generate(ids, 12, enable_trace=True, stop_on_eos=False, host_sampling=True)
    assert gen.counters["host_argmax_calls"] > 0
    assert gen.counters["full_logits_readbacks"] > 0
    assert host_tokens == device_tokens, (host_tokens, device_tokens)
    assert gen.host_sampling is False, "host sampling leaked out of the compatibility call"


def test_topk_topp_sampling_runs_and_greedy_still_works(gen):
    """The same split-sampling path serves top-k/top-p, and switching back to
    greedy still produces the greedy token."""
    ids = _prompt_ids(gen, 64)
    gen.reset()
    greedy_tokens = gen.generate(ids, 8, enable_trace=True, stop_on_eos=False)

    sampled_params = SamplingParams(temperature=0.8, top_k=20, top_p=0.9, seed=None)
    gen.reset()
    sampled = gen.generate(ids, 8, enable_trace=True, stop_on_eos=False, sampling_params=sampled_params)
    assert len(sampled) == 8
    assert all(0 <= t < gen.model.vocab_size for t in sampled)

    gen.set_sampling_params(GREEDY)
    gen.reset()
    again = gen.generate(ids, 8, enable_trace=True, stop_on_eos=False)
    assert again == greedy_tokens, "greedy decode changed after a sampled request"


# --------------------------------------------------------------------- low-level serving API


def test_low_level_prefill_decode_api(gen):
    """The caller-owned cache/page-table API: mixed prompt lengths in a fixed
    slot layout, explicit positions, logits out."""
    model = gen.model
    kv_cache = gen._kv_cache
    page_table = gen._page_table_torch
    gen.reset()
    prompt_lens = [40]
    tokens = torch.tensor([_prompt_ids(gen, 40)], dtype=torch.int32)
    logits = gen.prefill_forward(tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens)
    assert logits.shape == (1, 1, model.vocab_size)
    next_token = int(logits[0, 0].argmax())

    step_logits = gen.decode_forward(
        torch.tensor([[next_token]]),
        torch.tensor([40]),
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
    )
    assert step_logits.shape == (1, model.vocab_size)
    assert torch.isfinite(step_logits).all()

    all_logits = gen.prefill_forward(
        tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens, return_all_logits=True
    )
    assert all_logits.shape == (1, 40, model.vocab_size)


def test_reset_clears_cache_and_state(gen):
    a_ids = _prompt_ids(gen, 32)
    gen.reset()
    a = gen.generate(a_ids, 8, enable_trace=True, stop_on_eos=False)
    gen.reset()
    _ = gen.generate(_prompt_ids(gen, 96), 8, enable_trace=True, stop_on_eos=False)
    gen.reset()
    assert gen.generate(a_ids, 8, enable_trace=True, stop_on_eos=False) == a


def test_reset_zeroes_cache_rows_the_request_never_wrote(gen):
    """``reset()`` really zeroes, including the rows attention never reads.

    This is the observable form of a real hazard the trace allocation tracker
    caught (work log FM-016): the shared cache-reset zero *source* used to be
    allocated lazily on the first reset, i.e. after the decode traces were
    captured, which Metal flags as an unsafe allocation because a replay can
    write over a post-capture buffer. A corrupted zero source would leave
    garbage in the cache instead of zeros, and no existing test could see it,
    because attention only ever reads rows that prefill or decode has already
    written. Reading a far, untouched row after a traced generation plus a
    reset is what makes it visible.
    """
    model = gen.model
    gen.reset()
    gen.generate(_prompt_ids(gen, 64), 8, enable_trace=True, stop_on_eos=False)
    gen.reset()
    cache = gen._kv_cache[0]
    blocks = int(cache.shape[0])
    # A block no request in this test came near, and the last one.
    for block in (blocks // 2, blocks - 1):
        row = ttnn.slice(
            cache, [block, 0, 0, 0], [block + 1, 1, int(cache.shape[2]), int(cache.shape[3])], [1, 1, 1, 1]
        )
        host = ttnn.to_torch(row).to(torch.float32)
        ttnn.deallocate(row)
        assert torch.count_nonzero(host) == 0, (block, float(host.abs().max()))
    assert model is gen.model


def test_single_chunk_prompt_shape_does_not_recapture(gen):
    """No first-use cost for any prompt inside one prefill chunk.

    The terminal path slices the 32-row tile holding the last prompt position,
    so its program is keyed by that tile offset, and
    ``GLM47FlashModel.warmup_terminal_shapes`` compiles every offset for the
    five buckets at construction, before the decode traces are captured. A
    length that is not itself a bucket must therefore still compile nothing
    and recapture nothing (work log FM-017).
    """
    seq = 173  # 173 % 32 = 13, not a bucket, but its terminal tile offset is warmed
    ids = _prompt_ids(gen, seq)
    # Absorb anything an earlier test in this module compiled (several read the
    # cache back through fresh `ttnn.slice` shapes), so the assertion below is
    # about this prompt and not about test order.
    gen._maybe_recapture_after_compile()
    gen.reset()
    gen.reset_counters()
    entries = gen.mesh_device.num_program_cache_entries()
    out = gen.generate(ids, 6, enable_trace=True, stop_on_eos=False)
    assert (
        gen.mesh_device.num_program_cache_entries() == entries
    ), "a prompt inside one chunk must compile no new program"
    assert gen.counters["trace_recaptures"] == 0, "a warmed terminal offset must not recapture"
    assert len(out) == 6
    assert gen.counters["full_logits_readbacks"] == 0
    assert gen.counters["host_argmax_calls"] == 0
    assert gen.counters["model_trace_replays"] == 5


def test_host_logits_paths_compile_nothing_at_an_unaligned_length(gen):
    """The two *host*-logits entry points must be length-independent too.

    `prefill_logits` (the readiness accuracy gate) and the low-level
    `prefill_forward` (what a vLLM adapter drives) go through
    `_logits_host_rows`, which used to slice `[s0, seq)`, pad a non-tile tail
    and then slice the wanted rows on device: three programs keyed on the
    logical prompt length, compiled while the decode traces were live, and a
    trace recapture to clean up after them. The walk is now tile-aligned at
    both ends and the single-row slice has a fixed size, so both families are
    warmed at construction (work log FM-018).
    """
    import torch as _torch

    seq = 251  # 251 % 32 = 27: neither a bucket, nor a tile multiple, nor used elsewhere
    ids = _prompt_ids(gen, seq)
    gen._maybe_recapture_after_compile()  # absorb anything an earlier test compiled
    gen.reset()
    gen.reset_counters()
    entries = gen.mesh_device.num_program_cache_entries()

    logits = gen.prefill_logits(ids)
    assert logits.shape == (1, seq, gen.model.vocab_size), logits.shape
    assert _torch.isfinite(logits).all()
    assert gen.mesh_device.num_program_cache_entries() == entries, "prefill_logits compiled a new program"
    assert gen.counters["trace_recaptures"] == 0

    low = gen.prefill_forward(
        _torch.tensor([ids], dtype=_torch.int32),
        page_table=gen._page_table_torch,
        kv_cache=gen._kv_cache,
        prompt_lens=[seq],
    )
    assert low.shape == (1, 1, gen.model.vocab_size), low.shape
    assert gen.mesh_device.num_program_cache_entries() == entries, "low-level prefill_forward compiled a new program"
    assert gen.counters["trace_recaptures"] == 0
    # the two paths agree on the final position
    assert int(low[0, 0].argmax()) == int(logits[0, -1].argmax())


def test_low_level_prefill_rejects_the_old_slot_kwarg(gen, expect_error):
    """``user_id=`` used to vanish into ``**kwargs`` and prefill slot 0."""
    import torch as _torch

    ids = _prompt_ids(gen, 40)
    with expect_error(TypeError, "user_ids"):
        gen.prefill_forward(
            _torch.tensor([ids], dtype=_torch.int32),
            page_table=gen._page_table_torch,
            kv_cache=gen._kv_cache,
            prompt_lens=[40],
            user_id=0,
        )


def test_first_use_multi_chunk_shape_recaptures_and_stays_correct(gen):
    """A first-use *multi-chunk* prompt does still recapture, and correctly.

    The chunk-offset-dependent programs (the RoPE-table ``ttnn.slice`` offsets
    are compile-time constants) cannot be enumerated cheaply at 99 offsets, so
    they compile on first use, while the decode traces are live. What has to
    hold is that the recapture is (a) triggered and (b) invisible in the
    output: the same prompt again, everything now cached, must give identical
    tokens from the newly captured traces.
    """
    seq = 4300  # two whole 2048 chunks plus a 256 bucket tail; used nowhere else
    ids = _prompt_ids(gen, seq)
    gen._maybe_recapture_after_compile()
    gen.reset()
    gen.reset_counters()
    first = gen.generate(ids, 4, enable_trace=True, stop_on_eos=False)
    assert gen.counters["trace_recaptures"] >= 1, "a first-use chunk offset must recapture before replaying"

    gen.reset()
    gen.reset_counters()
    again = gen.generate(ids, 4, enable_trace=True, stop_on_eos=False)
    assert again == first, (first, again)
    assert gen.counters["trace_recaptures"] == 0, "the second request at a cached shape must not recapture"
    assert gen.counters["full_logits_readbacks"] == 0
    assert gen.counters["host_argmax_calls"] == 0


def test_sampling_trace_is_captured_on_demand_if_capture_skipped_it(gen):
    """A generator whose first capture happened in host-sampling mode must not
    keep sampling untraced forever.

    ``capture_decode_trace`` skips the sampling capture when
    ``host_sampling`` is set, and nothing captured it afterwards, so
    ``build_generator(host_sampling=True)`` followed by on-device sampling ran
    the sampler untraced for the rest of the process: right tokens, silently
    slower, no error. A second full model does not fit alongside this one, so
    the condition is reproduced on this generator by releasing the sampling
    trace and clearing the flag, which is exactly the state that path leaves.
    Leaves the generator with both traces captured, as it found it.
    """
    ids = _prompt_ids(gen, 64)
    gen.reset()
    expected = gen.generate(ids, 4, enable_trace=True, stop_on_eos=False)

    gen.sampling.reset_trace()
    gen._sampling_traced = False
    gen.reset()
    gen.reset_counters()
    got = gen.generate(ids, 4, enable_trace=True, stop_on_eos=False)

    assert gen._sampling_traced, "on-device sampling must not stay untraced"
    assert gen.counters["trace_recaptures"] == 1
    assert got == expected, (expected, got)
    assert gen.counters["full_logits_readbacks"] == 0
    assert gen.counters["host_argmax_calls"] == 0


def test_decode_positions_reject_more_rows_than_slots(gen, expect_error):
    """Too many positions is an error. The *padding* branch needs batch > 1 to
    mean anything, so it is covered in ``test_full_model_batch.py``
    (`test_batch_decode_positions_pad_inactive_slots`)."""
    slots = gen.max_batch_size
    with expect_error(ValueError, "at most"):
        gen.set_decode_positions([0] * (slots + 1))
    gen.reset()


# --------------------------------------------------------------------- helpers


def _prompt_ids(gen, seq):
    """Deterministic in-vocabulary prompt of the requested length."""
    text = (
        "Tenstorrent builds AI accelerators. "
        "This paragraph exists so the tokenizer produces a long, ordinary, in-distribution prompt "
        "for the full-model prompt-length and determinism tests. "
    ) * 200
    ids = gen.tokenizer.encode(text, add_special_tokens=True)
    while len(ids) < seq:
        ids = ids + ids
    return ids[:seq]


def _topk_stats(prediction_logits, topk_reference):
    preds = torch.argmax(prediction_logits, dim=-1).tolist()
    return _topk_stats_from_tokens(preds, topk_reference)


def _topk_stats_from_tokens(preds, topk_reference):
    total = len(preds)
    m1 = m5 = mk = 0
    for i, pred in enumerate(preds):
        row = topk_reference[i].tolist()
        m1 += int(pred == row[0])
        m5 += int(pred in row[:5])
        mk += int(pred in row)
    return {
        "top1": m1 / total,
        "top5": m5 / total,
        "top100": mk / total,
        "matches_top1": m1,
        "matches_top5": m5,
        "matches_top100": mk,
        "total": total,
    }


def _write_stats(name, stats):
    out = MODEL_DIR / "doc" / "full_model" / "accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.is_file() else {}
    data[name] = stats
    # Both halves are written by separate tests through read-modify-write, so
    # the manifest is refreshed on every write and a mixed-session file would
    # show it (FM-016).
    data["source_manifest"] = source_manifest([__file__])
    out.write_text(json.dumps(data, indent=2) + "\n")


def _permuted_page_table(model):
    gen_t = torch.Generator().manual_seed(7)
    total = model.paged_config.max_num_blocks
    return torch.randperm(total, generator=gen_t, dtype=torch.int32).reshape(model.max_batch_size, -1)


def _decode_after_prefill(gen, ids):
    """Prefill and take 3 traced decode steps through the current page table."""
    gen.reset()
    gen._prefill_and_sample_first(ids)
    gen.set_decode_positions([len(ids)])
    out = []
    for _ in range(3):
        gen.decode_step_traced()
        out.append(gen.read_decode_tokens(1)[0])
    return out
