# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched-prefill helpers (tt/batched_prefill.py) vs the batch-1 ops they stand in for, on a 1x4 Blackhole mesh.

The batched prefill must give every user the rows of its own batch-1 prefill, bit for bit (the batch-32 gate,
tests/perf/prefill_gate_b32.py, compares KV pages at PCC >= 0.999, which one-ulp CCL differences at layer 0 already
fail by layer 23). Checked here on random partial sums / rows:
  * per_user_reduce of B users' [1, 1, B*S, 2880] partial sums vs ttnn.all_reduce (and, scatter, ttnn.reduce_scatter)
    of each user's [1, 1, S, 2880] alone, on both sides of the direct reduce-scatter size gate (bf8 S = 128 takes the
    direct op, S = 256 the ring);
  * the batched router linear (TopKRouter with batch_size) vs the batch-1 router of each user (S = 128 writes L1);
  * last_token_rows (one-hot selection matmul) vs the selected rows, replicated and sequence-parallel.

Run: pytest models/demos/gpt_oss/tests/unit/test_batched_prefill.py -k 1x4
"""

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.batched_prefill import (
    BATCHED_REDUCE_LINKS,
    DIRECT_REDUCE_SCATTER_MAX_BYTES,
    batched_reduce_links,
    fill_batch_idx,
    last_token_rows,
    partition_users,
    per_user_reduce,
    takes_direct_reduce_scatter,
)
from models.demos.gpt_oss.tt.experts.prefill_routed import ROUTED_PREFILL_NUM_EXPERTS

from ..test_factory import parametrize_mesh_with_fabric, real_checkpoint_path, skip_1x4_on_galaxy

H = 2880
USERS = (0, 5, 31)


def test_direct_reduce_scatter_gate():
    # bf8 [128, 2880]: 4 x 90 tiles x 1088 B = 391680 B <= 512 KiB; [256, 2880] and bf16 [128, 2880] are above it.
    assert DIRECT_REDUCE_SCATTER_MAX_BYTES == 512 << 10
    assert takes_direct_reduce_scatter(128, H, ttnn.bfloat8_b)
    assert not takes_direct_reduce_scatter(256, H, ttnn.bfloat8_b)
    assert not takes_direct_reduce_scatter(128, H, ttnn.bfloat16)


def _shard(mesh_device, x, dtype=ttnn.bfloat8_b):
    """[N, 1, R, W] host -> device d holds x[d] as [1, 1, R, W]."""
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _per_device(t):
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def _skip_unless_bh_1x4(mesh_device):
    if not ttnn.device.is_blackhole(mesh_device) or tuple(mesh_device.shape) != (1, 4):
        pytest.skip("batched prefill runs on Blackhole 1x4")


@pytest.mark.parametrize("B, S, scatter", [(32, 128, False), (32, 256, False), (4, 1024, True)])
@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_per_user_reduce_matches_batch1(mesh_device, device_params, B, S, scatter):
    _skip_unless_bh_1x4(mesh_device)
    torch.manual_seed(0)
    n = mesh_device.shape[1]
    x = torch.randn(n, 1, B * S, H)
    # The batched call on the links the model uses on 1x4 (batched_reduce_links), the batch-1 reference on 1 link.
    links = BATCHED_REDUCE_LINKS
    out = _per_device(per_user_reduce(_shard(mesh_device, x), B, cluster_axis=1, num_links=links, scatter=scatter))
    for u in sorted({min(v, B - 1) for v in USERS}):
        xu = _shard(mesh_device, x[:, :, u * S : (u + 1) * S].contiguous())
        if scatter:
            ref = _per_device(ttnn.reduce_scatter(xu, 2, cluster_axis=1, num_links=1, topology=ttnn.Topology.Ring))
            rows = S // n
            for d in range(n):
                got = out[d][0, 0, u * rows : (u + 1) * rows]
                assert torch.equal(got, ref[d][0, 0]), f"user {u} device {d}: reduce-scatter rows differ"
        else:
            ref = _per_device(ttnn.all_reduce(xu, num_links=1, topology=ttnn.Topology.Ring, cluster_axis=1))[0]
            got = out[0][0, 0, u * S : (u + 1) * S]
            assert torch.equal(
                got, ref[0, 0]
            ), f"user {u}: all-reduce rows differ (max {(got - ref[0, 0]).abs().max()})"


class _Manager:
    """The CCLManager attributes the batched-prefill helpers read."""

    def __init__(self, mesh_device, num_links=1):
        self.mesh_device = mesh_device
        self.num_links = num_links


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_batched_reduce_links(mesh_device, device_params, monkeypatch):
    _skip_unless_bh_1x4(mesh_device)
    from models.demos.gpt_oss.tt import batched_prefill

    assert batched_reduce_links(_Manager(mesh_device), 1) == BATCHED_REDUCE_LINKS == 2  # what BH 1x4 has
    assert batched_reduce_links(_Manager(mesh_device, num_links=4), 1) == 4  # never below the model's links
    monkeypatch.setattr(batched_prefill, "get_num_links", lambda *a: 1)  # a fabric with 1 link: 1, no TT_FATAL
    assert batched_reduce_links(_Manager(mesh_device), 1) == 1

    def unknown(*a):
        raise ValueError("Unsupported number of devices")

    monkeypatch.setattr(batched_prefill, "get_num_links", unknown)  # a mesh the link table does not know
    assert batched_reduce_links(_Manager(mesh_device), 1) == 1


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_fill_batch_idx_per_model(mesh_device, device_params):
    _skip_unless_bh_1x4(mesh_device)
    a, b = _Manager(mesh_device), _Manager(mesh_device)
    t = fill_batch_idx(a, 32)
    assert fill_batch_idx(a, 32) is t  # one tensor per model and batch size
    assert fill_batch_idx(b, 32) is not t  # a new model (new mesh) gets its own
    assert ttnn.to_torch(ttnn.get_device_tensors(t)[0]).flatten().tolist() == list(range(32))


@pytest.mark.parametrize("S", [128, 256])
@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_batched_router_matches_batch1(mesh_device, device_params, S):
    _skip_unless_bh_1x4(mesh_device)
    from models.demos.gpt_oss.tt.topk import TopKRouter

    class _Cfg:
        num_experts_per_tok = 4
        num_local_experts = 32
        hidden_size = H

    torch.manual_seed(0)
    B = 32
    sd = {"weight": torch.randn(32, H) * 0.02, "bias": torch.randn(32) * 0.1}
    router = TopKRouter(mesh_device, _Cfg(), sd)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    x = torch.randn(1, 1, B * S, H)
    xt = ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
    idx_b, w_b = (ttnn.to_torch(ttnn.get_device_tensors(t)[0]) for t in router(xt, True, B))
    for u in USERS:
        xu = ttnn.from_torch(
            x[:, :, u * S : (u + 1) * S].contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        idx_1, w_1 = (ttnn.to_torch(ttnn.get_device_tensors(t)[0]) for t in router(xu, True))
        assert torch.equal(idx_b.reshape(B * S, -1)[u * S : (u + 1) * S], idx_1.reshape(S, -1)), f"user {u}: ids"
        assert torch.equal(w_b.reshape(B * S, -1)[u * S : (u + 1) * S], w_1.reshape(S, -1)), f"user {u}: weights"


@pytest.mark.parametrize("sequence_parallel", [False, True])
@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_last_token_rows(mesh_device, device_params, sequence_parallel):
    _skip_unless_bh_1x4(mesh_device)
    torch.manual_seed(0)
    B, S = 32, 1024
    hidden = torch.randn(1, 1, B * S, H)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    ht = ttnn.from_torch(hidden, device=mesh_device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
    expect_all = ttnn.to_torch(ttnn.get_device_tensors(ht)[0]).float()  # bf8-rounded rows
    last = [int(t) for t in torch.randint(0, S, (B,))]
    last[0], last[1] = 0, S - 1
    if sequence_parallel:
        ht = partition_users(ht, B, cluster_axis=1)
    out = last_token_rows(ht, last, S, mesh_device, cluster_axis=1 if sequence_parallel else None)
    assert out.dtype == ttnn.bfloat8_b
    for d, got in enumerate(_per_device(out)):
        for b, t in enumerate(last):
            assert torch.equal(got[0, 0, b], expect_all[0, 0, b * S + t]), f"device {d} user {b} (token {t})"


# ---- model level: Generator.prefill_forward_text, batched vs batch 1 (24 layers, real weights from HF_MODEL) --------
MODEL_USERS = (0, 1, 2, 3, 17, 31)
MODEL_KV_LEN = 2048  # KV tokens per user (32 users)


def _model_ctx(mesh_device):
    _skip_unless_bh_1x4(mesh_device)
    checkpoint = real_checkpoint_path()
    if checkpoint is None:
        pytest.skip("model-level batched prefill checks need HF_MODEL with the real checkpoint (weights, tokenizer)")
    if json.loads((Path(checkpoint) / "config.json").read_text())["num_local_experts"] != ROUTED_PREFILL_NUM_EXPERTS:
        pytest.skip(f"batched prefill is built for {ROUTED_PREFILL_NUM_EXPERTS} experts (gpt-oss-20b)")
    from models.demos.gpt_oss.tests.perf.prefill_common import build_prefill_generator_b32, load_or_build_prompt_tokens

    ctx = build_prefill_generator_b32(mesh_device, MODEL_KV_LEN)
    # the real-text prompt stream (b32_prompt_tokens "text"): built from the repo's tech reports on first use
    load_or_build_prompt_tokens(ctx["tokenizer"], create=True)
    # Record the batch size of every model forward, to prove which path (batched or per user) ran.
    model = ctx["model"][0]
    ctx["forward_batches"] = calls = []
    ctx["forward_skip_lm_head"] = skips = []
    orig = model.ttnn_prefill_forward

    def spy(*args, batch_size=1, **kwargs):
        calls.append(batch_size)
        skips.append(kwargs.get("skip_lm_head", False))
        return orig(*args, batch_size=batch_size, **kwargs)

    model.ttnn_prefill_forward = spy
    return ctx


def _prefill(ctx, tokens, lens, page_table, sampling_params=None):
    """One Generator.prefill_forward_text call (untraced); logits [B, vocab] float, or the sampled tokens [B]."""
    ctx["forward_batches"].clear()
    ctx["forward_skip_lm_head"].clear()
    out = ctx["generator"].prefill_forward_text(
        tokens,
        page_table=page_table,
        kv_cache=ctx["kv_cache"],
        prompt_lens=list(lens),
        enable_trace=False,
        warmup_prefill=False,
        sampling_params=sampling_params,
    )
    if sampling_params is not None:
        assert isinstance(out, tuple), "on-device sampling returns (tokens, log_probs)"
        return out[0].reshape(-1)[: tokens.shape[0]]
    return out.reshape(tokens.shape[0], -1).float()


@pytest.mark.parametrize("L", [512, 2048])
@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_model_batched_matches_batch1(mesh_device, device_params, L):
    """32 users of L tokens in one batched forward: each checked user's last-token logits equal its batch-1 prefill
    bit for bit (512: replicated residual; 2K: sequence-parallel residual)."""
    from models.demos.gpt_oss.tests.perf.prefill_common import b32_prompt_tokens

    ctx = _model_ctx(mesh_device)
    pt = ctx["page_table"]
    tokens = b32_prompt_tokens(L, "text")
    logits32 = _prefill(ctx, tokens, [L] * 32, pt)
    assert ctx["forward_batches"] == [32], f"batched path not taken: {ctx['forward_batches']}"
    for u in MODEL_USERS:
        logits1 = _prefill(ctx, tokens[u : u + 1], [L], pt[u : u + 1])
        assert ctx["forward_batches"] == [1]
        assert torch.equal(logits32[u], logits1[0]), f"L={L} user {u}: max {(logits32[u] - logits1[0]).abs().max()}"


@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_model_batched_unequal_lengths(mesh_device, device_params):
    """Prompts of 128, 100, 65 and 33 tokens in one 128 bucket: one batched forward, and each user's last-token
    logits (token len - 1, not the bucket end) equal its own batch-1 prefill of len tokens."""
    from models.demos.gpt_oss.tests.perf.prefill_common import b32_prompt_tokens

    ctx = _model_ctx(mesh_device)
    pt = ctx["page_table"]
    tokens = b32_prompt_tokens(128, "text")
    lens = [(128, 100, 65, 33)[u % 4] for u in range(32)]
    logits32 = _prefill(ctx, tokens, lens, pt)
    assert ctx["forward_batches"] == [32], f"batched path not taken: {ctx['forward_batches']}"
    for u in MODEL_USERS:
        n = lens[u]
        logits1 = _prefill(ctx, tokens[u : u + 1, :n], [n], pt[u : u + 1])
        assert torch.equal(
            logits32[u], logits1[0]
        ), f"user {u} ({n} tokens): max {(logits32[u] - logits1[0]).abs().max()}"


@pytest.mark.parametrize("B, L", [(4, 128), (32, 128), (32, 2048)])
@skip_1x4_on_galaxy
@parametrize_mesh_with_fabric([(1, 4)])
def test_model_on_device_sampling(mesh_device, device_params, B, L, expect_error):
    """Greedy on-device sampling with B users of L tokens (128: the text demo's prefill call; 2K: sequence-parallel
    residual): one batched forward (extract_last_tokens_batched_prefill + _apply_norm_and_lm_head), and each user's
    token is the argmax of its logits from the same prompts without sampling (also one batched forward)."""
    from models.common.sampling import SamplingParams
    from models.demos.gpt_oss.tests.perf.prefill_common import b32_prompt_tokens

    ctx = _model_ctx(mesh_device)
    if ctx["model"][0].sampling is None:
        pytest.skip("no on-device sampling on this mesh")
    pt = ctx["page_table"][:B]
    tokens = b32_prompt_tokens(L, "text")[:B]
    n = 32  # the text demo's SAMPLING_BATCH_SIZE
    greedy = SamplingParams(temperature=[0.0] * n, top_k=[1] * n, top_p=[1.0] * n)
    sampled = _prefill(ctx, tokens, [L] * B, pt, sampling_params=greedy)
    assert ctx["generator"]._batched_prefill_samples_on_device(0)
    assert ctx["forward_batches"] == [B], f"batched path not taken with sampling: {ctx['forward_batches']}"
    # The residual return is asked for explicitly by the sampling path (batched_skip_lm_head), never inferred.
    assert ctx["forward_skip_lm_head"] == [True], ctx["forward_skip_lm_head"]
    logits = _prefill(ctx, tokens, [L] * B, pt)
    assert ctx["forward_batches"] == [B], f"batched path not taken without sampling: {ctx['forward_batches']}"
    assert ctx["forward_skip_lm_head"] == [False], ctx["forward_skip_lm_head"]
    expect = logits.argmax(dim=-1)
    assert torch.equal(sampled.long(), expect.long()), f"sampled {sampled.tolist()} != argmax {expect.tolist()}"

    gen = ctx["generator"]
    if (B, L) == (4, 128):
        # A batched call with the slot list and get_last_token = -1 but no skip_lm_head (the shape of the traced
        # batched prefill without sampling, whose caller slices logits per user) still returns the logits of every
        # row, not the residual; extract_last_tokens_batched_prefill rejects such logits.
        model = ctx["model"][0]
        out = gen.prefill_forward_single_user_text(
            tokens,
            page_table=pt,
            user_id=list(range(B)),
            last_token_idx=[L - 1] * B,
            kv_cache=ctx["kv_cache"][0],
            model_id=0,
            batch_size=B,
        )
        assert out.shape[-1] == model.lm_head_weight.shape[-1] != model.hf_config.hidden_size, tuple(out.shape)
        assert out.shape[-2] == B * L, tuple(out.shape)
        full = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1)).float()
        rows = full[0, 0, [b * L + L - 1 for b in range(B)], : logits.shape[-1]]
        assert torch.equal(rows.argmax(dim=-1), expect), f"{rows.argmax(dim=-1).tolist()} != {expect.tolist()}"
        with expect_error(AssertionError, "final residual"):
            model.extract_last_tokens_batched_prefill(
                ttnn.reshape(out, [B, 1, L, out.shape[-1]]), [L - 1] * B, B, L, target_batch=32
            )

    # Warmup agrees: batch sizes up to 32 are warmed with and without on-device sampling (the prefill batches both).
    warmed = []
    gen._warmup_prefill_sweep = lambda **kw: warmed.append(kw["warmup_batch_sizes"])
    for can_sample in (True, False):
        gen.already_warmed_up_prefill = False
        gen.warmup_model_prefill(ctx["kv_cache"], enable_trace=False, can_sample_on_device=can_sample)
    assert max(warmed[0]) == 32 and max(warmed[1]) == 32, warmed
