# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Attention-sink coverage for the Quasar SDPA forks
(ttnn.experimental.quasar.transformer.{scaled_dot_product_attention,
[paged_]scaled_dot_product_attention_decode}).

Shapes follow GPT-OSS-20B's attention call sites (models/demos/gpt_oss/tt/attention/{prefill,decode}.py):
64 query heads / 8 kv heads, head_dim 64, one sink logit per head, alternating layers with a 128-token
sliding window. Prefill uses 32x32 chunks; decode uses k_chunk_size=128 over a paged 64-token-block cache
with height-sharded Q. The KV cache is bf16 (the model's bfloat8_b is not supported on Quasar).

Sink convention (same as the model and the mainline tests): the kernel multiplies the sink by `scale`
together with QK, so the effective sink logit is sink * scale.
"""

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

N_HEADS = 64
N_KV_HEADS = 8
HEAD_DIM = 64
SCALE = HEAD_DIM**-0.5


def _to_device(t, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    # Quasar: tilize on host, then upload (the device-side tilize kernel is not the path under test).
    if ttnn.get_arch_name() == "quasar":
        return ttnn.from_torch(t, dtype=dtype, layout=layout).to(device, memory_config)
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _compute_kernel_config():
    # models/demos/gpt_oss/tt/attention/config.py
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )


def _sink_values(nh):
    # Effective sink logits ~ N(0, 4) (closer to the trained distribution), pre-divided by scale the way
    # models/demos/gpt_oss/tt/attention/weights.py hands them to the op.
    return torch.randn(nh) * 4.0 / SCALE


def _check(ref, out):
    # PCC barely moves when the sink is dropped (that mostly rescales each row), so also bound the
    # mean relative error: dropping the sink costs >= 8% on these cases, bf16 noise is ~1%.
    assert_with_pcc(ref, out, 0.99)
    rel = ((out - ref).abs().mean() / ref.abs().mean()).item()
    assert rel < 0.03, f"mean relative error {rel:.4f}"


def _softmax_with_sink(logits, mask, sink_logit):
    """softmax over [logits | sink] with the sink column dropped. logits/mask: [..., k]; sink_logit: [..., 1]."""
    w = torch.softmax(torch.cat([logits + mask, sink_logit], dim=-1), dim=-1)
    return w[..., :-1]


# ------------------------------------------------------------------------------------------------
# Prefill
# ------------------------------------------------------------------------------------------------


def _prefill_ref(q, k, v, sink, sliding_window):
    b, nh, s, d = q.shape
    rep = nh // k.shape[1]
    k = k.float().repeat_interleave(rep, dim=1)
    v = v.float().repeat_interleave(rep, dim=1)
    logits = torch.matmul(q.float(), k.transpose(-2, -1)) * SCALE
    qi = torch.arange(s)[:, None]
    ki = torch.arange(s)[None, :]
    allowed = ki <= qi
    if sliding_window:
        allowed &= ki > qi - sliding_window
    mask = torch.where(allowed, 0.0, float("-inf"))
    sink_logit = (sink.float() * SCALE).reshape(1, nh, 1, 1).expand(b, nh, s, 1)
    return torch.matmul(_softmax_with_sink(logits, mask, sink_logit), v)


@pytest.mark.parametrize(
    "nh, nkv, seq",
    [(N_HEADS, N_KV_HEADS, 128), (8, 1, 128)],
    ids=["gptoss_nh64", "nh8"],
)
@pytest.mark.parametrize("sliding_window", [None, 128, 64], ids=["full", "sw128", "sw64"])
def test_sdpa_prefill_attention_sink(device, nh, nkv, seq, sliding_window):
    torch.manual_seed(1234)
    q = torch.randn(1, nh, seq, HEAD_DIM).bfloat16()
    k = torch.randn(1, nkv, seq, HEAD_DIM).bfloat16()
    v = torch.randn(1, nkv, seq, HEAD_DIM).bfloat16()
    sink = _sink_values(nh).bfloat16()

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )
    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention(
        _to_device(q, device),
        _to_device(k, device),
        _to_device(v, device),
        is_causal=True,
        scale=SCALE,
        sliding_window_size=sliding_window,
        attention_sink=_to_device(sink.reshape(1, nh, 1, 1), device),
        program_config=program_config,
        compute_kernel_config=_compute_kernel_config(),
    )
    out = ttnn.to_torch(out)[:, :, :seq, :]
    _check(_prefill_ref(q, k, v, sink, sliding_window), out.float())


# ------------------------------------------------------------------------------------------------
# Decode
# ------------------------------------------------------------------------------------------------

BLOCK_SIZE = 64
CACHE_SEQ = 512
K_CHUNK = 128


def _decode_ref(q, k, v, sink, cur_pos, sliding_window):
    # q: [1, b, nh, d]; k/v: [b, nkv, S, d]; returns [1, b, nh, d]
    _, b, nh, d = q.shape
    rep = nh // k.shape[1]
    out = []
    for i in range(b):
        lo = max(0, cur_pos[i] + 1 - sliding_window) if sliding_window else 0
        hi = cur_pos[i] + 1
        kk = k[i, :, lo:hi].float().repeat_interleave(rep, dim=0)  # [nh, n, d]
        vv = v[i, :, lo:hi].float().repeat_interleave(rep, dim=0)
        logits = torch.einsum("hd,hnd->hn", q[0, i].float(), kk) * SCALE
        w = _softmax_with_sink(logits, torch.zeros_like(logits), (sink.float() * SCALE)[:, None])
        out.append(torch.einsum("hn,hnd->hd", w, vv))
    return torch.stack(out)[None]


def _decode_sink_tensor(sink, device):
    # [padded_heads, 32] with the per-head sink in column 0 (models/demos/gpt_oss/tt/attention/weights.py)
    nh = sink.shape[0]
    padded = math.ceil(nh / 32) * 32
    t = torch.zeros(padded, 32, dtype=torch.bfloat16)
    t[:nh, 0] = sink
    return _to_device(t, device)


def _height_sharded_q(q, device):
    # One core per user, (padded_heads, head_dim) per shard, as in the gpt_oss decode path.
    _, b, nh, d = q.shape
    padded = math.ceil(nh / 32) * 32
    grid = ttnn.num_cores_to_corerangeset(b, device.compute_with_storage_grid_size(), row_wise=True)
    mem = ttnn.create_sharded_memory_config(
        shape=(padded, d),
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return _to_device(q, device, memory_config=mem)


def _decode_inputs(batch):
    torch.manual_seed(4321)
    q = torch.randn(1, batch, N_HEADS, HEAD_DIM).bfloat16()
    k = torch.randn(batch, N_KV_HEADS, CACHE_SEQ, HEAD_DIM).bfloat16()
    v = torch.randn(batch, N_KV_HEADS, CACHE_SEQ, HEAD_DIM).bfloat16()
    sink = _sink_values(N_HEADS).bfloat16()
    return q, k, v, sink


def _decode_program_config(device):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=0,
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )


# cur_pos 63: one K chunk. cur_pos 300: three K chunks, and a sliding window that starts mid-chunk.
DECODE_POSITIONS = [63, 300]


@pytest.mark.parametrize("cur_pos", DECODE_POSITIONS, ids=[f"pos{p}" for p in DECODE_POSITIONS])
@pytest.mark.parametrize("sliding_window", [None, 128], ids=["full", "sw128"])
def test_paged_sdpa_decode_attention_sink(device, cur_pos, sliding_window):
    batch = 1
    q, k, v, sink = _decode_inputs(batch)

    blocks_per_seq = CACHE_SEQ // BLOCK_SIZE
    num_blocks = batch * blocks_per_seq

    def to_paged(cache):
        return (
            cache.reshape(batch, N_KV_HEADS, blocks_per_seq, BLOCK_SIZE, HEAD_DIM)
            .transpose(1, 2)
            .reshape(num_blocks, N_KV_HEADS, BLOCK_SIZE, HEAD_DIM)
        )

    permutation = torch.randperm(num_blocks)
    page_table = torch.argsort(permutation).reshape(batch, blocks_per_seq).to(torch.int32)

    out = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
        _height_sharded_q(q, device),
        _to_device(to_paged(k)[permutation], device),
        _to_device(to_paged(v)[permutation], device),
        _to_device(page_table, device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        cur_pos_tensor=_to_device(
            torch.tensor([cur_pos] * batch, dtype=torch.int32), device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        sliding_window_size=sliding_window,
        attention_sink=_decode_sink_tensor(sink, device),
        scale=SCALE,
        program_config=_decode_program_config(device),
        compute_kernel_config=_compute_kernel_config(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(out)[:, :, :N_HEADS, :]
    _check(_decode_ref(q, k, v, sink, [cur_pos] * batch, sliding_window), out.float())


@pytest.mark.parametrize("cur_pos", DECODE_POSITIONS, ids=[f"pos{p}" for p in DECODE_POSITIONS])
@pytest.mark.parametrize("sliding_window", [None, 128], ids=["full", "sw128"])
def test_sdpa_decode_attention_sink(device, cur_pos, sliding_window):
    batch = 1
    q, k, v, sink = _decode_inputs(batch)

    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
        _to_device(q, device),
        _to_device(k, device),
        _to_device(v, device),
        cur_pos_tensor=_to_device(
            torch.tensor([cur_pos] * batch, dtype=torch.int32), device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        ),
        sliding_window_size=sliding_window,
        attention_sink=_decode_sink_tensor(sink, device),
        scale=SCALE,
        program_config=_decode_program_config(device),
        compute_kernel_config=_compute_kernel_config(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.to_torch(out)[:, :, :N_HEADS, :]
    _check(_decode_ref(q, k, v, sink, [cur_pos] * batch, sliding_window), out.float())
