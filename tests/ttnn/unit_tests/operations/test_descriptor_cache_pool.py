# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Cache-correctness sweep for the pool / reduction / nlp-transformer ops migrated to the
descriptor framework (#46506). Companion to test_descriptor_cache.py — same fixture + pattern.

For each migrated op we run its SIMPLEST existing test invocation (copied verbatim from the
op's own test file — not invented), but under an ENABLED program cache. Two things must hold:

  * not stale  : every call's result matches the op's golden, even on a program-cache HIT.
                 The strongest staleness probe is to re-run with a FRESH input tensor each
                 call (fresh allocation -> fresh buffer address): if the framework froze the
                 first call's address into the rt-args instead of patching it on the hit, the
                 later calls read stale memory and the golden check fails.
  * not over-caching : holding the op's attributes constant (only the input DATA / address
                 varies), the op must NOT create a new program-cache entry per call. A new
                 entry per fresh allocation means an address (or address-derived value) leaked
                 into the program hash — the "cache too restrictive" failure that rebuilds
                 every call.

NOTE on what is a legitimate new entry: in the modern device-operation framework the DEFAULT
program hash reflects over the WHOLE operation_attributes struct (ttnn/api/ttnn/device_operation.hpp
compute_program_hash -> hash_objects_with_default_seed(..., operation_attributes, tensor_args)).
So a per-call value that lives in operation_attributes (topk's `k`, upsample's `scale_factor`,
pool's `divisor_override`/`count_include_pad`) is ALREADY hashed -> varying it SHOULD add a cache
entry, and that is correct, not over-caching. Therefore the entries<=1 assertion is only made for
the address-only (fixed-attribute) loop. Loops that vary an attribute assert correctness only.

These tests intentionally MIRROR the verdicts predicted from the device factories:
  max_pool2d / avg_pool2d  : OK  (empty get_dynamic_runtime_args; divisor_override &
                                  count_include_pad are hashed compile-time config)
  grid_sample              : OK  (grid is a tensor input; only addresses ride per call)
  upsample                 : OK  (scale_factor is a hashed attribute; addresses patched)
  topk                     : OK  (k is a hashed attribute; addresses patched)
  nlp_create_qkv_heads     : OK  (get_dynamic_runtime_args patches q/k/v base addresses)
  nlp_create_qkv_heads_boltz: sharded path not constructible single-device (see skip)

A FAILURE of any of these tests (stale golden or entry blow-up) flags that op as a descriptor
fix candidate.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.common.utility_functions import comp_pcc

from ttnn.operations.pool import golden_grid_sample, golden_upsample


@pytest.fixture(scope="module")
def cache_device():
    # l1_small_size matches the pool descriptor-class2 test fixture (pool needs the L1 scratch).
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


# ---------------------------------------------------------------------------
# pool / generic — ttnn.max_pool2d. Config copied from test_descriptor_class2_pool.py
# (BATCH=16, 16x16x32, 3x3 / stride 2 / pad 1). Pool declares an EMPTY get_dynamic_runtime_args
# and lets the framework patch the sharded CB buffer bindings on a hit -> verdict OK.
# We vary the input DATA (fresh address) across calls: addresses must be patched (no stale,
# no entry growth).
# ---------------------------------------------------------------------------
_BATCH, _IN_H, _IN_W, _CH = 16, 16, 16, 32
_KERNEL, _STRIDE, _PADDING, _DILATION = (3, 3), (2, 2), (1, 1), (1, 1)


def _pool_out_hw():
    out_h = (_IN_H + 2 * _PADDING[0] - _DILATION[0] * (_KERNEL[0] - 1) - 1) // _STRIDE[0] + 1
    out_w = (_IN_W + 2 * _PADDING[1] - _DILATION[1] * (_KERNEL[1] - 1) - 1) // _STRIDE[1] + 1
    return out_h, out_w


def _make_pool_input(device, seed):
    # SHAPE + all pool config are FIXED across calls (module constants); only the random DATA
    # differs per iteration (fresh seed -> fresh values at a fresh allocation/address). This keeps
    # the entry-count check valid: a new entry could only come from an address leaking into the hash,
    # not from a shape/config change (a shape change would legitimately add an entry).
    torch.manual_seed(seed)
    torch_nhwc = torch.randn(_BATCH, _IN_H, _IN_W, _CH, dtype=torch.bfloat16)
    flat = torch_nhwc.reshape(1, 1, _BATCH * _IN_H * _IN_W, _CH)
    ttnn_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    torch_nchw = torch_nhwc.permute(0, 3, 1, 2).float()
    return ttnn_input, torch_nchw


def _pool_to_nchw(ttnn_out, out_h, out_w):
    t = ttnn.to_torch(ttnn_out).float().reshape(_BATCH, out_h, out_w, _CH)
    return t.permute(0, 3, 1, 2)


def test_max_pool2d_cache(cache_device):
    out_h, out_w = _pool_out_hw()

    def _run(seed):
        ttnn_input, torch_nchw = _make_pool_input(cache_device, seed)
        out = ttnn.max_pool2d(
            input_tensor=ttnn_input,
            batch_size=_BATCH,
            input_h=_IN_H,
            input_w=_IN_W,
            channels=_CH,
            kernel_size=_KERNEL,
            stride=_STRIDE,
            padding=_PADDING,
            dilation=_DILATION,
        )
        ref = F.max_pool2d(torch_nchw, kernel_size=_KERNEL, stride=_STRIDE, padding=_PADDING, dilation=_DILATION)
        passing, pcc = check_with_pcc(ref, _pool_to_nchw(out, out_h, out_w), 0.999)
        assert passing, f"max_pool2d stale on cache hit (fresh input): PCC {pcc}"

    # WARM-UP: first call populates the cache with all of the op's sub-programs (max_pool2d
    # legitimately builds several: halo / reshard / pool). Record the settled entry count, then
    # assert ZERO growth across further calls with the same config (only input DATA / addresses
    # differ). Zero growth = the cache is reused regardless of how many sub-programs the op uses.
    _run(0)
    base = cache_device.num_program_cache_entries()
    for seed in range(1, 4):
        _run(seed)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"max_pool2d: cache grew past {base} across fresh allocations (addr not patched / over-caching)"


def test_avg_pool2d_cache(cache_device):
    # avg_pool2d with count_include_pad=False + non-zero padding materializes the per-stick scalar
    # config tensor (workload-owned, baked into compile-time args, and count_include_pad is hashed).
    out_h, out_w = _pool_out_hw()

    def _run(seed):
        ttnn_input, torch_nchw = _make_pool_input(cache_device, seed)
        out = ttnn.avg_pool2d(
            input_tensor=ttnn_input,
            batch_size=_BATCH,
            input_h=_IN_H,
            input_w=_IN_W,
            channels=_CH,
            kernel_size=_KERNEL,
            stride=_STRIDE,
            padding=[_PADDING[0], _PADDING[0], _PADDING[1], _PADDING[1]],
            count_include_pad=False,
        )
        ref = F.avg_pool2d(torch_nchw, kernel_size=_KERNEL, stride=_STRIDE, padding=_PADDING, count_include_pad=False)
        passing, pcc = check_with_pcc(ref, _pool_to_nchw(out, out_h, out_w), 0.999)
        assert passing, f"avg_pool2d stale on cache hit (fresh input): PCC {pcc}"

    # WARM-UP then assert ZERO growth (avg_pool2d builds several sub-programs on first call).
    _run(0)
    base = cache_device.num_program_cache_entries()
    for seed in range(1, 4):
        _run(seed)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"avg_pool2d: cache grew past {base} across fresh allocations (addr not patched / over-caching)"


# ---------------------------------------------------------------------------
# pool / grid_sample. Copied from pool/test_grid_sample.py test_grid_sample_random_grid:
# NHWC row-major input + a random grid TENSOR in [-1, 1], golden_grid_sample reference,
# bilinear / align_corners=False, float32 grid (atol=0.02, rtol=1e-2). The grid AND the input
# are both tensors -> only addresses ride per call. We vary BOTH input data and grid data each
# call (fresh addresses): must not go stale and must not grow entries.
# ---------------------------------------------------------------------------
def test_grid_sample_cache(cache_device):
    input_shape, grid_shape = (1, 32, 8, 8), (1, 6, 6, 2)
    mode, align_corners, grid_dtype = "bilinear", False, ttnn.float32
    batch_size, channels, height, width = input_shape

    def _run(i):
        torch.manual_seed(i)
        torch_input_nhwc = torch.randn((batch_size, height, width, channels), dtype=torch.bfloat16)
        torch_grid_f32 = torch.rand(grid_shape, dtype=torch.float32) * 2.0 - 1.0
        torch_output_nhwc = golden_grid_sample(
            input_tensor=torch_input_nhwc,
            grid=torch_grid_f32.to(torch.float32),
            mode=mode,
            padding_mode="zeros",
            align_corners=align_corners,
        )
        ttnn_input = ttnn.from_torch(torch_input_nhwc, layout=ttnn.ROW_MAJOR_LAYOUT, device=cache_device)
        ttnn_grid = ttnn.from_torch(torch_grid_f32, layout=ttnn.ROW_MAJOR_LAYOUT, device=cache_device, dtype=grid_dtype)
        ttnn_output = ttnn.grid_sample(ttnn_input, ttnn_grid, mode=mode, align_corners=align_corners)
        ttnn_output_torch = ttnn.to_torch(ttnn_output)
        assert torch.allclose(
            torch_output_nhwc, ttnn_output_torch, atol=0.02, rtol=1e-2
        ), f"grid_sample stale on cache hit (fresh input/grid), call {i}"

    # WARM-UP then assert ZERO growth across fresh input/grid allocations.
    _run(0)
    base = cache_device.num_program_cache_entries()
    for i in range(1, 4):
        _run(i)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"grid_sample: cache grew past {base} across fresh allocations (addr not patched)"


# ---------------------------------------------------------------------------
# pool / upsample. Copied from pool/test_upsample.py test_upsample_nearest_interleaved:
# NHWC row-major DRAM input, mode="nearest", golden_upsample reference, torch.equal.
# scale_factor=(scale_h, scale_w) is a HASHED attribute (lives in operation_attributes), so
# varying it legitimately adds a cache entry -> we do NOT bound entries across scales.
#   - test_upsample_cache       : fix scale, vary input DATA -> not stale + entries<=1.
#   - test_upsample_cache_scale : vary scale_factor (correctness only across a re-used input).
# ---------------------------------------------------------------------------
_UPSAMPLE_SHAPE = [1, 64, 32, 32]  # N C H W (from the test's interleaved param list)


def _upsample_once(device, scale_factor, seed):
    batch_size, num_channels, height, width = _UPSAMPLE_SHAPE
    torch.manual_seed(seed)
    torch_input_nhwc = torch.rand((batch_size, height, width, num_channels), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_nhwc, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    torch_result = golden_upsample(input_tensor=torch_input_nhwc, scale_factor=scale_factor, mode="nearest")
    output_tensor = ttnn.to_torch(ttnn.upsample(input_tensor, scale_factor))
    return output_tensor, torch_result


def test_upsample_cache(cache_device):
    scale_factor = (2, 2)
    # WARM-UP then assert ZERO growth across fresh input allocations (scale_factor held FIXED).
    out, ref = _upsample_once(cache_device, scale_factor, seed=0)
    assert torch.equal(out, ref), "upsample stale on cache hit (fresh input), seed=0"
    base = cache_device.num_program_cache_entries()
    for seed in range(1, 4):
        out, ref = _upsample_once(cache_device, scale_factor, seed)
        assert torch.equal(out, ref), f"upsample stale on cache hit (fresh input), seed={seed}"
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"upsample: cache grew past {base} across fresh allocations (addr not patched)"


def test_upsample_cache_scale(cache_device):
    # scale_factor is a hashed attribute: each value is allowed its own entry; we only require
    # that no value goes stale (the result tracks the scale actually requested on that call).
    for scale_factor in [(2, 2), (3, 2), (2, 3), (3, 3)]:
        out, ref = _upsample_once(cache_device, scale_factor, seed=0)
        assert torch.equal(out, ref), f"upsample stale at scale_factor={scale_factor}"


# ---------------------------------------------------------------------------
# reduction / topk. Copied from reduce/test_topk.py run_topk_test: TILE input padded with
# TEST_PADDING_VALUE, torch.topk reference, assert_with_pcc on values. `k` is a HASHED attribute
# (in operation_attributes) so varying it legitimately adds an entry -> not bounded across k.
#   - test_topk_cache       : fix k, vary input DATA -> not stale + entries<=1.
#   - test_topk_cache_k     : vary k (correctness only).
# ---------------------------------------------------------------------------
_TOPK_N, _TOPK_C, _TOPK_H, _TOPK_W, _TOPK_DIM = 1, 1, 64, 64, 2
_TEST_PADDING_VALUE = -42


def _topk_once(device, k, seed):
    torch.manual_seed(seed)
    shape = [_TOPK_N, _TOPK_C, _TOPK_H, _TOPK_W]
    input = torch.randn(shape, dtype=torch.bfloat16) * 0.9
    ttnn_input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, _TEST_PADDING_VALUE)
    pyt_values, _ = torch.topk(input, k, dim=_TOPK_DIM, largest=True, sorted=True)
    ttnn_values, _ = ttnn.topk(ttnn_input, k, dim=_TOPK_DIM, largest=True, sorted=True)
    return ttnn.to_torch(ttnn_values), pyt_values


def test_topk_cache(cache_device):
    k = 32
    # WARM-UP then assert ZERO growth across fresh input allocations (k held FIXED). topk builds
    # several sub-programs on its first call, then reuses them.
    ttnn_values, pyt_values = _topk_once(cache_device, k, seed=0)
    assert_with_pcc(pyt_values, ttnn_values, 0.9999)
    base = cache_device.num_program_cache_entries()
    for seed in range(1, 4):
        ttnn_values, pyt_values = _topk_once(cache_device, k, seed)
        assert_with_pcc(pyt_values, ttnn_values, 0.9999)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"topk: cache grew past {base} across fresh allocations (addr not patched)"


def test_topk_cache_k(cache_device):
    # k is a hashed attribute: each k may get its own entry; require only that none go stale.
    for k in [2, 4, 32, 64]:
        ttnn_values, pyt_values = _topk_once(cache_device, k, seed=0)
        assert_with_pcc(pyt_values, ttnn_values, 0.9999)


# ---------------------------------------------------------------------------
# experimental / transformer / nlp_create_qkv_heads. Interleaved (DRAM) path copied from
# tests/.../misc/test_nlp_create_qkv_heads.py run_nlp_create_qkv_heads_test. The sharded path
# bakes q/k/v base addresses into rt-args but declares get_dynamic_runtime_args to patch them;
# the interleaved path carries only addresses. We vary the input DATA each call (fresh address):
# Q/K/V must not go stale and entries must not grow.
# ---------------------------------------------------------------------------
def _nlp_qkv_once(device, batch, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads, seed):
    torch.manual_seed(seed)
    in0_shape = [batch, 1, seq_len, (num_q_heads + 2 * num_kv_heads) * head_dim]
    A = torch.randn(in0_shape)
    in0_t = ttnn.Tensor(A, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device, ttnn.DRAM_MEMORY_CONFIG)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        None,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ref_q, ref_k, ref_v = torch.split(
        A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    if transpose_k_heads:
        ref_k = ref_k.transpose(-2, -1)
    return (
        (ttnn.to_torch(q), ref_q),
        (ttnn.to_torch(k), ref_k),
        (ttnn.to_torch(v), ref_v),
    )


def test_nlp_create_qkv_heads_cache(cache_device):
    # Smallest generic case from the op's own test list: (1, 128, 64, 71, 1, False).
    batch, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads = 1, 128, 64, 71, 1, False

    def _run(seed):
        for got, ref in _nlp_qkv_once(
            cache_device, batch, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads, seed
        ):
            passing, pcc = comp_pcc(got, ref, 1.0)
            assert passing, f"nlp_create_qkv_heads stale on cache hit (fresh input), seed={seed}, PCC {pcc}"

    # WARM-UP then assert ZERO growth across fresh input allocations.
    _run(0)
    base = cache_device.num_program_cache_entries()
    for seed in range(1, 4):
        _run(seed)
    assert (
        cache_device.num_program_cache_entries() == base
    ), f"nlp_create_qkv_heads: cache grew past {base} across fresh allocations (addr not patched)"


# ---------------------------------------------------------------------------
# experimental / transformer / nlp_create_qkv_heads_boltz. SKIPPED for the same reason the
# descriptor-class2 nlp test skips its sharded boltz case: the boltz output spec shards each head
# as {TILE_HEIGHT, head_dim} over num_q_heads * (seq*seq / TILE_HEIGHT) shards, which exceeds the
# available cores for any tile-aligned seq, so a valid single-device sharded call cannot be built.
# The interleaved boltz path's get_dynamic_runtime_args fix is identical to the base op above; its
# minimal correctness shapes (seq>=704) are large/slow for a cache unit test, so coverage is left
# to the op's own test. The descriptor cache behaviour is exercised by test_nlp_create_qkv_heads_cache.
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="nlp_create_qkv_heads_boltz: sharded path not constructible single-device (shard count "
    "exceeds cores); interleaved boltz uses the same get_dynamic_runtime_args fix as the base op, "
    "which is covered by test_nlp_create_qkv_heads_cache. Its own minimal shapes (seq>=704) are too "
    "large/slow for a cache unit test."
)
def test_nlp_create_qkv_heads_boltz_cache(cache_device):
    pass
