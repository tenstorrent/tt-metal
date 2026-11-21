# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design (op_design.md, Work Distribution and Blocking Model) asks for beyond the acceptance test:

- `select_blocking` pinned for every first-phase INPUTS entry on a 32-cluster and a 2-cluster grid;
- every NEO ran (the Llama call: its 4 column tiles belong to the 4 NEOs of one cluster), and the same call
  under other thread plans gives the same bits;
- ResNet-50's call with K in two chunks, and the 3-by-5-tile call with padded rounds;
- one device session: the op after another program and before a tt-metal program (`to_layout`).
"""

import pytest
import torch

import ttnn
from ttnn.operations.linear import linear
from ttnn.operations.linear import linear_program_artifacts as artifacts
from ttnn.operations.linear.linear_program_artifacts import ThreadPlan, select_blocking

BUDGET = 4_194_304 - 2 * 65536  # a budget below the unreserved L1 of this target; the device query decides at run time


def _pin(m_t, n_t, k_t, cores, fp32=True, bias=True):
    p = select_blocking(m_t, n_t, k_t, fp32, artifacts.THREAD_PLAN, cores, BUDGET, 2048, bias)
    return (p.block_h, p.block_w, p.block_k, p.num_rounds)


# (m_t, k_t, n_t) of the first-phase INPUTS entries -> (block_h, block_w, block_k, num_rounds)
PINS_32 = {
    (1, 64, 32): (1, 1, 32, 8),  # ResNet-50 fc: K in two chunks (64 fails the ring cap by 16 B)
    (1, 2, 4): (1, 1, 2, 1),  # Llama 3.2 1B
    (4, 8, 16): (1, 1, 8, 16),  # Qwen3-VL
    (1, 1, 1): (1, 1, 1, 1),
    (2, 3, 5): (1, 1, 3, 4),  # padded rounds: 3 of the 8 NEO slots of each row block
}
PINS_2 = {
    (1, 64, 32): (1, 4, 8, 2),
    (1, 2, 4): (1, 1, 2, 1),
    (4, 8, 16): (4, 1, 8, 4),  # occupancy ties at 2 clusters: the coarsest, tallest block
    (1, 1, 1): (1, 1, 1, 1),
    (2, 3, 5): (2, 1, 3, 2),
}


@pytest.mark.parametrize("cores,pins", [(32, PINS_32), (2, PINS_2)], ids=["32_clusters", "2_clusters"])
def test_select_blocking_pins(cores, pins):
    for (m_t, k_t, n_t), expected in pins.items():
        assert _pin(m_t, n_t, k_t, cores) == expected, (m_t, k_t, n_t)


def test_select_blocking_respects_dest_and_rings():
    for fp32, dest in ((True, 4), (False, 8)):
        for m_t, k_t, n_t in [(64, 2, 2), (4, 128, 64), (16, 16, 128)]:
            p = select_blocking(m_t, n_t, k_t, fp32, artifacts.THREAD_PLAN, 32, BUDGET, 2048, True)
            assert p.block_h * p.block_w <= dest
            assert m_t % p.block_h == 0 and n_t % p.block_w == 0 and k_t % p.block_k == 0
            assert p.weight_entries * 2048 <= artifacts.DFB_RING_CAP_BYTES
            assert p.footprint_bytes() <= BUDGET


def _operands(device, a_shape, b_shape, bias_n=None, seed=0):
    torch.manual_seed(seed)
    ta = torch.randn(a_shape).to(torch.bfloat16)
    tb = torch.randn(b_shape).to(torch.bfloat16)
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ref = ta.double() @ tb.double()
    bias = None
    if bias_n is not None:
        tbias = (torch.randn(bias_n) * 4).to(torch.bfloat16)
        bias = ttnn.from_torch(tbias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ref = ref + tbias.double()
    return a, b, bias, ref


def _check(got, ref):
    got = got.double()
    err = (got - ref).abs()
    # bfloat16 rounding of the result: half an ulp of the largest magnitude, plus the accumulation.
    assert err.max() <= 0.02 * ref.abs().max() + 0.05, f"max err {err.max().item()} vs ref max {ref.abs().max().item()}"


@pytest.fixture
def thread_plan(monkeypatch):
    def use(plan):
        monkeypatch.setattr(artifacts, "THREAD_PLAN", plan)

    return use


def test_every_neo_computes_its_column(device):
    """Llama's call: one cluster, four column tiles, one per NEO. A NEO that did not run leaves its column
    unwritten (or the ring waiting); each column is checked on its own."""
    a, b, _, ref = _operands(device, (1, 1, 32, 64), (1, 1, 64, 128))
    got = ttnn.to_torch(linear(a, b)).double()
    for neo in range(4):
        cols = slice(32 * neo, 32 * neo + 32)
        _check(got[..., cols], ref[..., cols])


PLANS = [
    ThreadPlan(act_readers=1, weight_readers=1, compute=1, writers=1),
    ThreadPlan(act_readers=1, weight_readers=1, compute=2, writers=2),
    ThreadPlan(act_readers=1, weight_readers=1, compute=4, writers=1),
    ThreadPlan(act_readers=1, weight_readers=1, compute=4, writers=2),
]


@pytest.mark.parametrize("plan", PLANS, ids=lambda p: f"R{p.readers}C{p.compute}W{p.writers}")
@pytest.mark.parametrize(
    "a_shape,b_shape,bias_n",
    [((1, 1, 32, 64), (1, 1, 64, 128), 128), ((1, 1, 64, 96), (96, 160), 160)],
    ids=["llama", "padded_rounds"],
)
def test_thread_plans_agree(device, thread_plan, plan, a_shape, b_shape, bias_n):
    """The ALL and BLOCKED rings at C = 1, 2 and 4 against the plan of the design: the K order is the same in
    every plan, so the bits are too."""
    a, b, bias, ref = _operands(device, a_shape, b_shape, bias_n)
    thread_plan(artifacts.THREAD_PLAN)
    expected = ttnn.to_torch(linear(a, b, bias=bias))
    thread_plan(plan)
    got = ttnn.to_torch(linear(a, b, bias=bias))
    _check(got, ref)
    assert torch.equal(got, expected)


def test_resnet_fc_two_k_chunks(device):
    """ResNet-50's fc call: K = 64 tiles in two chunks of 32 (the bias preload then two chunks into DEST),
    8 clusters; the activation width-sharded in L1 over two cores, the result width-sharded with no shard spec."""
    a_shape, b_shape = (1, 1, 32, 2048), (1, 1, 2048, 1024)
    plan = select_blocking(1, 32, 64, False, artifacts.THREAD_PLAN, 32, BUDGET, 2048, True)
    assert plan.block_k < plan.k_t
    torch.manual_seed(1)
    ta = torch.randn(a_shape).to(torch.bfloat16)
    tb = torch.randn(b_shape).to(torch.bfloat16)
    tbias = (torch.randn(1, 1, 1, 1024) * 8).to(torch.bfloat16)
    shard = ttnn.create_sharded_memory_config(
        shape=(32, 1024),
        core_grid=ttnn.num_cores_to_corerangeset(2, device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    a = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard)
    b = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(tbias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    out = linear(a, b, bias=bias, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=config)
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    ref = ta.double() @ tb.double() + tbias.double()
    got = ttnn.to_torch(out).double()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1]
    assert pcc > 0.999, pcc


@pytest.mark.parametrize(
    "block_h,block_w,block_k,fp32",
    [(4, 1, 4, True), (1, 4, 2, True), (2, 2, 8, True), (4, 2, 4, False), (2, 4, 1, False)],
    ids=lambda v: str(v),
)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
def test_forced_blocks(device, monkeypatch, block_h, block_w, block_k, fp32, with_bias):
    """The block knobs are live: Qwen's call (4 x 8 x 16 tiles) with blocks the 32-cluster grid never picks,
    so the in0 row stride, the row-major DEST block, the bias preload of every block tile and the writer's
    block walk all run with more than one tile."""
    import sys

    linear_module = sys.modules["ttnn.operations.linear.linear"]  # the module, not the function of that name

    def forced(a, b, config, device, threads=None, has_bias=False):
        plan = artifacts.plan_blocking(a, b, config, device, threads, has_bias)
        return artifacts.BlockPlan(
            plan.m_t, plan.n_t, plan.k_t, block_h, block_w, block_k, plan.threads, has_bias, plan.tile_bytes
        )

    monkeypatch.setattr(linear_module, "plan_blocking", forced)
    a, b, bias, ref = _operands(device, (1, 1, 128, 256), (1, 1, 256, 512), 512 if with_bias else None, seed=5)
    if bias is not None:
        bias = ttnn.reshape(bias, (1, 1, 1, 512))
    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32)
    _check(ttnn.to_torch(linear(a, b, bias=bias, compute_kernel_config=config)), ref)


def test_device_config_is_taken(device):
    """Rule R4: tt-metal's device compute kernel config, with fields a descriptor does not have."""
    a, b, bias, ref = _operands(device, (1, 1, 32, 64), (64, 128), 128)
    config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    got = ttnn.to_torch(linear(a, b, bias=bias, compute_kernel_config=config)).double()
    pcc = torch.corrcoef(torch.stack([got.flatten(), ref.flatten()]))[0, 1]
    assert pcc > 0.99, pcc


def test_one_device_session(device):
    """The op after another program (a first call of its own) and before a tt-metal program (`to_layout`,
    an untilize with untagged transfers), with a bias and a width-sharded result, in one device session.
    Every data-movement kernel must hand its transaction tag back."""
    a, b, bias, ref = _operands(device, (1, 1, 128, 256), (256, 512), 512, seed=3)
    first = ttnn.to_torch(linear(a, b)).double()
    _check(first, ref - ttnn.to_torch(bias).double().reshape(-1)[:512])
    sharded = ttnn.create_sharded_memory_config(
        shape=(128, 64),
        core_grid=ttnn.num_cores_to_corerangeset(8, device.compute_with_storage_grid_size(), row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    out = linear(a, b, bias=bias, memory_config=sharded)
    assert out.memory_config() == sharded
    rm = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    _check(ttnn.to_torch(ttnn.from_device(rm)), ref)
