# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""dit_fused_distributed_rmsnorm all-gather path on a 4-device mesh (QuietBox 2): every call checked.

At num_links=1 on Blackhole, an even tile-row count from 4 to 32 takes the two-wave column split;
num_links=2 (what the models use) and the other row counts take the one-row-per-worker path.
Every call gets a fresh input, goes through the CCLManager stats-buffer and semaphore pools the
models use, and is checked against torch, so a stale gathered stat or a late output write fails.
"""

import pytest
import torch

import ttnn

from ...parallel.manager import CCLManager

EPS = 1e-5
PCC_MIN = 0.9999
MAX_ABS = 0.06
CALLS = 4
DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
FABRIC = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 400000}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    # float64: in float32 the correlation of two 8M-element tensors loses about 1e-4, which is the
    # size of the margin being tested (a perfectly rounded bf16 output scores 0.99987 in float32).
    return float(torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1])


def _reference(x: torch.Tensor, weight) -> torch.Tensor:
    w = weight.float() if weight is not None else None
    return torch.nn.functional.rms_norm(x.float(), normalized_shape=(x.shape[-1],), weight=w, eps=EPS)


def _shard(mesh_device, t, dtype=ttnn.bfloat16, memory_config=DRAM):
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 3))
    return ttnn.from_torch(
        t, mesh_mapper=mapper, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=dtype, memory_config=memory_config
    )


def _gather(mesh_device, t) -> torch.Tensor:
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 3))
    return ttnn.to_torch(t, mesh_composer=composer).float()


def _random(seq_len, hidden, scale):
    return torch.randn(1, 1, seq_len, hidden).to(torch.bfloat16) * scale


class _Norm:
    """The op called the way models/tt_dit/layers/normalization.py calls it."""

    def __init__(self, mesh_device, num_links, weight_tt, out_mem=DRAM, out_dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.weight_tt = weight_tt
        self.out_mem = out_mem
        self.out_dtype = out_dtype
        self.ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)

    def stats_buffer(self, x_tt):
        return self.ccl.get_fused_norm_stats_buffer(
            ("prefill", tuple(x_tt.shape), self.weight_tt is not None),
            lambda: ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                x_tt, 1, self.mesh_device, num_links=self.num_links, weight=self.weight_tt
            ),
        )

    def __call__(self, x_tt):
        return ttnn.experimental.dit_fused_distributed_rmsnorm(
            x_tt,
            1,
            self.mesh_device,
            self.ccl.get_ag_ping_pong_semaphore(1),
            topology=ttnn.Topology.Linear,
            epsilon=EPS,
            weight=self.weight_tt,
            persistent_output_buffer=self.stats_buffer(x_tt),
            num_preferred_links=self.num_links,
            dtype=self.out_dtype,
            memory_config=self.out_mem,
        )


def _check(expected, got, what, max_abs=MAX_ABS):
    rows = expected.shape[0]
    for copy in range(got.shape[0] // rows):
        part = got[copy * rows : (copy + 1) * rows]
        pcc = _pcc(expected, part)
        err = (expected - part).abs().max().item()
        assert pcc >= PCC_MIN and err <= max_abs, f"{what} (mesh row {copy}): pcc={pcc:.7f} max_abs={err:.5f}"


def _weight(mesh_device, hidden, dtype=ttnn.bfloat16):
    weight = (torch.randn(hidden) * 0.2 + 1.0).to(torch.bfloat16)
    return weight, _shard(mesh_device, weight.reshape(1, 1, 1, -1), dtype)


def _run_calls(mesh_device, norm, weight, seq_len, hidden, in_dtype=ttnn.bfloat16, in_mem=DRAM, max_abs=MAX_ABS):
    for i in range(CALLS):
        x_tt = _shard(mesh_device, _random(seq_len, hidden, 1.0 + i), in_dtype, in_mem)
        x = _gather(mesh_device, x_tt)[:1]
        _check(_reference(x, weight), _gather(mesh_device, norm(x_tt)), f"call {i}", max_abs)


# (seq_len, hidden) on a 1x4 mesh: tile rows R = seq_len / 32, local width C = hidden / 4 / 32 tiles.
TWO_WAVE_AND_EDGES = [
    pytest.param(640, 3584, id="R20-C28"),
    pytest.param(640, 7168, id="R20-C56"),
    pytest.param(128, 4096, id="R4-C32"),
    pytest.param(320, 5376, id="R10-C42-odd-half"),
    pytest.param(1024, 4096, id="R32-C32"),
    pytest.param(1056, 4096, id="R33-C32-fallback"),
    pytest.param(32, 4096, id="R1-C32-single-worker"),
    pytest.param(640, 3712, id="R20-C29-odd-width"),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize(("seq_len", "hidden"), TWO_WAVE_AND_EDGES)
def test_prefill_links1(mesh_device, seq_len, hidden):
    torch.manual_seed(0)
    weight, weight_tt = _weight(mesh_device, hidden)
    _run_calls(mesh_device, _Norm(mesh_device, 1, weight_tt), weight, seq_len, hidden)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize(
    ("seq_len", "hidden"),
    [
        pytest.param(640, 3584, id="R20-C28"),
        pytest.param(640, 7168, id="R20-C56"),
        pytest.param(2048, 4096, id="R64-C32"),
    ],
)
def test_prefill_links2(mesh_device, seq_len, hidden):
    torch.manual_seed(0)
    weight, weight_tt = _weight(mesh_device, hidden)
    _run_calls(mesh_device, _Norm(mesh_device, 2, weight_tt), weight, seq_len, hidden)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
def test_prefill_no_gamma(mesh_device):
    torch.manual_seed(0)
    _run_calls(mesh_device, _Norm(mesh_device, 1, None), None, 640, 4096)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize(
    ("in_dtype", "out_dtype", "gamma_dtype", "in_mem", "out_mem", "max_abs"),
    [
        pytest.param(ttnn.float32, ttnn.float32, ttnn.float32, DRAM, DRAM, MAX_ABS, id="fp32"),
        pytest.param(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, DRAM, DRAM, 0.25, id="bfp8-out"),
        pytest.param(ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, L1, L1, MAX_ABS, id="l1-in-out"),
    ],
)
def test_prefill_formats(mesh_device, in_dtype, out_dtype, gamma_dtype, in_mem, out_mem, max_abs):
    torch.manual_seed(0)
    weight, weight_tt = _weight(mesh_device, 4096, gamma_dtype)
    norm = _Norm(mesh_device, 1, weight_tt, out_mem, out_dtype)
    _run_calls(mesh_device, norm, weight, 320, 4096, in_dtype, in_mem, max_abs)


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
def test_prefill_ring2(mesh_device):
    torch.manual_seed(0)
    weight, weight_tt = _weight(mesh_device, 3584)
    _run_calls(mesh_device, _Norm(mesh_device, 1, weight_tt), weight, 640, 3584)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
@pytest.mark.parametrize(
    ("num_links", "out_mem"),
    [
        pytest.param(1, DRAM, id="links1-dram-out"),
        pytest.param(1, L1, id="links1-l1-out"),
        pytest.param(2, DRAM, id="links2-dram-out"),
    ],
)
def test_prefill_next_op_reads_output(mesh_device, num_links, out_mem):
    """The next op reads the norm's output with no host sync in between, eagerly and inside a trace."""
    torch.manual_seed(0)
    seq_len, hidden = 640, 7168
    weight, weight_tt = _weight(mesh_device, hidden)
    norm = _Norm(mesh_device, num_links, weight_tt, out_mem)

    def step(x_tt):
        return ttnn.multiply(norm(x_tt), 2.0, memory_config=DRAM)

    for i in range(CALLS):
        x = _random(seq_len, hidden, 1.0 + i)
        _check(_reference(x, weight), _gather(mesh_device, step(_shard(mesh_device, x))) / 2.0, f"eager call {i}")

    x_tt = _shard(mesh_device, _random(seq_len, hidden, 1.0))
    step(step(x_tt))  # compile, and fill both stats-buffer pool slots before capture
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_a = step(x_tt)
    out_b = step(out_a)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 3))
    for i in range(CALLS):
        x = _random(seq_len, hidden, 1.0 + i)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x, mesh_mapper=mapper, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16), x_tt
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        a = _gather(mesh_device, out_a)
        _check(_reference(x, weight), a / 2.0, f"trace replay {i}, first norm")
        _check(_reference(a, weight), _gather(mesh_device, out_b) / 2.0, f"trace replay {i}, second norm")
    ttnn.release_trace(mesh_device, tid)


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [FABRIC], indirect=True)
def test_prefill_stats_buffer_placement(mesh_device):
    """The stats scratch lives in DRAM, except on the Blackhole two-wave path (num_links=1, R20 here)."""
    x_tt = _shard(mesh_device, _random(640, 4096, 1.0))
    expect_l1 = mesh_device.arch() == ttnn.device.Arch.BLACKHOLE
    for num_links, l1 in [(1, expect_l1), (2, False)]:
        stats = ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
            x_tt, 1, mesh_device, num_links=num_links
        )
        want = ttnn.BufferType.L1 if l1 else ttnn.BufferType.DRAM
        assert stats.memory_config().buffer_type == want, f"num_links={num_links}"
