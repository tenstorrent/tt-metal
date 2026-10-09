# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness probe for dit_fused_distributed_rmsnorm on a Blackhole QuietBox 2.

Covers the shapes and settings around the two-wave column-split prefill path: tile-row counts on both
sides of its eligibility window, odd half-widths, gamma present / absent / fp32, bf16 / bfp8 / fp32
tensors, L1 input and output, num_links 1 and 2, ring 2 and 4, batch 2, a consumer op reading the output
right away (eager and traced), and trace replay. Calls go through the production CCLManager pools with a
distinct input every call, and every call's output is checked.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_and_get_pcc

_EPS = 1e-5
_PCC_MIN = 0.9999
_MAX_ABS = 0.06
_CALLS = 6
_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG


def _reference(x: torch.Tensor, weight) -> torch.Tensor:
    w = weight.float() if weight is not None else None
    return torch.nn.functional.rms_norm(x.float(), normalized_shape=(x.shape[-1],), weight=w, eps=_EPS)


def _mapper(mesh_device):
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, 3))


def _shard(mesh_device, t, dtype, memory_config):
    return ttnn.from_torch(
        t,
        mesh_mapper=_mapper(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=dtype,
        memory_config=memory_config,
    )


def _gather(mesh_device, t) -> torch.Tensor:
    return ttnn.to_torch(
        t,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 3)),
    ).float()


class _Runner:
    def __init__(self, mesh_device, axis, num_links, weight_tt, out_mem, out_dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.axis = axis
        self.num_links = num_links
        self.weight_tt = weight_tt
        self.out_mem = out_mem
        self.out_dtype = out_dtype
        self.ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)

    def __call__(self, x_tt):
        stats = self.ccl.get_fused_norm_stats_buffer(
            ("probe", tuple(x_tt.shape), self.weight_tt is not None),
            lambda: ttnn.experimental.dit_fused_distributed_rmsnorm_create_stats_buffer(
                x_tt, self.axis, self.mesh_device, num_links=self.num_links, weight=self.weight_tt
            ),
        )
        return ttnn.experimental.dit_fused_distributed_rmsnorm(
            x_tt,
            self.axis,
            self.mesh_device,
            self.ccl.get_ag_ping_pong_semaphore(self.axis),
            topology=ttnn.Topology.Linear,
            epsilon=_EPS,
            weight=self.weight_tt,
            persistent_output_buffer=stats,
            num_preferred_links=self.num_links,
            dtype=self.out_dtype,
            memory_config=self.out_mem,
        )


def _check(ref, out, tag, max_abs_limit=_MAX_ABS):
    """`out` is a gathered torch tensor; replicated mesh rows are stacked on dim 0."""
    copies = out.shape[0] // ref.shape[0]
    worst_pcc, worst_abs = 1.0, 0.0
    for c in range(copies):
        got = out[c * ref.shape[0] : (c + 1) * ref.shape[0]]
        _, _, pcc = comp_and_get_pcc(ref, got, _PCC_MIN)
        worst_pcc = min(worst_pcc, float(pcc))
        worst_abs = max(worst_abs, torch.max(torch.abs(ref - got)).item())
    logger.info(f"PROBE {tag} pcc={worst_pcc:.7f} max_abs={worst_abs:.5f}")
    assert worst_pcc >= _PCC_MIN and worst_abs <= max_abs_limit, f"{tag}: pcc={worst_pcc} max_abs={worst_abs}"


def _make_weight(mesh_device, hidden, affine, gamma_dtype=ttnn.bfloat16):
    if not affine:
        return None, None
    weight = (torch.randn(hidden, dtype=torch.float32) * 0.2 + 1.0).to(torch.bfloat16)
    weight_tt = _shard(mesh_device, weight.reshape(1, 1, 1, -1), gamma_dtype, _DRAM)
    return weight, weight_tt


def _random_input(batch, seq_len, hidden, scale):
    return torch.randn(batch, 1, seq_len, hidden, dtype=torch.float32).to(torch.bfloat16) * scale


def _run_eager(
    mesh_device,
    seq_len,
    hidden,
    affine,
    num_links,
    tag,
    batch=1,
    in_dtype=ttnn.bfloat16,
    out_dtype=ttnn.bfloat16,
    gamma_dtype=ttnn.bfloat16,
    in_mem=_DRAM,
    out_mem=_DRAM,
    max_abs_limit=_MAX_ABS,
):
    torch.manual_seed(1234)
    weight, weight_tt = _make_weight(mesh_device, hidden, affine, gamma_dtype)
    run = _Runner(mesh_device, 1, num_links, weight_tt, out_mem, out_dtype)
    for i in range(_CALLS):
        x_tt = _shard(mesh_device, _random_input(batch, seq_len, hidden, 1.0 + i), in_dtype, in_mem)
        # Reference from the values the device actually holds (bfp8 input is quantized on upload).
        x_dev = _gather(mesh_device, x_tt)[:batch]
        out = _gather(mesh_device, run(x_tt))
        _check(_reference(x_dev, weight), out, f"{tag} call{i}", max_abs_limit)


SHAPES = [
    pytest.param(640, 3584, id="R20-C28-campaign-h3584"),
    pytest.param(640, 7168, id="R20-C56-campaign-h7168"),
    pytest.param(32, 4096, id="R1-C32"),
    pytest.param(128, 4096, id="R4-C32"),
    pytest.param(192, 4096, id="R6-C32"),
    pytest.param(320, 5376, id="R10-C42-oddhalf"),
    pytest.param(384, 4608, id="R12-C36-oddhalf"),
    pytest.param(512, 5376, id="R16-C42-oddhalf"),
    pytest.param(896, 4096, id="R28-C32"),
    pytest.param(960, 4096, id="R30-C32"),
    pytest.param(1024, 4096, id="R32-C32"),
    pytest.param(1056, 4096, id="R33-C32"),
    pytest.param(1088, 4096, id="R34-C32"),
    pytest.param(2048, 4096, id="R64-C32"),
    pytest.param(640, 1024, id="R20-C8"),
    pytest.param(640, 2560, id="R20-C20"),
    pytest.param(640, 8192, id="R20-C64"),
    pytest.param(640, 8448, id="R20-C66"),
    pytest.param(640, 3712, id="R20-C29-oddwidth"),
]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize("affine", [True, False], ids=["gamma", "nogamma"])
@pytest.mark.parametrize("seq_len, hidden", SHAPES)
def test_probe_ring4(mesh_device, seq_len, hidden, affine, num_links):
    _run_eager(
        mesh_device, seq_len, hidden, affine, num_links, f"ring4 R{seq_len // 32} H{hidden} {affine=} {num_links=}"
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "batch, in_dtype, out_dtype, gamma_dtype, in_mem, out_mem, max_abs_limit",
    [
        pytest.param(1, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16, _DRAM, _DRAM, _MAX_ABS, id="bfp8-in"),
        pytest.param(1, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat16, _DRAM, _DRAM, 0.25, id="bfp8-out"),
        pytest.param(1, ttnn.float32, ttnn.float32, ttnn.float32, _DRAM, _DRAM, _MAX_ABS, id="fp32-in-out-gamma"),
        pytest.param(1, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, _DRAM, _DRAM, _MAX_ABS, id="fp32-gamma"),
        pytest.param(1, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, _L1, _L1, _MAX_ABS, id="l1-in-out"),
        pytest.param(1, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, _DRAM, _L1, _MAX_ABS, id="l1-out"),
        pytest.param(2, ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat16, _DRAM, _DRAM, _MAX_ABS, id="batch2"),
    ],
)
@pytest.mark.parametrize(
    "seq_len, hidden", [pytest.param(320, 4096, id="S320-C32"), pytest.param(320, 7168, id="S320-C56")]
)
def test_probe_ring4_formats(
    mesh_device, seq_len, hidden, batch, in_dtype, out_dtype, gamma_dtype, in_mem, out_mem, max_abs_limit
):
    _run_eager(
        mesh_device,
        seq_len,
        hidden,
        True,
        1,
        f"ring4 S{seq_len} H{hidden} b{batch} {in_dtype}->{out_dtype} gamma={gamma_dtype} "
        f"{in_mem.buffer_type}->{out_mem.buffer_type}",
        batch=batch,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        gamma_dtype=gamma_dtype,
        in_mem=in_mem,
        out_mem=out_mem,
        max_abs_limit=max_abs_limit,
    )


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize(
    "seq_len, hidden", [pytest.param(640, 2048, id="R20-C32"), pytest.param(640, 3584, id="R20-C56")]
)
def test_probe_ring2(mesh_device, seq_len, hidden, num_links):
    _run_eager(mesh_device, seq_len, hidden, True, num_links, f"ring2 R{seq_len // 32} H{hidden} {num_links=}")


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 400000}], indirect=True
)
@pytest.mark.parametrize("out_mem", [_DRAM, _L1], ids=["out-dram", "out-l1"])
@pytest.mark.parametrize("num_links", [1, 2], ids=["links1", "links2"])
@pytest.mark.parametrize(
    "seq_len, hidden", [pytest.param(640, 4096, id="R20-C32"), pytest.param(640, 7168, id="R20-C56")]
)
def test_probe_consumer(mesh_device, seq_len, hidden, num_links, out_mem):
    """The norm output is read by the next op with no host sync in between, eagerly and inside a trace."""
    torch.manual_seed(7)
    weight, weight_tt = _make_weight(mesh_device, hidden, True)
    run = _Runner(mesh_device, 1, num_links, weight_tt, out_mem)
    tag = f"consumer R{seq_len // 32} H{hidden} {num_links=} {out_mem.buffer_type}"

    def step(x_tt):
        return ttnn.multiply(run(x_tt), 2.0, memory_config=_DRAM)

    for i in range(_CALLS):
        x = _random_input(1, seq_len, hidden, 1.0 + i)
        out = _gather(mesh_device, step(_shard(mesh_device, x, ttnn.bfloat16, _DRAM))) / 2.0
        _check(_reference(x, weight), out, f"{tag} eager call{i}")

    x_tt = _shard(mesh_device, _random_input(1, seq_len, hidden, 1.0), ttnn.bfloat16, _DRAM)
    for _ in range(2):  # compile and fill both pool slots before capture
        step(x_tt)
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_a = step(x_tt)
    out_b = step(out_a)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for i in range(_CALLS):
        x = _random_input(1, seq_len, hidden, 1.0 + i)
        host = ttnn.from_torch(x, mesh_mapper=_mapper(mesh_device), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.copy_host_to_device_tensor(host, x_tt)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        a = _gather(mesh_device, out_a)
        _check(_reference(x, weight), a / 2.0, f"{tag} trace replay{i} a")
        _check(_reference(a, weight), _gather(mesh_device, out_b) / 2.0, f"{tag} trace replay{i} b")
    ttnn.release_trace(mesh_device, tid)
