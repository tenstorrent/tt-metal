# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.flat_routed_expert on one Blackhole chip (a p150, or one chip of a Galaxy): accuracy,
determinism and device time, each its own test over the same cases.

A case is a model config (flat_cases.MODELS: the expert shapes and activation) at a test config: the stage and chunk
that set the chip's load (experts per chip, capacity), how the routing spreads the tokens over those experts, and the
device time the op is expected to take there.

* test_flat_expert_accuracy: every routed row against a torch reference of the bfp4-quantized weights.
* test_flat_expert_determinism: CALLS calls on the same inputs, bit-identical on the device (op_checks.DeterminismMarker).
* test_flat_expert_perf: the op's device time from the real-time profiler (the median of op_checks.PERF_RUNS (5)
  calls) against its baseline in FLAT_US (a Galaxy chip's; on a p150 the time is logged, not gated).

Weights: flat_cases.weight_sets (seeded random, N_W sets round-robin over the experts)."""

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
sys.path.insert(0, str(HERE.parent))
from op_checks import PERF_RUNS, DeterminismMarker, check_perf, machine, median_us, op_device_times  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cases = _load("flat_cases", HERE / "flat_cases.py")
weights_py = _load(
    "flat_routed_expert_weights",
    REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py",
)

FLAT_OPS = {"flat_routed_expert": "moe_block/flat_routed_expert/device/kernels"}
CALLS = 10  # (determinism calls; perf: PERF_RUNS)
PCC = 0.99
# The op's device time (us; the median of PERF_RUNS calls) by case id, on one chip of a Blackhole Galaxy - gated there
# only (a p150's chip runs at its own clock: its numbers are logged, not checked).
FLAT_US = {
    "dsv41_flash-8x4-10240-uniform": 730.4,
    "dsv41_flash-8x4-5120-hot": 1736.5,
    "dsv41_flash-8x4-5120-uniform": 641.9,
    "dsv41_flash-8x4-5120-zipf": 684.8,
    "glm53-4x4-5120-uniform": 1005.8,
    "glm53-4x4-7680-uniform": 1115.4,
    "glm53-8x4-10240-uniform": 721.2,
    "glm53-8x4-10240-zipf": 714.5,
    "glm53-8x4-5120-hot": 1426.4,
    "glm53-8x4-5120-uniform": 529.4,
    "glm53-8x4-5120-zipf": 551.1,
    "glm53_flash-8x4-5120-uniform": 387.9,
    "k27-8x4-5120-uniform": 848.7,
    "k3-4x4-2560-uniform": 2606.5,
    "k3-8x4-10240-uniform": 1446.8,
    "k3-8x4-5120-hot": 1860.7,
    "k3-8x4-5120-uniform": 1348.8,
    "k3-8x4-5120-zipf": 1460.9,
    "m3-8x4-5120-uniform": 415.5,
}


# ---------------------------------------------------------------------------------------------------------------------
# Routing: how a chip's tokens spread over its experts. Each returns the per-expert row counts (each <= capacity).
# ---------------------------------------------------------------------------------------------------------------------


def _uniform(md, pc, chunk, seed):
    return cases.uniform_counts(md["E"], md["K"], pc["epc"], chunk, seed)


def _hot(md, pc, chunk, seed):
    """Expert 0 takes every token of the chunk (the capacity), the rest a uniform share: the slowest expert's worst."""
    return [chunk] + _uniform(md, pc, chunk, seed)[1:]


def _zipf(md, pc, chunk, seed):
    """The uniform total, spread by a Zipf(1.2) over a random order of the experts (hot and cold experts)."""
    total, E = sum(_uniform(md, pc, chunk, seed)), pc["epc"]
    g = torch.Generator().manual_seed(seed)
    w = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** 1.2
    w = w[torch.randperm(E, generator=g)]
    return [min(chunk, int(v)) for v in (w / w.sum() * total).round()]


EDGE_COUNTS = (1, 31, 32, 33, 63, 64, 255, 256, 257, 0)


def _edges(md, pc, chunk, seed):
    """Counts at the tile (32) and helper-relay (256) edges, plus empty experts."""
    return [EDGE_COUNTS[e % len(EDGE_COUNTS)] for e in range(pc["epc"])]


def _sparse(md, pc, chunk, seed):
    """Every other expert empty (a chip whose routing skips half its experts)."""
    return [c if e % 2 else 0 for e, c in enumerate(_uniform(md, pc, chunk, seed))]


def _single(md, pc, chunk, seed):
    """One token on the last expert, the rest empty."""
    return [0] * (pc["epc"] - 1) + [1]


ROUTINGS = {"uniform": _uniform, "hot": _hot, "zipf": _zipf, "edges": _edges, "sparse": _sparse, "single": _single}


# ---------------------------------------------------------------------------------------------------------------------
# Test configs
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FlatCase:
    model: str  # flat_cases.MODELS key
    stage: str  # flat_cases.STAGES key: the mesh the per-chip load is taken from
    chunk: int  # tokens a chunk (the capacity per expert: an expert can take every token)
    routing: str  # ROUTINGS key
    perf_margin: float = 0.05  # (over the baseline in FLAT_US)
    seed: int = 1
    marks: tuple = field(default=())

    @property
    def id(self):
        return f"{self.model}-{self.stage}-{self.chunk}-{self.routing}"


CASES = [
    # GLM-5.3: the Galaxy block's stages and chunks
    *(FlatCase("glm53", "8x4", 5120, r) for r in ROUTINGS),
    FlatCase("glm53", "8x4", 10240, "uniform"),
    FlatCase("glm53", "8x4", 10240, "zipf"),
    FlatCase("glm53", "4x4", 5120, "uniform"),
    FlatCase("glm53", "4x4", 7680, "uniform"),
    # Kimi K3
    *(FlatCase("k3", "8x4", 5120, r) for r in ("uniform", "hot", "zipf", "edges")),
    FlatCase("k3", "8x4", 10240, "uniform"),
    FlatCase("k3", "4x4", 2560, "uniform"),
    # DeepSeek V4.1 Flash
    *(FlatCase("dsv41_flash", "8x4", 5120, r) for r in ("uniform", "hot", "zipf", "edges")),
    FlatCase("dsv41_flash", "8x4", 10240, "uniform"),
    # the other activations, one case each
    FlatCase("glm53_flash", "8x4", 5120, "uniform"),
    FlatCase("k27", "8x4", 5120, "uniform"),
    FlatCase("m3", "8x4", 5120, "uniform"),
]
# (the perf test runs the uniform and hot loads; the other routings are accuracy / determinism shapes)
PERF_ROUTINGS = ("uniform", "hot", "zipf")


def _params(cases_, *, routings=None):
    return [pytest.param(c, id=c.id, marks=list(c.marks)) for c in cases_ if routings is None or c.routing in routings]


# ---------------------------------------------------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------------------------------------------------


def _skip_unless_supported(device):
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")
    if device.compute_with_storage_grid_size().y < 10:
        pytest.skip("laid out for a 10-row Blackhole grid")


def _rm(t, dt, device):
    return ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _inputs(case):
    """(x [rows, H] float32, counts [1, NG], regions [1, NG], gids, spans [(offset, count)] a local expert)."""
    md, pc = cases.MODELS[case.model], cases.per_chip(case.model, case.stage, case.chunk)
    E, NG, H = pc["epc"], md["E"], md["H"]
    counts = ROUTINGS[case.routing](md, pc, case.chunk, case.seed)
    assert len(counts) == E and all(0 <= c <= case.chunk for c in counts), counts
    gids = [(NG // E) * e + 1 for e in range(E)]
    offs, o = [], 0
    for c in counts:
        offs.append(o)
        o += -(-c // 32) * 32
    rows = o + 64
    g = torch.Generator().manual_seed(1000 + case.seed)
    x = torch.zeros(rows, H)
    c_ = torch.zeros(1, NG, dtype=torch.int32)
    r_ = torch.zeros(1, NG, dtype=torch.int32)
    for e, gid in enumerate(gids):
        x[offs[e] : offs[e] + counts[e]] = torch.randn(counts[e], H, generator=g)
        c_[0, gid], r_[0, gid] = counts[e], offs[e]
    return x, c_, r_, gids, list(zip(offs, counts))


def _setup(case, device):
    """(op, device inputs, host inputs, weight sets)."""
    md = cases.MODELS[case.model]
    x, c_, r_, gids, spans = _inputs(case)
    wsets = cases.weight_sets(md["H"], md["I"])
    op = weights_py.FlatRoutedExpert(
        device,
        [[wsets[e % cases.N_W] for e in range(len(gids))]],
        m=case.chunk,
        H=md["H"],
        I=md["I"],
        gids=[gids],
        n_global=md["E"],
        act=md["act"],
        pin=1,
    )
    args = (_rm(x, ttnn.bfloat16, device), _rm(c_, ttnn.uint32, device), _rm(r_, ttnn.uint32, device))
    logger.info(f"{case.id}: {len(gids)} experts, rows {[c for _, c in spans]} (total {sum(c for _, c in spans)})")
    return op, args, (x, spans), wsets


DEVICE_PARAMS = [{"l1_small_size": 0}]


# ---------------------------------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("case", _params(CASES))
def test_flat_expert_accuracy(device, case):
    _skip_unless_supported(device)
    md = cases.MODELS[case.model]
    op, args, (x, spans), wsets = _setup(case, device)
    y = ttnn.to_torch(op(*args)).float()
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa: E731
    wq = [tuple(q(w) for w in ws) for ws in wsets]
    worst = (1.0, None)
    for e, (o, c) in enumerate(spans):
        if not c:
            continue
        xe = x[o : o + c].to(torch.bfloat16).float()
        wg, wu, wd = wq[e % cases.N_W]
        ref = weights_py.act_ref(xe @ wg, xe @ wu, md["act"]) @ wd
        got = y[o : o + c]
        assert torch.isfinite(got).all(), f"{case.id}: expert {e} ({c} rows) has non-finite outputs"
        if c == 1:  # (one row: PCC over its H values)
            ref, got = ref.reshape(1, -1), got.reshape(1, -1)
        ok, pcc = comp_pcc(ref, got, PCC)
        worst = min(worst, (float(pcc), e)) if isinstance(pcc, float) else worst
        assert ok, f"{case.id}: expert {e} ({c} rows at {o}): pcc {pcc}"
    logger.info(f"ACCURACY {case.id}: worst expert pcc {worst[0]:.5f} (expert {worst[1]})")
    op.close()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("case", _params(CASES))
def test_flat_expert_determinism(device, case):
    _skip_unless_supported(device)
    op, args, (x, spans), _ = _setup(case, device)
    det = DeterminismMarker(device)
    # (only the routed experts' tiles are written - x is zero past each expert's rows, so their padding rows are
    # zero: compare those tiles; the rest of y is whatever its allocation held)
    tiles = [(o, o + -(-c // 32) * 32) for o, c in spans if c]
    for _ in range(CALLS):
        y = op(*args)
        det.add({f"r{a}": ttnn.slice(y, [a, 0], [b, y.shape[-1]]) for a, b in tiles})
        ttnn.deallocate(y)
    det.assert_identical(f"flat expert {case.id}")
    op.close()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("case", _params(CASES, routings=PERF_ROUTINGS))
def test_flat_expert_perf(device, case):
    _skip_unless_supported(device)
    op, args, _, _ = _setup(case, device)
    ttnn.deallocate(op(*args))  # (warm: program build)

    def run():
        for _ in range(PERF_RUNS):
            ttnn.deallocate(op(*args))

    times = op_device_times(device, run, FLAT_OPS)
    t, other = times["flat_routed_expert"], times["other"]
    if other:
        logger.info(f"PERF {case.id}: {len(other)} other programs (the lead's), {sum(other) / 1e3:.0f} us in all")
    # (the profiler window can also catch the warm call's record: keep the last PERF_RUNS)
    assert len(t) >= PERF_RUNS, f"{case.id}: {len(t)} flat_routed_expert programs profiled, expected {PERF_RUNS}"
    t = t[-PERF_RUNS:]
    us = median_us(t)
    logger.info(
        f"PERF {case.id}: flat_routed_expert median {us:.1f} us (min {min(t) / 1e3:.1f}, max {max(t) / 1e3:.1f})"
    )
    op.close()
    baseline = FLAT_US.get(case.id) if machine() == "BLACKHOLE_GALAXY" else None
    check_perf(f"flat expert {case.id}", us, baseline, case.perf_margin, where=f'FLAT_US["{case.id}"]')


# ---------------------------------------------------------------------------------------------------------------------
# Quantized y (y_quant: moe_block/common/kernels/quantized_row.hpp) - the GLM cases
# ---------------------------------------------------------------------------------------------------------------------

_qspec = importlib.util.spec_from_file_location("quantized_row_golden", HERE.parent / "quantized_row_golden.py")
quantized_row_golden = importlib.util.module_from_spec(_qspec)
_qspec.loader.exec_module(quantized_row_golden)

QUANT_FORMATS = {"fp8": 1, "int8": 2}  # y_quant (quantized_row::Format)
QUANT_CASES = [c for c in CASES if c.model == "glm53" and c.chunk == 5120 and c.routing in ("uniform", "zipf", "edges")]
QUANT_VARIANTS = [
    pytest.param("fp8", 0, id="fp8-1x96"),
    pytest.param("fp8", 32, id="fp8-1x32"),
    pytest.param("int8", 0, id="int8-1x96"),
    pytest.param("int8", 32, id="int8-1x32"),
]
# relative RMS error of the dequantized y vs the op's own bf16 y on the same inputs (the quantization's alone):
# measured on randn-like rows, e4m3 ~2.7%, int8 1x32 ~0.8% / 1x96 ~0.9% (moe_block/quantized_row_golden.py)
QUANT_REL_ERR = {"fp8": 0.035, "int8": 0.0125}
QUANT_PCC = {"fp8": 0.985, "int8": PCC}


def _dequantize(yq, H, producers, fmt):
    """y [rows, H + 16 P] uint8 (values, then a 16-byte cell a down core) -> fp32 [rows, H]."""
    yq = yq.to(torch.uint8)
    codes, cells = yq[:, :H], yq[:, H:].reshape(yq.shape[0], producers, 16)
    e16 = cells[:, :, : H // producers // 16].reshape(yq.shape[0], -1).int()  # one byte a 16 values
    if fmt == "fp8":
        vals, bias = codes.view(torch.float8_e4m3fn).float(), 127
    else:
        vals, bias = ((codes & 0x7F).int() * (1 - 2 * (codes >> 7).int())).float(), 133
    return vals * torch.pow(2.0, (e16 - bias).float()).repeat_interleave(16, dim=1)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("fmt, block", QUANT_VARIANTS)
@pytest.mark.parametrize("case", _params(QUANT_CASES))
def test_flat_expert_quant_accuracy(device, case, fmt, block):
    """The dequantized y against torch (PCC) and against the op's bf16 row-major y on the same inputs."""
    _skip_unless_supported(device)
    md = cases.MODELS[case.model]
    H = md["H"]
    op, args, (x, spans), wsets = _setup(case, device)
    y_bf16 = ttnn.to_torch(op(*args, y_row_major=True, down_fp32=True)).float()
    yq = ttnn.to_torch(op(*args, y_quant=QUANT_FORMATS[fmt], y_quant_block=block))
    producers = (yq.shape[-1] - H) // 16
    assert yq.shape[-1] == H + 16 * producers and producers > 0, yq.shape
    y = _dequantize(yq, H, producers, fmt)
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa: E731
    wq = [tuple(q(w) for w in ws) for ws in wsets]
    worst, err, ref_norm = 1.0, 0.0, 0.0
    for e, (o, c) in enumerate(spans):
        if not c:
            continue
        xe = x[o : o + c].to(torch.bfloat16).float()
        wg, wu, wd = wq[e % cases.N_W]
        ref = weights_py.act_ref(xe @ wg, xe @ wu, md["act"]) @ wd
        got = y[o : o + c]
        assert torch.isfinite(got).all(), f"{case.id}: expert {e} has non-finite outputs"
        ok, pcc = comp_pcc(ref.reshape(c, -1), got.reshape(c, -1), QUANT_PCC[fmt])
        worst = min(worst, float(pcc)) if isinstance(pcc, float) else worst
        assert ok, f"{case.id} {fmt}: expert {e} ({c} rows at {o}): pcc {pcc}"
        err += float((got - y_bf16[o : o + c]).pow(2).sum())
        ref_norm += float(y_bf16[o : o + c].pow(2).sum())
    rel = (err / ref_norm) ** 0.5
    logger.info(
        f"QUANT ACCURACY {case.id} {fmt} 1x{block or 96}: worst expert pcc {worst:.5f}, rel err vs bf16 y {rel:.4f}"
    )
    assert rel <= QUANT_REL_ERR[fmt], f"{case.id} {fmt}: relative error vs the op's bf16 y {rel:.4f}"
    op.close()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("fmt, block", QUANT_VARIANTS)
@pytest.mark.parametrize("case", _params(QUANT_CASES[:1]))
def test_flat_expert_quant_determinism(device, case, fmt, block):
    _skip_unless_supported(device)
    op, args, (x, spans), _ = _setup(case, device)
    det = DeterminismMarker(device)
    tiles = [(o, o + c) for o, c in spans if c]  # (the routed rows: the rest of y is never written)
    for _ in range(CALLS):
        y = op(*args, y_quant=QUANT_FORMATS[fmt], y_quant_block=block)
        det.add({f"r{a}": ttnn.slice(y, [a, 0], [b, y.shape[-1]]) for a, b in tiles})
        ttnn.deallocate(y)
    det.assert_identical(f"flat expert {case.id} {fmt}")
    op.close()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("fmt, block", [pytest.param(None, 0, id="bf16-row-major")] + QUANT_VARIANTS)
@pytest.mark.parametrize("case", _params([c for c in QUANT_CASES if c.routing == "uniform"]))
def test_flat_expert_quant_perf(device, case, fmt, block):
    """Device time with quantized y against the bf16 row-major y (down_fp32) it replaces: logged."""
    _skip_unless_supported(device)
    op, args, _, _ = _setup(case, device)
    kw = (
        dict(y_row_major=True, down_fp32=True) if fmt is None else dict(y_quant=QUANT_FORMATS[fmt], y_quant_block=block)
    )
    ttnn.deallocate(op(*args, **kw))

    def run():
        for _ in range(PERF_RUNS):
            ttnn.deallocate(op(*args, **kw))

    t = op_device_times(device, run, FLAT_OPS)["flat_routed_expert"][-PERF_RUNS:]
    logger.info(f"QUANT PERF {case.id} {fmt or 'bf16'} 1x{block or 96}: median {median_us(t):.1f} us")
    op.close()
