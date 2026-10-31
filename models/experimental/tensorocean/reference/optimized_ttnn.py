"""Tenstorrent TT-NN port of TensorOcean's ``optimized_cpu.py``.

This is a derivative implementation of the LANL TensorOcean optimized
horizontal tracer-advection benchmark.  The numerical work is performed with
TT-NN tensors on a Tenstorrent device.  PyTorch is used only to create the
synthetic input data and, when ``--validate`` is requested, to calculate the
CPU reference result.

The PyTorch implementation builds 10-cell stencils with ``unfold`` and boolean
masks.  TT-NN does not expose an equivalent ``unfold`` operation, so this port
expresses each stencil as ten device-side strided slices and accumulates the
ten products on the device.

Example (from an activated tt-metal/TT-NN environment):

    python3 optimized_ttnn.py 100 100 1 --validate

The original optimized benchmark currently models one tracer; NTRACERS is
retained for command-line compatibility.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Iterable, Sequence, Tuple

import torch

try:
    import ttnn
except ImportError as exc:  # Give a more useful error than a later NameError.
    raise SystemExit(
        "TT-NN is not importable. Activate the tt-metal Python environment "
        "or install the matching TT-NN wheel before running this program."
    ) from exc


TensorMap = Dict[str, torch.Tensor]
Offset = Tuple[int, int]


# These are the offsets selected by the boolean masks in optimized_cpu.py.
# For edge family 1, the source window is 4 rows x 4 columns.
EDGE1_EVEN_ROW_EVEN_COL: Tuple[Offset, ...] = (
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
)
EDGE1_EVEN_ROW_ODD_COL: Tuple[Offset, ...] = (
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
)
EDGE1_ODD_ROW_EVEN_COL: Tuple[Offset, ...] = (
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 2),
)
EDGE1_ODD_ROW_ODD_COL: Tuple[Offset, ...] = (
    (0, 1),
    (0, 2),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
)

# For edge family 2, optimized_cpu.py first removes one halo row, so the
# offsets below are relative to cell row 1 rather than cell row 0.
EDGE2_EVEN_ROW: Tuple[Offset, ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
)
EDGE2_ODD_ROW: Tuple[Offset, ...] = (
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
)


def _slice4(
    tensor,
    starts: Sequence[int],
    ends: Sequence[int],
    steps: Sequence[int] = (1, 1, 1, 1),
):
    """Call TT-NN's device slice with explicit rank-4 bounds."""
    return ttnn.slice(tensor, tuple(starts), tuple(ends), tuple(steps))


def _select_edges(tensor, row_parity: int, row_end: int, col_parity: int | None, col_end: int):
    """Select an even/odd edge sub-mesh while retaining the coefficient axis."""
    if col_parity is None:
        col_start, col_step = 0, 1
    else:
        col_start, col_step = col_parity, 2
    return _slice4(
        tensor,
        (0, row_parity, col_start, 0),
        (tensor.shape[0], row_end, col_end, tensor.shape[3]),
        (1, 2, col_step, 1),
    )


def _cell_slice(
    cell,
    row_start: int,
    row_count: int,
    row_step: int,
    col_start: int,
    col_count: int,
):
    """Return a rank-4 ``[L, rows, cols, 1]`` cell view on the device."""
    return _slice4(
        cell,
        (0, row_start, col_start, 0),
        (
            cell.shape[0],
            row_start + (row_count - 1) * row_step + 1,
            col_start + col_count,
            1,
        ),
        (1, row_step, 1, 1),
    )


def _stencil_dot(
    weights,
    cell,
    offsets: Iterable[Offset],
    *,
    base_row: int,
    row_count: int,
    row_step: int,
    base_col: int,
    col_count: int,
):
    """Compute a 10-point stencil dot product entirely with TT-NN operations."""
    result = None
    for coefficient, (row_offset, col_offset) in enumerate(offsets):
        weight = _slice4(
            weights,
            (0, 0, 0, coefficient),
            (weights.shape[0], weights.shape[1], weights.shape[2], coefficient + 1),
        )
        neighbor = _cell_slice(
            cell,
            base_row + row_offset,
            row_count,
            row_step,
            base_col + col_offset,
            col_count,
        )
        term = ttnn.mul(weight, neighbor)
        result = term if result is None else ttnn.add(result, term)
    return result


def _sum_tensors(tensors: Sequence):
    result = tensors[0]
    for tensor in tensors[1:]:
        result = ttnn.add(result, tensor)
    return result


def make_inputs(n: int, levels: int, seed: int) -> TensorMap:
    """Create deterministic synthetic inputs with the original tensor shapes."""
    generator = torch.Generator().manual_seed(seed)
    mesh_with_halo = n + 4

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float32)

    inputs: TensorMap = {
        "edgeSignOnCell1": randn(1, n + 1, n * 2 + 1),
        "edgeSignOnCell2": randn(1, n, n + 1),
        # A positive, non-near-zero denominator makes BF16 validation stable.
        "areaCell": torch.rand((n,), generator=generator, dtype=torch.float32) + 0.5,
        "dvEdge1": randn(1, n + 1, n * 2 + 1, 1),
        "dvEdge2": randn(1, n, n + 1, 1),
        "advMaskHighOrder1": randn(levels, n + 1, n * 2 + 1, 1),
        "advMaskHighOrder2": randn(levels, n, n + 1, 1),
        "advCoefs1": randn(1, n + 1, n * 2 + 1, 10),
        "advCoefs2": randn(1, n, n + 1, 10),
        "advCoefs3rd1": randn(1, n + 1, n * 2 + 1, 10),
        "advCoefs3rd2": randn(1, n, n + 1, 10),
        "normalThicknessFlux1": randn(levels, n + 1, n * 2 + 1, 1),
        "normalThicknessFlux2": randn(levels, n, n + 1, 1),
        "cell": randn(levels, mesh_with_halo, mesh_with_halo),
    }
    return inputs


def to_device_inputs(host: TensorMap, device, dtype) -> Dict[str, object]:
    """Transfer inputs to interleaved DRAM in tile layout."""
    converted: Dict[str, object] = {}
    levels = host["cell"].shape[0]
    n = host["areaCell"].numel()
    for name, tensor in host.items():
        if name == "cell":
            tensor = tensor.unsqueeze(-1)
        elif name.startswith("edgeSignOnCell"):
            tensor = tensor.unsqueeze(-1).expand(levels, -1, -1, -1)
        elif name == "areaCell":
            tensor = tensor.reshape(1, 1, n, 1).expand(levels, n // 2, n, 1)
        elif name.startswith("advCoefs") or name.startswith("dvEdge"):
            tensor = tensor.expand(levels, -1, -1, -1)

        # from_torch requires materialized host storage; expand() itself is a
        # zero-stride view.
        tensor = tensor.contiguous()

        converted[name] = ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    return converted


def horizontal_flux_ttnn(inputs: Dict[str, object], n: int, levels: int, device):
    """Run the optimized horizontal-flux calculation on a Tenstorrent device."""
    coef_3rd_order = 0.250
    half = n // 2

    edge_sign_1 = inputs["edgeSignOnCell1"]
    edge_sign_2 = inputs["edgeSignOnCell2"]
    area_cell = inputs["areaCell"]
    dv_edge_1 = inputs["dvEdge1"]
    dv_edge_2 = inputs["dvEdge2"]
    adv_mask_1 = inputs["advMaskHighOrder1"]
    adv_mask_2 = inputs["advMaskHighOrder2"]
    adv_coefs_1 = inputs["advCoefs1"]
    adv_coefs_2 = inputs["advCoefs2"]
    adv_coefs_3rd_1 = inputs["advCoefs3rd1"]
    adv_coefs_3rd_2 = inputs["advCoefs3rd2"]
    normal_flux_1 = inputs["normalThicknessFlux1"]
    normal_flux_2 = inputs["normalThicknessFlux2"]
    cell = inputs["cell"]

    ttnn.synchronize_device(device)
    flux_start = time.perf_counter()

    # Third-order weights.  TT-NN's binary operators perform the same singleton
    # broadcasting as torch.broadcast_to in the CPU implementation.
    signed_1 = ttnn.mul(ttnn.sign(normal_flux_1), coef_3rd_order)
    signed_2 = ttnn.mul(ttnn.sign(normal_flux_2), coef_3rd_order)
    masked_flux_1 = ttnn.mul(normal_flux_1, adv_mask_1)
    masked_flux_2 = ttnn.mul(normal_flux_2, adv_mask_2)
    tracer_wgt_1 = ttnn.mul(
        ttnn.mul(ttnn.add(adv_coefs_1, signed_1), masked_flux_1),
        adv_coefs_3rd_1,
    )
    tracer_wgt_2 = ttnn.mul(
        ttnn.mul(ttnn.add(adv_coefs_2, signed_2), masked_flux_2),
        adv_coefs_3rd_2,
    )

    w2_even = _select_edges(tracer_wgt_2, 0, n, None, n + 1)
    w2_odd = _select_edges(tracer_wgt_2, 1, n, None, n + 1)
    w1_ee = _select_edges(tracer_wgt_1, 0, n + 1, 0, n * 2 + 1)
    w1_eo = _select_edges(tracer_wgt_1, 0, n + 1, 1, n * 2 + 1)
    w1_oe = _select_edges(tracer_wgt_1, 1, n + 1, 0, n * 2 + 1)
    w1_oo = _select_edges(tracer_wgt_1, 1, n + 1, 1, n * 2 + 1)

    flx2_even = _stencil_dot(
        w2_even,
        cell,
        EDGE2_EVEN_ROW,
        base_row=1,
        row_count=half,
        row_step=2,
        base_col=0,
        col_count=n + 1,
    )
    flx2_odd = _stencil_dot(
        w2_odd,
        cell,
        EDGE2_ODD_ROW,
        base_row=2,
        row_count=half,
        row_step=2,
        base_col=0,
        col_count=n + 1,
    )
    flx1_ee = _stencil_dot(
        w1_ee,
        cell,
        EDGE1_EVEN_ROW_EVEN_COL,
        base_row=0,
        row_count=half + 1,
        row_step=2,
        base_col=0,
        col_count=n + 1,
    )
    flx1_eo = _stencil_dot(
        w1_eo,
        cell,
        EDGE1_EVEN_ROW_ODD_COL,
        base_row=0,
        row_count=half + 1,
        row_step=2,
        base_col=0,
        col_count=n,
    )
    flx1_oe = _stencil_dot(
        w1_oe,
        cell,
        EDGE1_ODD_ROW_EVEN_COL,
        base_row=1,
        row_count=half,
        row_step=2,
        base_col=0,
        col_count=n + 1,
    )
    flx1_oo = _stencil_dot(
        w1_oo,
        cell,
        EDGE1_ODD_ROW_ODD_COL,
        base_row=1,
        row_count=half,
        row_step=2,
        base_col=0,
        col_count=n,
    )

    # Second-order weights and contributions.
    low_order_mask_1 = ttnn.add(ttnn.neg(adv_mask_1), 1.0)
    low_order_mask_2 = ttnn.add(ttnn.neg(adv_mask_2), 1.0)
    tracer_wgt_1_low = ttnn.mul(
        ttnn.mul(dv_edge_1, 0.500),
        ttnn.mul(low_order_mask_1, normal_flux_1),
    )
    tracer_wgt_2_low = ttnn.mul(
        ttnn.mul(dv_edge_2, 0.500),
        ttnn.mul(low_order_mask_2, normal_flux_2),
    )

    # tracer_wgt_2_low is intentionally computed for fidelity with the source.
    # optimized_cpu.py does not add it to flx2_even/flx2_odd.
    del tracer_wgt_2_low

    w1_low_ee = _select_edges(tracer_wgt_1_low, 0, n + 1, 0, n * 2 + 1)
    w1_low_eo = _select_edges(tracer_wgt_1_low, 0, n + 1, 1, n * 2 + 1)
    w1_low_oe = _select_edges(tracer_wgt_1_low, 1, n + 1, 0, n * 2 + 1)
    w1_low_oo = _select_edges(tracer_wgt_1_low, 1, n + 1, 1, n * 2 + 1)

    ee_cells = ttnn.add(
        _cell_slice(cell, 1, half + 1, 2, 1, n + 1),
        _cell_slice(cell, 2, half + 1, 2, 2, n + 1),
    )
    eo_cells = ttnn.add(
        _cell_slice(cell, 1, half + 1, 2, 2, n),
        _cell_slice(cell, 2, half + 1, 2, 2, n),
    )
    oe_cells = ttnn.add(
        _cell_slice(cell, 3, half, 2, 1, n + 1),
        _cell_slice(cell, 2, half, 2, 2, n + 1),
    )
    oo_cells = ttnn.add(
        _cell_slice(cell, 3, half, 2, 2, n),
        _cell_slice(cell, 2, half, 2, 2, n),
    )

    flx1_ee = ttnn.add(flx1_ee, ttnn.mul(ee_cells, w1_low_ee))
    flx1_eo = ttnn.add(flx1_eo, ttnn.mul(eo_cells, w1_low_eo))
    flx1_oe = ttnn.add(flx1_oe, ttnn.mul(oe_cells, w1_low_oe))
    flx1_oo = ttnn.add(flx1_oo, ttnn.mul(oo_cells, w1_low_oo))

    ttnn.synchronize_device(device)
    flux_seconds = time.perf_counter() - flux_start

    accumulation_start = time.perf_counter()

    flx1_ee = ttnn.mul(flx1_ee, _select_edges(edge_sign_1, 0, n + 1, 0, n * 2 + 1))
    flx1_eo = ttnn.mul(flx1_eo, _select_edges(edge_sign_1, 0, n + 1, 1, n * 2 + 1))
    flx1_oe = ttnn.mul(flx1_oe, _select_edges(edge_sign_1, 1, n + 1, 0, n * 2 + 1))
    flx1_oo = ttnn.mul(flx1_oo, _select_edges(edge_sign_1, 1, n + 1, 1, n * 2 + 1))
    flx2_even = ttnn.mul(flx2_even, _select_edges(edge_sign_2, 0, n, None, n + 1))
    flx2_odd = ttnn.mul(flx2_odd, _select_edges(edge_sign_2, 1, n, None, n + 1))

    even = _sum_tensors(
        (
            _slice4(flx1_ee, (0, 0, 0, 0), (levels, half, n, 1)),
            _slice4(flx1_eo, (0, 0, 0, 0), (levels, half, n, 1)),
            _slice4(flx1_oe, (0, 0, 0, 0), (levels, half, n, 1)),
            flx1_oo,
            _slice4(flx2_even, (0, 0, 0, 0), (levels, half, n, 1)),
            _slice4(flx2_even, (0, 0, 1, 0), (levels, half, n + 1, 1)),
        )
    )
    even = ttnn.div(even, area_cell)

    odd = _sum_tensors(
        (
            _slice4(flx1_ee, (0, 1, 0, 0), (levels, half + 1, n, 1)),
            _slice4(flx1_eo, (0, 1, 0, 0), (levels, half + 1, n, 1)),
            _slice4(flx1_oe, (0, 0, 0, 0), (levels, half, n, 1)),
            flx1_oo,
            _slice4(flx2_odd, (0, 0, 0, 0), (levels, half, n, 1)),
            _slice4(flx2_odd, (0, 0, 1, 0), (levels, half, n + 1, 1)),
        )
    )
    odd = ttnn.div(odd, area_cell)

    ttnn.synchronize_device(device)
    accumulation_seconds = time.perf_counter() - accumulation_start
    return even, odd, flux_seconds, accumulation_seconds


def horizontal_flux_torch(inputs: TensorMap, n: int):
    """Original PyTorch algorithm, factored into a validation function."""
    edge_sign_1 = inputs["edgeSignOnCell1"]
    edge_sign_2 = inputs["edgeSignOnCell2"]
    area_cell = inputs["areaCell"]
    dv_edge_1 = inputs["dvEdge1"]
    adv_mask_1 = inputs["advMaskHighOrder1"]
    adv_mask_2 = inputs["advMaskHighOrder2"]
    adv_coefs_1 = inputs["advCoefs1"]
    adv_coefs_2 = inputs["advCoefs2"]
    adv_coefs_3rd_1 = inputs["advCoefs3rd1"]
    adv_coefs_3rd_2 = inputs["advCoefs3rd2"]
    normal_flux_1 = inputs["normalThicknessFlux1"]
    normal_flux_2 = inputs["normalThicknessFlux2"]
    cell = inputs["cell"]
    levels = cell.shape[0]

    even_row_2_mask = torch.tensor([False, True, True, True, True, True, True, True, False, True, True, True])
    odd_row_2_mask = torch.tensor([True, True, True, False, True, True, True, True, True, True, True, False])
    ee_1_mask = torch.tensor(
        [False, True, True, False, True, True, True, False, False, True, True, True, False, True, True, False]
    )
    eo_1_mask = torch.tensor(
        [False, False, True, True, False, True, True, True, False, True, True, True, False, True, True, False]
    )
    oe_1_mask = torch.tensor(
        [False, True, True, False, False, True, True, True, True, True, True, False, False, True, True, False]
    )
    oo_1_mask = torch.tensor(
        [False, True, True, False, False, True, True, True, False, True, True, True, False, False, True, True]
    )

    a = cell[:, 1:-1, :].unfold(2, 4, 1)
    b = torch.cat((a[:, :-2, :], a[:, 1:-1, :], a[:, 2:, :]), 3)
    even_row_2_cells = b[:, ::2, :, even_row_2_mask]
    odd_row_2_cells = b[:, 1::2, :, odd_row_2_mask]

    a = cell.unfold(2, 4, 1)
    b = torch.cat((a[:, :-3, :], a[:, 1:-2, :], a[:, 2:-1, :], a[:, 3:, :]), 3)
    even_row = b[:, ::2, :, :]
    odd_row = b[:, 1::2, :, :]
    ee_cells = even_row[:, :, :, ee_1_mask]
    eo_cells = even_row[:, :, :-1, eo_1_mask]
    oe_cells = odd_row[:, :, :, oe_1_mask]
    oo_cells = odd_row[:, :, :-1, oo_1_mask]

    tracer_wgt_1 = (
        (
            torch.broadcast_to(adv_coefs_1, (levels, -1, -1, -1))
            + torch.broadcast_to(torch.sign(normal_flux_1) * 0.250, (levels, -1, -1, -1))
        )
        * torch.broadcast_to(normal_flux_1 * adv_mask_1, (-1, -1, -1, 10))
        * torch.broadcast_to(adv_coefs_3rd_1, (levels, -1, -1, -1))
    )
    tracer_wgt_2 = (
        (
            torch.broadcast_to(adv_coefs_2, (levels, -1, -1, -1))
            + torch.broadcast_to(torch.sign(normal_flux_2) * 0.250, (levels, -1, -1, -1))
        )
        * torch.broadcast_to(normal_flux_2 * adv_mask_2, (-1, -1, -1, 10))
        * torch.broadcast_to(adv_coefs_3rd_2, (levels, -1, -1, -1))
    )

    flx2_even = torch.sum(tracer_wgt_2[:, ::2] * even_row_2_cells, 3)
    flx2_odd = torch.sum(tracer_wgt_2[:, 1::2] * odd_row_2_cells, 3)
    flx1_ee = torch.sum(ee_cells * tracer_wgt_1[:, ::2, ::2], 3)
    flx1_eo = torch.sum(eo_cells * tracer_wgt_1[:, ::2, 1::2], 3)
    flx1_oe = torch.sum(oe_cells * tracer_wgt_1[:, 1::2, ::2], 3)
    flx1_oo = torch.sum(oo_cells * tracer_wgt_1[:, 1::2, 1::2], 3)

    even_row = cell[:, 1:-1:2, 1:-1]
    odd_row = cell[:, 2:-1:2, 1:-1]
    tracer_wgt_1_low = torch.broadcast_to(dv_edge_1 * 0.500, (levels, -1, -1, -1)) * torch.broadcast_to(
        (1.000 - adv_mask_1) * normal_flux_1, (levels, -1, -1, -1)
    )

    flx1_ee += (even_row[:, : flx1_ee.shape[1], :-1] + odd_row[:, : flx1_ee.shape[1], 1:]) * tracer_wgt_1_low[
        :, ::2, ::2
    ].squeeze(-1)
    flx1_eo += (even_row[:, : flx1_eo.shape[1], 1:-1] + odd_row[:, : flx1_eo.shape[1], 1:-1]) * tracer_wgt_1_low[
        :, ::2, 1::2
    ].squeeze(-1)
    flx1_oe += (even_row[:, 1 : 1 + flx1_oe.shape[1], :-1] + odd_row[:, : flx1_oe.shape[1], 1:]) * tracer_wgt_1_low[
        :, 1::2, ::2
    ].squeeze(-1)
    flx1_oo += (even_row[:, 1 : 1 + flx1_oo.shape[1], 1:-1] + odd_row[:, : flx1_oo.shape[1], 1:-1]) * tracer_wgt_1_low[
        :, 1::2, 1::2
    ].squeeze(-1)

    flx1_ee *= edge_sign_1[:, ::2, ::2]
    flx1_eo *= edge_sign_1[:, ::2, 1::2]
    flx1_oe *= edge_sign_1[:, 1::2, ::2]
    flx1_oo *= edge_sign_1[:, 1::2, 1::2]
    flx2_even *= edge_sign_2[:, ::2]
    flx2_odd *= edge_sign_2[:, 1::2]

    even = (
        flx1_ee[:, : flx2_even.shape[1], : flx1_eo.shape[2]]
        + flx1_eo[:, : flx2_even.shape[1]]
        + flx1_oe[:, :, : flx1_eo.shape[2]]
        + flx1_oo
        + flx2_even[:, :, :-1]
        + flx2_even[:, :, 1:]
    ) / area_cell
    odd = (
        flx1_ee[:, 1:, : flx1_eo.shape[2]]
        + flx1_eo[:, 1:]
        + flx1_oe[:, :, : flx1_eo.shape[2]]
        + flx1_oo
        + flx2_odd[:, :, :-1]
        + flx2_odd[:, :, 1:]
    ) / area_cell
    return even, odd


def _validate_output(name: str, expected: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float):
    actual = actual.squeeze(-1).float()
    expected = expected.float()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    correlation = torch.corrcoef(torch.stack((expected.flatten(), actual.flatten())))[0, 1].item()
    max_error = (expected - actual).abs().max().item()
    print(f"{name}: validation passed; PCC={correlation:.6f}, max_abs_error={max_error:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", type=int, help="mesh length (must be positive and even)")
    parser.add_argument("depth", type=int, help="number of vertical levels")
    parser.add_argument("ntracers", type=int, help="retained for compatibility; current kernel uses one tracer")
    parser.add_argument("--device-id", type=int, default=0, help="Tenstorrent PCIe device index (default: 0)")
    parser.add_argument("--seed", type=int, default=0, help="random input seed (default: 0)")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="TT-NN compute/storage dtype (default: bfloat16)",
    )
    parser.add_argument("--validate", action="store_true", help="compare TT-NN output with the PyTorch CPU reference")
    parser.add_argument("--atol", type=float, default=0.35, help="validation absolute tolerance")
    parser.add_argument("--rtol", type=float, default=0.08, help="validation relative tolerance")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.length <= 0 or args.length % 2:
        raise SystemExit("LENGTH must be a positive even integer; the original optimized layout assumes even N.")
    if args.depth <= 0:
        raise SystemExit("DEPTH must be positive.")
    if args.ntracers <= 0:
        raise SystemExit("NTRACERS must be positive.")
    if args.ntracers != 1:
        print("Note: like optimized_cpu.py, this benchmark currently computes one tracer.")

    dtype = ttnn.bfloat16 if args.dtype == "bfloat16" else ttnn.float32
    host_inputs = make_inputs(args.length, args.depth, args.seed)
    device = ttnn.open_device(device_id=args.device_id)
    try:
        device_inputs = to_device_inputs(host_inputs, device, dtype)
        even_tt, odd_tt, flux_seconds, accumulation_seconds = horizontal_flux_ttnn(
            device_inputs, args.length, args.depth, device
        )

        print(f"Horz flux computation time:  {flux_seconds:.6f} secs")
        print(f"Horz flux accumulation time: {accumulation_seconds:.6f} secs")

        if args.validate:
            if args.dtype == "bfloat16":
                reference_inputs = {name: tensor.to(torch.bfloat16).float() for name, tensor in host_inputs.items()}
            else:
                reference_inputs = host_inputs
            expected_even, expected_odd = horizontal_flux_torch(reference_inputs, args.length)
            actual_even = ttnn.to_torch(even_tt)
            actual_odd = ttnn.to_torch(odd_tt)
            _validate_output("Even rows", expected_even, actual_even, args.atol, args.rtol)
            _validate_output("Odd rows", expected_odd, actual_odd, args.atol, args.rtol)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
