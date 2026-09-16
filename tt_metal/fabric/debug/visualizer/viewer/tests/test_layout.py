# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Mirror of viewer/js/layout.js chip-grid math. Keep constants and formulas in lockstep."""

from __future__ import annotations

import unittest

# From layout.js
CHIP_W = 72
CHIP_H = 72
CHIP_GAP = 28
PORT = 10
MESH_GAP = 96
WRAP_BOW = 36
PORT_GAP = 4


def chip_position(coord: list[int]) -> tuple[int, int]:
    return coord[1] * (CHIP_W + CHIP_GAP), coord[0] * (CHIP_H + CHIP_GAP)


def port_side(direction: str) -> str:
    return {"N": "top", "S": "bottom", "E": "right", "W": "left"}.get(direction, "inside")


def wrap_control(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    mesh_bounds: dict[str, float],
    axis: str,
) -> tuple[float, float]:
    if axis == "ew":
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        center = (mesh_bounds["minX"] + mesh_bounds["maxX"]) / 2
        cx = mesh_bounds["minX"] - WRAP_BOW if mid_x < center else mesh_bounds["maxX"] + WRAP_BOW
        return cx, mid_y
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    center = (mesh_bounds["minY"] + mesh_bounds["maxY"]) / 2
    cy = mesh_bounds["minY"] - WRAP_BOW if mid_y < center else mesh_bounds["maxY"] + WRAP_BOW
    return mid_x, cy


class LayoutMathTest(unittest.TestCase):
    def test_chip_origin_and_diagonal(self):
        self.assertEqual(chip_position([0, 0]), (0, 0))
        self.assertEqual(chip_position([1, 1]), (CHIP_W + CHIP_GAP, CHIP_H + CHIP_GAP))
        self.assertEqual(chip_position([0, 3]), (3 * (CHIP_W + CHIP_GAP), 0))

    def test_line_is_a_one_row_grid(self):
        xs = [chip_position([0, index])[0] for index in range(4)]
        ys = [chip_position([0, index])[1] for index in range(4)]
        self.assertEqual(xs, [0, 100, 200, 300])
        self.assertEqual(ys, [0, 0, 0, 0])

    def test_port_sides(self):
        self.assertEqual(port_side("N"), "top")
        self.assertEqual(port_side("E"), "right")
        self.assertEqual(port_side("S"), "bottom")
        self.assertEqual(port_side("W"), "left")
        self.assertEqual(port_side("Z"), "inside")
        self.assertEqual(port_side("NONE"), "inside")

    def test_east_west_wrap_control_is_outside_chip_box(self):
        chips = [chip_position([0, 0]), chip_position([0, 1])]
        bounds = {
            "minX": 0,
            "minY": 0,
            "maxX": CHIP_W + CHIP_GAP + CHIP_W,
            "maxY": CHIP_H,
        }
        left = (chips[0][0] + CHIP_W, chips[0][1] + CHIP_H / 2)
        right = (chips[1][0], chips[1][1] + CHIP_H / 2)
        cx, cy = wrap_control(*left, *right, bounds, "ew")
        self.assertTrue(cx < bounds["minX"] or cx > bounds["maxX"])
        self.assertGreaterEqual(cy, bounds["minY"])
        self.assertLessEqual(cy, bounds["maxY"])

    def test_north_south_wrap_bows_vertically(self):
        bounds = {"minX": 0, "minY": 0, "maxX": CHIP_W, "maxY": CHIP_H + CHIP_GAP + CHIP_H}
        top = (CHIP_W / 2, 0)
        bottom = (CHIP_W / 2, bounds["maxY"])
        cx, cy = wrap_control(*top, *bottom, bounds, "ns")
        self.assertTrue(cy < bounds["minY"] or cy > bounds["maxY"])
        self.assertEqual(cx, CHIP_W / 2)

    def test_mesh_gap_constant(self):
        self.assertEqual(MESH_GAP, 96)
        self.assertEqual(PORT + PORT_GAP, 14)


if __name__ == "__main__":
    unittest.main()
