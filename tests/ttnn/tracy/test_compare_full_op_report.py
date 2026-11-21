# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from tracy.compare_full_op_report import compare_cells


def test_compare_cells_reports_a_row_count_mismatch():
    headers = ["GLOBAL CALL COUNT", "OP CODE"]
    differences, allowed_headers = compare_cells(headers, [["1", "add"]], headers, [], max_differences=1)

    assert differences == ["Row count mismatch: 2 rows vs 1 rows"]
    assert allowed_headers == set()


def test_compare_cells_reports_every_difference_up_to_the_limit():
    headers = ["GLOBAL CALL COUNT", "OP CODE"]
    differences, allowed_headers = compare_cells(headers, [["1", "add"]], headers, [], max_differences=20)

    assert differences == [
        "Row count mismatch: 2 rows vs 1 rows",
        "Extra row 2 only present in first file: ['1', 'add']",
    ]
    assert allowed_headers == set()


def test_compare_cells_reports_a_column_count_mismatch():
    headers = ["GLOBAL CALL COUNT", "OP CODE"]
    differences, allowed_headers = compare_cells(headers, [["1", "add"]], headers, [["1"]], max_differences=1)

    assert differences == ["Row 2 column count mismatch: 2 columns vs 1 columns"]
    assert allowed_headers == set()


def test_compare_cells_reports_a_cell_mismatch():
    headers = ["GLOBAL CALL COUNT", "OP CODE"]
    differences, allowed_headers = compare_cells(headers, [["1", "add"]], headers, [["1", "mul"]], max_differences=1)

    assert differences == ["Row 2, Column 2 (header: 'OP CODE'): 'add' != 'mul'"]
    assert allowed_headers == set()
