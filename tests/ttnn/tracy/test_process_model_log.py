# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import csv

from tracy.process_model_log import merge_pass_csv

HEADERS = [
    "GLOBAL CALL COUNT",
    "METAL TRACE ID",
    "METAL TRACE REPLAY SESSION ID",
    "DEVICE ID",
    "OP CODE",
]
L1_1 = "L1_1 ACCUM [%]"


def _write_csv(path, headers, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def test_merge_pass_csv_keeps_devices_separate(tmp_path):
    pass1 = tmp_path / "pass1.csv"
    pass2 = tmp_path / "pass2.csv"
    _write_csv(
        pass1,
        HEADERS,
        [["0", "", "", "0", "add"], ["0", "", "", "1", "add"]],
    )
    _write_csv(
        pass2,
        HEADERS + [L1_1],
        [["0", "", "", "0", "add", "10"], ["0", "", "", "1", "add", "30"]],
    )

    merge_pass_csv(pass1, pass2)

    with pass1.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert [(row["DEVICE ID"], row[L1_1]) for row in rows] == [("0", "10"), ("1", "30")]
