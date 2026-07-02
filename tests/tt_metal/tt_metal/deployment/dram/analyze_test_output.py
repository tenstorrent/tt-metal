# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, asdict
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import pprint
import json
import csv

patterns = [
    "Checkerboard",
    "Address",
    "MarchingOnes",
    "MarchingZeroes",
    "MarchingOneBits",
    "MarchingZeroBits",
    "ToggleBits",
    "Saturation",
    "ReversibleRandom",
    "Random",
    "RandomXoshiro128pp",
    "ByteWiseSsn",
]

columns = [
    "test",
    "chip",
    "bdf",
    "bank",
    "status",
    "error rate",
] + patterns


@dataclass
class Bank:
    errors: int
    total_bytes: int
    passed: bool
    patterns: dict


@dataclass
class Chip:
    id: str
    bdf: str
    banks: dict[str, Bank]


@dataclass
class Test:
    name: str
    short_name: str
    passed: bool
    chips: dict[str, Chip]


# {'bank': 0,
#  'bdf': '0000:01:00.0',
#  'core': '(x=7,y=0)',
#  'dev_id': '0',
#  'failures': 20,
#  'first_fail_classified_as': 'read',
#  'pass': 0,
#  'pattern': 'Random',
#  'read_failures': 1,
#  'repeat': 0,
#  'words_checked': 7168,
#  'write_failures': 0},


def extract_errors(errors: list[dict], chip: str, bank: str) -> dict:
    ret: dict = {}
    for p in patterns:
        ret[p] = []
        for e in errors:
            if e["dev_id"] != chip:
                continue
            if e["bank"] != int(bank):
                continue
            if e["pattern"] != p:
                continue

            ret[p].append(e)
    return ret


def process_test(test: dict) -> Test:
    global patterns

    chips: dict[str, Chip] = {
        str(i): Chip(str(i), "", {str(i): Bank(0, 1, True, {}) for i in range(8)}) for i in range(32)
    }
    for c, ch in test["chips"].items():
        bdf = ch["bdf"]
        banks: dict[str, Bank] = {}
        for b, bank in ch["banks"].items():
            pat: dict = extract_errors(test["errors"], c, b)
            errors = bank["read_errors"] + bank["write_errors"]
            banks[b] = Bank(errors, bank["checked_bytes"], errors == 0, pat)

        chips[c] = Chip(c, bdf, banks)

    return Test(test["name"], test["name"].split("_")[1], test["status"] != "FAILED", chips)


def prepare_rows(tests: list[Test]):
    rows = []
    for t in tests:
        for c in t.chips:
            ch = t.chips[c]
            for b in ch.banks:
                bank = ch.banks[b]
                error_rate = bank.errors / bank.total_bytes * 1e6
                row = [
                    t.short_name,
                    ch.id,
                    ch.bdf,
                    b,
                    "OK" if bank.passed else "FAIL",
                    f"{error_rate:.2f}ppm" if error_rate else "0",
                ]

                row += ["FAIL" if bank.patterns.get(p, False) else "OK" for p in patterns]

                rows.append(row)

    return rows


def write_csv(path: str | Path, rows: list[list], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    inf = "out.json"
    outf = "out.csv"

    parser = ArgumentParser()
    parser.add_argument("-i", type=str, help="Input JSON file to analyze")
    parser.add_argument("-o", type=str, help="Output file to write CSV data to")
    opts = parser.parse_args()

    if opts.i:
        inf = opts.i

    if opts.o:
        outf = opts.o

    print(f"Reading {inf} and writing to {outf}")

    with open(inf, "r") as f:
        data = json.load(f)

    tests = [process_test(t) for t in data]
    rows = prepare_rows(tests)
    write_csv(outf, rows, columns)


if __name__ == "__main__":
    main()
