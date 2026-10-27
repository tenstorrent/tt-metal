# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

TEST_RENAME = {
    "MeshDispatchFixture.TensixDeploymentEthernet00LinkUp": "LINKUP",
    "MeshDispatchFixture.TensixDeploymentEthernet01Bandwidth": "BW",
    "MeshDispatchFixture.TensixDeploymentEthernet02BandwidthBidir": "BWBIDIR",
    "MeshDispatchFixture.TensixDeploymentEthernet03DataIntegrityDram": "INTEGRITY",
    "MeshDispatchFixture.TensixDeploymentEthernet04DataIntegrityDramBidir": "INTEGRITYBIDIR",
    "MeshDispatchFixture.TensixDeploymentEthernet05StressTest": "STRESS",
}

EXPECTED_TESTS = [
    "LINKUP",
    "BW",
    "BWBIDIR",
    "INTEGRITY",
    "INTEGRITYBIDIR",
    "STRESS",
]

RAW_COLUMNS = [
    "LinkID",
    "NormalizedLinkID",
    "LinkType",
    "Test",
    "TestStatus",
    "PassPercent",
    "SrcUBB",
    "DstUBB",
    "SrcDev",
    "DstDev",
    "SrcLocalChip",
    "DstLocalChip",
    "SrcCore",
    "DstCore",
    "Processor",
    "ProcessorShort",
    "Bandwidth",
    "ElapsedMs",
    "ErrorCount",
    "ErrorsJson",
]

INDEX_COLUMNS = [
    "LinkID",
    "NormalizedLinkID",
    "LinkType",
    "SrcUBB",
    "DstUBB",
    "SrcDev",
    "DstDev",
    "SrcLocalChip",
    "DstLocalChip",
    "SrcCore",
    "DstCore",
    "ProcessorShort",
]

FINAL_COLUMNS = INDEX_COLUMNS + EXPECTED_TESTS + [f"{test}_BW" for test in EXPECTED_TESTS]


def ubb_id(dev: int) -> int:
    if 0 <= dev <= 7:
        return 1
    if 8 <= dev <= 15:
        return 2
    if 16 <= dev <= 23:
        return 3
    if 24 <= dev <= 31:
        return 4
    return -1


def local_chip(dev: int) -> int:
    return dev % 8


def proc_short(proc: str) -> str:
    if proc.endswith("RISCV_0"):
        return "ERISC0"
    if proc.endswith("RISCV_1"):
        return "ERISC1"
    return proc


def normalize_internal_key(link: dict[str, Any]) -> tuple[Any, ...]:
    """Same physical link pattern across UBBs gets same key."""
    src = int(link["src_dev"])
    dst = int(link["dst_dev"])

    return (
        local_chip(src),
        link.get("src_core"),
        local_chip(dst),
        link.get("dst_core"),
        proc_short(link.get("proc", "")),
    )


def normalize_ubb_to_ubb_key(link: dict[str, Any]) -> tuple[Any, ...]:
    """UBB-to-UBB links keep actual UBB relationship."""
    src = int(link["src_dev"])
    dst = int(link["dst_dev"])
    proc = link.get("proc", "ERISC0")

    return (
        ubb_id(src),
        local_chip(src),
        link.get("src_core"),
        ubb_id(dst),
        local_chip(dst),
        link.get("dst_core"),
        proc_short(proc),
    )


def pass_percent(errors: list[Any]) -> float:
    """
    Per result:
    - 100% if no errors
    - 0% if errors exist

    Later we average these per LinkID/test.
    """
    if errors:
        return (1 - max(map(lambda x: x["data"]["rate"] if "data" in x else 0.0, errors))) * 100.0
    return 100.0


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_raw_rows(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    internal_id_map: dict[tuple[Any, ...], int] = {}
    external_id_map: dict[tuple[Any, ...], int] = {}
    rows: list[dict[str, Any]] = []

    for test in data:
        raw_name = test.get("name", "")
        test_name = TEST_RENAME.get(raw_name, raw_name)
        test_status = test.get("status", "")

        for link in test.get("links", []):
            src_dev = int(link["src_dev"])
            dst_dev = int(link["dst_dev"])
            if not link["proc"]:
                link["proc"] = "ERISC0"

            src_ubb = ubb_id(src_dev)
            dst_ubb = ubb_id(dst_dev)

            is_internal = src_ubb == dst_ubb
            link_type = "UBB_INTERNAL" if is_internal else "UBB_TO_UBB"

            if is_internal:
                key = normalize_internal_key(link)
                if key not in internal_id_map:
                    internal_id_map[key] = len(internal_id_map) + 1

                base_id = internal_id_map[key]
                link_id = f"UBB{src_ubb}-{base_id:03d}"
                normalized_link_id = f"INTERNAL-{base_id:03d}"
            else:
                key = normalize_ubb_to_ubb_key(link)
                if key not in external_id_map:
                    external_id_map[key] = len(external_id_map) + 1

                base_id = external_id_map[key]
                link_id = f"X-{base_id:03d}"
                normalized_link_id = f"UBB_TO_UBB-{base_id:03d}"

            bw = link.get("bw", {}) or {}
            errors = link.get("errors", []) or []

            rows.append(
                {
                    "LinkID": link_id,
                    "NormalizedLinkID": normalized_link_id,
                    "LinkType": link_type,
                    "Test": test_name,
                    "TestStatus": test_status,
                    "PassPercent": pass_percent(errors),
                    "SrcUBB": src_ubb,
                    "DstUBB": dst_ubb,
                    "SrcDev": src_dev,
                    "DstDev": dst_dev,
                    "SrcLocalChip": local_chip(src_dev),
                    "DstLocalChip": local_chip(dst_dev),
                    "SrcCore": link.get("src_core"),
                    "DstCore": link.get("dst_core"),
                    "Processor": link.get("proc"),
                    "ProcessorShort": proc_short(link.get("proc", "")),
                    "Bandwidth": bw.get("bw"),
                    "ElapsedMs": bw.get("elapsed_ms"),
                    "ErrorCount": len(errors),
                    "ErrorsJson": json.dumps(errors),
                }
            )

    return rows


def build_final_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    pass_values: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    bw_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in raw_rows:
        group_key = tuple(row[col] for col in INDEX_COLUMNS)
        test_name = row["Test"]
        link_id = row["LinkID"]

        if group_key not in grouped:
            grouped[group_key] = {col: row[col] for col in INDEX_COLUMNS}

        pass_value = row.get("PassPercent")
        if pass_value is not None:
            pass_values[group_key][test_name].append(float(pass_value))

        bandwidth = row.get("Bandwidth")
        if bandwidth is not None:
            bw_values[link_id][test_name].append(float(bandwidth))

    final_rows: list[dict[str, Any]] = []
    for group_key, base_row in grouped.items():
        output_row = dict(base_row)
        link_id = output_row["LinkID"]

        for test in EXPECTED_TESTS:
            output_row[test] = mean(pass_values[group_key].get(test, []))
            output_row[f"{test}_BW"] = mean(bw_values[link_id].get(test, []))

        final_rows.append(output_row)

    final_rows.sort(key=lambda row: row["LinkID"])
    return final_rows


def parse_json_to_csvs(
    json_path: str | Path,
    final_csv: str | Path = "parsed_link_results.csv",
    raw_csv: str | Path = "parsed_link_results_raw.csv",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    json_path = Path(json_path)

    with json_path.open("r") as f:
        data = json.load(f)

    raw_rows = build_raw_rows(data)
    final_rows = build_final_rows(raw_rows)

    write_csv(final_csv, final_rows, FINAL_COLUMNS)
    write_csv(raw_csv, raw_rows, RAW_COLUMNS)

    return final_rows, raw_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse link test JSON results and write summary/raw CSV files.")
    parser.add_argument(
        "--input",
        "-i",
        default="out.json",
        help="Input JSON file path. Default: %(default)s",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="parsed_link_results.csv",
        help="Final LinkID summary CSV output path. Default: %(default)s",
    )
    parser.add_argument(
        "--raw-output",
        default="parsed_link_results_raw.csv",
        help="Raw expanded results CSV output path. Default: %(default)s",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    final_rows, raw_rows = parse_json_to_csvs(
        json_path=args.input,
        final_csv=args.output,
        raw_csv=args.raw_output,
    )

    print(f"Created {args.output}")
    print(f"Created {args.raw_output}")
    print(f"Final LinkID rows: {len(final_rows)}")
    print(f"Raw test-result rows: {len(raw_rows)}")
