#!/usr/bin/env python3
"""Parse Ethernet Link Metrics log files and emit a per-run, per-link CSV.

Each row in the output is one (run, link) pair, organized run-by-run and then
link-by-link within each run. Each link is identified by the tuple
(hostname, tray, ASIC, channel, port type).

Usage
-----
    python3 ethernet_metrics_to_csv.py \\
        --output ethernet_metrics.csv \\
        [LABEL=]path/to/log1.txt  [LABEL=]path/to/log2.txt ...

A positional argument may include a ``LABEL=`` prefix to give that log an
explicit run name in the CSV. Without a prefix, the label defaults to the
filename stem (e.g. ``log_revc_2.txt`` becomes ``log_revc_2``).

CSV columns
-----------
    run, host, tray, asic, channel, port_id, port_type,
    txq0_resends, txq1_resends, txq2_resends, total_txq_resends,
    corrected_cw, uncorrected_cw, crc_err, retrains, unique_id
"""

import argparse
import csv
import os
import re
import sys


ROW_PAT = re.compile(
    r"(?:\]<stdout>:)?(bh-glx\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+?)\s*"
    r"(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+"
    r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s*B\s+\d+\s*B"
)


def parse_rows(path):
    """Yield one dict per ethernet-metrics row in *path*.

    Handles the quirk in the log format where the board Unique_ID and
    CRC_Err columns are printed glued together into a single hex token
    of the form ``0x<uid_hex><crc_hex>``. The board UID is the leading
    hex chars up to length 16; the CRC_Err is the trailing hex digits
    (in practice always a single digit in observed logs).
    """
    with open(path) as f:
        for line in f:
            m = ROW_PAT.search(line)
            if not m:
                continue
            host = m.group(1)
            tray = int(m.group(2))
            asic = int(m.group(3))
            channel = int(m.group(4))
            port_id = int(m.group(5))
            port_type = m.group(6).strip()

            # First hex token is UID+CRC_Err concatenated.
            uid_crc_token = m.group(7)
            hex_payload = uid_crc_token[2:]  # strip "0x"
            # Board UID is 16 hex chars; anything trailing is CRC_Err.
            if len(hex_payload) >= 16:
                unique_id_hex = "0x" + hex_payload[:16]
                crc_hex = hex_payload[16:] or "0"
            else:
                # UID shorter than 16 hex chars — treat the whole token as UID
                # and CRC_Err as 0. (Observed in some logs where leading
                # zero nibbles are suppressed.)
                unique_id_hex = uid_crc_token
                crc_hex = "0"
            crc_err = int(crc_hex, 16)

            retrains = int(m.group(8), 16)
            corrected_cw = int(m.group(9), 16)
            uncorrected_cw = int(m.group(10), 16)

            txq0 = int(m.group(11))
            txq1 = int(m.group(12))
            txq2 = int(m.group(13))

            yield {
                "host": host,
                "tray": tray,
                "asic": asic,
                "channel": channel,
                "port_id": port_id,
                "port_type": port_type,
                "unique_id": unique_id_hex,
                "retrains": retrains,
                "crc_err": crc_err,
                "corrected_cw": corrected_cw,
                "uncorrected_cw": uncorrected_cw,
                "txq0_resends": txq0,
                "txq1_resends": txq1,
                "txq2_resends": txq2,
                "total_txq_resends": txq0 + txq1 + txq2,
            }


def sort_key(row):
    """Stable link ordering within a run: host, tray, asic, channel, port_id."""
    return (row["host"], row["tray"], row["asic"], row["channel"], row["port_id"])


def parse_log_spec(spec):
    if "=" in spec:
        label, path = spec.split("=", 1)
        return (label.strip() or os.path.splitext(os.path.basename(path))[0], path)
    return (os.path.splitext(os.path.basename(spec))[0], spec)


COLUMNS = [
    "run",
    "host",
    "tray",
    "asic",
    "channel",
    "port_id",
    "port_type",
    "txq0_resends",
    "txq1_resends",
    "txq2_resends",
    "total_txq_resends",
    "corrected_cw",
    "uncorrected_cw",
    "crc_err",
    "retrains",
    "unique_id",
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "logs",
        nargs="+",
        metavar="[LABEL=]PATH",
        help="One or more log files. Optional LABEL= prefix sets the run name; "
        "without it, the filename stem is used.",
    )
    ap.add_argument("-o", "--output", required=True, help="Path to output CSV file")
    args = ap.parse_args(argv)

    logs = [parse_log_spec(s) for s in args.logs]

    total_rows = 0
    with open(args.output, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()

        for label, path in logs:
            if not os.path.exists(path):
                print(f"WARN: missing {path}", file=sys.stderr)
                continue

            # Sort within the run for consistent ordering
            rows = sorted(parse_rows(path), key=sort_key)
            if not rows:
                print(f"WARN: no ethernet-metrics rows found in {path}", file=sys.stderr)
                continue

            for r in rows:
                writer.writerow({"run": label, **{k: r[k] for k in COLUMNS if k != "run"}})

            total_rows += len(rows)
            print(f"  {label:20s}  {path}  ({len(rows):,} links)")

    print(f"\nWrote {total_rows:,} rows across {len(logs)} runs to {args.output}")


if __name__ == "__main__":
    main()
