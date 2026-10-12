# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import argparse
import json
import sys


def lookup(table, model, config, hosts):
    entry = table["baselines"].get(model, {}).get(config)
    if entry is None:
        return None
    power = {host: cls for cls, members in table["power"].items() for host in members}
    names = [host.split(".")[0] for host in hosts.split(",") if host]
    unknown = [host for host in names if host not in power]
    if unknown:
        print(f"TPS baseline WARN: {unknown} not in the power table; treating them as 8kw", file=sys.stderr)
    # A pipeline runs at its slowest galaxy's pace, so one 8kw host makes the whole set 8kw.
    cls = "14kw" if names and all(power.get(host) == "14kw" for host in names) else "8kw"
    if cls not in entry:
        print(f"TPS baseline: none for {model} {config} on {cls}", file=sys.stderr)
        return None
    print(f"TPS baseline: {model} {config} on {cls}: {entry[cls]} tok/s {entry['probe']}", file=sys.stderr)
    return entry[cls], entry["probe"], table["margin"]


def main():
    parser = argparse.ArgumentParser(description="Print '<tok/s> <probe> <margin>' for a run's hosts, or nothing.")
    parser.add_argument("table")
    parser.add_argument("model")
    parser.add_argument("config")
    parser.add_argument("hosts", help="comma-separated hostnames")
    args = parser.parse_args()
    with open(args.table) as f:
        baseline = lookup(json.load(f), args.model, args.config, args.hosts)
    if baseline is not None:
        print(*baseline)


if __name__ == "__main__":
    main()
