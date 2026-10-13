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
        print(f"perf baseline WARN: {unknown} not in the power table; treating them as 8kw", file=sys.stderr)
    cls = "14kw" if names and all(power.get(host) == "14kw" for host in names) else "8kw"
    if cls not in entry:
        print(f"perf baseline: none for {model} {config} on {cls}", file=sys.stderr)
        return None
    print(f"perf baseline: {model} {config} on {cls} {entry['probe']}: {entry[cls]}", file=sys.stderr)
    metrics = {metric: [value, table["margin"][metric]] for metric, value in entry[cls].items()}
    return {"probe": entry["probe"], "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(
        description="Print a run's baseline as JSON {probe, metrics: {name: [expected, margin]}}, or nothing."
    )
    parser.add_argument("table")
    parser.add_argument("model")
    parser.add_argument("config")
    parser.add_argument("hosts", help="comma-separated hostnames")
    args = parser.parse_args()
    with open(args.table) as f:
        baseline = lookup(json.load(f), args.model, args.config, args.hosts)
    if baseline is not None:
        print(json.dumps(baseline))


if __name__ == "__main__":
    main()
