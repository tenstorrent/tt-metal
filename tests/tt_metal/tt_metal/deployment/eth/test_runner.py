# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from result_parser import TestRun, TestedLink, process_logs

from argparse import ArgumentParser
from typing import Optional, TextIO
from asyncio import StreamReader
from threading import Thread
from queue import Queue
from time import sleep
from sys import stdout
from enum import Enum
import requests
import asyncio
import json
import re
import os

print_logs = True
missing_links: dict[str, dict] = {}


class TestCase(str, Enum):
    LINK_UP = "LinkUp"
    BANDWIDTH = "Bandwidth"
    BANDWIDTH_BIDIR = "BandwidthBidir"
    DATA_INTEGRITY = "DataIntegrityDram"
    DATA_INTEGRITY_BIDIR = "DataIntegrityDramBidir"
    STRESS_TEST = "StressTest"


class DistTestCase(str, Enum):
    BANDWIDTH = "DistBandwidth"


def runs_to_json(runs: list[TestRun], **kwargs) -> str:
    return json.dumps([r.to_dict() for r in runs], **kwargs)


def runs_to_jsonf(fname: str, runs: list[TestRun], **kwargs) -> None:
    with open(fname, "w") as f:
        json.dump([r.to_dict() for r in runs], f, **kwargs)


failed_requests = 0


def batch_worker(q: Queue, s: requests.Session, http_post_url: Optional[str]):
    done = False
    while not done:
        batch = []
        while not q.empty():
            data = q.get_nowait()
            if data == "DONE":
                done = True
                break
            batch.append(data)

        if batch and http_post_url:
            try:
                s.post(http_post_url, json=batch)
            except Exception as e:
                print(f"Batch request failed: {e}")
                global failed_requests
                failed_requests += 1
        else:
            sleep(0.1)


def prepare_filter(tests: list[TestCase]) -> str:
    return ":".join(map(lambda x: f"*TensixDeploymentEthernet*{x}", tests))


def table(title: str, headers: list[str], rows: list[list]) -> str:
    rows_str = [[str(x) for x in row] for row in rows]
    all_rows = [headers] + rows_str
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = []
    out.append("")
    out.append(title)
    out.append(sep)
    out.append("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    out.append(sep)
    for row in rows_str:
        out.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    out.append(sep)
    return "\n".join(out)


def shortname(n: str) -> str:
    return n if "." not in n else n.split(".")[1]


def print_summary(runs: list[TestRun], logf: TextIO):
    headers = ["test name", "status"]
    rows = []
    for r in runs:
        rows.append([shortname(r.name), r.status])

    print_table(table("Test summary", headers, rows), logf)


def link_name(l: TestedLink) -> str:
    names = [
        f"{l.src_dev}:{l.src_core}",
        f"{l.dst_dev}:{l.dst_core}",
    ]
    return "->".join(sorted(names))


def format_rate(r: float) -> str:
    s = f"{r * 1e6:.1f}ppm"

    if s == "0.0ppm" and r > 0.0:
        return "<0.1ppm"

    return s


def format_fail(fail: bool) -> str:
    return "FAIL" if fail else "OK"


refail = re.compile("(FAILED|FAIL)")


def print_table(t: str, logf: TextIO):
    if not logf.isatty():
        print(t, file=logf)
    else:
        print(re.sub(refail, "\x1b[31m\\1\x1b[0m", t), file=logf)


def print_test_summary(t: TestCase, runs: list[TestRun], logf: TextIO):
    for r in runs:
        if not r.name.endswith(t):
            continue

        links: dict[str, dict] = {}
        bw = False

        for l in r.links:
            name = link_name(l)
            if name not in links:
                links[name] = {"count": 0, "pass": True, "errors": [], "bw": 0.0, "ltype": l.ltype}

            links[name]["count"] += 1
            if "bw" in l.bw:
                links[name]["bw"] += l.bw["bw"]
                bw = True

            links[name]["errors"] += l.errors

        headers = ["link name", "link type", "test count"]
        if bw:
            headers += ["avg bw"]
        headers += ["errors", "error rate", "pass"]
        rows = []
        for k in sorted(links.keys()):
            row = [
                k,
                links[k]["ltype"],
                links[k]["count"],
            ]

            if bw:
                row += [f'{links[k]["bw"] / links[k]["count"]:.3f}']

            rate = 0
            for e in links[k]["errors"]:
                if "data" not in e:
                    continue

                rate = max(rate, e["data"]["rate"])

            row += [len(links[k]["errors"]), format_rate(rate), format_fail(bool(links[k]["errors"]))]
            rows.append(row)

        print_table(table(f"{t} test summary", headers, rows), logf)


def print_test_summary_per_chip(t: TestCase, runs: list[TestRun], logf: TextIO):
    for r in runs:
        if not r.name.endswith(t):
            continue

        chips: dict = {}
        have_bws = False

        def ensure_dev(ch: str):
            if ch not in chips:
                chips[ch] = {"bdf": "", "tests": 0, "bws": [], "errors": 0, "ubb": -1, "chip": -1}

        for l in r.links:
            ensure_dev(l.src_dev)
            ensure_dev(l.dst_dev)

            chips[l.src_dev]["tests"] += 1
            if l.src_dev != l.dst_dev:
                chips[l.dst_dev]["tests"] += 1
            if len(l.bw):
                have_bws = True
                chips[l.src_dev]["bws"].append(l.bw["bw"])
                if l.src_dev != l.dst_dev:
                    chips[l.dst_dev]["bws"].append(l.bw["bw"])

            chips[l.src_dev]["errors"] += len(l.errors)
            if l.src_dev != l.dst_dev:
                chips[l.dst_dev]["errors"] += len(l.errors)

            chips[l.src_dev]["bdf"] = l.src_devbdf
            if l.src_dev != l.dst_dev:
                chips[l.dst_dev]["bdf"] = l.dst_devbdf

            chips[l.src_dev]["ubb"] = l.src_ubb
            if l.src_dev != l.dst_dev:
                chips[l.dst_dev]["ubb"] = l.dst_ubb

            chips[l.src_dev]["chip"] = l.src_chip
            if l.src_dev != l.dst_dev:
                chips[l.dst_dev]["chip"] = l.dst_chip

        avg = lambda x: sum(x) / len(x)
        headers = ["chip id", "chip bdf", "ubb", "chip id", "tests"]
        if have_bws:
            headers += ["bw (min)", "bw (max)", "bw (avg)"]
        headers += ["errors", "status"]

        rows = []
        for c in sorted(chips.keys(), key=int):
            ch = chips[c]
            bdf = ch["bdf"]
            bws = ch["bws"]
            ubb = ch["ubb"]
            chip = ch["chip"]
            err = ch["errors"]
            msg = format_fail(err > 0)

            row = [c, bdf, ubb, chip, ch["tests"]]
            if have_bws:
                row += [f"{min(bws):.3f}", f"{max(bws):.3f}", f"{avg(bws):.3f}"]
            row += [err, msg]

            rows.append(row)

        rows.sort(key=lambda x: x[1])

        print_table(table(f"{t} per chip test summary", headers, rows), logf)


def print_failing(runs: list[TestRun], logf: TextIO):
    headers = ["test name", "link", "direction", "processor", "errors"]
    rows = []
    for r in runs:
        for l in r.links:
            if len(l.errors):
                name = link_name(l)
                d = f"{l.src_dev}->{l.dst_dev}"
                errors = "; ".join(json.dumps(e) for e in l.errors)
                # print(name, l)
                rows.append([r.name, name, d, l.proc, errors])

    print_table(table("Failing tests/links", headers, rows), logf)


def print_missing_links(logf: TextIO = stdout):
    global missing_links
    links = sorted([v for v in missing_links.values()], key=lambda x: x["bdf"])

    headers = ["chip id", "bdf", "ubb", "chip number", "expected", "found"]
    rows = []
    for l in links:
        rows.append(
            [
                l["id"],
                l["bdf"],
                l["ubb"],
                l["chip"],
                l["expected"],
                l["got"],
            ]
        )
    print_table(table("Missing links", headers, rows), logf)


def print_results(runs: list[TestRun], logf: TextIO = stdout):
    global missing_links
    if len(missing_links):
        print_missing_links(logf)
        print_summary(runs, logf)
        return

    print_failing(runs, logf)
    for t in TestCase:
        print_test_summary(t, runs, logf)
    for t in TestCase:
        print_test_summary_per_chip(t, runs, logf)
    print_summary(runs, logf)


async def file_to_streamreader(path: str, chunk_size: int = 16) -> StreamReader:
    loop = asyncio.get_running_loop()
    reader = StreamReader()

    def feed_blocking():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                data = chunk
                loop.call_soon_threadsafe(reader.feed_data, data)
        loop.call_soon_threadsafe(reader.feed_eof)

    loop.run_in_executor(None, feed_blocking)
    return reader


async def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-t", type=str, help="Comma separated list of tests to run")
    parser.add_argument("-l", action="store_true", help="List all of the available tests")
    parser.add_argument("-q", action="store_true", help="Don't print the logs as the tests run")
    parser.add_argument("-n", action="store_true", help="Don't print the summary")
    parser.add_argument("-v", action="store_true", help="Verbose output")
    parser.add_argument("-i", type=str, help="Input file to parse instead of running the tests")
    parser.add_argument("-o", type=str, help="Output path for the json file")
    parser.add_argument("-f", type=str, help="File to use for logging")
    parser.add_argument("-c", type=str, help="Path to the MGD config file to use")
    parser.add_argument("-d", type=str, help="Optional host to send the test data to")
    opts = parser.parse_args()

    logpath = opts.f if opts.f else None
    logf = open(logpath, "w") if logpath else None

    tests = [TestCase.LINK_UP, TestCase.BANDWIDTH_BIDIR]
    exit_status = 0

    http_post_url: Optional[str] = None

    if opts.l:
        print("Available tests:")
        for t in (t.value for t in TestCase):
            print(f"\t{t}")

        print("Default tests:", ",".join(t.value for t in tests))

        exit()

    if opts.q:
        global print_logs
        print_logs = False

    if opts.t:
        if opts.t == "all":
            tests = list(TestCase)
        else:
            tests = [TestCase(t) for t in opts.t.split(",")]

    outfile = opts.o if opts.o else "out.json"
    print(f"Writing results to '{outfile}'")
    if logpath:
        print(f"Writing logs to '{logpath}'")

    if opts.d:
        http_post_url = opts.d
        print("Sending data to:", http_post_url)

    if opts.i:
        evs = await parse_file(opts.i, logf)
    else:
        print("Running tests: ", ", ".join(t.value for t in tests))

        filters = prepare_filter([t for t in tests])
        program = "build/test/tt_metal/unit_tests_deployment"
        args = [f"--gtest_filter={filters}"]

        env = os.environ.copy()
        env["TT_METAL_SLOW_DISPATCH_MODE"] = "1"
        # env["ETH_TEST_EXPECTED_LINKS"] = os.environ.get("ETH_TEST_EXPECTED_LINKS", str(10))
        if not opts.v:
            env["TT_LOGGER_TYPES"] = "Test"

        if opts.c:
            env["TT_MESH_GRAPH_DESC_PATH"] = opts.c

        proc = await asyncio.create_subprocess_exec(program, *args, stdout=asyncio.subprocess.PIPE, env=env)

        if not proc.stdout:
            print("Couldn't get the test's stdout")  # TODO better logging
            exit(1)

        q: Queue = Queue()
        s: requests.Session = requests.Session()
        Thread(target=batch_worker, daemon=True, args=[q, s, http_post_url]).start()
        exit_status, runs = await asyncio.gather(proc.wait(), process_logs(proc.stdout, logf, q))
        q.put("DONE")

    # pprint.pp(evs)
    # runs = list(e async for e in parse_evs(evs))
    # pprint.pp(runs)
    # print(runs_to_json(runs, sort_keys=True, indent=4))

    if not opts.n:
        print_results(runs)
    if logf:
        print_results(runs, logf)

    runs_to_jsonf(outfile, runs, sort_keys=True, indent=4)

    if logf:
        logf.close()

    exit(exit_status)


if __name__ == "__main__":
    asyncio.run(main())
