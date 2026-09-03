# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, AsyncIterator, Iterator, TextIO
from argparse import ArgumentParser
from dataclasses import dataclass
from asyncio import StreamReader
from enum import Enum, auto
from sys import stdout
import asyncio
import json
import sys
import re
import os

timeregex = "\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+"
teststart = re.compile("\[\s+RUN\s+\] (.*)")
testend = re.compile("\[\s+([^ ]*)\s+\] (.*) \(.*\)")
testdevices = re.compile(
    f"({timeregex}).*Test \| sender device id: (.*) \((.*), ubb: (.*), chip: (.*)\), receiver device id: (.*) \((.*), ubb: (.*), chip: (.*)\) .*"
)
testcores = re.compile(f"({timeregex}).*Test \|   sender core: (\S*), receiver core: (\S*) \((\S*)\).*")
testprocs = re.compile(f"({timeregex}).*Test \|     running on (.*) .*")
testruns = re.compile(f"({timeregex}).*Test \| Ran (\d+) tests .*")
testbw = re.compile(f"({timeregex}).*Test \|       Bandwidth (.*) Gbps, (.*) ms .*")
testbwfail = re.compile(f"({timeregex}).*Test \|       Expected at least: (.*) Gbps, got (.*) Gbps .*")
testsetup = re.compile(f"({timeregex}).*Test \|       set up .*")
testdone = re.compile(f"({timeregex}).*Test \|     done.*")
testdatacmp = re.compile(
    f"({timeregex}).*Test \|       done comparing bank (.*) and (.*) with (\d+) "
    + "mismatched words starting at (.*), ending at (.*), out of (.*) .*"
)
testl1datacmp = re.compile(
    f"({timeregex}).*Test \|       \[device: (.*), core: (.*)\] (.*) "
    + "mismatched words starting at (.*), ending at (.*), out of (.*) .*"
)
testmissinglinks = re.compile(
    f"({timeregex}).*Test \| missing links: chip\[(.*) \((.*)\), ubb: (.*), chip: (.*)\]: expected (\d*) links, got (\d*) .*"
)
testtimeout = re.compile(f"({timeregex}).*Test \| Timed out! You probably need to reset the device .*")
locinfo = "sdev: \[(.*) \((.*)\), ubb: (.*), chip: (.*)\], rdev: \[(.*) \((.*)\), ubb: (.*), chip: (.*)\], score: \[(.*)\], rcore: \[(.*)\], processor: \[(.*)\], link: \[(.*)\]"
testcheck = re.compile(f"({timeregex}).*Test \| core_check: {locinfo} .*")

print_logs = True
missing_links: dict[str, dict] = {}
exit_status = 0


class TestCase(str, Enum):
    LINK_UP = "LinkUp"
    BANDWIDTH = "Bandwidth"
    BANDWIDTH_BIDIR = "BandwidthBidir"
    DATA_INTEGRITY = "DataIntegrityDram"
    DATA_INTEGRITY_BIDIR = "DataIntegrityDramBidir"
    STRESS_TEST = "StressTest"


@dataclass
class TestedLink:
    src_dev: str
    src_devbdf: str
    src_core: str
    src_ubb: str
    src_chip: str

    dst_dev: str
    dst_devbdf: str
    dst_core: str
    dst_ubb: str
    dst_chip: str

    ltype: str
    proc: str
    bw: dict
    errors: list

    def to_dict(self):
        return self.__dict__


@dataclass
class TestRun:
    name: str
    links: list[TestedLink]
    status: str

    def to_dict(self):
        return {
            "name": self.name,
            "links": [l.to_dict() for l in self.links],
            "status": self.status,
        }


def runs_to_json(runs: list[TestRun], **kwargs) -> str:
    return json.dumps([r.to_dict() for r in runs], **kwargs)


def runs_to_jsonf(fname: str, runs: list[TestRun], **kwargs) -> None:
    with open(fname, "w") as f:
        json.dump([r.to_dict() for r in runs], f, **kwargs)


class EventType(Enum):
    TESTSTART = auto()
    TESTEND = auto()
    DEVICES = auto()
    CORES = auto()
    PROCS = auto()
    RUNS = auto()
    BW = auto()
    BWFAIL = auto()
    DATACMP = auto()
    L1DATACMP = auto()
    TESTSETUP = auto()
    TESTCHECK = auto()
    TESTDONE = auto()
    MISSINGLINKS = auto()
    TIMEOUT = auto()


@dataclass
class Event:
    typ: EventType
    extra: dict

    def to_dict(self):
        return {"typ": self.typ.name, "extra": self.extra}


def parse_teststart(l: str) -> Optional[Event]:
    m = teststart.match(l)
    if m is None:
        return None

    testname = m.group(1)
    # print(f"\tTESTSTART '{testname}'")

    return Event(EventType.TESTSTART, {"name": testname})


def parse_testend(l: str) -> Optional[Event]:
    m = testend.match(l)
    if m is None:
        return None

    status = m.group(1)
    testname = m.group(2)
    # print(f"\tTESTEND '{testname}' '{status}'")

    return Event(EventType.TESTEND, {"name": testname, "status": status})


def parse_testdevices(l: str) -> Optional[Event]:
    m = testdevices.match(l)
    if m is None:
        return None

    extra = {
        "sdev": m.group(2),
        "sdevbdf": m.group(3),
        "subb": m.group(4),
        "schip": m.group(5),
        "rdev": m.group(6),
        "rdevbdf": m.group(7),
        "rubb": m.group(8),
        "rchip": m.group(9),
    }
    # print(f"\tDEVICES {extra}")

    return Event(EventType.DEVICES, extra)


def parse_cores(l: str) -> Optional[Event]:
    m = testcores.match(l)
    if m is None:
        return None

    extra = {
        "score": m.group(2),
        "rcore": m.group(3),
        "type": m.group(4),
    }
    # print(f"\tCORES {extra}")
    return Event(EventType.CORES, extra)


def parse_procs(l: str) -> Optional[Event]:
    m = testprocs.match(l)
    if m is None:
        return None

    proc = m.group(2)
    # print(f"\tPROCS '{proc}'")
    return Event(EventType.PROCS, {"proc": proc})


def parse_runs(l: str) -> Optional[Event]:
    m = testruns.match(l)
    if m is None:
        return None

    runs = int(m.group(2))
    # print(f"\tRUNS {runs}")
    return Event(EventType.RUNS, {"runs": runs})


def parse_bw(l: str) -> Optional[Event]:
    m = testbw.match(l)
    if m is None:
        return None

    bw = float(m.group(2))
    elapsed_ms = float(m.group(3))
    # print(f"\tBANDWIDTH {bw} Gbps, {elapsed_ms} ms")
    return Event(EventType.BW, {"bw": bw, "elapsed_ms": elapsed_ms})


def parse_bwfail(l: str) -> Optional[Event]:
    m = testbwfail.match(l)
    if m is None:
        return None

    expected = float(m.group(2))
    actual = float(m.group(3))
    # print(f"\tBWFAIL {expected} Gbps, {actual} ms")
    return Event(EventType.BWFAIL, {"expected": expected, "actual": actual})


def parse_l1datacmp(l: str) -> Optional[Event]:
    m = testl1datacmp.match(l)
    if m is None:
        return None

    dev = m.group(2)
    core = m.group(3)
    errors = int(m.group(4))
    starting = m.group(5)
    ending = m.group(6)
    total = m.group(7)

    rate = errors * 4 / int(total, 16)
    # print(f"\tL1DATACMP {dev}, {core}, {errors}, {starting}, {ending}, {total}")
    return Event(
        EventType.L1DATACMP,
        {
            "dev": dev,
            "core": core,
            "errors": errors,
            "starting": starting,
            "ending": ending,
            "total": total,
            "rate": rate,
        },
    )


def parse_datacmp(l: str) -> Optional[Event]:
    m = testdatacmp.match(l)
    if m is None:
        return None

    bank0 = int(m.group(2))
    bank1 = int(m.group(3))
    errors = int(m.group(4))
    starting = m.group(5)
    ending = m.group(6)
    total = m.group(7)
    rate = errors * 4 / int(total, 16)

    # print(f"\tDATACMP {bank0}, {bank1}, {errors}, {starting}, {ending}, {total}")
    return Event(
        EventType.DATACMP,
        {
            "bank0": bank0,
            "bank1": bank1,
            "errors": errors,
            "starting": starting,
            "ending": ending,
            "total": total,
            "rate": rate,
        },
    )


def parse_check(l: str) -> Optional[Event]:
    m = testcheck.match(l)
    if m is None:
        return None

    extra = {
        "sdev": m.group(2),
        "sdevbdf": m.group(3),
        "sdevubb": m.group(4),
        "sdevchip": m.group(5),
        "rdev": m.group(6),
        "rdevbdf": m.group(7),
        "rdevubb": m.group(8),
        "rdevchip": m.group(9),
        "score": m.group(10),
        "rcore": m.group(11),
        "proc": m.group(12),
        "link": m.group(13),
    }
    # print(f"\tTESTCHECK {extra}")
    return Event(EventType.TESTCHECK, extra)


def parse_setup(l: str) -> Optional[Event]:
    m = testsetup.match(l)
    if m is None:
        return None

    # print(f"\tSETUP")
    return Event(EventType.TESTSETUP, {})


def parse_timeout(l: str) -> Optional[Event]:
    m = testtimeout.match(l)
    if m is None:
        return None

    # print(f"\tTIMEDOUT")
    return Event(EventType.TIMEOUT, {})


def parse_done(l: str) -> Optional[Event]:
    m = testdone.match(l)
    if m is None:
        return None

    # print(f"\tDONE")
    return Event(EventType.TESTDONE, {})


def parse_missinglinks(l: str) -> Optional[Event]:
    m = testmissinglinks.match(l)
    if m is None:
        return None

    extra = {
        "id": m.group(2),
        "bdf": m.group(3),
        "ubb": m.group(4),
        "chip": m.group(5),
        "expected": m.group(6),
        "got": m.group(7),
    }
    # print(f"\tMISSINGLINKS {extra}")
    return Event(EventType.MISSINGLINKS, extra)


def parse_line(l: str, logf: Optional[TextIO]) -> Optional[Event]:
    if print_logs:
        print(f"l: {l}")

    if logf:
        print(f"l: {l}", file=logf)

    parsers = [
        parse_missinglinks,
        parse_testdevices,
        parse_teststart,
        parse_l1datacmp,
        parse_testend,
        parse_datacmp,
        parse_timeout,
        parse_bwfail,
        parse_cores,
        parse_check,
        parse_setup,
        parse_procs,
        parse_runs,
        parse_done,
        parse_bw,
    ]

    for p in parsers:
        if r := p(l):
            return r

    # print(f"l: {l}")
    return None


async def parse_logs_stream(inf: asyncio.StreamReader, logf: Optional[TextIO]) -> AsyncIterator[Event]:
    while True:
        l = await inf.readline()
        if l == b"":
            break

        r = parse_line(l.decode("utf-8").strip(), logf)
        if r is not None:
            yield r


async def parse_logs(inf: asyncio.StreamReader, logf: Optional[TextIO]) -> list[Event]:
    evs = []
    async for e in parse_logs_stream(inf, logf):
        evs.append(e)

    return evs


def parse_evs(evs: list[Event]) -> Iterator[TestRun]:
    test: str = ""
    sdev: str = ""
    sdevbdf: str = ""
    sdevubb: str = ""
    sdevchip: str = ""
    score: str = ""
    rdev: str = ""
    rdevbdf: str = ""
    rdevubb: str = ""
    rdevchip: str = ""
    rcore: str = ""
    ltype: str = ""
    proc: str = ""
    bw: dict = {}

    runs: list[TestRun] = []
    links: list[TestedLink] = []
    errors: list = []

    stresstest = "MeshDispatchFixture.TensixDeploymentEthernet05StressTest"
    drambidir = "MeshDispatchFixture.TensixDeploymentEthernet04DataIntegrityDramBidir"
    noprocs = [drambidir]

    it = iter(evs)
    for e in it:
        if e.typ == EventType.TESTSTART:
            test = e.extra["name"]
            sdev = sdevbdf = rdev = rdevbdf = score = rcore = ltype = proc = ""
            sdevubb = sdevchip = rdevubb = rdevchip = ""
            links = []
            errors = []
            bw = {}
        elif e.typ == EventType.TESTEND:
            runs.append(TestRun(test, links, e.extra["status"]))
            test = sdev = rdev = score = rcore = ltype = proc = ""
            sdevubb = sdevchip = rdevubb = rdevchip = ""
            links = []
            errors = []
            bw = {}
        elif e.typ == EventType.DEVICES:
            sdev = e.extra["sdev"]
            sdevbdf = e.extra["sdevbdf"]
            sdevubb = e.extra["subb"]
            sdevchip = e.extra["schip"]
            rdev = e.extra["rdev"]
            rdevbdf = e.extra["rdevbdf"]
            rdevubb = e.extra["rubb"]
            rdevchip = e.extra["rchip"]
            score = rcore = ltype = proc = ""
            errors = []
            bw = {}
        elif e.typ == EventType.CORES:
            score = e.extra["score"]
            rcore = e.extra["rcore"]
            ltype = e.extra["type"]
            proc = ""
            errors = []
            bw = {}
        elif e.typ == EventType.PROCS:
            proc = e.extra["proc"]
            bw = {}
            errors = []
        elif e.typ == EventType.BW:
            bw = e.extra
        elif e.typ == EventType.BWFAIL:
            errors.append({"bw": e.extra})
        elif e.typ == EventType.DATACMP:
            errors.append({"data": e.extra})
        elif e.typ == EventType.L1DATACMP:
            errors.append({"data": e.extra})
        elif e.typ == EventType.TESTSETUP:
            sdev = sdevbdf = rdev = rdevbdf = score = rcore = ltype = proc = ""
            sdevubb = sdevchip = rdevubb = rdevchip = ""
            links = []
            errors = []
            bw = {}
        elif e.typ == EventType.TESTCHECK:
            sdev = e.extra["sdev"]
            sdevbdf = e.extra["sdevbdf"]
            sdevubb = e.extra["sdevubb"]
            sdevchip = e.extra["sdevchip"]
            rdev = e.extra["rdev"]
            rdevbdf = e.extra["rdevbdf"]
            rdevubb = e.extra["rdevubb"]
            rdevchip = e.extra["rdevchip"]
            score = e.extra["score"]
            rcore = e.extra["rcore"]
            ltype = e.extra["link"]
            proc = e.extra["proc"]
            errors = []
            bw = {}
        elif e.typ == EventType.TESTDONE:
            links.append(
                TestedLink(
                    sdev,
                    sdevbdf,
                    score,
                    sdevubb,
                    sdevchip,
                    rdev,
                    rdevbdf,
                    rcore,
                    rdevubb,
                    rdevchip,
                    ltype,
                    proc,
                    bw,
                    errors,
                )
            )
        elif e.typ == EventType.MISSINGLINKS:
            global missing_links
            missing_links[e.extra["id"]] = e.extra
        elif e.typ == EventType.TIMEOUT:
            print("The test timed out, you should reset the card! (tt-smi -r)", file=sys.stderr)
            exit(1)

    yield from runs


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


async def file_to_streamreader(path: str, chunk_size: int = 8192) -> StreamReader:
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


async def parse_file(file: str, logf: Optional[TextIO]) -> list[Event]:
    sr = await file_to_streamreader(file)
    return await parse_logs(sr, logf)


async def main():
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
    opts = parser.parse_args()

    logpath = opts.f if opts.f else None
    logf = open(logpath, "w") if logpath else None

    tests = [TestCase.LINK_UP, TestCase.BANDWIDTH_BIDIR]

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

    if opts.i:
        evs = await parse_file(opts.i, logf)
    else:
        print("Running tests: ", ", ".join(t.value for t in tests))

        filters = prepare_filter([t for t in tests])
        program = "build/test/tt_metal/unit_tests_deployment"
        args = [f"--gtest_filter={filters}"]

        env = os.environ.copy()
        env["ETH_TEST_EXPECTED_LINKS"] = str(10)
        if not opts.v:
            env["TT_LOGGER_TYPES"] = "Test"

        if opts.c:
            env["TT_MESH_GRAPH_DESC_PATH"] = opts.c

        proc = await asyncio.create_subprocess_exec(program, *args, stdout=asyncio.subprocess.PIPE, env=env)

        p, evs = await asyncio.gather(proc.wait(), parse_logs(proc.stdout, logf))

    # pprint.pp(evs)
    runs = list(parse_evs(evs))
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
