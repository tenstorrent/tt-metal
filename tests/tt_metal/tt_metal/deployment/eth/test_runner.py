from typing import Optional, AsyncIterator, Iterator
from dataclasses import dataclass, asdict
from argparse import ArgumentParser
from asyncio import StreamReader
from dateutil import parser
from enum import Enum, auto
import fileinput
import asyncio
import pprint
import json
import re

timeregex = "\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+"
teststart = re.compile("\[\s+RUN\s+\] (.*)")
testend = re.compile("\[\s+([^ ]*)\s+\] (.*) \(.*\)")
testdevices = re.compile(
    f"({timeregex}).*Test \| sender device id: (.*) \((.*)\), receiver device id: (.*) \((.*)\) .*"
)
testcores = re.compile(f"({timeregex}).*Test \|   sender core: (.*), receiver core: (.*) .*")
testprocs = re.compile(f"({timeregex}).*Test \|     running on (.*) .*")
testruns = re.compile(f"({timeregex}).*Test \| Ran (\d+) tests .*")
testbw = re.compile(f"({timeregex}).*Test \|       Bandwidth (.*) Gbps, (.*) ms .*")
testbwfail = re.compile(f"({timeregex}).*Test \|       Expected at least: (.*) Gbps, got (.*) Gbps .*")
testsetup = re.compile(f"({timeregex}).*Test \|       set up .*")
testdatacmp = re.compile(
    f"({timeregex}).*Test \|       done comparing bank (.*) and (.*) with (\d+) "
    + "mismatched words starting at (.*), ending at (.*) .*"
)
testl1datacmp = re.compile(
    f"({timeregex}).*Test \|       \[device: (.*), core: (.*)\] (.*) "
    + "mismatched words starting at (.*), ending at (.*) .*"
)
locinfo = "sdev: \[(.*) \((.*)\)\], rdev: \[(.*) \((.*)\)\], score: \[(.*)\], rcore: \[(.*)\], processor: \[(.*)\]"
testcheck = re.compile(f"({timeregex}).*Test \| core_check: {locinfo} .*")

print_logs = True


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

    dst_dev: str
    dst_devbdf: str
    dst_core: str

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

    sdev = m.group(2)
    sdevbdf = m.group(3)
    rdev = m.group(4)
    rdevbdf = m.group(5)
    # print(f"\tDEVICES s: {sdev} ({sdevbdf}), r: {rdev} ({rdevbdf})")

    return Event(
        EventType.DEVICES,
        {
            "sdev": sdev,
            "sdevbdf": sdevbdf,
            "rdev": rdev,
            "rdevbdf": rdevbdf,
        },
    )


def parse_cores(l: str) -> Optional[Event]:
    m = testcores.match(l)
    if m is None:
        return None

    score = m.group(2)
    rcore = m.group(3)
    # print(f"\tCORES s: {score}, r: {rcore}")
    return Event(EventType.CORES, {"score": score, "rcore": rcore})


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
    # print(f"\tL1DATACMP {dev}, {core}, {errors}, {starting}, {ending}")
    return Event(
        EventType.L1DATACMP,
        {
            "dev": dev,
            "core": core,
            "errors": errors,
            "starting": starting,
            "ending": ending,
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
    # print(f"\tDATACMP {bank0}, {bank1}, {errors}, {starting}, {ending}")
    return Event(
        EventType.DATACMP,
        {
            "bank0": bank0,
            "bank1": bank1,
            "errors": errors,
            "starting": starting,
            "ending": ending,
        },
    )


def parse_check(l: str) -> Optional[Event]:
    m = testcheck.match(l)
    if m is None:
        return None

    sdev = m.group(2)
    sdevbdf = m.group(3)
    rdev = m.group(4)
    rdevbdf = m.group(5)
    score = m.group(6)
    rcore = m.group(7)
    proc = m.group(8)
    # print(f"\tTESTCHECK '{sdev}' ({sdevbdf}), '{rdev}' ({rdevbdf}), '{score}' '{rcore}' '{proc}'")
    return Event(
        EventType.TESTCHECK,
        {
            "sdev": sdev,
            "sdevbdf": sdevbdf,
            "rdev": rdev,
            "rdevbdf": rdevbdf,
            "score": score,
            "rcore": rcore,
            "proc": proc,
        },
    )


def parse_setup(l: str) -> Optional[Event]:
    m = testsetup.match(l)
    if m is None:
        return None

    # print(f"\tSETUP")
    return Event(EventType.TESTSETUP, {})


def parse_line(l: str) -> Optional[Event]:
    if print_logs:
        print(f"l: {l}")

    parsers = [
        parse_testdevices,
        parse_teststart,
        parse_l1datacmp,
        parse_testend,
        parse_datacmp,
        parse_bwfail,
        parse_cores,
        parse_check,
        parse_setup,
        parse_procs,
        parse_runs,
        parse_bw,
    ]

    for p in parsers:
        if r := p(l):
            return r

    # print(f"l: {l}")
    return None


async def parse_logs_stream(inf: asyncio.StreamReader) -> AsyncIterator[Event]:
    while True:
        l = await inf.readline()
        if l == b"":
            break

        r = parse_line(l.decode("utf-8").strip())
        if r is not None:
            yield r


async def parse_logs(inf: asyncio.StreamReader) -> list[Event]:
    evs = []
    async for e in parse_logs_stream(inf):
        evs.append(e)

    return evs


def parse_evs(evs: list[Event]) -> Iterator[TestRun]:
    test: str = ""
    sdev: str = ""
    sdevbdf: str = ""
    rdev: str = ""
    rdevbdf: str = ""
    score: str = ""
    rcore: str = ""
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
            sdev = sdevbdf = rdev = rdevbdf = score = rcore = proc = ""
            links = []
            errors = []
            bw = {}
        elif e.typ == EventType.TESTEND:
            if proc != "":
                links.append(TestedLink(sdev, sdevbdf, score, rdev, rdevbdf, rcore, proc, bw, errors))
            runs.append(TestRun(test, links, e.extra["status"]))
            test = sdev = rdev = score = rcore = proc = ""
            links = []
            errors = []
            bw = {}
        elif e.typ == EventType.DEVICES:
            sdev = e.extra["sdev"]
            sdevbdf = e.extra["sdevbdf"]
            rdev = e.extra["rdev"]
            rdevbdf = e.extra["rdevbdf"]
            score = rcore = proc = ""
            errors = []
            bw = {}
        elif e.typ == EventType.CORES:
            if (test in noprocs and score != "") or proc != "":
                links.append(TestedLink(sdev, sdevbdf, score, rdev, rdevbdf, rcore, proc, bw, errors))
            score = e.extra["score"]
            rcore = e.extra["rcore"]
            proc = ""
            errors = []
            bw = {}
        elif e.typ == EventType.PROCS:
            if proc != "" and test != stresstest:
                links.append(TestedLink(sdev, sdevbdf, score, rdev, rdevbdf, rcore, proc, bw, errors))
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
            sdev = sdevbdf = rdev = rdevbdf = score = rcore = proc = ""
            links = []
            errors = []
            bw = {}
        elif e.typ == EventType.TESTCHECK:
            if sdev != "":
                links.append(TestedLink(sdev, sdevbdf, score, rdev, rdevbdf, rcore, proc, bw, errors))
            sdev = e.extra["sdev"]
            sdevbdf = e.extra["sdevbdf"]
            rdev = e.extra["rdev"]
            rdevbdf = e.extra["rdevbdf"]
            score = e.extra["score"]
            rcore = e.extra["rcore"]
            proc = e.extra["proc"]
            errors = []
            bw = {}

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


def print_summary(runs: list[TestRun]):
    headers = ["test name", "status"]
    rows = []
    for r in runs:
        rows.append([shortname(r.name), r.status])

    print(table("Test summary", headers, rows))


def link_name(l: TestedLink) -> str:
    names = [
        f"{l.src_dev}:{l.src_core}",
        f"{l.dst_dev}:{l.dst_core}",
    ]
    return "->".join(sorted(names))


def print_test_summary(t: TestCase, runs: list[TestRun]):
    for r in runs:
        if not r.name.endswith(t):
            continue

        links: dict[str, dict] = {}
        bw = False

        for l in r.links:
            name = link_name(l)
            if name not in links:
                links[name] = {"count": 0, "pass": True, "errors": 0, "bw": 0.0}

            links[name]["count"] += 1
            if "bw" in l.bw:
                links[name]["bw"] += l.bw["bw"]
                bw = True

            if len(l.errors):
                links[name]["errors"] += 1
                links[name]["pass"] = False

        headers = ["link name", "test count"]
        if bw:
            headers += ["avg bw"]
        headers += ["errors", "pass"]
        rows = []
        for k in sorted(links.keys()):
            row = [
                k,
                links[k]["count"],
            ]

            if bw:
                row += [f'{links[k]["bw"] / links[k]["count"]:.3f}']

            row += [links[k]["errors"], "OK" if links[k]["pass"] else "FAIL"]
            rows.append(row)

        print(table(f"{t} test summary", headers, rows))


def print_test_summary_per_chip(t: TestCase, runs: list[TestRun]):
    for r in runs:
        if not r.name.endswith(t):
            continue

        chips: dict = {}
        have_bws = False

        def ensure_dev(ch: str):
            if ch not in chips:
                chips[ch] = {"bdf": "", "tests": 0, "bws": [], "errors": 0}

        for l in r.links:
            ensure_dev(l.src_dev)
            ensure_dev(l.dst_dev)

            chips[l.src_dev]["tests"] += 1
            chips[l.dst_dev]["tests"] += 1
            if len(l.bw):
                have_bws = True
                chips[l.src_dev]["bws"].append(l.bw["bw"])
                chips[l.dst_dev]["bws"].append(l.bw["bw"])

            chips[l.src_dev]["errors"] += len(l.errors)
            chips[l.dst_dev]["errors"] += len(l.errors)
            chips[l.src_dev]["bdf"] = l.src_devbdf
            chips[l.dst_dev]["bdf"] = l.dst_devbdf

        avg = lambda x: sum(x) / len(x)
        headers = ["chip id", "chip bdf", "tests"]
        if have_bws:
            headers += ["bw (min)", "bw (max)", "bw (avg)"]
        headers += ["errors", "status"]

        rows = []
        for c in sorted(chips.keys(), key=int):
            ch = chips[c]
            bdf = ch["bdf"]
            bws = ch["bws"]
            err = ch["errors"]
            msg = "FAIL" if err > 0 else "OK"

            row = [c, bdf, ch["tests"]]
            if have_bws:
                row += [f"{min(bws):.3f}", f"{max(bws):.3f}", f"{avg(bws):.3f}"]
            row += [err, msg]

            rows.append(row)

        print(table(f"{t} per chip test summary", headers, rows))


def print_failing(runs: list[TestRun]):
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

    print(table("Failing tests/links", headers, rows))


def print_results(runs: list[TestRun]):
    print_failing(runs)
    for t in TestCase:
        print_test_summary(t, runs)
    for t in TestCase:
        print_test_summary_per_chip(t, runs)
    print_summary(runs)


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


async def parse_file(file: str) -> list[Event]:
    sr = await file_to_streamreader(file)
    return await parse_logs(sr)


async def main():
    parser = ArgumentParser()
    parser.add_argument("-t", type=str, help="Comma separated list of tests to run")
    parser.add_argument("-l", action="store_true", help="List all of the available tests")
    parser.add_argument("-q", action="store_true", help="Don't print the logs as the tests run")
    parser.add_argument("-n", action="store_true", help="Don't print the summary")
    parser.add_argument("-i", type=str, help="Input file to parse instead of running the tests")
    parser.add_argument("-o", type=str, help="Output path for the json file")
    opts = parser.parse_args()

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
        tests = [TestCase(t) for t in opts.t.split(",")]

    outfile = opts.o if opts.o else "out.json"
    print(f"Writing results to '{outfile}'")

    if opts.i:
        evs = await parse_file(opts.i)
    else:
        print("Running tests: ", ", ".join(t.value for t in tests))

        filters = prepare_filter([t for t in tests])
        program = "build/test/tt_metal/unit_tests_deployment"
        args = [f"--gtest_filter={filters}"]

        proc = await asyncio.create_subprocess_exec(program, *args, stdout=asyncio.subprocess.PIPE)

        p, evs = await asyncio.gather(proc.wait(), parse_logs(proc.stdout))

    # pprint.pp(evs)
    runs = list(parse_evs(evs))
    # pprint.pp(runs)
    # print(runs_to_json(runs, sort_keys=True, indent=4))
    if not opts.n:
        print_results(runs)
    runs_to_jsonf(outfile, runs, sort_keys=True, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
