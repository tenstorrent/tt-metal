from typing import Optional, AsyncIterator, Iterable, TextIO, TypeVar

from dataclasses import dataclass
from enum import Enum, auto
from queue import Queue

import asyncio
import json
import re

T = TypeVar("T")

print_logs = True


async def async_iter(it: Iterable[T]) -> AsyncIterator[T]:
    for i in it:
        yield i


async def async_collect(f: AsyncIterator[T]) -> list[T]:
    return [i async for i in f]


timeregex = "\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+"
teststart = re.compile("\[\s+RUN\s+\] (.*)")
testend = re.compile("\[\s+([^ ]*)\s+\] (.*) \(.*\)")
testdevices = re.compile(
    f"({timeregex}).*Test \| sender device id: (.*) \((.*), ubb: (.*), chip: (.*)\), receiver device id: (.*) \((.*), ubb: (.*), chip: (.*)\) .*"
)
testcores = re.compile(
    f"({timeregex}).*Test \|   sender core: (\S*) \((\S*)\), receiver core: (\S*) \((\S*)\) \((\S*)\).*"
)
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
locinfo = "sdev: \[(.*) \((.*)\), ubb: (.*), chip: (.*)\], rdev: \[(.*) \((.*)\), ubb: (.*), chip: (.*)\], score: \[(.*) \((.*)\)\], rcore: \[(.*) \((.*)\)\], processor: \[(.*)\], link: \[(.*)\]"
testcheck = re.compile(f"({timeregex}).*Test \| core_check: {locinfo} .*")
testasic = re.compile(f"({timeregex}).*Test \|   UBB: (.*), Chip: (.*), BDF: (.*), SN: (.*) .*")


class EventType(Enum):
    TESTSTART = auto()
    TESTEND = auto()
    DEVICES = auto()
    CORES = auto()
    ASIC = auto()
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
class TestedLink:
    src_dev: str
    src_devbdf: str
    src_core: str
    src_core_uid: str
    src_ubb: str
    src_chip: str

    dst_dev: str
    dst_devbdf: str
    dst_core: str
    dst_core_uid: str
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


@dataclass
class Event:
    typ: EventType
    extra: dict

    def to_dict(self):
        return {"typ": self.typ.name, "extra": self.extra}


# NOTE this is unused for now, but might be useful
def publish_link(test, link: TestedLink, srcsn, dstsn, q: Queue):
    payload = {
        "test": test,
        "src_chip": srcsn,
        "src_core": link.src_core,
        "src_core_uid": link.src_core_uid,
        "dst_chip": dstsn,
        "dst_core": link.dst_core,
        "dst_core_uid": link.dst_core_uid,
        "proc": link.proc,
        "bw": link.bw,
        "errors": link.errors,
    }
    q.put(payload)
    print(json.dumps(payload, indent=4), link)


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
        "score_uid": m.group(3),
        "rcore": m.group(4),
        "rcore_uid": m.group(5),
        "type": m.group(6),
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
        "score_uid": m.group(11),
        "rcore": m.group(12),
        "rcore_uid": m.group(13),
        "proc": m.group(14),
        "link": m.group(15),
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


def parse_asic(l: str) -> Optional[Event]:
    m = testasic.match(l)
    if m is None:
        return None

    extra = {
        "ubb": m.group(2),
        "chip": m.group(3),
        "bdf": m.group(4),
        "sn": m.group(5),
    }
    # print(f"\tASIC {extra}")
    return Event(EventType.ASIC, extra)


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
        parse_asic,
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
    return await async_collect(parse_logs_stream(inf, logf))


async def parse_file(file: str, logf: Optional[TextIO]) -> list[Event]:
    sr = await file_to_streamreader(file)
    return await parse_logs(sr, logf)


async def process_logs(inf: asyncio.StreamReader, logf: Optional[TextIO], q: Queue):
    return await async_collect(parse_evs(parse_logs_stream(inf, logf), q))


async def parse_evs(evs: AsyncIterator[Event], q: Queue) -> AsyncIterator[TestRun]:
    test: str = ""
    sdev: str = ""
    sdevbdf: str = ""
    sdevubb: str = ""
    sdevchip: str = ""
    score: str = ""
    score_uid: str = ""
    rdev: str = ""
    rdevbdf: str = ""
    rdevubb: str = ""
    rdevchip: str = ""
    rcore: str = ""
    rcore_uid: str = ""
    ltype: str = ""
    proc: str = ""
    bw: dict = {}

    serialnums: dict[str, str] = {}
    runs: list[TestRun] = []
    links: list[TestedLink] = []
    errors: list = []

    drambidir = "MeshDispatchFixture.TensixDeploymentEthernet04DataIntegrityDramBidir"
    noprocs = [drambidir]

    # it = iter(evs)
    async for e in evs:
        if e.typ == EventType.TESTSTART:
            test = e.extra["name"]
            sdev = sdevbdf = rdev = rdevbdf = ltype = proc = ""
            score = score_uid = rcore = rcore_uid = ""
            sdevubb = sdevchip = rdevubb = rdevchip = ""
            links = []
            errors = []
            bw = {}
        elif e.typ == EventType.TESTEND:
            runs.append(TestRun(test, links, e.extra["status"]))
            test = sdev = rdev = ltype = proc = ""
            score = score_uid = rcore = rcore_uid = ""
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
            score = score_uid = rcore = rcore_uid = ltype = proc = ""
            errors = []
            bw = {}
        elif e.typ == EventType.CORES:
            score = e.extra["score"]
            score_uid = e.extra["score_uid"]
            rcore = e.extra["rcore"]
            rcore_uid = e.extra["rcore_uid"]
            ltype = e.extra["type"]
            proc = ""
            errors = []
            bw = {}
        elif e.typ == EventType.ASIC:
            serialnums[e.extra["bdf"]] = e.extra["sn"]
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
            sdev = sdevbdf = rdev = rdevbdf = ltype = proc = ""
            score = score_uid = rcore = rcore_uid = ""
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
            score_uid = e.extra["score_uid"]
            rcore = e.extra["rcore"]
            rcore_uid = e.extra["rcore_uid"]
            ltype = e.extra["link"]
            proc = e.extra["proc"]
            errors = []
            bw = {}
        elif e.typ == EventType.TESTDONE:
            tlink = TestedLink(
                sdev,
                sdevbdf,
                score,
                score_uid,
                sdevubb,
                sdevchip,
                rdev,
                rdevbdf,
                rcore,
                rcore_uid,
                rdevubb,
                rdevchip,
                ltype,
                proc,
                bw,
                errors,
            )
            links.append(tlink)
            publish_link(test, tlink, serialnums[sdevbdf], serialnums[rdevbdf], q)
        elif e.typ == EventType.MISSINGLINKS:
            global missing_links
            missing_links[e.extra["id"]] = e.extra
        elif e.typ == EventType.TIMEOUT:
            print("The test timed out, you should reset the card! (tt-smi -r)", file=sys.stderr)
            exit(1)

    for r in runs:
        yield r
