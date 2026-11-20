import os
import json
import shlex
import asyncio
import sqlite3
import threading
import configparser

import result_parser

from pathlib import Path
from time import time, sleep
from threading import Thread, Lock
from argparse import ArgumentParser
from collections.abc import Callable
from dataclasses import dataclass, field
from contextlib import contextmanager, closing
from typing import Union, Optional, Iterator, Iterable, TypeAlias, cast, Sequence

from pssh.config import HostConfig
from pssh.output import HostOutput
from pssh.utils import enable_host_logger
from pssh.clients import ParallelSSHClient

from gevent.queue import Queue
from gevent import joinall, spawn
from gevent.os import make_nonblocking, nb_write

from typed_configparser import ConfigParser

from result_parser import process_logs
from testenums import TestCase, DistTestCase, prepare_filter, ALLTESTS

from exceptiongroup import ExceptionGroup

result_parser.print_logs = False

Buffer: TypeAlias = bytes | bytearray | memoryview


@dataclass
class GeneralSettings:
    hosts: list[str]
    syseng: str


@dataclass
class HostSettings:
    user: str
    private_key: str
    workdir: str
    syseng: str
    hostname: str = "INVALID$!#%"
    port: int = 22
    host_config: HostConfig = HostConfig()

    def prepare_config(self, hostname: str):
        self.hostname = hostname
        keys = (
            "user",
            "port",
            "private_key",
        )
        d = {}
        for k in keys:
            d[k] = self.__dict__[k]

        self.host_config = HostConfig(**d)


def parse_config(p: str) -> tuple[GeneralSettings, list[HostSettings]]:
    parser = ConfigParser()
    parser.read(p)
    general = parser.parse_section(using_dataclass=GeneralSettings, section_name="general")
    host_configs = [parser.parse_section(using_dataclass=HostSettings, section_name=h) for h in general.hosts]
    for h, hn in zip(host_configs, general.hosts):
        h.prepare_config(hn)

    return general, host_configs


def validate_test_arg(t: str):
    if t not in ALLTESTS:
        raise Exception(f"Unknown test {t}")


asicsquery = """
WITH hosts AS (
	SELECT
		? AS host1,
		? AS host2
),
ports AS (
	SELECT p.*
	FROM eth_observed p
	JOIN cluster_member cm ON cm.SNAPSHOT_ID = p.SNAPSHOT_ID
	WHERE cm.MERGE_ID = (
		SELECT MERGE_ID
		FROM cluster_snapshot
		ORDER BY TIMESTAMP DESC, MERGE_ID DESC
		LIMIT 1
	)
)
SELECT DISTINCT
	-- a.GALAXY_ID AS SYSTEM_A,
	a.UID AS ENDPOINT_A,
	-- b.GALAXY_ID AS SYSTEM_B,
	b.UID AS ENDPOINT_B
FROM ports a
JOIN ports b
    ON a.OBSERVED_PARTNER = b.UID
    AND b.OBSERVED_PARTNER = a.UID
CROSS JOIN hosts h
WHERE a.GALAXY_ID = min(h.host1, h.host2)
    AND b.GALAXY_ID = max(h.host1, h.host2)
    AND a.GALAXY_ID <> b.GALAXY_ID
ORDER BY ENDPOINT_A, ENDPOINT_B;
"""

# TODO types
tableschema = """
CREATE TABLE IF NOT EXISTS eth_test_results(
    testrun_id,
    merge_id,
    testname,
    timestamp_ms,
    src_core_uid,
    dst_core_uid,
    proc,
    bw,
    errors,
    passed
)
"""


def getlinks(db: sqlite3.Connection, h1: str, h2: str) -> list[str]:
    cur = db.cursor()

    cur.execute(asicsquery, (h1, h2))
    links = cur.fetchall()
    return [",".join(l) for l in links]


def envtostr(e: dict[str, str | int]) -> str:
    return " ".join(f"{k}={shlex.quote(str(e[k]))}" for k in e)


def write_all(fd: int, data: Buffer):
    view = memoryview(data)
    while view:
        n = nb_write(fd, view)
        view = view[n:]


def handle_output(h: HostSettings, host_output: HostOutput, wfd: Optional[int] = None):
    host = h.hostname
    for line in host_output.stdout:
        assert isinstance(line, str)
        print(f"[{host:<15}] {line}")
        if wfd:
            write_all(wfd, (line + "\n").encode("utf-8"))
    for line in host_output.stderr:
        assert isinstance(line, str)
        print(f"[err {host:<11}] {line}")
    print(f"{host:<15} exit code: {host_output.exit_code}")


def run_single(host: HostSettings, argv: str, env: dict[str, str | int], wfd: Optional[int] = None) -> tuple[int]:
    hostnames = [host.hostname]
    hostconfigs = [host.host_config]

    args = [(shlex.quote(host.workdir),)]

    client = ParallelSSHClient(hosts=hostnames, host_config=hostconfigs, compress=True)

    e = envtostr(env)
    outputs = client.run_command(f"cd %s && {e} {argv}", host_args=args)

    readers = [spawn(handle_output, host, host_output, wfd) for host, host_output in zip([host], outputs)]

    joinall(readers)
    client.join(outputs)

    return tuple(host_output.exit_code for host_output in outputs)


def run_many(
    hosts: list[HostSettings],
    argv: str,
    env: dict[str, str | int],
    wfds: Optional[list[int]] = None,
    sync: bool = True,
) -> list[int]:
    hostnames = [h.hostname for h in hosts]
    hostconfigs = [h.host_config for h in hosts]

    args = [(shlex.quote(h.workdir),) for h in hosts]

    client = ParallelSSHClient(hosts=hostnames, host_config=hostconfigs, compress=True)

    if sync:
        now = int(time())  # TODO
        env["RUN_AT"] = now + 10
    e = envtostr(env)
    outputs = client.run_command(f"cd %s && {e} {argv}", host_args=args)
    # outputs = client.run_command("cd %s && for i in $(seq 10); do echo $i; sleep .4; done # sh run2.sh %d", host_args=args)

    fds: Sequence[Optional[int]] = [None for h in hosts]
    if wfds:
        fds = wfds
    readers = [spawn(handle_output, host, host_output, wfd) for host, host_output, wfd in zip(hosts, outputs, fds)]

    joinall(readers)
    client.join(outputs)

    return [host_output.exit_code for host_output in outputs]


def test_one(
    db: sqlite3.Connection,
    host: HostSettings,
    test: TestCase,
    links: Optional[str],
    precompile=False,
    wfd: Optional[int] = None,
) -> bool:
    # endpoint_pairs = getlinks(db, *[h.hostname for h in host])  # TODO
    # if len(endpoint_pairs) == 0:
    #     return False  # TODO
    # links = " ".join(endpoint_pairs)

    f = prepare_filter([test])
    cmd = f"sh -c '. ./run2.sh && ./build/test/tt_metal/unit_tests_deployment --gtest_filter={f}'"

    env: dict[str, str | int] = {
        "TT_METAL_SLOW_DISPATCH_MODE": 1,
    }
    if links:
        env["ETH_TEST_ENDPOINTS"] = links

    if precompile:
        e = env.copy()
        e["ETH_TEST_PRECOMPILE"] = 1
        exitcodes = run_single(host, cmd, e)
        if sum(exitcodes) > 0:
            print(f"Kernel precompilation failed with exit code: {', '.join(str(e) for e in exitcodes)}")
            return False

    exitcodes = run_single(host, cmd, env, wfd)
    if sum(exitcodes) > 0:
        print(f"Test run failed with exit code: {', '.join(str(e) for e in exitcodes)}")
        return False

    return True


def test_pair(
    db: sqlite3.Connection,
    hosts: tuple[HostSettings, HostSettings],
    test: DistTestCase,
    links: str,
    precompile=True,
    wfds: Optional[tuple[int, int]] = None,
) -> bool:
    endpoint_pairs = getlinks(db, *(h.hostname for h in hosts))  # TODO
    if len(endpoint_pairs) == 0:
        return False  # TODO

    if not links:  # TODO
        links = " ".join(endpoint_pairs)

    f = prepare_filter([test])
    cmd = f"sh -c '. ./run2.sh && ./build/test/tt_metal/unit_tests_deployment --gtest_filter={f}'"
    # cmd = "sh run2.sh"

    bw = 200  # TODO
    env: dict[str, str | int] = {
        "ETH_TEST_EXPECTED_BW": bw,
        "ETH_TEST_ENDPOINTS": links,
        "TT_METAL_SLOW_DISPATCH_MODE": 1,
    }

    if precompile:
        e = env.copy()
        e["ETH_TEST_PRECOMPILE"] = 1
        exitcodes = run_many(list(hosts), cmd, e, sync=False)
        if sum(exitcodes) > 0:
            print(f"Kernel precompilation failed with exit codes: {', '.join(str(e) for e in exitcodes)}")
            return False

    exitcodes = run_many(list(hosts), cmd, env, list(wfds) if wfds else None)
    if sum(exitcodes) > 0:
        print(f"Test run failed with exit codes: {', '.join(str(e) for e in exitcodes)}")
        return False

    return True


def pairs(hosts: list[HostSettings]) -> Iterator[tuple[HostSettings, HostSettings]]:
    for i in range(len(hosts) - 1):
        for j in range(i + 1, len(hosts)):
            yield hosts[i], hosts[j]


def test_single(
    db: sqlite3.Connection, tests: list[TestCase], hosts: list[HostSettings], links: Optional[str], wfd: Optional[int]
) -> bool:
    test_pass = True
    for h in hosts:
        for t in tests:
            print(f"Testing {t} on {h.hostname}")
            test_pass = test_pass and test_one(db, h, t, links, wfd=wfd)  # TODO: In parallel

    return test_pass


def test_dist(
    db: sqlite3.Connection,
    tests: list[DistTestCase],
    hosts: list[HostSettings],
    links: str,
    wfds: Optional[tuple[int, int]],
) -> bool:
    test_pass = True
    for p in pairs(hosts):
        for t in tests:
            print(f"Testing {t} on {p[0].hostname} <-> {p[1].hostname}")
            test_pass = test_pass and test_pair(db, p, t, links, wfds=wfds)  # TODO: In parallel

    return test_pass


def insert_test_db(db: sqlite3.Connection, t: Iterable[tuple]):
    query = """
INSERT INTO eth_test_results(
    testrun_id,
    merge_id,
    testname,
    timestamp_ms,
    src_core_uid,
    dst_core_uid,
    proc,
    bw,
    errors,
    passed
) VALUES (
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?,
    ?
)
"""

    with db:
        db.executemany(query, t)


def batch_worker(db: sqlite3.Connection, mergeid: str, runid: str, q: Queue[dict | str]) -> None:
    db.execute(tableschema)

    done = False
    while not done:
        batch = []
        while not q.empty():
            data = q.get_nowait()
            if data == "DONE":
                done = True
                break
            batch.append(data)

        if batch:
            print(batch)
            now = int(time() * 1000)
            insert_test_db(
                db,
                (
                    (
                        runid,
                        mergeid,
                        d["test"],
                        now,
                        d["src_core_uid"],
                        d["dst_core_uid"],
                        d["proc"],
                        json.dumps(d["bw"]),
                        json.dumps(d["errors"]),
                        len(d["errors"]) == 0,
                    )
                    for d in batch
                ),
            )
        else:
            sleep(0.1)


# Returns the write side of the pipe for the test output to be written to
@contextmanager
def prepare_test_runner(dbpath: str, mergeid: str, runid: str) -> Iterator[int]:
    rfd, wfd = os.pipe()
    make_nonblocking(wfd)

    q = Queue()
    lock: Lock = Lock()
    errors: list[Exception] = []

    async def runner():
        print("running in asyncio thread")
        pipe = os.fdopen(rfd, "rb", buffering=0)

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, pipe)

        try:
            runs = await process_logs(reader, None, q)
        except Exception as e:
            with lock:
                errors.append(e)
        finally:  # TODO error handling
            q.put("DONE")
            transport.close()

    def dbcontext():
        with closing(sqlite3.connect(f"file:{dbpath}?mode=rw")) as conn:
            try:
                batch_worker(conn, mergeid, runid, q)
            except Exception as e:
                with lock:
                    errors.append(e)

    threads = (
        Thread(target=lambda: asyncio.run(runner())),
        Thread(target=dbcontext),
    )

    for t in threads:
        t.start()

    try:
        yield wfd
    finally:  # TODO Error handling
        os.close(wfd)
        for t in threads:
            t.join(2)  # TODO

        if len(errors) > 0:
            raise ExceptionGroup("Errors happened when processing logs", errors)


def qiter(q: Queue[dict | str]):
    while True:
        while not q.empty():
            d = q.get_nowait()
            if d is None:
                continue

            if d == "DONE":
                return

            yield d
            d = None


def linkiter(ls: str) -> list[tuple[str, str]]:
    r = [l.split(",") for l in ls.split(" ")]
    for l in r:
        assert len(l) == 2

    return [(l[0], l[1]) for l in r]


def dist_batch_worker(
    db: sqlite3.Connection, mergeid: str, runid: str, links: str, q1: Queue[dict | str], q2: Queue[dict | str]
) -> None:
    db.execute(tableschema)

    for d1, d2, l in zip(qiter(q1), qiter(q2), linkiter(links), strict=True):
        batch: list[dict] = []

        bw1 = d1["bw"]
        bw2 = d2["bw"]

        d = {
            "test": d1["test"],
            "src_core_uid": d1["src_core_uid"],
            "dst_core_uid": d2["src_core_uid"],
            "proc": d1["proc"],
            "bw": {},
            "errors": d1["errors"] + d2["errors"],
        }
        if bw1 and bw2:  # TODO handle better
            d["bw"] = bw1 if bw1["bw"] > bw2["bw"] else bw2

        if d["src_core_uid"] != l[0]:
            d["src_core_uid"], d["dst_core_uid"] = d["dst_core_uid"], d["src_core_uid"]
        if d["src_core_uid"] != l[0] and d["dst_core_uid"] != l[1]:
            raise Exception(f"Unknown link {d}, expected {l}")

        print(d)
        now = int(time() * 1000)
        insert_test_db(
            db,
            (
                (
                    runid,
                    mergeid,
                    d["test"],
                    now,
                    d["src_core_uid"],
                    d["dst_core_uid"],
                    d["proc"],
                    json.dumps(d["bw"]),
                    json.dumps(d["errors"]),
                    len(d["errors"]) == 0,
                ),
            ),
        )


# Returns the write side of the pipe for the test output to be written to
@contextmanager
def prepare_dist_test_runner(dbpath: str, mergeid: str, runid: str, linkspec: str) -> Iterator[tuple[int, int]]:
    rfd1, wfd1 = os.pipe()
    make_nonblocking(wfd1)

    rfd2, wfd2 = os.pipe()
    make_nonblocking(wfd2)

    q1, q2 = Queue(), Queue()

    lock: Lock = Lock()
    errors: list[Exception] = []

    async def runner(rfd: int, q: Queue):
        print("running in asyncio thread")
        pipe = os.fdopen(rfd, "rb", buffering=0)

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, pipe)

        try:
            runs = await process_logs(reader, None, q)
        except Exception as e:
            with lock:
                errors.append(e)
        finally:  # TODO error handling
            q.put("DONE")
            transport.close()

    def dbcontext():
        with closing(sqlite3.connect(f"file:{dbpath}?mode=rw")) as conn:
            try:
                dist_batch_worker(conn, mergeid, runid, linkspec, q1, q2)
            except Exception as e:
                with lock:
                    errors.append(e)

    threads = (
        Thread(target=lambda rfd, q: asyncio.run(runner(rfd, q)), args=[rfd1, q1]),
        Thread(target=lambda rfd, q: asyncio.run(runner(rfd, q)), args=[rfd2, q2]),
        Thread(target=dbcontext),
    )

    for t in threads:
        t.start()

    try:
        yield wfd1, wfd2
    finally:  # TODO Error handling
        os.close(wfd1)
        os.close(wfd2)
        for t in threads:
            t.join(2)  # TODO

        if len(errors) > 0:
            raise ExceptionGroup("Errors happened when processing logs", errors)


def execute_many(hosts: list[HostSettings], argv: str, env: dict[str, str | int]) -> list[int]:
    return run_many(list(hosts), argv, env, None)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("-d", type=str, help="Path to the sqlite3 db with hardware info")
    parser.add_argument("-r", type=str, help="Test run ID")
    parser.add_argument("-s", type=str, help="Merged Snapshot ID")
    parser.add_argument("-c", type=str, help="INI host config file")
    parser.add_argument("-l", type=str, help="List of links to test")
    parser.add_argument("-t", type=str, help="Test to run")
    parser.add_argument("-e", type=str, help="Command to execute instead of tests")
    parser.add_argument("-m", type=str, help="Host or comma separated list of hosts to run the tests on")
    opts = parser.parse_args()

    hostsini = opts.c if opts.c else "hosts.ini"
    general, hosts = parse_config(hostsini)

    dbpath = opts.d if opts.d else "cluster.db"
    db: sqlite3.Connection = sqlite3.connect(f"file:{dbpath}?mode=rw")

    hostnames = opts.m.split(",")  # TODO validation
    hs = []
    for h in hostnames:
        for hc in hosts:
            if h == hc.hostname:
                hs.append(hc)
                break

    links = opts.l

    mergeid = opts.s if opts.s else "merge_id"
    runid = opts.r if opts.r else "run_id"

    if opts.e:
        assert not opts.t and not opts.l and not opts.s and not opts.r and not opts.d
        env: dict = {}  # TODO
        rets = run_many(hs, opts.e, env)
        return int(sum(rets) > 0)

    test_pass = True

    testname = opts.t
    validate_test_arg(testname)

    if testname in [t.value for t in TestCase]:
        with prepare_test_runner(dbpath, mergeid, runid) as wfd:
            test_pass = test_pass and test_single(db, [TestCase(opts.t)], hs, links, wfd)

    if testname in [t.value for t in DistTestCase]:
        with prepare_dist_test_runner(dbpath, mergeid, runid, links) as wfds:
            test_pass = test_pass and test_dist(db, [DistTestCase(opts.t)], hs, links, wfds)

    return n if test_pass else 1


if __name__ == "__main__":
    exit(main())
