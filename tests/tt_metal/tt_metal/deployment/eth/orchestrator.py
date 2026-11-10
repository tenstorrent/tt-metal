import shlex
import sqlite3
import threading
import configparser

from time import time
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional, Iterator

from pssh.clients import ParallelSSHClient
from pssh.config import HostConfig
from pssh.utils import enable_host_logger

from gevent import joinall, spawn
from gevent.queue import Queue

from typed_configparser import ConfigParser


@dataclass
class GeneralSettings:
    hosts: list[str]
    syseng: str
    dbpath: str


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
    parser.read("hosts.ini")
    general = parser.parse_section(using_dataclass=GeneralSettings, section_name="general")
    host_configs = [parser.parse_section(using_dataclass=HostSettings, section_name=h) for h in general.hosts]
    for h, hn in zip(host_configs, general.hosts):
        h.prepare_config(hn)

    return general, host_configs


hostsini = "hosts.ini"
dbpath = "cluster.db"

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


def getlinks(db, h1: str, h2: str) -> list[str]:
    cur = db.cursor()

    cur.execute(asicsquery, (h1, h2))
    links = cur.fetchall()
    return [",".join(l) for l in links]


def envtostr(e: dict[str, str]) -> str:
    return " ".join(f"{k}={shlex.quote(str(e[k]))}" for k in e)


def run_pair(hosts: tuple[HostSettings, HostSettings], argv: str, env: dict, sync: bool = True) -> tuple[int, int]:
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

    def handle_output(h: HostSettings, host_output):
        host = h.hostname
        for line in host_output.stdout:
            print(f"[{host:<15}] {line}")
        for line in host_output.stderr:
            print(f"[err {host:<11}] {line}")
        print(f"{host:<15} exit code: {host_output.exit_code}")

    readers = [spawn(handle_output, host, host_output) for host, host_output in zip(hosts, outputs)]

    joinall(readers)
    client.join(outputs)

    return tuple(host_output.exit_code for host_output in outputs)


def test_pair(db: sqlite3.Connection, hosts: tuple[HostSettings, HostSettings], precompile=True) -> bool:
    endpoint_pairs = getlinks(db, *[h.hostname for h in hosts])  # TODO
    if len(endpoint_pairs) == 0:
        return False  # TODO

    links = " ".join(endpoint_pairs)

    cmd = "sh run2.sh"
    args = [(shlex.quote(h.workdir),) for h in hosts]

    bw = 200  # TODO
    env = {
        "ETH_TEST_EXPECTED_BW": str(bw),
        "ETH_TEST_ENDPOINTS": links,
        "TT_METAL_SLOW_DISPATCH_MODE": 1,
    }

    if precompile:
        e = env.copy()
        e["ETH_TEST_PRECOMPILE"] = 1
        exitcodes = run_pair(hosts, cmd, e, sync=False)
        if sum(exitcodes) > 0:
            print(f"Kernel precompilation failed with exit codes: {', '.join(str(e) for e in exitcodes)}")
            return False

    exitcodes = run_pair(hosts, cmd, env)
    if sum(exitcodes) > 0:
        print(f"Test run failed with exit codes: {', '.join(str(e) for e in exitcodes)}")
        return False

    return True


def pairs(hosts: list[HostSettings]) -> Iterator[tuple[HostSettings, HostSettings]]:
    for i in range(len(hosts) - 1):
        for j in range(i + 1, len(hosts)):
            yield hosts[i], hosts[j]


def test_dist(db: sqlite3.Connection, hosts: list[HostSettings]) -> None:
    for p in pairs(hosts):
        print(f"Testing {p[0].hostname} <-> {p[1].hostname}")
        test_pair(db, p)  # TODO: In parallel


def main() -> None:
    db: sqlite3.Connection = sqlite3.connect(dbpath)
    general, hosts = parse_config(hostsini)

    test_dist(db, hosts)


if __name__ == "__main__":
    main()
