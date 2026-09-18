"""One source-owned etcd child; both endpoints verify its v3 API before TT imports."""

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from etcd_preflight import verify_etcd_api
from runner_support import require, wait_receipt, write_json


def status(url):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def resolve_client_listen_endpoint(plan):
    """Listen on local IPv4 interfaces; advertise the assigned source hostname."""
    endpoint = urlparse(plan["etcd_endpoint"])
    source = plan["source"]["host"]
    require(
        socket.gethostname() == source and endpoint.hostname == source,
        "etcd binding must run on the assigned source hostname",
    )
    require(
        endpoint.scheme == "http" and endpoint.port is not None and 1024 <= endpoint.port <= 65535,
        "A reviewed nonprivileged etcd HTTP port is required",
    )
    # The source resolves its own hostname to 127.0.1.1; peers resolve its network IP.
    # A wildcard listener accepts both routes without changing the advertised identity.
    return "http://0.0.0.0:" + str(endpoint.port)


class EtcdBootstrap:
    def __init__(self, plan, role, output):
        self.plan, self.role, self.output = plan, role, Path(output)
        self.process = self.log = None

    def start(self, check):
        plan, role, output = self.plan, self.role, self.output
        nonce, endpoint = plan["run_nonce"], plan["etcd_endpoint"]
        peer = "passive" if role == "source" else "source"
        if role == "source":
            listen_endpoint = resolve_client_listen_endpoint(plan)
            write_json(
                output / "etcd-binding.json",
                dict(
                    run_nonce=nonce,
                    role=role,
                    hostname=plan["source"]["host"],
                    listen_client_url=listen_endpoint,
                    advertised_client_url=endpoint,
                    listen_scope="all_local_ipv4_interfaces",
                ),
            )
            self.log = (output / "etcd.log").open("xb")
            name = "llama-transfer-" + nonce
            self.process = subprocess.Popen(
                [
                    plan["etcd_binary"],
                    "--name",
                    name,
                    "--data-dir",
                    str(output / "etcd-data"),
                    "--listen-client-urls",
                    listen_endpoint,
                    "--advertise-client-urls",
                    endpoint,
                    "--listen-peer-urls",
                    plan["etcd_peer_endpoint"],
                    "--initial-advertise-peer-urls",
                    plan["etcd_peer_endpoint"],
                    "--initial-cluster",
                    name + "=" + plan["etcd_peer_endpoint"],
                    "--initial-cluster-token",
                    nonce,
                ],
                stdout=self.log,
                stderr=subprocess.STDOUT,
                env=dict(os.environ, GOMAXPROCS="1"),
            )
            deadline = time.monotonic() + 30
            while True:
                check()
                require(self.process.poll() is None, "Source etcd exited before API preflight")
                try:
                    if status(endpoint + "/health")[0] == 200:
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                require(time.monotonic() < deadline, "Source etcd health timeout")
                time.sleep(0.1)
        else:
            wait_receipt(Path(plan["run_dir"]) / "source/bootstrap-ready.json", nonce, 120, check)
        result = verify_etcd_api(
            endpoint,
            nonce + "-" + role,
            output / "etcd-api-preflight.json",
            expected_version="3.5.13",
            check_stop=check,
        )
        if self.process is not None:
            require(self.process.poll() is None, "Source etcd exited after API preflight")
        write_json(
            output / "bootstrap-ready.json",
            dict(
                run_nonce=nonce,
                role=role,
                ok=True,
                endpoint=endpoint,
                child_pid=None if self.process is None else self.process.pid,
            ),
        )
        other = wait_receipt(Path(plan["run_dir"]) / peer / "bootstrap-ready.json", nonce, 120, check)
        require(
            other.get("role") == peer and other.get("endpoint") == endpoint, "Peer verified a different etcd endpoint"
        )
        check()
        if self.process is not None:
            require(self.process.poll() is None, "Source etcd exited before native imports")
        return result

    def close(self):
        # Only the source owns this CPU child. Caller first proves both managers stopped.
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log is not None:
            self.log.close()
