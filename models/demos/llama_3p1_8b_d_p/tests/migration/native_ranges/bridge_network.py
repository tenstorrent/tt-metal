"""Resolve the remote native rendezvous address before any TT or manager startup."""
import datetime
import ipaddress
import socket
from urllib.parse import urlparse

from runner_support import require


def _port(value):
    require(type(value) is int and 1024 <= value <= 65535, "Bridge port must be a nonprivileged uint16")
    return value


def _ipv4(value):
    address = ipaddress.IPv4Address(value)
    require(
        not (
            address.is_loopback
            or address.is_unspecified
            or address.is_multicast
            or address.is_link_local
            or address.is_reserved
        ),
        "Remote peer needs a concrete nonloopback IPv4",
    )
    return str(address)


def _endpoints(plan, role):
    require(role in ("source", "passive"), "Invalid bridge role")
    peer_role = "passive" if role == "source" else "source"
    local, peer = plan[role], plan[peer_role]
    require(local["host"] != peer["host"], "Remote bridge peer must be a different assigned host")
    for endpoint in (local, peer):
        require(
            type(endpoint["endpoint_id"]) is int and 0 <= endpoint["endpoint_id"] < 2**31,
            "Native endpoint ID must fit signed int32",
        )
        ports = [_port(endpoint[k]) for k in ("control_port", "health_port", "manager_port", "manager_control_port")]
        if endpoint is plan["source"]:
            ports += [_port(urlparse(plan[k]).port) for k in ("etcd_endpoint", "etcd_peer_endpoint")]
        require(len(ports) == len(set(ports)), "Task listener ports collide on one endpoint")
    require(local["endpoint_id"] != peer["endpoint_id"], "Native endpoint IDs collide")
    return peer_role, local, peer


def prepare_bridge_config(plan, role):
    peer_role, local, peer = _endpoints(plan, role)
    hostname = socket.gethostname()
    require(hostname == local["host"], "Bridge resolution must run on its assigned endpoint")
    # Resolve only the remote host. A node's own name can legitimately resolve to 127.0.1.1.
    rows = socket.getaddrinfo(
        peer["host"], peer["control_port"], socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP
    )
    answers = []
    for family, kind, protocol, canonical, address in rows:
        require(
            (family, kind, protocol) == (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP),
            "Peer resolver returned a non-IPv4 TCP answer",
        )
        require(address[1] == peer["control_port"], "Resolved peer port differs")
        answers.append(
            dict(
                family=int(family),
                socktype=int(kind),
                protocol=int(protocol),
                canonical_name=canonical,
                address=_ipv4(address[0]),
                port=address[1],
            )
        )
    addresses = sorted({row["address"] for row in answers})
    require(len(addresses) == 1, "Remote peer must resolve to exactly one concrete IPv4")
    config = dict(
        role=role,
        self_endpoint=local["endpoint_id"],
        peer_endpoint=peer["endpoint_id"],
        peer_host=addresses[0],
        peer_port=peer["control_port"],
        manager_endpoint=f"tcp://127.0.0.1:{local['manager_port']}",
        control_port=local["control_port"],
    )
    receipt = dict(
        run_nonce=plan["run_nonce"],
        role=role,
        hostname=hostname,
        job_id=str(local["job_id"]),
        peer_role=peer_role,
        peer_hostname=peer["host"],
        peer_job_id=str(peer["job_id"]),
        resolver="socket.getaddrinfo(AF_INET, SOCK_STREAM, IPPROTO_TCP)",
        answers=answers,
        peer_ipv4=addresses[0],
        peer_port=peer["control_port"],
        config=dict(config),
        recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    return config, receipt


def validate_bridge_network(plan, role, config, receipt):
    """Check saved resolver provenance against the exact file consumed by the native child."""
    peer_role, local, peer = _endpoints(plan, role)
    require(
        receipt["run_nonce"] == plan["run_nonce"]
        and receipt["role"] == role
        and receipt["hostname"] == local["host"]
        and receipt["job_id"] == str(local["job_id"])
        and receipt["peer_role"] == peer_role
        and receipt["peer_hostname"] == peer["host"]
        and receipt["peer_job_id"] == str(peer["job_id"]),
        "Bridge resolver identity differs",
    )
    require(
        receipt["resolver"] == "socket.getaddrinfo(AF_INET, SOCK_STREAM, IPPROTO_TCP)" and receipt["answers"],
        "Missing AF_INET resolver provenance",
    )
    for row in receipt["answers"]:
        require(
            (row["family"], row["socktype"], row["protocol"], row["port"])
            == (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, peer["control_port"]),
            "Saved peer resolver contract differs",
        )
        _ipv4(row["address"])
    address = _ipv4(receipt["peer_ipv4"])
    require(
        {row["address"] for row in receipt["answers"]} == {address} and receipt["peer_port"] == peer["control_port"],
        "Saved peer resolution differs",
    )
    expected = dict(
        role=role,
        self_endpoint=local["endpoint_id"],
        peer_endpoint=peer["endpoint_id"],
        peer_host=address,
        peer_port=peer["control_port"],
        manager_endpoint=f"tcp://127.0.0.1:{local['manager_port']}",
        control_port=local["control_port"],
    )
    require(
        receipt["config"] == expected
        and {key: value for key, value in config.items() if key not in ("journal", "ack_shm")} == expected,
        "Serialized native bridge config differs from pre-native resolution",
    )
