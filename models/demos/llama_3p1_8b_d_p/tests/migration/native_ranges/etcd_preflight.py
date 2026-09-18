"""Exercise the production v3 gateway on the same task-local etcd child before TT imports."""
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path


def verify_etcd_api(
    endpoint, nonce, receipt_path, *, expected_version, check_stop=lambda: None, open_url=urllib.request.urlopen
):
    """Keep failed status/body evidence; never treat /health alone as gateway compatibility."""
    report = dict(ok=False, endpoint=endpoint, expected_version=expected_version, calls=[])
    lease = None
    revoked = False

    def request(route, body=None):
        check_stop()
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(endpoint + route, data=data, headers={"Content-Type": "application/json"})
        event = dict(route=route, method="GET" if body is None else "POST")
        report["calls"].append(event)
        try:
            with open_url(req, timeout=2) as response:
                event["status"] = response.status
                raw = response.read(1048577)
        except urllib.error.HTTPError as error:
            event.update(status=error.code, body=error.read(4096).decode(errors="replace"))
            raise
        if len(raw) > 1048576:
            raise RuntimeError("Oversized etcd preflight response")
        event["body"] = raw.decode()
        if event["status"] != 200:
            raise RuntimeError("etcd preflight returned non-200")
        result = json.loads(raw)
        if not isinstance(result, dict) or "error" in result:
            raise RuntimeError("Malformed or failed etcd gateway response")
        return result

    try:
        version = request("/version")
        if version.get("etcdserver") != expected_version:
            raise RuntimeError("Unexpected etcd server version: " + repr(version))
        report["version"] = version
        grant = request("/v3/lease/grant", {"TTL": "30"})
        lease = str(grant.get("ID", ""))
        if not lease.isdecimal() or int(lease) == 0 or int(grant.get("TTL", 0)) <= 0:
            raise RuntimeError("etcd did not grant a valid positive-TTL lease")
        key = base64.b64encode(("llama-startup-preflight/" + nonce).encode()).decode()
        value = base64.b64encode(("gateway-ready/" + nonce).encode()).decode()
        request("/v3/kv/put", dict(key=key, value=value, lease=lease))
        found = request("/v3/kv/range", dict(key=key))
        rows = found.get("kvs", [])
        if len(rows) != 1 or any(rows[0].get(k) != v for k, v in dict(key=key, value=value, lease=lease).items()):
            raise RuntimeError("etcd put/range did not preserve exact key/value/lease")
        request("/v3/lease/revoke", {"ID": lease})
        revoked = True
        if request("/v3/kv/range", dict(key=key)).get("kvs", []):
            raise RuntimeError("etcd key remains after lease revoke")
        report["ok"] = True
        return report
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        # A failed probe must not leave its leased key behind while its child is alive.
        if lease is not None and not revoked:
            try:
                request("/v3/lease/revoke", {"ID": lease})
                report["failure_lease_revoked"] = True
            except BaseException as cleanup_error:
                report["cleanup_error"] = repr(cleanup_error)
        Path(receipt_path).write_text(json.dumps(report, indent=2) + "\n")
