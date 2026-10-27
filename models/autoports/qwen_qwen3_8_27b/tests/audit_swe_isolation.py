"""Fail closed on collected Harbor isolation evidence; never infer isolation from reward."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def audit(root, expected):
    errors = []
    seen = {}
    summaries = []
    inputs = {}

    def require(ok, message):
        if not ok:
            errors.append(message)

    def read(path):
        raw = path.read_bytes()
        inputs[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    configs = list(root.rglob("*harbor_config.json"))
    require(len(configs) == 1, f"Expected one Harbor config, got {len(configs)}")
    for path in configs:
        d = read(path)
        env = d.get("environment", {})
        kwargs = env.get("kwargs", {})
        require(env.get("type") == "docker", "Environment must be Docker")
        require(
            env.get("import_path") == "agentic_owned_networks.docker_environment:OwnedDockerEnvironment",
            "Owned environment hook missing",
        )
        require(kwargs.get("owned_agent_isolation") is True, "Saved Harbor config does not enable isolation")
        require(bool(kwargs.get("owned_owner") and kwargs.get("owned_job")), "Owned job identity missing")
        require(
            kwargs.get("owned_endpoint_host") in ["localhost", "127.0.0.1", "::1"],
            "API inventory endpoint is not host-local",
        )
        agents = d.get("agents", [])
        require(len(agents) == 1, "Expected one agent")
        for agent in agents:
            require(
                agent.get("import_path") == "agentic_owned_networks.mini_swe_agent:OwnedMiniSweAgent",
                "Phase-aware mini agent missing",
            )
            require(
                agent.get("env", {}).get("OPENAI_BASE_URL") == "http://host.docker.internal:8000/v1",
                "Unexpected model API URL",
            )
    for directory in root.rglob("owned_networks"):
        if "network-preflight" in directory.parts:
            continue
        records = []
        for path in directory.glob("*.json"):
            if re.fullmatch(r"[a-f0-9]{32}\.json", path.name):
                records.append((path, read(path)))
        for path, d in records:
            session = d.get("logical_session")
            if not session:
                continue
            matches = [name for name in expected if session.startswith(name + "__")]
            require(len(matches) == 1, f"Unexpected task session: {session}")
            name = matches[0] if len(matches) == 1 else session
            seen[name] = seen.get(name, 0) + 1
            require(d.get("internal") is True, f"{name}: primary network not internal")
            require(d.get("state") == "removed", f"{name}: primary network cleanup not verified")
            require(
                bool(re.fullmatch(r"[a-f0-9]{64}", d.get("network_id", ""))),
                f"{name}: primary network identity missing",
            )
            transitions = d.get("agent_network_transitions", [])
            require(
                [t.get("setup_egress") for t in transitions] == [True, False, True],
                f"{name}: missing/extra phase transitions",
            )
            require(
                all(t.get("internal_primary") is True for t in transitions),
                f"{name}: internal primary not verified at every transition",
            )
            ids = {t.get("container_id") for t in transitions}
            require(
                len(ids) == 1 and all(re.fullmatch(r"[a-f0-9]{64}", i or "") for i in ids),
                f"{name}: phase container identity changed/missing",
            )
            setup = [s for _, s in records if s.get("parent_attempt") == d.get("attempt")]
            require(len(setup) == 1, f"{name}: expected one linked setup network")
            for s in setup:
                require(s.get("internal", False) is False, f"{name}: setup network is not public")
                require(s.get("state") == "removed", f"{name}: setup cleanup not verified")
                require(
                    s.get("network_id") != d.get("network_id") and bool(s.get("network_id")),
                    f"{name}: setup/primary network identity not distinct",
                )
                require(
                    all(s.get(k) == d.get(k) and d.get(k) for k in ["owner", "job", "daemon_id"]),
                    f"{name}: setup ownership/daemon mismatch",
                )
            executions = d.get("workload_execs", [])
            require(bool(executions), f"{name}: supervised workload receipts missing")
            require(
                all(
                    e.get("state") == "settled"
                    and e.get("receipt", {}).get("status") == "settled"
                    and e.get("receipt", {}).get("supervisor_exited") is True
                    for e in executions
                ),
                f"{name}: unsettled workload",
            )
            require(
                not any(d.get(k) for k in ["cleanup_error", "structural_uncertainty"]),
                f"{name}: unresolved network uncertainty",
            )
            summaries.append(
                dict(
                    task=name,
                    primary_internal=d.get("internal", False),
                    phase_setup_egress=[t.get("setup_egress") for t in transitions],
                    primary_state=d.get("state"),
                    setup_networks=len(setup),
                    supervised_execs=len(executions),
                )
            )
    require(
        set(seen) == set(expected),
        f"Observed task identities differ: missing={sorted(set(expected)-set(seen))}, extra={sorted(set(seen)-set(expected))}",
    )
    require(
        all(n == 1 for n in seen.values()),
        "Expected one trial attempt per frozen task; duplicate attempts require explicit audit",
    )
    return dict(
        status="pass" if not errors else "fail",
        scope="Collected configuration/network phase evidence; not task accuracy or a packet-level runtime connectivity test",
        expected_tasks=expected,
        tasks=summaries,
        errors=errors,
        inputs_sha256=inputs,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    expected = manifest["task_names"]["swe_bench_verified"]
    assert len(expected) == len(set(expected)) == manifest["sample_count"]
    result = audit(a.directory, expected)
    result["manifest_sha256"] = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["status", "errors", "tasks"]}))
    raise SystemExit(result["status"] != "pass")


if __name__ == "__main__":
    main()
