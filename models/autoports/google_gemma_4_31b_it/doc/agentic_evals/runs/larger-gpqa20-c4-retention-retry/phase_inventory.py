"""Offline phase inventory; explicit completed artifacts only, no raw token output."""
import argparse
import hashlib
import json
import pathlib
import tarfile


def streams(path):
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive:
                if member.isfile() and "phase" in member.name and member.name.endswith((".json", ".jsonl")):
                    with archive.extractfile(member) as stream:
                        yield member.name, stream
    else:
        with path.open("rb") as stream:
            yield path.name, stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=pathlib.Path, action="append", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    groups = {}
    for path in args.phase:
        for name, stream in streams(path):
            for line in stream:
                if not line.strip():
                    continue
                frame = json.loads(line)
                if "events" not in frame:
                    raise ValueError("Unsupported frame schema: " + name)
                key = str((frame.get("pid"), frame.get("window_start_ns")))
                g = groups.setdefault(
                    key,
                    {
                        "frames": 0,
                        "events": {},
                        "conflicts": 0,
                        "duplicates": 0,
                        "loss_flags": set(),
                        "sources": set(),
                        "last": {},
                    },
                )
                g["frames"] += 1
                g["sources"].add(str(path) + ":" + name)
                retention = frame.get("token_retention", {})
                for flag in ("loss", "diagnostic_disabled", "export_failed", "file_exhausted"):
                    if retention.get(flag):
                        g["loss_flags"].add(flag)
                if frame.get("truncated") or frame.get("errors"):
                    g["loss_flags"].add("truncation_or_errors")
                if frame.get("boundary_ns", 0) >= g["last"].get("boundary_ns", 0):
                    g["last"] = {k: frame.get(k) for k in ("boundary_ns", "pending_steps", "token_retention")}
                for event in frame["events"]:
                    ek = (event["step_id"], event["event"])
                    digest = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest()
                    if ek in g["events"]:
                        g["duplicates"] += 1
                        g["conflicts"] += g["events"][ek] != digest
                    else:
                        g["events"][ek] = digest
    result = {}
    for key, g in groups.items():
        events = g.pop("events")
        submitted = {i for i, kind in events if kind == "submit"}
        completed = {i for i, kind in events if kind == "complete"}
        ids = submitted | completed
        g.update(
            unique_events=len(events),
            submitted=len(submitted),
            completed=len(completed),
            unmatched_steps=len(submitted ^ completed),
            missing_in_observed_range=(max(ids) - min(ids) + 1 - len(ids)) if ids else None,
            step_range=[min(ids), max(ids)] if ids else None,
        )
        g["sources"] = sorted(g["sources"])
        g["loss_flags"] = sorted(g["loss_flags"])
        result[key] = g
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump({"scope": "integrity_only_no_request_attribution", "windows": result}, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
