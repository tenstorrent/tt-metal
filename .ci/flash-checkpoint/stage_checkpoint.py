"""Finite-controller child: original downloader, immutable listing, separate checkpoint storage."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, sys

REVISION = "2741eec155d03a8ce151b993ccce1a7b1e398d6b"
MANIFEST = "2618cf53db8e04d539d96bb03d3f5c6e08f8ad3dddd0a54252d5d1b739d84ded"
TOTAL = 360023351829
CACHE_ESTIMATE = 92000000000
RESERVE = 4000000000
SOURCE_HASHES = {
    "tools/download_checkpoint.py": "e3019eedee5a57a1c851a7b987de2e86824f229d2574efaf28e42781600aed68",
    "tools/verify_release.py": "537f31ed9236c310ab6c1d76eca49375649d9e1243aa3701e592994fcce1bb35",
    "checkpoint.py": "8a444f4dee75b133dec59d0f4b39041e0ebe4c7aceedfe262e31a05980bdb53a",
    "config.py": "331c0e879805a9edc3a77be4a0e7a67bdcf177d38374ce1489b6ef2332329e3a",
}


def entries(raw):
    if hashlib.sha256(raw).hexdigest() != MANIFEST:
        raise ValueError("manifest digest")
    rows = [r for r in json.loads(raw)["Data"]["Files"] if r["Type"] == "blob"]
    if len(rows) != 145 or sum(r["Size"] for r in rows) != TOTAL:
        raise ValueError("manifest coverage")
    for r in rows:
        p = Path(r["Path"])
        if p.is_absolute() or ".." in p.parts or str(p) != r["Path"]:
            raise ValueError("manifest path")
    return rows


def capacity(free, allocated):
    if allocated < 0 or allocated > TOTAL + 100000000:
        raise ValueError("unexpected allocated checkpoint size")
    required = max(0, TOTAL - allocated) + CACHE_ESTIMATE + RESERVE
    if free < required:
        raise ValueError(f"insufficient capacity: {free} < {required}")
    return required


def owned_tree(root, uid):
    if root.is_symlink():
        raise ValueError("checkpoint root symlink")
    allocated = 0
    if root.exists():
        for p in [root, *root.rglob("*")]:
            s = p.lstat()
            if p.is_symlink() or s.st_uid != uid:
                raise ValueError("foreign or symlink checkpoint entry")
            if p.is_file():
                allocated += s.st_blocks * 512
    return allocated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--admit-only", action="store_true")
    a = ap.parse_args()
    assert a.job.isdigit() and os.uname().nodename == "bh-lb-41-special-moconnor-for-reservation-" + a.job
    assert os.getuid() == 1211402857
    drivers = sorted(Path("/proc/driver/tenstorrent").glob("*/pids"))
    assert [p.parent.name for p in drivers] == list(map(str, range(8))) and not any(p.read_text() for p in drivers)
    root = Path("/localdev/moconnor/qwen38-checkpoint-" + REVISION)
    assert root.parent.stat().st_uid == os.getuid()
    for parent in root.parents:
        if parent.is_symlink():
            raise ValueError("symlink parent")
    model = a.source / "models/demos/blackhole/qwen38_flash_next"
    for name, digest in SOURCE_HASHES.items():
        if hashlib.sha256((model / name).read_bytes()).hexdigest() != digest:
            raise ValueError("source digest " + name)
    rows = entries(a.manifest.read_bytes())
    allocated = owned_tree(root, os.getuid())
    marker = root / ".download/campaign-owner.json"
    identity = {"uid": os.getuid(), "revision": REVISION, "manifest": MANIFEST}
    if root.exists() and any(root.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError("unowned checkpoint namespace")
    free = shutil.disk_usage(root.parent).free
    required = capacity(free, allocated)
    sys.path.insert(0, str(a.source))
    from models.demos.blackhole.qwen38_flash_next.tools.download_checkpoint import Download, RELEASE_REVISION
    from models.demos.blackhole.qwen38_flash_next.tools.verify_release import verify

    assert RELEASE_REVISION == REVISION
    assert not any(p.read_text() for p in drivers)
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        assert not target.startswith("/dev/tenstorrent/")
    print(
        json.dumps(
            {
                "job": a.job,
                "root": str(root),
                "free": free,
                "allocated": allocated,
                "required": required,
                "files": len(rows),
                "source": str(a.source),
                "admit_only": a.admit_only,
            }
        ),
        flush=True,
    )
    if a.admit_only:
        return
    root.mkdir(exist_ok=True)
    work = root / ".download"
    work.mkdir(exist_ok=True)
    marker = work / "campaign-owner.json"
    identity = {"uid": os.getuid(), "revision": REVISION, "manifest": MANIFEST}
    if marker.exists():
        assert json.loads(marker.read_text()) == identity
    else:
        marker.write_text(json.dumps(identity))
    result = Download(REVISION, rows, root, work, 32).run()
    if result:
        raise SystemExit(result)
    # Original complete verifier is mandatory even when downloader resume markers exist.
    report = verify(root, a.manifest, 2)
    output = a.source.parent / "evidence" / ("checkpoint-verified-" + a.job + ".json")
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"verified": report["verified_files"], "bytes": report["verified_bytes"], "report": str(output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
