# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fetch the eval data that lmms-eval / openbench would otherwise pull in bulk, into $HF_HOME (the eval datasets dir).

  videomme  Video-MME (lmms-eval/Video-MME) videos of the chosen durations only. The repo is 101 GB of zip chunks
            and lmms-eval would download and unzip all of them; this reads each chunk's zip directory over HTTP range
            requests and extracts only the needed .mp4 files to $HF_HOME/videomme/data/<videoID>.mp4, the layout
            lmms-eval's videomme utils expect (it then skips its own download because the directory exists).
  tau2      tau2-bench's data assets at the installed tau2 commit (openbench's tau_bench_* tasks), into
            $MM_DATASETS/tau2 (run_eval.sh passes it as TAU2_DATA_DIR).
  gpqa      openbench's GPQA-Diamond source (nmayorga7/gpqa_diamond) into the HF datasets cache.

Every other benchmark is warmed by running it once through run_eval.sh with --dry-run --limit 5 (lmms-eval's own
loading code path). Writes a manifest (counts, bytes) to $MM_DATASETS/prefetch_manifest.json.

  HF_HOME=/data/.../datasets/hf python prefetch_datasets.py videomme --durations short
"""

import argparse
import json
import os
import shutil
import sys
import time
import zipfile


def _manifest_update(root, key, value):
    path = os.path.join(root, "prefetch_manifest.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[key] = value
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def fetch_videomme(hf_home, root, durations, workers):
    import pandas as pd
    from huggingface_hub import HfFileSystem, hf_hub_download

    parquet = hf_hub_download("lmms-eval/Video-MME", "videomme/test-00000-of-00001.parquet", repo_type="dataset")
    docs = pd.read_parquet(parquet)
    wanted = set(docs[docs.duration.isin(durations)].videoID)
    out_dir = os.path.join(hf_home, "videomme", "data")
    os.makedirs(out_dir, exist_ok=True)
    have = {os.path.splitext(n)[0] for n in os.listdir(out_dir) if n.lower().endswith(".mp4")}
    missing = wanted - have
    print(f"Video-MME {durations}: {len(wanted)} videos, {len(have & wanted)} present, {len(missing)} to fetch")
    fs = HfFileSystem()
    chunks = sorted(
        p for p in fs.ls("datasets/lmms-eval/Video-MME", detail=False) if p.endswith(".zip") and "videos_chunked" in p
    )
    t0 = time.time()
    fetched_bytes = 0
    for chunk in chunks:
        if not missing:
            break
        # 8 MiB blocks: the zip directory and each member are read with a handful of range requests.
        with fs.open(chunk, "rb", block_size=8 << 20) as remote, zipfile.ZipFile(remote) as zf:
            members = [m for m in zf.infolist() if os.path.splitext(os.path.basename(m.filename))[0] in missing]
            for m in members:
                vid = os.path.splitext(os.path.basename(m.filename))[0]
                dst = os.path.join(out_dir, vid + ".mp4")
                tmp = dst + ".part"
                with zf.open(m) as src, open(tmp, "wb") as f:
                    shutil.copyfileobj(src, f, 16 << 20)
                os.replace(tmp, dst)
                fetched_bytes += m.file_size
                missing.discard(vid)
            print(
                f"{os.path.basename(chunk)}: {len(members)} videos; {len(missing)} left; "
                f"{fetched_bytes / 1e9:.2f} GB in {time.time() - t0:.0f} s",
                flush=True,
            )
    present = [v for v in wanted if os.path.exists(os.path.join(out_dir, v + ".mp4"))]
    size = sum(os.path.getsize(os.path.join(out_dir, v + ".mp4")) for v in present)
    _manifest_update(
        root,
        f"videomme_{'_'.join(durations)}",
        {"videos": len(present), "of": len(wanted), "bytes": size, "missing": sorted(missing)},
    )
    if missing:
        print(f"WARNING: {len(missing)} videos not found in any chunk: {sorted(missing)[:10]}")


def _tau2_commit():
    """The tau2-bench commit the installed tau2 package was built from (pip's direct_url.json)."""
    from importlib.metadata import distribution

    info = json.loads(distribution("tau2").read_text("direct_url.json") or "{}")
    return info.get("vcs_info", {}).get("commit_id")


def fetch_tau2(root):
    """tau2's data at the commit of the installed tau2 code. openbench's own downloader clones tau2-bench main, whose
    data layout has moved on from the commit openbench pins (no tau2/user_simulator/ any more), which breaks the
    user simulator."""
    import subprocess
    import tempfile
    from pathlib import Path

    target = Path(root) / "tau2"
    commit = _tau2_commit()
    stamp = target / ".tau2_commit"
    if not (stamp.exists() and stamp.read_text().strip() == commit):
        if target.exists():
            shutil.rmtree(target)
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "tau2-bench")
            subprocess.run(
                [
                    "git",
                    "clone",
                    "-q",
                    "--filter=blob:none",
                    "--no-checkout",
                    "https://github.com/sierra-research/tau2-bench.git",
                    repo,
                ],
                check=True,
            )
            subprocess.run(["git", "-C", repo, "checkout", "-q", commit or "main"], check=True)
            shutil.copytree(os.path.join(repo, "data"), target)
        stamp.write_text(str(commit))
    tasks = target / "tau2" / "domains" / "airline" / "tasks.json"
    n = len(json.loads(tasks.read_text())) if tasks.exists() else None
    print(f"tau2 data (commit {commit}) in {target}; airline tasks: {n}")
    _manifest_update(root, "tau2", {"dir": str(target), "commit": commit, "airline_tasks": n})


def fetch_gpqa(root):
    import datasets

    ds = datasets.load_dataset("nmayorga7/gpqa_diamond", split="train")
    print(f"GPQA-Diamond (openbench source): {len(ds)} questions")
    _manifest_update(root, "gpqa_diamond", {"questions": len(ds)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", nargs="+", choices=["videomme", "tau2", "gpqa"])
    ap.add_argument("--durations", default="short", help="Video-MME durations, comma-separated (short,medium,long)")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        sys.exit("set HF_HOME to the eval datasets cache (run_eval.sh uses $MM_DATASETS/hf)")
    root = os.environ.get("MM_DATASETS", os.path.dirname(os.path.abspath(hf_home)))
    for what in args.what:
        if what == "videomme":
            fetch_videomme(hf_home, root, args.durations.split(","), args.workers)
        elif what == "tau2":
            fetch_tau2(root)
        else:
            fetch_gpqa(root)


if __name__ == "__main__":
    main()
