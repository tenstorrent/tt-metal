#!/usr/bin/env python3
"""Submit a Galaxy sequence job: several `run_group.py job` arms back to back in ONE allocation (regress_seq.sbatch).

    seq_submit.py NAME TIME_MIN ARM [ARM ...] [--go]
    ARM = "TREE GROUP LABEL [job args...]"   e.g. "main G-P ab_288afd2 --routers main --passes 1 --local-cache"

Same guards as `run_group.py submit`: trees must be staged at the manifest's sha, /data must have
min_data_inodes_free, the study Galaxy job count (running + pending, quad excepted) must be under the cap, standing
excludes (+ the ring exclude) apply, and without --go it only runs `sbatch --test-only`.
"""
import datetime
import os
import shlex
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_group as rg  # noqa: E402


def main():
    argv = sys.argv[1:]
    go = "--go" in argv
    argv = [a for a in argv if a != "--go"]
    if len(argv) < 3:
        print(__doc__)
        return 2
    name, minutes, arms = argv[0], int(argv[1]), [shlex.split(a) for a in argv[2:]]
    man = rg.load_manifest(rg.DEFAULT_MANIFEST)
    sl = man["systems"]["bh_galaxy"]["slurm"]
    if minutes > sl.get("max_time_min", 360):
        raise rg.RegressError(f"{minutes} min > the partition limit")
    lines, ring = [], False
    for a in arms:
        tree, group, label, extra = a[0], a[1], a[2], a[3:]
        t = man["trees"][tree]
        staged = {}
        try:
            import json
            staged = json.load(open(os.path.join(t["data_path"], ".regress_tree.json")))
        except (OSError, ValueError, TypeError):
            raise rg.RegressError(f"tree {tree} is not staged on /data")
        if t.get("sha") and staged.get("sha") != t["sha"]:
            raise rg.RegressError(f"staged {tree} is {staged.get('sha')}, the manifest says {t['sha']}")
        if "--local-cache" not in extra:
            raise rg.RegressError(f"arm {a}: every Galaxy arm uses --local-cache (coordinator 2026-11-04)")
        g = rg.get_group(man, group)
        only = extra[extra.index("--steps") + 1] if "--steps" in extra else None
        if any("ring_mapping" in s["needs"] for s in rg.group_steps(man, g, t["data_path"], only)):
            ring = True
        lines.append((tree, t["data_path"], group, label, " ".join(extra)))
    inodes = rg.data_inodes_free()
    rg.say(f"/data free inodes now: {inodes:,} (a submit needs >= {sl['min_data_inodes_free']:,})")
    if inodes < sl["min_data_inodes_free"] and go:
        raise rg.RegressError("/data is short of free inodes")
    jobs = rg.slurm_count_galaxy()
    cap = sl["max_concurrent"]
    if os.environ.get("REGRESS_GALAXY_CAP"):  # one-off allowance for this invocation only (the default stays in the manifest)
        cap = int(os.environ["REGRESS_GALAXY_CAP"])
        rg.say(f"cap override for this invocation: {cap} (manifest default {sl['max_concurrent']})")
    rg.say(f"study Galaxy jobs now, running + pending, quad excepted ({len(jobs)}/{cap}): {jobs}")
    if len(jobs) >= cap and go:
        raise rg.RegressError("at the Galaxy allocation cap: wait")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    stage = os.path.join("/data/aagarwal/regress/stage", f"{stamp}-SEQ-{name}")
    os.makedirs(stage)
    for f in ("run_group.py", "manifest.yaml", "regress_junit_plugin.py", "regress_seq.sbatch", "dit_ring_joint_perf.py"):
        shutil.copy2(os.path.join(HERE, f), stage)
    if os.environ.get("REGRESS_MANIFEST"):  # a per-invocation manifest variant (e.g. an A/B's raised timeout)
        shutil.copy2(os.environ["REGRESS_MANIFEST"], os.path.join(stage, "manifest.yaml"))
        rg.say(f"manifest for this job: {os.environ['REGRESS_MANIFEST']}")
    with open(os.path.join(stage, "seq.txt"), "w") as f:
        for tree, path, group, label, extra in lines:
            f.write(f"{stage}|{stage}/manifest.yaml|{tree}|{path}|{group}|{label}|{extra}\n")
    excl = list(sl.get("exclude") or []) + (list(sl.get("exclude_for_ring") or []) if ring else [])
    excl += [n for n in (os.environ.get("REGRESS_GLX_EXCLUDE") or "").split(",") if n]   # per-invocation extra excludes
    part, extra_env = os.environ.get("REGRESS_GLX_PARTITION") or sl["partition"], ""   # per-invocation partition list
    nodelist = os.environ.get("REGRESS_GLX_NODELIST")                                    # per-invocation node pin
    if os.environ.get("REGRESS_GLX_SC36_2") == "1":  # one-time exception (coordinator 2026-11-04): bh_sc36_2 too; the job's
        part += ",bh_sc36_2"                        # run_group.py then accepts bh-glx-120-c* (galaxy_node_regex)
        extra_env = ",REGRESS_GLX_SC36_2=1"
        rg.say("REGRESS_GLX_SC36_2=1: partitions " + part + " (bh-glx-120-c* accepted by the job's node guard)")
    sb = ["sbatch", "--nodes=1", "--exclusive", f"--time={rg.fmt_time(minutes)}", f"--job-name=regress-seq-{name}",
          "--output=/data/aagarwal/regress/logs/%x-%j.log", "--chdir=/data/aagarwal/regress",
          f"--partition={part}", f"--exclude={','.join(sorted(set(excl)))}"] + ([f"--nodelist={nodelist}"] if nodelist else []) + [
          
          f"--export=ALL,REGRESS_SEQ={stage}/seq.txt,REGRESS_DATA_INODES_AT_SUBMIT={inodes}{extra_env}",
          os.path.join(stage, "regress_seq.sbatch")]
    print(open(os.path.join(stage, "seq.txt")).read().rstrip())
    print(" ".join(shlex.quote(x) for x in sb))
    if not go:
        r = subprocess.run(sb[:1] + ["--test-only"] + sb[1:], capture_output=True, text=True)
        rg.say(f"dry run (sbatch --test-only) rc={r.returncode}: {(r.stdout + r.stderr).strip()}")
        shutil.rmtree(stage, ignore_errors=True)  # a dry run leaves no stage dir behind (inodes)
        return r.returncode
    r = subprocess.run(sb, capture_output=True, text=True)
    rg.say(f"sbatch rc={r.returncode}: {(r.stdout + r.stderr).strip()}")
    return r.returncode


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rg.RegressError as e:
        rg.say(f"!!! {e}")
        sys.exit(1)
