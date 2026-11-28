# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched DFlash, data parallel on a Blackhole Galaxy: one QuietBox 2 replica of gpt-oss-20b (1 x 4 tensor parallel)
per 2 x 2 ring of chips (demo/galaxy_dp.py), 8 on a Galaxy, each running tests/test_dflash_batched.py for its 32 - 256
users in its own process, all at once. Checks that every replica passes that test (its acceptance and its agreement
with the batch-32 decode), that the replicas, which serve the same prompts, decode the same tokens (they run the same
programs on identical rings, so any difference is a device-side race), and logs the aggregate tokens / s.

The replicas share the kernel cache. The first replica runs alone first, so that the programs are compiled once
(GALAXY_DP_WARM=0 skips it); GALAXY_DP_REPLICAS limits the number of replicas.

    pytest models/demos/gpt_oss/tests/test_dflash_galaxy_dp.py -k users_32
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from loguru import logger

from models.demos.gpt_oss.demo import galaxy_dp

BATCHED_TEST = Path(__file__).parent / "test_dflash_batched.py"
RESULT_ENV = "DFLASH_BATCHED_RESULT"  # (test_dflash_batched.RESULT_ENV, not imported: the parent opens no device)


def galaxy_rings():
    """The Galaxy's 2 x 2 rings, or skip (not a Blackhole Galaxy / no tt_umd)."""
    try:
        import tt_umd

        cluster = tt_umd.TopologyDiscovery.create_cluster_descriptor()
    except Exception as e:  # noqa: BLE001 - no devices / no bindings: not this system
        pytest.skip(f"no cluster descriptor: {e}")
    chips = cluster.get_all_chips()
    if len(chips) != 32 or str(cluster.get_arch(next(iter(chips)))).split(".")[-1].upper() != "BLACKHOLE":
        pytest.skip("data-parallel DFlash runs on a Blackhole Galaxy (32 chips)")
    rings = galaxy_dp.find_rings(galaxy_dp.chip_links(cluster))
    if not rings:
        pytest.skip("the Galaxy's chips do not tile into 2 x 2 rings with 2 links per edge")
    return rings


def run_replicas(rings, cpus, users, out_dir):
    """Run test_dflash_batched (users_<users>) on each ring at once; returns [(returncode, result or None, log)]."""
    procs = []
    for i, (ring, share) in enumerate(zip(rings, cpus)):
        env = galaxy_dp.replica_env(ring, share)
        env[RESULT_ENV] = str(out_dir / f"replica{i}.json")
        log = out_dir / f"replica{i}.log"
        cmd = [sys.executable, "-m", "pytest", str(BATCHED_TEST), "-k", f"users_{users}", "-s", "-p", "no:cacheprovider"]
        with open(log, "w") as f:
            procs.append((subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                           preexec_fn=galaxy_dp.pin(share)), log))  # fmt: skip
    out = []
    for i, (p, log) in enumerate(procs):
        rc = p.wait()
        path = out_dir / f"replica{i}.json"
        out.append((rc, json.loads(path.read_text()) if path.exists() else None, log))
    return out


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("users", [32, 64, 128, 256], ids=["users_32", "users_64", "users_128", "users_256"])
def test_dflash_galaxy_dp(users, tmp_path):
    rings = galaxy_rings()
    n = min(len(rings), int(os.getenv("GALAXY_DP_REPLICAS", len(rings))))
    rings = rings[:n]
    cpus = galaxy_dp.cpu_shares(n)
    logger.info(f"{n} replicas on rings {rings}, {len(cpus[0])} host cores each")

    if os.getenv("GALAXY_DP_WARM", "1") == "1":  # compile the programs once
        warm = tmp_path / "warm"
        warm.mkdir()
        ((rc, _, log),) = run_replicas(rings[:1], [cpus[0]], users, warm)
        assert rc == 0, f"warm-up replica failed:\n{log.read_text()[-4000:]}"

    t = time.perf_counter()
    results = run_replicas(rings, cpus, users, tmp_path)
    wall = time.perf_counter() - t
    for i, (rc, res, log) in enumerate(results):
        assert rc == 0 and res is not None, f"replica {i} on {rings[i]} failed:\n{log.read_text()[-4000:]}"
        logger.info(
            f"replica {i} {rings[i]}: {res['rounds']} rounds of {res['round_ms']:.1f} ms, "
            f"{res['tokens_per_round']:.2f} tokens per round, {res['tok_s_user']:.1f} tok/s/user, "
            f"agreement {res['agreement']}"
        )
    res = [r for _, r, _ in results]
    total_users = sum(r["users"] for r in res)
    total = sum(r["users"] * r["tok_s_user"] for r in res)
    logger.info(
        f"data-parallel DFlash: {n} replicas x {users} users = {total_users} users, "
        f"{total / total_users:.1f} tok/s/user, {total:.0f} tok/s in all ({wall:.0f} s wall)"
    )
    hashes = {r["tokens_sha256"] for r in res}
    assert len(hashes) == 1, f"the replicas decoded different tokens: {[r['tokens_sha256'][:12] for r in res]}"
