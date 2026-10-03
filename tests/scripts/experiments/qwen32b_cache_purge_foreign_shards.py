# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Remove Galaxy tensor-cache files whose stored DRAM shard grid belongs to the other architecture (#58871).

ttnn.as_tensor() caches the host tensor together with its memory config, and load_tensor re-applies that stored
config on the device, so a DRAM-sharded weight written on Blackhole (8 DRAM banks) loads with 8 shards on Wormhole
(12 banks) and overflows the prefetcher ring. This script scans <TT_CACHE_PATH>/TG/tensor_cache_* for DRAM-sharded
files whose shard grid does not match the running architecture and deletes them (with --delete; the default is a
dry run), so the next test run regenerates them from the current architecture. Needs a read-write /mnt/MLPerf.
"""
import argparse
import glob
import os
import socket

import ttnn

PATTERNS = ("*prefetcher*", "*sharded*", "*lm_head*", "*dram_shard*")


def expected_dram_cores() -> int:
    arch = ttnn.get_arch_name().lower()
    if "blackhole" in arch:
        return 8
    if "wormhole" in arch:
        return 12
    raise SystemExit(f"unsupported arch {arch}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--root",
        default=os.path.join(os.environ.get("TT_CACHE_PATH", "/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B"), "TG"),
    )
    parser.add_argument("--delete", action="store_true", help="actually delete the foreign files (default: dry run)")
    parser.add_argument("--all-files", action="store_true", help="scan every .tensorbin, not just the sharded patterns")
    args = parser.parse_args()

    expected = expected_dram_cores()
    print(
        f"host={socket.gethostname()} arch={ttnn.get_arch_name()} root={args.root} expected_dram_cores={expected}",
        flush=True,
    )
    candidates = set()
    for sub in glob.glob(os.path.join(args.root, "tensor_cache_*")):
        pats = ("*.tensorbin",) if args.all_files else PATTERNS
        for pat in pats:
            candidates.update(glob.glob(os.path.join(sub, pat)))
    print(f"scanning {len(candidates)} files", flush=True)

    foreign, unreadable = [], []
    for f in sorted(candidates):
        try:
            t = ttnn.load_tensor(f)
            mc = t.memory_config()
        except Exception as e:  # noqa: BLE001 - diagnostic tool
            unreadable.append((f, str(e)[:200]))
            continue
        ss = getattr(mc, "shard_spec", None)
        if ss is None or mc.buffer_type != ttnn.BufferType.DRAM:
            continue
        ncores = ss.grid.num_cores()
        if ncores != expected:
            foreign.append((f, ncores))

    for f, n in foreign:
        print(f"FOREIGN shard_cores={n} {os.path.relpath(f, args.root)}", flush=True)
    for f, e in unreadable:
        print(f"UNREADABLE {os.path.relpath(f, args.root)}: {e}", flush=True)
    print(f"{len(foreign)} foreign files, {len(unreadable)} unreadable", flush=True)

    if args.delete:
        for f, _ in foreign:
            os.remove(f)
            print(f"deleted {os.path.relpath(f, args.root)}", flush=True)
        print(f"deleted {len(foreign)} files; rerun the model tests with write access to regenerate them", flush=True)
    elif foreign:
        print("dry run: rerun with --delete to remove them", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
