#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that the fabric manifests one run's ranks wrote fit together.

Usage: check_manifest_ranks.py PATH [PATH ...]

Each directory holding fabric_manifest_rank_*.json files under a PATH is one run, and each tt-run sub-context in it
(run.subcontext.id) is one fabric: sub-contexts name their meshes and chips independently. A rank describes its own
chips in full and names other ranks' routers only by path, so a reader joins the files through those names. No single
rank can check them, so for each fabric this checks that:
  - every rank wrote the same run id, which names the run's folder
  - every chip is local in exactly one rank, so exactly one file describes it
  - every rank gives a chip the same host_rank, and each host rank of a mesh is one rank's: the ranks a mesh is split
    across each own whole host ranks
  - a router whose peer is on another rank's chip names a router that rank wrote, and that router names it back
  - a cross-host link ends at another rank's chip
Fails when a run doesn't, or when it finds no manifests.
"""

import json
import pathlib
import sys

MAX_ERRORS_SHOWN = 20


def find_fabrics(args):
    """Each fabric under args, keyed by its directory and sub-context id (None when the job isn't split), with its
    manifest paths."""
    fabrics = {}
    for arg in args:
        for path in sorted(pathlib.Path(arg).rglob("fabric_manifest_rank_*.json")):
            subcontext = json.loads(path.read_text())["run"].get("subcontext")
            subcontext_id = subcontext["id"] if subcontext is not None else None
            fabrics.setdefault((path.parent, subcontext_id), []).append(path)
    return fabrics


def find_chip_owners(manifests):
    """Each chip path, such as M0/C3, and the ranks whose manifests have it local."""
    owners = {}
    for rank, manifest in enumerate(manifests):
        for mesh_key, mesh in manifest["meshes"].items():
            for chip_key, chip in mesh["chips"].items():
                ranks = owners.setdefault(f"{mesh_key}/{chip_key}", [])
                if chip["is_local"]:
                    ranks.append(rank)
    return owners


def check_chip_owners(paths, owners):
    return [
        f"{chip_path} is local in {len(ranks)} ranks: {[paths[rank].name for rank in ranks]}"
        for chip_path, ranks in owners.items()
        if len(ranks) != 1
    ]


def check_host_ranks(paths, manifests, owners):
    """Every file gives each chip the same host_rank, and the chips of one host rank of a mesh are all one rank's."""
    problems = []
    host_ranks = {}
    for path, manifest in zip(paths, manifests):
        for mesh_key, mesh in manifest["meshes"].items():
            for chip_key, chip in mesh["chips"].items():
                host_ranks.setdefault(f"{mesh_key}/{chip_key}", {})[path.name] = chip["host_rank"]
    owner_of_host_rank = {}
    for chip_path, by_file in host_ranks.items():
        values = set(by_file.values())
        if len(values) != 1:
            problems.append(f"{chip_path} has host_rank {by_file} across files")
            continue
        chip_owners = owners.get(chip_path, [])
        if len(chip_owners) != 1:
            continue
        key = (chip_path.split("/")[0], values.pop())
        owner_of_host_rank.setdefault(key, set()).add(chip_owners[0])
    for (mesh_key, host_rank), ranks in owner_of_host_rank.items():
        if len(ranks) != 1:
            problems.append(
                f"{mesh_key} host rank {host_rank}'s chips are local in several ranks: {sorted(paths[r].name for r in ranks)}"
            )
    return problems


def find_router(manifest, router_path):
    mesh_key, chip_key, router_key = router_path.split("/")
    return manifest["meshes"][mesh_key]["chips"][chip_key].get("routers", {}).get(router_key)


def check_cross_rank_links(paths, manifests, owners):
    problems = []
    for rank, (path, manifest) in enumerate(zip(paths, manifests)):
        for mesh_key, mesh in manifest["meshes"].items():
            for chip_key, chip in mesh["chips"].items():
                for router_key, router in chip.get("routers", {}).items():
                    router_path = f"{mesh_key}/{chip_key}/{router_key}"
                    link = router["link"]
                    peer_path = link["peer"]
                    if peer_path is None:
                        continue
                    peer_owners = owners.get(peer_path.rsplit("/", 1)[0], [])
                    if len(peer_owners) != 1:
                        problems.append(f"{path.name}: {router_path}'s peer {peer_path} isn't on a chip one rank owns")
                        continue
                    peer_rank = peer_owners[0]
                    if peer_rank == rank:
                        if link["cross_host"]:
                            problems.append(
                                f"{path.name}: {router_path}'s link is cross-host, but its peer {peer_path} is local"
                            )
                        continue
                    peer = find_router(manifests[peer_rank], peer_path)
                    if peer is None:
                        problems.append(
                            f"{path.name}: {router_path}'s peer {peer_path} isn't in {paths[peer_rank].name}"
                        )
                    elif peer["link"]["peer"] != router_path:
                        problems.append(
                            f"{path.name}: {router_path}'s peer is {peer_path}, whose peer in "
                            f"{paths[peer_rank].name} is {peer['link']['peer']}"
                        )
    return problems


def check_run_ids(paths, manifests):
    """Every rank of the fabric wrote the same run id, and it's the name of the folder they're in."""
    run_ids = {path.name: manifest["run"]["run_id"] for path, manifest in zip(paths, manifests)}
    problems = []
    if len(set(run_ids.values())) != 1:
        problems.append(f"the ranks wrote different run ids: {run_ids}")
    for path, manifest in zip(paths, manifests):
        if path.parent.name != manifest["run"]["run_id"]:
            problems.append(f"{path.name} has run id {manifest['run']['run_id']} but is in {path.parent.name}")
    return problems


def check_fabric(paths):
    manifests = [json.loads(path.read_text()) for path in paths]
    owners = find_chip_owners(manifests)
    return (
        check_run_ids(paths, manifests)
        + check_chip_owners(paths, owners)
        + check_host_ranks(paths, manifests, owners)
        + check_cross_rank_links(paths, manifests, owners)
    )


def main(args):
    fabrics = find_fabrics(args)
    if not fabrics:
        print(f"Fabric manifest: no manifests found under {' '.join(args)}")
        return 1
    failed = 0
    for (directory, subcontext_id), paths in fabrics.items():
        name = str(directory) if subcontext_id is None else f"{directory} sub-context {subcontext_id}"
        problems = check_fabric(paths)
        if not problems:
            print(f"{name}: the {len(paths)}-rank run fits together")
            continue
        failed += 1
        print(f"Fabric manifest: the ranks in {name} don't fit together ({len(problems)} problems)")
        for problem in problems[:MAX_ERRORS_SHOWN]:
            print(f"  {problem}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
