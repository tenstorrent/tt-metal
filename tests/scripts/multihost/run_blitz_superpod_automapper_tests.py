#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Blitz superpod mapping determinism tests.

Mock / CPU sim (--mock-cluster-rank-binding): canonical run + randomized variations (mock mapping shuffle + MGD relabel + mesh_id permutation).
  Variation MGDs rename mesh/graph descriptors and permute mesh_id values consistently in instances and connections;
  connection graph structure is unchanged (topology-invariant determinism check).
  With --golden: mock canonical uses exact gtest compare; all variation mappings use topology-invariant Python compare vs golden.
  Without --golden: variations compare against the mapping captured from the canonical run.
  Real hardware: gtest skips YAML golden compare (real devices); Python compares every run to ``--golden``
  or the canonical (first) iteration. Canonical and variations use the same strict compare: hostname,
  tray_id, asic_location, mesh_id, chip_id, and asic_id must all match the reference.

  **Canonical** — first run with the original mock binding / ``--hosts`` order and canonical MGD.
  **Variation** — later runs that shuffle inputs (mock: rank-binding + permuted MGD mesh_ids;
  hardware: shuffled ``--hosts`` order plus a rotated tt-run launch host) to verify mapping output stays identical.

Mock and hardware tt-run invocations always pass ``--force-rediscovery`` so Phase 1 rank bindings are regenerated
  (avoids stale cache from a prior run with different hosts or mock mapping).
Real hardware (--hosts): canonical run + optional variations (same MGD and new-mode tt-run as canonical).
  Each variation passes a random ``--hosts`` order (seeded) to tt-run and launches tt-run from a different
  cluster node (rotated by variation index) to verify mapping output is invariant to launch host and
  host-list order.

  When launching from a login/jump host, use ``ssh-add`` and ``--ssh-bootstrap`` once. Variation runs SSH
  tt-run to a cluster node; MPI on that node uses node-local keys to reach peers.

Remote hardware launch (login node -> compute cluster):
  Use --ssh-bootstrap to verify each host, then --launch-host to run tt-run on a cluster
  node (MPI/tt-run still coordinates all ranks; OpenMPI SSHes from the launch host to workers).

Requires a built ``fabric_unit_tests`` binary and the same ``TT_METAL_HOME`` path on all hosts.

Examples:
  # Mock cluster / CI (canonical + 5 variations)
  python_env/bin/python3 tests/scripts/multihost/run_blitz_superpod_automapper_tests.py \\
      --mock-cluster-rank-binding tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/sp4_glx_cluster_desc_mapping.yaml \\
      --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/fabric_cpu_only_blitz_superpod_mesh_graph_descriptor.textproto

  # Real superpod (16 physical hosts; tt-run launched from first host if not already on-cluster)
  python_env/bin/python3 tests/scripts/multihost/run_blitz_superpod_automapper_tests.py \\
      --hosts bh-glx-d03u02,bh-glx-d03u08,bh-glx-d04u02,... \\
      --mesh-graph-descriptor models/demos/deepseek_v3_b1/scaleout_configs/blitz_decode_mesh_graph_descriptor_superpod.textproto \\
      --tcp-interface cnx1

  # Real single pod (4 hosts; canonical only)
  python_env/bin/python3 tests/scripts/multihost/run_blitz_superpod_automapper_tests.py \\
      --hosts bh-glx-c05u02,bh-glx-c05u08,bh-glx-c06u02,bh-glx-c06u08 \\
      --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/fabric_cpu_only_blitz_single_pod_mesh_graph_descriptor.textproto \\
      --tcp-interface cnx1 --num-variations 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
# This script lives at tests/scripts/multihost/, so the repo root is three levels up
# (parents[2]). Prefer TT_METAL_HOME when set (the suite exports it), matching the shell
# wrapper's "$SCRIPT_DIR/../../.." resolution.
REPO_ROOT = Path(os.environ.get("TT_METAL_HOME") or SCRIPT_DIR.parents[2]).resolve()

CANONICAL_REFERENCE_MAPPING = REPO_ROOT / "generated/blitz_superpod_automapper/canonical_reference_mapping.yaml"
DEFAULT_RUN_LOG = REPO_ROOT / "generated/blitz_superpod_automapper/automapper_test.log"

DEFAULT_CANONICAL_MESH_DESCRIPTOR = "M0"
DEFAULT_CANONICAL_GRAPH_DESCRIPTOR = "G0"
MESH_INSTANCES_PATTERN = re.compile(r"^\s*instances\s*\{\s*mesh\s*\{", re.MULTILINE)
MESH_ID_PATTERN = re.compile(r"mesh_id:\s*(\d+)")

# Shared TT_METAL_HOME (NFS): mapping YAML may appear on this host shortly after MPI exits.
MAPPING_SYNC_POLL_INTERVAL_SEC = 0.5
MAPPING_SYNC_MAX_WAIT_SEC = 1.0


class AutomapperTestLog:
    """Append structured test metadata, commands, mapping paths, and compare mismatches."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("w", encoding="utf-8")
        self._write(f"Blitz superpod mapping determinism test log")
        self._write(f"Started (UTC): {datetime.now(timezone.utc).isoformat()}")
        self._write(f"Log file: {self.path}")
        self._write(f"Script host: {socket.gethostname()}")
        self._write(f"TT_METAL_HOME: {REPO_ROOT}")
        self._write(f"Script invocation: {' '.join(shlex.quote(part) for part in sys.argv)}")
        self._write("")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def _write(self, line: str = "") -> None:
        self._fh.write(line + "\n")
        self._fh.flush()

    def log_session_config(
        self,
        *,
        hardware_mode: bool,
        mgd: Path,
        world_size: int,
        mock_mapping: Optional[Path],
        hosts: Optional[str],
        golden_path: Optional[Path],
        variations_dir: Path,
        num_variations: int,
        seed: int,
        launch_host: Optional[str],
        ssh_user: Optional[str],
        tcp_interface: Optional[str],
        build_dir: Path,
    ) -> None:
        self._write("=== Session configuration ===")
        self._write(f"Mode: {'hardware' if hardware_mode else 'mock'}")
        self._write(f"MGD (canonical): {mgd}")
        self._write(f"MPI world size: {world_size}")
        if mock_mapping is not None:
            self._write(f"Mock cluster rank binding: {mock_mapping}")
        if hosts is not None:
            self._write(f"--hosts: {hosts}")
        self._write(f"Golden / reference file: {golden_path if golden_path else '(canonical artifact)'}")
        self._write(f"Variations dir: {variations_dir}")
        self._write(f"num_variations: {num_variations}")
        self._write(f"seed: {seed}")
        self._write(f"Launch host (tt-run): {effective_launch_description(launch_host, ssh_user)}")
        self._write(f"tcp_interface: {tcp_interface or '(default)'}")
        self._write(f"Build dir: {build_dir}")
        self._write("")

    def log_run_start(
        self,
        *,
        run_label: str,
        mode: str,
        mgd: Path,
        launch_host: Optional[str],
        ssh_user: Optional[str],
        command: Sequence[str],
        mock_mapping: Optional[Path] = None,
        hosts: Optional[str] = None,
        extra: Optional[dict[str, object]] = None,
    ) -> None:
        self._write(f"--- Run: {run_label} ---")
        self._write(f"Mode: {mode}")
        self._write(f"MGD: {mgd}")
        self._write(f"Launch host: {effective_launch_description(launch_host, ssh_user)}")
        if mock_mapping is not None:
            self._write(f"Mock cluster rank binding: {mock_mapping}")
        if hosts is not None:
            self._write(f"--hosts: {hosts}")
        if extra:
            for key, value in extra.items():
                self._write(f"{key}: {value}")
        self._write(f"Command: {' '.join(shlex.quote(part) for part in command)}")

    def log_run_finished(self, *, run_label: str, exit_code: int, remote_launch: bool) -> None:
        self._write(f"Remote tt-run launch: {'yes' if remote_launch else 'no'}")
        self._write(f"Exit code: {exit_code}")
        self._write(f"Run result: {'PASSED' if exit_code == 0 else 'FAILED'}")
        self._write("")

    def log_mapping_artifact(
        self,
        *,
        run_label: str,
        generated_path: Path,
        artifact_path: Path,
        reference_path: Optional[Path] = None,
    ) -> None:
        self._write(f"Mapping ({run_label}):")
        self._write(f"  generated: {generated_path}")
        self._write(f"  saved artifact: {artifact_path}")
        if reference_path is not None:
            self._write(f"  reference: {reference_path}")

    def log_compare_result(
        self,
        *,
        run_label: str,
        mapping_artifact: Path,
        reference_mapping: Path,
        mismatches: List[str],
    ) -> None:
        self._write(f"Compare ({run_label}):")
        self._write(f"  artifact: {mapping_artifact}")
        self._write(f"  reference: {reference_mapping}")
        if mismatches:
            self._write(f"  result: FAIL ({len(mismatches)} mismatch(es))")
            self._write("  mismatches:")
            for line in mismatches:
                self._write(f"    - {line}")
        else:
            self._write("  result: PASS (exact match: hostname, tray_id, asic_location, mesh_id, chip_id, asic_id)")
        self._write("")

    def log_final_status(self, status: str, detail: Optional[str] = None) -> None:
        self._write("=== Final status ===")
        self._write(f"Status: {status}")
        if detail:
            self._write(f"Detail: {detail}")
        self._write(f"Finished (UTC): {datetime.now(timezone.utc).isoformat()}")


def effective_launch_description(launch_host: Optional[str], ssh_user: Optional[str]) -> str:
    local = socket.gethostname()
    if launch_host is None:
        return f"{local} (local)"
    if hostname_matches(local, launch_host):
        return f"{local} (local, matches launch host)"
    return f"{ssh_connection_target(launch_host, ssh_user)} (remote SSH launch from {local})"


def is_remote_tt_run_launch(launch_host: Optional[str]) -> bool:
    if launch_host is None:
        return False
    return not hostname_matches(socket.gethostname(), launch_host)


def generated_mapping_path(world_size: int, *, rank_one_based: int = 1) -> Path:
    return REPO_ROOT / "generated/fabric" / f"asic_to_fabric_node_mapping_rank_{rank_one_based}_of_{world_size}.yaml"


def cleanup_generated_mapping_files(world_size: int) -> None:
    """Remove stale per-rank mapping YAML before a run (shared FS may retain old rank-1)."""
    fabric_dir = REPO_ROOT / "generated/fabric"
    if not fabric_dir.is_dir():
        return
    pattern = f"asic_to_fabric_node_mapping_rank_*_of_{world_size}.yaml"
    for path in fabric_dir.glob(pattern):
        path.unlink(missing_ok=True)


def _mapping_relpath(world_size: int, rank_one_based: int) -> str:
    return f"generated/fabric/asic_to_fabric_node_mapping_rank_{rank_one_based}_of_{world_size}.yaml"


def _wait_for_local_mapping_file(
    world_size: int,
    *,
    max_wait_sec: float = MAPPING_SYNC_MAX_WAIT_SEC,
    poll_interval_sec: float = MAPPING_SYNC_POLL_INTERVAL_SEC,
) -> Optional[Path]:
    """Poll until rank-1 mapping appears locally (shared filesystem sync after MPI exit)."""
    local_path = generated_mapping_path(world_size)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max_wait_sec
    while time.monotonic() < deadline:
        if local_path.is_file():
            return local_path
        for rank_one_based in range(2, world_size + 1):
            alt = generated_mapping_path(world_size, rank_one_based=rank_one_based)
            if alt.is_file():
                shutil.copy2(alt, local_path)
                print(
                    f"Using rank-{rank_one_based} mapping after shared-fs sync as rank-1 reference",
                    flush=True,
                )
                return local_path
        time.sleep(poll_interval_sec)
    return None


def ensure_generated_mapping_local(
    *,
    world_size: int,
    hosts: list[str],
    launch_host: Optional[str] = None,
    ssh_user: Optional[str] = None,
    mapping_sync_wait_sec: float = MAPPING_SYNC_MAX_WAIT_SEC,
) -> Path:
    """Ensure rank-1 mapping exists locally after a hardware tt-run.

    Waits briefly for shared ``TT_METAL_HOME`` to sync, then falls back to fetching
    from cluster hosts over SSH when the file never appears locally.
    """
    local_path = generated_mapping_path(world_size)
    if local_path.is_file():
        return local_path

    synced = _wait_for_local_mapping_file(world_size, max_wait_sec=mapping_sync_wait_sec)
    if synced is not None:
        print(
            f"Generated mapping available locally after shared-fs sync: {synced}",
            flush=True,
        )
        return synced

    print(
        f"Mapping not visible locally after {mapping_sync_wait_sec:.0f}s; fetching from cluster hosts...",
        flush=True,
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tt_home = str(REPO_ROOT)

    search_hosts: list[str] = []
    if launch_host:
        search_hosts.append(launch_host)
    for host in hosts:
        if host not in search_hosts:
            search_hosts.append(host)

    local_hostname = socket.gethostname()
    errors: list[str] = []
    for host in search_hosts:
        for rank_one_based in range(1, world_size + 1):
            rel = _mapping_relpath(world_size, rank_one_based)
            remote_file = f"{tt_home}/{rel}"
            dest = REPO_ROOT / rel

            if hostname_matches(local_hostname, host):
                if dest.is_file():
                    if rank_one_based != 1:
                        shutil.copy2(dest, local_path)
                        print(
                            f"Using rank-{rank_one_based} mapping locally as rank-1 reference: {dest}",
                            flush=True,
                        )
                    return local_path
                continue

            target = ssh_connection_target(host, ssh_user)
            probe = subprocess.run(
                ssh_command(target, "test", "-f", remote_file),
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if probe.returncode != 0:
                continue

            scp_cmd = ["scp", *SSH_OPTS, f"{target}:{remote_file}", str(dest)]
            result = subprocess.run(scp_cmd, capture_output=True, text=True, env=os.environ.copy())
            if result.returncode != 0:
                errors.append(f"{host}:{rel}: scp failed ({result.stderr.strip() or result.stdout.strip()})")
                continue

            if rank_one_based != 1:
                shutil.copy2(dest, local_path)
                print(
                    f"Fetched rank-{rank_one_based} mapping from {target} as rank-1 reference",
                    flush=True,
                )
            else:
                print(f"Fetched generated mapping from {target}:{remote_file}", flush=True)
            return local_path

    tried = ", ".join(search_hosts)
    detail = "\n".join(errors) if errors else "No rank mapping file found on any host."
    raise FileNotFoundError(
        f"Missing generated mapping after hardware run: {local_path}\n"
        f"Searched hosts (MPI rank-0 is usually on the first --hosts entry): {tried}\n"
        f"{detail}"
    )


DEFAULT_MOCK_VARIATIONS_DIR = REPO_ROOT / "generated/blitz_superpod_variations"
DEFAULT_HARDWARE_VARIATIONS_DIR = REPO_ROOT / "generated/blitz_superpod_hardware_variations"
GTEST_FILTER = "MultiHost.TestBlitzSuperpodAutoMapperControlPlaneInit"

# --- MGD / mock variation generation ---------------------------------------------------------------


def count_mesh_instances(mgd_path: Path) -> int:
    """Return the number of mesh instances declared in an MGD (MPI / tt-run world size)."""
    text = mgd_path.read_text()
    count = len(MESH_INSTANCES_PATTERN.findall(text))
    if count == 0:
        raise ValueError(f"No mesh instances found in MGD: {mgd_path}")
    return count


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_canonical_mock_mapping(mapping_path: Path) -> Dict[int, str]:
    data = yaml.safe_load(mapping_path.read_text())
    raw = data["rank_to_cluster_mock_cluster_desc"]
    return {int(rank): str(path) for rank, path in raw.items()}


def write_mock_mapping(output_path: Path, rank_to_desc: Dict[int, str], comment: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [comment, "rank_to_cluster_mock_cluster_desc:"]
    for rank in sorted(rank_to_desc):
        lines.append(f'  {rank}: "{rank_to_desc[rank]}"')
    output_path.write_text("\n".join(lines) + "\n")


def randomize_mock_mapping(rng: random.Random, canonical: Dict[int, str]) -> Dict[int, str]:
    num_mock_ranks = len(canonical)
    descriptor_paths = [canonical[rank] for rank in range(num_mock_ranks)]
    shuffled = descriptor_paths.copy()
    rng.shuffle(shuffled)
    return {rank: shuffled[rank] for rank in range(num_mock_ranks)}


def apply_mesh_id_permutation(mgd_text: str, mesh_id_permutation: List[int], *, num_meshes: int) -> str:
    """Relabel mesh_id values consistently in instances, connections, and all other MGD references."""
    if len(mesh_id_permutation) != num_meshes:
        raise ValueError(f"Expected {num_meshes} mesh ids, got {len(mesh_id_permutation)}")
    if sorted(mesh_id_permutation) != list(range(num_meshes)):
        raise ValueError(f"mesh_id_permutation must be a permutation of 0..{num_meshes - 1}")

    old_to_new = {old_id: mesh_id_permutation[old_id] for old_id in range(num_meshes)}

    def repl(match: re.Match[str]) -> str:
        return f"mesh_id: {old_to_new[int(match.group(1))]}"

    return MESH_ID_PATTERN.sub(repl, mgd_text)


def generate_mesh_id_permutation(rng: random.Random, num_meshes: int) -> List[int]:
    perm = list(range(num_meshes))
    rng.shuffle(perm)
    return perm


def relabel_mgd_descriptors_only(
    mgd_text: str,
    *,
    old_mesh_name: str,
    new_mesh_name: str,
    old_graph_name: str,
    new_graph_name: str,
    variation_index: int,
    seed: int,
    mesh_id_permutation: Optional[List[int]] = None,
) -> str:
    """Build a variation MGD: new descriptor names; optional mesh_id permutation already applied."""
    perm_note = (
        f"mesh_id permutation: {mesh_id_permutation}" if mesh_id_permutation is not None else "mesh_id order unchanged"
    )
    header = (
        f"# Auto-generated variation {variation_index} (seed={seed}).\n"
        f"# Descriptor relabel + {perm_note}.\n"
        "# Connection graph unchanged; instances and connections use permuted mesh_id values.\n"
    )
    updated = mgd_text
    if updated.startswith("# Auto-generated variation"):
        updated = updated.split("\n", 2)[2] if updated.count("\n") >= 2 else updated

    updated = updated.replace(f'mesh_descriptor: "{old_mesh_name}"', f'mesh_descriptor: "{new_mesh_name}"')
    updated = updated.replace(f'graph_descriptor: "{old_graph_name}"', f'graph_descriptor: "{new_graph_name}"')
    updated = updated.replace(f'name: "{old_mesh_name}"', f'name: "{new_mesh_name}"', 1)
    updated = updated.replace(f'name: "{old_graph_name}"', f'name: "{new_graph_name}"', 1)
    return header + updated


def connection_edge_set(mgd_text: str) -> set[tuple[int, int, int, int]]:
    """Directed connection edges: (src_mesh_id, dst_mesh_id, channel_count, assign_z)."""
    edges: set[tuple[int, int, int, int]] = set()
    connection_blocks = re.findall(
        r"connections\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
        mgd_text,
        flags=re.DOTALL,
    )
    for block in connection_blocks:
        ids = [int(m.group(1)) for m in MESH_ID_PATTERN.finditer(block)]
        if len(ids) < 2:
            continue
        channels = re.search(r"count:\s*(\d+)", block)
        channel_count = int(channels.group(1)) if channels else -1
        assign_z = 1 if re.search(r"assign_z_direction:\s*true", block) else 0
        edges.add((ids[0], ids[1], channel_count, assign_z))
    return edges


def instance_mesh_ids(mgd_text: str) -> list[int]:
    """mesh_id values declared in graph_descriptors instance blocks (declaration order)."""
    start = mgd_text.find("graph_descriptors")
    if start < 0:
        return []
    top = mgd_text.find("top_level_instance", start)
    section = mgd_text[start:top] if top >= 0 else mgd_text[start:]
    return [int(m.group(1)) for m in re.finditer(r"instances\s*\{.*?mesh_id:\s*(\d+)", section, flags=re.DOTALL)]


def assert_mgd_mesh_id_permutation_preserves_topology(
    *,
    canonical_text: str,
    variant_text: str,
    mesh_id_permutation: List[int],
) -> None:
    old_to_new = {old_id: mesh_id_permutation[old_id] for old_id in range(len(mesh_id_permutation))}
    canon_edges = connection_edge_set(canonical_text)
    var_edges = connection_edge_set(variant_text)
    expected_edges = {
        (old_to_new[src], old_to_new[dst], channels, assign_z) for src, dst, channels, assign_z in canon_edges
    }
    if var_edges != expected_edges:
        raise ValueError("Variation MGD connection blocks do not match canonical topology under mesh_id permutation")

    canon_instances = instance_mesh_ids(canonical_text)
    var_instances = instance_mesh_ids(variant_text)
    expected_instances = [old_to_new[mesh_id] for mesh_id in canon_instances]
    if var_instances != expected_instances:
        raise ValueError("Variation MGD instance mesh_id values do not match canonical under mesh_id permutation")


def generate_variation_names(rng: random.Random, variation_index: int) -> tuple[str, str]:
    suffix = rng.randint(1000, 9999)
    return f"M_blitz_var{variation_index}_{suffix}", f"G_blitz_var{variation_index}_{suffix}"


def generate_mock_variations(
    *,
    output_dir: Path,
    num_variations: int,
    seed: int,
    canonical_mock_mapping: Path,
    canonical_mgd: Path,
) -> dict:
    if not canonical_mock_mapping.is_file():
        raise FileNotFoundError(f"Canonical mock mapping not found: {canonical_mock_mapping}")
    if not canonical_mgd.is_file():
        raise FileNotFoundError(f"Canonical MGD not found: {canonical_mgd}")

    canonical_mock = load_canonical_mock_mapping(canonical_mock_mapping)
    canonical_mgd_text = canonical_mgd.read_text()
    num_meshes = count_mesh_instances(canonical_mgd)
    num_mock_ranks = len(canonical_mock)

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "mode": "mock",
        "seed": seed,
        "num_variations": num_variations,
        "num_meshes": num_meshes,
        "canonical_mock_mapping": _repo_relative(canonical_mock_mapping),
        "canonical_mgd": _repo_relative(canonical_mgd),
        "variations": [],
    }

    for variation_index in range(num_variations):
        variation_rng = random.Random(rng.randint(0, 2**31 - 1))
        variation_dir = output_dir / f"variation_{variation_index}"
        variation_dir.mkdir(parents=True, exist_ok=True)

        mock_mapping = randomize_mock_mapping(variation_rng, canonical_mock)
        mesh_id_perm = generate_mesh_id_permutation(variation_rng, num_meshes)
        mesh_name, graph_name = generate_variation_names(variation_rng, variation_index)

        mock_mapping_path = variation_dir / "mock_cluster_desc_mapping.yaml"
        write_mock_mapping(
            mock_mapping_path,
            mock_mapping,
            (
                f"# Auto-generated variation {variation_index} (seed={seed}). "
                "Random MPI rank -> mock cluster descriptor shuffle."
            ),
        )

        mgd_variant = apply_mesh_id_permutation(canonical_mgd_text, mesh_id_perm, num_meshes=num_meshes)
        mgd_variant = relabel_mgd_descriptors_only(
            mgd_variant,
            old_mesh_name=DEFAULT_CANONICAL_MESH_DESCRIPTOR,
            new_mesh_name=mesh_name,
            old_graph_name=DEFAULT_CANONICAL_GRAPH_DESCRIPTOR,
            new_graph_name=graph_name,
            variation_index=variation_index,
            seed=seed,
            mesh_id_permutation=mesh_id_perm,
        )
        assert_mgd_mesh_id_permutation_preserves_topology(
            canonical_text=canonical_mgd_text,
            variant_text=mgd_variant,
            mesh_id_permutation=mesh_id_perm,
        )
        mgd_path = variation_dir / "mesh_graph_descriptor.textproto"
        mgd_path.write_text(mgd_variant)

        manifest["variations"].append(
            {
                "index": variation_index,
                "mock_mapping": _repo_relative(mock_mapping_path),
                "mgd": _repo_relative(mgd_path),
                "mesh_descriptor_name": mesh_name,
                "graph_descriptor_name": graph_name,
                "mesh_id_permutation": mesh_id_perm,
                "mock_rank_shuffle": [mock_mapping[rank] for rank in range(num_mock_ranks)],
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# --- Golden / mapping compare (Python; mock + hardware, canonical + variations) -----------------


def mapping_artifact_filename(world_size: int) -> str:
    return f"asic_to_fabric_node_mapping_rank_1_of_{world_size}.yaml"


def _iter_chip_entries(hostnames_node) -> List[Tuple[str, dict]]:
    """Yield (hostname, chip_dict) for every chip in the mapping YAML."""
    entries: List[Tuple[str, dict]] = []
    host_entries = hostnames_node if isinstance(hostnames_node, list) else []
    for host_entry in host_entries:
        if not isinstance(host_entry, dict):
            continue
        hostname = str(host_entry.get("hostname", ""))
        for mesh_block in host_entry.get("mesh", []):
            if not isinstance(mesh_block, dict):
                continue
            for chip in mesh_block.get("chips", []):
                if isinstance(chip, dict):
                    entries.append((hostname, chip))
    return entries


def _load_hostnames_node(yaml_path: Path):
    data = yaml.safe_load(yaml_path.read_text())
    return data["asic_to_fabric_node_mapping"]["hostnames"]


def collect_fabric_node_by_fabric_node_id(hostnames_node) -> Dict[str, Tuple[str, dict]]:
    """Index chips by ``mesh_id:chip_id`` (matches gtest golden compare)."""
    by_fabric: Dict[str, Tuple[str, dict]] = {}
    for hostname, chip in _iter_chip_entries(hostnames_node):
        fabric_node = chip.get("fabric_node_id")
        if not isinstance(fabric_node, dict):
            continue
        key = f"{int(fabric_node['mesh_id'])}:{int(fabric_node['chip_id'])}"
        by_fabric[key] = (hostname, chip)
    return by_fabric


def compare_asic_mapping_yaml_exact(generated_file: Path, reference_file: Path) -> List[str]:
    """Exact golden compare keyed by ``fabric_node_id`` (matches gtest ``compare_asic_mapping_files``)."""
    gen_by_fabric = collect_fabric_node_by_fabric_node_id(_load_hostnames_node(generated_file))
    ref_by_fabric = collect_fabric_node_by_fabric_node_id(_load_hostnames_node(reference_file))

    mismatches: List[str] = []
    if len(gen_by_fabric) != len(ref_by_fabric):
        mismatches.append(f"fabric_node entry count: generated={len(gen_by_fabric)}, reference={len(ref_by_fabric)}")

    for key in sorted(set(gen_by_fabric) - set(ref_by_fabric)):
        mismatches.append(f"fabric_node missing in reference: {key}")
    for key in sorted(set(ref_by_fabric) - set(gen_by_fabric)):
        mismatches.append(f"fabric_node missing in generated: {key}")

    for key in sorted(set(gen_by_fabric) & set(ref_by_fabric)):
        gen_host, gen_chip = gen_by_fabric[key]
        ref_host, ref_chip = ref_by_fabric[key]
        field_mismatches: List[str] = []
        if gen_host != ref_host:
            field_mismatches.append(f'hostname: generated="{gen_host}", reference="{ref_host}"')
        for field, yaml_key in (
            ("tray_id", ("asic_position", "tray_id")),
            ("asic_location", ("asic_position", "asic_location")),
            ("mesh_id", ("fabric_node_id", "mesh_id")),
            ("chip_id", ("fabric_node_id", "chip_id")),
        ):
            gen_val = gen_chip[yaml_key[0]][yaml_key[1]]
            ref_val = ref_chip[yaml_key[0]][yaml_key[1]]
            if int(gen_val) != int(ref_val):
                field_mismatches.append(f"{field}: generated={gen_val}, reference={ref_val}")
        if int(gen_chip["asic_id"]) != int(ref_chip["asic_id"]):
            field_mismatches.append(f"asic_id: generated={gen_chip['asic_id']}, reference={ref_chip['asic_id']}")
        if field_mismatches:
            mismatches.append(f"fabric_node {key}: " + ", ".join(field_mismatches))

    return mismatches


def _index_by_asic_id(hostnames_node) -> Dict[int, Tuple[str, dict]]:
    """Index chips by physical ``asic_id`` (stable identity across mock variations)."""
    by_asic: Dict[int, Tuple[str, dict]] = {}
    for hostname, chip in _iter_chip_entries(hostnames_node):
        if "asic_id" in chip:
            by_asic[int(chip["asic_id"])] = (hostname, chip)
    return by_asic


def compare_asic_mapping_topology_invariant(generated_file: Path, reference_file: Path) -> List[str]:
    """Topology-invariant determinism compare for mock variations, keyed by physical ``asic_id``.

    A mock variation shuffles the rank->descriptor binding and permutes the MGD ``mesh_id``
    labels (a consistent relabel that leaves the connection graph unchanged). A deterministic
    mapper must therefore select the *same physical chips* (identical ``asic_id`` set) and place
    each chip identically -- hostname, tray_id, asic_location, and chip_id preserved per
    ``asic_id`` -- with the only permitted difference being a *consistent, bijective* global
    ``mesh_id`` relabeling. Any physical reassignment, dropped chip, or non-bijective mesh_id
    relabel (i.e. real non-determinism) is reported as a mismatch.
    """
    gen = _index_by_asic_id(_load_hostnames_node(generated_file))
    ref = _index_by_asic_id(_load_hostnames_node(reference_file))

    mismatches: List[str] = []
    if len(gen) != len(ref):
        mismatches.append(f"asic_id entry count: generated={len(gen)}, reference={len(ref)}")
    for asic in sorted(set(gen) - set(ref)):
        mismatches.append(f"asic_id present in generated but missing in reference: {asic}")
    for asic in sorted(set(ref) - set(gen)):
        mismatches.append(f"asic_id present in reference but missing in generated: {asic}")

    ref_to_gen_mesh: Dict[int, int] = {}
    gen_to_ref_mesh: Dict[int, int] = {}
    for asic in sorted(set(gen) & set(ref)):
        gen_host, gen_chip = gen[asic]
        ref_host, ref_chip = ref[asic]
        field_mismatches: List[str] = []
        if gen_host != ref_host:
            field_mismatches.append(f'hostname: generated="{gen_host}", reference="{ref_host}"')
        for field, yaml_key in (
            ("tray_id", ("asic_position", "tray_id")),
            ("asic_location", ("asic_position", "asic_location")),
            ("chip_id", ("fabric_node_id", "chip_id")),
        ):
            gen_val = int(gen_chip[yaml_key[0]][yaml_key[1]])
            ref_val = int(ref_chip[yaml_key[0]][yaml_key[1]])
            if gen_val != ref_val:
                field_mismatches.append(f"{field}: generated={gen_val}, reference={ref_val}")
        gen_mesh = int(gen_chip["fabric_node_id"]["mesh_id"])
        ref_mesh = int(ref_chip["fabric_node_id"]["mesh_id"])
        if ref_to_gen_mesh.setdefault(ref_mesh, gen_mesh) != gen_mesh:
            field_mismatches.append(
                f"mesh_id relabel inconsistent: reference mesh {ref_mesh} maps to both "
                f"{ref_to_gen_mesh[ref_mesh]} and {gen_mesh}"
            )
        if gen_to_ref_mesh.setdefault(gen_mesh, ref_mesh) != ref_mesh:
            field_mismatches.append(
                f"mesh_id relabel not bijective: generated mesh {gen_mesh} maps from both "
                f"{gen_to_ref_mesh[gen_mesh]} and {ref_mesh}"
            )
        if field_mismatches:
            ref_chip_id = int(ref_chip["fabric_node_id"]["chip_id"])
            mismatches.append(f"asic_id {asic} (reference {ref_mesh}:{ref_chip_id}): " + ", ".join(field_mismatches))

    return mismatches


# --- SSH / tt-run launch ---------------------------------------------------------------------------

# Open MPI uses this command for node->node rank launch.
# StrictHostKeyChecking=accept-new rejects *changed* host keys (common after reprovisioning);
# use no + /dev/null like tools/scaleout/exabox/run_fabric_tests.sh for cluster automation.
MPI_SSH_OPTS = "-o BatchMode=yes " "-o StrictHostKeyChecking=no " "-o UserKnownHostsFile=/dev/null " "-o LogLevel=ERROR"
MPI_PLM_RSH_AGENT = f"ssh {MPI_SSH_OPTS} -o ForwardAgent=yes"
MPI_PLM_RSH_AGENT_NO_FORWARD = f"ssh {MPI_SSH_OPTS}"
SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ForwardAgent=yes",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
]


def ssh_command(*args: str) -> list[str]:
    return ["ssh", *SSH_OPTS, *args]


def warn_if_ssh_agent_missing(*, hardware_mode: bool) -> None:
    if hardware_mode and not os.environ.get("SSH_AUTH_SOCK"):
        print(
            "WARNING: SSH_AUTH_SOCK is not set. Hardware MPI needs an SSH agent with your cluster "
            "keys loaded (run `ssh-add` on the machine launching this script). Without it, mpirun "
            "cannot reach peer nodes unless a full node-to-node key mesh exists.",
            flush=True,
        )


def _python_executable() -> str:
    venv_python = REPO_ROOT / "python_env/bin/python3"
    return str(venv_python) if venv_python.is_file() else sys.executable


def resolve_build_dir(explicit: Optional[Path]) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    env_build = os.environ.get("TT_METAL_BUILD_DIR")
    if env_build:
        candidates.append(Path(env_build))
    candidates.extend([REPO_ROOT / "build", REPO_ROOT / "build_Debug"])
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "test/tt_metal/tt_fabric/fabric_unit_tests").is_file():
            return candidate
    raise FileNotFoundError("fabric_unit_tests not found. Build the target or set TT_METAL_BUILD_DIR.")


def default_mpi_args(*, hardware_mode: bool) -> list[str]:
    if not hardware_mode:
        # Mock / CPU simulation packs the full MPI world (one rank per mesh instance, e.g. 64
        # for the superpod MGD) onto a single CI host. There is no real device work, so allow
        # Open MPI to oversubscribe cores; otherwise it aborts with
        # "All nodes which are allocated for this job are already filled."
        return ["--allow-run-as-root", "--oversubscribe"]
    agent = MPI_PLM_RSH_AGENT if os.environ.get("SSH_AUTH_SOCK") else MPI_PLM_RSH_AGENT_NO_FORWARD
    return ["--mca", "plm_rsh_agent", agent]


def parse_hosts_csv(hosts_csv: str) -> list[str]:
    hosts = [h.strip() for h in hosts_csv.split(",") if h.strip()]
    if not hosts:
        raise ValueError("--hosts must contain at least one hostname")
    return hosts


def validate_mock_mapping(mock_mapping: Path, *, world_size: int) -> None:
    if not mock_mapping.is_file():
        raise FileNotFoundError(f"Mock cluster mapping not found: {mock_mapping}")
    rank_to_desc = load_canonical_mock_mapping(mock_mapping)
    if not rank_to_desc:
        raise ValueError(f"Mock cluster mapping has no ranks: {mock_mapping}")
    if max(rank_to_desc) >= world_size:
        raise ValueError(
            f"Mock mapping rank {max(rank_to_desc)} exceeds MGD world size {world_size} " f"({mock_mapping})"
        )


def validate_hardware_hosts(hosts_csv: str, world_size: int) -> list[str]:
    """Validate --hosts and return the host list in user order."""
    hosts = parse_hosts_csv(hosts_csv)
    if len(hosts) != len(set(hosts)):
        raise ValueError("--hosts contains duplicate hostnames; tt-run requires one entry per physical machine.")
    if len(hosts) != world_size:
        print(
            f"NOTE: {len(hosts)} host(s) in --hosts; MGD defines {world_size} MPI ranks "
            f"({world_size // len(hosts) if len(hosts) else '?'} rank(s) per host when evenly split).",
            flush=True,
        )
    return hosts


def format_hosts_csv(hosts: list[str]) -> str:
    return ",".join(hosts)


def generate_hardware_host_variations(host_list: list[str], *, num_variations: int, seed: int) -> list[list[str]]:
    """Random --hosts permutations (seeded); any host may appear first."""
    if not host_list:
        raise ValueError("Cannot generate hardware variations without hosts")
    rng = random.Random(seed)
    variations: list[list[str]] = []
    for _ in range(num_variations):
        variation_rng = random.Random(rng.randint(0, 2**31 - 1))
        shuffled = host_list.copy()
        variation_rng.shuffle(shuffled)
        variations.append(shuffled)
    return variations


def pick_hardware_variation_launch_host(
    host_list: list[str],
    *,
    variation_index: int,
) -> Optional[str]:
    """Pick the cluster node that runs tt-run for this hardware variation.

    Rotates deterministically through cluster hosts by variation index. When the script runs on one
    of the cluster nodes, only peer hosts are candidates so tt-run is launched remotely.
    """
    if not host_list:
        raise ValueError("host_list is empty")

    local = socket.gethostname()
    candidates = [host for host in host_list if not hostname_matches(local, host)]
    if not candidates:
        return None
    # +1 offset so variation 0 does not reuse the canonical launch host (typically host_list[0]).
    return candidates[(variation_index + 1) % len(candidates)]


def resolve_hardware_launch(
    *,
    hosts: list[str],
    explicit_launch_host: Optional[str],
    ssh_bootstrap: bool,
    hosts_csv: str,
) -> tuple[Optional[str], bool]:
    """Run tt-run from a cluster node when the current machine is not on the launch host."""
    if explicit_launch_host:
        launch = resolve_launch_host(explicit_launch_host, hosts_csv=hosts_csv)
        local = socket.gethostname()
        if hostname_matches(local, launch):
            return None, ssh_bootstrap
        return launch, ssh_bootstrap

    local = socket.gethostname()
    if any(hostname_matches(local, host) for host in hosts):
        return None, ssh_bootstrap

    launch = hosts[0]
    print(
        f"Current host {local!r} is not in --hosts; launching tt-run on cluster node {launch!r} via SSH.",
        flush=True,
    )
    return launch, True


def hostname_matches(local: str, host: str) -> bool:
    local_short = local.split(".")[0]
    host_short = host.split(".")[0]
    return local == host or local_short == host_short


def unique_hosts_in_order(hosts_csv: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for host in hosts_csv.split(","):
        host = host.strip()
        if not host or host in seen:
            continue
        seen.add(host)
        ordered.append(host)
    return ordered


def ssh_connection_target(host: str, ssh_user: Optional[str]) -> str:
    return f"{ssh_user}@{host}" if ssh_user else host


def remote_peer_ssh_shell(peer_host: str, ssh_user: Optional[str]) -> str:
    target = ssh_connection_target(peer_host, ssh_user)
    return (
        f"ssh -o BatchMode=yes -o ForwardAgent=yes -o ConnectTimeout=10 "
        f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "
        f"{shlex.quote(target)} hostname"
    )


def verify_mpi_peer_ssh_via_agent(
    *,
    launch_host: str,
    hosts_csv: str,
    ssh_user: Optional[str],
) -> None:
    """Verify launch host can reach MPI peers using the forwarded SSH agent."""
    unique_hosts = unique_hosts_in_order(hosts_csv)
    local = socket.gethostname()
    if any(hostname_matches(local, host) for host in unique_hosts):
        print(
            "  skip MPI peer SSH agent check (on-cluster launch; mpirun uses node-local keys)",
            flush=True,
        )
        return

    if not os.environ.get("SSH_AUTH_SOCK"):
        print(
            "  skip MPI peer SSH agent check (SSH_AUTH_SOCK unset; mpirun uses node-local keys)",
            flush=True,
        )
        return

    unique_hosts = unique_hosts_in_order(hosts_csv)
    peer_checks = [
        remote_peer_ssh_shell(host, ssh_user) for host in unique_hosts if not hostname_matches(launch_host, host)
    ]
    if not peer_checks:
        return

    launch_target = ssh_connection_target(launch_host, ssh_user)
    if hostname_matches(local, launch_host):
        print(f"  verifying MPI peer SSH via agent from {launch_target}...", flush=True)
        for check in peer_checks:
            result = subprocess.run(["bash", "-lc", check], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"MPI peer SSH via agent failed from {launch_target}.\n"
                    f"Ensure `ssh-add` loaded your keys and run this script from a login/jump host.\n"
                    f"cmd: {check}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
                )
        print("  MPI peer SSH via agent: OK", flush=True)
        return

    remote_mesh = " && ".join(["set -e"] + peer_checks)
    cmd = ssh_command(launch_target, "bash", "-lc", remote_mesh)
    print(f"  verifying MPI peer SSH via agent from {launch_target}...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    if result.returncode != 0:
        raise RuntimeError(
            f"MPI peer SSH via agent failed from {launch_target}.\n"
            f"Ensure `ssh-add` loaded your keys (SSH_AUTH_SOCK set) on this machine.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print("  MPI peer SSH via agent: OK", flush=True)


def resolve_launch_host(
    launch_host: Optional[str],
    *,
    hosts_csv: Optional[str],
) -> Optional[str]:
    if launch_host is None:
        return None
    if launch_host == "first":
        if hosts_csv:
            hosts = parse_hosts_csv(hosts_csv)
            return hosts[0]
        raise ValueError("--launch-host first requires --hosts")
    return launch_host


def ssh_bootstrap_cluster(
    *,
    hosts_csv: str,
    tt_metal_home: Path,
    fabric_unit_tests: Path,
    ssh_user: Optional[str],
    launch_host: Optional[str],
) -> None:
    """Verify passwordless SSH and a shared TT_METAL_HOME tree on each unique host."""
    unique_hosts = unique_hosts_in_order(hosts_csv)
    if not unique_hosts:
        raise ValueError("No hosts to bootstrap")

    executable_rel = fabric_unit_tests.relative_to(tt_metal_home).as_posix()
    python_rel = Path("python_env/bin/python3")
    ttrun_rel = Path("ttnn/ttnn/distributed/ttrun.py")

    print(f"\n=== SSH bootstrap ({len(unique_hosts)} unique host(s)) ===", flush=True)
    for host in unique_hosts:
        target = ssh_connection_target(host, ssh_user)
        remote_check = " && ".join(
            [
                f"test -d {shlex.quote(str(tt_metal_home))}",
                f"test -x {shlex.quote(str(tt_metal_home / executable_rel))}",
                f"test -x {shlex.quote(str(tt_metal_home / python_rel))}",
                f"test -f {shlex.quote(str(tt_metal_home / ttrun_rel))}",
                "hostname",
            ]
        )
        cmd = ssh_command(target, "bash", "-lc", remote_check)
        print(f"  checking {target}...", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH bootstrap failed on {target}\n"
                f"cmd: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        remote_hostname = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else host
        print(f"  OK: {target} ({remote_hostname})", flush=True)

    if launch_host is not None:
        verify_mpi_peer_ssh_via_agent(
            launch_host=launch_host,
            hosts_csv=hosts_csv,
            ssh_user=ssh_user,
        )


def _remote_env_shell(env: dict[str, str]) -> str:
    """Emit shell exports for remote tt-run (fixed keys first, then remaining TT_METAL_*)."""
    priority = [
        "TT_METAL_HOME",
        "TT_METAL_SLOW_DISPATCH_MODE",
        "TT_METAL_BLITZ_SUPERPOD_VARIATION",
        "TT_METAL_ASIC_MAPPING_GOLDEN_PATH",
        "TT_METAL_ASIC_MAPPING_GOLDEN_OPTIONAL",
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    ]
    lines: list[str] = []
    exported: set[str] = set()
    for key in priority:
        if key in env:
            lines.append(f"export {key}={shlex.quote(env[key])}")
            exported.add(key)
    for key, value in sorted(env.items()):
        if key.startswith("TT_METAL_") and key not in exported:
            lines.append(f"export {key}={shlex.quote(value)}")
    return "\n".join(lines)


def execute_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    launch_host: Optional[str] = None,
    ssh_user: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    if launch_host is not None:
        local = socket.gethostname()
        if hostname_matches(local, launch_host):
            launch_host = None

    if launch_host is None:
        return subprocess.run(cmd, cwd=cwd, env=env)

    tt_home = env.get("TT_METAL_HOME", str(cwd))
    remote_body = "\n".join(
        [
            _remote_env_shell(env),
            f"cd {shlex.quote(tt_home)}",
            "exec " + shlex.join(cmd),
        ]
    )
    target = ssh_connection_target(launch_host, ssh_user)
    ssh_cmd = ssh_command(target, "bash", "-lc", remote_body)
    print(f"  remote launch on {target} (ForwardAgent enabled)", flush=True)
    return subprocess.run(ssh_cmd, cwd=cwd, env=os.environ.copy())


def _runtime_env(
    build_dir: Path,
    *,
    variation_run: bool,
    golden_path: Optional[Path] = None,
    golden_optional: bool = False,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TT_METAL_HOME", str(REPO_ROOT))
    env.setdefault("TT_METAL_SLOW_DISPATCH_MODE", "1")
    if variation_run:
        env["TT_METAL_BLITZ_SUPERPOD_VARIATION"] = "1"
    else:
        env.pop("TT_METAL_BLITZ_SUPERPOD_VARIATION", None)

    env.pop("TT_METAL_ASIC_MAPPING_GOLDEN_PATH", None)
    env.pop("TT_METAL_ASIC_MAPPING_GOLDEN_OPTIONAL", None)
    if golden_path is not None:
        env["TT_METAL_ASIC_MAPPING_GOLDEN_PATH"] = str(golden_path)
    elif golden_optional:
        env["TT_METAL_ASIC_MAPPING_GOLDEN_OPTIONAL"] = "1"

    env["LD_LIBRARY_PATH"] = ":".join(
        filter(
            None,
            [
                str(build_dir / "tt_metal"),
                str(build_dir / "tt_metal/third_party/umd/lib"),
                str(build_dir / "tt_stl"),
                env.get("LD_LIBRARY_PATH", ""),
            ],
        )
    )
    return env


def run_superpod_control_plane(
    *,
    build_dir: Path,
    mgd: Path,
    run_label: str,
    variation_run: bool,
    mock_mapping: Optional[Path] = None,
    hosts: Optional[str] = None,
    tcp_interface: Optional[str] = None,
    mpi_args: Optional[list[str]] = None,
    golden_path: Optional[Path] = None,
    golden_optional: bool = False,
    launch_host: Optional[str] = None,
    ssh_user: Optional[str] = None,
    log: Optional[AutomapperTestLog] = None,
    log_extra: Optional[dict[str, object]] = None,
) -> None:
    launch_modes = sum(x is not None for x in (mock_mapping, hosts))
    if launch_modes != 1:
        raise ValueError("Exactly one of mock_mapping or hosts must be set")

    ttrun = REPO_ROOT / "ttnn/ttnn/distributed/ttrun.py"
    executable = build_dir / "test/tt_metal/tt_fabric/fabric_unit_tests"

    cmd = [_python_executable(), str(ttrun)]
    if tcp_interface:
        cmd.extend(["--tcp-interface", tcp_interface])

    if mock_mapping is not None:
        cmd.extend(["--mock-cluster-rank-binding", str(mock_mapping)])
    else:
        cmd.extend(["--hosts", hosts])

    cmd.extend(["--mesh-graph-descriptor", str(mgd), "--force-rediscovery"])

    merged_mpi_args = list(mpi_args or [])
    if merged_mpi_args:
        cmd.extend(["--mpi-args", shlex.join(merged_mpi_args)])

    cmd.extend([str(executable), f"--gtest_filter={GTEST_FILTER}"])

    mode = "mock" if mock_mapping is not None else "hardware"
    print(f"\n=== Blitz superpod mapping determinism tests ({mode}, {run_label}) ===", flush=True)
    print(" ".join(cmd), flush=True)

    if log is not None:
        log.log_run_start(
            run_label=run_label,
            mode=mode,
            mgd=mgd,
            launch_host=launch_host,
            ssh_user=ssh_user,
            command=cmd,
            mock_mapping=mock_mapping,
            hosts=hosts,
            extra=log_extra,
        )

    runtime_env = _runtime_env(
        build_dir,
        variation_run=variation_run,
        golden_path=golden_path if not variation_run else None,
        golden_optional=golden_optional if not variation_run else False,
    )

    remote_launch = is_remote_tt_run_launch(launch_host)
    result = execute_command(
        cmd,
        cwd=REPO_ROOT,
        env=runtime_env,
        launch_host=launch_host,
        ssh_user=ssh_user,
    )
    if log is not None:
        log.log_run_finished(
            run_label=run_label,
            exit_code=result.returncode,
            remote_launch=remote_launch,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Blitz superpod mapping determinism ({mode}, {run_label}) run failed (exit {result.returncode})"
        )


def sync_and_save_mapping_artifact(
    *,
    run_label: str,
    variations_dir: Path,
    world_size: int,
    hosts: Optional[list[str]] = None,
    launch_host: Optional[str] = None,
    ssh_user: Optional[str] = None,
    mapping_sync_wait_sec: float = MAPPING_SYNC_MAX_WAIT_SEC,
    log: Optional[AutomapperTestLog] = None,
    reference_path: Optional[Path] = None,
) -> Path:
    """Wait for/sync rank-1 mapping and copy into the run artifact directory."""
    if hosts is not None:
        ensure_generated_mapping_local(
            world_size=world_size,
            hosts=hosts,
            launch_host=launch_host,
            ssh_user=ssh_user,
            mapping_sync_wait_sec=mapping_sync_wait_sec,
        )
    generated = generated_mapping_path(world_size)
    if not generated.is_file():
        raise FileNotFoundError(f"Missing generated mapping after {run_label} run: {generated}")

    dest_dir = variations_dir / run_label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / mapping_artifact_filename(world_size)
    shutil.copy2(generated, dest)
    print(f"Saved {run_label} mapping artifact: {dest}", flush=True)

    if run_label == "canonical":
        CANONICAL_REFERENCE_MAPPING.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated, CANONICAL_REFERENCE_MAPPING)
        print(f"Captured canonical reference mapping: {CANONICAL_REFERENCE_MAPPING}", flush=True)

    if log is not None:
        log.log_mapping_artifact(
            run_label=run_label,
            generated_path=generated,
            artifact_path=dest,
            reference_path=reference_path,
        )

    return dest


def run_mapping_compare(
    *,
    run_label: str,
    mapping_artifact: Path,
    reference_mapping: Path,
    topology_invariant: bool = False,
    log: Optional[AutomapperTestLog] = None,
) -> None:
    """Compare a mapping to the reference.

    Exact compare (default) requires every field to match per ``fabric_node_id`` -- used for the
    canonical run and for hardware variations (which only shuffle --hosts order, not mesh_id
    labels). ``topology_invariant=True`` is used for mock variations, which permute MGD mesh_id
    labels: it keys by physical ``asic_id`` and allows a consistent bijective mesh_id relabel.
    """
    if topology_invariant:
        mismatches = compare_asic_mapping_topology_invariant(mapping_artifact, reference_mapping)
    else:
        mismatches = compare_asic_mapping_yaml_exact(mapping_artifact, reference_mapping)
    if log is not None:
        log.log_compare_result(
            run_label=run_label,
            mapping_artifact=mapping_artifact,
            reference_mapping=reference_mapping,
            mismatches=mismatches,
        )
    if mismatches:
        preview = "\n".join(mismatches[:20])
        extra = f"\n... and {len(mismatches) - 20} more" if len(mismatches) > 20 else ""
        raise RuntimeError(f"{run_label} mapping differs from reference ({reference_mapping}):\n{preview}{extra}")
    print(f"{run_label}: mapping matches reference ({reference_mapping})", flush=True)


def _load_or_generate_mock_manifest(
    *,
    variations_dir: Path,
    num_variations: int,
    seed: int,
    force_regenerate: bool,
    mock_mapping: Path,
    mgd: Path,
) -> dict:
    manifest_path = variations_dir / "manifest.json"
    expected_num_meshes = count_mesh_instances(mgd)
    if force_regenerate or not manifest_path.is_file():
        print(f"\n=== Generating {num_variations} mock variations (seed={seed}) ===", flush=True)
        return generate_mock_variations(
            output_dir=variations_dir,
            num_variations=num_variations,
            seed=seed,
            canonical_mock_mapping=mock_mapping,
            canonical_mgd=mgd,
        )

    manifest = json.loads(manifest_path.read_text())
    variations = manifest.get("variations", [])
    needs_mesh_id_perm = bool(variations) and not all("mesh_id_permutation" in v for v in variations)
    if (
        manifest.get("mode") != "mock"
        or manifest.get("num_variations") != num_variations
        or manifest.get("seed") != seed
        or manifest.get("num_meshes") != expected_num_meshes
        or needs_mesh_id_perm
    ):
        print("Regenerating mock variations (manifest stale or settings mismatch)", flush=True)
        return generate_mock_variations(
            output_dir=variations_dir,
            num_variations=num_variations,
            seed=seed,
            canonical_mock_mapping=mock_mapping,
            canonical_mgd=mgd,
        )
    return manifest


def run_automapper_tests(
    *,
    build_dir: Path,
    variations_dir: Path,
    num_variations: int,
    seed: int,
    force_regenerate: bool,
    hosts: Optional[str],
    mock_mapping: Optional[Path],
    mgd: Path,
    world_size: int,
    tcp_interface: Optional[str],
    mpi_args: Optional[list[str]],
    golden_path: Optional[Path],
    launch_host: Optional[str] = None,
    ssh_user: Optional[str] = None,
    ssh_bootstrap: bool = False,
    mapping_sync_wait_sec: float = MAPPING_SYNC_MAX_WAIT_SEC,
    log: Optional[AutomapperTestLog] = None,
) -> None:
    hardware_mode = hosts is not None
    use_golden_file = golden_path is not None
    host_list: Optional[list[str]] = None

    if use_golden_file and not golden_path.is_file():
        raise FileNotFoundError(f"Golden file not found: {golden_path}")

    if hardware_mode:
        host_list = validate_hardware_hosts(hosts, world_size)
        print(
            f"Hardware: {world_size} MPI ranks; {len(host_list)} host(s) in --hosts",
            flush=True,
        )
        if tcp_interface is None:
            print(
                "WARNING: --tcp-interface not set; tt-run uses default MPI TCP exclusions. "
                "For SP4 GLX clusters, try --tcp-interface cnx1.",
                flush=True,
            )
        if ssh_bootstrap:
            ssh_bootstrap_cluster(
                hosts_csv=hosts,
                tt_metal_home=REPO_ROOT,
                fabric_unit_tests=build_dir / "test/tt_metal/tt_fabric/fabric_unit_tests",
                ssh_user=ssh_user,
                launch_host=launch_host,
            )
        if launch_host:
            print(f"Launch host for tt-run: {launch_host}", flush=True)
    elif use_golden_file:
        print(f"Using golden mapping: {golden_path}", flush=True)
    else:
        print(
            "No --golden: canonical run captures reference; variations compare against first generated mapping.",
            flush=True,
        )

    run_superpod_control_plane(
        build_dir=build_dir,
        mgd=mgd,
        run_label="canonical",
        variation_run=False,
        mock_mapping=None if hardware_mode else mock_mapping,
        hosts=hosts,
        tcp_interface=tcp_interface,
        mpi_args=mpi_args,
        golden_path=golden_path,
        golden_optional=not use_golden_file and not hardware_mode,
        launch_host=launch_host if hardware_mode else None,
        ssh_user=ssh_user if hardware_mode else None,
        log=log,
    )
    print("Canonical run: PASSED", flush=True)

    canonical_artifact = sync_and_save_mapping_artifact(
        run_label="canonical",
        variations_dir=variations_dir,
        world_size=world_size,
        hosts=host_list if hardware_mode else None,
        launch_host=launch_host if hardware_mode else None,
        ssh_user=ssh_user if hardware_mode else None,
        mapping_sync_wait_sec=mapping_sync_wait_sec,
        log=log,
        reference_path=golden_path if use_golden_file else None,
    )

    if use_golden_file:
        reference_mapping = golden_path
        run_mapping_compare(
            run_label="canonical",
            mapping_artifact=canonical_artifact,
            reference_mapping=reference_mapping,
            log=log,
        )
    else:
        reference_mapping = canonical_artifact
        print(
            f"No --golden: using canonical mapping as reference for variations: {reference_mapping}",
            flush=True,
        )

    if num_variations <= 0:
        print("\nMapping determinism tests complete: canonical run passed.", flush=True)
        return

    if hardware_mode:
        assert host_list is not None
        shuffled_variations = generate_hardware_host_variations(host_list, num_variations=num_variations, seed=seed)
        for i, shuffled_hosts in enumerate(shuffled_variations):
            variation_hosts = format_hosts_csv(shuffled_hosts)
            variation_launch_host = pick_hardware_variation_launch_host(host_list, variation_index=i)
            print(
                f"\n=== Hardware variation {i}: --hosts={variation_hosts} "
                f"launch_host={variation_launch_host or socket.gethostname()} "
                f"(seed={seed}) ===",
                flush=True,
            )
            cleanup_generated_mapping_files(world_size)
            run_superpod_control_plane(
                build_dir=build_dir,
                mgd=mgd,
                run_label=f"variation_{i}",
                variation_run=True,
                hosts=variation_hosts,
                tcp_interface=tcp_interface,
                mpi_args=mpi_args,
                launch_host=variation_launch_host,
                ssh_user=ssh_user,
                log=log,
                log_extra={
                    "variation_index": i,
                    "seed": seed,
                    "variation_launch_host": variation_launch_host or socket.gethostname(),
                },
            )
            variation_artifact = sync_and_save_mapping_artifact(
                run_label=f"variation_{i}",
                variations_dir=variations_dir,
                world_size=world_size,
                hosts=shuffled_hosts,
                launch_host=variation_launch_host,
                ssh_user=ssh_user,
                mapping_sync_wait_sec=mapping_sync_wait_sec,
                log=log,
                reference_path=reference_mapping,
            )
            run_mapping_compare(
                run_label=f"variation_{i}",
                mapping_artifact=variation_artifact,
                reference_mapping=reference_mapping,
                log=log,
            )
            print(
                f"Hardware variation {i} (--hosts={variation_hosts}): PASSED",
                flush=True,
            )
    else:
        manifest = _load_or_generate_mock_manifest(
            variations_dir=variations_dir,
            num_variations=num_variations,
            seed=seed,
            force_regenerate=force_regenerate,
            mock_mapping=mock_mapping,
            mgd=mgd,
        )

        for variation in manifest["variations"]:
            index = variation["index"]
            variation_mock_mapping = REPO_ROOT / variation["mock_mapping"]
            variation_mgd = REPO_ROOT / variation["mgd"]
            cleanup_generated_mapping_files(world_size)

            log_extra: dict[str, object] = {"variation_index": index, "seed": seed}
            if "mesh_id_permutation" in variation:
                log_extra["mesh_id_permutation"] = variation["mesh_id_permutation"]

            run_superpod_control_plane(
                build_dir=build_dir,
                mgd=variation_mgd,
                run_label=f"variation_{index}",
                variation_run=True,
                mock_mapping=variation_mock_mapping,
                tcp_interface=tcp_interface,
                mpi_args=mpi_args,
                log=log,
                log_extra=log_extra,
            )
            variation_artifact = sync_and_save_mapping_artifact(
                run_label=f"variation_{index}",
                variations_dir=variations_dir,
                world_size=world_size,
                log=log,
                reference_path=reference_mapping,
            )
            run_mapping_compare(
                run_label=f"variation_{index}",
                mapping_artifact=variation_artifact,
                reference_mapping=reference_mapping,
                topology_invariant=True,
                log=log,
            )

    print(f"\nMapping determinism tests complete: canonical + {num_variations} variation(s) passed.", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    launch = parser.add_argument_group("launch (exactly one mode required)")
    mode = launch.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--hosts",
        type=str,
        default=None,
        help=(
            "Real hardware: comma-separated physical hostnames passed unchanged to tt-run "
            "(one entry per machine). Mutually exclusive with --mock-cluster-rank-binding."
        ),
    )
    mode.add_argument(
        "--mock-cluster-rank-binding",
        type=Path,
        metavar="PATH",
        help="Mock cluster mapping YAML. Mutually exclusive with --hosts.",
    )
    launch.add_argument(
        "--mesh-graph-descriptor",
        type=Path,
        required=True,
        help="Mesh graph descriptor textproto for this test.",
    )
    launch.add_argument(
        "--tcp-interface",
        type=str,
        default=None,
        help="Network interface for MPI TCP (real hardware; passed to tt-run)",
    )
    launch.add_argument(
        "--golden",
        type=Path,
        default=None,
        help=(
            "Golden ASIC mapping YAML. Every run (canonical and variations) must match this file exactly "
            "(hostname, tray_id, asic_location, mesh_id, chip_id, asic_id). "
            "If omitted, the canonical run artifact is the reference for variation compares."
        ),
    )
    launch.add_argument(
        "--ssh-bootstrap",
        action="store_true",
        help=(
            "Real hardware only: SSH to each unique host and verify TT_METAL_HOME, tt-run, and "
            "fabric_unit_tests exist. Also verify the launch host can SSH to peer hosts for MPI."
        ),
    )
    launch.add_argument(
        "--launch-host",
        type=str,
        default=None,
        help=(
            "Real hardware only: run tt-run on this cluster node via SSH (use 'first' for the first "
            "unique --hosts entry). If omitted and the current host is not in --hosts, the first "
            "cluster host is used automatically."
        ),
    )
    launch.add_argument(
        "--ssh-user",
        type=str,
        default=None,
        help="SSH username for --ssh-bootstrap and --launch-host (default: current user)",
    )

    variations = parser.add_argument_group("variations")
    variations.add_argument(
        "--variations-dir",
        type=Path,
        default=None,
        help="Output directory for generated variation artifacts (mode-specific default if omitted)",
    )
    variations.add_argument(
        "--num-variations",
        type=int,
        default=5,
        help="Mock: randomized mapping/MGD cases. Hardware: random --hosts order and rotated tt-run launch host.",
    )
    variations.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible variations")
    variations.add_argument(
        "--mapping-sync-wait-sec",
        type=float,
        default=MAPPING_SYNC_MAX_WAIT_SEC,
        help="Hardware only: seconds to wait for shared-fs mapping YAML before SSH fetch",
    )
    variations.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Always regenerate variation artifacts before running",
    )
    variations.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_RUN_LOG,
        help="Path to write structured test log (MGD, launch host, commands, mapping paths, mismatches)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log_path = args.log_file
    if not log_path.is_absolute():
        log_path = REPO_ROOT / log_path

    log = AutomapperTestLog(log_path)
    print(f"Writing test log to: {log_path}", flush=True)
    try:
        build_dir = resolve_build_dir(None)
        hosts = args.hosts
        hardware_mode = hosts is not None

        mgd = args.mesh_graph_descriptor
        if not mgd.is_absolute():
            mgd = REPO_ROOT / mgd
        if not mgd.is_file():
            raise FileNotFoundError(f"Mesh graph descriptor not found: {mgd}")

        world_size = count_mesh_instances(mgd)
        print(f"MGD defines {world_size} mesh instances (MPI world size)", flush=True)

        if args.num_variations < 0:
            raise ValueError("--num-variations must be >= 0")

        mock_mapping: Optional[Path] = None
        if not hardware_mode:
            mock_mapping = args.mock_cluster_rank_binding
            if mock_mapping is None:
                raise ValueError("Internal error: mock mode without --mock-cluster-rank-binding")
            if not mock_mapping.is_absolute():
                mock_mapping = REPO_ROOT / mock_mapping
            validate_mock_mapping(mock_mapping, world_size=world_size)

        variations_dir = args.variations_dir
        if variations_dir is None:
            variations_dir = DEFAULT_HARDWARE_VARIATIONS_DIR if hardware_mode else DEFAULT_MOCK_VARIATIONS_DIR
        elif not variations_dir.is_absolute():
            variations_dir = REPO_ROOT / variations_dir

        mpi_args = default_mpi_args(hardware_mode=hardware_mode)
        warn_if_ssh_agent_missing(hardware_mode=hardware_mode)

        golden_path = args.golden
        if golden_path is not None and not golden_path.is_absolute():
            golden_path = REPO_ROOT / golden_path

        if args.launch_host and not hardware_mode:
            raise ValueError("--launch-host requires --hosts")
        if args.ssh_bootstrap and not hardware_mode:
            raise ValueError("--ssh-bootstrap requires --hosts")

        launch_host = None
        ssh_bootstrap = args.ssh_bootstrap
        if hardware_mode:
            host_list = validate_hardware_hosts(hosts, world_size)
            launch_host, ssh_bootstrap = resolve_hardware_launch(
                hosts=host_list,
                explicit_launch_host=args.launch_host,
                ssh_bootstrap=ssh_bootstrap,
                hosts_csv=hosts,
            )

        log.log_session_config(
            hardware_mode=hardware_mode,
            mgd=mgd,
            world_size=world_size,
            mock_mapping=mock_mapping,
            hosts=hosts,
            golden_path=golden_path,
            variations_dir=variations_dir,
            num_variations=args.num_variations,
            seed=args.seed,
            launch_host=launch_host,
            ssh_user=args.ssh_user,
            tcp_interface=args.tcp_interface,
            build_dir=build_dir,
        )

        run_automapper_tests(
            build_dir=build_dir,
            variations_dir=variations_dir,
            num_variations=args.num_variations,
            seed=args.seed,
            force_regenerate=args.force_regenerate,
            hosts=hosts,
            mock_mapping=mock_mapping,
            mgd=mgd,
            world_size=world_size,
            tcp_interface=args.tcp_interface,
            mpi_args=mpi_args,
            golden_path=golden_path,
            launch_host=launch_host,
            ssh_user=args.ssh_user,
            ssh_bootstrap=ssh_bootstrap,
            mapping_sync_wait_sec=args.mapping_sync_wait_sec,
            log=log,
        )
        log.log_final_status("PASSED")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        log.log_final_status("FAILED", str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
