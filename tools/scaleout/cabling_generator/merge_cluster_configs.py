#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Recursively aggregate a tree of cluster configurations.

Walks <source> and, for every directory whose subdirectories contain cluster
descriptors, writes merged cabling/deployment/FSD descriptors to
<output>/<rel_path>/aggregated/.

Directory classification:
  - Leaf:      contains both cabling_descriptor.textproto AND
               deployment_descriptor.textproto directly.
  - Composite: contains at least one immediate subdirectory that is a Leaf or
               Composite. May also carry "glue" cabling files at its own level
               that wire its children together (filenames matching
               *inter*.textproto or *glue*.textproto, e.g.
               inter_superpod_cabling.textproto).
  - Ignored:   anything else, including pre-existing aggregated/ directories.

For every Composite, the script emits:

    <output>/<rel_path>/aggregated/
        cabling_descriptor.textproto
        deployment_descriptor.textproto
        factory_system_descriptor.textproto

The traversal is post-order: a Composite's aggregate is built from its
children's canonical descriptors (own files for Leaves, aggregated/ files for
nested Composites) plus any glue files at the Composite's own level.

Usage:
    ./merge_cluster_configs.py <source-dir> [--output <dir>] [options]

Defaults to in-place aggregation (output = source). Pass --output to mirror
the source tree structure to a different location.

Examples:
    # In-place: writes aggregated/ subdirs throughout the source tree.
    ./merge_cluster_configs.py /path/to/tt-cluster-configs/exabox

    # Mirrored output: writes aggregated/ subdirs under /tmp/out preserving
    # the source's relative paths. Source is not modified.
    ./merge_cluster_configs.py /path/to/exabox --output /tmp/out

    # Ad-hoc pairwise merge (BRINGUP workflow):
    #   mkdir -p merge_input/existing merge_input/new
    #   cp <old>/cabling_descriptor.textproto    merge_input/existing/
    #   cp <old>/deployment_descriptor.textproto merge_input/existing/
    #   cp <new>/cabling_descriptor.textproto    merge_input/new/
    #   cp <new>/deployment_descriptor.textproto merge_input/new/
    #   ./merge_cluster_configs.py merge_input
    # -> merge_input/aggregated/{cabling,deployment,factory_system}_descriptor.textproto
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

LEAF_CABLING = "cabling_descriptor.textproto"
LEAF_DEPLOYMENT = "deployment_descriptor.textproto"
LEAF_FSD = "factory_system_descriptor.textproto"
AGGREGATED_DIR = "aggregated"

# Glue cabling files at composite level: anything matching *inter*.textproto or
# *glue*.textproto, except the canonical descriptor names above.
_GLUE_RE = re.compile(r".*(?:inter|glue).*\.textproto$", re.IGNORECASE)
_RESERVED_TEXTPROTOS = {LEAF_CABLING, LEAF_DEPLOYMENT, LEAF_FSD}


@dataclass
class ClusterFiles:
    """Canonical descriptor paths for a Leaf or aggregated Composite."""

    cabling: Path
    deployment: Path
    fsd: Optional[Path] = None


class CablingGeneratorError(Exception):
    """Raised when the underlying run_cabling_generator binary fails."""


@dataclass
class FailureRecord:
    composite: Path
    error: str


def find_cabling_generator(repo_root: Path, build_dir: Optional[str]) -> Path:
    binary_subpath = Path("tools") / "scaleout" / "run_cabling_generator"
    candidates = [build_dir] if build_dir else ["build_Release", "build_Debug", "build"]
    for bd in candidates:
        candidate = repo_root / bd / binary_subpath
        if candidate.exists():
            return candidate
    print(
        f"Error: run_cabling_generator not found in: {', '.join(candidates)}",
        file=sys.stderr,
    )
    print("Build with: ./build_metal.sh --build-tests", file=sys.stderr)
    sys.exit(1)


def is_glue_textproto(name: str) -> bool:
    if name in _RESERVED_TEXTPROTOS:
        return False
    return bool(_GLUE_RE.match(name))


def immediate_subdirs(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.is_dir() and not p.is_symlink() and p.name != AGGREGATED_DIR)


def glue_files(d: Path) -> List[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and is_glue_textproto(p.name))


def concat_deployments(deployment_paths: List[Path], out_path: Path) -> None:
    with open(out_path, "w") as out:
        for i, p in enumerate(deployment_paths):
            with open(p, "r") as f:
                out.write(f.read())
            if i + 1 < len(deployment_paths):
                out.write("\n")


@contextmanager
def staging_dir(keep: bool, prefix: str) -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield path
    finally:
        if keep:
            print(f"  (kept staging dir: {path})")
        else:
            shutil.rmtree(path, ignore_errors=True)


def run_cabling_generator(
    cabling_gen: Path,
    repo_root: Path,
    cabling_dir: Path,
    deployment_path: Path,
    suffix: str,
    target_aggregated_dir: Path,
    verbose: bool,
) -> ClusterFiles:
    """Invoke run_cabling_generator and copy its outputs into target_aggregated_dir."""
    cmd = [
        str(cabling_gen),
        "--cabling",
        str(cabling_dir),
        "--deployment",
        str(deployment_path),
        "--output",
        suffix,
    ]
    if verbose:
        print(f"  $ {' '.join(cmd)}", flush=True)
    sys.stdout.flush()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    if result.returncode != 0:
        # Extract a short, human-readable error (last "Error: ..." line, falling back to stderr tail).
        tail = (result.stderr or result.stdout or "").strip().splitlines()
        short = next((ln for ln in reversed(tail) if ln.startswith("Error:")), "")
        if not short and tail:
            short = tail[-1]
        raise CablingGeneratorError(short or "run_cabling_generator failed")

    out_scaleout = repo_root / "out" / "scaleout"
    src_fsd = out_scaleout / f"factory_system_descriptor_{suffix}.textproto"
    src_cabling = out_scaleout / f"cabling_descriptor_{suffix}.textproto"
    src_deployment = out_scaleout / f"deployment_descriptor_{suffix}.textproto"
    src_csv = out_scaleout / f"cabling_guide_{suffix}.csv"

    for required in (src_fsd, src_cabling, src_deployment):
        if not required.exists():
            raise CablingGeneratorError(f"expected generator output missing: {required}")

    target_aggregated_dir.mkdir(parents=True, exist_ok=True)
    final = ClusterFiles(
        cabling=target_aggregated_dir / LEAF_CABLING,
        deployment=target_aggregated_dir / LEAF_DEPLOYMENT,
        fsd=target_aggregated_dir / LEAF_FSD,
    )
    shutil.copy(src_cabling, final.cabling)
    shutil.copy(src_deployment, final.deployment)
    shutil.copy(src_fsd, final.fsd)

    for transient in (src_fsd, src_cabling, src_deployment, src_csv):
        try:
            transient.unlink()
        except FileNotFoundError:
            pass

    return final


def _suffix_for(rel: Path, fallback_name: str) -> str:
    slug = str(rel) if str(rel) != "." else fallback_name
    return "agg_" + re.sub(r"[^A-Za-z0-9_.-]", "_", slug)


def aggregate_tree(
    source_dir: Path,
    output_dir: Path,
    source_root: Path,
    repo_root: Path,
    cabling_gen: Path,
    args: argparse.Namespace,
    failures: List[FailureRecord],
) -> Optional[ClusterFiles]:
    """Post-order DFS. Returns canonical files for source_dir, or None if ignored/failed."""
    if source_dir.name == AGGREGATED_DIR:
        return None
    if not source_dir.is_dir() or source_dir.is_symlink():
        return None

    rel = source_dir.relative_to(source_root)
    target_dir = output_dir / rel

    own_cabling = source_dir / LEAF_CABLING
    own_deployment = source_dir / LEAF_DEPLOYMENT
    own_fsd = source_dir / LEAF_FSD

    if own_cabling.is_file() and own_deployment.is_file():
        # Leaf: own descriptors are canonical. Children (if any) are ignored.
        if args.verbose:
            print(f"leaf:      {source_dir}", flush=True)
        return ClusterFiles(
            cabling=own_cabling,
            deployment=own_deployment,
            fsd=own_fsd if own_fsd.is_file() else None,
        )

    children: List[Tuple[str, ClusterFiles]] = []
    failed_children = 0
    for sub in immediate_subdirs(source_dir):
        child = aggregate_tree(sub, output_dir, source_root, repo_root, cabling_gen, args, failures)
        if child is not None:
            children.append((sub.name, child))
        elif classify_is_composite_candidate(sub):
            failed_children += 1

    if not children:
        return None

    glue = glue_files(source_dir)
    target_aggregated_dir = target_dir / AGGREGATED_DIR

    rel_str = str(rel) if str(rel) != "." else source_dir.name
    print(
        f"composite: {rel_str} -> " f"{target_aggregated_dir}  ({len(children)} children, {len(glue)} glue files)",
        flush=True,
    )
    if args.verbose:
        for name, _ in children:
            print(f"    + child: {name}", flush=True)
        for g in glue:
            print(f"    + glue:  {g.name}", flush=True)
    if failed_children:
        print(
            f"  WARNING: {failed_children} child composite(s) failed; aggregating remaining {len(children)}.",
            flush=True,
        )

    if args.dry_run:
        return ClusterFiles(
            cabling=target_aggregated_dir / LEAF_CABLING,
            deployment=target_aggregated_dir / LEAF_DEPLOYMENT,
            fsd=target_aggregated_dir / LEAF_FSD,
        )

    try:
        with staging_dir(keep=args.keep_temp, prefix=f"merge_{source_dir.name}_") as staging:
            staging_cabling = staging / "cabling"
            staging_cabling.mkdir(parents=True, exist_ok=True)

            used_names: set = set()
            for name, cf in children:
                # Disambiguate by child directory name to keep merged cabling filenames unique.
                base = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
                dest_name = f"{base}__cabling.textproto"
                n = 1
                while dest_name in used_names:
                    dest_name = f"{base}__cabling_{n}.textproto"
                    n += 1
                used_names.add(dest_name)
                shutil.copy(cf.cabling, staging_cabling / dest_name)

            for g in glue:
                dest_name = g.name
                n = 1
                while dest_name in used_names:
                    stem, ext = g.stem, g.suffix
                    dest_name = f"{stem}_{n}{ext}"
                    n += 1
                used_names.add(dest_name)
                shutil.copy(g, staging_cabling / dest_name)

            staged_deployment = staging / "deployment.textproto"
            concat_deployments([cf.deployment for _, cf in children], staged_deployment)

            return run_cabling_generator(
                cabling_gen=cabling_gen,
                repo_root=repo_root,
                cabling_dir=staging_cabling,
                deployment_path=staged_deployment,
                suffix=_suffix_for(rel, source_dir.name),
                target_aggregated_dir=target_aggregated_dir,
                verbose=args.verbose,
            )
    except CablingGeneratorError as e:
        print(f"  FAILED: {source_dir}\n    {e}", file=sys.stderr, flush=True)
        failures.append(FailureRecord(composite=source_dir, error=str(e)))
        return None


def classify_is_composite_candidate(d: Path) -> bool:
    """Cheap check: would this directory have produced a result under normal walking?"""
    if not d.is_dir() or d.is_symlink() or d.name == AGGREGATED_DIR:
        return False
    if (d / LEAF_CABLING).is_file() and (d / LEAF_DEPLOYMENT).is_file():
        return True  # leaf
    for sub in d.iterdir():
        if sub.is_dir() and not sub.is_symlink() and sub.name != AGGREGATED_DIR:
            return True  # potential composite
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursively aggregate a tree of cluster configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source", help="Source tree root (walked recursively)")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output tree root. Defaults to source (in-place aggregation).",
    )
    parser.add_argument(
        "--build-dir",
        default=None,
        help="Build directory containing run_cabling_generator (auto-detected if omitted).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without running the generator or writing files.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep staged input directories after each merge (for debugging).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"Error: source is not a directory: {source}", file=sys.stderr)
        sys.exit(1)
    output = Path(args.output).resolve() if args.output else source

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    cabling_gen = find_cabling_generator(repo_root, args.build_dir)

    failures: List[FailureRecord] = []
    result = aggregate_tree(
        source_dir=source,
        output_dir=output,
        source_root=source,
        repo_root=repo_root,
        cabling_gen=cabling_gen,
        args=args,
        failures=failures,
    )

    if result is None and not failures:
        print(f"Nothing classifiable under {source}", file=sys.stderr)
        sys.exit(1)

    if failures:
        print("", file=sys.stderr)
        print(f"=== {len(failures)} composite(s) failed to aggregate ===", file=sys.stderr)
        for fr in failures:
            print(f"  - {fr.composite}\n      {fr.error}", file=sys.stderr)
        if args.dry_run:
            sys.exit(1)
        # Non-fatal: partial outputs may still be useful.
        print(f"Partial aggregation completed. Outputs under: {output}", file=sys.stderr)
        sys.exit(2)

    if args.dry_run:
        print("(dry run; no files written)")
    else:
        print(f"Done. Aggregated outputs under: {output}")


if __name__ == "__main__":
    main()
