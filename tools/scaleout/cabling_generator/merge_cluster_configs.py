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
import os
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


def _is_safe_name(name: str) -> bool:
    # Reject directory traversal, absolute paths, and shell-meaningful characters in
    # filenames discovered on disk. Real cluster config filenames are alnum + - _ .
    if not name or name in (".", "..") or "/" in name or "\\" in name or "\x00" in name:
        return False
    return True


def _ensure_within(root: Path, candidate: Path) -> Path:
    """Resolve candidate and assert it is contained in root. Returns the resolved path."""
    resolved_root = root.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path escapes expected root: {candidate} not under {root}") from exc
    return resolved


def _safe_copy(src: Path, dst: Path, dst_root: Path) -> None:
    """Copy src -> dst after verifying dst stays under dst_root and src is a regular file (no symlinks).

    Both paths are validated immediately before the file system operation:
      - src must be a regular file (not a symlink, not a device, not a directory).
      - dst is asserted to live under dst_root using the OWASP-recommended
        os.path.abspath + startswith pattern (no traversal possible).
    The validated paths are then passed to shutil.copyfile, which (unlike shutil.copy)
    does not follow symlinks at the destination.
    """
    if src.is_symlink() or not src.is_file():
        raise ValueError(f"refusing to copy non-regular or symlinked source: {src}")
    base_directory = os.path.abspath(str(dst_root))
    abs_dst = os.path.abspath(str(dst))
    if not (abs_dst == base_directory or abs_dst.startswith(base_directory + os.sep)):
        raise ValueError(f"refusing path outside {base_directory}: {abs_dst}")
    abs_src = os.path.abspath(str(src))
    shutil.copyfile(abs_src, abs_dst)


def immediate_subdirs(d: Path) -> List[Path]:
    # Source-tree directories: exclude symlinks (defense against malicious PRs adding
    # symlinks that escape the tree) and pre-existing aggregated/ outputs.
    return sorted(
        p
        for p in d.iterdir()
        if p.is_dir() and not p.is_symlink() and p.name != AGGREGATED_DIR and _is_safe_name(p.name)
    )


def glue_files(d: Path) -> List[Path]:
    # Reject symlinks here too so a malicious glue symlink can't redirect into /etc/...
    return sorted(
        p
        for p in d.iterdir()
        if p.is_file() and not p.is_symlink() and _is_safe_name(p.name) and is_glue_textproto(p.name)
    )


def concat_deployments(deployment_paths: List[Path], out_path: Path, out_root: Path) -> None:
    """Concatenate deployment textprotos into out_path.

    out_path is asserted to live under out_root (the staging tempdir we created) using
    the OWASP-recommended absolute-path safelist check, applied inline at the I/O sink.
    Every input path is required to be a regular file (rejecting symlinks). All
    validation completes before any file is written.
    """
    base_directory = os.path.abspath(str(out_root))
    abs_out = os.path.abspath(str(out_path))
    if not (abs_out == base_directory or abs_out.startswith(base_directory + os.sep)):
        raise ValueError(f"refusing path outside {base_directory}: {abs_out}")

    chunks: List[str] = []
    for p in deployment_paths:
        if p.is_symlink() or not p.is_file():
            raise ValueError(f"refusing to read non-regular or symlinked deployment: {p}")
        abs_p = os.path.abspath(str(p))
        chunks.append(Path(abs_p).read_text())
    Path(abs_out).write_text("\n".join(chunks))


@contextmanager
def staging_dir(keep: bool, prefix: str) -> Iterator[Path]:
    # Path is created by tempfile.mkdtemp under the system temp root (trusted).
    # We resolve it once so subsequent _ensure_within checks operate on the canonical
    # form and won't be confused by symlinked temp dirs (e.g. /tmp -> /private/tmp).
    path = Path(tempfile.mkdtemp(prefix=prefix)).resolve()
    try:
        yield path
    finally:
        if keep:
            print(f"  (kept staging dir: {path})")
        else:
            # rmtree on a tempfile.mkdtemp path we just created; ignore_errors avoids
            # races with files vanishing during shutdown.
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
    # All argv values are trusted (binary path resolved from build dir, paths constructed
    # by us, suffix sanitized in _suffix_for). subprocess.run is invoked with shell=False
    # and an argv list, so there is no shell-injection surface.
    if not cabling_gen.is_file() or cabling_gen.is_symlink():
        raise CablingGeneratorError(f"run_cabling_generator binary missing or symlinked: {cabling_gen}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", suffix):
        raise CablingGeneratorError(f"refusing unsafe suffix: {suffix!r}")
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
    result = subprocess.run(  # noqa: S603 - argv list, shell=False, all inputs validated above
        cmd,
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        shell=False,
    )
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

    # If an in-place run encounters a pre-existing aggregated/ symlink, refuse to follow
    # it: _safe_copy's containment check would otherwise resolve against the symlink
    # target and could write outside the intended output tree.
    if target_aggregated_dir.is_symlink():
        raise CablingGeneratorError(f"refusing to write to symlinked aggregated directory: {target_aggregated_dir}")
    if target_aggregated_dir.exists() and not target_aggregated_dir.is_dir():
        raise CablingGeneratorError(f"refusing to write to non-directory aggregated path: {target_aggregated_dir}")
    target_aggregated_dir.mkdir(parents=True, exist_ok=True)
    final = ClusterFiles(
        cabling=target_aggregated_dir / LEAF_CABLING,
        deployment=target_aggregated_dir / LEAF_DEPLOYMENT,
        fsd=target_aggregated_dir / LEAF_FSD,
    )
    _safe_copy(src_cabling, final.cabling, target_aggregated_dir)
    _safe_copy(src_deployment, final.deployment, target_aggregated_dir)
    _safe_copy(src_fsd, final.fsd, target_aggregated_dir)

    for transient in (src_fsd, src_cabling, src_deployment, src_csv):
        try:
            transient.unlink()
        except FileNotFoundError:
            # Some generator artifacts (e.g. CSV) may be absent depending on flags;
            # cleanup is best-effort and a missing file is not an error.
            continue

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

    # If any child composite failed, do NOT produce a parent aggregate: it would use the
    # same canonical descriptor names (cabling_descriptor.textproto etc.) as a complete
    # aggregate and could easily be consumed downstream without anyone noticing it's
    # missing one of its children. Surface as a parent failure instead.
    if failed_children:
        msg = (
            f"skipping aggregation: {failed_children} child composite(s) failed; "
            f"fix the children first to produce a complete parent aggregate."
        )
        print(f"  SKIPPED: {source_dir}\n    {msg}", file=sys.stderr, flush=True)
        failures.append(FailureRecord(composite=source_dir, error=msg))
        return None

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
                _safe_copy(cf.cabling, staging_cabling / dest_name, staging_cabling)

            for g in glue:
                # g.name is the basename (Path.name strips any directory component).
                # Re-sanitize for defense in depth and to satisfy SAST.
                base = re.sub(r"[^A-Za-z0-9_.-]", "_", g.name)
                # Force glue files to sort lexicographically AFTER all child cabling
                # descriptors. run_cabling_generator constructs the base topology from
                # the first .textproto it finds (alphabetical order); a glue-only file
                # is not a valid base, so we must guarantee it never sorts first.
                stem, ext = Path(base).stem, Path(base).suffix
                dest_name = f"zz_glue__{stem}{ext}"
                n = 1
                while dest_name in used_names:
                    dest_name = f"zz_glue__{stem}_{n}{ext}"
                    n += 1
                used_names.add(dest_name)
                _safe_copy(g, staging_cabling / dest_name, staging_cabling)

            staged_deployment = staging / "deployment.textproto"
            concat_deployments([cf.deployment for _, cf in children], staged_deployment, staging)

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
    """Would this directory have produced a result under normal walking?

    Returns True only if d is itself a Leaf or has a classifiable descendant. This
    avoids reporting unrelated subdirectories (e.g. docs/, .git/) as failed composites
    in mixed-content trees.
    """
    if not d.is_dir() or d.is_symlink() or d.name == AGGREGATED_DIR:
        return False
    if (d / LEAF_CABLING).is_file() and (d / LEAF_DEPLOYMENT).is_file():
        return True
    for sub in d.iterdir():
        if not sub.is_dir() or sub.is_symlink() or sub.name == AGGREGATED_DIR:
            continue
        if classify_is_composite_candidate(sub):
            return True
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

    # Reject symlinked sources/outputs BEFORE resolving the path; resolve() follows
    # symlinks and would otherwise mask them.
    raw_source = Path(args.source)
    if raw_source.is_symlink():
        print(f"Error: source must not be a symlink: {raw_source}", file=sys.stderr)
        sys.exit(1)
    if not raw_source.exists():
        print(f"Error: source does not exist: {raw_source}", file=sys.stderr)
        sys.exit(1)
    source = raw_source.resolve(strict=False)
    if not source.is_dir():
        print(f"Error: source must be a directory: {source}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        raw_output = Path(args.output)
        if raw_output.exists() and raw_output.is_symlink():
            print(f"Error: output must not be a symlink: {raw_output}", file=sys.stderr)
            sys.exit(1)
        output = raw_output.resolve(strict=False)
    else:
        output = source

    # In dry-run we never write outputs and never invoke the C++ binary, so neither the
    # output tree nor the build product is required to exist.
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent.parent
    if args.dry_run:
        cabling_gen = Path("/nonexistent/cabling_generator_dry_run_placeholder")
    else:
        output.mkdir(parents=True, exist_ok=True)
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
