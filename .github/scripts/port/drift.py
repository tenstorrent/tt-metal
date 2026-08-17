#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Say what moved in the generator between the commit a port was written against and today's.

A port is a transliteration of tt-dm-codegen at one commit. That branch keeps moving, so every
port is drifting away from its source from the moment it is written, and the question this answers
is the one update mode exists to ask: *given this drift, is there work an agent can do?*

The answer is not always yes, and the expensive failure is assuming it is. Merged `untilize` came
from `7f2930ff`, where `writer_untilize_interleaved.cpp` took three compile-time arguments. A later
generator gave the same template a shard-split contract of two more. A port carrying the old
argument list against the new template compiles, routes, and writes uncorrelated data -- 76 of 112
cases, read as a kernel bug for a day. Nothing about that drift announced itself; it was a
template's content changing under a port that still looked right.

So drift is classified by *who can act on it*, which is the only distinction that changes what the
harness should do next:

  blocking   the harness cannot describe the op at the target commit at all -- no builder resolves,
             no shared template directory exists, or the kernel closure comes back empty. Starting
             an agent here spends a budget failing, because there is nothing coherent to work
             against and no edit to a port changes that.
  agent      the port is out of date and re-transliterating it is exactly the agent's job. Builder
             logic, kernel templates, the attributes struct.
  rescope    the graded case set moved. The port may be perfect and still measure differently, so
             a case count from before this drift cannot be compared against one from after it.

What is compared is the generator itself, not a description of it. An earlier version of this file
diffed the two commits' `agentic_port/manifests/<op>.yaml`, which meant it could only report drift
that somebody had remembered to write down -- inheriting exactly the staleness problem it exists to
catch. `tilize.yaml` proves the point: it omits two kernels its own builder selects, so a commit that
changed either one would have been classified as nothing to do. The kernel set here comes from
`discover.py`, the same walk `scaffold.py` vendors from, so a drift report and a vendoring pass cannot
disagree about what the port consists of.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BLOCKING, AGENT, RESCOPE, NOISE = "blocking", "agent", "rescope", "noise"

REFUSE, UPDATE, CLEAN = "refuse", "update", "clean"


def git(root: Path, *args: str) -> tuple[int, str]:
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return done.returncode, done.stdout


class Source:
    """One side of the comparison: a way to read the generator as it stood at one commit.

    Two implementations because the two callers genuinely differ. CI has the generator checked out
    twice, at the pin and at the target, and cannot ask one checkout about the other's commit -- the
    clones are shallow and single-branch, and authenticating a fetch inside the agent job would put
    a token in a `run:` step, which this repo's workflow linter refuses. A person investigating
    drift has the opposite situation: one full clone and two commits they want compared.
    """

    def read(self, path: str) -> str | None:
        raise NotImplementedError

    def tree(self) -> Path:
        """A real directory holding this side, because the kernel walk lists directories.

        Reading named paths is not enough to discover a kernel set: the walk has to enumerate a
        template directory and follow includes it has not been told about.
        """
        raise NotImplementedError

    @property
    def label(self) -> str:
        raise NotImplementedError


class GitSource(Source):
    """A commit in a checkout that has it, read without disturbing the working tree."""

    def __init__(self, root: Path, ref: str):
        self.root, self.ref = root, ref
        self._tree: Path | None = None
        # Up front rather than per file, because `git show` cannot be trusted to distinguish a
        # missing path from a missing commit: a sha the clone never fetched reports `path 'x' exists
        # on disk, but not in '<sha>'`, which is the *missing file* message. Inferring from that
        # would read every template as absent and conclude the port is entirely stale, with a
        # confident report and nothing pointing at the clone as the real problem.
        code, out = git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        if code != 0:
            raise SystemExit(
                f"{ref} is not a commit in {root}. Both commits have to be present before they "
                f"can be compared -- a shallow or single-branch clone will not have both."
            )
        self.sha = out.strip()

    def read(self, path: str) -> str | None:
        code, out = git(self.root, "show", f"{self.ref}:{path}")
        return out if code == 0 else None

    def tree(self) -> Path:
        """The commit extracted into a temporary directory, once.

        `git archive` rather than a second clone or a worktree: the objects are already here -- the
        constructor proved it -- and neither touching the caller's working tree nor leaving a worktree
        registered behind is acceptable in a tool someone runs on their own clone.
        """
        if self._tree is None:
            out = Path(tempfile.mkdtemp(prefix=f"drift-{self.sha[:8]}-"))
            done = subprocess.run(
                f"git -C {self.root} archive {self.sha} | tar -x -C {out}",
                shell=True,
                capture_output=True,
                text=True,
            )
            if done.returncode != 0:
                raise SystemExit(f"could not extract {self.sha} from {self.root}: {done.stderr.strip()}")
            self._tree = out
        return self._tree

    @property
    def label(self) -> str:
        return self.sha


class TreeSource(Source):
    """A checkout, read as it sits on disk, labelled by whatever commit it is at."""

    def __init__(self, root: Path):
        self.root = root
        if not root.is_dir():
            raise SystemExit(f"{root} is not a directory, so there is nothing to compare")
        code, out = git(root, "rev-parse", "HEAD")
        self.sha = out.strip() if code == 0 else ""

    def read(self, path: str) -> str | None:
        target = self.root / path
        try:
            return target.read_text()
        except (OSError, UnicodeDecodeError):
            return None

    def tree(self) -> Path:
        return self.root

    @property
    def label(self) -> str:
        return self.sha or str(self.root)


def describe(source: Source, op: str) -> dict:
    """What the port consists of at this side of the comparison, resolved from the tree.

    Delegates to `discover.py` rather than reimplementing the walk. That matters more than the code it
    saves: if this file had its own idea of which kernels belong to an op, a drift report could name a
    template that `scaffold.py` never vendors, or stay silent about one it does.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import discover

    root = source.tree()
    try:
        builder = discover.resolve_builder(root, op)
        kernels, unresolved = discover.kernel_closure(root, op, builder)
    except SystemExit as exc:
        return {"error": str(exc)}
    if not kernels:
        return {"error": f"no kernels resolve for {op} at {source.label}"}
    return {
        "builder": builder,
        # Keyed by basename because that is the identity a vendored copy has: the port holds
        # `codegen/kernels/<basename>`, so a template that moved between directories is the same file
        # to the port and a different path here. Keeping both lets a move be reported as a move.
        "kernels": {path.rsplit("/", 1)[-1]: path for path in kernels},
        "unresolved": unresolved,
    }


def classify_generator(op: str, before: Source, after: Source, old: dict, new: dict) -> dict[str, list[str]]:
    """Compare the two sides and bucket every difference by who can act on it."""
    found: dict[str, list[str]] = {BLOCKING: [], AGENT: [], RESCOPE: [], NOISE: []}

    if new.get("error"):
        # The target is the side that has to be describable; a run is about to work against it.
        found[BLOCKING].append(f"the target commit cannot be described: {new['error']}")
        return found
    if old.get("error"):
        found[BLOCKING].append(f"the pinned commit cannot be described: {old['error']}")
        return found

    if new["builder"] != old["builder"]:
        found[AGENT].append(
            f"the builder moved from {old['builder']} to {new['builder']}, so the port was "
            f"transliterated from a file that is no longer the source"
        )
    elif before.read(new["builder"]) != after.read(new["builder"]):
        found[AGENT].append(f"{new['builder']}: builder logic changed -- the port's program factory came from this")

    old_kernels, new_kernels = old["kernels"], new["kernels"]
    for name in sorted(set(new_kernels) - set(old_kernels)):
        found[AGENT].append(f"{new_kernels[name]}: new template, needs vendoring beside the port")
    for name in sorted(set(old_kernels) - set(new_kernels)):
        found[AGENT].append(f"{old_kernels[name]}: no longer reachable, the vendored copy is dead")
    for name in sorted(set(old_kernels) & set(new_kernels)):
        was, now = old_kernels[name], new_kernels[name]
        if was != now:
            # `f895be71b` moving the shared templates is this case. The vendored copy still compiles,
            # so nothing announces it; only the path it came from changed.
            found[AGENT].append(f"{name}: moved from {was} to {now} -- re-vendor it from the new location")
        elif before.read(was) != after.read(now):
            found[AGENT].append(
                f"{now}: template contents changed -- re-vendor and re-check its argument contract"
            )

    for name in sorted(set(new.get("unresolved") or []) - set(old.get("unresolved") or [])):
        found[BLOCKING].append(
            f"{name} is referenced at the target commit but is in no template directory, so the "
            f"kernel set there is incomplete"
        )

    # The sweep is conventional, so it is checked at the conventional path rather than being read from
    # a field. A sweep that moved shows up as both sides reading None, which is not a difference.
    sweep = f"common/sweeps/codegen_{op}.py"
    old_sweep, new_sweep = before.read(sweep), after.read(sweep)
    if old_sweep != new_sweep and not (old_sweep is None and new_sweep is None):
        found[RESCOPE].append(f"{sweep}: the sweep grid moved, so the case set is not the old case set")

    return found


def merge(*parts: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {BLOCKING: [], AGENT: [], RESCOPE: [], NOISE: []}
    for part in parts:
        for bucket, items in part.items():
            out[bucket].extend(items)
    return out


def verdict_for(found: dict[str, list[str]]) -> str:
    if found[BLOCKING]:
        return REFUSE
    if found[AGENT] or found[RESCOPE]:
        return UPDATE
    return CLEAN


HEADINGS = {
    BLOCKING: (
        "## The harness cannot run against this generator yet",
        "These are fields this repo's own Python reads. An agent cannot fix any of them from a "
        "port branch, so a run started against this drift would spend its budget failing. Migrate "
        "the harness, then re-run.",
    ),
    AGENT: (
        "## The port is out of date against the generator, and this is the work",
        "Re-read each of these at the target commit and bring the port back in line. Where a "
        "kernel template's contents moved, the argument contract is the thing to check first: a "
        "port carrying a stale compile-time argument list still compiles and still routes, and "
        "then writes uncorrelated data.",
    ),
    RESCOPE: (
        "## The set of cases being graded has moved",
        "The port is not implicated, but the grading is. A failing count from before this drift "
        "cannot be compared against one from after it, so treat the next measurement as a new "
        "baseline rather than as progress or regression.",
    ),
}


def render(op: str, before: str, after: str, found: dict[str, list[str]], verdict: str) -> str:
    def short(label: str) -> str:
        # Only a sha abbreviates. A checkout with no commit is labelled by its path, and cutting that
        # to twelve characters leaves `/private/tmp`, which names neither side.
        return label[:12] if len(label) == 40 and all(c in "0123456789abcdef" for c in label) else label

    lines = [
        f"# What moved in the generator under {op}\n",
        f"Comparing `{short(before)}`, which this port was written against, to `{short(after)}`.\n",
    ]
    if verdict == CLEAN:
        lines.append(
            "Nothing this port depends on has changed. The builder, every reachable template and the "
            "sweep grid are byte-identical at both commits, so there is no generator-drift work to do "
            "here.\n"
        )
    for bucket in (BLOCKING, AGENT, RESCOPE):
        if not found[bucket]:
            continue
        heading, preamble = HEADINGS[bucket]
        lines.append(f"{heading}\n\n{preamble}\n")
        lines.extend(f"- {item}" for item in found[bucket])
        lines.append("")
    return "\n".join(lines)


def sources(args) -> tuple[Source, Source]:
    """Whichever pair of sides the caller described, with the two forms not mixable.

    CI passes two checkouts because that is what it has; a person passes one clone and two commits
    because that is what they have. Refusing the half-specified combinations is worth the four
    lines: `--from-tree` with `--to` would silently compare a checkout against a commit resolved in
    a different repository, and the report would look entirely normal.
    """
    by_tree = bool(args.from_tree or args.to_tree)
    by_ref = bool(args.old_ref or args.new_ref)
    if by_tree and by_ref:
        raise SystemExit("compare two checkouts or two commits, not one of each")
    if by_tree:
        if not (args.from_tree and args.to_tree):
            raise SystemExit("--from-tree and --to-tree go together")
        return TreeSource(Path(args.from_tree).resolve()), TreeSource(Path(args.to_tree).resolve())
    if not (args.old_ref and args.new_ref and args.codegen_root):
        raise SystemExit("--from and --to need --codegen-root, or use --from-tree and --to-tree")
    root = Path(args.codegen_root).resolve()
    return GitSource(root, args.old_ref), GitSource(root, args.new_ref)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", required=True)
    ap.add_argument("--codegen-root", default=None, help="a tt-dm-codegen clone holding both commits")
    ap.add_argument("--from", dest="old_ref", default=None, help="the commit the port was written against")
    ap.add_argument("--to", dest="new_ref", default=None, help="the commit to bring it up to")
    ap.add_argument("--from-tree", default=None, help="a checkout at the pinned commit")
    ap.add_argument("--to-tree", default=None, help="a checkout at the target commit")
    ap.add_argument("--out", default=None, help="write JSON here as well as the report to stdout")
    ap.add_argument(
        "--verdict-only",
        action="store_true",
        help="print just refuse/update/clean, for a shell step to branch on",
    )
    args = ap.parse_args()

    before, after = sources(args)
    if before.label == after.label:
        # Not an error: a port already at the target is the desired state, and an update run asked
        # to move it nowhere should say so plainly rather than look like a failure.
        found, verdict = merge(), CLEAN
    else:
        old = describe(before, args.op)
        new = describe(after, args.op)
        found = merge(classify_generator(args.op, before, after, old, new))
        verdict = verdict_for(found)

    if args.verdict_only:
        print(verdict)
        return 0

    report = render(args.op, before.label, after.label, found, verdict)
    print(report)
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "op": args.op,
                    "from": before.label,
                    "to": after.label,
                    "verdict": verdict,
                    "blocking": found[BLOCKING],
                    "agent": found[AGENT],
                    "rescope": found[RESCOPE],
                    "report": report,
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
