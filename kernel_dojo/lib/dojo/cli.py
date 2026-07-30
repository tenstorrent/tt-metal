# SPDX-License-Identifier: Apache-2.0
"""The `dojo` command line."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from . import perf as perf_mod
from .runner import (
    Ctx,
    DeviceSession,
    kernel_dir_for,
    list_exercise_dirs,
    load_exercise,
    prepare,
    require_tt_metal_home,
    resolve_exercise_dir,
    run_case,
)

# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def bold(s):
    return _c("1", s)


def green(s):
    return _c("32", s)


def red(s):
    return _c("31", s)


def yellow(s):
    return _c("33", s)


def dim(s):
    return _c("2", s)


def cyan(s):
    return _c("36", s)


PASS = green("PASS")
FAIL = red("FAIL")


def rule(title: str = "") -> None:
    width = 72
    if title:
        pad = width - len(title) - 3
        print(bold(f"── {title} " + "─" * max(pad - 1, 0)))
    else:
        print(dim("─" * width))


# --------------------------------------------------------------------------
# Skeleton management
# --------------------------------------------------------------------------


def ensure_working_kernels(exercise_dir: Path) -> Path:
    """Populate kernels/ from skeleton/ the first time an exercise is opened.

    kernels/ is the learner's scratch space; skeleton/ is pristine, so `reset`
    always has something to restore from.
    """
    work = exercise_dir / "kernels"
    skel = exercise_dir / "skeleton"
    if not skel.is_dir():
        return work
    work.mkdir(exist_ok=True)
    for src in skel.glob("*.cpp"):
        dst = work / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
    return work


def cmd_reset(args) -> int:
    d = resolve_exercise_dir(args.exercise)
    skel = d / "skeleton"
    work = d / "kernels"
    if not skel.is_dir():
        print(red(f"no skeleton/ for {d.name}"))
        return 1
    work.mkdir(exist_ok=True)
    for src in skel.glob("*.cpp"):
        shutil.copyfile(src, work / src.name)
        print(f"restored {work.name}/{src.name}")
    return 0


def cmd_solution(args) -> int:
    d = resolve_exercise_dir(args.exercise)
    sol = d / "solution"
    files = sorted(sol.glob("*.cpp"))
    if not files:
        print(red(f"no solution for {d.name}"))
        return 1
    if args.apply:
        work = ensure_working_kernels(d)
        for f in files:
            shutil.copyfile(f, work / f.name)
            print(f"copied solution → kernels/{f.name}")
        return 0
    for f in files:
        rule(f"solution/{f.name}")
        print(f.read_text())
    return 0


# --------------------------------------------------------------------------
# Informational commands
# --------------------------------------------------------------------------


def cmd_list(args) -> int:
    dirs = list_exercise_dirs()
    if not dirs:
        print(red("no exercises found"))
        return 1
    print()
    print(bold("  tt-metal kernel dojo"))
    print()
    for d in dirs:
        try:
            ex = load_exercise(d)
            title = ex.title or d.name
            blurb = ex.blurb.strip().splitlines()[0] if ex.blurb.strip() else ""
        except Exception as exc:  # a broken task.py should not hide the rest
            title, blurb = d.name, red(f"(failed to load: {exc})")
        num = d.name.split("_")[0]
        print(f"  {cyan(num)}  {bold(title)}")
        if blurb:
            print(f"      {dim(blurb)}")
    print()
    print(dim("  dojo info <n>    read the lesson"))
    print(dim("  dojo test <n>    grade your kernels"))
    print(dim("  dojo bench <n>   measure performance"))
    print()
    return 0


THEORY_DIR = Path(__file__).resolve().parent.parent.parent / "theory"


def _theory_chapters() -> list[Path]:
    if not THEORY_DIR.is_dir():
        return []
    return sorted(THEORY_DIR.glob("*.md"))


def _chapter_title(path: Path) -> str:
    """First markdown H1, minus the leading number."""
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            # "00 — What is a kernel" -> "What is a kernel"
            parts = title.split("—", 1)
            return parts[1].strip() if len(parts) == 2 else title
    return path.stem


def cmd_theory(args) -> int:
    chapters = _theory_chapters()
    if not chapters:
        print(red(f"no chapters found in {THEORY_DIR}"))
        return 1

    if not args.chapter:
        print()
        print(bold("  theory"))
        print()
        for ch in chapters:
            num = ch.name.split("-")[0]
            print(f"  {cyan(num)}  {_chapter_title(ch)}")
        print()
        print(dim("  dojo theory <n>   read a chapter"))
        print()
        return 0

    key = args.chapter.strip().zfill(2)
    matched = [c for c in chapters if c.name.startswith(key + "-")]
    if not matched:
        matched = [c for c in chapters if args.chapter.lower() in c.name.lower()]
    if len(matched) != 1:
        names = ", ".join(c.name.split("-")[0] for c in chapters)
        print(red(f"no single chapter matching '{args.chapter}'. Available: {names}"))
        return 1

    text = matched[0].read_text()
    pager = os.environ.get("PAGER")
    if sys.stdout.isatty() and pager and not args.no_pager:
        import subprocess

        subprocess.run([pager], input=text, text=True)
    else:
        print(text)
    return 0


def cmd_info(args) -> int:
    d = resolve_exercise_dir(args.exercise)
    readme = d / "README.md"
    if not readme.is_file():
        print(red(f"no README.md in {d}"))
        return 1
    text = readme.read_text()
    pager = os.environ.get("PAGER")
    if sys.stdout.isatty() and pager and not args.no_pager:
        import subprocess

        subprocess.run([pager], input=text, text=True)
    else:
        print(text)

    work = ensure_working_kernels(d)
    print()
    rule("your files")
    for f in sorted(work.glob("*.cpp")):
        print(f"  {f}")
    print()
    return 0


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------


def _select_cases(ex, only: str | None, perf_only: bool):
    cases = ex.cases()
    if perf_only:
        cases = [c for c in cases if c.perf] or cases[-1:]
    if only:
        matched = [c for c in cases if only.lower() in c.name.lower()]
        if not matched:
            names = ", ".join(c.name for c in cases)
            raise SystemExit(f"no case matching '{only}'. Available: {names}")
        cases = matched
    return cases


def cmd_test(args) -> int:
    d = resolve_exercise_dir(args.exercise)
    ensure_working_kernels(d)
    ex = load_exercise(d)
    kdir = kernel_dir_for(d, args.solution)
    cases = _select_cases(ex, args.case, perf_only=False)

    print()
    rule(f"{ex.title or d.name}")
    if args.solution:
        print(yellow("  running the reference solution"))
    print(dim(f"  kernels: {kdir}"))
    print()

    require_tt_metal_home()
    failures = 0
    hung = False
    with DeviceSession(device_id=args.device) as device:
        ctx = Ctx(device=device, kernel_dir=kdir)
        for case in cases:
            if hung:
                # The device is wedged for the rest of this process, so any
                # further case would fail for reasons unrelated to its own
                # correctness. Reporting those as failures would mislead.
                print(f"  {case.name:<28}{dim('skipped (device needs a fresh run)')}")
                continue
            label = f"  {case.name:<28}"
            if sys.stdout.isatty():
                # Overwritten by the result line. Only useful on a terminal —
                # piped, the \r would leave both lines in the output.
                sys.stdout.write(label + dim("running...") + "\r")
                sys.stdout.flush()
            try:
                _out, _ref, cmp_ = run_case(ex, case, ctx)
            except Exception as exc:
                failures += 1
                if _is_hang(exc):
                    hung = True
                    print(label + FAIL + "  " + red("device hang (deadlock)"))
                else:
                    print(label + FAIL + "  " + red(_short_exc(exc)))
                if args.verbose:
                    import traceback

                    traceback.print_exc()
                continue
            status = PASS if cmp_.passed else FAIL
            print(f"{label}{status}  {dim(cmp_.summary())}")
            if not cmp_.passed:
                failures += 1
                if cmp_.detail:
                    print(f"      {red(cmp_.detail)}")

    print()
    total = len(cases)
    if hung:
        _hang_report()
        print()
        return 1
    if failures:
        print(f"  {red(f'{failures}/{total} cases failed')}")
        print()
        print(dim("  hints:  dojo info " + d.name.split('_')[0] + "   (re-read the lesson)"))
        print(dim("          dojo solution " + d.name.split('_')[0] + "   (show reference)"))
    else:
        print(f"  {green(f'all {total} cases passed')}")
        if any(c.perf for c in ex.cases()):
            print(dim(f"  now measure it:  dojo bench {d.name.split('_')[0]}"))
    print()
    return 1 if failures else 0


def _short_exc(exc: Exception) -> str:
    msg = str(exc).strip().splitlines()
    if not msg:
        return type(exc).__name__
    # tt-metal exceptions are verbose; the first line carries the signal.
    first = msg[0]
    return first if len(first) < 300 else first[:300] + " ..."


#: Substrings that mean the device stopped making progress rather than that the
#: kernel computed something wrong.
_HANG_MARKERS = ("TIMEOUT", "potential hang", "waiting for physical cores")


def _is_hang(exc: Exception) -> bool:
    text = str(exc)
    return any(m in text for m in _HANG_MARKERS)


def _hang_report() -> None:
    print()
    print(f"  {red('The device stopped responding — your kernels deadlocked.')}")
    print()
    print("  This is almost always circular-buffer accounting. Check that:")
    print(f"    {dim('- every cb_reserve_back has a matching cb_push_back')}")
    print(f"    {dim('- every cb_wait_front has a matching cb_pop_front')}")
    print(f"    {dim('- the counts match: reserving 1 and pushing 2 will hang')}")
    print(f"    {dim('- all your kernels agree on how many tiles they process')}")
    print(f"    {dim('- you never wait for more pages than the CB can hold')}")
    print()
    print(f"  {dim('The device recovers on its own; just run the test again.')}")


def cmd_bench(args) -> int:
    d = resolve_exercise_dir(args.exercise)
    ensure_working_kernels(d)
    ex = load_exercise(d)
    kdir = kernel_dir_for(d, args.solution)
    cases = _select_cases(ex, args.case, perf_only=True)

    print()
    rule(f"{ex.title or d.name} — performance")
    if args.solution:
        print(yellow("  running the reference solution"))
    if not perf_mod.profiler_enabled():
        print(yellow("  device profiler off — falling back to host timing"))
        print(yellow("  (run via the `dojo` wrapper, or set TT_METAL_DEVICE_PROFILER=1)"))
    print()

    require_tt_metal_home()
    import ttnn

    rc = 0
    with DeviceSession(device_id=args.device) as device:
        ctx = Ctx(device=device, kernel_dir=kdir)
        for case in cases:
            print(bold(f"  {case.name}"))
            io_tensors, prog, _inputs, ref = prepare(ex, case, ctx)

            # Verify before timing: a fast wrong kernel is not a result.
            out = ttnn.to_torch(ttnn.generic_op(io_tensors, prog))
            cmp_ = ex.compare(out, ref, case)
            if not cmp_.passed:
                print(f"    {FAIL}  {red('incorrect output — fix correctness first')}")
                print(f"    {dim(cmp_.summary())}")
                rc = 1
                continue

            result = perf_mod.benchmark(
                device, lambda: ttnn.generic_op(io_tensors, prog), iterations=args.iterations
            )
            wl = ex.workload(case)
            result.bytes_moved = wl.bytes_moved
            result.flops = wl.flops
            result.cores = case.get("cores_used", 0)

            for line in result.summary_lines():
                print(f"    {line}")

            target = ex.perf_targets.get(case.name)
            if target:
                us = result.ns / 1000.0
                ratio = us / target
                verdict = green("under target") if ratio <= 1.0 else yellow(f"{ratio:.2f}x target")
                print(f"    {'target':<12} {target:10.2f} us   {verdict}")
            print()
    return rc


def cmd_ide(args) -> int:
    from . import ide

    path, n = ide.generate(arch=args.arch)
    print()
    print(f"  wrote {path}")
    print(f"  {n} kernel files indexed  (arch: {args.arch})")
    print()
    print(dim("  Restart your editor's language server to pick it up."))
    print(dim("  VS Code + clangd:  Ctrl-Shift-P -> 'clangd: Restart language server'"))
    print()
    print(dim("  Kernels target RISC-V, so a host parse cannot fully typecheck the"))
    print(dim("  SFPU vector headers. Expect your own kernel files to be clean, with"))
    print(dim("  any residual errors confined to sfpi.h. See theory 09."))
    print()
    return 0


def cmd_doctor(args) -> int:
    print()
    rule("environment")
    ok = True

    try:
        import ttnn  # noqa: F401

        print(f"  {green('ok')}    ttnn imports")
    except Exception as exc:
        print(f"  {red('fail')}  ttnn import: {exc}")
        return 1

    home = os.environ.get("TT_METAL_HOME")
    print(f"  {green('ok') if home else yellow('warn')}    TT_METAL_HOME={home or '(unset)'}")

    prof = perf_mod.profiler_enabled()
    print(f"  {green('ok') if prof else yellow('off')}   device profiler {'enabled' if prof else 'disabled'}")

    try:
        import ttnn

        n = ttnn.GetNumAvailableDevices()
        print(f"  {green('ok')}    {n} device(s) visible")
        with DeviceSession(device_id=args.device) as dev:
            print(f"  {green('ok')}    opened device {args.device}: arch={dev.arch()}")
            try:
                grid = dev.compute_with_storage_grid_size()
                print(f"  {green('ok')}    compute grid {grid.x} x {grid.y}")
            except Exception:
                pass
    except Exception as exc:
        ok = False
        print(f"  {red('fail')}  device: {_short_exc(exc)}")

    print()
    return 0 if ok else 1


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dojo", description="tt-metal kernel dojo")
    p.add_argument("--device", type=int, default=0, help="device id (default 0)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list exercises").set_defaults(func=cmd_list)

    d = sub.add_parser("doctor", help="check the environment")
    d.set_defaults(func=cmd_doctor)

    ide_p = sub.add_parser("ide", help="generate compile_commands.json for editor support")
    ide_p.add_argument("--arch", default="wormhole_b0", help="target arch (default wormhole_b0)")
    ide_p.set_defaults(func=cmd_ide)

    th = sub.add_parser("theory", help="read the theory chapters")
    th.add_argument("chapter", nargs="?", help="chapter number; omit to list")
    th.add_argument("--no-pager", action="store_true")
    th.set_defaults(func=cmd_theory)

    i = sub.add_parser("info", help="read an exercise's lesson")
    i.add_argument("exercise")
    i.add_argument("--no-pager", action="store_true")
    i.set_defaults(func=cmd_info)

    t = sub.add_parser("test", help="grade your kernels")
    t.add_argument("exercise")
    t.add_argument("--solution", action="store_true", help="grade the reference instead")
    t.add_argument("--case", help="run only cases matching this substring")
    t.add_argument("-v", "--verbose", action="store_true")
    t.set_defaults(func=cmd_test)

    b = sub.add_parser("bench", help="measure kernel performance")
    b.add_argument("exercise")
    b.add_argument("--solution", action="store_true")
    b.add_argument("--case")
    b.add_argument("--iterations", type=int, default=20)
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("reset", help="restore the skeleton kernels")
    r.add_argument("exercise")
    r.set_defaults(func=cmd_reset)

    s = sub.add_parser("solution", help="show the reference solution")
    s.add_argument("exercise")
    s.add_argument("--apply", action="store_true", help="copy it into kernels/")
    s.set_defaults(func=cmd_solution)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
