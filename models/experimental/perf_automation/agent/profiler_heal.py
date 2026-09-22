# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_MARKER = "PERF_AUTOMATION_ORPHAN_SKIP"
_HEAL_ATTEMPTED = False

_BLOCKS = [
    (
        """                if (start_marker_stack.empty()) {
                    // Orphan ZONE_END from a dropped-marker run; skip instead of fatal.
                    if (!this->had_dropped_markers.load(std::memory_order_relaxed)) {
                        TT_FATAL(
                            false,
                            "End marker found without a corresponding start marker.\\nEnd marker: {}",
                            marker.to_string());
                    }
                    device_marker_it = next_device_marker_it;
                    continue;
                }""",
        """                if (start_marker_stack.empty()) {
                    // PERF_AUTOMATION_ORPHAN_SKIP: tolerate an orphan ZONE_END (mesh / high-volume
                    // marker imbalance) -- warn and skip, keep a partial report instead of aborting.
                    log_warning(
                        tt::LogMetal,
                        "PERF_AUTOMATION_ORPHAN_SKIP End marker found without a corresponding start "
                        "marker; skipping (report will be partial).\\nEnd marker: {}",
                        marker.to_string());
                    device_marker_it = next_device_marker_it;
                    continue;
                }""",
    ),
    (
        """                    if (start_marker_it->marker_id != marker.marker_id) {
                        if (!this->had_dropped_markers.load(std::memory_order_relaxed)) {
                            TT_FATAL(
                                false,
                                "Start and end marker IDs do not match.\\nStart marker: {}\\nEnd marker: {}",
                                start_marker_it->to_string(),
                                marker.to_string());
                        }
                        // Stack is misaligned due to drops; skip this end without popping.
                        device_marker_it = next_device_marker_it;
                        continue;
                    }""",
        """                    if (start_marker_it->marker_id != marker.marker_id) {
                        // PERF_AUTOMATION_ORPHAN_SKIP: stack misaligned -- warn and skip this end.
                        log_warning(
                            tt::LogMetal,
                            "Start and end marker IDs do not match; skipping this end (report will be "
                            "partial).\\nStart marker: {}\\nEnd marker: {}",
                            start_marker_it->to_string(),
                            marker.to_string());
                        device_marker_it = next_device_marker_it;
                        continue;
                    }""",
    ),
    (
        """    if (!start_marker_stack.empty()) {
        if (this->had_dropped_markers.load(std::memory_order_relaxed)) {
            log_warning(
                tt::LogMetal,
                "{} start markers detected without corresponding end markers (some end markers were "
                "dropped due to DRAM-buffer overflow; report will be partial). Marker at top of stack: {}",
                start_marker_stack.size(),
                start_marker_stack.top()->to_string());
        } else {
            TT_FATAL(
                false,
                "{} start markers detected without corresponding end markers. Marker at top of stack: {}",
                start_marker_stack.size(),
                start_marker_stack.top()->to_string());
        }
    }""",
        """    if (!start_marker_stack.empty()) {
        // PERF_AUTOMATION_ORPHAN_SKIP: leftover starts with no matching end -- warn + partial report.
        log_warning(
            tt::LogMetal,
            "{} start markers detected without corresponding end markers (marker imbalance; report will "
            "be partial). Marker at top of stack: {}",
            start_marker_stack.size(),
            start_marker_stack.top()->to_string());
    }""",
    ),
]


def _log(msg: str) -> None:
    print(f"  [profiler-heal] {msg}", file=sys.stderr, flush=True)


def _src_path(root: Path) -> Path:
    return root / "tt_metal" / "impl" / "profiler" / "profiler.cpp"


_INEFFECTIVE_SENTINEL = ".perfauto_heal_ineffective"


def _loaded_lib(root: Path) -> Path | None:
    """The libtt_metal.so the RUNTIME actually loads, resolved via ldd on the imported
    _ttnn.so. Guessing a hardcoded path was the bug: the heal rebuilt one copy and then
    verified a different one, reported "marker not found in lib", and reverted -- burning
    ~3 min on every run (observed 2026-07-25)."""
    ttnn_so = root / "ttnn" / "ttnn" / "_ttnn.so"
    if ttnn_so.is_file():
        try:
            r = subprocess.run(["ldd", str(ttnn_so)], capture_output=True, text=True, timeout=60)
            for ln in (r.stdout or "").splitlines():
                if "libtt_metal.so" in ln and "=>" in ln:
                    cand = Path(ln.split("=>", 1)[1].strip().split(" ")[0])
                    if cand.is_file():
                        return cand
        except Exception:  # noqa: BLE001
            pass
    for rel in ("build_Release/lib/libtt_metal.so", "build/lib/libtt_metal.so"):
        p = root / rel
        if p.is_file():
            return p
    hits = sorted(root.glob("build*/lib/libtt_metal.so"))
    return hits[0] if hits else None


def _lib_identity(lib: Path | None) -> str:
    """A cheap fingerprint of the library the runtime loads. An 'ineffective' verdict is
    only trusted while the library it was recorded against is still the one in place; once
    that library is rebuilt or replaced, the verdict is stale and the heal retries."""
    if lib is None:
        return "none"
    try:
        st = lib.stat()
        return f"{lib.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    except Exception:  # noqa: BLE001
        return "none"


def _heal_marked_ineffective(build: Path, lib: Path | None) -> bool:
    """Whether a prior run already proved patch+rebuild cannot land on THIS library.

    The verdict is keyed to the loaded library's identity, not just the build directory:
    worktrees share one build tree, so a verdict written from a throwaway worktree (or
    before the library was rebuilt) must not permanently disable healing for later runs.
    A verdict whose recorded identity no longer matches the current library is stale and
    is cleared so the heal is attempted again."""
    sentinel = build / _INEFFECTIVE_SENTINEL
    try:
        if not sentinel.is_file():
            return False
        recorded = sentinel.read_text().splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return False
    if recorded == _lib_identity(lib):
        return True
    try:
        sentinel.unlink()
    except Exception:  # noqa: BLE001
        pass
    return False


def _mark_heal_ineffective(build: Path, lib: Path | None, why: str) -> None:
    """Record that patch+rebuild does not land on this library, so later runs skip it
    instead of repeating a known-failing 3-minute rebuild every time. The verdict carries
    the library's identity so it expires automatically once the library changes."""
    try:
        (build / _INEFFECTIVE_SENTINEL).write_text(f"{_lib_identity(lib)}\n{why[:500]}")
    except Exception:  # noqa: BLE001
        pass


def _lib_has_marker(lib: Path) -> bool:
    try:
        return _MARKER.encode() in lib.read_bytes()
    except Exception:
        return False


def _build_dir(root: Path) -> Path | None:
    for rel in ("build_Release", "build"):
        d = root / rel
        if (d / "build.ninja").is_file():
            return d
    return None


def _rebuild(root: Path, build: Path) -> bool:
    for targets in (["tt_metal/libtt_metal.so", "ttnn/_ttnn.so"], []):
        try:
            r = subprocess.run(
                ["ninja", "-C", str(build), *targets],
                capture_output=True,
                text=True,
                # a full tt-metal rebuild on a slower machine can exceed a fixed 90 min
                timeout=_rebuild_timeout(),
            )
        except Exception as exc:
            _log(f"rebuild invocation failed: {exc}")
            return False
        if r.returncode == 0:
            break
    else:
        _log("ninja rebuild failed")
        return False
    installs = [
        (build / "tt_metal" / "libtt_metal.so", "build*/lib/libtt_metal.so"),
        (build / "ttnn" / "_ttnn.so", "build*/lib/_ttnn.so"),
        (build / "ttnn" / "_ttnn.so", "ttnn/ttnn/_ttnn.so"),
    ]
    for built, loaded_glob in installs:
        if not built.is_file():
            continue
        for loaded in sorted(root.glob(loaded_glob)):
            if built.resolve() != loaded.resolve():
                try:
                    loaded.write_bytes(built.read_bytes())
                except Exception as exc:
                    _log(f"could not install {loaded.name}: {exc}")
                    return False
    # The glob above is relative to root; when the build tree is shared/symlinked the
    # library the runtime actually loads (ldd-resolved) may sit outside root/build*/lib.
    # Install the freshly built libtt_metal.so directly over that exact path too, so a
    # rebuilt marker is guaranteed to reach the library that is loaded and verified.
    loaded_lib = _loaded_lib(root)
    built_lib = build / "tt_metal" / "libtt_metal.so"
    if loaded_lib is not None and built_lib.is_file() and built_lib.resolve() != loaded_lib.resolve():
        try:
            loaded_lib.write_bytes(built_lib.read_bytes())
        except Exception as exc:
            _log(f"could not install {loaded_lib}: {exc}")
            return False
    return True


def _rebuild_timeout() -> int:
    """Budget for the ninja rebuild, on the same adaptive chain as every other timer."""
    from .probes import adaptive_op_timeout

    return int(adaptive_op_timeout("build", env_key="PERF_HEAL_REBUILD_TIMEOUT_S"))


def ensure_profiler_patched(tt_metal_root) -> None:
    global _HEAL_ATTEMPTED
    if _HEAL_ATTEMPTED:
        return
    _HEAL_ATTEMPTED = True
    try:
        root = Path(tt_metal_root)
        src = _src_path(root)
        lib = _loaded_lib(root)
        if not src.is_file():
            return
        if lib is not None and _lib_has_marker(lib):
            return
        text = src.read_text()
        if _MARKER in text:
            patched = text
        else:
            patched = text
            matched = 0
            for old, new in _BLOCKS:
                if old in patched:
                    patched = patched.replace(old, new, 1)
                    matched += 1
            if matched != len(_BLOCKS):
                _log(
                    f"profiler.cpp did not match expected pattern ({matched}/{len(_BLOCKS)}); leaving stock, skipping heal"
                )
                return
            src.with_name("profiler.cpp.perfauto_bak").write_text(text)
            src.write_text(patched)
            _log(
                "detected unpatched tt-metal profiler (orphan-marker crash) -> applied fix, rebuilding libtt_metal (one-time, ~2-3 min)..."
            )
        build = _build_dir(root)
        if build is None:
            _log("no build dir found (wheel/prebuilt install) -> cannot rebuild; run will use stock profiler")
            return
        if _heal_marked_ineffective(build, lib):
            _log("patch+rebuild already proven ineffective on this build -> skipping (stock profiler)")
            if _MARKER not in text:
                src.write_text(text)
            return
        if not _rebuild(root, build):
            bak = src.with_name("profiler.cpp.perfauto_bak")
            if bak.is_file() and _MARKER not in text:
                src.write_text(text)
            return
        lib = _loaded_lib(root)
        if lib is not None and _lib_has_marker(lib):
            _log("profiler patched + rebuilt; mesh profiling will no longer crash on orphan markers")
        else:
            # The rebuild did not land in the library the runtime loads. Revert the source and
            # record it so the next run does not repeat this 3-minute round trip.
            _log(
                f"rebuild did not reach the loaded library ({lib}); reverting source and "
                "disabling further heal attempts on this build (stock profiler is used)"
            )
            _mark_heal_ineffective(build, lib, f"marker absent from {lib} after ninja rebuild")
            bak = src.with_name("profiler.cpp.perfauto_bak")
            if bak.is_file() and _MARKER not in text:
                src.write_text(text)
    except Exception as exc:
        _log(f"skipped ({type(exc).__name__}: {exc})")
