# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
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


def _loaded_lib(root: Path) -> Path | None:
    for rel in ("build_Release/lib/libtt_metal.so", "build/lib/libtt_metal.so"):
        p = root / rel
        if p.is_file():
            return p
    hits = sorted(root.glob("build*/lib/libtt_metal.so"))
    return hits[0] if hits else None


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
    for target in (["tt_metal/libtt_metal.so"], []):
        try:
            r = subprocess.run(
                ["ninja", "-C", str(build), *target],
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except Exception as exc:
            _log(f"rebuild invocation failed: {exc}")
            return False
        if r.returncode == 0:
            break
    else:
        _log("ninja rebuild failed")
        return False
    built = build / "tt_metal" / "libtt_metal.so"
    lib = _loaded_lib(root)
    if built.is_file() and lib is not None and built.resolve() != lib.resolve():
        try:
            lib.write_bytes(built.read_bytes())
        except Exception as exc:
            _log(f"could not install rebuilt lib: {exc}")
            return False
    return True


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
                _log(f"profiler.cpp did not match expected pattern ({matched}/{len(_BLOCKS)}); leaving stock, skipping heal")
                return
            src.with_name("profiler.cpp.perfauto_bak").write_text(text)
            src.write_text(patched)
            _log("detected unpatched tt-metal profiler (orphan-marker crash) -> applied fix, rebuilding libtt_metal (one-time, ~2-3 min)...")
        build = _build_dir(root)
        if build is None:
            _log("no build dir found (wheel/prebuilt install) -> cannot rebuild; run will use stock profiler")
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
            _log("rebuild done but marker not found in lib; proceeding")
    except Exception as exc:
        _log(f"skipped ({type(exc).__name__}: {exc})")
