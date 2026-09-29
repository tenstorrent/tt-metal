# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Profile-only forward drains; no TTNN imports or tensor copies in this helper."""

import json
import time
from collections import Counter


class ProfileSegmentComplete(Exception):
    """Controlled stop after the selected module's output has been drained."""


def walk_named(roots):
    seen = set()

    def visit(path, module):
        if id(module) in seen:
            return
        seen.add(id(module))
        yield path, module
        for name, child in module.named_children():
            yield from visit(f"{path}.{name}", child)

    for path, root in roots:
        yield from visit(path, root)


class ProfileDrains:
    """Drain after selected forwards and record actual device-operation gaps.

    Hooks accumulate small operations until the requested gap is reached; the
    segment stop always drains its tail. A single leaf may overshoot the threshold.
    Counts are observed operation IDs, not kernel or marker
    counts, and cannot on their own prove that no profiler markers were dropped.
    The caller must also validate the device log and emitted profiler data.
    """

    def __init__(
        self, roots, types, read_profiler, operation_id, directory, stop_module=None, minimum_operation_gap=32
    ):
        assert isinstance(minimum_operation_gap, int) and 1 <= minimum_operation_gap <= 64
        self.minimum_operation_gap = minimum_operation_gap
        self.roots = tuple(roots)
        self.nodes = tuple(walk_named(self.roots))
        self.read_profiler = read_profiler
        self.operation_id = operation_id
        self.stop_module = stop_module
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = directory / "profile-drains.jsonl"
        self.events_file = self.events_path.open("x")
        self.last_id = int(operation_id())
        self.events = []
        self.saved = []
        self.coverage = []
        self.phase = "setup"
        self.completed_segment = None
        try:
            for path, module in self.nodes:
                if isinstance(module, types) or module is stop_module:
                    self._hook(path, module)
            assert self.coverage, "no audio profile hooks installed"
            if stop_module is not None:
                assert any(module is stop_module for _, module in self.nodes), "stop module is outside audio tree"
            coverage = {"modules": self.coverage, "counts": dict(Counter(row["type"] for row in self.coverage))}
            (directory / "profile-hook-coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
            print("C03_PROFILE_HOOK_COUNTS=" + json.dumps(coverage["counts"], sort_keys=True), flush=True)
        except BaseException:
            self.close()
            raise

    def _hook(self, path, module):
        had_instance = "forward" in vars(module)
        instance_value = vars(module).get("forward")
        original = module.forward
        self.saved.append((module, had_instance, instance_value))
        self.coverage.append({"path": path, "type": type(module).__name__})

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            self.drain(path, force=module is self.stop_module)
            if module is self.stop_module:
                self.completed_segment = path
                raise ProfileSegmentComplete(path)
            return result

        module.forward = wrapped

    def assert_same_modules(self, roots):
        current = tuple(walk_named(roots))
        assert len(current) == len(self.nodes) and all(
            path == old_path and module is old_module
            for (path, module), (old_path, old_module) in zip(current, self.nodes)
        ), "audio module identities changed during weight preparation"

    def drain(self, path, force=False):
        end = int(self.operation_id())
        assert end >= self.last_id, "device operation ID moved backwards"
        gap = end - self.last_id
        if not force and gap < self.minimum_operation_gap:
            return
        start = time.perf_counter()
        self.read_profiler()
        row = {
            "phase": self.phase,
            "path": path,
            "op_id_start": self.last_id,
            "op_id_end": end,
            "operation_gap": gap,
            "drain_wall_ms": (time.perf_counter() - start) * 1000,
        }
        self.events_file.write(json.dumps(row, sort_keys=True) + "\n")
        self.events_file.flush()
        self.events.append(row)
        self.last_id = end
        print("C03_PROFILE_DRAIN=" + json.dumps(row, sort_keys=True), flush=True)

    def summary(self):
        return {
            "drain_count": len(self.events),
            "minimum_operation_gap": self.minimum_operation_gap,
            "max_operation_gap": max((row["operation_gap"] for row in self.events), default=0),
            "completed_segment": self.completed_segment,
            "note": "operation gaps are not marker counts; independently reject dropped-marker logs",
        }

    def close(self):
        for module, had_instance, value in reversed(self.saved):
            if had_instance:
                module.forward = value
            else:
                del module.forward
        self.saved.clear()
        self.events_file.close()
