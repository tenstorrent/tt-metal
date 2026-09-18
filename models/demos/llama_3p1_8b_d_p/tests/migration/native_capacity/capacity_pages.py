"""Bounded packed-page capture before readiness; no full-cache snapshot or tensor import."""

import hashlib
import os
from pathlib import Path

from capacity_execution import PAGE, require, selected_keys
from page_io import read_page


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


class SelectedWriter:
    def __init__(self, path, phase):
        self.path = Path(path)
        self.phase = phase
        self.expected = set(selected_keys(phase))
        self.keys = []
        self.seen = set()
        self.stream = self.path.open("xb")
        self.closed = False

    def capture(self, table, begin, end):
        spec = self.phase["source_command"]
        lo = max(begin, spec["from"])
        hi = min(end, spec["to"])
        count = 0
        for c in range(16):
            for layer in range(32):
                for pos in range(lo, hi, 32):
                    key = (c, self.phase["source_slot"], layer, pos)
                    require(key in self.expected and key not in self.seen, "Unexpected or duplicate selected key")
                    raw = read_page(table, key)
                    self.stream.write(raw)
                    self.keys.append(key)
                    self.seen.add(key)
                    count += 1
        # Flush selected bytes before the callback can publish any readiness counter.
        self.stream.flush()
        if count:
            os.fsync(self.stream.fileno())
        return count

    def finish(self):
        require(not self.closed and self.seen == self.expected, "Missing selected keys")
        self.close()
        return dict(
            path=str(self.path),
            sha256=digest(self.path),
            keys=self.keys,
            pages=len(self.keys),
            bytes=len(self.keys) * PAGE,
            source_slot=self.phase["source_slot"],
            destination_slot=self.phase["destination_slot"],
            begin=self.phase["source_command"]["from"],
            end=self.phase["source_command"]["to"],
            scope="actual packed source bytes saved before per-chunk layer readiness",
        )

    def close(self):
        if not self.closed:
            self.stream.close()
            self.closed = True


def compare_selected(receipt, phase, table, *, destination):
    expected = set(selected_keys(phase))
    keys = [tuple(k) for k in receipt["keys"]]
    require(
        len(keys) == len(expected) and len(set(keys)) == len(keys) and set(keys) == expected,
        "Missing, duplicate or foreign selected key",
    )
    path = Path(receipt["path"])
    require(
        receipt["pages"] == len(keys)
        and receipt["bytes"] == len(keys) * PAGE
        and path.stat().st_size == receipt["bytes"]
        and digest(path) == receipt["sha256"],
        "Selected snapshot changed",
    )
    groups = {}
    count = 0
    with path.open("rb") as stream:
        for key in keys:
            raw = stream.read(PAGE)
            require(len(raw) == PAGE, "Truncated saved page")
            c, s, l, p = key
            target = (c, phase["destination_slot"] if destination else s, l, p)
            require(raw == read_page(table, target), "Selected packed page mismatch: " + str(target))
            groups[c] = groups.get(c, 0) + 1
            count += 1
        require(not stream.read(1), "Extra selected bytes")
    require(digest(path) == receipt["sha256"], "Selected bytes changed during comparison")
    require(groups == {c: 1024 for c in range(16)}, "Incomplete config/layer coverage")
    return dict(
        pages=count,
        bytes=count * PAGE,
        configs=16,
        layers=32,
        exact=True,
        kind="destination_landing" if destination else "source_unchanged_after_transfer",
    )


def sample_pages(table, keys):
    return {",".join(map(str, k)): hashlib.sha256(read_page(table, k)).hexdigest() for k in keys}


def check_samples(table, values):
    for key, raw in values.items():
        require(
            hashlib.sha256(read_page(table, tuple(map(int, key.split(","))))).hexdigest() == raw,
            "Untouched sentinel changed: " + key,
        )
    return dict(pages=len(values), bytes=len(values) * PAGE, unchanged_sha256=True)


def selected_config_hashes(receipt):
    # Normalize capture order: the ragged range arrives in two chunks, the full range in one.
    groups = saved_selected_group_hashes(receipt)
    return {
        str(c): hashlib.sha256(
            b"".join(bytes.fromhex(groups[str(c) + ":" + str(layer)]) for layer in range(32))
        ).hexdigest()
        for c in range(16)
    }


def live_selected_group_hashes(table, phase):
    """Bound every config/layer separately; a stale warmup group cannot hide behind another."""
    from capacity_execution import selected_keys

    hashes = {}
    for key in selected_keys(phase):
        config, _, layer, _ = key
        digest = hashes.setdefault(str(config) + ":" + str(layer), hashlib.sha256())
        digest.update(read_page(table, key))
    return {key: value.hexdigest() for key, value in hashes.items()}


def saved_selected_group_hashes(receipt):
    hashes = {}
    require(digest(receipt["path"]) == receipt["sha256"], "Selected snapshot changed")
    with Path(receipt["path"]).open("rb") as stream:
        for config, _, layer, _ in receipt["keys"]:
            raw = stream.read(PAGE)
            require(len(raw) == PAGE, "Truncated selected snapshot")
            hashes.setdefault(str(config) + ":" + str(layer), hashlib.sha256()).update(raw)
        require(not stream.read(1), "Extra selected snapshot bytes")
    return {key: value.hexdigest() for key, value in hashes.items()}


def require_changed_groups(before, after):
    expected = {str(config) + ":" + str(layer) for config in range(16) for layer in range(32)}
    require(set(before) == set(after) == expected, "Incomplete selected config/layer groups")
    require(
        all(before[key] != after[key] for key in expected),
        "Real request left a selected config/layer group unchanged from its warmup state",
    )
    return dict(groups=512, all_changed=True, before=before, after=after)
