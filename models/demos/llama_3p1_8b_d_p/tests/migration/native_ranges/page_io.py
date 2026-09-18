"""Packed BFP8 snapshots for a bounded source-to-passive transport test; no native imports."""
import hashlib
import json
import os
from pathlib import Path

PAGE = 4352
NAMES = tuple(f"{kind}_h{head}" for kind in ("k", "v") for head in range(8))


def geometry(table):
    if table.num_configs() != 16 or tuple(table.config_name(i) for i in range(16)) != NAMES:
        raise ValueError("Incomplete or reordered K/V configs")
    rows = [
        (c.num_layers, c.max_sequence_length, c.num_slots, c.chunk_n_tokens, c.chunk_size_bytes)
        for c in (table.config(i) for i in range(16))
    ]
    layers, length, slots, tokens, page = rows[0]
    if len(set(rows)) != 1 or (layers, slots, tokens, page) != (32, 2, 32, PAGE):
        raise ValueError("Wrong native table geometry")
    if type(length) is not int or not 0 < length <= 131072 or length % 32:
        raise ValueError("Invalid table capacity")
    if table.total_entries() != 16 * 2 * 32 * (length // 32):
        raise ValueError("Incomplete native address table")
    return length


def range_keys(slot, begin, end):
    if (
        any(type(x) is not int for x in (slot, begin, end))
        or slot not in (0, 1)
        or not 0 <= begin < end <= 131072
        or begin % 32
        or end % 32
    ):
        raise ValueError("Packed snapshots require complete, aligned tiles")
    for config in range(16):
        for layer in range(32):
            for position in range(begin, end, 32):
                yield config, slot, layer, position


def read_page(table, key):
    config, slot, layer, position = key
    if table.lookup(layer, position, slot, config).size_bytes != PAGE:
        raise ValueError("Wrong native packed page size")
    raw = table.read_device_chunk(layer, position, slot, config)
    if not isinstance(raw, bytes) or len(raw) != PAGE:
        raise ValueError("Incomplete native packed read")
    return raw


def snapshot(table, path, slot, begin, end):
    """Publish the receipt only after every page is durably saved; partial files remain on failure."""
    if end > geometry(table):
        raise ValueError("Snapshot exceeds actual allocation")
    path = Path(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("xb") as stream:
        for key in range_keys(slot, begin, end):
            raw = read_page(table, key)
            stream.write(raw)
            digest.update(raw)
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    result = dict(
        path=str(path.resolve()),
        slot=slot,
        begin=begin,
        end=end,
        pages=count,
        bytes=count * PAGE,
        sha256=digest.hexdigest(),
    )
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


class SavedPages:
    def __init__(self, receipts, length):
        self.length = length
        self.rows = sorted(receipts, key=lambda r: (r["slot"], r["begin"]))
        self.handles = {}
        if not self.rows:
            raise ValueError("Empty snapshot cannot establish byte transport")
        for row in self.rows:
            next(range_keys(row["slot"], row["begin"], row["end"]))
        for slot in (0, 1):
            cursor = 0
            for row in (r for r in self.rows if r["slot"] == slot):
                if row["begin"] != cursor or row["end"] > length:
                    raise ValueError("Snapshot coverage has a gap, duplicate, or overlap")
                cursor = row["end"]
            if cursor != length:
                raise ValueError("Incomplete slot snapshot")
        for row in self.rows:
            path = Path(row["path"])
            expected = 16 * 32 * ((row["end"] - row["begin"]) // 32) * PAGE
            if row["bytes"] != expected or row["pages"] * PAGE != expected or path.stat().st_size != expected:
                raise ValueError("Snapshot byte inventory differs")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("Snapshot hash differs")
            self.handles[str(path)] = path.open("rb")

    def get(self, key):
        config, slot, layer, position = key
        row = next(r for r in self.rows if r["slot"] == slot and r["begin"] <= position < r["end"])
        tiles = (row["end"] - row["begin"]) // 32
        offset = ((config * 32 + layer) * tiles + (position - row["begin"]) // 32) * PAGE
        stream = self.handles[row["path"]]
        stream.seek(offset)
        raw = stream.read(PAGE)
        if len(raw) != PAGE:
            raise ValueError("Saved page was truncated after validation")
        return raw

    def close(self):
        for stream in self.handles.values():
            stream.close()


def compare(saved, table, slot_map=(0, 1), *, require_equal=True):
    if geometry(table) != saved.length or sorted(slot_map) != [0, 1]:
        raise ValueError("Source/destination geometry or slot mapping differs")
    mismatches, pages, groups = [], 0, []
    raw_slot_hashes = {}
    for config in range(16):
        for slot in (0, 1):
            expected_hash, actual_hash = hashlib.sha256(), hashlib.sha256()
            bad = 0
            for layer in range(32):
                for position in range(0, saved.length, 32):
                    key = config, slot, layer, position
                    a = saved.get(key)
                    b = read_page(table, (config, slot_map[slot], layer, position))
                    expected_hash.update(a)
                    actual_hash.update(b)
                    pages += 1
                    if a != b:
                        bad += 1
                        if len(mismatches) < 16:
                            mismatches.append(dict(key=key, first_byte=next(i for i in range(PAGE) if a[i] != b[i])))
            raw_slot_hashes[config, slot] = expected_hash.hexdigest()
            groups.append(
                dict(
                    config=config,
                    source_slot=slot,
                    destination_slot=slot_map[slot],
                    expected_sha256=expected_hash.hexdigest(),
                    actual_sha256=actual_hash.hexdigest(),
                    mismatched_pages=bad,
                )
            )
    result = dict(
        pages=pages,
        bytes=pages * PAGE,
        groups=groups,
        mismatches=mismatches,
        packed_bytes_equal=not any(r["mismatched_pages"] for r in groups),
        decoded_value_comparison=False,
        golden_comparison=False,
    )
    if any(raw_slot_hashes[c, 0] == raw_slot_hashes[c, 1] for c in range(16)):
        raise ValueError("Source slots are not observably distinct in every config")
    if require_equal and not result["packed_bytes_equal"]:
        raise ValueError(json.dumps(result))
    return result
