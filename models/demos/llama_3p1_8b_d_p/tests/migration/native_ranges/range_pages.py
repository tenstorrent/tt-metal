"""Exact full-page effects with logical valid/padding rows kept separate."""
import ast
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from edge_checks import check_values, seed_value
from owned_cleanup import cleanup_each
from page_io import PAGE, SavedPages, geometry, range_keys, read_page
from runner_support import require, sha256, write_json
from writer_boundaries import BoundaryCase, case_page_positions


def seed_cache(source_path, expected_hash, ttnn, torch, mesh, cache, writer, errors):
    require(sha256(source_path) == expected_hash, "frozen seed helper changed")
    tree = ast.parse(Path(source_path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "seed_cache")
    namespace = dict(seed_value=seed_value, cleanup_each=cleanup_each)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["seed_cache"](ttnn, torch, mesh, cache, writer, cleanup_errors=errors)


def validate_new_content_fixture(fixtures):
    require(
        len(fixtures["A"]) >= 33
        and len(fixtures["C"]) >= 65
        and all(a != c for a, c in zip(fixtures["A"][:33], fixtures["C"][:33])),
        "frozen A/C new-content precondition differs",
    )


@lru_cache(maxsize=1)
def progress_contract():
    from range_contract import scenario

    doc = scenario(Path(__file__).with_name("scenario.json"))
    fixture = Path(__file__).with_name("fixtures.json")
    require(sha256(fixture) == doc["fixture_sha256"], "progress fixture differs from frozen token identity")
    validate_new_content_fixture(json.loads(fixture.read_bytes())["tokens"])
    return doc


def source_progress_policy(phase, call):
    doc = progress_contract()
    matches = [index for index, expected in enumerate(doc["phases"]) if phase == expected]
    require(len(matches) == 1 and call in phase["compute_calls"], "unknown source phase/call identity")
    index = matches[0]
    replay = index in (1, 2)
    # C105 overlaps an already-valid row32. Its progress must come from newly valid33..64.
    begin = 33 if index == 5 else call["begin"]
    return dict(
        valid_change_required=not replay,
        progress_begin=begin,
        progress_end=call["end"],
        expected_valid_change_groups=0 if replay else 512,
        replay_phase=phase["name"] if replay else None,
    )


class PageEffect:
    def __init__(self, role, phase=None, call=None, decode=None):
        self.role, self.phase, self.call, self.decode = role, phase, call, decode
        self.count = self.selected = self.untouched = self.padding_values = self.valid_values = 0
        self.progress = source_progress_policy(phase, call) if role == "source" else None
        self.changed_valid_groups = set()
        if role == "source":
            self.slot = call["slot"]
            self.begin = call["begin"]
            self.end = call["end"]
        else:
            self.slot = phase["destination_slot"]
            self.begin = phase["source_command"]["from"]
            self.end = phase["source_command"]["to"]
        self.positions = set(case_page_positions(BoundaryCase(0, self.slot, self.begin, self.end)))

    def accept(self, key, before, after, source=None):
        config, slot, layer, position = key
        require(
            0 <= config < 16 and slot in (0, 1) and 0 <= layer < 32 and position in range(0, 2048, 32),
            "invalid page coordinate",
        )
        require(
            ((slot * 16 + config) * 32 + layer) * 64 + position // 32 == self.count, "missing/duplicate/reordered page"
        )
        require(all(isinstance(x, bytes) and len(x) == PAGE for x in (before, after)), "invalid packed page")
        if slot == self.slot and position in self.positions:
            if self.role == "passive":
                require(
                    isinstance(source, bytes) and len(source) == PAGE and after == source,
                    "selected destination packed bytes differ",
                )
            else:
                decoded = self.decode(after)
                values = check_values(
                    decoded, valid_rows=min(32, self.end - position), forbidden_seed=seed_value(config, slot, layer)
                )
                self.valid_values += values["valid_values"]
                self.padding_values += values["padding_values"]
                if self.progress["valid_change_required"] and (config, layer) not in self.changed_valid_groups:
                    old = self.decode(before)
                    require(len(old) == 32 and all(len(row) == 128 for row in old), "invalid prior decoded page")
                    lo = max(position, self.progress["progress_begin"]) - position
                    hi = min(position + 32, self.progress["progress_end"]) - position
                    # Require one changed valid value per config/layer across the write, not per row/value.
                    # Decoding excludes padding and encoding-only changes. This detects stale writes;
                    # it is not a numerical golden or evidence that every intended value is correct.
                    if any(decoded[row] != old[row] for row in range(lo, hi)):
                        self.changed_valid_groups.add((config, layer))
            self.selected += 1
        else:
            require(before == after, "untouched prefix/suffix/other-slot bytes changed")
            self.untouched += 1
        self.count += 1

    def finish(self):
        require(self.count == 65536 and self.selected == len(self.positions) * 512, "incomplete page-effect inventory")
        if self.progress is not None:
            require(
                len(self.changed_valid_groups) == self.progress["expected_valid_change_groups"],
                "missing valid-region progress for one or more config/layer writes",
            )
        result = dict(
            pages=self.count,
            selected_pages=self.selected,
            untouched_pages=self.untouched,
            selected_bytes=self.selected * PAGE,
            semantic_end=self.end,
            packed_end=max(self.positions) + 32,
            valid_values=self.valid_values,
            padding_values=self.padding_values,
            packed_bytes_equal=self.role == "passive",
            numerical_golden=False,
            structural_source_check=self.role == "source",
        )
        if self.progress is not None:
            result.update(self.progress, valid_change_groups=len(self.changed_valid_groups))
        return result


def capture_effect(table, directory, identity, previous, checker, source=None):
    require(geometry(table) == 2048, "wrong table geometry")
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    receipts = []
    for slot in (0, 1):
        path = directory / f"slot{slot}.bin"
        digest = hashlib.sha256()
        count = 0
        with path.open("xb") as output:
            for key in range_keys(slot, 0, 2048):
                raw = read_page(table, key)
                expected = None
                if source is not None and slot == checker.slot and key[3] in checker.positions:
                    expected = source.get((key[0], checker.phase["source_slot"], key[2], key[3]))
                checker.accept(key, previous.get(key), raw, expected)
                output.write(raw)
                digest.update(raw)
                count += 1
            output.flush()
            os.fsync(output.fileno())
        receipts.append(
            dict(
                path=str(path.resolve()),
                slot=slot,
                begin=0,
                end=2048,
                pages=count,
                bytes=count * PAGE,
                sha256=digest.hexdigest(),
            )
        )
    receipt = dict(identity=identity, files=receipts, checks=checker.finish())
    write_json(directory / "snapshot.json", receipt)
    summary = dict(
        receipt_path=str((directory / "snapshot.json").resolve()),
        receipt_sha256=sha256(directory / "snapshot.json"),
        **receipt,
    )
    return SavedPages(receipts, 2048), summary


def verify_final_cache(saved, table):
    require(geometry(table) == 2048, "wrong final allocation")
    count = 0
    for slot in (0, 1):
        for key in range_keys(slot, 0, 2048):
            require(saved.get(key) == read_page(table, key), "cache changed after terminal snapshot/native shutdown")
            count += 1
    return dict(pages=count, bytes=count * PAGE, packed_bytes_equal=True)
