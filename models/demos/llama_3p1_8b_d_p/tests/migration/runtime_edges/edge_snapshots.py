"""Stream immutable full-cache snapshots using the accepted same-owner page reader."""
import ast
import hashlib
import os
import time
from pathlib import Path

from edge_checks import EffectCheck, check_values, seed_value
from page_io import PAGE, SavedPages, geometry, range_keys, read_page
from support import sha256, write_json


def load_decoder(path, expected_hash, torch, np):
    if sha256(path) != expected_hash:
        raise ValueError("packed decoder source changed")
    source = Path(path).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_decode_bfp8_chunk")
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return lambda raw: namespace["_decode_bfp8_chunk"](raw, 128).tolist()


def capture(table, directory, identity, decode, *, previous=None, call=None):
    if geometry(table) != 2048:
        raise ValueError("runtime edges require exact2K table")
    if (previous is None) != (call is None):
        raise ValueError("baseline/call identity mismatch")
    directory = Path(directory)
    directory.mkdir(exist_ok=False)
    checker = EffectCheck(call, decode) if call is not None else None
    receipts = []
    seed_groups = 0
    for slot in (0, 1):
        path = directory / f"slot{slot}.bin"
        digest = hashlib.sha256()
        count = 0
        baseline_page = None
        with path.open("xb") as output:
            for key in range_keys(slot, 0, 2048):
                raw = read_page(table, key)
                if checker is None:
                    if key[3] == 0:
                        check_values(decode(raw), seed=seed_value(key[0], slot, key[2]))
                        if not any(raw):
                            raise ValueError("baseline seed is zero")
                        baseline_page = raw
                        seed_groups += 1
                    elif raw != baseline_page:
                        raise ValueError("seed differs within a layer/config/slot")
                else:
                    checker.accept(key, previous.get(key), raw)
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
    checks = (
        checker.finish() if checker is not None else dict(pages=65536, nonzero_pages=65536, seed_groups=seed_groups)
    )
    receipt = dict(identity=identity, files=receipts, checks=checks, capture_finished_ns=time.monotonic_ns())
    write_json(directory / "snapshot.json", receipt)
    summary = dict(
        receipt_path=str((directory / "snapshot.json").resolve()),
        receipt_sha256=sha256(directory / "snapshot.json"),
        **receipt,
    )
    return SavedPages(receipts, 2048), summary
