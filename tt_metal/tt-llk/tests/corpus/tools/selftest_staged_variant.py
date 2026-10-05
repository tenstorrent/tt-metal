#!/usr/bin/env python3
"""Host-only regression for cross-path compile-consumer variant selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile


HELPER = (
    Path(__file__).resolve().parents[2]
    / "python_tests/helpers/staged_variant.py"
)
spec = importlib.util.spec_from_file_location("staged_variant", HELPER)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def main() -> int:
    quietbox_id = "c" * 64
    exabox_recomputed_id = "2" * 64
    assert module.resolve_staged_variant(
        exabox_recomputed_id, quietbox_id, compile_consumer=True
    ) == quietbox_id
    assert module.resolve_staged_variant(
        exabox_recomputed_id, None, compile_consumer=True
    ) == exabox_recomputed_id

    for explicit, consumer in ((quietbox_id, False), ("not-a-variant", True)):
        try:
            module.resolve_staged_variant(
                exabox_recomputed_id, explicit, compile_consumer=consumer
            )
            raise AssertionError("invalid staged variant override was accepted")
        except RuntimeError:
            pass
    with tempfile.TemporaryDirectory() as temporary:
        elf_dir = Path(temporary)
        for component in ("unpack", "math", "pack"):
            (elf_dir / f"{component}.elf").write_bytes(component.encode())
        module.require_complete_staged_variant(
            elf_dir, ["unpack", "math", "pack"], quietbox_id
        )
        (elf_dir / "math.elf").unlink()
        try:
            module.require_complete_staged_variant(
                elf_dir, ["unpack", "math", "pack"], quietbox_id
            )
            raise AssertionError("incomplete staged variant was accepted")
        except RuntimeError as error:
            assert "math" in str(error)
    print("PASS explicit producer variant overrides cross-host hash only in consumer mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
