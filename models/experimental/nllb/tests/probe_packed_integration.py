# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Untimed portable native packed observer; no FP32 or speed acceptance.

Run with explicit checkpoint/config/tokenizer/device/output arguments. CPU fault
coverage remains in test_packed_integration.py. Controlled EOS logits below are
lifecycle coverage, never ordinary translation evidence.
"""

import argparse
import json
import os
from pathlib import Path
import sys


def observer_trace_id(trace_id):
    """Snapshot a native handle for JSON without changing ownership."""
    return None if trace_id is None else str(trace_id)


def atomic_record(directory, name, value):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def package_identity(root):
    """Verify the three runtime files against the sibling export inventory."""
    import hashlib
    import re

    manifest = json.loads((root / "PACKAGE_FILES.json").read_text())
    entries = manifest["files"]
    if not isinstance(entries, list):
        raise ValueError("PACKAGE_FILES.json files must be a list")
    pins = {}
    for name in ("tt/backend.py", "tt/trace_decode.py", "tt/runtime_setup.py"):
        matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("path") == name]
        if len(matches) != 1:
            raise ValueError(f"inventory requires exactly one entry for {name}")
        entry = matches[0]
        digest, size = entry.get("sha256"), entry.get("bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"invalid SHA256 for {name}")
        if type(size) is not int or size < 0:
            raise ValueError(f"invalid byte count for {name}")
        actual = (root / name).read_bytes()
        if len(actual) != size or hashlib.sha256(actual).hexdigest() != digest.lower():
            raise ValueError(f"inventory identity mismatch for {name}")
        pins[name] = digest.lower()
    return pins


def exercise(args):
    from models.experimental.nllb.tt import runtime_setup

    runtime_setup.configure_tracking()
    from models.experimental.nllb.tt import backend
    from models.experimental.nllb.tt import trace_decode
    from models.experimental.nllb.demo import translate
    from models.experimental.nllb.reference.envelope_regression import learned_compute_guard
    import numpy as np
    import torch
    import ttnn
    from unittest.mock import patch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads(Path(args.config).read_text())
    root = Path(__file__).resolve().parents[1]
    pins = package_identity(root)
    source = (root / "tt/trace_decode.py").read_text()
    assert source.count('#include "ttnn/kernel/dataflow/moreh_common.hpp"') == 2
    atomic_record(
        args.output,
        "identity",
        dict(
            source=pins,
            python=sys.executable,
            ttnn=ttnn.__file__,
            precision="bf16",
            timed=False,
            independent_fp32=False,
            cache=os.environ.get("TT_METAL_CACHE"),
        ),
    )
    texts = [
        "The students visited the library yesterday and borrowed several interesting books about the history of France.",
        "The sun provides light and heat for life on Earth.",
        "Hello world.",
        "We are learning to translate different languages.",
    ]
    _, ids, mask, target = translate.load_text_inputs(
        args.checkpoint, config, "eng_Latn", "fra_Latn", texts, tokenizer_directory=args.tokenizer_directory
    )
    runtime = runtime_setup.RuntimeOwner()
    models, owners, projections, packs = [], [], [], []
    label = ["public_b4"]
    original_init = trace_decode.PackedLMRequest.initialize
    original_finish = trace_decode.PackedLMRequest.finish
    original_pack = trace_decode.pack_last_rows
    original_linear = ttnn.linear
    original_bind = runtime_setup.RuntimeOwner.bind

    def bind(owner, model):
        result = original_bind(owner, model)
        if not any(model is m for m in models):
            models.append(model)
        return result

    def initialize(owner):
        original_init(owner)
        owners.append((owner, list(owner.rows), label[0]))
        assert owner.model.decode.__func__ is backend._CANONICAL_DECODE

    def finish(owner, **kw):
        handles = [observer_trace_id(x.trace_id) for x in [owner] + owner.rows]
        original_finish(owner, **kw)
        try:
            atomic_record(
                args.output,
                "cleanup_" + str(len(owners)),
                dict(
                    case=label[0],
                    before_handles=handles,
                    after_handle=observer_trace_id(owner.trace_id),
                    unresolved=owner.unresolved,
                    remaining_rows=len(owner.rows),
                    shared_replays=owner.replays,
                    warm_results=owner.warm_results,
                ),
            )
        except BaseException:
            if not kw.get("preserve_exception"):
                raise

    def pack(hidden, indices, output):
        result = original_pack(hidden, indices, output)
        packs.append(
            dict(case=label[0], batch=len(hidden), logical=list(output.shape), physical=list(output.padded_shape))
        )
        return result

    def linear(x, w, **kw):
        result = original_linear(x, w, **kw)
        if models and w is models[-1].lm_weight:
            projections.append(
                dict(
                    case=label[0],
                    logical=list(x.shape),
                    physical=list(x.padded_shape),
                    output=list(result.shape),
                    captured=bool(ttnn.is_trace_capture_active(runtime.device)),
                )
            )
        return result

    def clean(model):
        assert model.decode.__func__ is backend._CANONICAL_DECODE
        assert not model._trace_failures and not model._last_warmup_owners
        assert getattr(model, "_batch_trace", None) is None
        assert getattr(model, "_decode_trace", None) is None
        assert not ttnn.is_trace_capture_active(runtime.device)
        assert not runtime_setup.retained_owners()
        for owner, rows, _ in owners:
            assert not owner.unresolved and owner.trace_id is None
            assert not owner.rows and not owner.inputs and owner.output is None
            assert owner.selected is None and owner.input_ids is None and owner.attention_mask is None
            for row in rows:
                assert not row.unresolved and row.trace_id is None
                assert not row.inputs and row.output is None and not row.cross_kv and row.encoder is None

    def request(model, name, x, m):
        label[0] = name
        saved = x.copy(), m.copy()
        result = model.generate(x, m, target, 4)
        ttnn.synchronize_device(runtime.device)
        clean(model)
        for a, b in zip((x, m), saved):
            np.testing.assert_array_equal(a, b)
        atomic_record(args.output, name, dict(tokens=result.tolist(), inputs_unchanged=True, cleanup=True, timed=False))
        return result

    try:
        with runtime:
            device = runtime.open(args.device)
            assert ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()
            assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"

            # Protected donor raw-bit boundaries, including nonfinite payloads.
            def upload(x):
                return ttnn.from_torch(
                    x,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            def index(n):
                return ttnn.from_torch(
                    torch.tensor([[n]], dtype=torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            def physical(x):
                return ttnn.from_device(x).to_torch_with_padded_shape().contiguous().view(torch.int16)

            rng = np.random.default_rng(731)
            raw_cases = 0
            for batch in (2, 4):
                tables = [
                    torch.from_numpy(rng.integers(0, 65536, (1, 1, 64, 1024), dtype=np.uint16).view(np.int16)).view(
                        torch.bfloat16
                    )
                    for _ in range(batch)
                ]
                for table in tables:
                    table.view(torch.int16)[0, 0, :, :8] = torch.tensor(
                        [0, -32768, 1, -32767, 0x7F80, -128, 0x7FC1, 0x7FA1], dtype=torch.int16
                    )
                hidden = [upload(x) for x in tables]
                output = upload(torch.full((1, 1, batch, 1024), 3.0, dtype=torch.bfloat16))
                selected = upload(torch.ones(1, 1, 1, 1024))
                for length in (2, 31, 32, 33, 63, 64):
                    pos = length - 1
                    trace_decode.raw_selector(hidden[0], index(pos), selected)
                    bits = physical(selected)
                    assert torch.equal(bits[0, 0, 0], tables[0].view(torch.int16)[0, 0, pos])
                    assert bool((bits[0, 0, 1:] == 0).all())
                    for order in (list(range(batch)), list(reversed(range(batch)))):
                        positions = [pos if j % 2 == 0 else (length + 16) % 64 for j in range(batch)]
                        trace_decode.pack_last_rows([hidden[j] for j in order], [index(n) for n in positions], output)
                        bits = physical(output)
                        for slot, j in enumerate(order):
                            assert torch.equal(bits[0, 0, slot], tables[j].view(torch.int16)[0, 0, positions[slot]])
                        assert bool((bits[0, 0, batch:] == 0).all())
                        raw_cases += 1
                trace_decode.pack_last_rows(hidden, [index(64) for _ in hidden], output)
                assert bool((physical(output) == 0).all())
                for x, table in zip(hidden, tables):
                    assert torch.equal(physical(x), table.view(torch.int16))
            atomic_record(
                args.output,
                "raw_selector_pack",
                dict(
                    passed=True,
                    raw_bit_cases=raw_cases,
                    boundaries=[2, 31, 32, 33, 63, 64],
                    batch=[2, 4],
                    invalid_lanes_zero=True,
                    unused_rows_zero=True,
                    input_unchanged=True,
                    relative_include=True,
                ),
            )
            with (
                patch.object(runtime_setup.RuntimeOwner, "bind", bind),
                patch.object(trace_decode.PackedLMRequest, "initialize", initialize),
                patch.object(trace_decode.PackedLMRequest, "finish", finish),
                patch.object(trace_decode, "pack_last_rows", pack),
                patch.object(ttnn, "linear", linear),
                learned_compute_guard(torch, ttnn),
            ):
                public = translate.translate(
                    args.checkpoint,
                    config,
                    device,
                    "eng_Latn",
                    "fra_Latn",
                    texts,
                    max_new_tokens=4,
                    tokenizer_directory=args.tokenizer_directory,
                )
                model = models[-1]
                original_bind(runtime, model)
                clean(model)
                observed = [p for p in projections if p["case"] == "public_b4"]
                assert observed and packs and owners
                assert all(
                    p["logical"] == [1, 1, 4, model.dim] and p["physical"] == [1, 1, 32, model.dim] for p in observed
                )
                assert owners[0][0].replays > 0
                atomic_record(
                    args.output,
                    "public_b4",
                    dict(
                        tokens=public["token_ids"],
                        packed=packs[:],
                        projections=observed,
                        cleanup=True,
                        decode_canonical=True,
                        timed=False,
                    ),
                )
                singles = []
                count = len(owners)
                for i in range(4):
                    singles.append(request(model, "b1_" + str(i), ids[i : i + 1], mask[i : i + 1])[0])
                assert len(owners) == count
                expected = np.full((4, max(map(len, singles))), 1, dtype=np.int64)
                for i, row in enumerate(singles):
                    expected[i, : len(row)] = row
                np.testing.assert_array_equal(public["token_ids"], expected)
                x, m = ids.copy(), mask.copy()
                m[0, :2] = 0
                m[1, [2, 5]] = 0
                baseline = request(model, "mask_holes", x, m)
                padded = np.pad(x, ((0, 0), (0, 33)), constant_values=1)
                padded_mask = np.pad(m, ((0, 0), (0, 33)))
                np.testing.assert_array_equal(request(model, "pad_suffix", padded, padded_mask), baseline)
                padded[:, -33:] = 7
                np.testing.assert_array_equal(request(model, "masked_nonpad_suffix", padded, padded_mask), baseline)
                # Execute genuine native bodies, then control host selections only.
                read = trace_decode.PackedLMRequest.read
                active_rows = []

                def controlled(owner):
                    values = read(owner)
                    assert np.isfinite(values).all()
                    active_rows.append(list(owner.active))
                    values[:] = 0
                    for slot, row in enumerate(owner.active):
                        values[slot, 2 if row == 0 or owner.replays >= 2 else 7] = 1
                    return values

                with patch.object(trace_decode.PackedLMRequest, "read", controlled):
                    mixed = request(model, "controlled_eos", ids, mask)
                np.testing.assert_array_equal(mixed[0], [2, target, 2, 1, 1])
                for row in mixed[1:]:
                    np.testing.assert_array_equal(row, [2, target, 7, 7, 2])
                assert active_rows == [[0, 1, 2, 3], [1, 2, 3], [1, 2, 3]]
                atomic_record(
                    args.output,
                    "mixed_eos",
                    dict(active_rows=active_rows, controlled_host_logits=True, native_bodies=True, cleanup=True),
                )
                clean(model)
        assert runtime.closed and not runtime.cleanup_errors
        atomic_record(
            args.output,
            "complete",
            dict(
                passed=True, device_closed=True, retained=False, timed=False, independent_fp32=False, speed_claim=False
            ),
        )
    except BaseException as error:
        try:
            atomic_record(
                args.output,
                "failure",
                dict(
                    type=type(error).__name__,
                    error=str(error),
                    retained=bool(runtime_setup.retained_owners()),
                    device_closed=runtime.closed,
                    cleanup_errors=[str(e) for e in runtime.cleanup_errors],
                ),
            )
        except BaseException:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "tokenizer-directory", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", required=True, type=int)
    args = parser.parse_args(argv)
    from models.experimental.nllb.tt import runtime_setup

    runtime_setup.configure_tracking()
    exercise(args)


if __name__ == "__main__":
    main()
