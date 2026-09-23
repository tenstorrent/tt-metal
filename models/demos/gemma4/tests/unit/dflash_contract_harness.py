# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stubs and step helpers shared by the Gemma4 dFlash contract host tests.

The fixtures that build the stubbed environment live in ``conftest.py``; this
module holds the plain classes and functions those fixtures and the tests use,
so the unused-import hook has nothing to strip from the fixture module.
"""

from __future__ import annotations

import contextlib
from types import ModuleType, SimpleNamespace

import pytest
import torch


def make_expect_error():
    """The per-module ``expect_error`` fixture body.

    The repository fixture of that name lives in the root ``conftest.py``,
    which ``--confcutdir`` excludes for the host run, and a copy in this
    directory's ``conftest.py`` would shadow it for the device tests here.
    """

    @contextlib.contextmanager
    def expect_error_(error, message):
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: per-module expect_error body
            yield exc_info

    return expect_error_


def _unused(*args, **kwargs):
    raise AssertionError("A host contract test reached an unstubbed device operation")


def _padded_length(length):
    return 128 if length <= 128 else max(1024, 1 << (int(length) - 1).bit_length())


def _align_down(values, step=128):
    return [int(v) - int(v) % step for v in values]


def _module(name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    return module


class Tap:
    """A captured residual tap: ``[1, 1, rows, hidden]`` on the device."""

    def __init__(self, rows, tag):
        self.shape = (1, 1, int(rows), 8)
        self.tag = tag
        self.releases = 0

    def deallocate(self, force):
        assert force
        self.releases += 1


class Target:
    """``model[0]``: layers with attention configs, the tap hook and the output converter."""

    def __init__(self, events, ring=None, window=1024):
        self.events = events
        self.layers = [
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    config=SimpleNamespace(
                        cache_position_modulo=ring if kind == "sliding" else None, sliding_window=window
                    )
                )
            )
            for kind in ("sliding", "full")
        ]
        self.tap_layers = None
        self.keep_last = None
        self.taps = []
        self.convert_calls = []

    def dflash_capture_taps(self, layer_ids, buffers=None, keep_last=None):
        self.tap_layers = None if layer_ids is None else list(layer_ids)
        if layer_ids is not None:
            self.keep_last = keep_last
        self.taps = []
        self.events.append(("capture", self.tap_layers))

    def pop_dflash_taps(self):
        taps, self.taps = self.taps, []
        return taps

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        self.convert_calls.append((int(B), bool(is_tokens)))
        value = tt_out.value if isinstance(tt_out, DeviceResult) else tt_out
        if is_tokens:
            return value.reshape(-1)[:B]
        return value.reshape(-1, value.shape[-1])[:B].reshape(B, S, -1)


class DeviceResult:
    """What the plain decode leaves on the device before ``read_decode_output``."""

    def __init__(self, value):
        self.value = value


class Decoder:
    """Fused decoder stub: records commits, refreshes and replays; scripted outputs."""

    def __init__(self, events, widths=(1024,), P_v=6, cap=2048):
        self.events = events
        self.start = 0
        self.anchor = 0
        self.P_v = P_v
        self.K = P_v - 1
        self.V = P_v - 1
        self.cap = cap
        self.use_packed = True
        self.pv_sk = min(widths)
        self._pv_widths = {int(w): {"pv_sk": int(w), "trace": object()} for w in widths}
        self.page_table_torch = None
        self.script = []
        self.replays = 0

    def width_for(self, start):
        need = int(start) + self.P_v + 64
        fits = [w for w, r in self._pv_widths.items() if r["trace"] is not None and w >= need]
        return min(fits) if fits else None

    def select_width(self, start):
        self.events.append(("select_width", int(start)))
        return self.width_for(start)

    def refresh_page_tables(self, page_table):
        self.page_table_torch = page_table
        self.events.append(("refresh", page_table.reshape(-1).tolist()))

    def prefill_ingest(self, taps, n):
        self.events.append(("ingest", int(n), [tap.tag for tap in taps]))

    def reseed(self, anchor, start):
        self.anchor, self.start = int(anchor), int(start)
        self.events.append(("reseed", self.anchor, self.start))

    def contract_replay(self, first=False):
        self.replays += 1
        self.events.append(("replay", bool(first), self.start))
        if self.script:
            return self.script.pop(0)
        base = 100 * self.replays
        return [base + i for i in range(1, self.K + 1)], [base + 50 + i for i in range(self.P_v)]

    def contract_commit(self, produced, anchor):
        produced = max(1, min(int(produced), self.P_v))
        self.events.append(("commit", produced, int(anchor)))
        self.start += produced
        self.anchor = int(anchor)
        return self.start


def _tensor(values):
    return torch.tensor(values, dtype=torch.int32)


def _table(keys, width=4):
    """One block-table row per key: ``[key, key + 1, 0, ...]``."""
    rows = []
    for key in keys:
        row = [0] * width
        row[0], row[1] = int(key), int(key) + 1
        rows.append(row)
    return _tensor(rows)


def _prefill(model, prompt_len=2, slot=0, key=10, start=0, rows=1):
    """One prefill call as the runner makes it for ``rows`` prompts."""
    tokens = torch.full((rows, prompt_len), 7, dtype=torch.int32)
    kwargs = dict(
        tokens=tokens,
        prompt_lens=[prompt_len] * rows,
        empty_slots=[slot + r for r in range(rows)],
        page_table=_table([key + 2 * r for r in range(rows)]),
        kv_cache=model.kv_cache,
        warmup_prefill=False,
    )
    if start:
        kwargs["start_pos"] = [start] * rows
    return model.prefill_forward(**kwargs)


def _ordinary(model, anchors, positions, keys, result=None, sampling=True, **kwargs):
    """An ordinary decode step: ``[rows, 1]`` tokens, ``[rows]`` positions, -1 pads."""
    if result is not None:
        model.results.append(result)
    if sampling:
        kwargs.setdefault("sampling_params", object())
    return model.decode_forward(
        tokens=_tensor(anchors).reshape(-1, 1),
        start_pos=_tensor(positions),
        page_table=_table(keys),
        kv_cache=model.kv_cache,
        **kwargs,
    )


def _propose(model, committed, positions, counts=None):
    committed = _tensor(committed)
    positions = _tensor(positions)
    if counts is None:
        counts = [1] * int(committed.shape[0])
    return model.propose_draft_tokens(5, committed, positions, _tensor(counts))


def _verify(model, blocks, positions, valid, keys, result=None, sampling=True, **kwargs):
    if result is not None:
        model.results.append(result)
    if sampling:
        kwargs.setdefault("sampling_params", object())
    return model.decode_forward(
        tokens=_tensor(blocks),
        start_pos=_tensor(positions),
        spec_mode="argmax_ids",
        num_valid_drafts=_tensor(valid),
        accepted_counts=_tensor([1] * len(blocks)),
        page_table=_table(keys),
        kv_cache=model.kv_cache,
        **kwargs,
    )


def _start_solo(model, prompt_len=2, key=10):
    """Prefill alone, take the first ordinary step, then the first proposal."""
    _prefill(model, prompt_len=prompt_len, key=key)
    first = _ordinary(model, [3], [prompt_len], [key])
    assert first.tolist() == [150]  # posterior[0] of the first replay
    proposal = _propose(model, [[150, -1, -1, -1, -1, -1]], [[prompt_len + 1] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5]
    return proposal
