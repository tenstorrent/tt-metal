# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The Rust host runtime (``host_rs/``, the ``dsv41_host`` extension) from Python.

With the Rust host runtime on (the default; only ``DEEPSEEK_V41_RUST_HOST=0`` turns it off, and a missing
``dsv41_host`` extension is an error, not a fallback) the model hands its per-step host work to native threads that
never take the GIL:

* :class:`RustEngramLookup` is :class:`~.engram_lookup.EngramHostLookup` with the hashing, the per-session token
  histories and the table reads in Rust (bit-exact; the checkpoint-derived constants still come from
  ``NgramHashState``). Prefill and decode share it, as they share the Python one.
* :meth:`DeepSeekV41Model.start_host_runtime <.model.DeepSeekV41Model.start_host_runtime>` starts a ``DecodeHost``
  once the decode traces are captured: one trace-dispatch thread per pipeline stage, a packet writer, one Engram
  writer per layer and an output reader, on the model's own submeshes and H2D / D2H sockets (handed over by
  address, :func:`native_handle`). A socket can instead be given by the id it was exported under
  (``export_descriptor``), which ``DecodeHost`` connects to: that is how another process would attach.

``demo/server.py`` builds the extension at startup (:func:`build`); elsewhere, build it first (host-only)::

    models/experimental/deepseek_v41_flash/host_rs/build.sh
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import torch

import ttnn
from models.experimental.deepseek_v41_flash.tt.engram_lookup import EngramHostLookup

_EXTENSION = Path(__file__).resolve().parents[1] / "host_rs" / "dsv41_host.abi3.so"
_module = None


def enabled() -> bool:
    """Whether the Rust host runtime is used: the default, unless ``DEEPSEEK_V41_RUST_HOST`` turns it off (then the
    Python lookup and the traced-replay path serve). An unbuilt extension does not turn it off: :func:`module`
    raises instead."""
    return os.environ.get("DEEPSEEK_V41_RUST_HOST", "1") not in ("0", "", "false", "False")


def build() -> None:
    """Build the extension with ``host_rs/build.sh`` (incremental: a no-op when it is up to date), against this
    interpreter. Raises if the build fails or leaves no extension behind."""
    script = _EXTENSION.parent / "build.sh"
    try:
        subprocess.run([str(script)], env={**os.environ, "PYTHON": sys.executable}, check=True)
    except (OSError, subprocess.CalledProcessError) as err:
        raise RuntimeError(
            f"building the Rust host runtime ({script}) failed: {err}; "
            "--no-rust-host (or DEEPSEEK_V41_RUST_HOST=0) serves on the Python host runtime instead"
        ) from err
    if not _EXTENSION.is_file():
        raise RuntimeError(f"{script} succeeded but left no {_EXTENSION}")


def module():
    """The ``dsv41_host`` extension, loaded once, with ttnn's ``ttnn_hr_*`` C ABI bound."""
    global _module
    if _module is None:
        if not _EXTENSION.is_file():
            raise ImportError(f"{_EXTENSION} is not built: run {_EXTENSION.parent / 'build.sh'}")
        spec = importlib.util.spec_from_file_location("dsv41_host", _EXTENSION)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        loaded.load_ttnn(ttnn._ttnn.__file__)
        _module = loaded
    return _module


def native_handle(obj) -> int:
    """The address of the C++ ``MeshDevice`` / ``H2DSocket`` / ``D2HSocket`` behind ``obj``."""
    return ttnn._ttnn.hd_socket.native_handle(obj)


def _bf16(data: bytearray, shape) -> torch.Tensor:
    return torch.frombuffer(data, dtype=torch.bfloat16).reshape(shape)


class RustEngramLookup(EngramHostLookup):
    """:class:`EngramHostLookup` computed by ``dsv41_host.Engram``: the same results, bit for bit, with the GIL
    released and a prompt's positions spread over threads. ``sessions`` token histories live in Rust; the tables
    stay the torch tensors in ``tables``, which this object keeps alive."""

    def __init__(self, config, snapshot_dir, tables: dict, batch: int, max_seq: int, tokenizer=None, sessions: int = 1):
        super().__init__(config, snapshot_dir, tables, batch, max_seq, tokenizer, sessions=1)
        state, layout = self.hash, self.layout
        for layer_id, (weight, scale) in tables.items():
            if not (weight.is_contiguous() and scale.is_contiguous()):
                raise ValueError(f"layer {layer_id}'s Engram table must be contiguous")
        self.batch = batch
        self.rust = module().Engram(
            token_map=state.token_map.tolist(),
            vocab_size=self.vocab_size,
            pad_id=int(state.pad_id),
            max_ngram=layout.max_ngram_size,
            n_heads=layout.n_heads,
            head_dim=layout.head_dim,
            layout_layers=list(layout.layer_ids),
            multipliers=state.multipliers.tolist(),
            primes=state.primes.reshape(len(layout.layer_ids), -1).tolist(),
            offsets=state.offsets.tolist(),
            tables=[(layer_id, w.data_ptr(), s.data_ptr(), w.shape[0]) for layer_id, (w, s) in tables.items()],
            batch=batch,
            max_seq=max_seq,
            sessions=sessions,
        )
        self._order = self.rust.table_layers

    def use_session(self, slot: int) -> None:
        self.rust.use_session(slot)

    def export_history(self, slot: int, pos: int) -> torch.Tensor:
        return torch.tensor(self.rust.export_history(slot, pos), dtype=torch.int64).reshape(self.batch, pos)

    def import_history(self, slot: int, history: torch.Tensor) -> None:
        self.rust.import_history(slot, history.reshape(-1).tolist(), history.shape[1])

    def _hashes(self, slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        b, length = input_ids.shape
        flat = self.rust.hash(slot, input_ids.reshape(-1).tolist(), int(start_pos))
        return torch.tensor(flat, dtype=torch.int64).reshape(b, length, len(self.layout.layer_ids), -1)

    @torch.no_grad()
    def hash_session(self, slot: int, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        return self._hashes(slot, input_ids, start_pos)

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, start_pos: int) -> dict:
        b, length = input_ids.shape
        rows = {layer_id: torch.empty(b, length, self.rust.kin, dtype=torch.bfloat16) for layer_id in self._order}
        self.rust.rows_into(input_ids.reshape(-1).tolist(), int(start_pos), [r.data_ptr() for r in rows.values()])
        return rows

    @torch.no_grad()
    def lookup_history(self, history: torch.Tensor, input_ids: torch.Tensor, start_pos: int) -> dict:
        """:meth:`EngramHostLookup.lookup_history` in Rust, on a copy of ``history`` (a live session is not written)."""
        if history.ndim != 2 or input_ids.ndim != 2 or history.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"history {tuple(history.shape)} and ids {tuple(input_ids.shape)} must both be [batch, length]"
            )
        history = history.contiguous()
        input_ids = input_ids.contiguous()
        b, length = input_ids.shape
        rows = {layer_id: torch.empty(b, length, self.rust.kin, dtype=torch.bfloat16) for layer_id in self._order}
        self.rust.rows_from_history(
            history.data_ptr(),
            int(history.shape[1]),
            input_ids.data_ptr(),
            int(length),
            int(start_pos),
            [r.data_ptr() for r in rows.values()],
        )
        return rows

    def gather(self, layer_id: int, ids: torch.Tensor) -> torch.Tensor:
        data = self.rust.gather(layer_id, ids.reshape(-1).tolist())
        return _bf16(data, (*ids.shape, self.layout.head_dim))
