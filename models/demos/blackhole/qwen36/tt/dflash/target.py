# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The device DFlash target: :class:`TtTarget` wraps :class:`~...tt.model.Qwen36Model`.

It implements the :class:`~...reference.dflash.targets.SpeculativeTarget` protocol that
:func:`~...reference.dflash.generate.dflash_generate` drives, so the same loop runs against the
host reference (:class:`~...reference.dflash.targets.HFTarget`) and against the device.

A speculative block of ``T`` slots (1 confirmed token + drafts) runs through
``Qwen36Model.verify_traced``:

* full-attention layers run in decode mode, one pseudo-user per slot, each writing its own KV row at
  its exact position and attending causally up to it (``spec_verify_mode``);
* Gated DeltaNet layers run the fused recurrent op over the ``T`` rows and stash the state after
  every row;
* the residual stream after each tap layer is cloned inside the trace, so the drafter's taps come
  out of the same replay as the logits;
* the LM head and argmax run in the trace, so only ``T`` token ids cross PCIe.

A step therefore costs the same at any position. Rolling back a rejected tail is :meth:`commit`:
KV needs nothing (rows past the accepted prefix are never read and are rewritten by the next
block), and each GDN layer points its durable state at the stashed slot of the last accepted token.

The drafter has neither an input embedding nor an LM head of its own; it borrows the target's.
:class:`TtTarget` serves the embedding either as a host gather of a few rows or on the mesh, and runs
the LM head on the mesh, where the target already holds it.
"""

from __future__ import annotations

import os

import torch
from loguru import logger


def load_target_embedding(path: str) -> torch.Tensor:
    """Pull just ``embed_tokens.weight`` out of the target checkpoint (~2.5 GB bf16).

    The host drafter has no embedding of its own, so it needs this as a torch tensor. ``lm_head`` is
    not loaded here because the device already holds it (see :meth:`TtTarget.lm_head`).
    """
    import json

    from safetensors import safe_open

    index = os.path.join(path, "model.safetensors.index.json")
    assert os.path.exists(index), f"no safetensors index at {index}"
    weight_map = json.load(open(index))["weight_map"]
    key = next(k for k in weight_map if k.endswith("embed_tokens.weight"))
    with safe_open(os.path.join(path, weight_map[key]), framework="pt") as f:
        return f.get_tensor(key)


class TtTarget:
    """Device :class:`Qwen36Model` behind the speculative loop's target protocol."""

    #: :meth:`commit` selects the GDN slot; the loop must not restore or replay.
    commits_slot = True
    replays_after_rollback = False
    #: Greedy verification reads only the argmax, which the verify trace already reduces on device.
    device_posterior = True

    def __init__(self, model, tap_layer_ids, page_table, *, verify_len=16, checkpoint_path=None, block_size=64):
        assert model.num_devices > 1, "the speculative verify is the tensor-parallel path"
        self.model = model
        self.tap_layer_ids = list(tap_layer_ids)
        self.page_table = page_table
        self.capacity = page_table.shape[1] * block_size
        self.verify_len = int(verify_len)
        model.set_residual_taps(self.tap_layer_ids, keep_on_device=True)
        self._checkpoint_path = checkpoint_path  # loaded lazily; only the embedding table
        self._embed = None
        self._commit_traced = False
        #: Rows accepted by the last verify, while its tap buffers are still the newest; else None.
        self.last_produced = None

    @property
    def hidden_size(self) -> int:
        return self.model.args.dim

    def max_block(self, start: int) -> int:
        return self.verify_len

    def reset(self) -> None:
        self.model._reset_gdn_state_for_new_sequence()
        self.last_produced = None

    def _taps_cat(self, parts):
        """Join consecutive tap sets: device taps concatenate per tap along the row axis."""
        if len(parts) == 1:
            return parts[0]
        import ttnn

        return [
            ttnn.concat([p[j] for p in parts], dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for j in range(len(parts[0]))
        ]

    def taps_head(self, taps, rows: int):
        """The first ``rows`` rows — the accepted prefix of a block's taps."""
        import ttnn

        return [
            (
                t
                if t.shape[-2] == rows
                else ttnn.slice(t, (0, 0, 0, 0), (1, 1, rows, t.shape[-1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            for t in taps
        ]

    def _set_slot_capture(self, on: bool) -> None:
        for dn in getattr(self.model, "_vfy_gdn", ()):
            dn._capture_slots = on

    def _prepare(self) -> None:
        if getattr(self.model, "_vfy_token_buf", None) is None:
            self.model.prepare_verify(self.page_table, self.verify_len, decode_cfg=True)

    def enable_traced_verify(self):
        """Capture the verify trace. Run one full generation first so every other program is compiled."""
        self._prepare()
        self.model.capture_verify_trace(
            self.page_table, self.verify_len, decode_cfg=True, warm_start=0, commit_warmup=True
        )
        self._commit_traced = bool(self.model.capture_commit_traces())
        logger.info(
            f"[dflash] verify trace captured (T={self.verify_len}), commit={'traced' if self._commit_traced else 'eager'}"
        )

    def release_verify_trace(self) -> None:
        import ttnn

        self.model.release_commit_traces()
        self._commit_traced = False
        if getattr(self.model, "_vfy_trace_id", None) is not None:
            ttnn.release_trace(self.model.device, self.model._vfy_trace_id)
            self.model._vfy_trace_id = None

    def _prefill(self, ids):
        """Prompt through the chunked spec prefill; taps are collected chunk by chunk on device."""
        import ttnn

        model = self.model
        S = ids.shape[1]
        assert S <= self.capacity, f"prompt of {S} tokens exceeds the {self.capacity}-token page table"
        parts = []

        def _on_chunk(hidden, chunk_start, valid_len):
            parts.append(model.take_taps(valid_len))

        self.last_produced = None
        self._set_slot_capture(False)  # the eager prompt must not write the trace's slot buffers
        logits_dev = model.prefill_for_spec(ids.to(torch.int32).cpu(), self.page_table, S, _on_chunk)
        self._set_slot_capture(True)
        # A traced replay cannot sync the conv-window mirror itself; the prompt left it stale.
        for dn in getattr(model, "_vfy_gdn", ()):
            dn.sync_conv_win()
        host = ttnn.to_torch(ttnn.get_device_tensors(logits_dev)[0])
        ttnn.deallocate(logits_dev)
        logits = host.reshape(1, 1, -1)[..., : model.vocab_size].float()
        return logits, self._taps_cat(parts)

    def forward(self, ids, start, *, all_logits=True, posterior=False):
        """Prompt (``start == 0``) or one verify block. Returns ``(logits | ids, device taps)``.

        A block shorter than ``verify_len`` is padded; its extra rows are computed and ignored.
        With ``posterior`` the first element is the argmax ids ``[1, S]`` instead of logits.
        """
        if start == 0:
            return self._prefill(ids)
        S = ids.shape[1]
        T = self.verify_len
        assert S <= T, f"block of {S} exceeds the {T}-slot verify"
        assert start + T <= self.capacity, f"block at {start} overruns the {self.capacity}-token page table"
        self._prepare()
        tokens = [int(t) for t in ids[0].tolist()] + [0] * (T - S)
        lt, _, vids = self.model.verify_traced(
            tokens, start, read_logits=not posterior, clone_rows=False, page_table=self.page_table
        )
        taps = self.model.verify_taps()
        if posterior:
            return torch.tensor([vids[:S]], dtype=torch.long), taps
        return lt[:S].unsqueeze(0), taps

    def commit(self, produced: int) -> None:
        """Make the durable GDN state the one after the first ``produced`` slots of the last block."""
        self.last_produced = produced
        mi = produced - 1
        if self._commit_traced and mi < self.verify_len - 1 and self.model.replay_commit_trace(mi):
            for dn in self.model._vfy_gdn:
                dn.commit_verify_slot_host(mi)
            return
        for dn in self.model._vfy_gdn:
            dn.commit_verify_slot(mi)

    def snapshot(self):
        """Nothing to capture: :meth:`commit` selects among the states the verify already stashed."""
        return None

    def restore(self, snap, length):
        raise AssertionError("TtTarget rolls back with commit(); the loop must not call restore")

    def embed_device(self, ids):
        """The target's input embedding of ``ids``, on the mesh, replicated at full hidden width.

        ``model.embd`` returns the embedding fractured on the hidden dim (TP), so it is gathered back
        to full width for the drafter, which keeps its hidden replicated.
        """
        import ttnn

        seq = ids.shape[1]
        multi = self.model.num_devices > 1
        tok = ttnn.from_torch(
            ids.to(torch.int32).cpu(),
            dtype=ttnn.uint32,
            device=self.model.device,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.device)) if multi else {}),
        )
        return self._embed_tokens(tok, seq, own=True)

    def _embed_tokens(self, tok, seq, *, own):
        import ttnn
        from models.tt_transformers.tt.ccl import tt_all_gather

        multi = self.model.num_devices > 1
        x = self.model.embd(tok)
        if own:
            ttnn.deallocate(tok)
        x = ttnn.reshape(x, (1, 1, seq, x.shape[-1]))
        if multi:
            x = tt_all_gather(
                x,
                self.model.device,
                self.model.tt_ccl,
                cluster_axis=None,
                dim=3,
                topology=ttnn.Topology.Linear,
                num_workers_per_link=2,
                chunks_per_sync=10,
            )
        return x

    def draft_ids_device(self, hidden, *, keep_rows):
        """Greedy draft token ids, argmaxed on device -- ``keep_rows`` ints cross PCIe, not logits.

        Under greedy decoding the host only ever argmaxes the draft logits, so the argmax runs on
        the device and the transfer is a handful of uint32s instead of ``[rows, vocab]`` logits.

        This is one op, not a reduction tree: ``Qwen36Model._lm_head`` all-gathers its vocab shards
        before returning, so every device holds the full-width row and device 0's argmax is the
        global one.

        Sampling still needs the distribution, so ``temperature > 0`` keeps :meth:`lm_head_device`.
        """
        import ttnn

        logits = self.model._lm_head(hidden)
        rows, width = logits.shape[-2], logits.shape[-1]
        vocab = self.model.vocab_size
        owned = []

        keep = logits
        if keep_rows is not None and keep_rows < rows:
            keep = ttnn.slice(keep, (0, 0, rows - keep_rows, 0), (1, 1, rows, width))
            owned.append(keep)
        if width > vocab:
            # Must trim before the argmax: the head's output is padded past vocab_size, and an
            # argmax over the padded columns can silently return an index that is not a token.
            keep = ttnn.slice(keep, (0, 0, 0, 0), (1, 1, keep.shape[-2], vocab))
            owned.append(keep)

        # Argmax in ROW_MAJOR, not TILE: ttnn.argmax over a vocab-wide row is far slower in TILE
        # layout, even counting the untilize.
        rm = ttnn.to_layout(keep, ttnn.ROW_MAJOR_LAYOUT)
        ids = ttnn.argmax(rm, dim=-1)
        ttnn.deallocate(rm)
        if self.model.num_devices > 1:
            host = ttnn.to_torch(ttnn.get_device_tensors(ids)[0])
        else:
            host = ttnn.to_torch(ids)
        # Never test membership with `in` here: `x in owned` calls `==`, which on ttnn.Tensor is an
        # elementwise device op and fails against already-freed tensors. `owned` only ever holds
        # slices this function made, so `logits` is not in it by construction.
        for t in owned:
            ttnn.deallocate(t)
        ttnn.deallocate(logits)
        ttnn.deallocate(ids)
        return host.reshape(1, -1)[:, -keep_rows:].long()

    def draft_ids_traced(self, hidden):
        """Greedy draft ids for every row of ``hidden``, left on device (``[1, 1, rows]`` uint32).

        The trace-safe form of :meth:`draft_ids_device`: no readback, no host-dependent slicing."""
        import ttnn

        logits = self.model._lm_head(hidden)
        vocab = self.model.vocab_size
        if logits.shape[-1] > vocab:
            # The head's output is padded past vocab_size; an argmax over the pad can return a
            # non-token index.
            trimmed = ttnn.slice(logits, (0, 0, 0, 0), (1, 1, logits.shape[-2], vocab))
            ttnn.deallocate(logits)
            logits = trimmed
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(logits)
        ids = ttnn.argmax(rm, dim=-1)
        ttnn.deallocate(rm)
        return ids

    def lm_head_device(self, hidden, *, keep_rows=None):
        """Draft logits for a device hidden tensor, via the mesh-resident LM head.

        The drafter borrows the target's head, and on this backend that head already lives on the
        mesh (``Qwen36Model.lm_head_weight``, vocab-sharded with a full-width replicated input —
        exactly the shape the drafter's hidden states have). ``keep_rows`` returns only the trailing
        rows, which is what the drafted slots are.

        :meth:`lm_head` is the same thing for a host tensor.
        """
        import ttnn

        logits = self.model._lm_head(hidden)
        # Slice on device, then read one device's copy (as Qwen36Model._read_verify_logits does):
        # the LM head all-gathers its vocab shards, so every device already holds the full row.
        # `keep_rows` trims the row axis: slot 0 of a block is the confirmed anchor, so only the
        # trailing q_len-1 rows are ever read.
        keep, sliced = logits, None
        rows = logits.shape[-2]
        if keep_rows is not None and keep_rows < rows:
            sliced = ttnn.slice(logits, (0, 0, rows - keep_rows, 0), (1, 1, rows, logits.shape[-1]))
            keep = sliced
        if self.model.num_devices > 1:
            host = ttnn.to_torch(ttnn.get_device_tensors(keep)[0])
        else:
            host = ttnn.to_torch(keep)
        if sliced is not None:
            ttnn.deallocate(sliced)
        ttnn.deallocate(logits)
        host = host.reshape(-1, host.shape[-1])[:, : self.model.vocab_size].float()
        # Already trimmed on device when keep_rows applied; the tail slice is now a no-op guard.
        return (host if keep_rows is None else host[-keep_rows:]).unsqueeze(0)

    def embed(self, ids):
        """Host gather from the target's embedding table — a handful of rows, so host is fine."""
        if self._embed is None:
            from models.demos.blackhole.qwen36.tt.dflash.config import resolve_target_path

            self._embed = load_target_embedding(self._checkpoint_path or resolve_target_path())
        return torch.nn.functional.embedding(ids.cpu(), self._embed)

    def lm_head(self, hidden):
        """Draft logits for a host hidden tensor: upload, then :meth:`lm_head_device`.

        Uploading the hidden states and reading the logits back avoids a 5120 x vocab CPU matmul,
        which is why the host drafter routes through here too.
        """
        import ttnn

        seq = hidden.shape[-2]
        multi = self.model.num_devices > 1
        x = ttnn.from_torch(
            hidden.reshape(1, 1, seq, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.model.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.device)) if multi else {}),
        )
        out = self.lm_head_device(x)
        ttnn.deallocate(x)
        return out
