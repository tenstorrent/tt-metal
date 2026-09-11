# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What DFlash needs from a target model, and the two implementations of it.

:class:`SpeculativeTarget` is deliberately small — six methods. That list *is* the port spec: it is
everything the speculative loop asks of Qwen3.6-27B, with no HF cache objects or ttnn tensors
leaking into :mod:`.generate`.

* :class:`HFTarget` wraps the host ``Qwen3_5ForCausalLM`` — the golden reference.
* :class:`TtTarget` wraps the device :class:`~...tt.model.Qwen36Model` — same loop, real hardware.

Running the same loop against both is the point: the device's tokens must match the host's, and
where they do not, the six methods say exactly which one to look at.

The drafter always stays on host in both configurations, and it has neither an input embedding nor
an LM head of its own — it borrows the target's. :class:`TtTarget` splits those two: the embedding
is a host gather of a few rows (cheap), while the LM head runs **on the mesh**, where the target
already holds it.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class SpeculativeTarget(Protocol):
    """The target-model surface :func:`~.generate.dflash_generate` drives."""

    def reset(self) -> None:
        """Drop all cached state and start a new sequence at position 0."""

    def forward(self, ids: torch.Tensor, start: int, *, all_logits: bool = True):
        """Run ``ids`` (``[1, S]``) at absolute position ``start``, advancing cached state.

        Returns ``(logits, taps)``: logits ``[1, S, V]`` fp32 (or ``[1, 1, V]`` when
        ``all_logits=False`` — prefill only wants the last row), and taps ``[1, S, n*H]`` fp32, the
        target's residual stream at the drafter's tap layers concatenated in the drafter's order.
        """

    def snapshot(self) -> Any:
        """Capture whatever :meth:`restore` needs to undo the next :meth:`forward`."""

    def restore(self, snap: Any, length: int) -> None:
        """Roll cached state back to exactly ``length`` tokens, undoing a rejected block."""

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """The target's raw input embedding of ``ids`` — the drafter's noise input. ``[1, S, H]``."""

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """The target's output head applied to drafter hidden states. ``[1, S, V]`` fp32."""


class HFTarget:
    """Host ``Qwen3_5ForCausalLM``. Rolls GDN state back by snapshot + replay — see :mod:`.generate`."""

    #: `restore` only rewinds; the accepted prefix must be replayed to re-advance the GDN state.
    replays_after_rollback = True

    def __init__(self, model, tap_layer_ids):
        self.model = model
        self.tap_layer_ids = list(tap_layer_ids)
        self.cache = None

    @property
    def hidden_size(self) -> int:
        return self.model.config.get_text_config().hidden_size

    def reset(self) -> None:
        from transformers import DynamicCache

        self.cache = DynamicCache(config=self.model.config)

    @torch.inference_mode()
    def forward(self, ids, start, *, all_logits=True):
        from models.demos.blackhole.qwen36.reference.dflash.dflash import extract_context_feature

        ids = ids.to(self.model.device)
        position_ids = torch.arange(start, start + ids.shape[1], device=ids.device).unsqueeze(0)
        out = self.model(
            ids,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            output_hidden_states=True,
            **({} if all_logits else {"logits_to_keep": 1}),
        )
        taps = extract_context_feature(out.hidden_states, self.tap_layer_ids).float()
        return out.logits.float(), taps

    def snapshot(self):
        """Clone every linear-attention layer's conv + recurrent state.

        Only the GDN layers need cloning: attention KV truncates exactly under ``crop``, but
        ``crop`` is a documented no-op for ``linear_attention``. ~200 MB for the 27B, negligible
        against the forward it guards.
        """
        snapshot: dict[int, dict] = {}
        for idx, layer in enumerate(self.cache.layers):
            if not (hasattr(layer, "recurrent_states") and hasattr(layer, "conv_states")):
                continue
            conv, rec = layer.conv_states, layer.recurrent_states
            snapshot[idx] = {
                "conv": conv.clone() if isinstance(conv, torch.Tensor) else None,
                "recurrent": rec.clone() if isinstance(rec, torch.Tensor) else None,
                "has_previous_state": getattr(layer, "has_previous_state", False),
            }
        return snapshot

    def restore(self, snap, length):
        for idx, saved in snap.items():
            layer = self.cache.layers[idx]
            for attr, key in (("conv_states", "conv"), ("recurrent_states", "recurrent")):
                current, previous = getattr(layer, attr), saved[key]
                if not isinstance(current, torch.Tensor):
                    continue
                if previous is not None:
                    # copy_ rather than assignment, to keep the (cudagraph-static) address.
                    current.copy_(previous)
                else:
                    # Uninitialised when snapshotted but written since: the pre-block truth is "no
                    # state", so zero it rather than leaving this forward's values behind.
                    current.zero_()
            layer.has_previous_state = saved["has_previous_state"]
        # Qwen3.6-27B has no sliding layers, so this is exact for every attention layer.
        self.cache.crop(length)

    def embed(self, ids):
        weight = self.model.get_input_embeddings().weight
        return torch.nn.functional.embedding(ids.to(weight.device), weight)

    def lm_head(self, hidden):
        head = getattr(self.model, "lm_head", None) or self.model.get_output_embeddings()
        return head(hidden.to(head.weight.dtype)).float()


def load_target_embedding(path: str) -> torch.Tensor:
    """Pull just ``embed_tokens.weight`` out of the target checkpoint (~2.5 GB bf16).

    The drafter has no embedding of its own and runs on host, so it needs this as a torch tensor.
    Only the embedding: ``lm_head`` is NOT loaded here, because the device already holds it (see
    :meth:`TtTarget.lm_head`) and a second host copy would be 2.5 GB of pure waste.
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
    """Device :class:`Qwen36Model`, driven through its masked-bucket prefill path.

    Every forward — the prompt and each speculative block — is a masked-bucket prefill at an
    absolute ``chunk_start``. That path already carries GDN state across an offset and writes paged
    KV, but it has one hard constraint, measured on T3K (``test_tt_block_forward_continues_state``):

        ``chunk_start`` must be a multiple of the **bucket size**, not of the paged block size.

    PCC against a one-shot prefill is exactly 1.0 at offsets 128 and 256, and 0.16 / 0.27 / 0.25 at
    64, 192 and 320. The cause is the KV fill: ``paged_fill_cache`` writes the whole padded bucket
    starting at block ``chunk_start // 64``, so consecutive segments are spaced by the bucket, not
    by their ``valid_len``. There is no device primitive that writes a multi-token run at an
    arbitrary offset — ``paged_fill_cache`` starts at a block boundary and ``paged_update_cache``
    writes one token per batch element.

    Speculation advances by 1..16 tokens a step, so ``start`` is arbitrary and cannot be used as
    ``chunk_start`` directly. This class **anchors** instead: it keeps a 128-aligned ``anchor``
    with a GDN snapshot, and every forward re-runs the whole span ``[anchor, start + S)`` as one
    bucket at ``chunk_start=anchor``. Re-running up to 127 already-computed tokens is free — the
    bucket costs 128 positions either way.

    Anchoring also makes rollback disappear. Each forward restores GDN to the anchor and rewrites
    the entire ``[anchor, anchor + 128)`` KV span, so a rejected block is simply overwritten; there
    is nothing to undo and no replay to pay for. That is why :attr:`replays_after_rollback` is
    False here and True for :class:`HFTarget`.

    The one thing the caller must respect is :meth:`max_block`: a block may not cross the anchor's
    128-token boundary, so near one the loop drafts a shorter block.
    """

    #: Both the smallest masked bucket and the required chunk_start alignment.
    ANCHOR = 128
    #: Every forward recomputes from the anchor, so a rejected block needs no replay.
    replays_after_rollback = False

    def __init__(self, model, tap_layer_ids, page_table, *, checkpoint_path=None, block_size=64, device_taps=False):
        self.model = model
        self.tap_layer_ids = list(tap_layer_ids)
        self.page_table = page_table
        self.capacity = page_table.shape[1] * block_size
        # device_taps: keep the residual taps on the mesh for a ttnn drafter, instead of reading
        # them back to host. Five fewer PCIe round-trips per step.
        self.device_taps = device_taps
        model.set_residual_taps(self.tap_layer_ids, keep_on_device=device_taps)

        # Loaded lazily, and only the embedding: the LM head stays on the mesh.
        self._checkpoint_path = checkpoint_path
        self._embed = None
        self._anchor = 0
        self._anchor_gdn = None
        self._tokens = torch.zeros(1, self.capacity + self.ANCHOR, dtype=torch.long)

    @property
    def hidden_size(self) -> int:
        return self.model.args.dim

    def max_block(self, start: int) -> int:
        """Largest block that fits without crossing the anchor's bucket boundary."""
        return self.ANCHOR - (start % self.ANCHOR)

    def reset(self) -> None:
        self.model._reset_gdn_state_for_new_sequence()
        self._anchor = 0
        self._anchor_gdn = self.model.save_gdn_state()
        self._tokens.zero_()

    def _taps_cat(self, parts):
        """Join per-bucket taps. Host taps concat on the row axis; device taps concat per tap."""
        if len(parts) == 1:
            return parts[0]
        if not self.device_taps:
            return torch.cat(parts, dim=1)
        import ttnn

        return [
            ttnn.concat([p[j] for p in parts], dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for j in range(len(parts[0]))
        ]

    def taps_head(self, taps, rows: int):
        """The FIRST ``rows`` rows — the accepted prefix of a block's taps."""
        if not self.device_taps:
            return taps[:, :rows]
        import ttnn

        return [
            (
                t
                if t.shape[-2] == rows
                else ttnn.slice(t, (0, 0, 0, 0), (1, 1, rows, t.shape[-1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            )
            for t in taps
        ]

    def taps_tail(self, taps, rows: int):
        """The trailing ``rows`` rows of a tap set — a block forward's own positions inside a bucket."""
        if not self.device_taps:
            return taps[:, -rows:]
        import ttnn

        out = []
        for t in taps:
            have = t.shape[-2]
            out.append(
                t
                if have == rows
                else ttnn.slice(
                    t, (0, 0, have - rows, 0), (1, 1, have, t.shape[-1]), memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            )
        return out

    def embed_device(self, ids):
        """The target's input embedding of ``ids``, on the mesh, replicated at full hidden width.

        ``model.embd`` returns the embedding **fractured** on the hidden dim (TP), so it is gathered
        back to full width for the drafter, which keeps its hidden replicated. ~160 KB for a block.
        """
        import ttnn
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        seq = ids.shape[1]
        multi = self.model.num_devices > 1
        tok = ttnn.from_torch(
            ids.to(torch.int32).cpu(),
            dtype=ttnn.uint32,
            device=self.model.device,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.device)) if multi else {}),
        )
        x = self.model.embd(tok)
        ttnn.deallocate(tok)
        x = ttnn.reshape(x, (1, 1, seq, x.shape[-1]))
        if multi:
            x = tpc.tuned_vocab_all_gather(
                x,
                self.model.device,
                self.model.tt_ccl,
                dim=3,
                topology=ttnn.Topology.Linear,
                num_workers_per_link=2,
                chunks_per_sync=10,
            )
        return x

    def lm_head_device(self, hidden, *, keep_rows=None):
        """Draft logits for a **device** hidden tensor, via the mesh-resident LM head.

        The drafter borrows the target's head, and on this backend that head already lives on the
        mesh (``Qwen36Model.lm_head_weight``, vocab-sharded with a full-width replicated input —
        exactly the shape the drafter's hidden states have). ``keep_rows`` returns only the trailing
        rows, which is what the drafted slots are.

        :meth:`lm_head` is the same thing for a host tensor. Doing this projection on host instead
        cost 56% of a speculative run's wall clock, against a redundant 2.5 GB CPU copy of a weight
        the device was already holding.
        """
        import ttnn

        logits = self.model._lm_head(hidden)
        if self.model.num_devices > 1:
            host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.model.device, dim=0))[0]
        else:
            host = ttnn.to_torch(logits)
        ttnn.deallocate(logits)
        host = host.reshape(-1, host.shape[-1])[:, : self.model.vocab_size].float()
        return (host if keep_rows is None else host[-keep_rows:]).unsqueeze(0)

    def _run(self, lo: int, hi: int):
        """Run ``[lo, hi)`` from the anchor's GDN state as one bucket at ``chunk_start=lo``."""
        assert lo % self.ANCHOR == 0, f"chunk_start {lo} is not {self.ANCHOR}-aligned"
        length = hi - lo
        assert 1 <= length <= self.ANCHOR, f"span [{lo}, {hi}) does not fit one {self.ANCHOR} bucket"
        self.model.restore_gdn_state(self._anchor_gdn)
        # length == ANCHOR is a WHOLE bucket (prompt prefill), and gdn/tp.py::_normalize_valid_len
        # turns valid_len >= T into None -- masking skipped, different programs than the capture.
        # The trace serves partial buckets only; a full one falls back to eager.
        if getattr(self, "_traced_verify", False) and length < self.ANCHOR:
            # Trace replay instead of ~N eager dispatches. Same all-row logits, verified
            # bit-identical to the eager entry point in
            # tests/reference/test_dflash_target_trace_replay.py. One capture serves every
            # `length` below the bucket -- valid_len lives in the staged GDN mask's contents.
            token_buf = torch.zeros(1, self.ANCHOR, dtype=self._tokens.dtype)
            token_buf[:, :length] = self._tokens[:, lo:hi]
            logits = self.model.verify_traced(token_buf, length, lo, self.page_table, self.ANCHOR)
        else:
            logits = self.model.prefill_block_all_logits(
                self._tokens[:, lo:hi], self.page_table, actual_len=length, chunk_start=lo, bucket=self.ANCHOR
            )
        return logits, self.model.take_taps(length)

    def enable_traced_verify(self, warm_tokens=None):
        """Capture the verify trace and route :meth:`_run` through it.

        Off by default: the eager path stays the default until the throughput case is measured, and
        a capture costs trace memory. ``release_verify_trace`` on the model undoes it.
        """
        self.model.capture_verify_trace(self.page_table, self.ANCHOR, warm_tokens=warm_tokens)
        self._traced_verify = True

    def forward(self, ids, start, *, all_logits=True):
        """Run ``ids`` at absolute ``start``; ``all_logits`` is ignored (a bucket computes all rows).

        A long prompt is fed as consecutive whole buckets, each of which also re-anchors, so prompts
        are not limited to 128 tokens — only a single speculative block is, by :meth:`max_block`.
        """
        S = ids.shape[1]
        end = start + S
        assert end <= self.capacity, f"position {end} exceeds the {self.capacity}-token page table"
        assert self._anchor_gdn is not None, "call reset() before the first forward"
        assert start <= self._anchor + self.ANCHOR, (
            f"start {start} skips past the anchor at {self._anchor}; a block must not cross a bucket "
            "boundary — see max_block()"
        )
        self._tokens[:, start:end] = ids.to(self._tokens.dtype).cpu()

        first_anchor = self._anchor
        logit_parts, tap_parts = [], []
        # Whole buckets: all-real and already committed, so each one also re-anchors the snapshot.
        while end - self._anchor > self.ANCHOR:
            lg, tp = self._run(self._anchor, self._anchor + self.ANCHOR)
            logit_parts.append(lg)
            tap_parts.append(tp)
            self._anchor += self.ANCHOR
            # Reuse the snapshot's buffers rather than reallocating every bucket.
            self._anchor_gdn = self.model.save_gdn_state(into=self._anchor_gdn)
        # The partial tail bucket — where a speculative block always lands.
        lg, tp = self._run(self._anchor, end)
        logit_parts.append(lg)
        tap_parts.append(tp)

        logits = torch.cat(logit_parts, dim=1) if len(logit_parts) > 1 else logit_parts[0]
        return logits[:, -S:], self.taps_tail(self._taps_cat(tap_parts), S)

    def snapshot(self):
        """Nothing to capture: every forward recomputes from the anchor."""
        return None

    def restore(self, snap, length):
        """No-op — see the class docstring. The next forward rewrites the whole bucket."""

    def embed(self, ids):
        """Host gather from the target's embedding table — a handful of rows, so host is fine."""
        if self._embed is None:
            from models.demos.blackhole.qwen36.reference.dflash.loader import resolve_target_path

            self._embed = load_target_embedding(self._checkpoint_path or resolve_target_path())
        return torch.nn.functional.embedding(ids.cpu(), self._embed)

    def lm_head(self, hidden):
        """Draft logits for a **host** hidden tensor: upload, then :meth:`lm_head_device`.

        Uploading ~150 KB of hidden states and reading the logits back beats a 5120 x 248,320 CPU
        matmul by a wide margin, which is why the host drafter routes through here too.
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
