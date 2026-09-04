# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which Kimi-K3 layers are MLA, and which KV slot each one owns.

The single place the 24-of-93 map lives. Two numberings meet here and confusing them is silent:

* a **global** model layer index, 0..92, which names the weights in the checkpoint and decides
  whether a layer is MLA or KDA;
* a **rank-local** KV slot, which indexes the KV cache this rank allocated.

They are not the same and the difference is not cosmetic. `ttMLA._cache_batch_idx` computes
`cache_user_id * self.layer_num + cache_layer_idx` against a cache sized to *this rank's* layer
count, so a rank starting at layer 24 owns MLA layers 27, 31, ... whose slots must be 0, 1, ... —
not the 6, 7, ... that `KimiK3Config.mla_kv_slot()` returns. That classmethod is the model-wide
map; `adapters/kimi_k3.py`'s `NotImplementedError` points at it, and taking that hint literally
gives a pipeline-parallel rank a plausible, wrong, out-of-range slot. Use `kv_slot_of_local` here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KimiK3LayerSchedule:
    """One rank's slice of the hybrid layer schedule.

    Attributes:
        first_layer_idx: global index of this rank's first layer.
        num_layers: how many layers this rank holds.
        mla_layer_ids: global indices of every MLA layer in the whole model.
        kv_slot_of_local: one entry per local layer — its rank-local KV slot, or `None` on a KDA
            layer, which writes no KV slab at all.
        num_mla_layers: how many of this rank's layers are MLA. This is what `ttMLA` must be given
            as `layer_num`, and what the layer-ack channel must be configured with — not the model's
            93, and not the rank's layer count.
    """

    first_layer_idx: int
    num_layers: int
    mla_layer_ids: tuple[int, ...]
    kv_slot_of_local: tuple[int | None, ...]
    num_mla_layers: int

    @classmethod
    def build(cls, model_cfg: type, first_layer_idx: int = 0, num_layers: int | None = None) -> "KimiK3LayerSchedule":
        """Derive the schedule from `KimiK3Config` (or anything exposing `mla_layer_ids()`)."""
        mla_layer_ids = tuple(model_cfg.mla_layer_ids())
        if num_layers is None:
            num_layers = model_cfg.NUM_LAYERS - first_layer_idx
        if first_layer_idx < 0 or num_layers <= 0:
            raise ValueError(f"invalid slice: first_layer_idx={first_layer_idx}, num_layers={num_layers}")
        if first_layer_idx + num_layers > model_cfg.NUM_LAYERS:
            raise ValueError(
                f"slice [{first_layer_idx}, {first_layer_idx + num_layers}) runs past the model's "
                f"{model_cfg.NUM_LAYERS} layers"
            )

        slots: list[int | None] = []
        next_slot = 0
        for local_idx in range(num_layers):
            if first_layer_idx + local_idx in mla_layer_ids:
                slots.append(next_slot)
                next_slot += 1
            else:
                slots.append(None)

        return cls(
            first_layer_idx=first_layer_idx,
            num_layers=num_layers,
            mla_layer_ids=mla_layer_ids,
            kv_slot_of_local=tuple(slots),
            num_mla_layers=next_slot,
        )

    def is_mla(self, global_layer_idx: int) -> bool:
        """Whether a *global* model layer index is a full-attention layer."""
        return global_layer_idx in self.mla_layer_ids

    def local_is_mla(self, local_idx: int) -> bool:
        return self.kv_slot_of_local[local_idx] is not None

    def global_index(self, local_idx: int) -> int:
        return self.first_layer_idx + local_idx

    def kv_slot(self, local_idx: int) -> int | None:
        """This rank's KV slot for a local layer, or `None` if the layer writes no KV."""
        return self.kv_slot_of_local[local_idx]

    def validate_kv_only_last_layer(self) -> None:
        """A `kv_only` last layer must be an MLA layer.

        `kv_only_last_layer` builds the final block as attention-plus-KV-write only, to save the FFN
        on a chunk whose output nobody reads. On K3 that only makes sense if the layer writes KV at
        all: asking a KDA layer to be kv_only produces a block that computes a recurrence, discards
        it, and writes nothing — pure cost. `PREFILL_KV_ONLY_LAST_LAYER` defaults to on in serving,
        so this is a real configuration, not a hypothetical.
        """
        if self.num_layers and not self.local_is_mla(self.num_layers - 1):
            raise ValueError(
                f"kv_only_last_layer needs an MLA last layer, but layer "
                f"{self.global_index(self.num_layers - 1)} is KDA; MLA layers are {self.mla_layer_ids}"
            )
