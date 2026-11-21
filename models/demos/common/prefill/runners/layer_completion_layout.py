# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Which layer completions each pipelined-prefill rank reports per chunk, and where its records sit in the
chunk's record sequence. A dense model reports once per layer; a hybrid stack only on KV-writing layers; MTP and
a DFlash drafter add rows past the last model layer, on the last rank. ttnn-free, so unit-testable."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerCompletionLayout:
    completions_per_chunk: int  # layers one chunk covers across all ranks; published in the ring header
    rank_first_record: int  # index of this rank's first record in a chunk's record sequence
    rank_record_count: int  # records this rank emits per chunk
    rank_layer_ids: list  # this rank's records as global layers, in emission order; [] when dense


def layer_completion_layout(
    layer_split: list, rank: int, *, kv_slot_layer_ids=None, extra_layers: int = 0
) -> LayerCompletionLayout:
    """`layer_split` is compute_layer_split's [(first_layer, count)] per rank; `kv_slot_layer_ids`
    the global layers that write KV (None: every layer); `extra_layers` the rows the last rank
    reports past the trunk (MTP levels plus drafter layers)."""
    first_layer_idx, num_my_layers = layer_split[rank]
    trunk_layers = sum(count for _, count in layer_split)
    extra_layer_ids = list(range(trunk_layers, trunk_layers + extra_layers))
    last_rank = rank == len(layer_split) - 1

    if kv_slot_layer_ids is None:
        records_per_rank = [count for _, count in layer_split]
        my_layer_ids = []
    else:
        ids = sorted(kv_slot_layer_ids)
        records_per_rank = [sum(1 for layer in ids if first <= layer < first + count) for first, count in layer_split]
        my_layer_ids = [layer for layer in ids if first_layer_idx <= layer < first_layer_idx + num_my_layers]
        if last_rank:
            my_layer_ids += extra_layer_ids

    rank_first_record = sum(records_per_rank[:rank])
    rank_record_count = records_per_rank[rank]
    if rank_record_count == 0:
        raise ValueError(
            f"rank {rank} holds layers [{first_layer_idx}, {first_layer_idx + num_my_layers}) and none of them "
            "writes a KV slab, so it would emit no layer-completion records; give every rank a KV-writing layer "
            "(PREFILL_PP_LAYER_COUNTS) or run without migration"
        )
    return LayerCompletionLayout(
        completions_per_chunk=sum(records_per_rank) + extra_layers,
        rank_first_record=rank_first_record,
        rank_record_count=rank_record_count + (extra_layers if last_rank else 0),
        rank_layer_ids=my_layer_ids,
    )
