# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""ACK space: the per-chunk records a pipelined-prefill rank emits, as both layer-completion
transports count them. A dense model acks once per layer, so ACK space is layer space; a hybrid
stack acks only on KV-writing layers; MTP and a DFlash drafter ack extra rows past the last model
layer, on the last rank. Kept ttnn-free so the derivation is unit-testable."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AckSpaceLayout:
    num_ack_layers: int  # records one chunk produces across all ranks (the seq stride)
    ack_first_idx: int  # this rank's first ACK index
    ack_local_count: int  # records this rank emits per chunk
    ack_idx_of_layer: dict | None  # global layer -> ACK index; None when the two coincide (dense)
    ack_layer_ids: list  # this rank's records as global layers, in emission order; [] when dense


def ack_space_layout(
    layer_split: list, rank: int, *, kv_slot_layer_ids=None, extra_ack_layers: int = 0
) -> AckSpaceLayout:
    """`layer_split` is compute_layer_split's [(first_layer, count)] per rank; `kv_slot_layer_ids`
    the global layers that write KV (None: every layer); `extra_ack_layers` the rows the last rank
    acks past the trunk (MTP levels plus drafter layers)."""
    first_layer_idx, num_my_layers = layer_split[rank]
    trunk_layers = sum(count for _, count in layer_split)
    extra_layers = list(range(trunk_layers, trunk_layers + extra_ack_layers))
    last_rank = rank == len(layer_split) - 1

    if kv_slot_layer_ids is None:
        acks_per_rank = [count for _, count in layer_split]
        ack_idx_of_layer = None
        my_layer_ids = []
    else:
        ids = sorted(kv_slot_layer_ids)
        acks_per_rank = [sum(1 for layer in ids if first <= layer < first + count) for first, count in layer_split]
        ack_idx_of_layer = {layer: idx for idx, layer in enumerate(ids + extra_layers)}
        my_layer_ids = [layer for layer in ids if first_layer_idx <= layer < first_layer_idx + num_my_layers]
        if last_rank:
            my_layer_ids += extra_layers

    ack_first_idx = sum(acks_per_rank[:rank])
    ack_local_count = acks_per_rank[rank]
    if ack_local_count == 0:
        raise ValueError(
            f"rank {rank} holds layers [{first_layer_idx}, {first_layer_idx + num_my_layers}) and none of them "
            "writes a KV slab, so it would emit no layer-completion records; give every rank a KV-writing layer "
            "(PREFILL_PP_LAYER_COUNTS) or run without migration"
        )
    return AckSpaceLayout(
        num_ack_layers=sum(acks_per_rank) + extra_ack_layers,
        ack_first_idx=ack_first_idx,
        ack_local_count=ack_local_count + (extra_ack_layers if last_rank else 0),
        ack_idx_of_layer=ack_idx_of_layer,
        ack_layer_ids=my_layer_ids,
    )
