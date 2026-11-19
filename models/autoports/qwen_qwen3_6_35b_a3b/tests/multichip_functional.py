# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only logical traffic bound for the default TP4 batch-one decoder.

As in the single-chip bound, this excludes tile padding, redundant reads,
intermediates and collective traffic. It is an optimistic useful-byte bound,
not measured device traffic. Resident L1 output-norm gamma is excluded. Replicated work is charged to each rank; sharded
work is divided by four, except duplicated GQA K/V heads, which divide by two.
"""


def byte_model(layer, position):
    if layer not in (0, 3) or not isinstance(position, int) or position < 0:
        raise ValueError("expected layer 0/3 and a nonnegative decode position")
    linear = layer == 0
    bfp8 = 17 / 16
    weights = {
        # Linear: 2048 Q + 2048 K + 4096 V + 4096 Z + 32 A + 32 B,
        # partitioned TP4. Full: 1024 Q + 1024 gate + 256 K + 256 V
        # per rank; K/V duplication is intentional and cannot divide by TP4.
        "attention_projection": 2048 * (12352 / 4 if linear else 2560) * (2 if linear else bfp8),
        "attention_output": (4096 / 4) * 2048 * (2 if linear else bfp8),
        "eight_expert_gate_up_bfp8": 8 * 2048 * (1024 / 4) * bfp8,
        "eight_expert_down_bfp8": 8 * (512 / 4) * 2048 * bfp8,
        # Ordered linear router reads BF16 weights with FP32 products/logits.
        # Shared gate/up are TP4; scalar gate is replicated on each rank.
        "shared_gate_up_scalar_bfp8": 2048 * (1024 / 4 + 1) * bfp8,
        "shared_down_bfp8": (512 / 4) * 2048 * bfp8,
        "router_weights": 2048 * 256 * (2 if linear else 4),
        "layer_norm_weights_fp32": 2 * 2048 * 4,
    }
    if linear:
        weights.update(convolution_taps_fp32=4 * (8192 / 4) * 4, selected_decay_fp32=(32 / 4) * 4)
        cache = dict(
            recurrent_read_write_fp32=2 * (32 / 4) * 128 * 128 * 4, conv_read_write_bf16=2 * 3 * (8192 / 4) * 2
        )
    else:
        weights["qk_norm_weights_fp32"] = 2 * 256 * 4
        cache = dict(kv_reads_bf16=(position + 1) * 2 * 1 * 256 * 2, kv_write_bf16=2 * 1 * 256 * 2)
    per_rank = sum(weights.values()) + sum(cache.values())
    return dict(
        weight_bytes_per_rank=weights,
        cache_bytes_per_rank=cache,
        logical_bytes_per_rank=per_rank,
        rank_count=4,
        aggregate_logical_bytes=4 * per_rank,
        bandwidth_gb_s_per_rank=512,
        roofline_ms=per_rank / 512e6,
        scope="logical useful traffic on each TP4 rank; no padding, rereads, intermediates or CCL",
    )


def sequence_bound(layer):
    rows = [byte_model(layer, position) for position in range(128, 256)]
    return dict(
        positions=list(range(128, 256)),
        roofline_ms_per_token=sum(row["roofline_ms"] for row in rows) / 128,
        first_position=rows[0],
        last_position=rows[-1],
    )
