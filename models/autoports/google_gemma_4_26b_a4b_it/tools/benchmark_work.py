# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Source-derived work estimates, not measurements; imports no device libraries.

See doc/benchmark/WORK_ACCOUNTING.md. Decoder rows execute separately; the
vocabulary projection executes once per model invocation. All totals aggregate
four ASICs. The caller must supply *actual* invocation membership and positions.
"""

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TILE_BYTES = {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048, "float32": 4096}
SOURCES = (
    "tt/model.py",
    "tt/generator.py",
    "tt/generator_vllm.py",
    "tt/multichip_decoder.py",
    "tt/optimized_decoder.py",
    "tests/config.json",
    "doc/datatype_sweep/selected_precision_config.json",
)


def payload(rows, cols, dtype):
    """Physical tiled payload, including BFP shared exponents and tile padding."""
    if type(rows) is not int or type(cols) is not int or min(rows, cols) <= 0:
        raise ValueError("positive integer matrix dimensions required")
    return math.ceil(rows / 32) * math.ceil(cols / 32) * TILE_BYTES[dtype]


def kv_read_tokens(position, sliding):
    if type(position) is not int or not 0 <= position < 262144:
        raise ValueError("position outside full context")
    # NativePagedAttention uses fp32_dest_acc_en and dynamic chunks: max 4 tiles.
    chunk = 32 * min(4, 1 << ((position // 32 + 1) - 1).bit_length())
    end = position + 1
    start = max(0, end - 1024) if sliding else 0
    return math.ceil(end / chunk) * chunk - start // chunk * chunk


class WorkAccounting:
    def __init__(self, config=None, precision=None, *, mesh=(1, 4), layers=30, context=262144):
        expected = json.loads((ROOT / "tests/config.json").read_text())["text_config"]
        selected = json.loads((ROOT / "doc/datatype_sweep/selected_precision_config.json").read_text())
        self.config = expected if config is None else config
        self.precision = selected if precision is None else precision
        if self.config != expected or self.precision != selected:
            raise ValueError("Accounting supports only the checked-in architecture and selected precision policy")
        if tuple(mesh) != (1, 4) or layers != 30 or context != 262144:
            raise ValueError("Accounting requires full 30-layer TP4/EP4 context262144 model")
        if (
            expected["hidden_size"],
            expected["num_attention_heads"],
            expected["top_k_experts"],
            expected["num_experts"],
            expected["moe_intermediate_size"],
            expected["intermediate_size"],
        ) != (2816, 16, 8, 128, 704, 2112):
            raise ValueError("Architecture changed; re-audit accounting")
        self.source_sha256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}

    def _policy(self, index, kind):
        return {**self.precision["layer_types"][kind], **self.precision["layer_overrides"].get(str(index), {})}

    def prefill(self, sequence_lengths):
        """Useful FLOPs for initial prefill(s), with one last-token head per request.

        Does not support cached continuation or return_all_logits. Padding,
        replicated K/V projections and union-expert work are not useful FLOPs.
        """
        lengths = list(sequence_lengths)
        if not lengths or any(type(s) is not int or not 1 <= s <= 262144 for s in lengths):
            raise ValueError("nonempty positive sequence lengths within context required")
        terms = defaultdict(int)
        h, q, shared, expert, topk = 2816, 16, 2112, 704, 8
        for s in lengths:
            for kind in self.config["layer_types"]:
                sliding = kind == "sliding_attention"
                d, kv = (256, 8) if sliding else (512, 2)
                w = min(s, 1024)
                pairs = s * w - w * (w - 1) // 2 if sliding else s * (s + 1) // 2
                # Global K=V uses one useful projection, although runtime stores both.
                terms["qkv_output"] += 2 * s * h * (2 * q * d + kv * d * (2 if sliding else 1))
                terms["shared_mlp"] += 6 * s * h * shared
                terms["active_top8_experts"] += 6 * s * h * expert * topk
                terms["router"] += 2 * s * h * 128
                terms["causal_qk_pv"] += 4 * q * d * pairs
            terms["last_token_lm_head"] += 2 * h * 262144
        return {
            "useful_flops": sum(terms.values()),
            "terms": dict(terms),
            "request_count": len(lengths),
            "input_tokens": sum(lengths),
            "scope": "full_model_useful_matmul_flops_all_four_asics",
        }

    def decode(self, positions, *, batch_slots=None):
        """Estimated aggregate DRAM bytes for ONE actual model invocation.

        positions contains only executed active rows. batch_slots is model head
        logical batch (including zero rows when active_slots omits slots), not
        server capacity. Decoder sparse weights are reread for each active row.
        """
        positions = list(positions)
        batch_slots = len(positions) if batch_slots is None else batch_slots
        if not positions or type(batch_slots) is not int or not len(positions) <= batch_slots <= 32:
            raise ValueError("1..32 actual model slots with nonempty active positions required")
        for pos in positions:
            kv_read_tokens(pos, False)
        terms = defaultdict(int)
        h = 2816
        for index, kind in enumerate(self.config["layer_types"]):
            p = self._policy(index, kind)
            sliding = kind == "sliding_attention"
            d, local_kv = (256, 2) if sliding else (512, 1)
            for pos in positions:
                # Actual TP geometry duplicates global KV heads on pairs of ASICs.
                terms["qkv_output_weights"] += 4 * (
                    payload(h, (4 + 2 * local_kv) * d, p["qkv_weight_dtype"])
                    + payload(4 * d, h, p["output_weight_dtype"])
                )
                # Expert local width ceil(704/4/32)*32=192; shared width 544.
                terms["indexed_top8_expert_weights"] += (
                    4 * 8 * (payload(h, 2 * 192, p["expert_gate_dtype"]) + payload(192, h, p["expert_down_dtype"]))
                )
                terms["shared_weights"] += 4 * (
                    payload(h, 2 * 544, p["shared_gate_dtype"]) + payload(544, h, p["shared_down_dtype"])
                )
                terms["replicated_router_weights"] += 4 * payload(h, 128, "bfloat16")
                terms["kv_reads"] += 4 * 2 * local_kv * payload(kv_read_tokens(pos, sliding), d, p["kv_cache_dtype"])
                # Read-modify-write entire 32-token cache tile for each updated row.
                terms["kv_update_read_write"] += 4 * 2 * 2 * local_kv * payload(32, d, p["kv_cache_dtype"])
                terms["norm_weights_estimate"] += 4 * (8 * h + 2 * d) * 4
                # Explicit approximation for non-weight DRAM operands, see ledger.
                terms["other_layer_activation_estimate"] += 4 * (
                    24 * payload(32, h, "bfloat16")
                    + 12 * payload(32, h, "float32")
                    + 4 * payload(32, (4 + 2 * local_kv) * d, "float32")
                )
                # Row-major page table, positions, route indices/weights, scalar state.
                terms["metadata_estimate"] += 4 * (4 * math.ceil((pos + 1) / 32) + 8 * 2 + 128 * 2 + 64)
        terms["lm_head_weights"] = 4 * payload(h, 262144 // 4, self.precision["model"]["head_weight_dtype"])
        terms["embedding_rows"] = batch_slots * h * 2
        terms["final_norm_estimate"] = 4 * (h * 4 + 3 * payload(batch_slots, h, "float32"))
        # Linear write, three softcap unary read/write passes, and sampler read.
        terms["logits_softcap_sampler"] = 4 * 8 * payload(batch_slots, 262144 // 4, "bfloat16")
        return {
            "dram_bytes": sum(terms.values()),
            "terms": dict(terms),
            "active_rows": len(positions),
            "batch_slots": batch_slots,
            "positions": positions,
            "scope": "estimated_full_model_dram_all_four_asics",
        }

    @staticmethod
    def peaks(clock_hz=1.35e9):
        if not isinstance(clock_hz, (int, float)) or not math.isfinite(clock_hz) or clock_hz <= 0:
            raise ValueError("positive finite clock required")
        return {
            "peak_flops_per_s": 4 * 120 * 4096 * clock_hz,
            "peak_dram_bytes_per_s": 4 * 512e9,
            "clock_hz": clock_hz,
            "peak_basis": "Four Blackhole ASICs,120 physical cores/ASIC,4096 LoFi FLOP/core/cycle; "
            "mixed-fidelity useful-work ratio against LoFi upper envelope, not FPU utilization; "
            "512GB/s DRAM/ASIC (P3001024GB/s per dual-ASIC card)",
            "reference": "https://docs.tenstorrent.com/aibs/blackhole/p300.html",
        }
