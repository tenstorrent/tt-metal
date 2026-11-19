"""CPU-only work ledger for full48 Qwen3.8-Flash-Next TP4/EP4 serving.

This models useful arithmetic and source-visible DRAM payload/materializations,
not hardware transaction counters. The expert-union interval bounds uncertainty
in that category only; it is not a bound on all physical DRAM transactions.
No trace, tensor readback, synchronization or TTNN import is performed here.
"""
import math
from collections import defaultdict

H, RANKS, LAYERS, EXPERTS, TOPK, EXPERT_I = 2560, 4, 48, 512, 10, 640
CAPACITY = 262144
PEAK_LOFI = 4 * 120 * 1.35e9 * 4096
PEAK_DRAM = 4 * 512e9
HARDWARE_SOURCE = "https://docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/specifications.html"
FLOP_SOURCE = "tech_reports/GEMM_FLOPS/GEMM_FLOPS.md"
SOURCES = {
    "parallelism": "tt/factored_decode.py: attention/terminal execute once per active slot; moe executes once per bucket",
    "prepared_weights": "tt/dram_projection.py: weight_memory_plan and _PreparedGroup; 8-bank N padding",
    "precision": "live model.precision_summary(): per-layer constructed dtype/fidelity/weight_shapes",
    "fixed_weights": "tt/multichip_decoder.py: MeshResidual, MeshGDN, MeshMoe, MeshQSA, MeshPLE; tt/model.py: FinalMixer",
    "exact_products": "tt/multichip_decoder.py: MeshExactProjection native_q1; tt/fused_decoder.py: FusedPreciseProjection",
    "recurrence": "tt/multichip_decoder.py: mesh_recurrent_gated_delta_rule, state read/final write and copy back",
    "indexer": "tt/causal_indexer_scores.py; tt/kernels/causal_score_packed_q1_{reader,writer}.cpp",
    "selected_kv": "tt/fused_decoder.py: fused_selected_token_ids; tt/multichip_decoder.py: LocalSelectedPagedKV",
    "ple_lookup": "tt/pipelined_striped_gather.py; MeshPackedEmbedding; 16*(20+4) uint32 selections/rank",
    "collective_boundaries": "tt/multichip_decoder.py: Collectives.gather/sum, GatheredProjection, MeshResidual, MeshMoe, MeshPackedEmbedding; tt/model.py: embed/FinalMixer",
    "residual_materializations": "tt/fused_decoder.py: merged_rms; tt/multichip_decoder.py: MeshResidual.read/write; tt/kernels/exact_residual_write_{reader,writer}.cpp",
    "moe_materializations": "tt/multichip_decoder.py: MeshMoe.__call__, static combine_capacity untilize, _reduce/_reduce_single",
    "sampler_canvas": "tt/generator.py: _decode_body/_replay_sampling; tt/factored_decode.py: clear/terminal; tt/chunk_output.py; models/common/sampling/tt_sampling.py: forward local topk and candidate gathers",
    "factored_slots": "tt/factored_decode.py: attention copies residual/mixed/injection, moe indexed pack/scatter; tt/indexed_rows.py native reader/writer",
}

TRAFFIC_DERIVATIONS = {
    "collectives": "All-gather BF16 boundary = source read2 plus four-rank output write8 =10 B/local element/rank. exact_rank_sum then reads four BF16 rank inputs8 and writes FP32 output4, so ordered-sum boundary totals22 B/local element/rank. Caller FP32->BF16 casts are separate. Interconnect traffic is not added to DRAM bytes. Internal staging/reloads excluded.",
    "residual_norm": "Per physical H-wide row: BF16->FP32 cast 6 B/element; square read+write 8; variance reduction input read4; normalization read+write8; norm-weight multiply two reads+write12; BF16 cast6 =44 B/element. Small variance buffers excluded; two layer mixers plus final mixer.",
    "residual_mix": "Up output BF16->FP32/sigmoid/BF16 pipeline20 B/element; mixing*normalized two reads+write6; post-sum FP32 scalar-scale read+write8 and FP32->BF16 cast6 =40 B/element. CCL is a distinct category; projection products/weights remain separate.",
    "residual_write": "Two per-layer exact writers read residual and block and write output, plus one unique injection tile. Repeated injection-tile requests per worker are excluded from conservative payload.",
    "moe_untilize": "Explicit static combine_capacity=32*min(512,10*b) by2560 BF16 tensor, one read+write during untilize on each rank, including static padding; do not charge full resident workspace capacity.",
    "moe_payload": "Rank-local minimum: dispatch input read/routed write, combine input read/output write, reduction input read/output write =2*b*H BF16 input/output plus four*b*10*H BF16 routing/intermediate passes per rank. Expert output write adds b*10*H BF16 elements ONCE over EP4 because only owning ranks execute experts; unlike combine's full local buffer it is not replicated. Other padded post-combine/layout temporaries excluded.",
    "sampler": "Old head logits write+read category is specifically head output write plus chunk-copy input read. Separately count once-per-step canvas clear, valid-row canvas writes, and one full physical32-row canvas read for local topk. Local top32 candidate writes and DRAM all-gather boundaries use conservative BF16+uint16 sizes; sort/sampling internals excluded.",
    "factored_slots": "Only servers with slots>1: each layer copies per-slot residual, mixed and injection buffers; MoE packing reads one whole H-wide tile-row per active slot, scatter reads packed tiles once, both write valid rows only. Bucket1 on a two-slot server retains these copies.",
}


def dtype_name(value):
    return str(value).split(".")[-1].lower()


def item_bytes(dtype):
    # Exponents are included: 1024-element BFP4/BFP8 tiles are 576/1088 B.
    return {"bfloat4_b": 576 / 1024, "bfloat8_b": 1088 / 1024, "bfloat16": 2, "float32": 4, "uint32": 4}[
        dtype_name(dtype)
    ]


def fidelity(value):
    return {"lofi": 1, "hifi2": 2, "hifi3": 3, "hifi4": 4}[str(value).split(".")[-1].lower()]


def padded(n, multiple=32):
    return math.ceil(n / multiple) * multiple


def projection_bytes(projection, phase):
    p = projection[phase]
    return RANKS * sum(
        math.prod(shape[:-2]) * padded(shape[-2]) * padded(shape[-1], 256) * item_bytes(dtype)
        for shape, dtype in zip(p["weight_shapes"], p["weight_dtypes"], strict=True)
    )


def validate_identity(identity):
    precision = identity.get("runtime_precision", identity.get("precision_summary", identity.get("precision")))
    if not isinstance(precision, dict) or "layers" not in precision:
        raise ValueError("Accounting requires live constructed model.precision_summary(), not requested policy alone")
    if [row["layer"] for row in precision["layers"]] != list(range(LAYERS)):
        raise ValueError("Accounting requires all 48 layers in original order")
    if precision["requested"]["config_id"] != "combined_shared4_mixed":
        raise ValueError("Work ledger is audited for combined_shared4_mixed")
    if precision["requested"]["max_context"] != CAPACITY:
        raise ValueError("Full-context identity changed")
    if "devices" in identity and len(identity["devices"]) != RANKS:
        raise ValueError("TP4/EP4 requires four participating devices")
    if "mesh_shape" in identity and list(identity["mesh_shape"]) != [1, 4]:
        raise ValueError("Expected 1x4 model mesh")
    for layer in precision["layers"]:
        if layer["experts"]["active_experts_per_token"] != TOPK or layer["experts"]["local_experts"] != 128:
            raise ValueError("Expert selection/partition changed")
        if dtype_name(layer["kv_dtype"]) != "bfloat16":
            raise ValueError("This KV ledger requires selected BF16 cache")
        if "shared" not in layer["groups"] or len(layer["groups"]["gr"]) != 2:
            raise ValueError("Missing constructed projection groups")
        expected_shapes = {
            "gr": [1, 1, 10240, 32],
            "shared": [1, 1, 2560, 320],
            "gdn_zab": [1, 1, 2560, 1568],
            "qsa_qkv": [1, 1, 2560, 3584],
            "qsa_out": [1, 1, 1536, 2560],
        }
        for role, groups in layer["groups"].items():
            if role not in expected_shapes:
                raise ValueError(f"Unaudited projection role: {role}")
            for group in groups:
                for phase in ("decode", "rows"):
                    if group[phase]["weight_shapes"] != [expected_shapes[role]]:
                        raise ValueError(f"Constructed {role}/{phase} shape changed")
    if sum("gdn_zab" in row["groups"] for row in precision["layers"]) != 36:
        raise ValueError("Expected 36 GDN and 12 QSA layers")
    for phase in ("decode", "rows"):
        if precision["lm_head"][phase]["weight_shapes"] != [[1, 1, 2560, 62080]]:
            raise ValueError("Constructed vocabulary projection shape changed")
    return precision


def selected_rows(position):
    complete, tail = divmod(position + 1, 4)
    return 4 * min(512, complete) + tail


class Ledger:
    def __init__(self):
        self.values = defaultdict(float)
        self.fidelity_cost = defaultdict(float)

    def add(self, category, value, divisor=None):
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"Invalid accounting contribution {category}: {value}")
        self.values[category] += value
        if divisor is not None:
            self.fidelity_cost[category] += value * divisor

    def total(self):
        return sum(self.values.values())


def prefill_chunk(ledger, precision, start, end):
    """Useful logical rows including rollback replay; exclude tile padding FLOPs."""
    rows = end - start
    # Four local matrices are summed. Replicated QSA indexer/PLE value work is
    # included four times because every rank executes it, unlike EP experts.
    for layer in precision["layers"]:
        groups = layer["groups"]
        # Two hyperconnection mixers; down has 80 useful outputs/rank (96 padded).
        ledger.add("hyperconnection_exact_down_up", 2 * rows * RANKS * 2 * (80 * 10240 + 2560 * 320), 4)
        ledger.add(
            "hyperconnection_injection",
            2 * rows * RANKS * 2 * 10240,
            fidelity(groups["gr"][0]["rows"]["compute"]["math_fidelity"]),
        )
        ledger.add("router", 2 * rows * H * EXPERTS, 4)
        ledger.add(
            "routed_expert_gate_up_down",
            2 * rows * TOPK * 3 * H * EXPERT_I,
            fidelity(layer["experts"]["compute"]["math_fidelity"]),
        )
        ledger.add(
            "shared_gate_up",
            2 * rows * RANKS * H * 320,
            fidelity(groups["shared"][0]["rows"]["compute"]["math_fidelity"]),
        )
        ledger.add(
            "shared_down", 2 * rows * RANKS * 160 * H, fidelity(layer["shared_down"]["compute"]["math_fidelity"])
        )
        ledger.add("shared_route", 2 * rows * RANKS * H, 4)
        if "gdn_zab" in groups:
            ledger.add("gdn_exact_qkv_out", 2 * rows * RANKS * (H * 2560 + 640 * 6144), 4)
            ledger.add(
                "gdn_zab",
                2 * rows * RANKS * H * 1560,
                fidelity(groups["gdn_zab"][0]["rows"]["compute"]["math_fidelity"]),
            )
            ledger.add("gdn_recurrence_and_conv", rows * RANKS * (12 * (7 * 128 * 128 + 2 * 128) + 7 * 2560), 4)
        else:
            ledger.add(
                "qsa_qgate_k_v",
                2 * rows * RANKS * H * 3584,
                fidelity(groups["qsa_qkv"][0]["rows"]["compute"]["math_fidelity"]),
            )
            ledger.add(
                "qsa_output",
                2 * rows * RANKS * 1536 * H,
                fidelity(groups["qsa_out"][0]["rows"]["compute"]["math_fidelity"]),
            )
            ledger.add("qsa_index_projection_replicated", 2 * rows * RANKS * H * 640, 4)
            ledger.add("qsa_causal_index_dots", RANKS * 2 * 4 * 128 * sum((p + 1) // 4 for p in range(start, end)), 4)
            ledger.add(
                "qsa_selected_attention", RANKS * 4 * 6 * 256 * sum(selected_rows(p) for p in range(start, end)), 2
            )
    ledger.add("ple_key_and_replicated_value", 2 * rows * RANKS * 2 * H * H, 4)
    ledger.add("final_mixer", 2 * rows * RANKS * (80 * 10240 + H * 320), 4)
    # _prefill_body writes all logits for every executed chunk, even a replay.
    ledger.add(
        "vocabulary_projection",
        2 * rows * H * 248320,
        fidelity(precision["lm_head"]["rows"]["compute"]["math_fidelity"]),
    )


def decode_step(ledger, precision, positions, server_slots=None):
    """Conservative materialized-data ledger over all four ASICs for one step."""
    b = len(positions)
    if b not in (1, 2):
        raise ValueError("Audited serving profiles have one or two active slots")
    server_slots = b if server_slots is None else server_slots
    if server_slots not in (1, 2) or b > server_slots:
        raise ValueError("Invalid active batch/server slot relationship")
    phase = "decode" if b == 1 else "rows"
    expert_extra = 0
    row_elements = 32 * H

    def gather_boundary(width, count=1):
        return count * RANKS * 5 * 32 * padded(width) * 2

    def sum_boundary(width=H, count=1):
        # BF16 all-gather (2+4*2), BF16 reduction read (4*2), FP32 output (4).
        return count * RANKS * 32 * padded(width) * 22

    for layer in precision["layers"]:
        groups = layer["groups"]
        # Attention and its two hyperconnection mixers are per-slot in FactoredDecode.
        for name, projections in groups.items():
            for projection in projections:
                multiplier = 1 if name == "shared" else b
                ledger.add(
                    "prepared_" + name + "_weights",
                    multiplier * projection_bytes(projection, phase if name == "shared" else "decode"),
                )
        ledger.add("hyperconnection_exact_weights", b * RANKS * 2 * (96 * 10240 + H * 320) * 4)
        ledger.add(
            "gdn_exact_weights" if "gdn_zab" in groups else "qsa_indexer_weights",
            b * RANKS * ((H * 2560 + 640 * 6144) * 2 if "gdn_zab" in groups else H * 640 * 2),
        )
        # Router and shared expert run once for the packed MoE bucket. Their
        # payload assumes one matrix scan, not one full scan per token.
        ledger.add("router_weights", RANKS * 128 * H * 2)
        ledger.add("shared_down_weights", RANKS * 160 * H * item_bytes(layer["shared_down"]["dtype"]))
        ledger.add("shared_gate_weights_padded", RANKS * H * 32 * item_bytes("bfloat8_b"))
        one_expert = (
            H * EXPERT_I * sum(item_bytes(layer["experts"]["weights"][name][0]) for name in ("gate", "up", "down"))
        )
        ledger.add("routed_expert_weights_union_lower", TOPK * one_expert)
        expert_extra += (min(TOPK * b, EXPERTS) - TOPK) * one_expert
        # Each slot's GR-down still emits FP32 products; GR-up and GDN-out
        # now use exact_projection_q1 and do not emit the old product tensor.
        ledger.add("gr_down_fp32_products_write_read", b * RANKS * 2 * 2 * 96 * 10240 * 4)
        ledger.add("router_fp32_products_write_read", b * RANKS * 2 * 128 * H * 4)
        # Separate producer/consumer operations: none of these source/output
        # buffers are the weight scans or FP32 product arrays counted above.
        ledger.add(
            "hyperconnection_ccl_dram_boundaries", 2 * b * (gather_boundary(H) + gather_boundary(96) + sum_boundary())
        )
        ledger.add("moe_router_and_output_ccl_dram_boundaries", gather_boundary(128) + sum_boundary())
        ledger.add("residual_norm_materializations", 2 * b * RANKS * row_elements * 44)
        ledger.add("residual_mixing_materializations", 2 * b * RANKS * row_elements * 40)
        ledger.add("residual_write_payload", 2 * b * RANKS * (row_elements * 6 + 2048))
        combine_capacity = 32 * min(EXPERTS, TOPK * b)
        ledger.add("moe_static_combine_untilize", RANKS * combine_capacity * H * 4)
        ledger.add(
            "moe_live_dispatch_combine_reduce_payload", RANKS * (2 * b * H + 4 * b * TOPK * H) * 2 + b * TOPK * H * 2
        )
        ledger.add("moe_ordered_sum_output_bf16_cast", RANKS * row_elements * 6)
        if server_slots > 1:
            ledger.add("factored_per_slot_buffer_copies", b * RANKS * (2 * row_elements + 32 * 32) * 4)
            ledger.add("factored_moe_pack_scatter", RANKS * ((b + 1) * row_elements * 2 + 2 * b * H * 2))
        if "gdn_zab" in groups:
            ledger.add("gdn_output_ccl_dram_boundaries", b * (gather_boundary(1536) + gather_boundary(640)))
            # Initial state -> kernel final -> persistent copy = four full passes.
            ledger.add("gdn_recurrent_state_and_copy", b * RANKS * 4 * 12 * 128 * 128 * 4)
            ledger.add("gdn_convolution_state_read_write", b * RANKS * 2 * 32 * 2560 * 2)
        else:
            ledger.add("qsa_output_ccl_dram_boundaries", b * sum_boundary())
            ledger.add("qsa_ordered_sum_output_bf16_cast", b * RANKS * row_elements * 6)
            ledger.add("selected_kv_payload_reads", RANKS * sum(selected_rows(p) for p in positions) * 2 * 256 * 2)
            ledger.add("kv_append_payload_writes", b * RANKS * 2 * 256 * 2)
            ledger.add("selected_kv_fixed_gather_write_sdpa_read", b * RANKS * 2 * 2080 * 2 * 256 * 2)
            # Q1 per-slot packed scorer: 8 MiB dot output and 3 read+write passes.
            score_bytes = 32 * (CAPACITY // 4) * 4
            ledger.add("indexer_dot_relu_reduce_scale_materializations", b * RANKS * 7 * score_bytes)
            # Causal tiled key payload once; repeated query read requests are
            # retained separately as an explanatory kernel-request estimate.
            ledger.add(
                "indexer_causal_key_tile_reads",
                RANKS * sum(math.ceil(((p + 1) // 4) / 32) for p in positions) * 4 * 2048,
            )
            ledger.add("indexer_query_unique_tiles", b * RANKS * 4 * 4 * 2048)
            ledger.add(
                "selector_scores_and_selected_id_payload",
                RANKS * sum(padded((p + 1) // 4) * 4 + 2080 * 4 for p in positions),
            )
            ledger.add("compressed_key_writes", RANKS * sum((p + 1) % 4 == 0 for p in positions) * 128 * 2)
    ledger.add("ple_key_value_weights", b * RANKS * 2 * H * H * item_bytes("bfloat8_b"))
    ledger.add("ple_lookup_minimum_useful_payload", b * RANKS * 16 * (20 + 4) * 4)
    ledger.add("token_embedding_lookup", b * H * 2)
    ledger.add("embedding_and_ple_ccl_dram_boundaries", b * (gather_boundary(640) + sum_boundary()))
    ledger.add("ple_ordered_sum_output_bf16_cast", b * RANKS * row_elements * 6)
    ledger.add("final_mixer_weights", b * RANKS * (96 * 10240 + H * 320) * 4)
    ledger.add("final_mixer_products_write_read", b * RANKS * 2 * (96 * 10240 + H * 320) * 4)
    ledger.add("final_mixer_ccl_dram_boundaries", b * (gather_boundary(H) + gather_boundary(96) + sum_boundary()))
    ledger.add("final_mixer_norm_mix_materializations", b * RANKS * row_elements * (44 + 40))
    ledger.add("vocabulary_projection_weights", b * projection_bytes(precision["lm_head"], "decode"))
    # chunk_output reads whole source tiles but writes only active canvas rows.
    # The native sampler then consumes physical32 rows even for B1 or B2.
    ledger.add("head_logits_write_copy_read", b * RANKS * 2 * 32 * padded(62080) * 2)
    ledger.add("sampler_canvas_clear_and_topk_read", RANKS * 2 * 32 * 62080 * 2)
    ledger.add("sampler_canvas_active_row_writes", b * RANKS * 62080 * 2)
    ledger.add("sampler_minimum_top32_candidate_writes", RANKS * 32 * 32 * (2 + 2))
    ledger.add("sampler_candidate_ccl_dram_boundaries", 5 * RANKS * 32 * 32 * (2 + 2))
    return expert_extra


def account(events, runtime_identity):
    """Return work totals/categories for already selected measured phase events.

    Event layout is PhaseRecorder's {phase, metadata:{requests,chunks}}. This
    function does not select requests or assign time; the collector must supply
    exactly the matched requests and their full-phase completion intervals.
    """
    precision = validate_identity(runtime_identity)
    prefill, decode = Ledger(), Ledger()
    chunks, positions_seen, decode_steps, prefill_calls, expert_extra = [], [], 0, 0, 0
    for event in events:
        metadata = event.get("metadata", event)
        if event["phase"] == "prefill":
            prefill_calls += 1
            if not metadata.get("chunks"):
                raise ValueError("Prefill requires actual executed chunk boundaries, including rollback replays")
            for chunk in metadata["chunks"]:
                start, end = int(chunk["start"]), int(chunk["end"])
                if not 0 <= start < end <= CAPACITY:
                    raise ValueError("Invalid executed prefill chunk positions")
                prefill_chunk(prefill, precision, start, end)
                chunks.append(dict(chunk))
        elif event["phase"] == "decode":
            positions = [int(request["position"]) for request in metadata.get("requests", [])]
            if not positions or any(p < 0 or p >= CAPACITY for p in positions):
                raise ValueError("Decode requires actual active-slot positions")
            expert_extra += decode_step(decode, precision, positions, runtime_identity.get("slots"))
            decode_steps += 1
            positions_seen.extend(positions)
        else:
            raise ValueError("Unknown phase")
    if not prefill.total() or not decode.total():
        raise ValueError("Matched profile must contain both prefill and decode work")
    useful_flops = prefill.total()
    divisor = sum(prefill.fidelity_cost.values()) / useful_flops
    return {
        "prefill": {
            "flops": useful_flops,
            "categories_flops": dict(prefill.values),
            "fidelity_weighted_flops": dict(prefill.fidelity_cost),
            "phase_fidelity_divisor": divisor,
            "peak_flops_per_second": PEAK_LOFI / divisor,
            "nominal_lofi_peak_flops_per_second": PEAK_LOFI,
            "peak_source": HARDWARE_SOURCE + "; " + FLOP_SOURCE,
            "work_method": "Sum 2*M*K*N over full48 executed logical chunk rows, active top10 experts, replicated rank work, actual causal index/selected-attention positions, and useful recurrence; include replay tails and every chunk's logits. Exclude tile padding and minor scalar/nonlinear FLOPs.",
            "peak_method": "4 ASICs * 120 cores * 1.35GHz * 4096 FLOP/cycle / work-weighted fidelity divisor (LoFi1, HiFi2=2, HiFi4=4). SFPU exact/recurrent work is conservatively normalized at divisor4; this mixed-path nominal matrix-engine reference is not an SFPU achievable ceiling.",
            "executed_chunks": len(chunks),
            "executed_rows_including_replay": sum(c["end"] - c["start"] for c in chunks),
            "prefill_calls": prefill_calls,
        },
        "decode": {
            "dram_bytes": decode.total(),
            "dram_bytes_expert_union_upper": decode.total() + expert_extra,
            "categories_bytes": dict(decode.values),
            "expert_union_uncertainty_bytes": expert_extra,
            "peak_dram_bytes_per_second": PEAK_DRAM,
            "peak_source": HARDWARE_SOURCE,
            "work_method": "Conservative source-derived payload plus explicit DRAM materializations over TP4/EP4, padded prepared weight storage and BFP exponents, active expert union lower estimate, selected KV, recurrent state/copies, indexer terminal materializations and output traffic. Per-slot attention/head; shared bucket MoE. Not measured DRAM transactions.",
            "expert_union_method": "For each layer/step, distinct active expert count in [10,min(10*active_slots,512)]; lower assumes identical routing, upper disjoint. No routing readback added. EP weights counted once over all ranks, never multiplied by four. Upper is an expert-union sensitivity bound, not a bound on total physical transactions.",
            "decode_steps": decode_steps,
            "active_token_steps": len(positions_seen),
            "positions_min": min(positions_seen),
            "positions_max": max(positions_seen),
            "kernel_request_extras_not_in_headline": {
                "ple_lookup_requested_bytes": len(positions_seen) * RANKS * 16 * 24 * 64,
                "indexer_repeated_query_read_requests_bytes": RANKS
                * 12
                * sum(math.ceil(((p + 1) // 4) / 32) for p in positions_seen)
                * 32768,
            },
        },
        "scope": "Conservative modeled work over complete measured phase wall times, not complete hardware transaction counts",
        "parallelism": "TP4/EP4; sum all participating ASIC payload then divide by aggregate 2.048 TB/s; shared MoE bucket amortization; QSA indexer and PLE value replication explicitly included",
        "sources": SOURCES,
        "material_traffic_derivations": TRAFFIC_DERIVATIONS,
        "omissions_and_uncertainties": [
            "No live profiler or bandwidth counters: allocator sizes, requested NoC reads and physical DRAM transactions differ; caching/reloads/bank imbalance are unmeasured.",
            "Conservative ledger includes major norm/mix materializations, explicit CCL DRAM boundaries, MoE untilize/live intermediate payload and sampler canvas/candidate boundaries; still omits some small nonlinear/layout/metadata traffic, trace dispatch, internal collective staging, sampler/sort scratch and implementation-dependent repeated weight reads. Not a complete transaction count.",
            "Prefill omits small norm, nonlinear, softmax, routing-sort and sampling FLOPs; exact/SFPU operations share a nominal matrix-equivalent fidelity reference rather than a separate SFPU ceiling.",
            "Expert union is unknown without an intrusive readback; both lower and upper category estimates are retained; headline uses lower.",
            "GR-up and GDN-output native Q1 avoid old Stage7 FP32 product tensors; final-mixer-up remains materialized. Stale Stage7 full-product traffic is intentionally excluded.",
        ],
    }
