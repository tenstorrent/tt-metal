# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MoE expert-routing capture for Kimi-K2.7 chunked prefill (single Galaxy, no mpirun).

Drives the SAME chunked-prefill path as
``test_prefill_transformer_chunked.py::test_kimi_prefill_transformer_chunked_perf``
(L61, 11 chunks x 5120 = 56320 tokens, preload0, notrace) but with **num_iters=1**, and records
the MoE routing histograms for every (layer, chunk, chip, expert).

What is captured, per MoE layer and per chunk, straight off the device:

  totals[group][row][e]  -- ``total_counts_per_expert``: tokens dispatched to routed expert ``e``
                            summed over all 8 chips of the dispatch group. Replicated across the
                            8 rows of a group (all 8 hold the same vector); we keep all 32 copies
                            so the replication can be verified rather than assumed.
  hist[group][row][e]    -- ``expert_histograms``: the per-chip contribution, i.e. how many tokens
                            *resident on chip (group,row)* were routed to expert ``e``.
                            sum over rows within a group == totals for that group.

Mesh/expert geometry (8x4 Blackhole galaxy, 384 routed experts, top-8):
  mesh axis 0 = 8 rows  = chips within a dispatch group (sequence-parallel)
  mesh axis 1 = 4 cols  = dispatch groups (expert-parallel; ``expert_dispatch_table`` is sharded here)
  each chip hosts 12 experts -> 4 groups x 8 chips x 12 experts = 384
  expert e lives on global chip e // 12, local slot e % 12, dispatch group (e // 12) // 8

The capture is pure instrumentation: ``TtMoe.__init__`` is wrapped only to tag each routing-setup
instance with its layer index, and ``TtMoERoutingSetup.forward`` is wrapped to read its outputs back
to host. No production code is modified, and both wrappers are restored on exit.

Output (JSONL, one record per (layer, chunk)) goes to ``$KIMI_EXPERT_DUMP`` (default:
``generated/kimi_k2_7_expert_counts.jsonl`` under the repo).

Env (same shape as the CI "Blaze - Chunked Kimi perf" job, K2.7 paths):
    MESH_DEVICE=TG
    KIMI_K2_7_HF_MODEL=/mnt/models/moonshotai/Kimi-K2_7-Code-dequantized
    TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/moonshotai/Kimi-K2_7-Code-Cache/Kimi-K2_7-Code-Cache-prefill
    PREFILL_TRACE_DIR=/mnt/models/deepseek-prefill-cache/golden/structured_traces/vllm-kimi-k27-codedebug-56320
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer_chunked import CHUNK, run_chunked_transformer_updated
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe import TtMoe
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer

# The perf runner builds the transformer with kv_only_last_layer=True: the LAST block computes only
# KV and its MoE is never built, so layer num_layers-1 contributes no routing data. That is the right
# call for a timing run (the tail is dead work) but it would leave a hole in a per-layer routing map,
# so this capture forces the full block. Set KIMI_CAPTURE_KV_ONLY_LAST=1 to keep the perf harness's
# exact shape instead and accept the missing last layer.
_CAPTURE_ALL_LAYERS = os.environ.get("KIMI_CAPTURE_KV_ONLY_LAST", "0") != "1"


class ExpertCountCollector:
    """Reads ``total_counts_per_expert`` / ``expert_histograms`` back to host on every MoE forward.

    Chunk index is derived per layer from the call ordering: with num_iters=1, notrace and no
    profile warmup, MoE layer L is entered exactly once per chunk, in chunk order. The test asserts
    the resulting shape (n_moe_layers x n_chunks) at the end, so a stray extra forward (a warmup
    pass, a second iteration) is caught rather than silently mislabelled.
    """

    def __init__(self, mesh_device, out_path: Path, n_chunks: int):
        self.mesh_device = mesh_device
        self.out_path = out_path
        self.n_chunks = n_chunks
        self.calls_per_layer: dict[int, int] = {}
        self.records: list[dict] = []
        self._fh = None
        self._orig_moe_init = None
        self._orig_setup_forward = None
        self._orig_transformer_init = None
        # mesh axis 0 -> tensor dim 1 (rows / chips in group), mesh axis 1 -> tensor dim 0 (dispatch groups)
        self._composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(dims=[1, 0]))

    def _to_host(self, tensor) -> torch.Tensor:
        """(num_groups, group_size, num_routed_experts) int64 on host."""
        composed = ttnn.to_torch(ttnn.unsqueeze_to_4D(tensor), mesh_composer=self._composer)
        return composed.reshape(composed.shape[0], composed.shape[1], -1).to(torch.int64)

    def install(self):
        self._fh = self.out_path.open("w")
        collector = self

        self._orig_moe_init = TtMoe.__init__
        self._orig_setup_forward = TtMoERoutingSetup.forward

        def moe_init(moe_self, *args, **kwargs):
            collector._orig_moe_init(moe_self, *args, **kwargs)
            # Every call site passes layer_idx by keyword (tt_prefill_block.build_moe); fall back to
            # construction order (MoE blocks are built in layer order) if that ever changes.
            moe_self.routing_setup._capture_layer_idx = kwargs.get("layer_idx", len(collector.calls_per_layer))

        def setup_forward(setup_self, *args, **kwargs):
            out = collector._orig_setup_forward(setup_self, *args, **kwargs)
            _, total_counts, _, histograms = out
            collector._record(getattr(setup_self, "_capture_layer_idx", -1), total_counts, histograms)
            return out

        TtMoe.__init__ = moe_init
        TtMoERoutingSetup.forward = setup_forward

        if _CAPTURE_ALL_LAYERS:
            self._orig_transformer_init = TtPrefillTransformer.__init__

            def transformer_init(tf_self, *args, **kwargs):
                kwargs["kv_only_last_layer"] = False
                collector._orig_transformer_init(tf_self, *args, **kwargs)

            TtPrefillTransformer.__init__ = transformer_init

        logger.info(f"[expert-capture] installed (all_layers={_CAPTURE_ALL_LAYERS}); writing {self.out_path}")

    def restore(self):
        if self._orig_moe_init is not None:
            TtMoe.__init__ = self._orig_moe_init
        if self._orig_setup_forward is not None:
            TtMoERoutingSetup.forward = self._orig_setup_forward
        if self._orig_transformer_init is not None:
            TtPrefillTransformer.__init__ = self._orig_transformer_init
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _record(self, layer_idx: int, total_counts, histograms):
        chunk = self.calls_per_layer.get(layer_idx, 0)
        self.calls_per_layer[layer_idx] = chunk + 1

        totals = self._to_host(total_counts)
        hist = self._to_host(histograms)
        rec = {
            "layer": layer_idx,
            "chunk": chunk,
            "totals": totals.tolist(),  # [group][row][expert] - replicated across rows
            "hist": hist.tolist(),  # [group][row][expert] - per-source-chip contribution
        }
        self.records.append({"layer": layer_idx, "chunk": chunk, "totals": totals, "hist": hist})
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._fh.flush()
        if layer_idx <= 1 or layer_idx % 20 == 0:
            logger.info(
                f"[expert-capture] layer={layer_idx} chunk={chunk} "
                f"routed_tokens={int(totals[:, 0, :].sum())} max_expert={int(totals[:, 0, :].max())}"
            )


@pytest.mark.parametrize("num_iters", [1], ids=["iters1"])
# chunks1 / L10 exist only as a fast end-to-end smoke of the capture path; the real capture is
# "L61 and chunks_eleven" (the full 56320-token code_debug prefill).
@pytest.mark.parametrize("n_chunks", [1, 11], ids=["chunks1", "chunks_eleven"])
@pytest.mark.parametrize("preload_isl", [0], ids=["preload0"])
@pytest.mark.parametrize("num_layers", [10, 61], ids=["L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
                "l1_small_size": 768,
                "trace_region_size": 256 * 1024 * 1024,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_7"], indirect=True, ids=["kimi_k2_7"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_k2_7_expert_token_counts(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_iters,
    num_links,
    topology,
    preload_isl,
):
    out_path = Path(os.environ.get("KIMI_EXPERT_DUMP", "generated/kimi_k2_7_expert_counts.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collector = ExpertCountCollector(mesh_device, out_path, n_chunks)
    collector.install()
    try:
        run_chunked_transformer_updated(
            variant,
            config_only,
            mesh_device,
            weight_cache_path,
            num_layers,
            n_chunks,
            GateComputeMode.DEVICE_FP32,
            num_links,
            topology,
            num_iters,
            routing_use_l1_small_for_semaphores=True,
            baseline_chunk_times_s=None,  # readback per layer perturbs timing; data run, not a perf gate
            perf_margin=None,
            preload_isl=preload_isl,
            check_pcc=False,
            use_trace=False,
        )
    finally:
        collector.restore()

    # --- structural checks on what we captured -------------------------------------------------
    layers = sorted(collector.calls_per_layer)
    assert layers, "no MoE forwards were captured"
    # layers [first_k_dense_replace, num_layers) are MoE; the last block loses its MoE when the
    # perf harness's kv_only_last_layer=True is left in place.
    n_moe_layers = num_layers - int(config_only.first_k_dense_replace) - (0 if _CAPTURE_ALL_LAYERS else 1)
    assert len(layers) == n_moe_layers, f"expected {n_moe_layers} MoE layers, captured {len(layers)}: {layers}"
    for layer in layers:
        assert (
            collector.calls_per_layer[layer] == n_chunks
        ), f"layer {layer}: {collector.calls_per_layer[layer]} forwards, expected {n_chunks}"

    num_groups, group_size, num_experts = collector.records[0]["totals"].shape
    experts_per_chip = num_experts // (num_groups * group_size)
    logger.info(
        f"[expert-capture] {len(collector.records)} records: layers {layers[0]}..{layers[-1]}, "
        f"{n_chunks} chunks, {num_groups} groups x {group_size} chips x {experts_per_chip} experts "
        f"= {num_experts} routed experts"
    )

    expected_routings = CHUNK * int(config_only.num_experts_per_tok)
    for rec in collector.records:
        totals, hist = rec["totals"], rec["hist"]
        where = f"layer {rec['layer']} chunk {rec['chunk']}"
        # totals is replicated across the 8 chips of a dispatch group
        assert torch.equal(
            totals, totals[:, :1, :].expand_as(totals)
        ), f"{where}: total_counts_per_expert differs across chips within a dispatch group"
        # per-chip histograms sum to the group total
        assert torch.equal(hist.sum(dim=1), totals[:, 0, :]), f"{where}: per-chip histograms do not sum to the total"
        # every token routed to exactly top_k experts, across all four groups
        assert (
            int(totals[:, 0, :].sum()) == expected_routings
        ), f"{where}: {int(totals[:, 0, :].sum())} routings, expected {expected_routings}"
        # each dispatch group only counts the experts it hosts
        per_group = num_experts // num_groups
        for g in range(num_groups):
            outside = totals[g, 0, :].clone()
            outside[g * per_group : (g + 1) * per_group] = 0
            assert int(outside.sum()) == 0, f"{where}: group {g} counted experts outside its own range"

    logger.info(f"[expert-capture] wrote {len(collector.records)} records to {out_path}")
