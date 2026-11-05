# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""The persistent TP12 MoE: one ``fused_experts`` launch serving several layers over host sockets.

Three layers of random weights at V4.1's shapes (D = 5120, I = 2304, 384 experts, top 6) are loaded on a 3x4
submesh, each split TP12 as the model splits it. The op is launched once. Every step the host pushes a
``[64-byte header | bf16 row]`` page to each chip's H2D socket, the header's first word naming the layer; the
chips run that layer's whole MoE (router, routed experts, shared expert), reduce it onto mesh coordinate (0, 0),
and the host reads the row back from that chip's D2H socket. The layers are visited interleaved. Each step is
checked against a torch reference and against the same layer's non-persistent op on the same row, and then a
stop page ends every kernel so the device closes normally.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe_persistent.py
"""

import itertools
import os
import time
import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts, DeepSeekV41SparseMoeBlock
from models.experimental.deepseek_v4_flash.tt.common import CCL_TOPOLOGY, width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import resolve_snapshot_dir

PCC = 0.95
# The persistent op runs the very same kernels as the one-shot op, so the two should agree to rounding.
PCC_VS_OP = 0.9999
PARENT_MESH = (8, 4)
SUBMESH = (3, 4)
TP_SIZE = 12
NUM_LAYERS = 3
# The layer of each step: every layer is visited more than once, never in a fixed order.
STEPS = (0, 2, 1, 1, 0, 2, 0, 1, 2)
# Distinct base experts per layer; expert e is base e % EXPERT_POOL scaled by its own factor, which keeps every
# expert distinct without generating 384 full-size random experts per layer.
EXPERT_POOL = 4
# Biased router scores within this of the k-th one are a tie the device's bf16 router (one ulp ~ 0.008 at
# these scores) may break either way; the torch reference accepts any top-k drawn from that band.
ROUTING_TIE = 0.02
STOP_LAYER = 0xFFFFFFFF
HEADER_BYTES = 64
# The worker grid and router row the op multicasts each row over: 12 x (8 + 1).
REPLICA_GRID = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8))])


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _random_layer(config, seed):
    """``(provider, weights)`` in the shapes ``expert_provider`` / ``moe_weights`` return, random."""
    g = torch.Generator().manual_seed(seed)
    d, i, e = config.hidden_size, config.moe_intermediate_size, config.num_local_experts
    pool_gate_up = [torch.randn(2 * i, d, generator=g) * d**-0.5 for _ in range(EXPERT_POOL)]
    pool_down = [torch.randn(d, i, generator=g) * i**-0.5 for _ in range(EXPERT_POOL)]

    def provider(x: int):
        scale = 1.0 + x / e
        return pool_gate_up[x % EXPERT_POOL] * scale, pool_down[x % EXPERT_POOL] * scale

    weights = {
        "gate.weight": torch.randn(e, d, generator=g) * d**-0.5,
        "gate.e_score_correction_bias": torch.randn(e, generator=g) * 0.1,
        "shared_experts.gate_proj.weight": torch.randn(i, d, generator=g) * d**-0.5,
        "shared_experts.up_proj.weight": torch.randn(i, d, generator=g) * d**-0.5,
        "shared_experts.down_proj.weight": torch.randn(d, i, generator=g) * i**-0.5,
    }
    return provider, weights


def _reference(x, provider, weights, config, experts):
    """The MoE of one row ``x`` ``[D]`` in fp32: the ``sqrtsoftplus`` router ranked on the biased scores, the
    renormalized top-k experts, and the shared expert, all with the ``swiglu_limit`` clamp.

    Returns ``(candidates, margin)``. ``candidates[0]`` is the output for the exact top-k; the rest are the
    outputs for every other top-k the device could pick by breaking a ``ROUTING_TIE`` differently. ``margin``
    is the gap between the k-th and (k+1)-th ranked score."""
    k = experts.top_k
    gate_w = weights["gate.weight"] / getattr(config, "gate_temp", 1.0)
    scores = torch.sqrt(torch.nn.functional.softplus(gate_w @ x))
    biased = scores + weights["gate.e_score_correction_bias"]
    order = biased.argsort(descending=True).tolist()
    margin = (biased[order[k - 1]] - biased[order[k]]).item()
    boundary = biased[order[k - 1]].item()
    band = [e for e in order if abs(biased[e].item() - boundary) <= ROUTING_TIE]
    sure = [e for e in order[:k] if e not in band]
    limit = experts.limit
    i = config.moe_intermediate_size

    def swiglu(gate, up, down):
        act = torch.nn.functional.silu((gate @ x).clamp(max=limit)) * (up @ x).clamp(-limit, limit)
        return down @ act

    shared = swiglu(
        weights["shared_experts.gate_proj.weight"],
        weights["shared_experts.up_proj.weight"],
        weights["shared_experts.down_proj.weight"],
    )
    routed = {}
    for e in sure + band:
        gate_up, down = provider(e)
        routed[e] = swiglu(gate_up[:i], gate_up[i:], down)

    # ``band`` is in rank order, so the first combination completes the exact top-k.
    candidates = []
    for chosen in itertools.combinations(band, k - len(sure)):
        ids = sure + list(chosen)
        w = scores[ids]
        w = experts.routed_scaling_factor * w / (w.sum() + experts.routing_eps)
        candidates.append(shared + sum(wi * routed[e] for wi, e in zip(w, ids)))
    return candidates, margin


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()


def _as_int32(values):
    return torch.tensor([v - (1 << 32) if v >= 1 << 31 else v for v in values], dtype=torch.int32)


class PersistentMoE:
    """The persistent op over ``moes`` (one ``DeepSeekV41SparseMoeBlock`` per layer, all at TP12 on ``device``).

    The layer table holds each layer's buffer addresses in the row layout the op reads; the op itself is launched
    on layer 0's tensors, whose layouts every layer shares. Call it with ``(layer, row)`` to run that layer on one
    ``[D]`` row, and :meth:`stop` once to end the kernels. ``h2d_mode`` is the H2D sockets' mode: under
    ``DEVICE_PULL`` a host write is a memcpy into pinned memory and the control kernel reads the page over PCIe.
    """

    def __init__(self, device, moes, socket_core, h2d_mode=ttnn.H2DMode.HOST_PUSH):
        self.device = device
        self.hidden = moes[0].hidden
        self.row_bytes = self.hidden * 2
        first = moes[0]
        experts = first.experts
        for moe in moes:
            assert moe.fused_router and moe.fused_shared is not None, "the persistent op runs the whole block"
        assert experts.fused_all_reduce_axes is not None, "the persistent op reduces over the whole mesh"

        rows = []
        for layer, moe in enumerate(moes):
            header = [
                layer,
                moe.fused_shared.gate_up.buffer_address(),
                moe.fused_shared.down.buffer_address(),
                moe.gate.fused_gate_weight.buffer_address(),
                moe.gate.ranking_bias().buffer_address(),
            ]
            header += [0] * (HEADER_BYTES // 4 - len(header))
            rows.append(
                _as_int32(
                    header
                    + [w.buffer_address() for w in moe.experts.gate_up_weights]
                    + [w.buffer_address() for w in moe.experts.down_weights]
                )
            )
        self.layer_table = ttnn.from_torch(
            torch.stack(rows),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        # The row lands here on every core of the grid, replacing the all_gather_for_matmul replica.
        self.replica = ttnn.from_torch(
            torch.zeros(1, 1, REPLICA_GRID.num_cores(), self.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(REPLICA_GRID, [1, self.hidden], ttnn.ShardOrientation.ROW_MAJOR),
            ),
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        page_bytes = HEADER_BYTES + self.row_bytes
        rows_, cols = tuple(device.shape)
        self.h2d = []
        for r in range(rows_):
            for c in range(cols):
                socket = ttnn.H2DSocket(
                    device,
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(r, c), socket_core),
                    ttnn.BufferType.L1,
                    page_bytes,
                    h2d_mode,
                )
                socket.set_page_size(page_bytes)
                self.h2d.append(socket)
        self.d2h = ttnn.D2HSocket(
            device, ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), socket_core), 2 * self.row_bytes
        )
        self.d2h.set_page_size(self.row_bytes)

        shared = first.fused_shared
        self.scratch = ttnn.experimental.deepseek.moe.fused_experts(
            self.replica,
            router_weight=first.gate.fused_gate_weight,
            router_bias=first.gate.ranking_bias(),
            gate_up_weights=experts.gate_up_weights,
            down_weights=experts.down_weights,
            num_experts=experts.top_k,
            intermediate_size=experts.intermediate,
            swiglu_limit=experts.limit,
            top_k=experts.top_k,
            routed_scaling_factor=experts.routed_scaling_factor,
            routing_eps=experts.routing_eps,
            experts_block_size=experts.experts_block_size,
            shared_gate_up_weights=shared.gate_up,
            shared_down_weights=shared.down,
            shared_swiglu_limit=shared.swiglu_limit,
            memory_config=experts.output_memory_config,
            all_reduce_cluster_axes=experts.fused_all_reduce_axes,
            all_reduce_num_links=2,
            all_reduce_topology=CCL_TOPOLOGY,
            persistent_layer_table=self.layer_table,
            persistent_socket_core=socket_core,
            persistent_h2d_socket_addresses=[s.get_config_buffer_address() for s in self.h2d],
            persistent_d2h_socket_address=self.d2h.get_config_buffer_address(),
            persistent_h2d_device_pull=h2d_mode == ttnn.H2DMode.DEVICE_PULL,
        )

    def _send(self, layer: int, row: torch.Tensor):
        header = torch.zeros(HEADER_BYTES // 4, dtype=torch.int32)
        header[0] = _as_int32([layer])[0]
        page = torch.cat([header.view(torch.uint8), row.to(torch.bfloat16).contiguous().view(torch.uint8)])
        for socket in self.h2d:
            socket.write_tensor(page)

    def __call__(self, layer: int, row: torch.Tensor) -> torch.Tensor:
        self._send(layer, row)
        out = torch.zeros(self.hidden, dtype=torch.bfloat16)
        self.d2h.read_tensor(out)
        return out.float()

    def stop(self):
        self._send(STOP_LAYER, torch.zeros(self.hidden))
        ttnn.synchronize_device(self.device)


def _setup(mesh_device):
    """``(device, config, socket_core)``: the TP12 submesh, the V4.1 config, and the persistent op's socket core."""
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    device = mesh_device.create_submesh(ttnn.MeshShape(*SUBMESH), ttnn.MeshCoordinate(0, 0))
    grid = device.compute_with_storage_grid_size()
    # Bottom-right: outside the 12x9 grid the op multicasts over, and away from the reduce senders, which the
    # op picks from the bottom row's left end.
    return device, load_config(_snapshot()), ttnn.CoreCoord(grid.x - 1, grid.y - 1)


def _load_layers(device, config):
    """The ``NUM_LAYERS`` random layers: ``(layers, moes)``, ``layers[i]`` the torch ``(provider, weights)``."""
    layers = []
    moes = []
    for layer in range(NUM_LAYERS):
        start = time.perf_counter()
        provider, weights = _random_layer(config, seed=layer)
        # The random weights are a function of the seed alone, so their converted tensors are cached like
        # real ones: only the first run pays the bfloat4_b conversion of 384 experts per layer.
        cache = WeightCache(CACHE_DIR).sub(f"persistent_test.random{layer}.ffn")

        def loading_provider(x, provider=provider, layer=layer):
            # Called only on a weight-cache miss, once per expert being converted.
            if x % 64 == 0:
                logger.info(f"layer {layer}: converting experts {x}..{x + 63}")
            return provider(x)

        experts = DeepSeekV41PreloadedExperts(
            config, loading_provider, device, dtype=ttnn.bfloat4_b, cache=cache, tp_size=TP_SIZE
        )
        moe = DeepSeekV41SparseMoeBlock(
            config, {k: (lambda v=v: v) for k, v in weights.items()}, device, experts, cache=cache, tp_size=TP_SIZE
        )
        assert moe.fused_router and moe.fused_shared is not None, "the fused TP12 MoE must run router and shared"
        layers.append((provider, weights))
        moes.append(moe)
        logger.info(f"layer {layer} loaded in {time.perf_counter() - start:.1f}s (cache {CACHE_DIR})")
    return layers, moes


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("h2d_mode", [ttnn.H2DMode.HOST_PUSH, ttnn.H2DMode.DEVICE_PULL], ids=["push", "pull"])
@torch.no_grad()
def test_moe_persistent_tp12(mesh_device, h2d_mode):
    device, config, socket_core = _setup(mesh_device)
    layers, moes = _load_layers(device, config)

    torch.manual_seed(0)
    hidden = torch.randn(len(STEPS), config.hidden_size).to(torch.bfloat16).float()
    candidates, margins = zip(
        *(_reference(hidden[s], *layers[layer], config, moes[layer].experts) for s, layer in enumerate(STEPS))
    )

    # The one-shot op on the same rows, before the persistent op takes over the cores.
    hidden_config = width_sharded_l1_config(1, config.hidden_size, device, tile_height=1)
    one_shot = []
    for s, layer in enumerate(STEPS):
        out = moes[layer].decode_static(
            ttnn.from_torch(
                hidden[s].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=hidden_config,
            )
        )
        one_shot.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1).float())
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    moe_op = PersistentMoE(device, moes, socket_core, h2d_mode)
    outs = []
    try:
        for s, layer in enumerate(STEPS):
            outs.append(moe_op(layer, hidden[s]))
            logger.info(f"step {s} layer {layer}: PCC {_pcc(one_shot[s], outs[-1])} vs the one-shot op")
    finally:
        moe_op.stop()

    # Every step in one report: the one-shot op against torch tells a reference / routing mismatch (both ops
    # off torch alike) apart from a persistent-op bug (persistent off the one-shot op).
    report = []
    failed = False
    for s, layer in enumerate(STEPS):
        pccs = [_pcc(c, outs[s]) for c in candidates[s]]
        best = max(range(len(pccs)), key=pccs.__getitem__)
        pcc_torch = pccs[best]
        pcc_one_shot_torch = max(_pcc(c, one_shot[s]) for c in candidates[s])
        pcc_op = _pcc(one_shot[s], outs[s])
        failed |= pcc_torch < PCC or pcc_op < PCC_VS_OP
        routing = "exact top-k" if best == 0 else f"tie-broken top-k {best} of {len(pccs) - 1} alternatives"
        report.append(
            f"step {s} layer {layer}: persistent vs torch {pcc_torch:.5f} ({routing}; exact top-k {pccs[0]:.5f}), "
            f"one-shot vs torch {pcc_one_shot_torch:.5f}, persistent vs one-shot {pcc_op:.6f}, "
            f"top-k margin {margins[s]:.2e}"
        )
    logger.info("\n" + "\n".join(report))
    assert not failed, f"PCC below {PCC} vs torch or {PCC_VS_OP} vs the one-shot op:\n" + "\n".join(report)


def _latency(name, seconds):
    us = torch.tensor(seconds, dtype=torch.float64) * 1e6
    return (
        f"{name:<44} median {us.median():8.1f} us   p10 {us.quantile(0.1):8.1f}   p90 {us.quantile(0.9):8.1f}"
        f"   (n={len(seconds)})"
    )


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash config not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": 32 << 20}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@torch.no_grad()
def test_moe_persistent_tp12_perf(mesh_device):
    """Per-token MoE latency over the interleaved layers, host row in to host row out, four ways:

    - one-shot eager: upload, ``decode_static`` dispatched op by op, read chip 0;
    - one-shot traced: upload, replay that layer's trace, read chip 0 -- the model's decode today;
    - one-shot traced, device only: back-to-back replays with no host I/O, total / count;
    - persistent: the H2D page in, the reduced row out of the D2H socket; with HOST_PUSH and DEVICE_PULL sockets.

    ``MOE_PERF_WARMUP`` / ``MOE_PERF_STEPS`` override the step counts, e.g. to keep a device-profiler capture
    (``TT_METAL_STREAMING_PROFILER=1``) small.
    """
    device, config, socket_core = _setup(mesh_device)
    _, moes = _load_layers(device, config)
    hidden = config.hidden_size
    warmup = int(os.environ.get("MOE_PERF_WARMUP", 20))
    steps = int(os.environ.get("MOE_PERF_STEPS", 270))
    order = [STEPS[i % len(STEPS)] for i in range(warmup + steps)]
    torch.manual_seed(0)
    rows = torch.randn(len(order), hidden).to(torch.bfloat16)
    hidden_config = width_sharded_l1_config(1, hidden, device, tile_height=1)
    inputs = [
        ttnn.from_torch(
            torch.zeros(1, 1, 1, hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=hidden_config,
        )
        for _ in range(NUM_LAYERS)
    ]
    results = []

    def upload(i, layer):
        host = ttnn.from_torch(
            rows[i].reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        ttnn.copy_host_to_device_tensor(host, inputs[layer])

    eager = []
    for i, layer in enumerate(order):
        start = time.perf_counter()
        upload(i, layer)
        out = moes[layer].decode_static(inputs[layer])
        ttnn.to_torch(ttnn.get_device_tensors(out)[0])
        eager.append(time.perf_counter() - start)
        ttnn.deallocate(out)
    results.append(_latency("one-shot eager (upload + ops + readback)", eager[warmup:]))

    tids = []
    trace_outs = []
    for layer in range(NUM_LAYERS):
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        trace_outs.append(moes[layer].decode_static(inputs[layer]))
        ttnn.end_trace_capture(device, tid, cq_id=0)
        tids.append(tid)
    ttnn.synchronize_device(device)

    traced = []
    for i, layer in enumerate(order):
        start = time.perf_counter()
        upload(i, layer)
        ttnn.execute_trace(device, tids[layer], cq_id=0, blocking=False)
        ttnn.to_torch(ttnn.get_device_tensors(trace_outs[layer])[0])
        traced.append(time.perf_counter() - start)
    results.append(_latency("one-shot traced (upload + replay + readback)", traced[warmup:]))

    for layer in order[:warmup]:
        ttnn.execute_trace(device, tids[layer], cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for layer in order[warmup:]:
        ttnn.execute_trace(device, tids[layer], cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    per_replay = (time.perf_counter() - start) / steps
    results.append(f"{'one-shot traced, device only (back-to-back)':<44} mean {per_replay * 1e6:8.1f} us")

    for tid in tids:
        ttnn.release_trace(device, tid)
    for t in trace_outs + inputs:
        ttnn.deallocate(t)
    ttnn.synchronize_device(device)

    out = torch.zeros(hidden, dtype=torch.bfloat16)
    for mode, mode_name in ((ttnn.H2DMode.HOST_PUSH, "push"), (ttnn.H2DMode.DEVICE_PULL, "pull")):
        moe_op = PersistentMoE(device, moes, socket_core, mode)
        persistent = []
        sends = []
        try:
            for i, layer in enumerate(order):
                start = time.perf_counter()
                moe_op._send(layer, rows[i])
                sent = time.perf_counter()
                moe_op.d2h.read_tensor(out)
                persistent.append(time.perf_counter() - start)
                sends.append(sent - start)
        finally:
            moe_op.stop()
        results.append(_latency(f"persistent, H2D {mode_name} (page + loop + D2H row)", persistent[warmup:]))
        results.append(_latency(f"  of which host writes to {len(moe_op.h2d)} H2D sockets", sends[warmup:]))
        results.append(
            _latency("  of which waiting on the D2H row", [p - s for p, s in zip(persistent[warmup:], sends[warmup:])])
        )
        # The next launch's sockets and scratch take the same L1.
        del moe_op
        ttnn.synchronize_device(device)

    logger.info("MoE TP12 per-token latency, layers interleaved " + str(STEPS) + ":\n" + "\n".join(results))
