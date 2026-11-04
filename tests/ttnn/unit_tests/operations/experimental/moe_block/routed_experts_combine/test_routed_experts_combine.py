# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.moe_block.routed_experts_combine on device, bit-exact against the saved reference cases (the
experts + combine, combine_snap 2^-14). Needs the case's mesh and fabric (Galaxy 8x4, FABRIC_2D_TORUS_XY); skips without
reference data ($MOE_BLOCK_REFS) or on another mesh.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

import ttnn

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
OP = "routed_experts_combine"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


harness = _load("moe_block_ref_harness", HERE.parent / "ref_harness.py")
weights_py = _load(
    "flat_routed_expert_weights",
    REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py",
)
mb = ttnn.experimental.moe_block
CASES = harness.list_cases(OP)


def _device_params(case):
    """The case's fabric (meta.json): the block's payload - a row and its trailer in one packet."""
    cfg = json.loads((harness.case_dir(OP, case) / "meta.json").read_text())["config"] if case != "none" else {}
    params = dict(fabric_config=getattr(ttnn.FabricConfig, cfg.get("fabric_config", "FABRIC_2D_TORUS_XY")))
    if "fabric_payload_bytes" in cfg:
        router = ttnn._ttnn.fabric.FabricRouterConfig()
        router.max_packet_payload_size_bytes = cfg["fabric_payload_bytes"]
        params.update(fabric_router_config=router, reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT)
    return params


def _params(cases):
    return [pytest.param((8, 4), _device_params(c), c, id=c) for c in cases]


class _Call:
    """One case on the mesh: the resources, the layer's flat expert (its weights regenerated from the dump's seed) and
    the op's inputs; call() runs the op."""

    def __init__(self, mesh_device, meta, ref):
        cfg = meta["config"]
        R, C = meta["mesh_shape"]
        N = R * C
        self.mesh, self.cfg, self.ref = mesh_device, cfg, ref
        dims = mb.MoEModelDims(
            num_experts=cfg["E"], top_k=cfg["K"], emb_dim=cfg["Hr"], routed_expert_hidden_dim=cfg["I"]
        )
        conf = mb.MoEBlockConfig(chunk_tokens=cfg["chunk_tokens"], combine_snap=cfg["combine_snap"])
        self.res = mb.MoEBlockResources(mesh_device, dims, conf)
        Hr, I, E = cfg["Hr"], cfg["I"], cfg["E"]
        EPC = E // N
        g = torch.Generator().manual_seed(meta["seed"])
        r = lambda *s: torch.randn(*s, generator=g) * 0.02  # noqa: E731  (the saved case's order)
        gids = [list(range(c * EPC, (c + 1) * EPC)) for c in range(N)]
        local = [(r(Hr, I), r(Hr, I), r(I, Hr)) for _ in range(EPC)]  # (every chip the same local experts)
        # (the saved case's layout fields the op still has)
        saved = cfg.get("flat_layout") or {}
        layout = weights_py.flat_layout({k: v for k, v in saved.items() if k in weights_py.FLAT_LAYOUT_FIELDS})
        self.layer = weights_py.FlatRoutedExpert(
            mesh_device, [local] * N, m=cfg["m"], H=Hr, I=I, gids=gids, n_global=E, act=cfg["activation"], pin=1,
            layout=layout,
        )  # fmt: skip
        rm = lambda name, dt: harness.to_mesh(ref, name, mesh_device, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT)  # noqa
        self.args = [
            rm("xg", ttnn.bfloat16),
            rm("counts", ttnn.uint32),
            rm("regions", ttnn.uint32),
            rm("token_index", ttnn.uint32),
            rm("table", ttnn.uint32),
            rm("plan", ttnn.uint32),
            self.layer.gidx,
            self.layer.w_gu,
            self.layer.w_d,
            self.res,
        ]
        combine = mb.CombineConfig()
        combine.n_acc = cfg["n_acc"]
        combine.mode = mb.CombineMode.AccOnPorts if cfg["acc_on_ports"] else mb.CombineMode.PortsApart
        combine.arena_bytes = cfg["arena_bytes"]
        self.kw = dict(
            max_tokens_per_expert=cfg["m"],
            activation=weights_py.ACTS[cfg["activation"]],
            layout=layout,
            combine=combine,
        )

    def call(self, arena=None):
        return mb.routed_experts_combine(*self.args, arena=arena, **self.kw)

    def arena_spec(self):
        """The block's shared arena (here: the op's need alone)."""
        cfg = self.cfg
        rows_cap = int(self.ref[(0, 0)]["table"].shape[-2]) * 32
        return mb.routed_experts_combine_arena_spec(
            self.res, rows_cap=rows_cap, max_tokens_per_expert=cfg["m"], activation=self.kw["activation"],
            layout=self.kw["layout"], combine=self.kw["combine"],
        )  # fmt: skip

    def close(self):
        self.layer.close()
        self.res.release()


def _case(mesh_device, case):
    if case == "none":
        pytest.skip(f"no saved {OP} case ($MOE_BLOCK_REFS)")
    meta, ref = harness.load_case(OP, case)
    if list(mesh_device.shape) != meta["mesh_shape"]:
        pytest.skip(f"case {case} needs a {meta['mesh_shape']} mesh")
    return meta, ref


@pytest.mark.parametrize(
    "mesh_device, device_params, case", _params(CASES or ["none"]), indirect=["mesh_device", "device_params"]
)
def test_routed_experts_combine(mesh_device, device_params, case):
    meta, ref = _case(mesh_device, case)
    run = _Call(mesh_device, meta, ref)
    try:
        # a miss, hits on the same addresses (the per-call zeroing and counters), then hits on the block's arena
        # (another address: the patches of the moved source only)
        shared = ttnn.allocate_tensor_on_device(run.arena_spec(), mesh_device)
        for arena in (None, None, None, shared, shared):
            out = run.call(arena)
            got = harness.from_mesh(out, tuple(meta["mesh_shape"]))
            harness.assert_bit_exact({ch: {"y_own": t} for ch, t in got.items()}, ref, ["y_own"])
        ttnn.deallocate(shared)
    finally:
        run.close()
