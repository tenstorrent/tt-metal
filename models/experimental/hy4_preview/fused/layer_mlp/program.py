# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MLP half of a Hy4-preview decoder layer for 8 token rows as ONE program on the 8x4 mesh (PLAN.md has the
plan, the cores and the wait graph; this builds it).

    root chip, 16 site cores   iHC pass (fused/ihc_site): post of the sublayer before (optional) + pre of the MLP
                               site, the RMSNorm on the same cores, the result x [8, 6144] BF16 to the sender
    X1'                        the root's sender -> every chip's hub x buffer (fused/layer_common/transport.py)
    every chip                 the MoE piece as it is (fused/moe RoutedExperts with the router on the chip): its
                               1/32 share of the union's experts and of the shared expert, streamed from DRAM
    X6                         the 32 partial outputs summed to the root in two stages; the sum F [8, 6144] BF16 lands
                               in the site cores' F tensor
    (the next program)         its leading iHC pass does the post of this F; `post_program` is that pass alone

Pieces and what is adapted here, without editing them:
  fused/moe      two RoutedExperts: one for 31 chips (mode "rect", 6 gate|up and 4 down cores per bank) and one for
                 the root (4 gate|up cores per bank; its hub, finishers and router cores moved so that the site fits),
                 sharing the hub's tensors. The hub kernel is replaced by kernels/mlp_hub.cpp, which includes the
                 piece's hub_router.cpp. The cores' constants and the router's bias are written here with a
                 replicating mesh mapper; the weight streams are one mesh tensor with a shard per chip.
  fused/ihc_site IhcSite.program(norm_out=...) on the root: 16 cores as 2 columns x 8 rows, CB ids from 32.
  fused/collective  its sender kernel through layer_common.sender (explicit ops).

State (2026-11-19, board): runs on all 32 chips with checkpoint weights of layers 1 to 5 (PLAN.md section 10 has the
numbers and the logs): 188 / 204 / 244 / 276 / 302 us per call traced for unions of 8 / 16 / 32 / 48 / 64 experts
against 1752 us for the ttnn form of the same stages; picks as a host router's on 223 of 224 rows (one near tie);
F against exact arithmetic on the BFP4 weights PCC 0.99998, gain 0.986 to 0.988 (LoFi custom_mm), against the
reference PCC 0.9917 to 0.9991 (the BFP4 format's own). Layer 0's dense FFN (dense=True) passes one step on 32 chips
(F against the reference PCC 0.99997, gain 0.988); its latency is not measured. Not built: the post-mix of this
program's F inside the same program (it is the next program's leading pass, post_program here).

Precision: the iHC pass and the norm are fp32 (SFPU); x travels as BF16; the MoE piece's matmuls are custom_mm at LoFi
on BFP4 weights (two BFP4 terms for the shared expert); the partial outputs are BF16, summed in Float32 and rounded
to BF16 once per reduce stage; F is BF16.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.experimental.hy4_preview.fused.ihc_site.op import IhcSite, NormOut, SiteRef
from models.experimental.hy4_preview.fused.ihc_site import layout as ihc_layout
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe import prepare
from models.experimental.hy4_preview.fused.moe import program as moe
from models.experimental.hy4_preview.reference.ihc import IHCConfig

KERNEL_DIR = Path(__file__).parent / "kernels"
E = 256
DENSE_SLICES = 9  # layer 0's FFN: 18432 intermediate channels as 9 slices of 2048
HIDDEN = L.HIDDEN
PAYLOAD = 2 * L.ROWS * HIDDEN  # [8, 6144] BF16

# Cores (PLAN.md section 2). The MoE piece's rectangular placement leaves (5, 6..8), (6, 8), (6, 9) free.
SENDER = (6, 9)
REDUCE = [(5, 6), (5, 7), (5, 8), (6, 8)]
ROOT_REDUCE = [(4, 6), (4, 7), (4, 8), (4, 9)]
ROOT_FINISHERS = [(4, y) for y in range(6)]
ROOT_ROUTER = [(11, y) for y in range(8)]
SITE = dict(num_cores=16, origin=(5, 0), per_row=2, cb_base=32)


def is_mock() -> bool:
    return tp.is_mock()


def replicated(mesh_device, data, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(data, dtype=dtype, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device))


def load_streams(mesh_device, layer, split="tp32", root=prepare.DEFAULT_OUT):
    """The layer's two weight streams as mesh tensors with one shard per chip (fused/moe/prepare.py's files)."""
    chips = mesh_device.get_num_devices()
    out = []
    for which in (0, 1):
        shards = [ttnn.load_tensor(str(prepare.stream_paths(root, layer, split, chip)[which])) for chip in range(chips)]
        rows = shards[0].shape[-2]
        host = ttnn.from_host_shards(shards, mesh_device.shape)
        out.append(ttnn.to_device(host, mesh_device, memory_config=L.stream_memory_config(rows)))
    return out


class MlpLayer:
    """The program and its persistent state. ``chips``: the chips that take part (default all 32; fewer only for
    bring-up: the sum is then over those chips' shares). ``root_gate_up``: gate|up cores per bank on the root."""

    def __init__(
        self,
        mesh_device,
        root=(0, 0),
        chips=None,
        root_gate_up=4,
        dense=False,
        timing=False,
        wait_limit=30_000_000,
        moe_defines=(),
        moe_kwargs=None,
        layer_args=False,
        site=None,
        site_tensors=None,
        transport=None,
        hub_tensors=None,
    ):
        """For a caller that chains layers (fused/model); every default is the half as it was:
        layer_args: ONE MlpLayer serves many layers: the expert and router streams' addresses and the router's bias
            travel as runtime arguments of each launch (RoutedExperts(stream_rt, bias_rt)), so descriptors built
            after different set_layer calls are the same program; nothing per layer is left in L1 but the norm's
            gamma, which set_layer(gamma=None) takes from the site table (IhcSite.site_table(gammas=...)).
        site / site_tensors: an IhcSite and its {"streams", "f", "y", "xn"} tensors that another half already has
            (the root's site state is one for the whole layer), instead of making them.
        transport: a function (mesh_device, topology, hub, x) -> Transport, in place of a transport of this half's
            own (for one on another half's arena).
        hub_tensors: RoutedExperts(hub_tensors=...) for the chips' instance (the root's takes the same ones)."""
        self.mesh = mesh_device
        self.topo = tp.Topology(tuple(mesh_device.shape), root, chips)
        self.timing = timing
        # dense: layer 0's FFN as DENSE_SLICES always-on plain slots of two BFP4 terms each (fused/moe/prepare.py
        # dense_experts), no router: the slots are the same on every step and are written to the hubs' meta buffers
        self.dense = dense
        share = L.tp_share(32)
        defines = list(moe_defines) + ([("HY4_TIMING", "1")] if timing else [])
        # the MoE piece's measured best: rectangular placement, 4 down cores per bank, the router cores cutting
        # their 32 experts to each row's 8 best so that the hub merges 8 short lists (25 us of software against 82)
        kwargs = dict(
            router=True,
            num_experts=E,
            n_plain=1,
            plain_id0=E,
            plain_terms=2,
            mode="rect",
            down_per_bank=4,
            router_candidates=True,
        )
        if dense:
            kwargs = dict(router=False, plain_terms=2, mode="rect", down_per_bank=4)
        kwargs.update(moe_kwargs or {})
        self.layer_args = layer_args
        if layer_args:
            kwargs.update(stream_rt=True, bias_rt=not dense)
        self.op = moe.RoutedExperts(mesh_device, share, defines=defines, hub_tensors=hub_tensors, **kwargs)
        self._hub_names = ("x", "out", "meta", "status") + (
            ("scores", "bias", "picks", "cand") if self.op.router else ()
        )
        # (the root's instance takes the first one's hub tensors: given to it at once, nothing to free)
        root_tensors = {name: getattr(self.op, name) for name in self._hub_names}
        self.op_root = moe.RoutedExperts(
            mesh_device, share, defines=defines, hub_tensors=root_tensors, **dict(kwargs, gate_up_per_bank=root_gate_up)
        )
        self._adopt_root()
        hub = (self.op.hub.x, self.op.hub.y)
        if transport is not None:
            self.t = transport(mesh_device, self.topo, hub, self.op.x)
        else:
            self.t = tp.Transport(
                mesh_device,
                self.topo,
                hub,
                SENDER,
                REDUCE,
                PAYLOAD,
                x=self.op.x,
                root_reduce_cores=ROOT_REDUCE,
                wait_limit=wait_limit,
                timing=timing,
            )
        self.cfg = IHCConfig(dim=HIDDEN)
        self.site = site if site is not None else IhcSite(mesh_device, self.cfg, f_dtype=ttnn.bfloat16, **SITE)
        self._check_cores()
        given = site_tensors or {}
        self.xn = given["xn"] if "xn" in given else self.site.empty_rows(ttnn.bfloat16)
        self.y = given["y"] if "y" in given else self.site.empty_rows(ttnn.float32)
        self.f = given["f"] if "f" in given else self.site.empty_rows(ttnn.bfloat16)
        self.streams = (
            given["streams"] if "streams" in given else self.site.streams_to_device(torch.zeros(L.ROWS, 4, HIDDEN))
        )
        self.gamma = None
        self.table = None
        self.w_gate_up = self.w_down = None
        self._programs = {}
        ttnn.synchronize_device(mesh_device)

    # ------------------------------------------------------------------------------------------ construction
    def _adopt_root(self):
        """The root's MoE instance: its hub on the common hub core, its finishers and router cores out of the
        site's way, and the first instance's hub tensors in place of its own."""
        op, root = self.op, self.op_root
        for name in self._hub_names:
            assert getattr(root, name) is getattr(op, name)
        cc = lambda cores: [ttnn.CoreCoord(*c) for c in cores]
        root.hub = op.hub
        root.fin = cc(ROOT_FINISHERS)[: root.NF]
        root.r_cores = cc(ROOT_ROUTER)[: len(root.r_cores)]

    def _check_cores(self):
        xy = lambda cores: {(c.x, c.y) for c in cores}
        hub = {(self.op.hub.x, self.op.hub.y)}
        for name, op, extra in (
            ("chip", self.op, set(REDUCE) | {SENDER}),
            ("root", self.op_root, set(ROOT_REDUCE) | {SENDER} | xy(self.site.cores)),
        ):
            used = xy(op.g_kernel_cores) | xy(op.d_kernel_cores) | xy(op.fin) | xy(op.r_cores)
            assert not (used & hub) and not (used & extra) and not (hub & extra), f"{name}: core roles overlap"
            assert len(xy(op.fin)) == op.NF and len(xy(op.r_cores)) == (op.router_cols if op.router else 0)

    def n_down(self, coord):
        return len((self.op_root if tuple(coord) == self.topo.root else self.op).all_d)

    # ------------------------------------------------------------------------------------------ weights
    def set_layer(
        self, layer, sites, site_index, gamma, router_weight=None, router_bias=None, streams=None, table=None
    ):
        """The layer's data: ``sites`` [(fn, base, scale), ...] the iHC sites in the DRAM table (``site_index``: the
        MLP site of this layer), ``gamma`` the pre-norm's weight [6144], the router's weight [256, 6144] and bias
        [256], and the expert weight streams (default: the prepared files of ``layer``).
        table: a site table that exists (IhcSite.site_table) in place of ``sites``; gamma None: that table holds the
        norms' weights too. With layer_args the object can be given one layer after another: ``layer_state()``
        returns what a later ``use(state)`` puts back before that layer's descriptors are built."""
        self.site_index = site_index
        self.table = table if table is not None else self.site.site_table(sites)
        self.gamma = self.site.gamma_to_device(gamma) if gamma is not None else None
        op = self.op
        if self.dense:
            ids = [2 * j for j in range(DENSE_SLICES)]  # slice j: stream entries 2 j and 2 j + 1 (its two terms)
            meta = L.meta_words(ids, torch.ones(L.ROWS, DENSE_SLICES), n_plain=DENSE_SLICES)
            ttnn.copy_host_to_device_tensor(replicated(self.mesh, meta, ttnn.uint16), op.meta)
        else:
            self._set_router(router_weight, router_bias)
        self.w_gate_up, self.w_down = streams if streams is not None else load_streams(self.mesh, layer)
        router_address = 0 if self.dense else op.w_router.buffer_address()
        for one in (self.op, self.op_root):  # the cores' constants: hand-over words zero, roles, stream addresses
            words = one._core_words(self.w_gate_up, self.w_down).to(torch.int32)
            ttnn.copy_host_to_device_tensor(replicated(self.mesh, words, ttnn.uint32), one.sync)
            one._core_words_key = (
                ("stream_rt",)
                if one.stream_rt
                else (self.w_gate_up.buffer_address(), self.w_down.buffer_address(), router_address)
            )
        self._programs = {}
        ttnn.synchronize_device(self.mesh)

    def _set_router(self, router_weight, router_bias):
        op = self.op
        op.w_router = ttnn.from_torch(
            router_weight.float().T.contiguous()[None, None],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=L.stream_memory_config(L.K_TILES * L.TILE),
        )
        q = torch.round(router_bias.double() * (1 << 30)).to(torch.int32).reshape(1, E)
        ttnn.copy_host_to_device_tensor(replicated(self.mesh, q, ttnn.int32), op.bias)
        self.op_root.w_router = op.w_router
        op.bias_values = self.op_root.bias_values = q.flatten().tolist()

    # ------------------------------------------------------------------------------------------ programs
    def _norm_out(self):
        root = self.site.cores[0]
        sender, staging, ready, bit = self.t.x_source_args()
        return NormOut(
            gamma=self.gamma,
            xn=self.xn,
            sender=sender,
            staging=staging,
            ready=ready,
            ready_bit=bit,
            gather=self.t.word_address(tp.W_GATHER),
            status=self.t.word_address(tp.W_STATUS),
            wait_limit=self.t.wait_limit,
            stamps=self.t.word_address(tp.W_STAMP) if self.timing else 0,
        )

    def _chip_program(self, coord, post):
        topo, t = self.topo, self.t
        root = coord == topo.root
        op = self.op_root if root else self.op
        desc, io = op.descriptor(self.w_gate_up, self.w_down)
        hub = desc.kernels[0]
        assert str(hub.kernel_source).endswith("hub.cpp" if self.dense else "hub_router.cpp"), hub.kernel_source
        hub.kernel_source = str(KERNEL_DIR / "mlp_hub.cpp")
        hub.defines = list(hub.defines) + t.defines + ([("HY4_MLP_GIVEN", "1")] if self.dense else [])
        hub.compiler_include_paths = [str(moe.KERNEL_DIR), str(tp.KERNEL_DIR)]
        if self.dense:
            rt = [op.x.buffer_address(), op.meta.buffer_address(), op.status.buffer_address()]
        else:
            rt = [
                op.x.buffer_address(),
                op.status.buffer_address(),
                op.scores.buffer_address(),
                op.bias.buffer_address(),
            ]
            rt += [op.picks.buffer_address(), op.cand.buffer_address()]
            if op.bias_rt:  # the bias follows the six addresses; this half's own arguments move behind it
                assert len(rt) == op.BIAS_ARG
                rt += op.bias_args()
                hub.defines = list(hub.defines) + [("HY4_MLP_BASE", str(len(rt)))]
        rt += [self.n_down(coord), op.out.buffer_address()] + t.feed_args(coord)
        hub.runtime_args[op.hub.x][op.hub.y] = rt
        if topo.is_head(coord):
            targets = None
            if root:
                site = [(c.x, c.y) for c in self.site.cores]
                per = len(site) // t.parts
                targets = [site[k * per : (k + 1) * per] for k in range(t.parts)]
            t.add_reduce(desc, coord, targets, self.f.buffer_address())
        if root:
            s_io, s_desc = self.site.program(
                self.streams,
                f=self.f if post else None,
                params=SiteRef(self.table, self.site_index),
                y=self.y,
                norm_out=self._norm_out(),
            )
            desc.kernels = list(desc.kernels) + list(s_desc.kernels)
            desc.cbs = list(desc.cbs) + list(s_desc.cbs)
            desc.semaphores = list(desc.semaphores) + list(s_desc.semaphores)
            io = io + s_io
        t.add_sender(desc, coord)
        return desc, io

    def layer_state(self):
        """layer_args: what set_layer left, to be put back with use() before another layer's descriptors are built."""
        op = self.op
        return dict(
            site_index=self.site_index,
            table=self.table,
            gamma=self.gamma,
            w_gate_up=self.w_gate_up,
            w_down=self.w_down,
            w_router=None if self.dense else op.w_router,
            bias_values=None if self.dense else list(op.bias_values),
        )

    def use(self, state):
        assert self.layer_args, "one MlpLayer for many layers needs layer_args"
        self.site_index, self.table, self.gamma = state["site_index"], state["table"], state["gamma"]
        self.w_gate_up, self.w_down = state["w_gate_up"], state["w_down"]
        for one in (self.op, self.op_root):
            one.w_router, one.bias_values = state["w_router"], state["bias_values"]
        self._programs = {}

    def program(self, post=False):
        """(io tensors, MeshProgramDescriptor). post: the root's leading iHC pass also does the post-mix of the F in
        the site's F tensor (the sublayer before) with the coefficients of the pre before."""
        if post not in self._programs:
            mesh_desc = ttnn.MeshProgramDescriptor()
            io = []
            for coord in self.topo.chips:
                desc, chip_io = self._chip_program(coord, post)
                io += chip_io
                mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
            io = self.t.io() + io + [self.f]
            unique = []
            for tensor in io:
                if not any(tensor is u for u in unique):
                    unique.append(tensor)
            self._programs[post] = (unique, mesh_desc)
        return self._programs[post]

    def post_program(self):
        """The post-mix of the F this program left, alone, on the root: what the next program's leading iHC pass
        does first. (io, MeshProgramDescriptor)."""
        if "post" not in self._programs:
            io, desc = self.site.program(self.streams, f=self.f)
            mesh_desc = ttnn.MeshProgramDescriptor()
            root = ttnn.MeshCoordinate(*self.topo.root)
            mesh_desc[ttnn.MeshCoordinateRange(root, root)] = desc
            self._programs["post"] = (io, mesh_desc)
        return self._programs["post"]

    def run(self, post=False):
        io, desc = self.program(post)
        (ttnn.experimental.prepare_generic_op if is_mock() else ttnn.generic_op)(io, desc)

    def run_post(self):
        io, desc = self.post_program()
        (ttnn.experimental.prepare_generic_op if is_mock() else ttnn.generic_op)(io, desc)

    # ------------------------------------------------------------------------------------------ host side
    def _root_matrix(self, tensor):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[self.topo.chip(self.topo.root)]).float()[0, 0]

    def set_streams(self, x):
        """[8, 4, 6144] -> the root's streams (written to every chip: only the root's are used)."""
        pages = ihc_layout.streams_to_pages(x).reshape(1, 1, 32, -1)
        ttnn.copy_host_to_device_tensor(replicated(self.mesh, pages, ttnn.float32, ttnn.TILE_LAYOUT), self.streams)

    def set_f(self, f):
        """[8, 6144] -> the site's F tensor (the output of the sublayer before, for program(post=True))."""
        pages = ihc_layout.rows_to_pages(f).reshape(1, 1, 32, -1)
        ttnn.copy_host_to_device_tensor(replicated(self.mesh, pages, ttnn.bfloat16, ttnn.TILE_LAYOUT), self.f)

    def read_streams(self):
        return ihc_layout.pages_to_streams(self._root_matrix(self.streams))

    def read_root(self):
        """What the root's site holds: y [8, 6144] fp32, 1 / RMS [8], x [8, 6144] (the BF16 it broadcast), F [8, 6144]."""
        rows = lambda t: ihc_layout.pages_to_rows(self._root_matrix(t))
        stat = ihc_layout.stat_from_page(self._root_matrix(self.site.stat)[:, :32])
        return {"y": rows(self.y), "inv_rms": stat, "x": rows(self.xn), "f": rows(self.f)}

    def read_chip(self, coord):
        """A chip's hub buffers: x as it arrived [8, 6144], its partial output [8, 6144], the MoE piece's status
        words, the router's scores [8, 256] (sigmoid, FP32), picks [8, 8] and weights [8, 8]."""
        chip = self.topo.chip(coord)
        dev = lambda t: ttnn.to_torch(ttnn.get_device_tensors(t)[chip])
        out = {
            "x": L.from_row_tiles(dev(self.op.x).float()),
            "out": L.from_row_tiles(dev(self.op.out).float()),
            "status": [int(v) & 0xFFFFFFFF for v in dev(self.op.status).flatten().tolist()],
        }
        if self.dense:
            return out
        picks = dev(self.op.picks).to(torch.int32) & 0xFFFF
        return {
            **out,
            "scores": L.from_row_tiles(dev(self.op.scores).float()),
            "ids": picks[:, :8],
            "weights": picks[:, 8:].contiguous().to(torch.int16).view(torch.bfloat16).float(),
        }

    def status(self) -> dict:
        """Everything that is not at rest after a run: the transport's words (Transport.status) and the MoE piece's
        status words 1..5 of every chip. Empty when the run completed."""
        out = dict(self.t.status())
        ttnn.synchronize_device(self.mesh)
        tensors = ttnn.get_device_tensors(self.op.status)
        for coord in self.topo.chips:
            words = [int(v) & 0xFFFFFFFF for v in ttnn.to_torch(tensors[self.topo.chip(coord)]).flatten().tolist()[:6]]
            if any(words[1:6]):
                out[(coord, "moe")] = [hex(w) for w in words[:6]]
        return out

    def reset(self):
        """After a run that left a status: the transport's words and both MoE instances' hand-over words to zero."""
        self.t.reset()
        for one in (self.op, self.op_root):
            words = one._core_words(self.w_gate_up, self.w_down).to(torch.int32)
            ttnn.copy_host_to_device_tensor(replicated(self.mesh, words, ttnn.uint32), one.sync)
        ttnn.synchronize_device(self.mesh)

    def l1_bytes(self) -> int:
        return ttnn.get_memory_view(self.mesh, ttnn.BufferType.L1).total_bytes_allocated_per_bank
