# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""tencent/Hy4-preview decode on the 8x4 mesh that generates from an EMPTY context (PLAN.md section 54):

    model = DecodeModel(mesh_device, layers=78)
    model.reset_row(r)                                # row r's conversation starts empty: its next position is 0
    ids = model.step(tokens, positions)               # 8 entries each; None = an idle row

A step is ONE pass of the fused programs over the 8 token rows (program.py Stack with set_indexer, ends_fused.py):

    IN | per layer: P1 [| scan | merge | cut] | P2 | MLP | HEAD

The host gives token ids and positions and reads token ids. Nothing of the selection, the index query or the cache
comes from the host: every layer's P1 appends the row's latent row at the row's own position (its own
round-to-nearest pack: AttnLayer(append="nearest"), so NO latent_scale), a full-indexer layer's P1 also appends the
index key and writes the scan's query, the scan / merge / cut make the selection on the chips, and a shared layer
reads the selection the full layer before it left. There is no prefill stack in this form: a prompt is fed through
``step`` token by token (the caller ignores the outputs until the last prompt token).

Storage: two slot-major DRAM tensors per chip in the model's striping (position p on chip (p // 128) % 32 at stripe row
(p // 4096) * 128 + p % 128; fused/kv_store frames.py), zeroed on the device at start (the prefill stack's
DRAMZeroFill): the latent rows [8 x layers, 1, R, 656] and the index keys [8 x full layers, 1, R, 128], R = 256 x
(max_positions / 8,192) stripe rows a slot. ``reset_row`` writes nothing: a row only ever reads positions below its
own next position, and each of those was appended by one of its steps since the reset.

Per step from the host (in place, DRAM or L1 tensors the programs name): the 8 token ids, the rows' positions
(0xFFFFFFFF for an idle row: no append writes), the RoPE tiles of the positions, the scan's context lengths
(position + 1; 0 for an idle row: nothing is selected, its attention is the sink alone). An idle row still runs
through every program on token 0 and its output is dropped.

Behind the prefill stack (``stores=``): the two stores are the prefill stack's own cache tensors (its packed latent rows
and its index keys have this layout: slot = user x layers + layer, the same stripe), not allocated or zeroed here. A row
whose slots the prefill stack filled with a prompt of N tokens is taken over with ``adopt_row(row, N - 1)`` and its
first step feeds the LAST PROMPT TOKEN at position N - 1 (tt/prefill_bridge.py "What a prefill hands over": the step
rewrites row N - 1 and gives the first new token). The prefill stack's pack cuts the latent values toward zero where
this model's own append rounds, so such a model is built with ``append="truncate", latent_scale=1.0430`` (every row of a
context then has the same bias and the one scalar in the weights answers it) until the stack's pack rounds.

The RoPE pairing is the process's (HY4_ROPE_PAIRING, reference/attention.py rope_pairing): this checkpoint's is
"interleaved", and the constructor refuses any other (the weight streams, the tables and the stored rows' order all
follow it).
"""

from __future__ import annotations

import time

import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.kv_store import device as kv_device
from models.experimental.hy4_preview.fused.kv_store import frames as kv_frames
from models.experimental.hy4_preview.fused.model import ends_fused as EF
from models.experimental.hy4_preview.fused.model.program import ROWS, Stack, is_mock
from models.experimental.hy4_preview.fused.projections import layout as L
from models.experimental.hy4_preview.fused.projections import prepare as PP
from models.experimental.hy4_preview.fused.sparse_mla import layout as SL
from models.experimental.hy4_preview.reference.attention import AttentionConfig, rope_cos_sin, rope_pairing
from models.experimental.hy4_preview.tt import frames

IDLE = 0xFFFFFFFF  # an idle row's position (AttnLayer.set_positions)
MERGE_TIMEOUT = 2  # fused/merges topk_merge FLAG_TIMEOUT
MESH = (8, 4)
TRACE_REGION = 64 << 20  # what the tests open the mesh with


def mesh_requirements() -> dict:
    """What the 8x4 mesh must be opened with for a DecodeModel (what tests/test_decode_model.py's fixture and
    DecodeModel.open do), as data: a caller that opens the mesh itself (a vLLM plugin class) passes the same.
    Everything not named is the device's default (worker_l1_size; dispatch cores: type WORKER, axis COL, which is
    what tt_py_test_utils_common.device_params.get_updated_device_params gives on Blackhole without a
    fabric_tensix_config)."""
    from models.experimental.hy4_preview.tt import prefill_bridge as pb

    return {
        "mesh_shape": MESH,
        "mesh_device": "BHGLX",  # (models/common/device_utils.py: 32 Blackhole chips; the tests here give the shape)
        "fabric_config": "FABRIC_2D_TORUS_XY",
        "reliability_mode": "RELAXED_INIT",
        "fabric_tensix_config": "DISABLED",
        "fabric_udm_mode": "DISABLED",
        "fabric_manager": "DEFAULT",
        "fabric_max_packet_payload_size_bytes": pb.fabric_payload_size(),  # (12,352: the sender's ops are sized by it)
        "l1_small_size": pb.SHARED_L1_SMALL,  # 32,768
        "trace_region_size": TRACE_REGION,  # (with trace=True; a step's trace is far smaller)
        "worker_l1_size": None,
        "dispatch_core_type": "WORKER",
        "dispatch_core_axis": "COL",
    }


def zero_fill(tensor):
    """Zeros in every page of a DRAM tensor, written by the chips (models/demos/deepseek_v3_b1 DRAMZeroFill: what
    the prefill stack zeroes its caches with). On the mock the program is compiled and not run."""
    from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

    if not is_mock():
        return DRAMZeroFill.op(tensor)
    run = ttnn.generic_op
    ttnn.generic_op = ttnn.experimental.prepare_generic_op
    try:
        DRAMZeroFill.op(tensor)
    finally:
        ttnn.generic_op = run


class DecodeModel:
    rows: int = ROWS  # rows per pass
    max_positions: int  # tokens one row can hold
    supports_logits: bool = False  # (an instance built with keep_logits=True: step(.., want_logits=True))
    # reset_row(row, keep=k): the row goes back to position k. By the storage's rule (a row reads only positions
    # below its next one; the scan masks by the context length; every full layer's cut rewrites the selection table
    # before any P2 of the step reads it) nothing has to be undone. Shown on the board at 6 layers (tests/
    # test_decode_model.py test_rewind, log mf_dm_L6_rw_tr: the continuation after a rewind equals a fresh feed in
    # every bit of the logits; PLAN.md sections 58, 59).
    supports_rewind: bool = True

    def __init__(
        self,
        mesh_device,
        layers=78,
        max_positions=65_536,
        trace=False,
        keep_logits=False,
        fidelity="recommended",
        moe_fidelity="p02",
        wait_limit=30_000_000,
        append="nearest",
        zero=True,
        check_every=64,
        slow_pass_s=1.0,
        mlp_split="tp32",
        latent_scale=None,
        stores=None,
    ):
        """layers: the checkpoint's first ``layers`` layers with the model's head (78: the model; fewer: the cut
        models the reference's goldens are recorded for). max_positions: rounded up to frames of 8,192.
        trace: the step's programs as ONE trace (the device must be opened with a trace region):
        the first step runs program by program (it compiles), at the second the programs are captured and the step
        is the capture's first replay, every later step is a replay.
        keep_logits: HEAD also writes the step's logits to a DRAM tensor (read_logits; step(want_logits=True)).
        check_every, slow_pass_s: ``status()`` is read, and a status that is not at rest raises out of ``step``,
        after every pass that took longer than slow_pass_s on the host and after every check_every-th pass (0:
        only the slow ones). A bounded wait that ran out leaves a status and wrong data and takes seconds (its
        bound is 30,000,000 polls); one status() read costs 0.42 s on the host (32 chips' words, measured:
        mf_dm_L6_rw_tr), so it is not read after every pass. A stuck program raises by the operation timeout.
        mlp_split: the MoE layers' expert layout (Stack(mlp_split=)): "tp32", or "tp8g4" (a mesh column owns 64
        routed experts, 1/8 of each on each of its 8 chips); the prepared streams of that split must be on disk
        (fused/model/convert.py --split). Layer 0's dense FFN is tp32 either way.
        stores: (latent rows, index keys), two device tensors in this model's layout that someone else allocated and
        filled (the prefill stack's caches for 8 users of max_positions positions); nothing is zeroed. Rows are
        taken over with adopt_row.
        latent_scale: AttnLayer(latent_scale=): the scalar for rows a truncating pack wrote; only with
        append="truncate" (the rows this model appends are then cut the same way)."""
        assert (
            rope_pairing() == "interleaved"
        ), "this checkpoint's RoPE pairing is interleaved: HY4_ROPE_PAIRING=interleaved"
        assert append in ("nearest", "truncate")
        t0 = time.perf_counter()
        self.mesh, self.layers, self.traced = mesh_device, int(layers), bool(trace)
        self.supports_logits, self.check_every, self._opened = bool(keep_logits), int(check_every), False
        self.slow_pass_s = float(slow_pass_s)
        self.mlp_split = mlp_split
        self.full = [n for n in range(self.layers) if PP.is_full(n)]
        self.frames = -(-int(max_positions) // kv_frames.FRAME_TOKENS)
        assert 1 <= self.frames <= kv_frames.TABLE_FRAMES
        self.max_positions = self.frames * kv_frames.FRAME_TOKENS
        self.acfg = AttentionConfig()
        self.st = st = Stack(
            mesh_device,
            fidelity=fidelity,
            moe_fidelity=moe_fidelity,
            wait_limit=wait_limit,
            attn_kwargs=dict(append=append, **({} if latent_scale is None else {"latent_scale": float(latent_scale)})),
            mlp_split=mlp_split,
        )
        # the two stores (allocated here, zeroed after every L1 tensor of the step exists)
        lat = kv_frames.Kind(
            "latent", self.layers, kv_frames.LATENT.width, kv_frames.LATENT.shard_bytes, kv_frames.LATENT.row_stride
        )
        idx = kv_frames.Kind("index", len(self.full), kv_frames.INDEX.width, kv_frames.INDEX.shard_bytes, 0)
        self.own_stores = stores is None
        if self.own_stores:
            self.kv = kv_device.allocate(mesh_device, kv_frames.Geometry.contiguous(lat, self.frames), ROWS)
            self.keys = kv_device.allocate(mesh_device, kv_frames.Geometry.contiguous(idx, self.frames), ROWS)
        else:
            self.kv, self.keys = stores
            for tensor, kind in ((self.kv, lat), (self.keys, idx)):
                geometry = kv_frames.Geometry.contiguous(kind, self.frames)
                assert tuple(tensor.shape) == geometry.tensor_shape(ROWS), (
                    f"the {kind.name} store handed in is {tuple(tensor.shape)}; {ROWS} rows of {self.max_positions} "
                    f"positions over {kind.layers} layers are {geometry.tensor_shape(ROWS)}"
                )
                kv_device.check_layout(tensor, geometry, ROWS)
        st.set_indexer(self.keys, len(self.full))
        self._load()
        self.cfg = SL.SparseMlaConfig(rows_per_slot=self.frames * kv_frames.FRAME_ROWS, num_layers=self.layers)
        st.set_cache(self.cfg, self.kv)
        # the ends: the embedding table, the LM head's streams
        table, head_w = PP.checkpoint_tensors(["model.embed_tokens.weight", "lm_head.weight"])
        self.vocab = int(table.shape[0])
        assert self.vocab == EF.VOCAB_PER_CHIP * mesh_device.get_num_devices(), self.vocab
        self.ein = EF.EndsIn(st, table)
        stream = EF.head_stream(mesh_device, lambda c: head_w[c * EF.VOCAB_PER_CHIP : (c + 1) * EF.VOCAB_PER_CHIP])
        self.head = EF.EndsHead(st, 2 * self.layers, stream, K=1)
        self.ids_in, self.ids_out = EF.id_tensor(mesh_device), EF.id_tensor(mesh_device)
        self.kept = EF.logits_tensor(mesh_device) if keep_logits else None
        ttnn.synchronize_device(mesh_device)
        if zero and self.own_stores:
            zero_fill(self.kv)
            zero_fill(self.keys)
            ttnn.synchronize_device(mesh_device)
        self.root = st.topo.chip(st.topo.root)
        self.programs = None  # the layers' programs, built at the first step
        self.trace_id = None
        self.steps = 0
        self._next = [0] * ROWS
        self._last = None
        logger.info(
            f"HY4_MF DecodeModel: {self.layers} layers ({len(self.full)} full-indexer: {self.full[:6]}{'...' if len(self.full) > 6 else ''}), "
            f"{self.max_positions} positions a row ({self.frames} frames), append {append}"
            f"{'' if latent_scale is None else f', latent_scale {latent_scale}'}, stores {'its own' if self.own_stores else 'handed in'}, experts {mlp_split}, built in {time.perf_counter() - t0:.0f} s; "
            f"L1 tensors {st.l1_bytes()} B per core"
        )

    # ------------------------------------------------------------------------------------------ weights
    def _load(self):
        st, n_layers = self.st, self.layers
        prefix = "model.layers.{}."
        site = lambda n, name: tuple(
            PP.checkpoint_tensors(
                [f"model.layers.{n}.{name}.hc_pre.{part}" for part in ("hc_fn", "hc_base", "hc_scale")]
            )
        )
        gammas = PP.checkpoint_tensors(
            [
                prefix.format(n) + name
                for n in range(n_layers)
                for name in ("input_layernorm.weight", "post_attention_layernorm.weight")
            ]
        )
        sites = [s for n in range(n_layers) for s in (site(n, "hc_attn_layer"), site(n, "hc_mlp_layer"))]
        fn, base, scale, norm_w = PP.checkpoint_tensors(
            [
                "model.hc_head.hc_head_fn",
                "model.hc_head.hc_head_base",
                "model.hc_head.hc_head_scale",
                "model.norm.weight",
            ]
        )
        st.set_sites(sites + [EF.head_site(fn, base, scale)], list(gammas) + [norm_w])  # (the head's site: index 2 N)
        t0 = time.perf_counter()
        for n in range(n_layers):
            attention, _, indexer = PP.checkpoint_layer(n)  # (indexer: None on a shared-indexer layer)
            assert (indexer is not None) == (n in self.full)
            lw = L.layer_weights(attention, None, indexer)
            sink = attention["learnable_sink_param"].float().flatten()
            dense = n == 0
            rw = rb = None
            if not dense:
                rw, rb = PP.checkpoint_tensors(
                    [prefix.format(n) + "mlp.gate.weight", prefix.format(n) + "mlp.gate.e_score_correction_bias"]
                )
            st.add_layer(n, lw, sink, 2 * n, 2 * n + 1, cache_layer=n, dense=dense, router_weight=rw, router_bias=rb)
            if n < 6 or n % 12 == 5 or n == n_layers - 1:
                logger.info(
                    f"HY4_MF DecodeModel: layer {n} loaded ({time.perf_counter() - t0:.0f} s), L1 tensors {st.l1_bytes()} B per core"
                )

    # ------------------------------------------------------------------------------------------ the interface
    def reset_row(self, row: int, keep: int = 0) -> None:
        """keep == 0: row ``row``'s conversation starts empty; its next position is 0. 0 < keep <= the row's next
        position: the row keeps positions [0, keep) and its next position is ``keep`` (a rewind). Nothing is written
        either way: positions below a row's next one are the only ones it reads, and its own steps append them (a
        step at position k overwrites the row's latent row and index key of position k in every layer before
        anything of that step reads them)."""
        assert 0 <= row < ROWS and 0 <= int(keep) <= self._next[row], (row, keep, self._next[row])
        self._next[row] = int(keep)

    def adopt_row(self, row: int, position: int) -> None:
        """Row ``row``'s slots hold positions [0, position) written by someone else (the prefill stack, into the
        stores this model was built on): its next position is ``position``. Nothing is written or checked: what the
        slots hold is the caller's word."""
        assert not self.own_stores, "adopt_row: a model built on stores=(the prefill stack's caches)"
        assert 0 <= row < ROWS and 0 <= int(position) < self.max_positions, (row, position, self.max_positions)
        self._next[row] = int(position)

    def step(self, tokens, positions, want_logits: bool = False):
        """One pass. tokens[r] None: row r is idle. An active row r runs token tokens[r] at position positions[r]
        (0 after reset_row, then one more each call), appends its latent row and index key in every layer, and gets
        the greedy next-token id back. Returns a list of 8: ids, None for idle rows; with want_logits (an instance
        built with keep_logits=True) the pair (that list, the pass's logits: a host fp32 tensor [8, vocab]; an idle
        row's are of no use). A pass that failed raises: a stuck program by the operation timeout, an id outside the
        vocabulary, a status that is not at rest (check_every)."""
        ids = self._step(tokens, positions)
        return (ids, self.read_logits()) if want_logits else ids

    @classmethod
    def open(cls, trace_region_size=TRACE_REGION, **kwargs) -> "DecodeModel":
        """Open the 8x4 mesh the way the tests' fixture does (mesh_requirements) and build the model on it; the mesh
        is then this object's to close (close()). trace defaults to True here (the mesh has its trace region)."""
        from models.experimental.hy4_preview.tt import prefill_bridge as pb
        from tt_py_test_utils_common.device_params import get_updated_device_params

        params = get_updated_device_params(dict(pb.device_params(), trace_region_size=int(trace_region_size)))
        fabric_config, reliability = params.pop("fabric_config"), params.pop("reliability_mode")
        router = params.pop("fabric_router_config")
        ttnn.set_fabric_config(
            fabric_config,
            reliability,
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
            router,
        )
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH), **params)
        try:
            model = cls(mesh, **dict({"trace": True}, **kwargs))
        except BaseException:
            ttnn.close_mesh_device(mesh)
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            raise
        model._opened = True
        return model

    def close(self) -> None:
        """Release the trace; take the host-write registry off ttnn; close the mesh if open() opened it. Plain calls
        in the calling thread: call it from the thread that built the model, once, after the last pass."""
        from models.experimental.hy4_preview.fused.model import hostwrites

        try:
            self.release_trace()
        finally:
            hostwrites.uninstall()
            if self._opened:
                self._opened = False
                for submesh in self.mesh.get_submeshes():
                    ttnn.close_mesh_device(submesh)
                ttnn.close_mesh_device(self.mesh)
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    # ------------------------------------------------------------------------------------------ the pass
    def _inputs(self, tokens, positions):
        assert len(tokens) == len(positions) == ROWS, (len(tokens), len(positions))
        active = [t is not None for t in tokens]
        for r in range(ROWS):
            if active[r]:
                p = int(positions[r])
                assert (
                    p == self._next[r]
                ), f"row {r}: position {p}, its next one is {self._next[r]} (reset_row starts a row at 0)"
                assert p < self.max_positions, f"row {r}: position {p} is beyond the {self.max_positions} a row holds"
                assert 0 <= int(tokens[r]) < self.vocab, f"row {r}: token id {tokens[r]}"
        pos = [int(positions[r]) if active[r] else None for r in range(ROWS)]
        st = self.st
        self.ein.write_ids(self.ids_in, [int(tokens[r]) if active[r] else 0 for r in range(ROWS)])
        st.set_positions([IDLE if p is None else p for p in pos])
        cos, sin = rope_cos_sin(torch.tensor([0 if p is None else p for p in pos]), self.acfg)
        st.set_rope(*frames.rope_tables(cos, sin))
        st.set_contexts([0 if p is None else p + 1 for p in pos])
        return active

    def _launch(self, after=None):
        """IN, the layers' programs, HEAD. after(name): called after each program has ended on every chip (the
        host may read between programs: an untraced step only)."""
        st, mesh = self.st, self.mesh
        if self.programs is None:
            self.programs = st.step_programs(first_post=True, last_post=False)
        if after is None:
            self.ein.launch(self.ids_in)
            Stack.launch(self.programs)
            self.head.launch(self.ids_out, self.kept)
            return
        self.ein.launch(self.ids_in)
        ttnn.synchronize_device(mesh)
        after("in")
        for name, io, desc in self.programs:
            Stack.launch([(name, io, desc)])
            ttnn.synchronize_device(mesh)
            after(name)
        self.head.launch(self.ids_out, self.kept)
        ttnn.synchronize_device(mesh)
        after("head")

    def _step(self, tokens, positions, after=None):
        active = self._inputs(tokens, positions)
        t0 = time.perf_counter()
        if after is not None or not self.traced or self.steps == 0 or is_mock():
            self._launch(after)  # (the first step compiles every program)
        elif self.trace_id is None:
            # (a capture RECORDS the step's programs and runs none of them: on the board the ids, the logits and the
            # stores were those of the step before after it, log mf_dm_L6_P31_tr; this step is the first replay)
            self.trace_id = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            try:
                self._launch()
            finally:
                ttnn.end_trace_capture(self.mesh, self.trace_id, cq_id=0)
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        else:
            ttnn.execute_trace(self.mesh, self.trace_id, cq_id=0, blocking=False)
        self.steps += 1
        if is_mock():
            out = [0] * ROWS
        else:
            ttnn.synchronize_device(self.mesh)
            out = EF.read_ids(self.ids_out, chip=self.root)
            bad = [r for r in range(ROWS) if active[r] and not 0 <= out[r] < self.vocab]
            if bad:
                raise RuntimeError(
                    f"DecodeModel: step {self.steps}: rows {bad} got ids outside the vocabulary: {[out[r] for r in bad]}"
                )
            slow = time.perf_counter() - t0 > self.slow_pass_s
            if after is None and (slow or (self.check_every and self.steps % self.check_every == 0)):
                status = self.status()
                if status:
                    raise RuntimeError(
                        f"DecodeModel: step {self.steps} left a status (a wait ran out; the pass's data are not valid): {str(status)[:600]}"
                    )
        for r in range(ROWS):
            if active[r]:
                self._next[r] += 1
        self._last = [out[r] if active[r] else None for r in range(ROWS)]
        return list(self._last)

    def step_probed(self, tokens, positions, after):
        """A step launched program by program with ``after(name)`` between programs (names: "in", "p1[n]",
        "scan[n]", "merge[n]", "cut[n]", "p2[n]", "mlp[n]", "head"): tests read the device between programs."""
        return self._step(tokens, positions, after=after)

    # ------------------------------------------------------------------------------------------ host views
    def next_position(self, row: int) -> int:
        return self._next[row]

    def read_logits(self) -> torch.Tensor:
        """[8, vocab] fp32: the last step's logits (keep_logits=True)."""
        assert self.kept is not None, "DecodeModel(keep_logits=True)"
        ttnn.synchronize_device(self.mesh)
        return EF.read_logits(self.kept).permute(1, 0, 2).reshape(ROWS, -1)

    def read_selection(self) -> list:
        """The selection table as the last cut left it: per row the sorted positions picked (int64)."""
        ttnn.synchronize_device(self.mesh)
        picked = [[] for _ in range(ROWS)]
        for d in range(self.mesh.get_num_devices()):
            counts, rows = self.st.cut.read(d)
            for u in range(ROWS):
                r = rows[u, : int(counts[u])]
                picked[u].append(
                    (r // kv_frames.STRIPE_BLOCK * kv_frames.CHIPS + d) * kv_frames.STRIPE_BLOCK
                    + r % kv_frames.STRIPE_BLOCK
                )
        return [torch.cat(p).sort().values for p in picked]

    def status(self) -> dict:
        """Everything that is not at rest after a step (empty when the step completed): both halves' words, the
        merge's time-out flag, the HEAD's canary."""
        out = dict(self.st.status())
        flags = self.st.merge.read()["flags"]
        if bool((flags & MERGE_TIMEOUT).any()):
            out[("merge", "timeout")] = sorted(set(flags.flatten().tolist()))
        canary = self.head.canary_summary()
        if canary["failed"]:
            out[("head", "canary")] = canary
        return out

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh, self.trace_id)
            self.trace_id = None
