# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""32-row target verify pass for DFlash2: row groups of consecutive positions per user (a RowLayout), with layer taps.

Mode "argmax" returns one token per row, "topk" the per-device top-32 candidates of the target logits for each row
(host rejection sampling), "logits" the full logits of each row."""

import torch

import ttnn
from models.demos.muse_glimmer_30b.tt.attention import kv_page_table
from models.demos.muse_glimmer_30b.tt.common import replicate_mapper
from models.demos.muse_glimmer_30b.tt.dflash.layout import RowLayout
from models.demos.muse_glimmer_30b.tt.dflash.topk import TOPK_PER_DEVICE, device_topk, make_vocab_bias, read_topk
from models.demos.muse_glimmer_30b.tt.model import DECODE_B_PAD

MODES = ("argmax", "topk", "logits")


def set_greedy_sampling(model):
    """Load greedy parameters (temperature 0, top_k 1, top_p 1; 32 rows) into the model's device sampler.

    This changes the model's shared sampler to greedy; the DFlash2 path leaves it greedy afterwards."""
    from models.common.sampling.generator import SamplingParams, format_sampling_params

    model.sampling.reset_sampling_params(
        format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), DECODE_B_PAD)
    )


class TargetVerifier:
    """32-row verify pass of the target for several users with layer taps, for DFlash2. The rows follow a RowLayout."""

    def __init__(self, model, tap_layers, page_rows):
        """page_rows: host int32 [n_slots, nb]; row u is the target page row of slot u (generator.page_table for one
        user). Starts with the single-user layout RowLayout(16, (0,))."""
        self.model = model
        self.tap_layers = list(tap_layers)
        args = model.args
        mesh = model.mesh_device
        nb = args.max_num_blocks_per_user
        page_rows = page_rows.to(torch.int32)
        assert page_rows.dim() == 2 and page_rows.shape[1] == nb
        self.nb = nb
        self.page_rows = page_rows
        self.n_slots = page_rows.shape[0]
        self.sliding_rows = kv_page_table(args, self.n_slots, sliding=True)
        self._mapper = replicate_mapper(mesh)

        self.taps = {
            i: self._dev(torch.zeros(1, 1, DECODE_B_PAD, args.hidden_size), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            for i in self.tap_layers
        }
        zeros_pt = torch.zeros(DECODE_B_PAD, nb, dtype=torch.int32)
        # 32-row tables for SDPA (sliding) and the compact tables for the KV write (row g = group g's slot)
        self.sliding_page_table = self._dev(zeros_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.kv_write_page_table = self._dev(zeros_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.kv_write_sliding_page_table = self._dev(zeros_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.decode_inputs = model.allocate_decode_inputs(DECODE_B_PAD, nb)
        self._write_rows = {}  # (R, B) -> list of R device tensors [B] int32
        self._rows_host = (
            {}
        )  # (R, B) -> the last values copied into each write-row tensor (unchanged copies are skipped)
        self.layout = None
        self._pt_copied = False
        self.vocab_bias = make_vocab_bias(mesh, args.padded_vocab_size, args.vocab_size)
        set_greedy_sampling(model)
        self.set_layout(RowLayout(16, (0,)))

    def _dev(self, t, dtype, layout):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=layout,
            device=self.model.mesh_device,
            mesh_mapper=self._mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _copy(self, t, dtype, dev, layout=ttnn.ROW_MAJOR_LAYOUT):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self._mapper), dev)

    @property
    def rows(self) -> int:
        """SDPA row count: the rows of the current layout's groups."""
        return self.layout.n_rows

    @property
    def kv_write_batch(self) -> int:
        return self.layout.groups

    @property
    def kv_write_rows(self) -> list:
        """The current layout's R write-position tensors [B] int32."""
        return self._rows_for(self.layout)

    def _rows_for(self, layout):
        key = (layout.R, layout.groups)
        if key not in self._write_rows:  # created once, kept (persistent for trace replay)
            self._write_rows[key] = [
                self._dev(torch.full((layout.groups,), -1, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
                for _ in range(layout.R)
            ]
        return self._write_rows[key]

    def reserve(self, R, groups):
        """Allocate the write rows of an (R, groups) layout now: persistent tensors must exist before any trace capture."""
        self._rows_for(RowLayout(R, (None,) * groups))

    def write_rows(self, positions):
        """Copy the current layout's write rows from 32 row positions (entry g of tensor p = positions[g * R + p]).
        A tensor whose values did not change since its last copy is skipped: the DFlash2 drafter block pass uses
        these tensors with the same positions just before the verify pass."""
        lay = self.layout
        last = self._rows_host.setdefault((lay.R, lay.groups), [None] * lay.R)
        for p, d in enumerate(self._rows_for(lay)):
            vals = tuple(positions[g * lay.R + p] for g in range(lay.groups))
            if vals != last[p]:
                self._copy(torch.tensor(vals, dtype=torch.int32), ttnn.int32, d)
                last[p] = vals

    def set_layout(self, layout):
        """Switch to `layout` (host copies only): the decode-inputs page table, the 32-row sliding table and the compact
        full / sliding write tables."""
        assert max((s for s in layout.slots if s is not None), default=0) < self.n_slots
        nb = self.nb
        self._rows_for(layout)
        full = torch.zeros(DECODE_B_PAD, nb, dtype=torch.int32)
        sliding = torch.zeros(DECODE_B_PAD, nb, dtype=torch.int32)
        for r in range(layout.n_rows):
            u = layout.slot_of_row(r)
            if u is not None:
                full[r] = self.page_rows[u]
                sliding[r] = self.sliding_rows[u]
        cfull = torch.zeros(DECODE_B_PAD, nb, dtype=torch.int32)
        csliding = torch.zeros(DECODE_B_PAD, nb, dtype=torch.int32)
        for g, u in enumerate(layout.slots):
            if u is not None:
                cfull[g] = self.page_rows[u]
                csliding[g] = self.sliding_rows[u]
        self._copy(sliding, ttnn.int32, self.sliding_page_table)
        self._copy(cfull, ttnn.int32, self.kv_write_page_table)
        self._copy(csliding, ttnn.int32, self.kv_write_sliding_page_table)
        self._decode_pt_host = full
        self._pt_copied = False
        self.layout = layout

    def tap_list(self) -> list:
        return [self.taps[i] for i in self.tap_layers]

    def prepare(self, tokens, positions):
        """Host-side input copies only. tokens, positions: lists of 32 ints (position -1: row without a slot; SDPA
        skips it). Writes the decode inputs and the current layout's write rows (see write_rows)."""
        m = self.model
        assert len(tokens) == DECODE_B_PAD and len(positions) == DECODE_B_PAD
        host = m.prepare_decode_inputs_host(
            torch.tensor(tokens, dtype=torch.int64), torch.tensor(positions, dtype=torch.int64), self._decode_pt_host
        )
        m.copy_decode_inputs(host, self.decode_inputs, copy_page_table=not self._pt_copied)
        self._pt_copied = True
        self.write_rows(positions)
        m.sampling.seed_manager.get_new_values()

    def prepare_single(self, tokens, start_pos):
        """Single-user step: `tokens` (16 ints) at positions start_pos.. of slot 0, layout RowLayout(16, (0,))."""
        if self.layout != RowLayout(16, (0,)):
            self.set_layout(RowLayout(16, (0,)))
        assert len(tokens) == 16
        pad = DECODE_B_PAD - 16
        self.prepare(list(tokens) + [0] * pad, list(range(start_pos, start_pos + 16)) + [-1] * pad)

    def forward(self, mode="argmax"):
        """Device ops only (traceable): the verify pass, then by mode: "argmax" -> greedy token tensor; "topk" ->
        per-device top-32 (vals, ids) tensors; "logits" -> the logits tensor ([1,1,32,vocab_local] per device, DRAM).
        The caller frees the outputs if eager."""
        assert mode in MODES, mode
        m = self.model
        logits = m.decode_forward(*self.decode_inputs, verify=self, taps=self.taps)
        if mode == "logits":
            return logits
        if mode == "topk":
            out = device_topk(logits, self.vocab_bias)
        else:
            out = m.sample_on_device(logits, refresh_seeds=False)
        ttnn.deallocate(logits)
        return out

    def close(self):
        """Free the persistent buffers this object allocated (taps, page tables, decode inputs, write rows)."""
        tensors = [
            *self.taps.values(),
            self.sliding_page_table,
            self.kv_write_page_table,
            self.kv_write_sliding_page_table,
            *self.decode_inputs,
            self.vocab_bias,
            *[t for rows in self._write_rows.values() for t in rows],
        ]
        for t in tensors:
            ttnn.deallocate(t)
        self.taps = {}
        self._write_rows = {}
        self._rows_host = {}

    def read(self, tok) -> list:
        return self.model.tokens_to_torch(tok, self.rows).tolist()

    def read_topk(self, out):
        """out: the (vals, ids) pair from forward(mode="topk") -> (values [rows, 32] f32, global ids [rows, 32] i64)."""
        vals, ids = out
        args = self.model.args
        return read_topk(
            vals,
            ids,
            range(self.rows),
            vocab_local=args.vocab_local,
            vocab_size=args.vocab_size,
            k=TOPK_PER_DEVICE,
        )

    def read_logits(self, out):
        """out: the logits tensor from forward(mode="logits") -> torch float32 [rows, vocab_size]."""
        return self.model.logits_to_torch(out, self.rows)

    def read_out(self, out, mode):
        if mode == "topk":
            return self.read_topk(out)
        if mode == "logits":
            return self.read_logits(out)
        return self.read(out)

    def run(self, tokens, start_pos, mode="argmax"):
        """tokens: `rows` ints (anchor + drafts) at positions start_pos..start_pos+rows-1.
        argmax: list of one token per row. topk: (values [rows, 32], ids [rows, 32]) top-32 candidates per row.
        logits: torch float32 [rows, vocab_size]."""
        self.prepare_single(tokens, start_pos)
        out = self.forward(mode)
        res = self.read_out(out, mode)
        for t in out if isinstance(out, tuple) else [out]:
            ttnn.deallocate(t)
        return res
