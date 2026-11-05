# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""16-row target verify pass for DFlash2: anchor + 15 drafts at consecutive positions of one user, with layer taps.

Mode "argmax" returns one token per row, "topk" the per-device top-32 candidates of the target logits for each row
(host rejection sampling), "logits" the full logits of each row."""

import torch

import ttnn
from models.demos.muse_glimmer_30b.tt.attention import kv_page_table
from models.demos.muse_glimmer_30b.tt.common import replicate_mapper
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
    """16-row verify pass of the target for one user (batch 1) with layer taps, for DFlash2."""

    def __init__(self, model, tap_layers, user_page_row, rows=16, shared_write_rows=None):
        """user_page_row: torch int32 [nb], the user's row of the global page table (generator.page_table[0]).
        shared_write_rows: optional list of `rows` persistent device tensors [1] int32 owned by the caller; then
        prepare() does not write them (the caller sets them to start_pos + i)."""
        self.model = model
        self.rows = rows
        self.tap_layers = list(tap_layers)
        args = model.args
        mesh = model.mesh_device
        nb = args.max_num_blocks_per_user
        assert user_page_row.numel() == nb
        self.user_page_row = user_page_row.reshape(-1).to(torch.int32)
        mapper = replicate_mapper(mesh)

        def dev(t, dtype, layout):
            return ttnn.from_torch(
                t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        self.taps = {
            i: dev(torch.zeros(1, 1, DECODE_B_PAD, args.hidden_size), ttnn.bfloat16, ttnn.TILE_LAYOUT)
            for i in self.tap_layers
        }
        spt = torch.zeros(DECODE_B_PAD, nb, dtype=torch.int32)
        spt[:rows] = kv_page_table(args, 1, sliding=True)[0]
        self.sliding_page_table = dev(spt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        self.decode_inputs = model.allocate_decode_inputs(rows, nb)
        self._owns_rows = shared_write_rows is None
        if shared_write_rows is None:
            shared_write_rows = [
                dev(torch.tensor([-1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT) for _ in range(rows)
            ]
        assert len(shared_write_rows) == rows
        self.kv_write_rows = shared_write_rows
        self._pt_copied = False
        self.vocab_bias = make_vocab_bias(mesh, args.padded_vocab_size, args.vocab_size)
        set_greedy_sampling(model)

    def tap_list(self) -> list:
        return [self.taps[i] for i in self.tap_layers]

    def prepare(self, tokens, start_pos):
        """Host-side input copies only (decode inputs, seeds, and the write rows when this object owns them)."""
        m = self.model
        assert len(tokens) == self.rows
        positions = torch.arange(start_pos, start_pos + self.rows, dtype=torch.int64)
        pt = self.user_page_row[None].repeat(self.rows, 1)
        host = m.prepare_decode_inputs_host(torch.tensor(tokens, dtype=torch.int64), positions, pt)
        m.copy_decode_inputs(host, self.decode_inputs, copy_page_table=not self._pt_copied)
        self._pt_copied = True
        if self._owns_rows:
            mapper = replicate_mapper(m.mesh_device)
            for i, d in enumerate(self.kv_write_rows):
                h = ttnn.from_torch(
                    torch.tensor([start_pos + i], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mapper,
                )
                ttnn.copy_host_to_device_tensor(h, d)
        m.sampling.seed_manager.get_new_values()

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
        """Free the persistent buffers this object allocated (taps, sliding page table, decode inputs, owned rows)."""
        for t in [*self.taps.values(), self.sliding_page_table, *self.decode_inputs, self.vocab_bias]:
            ttnn.deallocate(t)
        if self._owns_rows:
            for t in self.kv_write_rows:
                ttnn.deallocate(t)
        self.taps = {}

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
        self.prepare(tokens, start_pos)
        out = self.forward(mode)
        res = self.read_out(out, mode)
        for t in out if isinstance(out, tuple) else [out]:
            ttnn.deallocate(t)
        return res
