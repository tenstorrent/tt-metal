# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-resident decode stepping for the TP serving path (QWEN36_ASYNC_DECODE_STEP=1).

The stock traced decode step spends ~2.8 ms/step on the host: building
tokens/pos/rope tensors, three copy_host_to_device writes, a per-shard argmax
idx/val readback, and a host winner-pick across the vocab shards. This module
folds all of that into the trace itself, so a steady-state step is
execute_trace + sync + one tiny token readback:

  * cross-shard winner-pick on device: the per-shard argmax (idx, val) pairs
    are all-gathered ([1,1,1,nd*Bn], KBs — not the full-vocab gather the
    force-argmax sampler does) and the winner is picked with the same int32
    min-sentinel recipe tt_sampling's tie-break uses. The winning GLOBAL
    vocab index is written straight into the trace's token input buffer, so
    the next replay consumes it with no host round-trip (token feedback).
  * position increment on device: in-place ttnn.plus_one on the cur_pos input
    (prior art: models/common/models/executor.py ondevice_decode_loop).
  * rope refresh on device: a persistent uint32 rope-index tensor is
    plus_one'd and gathers rows from a precomputed cos/sin table via
    ttnn.embedding — bit-identical to the host-computed rope values, so token
    sequences match the stock path exactly.

Tie-break equivalence with the stock host pick: the host does
torch.argmax(vals, dim=0) — the LOWEST shard among exact-value ties. Shard d's
global indices live in [d*per_shard, (d+1)*per_shard), so the minimum global
index among tied shard-winners IS the lowest-shard winner; both paths also use
the same per-shard ttnn.argmax. Greedy sequences are therefore exactly equal.

Constraint: this keeps tokens/positions/rope ON DEVICE between steps, so any
host-side slot mutation (vLLM batch condense -> model._remap_gdn_slots) makes
the device state stale. The model invalidates a registered helper on remap;
the owner must call resync() before the next step or step-readiness asserts.
"""

import torch

import ttnn

# Added to every non-maximum candidate's global index before the min-reduce.
# Must exceed the largest real global vocab index (qwen3.x: 248319) and stay
# bf16-exact (power of two), since the mask is scaled in bf16 before the int32
# typecast. 2**18 = 262144 satisfies both for vocab sizes up to 262k.
_TIE_SENTINEL = float(2**18)
# Row-pad marker for the gathered candidate grid (rows nd..31). Large enough
# that a padded row can never win the min-reduce, small enough that adding the
# tie sentinel cannot overflow int32.
_PAD_OFFSET = 2**30


class AsyncDecodeStep:
    """In-trace decode stepping: winner-pick + token feedback + pos/rope increment.

    Lifecycle (mirrors the stage-before-capture rule for GDN restore):
      1. construct BEFORE ttnn.begin_trace_capture (allocates persistent/static
         device tensors; allocating under a live trace is unsafe),
      2. run emit_step_tail once eagerly in the warm/compile pass,
      3. run emit_step_tail again inside the capture,
      4. resync() after capture (the warm+capture passes garble the inputs),
      5. per measured step: execute_trace + sync + read_tokens().
    """

    def __init__(self, model, dev_tokens, dev_pos, dev_rope, batch, table_len):
        assert model.num_devices > 1, "async decode step is the TP serving path"
        assert model.sampling is not None, "async decode step needs the vocab-sharded lm_head (model.sampling)"
        assert model.rope.rope_delta == 0, "async decode step is text-only (rope_delta must be 0)"
        args = model.args
        assert args.vocab_size % model.num_devices == 0
        self.model = model
        self.mesh = model.mesh_device
        self.tt_ccl = model.tt_ccl
        self.nd = model.num_devices
        self.per_shard = args.vocab_size // self.nd
        assert self.per_shard + _TIE_SENTINEL < 2**24, "sentinel arithmetic assumes vocab < 16M"
        self.B = batch
        self.rd = args.rope_head_dim
        self.table_len = table_len
        self.dev_tokens = dev_tokens  # [B,1] uint32 ROW_MAJOR (trace token input)
        self.dev_pos = dev_pos  # [B]   int32  ROW_MAJOR (trace cur_pos input)
        self.dev_rope = dev_rope  # [2,B,1,rd] bf16 TILE  (trace packed-rope input)
        self._stale_reason = None

        # cos/sin lookup table, one row per position: rows [0,T) cos, rows
        # [T,2T) sin. Same torch math as prepare_decode_inputs_host, so the
        # gathered bf16 rows are bit-identical to the host-computed rope.
        inv_freq = 1.0 / (args.rope_theta ** (torch.arange(0, self.rd, 2).float() / self.rd))
        freqs = torch.outer(torch.arange(table_len).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        table = torch.cat([emb.cos(), emb.sin()], dim=0).to(torch.bfloat16)
        self.rope_table = ttnn.from_torch(
            table,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        # Persistent rope indices [1, 2B]: [pos_0..pos_{B-1}, T+pos_0..T+pos_{B-1}].
        # One plus_one advances cos and sin rows together (the +T offset is preserved).
        self.rope_idx = ttnn.from_torch(
            torch.zeros(1, 2 * batch, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        # Shard-offset grid for the gathered candidates [1,1,32,Bn]: row d holds
        # d*per_shard (global-index rebase); pad rows hold _PAD_OFFSET so they
        # can never win the min-reduce. Bn = sampler width (32).
        self.Bn = model.sampling.tt_sampling.max_batch_size
        off = torch.full((1, 1, 32, self.Bn), _PAD_OFFSET, dtype=torch.int32)
        for d in range(self.nd):
            off[:, :, d, :] = d * self.per_shard
        self.offsets = ttnn.from_torch(
            off,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        # Batch condense moves GDN slots under us -> device tokens/pos/rope are stale.
        model._async_decode_step = self

    def _gather(self, t):
        """Tiny width-dim all-gather (the decode-norm CCL call shape from rmsnorm)."""
        return ttnn.experimental.all_gather_async(
            t,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            topology=self.model.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def emit_step_tail(self, shard_idx, shard_val):
        """Append the winner-pick + feedback + increment ops to the current stream.

        shard_idx: per-shard argmax [1,1,Bn] uint32 ROW_MAJOR (local vocab index).
        shard_val: per-shard max    [1,1,Bn] bf16 TILE.
        Must run once eagerly before capture (program compile) and once inside it.
        """
        Bn, nd = self.Bn, self.nd
        # Per-shard candidates -> gatherable [1,1,1,Bn].
        idx4 = ttnn.reshape(shard_idx, (1, 1, 1, Bn))
        idx4 = ttnn.to_layout(idx4, ttnn.TILE_LAYOUT)
        idx4 = ttnn.typecast(idx4, ttnn.int32)
        val4 = ttnn.reshape(shard_val, (1, 1, 1, Bn))
        g_idx = self._gather(idx4)  # [1,1,1,nd*Bn]
        g_val = self._gather(val4)
        ttnn.deallocate(idx4)
        ttnn.deallocate(val4)

        # Regroup device blocks into candidate rows [1,1,nd,Bn], pad rows to a
        # full tile so the dim=-2 reduces never see undefined tile padding.
        g_idx = ttnn.to_layout(g_idx, ttnn.ROW_MAJOR_LAYOUT)
        g_idx = ttnn.reshape(g_idx, (1, 1, nd, Bn))
        g_idx = ttnn.to_layout(g_idx, ttnn.TILE_LAYOUT)
        g_idx = ttnn.pad(g_idx, [(0, 0), (0, 0), (0, 32 - nd), (0, 0)], value=0)
        gidx = ttnn.add(g_idx, self.offsets)  # global vocab index per candidate
        ttnn.deallocate(g_idx)
        g_val = ttnn.to_layout(g_val, ttnn.ROW_MAJOR_LAYOUT)
        g_val = ttnn.reshape(g_val, (1, 1, nd, Bn))
        g_val = ttnn.to_layout(g_val, ttnn.TILE_LAYOUT)
        g_val = ttnn.pad(g_val, [(0, 0), (0, 0), (0, 32 - nd), (0, 0)], value=-1e30)

        # Winner = min global index among max-value candidates (the host rule:
        # lowest shard wins ties). int32 min/max reduces take the exact SFPU
        # path; the bf16 mask arithmetic is exact (0/1 * 2**18).
        maxv = ttnn.max(g_val, dim=-2, keepdim=True)  # [1,1,1,Bn] bf16
        not_max = ttnn.lt(g_val, maxv)
        ttnn.deallocate(maxv)
        ttnn.deallocate(g_val)
        sent = ttnn.multiply(not_max, _TIE_SENTINEL)
        ttnn.deallocate(not_max)
        sent = ttnn.typecast(sent, ttnn.int32)
        masked = ttnn.add(gidx, sent)
        ttnn.deallocate(gidx)
        ttnn.deallocate(sent)
        win = ttnn.min(masked, dim=-2, keepdim=True)  # [1,1,1,Bn] int32
        ttnn.deallocate(masked)

        # Token feedback: winner -> [B,1] uint32 ROW_MAJOR into the trace's
        # token input. All later replays read the token this replay produced.
        tok = ttnn.typecast(win, ttnn.uint32)
        ttnn.deallocate(win)
        tok = ttnn.untilize_with_unpadding(tok, [0, 0, 0, self.B - 1])
        tok = ttnn.reshape(tok, (self.B, 1))
        ttnn.copy(tok, self.dev_tokens)

        # Position + rope advance for the NEXT replay. These run after every
        # read of the buffers in the captured body, so no intra-trace hazard.
        ttnn.plus_one(self.dev_pos)
        ttnn.plus_one(self.rope_idx)
        rope = ttnn.embedding(self.rope_idx, self.rope_table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1,2B,rd]
        rope = ttnn.reshape(rope, (2, self.B, 1, self.rd))
        rope = ttnn.to_layout(rope, ttnn.TILE_LAYOUT)
        ttnn.copy(rope, self.dev_rope)

    def resync(self, tokens, positions, page_table=None):
        """Host-write tokens/pos/rope (+ rope indices) into the live device buffers.

        Required after trace capture (the warm+capture passes advanced the
        buffers) and after any invalidation (GDN slot remap).
        """
        from models.tt_transformers.tt.common import copy_host_to_device

        pos_vec = positions if isinstance(positions, torch.Tensor) else torch.tensor(positions, dtype=torch.int32)
        pos_vec = pos_vec.to(torch.int32).reshape(-1)
        assert int(pos_vec.max()) + 1 < self.table_len, "position would overrun the rope table"
        host = self.model.prepare_decode_inputs_host(
            torch.tensor(tokens, dtype=torch.int32).reshape(self.B, 1), pos_vec, page_table=None
        )
        copy_host_to_device(list(host[:3]), device_tensors=[self.dev_tokens, self.dev_pos, self.dev_rope])
        idx = torch.cat([pos_vec, pos_vec + self.table_len]).reshape(1, 2 * self.B)
        idx_host = ttnn.from_torch(idx.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        copy_host_to_device([idx_host], device_tensors=[self.rope_idx])
        self._stale_reason = None

    def read_tokens(self):
        """This step's winning token ids (host list of B ints); one tiny readback."""
        assert self._stale_reason is None, f"async decode state is stale ({self._stale_reason}); resync() first"
        return ttnn.to_torch(ttnn.get_device_tensors(self.dev_tokens)[0]).reshape(-1)[: self.B].tolist()

    def invalidate(self, reason):
        self._stale_reason = reason

    def release(self):
        if getattr(self.model, "_async_decode_step", None) is self:
            self.model._async_decode_step = None
