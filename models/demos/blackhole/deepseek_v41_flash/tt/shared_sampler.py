# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The shared tt_transformers sampler (``models.common.sampling.TTSampling``) behind the interface of ``tt/device_sampler.DSV41DeviceSampler``, for the PLAIN decode traces
(DSV41_SHARED_SAMPLER=1; the speculative verify keeps our own sampler, which draws one token per verify row from the request's uniform).

What the shared sampler is: per mesh column a top-``max_top_k`` (32) of the bf16 logits, all-gathered over the columns, then ``ttnn.sampling`` (top-k <= 32, top-p, inverse temperature over
those candidates, RNG seeded on the device per user slot by ``ttnn.manual_seed``). ~0.28 ms per step against ~1-3.4 ms for ours, with this price: top-k above 32 and "no top-k" (0 / -1 / None) are
CLAMPED to 32 (the law is then the top-32 truncation of the requested one), the candidates are bf16 roundings of the logits, and ``top_p`` acts inside the 32 candidates.

Adapter details:
* the shared sampler works on 32 user slots per mesh row; the model has ``T`` users per mesh row (T = 4 at 16 users, 1 at 4 users per row ...): the fp32 logits [1,1,T,shard] are rounded to bf16 and
  padded to 32 rows (the padding slots get k = 1, temperature 1, and their tokens are never read);
* ``set_params(invT, top_k, top_p, u, greedy)`` takes the same per-row tensors as our sampler (``tt.device_sampler.sampling_rows``): k / p / 1/T go to ``TTSampling.reset_params`` (skipped when
  unchanged), the uniform ``u`` of the row's own generator becomes the device seed of the row's slot (a fresh seed every step: ``ttnn.manual_seed`` re-seeds at every replay), so a seeded request
  keeps its stream whatever slot / bucket it runs in (the draw of a seed on the device does not depend on the slot, tests/test_shared_sampler_integration.py);
* greedy rows (temperature 0 / top_k 1) are the exact fp32 argmax: the step trace already wrote it into the token buffer (``base_tok``), the sampler trace keeps it for those rows.
"""

import types

import torch
from loguru import logger

import ttnn

VOCAB = 129280
SLOTS = 32  # user slots per mesh row of the shared sampler (its minimum, a tile of rows)
MAX_TOP_K = 32

_logged = set()


def _log_once(key, msg):
    if key not in _logged:
        _logged.add(key)
        logger.info(msg)


class SharedSamplerAdapter:
    shared = True  # decoder.forward_sampler passes the step trace's greedy token buffer (``base_tok``)
    variants = (False,)  # one trace (no separate top-k variant: the shared sampler is always the top-32 path)

    def __init__(self, mesh_device, mesh_config, ccl, users_per_row, exact_greedy=False):
        from models.common.sampling.tt_sampling import TTSampling

        self.md, self.mc, self.ccl = mesh_device, mesh_config, ccl
        rows, cols = tuple(mesh_device.shape)
        self.rows, self.cols, self.T = rows, cols, users_per_row
        assert users_per_row <= SLOTS
        self.fast = False
        self.params = None
        args = types.SimpleNamespace(
            vocab_size=VOCAB,
            padded_vocab_size=VOCAB,
            cluster_shape=(rows, cols),
            max_batch_size=SLOTS,
            max_top_k=MAX_TOP_K,
            sampling_dp=rows,
            sampling_all_gather_axis=1,
        )
        self.ts = TTSampling(mesh_device, ccl, args)
        self._last_kpt = None
        self._last_greedy = None
        # spec verify (no step trace wrote the greedy tokens): the exact fp32 argmax of every row comes from our sampler's greedy part (its other constants are tiny and unused)
        self._greedy_helper = None
        if exact_greedy:
            from models.demos.blackhole.deepseek_v41_flash.tt.device_sampler import DSV41DeviceSampler

            self._greedy_helper = DSV41DeviceSampler(mesh_device, mesh_config, ccl, users_per_row, J=2, levels=1)

    # host side -------------------------------------------------------------------------------------------------------------------------
    def alloc_params(self):
        """Persistent device tensor [1,1,rows*T,32] fp32 (column 4 = the greedy flag of every row), row sharded, like ``DSV41DeviceSampler.alloc_params``."""
        self._pmap = ttnn.ShardTensor2dMesh(self.md, dims=(2, None), mesh_shape=(self.rows, self.cols))
        host = torch.zeros(1, 1, self.rows * self.T, 32)
        host[..., 4] = 1.0
        self.params = ttnn.from_torch(
            host,
            device=self.md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._pmap,
        )
        return self.params

    def set_params(self, invT, top_k, top_p, u, greedy):
        """Each argument a [B] tensor / list (B = rows * T, mesh-row order), as ``DSV41DeviceSampler.set_params``. top_k <= 0 / > 32: clamped to 32 (logged once)."""
        rows, T = self.rows, self.T
        B = rows * T
        invT = torch.as_tensor(invT, dtype=torch.float32).reshape(B)
        kk = torch.as_tensor(top_k).reshape(B).float()
        pp = torch.as_tensor(top_p, dtype=torch.float32).reshape(B)
        gr = torch.as_tensor(greedy, dtype=torch.float32).reshape(B) > 0
        uu = torch.as_tensor(u, dtype=torch.float32).reshape(B)
        smp = ~gr
        if bool((smp & ((kk <= 0) | (kk > MAX_TOP_K))).any()):
            _log_once(
                "kclamp",
                f"DSV4.1 shared sampler: top_k {'off / ' if bool((smp & (kk <= 0)).any()) else ''}> {MAX_TOP_K} is clamped to {MAX_TOP_K} candidates "
                f"(no top_k = the top-{MAX_TOP_K} of the vocabulary; DSV41_SHARED_SAMPLER=0 selects the full-vocabulary sampler)",
            )
        # [rows, SLOTS] slot layout: the T users of a mesh row first, padding slots greedy (k = 1, temperature 1)
        k_s = torch.ones(rows, SLOTS, dtype=torch.long)
        p_s = torch.ones(rows, SLOTS)
        t_s = torch.ones(rows, SLOTS)  # inverse temperature
        seeds = torch.zeros(rows, SLOTS, dtype=torch.long)
        k_eff = torch.where(
            gr, torch.ones_like(kk), torch.where((kk <= 0) | (kk > MAX_TOP_K), torch.full_like(kk, MAX_TOP_K), kk)
        )
        k_s[:, :T] = k_eff.reshape(rows, T).long()
        p_s[:, :T] = torch.where(gr, torch.ones_like(pp), pp).reshape(rows, T)
        t_s[:, :T] = torch.where(gr, torch.ones_like(invT), invT).reshape(rows, T)
        seeds[:, :T] = (uu.double() * (2**31 - 1)).long().reshape(rows, T)
        kpt = (k_s.flatten().tolist(), p_s.flatten().tolist(), t_s.flatten().tolist())
        if kpt != self._last_kpt:
            self.ts.reset_params(
                k=k_s.flatten(), p=p_s.flatten().tolist(), temp=t_s.flatten()
            )  # (ttnn.sampling takes the INVERSE temperature)
            self._last_kpt = kpt
        host_seeds = ttnn.from_torch(
            seeds.flatten().to(torch.uint32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=self.ts._param_dims, mesh_shape=self.ts.cluster_shape),
        )
        ttnn.copy_host_to_device_tensor(host_seeds, self.ts.seeds_tt_tensor)
        gkey = gr.tolist()
        if (
            gkey != self._last_greedy
        ):  # the greedy flags change rarely: upload them only then (one host-to-device copy less per step)
            gh = torch.zeros(1, 1, B, 32)
            gh[0, 0, :, 4] = gr.float()
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(gh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._pmap), self.params
            )
            self._last_greedy = gkey

    # device side (inside the trace) -----------------------------------------------------------------------------------------------------
    def run(self, logits, params, fast=False, base_tok=None):
        """logits [1,1,T,shard] fp32 -> token ids [T,1] uint32. ``base_tok`` [T,1] uint32: the greedy tokens of the step trace, kept for the greedy rows."""
        T = self.T
        xb = ttnn.typecast(logits, ttnn.bfloat16)
        if T < SLOTS:
            xb = ttnn.pad(xb, [(0, 0), (0, 0), (0, SLOTS - T), (0, 0)], value=0.0)
        out = self.ts.forward(xb)
        out = out[0] if isinstance(out, tuple) else out
        tok_s = ttnn.reshape(ttnn.slice(ttnn.reshape(out, [1, 1, 1, SLOTS]), [0, 0, 0, 0], [1, 1, 1, T]), [T, 1])
        tok_s = ttnn.typecast(ttnn.to_layout(ttnn.typecast(tok_s, ttnn.uint32), ttnn.TILE_LAYOUT), ttnn.float32)
        if base_tok is None and self._greedy_helper is None:
            return ttnn.to_layout(ttnn.typecast(tok_s, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        greedy = ttnn.reshape(ttnn.slice(params, [0, 0, 0, 4], [1, 1, T, 5]), [T, 1])
        if base_tok is not None:
            base = ttnn.typecast(ttnn.to_layout(base_tok, ttnn.TILE_LAYOUT), ttnn.float32)
        else:
            base = ttnn.reshape(self._greedy_helper._greedy(logits, with_min=False)[2], [T, 1])
        tok = ttnn.add(ttnn.multiply(greedy, base), ttnn.multiply(ttnn.subtract(1.0, greedy), tok_s))
        return ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
