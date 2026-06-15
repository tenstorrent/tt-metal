# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact greedy text generator for the Mistral-Small-4 text core.

Mirrors the deepseek_v3 demo pattern (dedicated model-local generator, not the tt_transformers
Generator): a parallel prefill that populates the KV cache, then a token-by-token decode loop with
on-device argmax. Embedding is a host row-gather (the embedding table is not on device); everything
else — attention, MoE, LM head, and the argmax sampling — runs on the mesh.
"""
import torch

import ttnn


def _repl(t, mesh):
    return ttnn.from_torch(
        t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _pos(positions, mesh):
    # int32 device tensor [B] of current cache positions (one per user) for paged_update_cache / SDPA decode
    return ttnn.from_torch(
        torch.tensor(positions, dtype=torch.int32), device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _host(t, mesh, layout):
    return ttnn.from_torch(
        t.to(torch.bfloat16), device=None, layout=layout, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


class TracedDecode:
    """Captures TtMistral4TextModel.forward_decode as a replayable trace over persistent device
    input buffers (x, cos, sin, position). Each decode step copies the new inputs into those buffers
    and replays the trace — the whole decode graph runs without per-op host dispatch.

    The capture pass writes the (zero) warmup input at position 0; the first real step writes
    position 0 again with real k/v, so no cache reset is needed before replay.
    """

    def __init__(self, model, mesh, B, hidden, rope, kv_caches):
        self.model, self.mesh, self.B, self.kv = model, mesh, B, kv_caches
        self.tt_x = _repl(torch.zeros(B, 1, hidden), mesh)
        self.tt_cos = _repl(torch.zeros(B, 1, 1, rope), mesh)
        self.tt_sin = _repl(torch.zeros(B, 1, 1, rope), mesh)
        self.tt_pos = ttnn.from_torch(
            torch.zeros(B, dtype=torch.int32), device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
        )
        # warmup (compile kernels) then capture the graph
        self.model.forward_decode(self.tt_x, self.tt_pos, self.tt_cos, self.tt_sin, self.kv)
        self.tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.out = self.model.forward_decode(self.tt_x, self.tt_pos, self.tt_cos, self.tt_sin, self.kv)
        ttnn.end_trace_capture(mesh, self.tid, cq_id=0)

    def step(self, x, cos, sin, positions):
        ttnn.copy_host_to_device_tensor(_host(x, self.mesh, ttnn.TILE_LAYOUT), self.tt_x)
        ttnn.copy_host_to_device_tensor(_host(cos, self.mesh, ttnn.TILE_LAYOUT), self.tt_cos)
        ttnn.copy_host_to_device_tensor(_host(sin, self.mesh, ttnn.TILE_LAYOUT), self.tt_sin)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor(positions, dtype=torch.int32),
                device=None,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            ),
            self.tt_pos,
        )
        ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=True)
        return self.out


class TracedPrefill:
    """Captures TtMistral4TextModel.forward_prefill at a FIXED input length S as a replayable trace
    over persistent device buffers (x, cos, sin). The KV-cache fill (fill_cache, fixed batch_idx 0 +
    positions 0..S-1) is captured as a constant side effect; only x/cos/sin vary per prefill. Upgrades
    trace coverage to prefill AND decode (one trace per supported ISL bucket)."""

    def __init__(self, model, mesh, S, hidden, rope, kv_caches):
        self.model, self.mesh, self.kv = model, mesh, kv_caches
        self.tt_x = _repl(torch.zeros(1, S, hidden), mesh)
        self.tt_cos = _repl(torch.zeros(1, 1, S, rope), mesh)
        self.tt_sin = _repl(torch.zeros(1, 1, S, rope), mesh)
        self.model.forward_prefill(self.tt_x, self.tt_cos, self.tt_sin, self.kv)  # warmup/compile
        self.tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.out = self.model.forward_prefill(self.tt_x, self.tt_cos, self.tt_sin, self.kv)
        ttnn.end_trace_capture(mesh, self.tid, cq_id=0)

    def run(self, x, cos, sin):
        ttnn.copy_host_to_device_tensor(_host(x, self.mesh, ttnn.TILE_LAYOUT), self.tt_x)
        ttnn.copy_host_to_device_tensor(_host(cos, self.mesh, ttnn.TILE_LAYOUT), self.tt_cos)
        ttnn.copy_host_to_device_tensor(_host(sin, self.mesh, ttnn.TILE_LAYOUT), self.tt_sin)
        ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=True)
        return self.out


class Mistral4Generator:
    """Greedy generator over a TtMistral4TextModel. cos_full/sin_full hold per-position RoPE
    [B, max_pos, rope_dim] for the whole horizon (prompt + generated)."""

    def __init__(self, model, embed_weight, mesh, max_seq=256):
        self.model = model
        self.embed = embed_weight  # [vocab, hidden] host tensor (row gather)
        self.mesh = mesh
        self.max_seq = max_seq

    def _argmax_ids(self, logits, B):
        # logits [B,*,vocab] replicated -> on-device argmax over vocab -> host ids [B]
        idx = ttnn.argmax(logits, dim=-1)
        return ttnn.to_torch(idx, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=0)).long().reshape(-1)[:B]

    def greedy(self, prompt_ids, cos_full, sin_full, max_new):
        B, S = prompt_ids.shape
        rope = cos_full.shape[-1]
        kv = self.model.init_kv_caches(B, self.max_seq)

        h = _repl(self.embed[prompt_ids], self.mesh)  # [B,S,hidden]
        c = _repl(cos_full[:, :S].reshape(B, 1, S, rope), self.mesh)
        s = _repl(sin_full[:, :S].reshape(B, 1, S, rope), self.mesh)
        logits = self.model.forward_prefill(h, c, s, kv)  # [B,S,vocab]
        last = ttnn.slice(logits, [0, S - 1, 0], [B, S, logits.shape[-1]])
        nxt = self._argmax_ids(last, B)
        out = [nxt]

        cur = S
        for _ in range(max_new - 1):
            h = _repl(self.embed[nxt].reshape(B, 1, -1), self.mesh)
            c = _repl(cos_full[:, cur : cur + 1].reshape(B, 1, 1, rope), self.mesh)
            s = _repl(sin_full[:, cur : cur + 1].reshape(B, 1, 1, rope), self.mesh)
            logits = self.model.forward_decode(h, _pos([cur] * B, self.mesh), c, s, kv)  # [B,1,vocab]
            nxt = self._argmax_ids(logits, B)
            out.append(nxt)
            cur += 1
        return torch.stack(out, dim=1)  # [B, max_new]
