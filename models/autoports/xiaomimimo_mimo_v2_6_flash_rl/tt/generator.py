# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TP=8 x DP=4 generator for MiMo on a Blackhole Galaxy.

The 4x8 mesh is split into its four 1x8 rows (``create_submeshes``, the same split vLLM's
data-parallel path uses). Each row holds a full TP=8 replica: its own weights, paged KV cache and
users. Replicas never communicate. Every step enqueues work on all four rows before reading any
result, so the rows run concurrently.

Global user ``u`` lives on replica ``u // users_per_replica`` in decode slot ``u % users_per_replica``.
"""

import time

import torch
from loguru import logger

import ttnn

from .model import TtMiMoModel


def create_dp_submeshes(mesh_device, dp):
    rows, cols = tuple(mesh_device.shape)
    assert rows * cols % dp == 0
    if (rows, cols) != (dp, rows * cols // dp):
        mesh_device.reshape(ttnn.MeshShape(dp, rows * cols // dp))
    return mesh_device.create_submeshes(ttnn.MeshShape(1, rows * cols // dp))


class MiMoGalaxyGenerator:
    def __init__(
        self,
        mesh_device,
        cfg,
        ckpt,
        cache_dir,
        dp=4,
        users_per_replica=4,
        max_seq_len=4096,
        num_layers=None,
        max_batch=32,
        block_size=64,
        **model_kwargs,
    ):
        self.cfg = cfg
        self.dp = dp
        self.users_per_replica = users_per_replica
        self.max_batch = max_batch
        self.submeshes = create_dp_submeshes(mesh_device, dp)
        blocks_per_user = (max_seq_len + block_size - 1) // block_size
        num_blocks = users_per_replica * blocks_per_user
        # Slots that hold users get their own blocks; empty slots point at block 0 and are never written
        # (their position is -1).
        page_table = torch.zeros(max_batch, blocks_per_user, dtype=torch.int32)
        page_table[:users_per_replica] = torch.arange(num_blocks, dtype=torch.int32).reshape(users_per_replica, -1)

        self.models = []
        for r, sm in enumerate(self.submeshes):
            t0 = time.time()
            # Replica 0 converts the checkpoint into the tensor cache; rows 1-3 load the cached tensors.
            model = TtMiMoModel(
                sm,
                cfg,
                ckpt=ckpt if r == 0 else None,
                cache_dir=cache_dir,
                num_layers=num_layers,
                max_seq_len=max_seq_len,
                max_batch=max_batch,
                block_size=block_size,
                **model_kwargs,
            )
            model.allocate_kv_cache(num_blocks)
            model.set_page_table(page_table)
            self.models.append(model)
            logger.info(f"replica {r} (row {r} of the Galaxy) ready in {time.time() - t0:.0f}s")

    @property
    def num_users(self):
        return self.dp * self.users_per_replica

    def locate(self, user):
        return user // self.users_per_replica, user % self.users_per_replica

    def prefill(self, prompts):
        """prompts: list of 1D token tensors, one per global user (len <= num_users). Returns last-token logits."""
        assert len(prompts) <= self.num_users
        pending = {}
        # Interleave replicas so all four rows have work in flight.
        for slot in range(self.users_per_replica):
            for r in range(self.dp):
                user = r * self.users_per_replica + slot
                if user < len(prompts):
                    pending[user] = self.models[r].prefill_enqueue(prompts[user], slot)
        logits = {}
        for user, handle in pending.items():
            r, _ = self.locate(user)
            logits[user] = self.models[r].prefill_read(handle)[0]
        return [logits[u] for u in range(len(prompts))]

    def decode(self, tokens, positions):
        """tokens, positions: dict user -> int. Users not listed stay idle. Returns dict user -> logits."""
        pending = []
        for r, model in enumerate(self.models):
            tok = torch.zeros(self.max_batch, dtype=torch.int32)
            pos = torch.full((self.max_batch,), -1, dtype=torch.int32)
            for slot in range(self.users_per_replica):
                user = r * self.users_per_replica + slot
                if user in tokens:
                    tok[slot], pos[slot] = tokens[user], positions[user]
            pending.append(model.decode_enqueue(tok, pos))
        out = {}
        for r, handle in enumerate(pending):
            logits = self.models[r].decode_read(handle)
            for slot in range(self.users_per_replica):
                user = r * self.users_per_replica + slot
                if user in tokens:
                    out[user] = logits[slot]
        return out

    def generate_greedy(self, prompts, max_new_tokens):
        """Greedy decode for every user; returns (generated token lists, seconds per decode step)."""
        first = self.prefill(prompts)
        generated = {u: [int(first[u].argmax())] for u in range(len(prompts))}
        positions = {u: prompts[u].numel() for u in range(len(prompts))}
        step_times = []
        for _ in range(max_new_tokens - 1):
            t0 = time.time()
            logits = self.decode({u: generated[u][-1] for u in generated}, positions)
            step_times.append(time.time() - t0)
            for u, lg in logits.items():
                generated[u].append(int(lg.argmax()))
                positions[u] += 1
        return [generated[u] for u in range(len(prompts))], step_times
