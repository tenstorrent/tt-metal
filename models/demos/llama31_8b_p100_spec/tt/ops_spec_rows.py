# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

_BY_CACHE = {}


def _addr(t):
    return t.buffer_address()


def active(cache):
    return _BY_CACHE.get(_addr(cache))


class RowState:
    def __init__(self, mesh_device, rows, page_table, kv_cache, sdpa_program_config, per_row_sdpa=True):
        self.mesh_device = mesh_device
        self.rows = rows
        self.per_row_sdpa = per_row_sdpa
        self.program_config = sdpa_program_config
        self._last = [None] * rows
        self.update_idx = [self._dev(torch.full((rows,), -1, dtype=torch.int32)) for _ in range(rows)]
        self.pos1 = [self._dev(torch.full((1,), -1, dtype=torch.int32)) for _ in range(rows)]
        self.page_table1 = self._dev(page_table[0:1].contiguous())
        for layer in kv_cache:
            for t in layer:
                _BY_CACHE[_addr(t)] = self

    def _host(self, v):
        return ttnn.from_torch(v, dtype=ttnn.int32, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device))

    def _dev(self, v):
        return self._host(v).to(self.mesh_device)

    def set(self, positions):
        for i in range(self.rows):
            p = int(positions[i])
            if self._last[i] == p:
                continue
            v = torch.full((self.rows,), -1, dtype=torch.int32)
            v[i] = p
            ttnn.copy_host_to_device_tensor(self._host(v), self.update_idx[i])
            ttnn.copy_host_to_device_tensor(self._host(torch.tensor([p], dtype=torch.int32)), self.pos1[i])
            self._last[i] = p

    def update_cache(self, keys, k_heads, values, v_heads, page_table, fused):
        for t in self.update_idx:
            if fused:
                ttnn.experimental.paged_fused_update_cache(
                    keys, k_heads, values, v_heads, update_idxs_tensor=t, page_table=page_table
                )
            else:
                ttnn.experimental.paged_update_cache(keys, k_heads, update_idxs_tensor=t, page_table=page_table)
                ttnn.experimental.paged_update_cache(values, v_heads, update_idxs_tensor=t, page_table=page_table)

    def sdpa(
        self,
        q,
        keys,
        values,
        scale,
        compute_kernel_config,
        sliding_window=None,
        page_table=None,
        cur_pos=None,
    ):
        qd = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        sh = list(qd.shape)
        outs = []
        for i in range(self.rows):
            qi = ttnn.slice(qd, [0, i, 0, 0], [sh[0], i + 1, sh[2], sh[3]])
            outs.append(
                ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    qi,
                    keys,
                    values,
                    page_table_tensor=self.page_table1,
                    cur_pos_tensor=self.pos1[i],
                    scale=scale,
                    program_config=self.program_config,
                    compute_kernel_config=compute_kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            ttnn.deallocate(qi)
        ttnn.deallocate(qd)
        out = ttnn.concat(outs, dim=1)
        for o in outs:
            ttnn.deallocate(o)
        return out
