# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The z-lab/gpt-oss-20b-DFlash drafter's state on a 1 x D mesh (models/demos/gpt_oss/reference/dflash.py describes
the model; the draft and context passes run in dflash/fused_draft.py).

Tensor parallel like a dense target layer: per device the drafter holds 64 / D query heads and 8 / D KV heads. RoPE is
the target's own (identical YaRN parameters), so k / q_norm weights are permuted to the interleaved layout of ttnn's
rotary_embedding_llama, as the target's.

Held here: the config, the norm weights the fused passes read, the paged BF16 draft K / V cache (one page table, one
sequence), the prompt context (add_prompt_context: the prompt's target features -> every layer's context K / V, with
ttnn ops, once per prompt) and the per-round device inputs."""

import json
import os
from pathlib import Path

import torch

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute, reverse_permute_1d

TILE = ttnn.TILE_SIZE
ROWS = 32  # one tile of rows: the per-round inputs hold the block's 8 rows
DRAFT_ID = "z-lab/gpt-oss-20b-DFlash"
# a local directory with the drafter's checkpoint, used instead of the HF cache (CI: the shared HF cache is read-only
# and offline, so the job downloads the drafter into its own cache: tests/pipeline_reorg/models_e2e_tests.yaml)
DRAFT_PATH_ENV = "DFLASH_DRAFT_PATH"


def draft_snapshot():
    """The drafter's checkpoint directory: $DFLASH_DRAFT_PATH if set, else the HF cache (downloaded on first use)."""
    if os.getenv(DRAFT_PATH_ENV):
        return Path(os.environ[DRAFT_PATH_ENV])
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(DRAFT_ID))


def load_draft_config(path=None):
    path = Path(path or draft_snapshot())
    return json.loads((path / "config.json").read_text())


class DFlashDraft:
    def __init__(self, mesh_device, target, max_positions, path=None, block_size=64):
        """target: the gpt-oss Model (embedding table, LM head, RoPE config). max_positions: the longest sequence
        drafted for. window (DFlashRunner.set_draft_window): the draft pass attends to the last `window` positions only
        (SGLang's --speculative-dflash-draft-window-size; None, the checkpoint's setting: full attention)."""
        from safetensors.torch import load_file

        self.mesh_device = mesh_device
        self.target = target
        self.path = path or draft_snapshot()
        c = load_draft_config(self.path)
        D = mesh_device.shape[1]
        self.D = D
        self.hidden = c["hidden_size"]
        self.hd = c["head_dim"]
        self.nq, self.nkv = c["num_attention_heads"] // D, c["num_key_value_heads"] // D
        self.inter = c["intermediate_size"] // D
        self.layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.block = c["block_size"]
        self.mask_token_id = c["dflash_config"]["mask_token_id"]
        self.target_layer_ids = c["dflash_config"]["target_layer_ids"]
        self.ctx_in = len(self.target_layer_ids) * self.hidden
        self.window = None
        w = load_file(str(self.path / "model.safetensors"))
        mesh = mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def put(t, mapper=rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def norm_w(t):
            return put(t.float().reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)

        cols = ttnn.ShardTensorToMesh(mesh, dim=-1)
        hd, nkv = self.hd, self.nkv
        self.fc_full = put(w["fc.weight"].T)  # replicated (prompt context: no all-reduce over thousands of rows)
        self.hidden_norm = norm_w(w["hidden_norm.weight"])
        self.hidden_norm_row = put(w["hidden_norm.weight"].float().reshape(1, 1, 1, -1), dtype=ttnn.float32)
        self.norm = norm_w(w["norm.weight"])
        self.L = []
        for i in range(self.layers):
            p = f"layers.{i}."
            a = p + "self_attn."
            k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, self.hidden)
            kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0]
            v, vb = w[a + "v_proj.weight"], w[a + "v_proj.bias"]
            # per device: [k heads | v heads] columns (the prompt context's K / V projection)
            kv = torch.cat(
                [
                    torch.cat([k[d * nkv * hd : (d + 1) * nkv * hd], v[d * nkv * hd : (d + 1) * nkv * hd]])
                    for d in range(D)
                ]
            )
            kvb = torch.cat(
                [
                    torch.cat([kb[d * nkv * hd : (d + 1) * nkv * hd], vb[d * nkv * hd : (d + 1) * nkv * hd]])
                    for d in range(D)
                ]
            )
            self.L.append(
                dict(
                    kv=put(kv.T, cols),
                    kv_b=put(kvb.reshape(1, -1), cols),
                    ln1=norm_w(w[p + "input_layernorm.weight"]),
                    ln2=norm_w(w[p + "post_attention_layernorm.weight"]),
                    q_norm=norm_w(reverse_permute_1d(w[a + "q_norm.weight"])),
                    k_norm=norm_w(reverse_permute_1d(w[a + "k_norm.weight"])),
                )
            )
        # paged draft K / V cache of the one sequence
        self.block_size = block_size
        self.num_blocks = -(-(max_positions + self.block) // block_size)
        self.user_table = torch.arange(self.num_blocks, dtype=torch.int32)[None]
        self.caches = []
        heads_split = ttnn.ShardTensorToMesh(mesh, dim=1)
        for _ in range(self.layers):
            z = torch.zeros(self.num_blocks, D * nkv, block_size, hd)
            self.caches.append(
                tuple(
                    ttnn.from_torch(
                        z,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=heads_split,
                    )
                    for _ in range(2)
                )
            )
        self.page_table = ttnn.from_torch(
            self.user_table.repeat(ROWS, 1), dtype=ttnn.int32, device=mesh, mesh_mapper=rep
        )
        from ..model import create_rope_setup

        self.rope = create_rope_setup(mesh, target.hf_config, max_local_batch_size=ROWS, datatype=ttnn.bfloat16)
        self.head_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, hd),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.mm_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self._alloc_round_inputs()

    def _rms_norm_small(self, x, w_row):
        """RMSNorm of [1, 1, rows, hidden] as elementwise / reduce ops (FP32): small circular buffers, unlike
        ttnn.rms_norm's full-width ones, which do not fit next to the verify pass's persistent L1 buffers (prompt
        context only, once per prompt)."""
        xf = ttnn.typecast(x, ttnn.float32)
        x.deallocate(True)
        ms = ttnn.mean(ttnn.multiply(xf, xf), dim=-1, keepdim=True)
        r = ttnn.rsqrt(ttnn.add(ms, self.eps))
        ms.deallocate(True)
        y = ttnn.multiply(ttnn.multiply(xf, r), w_row, dtype=ttnn.bfloat16)
        xf.deallocate(True)
        r.deallocate(True)
        return y

    def add_prompt_context(self, features, chunk):
        """Context of prompt positions 0 .. n - 1 (prefill-style: [1, heads, rows, head_dim] K / V, paged_fill_cache of
        block-aligned chunks; positions past n in the last chunk are rewritten before they are read). features: host
        [n, 5 * hidden] target hidden states (target_layer_ids order)."""
        n = features.shape[0]
        mesh = self.mesh_device
        for s0 in range(0, n, chunk):
            rows = min(chunk, -(-(n - s0) // self.block_size) * self.block_size)
            x = torch.zeros(1, 1, rows, self.ctx_in)
            x[0, 0, : min(rows, n - s0)] = features[s0 : s0 + rows].float()
            tx = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ctx = ttnn.linear(tx, self.fc_full, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            tx.deallocate(True)
            ctx = self._rms_norm_small(ctx, self.hidden_norm_row)
            cos = self.rope.cos_matrix_prefill[:, :, s0 : s0 + rows, :]
            sin = self.rope.sin_matrix_prefill[:, :, s0 : s0 + rows, :]
            trans = self.rope.get_both_trans_mats()["prefill"]
            b0 = s0 // self.block_size
            chunk_pt = ttnn.from_torch(
                self.user_table[:, b0 : b0 + rows // self.block_size].contiguous(),
                dtype=ttnn.int32,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for i in range(self.layers):
                L = self.L[i]
                kv = ttnn.linear(ctx, L["kv"], bias=L["kv_b"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
                width = self.nkv * self.hd
                for j, cache in enumerate(self.caches[i]):
                    t = ttnn.slice(kv, (0, 0, 0, j * width), (1, 1, rows, (j + 1) * width))
                    t = ttnn.transpose(ttnn.reshape(t, [1, rows, self.nkv, self.hd]), 1, 2)  # [1, heads, rows, hd]
                    if j == 0:
                        t = ttnn.rms_norm(t, epsilon=self.eps, weight=L["k_norm"], compute_kernel_config=self.ck)
                        t = ttnn.experimental.rotary_embedding_llama(t, cos, sin, trans, is_decode_mode=False)
                    ttnn.experimental.paged_fill_cache(cache, t, chunk_pt, batch_idx=0)
                    t.deallocate(True)
                kv.deallocate(True)
            chunk_pt.deallocate(True)
            ctx.deallocate(True)

    def _alloc_round_inputs(self):
        """The per-round device inputs (written by the host for the first round, then by the verify pass on device,
        dflash/argmax.py ACCEPT): the block's token ids, positions, RoPE rows and attention bound, and the context
        rows' positions and RoPE rows; and the K / V row writer (dflash/kv_write.py)."""
        from .kv_write import DraftKVWrite

        mesh = self.mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def dev(t, dtype):
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )

        self.t_toks = dev(torch.full((1, ROWS), self.mask_token_id, dtype=torch.int32), ttnn.uint32)
        self.t_pos = dev(torch.full((ROWS,), -1, dtype=torch.int32), ttnn.int32)
        self.t_rot = dev(torch.zeros(1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.t_bound = dev(torch.full((ROWS,), -1, dtype=torch.int32), ttnn.int32)
        self.t_cpos = dev(torch.full((ROWS,), -1, dtype=torch.int32), ttnn.int32)
        self.t_crot = dev(torch.zeros(1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.kv_write = DraftKVWrite(mesh, self.block, self.nkv, self.hd, self.block_size)
        self.vd = self.target.decode_terminal.vd

    def _host_rows(self, values, dtype, shape=None):
        t = torch.tensor(values, dtype=torch.int32)
        return ttnn.from_torch(
            t.reshape(shape) if shape else t,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def set_draft_inputs(self, last_token, start):
        """The first round's inputs: the block [last_token, mask x 7] at positions start .. start + 7."""
        B = self.block
        toks = [last_token] + [self.mask_token_id] * (ROWS - 1)
        pos = [start + i for i in range(B)] + [-1] * (ROWS - B)
        rot = [start + i for i in range(B)] + [0] * (ROWS - B)
        bound = [start + B - 1] * B + [-1] * (ROWS - B)
        ttnn.copy_host_to_device_tensor(self._host_rows(toks, ttnn.uint32, (1, ROWS)), self.t_toks)
        ttnn.copy_host_to_device_tensor(self._host_rows(pos, ttnn.int32), self.t_pos)
        ttnn.copy_host_to_device_tensor(self._host_rows(rot, ttnn.uint32, (1, ROWS)), self.t_rot)
        ttnn.copy_host_to_device_tensor(self._host_rows(bound, ttnn.int32), self.t_bound)
